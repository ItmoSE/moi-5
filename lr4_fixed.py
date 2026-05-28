#!/usr/bin/env python3
"""
ЛР4. Формирование изображения методом трассировки путей.

Требования:
- Python 3.10+
- numpy
- Pillow (PIL)
- tkinter

Программа строит изображение треугольной сцены методом трассировки путей
с учетом диффузного и зеркального отражения, протяженных источников,
сглаживания, русской рулетки и простого интерфейса.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


EPS = 1e-5
INF = 1e30


# ----------------------------- math helpers -----------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    # Нормировка: v^ = v / ||v||, чтобы работать с единичными направлениями.
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    # Зеркальное отражение: r = v - 2 (v·n) n.
    return v - 2.0 * np.dot(v, n) * n


def clamp01(x: np.ndarray) -> np.ndarray:
    # Ограничение относительной яркости в диапазон [0, 1].
    return np.clip(x, 0.0, 1.0)


def luminance(rgb: np.ndarray) -> float:
    # Яркость RGB в линейной аппроксимации Y = 0.2126R + 0.7152G + 0.0722B.
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def random_in_unit_square(rng: np.random.Generator) -> Tuple[float, float]:
    # Равномерная выборка внутри пикселя для сглаживания.
    return float(rng.random()), float(rng.random())


def build_orthonormal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Строим базис t, b, n, чтобы выборку в локальной полусфере перевести в мировые координаты.
    if abs(n[0]) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t = normalize(np.cross(a, n))
    b = normalize(np.cross(n, t))
    return t, b


def sample_cosine_hemisphere(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Косинусная выборка Ламберта: p(w) = cos(theta) / pi, что уменьшает шум диффузного BRDF.
    r1 = float(rng.random())
    r2 = float(rng.random())
    phi = 2.0 * math.pi * r1
    r = math.sqrt(r2)
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r2))

    t, b = build_orthonormal_basis(normal)
    d = x * t + y * b + z * normal
    return normalize(d)


def sample_triangle(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    # Равномерная выборка точки по площади треугольника; площадь нужна для PDF = 1 / S.
    u = float(rng.random())
    v = float(rng.random())
    su = math.sqrt(u)
    b0 = 1.0 - su
    b1 = su * (1.0 - v)
    b2 = su * v
    p = b0 * v0 + b1 * v1 + b2 * v2
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return p, area


# ----------------------------- scene data -----------------------------


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray


@dataclass
class Material:
    # Материал: суммарное отражение по каждой компоненте не должно превышать 1.
    diffuse: np.ndarray
    mirror: np.ndarray
    emission: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    roughness: float = 0.35
    metallic: float = 0.0

    def validate(self) -> None:
        # Проверка физичности: k_d + k_s <= 1 по каждой компоненте RGB.
        total = self.diffuse + self.mirror
        if np.any(total > 1.0 + 1e-9):
            raise ValueError(
                f"Нефизичный материал: diffuse + mirror = {total}, а должно быть <= 1 по компонентам."
            )
        if not (0.0 < self.roughness <= 1.0):
            raise ValueError("Roughness должна быть в диапазоне (0, 1].")
        if not (0.0 <= self.metallic <= 1.0):
            raise ValueError("Metallic должен быть в диапазоне [0, 1].")


@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material_id: int
    object_id: int = 0
    normal: np.ndarray = field(init=False)
    area: float = field(init=False)

    def __post_init__(self) -> None:
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        n = np.cross(e1, e2)
        self.area = 0.5 * np.linalg.norm(n)
        self.normal = normalize(n)


@dataclass
class Hit:
    t: float
    position: np.ndarray
    normal: np.ndarray
    material_id: int
    tri_id: int
    object_id: int


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_deg: float

    def generate_ray(self, x: float, y: float, width: int, height: int) -> Ray:
        # Луч камеры через точку изображения; точечная камера без толщины линзы.
        forward = normalize(self.target - self.eye)
        right = normalize(np.cross(forward, self.up))
        true_up = normalize(np.cross(right, forward))

        aspect = width / height
        fov_y = math.radians(self.fov_y_deg)
        half_h = math.tan(fov_y * 0.5)
        half_w = aspect * half_h

        sx = (2.0 * x - 1.0) * half_w
        sy = (1.0 - 2.0 * y) * half_h
        d = normalize(forward + sx * right + sy * true_up)
        return Ray(self.eye.copy(), d)


@dataclass
class Scene:
    materials: List[Material]
    triangles: List[Triangle]
    camera: Camera
    background: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    tri_v0: Optional[np.ndarray] = None
    tri_e1: Optional[np.ndarray] = None
    tri_e2: Optional[np.ndarray] = None
    tri_normals: Optional[np.ndarray] = None
    tri_mat_ids: Optional[np.ndarray] = None
    tri_obj_ids: Optional[np.ndarray] = None
    emissive_ids_cache: Optional[List[int]] = None
    emissive_id_to_index: Optional[dict[int, int]] = None
    light_power_cache: Optional[np.ndarray] = None

    def emissive_triangles(self) -> List[int]:
        # Источники света — это треугольники с ненулевой собственной яркостью.
        if self.emissive_ids_cache is None:
            result = []
            for i, tri in enumerate(self.triangles):
                mat = self.materials[tri.material_id]
                if np.any(mat.emission > 0.0):
                    result.append(i)
            self.emissive_ids_cache = result
            self.emissive_id_to_index = {tri_id: idx for idx, tri_id in enumerate(result)}
        return self.emissive_ids_cache

    def finalize(self) -> None:
        # Предрасчет массивов: пересечение одного луча со всей сценой делаем векторно через NumPy.
        self.tri_v0 = np.array([t.v0 for t in self.triangles], dtype=np.float64)
        self.tri_e1 = np.array([t.v1 - t.v0 for t in self.triangles], dtype=np.float64)
        self.tri_e2 = np.array([t.v2 - t.v0 for t in self.triangles], dtype=np.float64)
        self.tri_normals = np.array([t.normal for t in self.triangles], dtype=np.float64)
        self.tri_mat_ids = np.array([t.material_id for t in self.triangles], dtype=np.int32)
        self.tri_obj_ids = np.array([t.object_id for t in self.triangles], dtype=np.int32)
        tri_ids = self.emissive_triangles()
        if tri_ids:
            self.light_power_cache = np.array([
                self.triangles[i].area * max(1e-9, luminance(self.materials[self.triangles[i].material_id].emission))
                for i in tri_ids
            ], dtype=np.float64)


# ----------------------------- geometry -----------------------------


def intersect_scene(scene: Scene, ray: Ray) -> Optional[Hit]:
    # Векторный Moller-Trumbore сразу по всем треугольникам: это заметно ускоряет Python-версию.
    v0 = scene.tri_v0
    e1 = scene.tri_e1
    e2 = scene.tri_e2
    if v0 is None or e1 is None or e2 is None:
        raise RuntimeError("Scene.finalize() must be called before rendering.")

    d = ray.direction
    pvec = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    valid = np.abs(det) > 1e-10
    if not np.any(valid):
        return None

    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = ray.origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    valid &= (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return None

    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", d, qvec) * inv_det
    valid &= (v >= 0.0) & ((u + v) <= 1.0)
    if not np.any(valid):
        return None

    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    valid &= t > EPS
    if not np.any(valid):
        return None

    masked = np.where(valid, t, INF)
    tri_id = int(np.argmin(masked))
    best_t = float(masked[tri_id])
    if best_t >= INF:
        return None

    pos = ray.origin + best_t * d
    n = scene.tri_normals[tri_id].copy()
    if np.dot(n, d) > 0.0:
        n = -n
    obj_id = -1 if scene.tri_obj_ids is None else int(scene.tri_obj_ids[tri_id])
    return Hit(
        t=best_t,
        position=pos,
        normal=n,
        material_id=int(scene.tri_mat_ids[tri_id]),
        tri_id=tri_id,
        object_id=obj_id,
    )


def visible(scene: Scene, p: np.ndarray, q: np.ndarray, ignore_tri_id: int) -> bool:
    # Проверка тени: между точками не должно быть пересечения ближе, чем источник.
    d = q - p
    dist = np.linalg.norm(d)
    if dist < 1e-10:
        return False
    direction = d / dist
    ray = Ray(p + direction * EPS * 4.0, direction)
    hit = intersect_scene(scene, ray)
    if hit is None:
        return False
    if hit.tri_id == ignore_tri_id:
        return hit.t >= dist - 2e-4
    return hit.t >= dist - 2e-4


# ----------------------------- lighting -----------------------------


def fresnel_schlick(cos_theta: float, f0: np.ndarray) -> np.ndarray:
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5


def ggx_D(alpha: float, cos_h: float) -> float:
    a2 = alpha * alpha
    denom = cos_h * cos_h * (a2 - 1.0) + 1.0
    return a2 / max(1e-12, math.pi * denom * denom)


def ggx_G(alpha: float, cos_i: float, cos_o: float) -> float:
    k = (alpha + 1.0) ** 2 / 8.0
    g1_i = cos_i / max(1e-12, cos_i * (1.0 - k) + k)
    g1_o = cos_o / max(1e-12, cos_o * (1.0 - k) + k)
    return g1_i * g1_o


def sample_ggx(normal: np.ndarray, wo: np.ndarray, roughness: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    # GGX выборка полувектора для отражения.
    r1 = float(rng.random())
    r2 = float(rng.random())
    alpha = max(1e-3, roughness * roughness)
    phi = 2.0 * math.pi * r1
    # Инверсия CDF для GGX распределения по полярному углу.
    cos_theta = math.sqrt((1.0 - r2) / (1.0 + (alpha * alpha - 1.0) * r2))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    t, b = build_orthonormal_basis(normal)
    h = normalize(t * (sin_theta * math.cos(phi)) + b * (sin_theta * math.sin(phi)) + normal * cos_theta)
    # Отражаем выходящее направление относительно микронормали h.
    wi = normalize(reflect(-wo, h))
    if np.dot(wi, normal) <= 0.0:
        return wi, 0.0

    cos_h = max(0.0, float(np.dot(normal, h)))
    wo_dot_h = max(1e-12, float(np.dot(wo, h)))
    pdf = ggx_D(alpha, cos_h) * cos_h / (4.0 * wo_dot_h)
    return wi, pdf


def diffuse_albedo(mat: Material, brdf_mode: str) -> np.ndarray:
    if brdf_mode == "cook_torrance":
        f0 = np.clip(mat.mirror, 0.0, 1.0)
        return mat.diffuse * (1.0 - mat.metallic) * (1.0 - float(np.max(f0)))
    return mat.diffuse


def eval_diffuse_brdf(mat: Material, brdf_mode: str) -> np.ndarray:
    return diffuse_albedo(mat, brdf_mode) / math.pi


def eval_specular_brdf(mat: Material, wi: np.ndarray, wo: np.ndarray, normal: np.ndarray) -> np.ndarray:
    cos_i = max(0.0, float(np.dot(normal, wi)))
    cos_o = max(0.0, float(np.dot(normal, wo)))
    if cos_i <= 0.0 or cos_o <= 0.0:
        return np.zeros(3, dtype=np.float64)

    # Cook-Торренс: D (GGX) * G (маскирование/затенение) * F (Schlick) / (4 cos_i cos_o).
    h = normalize(wi + wo)
    cos_h = max(0.0, float(np.dot(normal, h)))
    cos_wo_h = max(0.0, float(np.dot(wo, h)))
    alpha = max(1e-3, mat.roughness * mat.roughness)
    d = ggx_D(alpha, cos_h)
    g = ggx_G(alpha, cos_i, cos_o)
    f0 = np.clip(mat.mirror, 0.0, 1.0)
    f = fresnel_schlick(cos_wo_h, f0)
    return f * (d * g / max(1e-12, 4.0 * cos_i * cos_o))


def eval_brdf(mat: Material, wi: np.ndarray, wo: np.ndarray, normal: np.ndarray, brdf_mode: str) -> np.ndarray:
    if brdf_mode == "cook_torrance":
        # Сумма диффузного и микрофацетного зеркального вклада.
        return eval_diffuse_brdf(mat, brdf_mode) + eval_specular_brdf(mat, wi, wo, normal)
    return eval_diffuse_brdf(mat, brdf_mode)


def choose_event(material: Material, rng: np.random.Generator, brdf_mode: str) -> Tuple[str, float]:
    # Выбор события по значимости: вероятность пропорциональна энергии диффузной и зеркальной части.
    kd = float(np.mean(diffuse_albedo(material, brdf_mode)))
    ks = float(np.mean(np.clip(material.mirror, 0.0, 1.0)))
    s = kd + ks
    if s <= 1e-12:
        return "absorb", 1.0
    xi = float(rng.random()) * s
    if xi < kd:
        return "diffuse", max(1e-12, kd / s)
    return "specular", max(1e-12, ks / s)


def light_pdf_at_point(scene: Scene, from_pos: np.ndarray, light_hit: Hit) -> float:
    # PDF выбора той же точки на источнике, но выраженный по направлению wi.
    tri_ids = scene.emissive_triangles()
    if not tri_ids or scene.light_power_cache is None or scene.emissive_id_to_index is None:
        return 0.0

    idx = scene.emissive_id_to_index.get(light_hit.tri_id)
    if idx is None:
        return 0.0

    powers = scene.light_power_cache
    total_power = float(np.sum(powers))
    if total_power <= 0.0:
        return 0.0

    tri = scene.triangles[light_hit.tri_id]
    to_light = light_hit.position - from_pos
    dist2 = float(np.dot(to_light, to_light))
    if dist2 <= 1e-12:
        return 0.0

    wi = normalize(to_light)
    cos_light = max(0.0, float(np.dot(tri.normal, -wi)))
    if cos_light <= 0.0:
        return 0.0

    p_light = float(powers[idx] / total_power)
    p_area = 1.0 / max(1e-12, tri.area)
    return p_light * p_area * dist2 / cos_light


def direct_light(scene: Scene, hit: Hit, wo: np.ndarray, rng: np.random.Generator, brdf_mode: str) -> np.ndarray:
    # Оценка прямого освещения от одного случайного источника: L = Le * f_r * G / p(light) / p_A.
    tri_ids = scene.emissive_triangles()
    if not tri_ids or scene.light_power_cache is None:
        return np.zeros(3, dtype=np.float64)

    powers = scene.light_power_cache
    total_power = float(np.sum(powers))
    xi = float(rng.random()) * total_power
    chosen_idx = int(np.searchsorted(np.cumsum(powers), xi, side="left"))
    chosen_idx = min(chosen_idx, len(tri_ids) - 1)

    tri_id = tri_ids[chosen_idx]
    tri = scene.triangles[tri_id]
    light_mat = scene.materials[tri.material_id]

    q, area = sample_triangle(tri.v0, tri.v1, tri.v2, rng)
    light_n = tri.normal
    to_light = q - hit.position
    r2 = float(np.dot(to_light, to_light))
    if r2 < 1e-12:
        return np.zeros(3, dtype=np.float64)
    wi = normalize(to_light)

    cos_surface = max(0.0, float(np.dot(hit.normal, wi)))
    cos_light = max(0.0, float(np.dot(light_n, -wi)))
    if cos_surface <= 0.0 or cos_light <= 0.0:
        return np.zeros(3, dtype=np.float64)

    if not visible(scene, hit.position + hit.normal * EPS * 4.0, q, tri_id):
        return np.zeros(3, dtype=np.float64)

    # Вклад оценивается с учетом выбора источника и равномерной выборки по площади.
    p_light = float(powers[chosen_idx] / total_power)
    p_area = 1.0 / max(area, 1e-12)
    brdf = eval_brdf(scene.materials[hit.material_id], wi, wo, hit.normal, brdf_mode)
    geom = cos_surface * cos_light / r2
    return light_mat.emission * brdf * geom / max(p_light * p_area, 1e-12)


@dataclass
class TraceSample:
    radiance: np.ndarray
    direct: np.ndarray
    secondary: np.ndarray
    depth: float
    object_id: int
    normal: np.ndarray
    has_hit: bool


def trace_path(scene: Scene, ray: Ray, rng: np.random.Generator, max_depth: int, rr_start: int, brdf_mode: str) -> TraceSample:
    # Трассировка путей: L = sum beta * Le + beta * direct + beta * indirect, beta — накопленный вес пути.
    L = np.zeros(3, dtype=np.float64)
    first_direct = np.zeros(3, dtype=np.float64)
    secondary = np.zeros(3, dtype=np.float64)
    beta = np.ones(3, dtype=np.float64)
    prev_event = "camera"
    prev_hit_pos = None
    prev_bsdf_pdf = 0.0
    first_depth = 0.0
    first_object_id = -1
    first_normal = np.zeros(3, dtype=np.float64)
    has_first_hit = False

    for depth in range(max_depth):
        hit = intersect_scene(scene, ray)
        if hit is None:
            env = beta * scene.background
            L += env
            if depth == 0:
                first_direct += env
            else:
                secondary += env
            break

        if not has_first_hit:
            has_first_hit = True
            first_depth = hit.t
            first_object_id = hit.object_id
            first_normal = hit.normal.copy()

        mat = scene.materials[hit.material_id]
        if np.any(mat.emission > 0.0):
            if depth == 0 or prev_event == "specular":
                emit = beta * mat.emission
                L += emit
                if depth == 0:
                    first_direct += emit
                else:
                    secondary += emit
            elif prev_event == "diffuse" and prev_hit_pos is not None:
                # MIS: корректно учитываем попадание на источник после диффузного события.
                light_pdf = light_pdf_at_point(scene, prev_hit_pos, hit)
                mis_w = prev_bsdf_pdf / max(1e-12, prev_bsdf_pdf + light_pdf)
                emit = beta * mat.emission * mis_w
                L += emit
                secondary += emit

        if np.any(mat.diffuse > 0.0) or np.any(mat.mirror > 0.0):
            dl = beta * direct_light(scene, hit, -ray.direction, rng, brdf_mode)
            L += dl
            if depth == 0:
                first_direct += dl
            else:
                secondary += dl

        event, p_event = choose_event(mat, rng, brdf_mode)
        if event == "absorb":
            break

        if event == "diffuse":
            # Диффузная выборка: используем BRDF и pdf для корректного веса.
            new_dir = sample_cosine_hemisphere(hit.normal, rng)
            cos_i = max(0.0, float(np.dot(hit.normal, new_dir)))
            pdf = cos_i / math.pi
            brdf = eval_diffuse_brdf(mat, brdf_mode)
            beta = beta * brdf * cos_i / max(1e-12, pdf * p_event)
            prev_event = "diffuse"
            prev_bsdf_pdf = pdf * p_event
        else:
            if brdf_mode == "cook_torrance":
                # Микрофацетное отражение: выборка GGX задает PDF для MIS.
                wo = -ray.direction
                new_dir, pdf = sample_ggx(hit.normal, wo, mat.roughness, rng)
                if pdf <= 0.0:
                    break
                cos_i = max(0.0, float(np.dot(hit.normal, new_dir)))
                brdf = eval_specular_brdf(mat, new_dir, wo, hit.normal)
                beta = beta * brdf * cos_i / max(1e-12, pdf * p_event)
                prev_event = "specular"
                prev_bsdf_pdf = pdf * p_event
            else:
                # Идеальное зеркало — дельта-событие, направление определяется формулой отражения.
                new_dir = normalize(reflect(ray.direction, hit.normal))
                beta = beta * mat.mirror / max(1e-12, p_event)
                prev_event = "specular"
                prev_bsdf_pdf = 0.0

        prev_hit_pos = hit.position.copy()
        new_origin = hit.position + hit.normal * EPS * 4.0
        ray = Ray(new_origin, new_dir)

        if depth >= rr_start:
            # Русская рулетка: продолжаем путь с вероятностью p, а вклад делим на p для несмещенности оценки.
            p = min(0.99, max(0.05, float(np.max(beta))))
            if float(rng.random()) > p:
                break
            beta = beta / p

    return TraceSample(
        radiance=L,
        direct=first_direct,
        secondary=secondary,
        depth=first_depth,
        object_id=first_object_id,
        normal=first_normal,
        has_hit=has_first_hit,
    )


# ----------------------------- scene builders -----------------------------


def make_np3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def add_quad(
    tris: List[Triangle],
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    material_id: int,
    object_id: int,
) -> None:
    # Четырехугольник раскладываем на два треугольника.
    tris.append(Triangle(a, b, c, material_id, object_id=object_id))
    tris.append(Triangle(a, c, d, material_id, object_id=object_id))


def build_cornell_scene(
    width_room: float = 2.0,
    depth_room: float = 2.0,
    height_room: float = 2.0,
    light_power: float = 12.0,
    object_mirror_mix: float = 0.25,
    add_mesh: Optional[List[Triangle]] = None,
    mesh_material_id: Optional[int] = None,
) -> Scene:
    # Тестовая сцена на основе треугольной сетки: стены, пол, потолок, свет и два объекта.
    mats: List[Material] = [
        Material(diffuse=np.array([0.75, 0.75, 0.75]), mirror=np.array([0.0, 0.0, 0.0])),
        Material(diffuse=np.array([0.75, 0.15, 0.15]), mirror=np.array([0.0, 0.0, 0.0])),
        Material(diffuse=np.array([0.15, 0.75, 0.15]), mirror=np.array([0.0, 0.0, 0.0])),
        Material(diffuse=np.array([0.70, 0.70, 0.70]), mirror=np.array([object_mirror_mix] * 3)),
        Material(diffuse=np.array([0.0, 0.0, 0.0]), mirror=np.array([0.85, 0.85, 0.85])),
        Material(diffuse=np.array([0.0, 0.0, 0.0]), mirror=np.array([0.0, 0.0, 0.0]), emission=np.array([light_power] * 3)),
    ]
    for m in mats:
        m.validate()

    tris: List[Triangle] = []
    x0, x1 = -width_room / 2.0, width_room / 2.0
    z0, z1 = -depth_room / 2.0, depth_room / 2.0
    y0, y1 = 0.0, height_room

    add_quad(tris, make_np3(x0, y0, z1), make_np3(x1, y0, z1), make_np3(x1, y0, z0), make_np3(x0, y0, z0), 0, 1)
    add_quad(tris, make_np3(x0, y1, z0), make_np3(x1, y1, z0), make_np3(x1, y1, z1), make_np3(x0, y1, z1), 0, 2)
    add_quad(tris, make_np3(x0, y0, z0), make_np3(x0, y0, z1), make_np3(x0, y1, z1), make_np3(x0, y1, z0), 1, 3)
    add_quad(tris, make_np3(x1, y0, z1), make_np3(x1, y0, z0), make_np3(x1, y1, z0), make_np3(x1, y1, z1), 2, 4)
    add_quad(tris, make_np3(x0, y0, z0), make_np3(x1, y0, z0), make_np3(x1, y1, z0), make_np3(x0, y1, z0), 0, 5)

    ly = y1 - 1e-3
    add_quad(
        tris,
        make_np3(-0.35, ly, -0.35),
        make_np3(0.35, ly, -0.35),
        make_np3(0.35, ly, 0.35),
        make_np3(-0.35, ly, 0.35),
        5,
        6,
    )

    bx0, bx1 = -0.65, -0.10
    bz0, bz1 = -0.25, 0.40
    by0, by1 = 0.0, 0.85
    p000 = make_np3(bx0, by0, bz0)
    p100 = make_np3(bx1, by0, bz0)
    p110 = make_np3(bx1, by0, bz1)
    p010 = make_np3(bx0, by0, bz1)
    p001 = make_np3(bx0 + 0.08, by1, bz0 + 0.10)
    p101 = make_np3(bx1 + 0.10, by1, bz0 + 0.05)
    p111 = make_np3(bx1 + 0.05, by1, bz1 + 0.08)
    p011 = make_np3(bx0 + 0.02, by1, bz1 + 0.12)
    add_quad(tris, p000, p100, p110, p010, 3, 7)
    add_quad(tris, p001, p101, p111, p011, 3, 7)
    add_quad(tris, p000, p100, p101, p001, 3, 7)
    add_quad(tris, p100, p110, p111, p101, 3, 7)
    add_quad(tris, p110, p010, p011, p111, 3, 7)
    add_quad(tris, p010, p000, p001, p011, 3, 7)

    cx0, cx1 = 0.15, 0.65
    cz0, cz1 = -0.55, -0.10
    cy0, cy1 = 0.0, 1.20
    q000 = make_np3(cx0, cy0, cz0)
    q100 = make_np3(cx1, cy0, cz0)
    q110 = make_np3(cx1, cy0, cz1)
    q010 = make_np3(cx0, cy0, cz1)
    q001 = make_np3(cx0, cy1, cz0)
    q101 = make_np3(cx1, cy1, cz0)
    q111 = make_np3(cx1, cy1, cz1)
    q011 = make_np3(cx0, cy1, cz1)
    add_quad(tris, q000, q100, q110, q010, 4, 8)
    add_quad(tris, q001, q101, q111, q011, 4, 8)
    add_quad(tris, q000, q100, q101, q001, 4, 8)
    add_quad(tris, q100, q110, q111, q101, 4, 8)
    add_quad(tris, q110, q010, q011, q111, 4, 8)
    add_quad(tris, q010, q000, q001, q011, 4, 8)

    if add_mesh:
        mmid = 3 if mesh_material_id is None else mesh_material_id
        for tri in add_mesh:
            tris.append(Triangle(tri.v0.copy(), tri.v1.copy(), tri.v2.copy(), mmid, object_id=9))

    cam = Camera(
        eye=make_np3(0.0, 1.0, 3.2),
        target=make_np3(0.0, 0.9, 0.0),
        up=make_np3(0.0, 1.0, 0.0),
        fov_y_deg=40.0,
    )
    scene = Scene(materials=mats, triangles=tris, camera=cam, background=np.zeros(3, dtype=np.float64))
    scene.finalize()
    return scene


def load_obj_triangles(path: str, scale: float = 1.0, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Triangle]:
    # Загрузчик OBJ для вершин v и граней f; полигоны триангулируются веером -> треугольники.
    verts: List[np.ndarray] = []
    tris: List[Triangle] = []
    ox, oy, oz = offset

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if parts[0] == "v" and len(parts) >= 4:
                x, y, z = map(float, parts[1:4])
                verts.append(make_np3(scale * x + ox, scale * y + oy, scale * z + oz))
            elif parts[0] == "f" and len(parts) >= 4:
                idxs = []
                for p in parts[1:]:
                    token = p.split("/")[0]
                    if not token:
                        continue
                    i = int(token)
                    if i < 0:
                        i = len(verts) + i + 1
                    idxs.append(i - 1)
                if len(idxs) < 3:
                    continue
                for k in range(1, len(idxs) - 1):
                    v0 = verts[idxs[0]]
                    v1 = verts[idxs[k]]
                    v2 = verts[idxs[k + 1]]
                    tris.append(Triangle(v0.copy(), v1.copy(), v2.copy(), 3))
    return tris


# ----------------------------- image output -----------------------------


def tonemap_and_gamma(hdr: np.ndarray, normalize_mode: str = "max", gamma: float = 2.2) -> np.ndarray:
    # Переход к относительным яркостям: сначала нормировка, затем отсечение и гамма-коррекция I_out = I_in^(1/gamma).
    img = hdr.copy()
    if normalize_mode == "max":
        m = float(np.max(img))
        if m > 1e-12:
            img /= m
    elif normalize_mode == "mean05":
        m = float(np.mean(img))
        if m > 1e-12:
            img *= 0.5 / m

    img = clamp01(img)
    img = np.power(img, 1.0 / gamma)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def save_ppm(path: str, rgb8: np.ndarray) -> None:
    # Формат PPM P6: заголовок + бинарные RGB-байты.
    h, w, _ = rgb8.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())


def save_hdr_npy(path: str, hdr: np.ndarray) -> None:
    # Сохраняем линейные яркости для повторной визуализации и проверки.
    np.save(path, hdr)


# ----------------------------- renderer -----------------------------


@dataclass
class RenderParams:
    width: int = 500
    height: int = 500
    spp: int = 4
    max_depth: int = 4
    rr_start: int = 2
    gamma: float = 2.2
    normalize_mode: str = "max"
    seed: int = 42
    preview_every_rows: int = 16
    brdf_mode: str = "lambert"
    workers: int = 1


@dataclass
class RenderOutput:
    hdr: np.ndarray
    rgb8: np.ndarray
    direct: np.ndarray
    secondary: np.ndarray
    depth: np.ndarray
    object_id: np.ndarray
    normal: np.ndarray


class PathTracer:
    def __init__(self, scene: Scene, params: RenderParams) -> None:
        self.scene = scene
        self.params = params

    def render(self, progress_cb=None, preview_cb=None) -> RenderOutput:
        # Монте-Карло оценка: среднее по spp независимым лучам на пиксель; превью обновляем по строкам.
        w = self.params.width
        h = self.params.height
        spp = self.params.spp
        hdr = np.zeros((h, w, 3), dtype=np.float64)
        direct = np.zeros((h, w, 3), dtype=np.float64)
        secondary = np.zeros((h, w, 3), dtype=np.float64)
        depth = np.zeros((h, w), dtype=np.float64)
        object_id = np.full((h, w), -1, dtype=np.int32)
        normal = np.zeros((h, w, 3), dtype=np.float64)
        seeds = np.arange(h * w, dtype=np.int64) + self.params.seed * 10003

        def render_row(y: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            row_hdr = np.zeros((w, 3), dtype=np.float64)
            row_direct = np.zeros((w, 3), dtype=np.float64)
            row_secondary = np.zeros((w, 3), dtype=np.float64)
            row_depth = np.zeros(w, dtype=np.float64)
            row_obj = np.full(w, -1, dtype=np.int32)
            row_normal = np.zeros((w, 3), dtype=np.float64)
            row_offset = y * w
            for x in range(w):
                pixel = np.zeros(3, dtype=np.float64)
                pixel_direct = np.zeros(3, dtype=np.float64)
                pixel_secondary = np.zeros(3, dtype=np.float64)
                depth_acc = 0.0
                normal_acc = np.zeros(3, dtype=np.float64)
                hit_count = 0
                obj_samples = np.full(spp, -1, dtype=np.int32)
                rng = np.random.default_rng(int(seeds[row_offset + x]))
                for s in range(spp):
                    dx, dy = random_in_unit_square(rng)
                    # Нормализованные координаты в [0, 1] для генерации луча камеры.
                    sx = (x + dx) / w
                    sy = (y + dy) / h
                    ray = self.scene.camera.generate_ray(sx, sy, w, h)
                    sample = trace_path(
                        self.scene,
                        ray,
                        rng,
                        self.params.max_depth,
                        self.params.rr_start,
                        self.params.brdf_mode,
                    )
                    pixel += sample.radiance
                    pixel_direct += sample.direct
                    pixel_secondary += sample.secondary
                    if sample.has_hit:
                        depth_acc += sample.depth
                        normal_acc += sample.normal
                        obj_samples[s] = sample.object_id
                        hit_count += 1

                inv_spp = 1.0 / max(1, spp)
                row_hdr[x] = pixel * inv_spp
                row_direct[x] = pixel_direct * inv_spp
                row_secondary[x] = pixel_secondary * inv_spp
                if hit_count > 0:
                    row_depth[x] = depth_acc / hit_count
                    row_normal[x] = normalize(normal_acc / hit_count)
                    valid_obj = obj_samples[obj_samples >= 0]
                    if valid_obj.size > 0:
                        vals, counts = np.unique(valid_obj, return_counts=True)
                        row_obj[x] = int(vals[int(np.argmax(counts))])
            return y, row_hdr, row_direct, row_secondary, row_depth, row_obj, row_normal

        workers = max(1, self.params.workers)
        done_rows = 0
        preview_step = max(1, self.params.preview_every_rows)

        if workers == 1:
            for y in range(h):
                y0, row_hdr, row_direct, row_secondary, row_depth, row_obj, row_normal = render_row(y)
                hdr[y0] = row_hdr
                direct[y0] = row_direct
                secondary[y0] = row_secondary
                depth[y0] = row_depth
                object_id[y0] = row_obj
                normal[y0] = row_normal
                done_rows += 1
                if progress_cb is not None:
                    progress_cb(done_rows / h)
                if preview_cb is not None and ((done_rows % preview_step == 0) or (done_rows == h)):
                    preview_cb(tonemap_and_gamma(hdr, normalize_mode=self.params.normalize_mode, gamma=self.params.gamma))
        else:
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                tasks = [pool.submit(render_row, y) for y in range(h)]
                for task in futures.as_completed(tasks):
                    y0, row_hdr, row_direct, row_secondary, row_depth, row_obj, row_normal = task.result()
                    hdr[y0] = row_hdr
                    direct[y0] = row_direct
                    secondary[y0] = row_secondary
                    depth[y0] = row_depth
                    object_id[y0] = row_obj
                    normal[y0] = row_normal
                    done_rows += 1
                    if progress_cb is not None:
                        progress_cb(done_rows / h)
                    if preview_cb is not None and ((done_rows % preview_step == 0) or (done_rows == h)):
                        preview_cb(tonemap_and_gamma(hdr, normalize_mode=self.params.normalize_mode, gamma=self.params.gamma))

        rgb8 = tonemap_and_gamma(hdr, normalize_mode=self.params.normalize_mode, gamma=self.params.gamma)
        return RenderOutput(
            hdr=hdr,
            rgb8=rgb8,
            direct=direct,
            secondary=secondary,
            depth=depth,
            object_id=object_id,
            normal=normal,
        )


# ----------------------------- GUI -----------------------------


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ЛР4 - Path Tracer")
        self.root.geometry("1250x760")

        self.obj_path: Optional[str] = None
        self.last_hdr: Optional[np.ndarray] = None
        self.last_rgb8: Optional[np.ndarray] = None
        self.last_render: Optional[RenderOutput] = None
        self.preview_photo = None
        self.render_thread: Optional[threading.Thread] = None
        self.progress_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self._build_ui()
        self._schedule_poll()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y")
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        lf = ttk.LabelFrame(left, text="Параметры сцены и рендера", padding=8)
        lf.pack(fill="x", padx=4, pady=4)

        self.width_var = tk.StringVar(value="500")
        self.height_var = tk.StringVar(value="500")
        self.spp_var = tk.StringVar(value="4")
        self.depth_var = tk.StringVar(value="4")
        self.rr_var = tk.StringVar(value="2")
        self.gamma_var = tk.StringVar(value="2.2")
        self.seed_var = tk.StringVar(value="42")
        self.workers_var = tk.StringVar(value="1")
        self.light_var = tk.StringVar(value="12.0")
        self.mirror_mix_var = tk.StringVar(value="0.25")
        self.norm_var = tk.StringVar(value="max")
        self.brdf_var = tk.StringVar(value="lambert")
        self.obj_scale_var = tk.StringVar(value="0.6")
        self.obj_offx_var = tk.StringVar(value="0.0")
        self.obj_offy_var = tk.StringVar(value="0.0")
        self.obj_offz_var = tk.StringVar(value="0.0")

        rows = [
            ("Ширина", self.width_var),
            ("Высота", self.height_var),
            ("SPP", self.spp_var),
            ("Max depth", self.depth_var),
            ("RR start", self.rr_var),
            ("Gamma", self.gamma_var),
            ("Seed", self.seed_var),
            ("Workers", self.workers_var),
            ("Сила света", self.light_var),
            ("Mirror mix", self.mirror_mix_var),
        ]
        for i, (label, var) in enumerate(rows):
            ttk.Label(lf, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(lf, textvariable=var, width=12).grid(row=i, column=1, sticky="ew", pady=2, padx=5)

        base_row = len(rows)
        ttk.Label(lf, text="Нормировка").grid(row=base_row, column=0, sticky="w", pady=2)
        norm_box = ttk.Combobox(lf, textvariable=self.norm_var, values=["max", "mean05"], width=10, state="readonly")
        norm_box.grid(row=base_row, column=1, sticky="ew", pady=2, padx=5)

        ttk.Label(lf, text="BRDF").grid(row=base_row + 1, column=0, sticky="w", pady=2)
        brdf_box = ttk.Combobox(
            lf,
            textvariable=self.brdf_var,
            values=["lambert", "cook_torrance"],
            width=14,
            state="readonly",
        )
        brdf_box.grid(row=base_row + 1, column=1, sticky="ew", pady=2, padx=5)

        objf = ttk.LabelFrame(left, text="OBJ (необязательно)", padding=8)
        objf.pack(fill="x", padx=4, pady=4)
        self.obj_label = ttk.Label(objf, text="OBJ: не выбран", wraplength=280)
        self.obj_label.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(objf, text="Открыть OBJ", command=self.load_obj).grid(row=1, column=0, pady=4, sticky="ew")
        ttk.Button(objf, text="Очистить OBJ", command=self.clear_obj).grid(row=1, column=1, pady=4, sticky="ew")
        ttk.Label(objf, text="Scale").grid(row=2, column=0, sticky="w")
        ttk.Entry(objf, textvariable=self.obj_scale_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(objf, text="Offset x y z").grid(row=3, column=0, sticky="w")
        off_frame = ttk.Frame(objf)
        off_frame.grid(row=3, column=1, columnspan=2, sticky="w")
        ttk.Entry(off_frame, textvariable=self.obj_offx_var, width=6).pack(side="left")
        ttk.Entry(off_frame, textvariable=self.obj_offy_var, width=6).pack(side="left", padx=2)
        ttk.Entry(off_frame, textvariable=self.obj_offz_var, width=6).pack(side="left")

        btnf = ttk.Frame(left)
        btnf.pack(fill="x", padx=4, pady=4)
        self.render_btn = ttk.Button(btnf, text="Рендер", command=self.start_render)
        self.render_btn.pack(fill="x", pady=2)
        ttk.Button(btnf, text="Сохранить PPM", command=self.save_ppm_dialog).pack(fill="x", pady=2)
        ttk.Button(btnf, text="Сохранить HDR (.npy)", command=self.save_hdr_dialog).pack(fill="x", pady=2)
        ttk.Button(btnf, text="Сохранить AOV (.npz)", command=self.save_aov_dialog).pack(fill="x", pady=2)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(left, textvariable=self.status_var).pack(fill="x", padx=4, pady=4)
        self.pb = ttk.Progressbar(left, orient="horizontal", mode="determinate")
        self.pb.pack(fill="x", padx=4, pady=4)

        pvf = ttk.LabelFrame(right, text="Изображение", padding=8)
        pvf.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas = tk.Canvas(pvf, bg="#222")
        self.canvas.pack(fill="both", expand=True)

        help_text = (
            "Подсказка:\n"
            "- Минимальное разрешение по ТЗ: 500x500.\n"
            "- Быстрый режим для проверки: 4 spp, depth 4.\n"
            "- Для финального кадра поднимите spp до 16-32.\n"
            "- Для 5 ЛР сохраните AOV (.npz): direct/secondary/depth/object_id/normal."
        )
        ttk.Label(right, text=help_text, justify="left").pack(anchor="w", padx=8, pady=4)

    def load_obj(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите OBJ",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")],
        )
        if not path:
            return
        self.obj_path = path
        self.obj_label.configure(text=f"OBJ: {path}")

    def clear_obj(self) -> None:
        self.obj_path = None
        self.obj_label.configure(text="OBJ: не выбран")

    def _parse_params(self) -> Tuple[RenderParams, Scene]:
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        spp = int(self.spp_var.get())
        max_depth = int(self.depth_var.get())
        rr_start = int(self.rr_var.get())
        gamma = float(self.gamma_var.get())
        seed = int(self.seed_var.get())
        workers = int(self.workers_var.get())
        light_power = float(self.light_var.get())
        mirror_mix = float(self.mirror_mix_var.get())
        normalize_mode = self.norm_var.get().strip()
        brdf_mode = self.brdf_var.get().strip()

        if width < 500 or height < 500:
            raise ValueError("По ТЗ разрешение должно быть не менее 500x500.")
        if spp <= 0:
            raise ValueError("SPP должно быть положительным.")
        if max_depth <= 0:
            raise ValueError("Max depth должно быть положительным.")
        if rr_start < 0:
            raise ValueError("RR start не может быть отрицательным.")
        if gamma <= 0.0:
            raise ValueError("Gamma должна быть положительной.")
        if workers <= 0:
            raise ValueError("Workers должно быть положительным целым.")
        if not (0.0 <= mirror_mix <= 1.0):
            raise ValueError("Mirror mix должен быть в диапазоне [0, 1].")
        if normalize_mode not in {"max", "mean05"}:
            raise ValueError("Режим нормировки должен быть max или mean05.")
        if brdf_mode not in {"lambert", "cook_torrance"}:
            raise ValueError("BRDF должен быть lambert или cook_torrance.")

        obj_tris = None
        if self.obj_path:
            scale = float(self.obj_scale_var.get())
            offx = float(self.obj_offx_var.get())
            offy = float(self.obj_offy_var.get())
            offz = float(self.obj_offz_var.get())
            obj_tris = load_obj_triangles(self.obj_path, scale=scale, offset=(offx, offy, offz))

        scene = build_cornell_scene(
            light_power=light_power,
            object_mirror_mix=mirror_mix,
            add_mesh=obj_tris,
            mesh_material_id=3,
        )
        params = RenderParams(
            width=width,
            height=height,
            spp=spp,
            max_depth=max_depth,
            rr_start=rr_start,
            gamma=gamma,
            normalize_mode=normalize_mode,
            seed=seed,
            brdf_mode=brdf_mode,
            workers=workers,
        )
        return params, scene

    def start_render(self) -> None:
        if self.render_thread is not None and self.render_thread.is_alive():
            messagebox.showinfo("Рендер", "Рендер уже запущен.")
            return

        try:
            params, scene = self._parse_params()
        except Exception as e:
            messagebox.showerror("Ошибка параметров", str(e))
            return

        self.render_btn.configure(state="disabled")
        self.pb["value"] = 0
        self.status_var.set("Рендер...")
        self.last_hdr = None
        self.last_rgb8 = None
        self.last_render = None
        self.canvas.delete("all")
        self.canvas.create_text(20, 20, anchor="nw", fill="white", text="Рендер запущен...")

        def worker() -> None:
            try:
                start = time.time()
                tracer = PathTracer(scene, params)

                def progress_cb(v: float) -> None:
                    # Обновления интерфейса отправляем через очередь, чтобы не трогать GUI из потока рендера.
                    self.progress_queue.put(("progress", v))

                def preview_cb(rgb8: np.ndarray) -> None:
                    self.progress_queue.put(("preview", rgb8))

                result = tracer.render(progress_cb=progress_cb, preview_cb=preview_cb)
                elapsed = time.time() - start
                self.progress_queue.put(("done", (result, elapsed)))
            except Exception as ex:
                self.progress_queue.put(("error", str(ex)))

        self.render_thread = threading.Thread(target=worker, daemon=True)
        self.render_thread.start()

    def _schedule_poll(self) -> None:
        self.root.after(100, self._poll_queue)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, data = self.progress_queue.get_nowait()
                if kind == "progress":
                    # Прогресс рендера (0..100%).
                    self.pb["value"] = float(data) * 100.0
                    self.status_var.set(f"Рендер... {float(data) * 100.0:.1f}%")
                elif kind == "preview":
                    # Промежуточный предпросмотр.
                    self.show_image(data)
                elif kind == "done":
                    result, elapsed = data
                    self.last_render = result
                    self.last_hdr = result.hdr
                    self.last_rgb8 = result.rgb8
                    self.show_image(result.rgb8)
                    self.pb["value"] = 100.0
                    self.status_var.set(f"Готово. Время: {elapsed:.2f} с")
                    self.render_btn.configure(state="normal")
                elif kind == "error":
                    self.status_var.set("Ошибка")
                    self.render_btn.configure(state="normal")
                    messagebox.showerror("Ошибка рендера", str(data))
        except queue.Empty:
            pass
        self._schedule_poll()

    def show_image(self, rgb8: np.ndarray) -> None:
        if not PIL_AVAILABLE:
            self.canvas.delete("all")
            self.canvas.create_text(
                20,
                20,
                anchor="nw",
                text="Pillow не найден. Сохраните PPM и откройте изображение внешней программой.",
                fill="white",
            )
            return

        img = Image.fromarray(rgb8, mode="RGB")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / img.width, ch / img.height)
        nw = max(1, int(img.width * scale))
        nh = max(1, int(img.height * scale))
        img = img.resize((nw, nh), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.preview_photo)

    def save_ppm_dialog(self) -> None:
        if self.last_rgb8 is None:
            messagebox.showinfo("Сохранение", "Сначала выполните рендер.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить PPM",
            defaultextension=".ppm",
            filetypes=[("PPM", "*.ppm"), ("All files", "*.*")],
        )
        if not path:
            return
        save_ppm(path, self.last_rgb8)
        messagebox.showinfo("Сохранение", f"Изображение сохранено:\n{path}")

    def save_hdr_dialog(self) -> None:
        if self.last_hdr is None:
            messagebox.showinfo("Сохранение", "Сначала выполните рендер.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить HDR в NPY",
            defaultextension=".npy",
            filetypes=[("NumPy", "*.npy"), ("All files", "*.*")],
        )
        if not path:
            return
        save_hdr_npy(path, self.last_hdr)
        messagebox.showinfo("Сохранение", f"HDR сохранен:\n{path}")

    def save_aov_dialog(self) -> None:
        if self.last_render is None:
            messagebox.showinfo("Сохранение", "Сначала выполните рендер.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить AOV в NPZ",
            defaultextension=".npz",
            filetypes=[("NumPy zip", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        np.savez_compressed(
            path,
            hdr=self.last_render.hdr,
            rgb8=self.last_render.rgb8,
            direct=self.last_render.direct,
            secondary=self.last_render.secondary,
            depth=self.last_render.depth,
            object_id=self.last_render.object_id,
            normal=self.last_render.normal,
        )
        messagebox.showinfo("Сохранение", f"AOV сохранен:\n{path}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
