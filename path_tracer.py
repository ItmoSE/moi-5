from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-6
RAY_EPS = 1e-4
PI = math.pi


def vec3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def luminance(c: np.ndarray) -> float:
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return direction - 2.0 * float(np.dot(direction, normal)) * normal


def max_component(v: np.ndarray) -> float:
    return float(np.max(v))


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class Material:
    kd: np.ndarray
    ks: np.ndarray
    emission: np.ndarray

    def __post_init__(self) -> None:
        kd = np.asarray(self.kd, dtype=np.float64)
        ks = np.asarray(self.ks, dtype=np.float64)
        emission = np.asarray(self.emission, dtype=np.float64)

        if kd.shape != (3,) or ks.shape != (3,) or emission.shape != (3,):
            raise ValueError("Material vectors must have shape (3,)")
        if np.any(kd < 0.0) or np.any(ks < 0.0) or np.any(emission < 0.0):
            raise ValueError("Material properties must be non-negative")
        if np.any(kd + ks > 1.0 + 1e-9):
            raise ValueError("Material violates energy conservation: kd + ks must be <= 1")

        object.__setattr__(self, "kd", kd)
        object.__setattr__(self, "ks", ks)
        object.__setattr__(self, "emission", emission)


@dataclass
class TrianglePrimitive:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    material_id: int


@dataclass(frozen=True)
class Hit:
    t: float
    tri_index: int
    position: np.ndarray
    normal: np.ndarray
    material: Material


@dataclass(frozen=True)
class Scene:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    mat_ids: np.ndarray
    materials: tuple[Material, ...]
    background: np.ndarray
    light_indices: np.ndarray
    light_pmf: np.ndarray
    light_cdf: np.ndarray

    def intersect(self, ray: Ray, t_min: float = RAY_EPS, t_max: float = float("inf")) -> Hit | None:
        direction = ray.direction

        pvec = np.cross(direction, self.e2)
        det = np.einsum("ij,ij->i", self.e1, pvec)
        valid = np.abs(det) > EPS
        if not np.any(valid):
            return None

        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]

        tvec = ray.origin - self.v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)
        if not np.any(valid):
            return None

        qvec = np.cross(tvec, self.e1)
        v = np.einsum("ij,j->i", qvec, direction) * inv_det
        valid &= (v >= 0.0) & (u + v <= 1.0)
        if not np.any(valid):
            return None

        t = np.einsum("ij,ij->i", self.e2, qvec) * inv_det
        valid &= (t > t_min) & (t < t_max)
        if not np.any(valid):
            return None

        candidate_t = np.where(valid, t, np.inf)
        tri_index = int(np.argmin(candidate_t))
        t_hit = float(candidate_t[tri_index])
        hit_pos = ray.origin + t_hit * direction

        normal = self.normals[tri_index]
        if float(np.dot(normal, direction)) > 0.0:
            normal = -normal

        material = self.materials[int(self.mat_ids[tri_index])]
        return Hit(t=t_hit, tri_index=tri_index, position=hit_pos, normal=normal, material=material)

    def occluded(self, origin: np.ndarray, direction: np.ndarray, max_distance: float) -> bool:
        hit = self.intersect(Ray(origin=origin, direction=direction), t_min=RAY_EPS, t_max=max_distance)
        return hit is not None

    def sample_light(
        self, rng: np.random.Generator
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float] | None:
        if self.light_indices.size == 0:
            return None

        xi = float(rng.random())
        light_slot = int(np.searchsorted(self.light_cdf, xi, side="right"))
        light_slot = min(light_slot, self.light_indices.size - 1)
        tri_index = int(self.light_indices[light_slot])

        u = float(rng.random())
        v = float(rng.random())
        su = math.sqrt(u)
        b0 = 1.0 - su
        b1 = su * (1.0 - v)
        b2 = su * v

        point = b0 * self.v0[tri_index] + b1 * self.v1[tri_index] + b2 * self.v2[tri_index]
        normal = self.normals[tri_index]
        emission = self.materials[int(self.mat_ids[tri_index])].emission

        pdf_area = 1.0 / float(self.areas[tri_index])
        pdf = float(self.light_pmf[light_slot]) * pdf_area
        return tri_index, point, normal, emission, pdf


@dataclass(frozen=True)
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_degrees: float
    _forward: np.ndarray = field(init=False, repr=False)
    _right: np.ndarray = field(init=False, repr=False)
    _true_up: np.ndarray = field(init=False, repr=False)
    _tan_half_fov: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        eye = np.asarray(self.eye, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        if eye.shape != (3,) or target.shape != (3,) or up.shape != (3,):
            raise ValueError("Camera vectors must have shape (3,)")
        if self.fov_degrees <= 0.0 or self.fov_degrees >= 179.0:
            raise ValueError("Field of view must be in (0, 179)")

        forward = normalize(target - eye)
        right_raw = np.cross(forward, up)
        right_norm = float(np.linalg.norm(right_raw))
        if right_norm <= EPS:
            raise ValueError("Camera up vector is parallel to viewing direction")
        right = right_raw / right_norm
        true_up = normalize(np.cross(right, forward))
        tan_half_fov = math.tan(math.radians(self.fov_degrees) * 0.5)

        object.__setattr__(self, "eye", eye)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "_forward", forward)
        object.__setattr__(self, "_right", right)
        object.__setattr__(self, "_true_up", true_up)
        object.__setattr__(self, "_tan_half_fov", tan_half_fov)

    def generate_ray(
        self, pixel_x: int, pixel_y: int, width: int, height: int, rng: np.random.Generator
    ) -> Ray:
        aspect = float(width) / float(height)

        sx = (2.0 * ((pixel_x + float(rng.random())) / float(width)) - 1.0) * aspect * self._tan_half_fov
        sy = (1.0 - 2.0 * ((pixel_y + float(rng.random())) / float(height))) * self._tan_half_fov

        direction = normalize(self._forward + sx * self._right + sy * self._true_up)
        return Ray(origin=self.eye.copy(), direction=direction)


def oriented_triangle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, desired_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.cross(b - a, c - a)
    if float(np.dot(normal, desired_normal)) < 0.0:
        return a, c, b
    return a, b, c


def add_quad(
    triangles: list[TrianglePrimitive],
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    material_id: int,
    desired_normal: np.ndarray,
) -> None:
    t0 = oriented_triangle(a, b, c, desired_normal)
    t1 = oriented_triangle(a, c, d, desired_normal)
    triangles.append(TrianglePrimitive(v0=t0[0], v1=t0[1], v2=t0[2], material_id=material_id))
    triangles.append(TrianglePrimitive(v0=t1[0], v1=t1[1], v2=t1[2], material_id=material_id))


def add_box(
    triangles: list[TrianglePrimitive],
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    material_id: int,
) -> None:
    x0, y0, z0 = map(float, min_corner)
    x1, y1, z1 = map(float, max_corner)

    p000 = vec3(x0, y0, z0)
    p001 = vec3(x0, y0, z1)
    p010 = vec3(x0, y1, z0)
    p011 = vec3(x0, y1, z1)
    p100 = vec3(x1, y0, z0)
    p101 = vec3(x1, y0, z1)
    p110 = vec3(x1, y1, z0)
    p111 = vec3(x1, y1, z1)

    add_quad(triangles, p000, p001, p011, p010, material_id, vec3(-1.0, 0.0, 0.0))
    add_quad(triangles, p100, p110, p111, p101, material_id, vec3(1.0, 0.0, 0.0))
    add_quad(triangles, p000, p100, p101, p001, material_id, vec3(0.0, -1.0, 0.0))
    add_quad(triangles, p010, p011, p111, p110, material_id, vec3(0.0, 1.0, 0.0))
    add_quad(triangles, p000, p010, p110, p100, material_id, vec3(0.0, 0.0, -1.0))
    add_quad(triangles, p001, p101, p111, p011, material_id, vec3(0.0, 0.0, 1.0))


def build_scene(
    primitives: list[TrianglePrimitive], materials: list[Material], background: np.ndarray
) -> Scene:
    if not primitives:
        raise ValueError("Scene must contain at least one triangle")

    v0 = np.stack([np.asarray(t.v0, dtype=np.float64) for t in primitives], axis=0)
    v1 = np.stack([np.asarray(t.v1, dtype=np.float64) for t in primitives], axis=0)
    v2 = np.stack([np.asarray(t.v2, dtype=np.float64) for t in primitives], axis=0)
    mat_ids = np.array([int(t.material_id) for t in primitives], dtype=np.int32)

    if np.any(mat_ids < 0) or np.any(mat_ids >= len(materials)):
        raise ValueError("Triangle uses material id out of range")

    e1 = v1 - v0
    e2 = v2 - v0
    raw_normals = np.cross(e1, e2)
    lengths = np.linalg.norm(raw_normals, axis=1)
    if np.any(lengths <= EPS):
        raise ValueError("Degenerate triangle detected in scene")

    normals = raw_normals / lengths[:, None]
    areas = 0.5 * lengths

    light_indices: list[int] = []
    light_weights: list[float] = []
    for tri_idx, material_id in enumerate(mat_ids):
        emission = materials[int(material_id)].emission
        weight = float(areas[tri_idx]) * max(0.0, luminance(emission))
        if weight > 0.0:
            light_indices.append(tri_idx)
            light_weights.append(weight)

    if light_weights:
        weights = np.asarray(light_weights, dtype=np.float64)
        pmf = weights / float(np.sum(weights))
        cdf = np.cumsum(pmf)
        cdf[-1] = 1.0
        light_idx_arr = np.asarray(light_indices, dtype=np.int32)
    else:
        pmf = np.zeros((0,), dtype=np.float64)
        cdf = np.zeros((0,), dtype=np.float64)
        light_idx_arr = np.zeros((0,), dtype=np.int32)

    return Scene(
        v0=v0,
        v1=v1,
        v2=v2,
        e1=e1,
        e2=e2,
        normals=normals,
        areas=areas,
        mat_ids=mat_ids,
        materials=tuple(materials),
        background=np.asarray(background, dtype=np.float64),
        light_indices=light_idx_arr,
        light_pmf=pmf,
        light_cdf=cdf,
    )


def load_obj(path: Path, material_id: int) -> list[TrianglePrimitive]:
    if not path.exists():
        raise FileNotFoundError(f"OBJ file not found: {path}")

    vertices: list[np.ndarray] = []
    triangles: list[TrianglePrimitive] = []

    with path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"Invalid vertex at line {line_number}: {line.rstrip()}")
                vertices.append(vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                continue

            if parts[0] != "f":
                continue

            if len(parts) < 4:
                raise ValueError(f"Face with fewer than 3 vertices at line {line_number}")

            face_indices: list[int] = []
            for token in parts[1:]:
                idx_token = token.split("/")[0]
                if not idx_token:
                    raise ValueError(f"Invalid face index at line {line_number}: {line.rstrip()}")
                raw_index = int(idx_token)
                if raw_index > 0:
                    vertex_index = raw_index - 1
                else:
                    vertex_index = len(vertices) + raw_index
                if vertex_index < 0 or vertex_index >= len(vertices):
                    raise ValueError(f"Face references missing vertex at line {line_number}")
                face_indices.append(vertex_index)

            for i in range(1, len(face_indices) - 1):
                triangles.append(
                    TrianglePrimitive(
                        v0=vertices[face_indices[0]].copy(),
                        v1=vertices[face_indices[i]].copy(),
                        v2=vertices[face_indices[i + 1]].copy(),
                        material_id=material_id,
                    )
                )

    if not triangles:
        raise ValueError(f"No triangles found in OBJ file: {path}")

    return triangles


def normalize_and_place_obj(
    triangles: list[TrianglePrimitive], max_size: float, target_center_xz: tuple[float, float]
) -> None:
    all_points = np.concatenate(
        [
            np.stack([triangle.v0, triangle.v1, triangle.v2], axis=0)
            for triangle in triangles
        ],
        axis=0,
    )
    bb_min = np.min(all_points, axis=0)
    bb_max = np.max(all_points, axis=0)
    extent = bb_max - bb_min
    longest = float(np.max(extent))
    if longest <= EPS:
        raise ValueError("OBJ mesh has zero extent")

    scale = max_size / longest
    center = 0.5 * (bb_min + bb_max)

    transformed_points: list[np.ndarray] = []
    for triangle in triangles:
        triangle.v0 = (triangle.v0 - center) * scale
        triangle.v1 = (triangle.v1 - center) * scale
        triangle.v2 = (triangle.v2 - center) * scale
        transformed_points.extend([triangle.v0, triangle.v1, triangle.v2])

    transformed = np.stack(transformed_points, axis=0)
    min_y = float(np.min(transformed[:, 1]))
    tx, tz = target_center_xz
    shift = vec3(tx, 0.01 - min_y, tz)

    for triangle in triangles:
        triangle.v0 = triangle.v0 + shift
        triangle.v1 = triangle.v1 + shift
        triangle.v2 = triangle.v2 + shift


def build_manual_scene() -> tuple[Scene, Camera]:
    materials = [
        Material(kd=vec3(0.75, 0.75, 0.75), ks=vec3(0.0, 0.0, 0.0), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.75, 0.15, 0.15), ks=vec3(0.0, 0.0, 0.0), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.15, 0.75, 0.15), ks=vec3(0.0, 0.0, 0.0), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.02, 0.02, 0.02), ks=vec3(0.90, 0.90, 0.90), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.0, 0.0, 0.0), ks=vec3(0.0, 0.0, 0.0), emission=vec3(18.0, 18.0, 15.0)),
    ]

    triangles: list[TrianglePrimitive] = []

    x0, x1 = 0.0, 1.0
    y0, y1 = 0.0, 1.0
    z0, z1 = 0.0, 2.0

    add_quad(
        triangles,
        vec3(x0, y0, z0),
        vec3(x1, y0, z0),
        vec3(x1, y0, z1),
        vec3(x0, y0, z1),
        material_id=0,
        desired_normal=vec3(0.0, 1.0, 0.0),
    )
    add_quad(
        triangles,
        vec3(x0, y1, z0),
        vec3(x0, y1, z1),
        vec3(x1, y1, z1),
        vec3(x1, y1, z0),
        material_id=0,
        desired_normal=vec3(0.0, -1.0, 0.0),
    )
    add_quad(
        triangles,
        vec3(x0, y0, z1),
        vec3(x1, y0, z1),
        vec3(x1, y1, z1),
        vec3(x0, y1, z1),
        material_id=0,
        desired_normal=vec3(0.0, 0.0, -1.0),
    )
    add_quad(
        triangles,
        vec3(x0, y0, z0),
        vec3(x0, y0, z1),
        vec3(x0, y1, z1),
        vec3(x0, y1, z0),
        material_id=1,
        desired_normal=vec3(1.0, 0.0, 0.0),
    )
    add_quad(
        triangles,
        vec3(x1, y0, z0),
        vec3(x1, y1, z0),
        vec3(x1, y1, z1),
        vec3(x1, y0, z1),
        material_id=2,
        desired_normal=vec3(-1.0, 0.0, 0.0),
    )

    add_box(
        triangles,
        min_corner=vec3(0.14, 0.00, 1.25),
        max_corner=vec3(0.40, 0.36, 1.60),
        material_id=3,
    )
    add_box(
        triangles,
        min_corner=vec3(0.58, 0.00, 0.72),
        max_corner=vec3(0.86, 0.66, 1.24),
        material_id=0,
    )

    light_y = 0.995
    add_quad(
        triangles,
        vec3(0.35, light_y, 0.82),
        vec3(0.65, light_y, 0.82),
        vec3(0.65, light_y, 1.18),
        vec3(0.35, light_y, 1.18),
        material_id=4,
        desired_normal=vec3(0.0, -1.0, 0.0),
    )

    scene = build_scene(primitives=triangles, materials=materials, background=vec3(0.0, 0.0, 0.0))
    camera = Camera(
        eye=vec3(0.50, 0.53, -1.45),
        target=vec3(0.50, 0.45, 1.00),
        up=vec3(0.0, 1.0, 0.0),
        fov_degrees=40.0,
    )
    return scene, camera


def build_obj_scene(obj_path: Path) -> tuple[Scene, Camera]:
    materials = [
        Material(kd=vec3(0.75, 0.75, 0.75), ks=vec3(0.0, 0.0, 0.0), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.75, 0.15, 0.15), ks=vec3(0.0, 0.0, 0.0), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.15, 0.75, 0.15), ks=vec3(0.0, 0.0, 0.0), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.50, 0.55, 0.70), ks=vec3(0.25, 0.25, 0.20), emission=vec3(0.0, 0.0, 0.0)),
        Material(kd=vec3(0.0, 0.0, 0.0), ks=vec3(0.0, 0.0, 0.0), emission=vec3(18.0, 18.0, 15.0)),
    ]

    triangles: list[TrianglePrimitive] = []

    x0, x1 = 0.0, 1.0
    y0, y1 = 0.0, 1.0
    z0, z1 = 0.0, 2.0

    add_quad(
        triangles,
        vec3(x0, y0, z0),
        vec3(x1, y0, z0),
        vec3(x1, y0, z1),
        vec3(x0, y0, z1),
        material_id=0,
        desired_normal=vec3(0.0, 1.0, 0.0),
    )
    add_quad(
        triangles,
        vec3(x0, y1, z0),
        vec3(x0, y1, z1),
        vec3(x1, y1, z1),
        vec3(x1, y1, z0),
        material_id=0,
        desired_normal=vec3(0.0, -1.0, 0.0),
    )
    add_quad(
        triangles,
        vec3(x0, y0, z1),
        vec3(x1, y0, z1),
        vec3(x1, y1, z1),
        vec3(x0, y1, z1),
        material_id=0,
        desired_normal=vec3(0.0, 0.0, -1.0),
    )
    add_quad(
        triangles,
        vec3(x0, y0, z0),
        vec3(x0, y0, z1),
        vec3(x0, y1, z1),
        vec3(x0, y1, z0),
        material_id=1,
        desired_normal=vec3(1.0, 0.0, 0.0),
    )
    add_quad(
        triangles,
        vec3(x1, y0, z0),
        vec3(x1, y1, z0),
        vec3(x1, y1, z1),
        vec3(x1, y0, z1),
        material_id=2,
        desired_normal=vec3(-1.0, 0.0, 0.0),
    )

    mesh = load_obj(obj_path, material_id=3)
    normalize_and_place_obj(mesh, max_size=0.8, target_center_xz=(0.5, 1.0))
    triangles.extend(mesh)

    light_y = 0.995
    add_quad(
        triangles,
        vec3(0.35, light_y, 0.82),
        vec3(0.65, light_y, 0.82),
        vec3(0.65, light_y, 1.18),
        vec3(0.35, light_y, 1.18),
        material_id=4,
        desired_normal=vec3(0.0, -1.0, 0.0),
    )

    scene = build_scene(primitives=triangles, materials=materials, background=vec3(0.0, 0.0, 0.0))
    camera = Camera(
        eye=vec3(0.50, 0.53, -1.45),
        target=vec3(0.50, 0.45, 1.00),
        up=vec3(0.0, 1.0, 0.0),
        fov_degrees=40.0,
    )
    return scene, camera


def sample_cosine_hemisphere(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r1 = float(rng.random())
    r2 = float(rng.random())
    phi = 2.0 * PI * r1
    r = math.sqrt(r2)
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r2))

    if abs(float(normal[0])) > 0.1:
        tangent = normalize(np.cross(vec3(0.0, 1.0, 0.0), normal))
    else:
        tangent = normalize(np.cross(vec3(1.0, 0.0, 0.0), normal))
    bitangent = np.cross(normal, tangent)

    direction = x * tangent + y * bitangent + z * normal
    return normalize(direction)


def offset_point(point: np.ndarray, normal: np.ndarray, direction: np.ndarray) -> np.ndarray:
    sign = 1.0 if float(np.dot(direction, normal)) >= 0.0 else -1.0
    return point + sign * RAY_EPS * normal


def estimate_direct_light(scene: Scene, hit: Hit, rng: np.random.Generator) -> np.ndarray:
    if scene.light_indices.size == 0:
        return np.zeros(3, dtype=np.float64)

    if max_component(hit.material.kd) <= 0.0:
        return np.zeros(3, dtype=np.float64)

    sampled = scene.sample_light(rng)
    if sampled is None:
        return np.zeros(3, dtype=np.float64)
    _, light_point, light_normal, light_emission, pdf = sampled
    if pdf <= EPS:
        return np.zeros(3, dtype=np.float64)

    to_light = light_point - hit.position
    dist2 = float(np.dot(to_light, to_light))
    if dist2 <= EPS:
        return np.zeros(3, dtype=np.float64)

    distance = math.sqrt(dist2)
    wi = to_light / distance

    cos_surface = float(np.dot(hit.normal, wi))
    if cos_surface <= 0.0:
        return np.zeros(3, dtype=np.float64)

    cos_light = float(np.dot(light_normal, -wi))
    if cos_light <= 0.0:
        return np.zeros(3, dtype=np.float64)

    shadow_origin = offset_point(hit.position, hit.normal, wi)
    if scene.occluded(shadow_origin, wi, max_distance=distance - RAY_EPS):
        return np.zeros(3, dtype=np.float64)

    brdf = hit.material.kd / PI
    return brdf * light_emission * (cos_surface * cos_light / (dist2 * pdf))


def trace_path(ray: Ray, scene: Scene, rng: np.random.Generator, max_depth: int) -> np.ndarray:
    radiance = np.zeros(3, dtype=np.float64)
    throughput = np.ones(3, dtype=np.float64)
    previous_was_specular = True

    for depth in range(max_depth):
        hit = scene.intersect(ray)
        if hit is None:
            radiance += throughput * scene.background
            break

        material = hit.material

        if max_component(material.emission) > 0.0:
            if depth == 0 or previous_was_specular:
                radiance += throughput * material.emission
            break

        radiance += throughput * estimate_direct_light(scene, hit, rng)

        p_diffuse = max_component(material.kd)
        p_specular = max_component(material.ks)
        if p_diffuse + p_specular > 1.0:
            norm = p_diffuse + p_specular
            p_diffuse /= norm
            p_specular /= norm

        xi = float(rng.random())

        if xi < p_diffuse and p_diffuse > EPS:
            new_direction = sample_cosine_hemisphere(hit.normal, rng)
            throughput *= material.kd / p_diffuse
            previous_was_specular = False
        elif xi < p_diffuse + p_specular and p_specular > EPS:
            new_direction = normalize(reflect(ray.direction, hit.normal))
            throughput *= material.ks / p_specular
            previous_was_specular = True
        else:
            break

        if depth >= 3:
            survive = float(np.clip(max_component(throughput), 0.05, 0.95))
            if float(rng.random()) > survive:
                break
            throughput /= survive

        new_origin = offset_point(hit.position, hit.normal, new_direction)
        ray = Ray(origin=new_origin, direction=new_direction)

    return radiance


def render(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    spp: int,
    max_depth: int,
    seed: int,
    max_seconds: float | None,
    progress_every: int,
) -> tuple[np.ndarray, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image resolution must be positive")
    if spp <= 0:
        raise ValueError("Samples per pixel must be positive")
    if max_depth <= 0:
        raise ValueError("Max depth must be positive")

    rng = np.random.default_rng(seed)
    accum = np.zeros((height, width, 3), dtype=np.float64)
    start = time.perf_counter()
    completed_spp = 0
    stopped_by_time = False

    for _sample_idx in range(spp):
        sample_accum = np.zeros((height, width, 3), dtype=np.float64)
        aborted_sample = False
        for y in range(height):
            for x in range(width):
                ray = camera.generate_ray(x, y, width, height, rng)
                sample_accum[y, x] = trace_path(ray, scene, rng, max_depth)

            if max_seconds is not None and (time.perf_counter() - start) >= max_seconds:
                aborted_sample = True
                stopped_by_time = True
                break

        if aborted_sample:
            break

        accum += sample_accum

        completed_spp += 1
        elapsed = time.perf_counter() - start

        if progress_every > 0 and (
            completed_spp == 1
            or completed_spp == spp
            or completed_spp % progress_every == 0
        ):
            total_primary_rays = completed_spp * width * height
            print(
                f"[render] spp={completed_spp}/{spp}, "
                f"elapsed={elapsed:.2f}s, primary-rays={total_primary_rays}",
                flush=True,
            )

        if max_seconds is not None and elapsed >= max_seconds:
            stopped_by_time = True
            break

    if completed_spp == 0:
        raise RuntimeError("No samples were rendered")

    if stopped_by_time:
        print("[render] Time limit reached, finishing early.", flush=True)

    linear_image = accum / float(completed_spp)
    elapsed_total = time.perf_counter() - start
    return linear_image, completed_spp, elapsed_total


def tonemap_to_srgb(
    linear_image: np.ndarray,
    normalization: str,
    gamma: float,
    fixed_scale: float,
) -> tuple[np.ndarray, float]:
    if gamma <= 0.0:
        raise ValueError("Gamma must be positive")

    if normalization == "max":
        scale = float(np.max(linear_image))
        if scale <= EPS:
            scale = 1.0
    elif normalization == "mean":
        lum = (
            0.2126 * linear_image[:, :, 0]
            + 0.7152 * linear_image[:, :, 1]
            + 0.0722 * linear_image[:, :, 2]
        )
        mean_lum = float(np.mean(lum))
        target_lum = 0.5
        scale = mean_lum / target_lum if mean_lum > EPS else 1.0
    elif normalization == "fixed":
        scale = max(fixed_scale, EPS)
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    mapped = np.clip(linear_image / scale, 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / gamma)
    return mapped, scale


def save_ppm(path: Path, srgb_image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = srgb_image.shape
    if channels != 3:
        raise ValueError("Image must have exactly 3 channels")

    image_u8 = np.clip(srgb_image * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    with path.open("wb") as ppm:
        ppm.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        ppm.write(image_u8.tobytes())


def save_pfm(path: Path, linear_image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = linear_image.shape
    if channels != 3:
        raise ValueError("Image must have exactly 3 channels")

    data = np.flipud(linear_image).astype(np.float32)
    with path.open("wb") as pfm:
        pfm.write(f"PF\n{width} {height}\n-1.0\n".encode("ascii"))
        pfm.write(data.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global illumination path tracer with triangle meshes and area lights."
    )
    parser.add_argument("--scene", choices=["manual", "obj"], default="manual")
    parser.add_argument("--obj-path", type=Path, default=None, help="OBJ mesh path when --scene obj")

    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--spp", type=int, default=8, help="Samples per pixel")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output", type=Path, default=Path("renders/render.ppm"))
    parser.add_argument("--pfm-output", type=Path, default=None, help="Optional HDR-like output in PFM")
    parser.add_argument("--normalization", choices=["max", "mean", "fixed"], default="max")
    parser.add_argument("--fixed-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--progress-every", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scene == "obj" and args.obj_path is None:
        raise SystemExit("For scene='obj' provide --obj-path /path/to/model.obj")

    if args.width < 500 or args.height < 500:
        print("[warning] For the lab report, use at least 500x500 resolution.", flush=True)

    if args.max_seconds is not None and args.max_seconds <= 0.0:
        raise SystemExit("--max-seconds must be positive")

    if args.scene == "manual":
        scene, camera = build_manual_scene()
    else:
        scene, camera = build_obj_scene(args.obj_path)

    print(
        f"[config] scene={args.scene}, resolution={args.width}x{args.height}, "
        f"spp={args.spp}, depth={args.max_depth}",
        flush=True,
    )
    if scene.light_indices.size == 0:
        print("[warning] Scene has no emissive triangles.", flush=True)

    linear_image, completed_spp, elapsed = render(
        scene=scene,
        camera=camera,
        width=args.width,
        height=args.height,
        spp=args.spp,
        max_depth=args.max_depth,
        seed=args.seed,
        max_seconds=args.max_seconds,
        progress_every=args.progress_every,
    )

    srgb_image, scale = tonemap_to_srgb(
        linear_image=linear_image,
        normalization=args.normalization,
        gamma=args.gamma,
        fixed_scale=args.fixed_scale,
    )

    save_ppm(args.output, srgb_image)
    if args.pfm_output is not None:
        save_pfm(args.pfm_output, linear_image)

    print(f"[done] rendered in {elapsed:.2f}s with effective spp={completed_spp}", flush=True)
    print(f"[done] normalization='{args.normalization}', scale={scale:.6f}", flush=True)
    print(f"[done] LDR output: {args.output}", flush=True)
    if args.pfm_output is not None:
        print(f"[done] HDR output: {args.pfm_output}", flush=True)


if __name__ == "__main__":
    main()