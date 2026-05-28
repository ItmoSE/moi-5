#!/usr/bin/env python3
"""
ЛР5. Фильтрация синтезированного изображения с использованием AOV-буферов.

Ключевые отличия синтезированного изображения от обычного:
1) Шум стохастический (Монте-Карло), а не только сенсорный.
2) Доступны дополнительные буферы сцены (depth, normal, object_id), поэтому можно
   делать edge-aware фильтрацию с сохранением границ объектов.
3) После приведения HDR -> 0..255 возможны: потеря динамического диапазона,
   клиппинг светов и потеря слабых деталей в тенях.
"""

from __future__ import annotations

import argparse
import math
from concurrent import futures
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


EPS = 1e-12


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def save_ppm(path: Path, rgb8: np.ndarray) -> None:
    h, w, _ = rgb8.shape
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())


def tonemap_and_gamma(hdr: np.ndarray, normalize_mode: str = "p99", gamma: float = 2.2) -> Tuple[np.ndarray, float]:
    img = hdr.copy()
    if normalize_mode == "max":
        m = float(np.max(img))
        if m > EPS:
            img /= m
    elif normalize_mode == "mean05":
        m = float(np.mean(img))
        if m > EPS:
            img *= 0.5 / m
    elif normalize_mode == "p99":
        p = float(np.percentile(img, 99.0))
        if p > EPS:
            img /= p
    else:
        raise ValueError("normalize_mode должен быть max, mean05 или p99.")

    clipped_ratio = float(np.mean((img < 0.0) | (img > 1.0)))
    img = clamp01(img)
    img = np.power(img, 1.0 / gamma)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8), clipped_ratio


def load_aov(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"direct", "secondary", "depth", "object_id", "normal"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"В NPZ отсутствуют ключи: {missing}. Нужен файл AOV из LR4.")
        direct = data["direct"].astype(np.float64)
        secondary = data["secondary"].astype(np.float64)
        hdr = data["hdr"].astype(np.float64) if "hdr" in data.files else direct + secondary
        depth = data["depth"].astype(np.float64)
        object_id = data["object_id"].astype(np.int32)
        normal = data["normal"].astype(np.float64)
    return {
        "hdr": hdr,
        "direct": direct,
        "secondary": secondary,
        "depth": depth,
        "object_id": object_id,
        "normal": normal,
    }


def make_spatial_kernel(radius: int, sigma_spatial: float) -> np.ndarray:
    side = 2 * radius + 1
    kernel = np.zeros((side, side), dtype=np.float64)
    denom = 2.0 * sigma_spatial * sigma_spatial
    for j in range(-radius, radius + 1):
        for i in range(-radius, radius + 1):
            kernel[j + radius, i + radius] = math.exp(-(i * i + j * j) / max(EPS, denom))
    return kernel


def _gaussian_row(y: int, img: np.ndarray, kernel: np.ndarray, radius: int) -> Tuple[int, np.ndarray]:
    h, w, _ = img.shape
    out_row = np.zeros((w, 3), dtype=np.float64)
    for x in range(w):
        acc = np.zeros(3, dtype=np.float64)
        wsum = 0.0
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        for yy in range(y0, y1):
            ky = yy - y + radius
            for xx in range(x0, x1):
                kx = xx - x + radius
                wgt = kernel[ky, kx]
                acc += wgt * img[yy, xx]
                wsum += wgt
        out_row[x] = img[y, x] if wsum <= EPS else acc / wsum
    return y, out_row


def gaussian_blur_rgb(img: np.ndarray, radius: int, sigma_spatial: float, workers: int = 1) -> np.ndarray:
    h, _, _ = img.shape
    kernel = make_spatial_kernel(radius, sigma_spatial)
    out = np.zeros_like(img)
    workers = max(1, workers)

    if workers == 1:
        for y in range(h):
            _, row = _gaussian_row(y, img, kernel, radius)
            out[y] = row
        return out

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_gaussian_row, y, img, kernel, radius) for y in range(h)]
        for job in futures.as_completed(jobs):
            y, row = job.result()
            out[y] = row
    return out


def _filter_row(
    y: int,
    img: np.ndarray,
    spatial: np.ndarray,
    radius: int,
    sigma_color: float,
    depth: np.ndarray | None,
    sigma_depth: float | None,
    normal: np.ndarray | None,
    sigma_normal: float | None,
    object_id: np.ndarray | None,
) -> Tuple[int, np.ndarray]:
    h, w, _ = img.shape
    out_row = np.zeros((w, 3), dtype=np.float64)
    color_denom = 2.0 * sigma_color * sigma_color
    depth_denom = None if sigma_depth is None else 2.0 * sigma_depth * sigma_depth
    normal_denom = None if sigma_normal is None else 2.0 * sigma_normal * sigma_normal

    for x in range(w):
        c0 = img[y, x]
        d0 = None if depth is None else depth[y, x]
        n0 = None if normal is None else normal[y, x]
        o0 = None if object_id is None else int(object_id[y, x])

        acc = np.zeros(3, dtype=np.float64)
        wsum = 0.0

        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)

        for yy in range(y0, y1):
            ky = yy - y + radius
            for xx in range(x0, x1):
                if object_id is not None and o0 >= 0 and int(object_id[yy, xx]) != o0:
                    continue

                kx = xx - x + radius
                wgt = spatial[ky, kx]

                dc = img[yy, xx] - c0
                dc2 = float(np.dot(dc, dc))
                wgt *= math.exp(-dc2 / max(EPS, color_denom))

                if depth is not None and depth_denom is not None and d0 > 0.0 and depth[yy, xx] > 0.0:
                    dd = float(depth[yy, xx] - d0)
                    wgt *= math.exp(-(dd * dd) / max(EPS, depth_denom))

                if normal is not None and normal_denom is not None:
                    n1 = normal[yy, xx]
                    n0n = float(np.linalg.norm(n0)) if n0 is not None else 0.0
                    n1n = float(np.linalg.norm(n1))
                    if n0n > EPS and n1n > EPS:
                        nd = float(np.dot(n0 / n0n, n1 / n1n))
                        nd = max(-1.0, min(1.0, nd))
                        ang = 1.0 - nd
                        wgt *= math.exp(-(ang * ang) / max(EPS, normal_denom))

                acc += wgt * img[yy, xx]
                wsum += wgt

        if wsum <= EPS:
            out_row[x] = c0
        else:
            out_row[x] = acc / wsum

    return y, out_row


def edge_aware_filter(
    img: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
    *,
    depth: np.ndarray | None = None,
    sigma_depth: float | None = None,
    normal: np.ndarray | None = None,
    sigma_normal: float | None = None,
    object_id: np.ndarray | None = None,
    workers: int = 1,
) -> np.ndarray:
    h, _, _ = img.shape
    spatial = make_spatial_kernel(radius, sigma_spatial)
    out = np.zeros_like(img)
    workers = max(1, workers)

    if workers == 1:
        for y in range(h):
            _, row = _filter_row(
                y, img, spatial, radius, sigma_color, depth, sigma_depth, normal, sigma_normal, object_id
            )
            out[y] = row
        return out

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(
                _filter_row,
                y,
                img,
                spatial,
                radius,
                sigma_color,
                depth,
                sigma_depth,
                normal,
                sigma_normal,
                object_id,
            )
            for y in range(h)
        ]
        for job in futures.as_completed(jobs):
            y, row = job.result()
            out[y] = row
    return out


def _median_row(y: int, img: np.ndarray, object_id: np.ndarray, radius: int) -> Tuple[int, np.ndarray]:
    h, w, _ = img.shape
    out_row = np.zeros((w, 3), dtype=np.float64)
    for x in range(w):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        patch = img[y0:y1, x0:x1]
        patch_obj = object_id[y0:y1, x0:x1]
        center_obj = int(object_id[y, x])

        if center_obj >= 0:
            vals = patch[patch_obj == center_obj]
        else:
            vals = patch.reshape(-1, 3)

        if vals.size == 0:
            out_row[x] = img[y, x]
        else:
            out_row[x] = np.median(vals, axis=0)
    return y, out_row


def object_aware_median_filter(img: np.ndarray, object_id: np.ndarray, radius: int, workers: int = 1) -> np.ndarray:
    h, _, _ = img.shape
    out = np.zeros_like(img)
    workers = max(1, workers)

    if workers == 1:
        for y in range(h):
            _, row = _median_row(y, img, object_id, radius)
            out[y] = row
        return out

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_median_row, y, img, object_id, radius) for y in range(h)]
        for job in futures.as_completed(jobs):
            y, row = job.result()
            out[y] = row
    return out


def normalize_median_per_object(filtered: np.ndarray, original: np.ndarray, object_id: np.ndarray) -> np.ndarray:
    # Нормировка медианного результата: для каждого объекта сохраняем суммарную энергию.
    out = filtered.copy()
    unique_ids = np.unique(object_id)
    for oid in unique_ids:
        mask = object_id == oid
        if not np.any(mask):
            continue
        src_sum = np.sum(original[mask], axis=0)
        dst_sum = np.sum(out[mask], axis=0)
        scale = np.ones(3, dtype=np.float64)
        nz = dst_sum > EPS
        scale[nz] = src_sum[nz] / dst_sum[nz]
        out[mask] *= scale
    return out


def run_pipeline(args: argparse.Namespace) -> None:
    inp = Path(args.input_aov)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aov = load_aov(inp)
    direct = aov["direct"]
    secondary = aov["secondary"]
    depth = aov["depth"]
    object_id = aov["object_id"]
    normal = aov["normal"]

    gaussian_secondary = gaussian_blur_rgb(
        secondary,
        radius=args.radius,
        sigma_spatial=args.sigma_spatial,
        workers=args.workers,
    )

    bilateral_secondary = edge_aware_filter(
        secondary,
        radius=args.radius,
        sigma_spatial=args.sigma_spatial,
        sigma_color=args.sigma_color,
        object_id=object_id,
        workers=args.workers,
    )

    multilateral_secondary = edge_aware_filter(
        secondary,
        radius=args.radius,
        sigma_spatial=args.sigma_spatial,
        sigma_color=args.sigma_color,
        depth=depth,
        sigma_depth=args.sigma_depth,
        normal=normal,
        sigma_normal=args.sigma_normal,
        object_id=object_id,
        workers=args.workers,
    )

    median_secondary = object_aware_median_filter(
        secondary,
        object_id=object_id,
        radius=args.radius,
        workers=args.workers,
    )
    median_secondary = normalize_median_per_object(median_secondary, secondary, object_id)

    results = {
        "raw": direct + secondary,
        "gaussian": direct + gaussian_secondary,
        "bilateral": direct + bilateral_secondary,
        "multilateral": direct + multilateral_secondary,
        "median": direct + median_secondary,
    }

    stem = inp.stem
    for name, hdr in results.items():
        np.save(out_dir / f"{stem}_{name}.npy", hdr)
        rgb8, clipped = tonemap_and_gamma(hdr, normalize_mode=args.normalize, gamma=args.gamma)
        save_ppm(out_dir / f"{stem}_{name}.ppm", rgb8)
        print(f"[{name}] clip_ratio_before_clamp={clipped:.4f}")

    np.savez_compressed(
        out_dir / f"{stem}_lr5_bundle.npz",
        **results,
        direct=direct,
        secondary=secondary,
        depth=depth,
        object_id=object_id,
        normal=normal,
    )

    print("[done] Сохранены .npy/.ppm и общий bundle NPZ.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ЛР5: gaussian/bilateral/multilateral/median фильтрация синтезированного изображения.")
    parser.add_argument("--input-aov", required=True, help="Путь к NPZ-файлу AOV, сохраненному из LR4.")
    parser.add_argument("--out-dir", default="renders/lr5", help="Каталог для результатов.")
    parser.add_argument("--radius", type=int, default=3, help="Радиус окна фильтра.")
    parser.add_argument("--sigma-spatial", type=float, default=2.0, help="Sigma по пространству.")
    parser.add_argument("--sigma-color", type=float, default=0.2, help="Sigma по цвету.")
    parser.add_argument("--sigma-depth", type=float, default=0.2, help="Sigma по глубине (multilateral).")
    parser.add_argument("--sigma-normal", type=float, default=0.2, help="Sigma по нормали (multilateral).")
    parser.add_argument("--normalize", choices=["max", "mean05", "p99"], default="p99", help="Режим HDR->LDR.")
    parser.add_argument("--gamma", type=float, default=2.2, help="Гамма для вывода.")
    parser.add_argument("--workers", type=int, default=1, help="Число потоков для bilateral/multilateral/median.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.radius < 1:
        raise ValueError("radius должен быть >= 1")
    if args.sigma_spatial <= 0.0 or args.sigma_color <= 0.0:
        raise ValueError("sigma_spatial и sigma_color должны быть положительными")
    if args.sigma_depth <= 0.0 or args.sigma_normal <= 0.0:
        raise ValueError("sigma_depth и sigma_normal должны быть положительными")
    if args.gamma <= 0.0:
        raise ValueError("gamma должна быть положительной")
    if args.workers <= 0:
        raise ValueError("workers должен быть положительным")
    run_pipeline(args)


if __name__ == "__main__":
    main()
