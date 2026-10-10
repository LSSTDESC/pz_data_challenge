#!/usr/bin/env python3
"""RAIL-style evaluation for Simple MLP conditional rectified flow outputs.

Expected model output:
  - NPZ with arrays:
      z_true: shape (n,)
      samples: shape (n, n_samples), posterior redshift samples
      mag_r or mag_i: optional shape (n,)
  - or CSV with columns:
      z_true, optional mag_r/mag_i, and sample columns named sample_0, sample_1, ...

This script computes overall metrics, redshift-binned metrics, magnitude-binned
metrics, PIT / Delta-Q curves, and lightweight SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np


METRICS = ["bias", "scatter", "outlier_rate", "std_delta_z", "mse", "energy_score"]


def read_input(path: Path, sample_prefix: str):
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        z_true = np.asarray(data["z_true"], dtype=float)
        samples = np.asarray(data["samples"], dtype=float)
        mag_r = np.asarray(data["mag_r"], dtype=float) if "mag_r" in data else None
        mag_i = np.asarray(data["mag_i"], dtype=float) if "mag_i" in data else None
        return z_true, samples, mag_r, mag_i

    if path.suffix.lower() != ".csv":
        raise ValueError("Input must be .npz or .csv")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        sample_cols = [c for c in reader.fieldnames or [] if c.startswith(sample_prefix)]
        if "z_true" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a z_true column")
        if not sample_cols:
            raise ValueError(f"CSV must contain posterior sample columns like {sample_prefix}0")

        z_true, mag_r, mag_i, samples = [], [], [], []
        has_mag_r = "mag_r" in reader.fieldnames
        has_mag_i = "mag_i" in reader.fieldnames
        for row in reader:
            z_true.append(float(row["z_true"]))
            if has_mag_r:
                mag_r.append(float(row["mag_r"]))
            if has_mag_i:
                mag_i.append(float(row["mag_i"]))
            samples.append([float(row[c]) for c in sample_cols])

    return (
        np.asarray(z_true, dtype=float),
        np.asarray(samples, dtype=float),
        np.asarray(mag_r, dtype=float) if mag_r else None,
        np.asarray(mag_i, dtype=float) if mag_i else None,
    )


def point_prediction(samples: np.ndarray, mode: str) -> np.ndarray:
    if mode == "median":
        return np.median(samples, axis=1)
    if mode == "mean":
        return np.mean(samples, axis=1)
    if mode == "mode":
        return sample_modes(samples)
    raise ValueError("point mode must be mean, median, or mode")


def sample_modes(samples: np.ndarray, n_bins: int = 64) -> np.ndarray:
    modes = np.empty(samples.shape[0], dtype=float)
    for i, row in enumerate(samples):
        counts, edges = np.histogram(row, bins=n_bins)
        j = int(np.argmax(counts))
        modes[i] = 0.5 * (edges[j] + edges[j + 1])
    return modes


def pairwise_abs_mean_1d(samples: np.ndarray) -> np.ndarray:
    """Mean |X-X'| per row using sorted samples, no O(m^2) allocation."""
    sorted_samples = np.sort(samples, axis=1)
    m = sorted_samples.shape[1]
    weights = 2 * np.arange(1, m + 1) - m - 1
    return 2.0 * np.sum(sorted_samples * weights, axis=1) / (m * m)


def compute_metrics(z_true: np.ndarray, samples: np.ndarray, point_mode: str) -> dict[str, float]:
    z_pred = point_prediction(samples, point_mode)
    dz = (z_pred - z_true) / (1.0 + z_true)
    abs_centered = np.abs(dz - np.median(dz))
    sigma_nmad = float(1.48 * np.median(abs_centered))
    es = np.mean(np.abs(samples - z_true[:, None]), axis=1) - 0.5 * pairwise_abs_mean_1d(samples)
    return {
        "n": float(len(z_true)),
        "bias": float(np.mean(dz)),
        "scatter": sigma_nmad,
        "sigma_nmad": sigma_nmad,
        "std_delta_z": float(np.std(dz)),
        "outlier_rate": float(np.mean(np.abs(dz) > 0.15)),
        "mse": float(np.mean((z_pred - z_true) ** 2)),
        "energy_score": float(np.mean(es)),
    }


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def fixed_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return np.linspace(lo, hi, n_bins + 1)


def binned_metrics(
    bin_values: np.ndarray,
    z_true: np.ndarray,
    samples: np.ndarray,
    n_bins: int,
    point_mode: str,
    strategy: str,
    drop_first: bool,
    max_value: float | None,
) -> list[dict[str, float | str]]:
    mask = np.isfinite(bin_values) & np.isfinite(z_true) & np.all(np.isfinite(samples), axis=1)
    if max_value is not None:
        mask &= bin_values <= max_value

    bin_values = bin_values[mask]
    z_true = z_true[mask]
    samples = samples[mask]
    edges = quantile_bins(bin_values, n_bins) if strategy == "quantile" else fixed_bins(bin_values, n_bins)

    rows = []
    start = 1 if drop_first else 0
    for i in range(start, len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        in_bin = (bin_values >= left) & (bin_values < right)
        if i == len(edges) - 2:
            in_bin = (bin_values >= left) & (bin_values <= right)
        if not np.any(in_bin):
            continue
        metrics = compute_metrics(z_true[in_bin], samples[in_bin], point_mode)
        center = float(np.mean(bin_values[in_bin]))
        row: dict[str, float | str] = {
            "bin_index": i,
            "bin_left": float(left) if np.isfinite(left) else float(np.min(bin_values[in_bin])),
            "bin_right": float(right) if np.isfinite(right) else float(np.max(bin_values[in_bin])),
            "bin_center": center,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def pit_curve(z_true: np.ndarray, samples: np.ndarray, n_grid: int) -> tuple[list[dict[str, float]], np.ndarray]:
    pit = np.mean(samples <= z_true[:, None], axis=1)
    qs = np.linspace(0.0, 1.0, n_grid)
    rows = []
    for q in qs:
        empirical = float(np.mean(pit <= q))
        rows.append({"q_th": float(q), "empirical_cdf": empirical, "delta_q": empirical - float(q)})
    return rows, pit


def pit_histogram_rows(pit: np.ndarray, n_bins: int) -> list[dict[str, float]]:
    counts, edges = np.histogram(pit, bins=np.linspace(0.0, 1.0, n_bins + 1))
    widths = np.diff(edges)
    density = counts / np.sum(counts) / widths
    rows = []
    for i in range(n_bins):
        rows.append(
            {
                "bin_index": float(i),
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "bin_center": float(0.5 * (edges[i] + edges[i + 1])),
                "count": float(counts[i]),
                "density": float(density[i]),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_summary(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def svg_polyline(points, width, height, margin):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_line_svg(
    path: Path,
    rows: list[dict[str, float | str]],
    x_key: str,
    y_keys: list[str],
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
    grey_band: tuple[float, float] | None = None,
) -> None:
    if not rows:
        return
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 78, 36, 64, 72
    x = np.asarray([float(r[x_key]) for r in rows])
    ys = {k: np.asarray([float(r[k]) for r in rows]) for k in y_keys}
    y_all = np.concatenate(list(ys.values()))
    if grey_band is not None:
        y_all = np.concatenate([y_all, np.asarray(grey_band)])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def sx(v):
        return margin_left + (v - x_min) / (x_max - x_min or 1.0) * (width - margin_left - margin_right)

    def sy(v):
        return height - margin_bottom - (v - y_min) / (y_max - y_min or 1.0) * (height - margin_top - margin_bottom)

    def fmt_tick(v):
        if abs(v) >= 10:
            return f"{v:.0f}"
        if abs(v) >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial" font-size="22" fill="#222">{title}</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#222"/>',
    ]
    for i in range(6):
        xv = x_min + i * (x_max - x_min) / 5
        xp = sx(xv)
        pieces.append(f'<line x1="{xp:.2f}" y1="{margin_top}" x2="{xp:.2f}" y2="{height-margin_bottom}" stroke="#e5e5e5" stroke-width="1"/>')
        pieces.append(f'<text x="{xp:.2f}" y="{height-margin_bottom+20}" text-anchor="middle" font-family="Arial" font-size="12" fill="#444">{fmt_tick(xv)}</text>')
    for i in range(6):
        yv = y_min + i * (y_max - y_min) / 5
        yp = sy(yv)
        pieces.append(f'<line x1="{margin_left}" y1="{yp:.2f}" x2="{width-margin_right}" y2="{yp:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        pieces.append(f'<text x="{margin_left-8}" y="{yp+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#444">{fmt_tick(yv)}</text>')

    if grey_band is not None:
        y1, y2 = sy(grey_band[0]), sy(grey_band[1])
        pieces.append(
            f'<rect x="{margin_left}" y="{min(y1,y2):.2f}" width="{width-margin_left-margin_right}" height="{abs(y2-y1):.2f}" fill="#d9d9d9" opacity="0.45"/>'
        )
    for idx, key in enumerate(y_keys):
        pts = [(sx(v), sy(y)) for v, y in zip(x, ys[key])]
        color = colors[idx % len(colors)]
        pieces.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{svg_polyline(pts, width, height, margin_left)}"/>')
        lx, ly = width - margin_right - 180, margin_top + 24 + idx * 22
        pieces.append(f'<line x1="{lx}" y1="{ly-5}" x2="{lx+28}" y2="{ly-5}" stroke="{color}" stroke-width="2.5"/>')
        pieces.append(f'<text x="{lx+36}" y="{ly}" font-family="Arial" font-size="14" fill="#222">{key}</text>')
    x_label = x_label or x_key
    y_label = y_label or (y_keys[0] if len(y_keys) == 1 else "value")
    pieces.append(f'<text x="{width/2:.2f}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="14" fill="#222">{x_label}</text>')
    pieces.append(
        f'<text x="18" y="{height/2:.2f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.2f})" font-family="Arial" font-size="14" fill="#222">{y_label}</text>'
    )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_pit_svg(path: Path, rows: list[dict[str, float]]) -> None:
    write_line_svg(
        path,
        rows,
        "q_th",
        ["delta_q"],
        "PIT / Delta Q vs Qth",
        x_label="Q_th",
        y_label="Delta Q",
        grey_band=(-0.02, 0.02),
    )


def write_pit_histogram_svg(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    width, height, margin = 900, 520, 70
    x_min, x_max = 0.0, 1.0
    y_vals = [float(r["density"]) for r in rows] + [1.0]
    y_min, y_max = 0.0, max(y_vals) * 1.12

    def sx(v):
        return margin + (v - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(v):
        return height - margin - (v - y_min) / (y_max - y_min or 1.0) * (height - 2 * margin)

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="34" font-family="Arial" font-size="22" fill="#222">PIT histogram</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
    ]
    # reference uniform density
    uy = sy(1.0)
    pieces.append(f'<line x1="{margin}" y1="{uy:.2f}" x2="{width-margin}" y2="{uy:.2f}" stroke="#cc2222" stroke-width="2" stroke-dasharray="7 4"/>')
    pieces.append(f'<text x="{width-margin-4}" y="{uy-8:.2f}" text-anchor="end" font-family="Arial" font-size="13" fill="#cc2222">Uniform density = 1</text>')

    for i in range(6):
        xv = x_min + i * (x_max - x_min) / 5
        x = sx(xv)
        pieces.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" y2="{height-margin}" stroke="#e3e3e3" stroke-width="1"/>')
        pieces.append(f'<text x="{x:.2f}" y="{height-margin+20}" text-anchor="middle" font-family="Arial" font-size="12" fill="#444">{xv:.1f}</text>')
    for i in range(6):
        yv = y_min + i * (y_max - y_min) / 5
        y = sy(yv)
        pieces.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" stroke="#e3e3e3" stroke-width="1"/>')
        pieces.append(f'<text x="{margin-8}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#444">{yv:.2f}</text>')

    for row in rows:
        x0 = sx(float(row["bin_left"])) + 1
        x1 = sx(float(row["bin_right"])) - 1
        y = sy(float(row["density"]))
        pieces.append(f'<rect x="{x0:.2f}" y="{y:.2f}" width="{max(1.0, x1-x0):.2f}" height="{height-margin-y:.2f}" fill="#4c78a8" opacity="0.85"/>')

    pieces.append(f'<text x="{width/2:.2f}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="14" fill="#222">PIT value</text>')
    pieces.append(
        f'<text x="18" y="{height/2:.2f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.2f})" font-family="Arial" font-size="14" fill="#222">Density</text>'
    )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_metric_svgs(outdir: Path, prefix: str, rows: list[dict[str, float | str]], x_key: str, x_label: str) -> None:
    specs = [
        ("bias", "Bias E[Delta z]", "E[Delta z]", (-0.003, 0.003)),
        ("scatter", "Scatter sigma_NMAD", "sigma_NMAD", None),
        ("outlier_rate", "Outlier rate eta_0.15", "eta_0.15", None),
        ("energy_score", "Energy Score", "Energy Score", None),
        ("mse", "MSE", "MSE", None),
    ]
    for metric, title, y_label, band in specs:
        write_line_svg(
            outdir / f"{prefix}_{metric}.svg",
            rows,
            x_key,
            [metric],
            f"{title} vs {x_label}",
            x_label=x_label,
            y_label=y_label,
            grey_band=band,
        )


def write_checklist(path: Path, args, has_mag: str | None) -> None:
    mag_line = has_mag if has_mag else "missing: provide mag_i or mag_r for magnitude dependence"
    text = f"""# Simple MLP Flow RAIL-style evaluation outputs

Model scope: Simple MLP + conditional rectified flow matching only.

Input used: `{args.input}`

Generated files:

- `overall_metrics.csv`: overall bias, scatter, outlier rate, sigma NMAD, MSE, Energy Score.
- `redshift_binned_metrics.csv`: redshift dependence of metrics.
- `redshift_binned_metrics.svg`: quick visual check for redshift dependence.
- `magnitude_binned_metrics.csv`: magnitude dependence of metrics, if magnitude is available.
- `magnitude_binned_metrics.svg`: quick visual check for magnitude dependence, if available.
- `pit_curve.csv`: PIT empirical CDF and Delta Q vs Qth.
- `pit_curve.svg`: PIT / calibration plot.
- `pit_histogram.csv`: binned PIT densities for calibration inspection.
- `pit_histogram.svg`: PIT histogram with uniform-density reference line.
- `pit_values.csv`: one PIT value per object.

RAIL-style formatting choices:

- Redshift binning variable: `z_true`.
- Drop lowest redshift bin: `{args.drop_first_z_bin}`.
- Max redshift cutoff: `{args.z_max if args.z_max is not None else "not set"}`.
- Magnitude variable: `{mag_line}`.
- Point prediction for point metrics: posterior `{args.point}`.
- Outlier threshold: `abs((z_pred - z_true)/(1 + z_true)) > 0.15`.

Still needed for FZB/DNF/BPZ comparison:

- Numeric curves or digitized values from the RAIL paper for the same metrics/bins.
- Once those are available, overlay them against the CSV files generated here.

Interpretation prompts for the next meeting:

1. Does Simple MLP Flow have low bias across redshift, or does bias grow at high z?
2. Does scatter / sigma NMAD increase for dimmer galaxies?
3. Does outlier rate rise in high-z or faint bins?
4. Is the PIT Delta-Q curve close to zero, or does it show over/under-confidence?
5. Does the PIT histogram look uniform, U-shaped, hump-shaped, or skewed?
6. Are the trends closer to FZB, DNF, or BPZ once external RAIL curves are overlaid?
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .npz or .csv model output")
    parser.add_argument("--outdir", default="simple_mlp_flow_eval_outputs")
    parser.add_argument("--sample-prefix", default="sample_")
    parser.add_argument("--point", choices=["mean", "median", "mode"], default="mode")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--binning", choices=["fixed", "quantile"], default="fixed")
    parser.add_argument("--drop-first-z-bin", action="store_true")
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--mag", choices=["auto", "mag_i", "mag_r"], default="auto")
    parser.add_argument("--pit-grid", type=int, default=101)
    parser.add_argument("--pit-bins", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z_true, samples, mag_r, mag_i = read_input(input_path, args.sample_prefix)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_objects, n_posterior_samples)")
    if len(z_true) != samples.shape[0]:
        raise ValueError("z_true length must match samples rows")

    overall = compute_metrics(z_true, samples, args.point)
    write_metric_summary(outdir / "overall_metrics.csv", overall)

    z_rows = binned_metrics(
        z_true,
        z_true,
        samples,
        args.bins,
        args.point,
        args.binning,
        args.drop_first_z_bin,
        args.z_max,
    )
    write_rows(outdir / "redshift_binned_metrics.csv", z_rows)
    write_line_svg(
        outdir / "redshift_binned_metrics.svg",
        z_rows,
        "bin_center",
        METRICS[:4],
        "Redshift dependence",
        x_label="Redshift",
        y_label="Metric value",
    )
    write_metric_svgs(outdir, "redshift", z_rows, "bin_center", "redshift")

    mag_values = None
    mag_name = None
    if args.mag == "mag_i" and mag_i is not None:
        mag_values, mag_name = mag_i, "mag_i"
    elif args.mag == "mag_r" and mag_r is not None:
        mag_values, mag_name = mag_r, "mag_r"
    elif args.mag == "auto":
        if mag_i is not None:
            mag_values, mag_name = mag_i, "mag_i"
        elif mag_r is not None:
            mag_values, mag_name = mag_r, "mag_r"

    if mag_values is not None:
        mag_rows = binned_metrics(mag_values, z_true, samples, args.bins, args.point, args.binning, False, None)
        write_rows(outdir / "magnitude_binned_metrics.csv", mag_rows)
        write_line_svg(
            outdir / "magnitude_binned_metrics.svg",
            mag_rows,
            "bin_center",
            METRICS[:4],
            f"{mag_name} dependence",
            x_label=mag_name,
            y_label="Metric value",
        )
        write_metric_svgs(outdir, "magnitude", mag_rows, "bin_center", mag_name)
    else:
        (outdir / "magnitude_binned_metrics.csv").write_text("", encoding="utf-8")

    pit_rows, pit = pit_curve(z_true, samples, args.pit_grid)
    write_rows(outdir / "pit_curve.csv", pit_rows)
    write_pit_svg(outdir / "pit_curve.svg", pit_rows)
    pit_hist_rows = pit_histogram_rows(pit, args.pit_bins)
    write_rows(outdir / "pit_histogram.csv", pit_hist_rows)
    write_pit_histogram_svg(outdir / "pit_histogram.svg", pit_hist_rows)
    write_rows(outdir / "pit_values.csv", [{"pit": float(v)} for v in pit])

    write_checklist(outdir / "README_outputs.md", args, mag_name)


if __name__ == "__main__":
    main()
