#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OVERALL_KEYS = ["bias", "scatter", "outlier_rate", "mse", "energy_score"]
LINE_METRICS = [
    ("bias", "Bias E[Delta z]", "E[Delta z]", True),
    ("scatter", "Scatter sigma_NMAD", "sigma_NMAD", False),
    ("outlier_rate", "Outlier rate eta_0.15", "eta_0.15", False),
]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_tick(v: float) -> str:
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if hi == lo:
        hi = lo + 1.0
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def parse_run_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"Invalid --run value: {spec}. Use label=/path/to/eval_dir")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser()
    if not label:
        raise SystemExit(f"Invalid empty label in --run {spec}")
    return label, path


def read_overall(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = float(row["value"])
    return out


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def read_pit_values(path: Path) -> list[float]:
    vals: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            vals.append(float(row["pit"]))
    return vals


def pit_hist_rows_from_values(values: list[float], n_bins: int = 20) -> list[dict[str, float]]:
    counts = [0] * n_bins
    for v in values:
        if v <= 0:
            idx = 0
        elif v >= 1:
            idx = n_bins - 1
        else:
            idx = min(n_bins - 1, int(v * n_bins))
        counts[idx] += 1
    total = max(1, len(values))
    rows: list[dict[str, float]] = []
    for i, count in enumerate(counts):
        left = i / n_bins
        right = (i + 1) / n_bins
        width = right - left
        density = count / total / width
        rows.append(
            {
                "bin_index": float(i),
                "bin_left": left,
                "bin_right": right,
                "bin_center": 0.5 * (left + right),
                "count": float(count),
                "density": float(density),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_overall_table(path: Path, overall_by_label: dict[str, dict[str, float]]) -> None:
    rows = []
    for label, metrics in overall_by_label.items():
        row: dict[str, str | float] = {"method": label}
        for key in OVERALL_KEYS:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)
    write_csv(path, rows)


def write_overall_markdown(path: Path, overall_by_label: dict[str, dict[str, float]]) -> None:
    labels = list(overall_by_label.keys())
    lines = [
        "# Simple MLP comparison table",
        "",
        "| Method | Bias | Scatter | Outlier rate | MSE | Energy score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        m = overall_by_label[label]
        lines.append(
            f"| {label} | {m.get('bias', float('nan')):.5f} | {m.get('scatter', float('nan')):.5f} | "
            f"{m.get('outlier_rate', float('nan')):.5f} | {m.get('mse', float('nan')):.5f} | {m.get('energy_score', float('nan')):.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def panel_line_overlay_svg(
    rows_by_label: dict[str, list[dict[str, float]]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    grey_band: tuple[float, float] | None = None,
    zero_line: bool = False,
    width: int = 520,
    height: int = 360,
    left: int = 78,
    right: int = 24,
    top: int = 44,
    bottom: int = 64,
) -> str:
    labels = list(rows_by_label.keys())
    x_vals = [r[x_key] for label in labels for r in rows_by_label[label]]
    y_vals = [r[y_key] for label in labels for r in rows_by_label[label]]
    if not x_vals or not y_vals:
        raise SystemExit(f"No rows available for {title}")

    x_min = min(x_vals) if xlim is None else xlim[0]
    x_max = max(x_vals) if xlim is None else xlim[1]
    if ylim is None:
        y_min = min(y_vals)
        y_max = max(y_vals)
        if zero_line:
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
        if math.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5
        pad = 0.14 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = ylim

    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="600">{esc(title)}</text>',
    ]

    if grey_band is not None:
        gy0 = sy(grey_band[0])
        gy1 = sy(grey_band[1])
        parts.append(
            f'<rect x="{left}" y="{min(gy0, gy1):.1f}" width="{plot_w}" height="{abs(gy1-gy0):.1f}" fill="#d9d9d9" opacity="0.35"/>'
        )

    for xv in ticks(x_min, x_max):
        x = sx(xv)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333">{fmt_tick(xv)}</text>')
    for yv in ticks(y_min, y_max):
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333">{fmt_tick(yv)}</text>')

    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111" stroke-width="1.4"/>')

    if zero_line and y_min <= 0.0 <= y_max:
        zy = sy(0.0)
        parts.append(f'<line x1="{left}" y1="{zy:.1f}" x2="{width-right}" y2="{zy:.1f}" stroke="#cc2222" stroke-width="1.6" stroke-dasharray="6 4"/>')

    for idx, label in enumerate(labels):
        color = COLORS[idx % len(COLORS)]
        pts = [(sx(r[x_key]), sy(r[y_key])) for r in rows_by_label[label] if x_min <= r[x_key] <= x_max]
        parts.append(
            '<polyline fill="none" stroke="{color}" stroke-width="2.6" points="{points}"/>'.format(
                color=color,
                points=" ".join(f"{x:.1f},{y:.1f}" for x, y in pts),
            )
        )
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{color}"/>')
        lx = width - right - 148
        ly = top + 18 + idx * 18
        parts.append(f'<line x1="{lx}" y1="{ly:.1f}" x2="{lx+24}" y2="{ly:.1f}" stroke="{color}" stroke-width="2.6"/>')
        parts.append(f'<text x="{lx+32}" y="{ly+4:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#222">{esc(label)}</text>')

    parts.append(f'<text x="{width/2:.1f}" y="{height-16}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="14">{esc(xlabel)}</text>')
    parts.append(
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.1f})" font-family="Arial, Helvetica, sans-serif" font-size="14">{esc(ylabel)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def panel_bar_svg(
    overall_by_label: dict[str, dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    width: int = 330,
    height: int = 320,
    left: int = 62,
    right: int = 20,
    top: int = 44,
    bottom: int = 70,
) -> str:
    labels = list(overall_by_label.keys())
    values = [overall_by_label[label][metric] for label in labels]
    y_min = min(values)
    y_max = max(values)
    include_zero = metric != "bias"
    if include_zero:
        y_min = min(y_min, 0.0)
    if metric == "bias":
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    pad = 0.16 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = plot_w / max(1, len(labels)) * 0.64

    def sx(i: int) -> float:
        return left + (i + 0.5) * plot_w / len(labels)

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="600">{esc(title)}</text>',
    ]
    for yv in ticks(y_min, y_max):
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#333">{fmt_tick(yv)}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111" stroke-width="1.4"/>')
    if y_min <= 0.0 <= y_max:
        zy = sy(0.0)
        parts.append(f'<line x1="{left}" y1="{zy:.1f}" x2="{width-right}" y2="{zy:.1f}" stroke="#666" stroke-width="1"/>')

    for i, (label, value) in enumerate(zip(labels, values)):
        color = COLORS[i % len(COLORS)]
        xc = sx(i)
        yv = sy(value)
        y0 = sy(0.0) if y_min <= 0.0 <= y_max else height - bottom
        rect_y = min(yv, y0)
        rect_h = abs(y0 - yv)
        parts.append(f'<rect x="{xc-bar_w/2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{max(rect_h,1):.1f}" fill="{color}" opacity="0.9"/>')
        parts.append(f'<text x="{xc:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#333">{esc(label)}</text>')
        parts.append(f'<text x="{xc:.1f}" y="{rect_y-6:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#333">{value:.4f}</text>')

    parts.append(
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.1f})" font-family="Arial, Helvetica, sans-serif" font-size="13">{esc(ylabel)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def combined_overall_bar_svg(overall_by_label: dict[str, dict[str, float]]) -> str:
    specs = [
        ("bias", "Overall Bias", "Bias"),
        ("scatter", "Overall Scatter", "sigma_NMAD"),
        ("outlier_rate", "Overall Outlier Rate", "eta_0.15"),
        ("energy_score", "Overall Energy Score", "Energy Score"),
    ]
    panel_w, panel_h = 330, 320
    gap = 18
    width = panel_w * 4 + gap * 3
    height = panel_h + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for i, (metric, title, ylabel) in enumerate(specs):
        inner = panel_bar_svg(overall_by_label, metric, title, ylabel, panel_w, panel_h)
        inner_body = inner.split("\n", 2)[2].rsplit("\n", 1)[0]
        x = i * (panel_w + gap)
        parts.append(f'<g transform="translate({x},40)">{inner_body}</g>')
    parts.append("</svg>")
    return "\n".join(parts)


def combined_metric_grid_svg(
    rows_by_label: dict[str, list[dict[str, float]]],
    xlabel: str,
    xlim: tuple[float, float] | None,
    title: str,
) -> str:
    panel_w, panel_h = 520, 360
    gap_x, gap_y = 28, 28
    width = panel_w * 3 + gap_x * 2
    height = panel_h + 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="600">{esc(title)}</text>',
    ]
    for i, (metric, metric_title, ylabel, zero_line) in enumerate(LINE_METRICS):
        inner = panel_line_overlay_svg(
            rows_by_label,
            "bin_center",
            metric,
            metric_title,
            xlabel,
            ylabel,
            xlim=xlim,
            zero_line=zero_line,
        )
        inner_body = inner.split("\n", 2)[2].rsplit("\n", 1)[0]
        x = i * (panel_w + gap_x)
        parts.append(f'<g transform="translate({x},58)">{inner_body}</g>')
    parts.append("</svg>")
    return "\n".join(parts)


def combined_pit_hist_svg(
    rows_by_label: dict[str, list[dict[str, float]]],
    width: int = 920,
    height: int = 560,
    margin: int = 74,
) -> str:
    labels = list(rows_by_label.keys())
    labels_ref = labels[0]
    ref_rows = rows_by_label[labels_ref]
    y_max = max(max(r["density"] for r in rows) for rows in rows_by_label.values())
    y_max = max(1.0, y_max * 1.12)
    x_min, x_max = 0.0, 1.0

    def sx(v: float) -> float:
        return margin + (v - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(v: float) -> float:
        return height - margin - v / y_max * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="600">PIT histogram comparison</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
    ]

    uy = sy(1.0)
    parts.append(f'<line x1="{margin}" y1="{uy:.2f}" x2="{width-margin}" y2="{uy:.2f}" stroke="#cc2222" stroke-width="2" stroke-dasharray="7 4"/>')
    parts.append(f'<text x="{width-margin-4}" y="{uy-8:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#cc2222">Uniform density = 1</text>')

    for xv in ticks(0.0, 1.0):
        x = sx(xv)
        parts.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" y2="{height-margin}" stroke="#e3e3e3" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-margin+20}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#444">{xv:.1f}</text>')
    for yv in ticks(0.0, y_max):
        y = sy(yv)
        parts.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" stroke="#e3e3e3" stroke-width="1"/>')
        parts.append(f'<text x="{margin-8}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#444">{yv:.2f}</text>')

    for idx, label in enumerate(labels):
        color = COLORS[idx % len(COLORS)]
        alpha = 0.18
        for row in rows_by_label[label]:
            x0 = sx(row["bin_left"])
            x1 = sx(row["bin_right"])
            y = sy(row["density"])
            parts.append(
                f'<rect x="{x0:.2f}" y="{y:.2f}" width="{max(1.0, x1-x0):.2f}" height="{height-margin-y:.2f}" fill="{color}" fill-opacity="{alpha}"/>'
            )
        pts = [(sx(r["bin_center"]), sy(r["density"])) for r in rows_by_label[label]]
        parts.append(
            '<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>'.format(
                color=color,
                points=" ".join(f"{x:.1f},{y:.1f}" for x, y in pts),
            )
        )
        lx, ly = width - margin - 180, margin + 18 + idx * 22
        parts.append(f'<line x1="{lx}" y1="{ly:.1f}" x2="{lx+28}" y2="{ly:.1f}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{lx+36}" y="{ly+5:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#222">{esc(label)}</text>')

    parts.append(f'<text x="{width/2:.2f}" y="{height-16}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#222">PIT value</text>')
    parts.append(
        f'<text x="18" y="{height/2:.2f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.2f})" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#222">Density</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def combined_pit_curve_svg(rows_by_label: dict[str, list[dict[str, float]]]) -> str:
    return panel_line_overlay_svg(
        rows_by_label,
        "q_th",
        "delta_q",
        "PIT / Delta Q comparison",
        "Q_th",
        "Delta Q",
        xlim=(0.0, 1.0),
        grey_band=(-0.02, 0.02),
        width=920,
        height=560,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline / longer-trained / ensemble MLP eval outputs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Comparison member in the form label=/path/to/eval_dir. Repeat three times or more.",
    )
    parser.add_argument("--outdir", required=True, help="Directory for comparison SVG / CSV outputs")
    parser.add_argument("--mag-xlabel", default="i-band Magnitude")
    parser.add_argument("--mag-xlim", nargs=2, type=float, default=None)
    parser.add_argument("--redshift-xlim", nargs=2, type=float, default=None)
    args = parser.parse_args()

    runs = [parse_run_arg(spec) for spec in args.run]
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    overall_by_label: dict[str, dict[str, float]] = {}
    redshift_by_label: dict[str, list[dict[str, float]]] = {}
    magnitude_by_label: dict[str, list[dict[str, float]]] = {}
    pit_hist_by_label: dict[str, list[dict[str, float]]] = {}
    pit_curve_by_label: dict[str, list[dict[str, float]]] = {}

    for label, eval_dir in runs:
        if not eval_dir.exists():
            raise SystemExit(f"Missing eval directory: {eval_dir}")
        overall_by_label[label] = read_overall(eval_dir / "overall_metrics.csv")
        redshift_by_label[label] = read_rows(eval_dir / "redshift_binned_metrics.csv")
        magnitude_by_label[label] = read_rows(eval_dir / "magnitude_binned_metrics.csv")
        pit_hist_path = eval_dir / "pit_histogram.csv"
        if pit_hist_path.exists():
            pit_hist_by_label[label] = read_rows(pit_hist_path)
        else:
            pit_vals_path = eval_dir / "pit_values.csv"
            if not pit_vals_path.exists():
                raise SystemExit(f"Missing both pit_histogram.csv and pit_values.csv in {eval_dir}")
            pit_hist_by_label[label] = pit_hist_rows_from_values(read_pit_values(pit_vals_path))
        pit_curve_by_label[label] = read_rows(eval_dir / "pit_curve.csv")

    write_overall_table(outdir / "overall_metrics_comparison.csv", overall_by_label)
    write_overall_markdown(outdir / "overall_metrics_comparison.md", overall_by_label)
    write_svg(outdir / "overall_metrics_comparison.svg", combined_overall_bar_svg(overall_by_label))
    write_svg(outdir / "pit_histogram_comparison.svg", combined_pit_hist_svg(pit_hist_by_label))
    write_svg(outdir / "pit_curve_comparison.svg", combined_pit_curve_svg(pit_curve_by_label))
    write_svg(
        outdir / "redshift_metrics_comparison.svg",
        combined_metric_grid_svg(
            redshift_by_label,
            xlabel="Redshift",
            xlim=tuple(args.redshift_xlim) if args.redshift_xlim else None,
            title="RAIL-style metrics vs redshift",
        ),
    )
    write_svg(
        outdir / "magnitude_metrics_comparison.svg",
        combined_metric_grid_svg(
            magnitude_by_label,
            xlabel=args.mag_xlabel,
            xlim=tuple(args.mag_xlim) if args.mag_xlim else None,
            title="RAIL-style metrics vs magnitude",
        ),
    )

    summary_lines = [
        "# Comparison outputs",
        "",
        "Generated files:",
        "- `overall_metrics_comparison.csv`",
        "- `overall_metrics_comparison.md`",
        "- `overall_metrics_comparison.svg`",
        "- `pit_histogram_comparison.svg`",
        "- `pit_curve_comparison.svg`",
        "- `redshift_metrics_comparison.svg`",
        "- `magnitude_metrics_comparison.svg`",
        "",
        "Compared runs:",
    ]
    for label, eval_dir in runs:
        summary_lines.append(f"- `{label}` -> `{eval_dir}`")
    (outdir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(outdir)


if __name__ == "__main__":
    main()
