#!/usr/bin/env python3
"""For Monte Carlo export structure:
- CSV1: col0 is shared x; col1..N are y curves (mcparamset columns)
- CSV2: repeated (x_i, y_i) pairs by columns
Pair by mcparamset index parsed from header; fallback to ordinal index.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from statistics import mean, stdev
from typing import Dict, List, Tuple

from snm_core import Point, XYCurve, compute_snm, sort_and_merge_x


def _is_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except Exception:
        return False


def _mc_id(text: str, default_id: int) -> str:
    m = re.search(r"mcparamset\s*(\d+)", text, re.IGNORECASE)
    return m.group(1) if m else str(default_id)


def load_mc_csv1(path: str) -> Dict[str, List[Point]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    data = rows[1:]
    if len(header) < 2:
        raise ValueError(f"{path} columns < 2")

    out: Dict[str, List[Point]] = {}
    for col in range(1, len(header)):
        mcid = _mc_id(header[col], col)
        pts: List[Point] = []
        for row in data:
            if len(row) <= col:
                continue
            a = row[0].strip()      # shared x
            b = row[col].strip()    # y_i
            if _is_float(a) and _is_float(b):
                pts.append((float(a), float(b)))
        if len(pts) >= 4:
            out[mcid] = pts
    return out


def load_mc_csv2(path: str) -> Dict[str, List[Point]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    data = rows[1:]

    out: Dict[str, List[Point]] = {}
    pair_idx = 0
    for col in range(0, len(header) - 1, 2):
        pair_idx += 1
        mcid = _mc_id(header[col + 1], pair_idx)
        pts: List[Point] = []
        for row in data:
            if len(row) <= col + 1:
                continue
            a = row[col].strip()
            b = row[col + 1].strip()
            if _is_float(a) and _is_float(b):
                pts.append((float(a), float(b)))
        if len(pts) >= 4:
            out[mcid] = pts
    return out


def write_results_csv(path: str, rows: List[Tuple[int, float, float, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mcparamset", "left_lobe_V", "right_lobe_V", "snm_V"])
        for mcid, left, right, snm in rows:
            w.writerow([mcid, f"{left:.9f}", f"{right:.9f}", f"{snm:.9f}"])


def write_probability_svg(path: str, values: List[float], mu: float, sigma: float) -> None:
    if not values:
        raise ValueError("empty values")

    n = len(values)
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-15:
        vmin -= 0.0005
        vmax += 0.0005

    bins = max(12, int(math.sqrt(n)))
    bw = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - vmin) / bw)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1

    # Density histogram (area = 1)
    densities = [(c / n) / bw for c in counts]

    # Gaussian-kernel density estimate
    h = sigma * (n ** (-1.0 / 5.0)) if sigma > 0 else (vmax - vmin) / 30.0
    if h <= 0:
        h = 1e-4
    x_grid = [vmin + (vmax - vmin) * i / 300 for i in range(301)]
    kde = []
    norm = 1.0 / (n * h * math.sqrt(2.0 * math.pi))
    for x in x_grid:
        s = 0.0
        for v in values:
            z = (x - v) / h
            s += math.exp(-0.5 * z * z)
        kde.append(norm * s)

    y_max = max(max(densities) if densities else 0.0, max(kde) if kde else 0.0) * 1.15
    if y_max <= 0:
        y_max = 1.0

    width, height = 1100, 720
    ml, mr, mt, mb = 90, 40, 70, 95
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x: float) -> float:
        return ml + (x - vmin) / (vmax - vmin) * pw

    def sy(y: float) -> float:
        return mt + ph - (y / y_max) * ph

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append('<text x="550" y="36" text-anchor="middle" font-size="24" font-family="Arial">SNM Probability Distribution (Monte Carlo)</text>')

    # Axes
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="black" stroke-width="2"/>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="black" stroke-width="2"/>')

    # Histogram density bars
    for i, d in enumerate(densities):
        x0 = vmin + i * bw
        x1 = x0 + bw
        px0, px1 = sx(x0), sx(x1)
        py = sy(d)
        hgt = mt + ph - py
        lines.append(
            f'<rect x="{px0:.2f}" y="{py:.2f}" width="{max(px1 - px0 - 1.0, 1.0):.2f}" '
            f'height="{hgt:.2f}" fill="#bfd5ff" stroke="#7ea4f7" stroke-width="0.8"/>'
        )

    # KDE curve
    pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_grid, kde))
    lines.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="3"/>')

    # Mean / sigma markers
    x_mu = sx(mu)
    lines.append(f'<line x1="{x_mu:.2f}" y1="{mt}" x2="{x_mu:.2f}" y2="{mt + ph}" stroke="#d62728" stroke-width="2.5"/>')
    if sigma > 0:
        x_l = sx(mu - sigma)
        x_r = sx(mu + sigma)
        lines.append(f'<line x1="{x_l:.2f}" y1="{mt}" x2="{x_l:.2f}" y2="{mt + ph}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="8,6"/>')
        lines.append(f'<line x1="{x_r:.2f}" y1="{mt}" x2="{x_r:.2f}" y2="{mt + ph}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="8,6"/>')

    # Ticks
    for i in range(6):
        xv = vmin + (vmax - vmin) * i / 5
        px = sx(xv)
        lines.append(f'<line x1="{px:.2f}" y1="{mt + ph}" x2="{px:.2f}" y2="{mt + ph + 8}" stroke="black" stroke-width="1"/>')
        lines.append(f'<text x="{px:.2f}" y="{mt + ph + 28}" text-anchor="middle" font-size="14" font-family="Arial">{xv:.4f}</text>')
    for i in range(6):
        yv = y_max * i / 5
        py = sy(yv)
        lines.append(f'<line x1="{ml - 8}" y1="{py:.2f}" x2="{ml}" y2="{py:.2f}" stroke="black" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 12}" y="{py + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial">{yv:.2f}</text>')

    # Labels and legend
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{height - 30}" text-anchor="middle" font-size="18" font-family="Arial">SNM (V)</text>')
    lines.append(f'<text x="30" y="{mt + ph / 2:.2f}" text-anchor="middle" font-size="18" font-family="Arial" transform="rotate(-90 30 {mt + ph / 2:.2f})">Probability Density</text>')
    lx, ly = ml + 16, mt + 14
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 36}" y="{ly + 5}" font-size="14" font-family="Arial">KDE curve</text>')
    lines.append(f'<line x1="{lx}" y1="{ly + 22}" x2="{lx + 28}" y2="{ly + 22}" stroke="#d62728" stroke-width="2.5"/>')
    lines.append(f'<text x="{lx + 36}" y="{ly + 27}" font-size="14" font-family="Arial">mean = {mu:.6f} V</text>')
    lines.append(f'<line x1="{lx}" y1="{ly + 44}" x2="{lx + 28}" y2="{ly + 44}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="8,6"/>')
    lines.append(f'<text x="{lx + 36}" y="{ly + 49}" font-size="14" font-family="Arial">sigma = {sigma:.6f} V</text>')
    lines.append(f'<text x="{lx}" y="{ly + 71}" font-size="14" font-family="Arial">N = {n}</text>')

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Compute SNM for MC curve pairs from two MC CSV files.")
    p.add_argument("csv1")
    p.add_argument("csv2")
    p.add_argument("--samples", type=int, default=20001)
    p.add_argument("--out-csv", default="mc_snm_results.csv")
    p.add_argument("--out-plot", default="mc_snm_probability.svg")
    args = p.parse_args()

    d1 = load_mc_csv1(args.csv1)
    d2 = load_mc_csv2(args.csv2)
    keys = sorted(set(d1.keys()) & set(d2.keys()), key=lambda s: int(s))
    if not keys:
        raise ValueError("No matched mcparamset between csv1 and csv2")

    snms: List[float] = []
    out_rows: List[Tuple[int, float, float, float]] = []
    print("=== SNM (MC CSV) ===")
    print(f"Matched mcparamset count: {len(keys)}")
    for k in keys:
        c1 = XYCurve.from_points(sort_and_merge_x(d1[k]))
        c2 = XYCurve.from_points(sort_and_merge_x(d2[k]))
        r = compute_snm(c1, c2, args.samples)
        snms.append(r.snm)
        out_rows.append((int(k), r.left_lobe, r.right_lobe, r.snm))
        print(f"mcparamset {int(k):03d}: left={r.left_lobe:.6f} V right={r.right_lobe:.6f} V SNM={r.snm:.6f} V")

    mn = min(snms)
    idx = snms.index(mn)
    mu = mean(snms)
    sig = stdev(snms) if len(snms) >= 2 else 0.0
    base_dir = os.path.dirname(os.path.abspath(args.csv1))
    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(base_dir, args.out_csv)
    out_plot = args.out_plot if os.path.isabs(args.out_plot) else os.path.join(base_dir, args.out_plot)
    write_results_csv(out_csv, out_rows)
    write_probability_svg(out_plot, snms, mu, sig)
    print("")
    print("Summary:")
    print(f"Minimum SNM : {mn:.6f} V (mcparamset {int(keys[idx]):03d})")
    print(f"Mean SNM    : {mu:.6f} V")
    print(f"Std Dev     : {sig:.6f} V")
    print(f"SNM CSV     : {out_csv}")
    print(f"SNM PDF SVG : {out_plot}")


if __name__ == "__main__":
    main()
