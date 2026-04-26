#!/usr/bin/env python3
"""
Monte Carlo butterfly SNM extractor from two CSV files.

What it does:
1) Parse both CSV files as multiple (x, y) curve pairs: (x1,y1,x2,y2,...).
2) Pair curve-set i from csv1 with curve-set i from csv2 and compute SNM_i.
3) Print SNM for each pair, then report min/mean/std.
4) Save SNM list to CSV and save SNM distribution histogram as SVG
   with mean and +/-1 sigma markers.

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from statistics import mean, stdev
from typing import List, Sequence, Tuple


Point = Tuple[float, float]


@dataclass
class SnmResult:
    left_lobe: float
    right_lobe: float
    snm: float
    intersections: List[float]


@dataclass
class XYCurve:
    xs: List[float]
    ys: List[float]

    @classmethod
    def from_points(cls, points_sorted_by_x: Sequence[Point]) -> "XYCurve":
        return cls(xs=[p[0] for p in points_sorted_by_x], ys=[p[1] for p in points_sorted_by_x])


@dataclass
class TXCurve:
    ts: List[float]
    xs: List[float]

    @classmethod
    def from_pairs(cls, pairs_sorted_by_t: Sequence[Point]) -> "TXCurve":
        return cls(ts=[p[0] for p in pairs_sorted_by_t], xs=[p[1] for p in pairs_sorted_by_t])


def _is_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except Exception:
        return False


def _extract_label(rows: Sequence[Sequence[str]], col: int, pair_index: int) -> str:
    corner_pat = re.compile(r"([a-z]{2}_lib)", re.IGNORECASE)
    for r in rows[:3]:
        if len(r) <= col + 1:
            continue
        text = (r[col] + " " + r[col + 1]).lower()
        m = corner_pat.search(text)
        if m:
            return m.group(1)
    return f"pair_{pair_index + 1}"


def load_curve_sets(path: str, min_points: int = 4) -> List[Tuple[str, List[Point]]]:
    rows: List[List[str]] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"{path} is empty")

    max_cols = max(len(r) for r in rows)
    curve_sets: List[Tuple[str, List[Point]]] = []

    for col in range(0, max_cols - 1, 2):
        points: List[Point] = []
        for row in rows:
            if len(row) <= col + 1:
                continue
            a = row[col].strip()
            b = row[col + 1].strip()
            if _is_float(a) and _is_float(b):
                points.append((float(a), float(b)))
        if len(points) >= min_points:
            label = _extract_label(rows, col, len(curve_sets))
            curve_sets.append((label, points))

    if not curve_sets:
        raise ValueError(f"{path} has no valid (x,y) curve pair with >= {min_points} points")
    return curve_sets


def sort_and_merge_x(points: Sequence[Point]) -> List[Point]:
    items = sorted(points, key=lambda p: p[0])
    merged: List[Point] = []
    i = 0
    n = len(items)
    while i < n:
        x0 = items[i][0]
        ys = [items[i][1]]
        i += 1
        while i < n and abs(items[i][0] - x0) <= 1e-15:
            ys.append(items[i][1])
            i += 1
        merged.append((x0, sum(ys) / len(ys)))
    return merged


def interp_y(curve: XYCurve, x: float) -> float:
    xs, ys = curve.xs, curve.ys
    eps = 1e-12 * max(1.0, abs(xs[0]), abs(xs[-1]))
    if x < xs[0] - eps or x > xs[-1] + eps:
        raise ValueError("x out of interpolation range")
    if x < xs[0]:
        x = xs[0]
    elif x > xs[-1]:
        x = xs[-1]

    lo, hi = 0, len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, y0 = xs[lo], ys[lo]
    x1, y1 = xs[hi], ys[hi]
    if abs(x1 - x0) < 1e-15:
        return y0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def make_t_x_pairs(curve: XYCurve, samples: int) -> List[Point]:
    x_min, x_max = curve.xs[0], curve.xs[-1]
    pairs: List[Point] = []
    for i in range(samples):
        x = x_min + (x_max - x_min) * i / (samples - 1)
        y = interp_y(curve, x)
        pairs.append((y - x, x))
    pairs.sort(key=lambda p: p[0])
    return pairs


def interp_x_from_t(tx: TXCurve, t: float) -> float:
    ts, xs = tx.ts, tx.xs
    eps = 1e-12 * max(1.0, abs(ts[0]), abs(ts[-1]))
    if t < ts[0] - eps or t > ts[-1] + eps:
        raise ValueError("t out of interpolation range")
    if t < ts[0]:
        t = ts[0]
    elif t > ts[-1]:
        t = ts[-1]

    lo, hi = 0, len(ts) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0 = ts[lo], xs[lo]
    t1, x1 = ts[hi], xs[hi]
    if abs(t1 - t0) < 1e-15:
        return x0
    return x0 + (t - t0) * (x1 - x0) / (t1 - t0)


def find_intersections_x(curve1: XYCurve, curve2: XYCurve, samples: int) -> List[float]:
    x_lo = max(curve1.xs[0], curve2.xs[0])
    x_hi = min(curve1.xs[-1], curve2.xs[-1])
    xs = [x_lo + (x_hi - x_lo) * i / (samples - 1) for i in range(samples)]
    diffs = [interp_y(curve1, x) - interp_y(curve2, x) for x in xs]
    hits: List[float] = []
    for i in range(samples - 1):
        d0, d1 = diffs[i], diffs[i + 1]
        x0, x1 = xs[i], xs[i + 1]
        if abs(d0) < 1e-12:
            hits.append(x0)
            continue
        if d0 * d1 < 0:
            hits.append(x0 + (0 - d0) * (x1 - x0) / (d1 - d0))
    if abs(diffs[-1]) < 1e-12:
        hits.append(xs[-1])

    hits.sort()
    merged: List[float] = []
    for v in hits:
        if not merged or abs(v - merged[-1]) > 1e-5:
            merged.append(v)
    return merged


def local_max(values: Sequence[float], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    m = values[start]
    for i in range(start + 1, end):
        if values[i] > m:
            m = values[i]
    return m


def compute_snm(curve1: XYCurve, curve2: XYCurve, samples: int) -> SnmResult:
    tx1 = TXCurve.from_pairs(make_t_x_pairs(curve1, samples))
    tx2 = TXCurve.from_pairs(make_t_x_pairs(curve2, samples))

    t_min = max(tx1.ts[0], tx2.ts[0])
    t_max = min(tx1.ts[-1], tx2.ts[-1])
    t_grid = [t_min + (t_max - t_min) * i / (samples - 1) for i in range(samples)]

    sides: List[float] = []
    for t in t_grid:
        x1 = interp_x_from_t(tx1, t)
        x2 = interp_x_from_t(tx2, t)
        sides.append(abs(x2 - x1))

    intersections = find_intersections_x(curve1, curve2, samples)
    if len(intersections) >= 3:
        x_mid = intersections[len(intersections) // 2]
        t_mid = interp_y(curve1, x_mid) - x_mid
        idx_mid = min(range(samples), key=lambda i: abs(t_grid[i] - t_mid))
    else:
        left = samples // 10
        right = samples - left
        idx_mid = min(range(left, right), key=lambda i: sides[i])

    left_peak = local_max(sides, 0, idx_mid + 1)
    right_peak = local_max(sides, idx_mid, samples)
    return SnmResult(left_lobe=left_peak, right_lobe=right_peak, snm=min(left_peak, right_peak), intersections=intersections)


def _to_xy(raw_points: Sequence[Point]) -> XYCurve:
    return XYCurve.from_points(sort_and_merge_x(raw_points))


def write_snm_csv(path: str, rows: Sequence[Tuple[int, str, str, float, float, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "csv1_label", "csv2_label", "left_lobe_V", "right_lobe_V", "snm_V"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.9f}", f"{r[4]:.9f}", f"{r[5]:.9f}"])


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def write_histogram_svg(path: str, values: Sequence[float], mu: float, sigma: float) -> None:
    if not values:
        raise ValueError("values is empty")

    n = len(values)
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-15:
        vmin -= 0.5e-3
        vmax += 0.5e-3

    bins = max(10, int(math.sqrt(n)))
    bw = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - vmin) / bw)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1

    width, height = 1100, 700
    ml, mr, mt, mb = 90, 40, 70, 90
    pw, ph = width - ml - mr, height - mt - mb
    cmax = max(counts) if counts else 1

    def sx(v: float) -> float:
        return ml + (v - vmin) / (vmax - vmin) * pw

    def sy(c: float) -> float:
        return mt + ph - c / cmax * ph

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append('<text x="550" y="35" text-anchor="middle" font-size="24" font-family="Arial">SNM Distribution (Monte Carlo)</text>')

    # Axes
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="black" stroke-width="2"/>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="black" stroke-width="2"/>')

    # Bars
    for i, c in enumerate(counts):
        x0 = vmin + i * bw
        x1 = x0 + bw
        px0, px1 = sx(x0), sx(x1)
        py = sy(c)
        h = mt + ph - py
        lines.append(
            f'<rect x="{px0:.2f}" y="{py:.2f}" width="{max(px1 - px0 - 1.0, 1.0):.2f}" height="{h:.2f}" '
            'fill="#8db7ff" stroke="#5d88d8" stroke-width="0.8"/>'
        )

    # Mean and sigma lines
    x_mu = sx(mu)
    lines.append(f'<line x1="{x_mu:.2f}" y1="{mt}" x2="{x_mu:.2f}" y2="{mt + ph}" stroke="#d62728" stroke-width="2.5"/>')
    if sigma > 0:
        x_m1 = sx(mu - sigma)
        x_p1 = sx(mu + sigma)
        lines.append(f'<line x1="{x_m1:.2f}" y1="{mt}" x2="{x_m1:.2f}" y2="{mt + ph}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="8,6"/>')
        lines.append(f'<line x1="{x_p1:.2f}" y1="{mt}" x2="{x_p1:.2f}" y2="{mt + ph}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="8,6"/>')

    # Ticks (x)
    for i in range(6):
        xv = vmin + (vmax - vmin) * i / 5
        px = sx(xv)
        lines.append(f'<line x1="{px:.2f}" y1="{mt + ph}" x2="{px:.2f}" y2="{mt + ph + 8}" stroke="black" stroke-width="1"/>')
        lines.append(f'<text x="{px:.2f}" y="{mt + ph + 28}" text-anchor="middle" font-size="14" font-family="Arial">{xv:.4f}</text>')

    # Ticks (y)
    for i in range(6):
        cv = cmax * i / 5
        py = sy(cv)
        lines.append(f'<line x1="{ml - 8}" y1="{py:.2f}" x2="{ml}" y2="{py:.2f}" stroke="black" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 14}" y="{py + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial">{int(round(cv))}</text>')

    # Labels and legend
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{height - 30}" text-anchor="middle" font-size="18" font-family="Arial">SNM (V)</text>')
    lines.append(f'<text x="28" y="{mt + ph / 2:.2f}" text-anchor="middle" font-size="18" font-family="Arial" transform="rotate(-90 28 {mt + ph / 2:.2f})">Count</text>')

    lx, ly = ml + 20, mt + 15
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 30}" y2="{ly}" stroke="#d62728" stroke-width="2.5"/>')
    lines.append(f'<text x="{lx + 38}" y="{ly + 5}" font-size="14" font-family="Arial">mean = {_fmt(mu)} V</text>')
    lines.append(f'<line x1="{lx}" y1="{ly + 24}" x2="{lx + 30}" y2="{ly + 24}" stroke="#2ca02c" stroke-width="2" stroke-dasharray="8,6"/>')
    lines.append(f'<text x="{lx + 38}" y="{ly + 29}" font-size="14" font-family="Arial">sigma = {_fmt(sigma)} V</text>')
    lines.append(f'<text x="{lx}" y="{ly + 54}" font-size="14" font-family="Arial">N = {n}</text>')

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SNM for each paired VTC (Monte Carlo style) from two CSV files."
    )
    parser.add_argument("csv1", help="CSV for one side VTCs (x,y pairs repeated by columns)")
    parser.add_argument("csv2", help="CSV for mirrored-side VTCs (x,y pairs repeated by columns)")
    parser.add_argument("--samples", type=int, default=20001, help="Interpolation samples (default: 20001)")
    parser.add_argument("--out-csv", default="snm_results.csv", help="Output SNM table CSV path")
    parser.add_argument("--out-plot", default="snm_distribution.svg", help="Output histogram SVG path")
    args = parser.parse_args()

    if args.samples < 1001:
        raise ValueError("--samples should be >= 1001")

    out_csv = args.out_csv
    out_plot = args.out_plot
    base_dir = os.path.dirname(os.path.abspath(args.csv1))
    if out_csv == "snm_results.csv":
        out_csv = os.path.join(base_dir, out_csv)
    if out_plot == "snm_distribution.svg":
        out_plot = os.path.join(base_dir, out_plot)

    sets1 = load_curve_sets(args.csv1)
    sets2 = load_curve_sets(args.csv2)
    pair_count = min(len(sets1), len(sets2))
    if pair_count == 0:
        raise ValueError("No pairable VTC sets found between csv1 and csv2")

    print("=== Monte Carlo SNM Extraction ===")
    print(f"CSV1: {args.csv1}")
    print(f"CSV2: {args.csv2}")
    print(f"Curve sets found: csv1={len(sets1)}, csv2={len(sets2)}, paired={pair_count}")
    if len(sets1) != len(sets2):
        print("Warning: counts differ, using the smaller one by index matching.")

    rows_out: List[Tuple[int, str, str, float, float, float]] = []
    snm_values: List[float] = []

    for i in range(pair_count):
        label1, raw1 = sets1[i]
        label2, raw2 = sets2[i]
        c1 = _to_xy(raw1)
        c2 = _to_xy(raw2)
        res = compute_snm(c1, c2, args.samples)
        rows_out.append((i + 1, label1, label2, res.left_lobe, res.right_lobe, res.snm))
        snm_values.append(res.snm)
        label = label1 if label1 == label2 else f"{label1}|{label2}"
        print(
            f"Pair {i + 1:04d} [{label}]  "
            f"left={res.left_lobe:.6f} V, right={res.right_lobe:.6f} V, SNM={res.snm:.6f} V"
        )

    min_snm = min(snm_values)
    min_idx = snm_values.index(min_snm) + 1
    mu = mean(snm_values)
    sigma = stdev(snm_values) if len(snm_values) >= 2 else 0.0

    write_snm_csv(out_csv, rows_out)
    write_histogram_svg(out_plot, snm_values, mu, sigma)

    print("")
    print("=== Monte Carlo Summary ===")
    print(f"Total pairs          : {pair_count}")
    print(f"Minimum SNM          : {min_snm:.6f} V (pair {min_idx})")
    print(f"Mean SNM             : {mu:.6f} V")
    print(f"Standard deviation   : {sigma:.6f} V")
    print(f"SNM table saved      : {os.path.abspath(out_csv)}")
    print(f"Distribution plot    : {os.path.abspath(out_plot)}")


if __name__ == "__main__":
    main()
