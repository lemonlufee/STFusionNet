#!/usr/bin/env python3
"""Shared SNM geometry utilities (no third-party dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


Point = Tuple[float, float]


@dataclass
class SnmResult:
    left_lobe: float
    right_lobe: float
    snm: float


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
    out: List[Point] = []
    for i in range(samples):
        x = x_min + (x_max - x_min) * i / (samples - 1)
        y = interp_y(curve, x)
        out.append((y - x, x))
    out.sort(key=lambda p: p[0])
    return out


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


def compute_snm(curve1: XYCurve, curve2: XYCurve, samples: int = 20001) -> SnmResult:
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
    return SnmResult(left_lobe=left_peak, right_lobe=right_peak, snm=min(left_peak, right_peak))
