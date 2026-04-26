#!/usr/bin/env python3
"""For PVT-like structure: each CSV has repeated (x,y) pairs by columns."""

from __future__ import annotations

import argparse
import csv
import re
from typing import List, Sequence, Tuple

from snm_core import Point, XYCurve, compute_snm, sort_and_merge_x


def _is_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except Exception:
        return False


def _corner_label(rows: Sequence[Sequence[str]], col: int, idx: int) -> str:
    pat = re.compile(r"([a-z]{2}_lib)", re.IGNORECASE)
    for r in rows[:3]:
        if len(r) <= col + 1:
            continue
        m = pat.search((r[col] + " " + r[col + 1]).lower())
        if m:
            return m.group(1)
    return f"pair_{idx + 1}"


def load_paired_curves(path: str) -> List[Tuple[str, List[Point]]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty")

    max_cols = max(len(r) for r in rows)
    sets: List[Tuple[str, List[Point]]] = []
    for col in range(0, max_cols - 1, 2):
        pts: List[Point] = []
        for row in rows:
            if len(row) <= col + 1:
                continue
            a = row[col].strip()
            b = row[col + 1].strip()
            if _is_float(a) and _is_float(b):
                pts.append((float(a), float(b)))
        if len(pts) >= 4:
            sets.append((_corner_label(rows, col, len(sets)), pts))
    return sets


def main() -> None:
    p = argparse.ArgumentParser(description="Compute SNM for each curve pair in PVT CSV files.")
    p.add_argument("csv1")
    p.add_argument("csv2")
    p.add_argument("--samples", type=int, default=20001)
    args = p.parse_args()

    s1 = load_paired_curves(args.csv1)
    s2 = load_paired_curves(args.csv2)
    n = min(len(s1), len(s2))
    if n == 0:
        raise ValueError("No pairable curve sets")

    print("=== SNM (PVT CSV) ===")
    print(f"CSV1 sets: {len(s1)}, CSV2 sets: {len(s2)}, using: {n}")
    for i in range(n):
        l1, p1 = s1[i]
        l2, p2 = s2[i]
        c1 = XYCurve.from_points(sort_and_merge_x(p1))
        c2 = XYCurve.from_points(sort_and_merge_x(p2))
        r = compute_snm(c1, c2, args.samples)
        label = l1 if l1 == l2 else f"{l1}|{l2}"
        print(f"Pair {i + 1:03d} [{label}] left={r.left_lobe:.6f} V right={r.right_lobe:.6f} V SNM={r.snm:.6f} V")


if __name__ == "__main__":
    main()
