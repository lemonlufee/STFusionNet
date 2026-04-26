#!/usr/bin/env python3
"""For simple structure: each CSV contains one curve with 2 columns (x,y)."""

from __future__ import annotations

import argparse
import csv
from typing import List, Tuple

from snm_core import Point, XYCurve, compute_snm, sort_and_merge_x


def _is_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except Exception:
        return False


def load_single_curve(path: str) -> List[Point]:
    out: List[Point] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if _is_float(a) and _is_float(b):
                out.append((float(a), float(b)))
    if len(out) < 4:
        raise ValueError(f"{path}: valid points < 4")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compute SNM for one curve pair from two simple CSV files.")
    p.add_argument("csv1")
    p.add_argument("csv2")
    p.add_argument("--samples", type=int, default=20001)
    args = p.parse_args()

    c1 = XYCurve.from_points(sort_and_merge_x(load_single_curve(args.csv1)))
    c2 = XYCurve.from_points(sort_and_merge_x(load_single_curve(args.csv2)))
    r = compute_snm(c1, c2, args.samples)

    print("=== SNM (Simple CSV) ===")
    print(f"CSV1: {args.csv1}")
    print(f"CSV2: {args.csv2}")
    print(f"Left lobe max square side : {r.left_lobe:.6f} V")
    print(f"Right lobe max square side: {r.right_lobe:.6f} V")
    print(f"SNM = min(left, right)    : {r.snm:.6f} V")


if __name__ == "__main__":
    main()
