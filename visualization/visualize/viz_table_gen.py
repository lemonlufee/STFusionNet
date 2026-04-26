import warnings
warnings.warn('DEPRECATED: use evaluation/eval_feature_regime.py for report generation', DeprecationWarning, stacklevel=2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


def read_csv_robust(path: str, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "big5"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV with common encodings. Last error: {last_err}")


def to_float_safe(x: Any) -> float:
    """
    Convert pandas/numpy scalar (or python numeric) to python float safely.
    - If x is complex: use real part if imag ~ 0, else return NaN.
    - If x cannot be converted: return NaN.
    """
    if x is None:
        return float("nan")

    # pandas NaN / numpy NaN
    try:
        if pd.isna(x):
            return float("nan")
    except Exception:
        pass

    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        x = x.item()

    # complex handling
    if isinstance(x, complex):
        if abs(x.imag) < 1e-12:
            return float(x.real)
        return float("nan")

    # normal numeric
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_cv(mean: Any, sd: Any) -> float:
    m = to_float_safe(mean)
    s = to_float_safe(sd)
    if np.isnan(m) or np.isnan(s):
        return float("nan")
    if np.isclose(m, 0.0):
        return float("nan")
    return s / m


def build_stats_table(
    df: pd.DataFrame,
    variables: List[str],
    desc_map: Optional[Dict[str, str]] = None,
    unit_map: Optional[Dict[str, str]] = None,
    ddof: int = 1,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    desc_map = desc_map or {}
    unit_map = unit_map or {}

    work = df.copy()

    # Force numeric conversion: non-numeric values become NaN to keep stats stable.
    for c in variables:
        if c not in work.columns:
            raise KeyError(f"Column not found in CSV: {c}")
        work[c] = pd.to_numeric(work[c], errors="coerce")

    def one_block(block: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for v in variables:
            s = block[v].dropna()
            if len(s) == 0:
                mean = mx = mn = var = sd = cv = float("nan")
                n_valid = 0
            else:
                mean = to_float_safe(s.mean())
                mx = to_float_safe(s.max())
                mn = to_float_safe(s.min())
                var = to_float_safe(s.var(ddof=ddof))
                sd = to_float_safe(s.std(ddof=ddof))
                cv = safe_cv(mean, sd)
                n_valid = int(s.shape[0])

            rows.append(
                {
                    "Variable": v,
                    "Description": desc_map.get(v, ""),
                    "Unit": unit_map.get(v, ""),
                    "Mean": mean,
                    "Maximum": mx,
                    "Minimum": mn,
                    "Variance": var,
                    "SD": sd,
                    "CV": cv,
                    "N_valid": n_valid,
                }
            )

        out = pd.DataFrame(rows)
        return out[
            ["Variable", "Description", "Unit", "Mean", "Maximum", "Minimum", "Variance", "SD", "CV", "N_valid"]
        ]

    if group_col is None:
        return one_block(work)

    if group_col not in work.columns:
        raise KeyError(f"group_col not found in CSV: {group_col}")

    parts = []
    # pandas-stubs compatibility: avoid groupby(..., dropna=...) here.
    # Keep NaN groups by filling a sentinel value first.
    grouped_series = work[group_col].astype(object).where(work[group_col].notna(), "__NA__")
    for g, sub in work.groupby(grouped_series):
        t = one_block(sub)
        t.insert(0, "Group", (np.nan if g == "__NA__" else g))
        parts.append(t)

    return pd.concat(parts, ignore_index=True)


def load_mapping_json(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    desc = obj.get("description", {}) or {}
    unit = obj.get("unit", {}) or {}
    return {"description": desc, "unit": unit}


def fmt_num(x: Any, float_fmt: str) -> str:
    val = to_float_safe(x)
    if np.isnan(val):
        return ""
    # Use format() when possible; generally friendlier for static type checking.
    try:
        # Default numeric format is "%.4f".
        # Convert old-style %.xf patterns to Python f-string formatting.
        if float_fmt.startswith("%.") and float_fmt.endswith("f"):
            digits = int(float_fmt[2:-1])
            return f"{val:.{digits}f}"
        return float_fmt % val  # legacy fallback
    except Exception:
        return str(val)


def main():
    ap = argparse.ArgumentParser(description="Generate Table-1-like stats table from a raw CSV.")
    ap.add_argument("--csv", required=True, help="Path to input CSV file.")
    ap.add_argument("--vars", required=True, help="Comma-separated variable column names to summarize.")
    ap.add_argument("--map_json", default="", help="Optional JSON providing description/unit mappings.")
    ap.add_argument("--group_col", default="", help="Optional column to group by (e.g., station/site).")
    ap.add_argument("--ddof", type=int, default=1, help="ddof for variance/std (1=sample, 0=population).")
    ap.add_argument("--out", default="table1_stats.csv", help="Output CSV path.")
    ap.add_argument("--out_xlsx", default="", help="Optional output Excel (.xlsx) path.")
    ap.add_argument("--float_fmt", default="", help="Optional float format, e.g., '%.4f'.")

    args = ap.parse_args()

    df = read_csv_robust(args.csv)

    variables = [v.strip() for v in args.vars.split(",") if v.strip()]
    if not variables:
        raise ValueError("--vars is empty after parsing")

    desc_map, unit_map = {}, {}
    if args.map_json:
        maps = load_mapping_json(args.map_json)
        desc_map = maps["description"]
        unit_map = maps["unit"]

    group_col = args.group_col.strip() or None

    table = build_stats_table(
        df=df,
        variables=variables,
        desc_map=desc_map,
        unit_map=unit_map,
        ddof=args.ddof,
        group_col=group_col,
    )

    # Optional formatting settings used by table/chart rendering.
    if args.float_fmt:
        for c in ["Mean", "Maximum", "Minimum", "Variance", "SD", "CV"]:
            table[c] = table[c].map(lambda x: fmt_num(x, args.float_fmt))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")

    if args.out_xlsx:
        os.makedirs(os.path.dirname(args.out_xlsx) or ".", exist_ok=True)
        with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as w:
            table.to_excel(w, index=False, sheet_name="Table1_stats")

    print(f"Saved: {args.out}")
    if args.out_xlsx:
        print(f"Saved: {args.out_xlsx}")


if __name__ == "__main__":
    main()
