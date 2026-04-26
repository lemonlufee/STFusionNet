import argparse
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_taihu import Config
from utils.util_common import save_json, configure_stdio_for_server, collect_runtime_env


def _safe_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "skew": float("nan"),
            "q05": float("nan"),
            "q50": float("nan"),
            "q95": float("nan"),
        }
    s = pd.Series(x)
    mean = float(s.mean())
    std = float(s.std(ddof=1))
    return {
        "count": float(x.size),
        "mean": mean,
        "std": std,
        "cv": float(std / (abs(mean) + 1e-12)),
        "skew": float(s.skew()),
        "q05": float(s.quantile(0.05)),
        "q50": float(s.quantile(0.50)),
        "q95": float(s.quantile(0.95)),
    }


def _regime_label(skew: float, cv: float) -> str:
    if (not np.isfinite(skew)) or (not np.isfinite(cv)):
        return "unknown"
    if abs(skew) < 0.6 and cv < 0.35:
        return "smooth-like"
    if abs(skew) >= 1.0 or cv >= 0.8:
        return "regime-dependent"
    return "mixed"


def main() -> None:
    configure_stdio_for_server()
    parser = argparse.ArgumentParser(description="Feature distribution/regime diagnostics for paper discussion.")
    parser.add_argument("--raw_data_file", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./Training_time_log")
    args = parser.parse_args()

    cfg = Config()
    raw_file = args.raw_data_file.strip() if args.raw_data_file else cfg.RAW_DATA_FILE
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"RAW file not found: {raw_file}")

    df = pd.read_csv(raw_file, low_memory=False)
    rows: List[Dict[str, Any]] = []
    for feat in cfg.TARGET_FEATURES:
        if feat not in df.columns:
            continue
        arr = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        st = _safe_stats(arr)
        label = _regime_label(st["skew"], st["cv"])
        rows.append({"feature": feat, "regime_label": label, **st})

    out_df = pd.DataFrame(rows).sort_values("feature")
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(collect_runtime_env(), os.path.join(args.out_dir, "runtime_env_feature_regime.json"))
    csv_path = os.path.join(args.out_dir, "feature_regime_report.csv")
    json_path = os.path.join(args.out_dir, "feature_regime_report.json")
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_json({"raw_data_file": raw_file, "rows": rows}, json_path)

    if "Cond" in out_df["feature"].values:
        cond = out_df[out_df["feature"] == "Cond"].iloc[0].to_dict()
        print(
            "Conductivity diagnostic: "
            f"skew={cond.get('skew'):.4f}, cv={cond.get('cv'):.4f}, regime={cond.get('regime_label')}"
        )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
