import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def merge_folder(run_dir: str, features: List[str]) -> List[str]:
    # Legacy single-image merging is removed.
    # This helper now only reports panel outputs if they already exist.
    outputs: List[str] = []
    for name in ("timeseries_panel_2x3.png", "density_scatter_panel_2x3.png"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            outputs.append(p)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-feature run images into 2x3 panels.")
    ap.add_argument("--root", type=str, default="Training_time_log")
    ap.add_argument("--features", type=str, default="DO,Tur,TN,TP,PI,Cond")
    ap.add_argument("--only_dirs", type=str, default="")
    args = ap.parse_args()

    feats = [x.strip() for x in args.features.split(",") if x.strip()]
    if args.only_dirs.strip():
        targets = [x.strip() for x in args.only_dirs.split(",") if x.strip()]
        dirs = [os.path.join(args.root, d) if not os.path.isabs(d) else d for d in targets]
    else:
        dirs = [
            os.path.join(args.root, d)
            for d in os.listdir(args.root)
            if os.path.isdir(os.path.join(args.root, d))
        ]

    total = 0
    for d in dirs:
        try:
            outs = merge_folder(d, feats)
            if outs:
                total += len(outs)
                print(f"[OK] {d}")
                for p in outs:
                    print(f"  - {p}")
        except Exception as e:
            print(f"[WARN] {d}: {e}")
    print(f"[DONE] merged panels: {total}")


if __name__ == "__main__":
    main()
