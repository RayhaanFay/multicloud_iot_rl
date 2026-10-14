"""Aggregate metrics.csv files from several seed folders into mean +/- std per policy.

Usage:
    python aggregate.py seed_1000 seed_2000 seed_3000 seed_4000 seed_5000
or, with a glob-like prefix:
    python aggregate.py --prefix seed_
"""
import argparse
import csv
from pathlib import Path

import numpy as np

SHOW = [
    ("deadline_hit_rate", "On-time %"),
    ("deadline_miss_rate", "Miss %"),
    ("avg_latency", "Latency"),
    ("avg_cost", "Cost"),
    ("recovery_rate", "Recovery %"),
    ("avg_reward", "Reward"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", help="result folders containing metrics.csv")
    ap.add_argument("--prefix", default=None, help="use all folders in cwd starting with this prefix")
    ap.add_argument("--out", default="aggregate.csv")
    args = ap.parse_args()

    folders = [Path(f) for f in args.folders]
    if args.prefix:
        folders += sorted(p for p in Path(".").iterdir() if p.is_dir() and p.name.startswith(args.prefix))
    folders = [f for f in folders if (f / "metrics.csv").exists()]
    if not folders:
        raise SystemExit("No metrics.csv files found.")

    data = {}  # policy -> metric -> list of per-seed values
    for folder in folders:
        with (folder / "metrics.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pol = row["policy"]
                for key, _ in SHOW:
                    data.setdefault(pol, {}).setdefault(key, []).append(float(row[key]))

    n = len(folders)
    print(f"Aggregated over {n} runs: {', '.join(f.name for f in folders)}\n")
    header = ["Policy"] + [label for _, label in SHOW]
    rows = []
    for pol, metrics in data.items():
        cells = [pol]
        for key, _ in SHOW:
            vals = np.array(metrics[key])
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            cells.append(f"{vals.mean():.2f} ± {sd:.2f}")
        rows.append(cells)

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
