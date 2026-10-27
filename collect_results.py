#
# Collects the decisive-experiment runs into CSV files.
#
# Expected layout:  <root>/<scene>/<arm>/seed<k>/{results.json, train_stats.json, eval_maps.json, sem_log.jsonl}
#
# Writes:
#   <out>/runs.csv     one row per run
#   <out>/summary.csv  mean / std over seeds per (scene, arm), plus scene="ALL" rows that average the
#                      per-scene means of each arm (scenes missing for an arm are reported in n_scenes)
#
# Usage:  python collect_results.py --root output/decisive [--out output/decisive/_summary]
#

import argparse
import csv
import json
import math
import os
from collections import defaultdict


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def latest_method(d):
    """Pick the 'ours_<iter>' entry with the largest iteration."""
    if not d:
        return None, None
    def it(k):
        try:
            return int(k.split("_")[-1])
        except ValueError:
            return -1
    key = max(d.keys(), key=it)
    return key, d[key]


def sem_log_summary(path):
    if not os.path.exists(path):
        return {}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}

    def mean_of(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "log_steps": len(rows),
        "sum_n_sem": sum(r.get("n_sem", 0) for r in rows),
        "sum_n_sem_only": sum(r.get("n_sem_only", 0) for r in rows),
        "sum_n_grad": sum(r.get("n_grad", 0) for r in rows),
        "mean_sem_radius_frac_ge_rmax": mean_of([(r.get("radius_sem") or {}).get("frac_ge_ref") for r in rows]),
        "mean_valid_radius_frac_ge_rmax": mean_of([(r.get("radius_valid") or {}).get("frac_ge_ref") for r in rows]),
        "mean_sem_frac_grad_below_tau": mean_of([r.get("sem_frac_grad_below_tau") for r in rows]),
        "mean_raw_negative_frac": mean_of([r.get("raw_negative_frac") for r in rows]),
        "mean_shadow_spearman": mean_of([r.get("shadow_spearman_mean") for r in rows]),
        "frac_steps_r9_cap_bound": mean_of([float(r["r9_cap_bound"]) for r in rows if "r9_cap_bound" in r]),
    }


def parse_run(run_dir, root):
    rel = os.path.relpath(run_dir, root).split(os.sep)
    if len(rel) >= 3:
        scene, arm, seed = rel[-3], rel[-2], rel[-1]
    else:
        scene, arm, seed = "?", os.path.relpath(run_dir, root), "?"
    seed = seed[4:] if seed.startswith("seed") else seed

    row = {"scene": scene, "arm": arm, "seed": seed, "path": run_dir}

    method, metrics = latest_method(read_json(os.path.join(run_dir, "results.json")))
    if metrics:
        row.update({"method": method, "PSNR": metrics.get("PSNR"), "SSIM": metrics.get("SSIM"), "LPIPS": metrics.get("LPIPS")})

    ts = read_json(os.path.join(run_dir, "train_stats.json")) or {}
    row["num_gaussians_M"] = ts["num_gaussians"] / 1e6 if "num_gaussians" in ts else None
    row["train_min"] = ts.get("wall_time_min")
    row["peak_vram_gb"] = ts.get("peak_vram_allocated_gb")

    em = read_json(os.path.join(run_dir, "eval_maps.json"))
    if em and em.get("methods"):
        _, m = latest_method(em["methods"])
        for sig, stats in (m or {}).get("summary", {}).items():
            for k, v in stats.items():
                row[f"map_{sig}_{k}"] = v

    row.update(sem_log_summary(os.path.join(run_dir, "sem_log.jsonl")))
    return row


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else None
    return m, s, len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/decisive")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.root, "_summary")
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for dirpath, dirnames, filenames in os.walk(args.root):
        if os.path.abspath(dirpath).startswith(os.path.abspath(out_dir)):
            continue
        if "train_stats.json" in filenames or "results.json" in filenames:
            runs.append(parse_run(dirpath, args.root))
            dirnames[:] = []  # do not descend into a run folder
    runs.sort(key=lambda r: (r["scene"], r["arm"], str(r["seed"])))
    if not runs:
        print(f"No runs found under {args.root}")
        return

    keys = []
    for r in runs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(runs)

    numeric = [k for k in keys if k not in ("scene", "arm", "seed", "path", "method")]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["scene"], r["arm"])].append(r)

    summary = []
    for (scene, arm), rs in sorted(groups.items()):
        row = {"scene": scene, "arm": arm, "n_seeds": len(rs)}
        for k in numeric:
            m, s, n = mean_std([r.get(k) for r in rs])
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s
        summary.append(row)

    by_arm = defaultdict(list)
    for row in summary:
        by_arm[row["arm"]].append(row)
    for arm, rows in sorted(by_arm.items()):
        row = {"scene": "ALL", "arm": arm, "n_seeds": None, "n_scenes": len(rows)}
        for k in numeric:
            m, _, _ = mean_std([r.get(f"{k}_mean") for r in rows])
            row[f"{k}_mean"] = m
        summary.append(row)

    skeys = []
    for r in summary:
        for k in r:
            if k not in skeys:
                skeys.append(k)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=skeys)
        w.writeheader()
        w.writerows(summary)

    print(f"{len(runs)} runs -> {out_dir}/runs.csv, {out_dir}/summary.csv")
    for row in summary:
        if row["scene"] == "ALL":
            lp = row.get("LPIPS_mean")
            gs = row.get("num_gaussians_M_mean")
            print(f"  ALL {row['arm']:<16s} scenes={row['n_scenes']}  LPIPS={lp if lp is None else round(lp, 4)}  #GS(M)={gs if gs is None else round(gs, 3)}")


if __name__ == "__main__":
    main()
