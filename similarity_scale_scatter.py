#
# Similarity vs projected radius / scale weight plots (review item 4).
#
# Input (either):
#   <model>/similarity_scale_statistics.json          from train_sim_rendersim.save_similarity_scale_statistics_json
#   <model>/sem_snapshots/scale_weight_iter_*.npz     from train_sim.py --sem_log (summary taken from sem_log.jsonl)
#
# Output: <model>/similarity_scale_scatter/
#   <iter>_radius.png          similarity (with scale weight) vs projected radius, colored by interval G1..G3
#   <iter>_raw_vs_weighted.png similarity without vs with the scale weight; the box below the weighted G3
#                              boundary and right of the unweighted G3 boundary = "G3 only because of the weight"
#   <iter>_radius_hist.png     projected-radius distribution: all valid / G3 with weight / G3 without weight
#   <iter>_scale_ratio.png     the previous plot (similarity vs min/max world scale), when available
#   summary.txt                the per-iteration summary numbers
#
# Usage:  python similarity_scale_scatter.py output/flowers_dino
#

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference palette (dataviz skill): ordinal one-hue ramp for G1..G3, neutral gray for the rest
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b5b4ae"
INTERVAL_COLORS = {1: "#6da7ec", 2: "#256abf", 3: "#0d366b"}
CONTRAST = "#eb6834"     # second categorical slot, used for the "without scale weight" series
INTERVAL_NAMES = {0: "other valid", 1: "G1", 2: "G2", 3: "G3"}


def load_records(model_path):
    json_path = os.path.join(model_path, "similarity_scale_statistics.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        records = []
        for rec in data:
            pts = rec.get("points", [])
            if not pts:
                continue
            keys = pts[0].keys()
            arrays = {k: np.asarray([p[k] for p in pts]) for k in keys}
            records.append({"iteration": rec["iteration"], "summary": rec.get("summary"), "points": arrays})
        return records

    summaries = {}
    log_path = os.path.join(model_path, "sem_log.jsonl")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    summaries[r["iteration"]] = r.get("scale_weight")
    records = []
    for path in sorted(glob.glob(os.path.join(model_path, "sem_snapshots", "scale_weight_iter_*.npz"))):
        it = int(os.path.basename(path).split("_")[-1].split(".")[0])
        z = np.load(path)
        records.append({"iteration": it, "summary": summaries.get(it), "points": {k: z[k] for k in z.files}})
    return records


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def boundaries(stats):
    if not stats or stats.get("mean") is None or stats.get("std") is None:
        return None
    m, s = stats["mean"], stats["std"]
    return [m - s, m - 2 * s, m - 3 * s]


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_radius(rec, out_dir, r_ref):
    p, it = rec["points"], rec["iteration"]
    x, y, lab = p["radius"], p["similarity"], p["interval"]
    xmax = max(2 * r_ref, float(np.percentile(x, 99.5)))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for k in (0, 1, 2, 3):  # draw the rare low intervals last
        m = lab == k
        if not m.any():
            continue
        ax.scatter(x[m], y[m], s=2 if k == 0 else 4, alpha=0.35 if k == 0 else 0.8, linewidths=0,
                   color=NEUTRAL if k == 0 else INTERVAL_COLORS[k], label=f"{INTERVAL_NAMES[k]} (n={int(m.sum())})",
                   rasterized=True)
    ax.axvline(r_ref, color=TEXT_2, linestyle="--", linewidth=1)
    ax.text(r_ref, 1.0, f" R_max = {r_ref:g}px", color=TEXT_2, fontsize=8, va="top", transform=ax.get_xaxis_transform())
    b = boundaries((rec["summary"] or {}).get("weighted"))
    if b:
        for v, name in zip(b, ("μ−σ", "μ−2σ", "μ−3σ")):
            ax.axhline(v, color=GRID, linestyle=":", linewidth=1)
            ax.text(xmax, v, f"{name} ", color=TEXT_2, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(0, xmax)
    style(ax, f"Iteration {it}: similarity (with scale weight) vs projected radius",
          "mean projected radius over valid views (px)", "similarity score S_k")
    ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=3, labelcolor=TEXT,
              title="n = plotted sample", title_fontsize=8)
    save(fig, os.path.join(out_dir, f"{it:06d}_radius.png"))


def plot_raw_vs_weighted(rec, out_dir, r_ref):
    p, it = rec["points"], rec["iteration"]
    x, y, r = p["similarity_raw"], p["similarity"], p["radius"]
    large = r >= r_ref
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(x[~large], y[~large], s=2, alpha=0.35, linewidths=0, color=NEUTRAL,
               label=f"radius < {r_ref:g}px (n={int((~large).sum())})", rasterized=True)
    ax.scatter(x[large], y[large], s=3, alpha=0.6, linewidths=0, color=INTERVAL_COLORS[2],
               label=f"radius ≥ {r_ref:g}px (n={int(large.sum())})", rasterized=True)
    lo = max(-1.0, float(min(np.percentile(x, 0.1), np.percentile(y, 0.1))) - 0.05)  # zoom to the data
    ax.plot([lo, 1], [lo, 1], color=TEXT_2, linewidth=1, label="no change (y = x)")
    w_min = float(np.min(p["scale_weight"])) if "scale_weight" in p else None
    if w_min is not None and w_min < 0.999:
        ax.plot([lo, 1], [lo * w_min, w_min], color=TEXT_2, linewidth=1, linestyle="--",
                label=f"largest reduction (y = {w_min:.2f}·x)")
    s = rec["summary"] or {}
    bw, br = boundaries(s.get("weighted")), boundaries(s.get("unweighted"))
    if bw and br:
        ax.axhline(bw[2], color=CONTRAST, linewidth=1, linestyle=":")
        ax.axvline(br[2], color=CONTRAST, linewidth=1, linestyle=":")
        ax.text(lo, bw[2], " G3 boundary (with weight)", color=TEXT_2, fontsize=8, ha="left", va="bottom")
        ax.text(br[2], 1.0, "G3 boundary (without weight) ", color=TEXT_2, fontsize=8, rotation=90, ha="right", va="top")
        frac = s.get("g3_only_by_scale_weight")
        if frac is not None:
            # the region above y = x is always empty (the weight never raises a score)
            ax.text(0.03, 0.58, f"G3 only because of\nthe scale weight: {100 * frac:.1f}%", color=TEXT, fontsize=9,
                    ha="left", va="top", transform=ax.transAxes)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    style(ax, f"Iteration {it}: score without vs with scale weight",
          "similarity without scale weight (same views)", "similarity with scale weight S_k")
    ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=3, labelcolor=TEXT,
              title="n = plotted sample", title_fontsize=8)
    save(fig, os.path.join(out_dir, f"{it:06d}_raw_vs_weighted.png"))


def plot_radius_hist(rec, out_dir, r_ref):
    p, it = rec["points"], rec["iteration"]
    r = p["radius"]
    xmax = max(2 * r_ref, float(np.percentile(r, 99.5)))
    bins = np.linspace(0, xmax, 60)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = [("all valid", np.ones_like(r, dtype=bool), NEUTRAL),
              ("G3 with scale weight", p["interval"] == 3, INTERVAL_COLORS[3]),
              ("G3 without scale weight", p["interval_raw"] == 3, CONTRAST)]
    for name, m, c in series:
        if m.sum() < 2:
            continue
        frac = float((r[m] >= r_ref).mean())
        ax.hist(np.clip(r[m], 0, xmax), bins=bins, density=True, histtype="step", linewidth=2, color=c,
                label=f"{name} (n={int(m.sum())}, ≥{r_ref:g}px: {100 * frac:.0f}%)")
    ax.axvline(r_ref, color=TEXT_2, linestyle="--", linewidth=1)
    ax.set_xlim(0, xmax)
    style(ax, f"Iteration {it}: projected radius distribution", "mean projected radius over valid views (px)", "density")
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT, title="n = plotted sample", title_fontsize=8)
    save(fig, os.path.join(out_dir, f"{it:06d}_radius_hist.png"))


def plot_scale_ratio(rec, out_dir):
    p, it = rec["points"], rec["iteration"]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(p["similarity"], p["scale_ratio"], s=2, alpha=0.3, linewidths=0, color=INTERVAL_COLORS[2], rasterized=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style(ax, f"Iteration {it}", "similarity score S_k", "scale ratio (min/max world scale)")
    save(fig, os.path.join(out_dir, f"{it:06d}_scale_ratio.png"))
    return float(np.corrcoef(p["similarity"], p["scale_ratio"])[0, 1])


def summary_lines(rec):
    s = rec["summary"]
    if not s or not s.get("diag_available"):
        return [f"[{rec['iteration']}] no summary"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    g = s["per_interval_weighted"]
    lines = [f"[{rec['iteration']}] valid={s['n_valid']}  large(>= {s['radius_ref']:g}px)={pct(s['frac_large_valid'])}",
             f"    G1/G2/G3 with weight    = {s['weighted']['counts']}",
             f"    G1/G2/G3 without weight = {s['unweighted']['counts']}",
             f"    large fraction G1/G2/G3 = {pct(g['G1']['frac_large'])} / {pct(g['G2']['frac_large'])} / {pct(g['G3']['frac_large'])}",
             f"    G3 only by scale weight = {pct(s['g3_only_by_scale_weight'])},  "
             f"G1..G3 only by scale weight = {pct(s['candidates_only_by_scale_weight'])}",
             f"    spearman(score, radius) with / without weight = "
             f"{s['spearman_weighted_vs_radius']} / {s['spearman_unweighted_vs_radius']}",
             f"    transition (rows: with weight none,G1,G2,G3 / cols: without) = {s['transition_weighted_x_unweighted']}"]
    return lines


def plot_similarity_scale_scatter(model_path, radius_ref=None):
    records = load_records(model_path)
    if not records:
        print(f"No similarity-scale statistics found in {model_path}")
        return
    out_dir = os.path.join(model_path, "similarity_scale_scatter")
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for rec in records:
        p = rec["points"]
        r_ref = radius_ref or ((rec["summary"] or {}).get("radius_ref") or 20.0)
        if "radius" in p:
            plot_radius(rec, out_dir, r_ref)
            plot_raw_vs_weighted(rec, out_dir, r_ref)
            plot_radius_hist(rec, out_dir, r_ref)
        if "scale_ratio" in p:
            corr = plot_scale_ratio(rec, out_dir)
            lines.append(f"[{rec['iteration']}] corr(similarity, scale_ratio) = {corr:.4f}")
        lines.extend(summary_lines(rec))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--radius_ref", type=float, default=None, help="R_max line (default: value stored in the summary)")
    args = parser.parse_args()
    plot_similarity_scale_scatter(args.model_path, args.radius_ref)
