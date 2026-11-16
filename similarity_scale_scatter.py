#
# Similarity vs projected radius / scale weight plots (review item 4).
#
# Input (either):
#   <model>/similarity_scale_statistics.json          from train_sim_rendersim.save_similarity_scale_statistics_json
#   <model>/sem_snapshots/scale_weight_iter_*.npz     from train_sim.py --sem_log (summary taken from sem_log.jsonl)
#
# Output: <model>/similarity_scale_scatter/
#   <iter>_radius.png          similarity (with scale weight) vs projected radius, colored by interval G1..G3
#   <iter>_raw_vs_weighted.png similarity without vs with the scale weight, one panel per boundary (μ−σ, μ−2σ,
#                              μ−3σ); lower-right box = beyond the boundary only because of the weight
#   <iter>_radius_hist.png     projected-radius distribution: all valid / G1–G3 with and without weight / G3
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


def boundary_stats(summary):
    """
    Per-boundary effect of the scale weight, from the full transition matrix
    T[w][r] (rows: interval with weight, cols: without; order none, G1, G2, G3).

    For boundary k (k=1: candidate boundary mean-sigma, k=2: mean-2sigma, k=3: mean-3sigma):
      added_k   : Gaussians at or beyond boundary k with the weight but not without it
      removed_k : Gaussians at or beyond boundary k without the weight but not with it
    and, per weighted interval G_m, the fraction that is not a candidate at all without the weight.
    """
    T = np.asarray((summary or {}).get("transition_weighted_x_unweighted") or [], dtype=float)
    if T.shape != (4, 4):
        return None
    out = {"boundary": {}, "interval_not_candidate_raw": {}}
    for k in (1, 2, 3):
        with_w = T[k:, :].sum()
        without_w = T[:, k:].sum()
        added = T[k:, :k].sum()
        removed = T[:k, k:].sum()
        out["boundary"][k] = {
            "n_with": int(with_w), "n_without": int(without_w),
            "added": int(added), "removed": int(removed),
            "added_frac": (added / with_w) if with_w else None,      # of the weighted set
            "removed_frac": (removed / without_w) if without_w else None,  # of the unweighted set
        }
    fracs = []
    for m in (1, 2, 3):
        row = T[m].sum()
        f = (T[m, 0] / row) if row else None
        out["interval_not_candidate_raw"][m] = f
        if f is not None:
            fracs.append(f)
    # stratified sampling draws (about) the same number from every non-empty interval
    out["selection_not_candidate_raw"] = float(np.mean(fracs)) if fracs else None
    return out


BOUNDARY_NAMES = {1: "G1 (candidate) boundary  μ−σ", 2: "G2 boundary  μ−2σ", 3: "G3 boundary  μ−3σ"}


def plot_raw_vs_weighted(rec, out_dir, r_ref):
    """
    One panel per boundary (μ−σ, μ−2σ, μ−3σ). Horizontal line: boundary of the weighted score
    (used for selection); vertical line: the same boundary of the unweighted score.
    Shaded lower-right box = beyond the boundary only because of the scale weight.
    Shaded upper-left box  = beyond the boundary without the weight, pushed out by the weight.
    Points are coloured by their weighted interval. Percentages use all valid Gaussians.
    """
    p, it = rec["points"], rec["iteration"]
    x, y = p["similarity_raw"], p["similarity"]
    lab = p["interval"]
    s = rec["summary"] or {}
    bw, br = boundaries(s.get("weighted")), boundaries(s.get("unweighted"))
    bs = boundary_stats(s)
    lo = max(-1.0, float(min(np.percentile(x, 0.1), np.percentile(y, 0.1))) - 0.05)  # zoom to the data
    w_min = float(np.min(p["scale_weight"])) if "scale_weight" in p else None

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    for k, ax in zip((1, 2, 3), axes):
        if bw and br:
            ax.fill_between([br[k - 1], 1], lo, bw[k - 1], color=CONTRAST, alpha=0.10, linewidth=0)
            ax.fill_between([lo, br[k - 1]], bw[k - 1], 1, color=INTERVAL_COLORS[1], alpha=0.12, linewidth=0)
        for m in (0, 1, 2, 3):  # rare low intervals drawn last
            mk = lab == m
            if not mk.any():
                continue
            ax.scatter(x[mk], y[mk], s=2 if m == 0 else 3, alpha=0.3 if m == 0 else 0.7, linewidths=0,
                       color=NEUTRAL if m == 0 else INTERVAL_COLORS[m], rasterized=True,
                       label=("not a candidate" if m == 0 else f"G{m} (with weight)") + f"  n={int(mk.sum())}")
        ax.plot([lo, 1], [lo, 1], color=TEXT_2, linewidth=1, label="no change (y = x)")
        if w_min is not None and w_min < 0.999:
            ax.plot([lo, 1], [lo * w_min, w_min], color=TEXT_2, linewidth=1, linestyle="--",
                    label=f"largest reduction (y = {w_min:.2f}·x)")
        if bw and br:
            ax.axhline(bw[k - 1], color=CONTRAST, linewidth=1.2, linestyle=":")
            ax.axvline(br[k - 1], color=CONTRAST, linewidth=1.2, linestyle=":")
            ax.text(lo, bw[k - 1], " with weight", color=TEXT_2, fontsize=8, ha="left", va="bottom")
            ax.text(br[k - 1], 1.0, "without weight ", color=TEXT_2, fontsize=8, rotation=90, ha="right", va="top")
        if bs:
            b = bs["boundary"][k]
            pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
            ax.text(0.98, 0.02,
                    f"beyond only because of weight: {b['added']:,}\n({pct(b['added_frac'])} of {b['n_with']:,} with weight)",
                    color=TEXT, fontsize=8.5, ha="right", va="bottom", transform=ax.transAxes)
            # the region above y = x is always empty, so the upper-left text never covers points
            ax.text(0.03, 0.60,
                    f"pushed out by weight: {b['removed']:,}\n({pct(b['removed_frac'])} of {b['n_without']:,} without weight)",
                    color=TEXT, fontsize=8.5, ha="left", va="top", transform=ax.transAxes)
        ax.set_xlim(lo, 1)
        ax.set_ylim(lo, 1)
        style(ax, BOUNDARY_NAMES[k], "similarity without scale weight (same views)", "similarity with scale weight S_k")
        if k == 1:
            ax.legend(loc="upper left", fontsize=7.5, frameon=False, markerscale=3, labelcolor=TEXT,
                      title="n = plotted sample", title_fontsize=7.5)
    title = f"Iteration {it}: score without vs with scale weight, per interval boundary"
    if bs and bs["selection_not_candidate_raw"] is not None:
        f = bs["interval_not_candidate_raw"]
        title += (f"  |  not a candidate without weight: G1 {100 * (f[1] or 0):.0f}%, G2 {100 * (f[2] or 0):.0f}%, "
                  f"G3 {100 * (f[3] or 0):.0f}%  →  ≈{100 * bs['selection_not_candidate_raw']:.0f}% of M_sem")
    fig.suptitle(title, fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, f"{it:06d}_raw_vs_weighted.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_radius_hist(rec, out_dir, r_ref):
    p, it = rec["points"], rec["iteration"]
    r = p["radius"]
    xmax = max(2 * r_ref, float(np.percentile(r, 99.5)))
    bins = np.linspace(0, xmax, 60)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = [("all valid", np.ones_like(r, dtype=bool), NEUTRAL),
              ("G1–G3 with scale weight", p["interval"] >= 1, INTERVAL_COLORS[2]),
              ("G1–G3 without scale weight", p["interval_raw"] >= 1, CONTRAST),
              ("G3 with scale weight", p["interval"] == 3, INTERVAL_COLORS[3])]
    for name, m, c in series:
        if m.sum() < 2:
            continue
        frac = float((r[m] >= r_ref).mean())
        ax.hist(np.clip(r[m], 0, xmax), bins=bins, density=True, histtype="step", linewidth=2, color=c,
                label=f"{name} (n={int(m.sum())}, ≥{r_ref:g}px: {100 * frac:.0f}%)")
    ax.axvline(r_ref, color=TEXT_2, linestyle="--", linewidth=1)
    ax.set_xlim(0, xmax)
    style(ax, f"Iteration {it}: projected radius distribution", "mean projected radius over valid views (px)", "density")
    ax.legend(fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95, labelcolor=TEXT,
              title="n = plotted sample", title_fontsize=8, loc="upper right")
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
    bs = boundary_stats(s)
    if bs:
        for k in (1, 2, 3):
            b = bs["boundary"][k]
            lines.append(f"    boundary mu-{k}sigma: with weight {b['n_with']}, without {b['n_without']} | "
                         f"added only by weight {b['added']} ({pct(b['added_frac'])} of with-weight set), "
                         f"pushed out {b['removed']} ({pct(b['removed_frac'])} of without-weight set)")
        f = bs["interval_not_candidate_raw"]
        lines.append(f"    not a candidate without weight: G1 {pct(f[1])}, G2 {pct(f[2])}, G3 {pct(f[3])}"
                     f"  ->  expected share of M_sem (equal draws per interval) {pct(bs['selection_not_candidate_raw'])}")
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
