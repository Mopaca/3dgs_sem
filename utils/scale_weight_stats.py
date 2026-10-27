#
# Scale-weight diagnostics (review item 4).
#
# Call at a densification step after GaussianModel.finalize_similarity_score() and before
# densification. It compares the score used for selection (with the scale weight) with the
# counterfactual score from the same views without the scale weight, and relates both to the
# projected radius that the scale weight acts on.
#
#   radius        : mean projected radius over the valid views of the interval (px). This is the
#                   rasterizer radius ceil(3 * sqrt(lambda_max)) of the projected 2D covariance,
#                   i.e. the quantity normalized by R_min / R_max in the scale weight.
#   scale_weight  : mean of 1 - lambda_s * clip((R - R_min) / (R_max - R_min), 0, 1) over the same views
#   similarity_raw: mean of clip(view score, 0, 1) over the same views (no scale weight)
#   interval      : 0 = valid but >= mean - std, 1..3 = G1..G3 (same definition as the selection),
#                   computed separately for the weighted and the unweighted score
#
# Requires the training loop to pass radius / scale_weight / raw_score to accumulate_similarity.
# No global RNG is used (subsampling is strided).
#

import torch

from utils.sem_select import stratified_plan, strided_subsample, spearman, summarize

INTERVALS = ["none", "G1", "G2", "G3"]


def interval_labels(score, valid):
    """-1 for invalid, 0 for valid but >= mean - std, 1..3 for G1..G3."""
    plan = stratified_plan(score, valid, 1.0)  # the sampling ratio is irrelevant for the masks
    labels = torch.zeros_like(score, dtype=torch.long)
    labels[~valid] = -1
    if plan["masks"] is not None:
        for i, m in enumerate(plan["masks"], start=1):
            labels[m] = i
    return labels, plan


@torch.no_grad()
def scale_weight_analysis(gaussians, radius_ref=20.0, max_points=100_000):
    """
    Returns (summary dict, points dict of numpy arrays for a strided subsample of the valid Gaussians).
    summary is None-filled if the diagnostic accumulators were not fed by the training loop.
    """
    count = gaussians.sim_view_count.squeeze(1)
    score = gaussians.get_similarity_score.squeeze(1).detach()
    valid = (count > 0) & (score >= 0)
    n_valid = int(valid.sum().item())

    diag_ok = gaussians._diag_ready(count.shape[0]) and n_valid > 0 and \
        bool((gaussians.sim_diag_accum[valid, 1] > 0).any().item())
    if not diag_ok:
        print("[scale_weight_analysis] diagnostic accumulators are empty: pass radius / scale_weight / raw_score "
              "to accumulate_similarity in the training loop")
        return {"n_valid": n_valid, "diag_available": False}, None

    raw = gaussians.get_unweighted_similarity()
    radius = gaussians.get_mean_projected_radius()
    weight = gaussians.get_mean_scale_weight()
    grads = (gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)).squeeze(1)
    grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
    scaling = gaussians.get_scaling.detach()
    scale_ratio = scaling.min(dim=1).values / (scaling.max(dim=1).values + 1e-12)

    lab_w, plan_w = interval_labels(score, valid)
    lab_r, plan_r = interval_labels(raw, valid)
    large = radius >= radius_ref

    # transition[i][j]: Gaussians in interval i with the scale weight and interval j without it
    lw = lab_w[valid].clamp_min(0)
    lr = lab_r[valid].clamp_min(0)
    transition = torch.bincount(lw * 4 + lr, minlength=16).view(4, 4).tolist()

    per_interval = {}
    for i, name in enumerate(INTERVALS):
        m = valid & (lab_w == i)
        n = int(m.sum().item())
        per_interval[name] = {
            "n": n,
            "frac_large": (large[m].float().mean().item() if n else None),
            "radius": summarize(radius, m),
            "scale_weight_mean": (weight[m].mean().item() if n else None),
            "score_drop_mean": ((raw[m] - score[m]).mean().item() if n else None),
        }

    def only_by_weight(target):
        """Fraction of weighted interval >= target whose unweighted interval is below target."""
        m = valid & (lab_w >= target)
        n = int(m.sum().item())
        if n == 0:
            return None
        return ((lab_r[m] < target).float().mean().item())

    summary = {
        "diag_available": True,
        "n_valid": n_valid,
        "radius_ref": float(radius_ref),
        "frac_large_valid": large[valid].float().mean().item(),
        "weighted": {"mean": plan_w["mean"], "std": plan_w["std"], "counts": plan_w["counts"]},
        "unweighted": {"mean": plan_r["mean"], "std": plan_r["std"], "counts": plan_r["counts"]},
        "per_interval_weighted": per_interval,
        "transition_weighted_x_unweighted": transition,   # rows/cols: none, G1, G2, G3
        "g3_only_by_scale_weight": only_by_weight(3),     # in G3 only because of the scale weight
        "candidates_only_by_scale_weight": only_by_weight(1),  # in G1..G3 only because of the scale weight
        "spearman_weighted_vs_radius": spearman(score[valid], radius[valid]),
        "spearman_unweighted_vs_radius": spearman(raw[valid], radius[valid]),
    }

    idx = strided_subsample(torch.nonzero(valid, as_tuple=False).squeeze(1), int(max_points))
    points = {
        "similarity": score[idx].float().cpu().numpy(),
        "similarity_raw": raw[idx].float().cpu().numpy(),
        "radius": radius[idx].float().cpu().numpy(),
        "scale_weight": weight[idx].float().cpu().numpy(),
        "interval": lab_w[idx].cpu().numpy(),
        "interval_raw": lab_r[idx].cpu().numpy(),
        "scale_ratio": scale_ratio[idx].float().cpu().numpy(),
        "grad": grads[idx].float().cpu().numpy(),
    }
    return summary, points


def format_summary(summary):
    if not summary.get("diag_available"):
        return "scale-weight diagnostics unavailable"
    g3 = summary["per_interval_weighted"]["G3"]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    return (f"valid={summary['n_valid']}, large(>= {summary['radius_ref']:g}px)={pct(summary['frac_large_valid'])} | "
            f"G1/G2/G3 with weight={summary['weighted']['counts']}, without={summary['unweighted']['counts']} | "
            f"G3 large={pct(g3['frac_large'])}, G3 only by scale weight={pct(summary['g3_only_by_scale_weight'])}")
