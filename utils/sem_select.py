#
# Selection rules for the semantic densification set M_sem.
#
# All functions are pure (tensors in, mask + info out) so they can be unit-tested on CPU.
#
#   select_stratified       : the proposed distribution-based stratified sampling. This is a
#                             line-by-line port of GaussianModel.build_semantic_densify_mask
#                             (same operations, same order of torch.randperm calls), so with the
#                             same RNG state it returns the same mask.
#   select_random_matched   : control. Same total count as the stratified plan, drawn uniformly
#                             at random from the valid candidates.
#   select_radius_matched   : control. Same total count as the stratified plan, the candidates
#                             with the largest projected radius (Mini-Splatting "blur split"-like).
#   select_threshold_cap    : [9]-style rule. Error score above a threshold, at most
#                             cap_ratio * N per step, keeping the highest scores when capped.
#
# Logging helpers (summaries, Spearman) never touch the global RNG: subsampling is strided,
# so turning logging on or off does not change the training randomness.
#

import torch


# ---------------------------------------------------------------------------
# Proposed rule
# ---------------------------------------------------------------------------

def stratified_plan(sim_score, sim_valid_mask, global_sample_ratio):
    """
    Interval masks and per-interval sample sizes of the stratified sampling, without sampling.

    Returns a dict:
        masks        : [mask_high, mask_mid, mask_low]  (G1, G2, G3)
        counts       : [c1, c2, c3] (int)
        num_samples  : [K1, K2, K3] (int, 0 for empty intervals)
        mean, std    : score statistics over the valid candidates (float or None)
    """
    plan = {"masks": None, "counts": [0, 0, 0], "num_samples": [0, 0, 0], "mean": None, "std": None}

    if not sim_valid_mask.any():
        return plan

    sim_valid = sim_score[sim_valid_mask]
    sim_mean = sim_valid.mean()
    sim_std = sim_valid.std(unbiased=False)
    plan["mean"] = sim_mean.item()
    plan["std"] = sim_std.item()

    mask_high = (sim_valid_mask & (sim_score >= sim_mean - 2.0 * sim_std) & (sim_score < sim_mean - 1.0 * sim_std))
    mask_mid = (sim_valid_mask & (sim_score >= sim_mean - 3.0 * sim_std) & (sim_score < sim_mean - 2.0 * sim_std))
    mask_low = (sim_valid_mask & (sim_score < sim_mean - 3.0 * sim_std))
    masks = [mask_high, mask_mid, mask_low]
    plan["masks"] = masks

    counts = torch.tensor([m.sum().item() for m in masks], dtype=torch.float32, device=sim_score.device)
    plan["counts"] = [int(c) for c in counts.tolist()]

    total_candidate = counts.sum()
    if total_candidate <= 0:
        return plan

    # Same arithmetic as the original code. The denominator (all Gaussians) cancels out
    # in the normalization below; it is kept only to reproduce the original floating point ops.
    region_ratio = counts / sim_score.numel()
    valid_region = counts > 0

    inv_ratio = torch.zeros_like(region_ratio)
    inv_ratio[valid_region] = (1.0 / region_ratio[valid_region])
    if not inv_ratio.sum() > 0:
        return plan
    sample_ratio = (inv_ratio / inv_ratio.sum())
    sample_ratio = (sample_ratio * global_sample_ratio).clamp(max=1.0)

    num_samples = []
    for mask, ratio in zip(masks, sample_ratio):
        num_candidate = int(mask.sum().item())
        if num_candidate == 0:
            num_samples.append(0)
            continue
        num_sample = max(1, int(num_candidate * ratio.item()))
        num_sample = min(num_sample, num_candidate)
        num_samples.append(num_sample)
    plan["num_samples"] = num_samples
    return plan


def select_stratified(sim_score, sim_valid_mask, global_sample_ratio):
    plan = stratified_plan(sim_score, sim_valid_mask, global_sample_ratio)
    mask_out = torch.zeros_like(sim_score, dtype=torch.bool)
    if plan["masks"] is None:
        return mask_out, plan

    for mask, num_sample in zip(plan["masks"], plan["num_samples"]):
        if num_sample == 0:
            continue
        candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        num_candidate = candidate_indices.numel()
        perm = torch.randperm(num_candidate, device=sim_score.device)
        selected_indices = candidate_indices[perm[:num_sample]]
        mask_out[selected_indices] = True
    return mask_out, plan


# ---------------------------------------------------------------------------
# Controls (same count as the stratified plan)
# ---------------------------------------------------------------------------

def select_random_matched(sim_score, sim_valid_mask, global_sample_ratio):
    plan = stratified_plan(sim_score, sim_valid_mask, global_sample_ratio)
    mask_out = torch.zeros_like(sim_score, dtype=torch.bool)
    k_total = int(sum(plan["num_samples"]))
    if k_total == 0:
        return mask_out, plan
    candidate_indices = torch.nonzero(sim_valid_mask, as_tuple=False).squeeze(1)
    k_total = min(k_total, candidate_indices.numel())
    perm = torch.randperm(candidate_indices.numel(), device=sim_score.device)
    mask_out[candidate_indices[perm[:k_total]]] = True
    return mask_out, plan


def select_radius_matched(sim_score, sim_valid_mask, global_sample_ratio, radius):
    """`radius`: [N] projected radius per Gaussian (mean over the valid views of the interval)."""
    plan = stratified_plan(sim_score, sim_valid_mask, global_sample_ratio)
    mask_out = torch.zeros_like(sim_score, dtype=torch.bool)
    k_total = int(sum(plan["num_samples"]))
    if k_total == 0:
        return mask_out, plan
    candidate_indices = torch.nonzero(sim_valid_mask, as_tuple=False).squeeze(1)
    k_total = min(k_total, candidate_indices.numel())
    top = torch.topk(radius[candidate_indices].float(), k_total, largest=True, sorted=False).indices
    mask_out[candidate_indices[top]] = True
    return mask_out, plan


# ---------------------------------------------------------------------------
# [9]-style rule
# ---------------------------------------------------------------------------

def select_threshold_cap(err_score, valid_mask, threshold, cap_ratio):
    """
    [9]-style: grow Gaussians whose error exceeds `threshold`, at most cap_ratio * N per step
    (N = current number of Gaussians), keeping the highest errors when the cap binds.
    """
    n_total = err_score.numel()
    mask_out = torch.zeros_like(err_score, dtype=torch.bool)
    candidates = valid_mask & (err_score > threshold)
    n_candidates = int(candidates.sum().item())
    n_cap = int(cap_ratio * n_total)
    info = {"n_candidates": n_candidates, "n_cap": n_cap, "cap_bound": False}
    if n_candidates == 0 or n_cap <= 0:
        return mask_out, info
    candidate_indices = torch.nonzero(candidates, as_tuple=False).squeeze(1)
    if n_candidates > n_cap:
        info["cap_bound"] = True
        top = torch.topk(err_score[candidate_indices], n_cap, largest=True, sorted=False).indices
        candidate_indices = candidate_indices[top]
    mask_out[candidate_indices] = True
    return mask_out, info


# ---------------------------------------------------------------------------
# Logging helpers (no global RNG)
# ---------------------------------------------------------------------------

def strided_subsample(x, max_n):
    n = x.numel()
    if n <= max_n:
        return x
    step = (n + max_n - 1) // max_n
    return x[::step]


def summarize(values, mask=None, ref=None, max_n=1_000_000):
    """mean / median / p90 (and fraction >= ref) of `values[mask]`; quantiles on a strided subsample."""
    v = values if mask is None else values[mask]
    v = v.float().flatten()
    out = {"n": int(v.numel())}
    if v.numel() == 0:
        return out
    out["mean"] = v.mean().item()
    vs = torch.sort(strided_subsample(v, max_n)).values
    m = vs.numel()
    out["median"] = vs[(m - 1) // 2].item()
    out["p90"] = vs[min(m - 1, int(0.9 * (m - 1)))].item()
    if ref is not None:
        out["frac_ge_ref"] = (v >= ref).float().mean().item()
    return out


def spearman(a, b, max_n=200_000):
    """Spearman rank correlation of two 1-D tensors (strided subsample, average ranks not used)."""
    a = strided_subsample(a.float().flatten(), max_n)
    b = strided_subsample(b.float().flatten(), max_n)
    n = a.numel()
    if n < 3:
        return None
    ra = torch.empty_like(a)
    rb = torch.empty_like(b)
    ra[torch.argsort(a)] = torch.arange(n, device=a.device, dtype=a.dtype)
    rb[torch.argsort(b)] = torch.arange(n, device=b.device, dtype=b.dtype)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom.item() == 0:
        return None
    return ((ra * rb).sum() / denom).item()
