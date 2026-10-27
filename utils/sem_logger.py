#
# Per-densification-step logging for the semantic densification experiments.
#
#   <model_path>/sem_log.jsonl        one JSON line per densification step
#   <model_path>/sem_snapshots/*.npz   per-Gaussian arrays at a few steps (strided subsample)
#
# Per-iteration observations (between two densification steps) are averaged and written
# together with the step statistics returned by GaussianModel.densify_and_prune_by_similarity.
# Nothing here uses the global RNG.
#

import json
import os

import numpy as np
import torch

from utils.sem_select import spearman


class SemLogger:
    def __init__(self, model_path, enabled=False, snapshot_iters=(1000, 3000, 6000, 9000, 12000, 14900),
                 snapshot_max_points=200_000, shadow_every=10):
        self.enabled = bool(enabled)
        self.model_path = model_path
        self.snapshot_iters = set(int(i) for i in snapshot_iters)
        self.snapshot_max_points = int(snapshot_max_points)
        self.shadow_every = max(1, int(shadow_every))
        self.log_path = os.path.join(model_path, "sem_log.jsonl")
        self.snapshot_dir = os.path.join(model_path, "sem_snapshots")
        self._reset_interval()
        if self.enabled:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            open(self.log_path, "w").close()  # truncate when a run is restarted

    def _reset_interval(self):
        self._n_views = 0
        self._neg_frac_sum = 0.0
        self._valid_frac_sum = 0.0
        self._shadow_corr = []

    def wants_snapshot(self, iteration):
        return self.enabled and int(iteration) in self.snapshot_iters

    @torch.no_grad()
    def observe_view(self, iteration, raw_view_score, valid_mask, shadow_view_score=None):
        """
        raw_view_score: [N, 1] view score before the scale weight and clamp
        valid_mask:     [N, 1] float, visible and contribution sum > tau_v
        """
        if not self.enabled:
            return
        valid = valid_mask.squeeze(1) > 0
        n_valid = int(valid.sum().item())
        self._n_views += 1
        self._valid_frac_sum += n_valid / max(1, valid.numel())
        if n_valid == 0:
            return
        raw = raw_view_score.squeeze(1)[valid]
        self._neg_frac_sum += (raw < 0).float().mean().item()
        if shadow_view_score is not None and iteration % self.shadow_every == 0:
            rho = spearman(raw, shadow_view_score.squeeze(1)[valid])
            if rho is not None:
                self._shadow_corr.append(rho)

    def log_step(self, iteration, stats, extra=None):
        if not self.enabled:
            self._reset_interval()
            return
        record = {"iteration": int(iteration)}
        if self._n_views > 0:
            record["views_in_interval"] = self._n_views
            record["raw_negative_frac"] = self._neg_frac_sum / self._n_views
            record["valid_frac_per_view"] = self._valid_frac_sum / self._n_views
        if self._shadow_corr:
            record["shadow_spearman_mean"] = float(np.mean(self._shadow_corr))
            record["shadow_spearman_n"] = len(self._shadow_corr)
        if extra:
            record.update(extra)

        snapshot = None
        if stats:
            stats = dict(stats)
            snapshot = stats.pop("_snapshot", None)
            record.update(stats)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if snapshot is not None:
            path = os.path.join(self.snapshot_dir, f"iter_{int(iteration):05d}.npz")
            np.savez_compressed(path, **snapshot)

        self._reset_interval()
