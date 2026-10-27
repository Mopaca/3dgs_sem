#
# CPU checks for the decisive-experiment code (no GPU, no CUDA extensions needed).
#
#   python tests/test_decisive_cpu.py
#
# 1. Full densification step: scene/gaussian_model.py of the reference commit (343c470) vs the
#    current file, both loaded with "cuda" -> "cpu", same inputs and seed. With the default config
#    the resulting Gaussians must be bit-identical, also when step statistics/snapshots are on.
# 2. accumulate_similarity("mean") equals the previous implementation; min/max aggregation.
# 3. Controls (random_matched / radius_matched) pick exactly sum(K_m) valid Gaussians.
# 4. [9]-style threshold + cap rule.
# 5. SSIM map averages to the training SSIM value; l1 / l1_blur ranges.
# 6. resolve_sem_config validation (parsed from train_sim.py without importing CUDA modules).
#

import ast
import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REFERENCE_COMMIT = "343c470"

# ---- stubs for CUDA-only / optional modules imported by scene/gaussian_model.py ----
_knn = types.ModuleType("simple_knn")
_knn_c = types.ModuleType("simple_knn._C")
_knn_c.distCUDA2 = lambda pts: torch.full((pts.shape[0],), 0.01)
_knn._C = _knn_c
sys.modules.setdefault("simple_knn", _knn)
sys.modules.setdefault("simple_knn._C", _knn_c)
_ply = types.ModuleType("plyfile")
_ply.PlyData = object
_ply.PlyElement = object
sys.modules.setdefault("plyfile", _ply)

from utils.graphics_utils import BasicPointCloud  # noqa: E402
from utils import sem_select  # noqa: E402
from utils.signal_maps import ssim_map, GuidanceMaps  # noqa: E402
from utils.loss_utils import ssim as train_ssim  # noqa: E402

FAILED = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


def _cpu_src(source):
    return (source.replace("\r\n", "\n").replace('device="cuda"', 'device="cpu"')
            .replace("device='cuda'", "device='cpu'").replace(".cuda()", ".cpu()"))


# utils.general_utils allocates on CUDA (build_rotation, build_scaling_rotation): register a CPU copy
with open(os.path.join(ROOT, "utils", "general_utils.py"), encoding="utf-8") as _f:
    _gu = types.ModuleType("utils.general_utils")
    _gu.__file__ = os.path.join(ROOT, "utils", "general_utils.py")
    exec(compile(_cpu_src(_f.read()), _gu.__file__, "exec"), _gu.__dict__)
    sys.modules["utils.general_utils"] = _gu


def cpu_module(source, name):
    src = _cpu_src(source)
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(ROOT, "scene", name + ".py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def reference_source():
    try:
        return subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:scene/gaussian_model.py"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return None


TRAIN_ARGS = SimpleNamespace(percent_dense=0.01, position_lr_init=1.6e-4, position_lr_final=1.6e-6,
                             position_lr_delay_mult=0.01, position_lr_max_steps=30000, feature_lr=0.0025,
                             opacity_lr=0.025, scaling_lr=0.005, rotation_lr=0.001, exposure_lr_init=0.01,
                             exposure_lr_final=0.001, exposure_lr_delay_steps=0, exposure_lr_delay_mult=0.0,
                             iterations=30000)


def make_model(mod, n=4000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    pcd = BasicPointCloud(points=rng.uniform(-1, 1, (n, 3)), colors=rng.uniform(0, 1, (n, 3)),
                          normals=np.zeros((n, 3)))
    m = mod.GaussianModel(3)
    m.create_from_pcd(pcd, [SimpleNamespace(image_name=f"im{i}") for i in range(4)], 1.0)
    m.training_setup(TRAIN_ARGS)
    with torch.no_grad():  # spread the sizes so that both clone and split happen
        m._scaling.copy_(torch.log(torch.exp(torch.empty_like(m._scaling).uniform_(-6.5, -3.0))))
    # one optimizer step so that Adam states exist (exercises the state-copy paths)
    for p in [m._xyz, m._features_dc, m._features_rest, m._opacity, m._scaling, m._rotation]:
        p.grad = torch.randn_like(p) * 1e-3
    m.optimizer.step()
    m.optimizer.zero_grad(set_to_none=True)
    return m


def random_step_inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    count = torch.randint(0, 6, (n, 1), generator=g).float()
    score = torch.rand((n, 1), generator=g) * 0.4 + 0.5
    score[torch.rand((n, 1), generator=g) < 0.02] *= 0.3          # a low tail
    return {
        "xyz_gradient_accum": torch.rand((n, 1), generator=g) * 6e-4 * count,
        "denom": count.clone(),
        "sim_score_accum": score * count,
        "sim_view_count": count,
        "sim_radius_accum": torch.rand((n, 1), generator=g) * 40 * count,
        "max_radii2D": torch.rand((n,), generator=g) * 40,
        "radii": (torch.rand((n,), generator=g) * 40).int(),
    }


def apply_inputs(m, inp):
    m.xyz_gradient_accum = inp["xyz_gradient_accum"].clone()
    m.denom = inp["denom"].clone()
    m.sim_score_accum = inp["sim_score_accum"].clone()
    m.sim_view_count = inp["sim_view_count"].clone()
    m.max_radii2D = inp["max_radii2D"].clone()
    if hasattr(m, "sim_radius_accum") and m.sim_radius_accum.numel() == m.sim_view_count.numel():
        m.sim_radius_accum = inp["sim_radius_accum"].clone()


def model_state(m):
    return {k: getattr(m, k).detach().clone() for k in
            ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_similarity_aux"]}


def test_full_step_equivalence():
    print("[1] full densification step vs reference commit")
    ref_src = reference_source()
    if ref_src is None:
        print(f"  skip (git history with {REFERENCE_COMMIT} not available)")
        return
    old_mod = cpu_module(ref_src, "gaussian_model_ref")
    with open(os.path.join(ROOT, "scene", "gaussian_model.py"), encoding="utf-8") as f:
        new_mod = cpu_module(f.read(), "gaussian_model_new")

    for collect in (False, True):
        old = make_model(old_mod)
        new = make_model(new_mod)
        check(all(torch.equal(a, b) for a, b in zip(model_state(old).values(), model_state(new).values())),
              f"identical initial models (collect_stats={collect})")
        extent = 2.0
        for step in range(3):
            n = old.get_xyz.shape[0]
            inp = random_step_inputs(n, seed=100 + step)
            apply_inputs(old, inp)
            apply_inputs(new, inp)

            torch.manual_seed(1234 + step)
            old.finalize_similarity_score()
            old.densify_and_prune_by_similarity(2e-4, 0.005, extent, 20, inp["radii"], 3100 + step * 100, model_path=None)

            torch.manual_seed(1234 + step)
            new.finalize_similarity_score(agg="mean")
            stats = new.densify_and_prune_by_similarity(
                2e-4, 0.005, extent, 20, inp["radii"], 3100 + step * 100, model_path=None,
                sem_cfg={"sample_ratio": 0.2, "collect_stats": collect, "want_snapshot": collect})

            so, sn = model_state(old), model_state(new)
            same = all(so[k].shape == sn[k].shape and torch.equal(so[k], sn[k]) for k in so)
            check(same, f"step {step}: identical Gaussians after densify+prune "
                        f"(N {n} -> {new.get_xyz.shape[0]}, collect_stats={collect})")
            check(new.sim_radius_accum.shape[0] == new.get_xyz.shape[0], f"step {step}: sim_radius_accum length tracks N")
            if collect:
                check(stats is not None and stats["n_union"] == stats["n_grad"] + stats["n_sem"] - stats["n_inter"],
                      f"step {step}: stats consistent (n_sem={stats['n_sem']}, n_sem_only={stats['n_sem_only']}, "
                      f"sem radius>=20 frac={stats['radius_sem'].get('frac_ge_ref')})")
                check("_snapshot" in stats and len(stats["_snapshot"]["score"]) > 0, f"step {step}: snapshot arrays present")
            old.reset_similarity_accum()
            new.reset_similarity_accum()

    # other modes run and keep the bookkeeping consistent
    for cfg in [{"select": "random_matched"}, {"select": "radius_matched"},
                {"mode": "r9", "r9_threshold": 0.05}, {"split": "mask2"}]:
        m = make_model(new_mod)
        inp = random_step_inputs(m.get_xyz.shape[0], seed=7)
        apply_inputs(m, inp)
        m.finalize_similarity_score(agg="max" if cfg.get("mode") == "r9" else "mean")
        st = m.densify_and_prune_by_similarity(2e-4, 0.005, 2.0, 20, inp["radii"], 3100, model_path=None,
                                               sem_cfg=dict(cfg, collect_stats=True))
        ok = all(t.shape[0] == m.get_xyz.shape[0] for t in
                 [m.sim_score_accum, m.sim_view_count, m.similarity_score, m.sim_radius_accum, m._similarity_aux])
        check(ok, f"{cfg}: runs, n_sem={st['n_sem']}, state lengths consistent")


def test_accumulate():
    print("[2] accumulate_similarity")
    with open(os.path.join(ROOT, "scene", "gaussian_model.py"), encoding="utf-8") as f:
        mod = cpu_module(f.read(), "gaussian_model_acc")
    m = mod.GaussianModel(3)
    n = 50
    m.sim_score_accum = torch.zeros((n, 1)); m.sim_view_count = torch.zeros((n, 1)); m.sim_radius_accum = torch.zeros((n, 1))
    ref_acc = torch.zeros((n, 1)); ref_cnt = torch.zeros((n, 1))
    g = torch.Generator().manual_seed(0)
    views = [(torch.rand((n, 1), generator=g), (torch.rand((n, 1), generator=g) > 0.3).float()) for _ in range(7)]
    for s, v in views:
        m.accumulate_similarity(s, v, agg="mean", radius=torch.ones((n, 1)) * 3)
        ref_acc += s * v; ref_cnt += v
    check(torch.equal(m.sim_score_accum, ref_acc) and torch.equal(m.sim_view_count, ref_cnt), "mean == previous implementation")
    check(torch.allclose(m.get_mean_projected_radius()[ref_cnt.squeeze(1) > 0], torch.tensor(3.0)), "mean projected radius")
    for agg, fn in (("min", torch.minimum), ("max", torch.maximum)):
        m.sim_score_accum = torch.zeros((n, 1)); m.sim_view_count = torch.zeros((n, 1))
        for s, v in views:
            m.accumulate_similarity(s, v, agg=agg)
        m.finalize_similarity_score(agg=agg)
        exp = torch.full((n, 1), float("nan"))
        for s, v in views:
            exp = torch.where(v > 0, torch.where(torch.isnan(exp), s, fn(exp, s)), exp)
        exp = torch.nan_to_num(exp, nan=0.0)
        check(torch.equal(m.similarity_score, exp), f"{agg} aggregation over valid views")


def test_controls_and_r9():
    print("[3] controls / [4] [9]-style rule")
    g = torch.Generator().manual_seed(3)
    score = torch.rand(20000, generator=g) * 0.3 + 0.6
    score[:300] *= 0.4
    valid = torch.rand(20000, generator=g) > 0.2
    radius = torch.rand(20000, generator=g) * 50
    plan = sem_select.stratified_plan(score, valid, 0.3)
    k = sum(plan["num_samples"])
    torch.manual_seed(0); ms, _ = sem_select.select_stratified(score, valid, 0.3)
    torch.manual_seed(0); mr, _ = sem_select.select_random_matched(score, valid, 0.3)
    mrad, _ = sem_select.select_radius_matched(score, valid, 0.3, radius)
    check(int(ms.sum()) == k, f"stratified selects sum(K)={k} (K={plan['num_samples']}, counts={plan['counts']})")
    check(int(mr.sum()) == k and bool((mr & ~valid).sum() == 0), "random_matched: same count, valid only")
    check(int(mrad.sum()) == k and bool((mrad & ~valid).sum() == 0), "radius_matched: same count, valid only")
    thr = radius[mrad].min()
    check(bool((radius[valid & ~mrad] <= thr).all()), "radius_matched picks the largest radii")
    err = torch.rand(20000, generator=g)
    m9, info = sem_select.select_threshold_cap(err, valid, 0.5, 0.05)
    check(info["cap_bound"] and int(m9.sum()) == int(0.05 * 20000), f"cap binds at 5% ({info})")
    m9b, info_b = sem_select.select_threshold_cap(err, valid, 0.99, 0.05)
    check(not info_b["cap_bound"] and int(m9b.sum()) == info_b["n_candidates"], f"below cap keeps all candidates ({info_b})")
    rho = sem_select.spearman(score, score * 2 + 1)
    check(rho is not None and abs(rho - 1.0) < 1e-6, "spearman of a monotone transform = 1")


def test_maps():
    print("[5] guidance maps")
    g = torch.Generator().manual_seed(0)
    gt = torch.rand((3, 64, 80), generator=g)
    render = (gt + 0.1 * torch.randn((3, 64, 80), generator=g)).clamp(0, 1)
    smap = ssim_map(render, gt)
    check(smap.shape == (1, 64, 80), "ssim map shape [1,H,W]")
    check(abs(smap.mean().item() - train_ssim(render, gt).item()) < 1e-5, "ssim map mean == training ssim")
    gm = GuidanceMaps(["ssim", "l1", "l1_blur"], device="cpu")
    for s in ("ssim", "l1", "l1_blur"):
        mp = gm.goodness(s, render, gt)
        check(mp.shape == (1, 64, 80) and mp.max().item() <= 1.0 + 1e-6, f"{s}: shape and upper bound")
    check(torch.allclose(gm.goodness("l1", gt, gt), torch.ones(1, 64, 80)), "l1 goodness of a perfect render = 1")


def test_resolve_config():
    print("[6] resolve_sem_config")
    with open(os.path.join(ROOT, "train_sim.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_sem_config"
            or isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("SEM_") for t in n.targets)]
    ns = {"json": json, "SIGNALS": ("dino", "ssim", "l1", "l1_blur")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "train_sim_subset", "exec"), ns)
    base = dict(sem_mode="ours", sem_signal="dino", sem_shadow="none", sem_select="stratified", sem_split="adc",
                sem_view_norm="auto", sem_view_agg="auto", sem_sample_ratio=0.3, sem_r9_threshold=0.1, sem_r9_cap=0.05,
                sem_radius_max=20.0, sem_lambda_scale=0.3, sem_log=False)
    cfg = ns["resolve_sem_config"](SimpleNamespace(**base))
    check(cfg["view_norm"] == "mean" and cfg["view_agg"] == "mean", "ours: auto -> mean / mean")
    cfg = ns["resolve_sem_config"](SimpleNamespace(**dict(base, sem_mode="r9", sem_lambda_scale=0.0)))
    check(cfg["view_norm"] == "sum" and cfg["view_agg"] == "max", "r9: auto -> sum / max")
    for bad in [dict(sem_view_norm="sum"), dict(sem_view_agg="max"), dict(sem_signal="vgg"),
                dict(sem_mode="r9", sem_select="radius_matched")]:
        try:
            ns["resolve_sem_config"](SimpleNamespace(**dict(base, **bad)))
            check(False, f"rejects {bad}")
        except ValueError:
            check(True, f"rejects {bad}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    test_full_step_equivalence()
    test_accumulate()
    test_controls_and_r9()
    test_maps()
    test_resolve_config()
    print("\nALL CHECKS PASSED" if not FAILED else f"\n{len(FAILED)} CHECK(S) FAILED")
    sys.exit(1 if FAILED else 0)
