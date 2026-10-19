#!/usr/bin/env python3
"""Controlled MNIST digit-1 v/u initial-state sweep for the R6B.3 MSDL readout.

Scope: uses the repository's exact P3 static feature operator and MSDL
Izhikevich dynamics/readout learning rule. Other P3 weights are frozen. The
repository's conventional KMeans patch codebook substitutes for the missing
R6B.1 SNN-VQ checkpoint; this is NOT the full R6B.3 end-to-end experiment.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maskgit_b0_mnist012 import encode as patch_encode
from maskgit_b0_mnist012 import fit_codebook
import SNN_TRANSFORM_R6B3_HG_DYNAMICS_STAGE4_DIRECT_SPIKE as p3

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
VALUES = np.linspace(0.0, 1000.0, 10)
# Exactly the same sampled features, image split, target sequence, and frozen
# P3 weights for all conditions. Conditions vary ONLY initial E/M v and u.
@njit(cache=True)
def force_path(W, ve, ue, vm, um, ze, zm, F, T, indices, lr):
    """Preserve R6B.3's W0-frozen 27-force-step MSDL update."""
    V, nin = W.shape
    W0 = W.copy()
    acc = np.zeros((V, nin), dtype=np.float64)
    spikes_e = 0
    spikes_m = 0
    for k in range(len(indices)):
        ci = indices[k]
        x = F[ci]
        z = W0 @ x
        p = p3.softmax_numba(z)
        E, ne = p3.fast_izh_step(ve, ue, x, ze, p3.G_E, p3.CV_E, p3.CU_E)
        tar = np.zeros(V, dtype=np.float64)
        tar[T[ci]] = 1.0
        M, nm = p3.fast_izh_step(vm, um, tar, p, p3.G_M, p3.CV_M, p3.CU_M)
        spikes_e += ne
        spikes_m += nm
        for o in range(V):
            for j in range(nin):
                acc[o, j] += M[o] * E[j]
    for o in range(V):
        for j in range(nin):
            W[o, j] += lr * acc[o, j] / len(indices)
    return spikes_e, spikes_m


def data_file(repo: Path, allow_download: bool) -> tuple[Path, str]:
    path = repo / "mnist.npz"
    if path.is_file():
        return path, "repository/mnist.npz"
    if not allow_download:
        raise FileNotFoundError("mnist.npz is absent; pass --allow-download to fetch canonical Keras MNIST")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MNIST_URL, path)
    return path, MNIST_URL


def load_ones(path: Path, train_limit: int, val_limit: int, seed: int):
    with np.load(path, allow_pickle=False) as z:
        for k in ("x_train", "y_train", "x_test", "y_test"):
            if k not in z:
                raise ValueError("Not a standard MNIST archive: missing " + k)
        train_x = z["x_train"][z["y_train"] == 1].astype(np.float32) / 255.0
        val_x = z["x_test"][z["y_test"] == 1].astype(np.float32) / 255.0
    if len(train_x) < train_limit or len(val_x) < val_limit:
        raise ValueError("Not enough digit-1 samples for the requested train/eval subsets")
    rng = np.random.default_rng(seed)
    return (train_x[rng.permutation(len(train_x))[:train_limit]],
            val_x[rng.permutation(len(val_x))[:val_limit]])


def fixed_p3_embeddings(code: np.ndarray, D: int):
    """Same centered-codebook/DCT geometry as R6B.3, on KMeans patch codes."""
    centered = code.astype(np.float64) - code.astype(np.float64).mean(axis=0)
    mapped = centered @ p3._dct_isometry(D, code.shape[1]).T
    mapped, scale = p3._scale_mean_norm(mapped, math.sqrt(D) * 0.70)
    tok = np.zeros((len(code) + 1, D), dtype=np.float64)
    tok[:len(code)] = mapped
    pos, _ = p3.position_embeddings_2d(7, D)
    return tok, pos, scale


def features_for_digit_one(tokens, model, cfg, tok, pos, nctx: int, seed: int):
    """A fixed, masked digit-1 target-token task, without target leakage."""
    rng = np.random.default_rng(seed)
    X = np.empty((nctx, cfg.D + 1), dtype=np.float64)
    Y = np.empty(nctx, dtype=np.int64)
    for i in range(nctx):
        truth = tokens[int(rng.integers(len(tokens)))]
        sm = p3.sample_mask_set(cfg.N, rng)
        t = int(sm.masked_positions[int(rng.integers(len(sm.masked_positions)))])
        context = truth.copy()
        context[sm.masked_positions] = p3.MASK
        xn, xraw = p3.context_from_arr_unconditional(context, tok, pos)
        X[i] = p3.static_feature(model, cfg, xn, xraw[t], t)
        Y[i] = truth[t]
    if not np.all(np.isfinite(X)):
        raise AssertionError("nonfinite features")
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


def evaluate(W, F, targets):
    logits = F @ W.T
    if not np.all(np.isfinite(logits)):
        return float("nan"), float("nan")
    delta = logits - logits.max(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        log_prob = delta - np.log(np.exp(delta).sum(axis=1, keepdims=True))
    ce = float(-np.mean(log_prob[np.arange(len(targets)), targets]))
    accuracy = float(np.mean(logits.argmax(axis=1) == targets))
    return ce, accuracy


def run_condition(v, u, train_F, train_T, val_F, val_T, indices, lr, checkpoint):
    W = np.zeros((p3.K, train_F.shape[1]), dtype=np.float64)
    ve = np.full(train_F.shape[1], v, dtype=np.float64)
    ue = np.full(train_F.shape[1], u, dtype=np.float64)
    vm = np.full(p3.K, v, dtype=np.float64)
    um = np.full(p3.K, u, dtype=np.float64)
    ze = np.zeros(train_F.shape[1]); zm = np.zeros(p3.K)
    start_ce, start_acc = evaluate(W, val_F, val_T)
    best_ce = start_ce
    best_acc = start_acc
    best_path = 0
    reached_path = None
    reached_s = None
    train_seconds = 0.0
    total_spikes_e = total_spikes_m = 0
    history = [{"path": 0, "val_ce": start_ce, "val_acc": start_acc, "train_seconds": 0.0}]
    status = "completed"
    for path_id, order in enumerate(indices, 1):
        tick = time.perf_counter()
        with np.errstate(over="ignore", invalid="ignore"):
            se, sm = force_path(W, ve, ue, vm, um, ze, zm, train_F, train_T, order, lr)
        train_seconds += time.perf_counter() - tick
        total_spikes_e += int(se); total_spikes_m += int(sm)
        if not all(np.all(np.isfinite(arr)) for arr in (W, ve, ue, vm, um)):
            status = "nonfinite_dynamics"
            break
        if max(float(np.max(np.abs(W))), float(np.max(np.abs(ve))),
               float(np.max(np.abs(vm))), float(np.max(np.abs(ue))),
               float(np.max(np.abs(um)))) > 1e100:
            status = "numerically_overrange"
            break
        if path_id % checkpoint == 0 or path_id == len(indices):
            ce, acc = evaluate(W, val_F, val_T)
            if not np.isfinite(ce):
                status = "nonfinite_validation"
                break
            history.append({"path": path_id, "val_ce": ce, "val_acc": acc,
                            "train_seconds": train_seconds})
            if ce < best_ce:
                best_ce = ce; best_acc = acc; best_path = path_id
            if reached_path is None and ce <= 0.95 * start_ce:
                reached_path, reached_s = path_id, train_seconds
    row = {"initial_v": float(v), "fixed_u": float(u), "status": status,
           "initial_val_ce": start_ce, "best_val_ce": best_ce,
           "min_error_delta": start_ce - best_ce, "best_val_acc": best_acc,
           "best_path": best_path, "paths_to_5pct_ce_improvement": reached_path,
           "train_seconds_to_5pct": reached_s, "total_train_seconds": train_seconds,
           "spikes_e": total_spikes_e, "spikes_m": total_spikes_m,
           "paths_completed": int(history[-1]["path"])}
    return row, history


def save_plots(rows, out: Path):
    grid = np.full((10, 10), np.nan)
    fast = np.full((10, 10), np.nan)
    for r in rows:
        if r.get("reference_baseline"):
            continue
        u_idx = int(np.argmin(abs(VALUES - r["fixed_u"])))
        v_idx = int(np.argmin(abs(VALUES - r["initial_v"])))
        if r["status"] == "completed":
            grid[u_idx, v_idx] = r["best_val_ce"]
            if r["paths_to_5pct_ce_improvement"] is not None:
                fast[u_idx, v_idx] = r["paths_to_5pct_ce_improvement"]
    for values, filename, heading in (
        (grid, "minimum_validation_error.png", "Best held-out CE (digit 1)"),
        (fast, "convergence_paths.png", "Paths to 5% held-out CE improvement")):
        fig, ax = plt.subplots(figsize=(9, 7))
        show = np.ma.masked_invalid(values)
        im = ax.imshow(show, aspect="auto", origin="lower")
        ax.set_xticks(range(10), [f"{x:.0f}" for x in VALUES], rotation=45)
        ax.set_yticks(range(10), [f"{x:.0f}" for x in VALUES])
        ax.set_xlabel("Initial v"); ax.set_ylabel("Fixed initial u")
        ax.set_title(heading + "\nBlank = unstable or threshold not reached")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path("digit1_vu_results"))
    ap.add_argument("--paths", type=int, default=50)
    ap.add_argument("--checkpoint", type=int, default=5)
    ap.add_argument("--train-contexts", type=int, default=640)
    ap.add_argument("--val-contexts", type=int, default=192)
    ap.add_argument("--train-images", type=int, default=512)
    ap.add_argument("--val-images", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=7311)
    ap.add_argument("--allow-download", action="store_true")
    a = ap.parse_args()
    if a.paths < 1 or a.checkpoint < 1:
        ap.error("paths and checkpoint must be positive")
    a.out.mkdir(parents=True, exist_ok=True)
    mnist, data_provenance = data_file(a.repo, a.allow_download)
    train_x, val_x = load_ones(mnist, a.train_images, a.val_images, a.seed)
    cp = a.repo / "codebook.npy"
    if cp.exists():
        code = np.load(cp)
        code_origin = "repository/codebook.npy (conventional KMeans tokenizer)"
    else:
        cp = a.out / "derived_kmeans_codebook.npy"
        code = fit_codebook(train_x, cp, a.seed)
        code_origin = "fitted on MNIST digit-1 train subset"
    if code.shape != (64, 16):
        raise ValueError("expected conventional KMeans codebook of shape (64,16)")
    train_tokens = patch_encode(train_x, code)
    val_tokens = patch_encode(val_x, code)
    cfg = p3.Config()
    frozen = p3.init_model(cfg, a.seed)
    tok, pos, scale = fixed_p3_embeddings(code, cfg.D)
    Ftrain, Ttrain = features_for_digit_one(
        train_tokens, frozen, cfg, tok, pos, a.train_contexts, a.seed + 101)
    Fval, Tval = features_for_digit_one(
        val_tokens, frozen, cfg, tok, pos, a.val_contexts, a.seed + 102)
    rng = np.random.default_rng(a.seed + 103)
    indices = rng.integers(0, len(Ftrain), size=(a.paths, p3.FORCE_STEPS), dtype=np.int64)
    # Compile Numba once, before timing conditions.
    fw = np.zeros((p3.K, Ftrain.shape[1]))
    ve = np.full(Ftrain.shape[1], p3.V0); ue = np.full(Ftrain.shape[1], p3.U0)
    vm = np.full(p3.K, p3.V0); um = np.full(p3.K, p3.U0)
    force_path(fw, ve, ue, vm, um, np.zeros_like(ve), np.zeros_like(vm),
               Ftrain, Ttrain, indices[0], a.lr)
    rows = []; curves = []
    # A distinct reference control preserves the repository's actual equilibrium.
    settings = [(float(v), float(u), False) for u in VALUES for v in VALUES]
    settings += [(float(p3.V0), float(p3.U0), True)]
    for v, u, is_reference in settings:
        row, history = run_condition(
            v, u, Ftrain, Ttrain, Fval, Tval, indices, a.lr, a.checkpoint)
        row["reference_baseline"] = is_reference
        rows.append(row)
        for c in history:
            curves.append({"initial_v": v, "fixed_u": u, "reference_baseline": is_reference, **c})
        print(json.dumps(row, allow_nan=False), flush=True)
    def write_csv(path, items):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader(); writer.writerows(items)
    write_csv(a.out / "conditions.csv", rows)
    write_csv(a.out / "learning_curves.csv", curves)
    save_plots(rows, a.out)
    successful = [r for r in rows if r["status"] == "completed" and not r["reference_baseline"]]
    best = min(successful, key=lambda r: r["best_val_ce"]) if successful else None
    metadata = {"scope": "frozen P3 encoder + R6B.3 MSDL readout only, NOT full MSDL",
                "data": data_provenance, "target": "MNIST train/test images with label=1 only",
                "codebook": code_origin, "embedding_source": "R6B.3 deterministic DCT mapping",
                "D": cfg.D, "H": cfg.H, "K": cfg.V, "input_geometry_scale": scale,
                "force_steps": p3.FORCE_STEPS, "substeps_per_force": p3.N_SUB,
                "init_reference": {"v": p3.V0, "u": p3.U0},
                "initial_u_values": VALUES.tolist(), "initial_v_values": VALUES.tolist(),
                "training_examples": a.train_contexts, "validation_examples": a.val_contexts,
                "train_images_digit1": a.train_images, "test_images_digit1": a.val_images,
                "paths_requested": a.paths, "lr": a.lr, "seed": a.seed,
                "loss": "held-out masked patch token cross entropy (nats)",
                "convergence": "first checkpoint with held-out CE <= 95% of its initial CE",
                "numerically_failed": sum(r["status"] != "completed" for r in rows),
                "best_setting": best}
    (a.out / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (a.out / "REPORT.md").write_text(
        "# Digit-1 v/u sweep\n\n"
        "This is a **readout-only** MSDL initial-state sensitivity experiment, not "
        "an end-to-end R6B.3 training claim. Other P3 weights remain frozen; "
        "the repository KMeans codebook substitutes for its missing SNN-VQ checkpoint.\n\n"
        "The experiment uses held-out MNIST digit-1 images, identical fixed masked "
        "train/evaluation feature sets and presentation order for all 100 conditions, "
        "and a separate reference run at the repository equilibrium (v0,u0).\n\n"
        "Results are in `conditions.csv`, complete checkpoint histories in "
        "`learning_curves.csv`, and heatmaps in the two PNG files.\n\n"
        "The number of training paths needed for 5% relative CE improvement "
        "is a convergence proxy, not generalization accuracy or simulated biological time.\n"
    )
    print("SUMMARY", json.dumps(metadata, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
