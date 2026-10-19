#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNN-Transform R6B.3 — single-file audit build
================================================

Purpose
-------
A standalone, review-oriented consolidation of the current label-free R6B.3 route:

    raw 28x28 image
      -> frozen sensory SNN encoder
      -> VQ64, 7x7 discrete tokens
      -> VQ-derived token representation + deterministic 2-D position representation
      -> exact P3 resident linear-attention SNN-Transform
      -> local persistent E/M MSDL updates
      -> MaskGIT-style simultaneous masked-token training
      -> MaskGIT scheduled-parallel iterative generation
      -> frozen VQ decoder

This file intentionally removes the historical KMeans path and class-conditioning path.
MNIST labels are used only to select/evaluate the 0/1/2 benchmark subset; labels never
enter the neural predictor, MSDL updates, or generation.

Scientific substitutions relative to conventional MaskGIT
----------------------------------------------------------
* VQGAN image encoder -> frozen sensory SNN-VQ encoder.
* Learned token/position lookup embeddings -> frozen VQ geometry + deterministic 2-D sinusoid.
* Dense softmax Transformer -> resident positive-random-feature linear-attention P3 block.
* Backpropagation -> local persistent dynamical MSDL: dW ~ M_post E_pre^T.
* A WR-only local MSDL scaffold is retained as a generic P3/MSDL bootstrap accelerator.

MaskGIT task semantics retained
-------------------------------
* Training: r~U[0,1), gamma(r)=cos(pi*r/2), uniformly mask ceil(gamma*N) positions.
* Every masked receiver is predicted from the same immutable corrupted context.
* No ground-truth masked token is progressively revealed inside a training batch.
* All masked-target forward predictions are cached before one forward-weight update.
* Generation starts all-MASK; all current MASK sites get proposals each round.
* Confidence is probability assigned to the sampled proposal token.
* Cosine remasking schedule + annealed Gumbel random-top-k; committed tokens stay fixed.

Audit status
------------
This is a consolidation/refactor for external review. It preserves the equations and
protocol of the current camp4 R6B.3 modules but is not itself a new scientific result.
Use --self-test for structural checks. Full training requires the frozen R6B.1 tokenizer
checkpoint and mnist.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from numba import njit

# =============================================================================
# 0. Global task constants
# =============================================================================
K = 64
MASK = 64
GRID = 7
N_TOKENS = GRID * GRID
LATENT_HW = GRID
CLASSES = (0, 1, 2)

# =============================================================================
# 1. Frozen sensory SNN-VQ tokenizer (R6B.1)
# =============================================================================
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 4.0 * x.abs()).pow(2)


def spike(x):
    return SpikeFn.apply(x)


def lif_step(current, mem, beta=0.85, threshold=1.0):
    mem = beta * mem + current
    spk = spike(mem - threshold)
    mem = mem - spk * threshold
    return spk, mem


class VectorQuantizer(nn.Module):
    def __init__(self, dim=16, k=K, commitment=0.25, entropy_weight=0.02, entropy_temp=0.5):
        super().__init__()
        self.dim = int(dim)
        self.k = int(k)
        self.commitment = float(commitment)
        self.entropy_weight = float(entropy_weight)
        self.entropy_temp = float(entropy_temp)
        self.codebook = nn.Embedding(self.k, self.dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.k, 1 / self.k)

    def forward(self, z):
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.dim)
        emb = self.codebook.weight
        dist = flat.square().sum(1, keepdim=True) + emb.square().sum(1)[None] - 2 * flat @ emb.t()
        ids = dist.argmin(1)
        q = torch.nn.functional.embedding(ids, emb)
        qloss = torch.nn.functional.mse_loss(q, flat.detach()) + self.commitment * torch.nn.functional.mse_loss(flat, q.detach())
        soft = (-dist / self.entropy_temp).softmax(1)
        p = soft.mean(0)
        entropy = -(p * (p + 1e-12).log()).sum() / math.log(self.k)
        usage_loss = self.entropy_weight * (1.0 - entropy)
        qst = flat + (q - flat).detach()
        qst = qst.view(z.shape[0], LATENT_HW, LATENT_HW, self.dim).permute(0, 3, 1, 2).contiguous()
        return qst, ids.view(z.shape[0], LATENT_HW, LATENT_HW), qloss + usage_loss, float(entropy.detach())


class Decoder(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class SNNVQ(nn.Module):
    def __init__(self, dim=16, steps=8, alpha=0.75, sensory_gain=1.6, beta=0.85,
                 commitment=0.25, entropy_weight=0.02, entropy_temp=0.5):
        super().__init__()
        self.kind = 'snn'
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.sensory_gain = float(sensory_gain)
        self.beta = float(beta)
        self.c1 = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, dim, 1, bias=False)
        self.b3 = nn.BatchNorm2d(dim)
        self.vq = VectorQuantizer(dim=dim, k=K, commitment=commitment,
                                  entropy_weight=entropy_weight, entropy_temp=entropy_temp)
        self.dec = Decoder(dim)

    def forward(self, x):
        ms = torch.zeros_like(x)
        m1 = m2 = m3 = None
        ss = s1 = s2 = s3 = 0.0
        for _ in range(self.steps):
            sp0, ms = lif_step(self.sensory_gain * x, ms, self.beta)
            cur1 = self.b1(self.c1(sp0)); m1 = torch.zeros_like(cur1) if m1 is None else m1
            sp1, m1 = lif_step(cur1, m1, self.beta)
            cur2 = self.b2(self.c2(sp1)); m2 = torch.zeros_like(cur2) if m2 is None else m2
            sp2, m2 = lif_step(cur2, m2, self.beta)
            cur3 = self.b3(self.c3(sp2)); m3 = torch.zeros_like(cur3) if m3 is None else m3
            sp3, m3 = lif_step(cur3, m3, self.beta)
            ss = ss + sp0; s1 = s1 + sp1; s2 = s2 + sp2; s3 = s3 + sp3
        rate0 = ss / self.steps; rate1 = s1 / self.steps; rate2 = s2 / self.steps; rate3 = s3 / self.steps
        z = self.alpha * rate3 + (1.0 - self.alpha) * torch.tanh(m3)
        q, ids, vql, ent = self.vq(z)
        r = self.dec(q)
        st = {
            'sensory_rate': float(rate0.detach().mean()),
            'h1_rate': float(rate1.detach().mean()),
            'h2_rate': float(rate2.detach().mean()),
            'latent_rate': float(rate3.detach().mean()),
        }
        return r, ids, vql, ent, st


def load_frozen_tokenizer(path: Path, device: torch.device):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck.get('metrics', {})
    cfg = meta.get('config', {})
    model = SNNVQ(
        dim=int(cfg.get('dim', 16)),
        steps=int(cfg.get('steps', 8)),
        alpha=float(cfg.get('alpha', 0.75)),
        sensory_gain=float(cfg.get('sensory_gain', 1.6)),
        beta=float(cfg.get('beta', 0.85)),
        commitment=float(cfg.get('commitment', 0.25)),
        entropy_weight=float(cfg.get('entropy_weight', 0.02)),
        entropy_temp=float(cfg.get('entropy_temp', 0.5)),
    ).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


@torch.no_grad()
def encode_snn(model: SNNVQ, x: np.ndarray, device: torch.device, batch: int = 256):
    out, sensory_rates, latent_rates = [], [], []
    xx = torch.from_numpy(x[:, None]).float()
    for st in range(0, len(xx), batch):
        inp = xx[st:st + batch].to(device)
        ms = torch.zeros_like(inp); m1 = m2 = m3 = None
        ss = s3 = 0.0
        for _ in range(model.steps):
            sp0, ms = lif_step(model.sensory_gain * inp, ms, model.beta)
            cur1 = model.b1(model.c1(sp0)); m1 = torch.zeros_like(cur1) if m1 is None else m1
            sp1, m1 = lif_step(cur1, m1, model.beta)
            cur2 = model.b2(model.c2(sp1)); m2 = torch.zeros_like(cur2) if m2 is None else m2
            sp2, m2 = lif_step(cur2, m2, model.beta)
            cur3 = model.b3(model.c3(sp2)); m3 = torch.zeros_like(cur3) if m3 is None else m3
            sp3, m3 = lif_step(cur3, m3, model.beta)
            ss = ss + sp0; s3 = s3 + sp3
        rate0 = ss / model.steps; rate3 = s3 / model.steps
        z = model.alpha * rate3 + (1.0 - model.alpha) * torch.tanh(m3)
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, model.vq.dim)
        emb = model.vq.codebook.weight
        dist = flat.square().sum(1, keepdim=True) + emb.square().sum(1)[None] - 2 * flat @ emb.t()
        ids = dist.argmin(1).view(len(inp), GRID, GRID)
        out.append(ids.cpu().numpy().reshape(len(inp), N_TOKENS).astype(np.int64))
        sensory_rates.append(float(rate0.mean().cpu())); latent_rates.append(float(rate3.mean().cpu()))
    return np.concatenate(out), {
        'sensory_rate_mean': float(np.mean(sensory_rates)),
        'latent_rate_mean': float(np.mean(latent_rates)),
    }


@torch.no_grad()
def decode_snn(model: SNNVQ, ids: np.ndarray, device: torch.device, batch: int = 256):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1, GRID, GRID)
    out = []
    emb = model.vq.codebook.weight
    for st in range(0, len(ids), batch):
        ii = torch.from_numpy(ids[st:st + batch]).to(device)
        q = emb[ii].permute(0, 3, 1, 2).contiguous()
        rec = model.dec(q).squeeze(1).clamp(0, 1)
        out.append(rec.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


# =============================================================================
# 2. Label-free token/position representation
# =============================================================================
def _dct_isometry(out_dim: int, in_dim: int) -> np.ndarray:
    if out_dim < in_dim:
        raise ValueError('P3 dimension must be >= VQ code dimension')
    j = np.arange(out_dim, dtype=np.float64)[:, None]
    k = np.arange(in_dim, dtype=np.float64)[None, :]
    P = np.sqrt(2.0 / out_dim) * np.cos(math.pi * (j + 0.5) * k / out_dim)
    P[:, 0] = 1.0 / math.sqrt(out_dim)
    if not np.allclose(P.T @ P, np.eye(in_dim), rtol=1e-10, atol=1e-10):
        raise AssertionError('DCT projection not isometric')
    return P


def _scale_mean_norm(x: np.ndarray, target_mean_norm: float):
    mean = float(np.mean(np.linalg.norm(x, axis=1)))
    if mean <= 1e-12:
        raise ValueError('cannot scale zero embedding set')
    scale = float(target_mean_norm / mean)
    return x * scale, scale


def token_embeddings_from_snnvq(tokenizer: SNNVQ, D: int, token_scale: float = 0.70):
    with torch.no_grad():
        code = tokenizer.vq.codebook.weight.detach().cpu().numpy().astype(np.float64)
    K0, d = code.shape
    centered = code - code.mean(axis=0, keepdims=True)
    P = _dct_isometry(D, d)
    emb = centered @ P.T
    emb, scale = _scale_mean_norm(emb, math.sqrt(D) * token_scale)
    dc = ((centered[:, None, :] - centered[None, :, :]) ** 2).sum(-1)
    de = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1)
    if not np.allclose(de, dc * scale ** 2, rtol=2e-10, atol=2e-10):
        raise AssertionError('VQ geometry not preserved')
    tok = np.zeros((K0 + 1, D), dtype=np.float64)
    tok[:K0] = emb
    tok[K0] = 0.0
    return tok, {
        'visual_vocab': int(K0), 'vq_dim': int(d), 'p3_dim': int(D),
        'source': 'frozen SNN-VQ codebook', 'projection': 'deterministic DCT isometry',
        'pairwise_vq_geometry_preserved_up_to_global_scale': True,
        'global_scale': scale, 'random_token_embedding': False,
        'mask_token': 'zero vector at centered VQ origin',
    }


def _axis_sinusoid(coords: np.ndarray, dim: int):
    if dim % 2:
        raise ValueError('axis dim must be even')
    half = dim // 2
    freq = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float64) / max(1, half))
    angle = np.asarray(coords, dtype=np.float64)[:, None] * freq[None, :]
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=1)


def position_embeddings_2d(grid: int, D: int, position_scale: float = 0.35):
    if D % 4:
        raise ValueError('D must be divisible by 4')
    rr = np.repeat(np.arange(grid, dtype=np.float64), grid)
    cc = np.tile(np.arange(grid, dtype=np.float64), grid)
    pos = np.concatenate([_axis_sinusoid(rr, D // 2), _axis_sinusoid(cc, D // 2)], axis=1)
    pos = pos - pos.mean(axis=0, keepdims=True)
    pos, scale = _scale_mean_norm(pos, math.sqrt(D) * position_scale)
    return pos, {
        'grid': f'{grid}x{grid}', 'positions': grid * grid, 'p3_dim': D,
        'source': 'deterministic 2-D sinusoidal coordinates', 'centered': True,
        'global_scale': scale, 'random_position_embedding': False,
    }


def build_input_embeddings(tokenizer: SNNVQ, D: int, grid: int = GRID):
    tok, ta = token_embeddings_from_snnvq(tokenizer, D)
    pos, pa = position_embeddings_2d(grid, D)
    return tok, pos, {
        'token': ta, 'position': pa, 'class_embedding_present': False,
        'labels_enter_neural_input': False, 'fixed_random_embedding_present': False,
    }


def context_from_arr_unconditional(arr: np.ndarray, tok: np.ndarray, pos: np.ndarray):
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if len(arr) != len(pos):
        raise ValueError('token/position length mismatch')
    x = tok[arr] + pos
    mu = x.mean(1, keepdims=True)
    var = np.mean((x - mu) ** 2, 1, keepdims=True)
    xn = (x - mu) / np.sqrt(var + 1e-5)
    return xn, x


# =============================================================================
# 3. Exact P3/MSDL resident SNN-Transform core
# =============================================================================
FORCE_DT_MS = 80.0
AF, AS = 0.62, 0.92
WF, WS = 0.70, 0.30
TAU_F_MS = -FORCE_DT_MS / math.log(AF)
TAU_S_MS = -FORCE_DT_MS / math.log(AS)
RF, RS = 1.0 / TAU_F_MS, 1.0 / TAU_S_MS
A_IZH = 1e-3
S_IZH = -(RF + RS)
P_IZH = RF * RS
Q_IZH = S_IZH + A_IZH
B_IZH = Q_IZH + P_IZH / A_IZH
V0 = (Q_IZH - 5.0) / 0.08
U0 = B_IZH * V0
I0 = -0.04 * V0 * V0 - 5.0 * V0 - 140.0 + U0
NUM_S = WF * RF + WS * RS
NUM_0 = RF * RS
G_E = 1e-5
G_M = 1e-6
N_SUB = 80
FORCE_STEPS = 27
MIRROR_SIGMA = 0.1
MIRROR_LEAK = 5e-5
ETA_BR = 0.0338
ETA_BO = 0.0335
ETA_B2 = 0.0335
BASE_LRQ = 0.135
BASE_LRK = 0.1125
BASE_LRV = 0.039375
BASE_LRO = 0.1125
BASE_LR1 = 0.225
BASE_LR2 = 0.225
BASE_LRR = 0.001
SIGQ = 0.05
SIGK = 0.05


def readout_coeff(gain):
    cv = NUM_S / gain
    cu = (NUM_0 / gain - cv * A_IZH) / (A_IZH * B_IZH)
    return cv, cu


CV_E, CU_E = readout_coeff(G_E)
CV_M, CU_M = readout_coeff(G_M)


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def softmax_numba(z):
    mx = np.max(z)
    out = np.exp(z - mx)
    return out / np.sum(out)


@njit(cache=True)
def fast_izh_step(v, u, target_input, pred_input, gain, cv, cu):
    spikes = 0
    for _ in range(N_SUB):
        for j in range(v.shape[0]):
            I = I0 + gain * target_input[j] - gain * pred_input[j]
            dv1 = 0.04 * v[j] * v[j] + 5.0 * v[j] + 140.0 - u[j] + I
            du1 = A_IZH * (B_IZH * v[j] - u[j])
            vm = v[j] + 0.5 * dv1
            um = u[j] + 0.5 * du1
            dv2 = 0.04 * vm * vm + 5.0 * vm + 140.0 - um + I
            du2 = A_IZH * (B_IZH * vm - um)
            v[j] += dv2
            u[j] += du2
            if v[j] >= 30.0:
                v[j] = -65.0
                u[j] += 8.0
                spikes += 1
    out = np.empty(v.shape[0], np.float64)
    for j in range(v.shape[0]):
        out[j] = cv * (v[j] - V0) + cu * (u[j] - U0)
    return out, spikes


@njit(cache=True)
def reciprocal_microsteps(BR, BO, B2, WR, WO, W2,
                          vm_br, um_br, ve_br, ue_br, zm_br, ze_br,
                          vm_bo, um_bo, ve_bo, ue_bo, zm_bo, ze_bo,
                          vm_b2, um_b2, ve_b2, ue_b2, zm_b2, ze_b2,
                          nsteps, sigma, eta_br, eta_bo, eta_b2, leak):
    D = WO.shape[0]
    ff = W2.shape[1]
    V = WR.shape[0]
    for _ in range(nsteps):
        df = np.random.normal(0.0, sigma, D)
        dlogit = np.zeros(V, np.float64)
        for vv in range(V):
            for o in range(D):
                dlogit[vv] += WR[vv, o] * df[o]
        M, _ = fast_izh_step(vm_br, um_br, df, zm_br, G_M, CV_M, CU_M)
        E, _ = fast_izh_step(ve_br, ue_br, dlogit, ze_br, G_E, CV_E, CU_E)
        for o in range(D):
            for vv in range(V):
                BR[o, vv] = (1.0 - leak) * BR[o, vv] + eta_br * M[o] * E[vv]

        da = np.random.normal(0.0, sigma, D)
        dz = np.zeros(D, np.float64)
        for o in range(D):
            for j in range(D):
                dz[o] += WO[o, j] * da[j]
        M, _ = fast_izh_step(vm_bo, um_bo, da, zm_bo, G_M, CV_M, CU_M)
        E, _ = fast_izh_step(ve_bo, ue_bo, dz, ze_bo, G_E, CV_E, CU_E)
        for j in range(D):
            for o in range(D):
                BO[j, o] = (1.0 - leak) * BO[j, o] + eta_bo * M[j] * E[o]

        dh = np.random.normal(0.0, sigma, ff)
        dy = np.zeros(D, np.float64)
        for o in range(D):
            for i in range(ff):
                dy[o] += W2[o, i] * dh[i]
        M, _ = fast_izh_step(vm_b2, um_b2, dh, zm_b2, G_M, CV_M, CU_M)
        E, _ = fast_izh_step(ve_b2, ue_b2, dy, ze_b2, G_E, CV_E, CU_E)
        for i in range(ff):
            for o in range(D):
                B2[i, o] = (1.0 - leak) * B2[i, o] + eta_b2 * M[i] * E[o]


@njit(cache=True)
def train_readout_msdl(features, targets, target_weights, n_paths, lr, seed):
    np.random.seed(seed)
    nctx, nin = features.shape
    V = target_weights.shape[0]
    W = np.zeros((V, nin), np.float64)
    ve = np.full(nin, V0); ue = np.full(nin, U0)
    vm = np.full(V, V0); um = np.full(V, U0)
    ze = np.zeros(nin); zm = np.zeros(V)
    for _p in range(n_paths):
        W0 = W.copy(); acc = np.zeros_like(W)
        for _k in range(FORCE_STEPS):
            ci = np.random.randint(0, nctx)
            x = features[ci]; target = int(targets[ci]); p = softmax_numba(W0 @ x)
            E, _ = fast_izh_step(ve, ue, x, ze, G_E, CV_E, CU_E)
            one = np.zeros(V); one[target] = 1.0
            ww = target_weights[target]
            M, _ = fast_izh_step(vm, um, ww * one, ww * p, G_M, CV_M, CU_M)
            acc += np.outer(M, E)
        W += lr * acc / FORCE_STEPS
    return W


def sparse_mask(rng, shape, density=0.25):
    out = np.zeros(shape, np.float64)
    k = max(1, int(round(shape[1] * density)))
    for i in range(shape[0]):
        out[i, rng.choice(shape[1], size=k, replace=False)] = 1.0
    return out


def phi_q(q, omega):
    hd = q.shape[0]
    xx = q * (hd ** -0.25)
    a = omega @ xx - 0.5 * np.dot(xx, xx)
    a -= a.max(); a = np.clip(a, -30, 0)
    return np.exp(a) / math.sqrt(omega.shape[0]) + 1e-8


def phi_k(k, omega):
    hd = k.shape[0]
    xx = k * (hd ** -0.25)
    a = omega @ xx - 0.5 * np.dot(xx, xx)
    a = np.clip(a, -30, 30)
    return np.exp(a) / math.sqrt(omega.shape[0]) + 1e-8


def rotating_groups(N, R, phase):
    if N % R:
        raise ValueError('R must divide N')
    shift = phase % N
    return np.array([((n + shift) % N) % R for n in range(N)], np.int64)


def make_dirs(rng, R, H, hd):
    return np.where(rng.random((R, H, hd)) < 0.5, -1.0, 1.0).astype(np.float64)


@dataclass
class Config:
    D: int = 96
    H: int = 8
    M: int = 24
    FF: int = 192
    N: int = N_TOKENS
    V: int = K
    R: int = 7
    density: float = 0.25
    reciprocal_micro: int = 4


def init_model(cfg: Config, seed: int):
    if cfg.D % cfg.H:
        raise ValueError('D/H mismatch')
    if cfg.N % cfg.R:
        raise ValueError('N/R mismatch')
    rng = np.random.default_rng(seed); hd = cfg.D // cfg.H
    def W(o, i):
        return rng.normal(0, 1 / math.sqrt(i), size=(o, i)).astype(np.float64)
    maskO = sparse_mask(rng, (cfg.D, cfg.D), cfg.density)
    mask1 = sparse_mask(rng, (cfg.FF, cfg.D), cfg.density)
    mask2 = sparse_mask(rng, (cfg.D, cfg.FF), cfg.density)
    WQ, WK, WV = W(cfg.D, cfg.D), W(cfg.D, cfg.D), W(cfg.D, cfg.D)
    WO = W(cfg.D, cfg.D) * maskO
    W1 = W(cfg.FF, cfg.D) * mask1
    W2 = W(cfg.D, cfg.FF) * mask2
    WR = np.zeros((cfg.V, cfg.D + 1))
    BR = np.zeros((cfg.D, cfg.V)); BO = np.zeros((cfg.D, cfg.D)); B2 = np.zeros((cfg.FF, cfg.D))
    omega = rng.normal(size=(cfg.H, cfg.M, hd))
    omega /= np.linalg.norm(omega, axis=2, keepdims=True) + 1e-12
    omega *= math.sqrt(hd)
    scale = math.sqrt(24.0 / cfg.D)
    return dict(
        WQ=WQ, WK=WK, WV=WV, WO=WO, W1=W1, W2=W2, WR=WR, BR=BR, BO=BO, B2=B2,
        maskO=maskO, mask1=mask1, mask2=mask2, omega=omega,
        bq=np.zeros(cfg.D), bk=np.zeros(cfg.D), bv=np.zeros(cfg.D), bo=np.zeros(cfg.D),
        b1=np.zeros(cfg.FF), b2=np.zeros(cfg.D), theta=np.zeros(cfg.FF),
        gamma2=np.ones(cfg.D), beta2=np.zeros(cfg.D), eps2=1e-5,
        lrq=BASE_LRQ * scale, lrk=BASE_LRK * scale, lrv=BASE_LRV * scale,
        lro=BASE_LRO * scale, lr1=BASE_LR1 * scale, lr2=BASE_LR2 * scale,
        lrr=BASE_LRR, recip_scale=scale, sigq=SIGQ, sigk=SIGK,
    )


def static_feature(model, cfg, xn, xres, t):
    WQ, WK, WV = model['WQ'], model['WK'], model['WV']; hd = cfg.D // cfg.H
    q = (WQ @ xn[t] + model['bq']).reshape(cfg.H, hd)
    kk = (xn @ WK.T + model['bk']).reshape(cfg.N, cfg.H, hd).transpose(1, 0, 2)
    vv = (xn @ WV.T + model['bv']).reshape(cfg.N, cfg.H, hd).transpose(1, 0, 2)
    a = np.empty(cfg.D)
    for h in range(cfg.H):
        pq = phi_q(q[h], model['omega'][h])
        P = np.array([phi_k(kk[h, n], model['omega'][h]) for n in range(cfg.N)])
        scores = P @ pq
        den = max(1e-8, float(scores.sum()))
        a[h * hd:(h + 1) * hd] = (scores[:, None] * vv[h]).sum(0) / den
    z = xres + model['bo'] + model['WO'] @ a
    mu = z.mean(); inv = 1 / math.sqrt(float(np.mean((z - mu) ** 2)) + model['eps2'])
    u = (z - mu) * inv * model['gamma2'] + model['beta2']
    hh = np.maximum(0, model['W1'] @ u + model['b1'] - model['theta'])
    y = z + model['W2'] @ hh + model['b2']
    rms = math.sqrt(float(np.mean(y * y))) + 1e-8
    rf = np.empty(cfg.D + 1); rf[:-1] = 0.5 * y / rms; rf[-1] = 0.5
    return rf


class P3ImageSystem:
    def __init__(self, cfg: Config, model: Dict, seed: int):
        self.c = cfg
        for k, v in model.items():
            setattr(self, k, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        c = cfg; D = c.D; ff = c.FF; V = c.V; P = c.R; hd = D // c.H
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed + 8181); seed_numba(seed + 9181)
        self.vm_br = np.full(D, V0); self.um_br = np.full(D, U0); self.ve_br = np.full(V, V0); self.ue_br = np.full(V, U0); self.zm_br = np.zeros(D); self.ze_br = np.zeros(V)
        self.vm_bo = np.full(D, V0); self.um_bo = np.full(D, U0); self.ve_bo = np.full(D, V0); self.ue_bo = np.full(D, U0); self.zm_bo = np.zeros(D); self.ze_bo = np.zeros(D)
        self.vm_b2 = np.full(ff, V0); self.um_b2 = np.full(ff, U0); self.ve_b2 = np.full(D, V0); self.ue_b2 = np.full(D, U0); self.zm_b2 = np.zeros(ff); self.ze_b2 = np.zeros(D)
        def pop(n): return np.full(n, V0), np.full(n, U0), np.zeros(n)
        self.ver, self.uer, self.zer = pop(D + 1); self.vmr, self.umr, self.zmr = pop(V)
        self.veq, self.ueq, self.zeq = pop(D); self.vmq, self.umq, self.zmq = pop(D)
        self.vek, self.uek, self.zek = pop(P * D); self.vmk, self.umk, self.zmk = pop(P * D)
        self.vev, self.uev, self.zev = pop(c.H * D); self.vmv, self.umv, self.zmv = pop(D)
        self.veo, self.ueo, self.zeo = pop(D); self.vmo, self.umo, self.zmo = pop(D)
        self.ve1, self.ue1, self.ze1 = pop(D); self.vm1, self.um1, self.zm1 = pop(ff)
        self.ve2, self.ue2, self.ze2 = pop(ff); self.vm2, self.um2, self.zm2 = pop(D)
        self.xcache = np.zeros((c.N, D))
        self.qraw = np.zeros((c.H, c.N, hd)); self.kraw = np.zeros_like(self.qraw); self.vcache = np.zeros_like(self.qraw)
        self.pq = np.zeros((c.H, c.N, c.M)); self.pk = np.zeros_like(self.pq)
        self.Hstate = np.zeros((c.H, c.M, hd)); self.gstate = np.zeros((c.H, c.M)); self.Xphi = np.zeros((c.H, c.M, D))
        self.groups = np.zeros(c.N, np.int64); self.dirs = np.ones((P, c.H, hd)); self.xgrp = np.zeros((P, D))
        self.Hp = np.zeros((P, c.H, c.M, hd)); self.Gp = np.zeros((P, c.H, c.M)); self.Hm = np.zeros_like(self.Hp); self.Gm = np.zeros_like(self.Gp)
        self.initialized = False; self.image_count = 0
        self._clear(); self.event_count = 0; self.update_count = 0; self.obs_nll = 0.0; self.obs_count = 0

    def _clear(self):
        self.aq = np.zeros_like(self.WQ); self.ak = np.zeros_like(self.WK); self.av = np.zeros_like(self.WV)
        self.ao = np.zeros_like(self.WO); self.a1 = np.zeros_like(self.W1); self.a2 = np.zeros_like(self.W2); self.ar = np.zeros_like(self.WR)
        self.accum = 0

    def weights(self):
        return {k: np.array(getattr(self, k), copy=True) for k in ('WQ', 'WK', 'WV', 'WO', 'W1', 'W2', 'WR', 'BR', 'BO', 'B2')}

    def _project(self, x):
        c = self.c; hd = c.D // c.H
        return ((self.WQ @ x + self.bq).reshape(c.H, hd),
                (self.WK @ x + self.bk).reshape(c.H, hd),
                (self.WV @ x + self.bv).reshape(c.H, hd))

    def _reld(self, h, k, v, xi):
        pk = phi_k(k, self.omega[h]); pp = phi_k(k + self.sigk * xi, self.omega[h]); pm = phi_k(k - self.sigk * xi, self.omega[h])
        dp = pp - pk; dm = pm - pk
        return dp[:, None] * v[None, :], dp, dm[:, None] * v[None, :], dm

    def rebuild(self, xn):
        c = self.c; hd = c.D // c.H
        self.groups = rotating_groups(c.N, c.R, self.image_count)
        self.dirs = make_dirs(self.rng, c.R, c.H, hd)
        self.image_count += 1
        self.Hstate.fill(0); self.gstate.fill(0); self.Xphi.fill(0); self.xgrp.fill(0)
        self.Hp.fill(0); self.Gp.fill(0); self.Hm.fill(0); self.Gm.fill(0)
        norm = math.sqrt(c.N // c.R)
        for n in range(c.N):
            x = np.asarray(xn[n], float); q, k, v = self._project(x); self.xcache[n] = x
            r = int(self.groups[n]); self.xgrp[r] += x / norm
            for h in range(c.H):
                pq = phi_q(q[h], self.omega[h]); pk = phi_k(k[h], self.omega[h])
                self.qraw[h, n] = q[h]; self.kraw[h, n] = k[h]; self.vcache[h, n] = v[h]
                self.pq[h, n] = pq; self.pk[h, n] = pk
                self.gstate[h] += pk; self.Hstate[h] += pk[:, None] * v[h]; self.Xphi[h] += pk[:, None] * x
                hp, gp, hm, gm = self._reld(h, k[h], v[h], self.dirs[r, h])
                self.Hp[r, h] += hp; self.Gp[r, h] += gp; self.Hm[r, h] += hm; self.Gm[r, h] += gm
        self.initialized = True

    def update_token(self, n, newx):
        c = self.c; old = self.xcache[n].copy()
        if np.array_equal(old, newx):
            return
        r = int(self.groups[n]); norm = math.sqrt(c.N // c.R)
        self.xgrp[r] += (newx - old) / norm
        for h in range(c.H):
            op = self.pk[h, n].copy(); ov = self.vcache[h, n].copy(); ok = self.kraw[h, n].copy()
            self.gstate[h] -= op; self.Hstate[h] -= op[:, None] * ov; self.Xphi[h] -= op[:, None] * old
            hp, gp, hm, gm = self._reld(h, ok, ov, self.dirs[r, h])
            self.Hp[r, h] -= hp; self.Gp[r, h] -= gp; self.Hm[r, h] -= hm; self.Gm[r, h] -= gm
        q, k, v = self._project(newx)
        for h in range(c.H):
            pq = phi_q(q[h], self.omega[h]); pk = phi_k(k[h], self.omega[h])
            self.qraw[h, n] = q[h]; self.kraw[h, n] = k[h]; self.vcache[h, n] = v[h]
            self.pq[h, n] = pq; self.pk[h, n] = pk
            self.gstate[h] += pk; self.Hstate[h] += pk[:, None] * v[h]; self.Xphi[h] += pk[:, None] * newx
            hp, gp, hm, gm = self._reld(h, k[h], v[h], self.dirs[r, h])
            self.Hp[r, h] += hp; self.Gp[r, h] += gp; self.Hm[r, h] += hm; self.Gm[r, h] += gm
        self.xcache[n] = newx

    def _read(self, t):
        c = self.c; hd = c.D // c.H
        a = np.empty(c.D); q = np.empty((c.H, hd)); num = np.empty((c.H, hd)); den = np.empty(c.H)
        for h in range(c.H):
            q[h] = self.qraw[h, t]; pq = self.pq[h, t]
            dd = max(1e-8, float(pq @ self.gstate[h])); den[h] = dd
            nn = pq @ self.Hstate[h]; num[h] = nn
            a[h * hd:(h + 1) * hd] = nn / dd
        return a, q, num, den

    def _qdrive(self, q, drive_a):
        c = self.c; hd = c.D // c.H; out = np.zeros(c.D)
        for h in range(c.H):
            xi = np.where(self.rng.random(hd) < .5, -1., 1.)
            p = phi_q(q[h] + self.sigq * xi, self.omega[h]); m = phi_q(q[h] - self.sigq * xi, self.omega[h])
            dp = max(1e-8, float(p @ self.gstate[h])); dm = max(1e-8, float(m @ self.gstate[h]))
            ap = (p @ self.Hstate[h]) / dp; am = (m @ self.Hstate[h]) / dm
            sc = float(drive_a[h * hd:(h + 1) * hd] @ (ap - am)) / (2 * self.sigq)
            out[h * hd:(h + 1) * hd] = sc * xi
        return out

    def _kdrive(self, q, num, den, drive_a):
        c = self.c; hd = c.D // c.H; out = np.zeros((c.R, c.D)); norm = math.sqrt(c.N // c.R)
        for r in range(c.R):
            for h in range(c.H):
                pq = phi_q(q[h], self.omega[h])
                dgp = float(pq @ self.Gp[r, h]); dgm = float(pq @ self.Gm[r, h])
                dHp = pq @ self.Hp[r, h]; dHm = pq @ self.Hm[r, h]
                dp = max(1e-8, float(den[h] + dgp)); dm = max(1e-8, float(den[h] + dgm))
                ap = (num[h] + dHp) / dp; am = (num[h] + dHm) / dm
                sc = float(drive_a[h * hd:(h + 1) * hd] @ (ap - am)) / (2 * self.sigk * norm)
                out[r, h * hd:(h + 1) * hd] = sc * self.dirs[r, h]
        return out

    def _vinput(self, t, den):
        c = self.c; s = np.empty(c.H * c.D)
        for h in range(c.H):
            s[h * c.D:(h + 1) * c.D] = (self.pq[h, t] @ self.Xphi[h]) / max(1e-8, float(den[h]))
        return s

    def _tail(self, a, xres):
        z = xres + self.bo + self.WO @ a
        mu = z.mean(); inv = 1 / math.sqrt(float(np.mean((z - mu) ** 2)) + self.eps2)
        u = (z - mu) * inv * self.gamma2 + self.beta2
        h = np.maximum(0, self.W1 @ u + self.b1 - self.theta)
        y = z + self.W2 @ h + self.b2
        rms = math.sqrt(float(np.mean(y * y))) + 1e-8
        rf = np.empty(self.c.D + 1); rf[:-1] = .5 * y / rms; rf[-1] = .5
        return u, h, rf, softmax_numba(self.WR @ rf)

    def forward_only(self, xn, xres, t, new_image=False, changed=None):
        if new_image or not self.initialized:
            self.rebuild(xn)
        elif changed is not None:
            self.update_token(int(changed), np.asarray(xn[int(changed)], float))
        a, q, num, den = self._read(int(t)); u, h, rf, p = self._tail(a, xres)
        return np.asarray(p, float)

    def apply_update(self):
        inv = 1 / max(1, self.accum)
        self.WR += self.lrr * self.ar * inv
        self.WQ += self.lrq * self.aq * inv
        self.WK += self.lrk * self.ak * inv
        self.WV += self.lrv * self.av * inv
        self.WO += self.lro * self.ao * inv; self.WO *= self.maskO
        self.W1 += self.lr1 * self.a1 * inv; self.W1 *= self.mask1
        self.W2 += self.lr2 * self.a2 * inv; self.W2 *= self.mask2
        self.update_count += 1
        self._clear()


# =============================================================================
# 4. MaskGIT-style training adapter for local P3/MSDL
# =============================================================================
@dataclass(frozen=True)
class MaskSample:
    ratio_r: float
    mask_ratio: float
    masked_positions: np.ndarray


def cosine_mask_ratio_train(r: float) -> float:
    r = float(r)
    if not (0.0 <= r < 1.0):
        raise ValueError('r outside [0,1)')
    return float(math.cos(math.pi * 0.5 * r))


def sample_mask_set(n_tokens: int, rng: np.random.Generator) -> MaskSample:
    r = float(rng.random())
    gamma = cosine_mask_ratio_train(r)
    nmask = int(math.ceil(gamma * int(n_tokens)))
    nmask = max(1, min(int(n_tokens), nmask))
    pos = np.sort(rng.choice(int(n_tokens), size=nmask, replace=False).astype(np.int64))
    return MaskSample(r, gamma, pos)


def masking_self_test(n_tokens=49, vocab_size=64, trials=128, seed=9173):
    rng = np.random.default_rng(seed); truth = np.arange(n_tokens, dtype=np.int64) % vocab_size; counts = []
    for _ in range(trials):
        sm = sample_mask_set(n_tokens, rng)
        if len(sm.masked_positions) != int(math.ceil(sm.mask_ratio * n_tokens)):
            raise AssertionError('training mask count mismatch')
        arr = truth.copy(); arr[sm.masked_positions] = vocab_size
        if not np.all(arr[sm.masked_positions] == vocab_size):
            raise AssertionError('mask replacement failed')
        counts.append(len(sm.masked_positions))
    return {'pass': True, 'mask_count_min': int(np.min(counts)), 'mask_count_max': int(np.max(counts)),
            'mask_count_mean': float(np.mean(counts)), 'ground_truth_reveal_inside_mask_context': False}


class MaskGITP3System(P3ImageSystem):
    def _reciprocal_batch_microsteps(self, n_targets: int):
        c = self.c
        reciprocal_microsteps(
            self.BR, self.BO, self.B2, self.WR, self.WO, self.W2,
            self.vm_br, self.um_br, self.ve_br, self.ue_br, self.zm_br, self.ze_br,
            self.vm_bo, self.um_bo, self.ve_bo, self.ue_bo, self.zm_bo, self.ze_bo,
            self.vm_b2, self.um_b2, self.ve_b2, self.ue_b2, self.zm_b2, self.ze_b2,
            int(c.reciprocal_micro) * int(n_targets), MIRROR_SIGMA,
            ETA_BR * self.recip_scale, ETA_BO * self.recip_scale, ETA_B2 * self.recip_scale, MIRROR_LEAK)

    def prepare_mask_batch(self, xn, xraw, masked_positions):
        if self.accum != 0:
            raise AssertionError('local accumulators must be empty at batch start')
        self.rebuild(np.asarray(xn, dtype=np.float64))
        cache = []
        for t0 in np.asarray(masked_positions, dtype=np.int64):
            t = int(t0); a, q, num, den = self._read(t); u, h, rf, p = self._tail(a, np.asarray(xraw[t], dtype=np.float64))
            cache.append({'t': t, 'a': a.copy(), 'q': q.copy(), 'num': num.copy(), 'den': den.copy(),
                          'u': u.copy(), 'h': h.copy(), 'rf': rf.copy(), 'p': p.copy()})
        return cache

    def accumulate_cached_target(self, rec: Dict, target: int, weight: float = 1.0):
        c = self.c; t = int(rec['t']); a = rec['a']; q = rec['q']; num = rec['num']; den = rec['den']
        u = rec['u']; h = rec['h']; rf = rec['rf']; p = rec['p']
        one = np.zeros(c.V, dtype=np.float64); one[int(target)] = 1.0
        ww = float(weight); err = ww * (one - p)
        self.obs_nll -= math.log(max(float(p[int(target)]), 1e-12)); self.obs_count += 1

        ER, _ = fast_izh_step(self.ver, self.uer, rf, self.zer, G_E, CV_E, CU_E)
        MR, _ = fast_izh_step(self.vmr, self.umr, ww * one, ww * p, G_M, CV_M, CU_M)
        self.ar += np.outer(MR, ER)

        dy = self.BR @ err; dh = self.B2 @ dy; da = self.BO @ dy
        E2, _ = fast_izh_step(self.ve2, self.ue2, h, self.ze2, G_E, CV_E, CU_E)
        M2, _ = fast_izh_step(self.vm2, self.um2, dy, self.zm2, G_M, CV_M, CU_M)
        self.a2 += np.outer(M2, E2) * self.mask2
        E1, _ = fast_izh_step(self.ve1, self.ue1, u, self.ze1, G_E, CV_E, CU_E)
        M1, _ = fast_izh_step(self.vm1, self.um1, dh, self.zm1, G_M, CV_M, CU_M)
        self.a1 += np.outer(M1, E1) * self.mask1
        EO, _ = fast_izh_step(self.veo, self.ueo, a, self.zeo, G_E, CV_E, CU_E)
        MO, _ = fast_izh_step(self.vmo, self.umo, dy, self.zmo, G_M, CV_M, CU_M)
        self.ao += np.outer(MO, EO) * self.maskO

        sv = self._vinput(t, den)
        EV, _ = fast_izh_step(self.vev, self.uev, sv, self.zev, G_E, CV_E, CU_E)
        MV, _ = fast_izh_step(self.vmv, self.umv, da, self.zmv, G_M, CV_M, CU_M)
        hd = c.D // c.H
        for rr in range(c.D):
            self.av[rr] += MV[rr] * EV[(rr // hd) * c.D:(rr // hd + 1) * c.D]

        EK, _ = fast_izh_step(self.vek, self.uek, self.xgrp.reshape(-1), self.zek, G_E, CV_E, CU_E)
        kd = self._kdrive(q, num, den, da)
        MK, _ = fast_izh_step(self.vmk, self.umk, kd.reshape(-1), self.zmk, G_M, CV_M, CU_M)
        self.ak += (MK.reshape(c.R, c.D).T @ EK.reshape(c.R, c.D)) / c.R

        qd = self._qdrive(q, da)
        EQ, _ = fast_izh_step(self.veq, self.ueq, self.xcache[t], self.zeq, G_E, CV_E, CU_E)
        MQ, _ = fast_izh_step(self.vmq, self.umq, qd, self.zmq, G_M, CV_M, CU_M)
        self.aq += np.outer(MQ, EQ)
        self.event_count += 1; self.accum += 1
        return np.asarray(p, dtype=np.float64)

    def learn_mask_batch(self, xn, xraw, masked_positions, truth_tokens, weights, rng):
        masked = np.asarray(masked_positions, dtype=np.int64)
        truth = np.asarray(truth_tokens, dtype=np.int64)
        if len(masked) < 1:
            raise ValueError('empty mask batch')
        self._reciprocal_batch_microsteps(len(masked))
        cache = self.prepare_mask_batch(xn, xraw, masked)
        order = rng.permutation(len(cache))
        nll = []
        for ci in order:
            rec = cache[int(ci)]; target = int(truth[rec['t']])
            p = self.accumulate_cached_target(rec, target, float(weights[target]))
            nll.append(-math.log(max(float(p[target]), 1e-12)))
        if self.accum != len(masked):
            raise AssertionError('not every masked target contributed once')
        self.apply_update()
        return {'masked_targets': int(len(masked)), 'mean_nll': float(np.mean(nll)),
                'same_corrupted_context': True, 'ground_truth_reveal_inside_batch': False,
                'forward_weights_fixed_until_all_targets_contribute': True,
                'one_forward_update_after_mask_batch': True}


def build_maskgit_scaffold_contexts_unconditional(model, cfg, tokens, tok, pos, nctx, seed):
    rng = np.random.default_rng(seed)
    F = np.empty((nctx, cfg.D + 1), dtype=np.float64); T = np.empty(nctx, dtype=np.int64)
    row = 0; batches = 0; counts = []
    while row < nctx:
        ii = int(rng.integers(len(tokens))); sm = sample_mask_set(cfg.N, rng); truth = np.asarray(tokens[ii], dtype=np.int64)
        arr = truth.copy(); arr[sm.masked_positions] = MASK
        xn, xraw = context_from_arr_unconditional(arr, tok, pos)
        counts.append(len(sm.masked_positions)); batches += 1
        for t in sm.masked_positions:
            if row >= nctx: break
            F[row] = static_feature(model, cfg, xn, xraw[int(t)], int(t)); T[row] = int(truth[int(t)]); row += 1
    return F, T, {'protocol': 'label-free MaskGIT-style WR scaffold', 'target_rows': int(nctx),
                  'mask_contexts_sampled': int(batches), 'mean_masked_targets_per_context': float(np.mean(counts)),
                  'progressive_ground_truth_reveal': False, 'class_labels_used': False}


def train_maskgit_msdl_unconditional(system: MaskGITP3System, cfg, tr, tok, pos, weights,
                                     target_events, seed, report_every=5000):
    rng = np.random.default_rng(seed); done = 0; batches = 0; hist = []; t0 = time.time(); next_report = int(report_every)
    mask_counts = []
    while done < int(target_events):
        ii = int(rng.integers(len(tr))); sm = sample_mask_set(cfg.N, rng); truth = np.asarray(tr[ii], dtype=np.int64)
        arr = truth.copy(); arr[sm.masked_positions] = MASK
        xn, xraw = context_from_arr_unconditional(arr, tok, pos)
        rec = system.learn_mask_batch(xn, xraw, sm.masked_positions, truth, weights, rng)
        nmask = int(rec['masked_targets']); done += nmask; batches += 1; mask_counts.append(nmask)
        if done >= next_report:
            row = {'target_events': int(done), 'mask_batches': int(batches),
                   'online_nll': system.obs_nll / max(1, system.obs_count), 'updates': int(system.update_count),
                   'mean_masked_targets_per_batch': float(np.mean(mask_counts)), 'elapsed_s': time.time() - t0,
                   'mean_abs_BR': float(np.mean(np.abs(system.BR))), 'mean_abs_BO': float(np.mean(np.abs(system.BO))),
                   'mean_abs_B2': float(np.mean(np.abs(system.B2)))}
            hist.append(row); print('TRAIN', json.dumps(row), flush=True)
            while next_report <= done: next_report += int(report_every)
    return hist, {'pass': True, 'requested_target_events': int(target_events), 'actual_target_events': int(done),
                  'mask_batches': int(batches), 'mean_masked_targets_per_batch': float(np.mean(mask_counts)),
                  'all_masked_targets_learned_from_same_corrupted_context': True,
                  'ground_truth_reveal_inside_batch': False, 'forward_predictions_cached_before_weight_update': True,
                  'one_forward_weight_update_per_mask_context': True, 'backpropagation': False, 'class_labels_used': False}


def eval_masked_unconditional(model, cfg, tokens, tok, pos, ratio, seed, nimages=128, targets_per_image=6):
    rng = np.random.default_rng(seed); ce = []; acc = []; fgce = []; fgacc = []; fgmass = []
    dom = int(np.argmax(np.bincount(np.asarray(tokens).reshape(-1), minlength=cfg.V)))
    system = P3ImageSystem(cfg, model, seed + 30000)
    for ii in rng.choice(len(tokens), size=min(nimages, len(tokens)), replace=False):
        nmask = max(1, int(round(float(ratio) * cfg.N)))
        masked = rng.choice(cfg.N, size=nmask, replace=False)
        arr = np.asarray(tokens[ii], dtype=np.int64).copy(); arr[masked] = MASK
        xn, xraw = context_from_arr_unconditional(arr, tok, pos)
        picks = rng.choice(masked, size=min(targets_per_image, nmask), replace=False)
        for t in picks:
            p = system.forward_only(xn, xraw[int(t)], int(t), new_image=True)
            z = int(tokens[ii, int(t)]); cur = -math.log(max(float(p[z]), 1e-12))
            ce.append(cur); acc.append(int(np.argmax(p) == z))
            if z != dom:
                fgce.append(cur); fgacc.append(int(np.argmax(p) == z)); fgmass.append(1 - float(p[dom]))
    return {'ce': float(np.mean(ce)), 'acc': float(np.mean(acc)),
            'foreground_ce': float(np.mean(fgce)) if fgce else None,
            'foreground_acc': float(np.mean(fgacc)) if fgacc else None,
            'foreground_non_dominant_mass': float(np.mean(fgmass)) if fgmass else None,
            'samples': len(ce), 'fg_samples': len(fgce), 'dominant_token': dom, 'class_labels_used': False}


# =============================================================================
# 5. MaskGIT scheduled-parallel decoder
# =============================================================================
PredictMasked = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def cosine_mask_ratio_decode(ratio: float) -> float:
    return float(np.clip(math.cos(math.pi * 0.5 * float(ratio)), 1e-6, 1.0))


def _normalise_rows(p: np.ndarray):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError('bad predictor probability matrix')
    s = p.sum(axis=1, keepdims=True)
    if np.any(s <= 0): raise ValueError('zero probability row')
    return p / s


def _sample_rows(p: np.ndarray, rng: np.random.Generator):
    return np.asarray([rng.choice(p.shape[1], p=row) for row in p], dtype=np.int64)


def _official_mask_len(initial_unknown: int, current_unknown: int, mask_ratio: float):
    raw = int(math.floor(float(initial_unknown) * float(mask_ratio)))
    return int(max(1, min(int(current_unknown) - 1, raw)))


def scheduled_parallel_decode(init_ids, mask_token_id, vocab_size, predict_masked: PredictMasked,
                              rng, num_iter=8, choice_temperature=4.5):
    cur = np.asarray(init_ids, dtype=np.int64).copy().reshape(-1)
    initial_unknown = int(np.sum(cur == mask_token_id)); trace = []; final_proposal = None
    for step in range(int(num_iter)):
        unknown = np.flatnonzero(cur == mask_token_id).astype(np.int64); current_unknown = len(unknown)
        if current_unknown == 0:
            final_proposal = cur.copy(); break
        context_before = cur.copy()
        p = _normalise_rows(predict_masked(context_before.copy(), unknown.copy(), step))
        if p.shape != (current_unknown, int(vocab_size)): raise ValueError('predictor shape mismatch')
        if not np.array_equal(cur, context_before): raise AssertionError('predictor mutated context')
        sampled = _sample_rows(p, rng)
        proposal = context_before.copy(); proposal[unknown] = sampled
        known = np.flatnonzero(context_before != mask_token_id)
        if len(known) and not np.array_equal(proposal[known], context_before[known]):
            raise AssertionError('committed token changed')
        selected_prob = p[np.arange(current_unknown), sampled]
        final_proposal = proposal.copy()
        ratio = float(step + 1) / float(num_iter); mask_ratio = cosine_mask_ratio_decode(ratio)
        mask_len = _official_mask_len(initial_unknown, current_unknown, mask_ratio)
        temp = float(choice_temperature) * (1.0 - ratio)
        score = np.log(np.clip(selected_prob, 1e-12, 1.0)) + (temp * rng.gumbel(size=len(selected_prob)) if temp != 0 else 0.0)
        order = np.argsort(score, kind='mergesort'); remask = unknown[order[:mask_len]]
        nxt = proposal.copy(); nxt[remask] = mask_token_id
        trace.append({'step': int(step + 1), 'masked_before': int(current_unknown), 'masked_after': int(np.sum(nxt == mask_token_id)),
                      'mask_len': int(mask_len), 'ratio': ratio, 'mask_ratio': mask_ratio, 'choice_temperature': temp})
        cur = nxt
    if final_proposal is None or np.any(final_proposal == mask_token_id):
        raise AssertionError('final proposal still masked')
    return final_proposal.astype(np.int64), trace


def controller_self_test():
    n = 49; v = 64; rng = np.random.default_rng(12345); init = np.full(n, v, dtype=np.int64)
    def uniform_predict(cur, masked, step): return np.full((len(masked), v), 1.0 / v, dtype=np.float64)
    out, tr = scheduled_parallel_decode(init, v, v, uniform_predict, rng, num_iter=8, choice_temperature=4.5)
    got = [x['masked_after'] for x in tr]; expected = [48, 45, 40, 34, 27, 18, 9, 1]
    if got != expected or np.any(out == v): raise AssertionError(f'MaskGIT controller self-test failed: {got}')
    return {'pass': True, 'masked_after_each_round': got, 'expected': expected}


def generation_tokens_maskgit(model, cfg, n_samples, tok, pos, seed, num_iter=8, choice_temperature=4.5):
    outputs, traces = [], []
    for gi in range(int(n_samples)):
        system = P3ImageSystem(cfg, model, seed + 100000 + gi)
        sample_rng = np.random.default_rng(seed + 200000 + gi)
        def predict_masked(cur_ids, masked_positions, step):
            xn, xraw = context_from_arr_unconditional(cur_ids, tok, pos)
            probs = np.empty((len(masked_positions), cfg.V), dtype=np.float64)
            for qi, t in enumerate(masked_positions):
                probs[qi] = system.forward_only(xn, xraw[int(t)], int(t), new_image=(qi == 0), changed=None)
            return probs
        initial = np.full(cfg.N, MASK, dtype=np.int64)
        final_ids, trace = scheduled_parallel_decode(initial, MASK, cfg.V, predict_masked, sample_rng,
                                                     num_iter=num_iter, choice_temperature=choice_temperature)
        outputs.append(final_ids); traces.append(trace)
    return np.asarray(outputs, dtype=np.int64), traces


# =============================================================================
# 6. Dataset helpers and end-to-end experiment
# =============================================================================
def load_mnist(path: Path):
    d = np.load(path, allow_pickle=False)
    return (d['x_train'].astype(np.float32) / 255., d['y_train'].astype(np.int64),
            d['x_test'].astype(np.float32) / 255., d['y_test'].astype(np.int64))


def subset012(x, y):
    m = np.isin(y, np.array(CLASSES, dtype=np.int64))
    return x[m], y[m]


def split_train_val(x, y, val_frac=0.1, seed=7):
    rng = np.random.default_rng(seed); tr = []; va = []
    for c in CLASSES:
        z = np.where(y == c)[0]; rng.shuffle(z); nv = max(1, int(round(val_frac * len(z))))
        va.append(z[:nv]); tr.append(z[nv:])
    tr = np.concatenate(tr); va = np.concatenate(va); rng.shuffle(tr); rng.shuffle(va)
    return x[tr], y[tr], x[va], y[va]


def token_stats(tokens):
    cnt = np.bincount(np.asarray(tokens).reshape(-1), minlength=K).astype(np.float64); p = cnt / cnt.sum(); nz = p[p > 0]
    return {'dominant_token': int(np.argmax(cnt)), 'dominant_fraction': float(p.max()),
            'active_codes': int((cnt > 0).sum()), 'perplexity': float(np.exp(-(nz * np.log(nz)).sum()))}


def run_full(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True); device = torch.device('cpu')

    xall, yall, xte0, yte0 = load_mnist(a.data)
    x012, y012 = subset012(xall, yall); xte, yte = subset012(xte0, yte0)
    xtr, ytr, xva, yva = split_train_val(x012, y012, 0.1, 7)

    tokenizer, tok_meta = load_frozen_tokenizer(a.tokenizer, device)
    tr, sp_tr = encode_snn(tokenizer, xtr, device, a.encode_batch)
    va, _ = encode_snn(tokenizer, xva, device, a.encode_batch)
    te, _ = encode_snn(tokenizer, xte, device, a.encode_batch)

    cfg = Config(D=a.D, H=a.heads, M=a.features, FF=a.ff, R=a.R)
    model = init_model(cfg, a.seed)
    for k in ('lrq', 'lrk', 'lrv', 'lro', 'lr1', 'lr2'):
        model[k] *= a.internal_lr_scale
    tok, pos, input_audit = build_input_embeddings(tokenizer, cfg.D, GRID)

    # Generic WR bootstrap scaffold: local MSDL only, no class labels.
    ones = np.ones(cfg.V, np.float64)
    F, T, scaffold_audit = build_maskgit_scaffold_contexts_unconditional(model, cfg, tr, tok, pos,
                                                                         a.scaffold_contexts, a.seed + 101)
    model['WR'] = train_readout_msdl(F, T, ones, a.scaffold_paths, a.scaffold_lr, a.seed + 202)

    scaffold_eval = {str(r): eval_masked_unconditional(model, cfg, va, tok, pos, r,
                                                        a.seed + 300 + int(100 * r), a.eval_images)
                     for r in (0.25, 0.5, 0.9, 1.0)}

    sysm = MaskGITP3System(cfg, model, a.seed + 303)
    hist, training_audit = train_maskgit_msdl_unconditional(sysm, cfg, tr, tok, pos, ones,
                                                            a.events, a.seed + 404, a.report_every)
    final = model.copy(); final.update(sysm.weights())
    final_eval = {str(r): eval_masked_unconditional(final, cfg, te, tok, pos, r,
                                                     a.seed + 500 + int(100 * r), a.eval_images)
                  for r in (0.25, 0.5, 0.9, 1.0)}

    gtok, gtrace = generation_tokens_maskgit(final, cfg, a.gen_samples, tok, pos, a.seed + 606,
                                             a.decode_iterations, a.choice_temperature)
    gimg = decode_snn(tokenizer, gtok, device)
    np.save(a.out / 'generated_images.npy', gimg)
    np.savetxt(a.out / 'generated_tokens.csv', gtok, fmt='%d', delimiter=',')
    np.savez_compressed(a.out / 'final_state.npz', **{k: v for k, v in final.items() if isinstance(v, np.ndarray)})

    diag = {
        'stage': 'R6B.3 standalone audit build',
        'labels_enter_neural_input': False,
        'labels_enter_msdl_learning': False,
        'labels_enter_generation': False,
        'tokenizer_metadata': tok_meta,
        'tokenizer_activity_train': sp_tr,
        'token_stats_train': token_stats(tr),
        'input_representation': input_audit,
        'scaffold': scaffold_audit,
        'training_protocol': training_audit,
        'scaffold_eval': scaffold_eval,
        'final_eval': final_eval,
        'history': hist,
        'generation_trace_first': gtrace[0] if gtrace else None,
        'config': vars(a),
    }
    (a.out / 'metrics.json').write_text(json.dumps(diag, indent=2, default=str))
    print(json.dumps({'done': True, 'out': str(a.out), 'final_eval': final_eval}, indent=2), flush=True)


def structural_self_test():
    c = controller_self_test()
    m = masking_self_test()
    # Algebraic P3 smoke test independent of tokenizer checkpoint.
    cfg = Config(); model = init_model(cfg, 7311)
    assert model['WR'].shape == (64, 97) and np.all(model['WR'] == 0)
    assert model['WQ'].shape == (96, 96) and model['W1'].shape == (192, 96)
    return {'pass': True, 'maskgit_controller': c, 'mask_sampler': m,
            'p3_shapes': {'WR': list(model['WR'].shape), 'WQ': list(model['WQ'].shape), 'W1': list(model['W1'].shape)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--data', type=Path)
    ap.add_argument('--tokenizer', type=Path)
    ap.add_argument('--out', type=Path, default=Path('r6b3_standalone_out'))
    ap.add_argument('--D', type=int, default=96); ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--features', type=int, default=24); ap.add_argument('--ff', type=int, default=192); ap.add_argument('--R', type=int, default=7)
    ap.add_argument('--scaffold-contexts', type=int, default=2000); ap.add_argument('--scaffold-paths', type=int, default=2500); ap.add_argument('--scaffold-lr', type=float, default=0.6)
    ap.add_argument('--events', type=int, default=30000); ap.add_argument('--eval-images', type=int, default=128); ap.add_argument('--gen-samples', type=int, default=36)
    ap.add_argument('--internal-lr-scale', type=float, default=0.5); ap.add_argument('--encode-batch', type=int, default=256)
    ap.add_argument('--decode-iterations', type=int, default=8); ap.add_argument('--choice-temperature', type=float, default=4.5)
    ap.add_argument('--seed', type=int, default=7311); ap.add_argument('--report-every', type=int, default=5000); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(structural_self_test(), indent=2)); return
    if a.data is None or a.tokenizer is None:
        ap.error('full run requires --data MNIST_NPZ and --tokenizer R6B1_TOKENIZER_PT')
    run_full(a)


if __name__ == '__main__':
    main()
