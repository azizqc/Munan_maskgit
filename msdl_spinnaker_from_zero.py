"""
MSDL / SpiNNaker -- single-file canonical from-zero experiment
===============================================================

Required user files
-------------------
1. msdl_spinnaker_from_zero.py   (this file)
2. mnist.npz                     (standard Keras MNIST database)

Run
---
    python msdl_spinnaker_from_zero.py

What this file does
-------------------
A) Reads raw MNIST 0/1/2 examples and generates every 9072-LIF forward/noising
   training trajectory fresh in this run.
B) Starts W=H=0 and performs the locked historical Phase-I V4 MSDL learning on
   SpiNNaker using the validated explicit 83-tree multicast transport.
C) Learns frozen-v5 slow depth/self-B from the same experienced trajectories.
D) Freezes learned state and generates images autonomously. The target prior is
   not supplied to generation; there is no explicit count controller and no
   classifier feedback loop.
E) Fast 9072-LIF + 1-ms capture/direct-entry dynamics remain resident on
   SpiNNaker for each image. Slow three-channel Izh state is evaluated only
   between images (functional HIL boundary).

Scientific freeze
-----------------
This standalone file is a packaging consolidation of the already validated
V4LOCKED + next-step gate-buffer implementation. It does not introduce a new
learning rule, routing scheme, controller, prior injection, or numerical model.
Exact frozen implementation snapshots are embedded below and hash-checked at
startup so no external Python/C source files are required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


# =============================================================================
# USER CONFIGURATION -- canonical validated defaults
# =============================================================================
# A new user normally edits nothing. Put standard Keras mnist.npz beside this
# script and run:  python msdl_spinnaker_from_zero.py
#
# These are DEFAULTS only. In JupyterLab / IPython, command-line overrides are
# supported through %run, e.g.
#   %run msdl_spinnaker_from_zero.py --mode smoke --smoke-prior 0.2,0.6,0.2
# Scientific code below is unchanged; the CLI only selects the requested run.
TARGET_PRIOR = (0.6, 0.2, 0.2)
TRAIN_CYCLES = 50
N_GENERATE = 100
PATH_SEED_START = 20000
GENERATION_SEED_START = 910000
DATA_SEED = 20261011
PATH_WORKERS = 4
PIXEL_PREPROCESS = "whitelam-rms"
BCAST_SLOTS = 64
TIMER_TICK_US = 100
GENERATION_TIMEOUT_S = 180.0
TRAINING_TIMEOUT_S = 7200.0
QUALITY_GATE = 0.70
OUTPUT_DIR_NAME = "MSDL_FROM_ZERO_RUN"



# Frozen v5 hardware choices -------------------------------------------------
V5_CONTROL_DT_MS = 1            # matches cloud capture update cadence
V5_LATE_FRAC = 0.35             # matches cloud late-activity window
V5_LATE_STRIDE = 4              # matches cloud `ii % 4 == 0`
DYN_PREF0 = 156                 # three global preference words: 156..158
OUT_V5_LATE_R0 = 37             # three Q24 late-activity words: 37..39
OUT_V5_CAPTURE = 40             # Q24 final capture state
Q24 = 24
ONE_Q24 = 1 << Q24

PRIORS = {
    "p0": np.asarray([0.6, 0.2, 0.2], dtype=np.float64),
    "p1": np.asarray([0.2, 0.6, 0.2], dtype=np.float64),
    "p2": np.asarray([0.2, 0.2, 0.6], dtype=np.float64),
    "uniform": np.asarray([1/3, 1/3, 1/3], dtype=np.float64),
}


def parse_prior_text(text: str) -> np.ndarray:
    vals = np.asarray([float(x.strip()) for x in str(text).split(",")], dtype=np.float64)
    if vals.shape != (3,) or np.any(~np.isfinite(vals)) or np.any(vals < 0) or float(vals.sum()) <= 0:
        raise ValueError(f"invalid prior {text!r}; expected three non-negative finite values")
    vals /= float(vals.sum())
    return vals


# =============================================================================
# Embedded frozen cloud-v5 scientific core
# =============================================================================
# This removes the runtime dependency on cloud_msdl_full_dynamics_v5.py while
# preserving exactly the subset of frozen-v5 equations used by the hardware
# bridge.  The optional --v5-script argument is audit-only.

_EMB_N = 1296
_EMB_NV = 784
_EMB_NH = 512
_EMB_DT = 1e-3
_EMB_GENERATION_S = 4.32
_EMB_EVENT_WINDOW_MS = 10.0
_EMB_AF, _EMB_AS = 0.62, 0.92
_EMB_WF, _EMB_WS = 0.70, 0.30
_EMB_FORCE_DT_MS = 80.0
_EMB_TAU_F_MS = -_EMB_FORCE_DT_MS / math.log(_EMB_AF)
_EMB_TAU_S_MS = -_EMB_FORCE_DT_MS / math.log(_EMB_AS)
_EMB_RF, _EMB_RS = 1.0/_EMB_TAU_F_MS, 1.0/_EMB_TAU_S_MS


def _emb_low_discrepancy_schedule(prior, n):
    p = np.asarray(prior, dtype=float)
    p = p / p.sum()
    c = np.zeros(3, dtype=int)
    out = []
    for t in range(int(n)):
        deficit = (t + 1) * p - c
        k = int(np.argmax(deficit))
        out.append(k)
        c[k] += 1
    return np.asarray(out, dtype=int)


class _EmbIzhFastRealization:
    def __init__(self, a_izh=1e-3):
        S = -(_EMB_RF + _EMB_RS)
        P = _EMB_RF * _EMB_RS
        self.a = float(a_izh)
        self.q = S + self.a
        self.b = self.q + P / self.a
        self.v0 = (self.q - 5.0) / 0.08
        self.u0 = self.b * self.v0
        self.I0 = -0.04*self.v0*self.v0 - 5.0*self.v0 - 140.0 + self.u0
        self.num_s = _EMB_WF*_EMB_RF + _EMB_WS*_EMB_RS
        self.num_0 = _EMB_RF*_EMB_RS

    def readout(self, gain):
        g = float(gain)
        cv = self.num_s / g
        cu = (self.num_0/g - cv*self.a) / (self.a*self.b)
        return np.asarray([cv, cu], dtype=float)


@njit(cache=True)
def _emb_fast_izh_step(v, u, target_input, pred_input, gain, a, b, v0, u0, I0, cv, cu, n_sub=80):
    spikes = 0
    for _ in range(n_sub):
        for j in range(v.shape[0]):
            I = I0 + gain*target_input[j] - gain*pred_input[j]
            dv1 = 0.04*v[j]*v[j] + 5.0*v[j] + 140.0 - u[j] + I
            du1 = a*(b*v[j] - u[j])
            vm = v[j] + 0.5*dv1
            um = u[j] + 0.5*du1
            dv2 = 0.04*vm*vm + 5.0*vm + 140.0 - um + I
            du2 = a*(b*vm - um)
            v[j] += dv2
            u[j] += du2
            if v[j] >= 30.0:
                v[j] = -65.0
                u[j] += 8.0
                spikes += 1
    out = np.empty(v.shape[0], dtype=np.float64)
    for j in range(v.shape[0]):
        out[j] = cv*(v[j]-v0) + cu*(u[j]-u0)
    return out, spikes


class _EmbPoleMatchedIzh:
    def __init__(self, discrete_poles, a_izh, window_ms=_EMB_EVENT_WINDOW_MS):
        p1, p2 = map(float, discrete_poles)
        T = float(window_ms)
        r1 = -math.log(p1)/T
        r2 = -math.log(p2)/T
        S = -(r1+r2)
        P = r1*r2
        self.a = float(a_izh)
        self.q = S + self.a
        self.b = self.q + P/self.a
        self.v0 = (self.q - 5.0)/0.08
        self.u0 = self.b*self.v0
        self.I0 = -0.04*self.v0*self.v0 - 5.0*self.v0 - 140.0 + self.u0
        self.A = np.asarray([[self.q, -1.0],[self.a*self.b, -self.a]], dtype=float)
        self.Phi = expm(self.A*T)
        self.T = T

    def transfer_readout(self, gain, num_s, num_0):
        g = float(gain)
        cv = float(num_s)/g
        cu = (float(num_0)/g - cv*self.a)/(self.a*self.b)
        return np.asarray([cv, cu], dtype=float)


class _EmbOnePoleIzhRealization:
    def __init__(self, target_a, gain=1e-4, extra=.5, a_izh=.005):
        self.gain = float(gain)
        self.izh = _EmbPoleMatchedIzh((target_a, extra), a_izh=a_izh)
        rt = -math.log(target_a)/_EMB_EVENT_WINDOW_MS
        re = -math.log(extra)/_EMB_EVENT_WINDOW_MS
        self.C = self.izh.transfer_readout(self.gain, rt, rt*re)


@njit(cache=True)
def _emb_slow_izh_window(v, u, inp, gain, a, b, I0, steps):
    spikes = 0
    for _ in range(steps):
        for k in range(v.shape[0]):
            I = I0 + gain*inp[k]
            dv1 = 0.04*v[k]*v[k] + 5.0*v[k] + 140.0 - u[k] + I
            du1 = a*(b*v[k] - u[k])
            vm = v[k] + 0.5*dv1
            um = u[k] + 0.5*du1
            dv2 = 0.04*vm*vm + 5.0*vm + 140.0 - um + I
            du2 = a*(b*vm - um)
            v[k] += dv2
            u[k] += du2
            if v[k] >= 30.0:
                v[k] = -65.0
                u[k] += 8.0
                spikes += 1
    return spikes


_EMB_FP = SimpleNamespace(
    N=_EMB_N, NV=_EMB_NV, NH=_EMB_NH, DT=_EMB_DT, GENERATION_S=_EMB_GENERATION_S,
    EVENT_WINDOW_MS=_EMB_EVENT_WINDOW_MS, low_discrepancy_schedule=_emb_low_discrepancy_schedule,
    IzhFastRealization=_EmbIzhFastRealization, fast_izh_step=_emb_fast_izh_step,
    OnePoleIzhRealization=_EmbOnePoleIzhRealization, slow_izh_window=_emb_slow_izh_window,
)

# Frozen v5 constants.
_EMB_FORMULA_VERSION = "msdl-full-dynamics-v5-selfB-IzhWTA-directentry"
_EMB_TRAIN_CYCLES_DEFAULT = 50
_EMB_G_E = 1e-5
_EMB_G_M = 1e-6
_EMB_ACT_WTA_I0 = -5.0
_EMB_ACT_WTA_GAIN = 18.0
_EMB_ACT_WTA_LATERAL = 4.0
_EMB_ACT_WTA_DECAY = 0.90
_EMB_ACT_WTA_STEPS = 150
_EMB_B_LEARN_ETA = 0.05
_EMB_B_TANGENT_WEIGHT = 2.0
_EMB_PREF_WTA_I0 = 2.0
_EMB_PREF_WTA_GAIN = 15.0
_EMB_PREF_WTA_LATERAL = 4.0
_EMB_PREF_WTA_DECAY = 0.90
_EMB_PREF_WTA_STEPS = 120
_EMB_DEPTH_BETA = 2.2
_EMB_ADAPT_GAIN = 2.8
_EMB_ADAPT_COMMON_RHO = 0.25
_EMB_ADAPT_DEPTH_KAPPA = 1.5
_EMB_DIRECT_GAIN = 0.8
_EMB_CAPTURE_TAU_S = 0.30
_EMB_LATE_ACTIVITY_FRAC = 0.35


def _emb_rmsnorm(v):
    x = np.asarray(v, dtype=np.float64).copy()
    x -= np.mean(x)
    rms = float(np.sqrt(np.mean(x*x)))
    if not np.isfinite(rms) or rms <= 1e-30:
        raise ValueError("degenerate vector in rmsnorm")
    return x/rms


class _EmbSpikingIzhWTA:
    def __init__(self, *, I0, gain, lateral, decay, steps):
        self.I0=float(I0); self.gain=float(gain); self.lateral=float(lateral)
        self.decay=float(decay); self.steps=int(steps)

    def transform(self, drive):
        r = np.asarray(drive, dtype=np.float64)
        if r.shape != (3,) or np.any(~np.isfinite(r)):
            raise ValueError("WTA drive must be finite length-3")
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        r = np.full(3, 1/3, dtype=float) if s <= 1e-30 else r/s
        v = np.full(3, -65.0, dtype=np.float64)
        u = 0.2*v.copy(); inh=np.zeros(3); counts=np.zeros(3)
        for _ in range(self.steps):
            I = self.I0 + self.gain*r - self.lateral*(float(np.sum(inh))-inh)
            for _half in range(2):
                v += 0.5*(0.04*v*v + 5.0*v + 140.0 - u + I)
            u += 0.02*(0.2*v-u)
            sp = v >= 30.0
            counts += sp.astype(np.float64)
            v[sp] = -65.0; u[sp] += 8.0
            inh = self.decay*inh + sp.astype(np.float64)
        total=float(np.sum(counts))
        return r if total <= 0 else counts/total


def _emb_make_activity_wta():
    return _EmbSpikingIzhWTA(I0=_EMB_ACT_WTA_I0, gain=_EMB_ACT_WTA_GAIN, lateral=_EMB_ACT_WTA_LATERAL, decay=_EMB_ACT_WTA_DECAY, steps=_EMB_ACT_WTA_STEPS)

def _emb_make_preference_wta():
    return _EmbSpikingIzhWTA(I0=_EMB_PREF_WTA_I0, gain=_EMB_PREF_WTA_GAIN, lateral=_EMB_PREF_WTA_LATERAL, decay=_EMB_PREF_WTA_DECAY, steps=_EMB_PREF_WTA_STEPS)


class _EmbSlowDepthLearner:
    def __init__(self):
        self.real = _EMB_FP.OnePoleIzhRealization(0.99, gain=1e-4)
        self.v = np.full(3, self.real.izh.v0, dtype=np.float64)
        self.u = np.full(3, self.real.izh.u0, dtype=np.float64)
    def update(self, drive):
        x=np.maximum(np.asarray(drive,dtype=float),0.0); x/=float(np.sum(x))+1e-30
        _EMB_FP.slow_izh_window(self.v,self.u,x,self.real.gain,self.real.izh.a,self.real.izh.b,self.real.izh.I0,int(round(_EMB_FP.EVENT_WINDOW_MS)))
    def normalized(self):
        z=np.stack([self.v-self.real.izh.v0,self.u-self.real.izh.u0],axis=-1)
        y=np.maximum(z@self.real.C,1e-12)
        return y/np.sum(y)


class _EmbSlowAdaptation:
    def __init__(self):
        self.real = _EMB_FP.OnePoleIzhRealization(0.82, gain=1e-4)
        self.v=np.full(3,self.real.izh.v0,dtype=float); self.u=np.full(3,self.real.izh.u0,dtype=float)
    def readout(self):
        z=np.stack([self.v-self.real.izh.v0,self.u-self.real.izh.u0],axis=-1)
        return z@self.real.C
    def update(self,drive):
        r=np.maximum(np.asarray(drive,dtype=float),0.0); r/=float(np.sum(r))+1e-30
        _EMB_FP.slow_izh_window(self.v,self.u,r,self.real.gain,self.real.izh.a,self.real.izh.b,self.real.izh.I0,int(round(_EMB_FP.EVENT_WINDOW_MS)))


def _emb_internal_msdl_activity(E_path, vE, uE, cE, v0, u0):
    E=np.asarray(E_path,dtype=float)
    z=np.stack([np.asarray(vE)-v0,np.asarray(uE)-u0],axis=-1)
    addr=(z@np.asarray(cE))[:,_EMB_NV:]
    addr=addr-np.mean(addr,axis=0,keepdims=True)
    an=np.linalg.norm(addr,axis=1)
    eh=E[:,_EMB_NV:]; ehn=np.linalg.norm(eh,axis=1)+1e-30
    score=np.zeros(3,dtype=float)
    for j in range(3):
        if an[j] > 1e-12:
            proj=(eh@addr[j])/(ehn*an[j]+1e-30)
            score[j]=float(np.mean(proj*proj))
    tot=float(np.sum(score))
    return np.full(3,1/3,dtype=float) if tot<=1e-30 else score/tot


def _emb_slow_score(depth, adaptation):
    d=np.asarray(depth,dtype=float); a=np.asarray(adaptation,dtype=float)
    a0=a-_EMB_ADAPT_COMMON_RHO*float(np.mean(a))
    scale=1.0+_EMB_ADAPT_DEPTH_KAPPA*(1/3-d)
    return _EMB_DEPTH_BETA*(d-1/3)-_EMB_ADAPT_GAIN*a0*scale


def _emb_learn_preference(score, wta):
    x=np.asarray(score,dtype=float)
    if x.shape!=(3,) or np.any(~np.isfinite(x)):
        raise ValueError("preference score must be finite length-3")
    x=x-float(np.mean(x))
    v=np.full(3,-65.0,dtype=float); u=0.2*v.copy(); inh=np.zeros(3); counts=np.zeros(3)
    for _ in range(wta.steps):
        I=wta.I0+wta.gain*x-wta.lateral*(float(np.sum(inh))-inh)
        for _half in range(2):
            v += 0.5*(0.04*v*v+5.0*v+140.0-u+I)
        u += 0.02*(0.2*v-u)
        sp=v>=30.0; counts += sp.astype(float); v[sp]=-65.0; u[sp]+=8.0
        inh=wta.decay*inh+sp.astype(float)
    total=float(np.sum(counts))
    if total <= 0:
        y=np.maximum(x-np.min(x),0.0); s=float(np.sum(y))
        return np.full(3,1/3,dtype=float) if s<=1e-30 else y/s
    return counts/total


_EMBEDDED_V5 = SimpleNamespace(
    __file__=None, _is_embedded=True, fp=_EMB_FP, FORMULA_VERSION=_EMB_FORMULA_VERSION,
    TRAIN_CYCLES_DEFAULT=_EMB_TRAIN_CYCLES_DEFAULT, G_E=_EMB_G_E, G_M=_EMB_G_M,
    B_LEARN_ETA=_EMB_B_LEARN_ETA, B_TANGENT_WEIGHT=_EMB_B_TANGENT_WEIGHT,
    DEPTH_BETA=_EMB_DEPTH_BETA, ADAPT_GAIN=_EMB_ADAPT_GAIN, ADAPT_COMMON_RHO=_EMB_ADAPT_COMMON_RHO,
    ADAPT_DEPTH_KAPPA=_EMB_ADAPT_DEPTH_KAPPA, DIRECT_GAIN=_EMB_DIRECT_GAIN, CAPTURE_TAU_S=_EMB_CAPTURE_TAU_S,
    LATE_ACTIVITY_FRAC=_EMB_LATE_ACTIVITY_FRAC, rmsnorm=_emb_rmsnorm, SpikingIzhWTA=_EmbSpikingIzhWTA,
    make_activity_wta=_emb_make_activity_wta, make_preference_wta=_emb_make_preference_wta,
    SlowDepthLearner=_EmbSlowDepthLearner, SlowAdaptation=_EmbSlowAdaptation,
    internal_msdl_activity=_emb_internal_msdl_activity, slow_score=_emb_slow_score, learn_preference=_emb_learn_preference,
)


def load_frozen_v5_optional(path):
    """Return embedded frozen-v5 core. If an external v5 file is supplied, audit parity."""
    if path is None:
        return _EMBEDDED_V5
    p=Path(path).expanduser().resolve()
    if not p.exists():
        print(f"[v5] external reference not found: {p}; using embedded frozen v5 core", flush=True)
        return _EMBEDDED_V5
    ext=load_module(p,"msdl_v5_cloud_ref_external")
    keys=["FORMULA_VERSION","DIRECT_GAIN","CAPTURE_TAU_S","B_LEARN_ETA","ADAPT_GAIN","B_TANGENT_WEIGHT"]
    for k in keys:
        a=getattr(ext,k); b=getattr(_EMBEDDED_V5,k)
        if isinstance(a,str):
            ok=(a==b)
        else:
            ok=abs(float(a)-float(b))<1e-15
        if not ok:
            raise RuntimeError(f"external v5 mismatch for {k}: {a!r} != embedded {b!r}")
    print(f"[v5] external reference parity PASS: {p}", flush=True)
    return _EMBEDDED_V5



def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod



# ---------------------------------------------------------------------------
# HARD-LOCKED historical Phase-I V4 training substrate
# ---------------------------------------------------------------------------
# These hashes are from the saved V4 files that previously established the
# explicit 83-tree training path.  Any mismatch aborts before board contact.
PHASE1_V4_EXPECTED_SHA256 = {
    "trainer": "6c238bbe40ae84c2936e22d77b6082850dcb6b412a5181bf92a1579e6e13b77a",
    "routes":  "63fe1aec976edd1851e25c3ecb60b4df0fedc9f8fa8bc3c72842c66b6447a635",
    "kernel":  "8eacc79d7cf32c31fa0bc8cb4407de1a521f276d5d0b9fbfc0ed28692effaf75",
}
_LOCKED_PHASE1_V4 = None
_LOCKED_PHASE1_KERNEL = None
_LOCKED_PHASE1_MANIFEST = None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_embedded_module(module_name: str, source: str, file_hint: str):
    """Execute one frozen source snapshot in its own module namespace."""
    import types
    mod = types.ModuleType(module_name)
    mod.__file__ = file_hint
    mod.__package__ = None
    sys.modules[module_name] = mod
    exec(compile(source, file_hint, "exec"), mod.__dict__)
    return mod


def configure_locked_phase1_v4(output_dir: Path) -> dict:
    """Configure the exact historical V4 substrate from embedded snapshots.

    This is packaging only. The V4 trainer, route helper, and C kernel bytes are
    unchanged from the locked files that were previously used on the board.
    """
    global _LOCKED_PHASE1_V4, _LOCKED_PHASE1_KERNEL, _LOCKED_PHASE1_MANIFEST

    actual = {
        "base": _sha256_text(_EMBEDDED_BASE_SOURCE),
        "trainer": _sha256_text(_EMBEDDED_PHASE1_TRAINER_SOURCE),
        "routes": _sha256_text(_EMBEDDED_PHASE1_ROUTES_SOURCE),
        "kernel": _sha256_text(_EMBEDDED_PHASE1_KERNEL_SOURCE),
    }
    for key, expected in _EMBEDDED_EXPECTED_SHA256.items():
        if actual[key] != expected:
            raise RuntimeError(
                f"embedded frozen component {key} hash mismatch: "
                f"expected {expected}, got {actual[key]}"
            )

    # Load the route helper under its historical import name, then execute the
    # historical trainer unchanged. No route function is monkey-patched.
    rmod = _load_embedded_module(
        "phase1_custom_broadcast_routes",
        _EMBEDDED_PHASE1_ROUTES_SOURCE,
        "<embedded:phase1_custom_broadcast_routes.py>",
    )
    tmod = _load_embedded_module(
        "msdl_locked_phase1_v4",
        _EMBEDDED_PHASE1_TRAINER_SOURCE,
        "<embedded:spinnaker_phase1_train_generate_v4.py>",
    )
    if not hasattr(tmod, "run_board_training"):
        raise RuntimeError("embedded V4 trainer has no run_board_training()")
    if tmod.build_explicit_broadcast_routes is not rmod.build_explicit_broadcast_routes:
        raise RuntimeError("embedded V4 trainer did not bind the frozen route planner")
    if tmod.install_and_verify_explicit_routes is not rmod.install_and_verify_explicit_routes:
        raise RuntimeError("embedded V4 trainer did not bind the frozen route installer")

    # The V4 trainer build API accepts a C-file path. Materialise only this exact
    # embedded kernel into the run output; it is not an external input.
    internal = Path(output_dir) / "_internal_locked_v4"
    internal.mkdir(parents=True, exist_ok=True)
    kernel_path = internal / "phase1_train_kernel_v4.c"
    kernel_path.write_text(_EMBEDDED_PHASE1_KERNEL_SOURCE, encoding="utf-8")
    if sha256_file(kernel_path) != _EMBEDDED_EXPECTED_SHA256["kernel"]:
        raise RuntimeError("materialised embedded V4 kernel hash mismatch")

    _LOCKED_PHASE1_V4 = tmod
    _LOCKED_PHASE1_KERNEL = kernel_path
    _LOCKED_PHASE1_MANIFEST = {
        "packaging": "single-file embedded frozen snapshots",
        "base_sha256": _EMBEDDED_EXPECTED_SHA256["base"],
        "trainer_sha256": _EMBEDDED_EXPECTED_SHA256["trainer"],
        "routes_sha256": _EMBEDDED_EXPECTED_SHA256["routes"],
        "kernel_sha256": _EMBEDDED_EXPECTED_SHA256["kernel"],
        "route_module_identity_check": True,
        "base_training_api_used": False,
        "router_cam_clear_used": False,
        "route_monkey_patch_used": False,
        "external_python_or_c_dependency": False,
    }
    return dict(_LOCKED_PHASE1_MANIFEST)


def load_embedded_base(single_file_path: Path):
    """Load the exact frozen raw-MNIST/generation base from this single file."""
    if _sha256_text(_EMBEDDED_BASE_SOURCE) != _EMBEDDED_EXPECTED_SHA256["base"]:
        raise RuntimeError("embedded generation base hash mismatch")
    # Set __file__ to this standalone script so the unchanged base locates
    # mnist.npz beside it exactly as a normal file-based module would.
    return _load_embedded_module(
        "msdl_v5_hw_base",
        _EMBEDDED_BASE_SOURCE,
        str(Path(single_file_path).resolve()),
    )

def run_locked_phase1_v4_training(payload: Path, outdir: Path, args) -> dict:
    """Call the historical V4 trainer directly; do not wrap/replace its routing."""
    if _LOCKED_PHASE1_V4 is None or _LOCKED_PHASE1_KERNEL is None:
        raise RuntimeError("locked Phase-I V4 was not configured")
    checkpoint, report = _LOCKED_PHASE1_V4.run_board_training(
        Path(payload), Path(_LOCKED_PHASE1_KERNEL), Path(outdir),
        args.build_tree, None,
        timeout_s=float(args.training_timeout_s),
        bcast_slots=int(args.bcast_slots),
    )
    if not report.get("train_pass", False):
        raise RuntimeError("locked Phase-I V4 training did not pass")
    if not report.get("weights_start_from_zero", False):
        raise RuntimeError("locked Phase-I V4 report does not certify exact-zero W/H start")
    if int(report.get("n_worker_errors", -1)) != 0:
        raise RuntimeError("locked Phase-I V4 reported worker errors despite train_pass")
    if not report.get("internal_consistency", {}).get("pass", False):
        raise RuntimeError("locked Phase-I V4 W/H internal consistency failed")

    raw_path = Path(outdir) / "PHASE1_BOARD_RAW_Q.npz"
    if not raw_path.is_file():
        raise FileNotFoundError(f"V4 raw fixed-point output missing: {raw_path}")
    with np.load(raw_path, allow_pickle=False) as raw:
        Wq = np.asarray(raw["W_q24"], dtype=np.int32).copy()
        Hq = np.asarray(raw["H_q24"], dtype=np.int32).copy()
        Ereadq = np.asarray(raw["Eread_q20"], dtype=np.int32).copy()
    with np.load(checkpoint, allow_pickle=False) as cp:
        exposure = np.asarray(cp["exposure"], dtype=np.int32).copy()

    report = dict(report)
    report["board_training_wall_s"] = report.get("custom_training_wall_s")
    report["locked_v4_manifest"] = dict(_LOCKED_PHASE1_MANIFEST)
    return {
        "checkpoint": Path(checkpoint),
        "Wq": Wq,
        "Hq": Hq,
        "Ereadq": Ereadq,
        "exposure": exposure,
        "report": report,
    }

def q24_scalar(x: float) -> int:
    y = int(np.rint(float(x) * float(1 << Q24)))
    if y < -2**31 or y > 2**31 - 1:
        raise OverflowError(f"Q24 scalar overflow: {x}")
    return y


def prior_tag(p: np.ndarray) -> str:
    p = np.asarray(p, float)
    for k, v in PRIORS.items():
        if np.max(np.abs(p - v)) < 1e-10:
            return k
    return "custom"


# ---------------------------------------------------------------------------
# Frozen-v5 scientific audit / input preparation
# ---------------------------------------------------------------------------

def validate_v5_contract(v5) -> dict:
    """Static contract checks. Does not run a large-N experiment."""
    if getattr(v5, "_is_embedded", False):
        forbidden_generation = {
            "offline_labeled_actuator_function": False,
            "explicit_power_competition": False,
            "classifier_mode_drives_adaptation": False,
        }
        zero_ok = True
        api_ok = True
    else:
        src = Path(v5.__file__).read_text(encoding="utf-8")
        forbidden_generation = {
            "offline_labeled_actuator_function": "def derive_rank1_actuators(" in src,
            "explicit_power_competition": "np.power(" in src,
            "classifier_mode_drives_adaptation": "adapt.update(mode)" in src,
        }
        zero_ok = ("W = np.zeros((fp.NV, fp.NH)" in src and
                   "H = np.zeros((fp.NH, fp.NH)" in src)
        api_ok = "def run_generation(checkpoint: Path, n: int, seed_start: int, output: Path)" in src
    return {
        "formula_version": str(v5.FORMULA_VERSION),
        "W_H_exact_zero_in_source": zero_ok,
        "generation_api_has_no_prior_argument": api_ok,
        "forbidden_interface_present": forbidden_generation,
        "forbidden_interface_check_pass": bool(not any(forbidden_generation.values())),
        "direct_gain": float(v5.DIRECT_GAIN),
        "capture_tau_s": float(v5.CAPTURE_TAU_S),
        "B_learn_eta": float(v5.B_LEARN_ETA),
        "adapt_gain": float(v5.ADAPT_GAIN),
    }


def make_prior_payload(common_payload: Path, prior: np.ndarray, cycles: int,
                       out_path: Path, v5, base) -> dict:
    """Build a no-reuse 50-cycle payload by selecting immutable cached paths.

    This does not generate or learn any state; it only chooses which already
    validated LIF forward/noising trajectory is presented at each exposure.
    """
    d = np.load(common_payload, allow_pickle=True)
    paths_all = np.asarray(d["paths_q20"], dtype=np.int32)
    labels_all = np.asarray(
        d["selected_train_labels"] if "selected_train_labels" in d.files else d["schedule"],
        dtype=np.int64,
    )
    if paths_all.ndim != 3 or paths_all.shape[1:] != (28, base.N):
        raise ValueError(f"unexpected common path shape {paths_all.shape}")
    if labels_all.shape != (len(paths_all),):
        raise ValueError("common payload labels/path mismatch")

    p = np.asarray(prior, dtype=np.float64)
    p /= p.sum()
    schedule = v5.fp.low_discrepancy_schedule(p, int(cycles)).astype(np.int64)
    groups = [np.flatnonzero(labels_all == k) for k in range(3)]
    need = np.bincount(schedule, minlength=3)
    have = np.asarray([len(x) for x in groups], dtype=int)
    if np.any(need > have):
        raise RuntimeError(f"payload would reuse trajectories: need={need.tolist()} have={have.tolist()}")

    cursor = np.zeros(3, dtype=int)
    chosen = []
    for k in schedule:
        k = int(k)
        chosen.append(int(groups[k][cursor[k]]))
        cursor[k] += 1
    chosen = np.asarray(chosen, dtype=np.int64)
    paths = paths_all[chosen]

    constants_json = d["constants_json"] if "constants_json" in d.files else np.asarray([json.dumps(base.train_izh_constants(), sort_keys=True)])
    seeds_all = np.asarray(d["seeds"], dtype=np.int64) if "seeds" in d.files else np.arange(len(paths_all), dtype=np.int64)
    idx_all = np.asarray(d["selected_train_indices"], dtype=np.int64) if "selected_train_indices" in d.files else chosen.copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        paths_q20=paths,
        schedule=schedule.astype(np.int16),
        prior=p,
        reuse=np.zeros(3, dtype=np.int16),
        constants_json=constants_json,
        seeds=seeds_all[chosen],
        selected_train_indices=idx_all[chosen],
        selected_train_labels=schedule.astype(np.int16),
        source=np.asarray(["immutable cached 9072-LIF paths; no learned state"]),
    )
    return {
        "prior": p.tolist(),
        "cycles": int(cycles),
        "schedule_counts": need.astype(int).tolist(),
        "available_per_class": have.astype(int).tolist(),
        "trajectory_reuse": False,
        "common_payload_sha256": sha256_file(common_payload),
        "derived_payload_sha256": sha256_file(out_path),
        "chosen_common_indices": chosen.astype(int).tolist(),
    }


def learn_slow_states_only(payload: Path, symbols: np.ndarray, v5) -> dict:
    """Exact frozen-v5 slow/depth/B learner, without duplicating W/H learning.

    This is valid because v5 internal attractor activity and B learning depend on
    the E-Izh path dynamics and immutable trajectory observables, not on W/H or M.
    Board Phase-I remains the only W/H learner.
    """
    d = np.load(payload, allow_pickle=True)
    paths = np.asarray(d["paths_q20"], dtype=np.float64) / float(1 << 20)
    schedule = np.asarray(d["schedule"], dtype=np.int64)
    symbols = np.asarray(symbols, dtype=np.float32)
    if symbols.shape != (3, v5.fp.NV):
        raise ValueError(f"bad symbols shape {symbols.shape}, expected {(3, v5.fp.NV)}")
    v5.fp.SYMBOLS = symbols.copy()

    real = v5.fp.IzhFastRealization(a_izh=1e-3)
    cE = real.readout(v5.G_E)
    vE = np.full((3, v5.fp.N), real.v0, dtype=np.float64)
    uE = np.full((3, v5.fp.N), real.u0, dtype=np.float64)
    zero = np.zeros(v5.fp.N, dtype=np.float64)
    depth = v5.SlowDepthLearner()
    wta = v5.make_activity_wta()
    T = np.zeros((3, v5.fp.NH), dtype=np.float64)
    P = np.zeros((3, v5.fp.NH), dtype=np.float64)
    C = np.zeros(v5.fp.NH, dtype=np.float64)
    diag = []

    for cyc, k0 in enumerate(schedule):
        k = int(k0)
        st = paths[cyc]
        X1 = st[1:]
        E = []
        for t in range(len(X1)):
            e, _ = v5.fp.fast_izh_step(
                vE[k], uE[k], X1[t], zero, v5.G_E,
                real.a, real.b, real.v0, real.u0, real.I0,
                cE[0], cE[1],
            )
            E.append(e)
        E = np.asarray(E)
        raw = v5.internal_msdl_activity(E, vE, uE, cE, real.v0, real.u0)
        r = wta.transform(raw)
        depth.update(r)

        tangent = np.mean(st[18:27, v5.fp.NV:] - st[19:28, v5.fp.NV:], axis=0)
        noisy = np.mean(st[22:28, v5.fp.NV:], axis=0)
        C += v5.B_LEARN_ETA * (noisy - C)
        for j in range(3):
            w = float(r[j])
            T[j] += v5.B_LEARN_ETA * w * (tangent - T[j])
            P[j] += v5.B_LEARN_ETA * w * (noisy - P[j])
        if cyc in (0, len(schedule) - 1):
            diag.append({"cycle": cyc + 1, "raw": raw.tolist(), "wta": r.tolist(), "depth": depth.normalized().tolist()})

    B = []
    for j in range(3):
        Bj = v5.rmsnorm(v5.B_TANGENT_WEIGHT * v5.rmsnorm(T[j]) + v5.rmsnorm(P[j] - C))
        B.append(Bj)
    B = np.stack(B)
    return {
        "depth": depth.normalized(),
        "actuators": B,
        "symbols": symbols,
        "diagnostic": diag,
    }


# ---------------------------------------------------------------------------
# Persistent generator kernel patch: v5 direct-entry + exact 1-ms capture
# ---------------------------------------------------------------------------

def patch_generator_kernel_v5(base, v5) -> str:
    """Patch the already validated packed persistent generator in memory only."""
    src = str(base.KERNEL_C)
    rho_q24 = q24_scalar(math.exp(-v5.fp.DT / v5.CAPTURE_TAU_S))
    one_minus_rho_q24 = ONE_Q24 - rho_q24
    direct_gain_q24 = q24_scalar(v5.DIRECT_GAIN)

    patches: list[tuple[str, str]] = []

    patches.append((
        "#define DYN_CTRL0         144u\n#define DYN_WORDS         160u",
        "#define DYN_CTRL0         144u\n"
        "#define DYN_PREF0         156u  /* 3 Q24 preference values */\n"
        "#define DYN_WORDS         160u",
    ))
    patches.append((
        "#define OUT_AXIS_FINAL0      34u\n#define OUT_WORDS            64u",
        "#define OUT_AXIS_FINAL0      34u\n"
        "#define OUT_V5_LATE_R0       37u  /* 3 Q24 normalized late basin activities */\n"
        "#define OUT_V5_CAPTURE       40u  /* Q24 final capture state */\n"
        "#define OUT_WORDS            64u",
    ))
    patches.append((
        "static int32_t ctrl_coeff_q24[PACK][N_AXES];\n"
        "static int32_t ctrl_z_q24[N_AXES];\n"
        "static uint32_t ctrl_seen_mask = 0u;",
        "static int32_t ctrl_coeff_q24[PACK][N_AXES]; /* learned B_j,h in Q24 */\n"
        "static int32_t ctrl_z_q24[N_AXES];\n"
        "static int32_t pref_q24[N_AXES];\n"
        "static int32_t capture_q24 = 0;\n"
        "static int32_t gate_q24 = (1 << 24);\n"
        "/* Transport-only one-step look-ahead buffer. A faster group-0 core may\n"
        "   deliver gate(step+1) while a hidden core is still finishing step.\n"
        "   Buffer by the packet step tag; never apply it to the wrong step. */\n"
        "static int32_t next_gate_q24 = (1 << 24);\n"
        "static uint32_t next_gate_step = 0u;\n"
        "static bool next_gate_valid = false;\n"
        "static int64_t late_r_acc_q24[N_AXES];\n"
        "static uint32_t late_r_count = 0u;\n"
        "static uint32_t ctrl_seen_mask = 0u;",
    ))

    # Output v5 late activity + capture.  Group 0 owns the global B@x axis state.
    patches.append((
        "    if (group_id == 0u) {\n"
        "        for (uint32_t a = 0; a < N_AXES; a++) outrec[OUT_AXIS_FINAL0 + a] = sat_i64_to_i32(axis_state_q24_wide[a]);\n"
        "    }\n"
        "    write_status(status);",
        "    if (group_id == 0u) {\n"
        "        for (uint32_t a = 0; a < N_AXES; a++) {\n"
        "            outrec[OUT_AXIS_FINAL0 + a] = sat_i64_to_i32(axis_state_q24_wide[a]);\n"
        "            outrec[OUT_V5_LATE_R0 + a] = (late_r_count > 0u)\n"
        "                ? sat_i64_to_i32(late_r_acc_q24[a] / (int64_t)late_r_count) : (int32_t)((1u << 24) / 3u);\n"
        "        }\n"
        "        outrec[OUT_V5_CAPTURE] = capture_q24;\n"
        "    }\n"
        "    write_status(status);",
    ))

    # Reset/load per-image v5 state.
    patches.append((
        "    ctrl_seen_mask = 0u;\n"
        "    for (uint32_t a = 0; a < N_AXES; a++) {\n"
        "        ctrl_z_q24[a] = 0;\n"
        "        axis_state_q24_wide[a] = 0;\n"
        "        axis_acc_next_q44[a] = 0;\n"
        "    }",
        "    ctrl_seen_mask = 0u;\n"
        "    capture_q24 = 0; gate_q24 = (1 << 24); late_r_count = 0u;\n"
        "    next_gate_q24 = (1 << 24); next_gate_step = 0u; next_gate_valid = false;\n"
        "    for (uint32_t a = 0; a < N_AXES; a++) {\n"
        "        ctrl_z_q24[a] = 0;\n"
        "        pref_q24[a] = dynrec[DYN_PREF0 + a];\n"
        "        axis_state_q24_wide[a] = 0;\n"
        "        axis_acc_next_q44[a] = 0;\n"
        "        late_r_acc_q24[a] = 0;\n"
        "    }",
    ))

    helper = f'''\n/* ---------------- frozen v5 capture/direct-entry helpers ---------------- */\n#define V5_ONE_Q24          ((int32_t)(1u << 24))\n#define V5_RHO_Q24          ((int32_t){rho_q24})\n#define V5_ONE_MINUS_RHO_Q24 ((int32_t){one_minus_rho_q24})\n#define V5_DIRECT_GAIN_Q24  ((int32_t){direct_gain_q24})\n#define V5_LATE_NUM         35u\n#define V5_LATE_DEN         100u\n#define V5_LATE_STRIDE      {int(V5_LATE_STRIDE)}u\n\nstatic inline void v5_axis_activity_q24(const int64_t z_in[N_AXES], int32_t r_out[N_AXES]) {{\n    uint64_t maxabs = 0u;\n    for (uint32_t a = 0; a < N_AXES; a++) {{\n        uint64_t aa = (z_in[a] < 0) ? (uint64_t)(-z_in[a]) : (uint64_t)z_in[a];\n        if (aa > maxabs) maxabs = aa;\n    }}\n    if (maxabs == 0u) {{\n        r_out[0] = r_out[1] = (int32_t)(V5_ONE_Q24 / 3);\n        r_out[2] = V5_ONE_Q24 - r_out[0] - r_out[1];\n        return;\n    }}\n    int64_t sq[N_AXES]; int64_t den = 0;\n    for (uint32_t a = 0; a < N_AXES; a++) {{\n        /* Common rescaling cancels in z^2/sum(z^2); Q15 avoids int64 overflow. */\n        int64_t zn = (z_in[a] * (int64_t)(1 << 15)) / (int64_t)maxabs;\n        sq[a] = zn * zn;\n        den += sq[a];\n    }}\n    if (den <= 0) {{\n        r_out[0] = r_out[1] = (int32_t)(V5_ONE_Q24 / 3);\n        r_out[2] = V5_ONE_Q24 - r_out[0] - r_out[1];\n        return;\n    }}\n    int64_t used = 0;\n    for (uint32_t a = 0; a < N_AXES - 1u; a++) {{\n        r_out[a] = (int32_t)((sq[a] << 24) / den);\n        used += r_out[a];\n    }}\n    r_out[N_AXES-1u] = (int32_t)((int64_t)V5_ONE_Q24 - used);\n}}\n\nstatic inline void v5_update_capture_owner(void) {{\n    if (group_id != 0u) return;\n    int32_t r[N_AXES];\n    v5_axis_activity_q24(axis_state_q24_wide, r);\n    int64_t chi = 0;\n    for (uint32_t a = 0; a < N_AXES; a++)\n        chi += floor_rshift64((int64_t)pref_q24[a] * (int64_t)r[a], 24);\n    int64_t c = floor_rshift64((int64_t)V5_RHO_Q24 * capture_q24, 24) +\n                floor_rshift64((int64_t)V5_ONE_MINUS_RHO_Q24 * chi, 24);\n    if (c < 0) c = 0;\n    if (c > V5_ONE_Q24) c = V5_ONE_Q24;\n    capture_q24 = (int32_t)c;\n    gate_q24 = V5_ONE_Q24 - capture_q24;\n\n    /* axis_state here is the state AFTER micro-step (step_idx-1), because\n       consume_previous_events() ran immediately before this helper. Cloud v5\n       records late activity after each micro-step, so index with prev_step. */\n    if (step_idx > 0u) {{\n        uint32_t prev_step = step_idx - 1u;\n        if ((prev_step * V5_LATE_DEN >= gen_steps * (V5_LATE_DEN - V5_LATE_NUM)) &&\n            ((prev_step % V5_LATE_STRIDE) == 0u)) {{\n            for (uint32_t a = 0; a < N_AXES; a++) late_r_acc_q24[a] += r[a];\n            late_r_count++;\n        }}\n    }}\n}}\n\nstatic inline int32_t v5_direct_feedback_q24(uint32_t l) {{\n    if (logical_base < NV) return 0;\n    int64_t entry = 0;\n    for (uint32_t a = 0; a < N_AXES; a++)\n        entry += floor_rshift64((int64_t)ctrl_coeff_q24[l][a] * (int64_t)pref_q24[a], 24);\n    int64_t g = floor_rshift64(entry * (int64_t)gate_q24, 24);\n    g = floor_rshift64(g * (int64_t)V5_DIRECT_GAIN_Q24, 24);\n    return sat_i64_to_i32(g);\n}}\n'''
    patches.append((
        "static inline int32_t feedback_q24_for_lane(uint32_t l) {\n"
        "    if (logical_base < NV) return 0;\n"
        "    int64_t fb = 0;\n"
        "    for (uint32_t a = 0; a < N_AXES; a++) {\n"
        "        fb += floor_rshift64((int64_t)ctrl_coeff_q24[l][a] *\n"
        "                             (int64_t)ctrl_z_q24[a], Q_STATE);\n"
        "    }\n"
        "    return sat_i64_to_i32(fb);\n"
        "}",
        helper + "\nstatic inline int32_t feedback_q24_for_lane(uint32_t l) {\n    return v5_direct_feedback_q24(l);\n}",
    ))

    # Forward declaration: begin_scientific_step() calls this helper before the
    # helper body is inserted later at feedback_q24_for_lane().  The SpiNNaker
    # ARM C compiler does not permit an implicit non-static declaration followed
    # by a static inline definition.  This is declaration-only; no math changes.
    patches.append((
        "static void begin_scientific_step(uint32_t new_step) {",
        "static inline void v5_update_capture_owner(void);\n\n"
        "static void begin_scientific_step(uint32_t new_step) {",
    ))

    # Every scientific ms becomes a v5 capture boundary. Group 0 computes the
    # scalar gate from its resident B@x axis states and broadcasts one packet.
    patches.append((
        "    control_due = (step_idx > 0u && control_steps > 0u &&\n"
        "                   (step_idx % control_steps) == 0u);\n"
        "    axis_broadcast_pending = (group_id == 0u && control_due);\n"
        "    if (logical_base >= NV && control_due) ctrl_seen_mask = 0u;",
        "    control_due = true;\n"
        "    if (group_id == 0u) v5_update_capture_owner();\n"
        "    axis_broadcast_pending = (group_id == 0u);\n"
        "    if (logical_base >= NV) {\n"
        "        ctrl_seen_mask = 0u;\n"
        "        uint32_t cur = step_idx & STEP_MASK;\n"
        "        if (next_gate_valid && next_gate_step == cur) {\n"
        "            gate_q24 = next_gate_q24;\n"
        "            ctrl_seen_mask = 1u;\n"
        "            next_gate_valid = false;\n"
        "        } else if (next_gate_valid) {\n"
        "            /* A buffered gate may only be consumed at its exact tagged step. */\n"
        "            a35_wrong_total++;\n"
        "            next_gate_valid = false;\n"
        "        }\n"
        "    }",
    ))

    # Hidden targets now expect one scalar gate packet, not three axis payloads.
    patches.append((
        "    if (raw_group >= AXIS_GROUP_BASE && raw_group < AXIS_GROUP_BASE + N_AXES) {\n"
        "        if (phase != PHASE_RUNNING || logical_base < NV) return;\n"
        "        uint32_t ps = packet_step(key);\n"
        "        uint32_t cur = step_idx & STEP_MASK;\n"
        "        if (ps != cur) {\n"
        "            a35_wrong_total++;\n"
        "            return;\n"
        "        }\n"
        "        uint32_t a = raw_group - AXIS_GROUP_BASE;\n"
        "        uint32_t bit = 1u << a;\n"
        "        if (ctrl_seen_mask & bit) {\n"
        "            a35_wrong_total++;\n"
        "            return;\n"
        "        }\n"
        "        ctrl_z_q24[a] = (int32_t)payload;\n"
        "        ctrl_seen_mask |= bit;\n"
        "        a35_rx_total++;\n"
        "        return;\n"
        "    }",
        "    if (raw_group == AXIS_GROUP_BASE) {\n"
        "        if (phase != PHASE_RUNNING || logical_base < NV) return;\n"
        "        uint32_t ps = packet_step(key);\n"
        "        uint32_t cur = step_idx & STEP_MASK;\n"
        "        uint32_t nxt = (cur + 1u) & STEP_MASK;\n"
        "        if (ps == cur) {\n"
        "            if (ctrl_seen_mask & 1u) {\n"
        "                /* Same-step duplicates are transport noise; the gate is idempotent. */\n"
        "                return;\n"
        "            }\n"
        "            gate_q24 = (int32_t)payload;\n"
        "            ctrl_seen_mask = 1u;\n"
        "            a35_rx_total++;\n"
        "            return;\n"
        "        }\n"
        "        if (ps == nxt) {\n"
        "            /* Exact transport fix: keep an early next-step gate, but do not\n"
        "               expose it to the current micro-step. */\n"
        "            if (!next_gate_valid) {\n"
        "                next_gate_q24 = (int32_t)payload;\n"
        "                next_gate_step = ps;\n"
        "                next_gate_valid = true;\n"
        "                a35_rx_total++;\n"
        "            }\n"
        "            return;\n"
        "        }\n"
        "        a35_wrong_total++;\n"
        "        return;\n"
        "    }",
    ))

    patches.append((
        "static inline void broadcast_axis_if_due(uint32_t slot) {\n"
        "    if (!axis_broadcast_pending || slot < AXIS_SEND_SLOT || group_id != 0u) return;\n"
        "    for (uint32_t a = 0; a < N_AXES; a++) {\n"
        "        uint32_t low = ((step_idx & STEP_MASK) << 11) | (AXIS_GROUP_BASE + a);\n"
        "        uint32_t key = key_vis | low;  /* hidden targets only */\n"
        "        /* The frozen dense golden computes z with an int64 accumulator and\n"
        "           clamps only the transient Q24 readout.  Match that semantics here:\n"
        "           the persistent wide state is never clipped, only this 40-ms payload. */\n"
        "        int32_t z_payload_q24 = sat_i64_to_i32(axis_state_q24_wide[a]);\n"
        "        if (!send_packet_retry(key, (uint32_t)z_payload_q24)) {\n"
        "            fail_and_wait(ERR_SEND);\n"
        "            return;\n"
        "        }\n"
        "        a35_tx_total++;\n"
        "    }\n"
        "    axis_broadcast_pending = false;\n"
        "}",
        "static inline void broadcast_axis_if_due(uint32_t slot) {\n"
        "    if (!axis_broadcast_pending || slot < AXIS_SEND_SLOT || group_id != 0u) return;\n"
        "    uint32_t low = ((step_idx & STEP_MASK) << 11) | AXIS_GROUP_BASE;\n"
        "    uint32_t key = key_vis | low;  /* hidden targets only */\n"
        "    if (!send_packet_retry(key, (uint32_t)gate_q24)) {\n"
        "        fail_and_wait(ERR_SEND); return;\n"
        "    }\n"
        "    a35_tx_total++;\n"
        "    axis_broadcast_pending = false;\n"
        "}",
    ))

    # One gate packet must be present before each hidden micro-step.
    patches.append((
        "    if (logical_base >= NV && control_due && ctrl_seen_mask != ((1u << N_AXES) - 1u)) {\n"
        "        return;\n"
        "    }",
        "    if (logical_base >= NV && ctrl_seen_mask != 1u) {\n"
        "        return;\n"
        "    }",
    ))
    patches.append((
        "    if (logical_base >= NV && control_due) a35_updates_total++;",
        "    if (logical_base >= NV) a35_updates_total++;",
    ))
    patches.append((
        "        if (logical_base >= NV) total_field += feedback_q24_for_lane(l);",
        "        if (logical_base >= NV) total_field += feedback_q24_for_lane(l);",
    ))

    # The end-of-step strict 3-axis receive check is replaced by a 1-gate check.
    patches.append((
        "        if (logical_base >= NV && control_due && ctrl_seen_mask != ((1u << N_AXES) - 1u)) {",
        "        if (logical_base >= NV && ctrl_seen_mask != 1u) {",
    ))

    for old, new in patches:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"v5 kernel patch anchor occurs {n} times, expected 1:\n{old[:180]}")
        src = src.replace(old, new, 1)

    base.KERNEL_C = src
    base.CONTROL_DT_MS = V5_CONTROL_DT_MS
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Board wrapper
# ---------------------------------------------------------------------------

def make_v5_board_class(base):
    class V5Board(base.G3Board):
        @staticmethod
        def _parse_out(data: bytes) -> dict[str, Any]:
            r = base.G3Board._parse_out(data)
            a = np.frombuffer(data, dtype="<i4", count=base.OUT_WORDS).copy()
            r["v5_late_r_q24"] = a[OUT_V5_LATE_R0:OUT_V5_LATE_R0 + 3].astype(np.int64)
            r["v5_capture_q24"] = int(a[OUT_V5_CAPTURE])
            return r

        def __init__(self, cp, actuators, output, **kwargs):
            # The validated base build_aplx() writes generated kernel sources
            # directly into its output directory and assumes that directory
            # already exists.  Ensure it here before delegating to G3Board.
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            self._v5_B = np.asarray(actuators, dtype=np.float64).copy()
            if self._v5_B.shape != (3, base.NH):
                raise ValueError(f"bad v5 B shape {self._v5_B.shape}")
            # RMS-normalized rows are part of the frozen v5 scientific checkpoint.
            rms = np.sqrt(np.mean(self._v5_B * self._v5_B, axis=1))
            if np.max(np.abs(rms - 1.0)) > 5e-4:
                raise RuntimeError(f"v5 B rows are not RMS-normalized: {rms}")
            super().__init__(cp, self._v5_B, output, **kwargs)

        def _make_dynamic_blobs(self, seed: int, command: np.ndarray, gamma: float):
            # Let the validated base create membranes/RNGs and zero old A3.5 coeffs.
            pref = np.asarray(command, dtype=np.float64)
            pref = np.maximum(pref, 0.0)
            pref /= float(np.sum(pref)) + 1e-30
            blobs = super()._make_dynamic_blobs(seed, np.zeros(3, dtype=float), 0.0)
            Bq = base.qround_i32(self._v5_B, base.Q_STATE, name="v5 learned B")
            pq = base.qround_i32(pref, base.Q_STATE, name="v5 preference")
            Bq = np.asarray(Bq, dtype=np.int32)
            pq = np.asarray(pq, dtype=np.int32)
            for gid, x, y, p, logical_base, _ in self.placements:
                core_off = int(p) * self.dyn_stride
                # preference is global and is loaded on every core because group0
                # needs it for capture and hidden cores need it for direct entry.
                for aa in range(3):
                    off = core_off + (DYN_PREF0 + aa) * 4
                    blobs[(x, y)][off:off+4] = struct.pack("<i", int(pq[aa]))
                if logical_base < base.NV:
                    continue
                local_ids = np.arange(logical_base, logical_base + base.PACK, dtype=int)
                for lane, logical_id in enumerate(local_ids):
                    h = int(logical_id - base.NV)
                    for aa in range(3):
                        off = core_off + (base.DYN_CTRL0 + lane*3 + aa) * 4
                        blobs[(x, y)][off:off+4] = struct.pack("<i", int(Bq[aa, h]))
            return blobs

        def run_v5_image(self, seed: int, preference: np.ndarray):
            xv, xh, events, timing, transport = super().run_image(
                int(seed), np.asarray(preference, dtype=float), gamma=0.0)
            # group0 owns the global B@x states and late-activity accumulator.
            g0 = self._read_group(0)
            late = np.asarray(g0["v5_late_r_q24"], dtype=np.float64) / float(1 << 24)
            late = np.maximum(late, 0.0)
            late /= float(np.sum(late)) + 1e-30
            transport = dict(transport)
            transport["v5_late_activity"] = late.tolist()
            transport["v5_capture_final"] = float(g0["v5_capture_q24"] / float(1 << 24))
            return xv, xh, late, events, timing, transport

    return V5Board


def transport_ok(t: dict, base) -> bool:
    return bool(
        t.get("send_fail_total", 1) == 0
        and t.get("duplicate_total", 1) == 0
        and t.get("wrong_route_total", 1) == 0
        and t.get("late_event_total", 1) == 0
        and t.get("queue_overflow_total", 1) == 0
        and t.get("a35_missing_total", 1) == 0
        and t.get("a35_wrong_total", 1) == 0
        and t.get("workers_complete", 0) == base.N_GROUPS
    )


def run_board_generation_v5(base, v5, cp: dict, outdir: Path, n: int,
                            seed_start: int, build_tree: Path | None,
                            timeout_s: float, timer_tick_us: int = 100) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(cp["depth"], dtype=np.float64)
    B = np.asarray(cp["actuators"], dtype=np.float64)
    symbols = np.asarray(cp["symbols"], dtype=np.float64)
    adapt = v5.SlowAdaptation()
    hidden_wta = v5.make_activity_wta()
    pref_wta = v5.make_preference_wta()

    Board = make_v5_board_class(base)
    board_dir = outdir / "board"
    board_dir.mkdir(parents=True, exist_ok=True)
    board = None
    trace = []
    samples = []
    try:
        board = Board(
            cp, B, board_dir,
            generation_ms=int(round(v5.fp.GENERATION_S * 1000.0)),
            timer_tick_us=int(timer_tick_us),
            build_tree=build_tree,
            timeout_s=float(timeout_s),
        )
        for j in range(int(n)):
            score = v5.slow_score(depth, adapt.readout())
            preference = v5.learn_preference(score, pref_wta)
            seed = int(seed_start + j)
            xv, xh, late, events, timing, transport = board.run_v5_image(seed, preference)

            # Frozen v5 slow feedback, evaluated between images.
            hidden_comp = hidden_wta.transform(late)
            adapt.update(hidden_comp)

            # post-hoc only
            mode, signed, corr = base.classify_visible(xv, symbols)
            q = float(abs(signed))
            row = {
                "j": j,
                "seed": seed,
                "mode_posthoc": int(mode),
                "Q_posthoc": q,
                "slow_score": np.asarray(score, float).tolist(),
                "preference_wta": np.asarray(preference, float).tolist(),
                "board_late_activity": np.asarray(late, float).tolist(),
                "hidden_wta": np.asarray(hidden_comp, float).tolist(),
                "timing": timing,
                "transport": transport,
                "transport_pass": transport_ok(transport, base),
            }
            trace.append(row)
            samples.append(np.asarray(xv, dtype=np.float32))
            json_dump(outdir / "generation_trace.json", trace)
            cnt = np.bincount([x["mode_posthoc"] for x in trace], minlength=3)
            print(
                f"v5 board {j+1:03d}/{n} seed={seed} mode={mode} Q={q:.4f} "
                f"counts={cnt.tolist()} cap={transport['v5_capture_final']:.3f} "
                f"board={timing.get('board_generation_s', float('nan')):.3f}s",
                flush=True,
            )
    finally:
        if board is not None:
            board.close()

    modes = np.asarray([r["mode_posthoc"] for r in trace], dtype=int)
    qs = np.asarray([r["Q_posthoc"] for r in trace], dtype=float)
    counts = np.bincount(modes, minlength=3) if len(modes) else np.zeros(3, dtype=int)
    arr = np.asarray(samples, dtype=np.float32)
    np.savez_compressed(outdir / "generated_samples.npz", visible=arr)
    try:
        base.save_montage(arr, outdir / "montage.png")
    except Exception:
        pass
    report = {
        "formula_version": str(v5.FORMULA_VERSION),
        "n": int(len(trace)),
        "counts_posthoc": counts.astype(int).tolist(),
        "P_posthoc": (counts / max(1, len(trace))).tolist(),
        "Q_mean_posthoc": float(qs.mean()) if len(qs) else None,
        "Q_min_posthoc": float(qs.min()) if len(qs) else None,
        "all_transport_pass": bool(all(r["transport_pass"] for r in trace)) if trace else False,
        "generation_target_prior_input": False,
        "explicit_count_controller": False,
        "classifier_in_feedback_loop": False,
        "hidden_reads_during_image": 0,
        "one_resident_board_run_per_image": True,
        "slow_izh_location": "host_between_images_functional_HIL",
        "fast_lif_and_capture_location": "SpiNNaker_resident",
    }
    json_dump(outdir / "V5_BOARD_GENERATION_REPORT.json", report)
    return report


# ---------------------------------------------------------------------------
# Raw-MNIST -> fresh path helpers (no external trajectory payload required)
# ---------------------------------------------------------------------------

def require_raw_mnist_base(base) -> None:
    needed = [
        "locate_mnist_npz", "load_standard_mnist_npz",
        "build_training_payload_from_mnist_npz", "MNIST012_SYMBOLS",
    ]
    missing = [name for name in needed if not hasattr(base, name)]
    if missing:
        raise RuntimeError(
            "The selected --base-script does not expose the validated raw-MNIST fresh-path "
            f"API; missing {missing}. Use mnist_npz_zero_to_generation_spinnaker.py."
        )


def resolve_mnist(args, base) -> Path:
    require_raw_mnist_base(base)
    return Path(base.locate_mnist_npz(args.mnist)).resolve()


def build_fresh_payload_for_prior(args, base, prior: np.ndarray, cycles: int,
                                  outdir: Path, seed_start: int) -> tuple[Path, dict, np.ndarray]:
    """Generate all LIF forward/noising trajectories fresh from raw MNIST this run."""
    mnist = resolve_mnist(args, base)
    payload, report, _ = base.build_training_payload_from_mnist_npz(
        mnist_path=mnist,
        prior=np.asarray(prior, dtype=np.float64),
        cycles=int(cycles),
        outdir=outdir,
        seed_start=int(seed_start),
        data_seed=int(args.data_seed),
        path_workers=int(args.path_workers),
        pixel_preprocess=str(args.pixel_preprocess),
    )
    symbols = np.asarray(base.MNIST012_SYMBOLS, dtype=np.float32).copy()
    if symbols.shape != (3, base.NV):
        raise RuntimeError(f"fresh-path builder produced invalid symbols {symbols.shape}")
    if not report.get("fresh_paths", False) or report.get("external_path_cache_used", True):
        raise RuntimeError("fresh-path provenance contract failed")
    return Path(payload), report, symbols


def build_fresh_common_bank(args, base, root: Path) -> tuple[Path, dict, np.ndarray]:
    """One fresh 33/33/33 bank per invocation, reused only as immutable input across priors."""
    bank_cycles = 99
    bank_prior = np.ones(3, dtype=np.float64) / 3.0
    bank_dir = root / "00_fresh_common_bank"
    payload, report, symbols = build_fresh_payload_for_prior(
        args, base, bank_prior, bank_cycles, bank_dir, int(args.path_seed_start)
    )
    counts = np.asarray(report.get("schedule_counts", []), dtype=int)
    if counts.shape != (3,) or not np.array_equal(counts, np.array([33, 33, 33])):
        raise RuntimeError(f"common fresh bank must be 33/33/33, got {counts.tolist()}")
    return payload, report, symbols

# ---------------------------------------------------------------------------
# True from-zero smoke and full-four orchestration
# ---------------------------------------------------------------------------

def smoke(args, base, v5) -> dict:
    """True end-to-end smoke from raw MNIST to fresh paths to exact-zero learning.

    No pretrained checkpoint and no external trajectory payload are accepted or
    required. All learned states used by generation (W, H, depth, self-B) are
    learned in this smoke run.
    """
    if int(args.cycles) != int(v5.TRAIN_CYCLES_DEFAULT):
        raise ValueError(
            f"frozen v5 smoke requires exactly {v5.TRAIN_CYCLES_DEFAULT} training cycles; "
            f"got {args.cycles}. Refusing to change the frozen scientific condition."
        )
    prior = parse_prior_text(args.smoke_prior)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    # 1) Generate the training trajectories FRESH from raw MNIST in this invocation.
    #    No external path payload or learned checkpoint is accepted.
    print("\n=== SMOKE STAGE 0: RAW MNIST -> FRESH LIF FORWARD/NOISING PATHS ===", flush=True)
    payload, fresh_report, symbols = build_fresh_payload_for_prior(
        args, base, prior, args.cycles, out, int(args.path_seed_start)
    )
    prep = {
        "raw_mnist": str(resolve_mnist(args, base)),
        "fresh_paths_generated_this_run": True,
        "external_path_cache_used": False,
        "schedule_counts": fresh_report["schedule_counts"],
        "payload": str(payload),
        "payload_sha256": fresh_report["payload_sha256"],
    }
    json_dump(out / "FRESH_PATH_REPORT.json", fresh_report)

    # 2) The board is the only W/H learner; the Phase-I kernel initializes every
    #    logical weight to exact zero internally before any exposure.
    print("\n=== SMOKE: BOARD W/H TRAINING FROM EXACT ZERO ===", flush=True)
    train_result = run_locked_phase1_v4_training(
        payload, out / "01_board_training", args
    )
    if not train_result["report"].get("train_pass", False):
        raise RuntimeError("smoke: board from-zero training did not pass")
    if not train_result["report"].get("weights_start_from_zero", False):
        raise RuntimeError("smoke: training report does not certify exact-zero start")

    # 3) Learn frozen-v5 slow depth and self-B from the SAME experienced cached
    #    trajectories.  This is not a checkpoint import and does not alter W/H.
    print("=== SMOKE: FROZEN-v5 SLOW DEPTH/self-B LEARNING FROM SAME PATHS ===", flush=True)
    slow = learn_slow_states_only(payload, symbols, v5)
    cp = make_combined_cp(train_result, slow)
    combined_cp = out / "training_checkpoint_v5_combined.npz"
    save_combined_v5_checkpoint(combined_cp, cp, prior, train_result["exposure"], v5)

    # 4) Only after all learning is complete do we patch/build the frozen v5
    #    generator and run prior-free generation.
    kernel_sha = patch_generator_kernel_v5(base, v5)
    static = {
        "mode": "smoke-from-zero",
        "formula_version": str(v5.FORMULA_VERSION),
        "training_prior_for_exposure_schedule_only": prior.tolist(),
        "cycles": int(args.cycles),
        "training_exposure": np.asarray(train_result["exposure"], int).tolist(),
        "weights_start_from_zero": True,
        "host_learning_updates": int(train_result["report"].get("host_learning_updates", -1)),
        "board_training_pass": True,
        "board_training_wall_s": train_result["report"].get("board_training_wall_s"),
        "combined_checkpoint": str(combined_cp),
        "combined_checkpoint_sha256": sha256_file(combined_cp),
        "patched_kernel_sha256": kernel_sha,
        "v5_contract": validate_v5_contract(v5),
        "B_rms": np.sqrt(np.mean(cp["actuators"] ** 2, axis=1)).tolist(),
        "depth": cp["depth"].tolist(),
        "n_images": int(args.n),
        "seed_start": int(args.seed_start),
        "pretrained_checkpoint_loaded": False,
        "external_path_cache_used": False,
        "fresh_paths_generated_this_run": True,
        "locked_phase1_v4": dict(_LOCKED_PHASE1_MANIFEST),
    }
    json_dump(out / "SMOKE_STATIC_AUDIT.json", static)

    print("=== SMOKE: PRIOR-FREE V5 BOARD GENERATION ===", flush=True)
    rep = run_board_generation_v5(
        base, v5, cp, out / "02_generation",
        n=args.n, seed_start=args.seed_start,
        build_tree=args.build_tree, timeout_s=args.timeout_s,
        timer_tick_us=args.timer_tick_us,
    )
    rep["from_zero_training_pass"] = True
    rep["training_prior_used_only_for_exposure_schedule"] = prior.tolist()
    rep["training_exposure"] = np.asarray(train_result["exposure"], int).tolist()
    rep["pretrained_checkpoint_loaded"] = False
    rep["external_path_cache_used"] = False
    rep["fresh_paths_generated_this_run"] = True
    # Smoke gate is deliberately technical, not a distribution claim.
    rep["smoke_pass"] = bool(
        rep["from_zero_training_pass"]
        and rep["all_transport_pass"]
        and rep["Q_mean_posthoc"] is not None
        and rep["Q_mean_posthoc"] >= args.smoke_q_gate
    )
    json_dump(out / "SMOKE_SUMMARY.json", rep)
    return rep


def make_combined_cp(train_result: dict, slow: dict) -> dict:
    W = np.asarray(train_result["Wq"], dtype=np.float64) / float(1 << 24)
    H = np.asarray(train_result["Hq"], dtype=np.float64) / float(1 << 24)
    return {
        "Wvh": W,
        "Whh": H,
        "bv": np.zeros(W.shape[0], dtype=np.float64),
        "bh": np.zeros(H.shape[0], dtype=np.float64),
        "symbols": np.asarray(slow["symbols"], dtype=np.float64),
        "depth": np.asarray(slow["depth"], dtype=np.float64),
        "actuators": np.asarray(slow["actuators"], dtype=np.float64),
    }


def save_combined_v5_checkpoint(path: Path, cp: dict, prior: np.ndarray, exposure: np.ndarray, v5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        W=np.asarray(cp["Wvh"]), H=np.asarray(cp["Whh"]),
        depth=np.asarray(cp["depth"]), actuators=np.asarray(cp["actuators"]),
        symbols=np.asarray(cp["symbols"], dtype=np.float32),
        training_prior_metadata=np.asarray(prior, dtype=np.float64),
        exposure=np.asarray(exposure, dtype=np.int32),
        formula_version=np.asarray([v5.FORMULA_VERSION]),
        hardware_training=np.asarray(["SpiNNaker Phase-I V4 W/H; exact v5 slow-state HIL"]),
    )


def full_four(args, base, v5) -> dict:
    if int(args.cycles) != int(v5.TRAIN_CYCLES_DEFAULT):
        raise ValueError(
            f"frozen v5 requires exactly {v5.TRAIN_CYCLES_DEFAULT} training cycles; "
            f"got {args.cycles}. Refusing to change the frozen scientific condition."
        )
    # Patch generator once in this process. Training kernel is a separate frozen string.
    kernel_sha = patch_generator_kernel_v5(base, v5)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    print("\n=== FULL-FOUR STAGE 0: RAW MNIST -> ONE FRESH 33/33/33 LIF PATH BANK ===", flush=True)
    common_payload, common_report, symbols = build_fresh_common_bank(args, base, root)
    json_dump(root / "FRESH_COMMON_BANK_REPORT.json", common_report)

    master = {
        "experiment": "raw-MNIST fresh-path frozen-v5 four-prior exact-zero SpiNNaker/HIL test",
        "formula_version": str(v5.FORMULA_VERSION),
        "generator_kernel_sha256": kernel_sha,
        "cycles": int(args.cycles),
        "n_generate_per_prior": int(args.n),
        "priors": {},
        "raw_mnist": str(resolve_mnist(args, base)),
        "fresh_common_bank": common_report,
        "external_path_cache_used": False,
        "locked_phase1_v4": dict(_LOCKED_PHASE1_MANIFEST),
        "scientific_boundaries": [
            "W/H exact-zero MSDL learning runs on SpiNNaker Phase-I V4.",
            "Slow depth/self-B states use exact frozen-v5 dynamics on host from fresh trajectories generated from raw MNIST in this invocation.",
            "Fast 9072-LIF generation and 1-ms capture/direct-entry run resident on SpiNNaker.",
            "3-channel preference-WTA/adaptation is evaluated on host between images; no hidden read occurs during an image.",
            "Visible classification is post-hoc only.",
        ],
    }

    for n_prior, (tag, prior) in enumerate(PRIORS.items()):
        pdir = root / tag
        pdir.mkdir(parents=True, exist_ok=True)
        payload = pdir / "00_training_payload" / "phase1_training_payload.npz"
        prep = make_prior_payload(common_payload, prior, args.cycles, payload, v5, base)
        prep["fresh_common_bank_generated_this_run"] = True
        prep["external_path_cache_used"] = False
        prep["raw_mnist"] = str(resolve_mnist(args, base))
        json_dump(pdir / "PAYLOAD_SELECTION_REPORT.json", prep)

        print(f"\n=== {tag}: BOARD W/H TRAINING FROM EXACT ZERO ===", flush=True)
        train_result = run_locked_phase1_v4_training(
            payload, pdir / "01_board_training", args
        )
        if not train_result["report"].get("train_pass", False):
            raise RuntimeError(f"{tag}: board training did not pass")

        print(f"=== {tag}: FROZEN-v5 SLOW DEPTH/B LEARNING FROM SAME PATHS ===", flush=True)
        slow = learn_slow_states_only(payload, symbols, v5)
        cp = make_combined_cp(train_result, slow)
        combined_cp = pdir / "training_checkpoint_v5_combined.npz"
        save_combined_v5_checkpoint(combined_cp, cp, prior, train_result["exposure"], v5)

        # Important: board W/H and slow v5 state are now frozen before generation.
        print(f"=== {tag}: PRIOR-FREE V5 BOARD GENERATION ===", flush=True)
        gen = run_board_generation_v5(
            base, v5, cp, pdir / "02_generation",
            n=args.n,
            seed_start=args.seed_start + n_prior * args.seed_stride,
            build_tree=args.build_tree,
            timeout_s=args.timeout_s,
            timer_tick_us=args.timer_tick_us,
        )
        counts = np.asarray(gen["counts_posthoc"], dtype=int)
        pgen = counts / max(1, counts.sum())
        tv = float(0.5 * np.abs(pgen - prior).sum())
        master["priors"][tag] = {
            "target_prior_for_training_only": prior.tolist(),
            "payload": prep,
            "training_exposure": np.asarray(train_result["exposure"], int).tolist(),
            "board_training_pass": True,
            "board_training_wall_s": train_result["report"].get("board_training_wall_s"),
            "learned_depth": np.asarray(slow["depth"], float).tolist(),
            "B_rms": np.sqrt(np.mean(np.asarray(slow["actuators"])**2, axis=1)).tolist(),
            "generation": gen,
            "TV_posthoc_to_training_prior": tv,
            "combined_checkpoint": str(combined_cp),
        }
        json_dump(root / "FULL_FOUR_PROGRESS.json", master)

    master["all_training_pass"] = bool(all(x["board_training_pass"] for x in master["priors"].values()))
    master["all_generation_transport_pass"] = bool(all(x["generation"]["all_transport_pass"] for x in master["priors"].values()))
    json_dump(root / "FULL_FOUR_SUMMARY.json", master)
    return master


def dry_run(args, base, v5) -> dict:
    # No hardware execution and no fresh-path generation. Validate raw MNIST,
    # schedules, dimensions, kernel anchors, and class-pool sufficiency.
    kernel_sha = patch_generator_kernel_v5(base, v5)
    mnist = resolve_mnist(args, base)
    xtr, ytr, _xte, _yte, mnist_report = base.load_standard_mnist_npz(mnist)
    available = np.bincount(np.asarray(ytr, dtype=int), minlength=3)[:3]
    feasibility = {}
    for tag, p in PRIORS.items():
        sched = v5.fp.low_discrepancy_schedule(np.asarray(p, dtype=float), int(args.cycles))
        need = np.bincount(np.asarray(sched, dtype=int), minlength=3)[:3]
        feasibility[tag] = {
            "schedule_counts": need.astype(int).tolist(),
            "available_raw_mnist_train": available.astype(int).tolist(),
            "no_reuse_required": bool(np.all(need <= available)),
        }
    report = {
        "mode": "dry-run",
        "v5_contract": validate_v5_contract(v5),
        "patched_kernel_sha256": kernel_sha,
        "base_dimensions": {"N": int(base.N), "NV": int(base.NV), "NH": int(base.NH), "PACK": int(base.PACK)},
        "raw_mnist": mnist_report,
        "external_path_payload_required": False,
        "fresh_paths_generated_in_smoke_or_full": True,
        "locked_phase1_v4": dict(_LOCKED_PHASE1_MANIFEST),
        "hardware_capture_dt_ms": int(V5_CONTROL_DT_MS),
        "cloud_capture_dt_ms": float(v5.fp.DT * 1000.0),
        "capture_cadence_exact_match": bool(abs(V5_CONTROL_DT_MS - v5.fp.DT*1000.0) < 1e-12),
        "late_stride_match": int(V5_LATE_STRIDE),
        "prior_feasibility": feasibility,
    }
    json_dump(args.output / "DRY_RUN_AUDIT.json", report)
    return report


def main() -> None:
    """Run the complete experiment from raw MNIST to generation.

    The scientific implementation is frozen. This outer CLI only selects the
    requested prior, seeds, output directory, and execution mode.
    """
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description=(
            "Single-file from-zero MSDL/SpiNNaker experiment: raw MNIST -> "
            "fresh LIF paths -> exact-zero Phase-I V4 learning -> frozen-v5 "
            "slow dynamics -> prior-free generation."
        )
    )
    ap.add_argument("--mode", choices=["smoke", "full-four", "dry-run"], default="smoke")
    ap.add_argument(
        "--mnist", type=Path, default=here / "mnist.npz",
        help="standard Keras-format mnist.npz (default: beside this script)",
    )
    ap.add_argument(
        "--smoke-prior",
        default=",".join(str(float(x)) for x in TARGET_PRIOR),
        help="training exposure prior for smoke mode, e.g. 0.2,0.6,0.2",
    )
    ap.add_argument("--cycles", type=int, default=int(TRAIN_CYCLES))
    ap.add_argument("--n", type=int, default=int(N_GENERATE), help="number of generated images")
    ap.add_argument("--path-workers", type=int, default=int(PATH_WORKERS))
    ap.add_argument("--path-seed-start", type=int, default=int(PATH_SEED_START))
    ap.add_argument("--data-seed", type=int, default=int(DATA_SEED))
    ap.add_argument("--pixel-preprocess", default=str(PIXEL_PREPROCESS))
    ap.add_argument("--seed-start", type=int, default=int(GENERATION_SEED_START))
    ap.add_argument("--seed-stride", type=int, default=10000,
                    help="generation seed stride between priors in full-four mode")
    ap.add_argument("--bcast-slots", type=int, default=int(BCAST_SLOTS))
    ap.add_argument("--timer-tick-us", type=int, default=int(TIMER_TICK_US))
    ap.add_argument("--timeout-s", type=float, default=float(GENERATION_TIMEOUT_S))
    ap.add_argument("--training-timeout-s", type=float, default=float(TRAINING_TIMEOUT_S))
    ap.add_argument("--smoke-q-gate", type=float, default=float(QUALITY_GATE))
    ap.add_argument("--build-tree", type=Path, default=None)
    ap.add_argument(
        "--output", type=Path, default=here / OUTPUT_DIR_NAME,
        help="output directory",
    )
    cli = ap.parse_args()

    mnist = Path(cli.mnist).expanduser().resolve()
    if not mnist.is_file():
        raise FileNotFoundError(
            f"Expected standard Keras-format MNIST at {mnist}. "
            "Place mnist.npz beside this script or pass --mnist PATH."
        )

    output = Path(cli.output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    else:
        output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 1) Verify/load exact frozen implementation snapshots before board contact.
    locked_manifest = configure_locked_phase1_v4(output)
    json_dump(output / "LOCKED_IMPLEMENTATION_MANIFEST.json", locked_manifest)
    base = load_embedded_base(Path(__file__))
    v5 = load_frozen_v5_optional(None)

    if base.N != 1296 or base.NV != 784 or base.NH != 512 or base.PACK != 4:
        raise RuntimeError("unexpected embedded SpiNNaker base dimensions")
    if str(v5.FORMULA_VERSION) != "msdl-full-dynamics-v5-selfB-IzhWTA-directentry":
        raise RuntimeError(f"unexpected frozen v5 version {v5.FORMULA_VERSION}")

    # The frozen scientific condition remains exactly 50 training exposures.
    if int(cli.cycles) != int(v5.TRAIN_CYCLES_DEFAULT):
        raise RuntimeError(
            f"frozen v5 requires --cycles {int(v5.TRAIN_CYCLES_DEFAULT)} exactly; "
            f"got {cli.cycles}"
        )

    # 2) Convert parsed CLI values into the unchanged internal argument contract.
    args = SimpleNamespace(
        mnist=mnist,
        output=output,
        build_tree=cli.build_tree,
        cycles=int(cli.cycles),
        n=int(cli.n),
        path_seed_start=int(cli.path_seed_start),
        data_seed=int(cli.data_seed),
        path_workers=int(cli.path_workers),
        pixel_preprocess=str(cli.pixel_preprocess),
        smoke_prior=str(cli.smoke_prior),
        seed_start=int(cli.seed_start),
        seed_stride=int(cli.seed_stride),
        timer_tick_us=int(cli.timer_tick_us),
        timeout_s=float(cli.timeout_s),
        training_timeout_s=float(cli.training_timeout_s),
        bcast_slots=int(cli.bcast_slots),
        smoke_q_gate=float(cli.smoke_q_gate),
    )

    requested_prior = parse_prior_text(args.smoke_prior)
    requested_schedule = np.asarray(
        v5.fp.low_discrepancy_schedule(requested_prior, int(args.cycles)), dtype=int
    )
    requested_exposure = np.bincount(requested_schedule, minlength=3)[:3]

    print("=" * 78)
    print("MSDL / SpiNNaker -- single-file from-zero experiment")
    print(f"Mode:          {cli.mode}")
    print(f"MNIST:         {mnist}")
    if cli.mode == "smoke":
        print(f"Target prior:  {requested_prior.tolist()}")
        print(f"Exposure:      {requested_exposure.astype(int).tolist()}")
    print(f"Training:      {args.cycles} exposures, W=H=0 at start")
    print(f"Generation:    {args.n} prior-free images")
    print(f"Output:        {output}")
    print("=" * 78, flush=True)

    # 3) Execute the already validated internal path without modifying it.
    if cli.mode == "smoke":
        result = smoke(args, base, v5)
        result["canonical_single_file_run"] = True
        result["requested_training_prior"] = requested_prior.tolist()
        result["requested_training_exposure"] = requested_exposure.astype(int).tolist()
        result["implementation_manifest"] = locked_manifest
        json_dump(output / "FINAL_RESULT.json", result)
        print("\nRUN PASS" if result.get("smoke_pass", False) else "\nRUN FAIL")
        print(json.dumps(result, indent=2))
        return

    if cli.mode == "full-four":
        result = full_four(args, base, v5)
        result["canonical_single_file_run"] = True
        result["implementation_manifest"] = locked_manifest
        json_dump(output / "FINAL_RESULT.json", result)
        print(json.dumps(result, indent=2))
        return

    result = dry_run(args, base, v5)
    result["canonical_single_file_run"] = True
    result["implementation_manifest"] = locked_manifest
    json_dump(output / "FINAL_RESULT.json", result)
    print(json.dumps(result, indent=2))


# =============================================================================
# SINGLE-FILE FROZEN IMPLEMENTATION SNAPSHOTS
# =============================================================================
# These four source blocks are exact snapshots of the hardware implementation
# that produced the validated 100-image run. They are embedded only to remove
# external .py/.c dependencies. Their SHA256 hashes are verified at startup.
# Do not edit these blocks unless intentionally creating a new validated version.

_EMBEDDED_BASE_SOURCE = (
    '#!/usr/bin/env python3\n'
    '# -*- coding: utf-8 -*-\n'
    '"""\n'
    'Standard MNIST -> from-zero SpiNNaker MSDL training -> dynamical generation\n'
    '===========================================================================\n'
    '\n'
    'One-file experiment using a standard Keras-format `mnist.npz` placed next to\n'
    'this script.  The scientific pipeline is:\n'
    '\n'
    '  mnist.npz (x_train/y_train; digits 0/1/2 only)\n'
    '      -> deterministic selection of distinct real MNIST training examples\n'
    '      -> fixed physical pixel normalization (no learned preprocessing)\n'
    '      -> fresh 9072-hard-LIF forward/noising trajectories generated on this run\n'
    '      -> exact-zero W/H\n'
    '      -> unchanged Phase-I V4 on-board SpiNNaker local dynamical learning\n'
    '      -> learned W/H checkpoint\n'
    '      -> unchanged persistent 9072-LIF SpiNNaker generator\n'
    '      -> 100 generated images\n'
    '      -> quality + 0/1/2 distribution + transport/integrity report\n'
    '\n'
    'Data boundary\n'
    '-------------\n'
    '* Only `x_train/y_train` participate in learning.  `x_test/y_test` never enter\n'
    '  the W/H update and are reserved for held-out reporting/provenance.\n'
    '* Labels are used only to implement the requested exposure prior and to keep the\n'
    '  three historical mode-specific dynamical states.  The SpiNNaker weight update\n'
    '  itself remains the unchanged local MSDL update.\n'
    '* Training examples are distinct until a class pool is exhausted; any reuse is\n'
    '  explicitly reported.\n'
    '* No HADAPT_REFERENCE_PATHS.npz is used.  Every 28x1296 trajectory is generated\n'
    '  fresh from the selected raw MNIST image on this invocation.\n'
    '\n'
    'Physical pixel normalization\n'
    '----------------------------\n'
    'Keras MNIST pixels are first converted from uint8 0..255 to float32 0..1.  The\n'
    'historical GTC/MSDL pipeline used scaled grayscale inputs.  To preserve that\n'
    'validated dynamical operating range while switching to standard mnist.npz, the\n'
    'default `whitelam-rms` preprocessing applies one fixed, non-learned transform:\n'
    '\n'
    '    z = (x - 0.130925363) * 1.744274831 / RMS(x - 0.130925363)\n'
    '\n'
    'This reproduces the scale of the historical three-symbol mnist_data.dat to\n'
    '~1e-5 when applied to those underlying raw images.  Use --pixel-preprocess unit\n'
    'only if you deliberately want the new [0,1] input-scale experiment.\n'
    '\n'
    'Default run (place this script and mnist.npz in the same directory):\n'
    '\n'
    '    python mnist_npz_zero_to_generation_spinnaker.py\n'
    '\n'
    'or in Jupyter:\n'
    '\n'
    '    %run mnist_npz_zero_to_generation_spinnaker.py\n'
    '\n'
    'Default experiment:\n'
    '  target prior       = 1/3, 1/3, 1/3\n'
    '  training cycles    = 99  (33/33/33 distinct training examples)\n'
    '  generation samples = 100\n'
    '  generation time    = 4320 ms / sample\n'
    '\n'
    'The embedded Phase-I V4 APLX training kernel and persistent generation APLX\n'
    'kernel are intentionally unchanged from the validated hardware version.\n'
    '"""\n'
    'from __future__ import annotations\n'
    '\n'
    'import argparse\n'
    'import base64\n'
    'import hashlib\n'
    'import json\n'
    'import math\n'
    'import os\n'
    'import shutil\n'
    'import struct\n'
    'import subprocess\n'
    'import sys\n'
    'import time\n'
    'import traceback\n'
    'import zlib\n'
    'from collections import defaultdict, deque\n'
    'from pathlib import Path\n'
    'from typing import Any\n'
    '\n'
    'import numpy as np\n'
    '\n'
    'try:\n'
    '    from scipy.linalg import expm\n'
    'except Exception as exc:\n'
    '    raise RuntimeError("scipy is required") from exc\n'
    '\n'
    'VALIDATED_PATH_CACHE_SHA256 = "e4e2ff05308f7778a3b05f55cb550ad258102032a5fc387b48c44e781788a315"\n'
    'VALIDATED_TRAIN_KERNEL_SHA256 = "8eacc79d7cf32c31fa0bc8cb4407de1a521f276d5d0b9fbfc0ed28692effaf75"\n'
    'VALIDATED_CLEAN_SOURCE_SHA256 = "fcb6fa348f275ce3366e5e8c57e883941907708a1a2d36e9f889f772a8a354c3"\n'
    'EMBEDDED_SYMBOLS_FLOAT32_SHA256 = "e402b7fa3c858cb1c484e6c310be51da923aa77d6527165a4838f27c6ab09566"\n'
    'EMBEDDED_AXES_ARRAY_SHA256 = "13b2a16be7e340384e33ebdb6272c1bdd3883d1c7bb8e6fd1122b10868d4febb"\n'
    'VALIDATED_AXES_FILE_SHA256 = "e9aa14ca79d9595cdeb438f7746ff3de27aef816ac49ba8f158d6b89e6bd056c"\n'
    '\n'
    "_AXES_B64 = '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'\n"
    'MNIST012_SYMBOLS = np.zeros((3, 784), dtype=np.float64)  # replaced from raw MNIST file in main()\n'
    'EMBEDDED_BOARD_AXES = np.frombuffer(\n'
    '    zlib.decompress(base64.b64decode(_AXES_B64)), dtype=np.float64\n'
    ').copy().reshape(3, 512)\n'
    '\n'
    'if hashlib.sha256(EMBEDDED_BOARD_AXES.tobytes()).hexdigest() != EMBEDDED_AXES_ARRAY_SHA256:\n'
    '    raise RuntimeError("embedded basin-axis checksum mismatch")\n'
    '\n'
    '# =============================================================================\n'
    '# Frozen scientific dimensions / numeric representation\n'
    '# =============================================================================\n'
    'N = 1296\n'
    'NV = 784\n'
    'NH = 512\n'
    'N_BASIS = 7\n'
    'N_AXES = 3\n'
    'PACK = 4\n'
    'N_GROUPS = N // PACK\n'
    'N_VIS_GROUPS = NV // PACK\n'
    'N_HID_GROUPS = NH // PACK\n'
    'MAX_CORES = 18\n'
    'Q_STATE = 24\n'
    'Q_AXIS = 20\n'
    'CONTROL_DT_MS = 40\n'
    'GAMMA = 4.0\n'
    '\n'
    'assert N % PACK == 0\n'
    'assert NV % PACK == 0\n'
    'assert NH % PACK == 0\n'
    'assert N_GROUPS == 324\n'
    'assert N_VIS_GROUPS == 196\n'
    'assert N_HID_GROUPS == 128\n'
    '\n'
    'CFG_WORDS = 64\n'
    'CFG_BYTES = CFG_WORDS * 4\n'
    'DYN_WORDS = 160\n'
    'OUT_WORDS = 64\n'
    'DYN_BYTES = DYN_WORDS * 4\n'
    'OUT_BYTES = OUT_WORDS * 4\n'
    'SDRAM_TAG = 1\n'
    '\n'
    'MAGIC_CFG = 0x45314431  # E1D1\n'
    'MAGIC_OUT = 0x45314F31  # E1O1\n'
    'CFG_VERSION = 1\n'
    '\n'
    'STATUS_IDLE = 1\n'
    'STATUS_ARMED = 2\n'
    'STATUS_RUNNING = 3\n'
    'STATUS_IMAGE_DONE = 4\n'
    'STATUS_APP_DONE = 5\n'
    'STATUS_ERROR = 0xDEAD0001\n'
    '\n'
    '# Config words. Must match embedded C.\n'
    'CFG_ARM_EPOCH = 15\n'
    'CFG_GO_EPOCH = 16\n'
    'CFG_STOP_FLAG = 17\n'
    'CFG_GROUP_BY_P = 32\n'
    '\n'
    '# Dynamic words. Must match embedded C.\n'
    'DYN_INIT_V0 = 0                     # 4 * 7 = 28 signed Q24\n'
    'DYN_SEED0 = 32                      # 4 * 7 * 4 = 112 uint32\n'
    'DYN_CTRL0 = 144                     # 4 lanes * 3 A3.5 coefficients\n'
    '\n'
    '# Output words. Must match embedded C.\n'
    'OUT_MAGIC = 0\n'
    'OUT_GROUP = 1\n'
    'OUT_STATUS = 2\n'
    'OUT_ERROR = 3\n'
    'OUT_EPOCH = 4\n'
    'OUT_STEPS = 5\n'
    'OUT_SEND_FAIL = 6\n'
    'OUT_DUPLICATE = 7\n'
    'OUT_WRONG_ROUTE = 8\n'
    'OUT_LATE_EVENT = 9\n'
    'OUT_EVENT_GENERATED = 10\n'
    'OUT_EVENT_SENT = 11\n'
    'OUT_EVENT_RX = 12\n'
    'OUT_EVENT_PROCESSED = 13\n'
    'OUT_LOOPBACK_IGNORED = 14\n'
    'OUT_SYNOP = 15\n'
    'OUT_MAX_RX_INTERVAL = 16\n'
    'OUT_TIMER_SLIP_MAX = 17\n'
    'OUT_QUEUE_OVERFLOW = 18\n'
    'OUT_MAX_QUEUE = 19\n'
    'OUT_FINAL_X0 = 20                    # 4 signed Q24\n'
    'OUT_LOCAL_EVENTS0 = 24               # 4 uint32\n'
    'OUT_A35_TX = 28\n'
    'OUT_A35_RX = 29\n'
    'OUT_A35_MISSING = 30\n'
    'OUT_A35_WRONG = 31\n'
    'OUT_AXIS_SYNOP = 32\n'
    'OUT_A35_UPDATES = 33\n'
    'OUT_AXIS_FINAL0 = 34                 # owner only, 3 signed Q24\n'
    '\n'
    '# Same successful calibrated intrinsic noise as the frozen dense custom APLX.\n'
    'SIGMA_LOCAL = 1.27\n'
    'TAU_M_MS = 7.5\n'
    'CM_NF = 7.5\n'
    'NOISE_CURRENT_SD = (\n'
    '    SIGMA_LOCAL * (TAU_M_MS / 1000.0) / math.sqrt(0.001) * (CM_NF / TAU_M_MS)\n'
    ')\n'
    'NOISE_SD_Q24 = int(np.rint(NOISE_CURRENT_SD * (1 << Q_STATE)))\n'
    '\n'
    '# Frozen 30-image dense G3 reference, used only for post-run comparison.\n'
    '# It never gates, clips, retunes, or changes execution.\n'
    'DENSE_G3_REFERENCE = {\n'
    '    "n": 30,\n'
    '    "counts": [15, 7, 8],\n'
    '    "P_gen": [0.5, 7 / 30, 8 / 30],\n'
    '    "TV": 0.10,\n'
    '    "Q_mean": 0.8843994140376564,\n'
    '    "Q_min": 0.7560624903418135,\n'
    '    "board_generation_mean_s": 30.867274841914575,\n'
    '}\n'
    '\n'
    '\n'
    '# Frozen A3 / A3.6 parameters (copied from the accepted Stage-S1 controller).\n'
    'AR = 0.82\n'
    'AH = 0.88\n'
    'AN = 0.92\n'
    'AA = 0.82\n'
    'HERD_GAIN = 2.5\n'
    'NOV_GAIN = 1.0\n'
    'ADAPT_GAIN = 0.5\n'
    'EVENT_WINDOW_MS = 10.0\n'
    'COUPLING_ETA = 0.05\n'
    'COUPLING_LEAK = 0.01\n'
    'DIAG_LO = 0.5\n'
    'DIAG_HI = 1.75\n'
    'OFF_ABS = 0.75\n'
    '\n'
    '\n'
    'class PoleMatchedIzh:\n'
    '    def __init__(self, discrete_poles, a_izh, window_ms=EVENT_WINDOW_MS):\n'
    '        p1, p2 = map(float, discrete_poles)\n'
    '        T = float(window_ms)\n'
    '        r1 = -math.log(p1) / T\n'
    '        r2 = -math.log(p2) / T\n'
    '        S = -(r1 + r2)\n'
    '        P = r1 * r2\n'
    '        self.window_ms = T\n'
    '        self.a = float(a_izh)\n'
    '        self.q = S + self.a\n'
    '        self.b = self.q + P / self.a\n'
    '        self.v0 = (self.q - 5.0) / 0.08\n'
    '        self.u0 = self.b * self.v0\n'
    '        self.I0 = -0.04 * self.v0 * self.v0 - 5.0 * self.v0 - 140.0 + self.u0\n'
    '        self.A = np.array([[self.q, -1.0], [self.a * self.b, -self.a]], dtype=np.float64)\n'
    '        self.Phi = expm(self.A * T)\n'
    '\n'
    '    def gamma(self, gain):\n'
    '        B = np.array([float(gain), 0.0], dtype=np.float64)\n'
    '        return np.linalg.solve(self.A, (self.Phi - np.eye(2)) @ B)\n'
    '\n'
    '    def transfer_readout(self, gain, num_s, num_0):\n'
    '        g = float(gain)\n'
    '        cv = float(num_s) / g\n'
    '        cu = (float(num_0) / g - cv * self.a) / (self.a * self.b)\n'
    '        return np.array([cv, cu], dtype=np.float64)\n'
    '\n'
    '\n'
    'class RHRealization:\n'
    '    def __init__(self, gain=1e-4, window_ms=EVENT_WINDOW_MS):\n'
    '        self.gain = float(gain)\n'
    '        self.izh = PoleMatchedIzh((AR, AH), a_izh=0.003, window_ms=window_ms)\n'
    '        br = 1.0 - AR\n'
    '        bh = 1.0 - AH\n'
    '        Aref = np.array([[AR, 0.0], [-bh * AR, AH]], dtype=np.float64)\n'
    '        Bref = np.array([br, -bh * br], dtype=np.float64)\n'
    '        Gamma = self.izh.gamma(self.gain)\n'
    '        Cref = np.column_stack([Bref, Aref @ Bref])\n'
    '        Cizh = np.column_stack([Gamma, self.izh.Phi @ Gamma])\n'
    '        T = Cizh @ np.linalg.inv(Cref)\n'
    '        Tinv = np.linalg.inv(T)\n'
    '        self.Cr = np.array([1.0, 0.0]) @ Tinv\n'
    '        self.Ch = np.array([0.0, 1.0]) @ Tinv\n'
    '\n'
    '\n'
    'class OnePoleIzhRealization:\n'
    '    def __init__(self, target_a, gain=1e-4, extra_discrete_pole=0.5,\n'
    '                 a_izh=0.005, window_ms=EVENT_WINDOW_MS):\n'
    '        self.gain = float(gain)\n'
    '        self.izh = PoleMatchedIzh((float(target_a), float(extra_discrete_pole)),\n'
    '                                  a_izh=a_izh, window_ms=window_ms)\n'
    '        rt = -math.log(float(target_a)) / float(window_ms)\n'
    '        re = -math.log(float(extra_discrete_pole)) / float(window_ms)\n'
    '        self.C = self.izh.transfer_readout(self.gain, num_s=rt, num_0=rt * re)\n'
    '\n'
    '\n'
    'def _izh_window(v, u, inp, gain, a, b, I0, steps):\n'
    '    spikes = 0\n'
    '    inp = np.asarray(inp, float)\n'
    '    for _ in range(int(steps)):\n'
    '        I = float(I0) + float(gain) * inp\n'
    '        dv1 = 0.04 * v * v + 5.0 * v + 140.0 - u + I\n'
    '        du1 = float(a) * (float(b) * v - u)\n'
    '        vm = v + 0.5 * dv1\n'
    '        um = u + 0.5 * du1\n'
    '        dv2 = 0.04 * vm * vm + 5.0 * vm + 140.0 - um + I\n'
    '        du2 = float(a) * (float(b) * vm - um)\n'
    '        v += dv2\n'
    '        u += du2\n'
    '        hit = v >= 30.0\n'
    '        if np.any(hit):\n'
    '            spikes += int(np.sum(hit))\n'
    '            v[hit] = -65.0\n'
    '            u[hit] += 8.0\n'
    '    return spikes\n'
    '\n'
    '\n'
    'class SlowHerdingPopulation:\n'
    '    def __init__(self, gain_rh=1e-4, gain_n=1e-4, gain_a=1e-4):\n'
    '        self.rh = RHRealization(gain_rh)\n'
    '        self.nn = OnePoleIzhRealization(AN, gain_n)\n'
    '        self.aa = OnePoleIzhRealization(AA, gain_a)\n'
    '        self.vrh = np.full(3, self.rh.izh.v0, dtype=float)\n'
    '        self.urh = np.full(3, self.rh.izh.u0, dtype=float)\n'
    '        self.vn = np.full(3, self.nn.izh.v0, dtype=float)\n'
    '        self.un = np.full(3, self.nn.izh.u0, dtype=float)\n'
    '        self.va = np.full(3, self.aa.izh.v0, dtype=float)\n'
    '        self.ua = np.full(3, self.aa.izh.u0, dtype=float)\n'
    '        self.spikes_rh = self.spikes_n = self.spikes_a = 0\n'
    '\n'
    '    def _zrh(self):\n'
    '        return np.column_stack([self.vrh - self.rh.izh.v0, self.urh - self.rh.izh.u0])\n'
    '\n'
    '    def _zn(self):\n'
    '        return np.column_stack([self.vn - self.nn.izh.v0, self.un - self.nn.izh.u0])\n'
    '\n'
    '    def _za(self):\n'
    '        return np.column_stack([self.va - self.aa.izh.v0, self.ua - self.aa.izh.u0])\n'
    '\n'
    '    def readouts(self):\n'
    '        z = self._zrh()\n'
    '        r = z @ self.rh.Cr\n'
    '        h = z @ self.rh.Ch\n'
    '        n = self._zn() @ self.nn.C\n'
    '        a = self._za() @ self.aa.C\n'
    '        return r, h, n, a\n'
    '\n'
    '    def score(self):\n'
    '        _, h, n, a = self.readouts()\n'
    '        return HERD_GAIN * h + NOV_GAIN * n - ADAPT_GAIN * a\n'
    '\n'
    '    def update(self, observed_mode: int, prior: np.ndarray):\n'
    '        q = -np.asarray(prior, float).copy()\n'
    '        q[int(observed_mode)] += 1.0\n'
    '        steps = int(round(EVENT_WINDOW_MS))\n'
    '        self.spikes_rh += _izh_window(\n'
    '            self.vrh, self.urh, q, self.rh.gain,\n'
    '            self.rh.izh.a, self.rh.izh.b, self.rh.izh.I0, steps)\n'
    '        r = self._zrh() @ self.rh.Cr\n'
    '        d = -r\n'
    '        self.spikes_n += _izh_window(\n'
    '            self.vn, self.un, np.maximum(d, 0.0), self.nn.gain,\n'
    '            self.nn.izh.a, self.nn.izh.b, self.nn.izh.I0, steps)\n'
    '        self.spikes_a += _izh_window(\n'
    '            self.va, self.ua, np.maximum(-d, 0.0), self.aa.gain,\n'
    '            self.aa.izh.a, self.aa.izh.b, self.aa.izh.I0, steps)\n'
    '        return self.readouts()\n'
    '\n'
    '\n'
    'def local_command(C: np.ndarray, score: np.ndarray) -> np.ndarray:\n'
    '    u = np.zeros(3, float)\n'
    '    for j in range(3):\n'
    '        for k in range(3):\n'
    '            u[j] += float(C[j, k]) * float(score[k])\n'
    '    return u\n'
    '\n'
    '\n'
    'def update_local_couplings(C: np.ndarray, score: np.ndarray, observed_mode: int,\n'
    '                           eta=COUPLING_ETA, lam=COUPLING_LEAK,\n'
    '                           diag_lo=DIAG_LO, diag_hi=DIAG_HI,\n'
    '                           off_abs=OFF_ABS, normalize=True) -> np.ndarray:\n'
    '    C = np.asarray(C, float).copy()\n'
    '    s = np.asarray(score, float)\n'
    '    o = np.zeros(3, float)\n'
    '    o[int(observed_mode)] = 1.0\n'
    '    for j in range(3):\n'
    '        for k in range(3):\n'
    '            scaffold = 1.0 if j == k else 0.0\n'
    '            pre = max(float(s[k]), 0.0)\n'
    '            C[j, k] += float(eta) * pre * (scaffold - o[j]) - float(lam) * (C[j, k] - scaffold)\n'
    '            if j == k:\n'
    '                C[j, k] = min(max(C[j, k], diag_lo), diag_hi)\n'
    '            else:\n'
    '                C[j, k] = min(max(C[j, k], -off_abs), off_abs)\n'
    '    if normalize:\n'
    '        for k in range(3):\n'
    '            C[:, k] /= np.linalg.norm(C[:, k]) + 1e-30\n'
    '    return C\n'
    '\n'
    '\n'
    'def tv_distance(p: np.ndarray, q: np.ndarray) -> float:\n'
    '    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())\n'
    '\n'
    '# =============================================================================\n'
    '# SpiNNaker event-key layout and selective routing\n'
    '# =============================================================================\n'
    '# key bits:\n'
    '#   31..28  0xA magic\n'
    '#   27      route kind: 0 = visible-source -> hidden targets only\n'
    '#                       1 = hidden-source/control -> all targets\n'
    '#   26..21  source worker-chip index (0..63)\n'
    '#   20..11  source model-step colour modulo 1024\n'
    '#   10..0   source packed-group id (0..2047; 2047 reserved by control)\n'
    'KEY_MAGIC = 0xA0000000\n'
    'KEY_MAGIC_MASK = 0xF0000000\n'
    'KIND_BIT = 0x08000000\n'
    'ROOT_SHIFT = 21\n'
    'ROOT_MASK_VALUE = 0x3F\n'
    'ROUTE_MASK = 0xFFE00000\n'
    'LOW_MASK = 0x001FFFFF\n'
    'CTRL_LOW = LOW_MASK\n'
    'MAX_ROOTS = 64\n'
    '\n'
    '\n'
    'def route_group_key(root_index: int, all_targets: bool) -> int:\n'
    '    root_index = int(root_index)\n'
    '    if not 0 <= root_index < MAX_ROOTS:\n'
    '        raise ValueError(f"root index out of range: {root_index}")\n'
    '    return KEY_MAGIC | (KIND_BIT if all_targets else 0) | (root_index << ROOT_SHIFT)\n'
    '\n'
    '\n'
    'def build_machine_adjacency(machine):\n'
    '    coords = {(int(ch.x), int(ch.y)) for ch in machine.chips}\n'
    '    adjacency = {xy: [] for xy in coords}\n'
    '    for ch in machine.chips:\n'
    '        src = (int(ch.x), int(ch.y))\n'
    '        for link in ch.router.links:\n'
    '            dst = (int(link.destination_x), int(link.destination_y))\n'
    '            if dst in coords:\n'
    '                adjacency[src].append((dst, int(link.source_link_id)))\n'
    '    return adjacency\n'
    '\n'
    '\n'
    'def bfs_parent_tree(adjacency, root):\n'
    '    root = tuple(root)\n'
    '    parent = {root: None}\n'
    '    parent_link = {}\n'
    '    q = deque([root])\n'
    '    while q:\n'
    '        u = q.popleft()\n'
    '        for v, link_id in adjacency.get(u, ()):\n'
    '            if v in parent:\n'
    '                continue\n'
    '            parent[v] = u\n'
    '            parent_link[v] = int(link_id)\n'
    '            q.append(v)\n'
    '    return parent, parent_link\n'
    '\n'
    '\n'
    'def pruned_tree_to_targets(adjacency, root, targets):\n'
    '    root = tuple(root)\n'
    '    targets = {tuple(t) for t in targets}\n'
    '    parent, parent_link = bfs_parent_tree(adjacency, root)\n'
    '    missing = sorted(t for t in targets if t not in parent)\n'
    '    if missing:\n'
    '        raise RuntimeError(\n'
    '            f"root {root} cannot reach target chips {missing[:8]}"\n'
    '            + ("..." if len(missing) > 8 else "")\n'
    '        )\n'
    '    required = {root}\n'
    '    for t in targets:\n'
    '        u = t\n'
    '        while u is not None and u not in required:\n'
    '            required.add(u)\n'
    '            u = parent[u]\n'
    '    child_links = defaultdict(list)\n'
    '    for node in required:\n'
    '        if node == root:\n'
    '            continue\n'
    '        p = parent[node]\n'
    '        if p not in required:\n'
    '            raise RuntimeError("internal routing-tree inconsistency")\n'
    '        child_links[p].append(parent_link[node])\n'
    '    return required, {k: sorted(set(v)) for k, v in child_links.items()}\n'
    '\n'
    '\n'
    'def build_selective_routes(machine, by_chip):\n'
    '    """Build two trees per source worker chip.\n'
    '\n'
    '    visible-source tree -> hidden target processors only\n'
    '    hidden-source tree  -> all target processors\n'
    '\n'
    '    This is not pruning: the omitted V->V block is exactly K_VV = 0.\n'
    '    """\n'
    '    from spinn_machine import MulticastRoutingEntry, RoutingEntry\n'
    '\n'
    '    worker_chips = sorted(tuple(xy) for xy in by_chip)\n'
    '    if len(worker_chips) >= MAX_ROOTS:\n'
    '        raise RuntimeError(\n'
    '            f"E1 key format supports <{MAX_ROOTS} worker chips; mapping used {len(worker_chips)}"\n'
    '        )\n'
    '    root_index = {xy: i for i, xy in enumerate(worker_chips)}\n'
    '    adjacency = build_machine_adjacency(machine)\n'
    '\n'
    '    all_processors = {}\n'
    '    hidden_processors = {}\n'
    '    for xy, rows in by_chip.items():\n'
    '        all_processors[xy] = sorted(int(p) for _gid, p, _base in rows)\n'
    '        hidden_processors[xy] = sorted(\n'
    '            int(p) for _gid, p, base in rows if int(base) >= NV\n'
    '        )\n'
    '\n'
    '    all_target_chips = {xy for xy, ps in all_processors.items() if ps}\n'
    '    hidden_target_chips = {xy for xy, ps in hidden_processors.items() if ps}\n'
    '    n_all = sum(len(v) for v in all_processors.values())\n'
    '    n_hidden = sum(len(v) for v in hidden_processors.values())\n'
    '    if n_all != N_GROUPS or n_hidden != N_HID_GROUPS:\n'
    '        raise RuntimeError(\n'
    '            f"target-core accounting mismatch: all={n_all}/{N_GROUPS}, "\n'
    '            f"hidden={n_hidden}/{N_HID_GROUPS}"\n'
    '        )\n'
    '\n'
    '    routes_by_chip = defaultdict(list)\n'
    '    tree_stats = {}\n'
    '\n'
    '    def add_tree(root, *, all_targets: bool):\n'
    '        ridx = root_index[root]\n'
    '        targets = all_target_chips if all_targets else hidden_target_chips\n'
    '        proc_map = all_processors if all_targets else hidden_processors\n'
    '        required, child_links = pruned_tree_to_targets(adjacency, root, targets)\n'
    '        gkey = route_group_key(ridx, all_targets)\n'
    '        delivered = 0\n'
    '        for node in sorted(required):\n'
    '            processors = proc_map.get(node, [])\n'
    '            links = child_links.get(node, [])\n'
    '            if processors or links:\n'
    '                routes_by_chip[node].append(\n'
    '                    MulticastRoutingEntry(\n'
    '                        gkey,\n'
    '                        ROUTE_MASK,\n'
    '                        RoutingEntry(processor_ids=processors, link_ids=links),\n'
    '                    )\n'
    '                )\n'
    '                delivered += len(processors)\n'
    '        expected = n_all if all_targets else n_hidden\n'
    '        if delivered != expected:\n'
    '            raise RuntimeError(\n'
    '                f"route root={root} all={all_targets} delivered {delivered}, expected {expected}"\n'
    '            )\n'
    '        tree_stats[f"{root[0]},{root[1]}:{\'ALL\' if all_targets else \'HID\'}"] = {\n'
    '            "root_index": ridx,\n'
    '            "group_key": int(gkey),\n'
    '            "nodes": len(required),\n'
    '            "edges": sum(len(v) for v in child_links.values()),\n'
    '            "target_processors": delivered,\n'
    '        }\n'
    '\n'
    '    for root in worker_chips:\n'
    '        add_tree(root, all_targets=False)\n'
    '        add_tree(root, all_targets=True)\n'
    '\n'
    '    return {\n'
    '        "routes_by_chip": dict(routes_by_chip),\n'
    '        "root_index": root_index,\n'
    '        "worker_chips": worker_chips,\n'
    '        "total_entries": sum(len(v) for v in routes_by_chip.values()),\n'
    '        "max_entries_per_chip": max((len(v) for v in routes_by_chip.values()), default=0),\n'
    '        "tree_stats": tree_stats,\n'
    '        "n_all_target_cores": n_all,\n'
    '        "n_hidden_target_cores": n_hidden,\n'
    '    }\n'
    '\n'
    '\n'
    'def _route_sig(e):\n'
    '    return (int(e.key), int(e.mask), int(e.spinnaker_route), bool(e.defaultable))\n'
    '\n'
    '\n'
    'def install_and_verify_routes(txrx, plan, app_id: int):\n'
    '    stats = {}\n'
    '    for (x, y), routes in sorted(plan["routes_by_chip"].items()):\n'
    '        txrx.load_multicast_routes(int(x), int(y), list(routes), int(app_id))\n'
    '        reread = list(txrx.get_multicast_routes(int(x), int(y), int(app_id)))\n'
    '        a = sorted(_route_sig(e) for e in routes)\n'
    '        b = sorted(_route_sig(e) for e in reread)\n'
    '        ok = a == b\n'
    '        stats[f"{x},{y}"] = {\n'
    '            "planned": len(a), "reloaded": len(b), "exact": ok\n'
    '        }\n'
    '        if not ok:\n'
    '            raise RuntimeError(f"explicit route verification failed on chip {(x, y)}")\n'
    '    return stats\n'
    '\n'
    '\n'
    '# =============================================================================\n'
    '# Frozen host-side data preparation / evaluation only\n'
    '# =============================================================================\n'
    'def align_up(n: int, a: int = 256) -> int:\n'
    '    return ((int(n) + a - 1) // a) * a\n'
    '\n'
    '\n'
    'def sha256_file(path: Path) -> str:\n'
    '    h = hashlib.sha256()\n'
    '    with path.open("rb") as f:\n'
    '        for b in iter(lambda: f.read(1 << 20), b""):\n'
    '            h.update(b)\n'
    '    return h.hexdigest()\n'
    '\n'
    '\n'
    'def json_dump(path: Path, obj: Any) -> None:\n'
    '    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")\n'
    '\n'
    '\n'
    'def qround_i32(x: np.ndarray | float, q: int, *, name: str = "value") -> np.ndarray:\n'
    '    """Quantize to signed int32 fixed point and reject only a true storage overflow."""\n'
    '    scaled = np.rint(np.asarray(x, dtype=np.float64) * float(1 << q))\n'
    '    lo = np.iinfo(np.int32).min\n'
    '    hi = np.iinfo(np.int32).max\n'
    '    if np.any(~np.isfinite(scaled)):\n'
    '        raise ValueError(f"{name}: non-finite value before fixed-point conversion")\n'
    '    if np.any((scaled < lo) | (scaled > hi)):\n'
    '        mn = float(np.min(scaled))\n'
    '        mx = float(np.max(scaled))\n'
    '        raise OverflowError(\n'
    '            f"{name}: signed-int32 fixed-point storage overflow "\n'
    '            f"(scaled range {mn:.6g}..{mx:.6g}, allowed {lo}..{hi})"\n'
    '        )\n'
    '    return scaled.astype(np.int32)\n'
    '\n'
    '\n'
    'def load_checkpoint(path: Path) -> dict[str, np.ndarray]:\n'
    '    d = np.load(path, allow_pickle=True)\n'
    '    required = ["Wvh", "Whh", "bv", "bh", "symbols", "prior_learned"]\n'
    '    missing = [k for k in required if k not in d.files]\n'
    '    if missing:\n'
    '        raise KeyError(f"checkpoint missing keys {missing}")\n'
    '    cp = {k: np.asarray(d[k]) for k in d.files}\n'
    '    if cp["Wvh"].shape != (NV, NH):\n'
    '        raise ValueError(f"Wvh shape {cp[\'Wvh\'].shape}, expected {(NV, NH)}")\n'
    '    if cp["Whh"].shape != (NH, NH):\n'
    '        raise ValueError(f"Whh shape {cp[\'Whh\'].shape}, expected {(NH, NH)}")\n'
    '    if cp["bv"].shape != (NV,) or cp["bh"].shape != (NH,):\n'
    '        raise ValueError("bias vector shape mismatch")\n'
    '    return cp\n'
    '\n'
    '\n'
    '\n'
    'def load_axes(path: Path) -> np.ndarray:\n'
    '    d = np.load(path, allow_pickle=True)\n'
    '    if "axes" not in d.files:\n'
    '        raise KeyError(f"{path} has no \'axes\'")\n'
    '    a = np.asarray(d["axes"], dtype=np.float64)\n'
    '    if a.shape != (N_AXES, NH):\n'
    '        raise ValueError(f"axes shape {a.shape}, expected {(N_AXES, NH)}")\n'
    '    return a\n'
    '\n'
    '\n'
    'def command_coefficients(command: np.ndarray, axes: np.ndarray, gamma: float) -> np.ndarray:\n'
    '    u = np.asarray(command, dtype=np.float64)\n'
    '    A = np.asarray(axes, dtype=np.float64)\n'
    '    den = np.sum(A * A, axis=1) + 1e-30\n'
    '    c = float(gamma) * (u / den)[:, None] * A\n'
    '    return qround_i32(c, Q_STATE)\n'
    '\n'
    'def classify_visible(sample: np.ndarray, symbols: np.ndarray):\n'
    '    # Evaluation only. This does not participate in E1 OFF dynamics.\n'
    '    P = np.asarray(symbols, float)\n'
    '    P = P - P.mean(axis=1, keepdims=True)\n'
    '    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)\n'
    '    q = np.asarray(sample, float) - float(np.mean(sample))\n'
    '    q = q / (np.linalg.norm(q) + 1e-12)\n'
    '    scores = q @ P.T\n'
    '    k = int(np.argmax(np.abs(scores)))\n'
    '    return k, float(scores[k]), np.asarray(scores, float)\n'
    '\n'
    '\n'
    'def _source_seed(run_seed: int, micro_id: int) -> int:\n'
    '    return int((104729 * int(run_seed) + 13007 * int(micro_id) + 7919) % 2147483646 + 1)\n'
    '\n'
    '\n'
    'def mars_kiss_seed_tuple(run_seed: int, micro_id: int) -> tuple[int, int, int, int]:\n'
    '    rng = np.random.RandomState(_source_seed(run_seed, micro_id))\n'
    '    vals = [int(rng.randint(-0x80000000, 0x7FFFFFFF)) + 0x80000000 for _ in range(4)]\n'
    '    if vals[1] == 0:\n'
    '        vals[1] = 13031301\n'
    '    vals[3] = vals[3] % 698769068 + 1\n'
    '    return tuple(int(v) for v in vals)\n'
    '\n'
    '\n'
    'def initial_membranes(seed: int) -> np.ndarray:\n'
    '    out = np.empty((N, N_BASIS), dtype=np.float64)\n'
    '    for basis in range(N_BASIS):\n'
    '        rv = np.random.default_rng(int(seed) + basis * 10007)\n'
    '        rh = np.random.default_rng(int(seed) + basis * 10007 + 500003)\n'
    '        out[:NV, basis] = rv.uniform(0.0, 1.0, NV)\n'
    '        out[NV:, basis] = rh.uniform(0.0, 1.0, NH)\n'
    '    return out\n'
    '\n'
    '\n'
    'def make_static_record(cp: dict[str, np.ndarray], axes: np.ndarray, logical_base: int) -> np.ndarray:\n'
    '    """Create only the exact coupling block needed by one 4-target core.\n'
    '\n'
    '    Visible target group: only hidden sources, K = -W.\n'
    '    Hidden target group : visible + hidden sources, K = [-W^T | -H].\n'
    '    No learned connection is approximated or removed.\n'
    '    """\n'
    '    logical_base = int(logical_base)\n'
    '    ids = np.arange(logical_base, logical_base + PACK, dtype=int)\n'
    '    W = np.asarray(cp["Wvh"], dtype=np.float64)\n'
    '    H = np.asarray(cp["Whh"], dtype=np.float64)\n'
    '    bias_all = -np.concatenate([\n'
    '        np.asarray(cp["bv"], dtype=np.float64),\n'
    '        np.asarray(cp["bh"], dtype=np.float64),\n'
    '    ])\n'
    '\n'
    '    if logical_base < NV:\n'
    '        if logical_base + PACK > NV:\n'
    '            raise RuntimeError("visible/hidden boundary crossed by a packed core")\n'
    '        block = -W[ids, :]                         # [4, 512]\n'
    '    else:\n'
    '        hids = ids - NV\n'
    '        block = np.concatenate([\n'
    '            -W[:, hids].T,                         # [4, 784]\n'
    '            -H[hids, :],                           # [4, 512]\n'
    '        ], axis=1)                                 # [4, 1296]\n'
    '\n'
    '    biases = qround_i32(bias_all[ids], Q_STATE).astype("<i4")\n'
    '    weights = qround_i32(block, Q_STATE).astype("<i4", copy=False)\n'
    '    parts = [biases, weights.reshape(-1)]\n'
    '    # Group 0 is the single A3.5 axis owner. It already receives every hidden-source\n'
    '    # event, so appending the 3x512 frozen axes here adds no network traffic.\n'
    '    if logical_base == 0:\n'
    '        parts.append(qround_i32(np.asarray(axes, float), Q_AXIS).astype("<i4", copy=False).reshape(-1))\n'
    '    return np.concatenate(parts).astype("<i4", copy=False)\n'
    '\n'
    '\n'
    '# =============================================================================\n'
    '# Embedded APLX kernel\n'
    '# =============================================================================\n'
    "KERNEL_C = r'''/*\n"
    ' * Efficient persistent dynamical generator — SpiNNaker-1\n'
    ' *\n'
    ' * Scientific dynamics are frozen. Hardware execution only:\n'
    ' *   - 4 logical variables/core, 7 hard-LIF states/logical variable\n'
    ' *   - persistent F_i recurrence; no global x and no dense Kx\n'
    ' *   - sparse packed multicast events\n'
    ' *   - MCPL callback only queues; USER_EVENT does synaptic processing\n'
    ' *   - 1-ms scientific step divided into physical send slots\n'
    ' *   - no global barrier / completion protocol\n'
    ' *   - no arbitrary dynamical-amplitude limits\n'
    ' *   - only hardware-capacity and protocol-integrity failures stop execution\n'
    ' */\n'
    '#include <spin1_api.h>\n'
    '#include <sark.h>\n'
    '#include <stdint.h>\n'
    '#include <stdbool.h>\n'
    '#include <random.h>\n'
    '#include <normal.h>\n'
    '#include <stdfix-full-iso.h>\n'
    '\n'
    '#define MAGIC_CFG        0x45314431u\n'
    '#define MAGIC_OUT        0x45314f31u\n'
    '#define CFG_VERSION      1u\n'
    '#define SDRAM_TAG        1u\n'
    '\n'
    '#define N_LOGICAL        1296u\n'
    '#define NV               784u\n'
    '#define NH               512u\n'
    '#define PACK             4u\n'
    '#define N_GROUPS         324u\n'
    '#define N_VIS_GROUPS     196u\n'
    '#define N_BASIS          7u\n'
    '#define N_AXES           3u\n'
    '#define MAX_CORES        18u\n'
    '#define SEEN_WORDS       11u\n'
    '\n'
    '#define SCIENTIFIC_US    1000u\n'
    '#define DRAIN_TICKS      2u\n'
    '#define AXIS_SEND_SLOT   1u\n'
    '#define MICRO_FIRST_SLOT 2u\n'
    '#define SEND_FIRST_SLOT  3u\n'
    '#define EVENT_QUEUE_CAP  512u\n'
    '#define EVENT_QUEUE_MASK (EVENT_QUEUE_CAP - 1u)\n'
    '\n'
    '#define Q_STATE          24\n'
    '#define Q_AXIS           20\n'
    '#define Q_SCALAR         30\n'
    '#define ONE_Q24          ((int32_t)16777216)\n'
    '\n'
    '#define STATUS_IDLE       1u\n'
    '#define STATUS_ARMED      2u\n'
    '#define STATUS_RUNNING    3u\n'
    '#define STATUS_IMAGE_DONE 4u\n'
    '#define STATUS_APP_DONE   5u\n'
    '#define STATUS_ERROR      0xdead0001u\n'
    '\n'
    '#define ERR_NONE          0u\n'
    '#define ERR_CONFIG        (1u << 0)\n'
    '#define ERR_SEND          (1u << 1)\n'
    '#define ERR_DUPLICATE     (1u << 2)\n'
    '#define ERR_WRONG_ROUTE   (1u << 3)\n'
    '#define ERR_LATE_EVENT    (1u << 4)\n'
    '#define ERR_NUMERIC_OVERFLOW (1u << 5)\n'
    '#define ERR_QUEUE         (1u << 6)\n'
    '\n'
    '#define PHASE_WAIT_COMMAND 0u\n'
    '#define PHASE_ARMED        1u\n'
    '#define PHASE_RUNNING      2u\n'
    '#define PHASE_FINISHED     3u\n'
    '\n'
    '/* key: magic/kind/root in high bits; low 21 bits = step[20:11]+group[10:0] */\n'
    '#define KEY_MAGIC          0xa0000000u\n'
    '#define KEY_MAGIC_MASK     0xf0000000u\n'
    '#define STEP_MASK          0x3ffu\n'
    '#define GROUP_MASK         0x7ffu\n'
    '#define CTRL_LOW           0x001fffffu\n'
    '#define AXIS_GROUP_BASE    2044u\n'
    '\n'
    '/* Header layout. */\n'
    '#define CFG_MAGIC           0u\n'
    '#define CFG_VER             1u\n'
    '#define CFG_NLOGICAL        2u\n'
    '#define CFG_NV              3u\n'
    '#define CFG_NH              4u\n'
    '#define CFG_PACK            5u\n'
    '#define CFG_NGROUPS         6u\n'
    '#define CFG_GEN_STEPS       7u\n'
    '#define CFG_KEY_VIS         8u\n'
    '#define CFG_KEY_ALL         9u\n'
    '#define CFG_STATIC_OFFSET  10u\n'
    '#define CFG_STATIC_STRIDE  11u\n'
    '#define CFG_DYN_OFFSET     12u\n'
    '#define CFG_DYN_STRIDE     13u\n'
    '#define CFG_TIMER_TICK_US  14u\n'
    '#define CFG_ARM_EPOCH      15u\n'
    '#define CFG_GO_EPOCH       16u\n'
    '#define CFG_STOP_FLAG      17u\n'
    '#define CFG_NOISE_SD_Q24   18u\n'
    '#define CFG_OUT_OFFSET     19u\n'
    '#define CFG_OUT_STRIDE     20u\n'
    '#define CFG_CONTROL_STEPS  21u\n'
    '#define CFG_GROUP_BY_P     32u\n'
    '#define CFG_WORDS          64u\n'
    '\n'
    '/* Dynamic record. */\n'
    '#define DYN_INIT_V0         0u\n'
    '#define DYN_SEED0          32u\n'
    '#define DYN_CTRL0         144u\n'
    '#define DYN_WORDS         160u\n'
    '\n'
    '/* Output record. */\n'
    '#define OUT_MAGIC             0u\n'
    '#define OUT_GROUP             1u\n'
    '#define OUT_STATUS            2u\n'
    '#define OUT_ERROR             3u\n'
    '#define OUT_EPOCH             4u\n'
    '#define OUT_STEPS             5u\n'
    '#define OUT_SEND_FAIL         6u\n'
    '#define OUT_DUPLICATE         7u\n'
    '#define OUT_WRONG_ROUTE       8u\n'
    '#define OUT_LATE_EVENT        9u\n'
    '#define OUT_EVENT_GENERATED  10u /* logical nonzero events */\n'
    '#define OUT_EVENT_SENT       11u /* packed packets */\n'
    '#define OUT_EVENT_RX         12u /* packed packet deliveries */\n'
    '#define OUT_EVENT_PROCESSED  13u /* packed packets processed */\n'
    '#define OUT_LOOPBACK_IGNORED 14u\n'
    '#define OUT_SYNOP            15u\n'
    '#define OUT_MAX_RX_INTERVAL  16u /* max packed packets / scientific ms */\n'
    '#define OUT_TIMER_SLIP_MAX   17u\n'
    '#define OUT_QUEUE_OVERFLOW   18u\n'
    '#define OUT_MAX_QUEUE        19u\n'
    '#define OUT_FINAL_X0         20u\n'
    '#define OUT_LOCAL_EVENTS0    24u\n'
    '#define OUT_A35_TX           28u\n'
    '#define OUT_A35_RX           29u\n'
    '#define OUT_A35_MISSING      30u\n'
    '#define OUT_A35_WRONG        31u\n'
    '#define OUT_AXIS_SYNOP       32u\n'
    '#define OUT_A35_UPDATES      33u\n'
    '#define OUT_AXIS_FINAL0      34u\n'
    '#define OUT_WORDS            64u\n'
    '\n'
    '/* Frozen validated 7-LIF mapping. */\n'
    'static const int32_t BASE_Q30[N_BASIS] = {\n'
    '    1033120077, 1117575745, 1102809712,\n'
    '    1033120077, 1117575745, 1102809712, 966367642};\n'
    'static const int32_t DECODER_Q30[N_BASIS] = {\n'
    '    10419142, 167257183, 42700659,\n'
    '    -10419142, -167257183, -42700659, 0};\n'
    'static const int32_t FIELD_GAIN_Q30[N_BASIS] = {\n'
    '    1696271, 189463, 77602895,\n'
    '    -1696271, -189463, -77602895, 0};\n'
    'static const int32_t W_LOCAL_Q30[N_BASIS][N_BASIS] = {\n'
    '    { 1835602, 29466691, 7522829,-1835602,-29466691,-7522829,-64424509},\n'
    '    { 3221488, 51714150,13202592,-3221488,-51714150,-13202592,-64424509},\n'
    '    {   13796,   221457,   56538,  -13796,  -221457,   -56538,-64424509},\n'
    '    {-1835602,-29466691,-7522829, 1835602, 29466691, 7522829,-64424509},\n'
    '    {-3221488,-51714150,-13202592,3221488, 51714150,13202592,-64424509},\n'
    '    {  -13796,  -221457,  -56538,   13796,   221457,   56538,-64424509},\n'
    '    {       0,        0,       0,       0,        0,       0,        0}\n'
    '};\n'
    '#define AM_Q30  ((int32_t)939710196)\n'
    '#define AQ_Q30  ((int32_t)1052480311)\n'
    '\n'
    'static volatile uint32_t *cfg = 0;\n'
    'static volatile int32_t *dynrec = 0;\n'
    'static volatile int32_t *outrec = 0;\n'
    'static uint32_t group_id = 0xffffffffu;\n'
    'static uint32_t logical_base = 0xffffffffu;\n'
    'static uint32_t input_count = 0u;\n'
    'static uint32_t key_vis = 0u;\n'
    'static uint32_t key_all = 0u;\n'
    'static uint32_t gen_steps = 4320u;\n'
    'static uint32_t noise_sd_q24 = 0u;\n'
    'static uint32_t base_tick_us = 100u;\n'
    'static uint32_t ticks_per_model = 10u;\n'
    'static uint32_t send_slots = 7u;\n'
    'static uint32_t start_guard_ticks = 20u;\n'
    'static uint32_t control_steps = 40u;\n'
    '\n'
    '/* Static couplings local to this packed target. */\n'
    'static int32_t weights[PACK][N_LOGICAL];\n'
    'static int32_t logical_bias_q24[PACK];\n'
    'static int32_t axes_q20[N_AXES][NH];   /* used only by group 0 axis owner */\n'
    '\n'
    '/* Persistent 4x7 microscopic state. */\n'
    'static int32_t v_q24[PACK][N_BASIS];\n'
    'static int32_t q_q24[PACK][N_BASIS];\n'
    'static uint8_t refractory[PACK][N_BASIS];\n'
    'static mars_kiss64_seed_t rng_seed[PACK][N_BASIS];\n'
    'static uint32_t local_event_count[PACK];\n'
    'static int32_t ctrl_coeff_q24[PACK][N_AXES];\n'
    'static int32_t ctrl_z_q24[N_AXES];\n'
    'static uint32_t ctrl_seen_mask = 0u;\n'
    '\n'
    '/* Persistent field and one next-step event accumulator. */\n'
    'static int32_t field_state_q24[PACK];\n'
    'static int64_t acc_next_q48[PACK];\n'
    'static int64_t axis_state_q24_wide[N_AXES];\n'
    'static int64_t axis_acc_next_q44[N_AXES];\n'
    'static uint32_t seen_groups[SEEN_WORDS];\n'
    '\n'
    '/* Packed packet awaiting its physical send slot. */\n'
    'static uint32_t pending_payload = 0u;\n'
    'static uint32_t pending_step = 0u;\n'
    'static bool pending_send = false;\n'
    'static bool micro_pending = false;\n'
    'static bool axis_broadcast_pending = false;\n'
    'static bool control_due = false;\n'
    '\n'
    '/* Native-style input queue: MCPL interrupt only enqueues. */\n'
    'static uint32_t event_q_key[EVENT_QUEUE_CAP];\n'
    'static uint32_t event_q_payload[EVENT_QUEUE_CAP];\n'
    'static volatile uint32_t q_head = 0u;\n'
    'static volatile uint32_t q_tail = 0u;\n'
    'static volatile bool process_loop_running = false;\n'
    '\n'
    'static volatile uint32_t phase = PHASE_WAIT_COMMAND;\n'
    'static volatile uint32_t step_idx = 0u;      /* current scientific interval */\n'
    'static uint32_t steps_done = 0u;\n'
    'static volatile uint32_t start_tick_target = 0u;\n'
    'static uint32_t start_broadcast_remaining = 0u;\n'
    'static uint32_t run_start_tick = 0u;\n'
    'static uint32_t active_epoch = 0u;\n'
    'static uint32_t last_timer_tick = 0u;\n'
    '\n'
    '/* Provenance / efficiency. */\n'
    'static uint32_t send_fail = 0u;\n'
    'static uint32_t duplicate_total = 0u;\n'
    'static uint32_t wrong_route_total = 0u;\n'
    'static uint32_t late_event_total = 0u;\n'
    'static uint32_t event_generated_total = 0u;\n'
    'static uint32_t event_sent_total = 0u;\n'
    'static uint32_t event_rx_total = 0u;\n'
    'static uint32_t event_processed_total = 0u;\n'
    'static uint32_t loopback_ignored_total = 0u;\n'
    'static uint32_t synop_total = 0u;\n'
    'static volatile uint32_t rx_this_model = 0u;\n'
    'static uint32_t max_rx_interval = 0u;\n'
    'static uint32_t timer_slip_max = 0u;\n'
    'static uint32_t queue_overflow_total = 0u;\n'
    'static uint32_t max_queue_depth = 0u;\n'
    'static uint32_t a35_tx_total = 0u;\n'
    'static uint32_t a35_rx_total = 0u;\n'
    'static uint32_t a35_missing_total = 0u;\n'
    'static uint32_t a35_wrong_total = 0u;\n'
    'static uint32_t axis_synop_total = 0u;\n'
    'static uint32_t a35_updates_total = 0u;\n'
    'static uint32_t error_flags = ERR_NONE;\n'
    '\n'
    'static inline int64_t floor_rshift64(int64_t x, uint32_t q) {\n'
    '    if (x >= 0) return x >> q;\n'
    '    int64_t a = -x;\n'
    '    int64_t d = ((int64_t)1 << q);\n'
    '    return -((a + d - 1) >> q);\n'
    '}\n'
    '\n'
    '/* Saturating conversion used only at frozen int32-Q24 interfaces.\n'
    '   This is representation semantics inherited from the validated DYN core,\n'
    '   not a dynamical-amplitude guard. Persistent A3.5 axis state itself is int64. */\n'
    'static inline int32_t sat_i64_to_i32(int64_t x) {\n'
    '    if (x > 2147483647LL) return 2147483647;\n'
    '    if (x < -2147483648LL) return (int32_t)0x80000000u;\n'
    '    return (int32_t)x;\n'
    '}\n'
    '\n'
    'static inline int32_t mul_q24_q30(int32_t a_q24, int32_t b_q30) {\n'
    '    return (int32_t)floor_rshift64((int64_t)a_q24 * (int64_t)b_q30, Q_SCALAR);\n'
    '}\n'
    '\n'
    '/* Exact floor((x_q24 * a_q30)/2^30) without forming a potentially\n'
    '   overflowing 64x32 product.  This is used only for the global A3.5 axis\n'
    '   projection, whose natural range is wider than signed int32 Q24. */\n'
    'static inline int64_t mul_i64_q24_q30(int64_t x_q24, int32_t a_q30) {\n'
    '    int64_t hi = floor_rshift64(x_q24, Q_SCALAR);\n'
    '    int64_t rem = x_q24 - (hi << Q_SCALAR);  /* 0 <= rem < 2^30 */\n'
    '    int64_t lo = floor_rshift64(rem * (int64_t)a_q30, Q_SCALAR);\n'
    '    return hi * (int64_t)a_q30 + lo;\n'
    '}\n'
    '\n'
    'static inline uint32_t packet_step(uint32_t key) {\n'
    '    return (key >> 11) & STEP_MASK;\n'
    '}\n'
    'static inline uint32_t packet_group(uint32_t key) {\n'
    '    return key & GROUP_MASK;\n'
    '}\n'
    'static inline bool source_group_is_local(uint32_t sg) {\n'
    '    return sg == group_id;\n'
    '}\n'
    'static inline void clear_seen(void) {\n'
    '    for (uint32_t i = 0; i < SEEN_WORDS; i++) seen_groups[i] = 0u;\n'
    '}\n'
    'static inline bool seen_test_set(uint32_t sg) {\n'
    '    uint32_t wi = sg >> 5;\n'
    '    uint32_t bit = 1u << (sg & 31u);\n'
    '    bool old = (seen_groups[wi] & bit) != 0u;\n'
    '    seen_groups[wi] |= bit;\n'
    '    return old;\n'
    '}\n'
    '\n'
    'static inline uint32_t queue_depth(void) {\n'
    '    return (q_head - q_tail) & EVENT_QUEUE_MASK;\n'
    '}\n'
    '\n'
    'static inline int32_t decoded_local(uint32_t l) {\n'
    '    int64_t a = 0;\n'
    '    for (uint32_t b = 0; b < N_BASIS; b++) {\n'
    '        a += (int64_t)q_q24[l][b] * (int64_t)DECODER_Q30[b];\n'
    '    }\n'
    '    return sat_i64_to_i32(floor_rshift64(a, Q_SCALAR));\n'
    '}\n'
    '\n'
    'static inline bool source_is_valid_for_target(uint32_t source, uint32_t *column) {\n'
    '    if (source >= N_LOGICAL) return false;\n'
    '    if (logical_base < NV) {\n'
    '        if (source < NV) return false;   /* exact K_VV=0 */\n'
    '        *column = source - NV;\n'
    '        return *column < NH;\n'
    '    }\n'
    '    *column = source;\n'
    '    return true;\n'
    '}\n'
    '\n'
    'static inline void add_logical_event(uint32_t source, int32_t ev) {\n'
    '    uint32_t column;\n'
    '    if (!source_is_valid_for_target(source, &column)) {\n'
    '        wrong_route_total++;\n'
    '        error_flags |= ERR_WRONG_ROUTE;\n'
    '        return;\n'
    '    }\n'
    '    if (group_id == 0u && source >= NV) {\n'
    '        uint32_t h = source - NV;\n'
    '        for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '            axis_acc_next_q44[a] += (int64_t)axes_q20[a][h] * (int64_t)ev;\n'
    '            axis_synop_total++;\n'
    '        }\n'
    '    }\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        int32_t w = weights[l][column];\n'
    '        if (w != 0) {\n'
    '            acc_next_q48[l] += (int64_t)w * (int64_t)ev;\n'
    '            synop_total++;\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    'static inline int32_t logical_event_q24(uint8_t spike_mask) {\n'
    '    int64_t a = 0;\n'
    '    for (uint32_t b = 0; b < N_BASIS; b++) {\n'
    '        if (spike_mask & (1u << b)) {\n'
    '            a += (int64_t)ONE_Q24 * (int64_t)DECODER_Q30[b];\n'
    '        }\n'
    '    }\n'
    '    return sat_i64_to_i32(floor_rshift64(a, Q_SCALAR));\n'
    '}\n'
    '\n'
    'static inline void process_packed_payload(uint32_t source_group, uint32_t payload) {\n'
    '    uint32_t base = source_group * PACK;\n'
    '    if (base >= N_LOGICAL) {\n'
    '        wrong_route_total++;\n'
    '        error_flags |= ERR_WRONG_ROUTE;\n'
    '        return;\n'
    '    }\n'
    '    for (uint32_t lane = 0; lane < PACK; lane++) {\n'
    '        uint8_t mask = (uint8_t)((payload >> (lane * 7u)) & 0x7fu);\n'
    '        if (mask == 0u) continue;\n'
    '        uint32_t source = base + lane;\n'
    '        int32_t ev = logical_event_q24(mask);\n'
    '        if (ev != 0) add_logical_event(source, ev);\n'
    '    }\n'
    '}\n'
    '\n'
    'static inline bool send_packet_retry(uint32_t key, uint32_t payload) {\n'
    '    for (uint32_t n = 0; n < 16u; n++) {\n'
    '        if (spin1_send_mc_packet(key, payload, true)) return true;\n'
    '    }\n'
    '    send_fail++;\n'
    '    error_flags |= ERR_SEND;\n'
    '    return false;\n'
    '}\n'
    '\n'
    'static inline int32_t gaussian_current_q24(uint32_t l, uint32_t b) {\n'
    '    uint32_t u = mars_kiss64_seed(rng_seed[l][b]);\n'
    '    s1615 g = norminv_urt(u);\n'
    '    int32_t g_q15 = bitsk(g);\n'
    '    return sat_i64_to_i32(\n'
    '        floor_rshift64((int64_t)(int32_t)noise_sd_q24 * (int64_t)g_q15, 15u));\n'
    '}\n'
    '\n'
    'static inline uint8_t micro_step(uint32_t l, int32_t field_q24) {\n'
    '    int32_t current[N_BASIS];\n'
    '    for (uint32_t post = 0; post < N_BASIS; post++) {\n'
    '        int64_t c = floor_rshift64((int64_t)BASE_Q30[post], Q_SCALAR - Q_STATE);\n'
    '        for (uint32_t pre = 0; pre < N_BASIS; pre++) {\n'
    '            c += (int64_t)mul_q24_q30(q_q24[l][pre], W_LOCAL_Q30[post][pre]);\n'
    '        }\n'
    '        c += (int64_t)mul_q24_q30(field_q24, FIELD_GAIN_Q30[post]);\n'
    '        c += (int64_t)gaussian_current_q24(l, post);\n'
    '        current[post] = sat_i64_to_i32(c);\n'
    '    }\n'
    '\n'
    '    uint8_t mask = 0u;\n'
    '    for (uint32_t b = 0; b < N_BASIS; b++) {\n'
    '        bool sp = false;\n'
    '        if (refractory[l][b]) {\n'
    '            v_q24[l][b] = 0;\n'
    '            refractory[l][b] = 0u;\n'
    '        } else {\n'
    '            int32_t dv = v_q24[l][b] - current[b];\n'
    '            int32_t vn = current[b] + mul_q24_q30(dv, AM_Q30);\n'
    '            v_q24[l][b] = vn;\n'
    '            if (vn >= ONE_Q24) {\n'
    '                sp = true;\n'
    '                refractory[l][b] = 1u;\n'
    '                v_q24[l][b] = 0;\n'
    '                mask |= (uint8_t)(1u << b);\n'
    '            }\n'
    '        }\n'
    '        q_q24[l][b] = mul_q24_q30(q_q24[l][b], AQ_Q30);\n'
    '        if (sp) q_q24[l][b] += ONE_Q24;\n'
    '    }\n'
    '    return mask;\n'
    '}\n'
    '\n'
    'static inline void write_status(uint32_t status) {\n'
    '    outrec[OUT_MAGIC] = (int32_t)MAGIC_OUT;\n'
    '    outrec[OUT_GROUP] = (int32_t)group_id;\n'
    '    outrec[OUT_STATUS] = (int32_t)status;\n'
    '    outrec[OUT_ERROR] = (int32_t)error_flags;\n'
    '    outrec[OUT_EPOCH] = (int32_t)active_epoch;\n'
    '}\n'
    '\n'
    'static void flush_output(uint32_t status) {\n'
    '    outrec[OUT_STEPS] = (int32_t)steps_done;\n'
    '    outrec[OUT_SEND_FAIL] = (int32_t)send_fail;\n'
    '    outrec[OUT_DUPLICATE] = (int32_t)duplicate_total;\n'
    '    outrec[OUT_WRONG_ROUTE] = (int32_t)wrong_route_total;\n'
    '    outrec[OUT_LATE_EVENT] = (int32_t)late_event_total;\n'
    '    outrec[OUT_EVENT_GENERATED] = (int32_t)event_generated_total;\n'
    '    outrec[OUT_EVENT_SENT] = (int32_t)event_sent_total;\n'
    '    outrec[OUT_EVENT_RX] = (int32_t)event_rx_total;\n'
    '    outrec[OUT_EVENT_PROCESSED] = (int32_t)event_processed_total;\n'
    '    outrec[OUT_LOOPBACK_IGNORED] = (int32_t)loopback_ignored_total;\n'
    '    outrec[OUT_SYNOP] = (int32_t)synop_total;\n'
    '    outrec[OUT_MAX_RX_INTERVAL] = (int32_t)max_rx_interval;\n'
    '    outrec[OUT_TIMER_SLIP_MAX] = (int32_t)timer_slip_max;\n'
    '    outrec[OUT_QUEUE_OVERFLOW] = (int32_t)queue_overflow_total;\n'
    '    outrec[OUT_MAX_QUEUE] = (int32_t)max_queue_depth;\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        outrec[OUT_FINAL_X0 + l] = decoded_local(l);\n'
    '        outrec[OUT_LOCAL_EVENTS0 + l] = (int32_t)local_event_count[l];\n'
    '    }\n'
    '    outrec[OUT_A35_TX] = (int32_t)a35_tx_total;\n'
    '    outrec[OUT_A35_RX] = (int32_t)a35_rx_total;\n'
    '    outrec[OUT_A35_MISSING] = (int32_t)a35_missing_total;\n'
    '    outrec[OUT_A35_WRONG] = (int32_t)a35_wrong_total;\n'
    '    outrec[OUT_AXIS_SYNOP] = (int32_t)axis_synop_total;\n'
    '    outrec[OUT_A35_UPDATES] = (int32_t)a35_updates_total;\n'
    '    if (group_id == 0u) {\n'
    '        for (uint32_t a = 0; a < N_AXES; a++) outrec[OUT_AXIS_FINAL0 + a] = sat_i64_to_i32(axis_state_q24_wide[a]);\n'
    '    }\n'
    '    write_status(status);\n'
    '}\n'
    '\n'
    'static void fail_and_wait(uint32_t err) {\n'
    '    error_flags |= err;\n'
    '    flush_output(STATUS_ERROR);\n'
    '    phase = PHASE_WAIT_COMMAND;\n'
    '}\n'
    '\n'
    '/* MC interrupt is intentionally minimal, matching SpiNNaker local-only style. */\n'
    'static inline void trigger_process_loop(void) {\n'
    '    if (spin1_trigger_user_event(0u, 0u)) process_loop_running = true;\n'
    '}\n'
    '\n'
    'static void packet_callback(uint32_t key, uint32_t payload) {\n'
    '    if (phase != PHASE_ARMED && phase != PHASE_RUNNING) return;\n'
    '    if ((key & KEY_MAGIC_MASK) != KEY_MAGIC) return;\n'
    '\n'
    '    if ((key & CTRL_LOW) == CTRL_LOW) {\n'
    '        if (phase == PHASE_ARMED) {\n'
    '            uint32_t target = payload;\n'
    '            if (start_tick_target == 0u) start_tick_target = target;\n'
    '            else if (start_tick_target != target) {\n'
    '                wrong_route_total++;\n'
    '                error_flags |= ERR_WRONG_ROUTE;\n'
    '            }\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    uint32_t raw_group = packet_group(key);\n'
    '    if (raw_group >= AXIS_GROUP_BASE && raw_group < AXIS_GROUP_BASE + N_AXES) {\n'
    '        if (phase != PHASE_RUNNING || logical_base < NV) return;\n'
    '        uint32_t ps = packet_step(key);\n'
    '        uint32_t cur = step_idx & STEP_MASK;\n'
    '        if (ps != cur) {\n'
    '            a35_wrong_total++;\n'
    '            return;\n'
    '        }\n'
    '        uint32_t a = raw_group - AXIS_GROUP_BASE;\n'
    '        uint32_t bit = 1u << a;\n'
    '        if (ctrl_seen_mask & bit) {\n'
    '            a35_wrong_total++;\n'
    '            return;\n'
    '        }\n'
    '        ctrl_z_q24[a] = (int32_t)payload;\n'
    '        ctrl_seen_mask |= bit;\n'
    '        a35_rx_total++;\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (phase != PHASE_RUNNING) {\n'
    '        wrong_route_total++;\n'
    '        error_flags |= ERR_WRONG_ROUTE;\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    uint32_t sg = packet_group(key);\n'
    '    event_rx_total++;\n'
    '    rx_this_model++;\n'
    '\n'
    '    /* Hidden packed source locally injects once; ignore its router loopback. */\n'
    '    if (logical_base >= NV && source_group_is_local(sg)) {\n'
    '        loopback_ignored_total++;\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    uint32_t next = (q_head + 1u) & EVENT_QUEUE_MASK;\n'
    '    if (next == q_tail) {\n'
    '        queue_overflow_total++;\n'
    '        error_flags |= ERR_QUEUE;\n'
    '        return;\n'
    '    }\n'
    '    event_q_key[q_head] = key;\n'
    '    event_q_payload[q_head] = payload;\n'
    '    q_head = next;\n'
    '    uint32_t d = queue_depth();\n'
    '    if (d > max_queue_depth) max_queue_depth = d;\n'
    '    if (!process_loop_running) trigger_process_loop();\n'
    '}\n'
    '\n'
    'static void process_callback(uint32_t unused0, uint32_t unused1) {\n'
    '    (void)unused0; (void)unused1;\n'
    '    while (true) {\n'
    '        uint32_t key, payload;\n'
    '        uint32_t cspr = spin1_int_disable();\n'
    '        if (q_tail == q_head) {\n'
    '            process_loop_running = false;\n'
    '            spin1_mode_restore(cspr);\n'
    '            break;\n'
    '        }\n'
    '        key = event_q_key[q_tail];\n'
    '        payload = event_q_payload[q_tail];\n'
    '        q_tail = (q_tail + 1u) & EVENT_QUEUE_MASK;\n'
    '        spin1_mode_restore(cspr);\n'
    '\n'
    '        uint32_t ps = packet_step(key);\n'
    '        uint32_t sg = packet_group(key);\n'
    '        if (sg >= N_GROUPS) {\n'
    '            wrong_route_total++;\n'
    '            error_flags |= ERR_WRONG_ROUTE;\n'
    '            continue;\n'
    '        }\n'
    '        uint32_t cur = step_idx & STEP_MASK;\n'
    '        if (ps != cur) {\n'
    '            uint32_t back = (cur - ps) & STEP_MASK;\n'
    '            if (back > 0u && back < 512u) {\n'
    '                late_event_total++;\n'
    '                error_flags |= ERR_LATE_EVENT;\n'
    '            } else {\n'
    '                wrong_route_total++;\n'
    '                error_flags |= ERR_WRONG_ROUTE;\n'
    '            }\n'
    '            continue;\n'
    '        }\n'
    '        if (seen_test_set(sg)) {\n'
    '            duplicate_total++;\n'
    '            error_flags |= ERR_DUPLICATE;\n'
    '            continue;\n'
    '        }\n'
    '        process_packed_payload(sg, payload);\n'
    '        event_processed_total++;\n'
    '    }\n'
    '}\n'
    '\n'
    'static void arm_image(uint32_t epoch) {\n'
    '    active_epoch = epoch;\n'
    '    error_flags = ERR_NONE;\n'
    '    send_fail = duplicate_total = wrong_route_total = late_event_total = 0u;\n'
    '    event_generated_total = event_sent_total = event_rx_total = 0u;\n'
    '    event_processed_total = loopback_ignored_total = synop_total = 0u;\n'
    '    rx_this_model = max_rx_interval = timer_slip_max = 0u;\n'
    '    queue_overflow_total = max_queue_depth = 0u;\n'
    '    a35_tx_total = a35_rx_total = a35_missing_total = a35_wrong_total = 0u;\n'
    '    axis_synop_total = a35_updates_total = 0u;\n'
    '    step_idx = 0u;\n'
    '    steps_done = 0u;\n'
    '    start_tick_target = 0u;\n'
    '    start_broadcast_remaining = 0u;\n'
    '    run_start_tick = 0u;\n'
    '    last_timer_tick = 0u;\n'
    '    pending_payload = pending_step = 0u;\n'
    '    pending_send = false;\n'
    '    micro_pending = false;\n'
    '    axis_broadcast_pending = false;\n'
    '    control_due = false;\n'
    '    q_head = q_tail = 0u;\n'
    '    process_loop_running = false;\n'
    '    clear_seen();\n'
    '    ctrl_seen_mask = 0u;\n'
    '    for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '        ctrl_z_q24[a] = 0;\n'
    '        axis_state_q24_wide[a] = 0;\n'
    '        axis_acc_next_q44[a] = 0;\n'
    '    }\n'
    '\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        field_state_q24[l] = 0;\n'
    '        acc_next_q48[l] = 0;\n'
    '        local_event_count[l] = 0u;\n'
    '        for (uint32_t b = 0; b < N_BASIS; b++) {\n'
    '            v_q24[l][b] = dynrec[DYN_INIT_V0 + l*N_BASIS + b];\n'
    '            q_q24[l][b] = 0;\n'
    '            refractory[l][b] = 0u;\n'
    '            uint32_t sbase = DYN_SEED0 + 4u*(l*N_BASIS + b);\n'
    '            for (uint32_t s = 0; s < 4u; s++) {\n'
    '                rng_seed[l][b][s] = ((volatile uint32_t *)dynrec)[sbase + s];\n'
    '            }\n'
    '            validate_mars_kiss64_seed(rng_seed[l][b]);\n'
    '        }\n'
    '        for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '            ctrl_coeff_q24[l][a] = dynrec[DYN_CTRL0 + l*N_AXES + a];\n'
    '        }\n'
    '    }\n'
    '\n'
    '    for (uint32_t i = 0; i < OUT_WORDS; i++) outrec[i] = 0;\n'
    '    phase = PHASE_ARMED;\n'
    '    write_status(STATUS_ARMED);\n'
    '}\n'
    '\n'
    'static void finish_image(void) {\n'
    '    flush_output(error_flags ? STATUS_ERROR : STATUS_IMAGE_DONE);\n'
    '    phase = PHASE_WAIT_COMMAND;\n'
    '}\n'
    '\n'
    'static inline void consume_previous_events(void) {\n'
    '    uint32_t cspr = spin1_int_disable();\n'
    '    int64_t snap[PACK];\n'
    '    int64_t axis_snap[N_AXES];\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        snap[l] = acc_next_q48[l];\n'
    '        acc_next_q48[l] = 0;\n'
    '    }\n'
    '    for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '        axis_snap[a] = axis_acc_next_q44[a];\n'
    '        axis_acc_next_q44[a] = 0;\n'
    '    }\n'
    '    clear_seen();\n'
    '    spin1_mode_restore(cspr);\n'
    '\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        field_state_q24[l] = mul_q24_q30(field_state_q24[l], AQ_Q30);\n'
    '        int64_t delta = floor_rshift64(snap[l], Q_STATE);\n'
    '        int64_t f = (int64_t)field_state_q24[l] + delta;\n'
    '        /* True storage overflow of the frozen int32-Q24 field state.\n'
    '           No scientific clipping threshold is imposed here. */\n'
    '        if (f > 2147483647LL || f < -2147483648LL) {\n'
    '            error_flags |= ERR_NUMERIC_OVERFLOW;\n'
    '            return;\n'
    '        }\n'
    '        field_state_q24[l] = (int32_t)f;\n'
    '    }\n'
    '\n'
    '    if (group_id == 0u) {\n'
    '        for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '            /* axis_snap is Q20(axis) * Q24(event) = Q44, so shifting by\n'
    '               Q_AXIS returns a Q24 increment.  Keep z=A@x as a WIDE Q24\n'
    '               persistent state: unlike a neuron-local field, this 512-way\n'
    '               global projection is not naturally bounded to +/-128. */\n'
    '            int64_t decayed = mul_i64_q24_q30(axis_state_q24_wide[a], AQ_Q30);\n'
    '            int64_t delta = floor_rshift64(axis_snap[a], Q_AXIS);\n'
    '            axis_state_q24_wide[a] = decayed + delta;\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    '/* Begin one scientific millisecond.  This advances persistent synaptic/axis\n'
    ' * states but deliberately delays the microscopic LIF update until a later\n'
    ' * hardware subslot.  At 40-ms A3.5 boundaries, group 0 first broadcasts the\n'
    ' * three already-updated axis states to hidden target cores. */\n'
    'static void begin_scientific_step(uint32_t new_step) {\n'
    '    step_idx = new_step;\n'
    '    if (step_idx > 0u) consume_previous_events();\n'
    '    if (error_flags != ERR_NONE) return;\n'
    '\n'
    '    pending_payload = 0u;\n'
    '    pending_step = step_idx;\n'
    '    pending_send = false;\n'
    '    micro_pending = true;\n'
    '    control_due = (step_idx > 0u && control_steps > 0u &&\n'
    '                   (step_idx % control_steps) == 0u);\n'
    '    axis_broadcast_pending = (group_id == 0u && control_due);\n'
    '    if (logical_base >= NV && control_due) ctrl_seen_mask = 0u;\n'
    '}\n'
    '\n'
    'static inline int32_t feedback_q24_for_lane(uint32_t l) {\n'
    '    if (logical_base < NV) return 0;\n'
    '    int64_t fb = 0;\n'
    '    for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '        fb += floor_rshift64((int64_t)ctrl_coeff_q24[l][a] *\n'
    '                             (int64_t)ctrl_z_q24[a], Q_STATE);\n'
    '    }\n'
    '    return sat_i64_to_i32(fb);\n'
    '}\n'
    '\n'
    'static void run_micro_step_if_ready(uint32_t slot) {\n'
    '    if (!micro_pending || slot < MICRO_FIRST_SLOT) return;\n'
    '    if (logical_base >= NV && control_due && ctrl_seen_mask != ((1u << N_AXES) - 1u)) {\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    uint32_t payload = 0u;\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        int64_t total_field = (int64_t)field_state_q24[l] + logical_bias_q24[l];\n'
    '        if (logical_base >= NV) total_field += feedback_q24_for_lane(l);\n'
    '        /* True storage/interface overflow of the frozen int32-Q24 LIF drive. */\n'
    '        if (total_field > 2147483647LL || total_field < -2147483648LL) {\n'
    '            error_flags |= ERR_NUMERIC_OVERFLOW;\n'
    '            return;\n'
    '        }\n'
    '        uint8_t smask = micro_step(l, (int32_t)total_field);\n'
    '        if (smask != 0u) {\n'
    '            int32_t ev = logical_event_q24(smask);\n'
    '            if (ev != 0) {\n'
    '                payload |= ((uint32_t)smask) << (l * 7u);\n'
    '                event_generated_total++;\n'
    '                local_event_count[l]++;\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '    steps_done++;\n'
    '    micro_pending = false;\n'
    '    if (logical_base >= NV && control_due) a35_updates_total++;\n'
    '\n'
    '    pending_payload = payload;\n'
    '    pending_step = step_idx;\n'
    '    pending_send = (payload != 0u) && ((step_idx + 1u) < gen_steps);\n'
    '\n'
    '    /* Hidden packet affects its own hidden packed target too; inject locally now\n'
    '       for next-step field recurrence and ignore router loopback later. */\n'
    '    if (pending_send && logical_base >= NV) {\n'
    '        process_packed_payload(group_id, pending_payload);\n'
    '    }\n'
    '\n'
    '    if (steps_done >= gen_steps) {\n'
    '        pending_send = false;\n'
    '        finish_image();\n'
    '    }\n'
    '}\n'
    '\n'
    'static inline void broadcast_axis_if_due(uint32_t slot) {\n'
    '    if (!axis_broadcast_pending || slot < AXIS_SEND_SLOT || group_id != 0u) return;\n'
    '    for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '        uint32_t low = ((step_idx & STEP_MASK) << 11) | (AXIS_GROUP_BASE + a);\n'
    '        uint32_t key = key_vis | low;  /* hidden targets only */\n'
    '        /* The frozen dense golden computes z with an int64 accumulator and\n'
    '           clamps only the transient Q24 readout.  Match that semantics here:\n'
    '           the persistent wide state is never clipped, only this 40-ms payload. */\n'
    '        int32_t z_payload_q24 = sat_i64_to_i32(axis_state_q24_wide[a]);\n'
    '        if (!send_packet_retry(key, (uint32_t)z_payload_q24)) {\n'
    '            fail_and_wait(ERR_SEND);\n'
    '            return;\n'
    '        }\n'
    '        a35_tx_total++;\n'
    '    }\n'
    '    axis_broadcast_pending = false;\n'
    '}\n'
    '\n'
    'static void timer_callback(uint32_t tick, uint32_t unused) {\n'
    '    (void)unused;\n'
    '\n'
    '    if (last_timer_tick != 0u && tick > last_timer_tick + 1u) {\n'
    '        uint32_t gap = tick - last_timer_tick - 1u;\n'
    '        if (gap > timer_slip_max) timer_slip_max = gap;\n'
    '    }\n'
    '    last_timer_tick = tick;\n'
    '\n'
    '    if (phase == PHASE_WAIT_COMMAND) {\n'
    '        if (cfg[CFG_STOP_FLAG]) {\n'
    '            write_status(STATUS_APP_DONE);\n'
    '            phase = PHASE_FINISHED;\n'
    '            spin1_exit(0);\n'
    '            return;\n'
    '        }\n'
    '        uint32_t arm = cfg[CFG_ARM_EPOCH];\n'
    '        if (arm != 0u && arm != active_epoch) arm_image(arm);\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (phase == PHASE_ARMED) {\n'
    '        if (cfg[CFG_STOP_FLAG]) {\n'
    '            write_status(STATUS_APP_DONE);\n'
    '            phase = PHASE_FINISHED;\n'
    '            spin1_exit(0);\n'
    '            return;\n'
    '        }\n'
    '        if (group_id == 0u && cfg[CFG_GO_EPOCH] == active_epoch) {\n'
    '            if (start_tick_target == 0u) {\n'
    '                start_tick_target = tick + start_guard_ticks;\n'
    '                start_broadcast_remaining = 3u;\n'
    '            }\n'
    '            if (start_broadcast_remaining > 0u) {\n'
    '                uint32_t ctrl_key = key_all | CTRL_LOW;\n'
    '                if (!send_packet_retry(ctrl_key, start_tick_target)) {\n'
    '                    fail_and_wait(ERR_SEND); return;\n'
    '                }\n'
    '                start_broadcast_remaining--;\n'
    '            }\n'
    '        }\n'
    '        if (start_tick_target != 0u && ((int32_t)(tick - start_tick_target) >= 0)) {\n'
    '            phase = PHASE_RUNNING;\n'
    '            run_start_tick = start_tick_target;\n'
    '            step_idx = 0u;\n'
    '            rx_this_model = 0u;\n'
    '            begin_scientific_step(0u);\n'
    '            if (phase != PHASE_RUNNING) return;\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (phase != PHASE_RUNNING) return;\n'
    '\n'
    '    uint32_t rel = tick - run_start_tick;\n'
    '    uint32_t slot = rel % ticks_per_model;\n'
    '    uint32_t model_step = rel / ticks_per_model;\n'
    '\n'
    '    /* Do not rely on observing the exact slot-0 callback.  If a heavy boundary\n'
    '       callback delays one base tick, advance on the first callback belonging to\n'
    '       the new scientific millisecond. */\n'
    '    if (model_step > step_idx) {\n'
    '        if (model_step != step_idx + 1u || pending_send || micro_pending || axis_broadcast_pending) {\n'
    '            fail_and_wait(ERR_SEND);\n'
    '            return;\n'
    '        }\n'
    '        if (rx_this_model > max_rx_interval) max_rx_interval = rx_this_model;\n'
    '        rx_this_model = 0u;\n'
    '        if (model_step >= gen_steps) return;\n'
    '        begin_scientific_step(model_step);\n'
    '        if (phase != PHASE_RUNNING) return;\n'
    '    }\n'
    '\n'
    '    broadcast_axis_if_due(slot);\n'
    '    if (phase != PHASE_RUNNING) return;\n'
    '    run_micro_step_if_ready(slot);\n'
    '    if (phase != PHASE_RUNNING) return;\n'
    '\n'
    '    uint32_t drain_start = ticks_per_model - DRAIN_TICKS;\n'
    '    if (micro_pending && slot >= drain_start) {\n'
    '        if (logical_base >= NV && control_due && ctrl_seen_mask != ((1u << N_AXES) - 1u)) {\n'
    '            a35_missing_total++;\n'
    '        }\n'
    '        fail_and_wait(ERR_LATE_EVENT);\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (pending_send) {\n'
    '        uint32_t my_slot = SEND_FIRST_SLOT + (group_id % send_slots);\n'
    '        /* If one sub-tick callback is delayed, send at the next available slot\n'
    '           rather than silently losing the packet.  Never send in the guard. */\n'
    '        if (slot >= my_slot && slot < drain_start) {\n'
    '            uint32_t low = ((pending_step & STEP_MASK) << 11) | group_id;\n'
    '            uint32_t key = (logical_base < NV ? key_vis : key_all) | low;\n'
    '            if (!send_packet_retry(key, pending_payload)) {\n'
    '                fail_and_wait(ERR_SEND); return;\n'
    '            }\n'
    '            event_sent_total++;\n'
    '            pending_send = false;\n'
    '        }\n'
    '    }\n'
    '\n'
    '    if (error_flags != ERR_NONE) fail_and_wait(error_flags);\n'
    '}\n'
    '\n'
    'void c_main(void) {\n'
    '    cfg = (volatile uint32_t *)sark_tag_ptr(SDRAM_TAG, sark_app_id());\n'
    '    if (cfg == 0 || cfg[CFG_MAGIC] != MAGIC_CFG || cfg[CFG_VER] != CFG_VERSION ||\n'
    '            cfg[CFG_NLOGICAL] != N_LOGICAL || cfg[CFG_NV] != NV ||\n'
    '            cfg[CFG_NH] != NH || cfg[CFG_PACK] != PACK ||\n'
    '            cfg[CFG_NGROUPS] != N_GROUPS) {\n'
    '        io_printf(IO_BUF, "DYN_BAD_CONFIG p=%u\\n", spin1_get_core_id());\n'
    '        spin1_exit(1); return;\n'
    '    }\n'
    '\n'
    '    uint32_t p = spin1_get_core_id();\n'
    '    if (p >= MAX_CORES) { spin1_exit(0); return; }\n'
    '    group_id = cfg[CFG_GROUP_BY_P + p];\n'
    '    if (group_id >= N_GROUPS) { spin1_exit(0); return; }\n'
    '    logical_base = group_id * PACK;\n'
    '    input_count = (logical_base < NV) ? NH : N_LOGICAL;\n'
    '\n'
    '    key_vis = cfg[CFG_KEY_VIS];\n'
    '    key_all = cfg[CFG_KEY_ALL];\n'
    '    gen_steps = cfg[CFG_GEN_STEPS];\n'
    '    noise_sd_q24 = cfg[CFG_NOISE_SD_Q24];\n'
    '    control_steps = cfg[CFG_CONTROL_STEPS];\n'
    '    if (control_steps == 0u) control_steps = 40u;\n'
    '    base_tick_us = cfg[CFG_TIMER_TICK_US];\n'
    '    if (base_tick_us == 0u || (SCIENTIFIC_US % base_tick_us) != 0u) {\n'
    '        io_printf(IO_BUF, "DYN_BAD_TICK tick_us=%u\\n", base_tick_us);\n'
    '        spin1_exit(1); return;\n'
    '    }\n'
    '    ticks_per_model = SCIENTIFIC_US / base_tick_us;\n'
    '    /* Derived protocol requirement only: one compute phase, send slots, and\n'
    '       DRAIN_TICKS before the next scientific millisecond. */\n'
    '    if (ticks_per_model <= (DRAIN_TICKS + SEND_FIRST_SLOT + 1u)) {\n'
    '        io_printf(IO_BUF, "DYN_TICK_TOO_COARSE tick_us=%u ticks/model=%u\\n",\n'
    '                  base_tick_us, ticks_per_model);\n'
    '        spin1_exit(1); return;\n'
    '    }\n'
    '    send_slots = ticks_per_model - DRAIN_TICKS - SEND_FIRST_SLOT;\n'
    '    start_guard_ticks = (2000u + base_tick_us - 1u) / base_tick_us;\n'
    '\n'
    '    volatile int32_t *srow = (volatile int32_t *)((uint8_t *)cfg +\n'
    '        cfg[CFG_STATIC_OFFSET] + p * cfg[CFG_STATIC_STRIDE]);\n'
    '    for (uint32_t l = 0; l < PACK; l++) logical_bias_q24[l] = srow[l];\n'
    '    volatile int32_t *sw = &srow[PACK];\n'
    '    for (uint32_t l = 0; l < PACK; l++) {\n'
    '        for (uint32_t j = 0; j < input_count; j++) {\n'
    '            weights[l][j] = sw[l*input_count + j];\n'
    '        }\n'
    '    }\n'
    '    if (group_id == 0u) {\n'
    '        volatile int32_t *sa = &sw[PACK * input_count];\n'
    '        for (uint32_t a = 0; a < N_AXES; a++) {\n'
    '            for (uint32_t h = 0; h < NH; h++) axes_q20[a][h] = sa[a*NH + h];\n'
    '        }\n'
    '    }\n'
    '\n'
    '    dynrec = (volatile int32_t *)((uint8_t *)cfg + cfg[CFG_DYN_OFFSET] +\n'
    '                                  p * cfg[CFG_DYN_STRIDE]);\n'
    '    outrec = (volatile int32_t *)((uint8_t *)cfg + cfg[CFG_OUT_OFFSET] +\n'
    '                                  p * cfg[CFG_OUT_STRIDE]);\n'
    '\n'
    '    active_epoch = 0u;\n'
    '    phase = PHASE_WAIT_COMMAND;\n'
    '    for (uint32_t i = 0; i < OUT_WORDS; i++) outrec[i] = 0;\n'
    '    write_status(STATUS_IDLE);\n'
    '\n'
    '    if (group_id == 0u) {\n'
    '        io_printf(IO_BUF,\n'
    '            "DYN_G3_START groups=%u pack=%u steps=%u control=%u base_tick_us=%u ticks/model=%u send_slots=%u\\n",\n'
    '            N_GROUPS, PACK, gen_steps, control_steps, base_tick_us, ticks_per_model, send_slots);\n'
    '    }\n'
    '\n'
    '    spin1_set_timer_tick(base_tick_us);\n'
    '    spin1_callback_on(MCPL_PACKET_RECEIVED, packet_callback, -1);\n'
    '    spin1_callback_on(USER_EVENT, process_callback, 0);\n'
    '    spin1_callback_on(TIMER_TICK, timer_callback, 2);\n'
    '    spin1_start(SYNC_WAIT);\n'
    '}\n'
    "'''\n"
    '\n'
    '\n'
    '# =============================================================================\n'
    '# Build / SpiNNaker runtime helpers\n'
    '# =============================================================================\n'
    'def carrier_imports():\n'
    '    try:\n'
    '        import pyNN.spiNNaker as sim\n'
    '        from spinn_front_end_common.data import FecDataView\n'
    '        from spinn_machine import CoreSubsets\n'
    '        from spinnman.model.enums import CPUState\n'
    '        try:\n'
    '            from spinnman.messages.scp.enums.signal import Signal\n'
    '        except ImportError:\n'
    '            from spinnman.messages.scp.enums import Signal\n'
    '        try:\n'
    '            from spynnaker.pyNN.extra_algorithms.splitter_components import (\n'
    '                SplitterPopulationVertexFixed,\n'
    '            )\n'
    '        except ImportError:\n'
    '            SplitterPopulationVertexFixed = None\n'
    '    except Exception as exc:\n'
    '        raise RuntimeError("Run this file inside the same working Jupyter/sPyNNaker environment") from exc\n'
    '    return sim, FecDataView, SplitterPopulationVertexFixed, CoreSubsets, CPUState, Signal\n'
    '\n'
    '\n'
    'def _get_new_app_id(FecDataView):\n'
    '    try:\n'
    '        return FecDataView.get_new_id()\n'
    '    except TypeError:\n'
    '        return FecDataView().get_new_id()\n'
    '\n'
    '\n'
    'def _read_iobuf(txrx, cores) -> str:\n'
    '    rows = []\n'
    '    for b in txrx.get_iobuf(cores):\n'
    '        rows.append(str(getattr(b, "iobuf", b)))\n'
    '    return "\\n".join(rows)\n'
    '\n'
    '\n'
    'def find_build_tree(explicit: Path | None) -> Path:\n'
    '    if explicit is not None:\n'
    '        if explicit.exists():\n'
    '            return explicit\n'
    '        raise FileNotFoundError(explicit)\n'
    '    candidates = [\n'
    '        Path(os.environ.get("MSDL_GATE3_BUILD", "/home/jovyan/PGI/_gate3_build")),\n'
    '        Path.home() / "PGI" / "_gate3_build",\n'
    '    ]\n'
    '    for p in candidates:\n'
    '        if p.exists():\n'
    '            return p\n'
    '    raise FileNotFoundError(\n'
    '        "Custom APLX build tree not found. Set MSDL_GATE3_BUILD or pass --build-tree."\n'
    '    )\n'
    '\n'
    '\n'
    'def build_aplx(output: Path, build_tree: Path | None) -> tuple[Path, Path]:\n'
    '    bt = find_build_tree(build_tree)\n'
    '    srcdir = bt / "src"\n'
    '    builddir = bt / "build"\n'
    '    srcdir.mkdir(parents=True, exist_ok=True)\n'
    '    target = srcdir / "msdl_gate3.c"\n'
    '    backup = target.read_bytes() if target.exists() else None\n'
    '    generated_c = output / "spinnaker_dynamics_g3_clean_kernel.c"\n'
    '    generated_c.write_text(KERNEL_C, encoding="utf-8")\n'
    '    try:\n'
    '        target.write_text(KERNEL_C, encoding="utf-8")\n'
    '        if builddir.exists():\n'
    '            shutil.rmtree(builddir)\n'
    '        builddir.mkdir(parents=True, exist_ok=True)\n'
    '        env = os.environ.copy()\n'
    '        env.setdefault(\n'
    '            "FEC_INSTALL_DIR",\n'
    '            "/home/jovyan/sPyNNakerGit/SpiNNFrontEndCommon/c_common/front_end_common_lib",\n'
    '        )\n'
    '        env.setdefault("SPINN_COMMON_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/spinn_common")\n'
    '        env.setdefault("SPINN_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/spinnaker_tools")\n'
    '        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")\n'
    '        p = subprocess.run(\n'
    '            ["make"], cwd=bt, env=env, text=True,\n'
    '            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n'
    '        )\n'
    '        (output / "aplx_build.log").write_text(p.stdout, encoding="utf-8")\n'
    '        if p.returncode != 0:\n'
    '            raise RuntimeError(\n'
    '                "APLX build failed:\\n" + "\\n".join(p.stdout.splitlines()[-160:])\n'
    '            )\n'
    '        cand = list(bt.rglob("msdl_gate3.aplx"))\n'
    '        if not cand:\n'
    '            raise FileNotFoundError("make succeeded but msdl_gate3.aplx was not found")\n'
    '        built = max(cand, key=lambda x: x.stat().st_mtime)\n'
    '        dest = output / "spinnaker_dynamics_kernel.aplx"\n'
    '        shutil.copy2(built, dest)\n'
    '        return dest, generated_c\n'
    '    finally:\n'
    '        if backup is None:\n'
    '            try:\n'
    '                target.unlink()\n'
    '            except FileNotFoundError:\n'
    '                pass\n'
    '        else:\n'
    '            target.write_bytes(backup)\n'
    '\n'
    '\n'
    'class G3Board:\n'
    '    def __init__(\n'
    '        self,\n'
    '        cp: dict[str, np.ndarray],\n'
    '        axes: np.ndarray,\n'
    '        output: Path,\n'
    '        *,\n'
    '        generation_ms: int = 4320,\n'
    '        timer_tick_us: int = 100,\n'
    '        build_tree: Path | None = None,\n'
    '        timeout_s: float = 120.0,\n'
    '    ):\n'
    '        self.cp = cp\n'
    '        self.axes = np.asarray(axes, dtype=np.float64)\n'
    '        self.output = output\n'
    '        self.generation_ms = int(generation_ms)\n'
    '        self.timer_tick_us = int(timer_tick_us)\n'
    '        self.timeout_s = float(timeout_s)\n'
    '        self._closed = False\n'
    '        self._epoch = 0\n'
    '        self.sim = self.FecDataView = self.CoreSubsets = self.CPUState = self.Signal = None\n'
    '        self.txrx = None\n'
    '        self.custom_app_id = None\n'
    '        self.sim_started = False\n'
    '\n'
    '        if self.generation_ms <= 0:\n'
    '            raise ValueError("--generation-ms must be positive")\n'
    '        if self.timer_tick_us <= 0 or (1000 % self.timer_tick_us) != 0:\n'
    '            raise ValueError("--timer-tick-us must be a positive divisor of 1000 us")\n'
    '        if (1000 // self.timer_tick_us) < 5:\n'
    '            raise ValueError("need at least 5 hardware sub-ticks inside each 1-ms scientific step")\n'
    '\n'
    '        self.aplx, self.generated_c = build_aplx(output, build_tree)\n'
    '        self._setup_board()\n'
    '\n'
    '    def _setup_board(self):\n'
    '        t_all = time.perf_counter()\n'
    '        (sim, FecDataView, SplitterPopulationVertexFixed,\n'
    '         CoreSubsets, CPUState, Signal) = carrier_imports()\n'
    '        self.sim, self.FecDataView = sim, FecDataView\n'
    '        self.CoreSubsets, self.CPUState, self.Signal = CoreSubsets, CPUState, Signal\n'
    '\n'
    '        sim.setup(timestep=1.0, min_delay=1.0, time_scale_factor=1)\n'
    '        self.sim_started = True\n'
    '        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, PACK)\n'
    '        carrier_kwargs = {}\n'
    '        if SplitterPopulationVertexFixed is not None:\n'
    '            carrier_kwargs["additional_parameters"] = {\n'
    '                "splitter": SplitterPopulationVertexFixed()\n'
    '            }\n'
    '\n'
    '        carrier = sim.Population(\n'
    '            N,\n'
    '            sim.IF_curr_exp(\n'
    '                cm=1.0, tau_m=20.0, tau_refrac=1.0,\n'
    '                tau_syn_E=5.0, tau_syn_I=5.0,\n'
    '                v_rest=0.0, v_reset=0.0, v_thresh=1.0, i_offset=0.0,\n'
    '            ),\n'
    '            label="dynamics_pack4_placement_carrier",\n'
    '            **carrier_kwargs,\n'
    '        )\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        sim.run(1.0)\n'
    '        self.map_load_s = float(time.perf_counter() - t0)\n'
    '\n'
    '        mverts = list(carrier._vertex.machine_vertices)\n'
    '        if len(mverts) != N_GROUPS:\n'
    '            raise RuntimeError(\n'
    '                f"expected {N_GROUPS} four-atom machine vertices, got {len(mverts)}; "\n'
    '                "do not continue because packing is not the requested architecture"\n'
    '            )\n'
    '\n'
    '        placements = []\n'
    '        for mv in mverts:\n'
    '            sl = mv.vertex_slice\n'
    '            if int(sl.n_atoms) != PACK:\n'
    '                raise RuntimeError(f"non-{PACK}-atom machine slice: {sl}")\n'
    '            base = int(sl.lo_atom)\n'
    '            if base % PACK != 0:\n'
    '                raise RuntimeError(f"unaligned packed slice: {sl}")\n'
    '            gid = base // PACK\n'
    '            pl = FecDataView.get_placement_of_vertex(mv)\n'
    '            placements.append((gid, int(pl.x), int(pl.y), int(pl.p), base, mv))\n'
    '        placements.sort(key=lambda r: r[0])\n'
    '        if [r[0] for r in placements] != list(range(N_GROUPS)):\n'
    '            raise RuntimeError("packed group IDs are not exactly 0..323")\n'
    '        self.placements = placements\n'
    '\n'
    '        self.txrx = FecDataView.get_transceiver()\n'
    '        self.pynn_app_id = int(FecDataView.get_app_id())\n'
    '        self.txrx.stop_application(self.pynn_app_id)\n'
    '        time.sleep(0.25)\n'
    '        self.custom_app_id = int(_get_new_app_id(FecDataView))\n'
    '\n'
    '        by_chip = defaultdict(list)\n'
    '        worker_core = CoreSubsets()\n'
    '        for gid, x, y, p, base, _ in placements:\n'
    '            by_chip[(x, y)].append((gid, p, base))\n'
    '            worker_core.add_processor(x, y, p)\n'
    '        self.by_chip = dict(by_chip)\n'
    '        self.worker_core = worker_core\n'
    '\n'
    '        machine = FecDataView.get_machine()\n'
    '        t0 = time.perf_counter()\n'
    '        self.route_plan = build_selective_routes(machine, self.by_chip)\n'
    '        self.route_install = install_and_verify_routes(\n'
    '            self.txrx, self.route_plan, self.custom_app_id\n'
    '        )\n'
    '        self.route_install_s = float(time.perf_counter() - t0)\n'
    '\n'
    '        # Largest record remains within the fixed packed-core SDRAM stride; group 0 also stores 3x512 axes.\n'
    '        static_words_max = max(PACK + PACK * N, PACK + PACK * NH + N_AXES * NH)\n'
    '        self.static_stride = align_up(static_words_max * 4, 256)\n'
    '        self.static_offset = CFG_BYTES\n'
    '        self.dyn_offset = align_up(\n'
    '            self.static_offset + MAX_CORES * self.static_stride, 256\n'
    '        )\n'
    '        self.dyn_stride = align_up(DYN_BYTES, 256)\n'
    '        self.out_offset = align_up(\n'
    '            self.dyn_offset + MAX_CORES * self.dyn_stride, 256\n'
    '        )\n'
    '        self.out_stride = align_up(OUT_BYTES, 256)\n'
    '        self.region_size = align_up(\n'
    '            self.out_offset + MAX_CORES * self.out_stride, 256\n'
    '        )\n'
    '\n'
    '        self.bases: dict[tuple[int, int], int] = {}\n'
    '        t0 = time.perf_counter()\n'
    '        for (x, y), groups in sorted(self.by_chip.items()):\n'
    '            base_addr = int(self.txrx.malloc_sdram(\n'
    '                x, y, self.region_size, self.custom_app_id, tag=SDRAM_TAG\n'
    '            ))\n'
    '            self.bases[(x, y)] = base_addr\n'
    '            ridx = int(self.route_plan["root_index"][(x, y)])\n'
    '\n'
    '            hdr = np.zeros(CFG_WORDS, dtype="<u4")\n'
    '            hdr[0] = MAGIC_CFG\n'
    '            hdr[1] = CFG_VERSION\n'
    '            hdr[2] = N\n'
    '            hdr[3] = NV\n'
    '            hdr[4] = NH\n'
    '            hdr[5] = PACK\n'
    '            hdr[6] = N_GROUPS\n'
    '            hdr[7] = self.generation_ms\n'
    '            hdr[8] = np.uint32(route_group_key(ridx, False))\n'
    '            hdr[9] = np.uint32(route_group_key(ridx, True))\n'
    '            hdr[10] = self.static_offset\n'
    '            hdr[11] = self.static_stride\n'
    '            hdr[12] = self.dyn_offset\n'
    '            hdr[13] = self.dyn_stride\n'
    '            hdr[14] = self.timer_tick_us\n'
    '            hdr[15] = 0\n'
    '            hdr[16] = 0\n'
    '            hdr[17] = 0\n'
    '            hdr[18] = np.uint32(NOISE_SD_Q24)\n'
    '            hdr[19] = self.out_offset\n'
    '            hdr[20] = self.out_stride\n'
    '            hdr[21] = CONTROL_DT_MS\n'
    '            hdr[CFG_GROUP_BY_P:CFG_GROUP_BY_P + MAX_CORES] = np.uint32(0xFFFFFFFF)\n'
    '            for gid, p, _base in groups:\n'
    '                hdr[CFG_GROUP_BY_P + int(p)] = np.uint32(gid)\n'
    '\n'
    '            static_blob = bytearray(MAX_CORES * self.static_stride)\n'
    '            for gid, p, logical_base in groups:\n'
    '                rec = make_static_record(self.cp, self.axes, logical_base)\n'
    '                raw = rec.tobytes()\n'
    '                if len(raw) > self.static_stride:\n'
    '                    raise RuntimeError("static record exceeds stride")\n'
    '                off = int(p) * self.static_stride\n'
    '                static_blob[off:off + len(raw)] = raw\n'
    '\n'
    '            self.txrx.write_memory(x, y, base_addr, hdr.tobytes(), n_bytes=CFG_BYTES)\n'
    '            self.txrx.write_memory(\n'
    '                x, y, base_addr + self.static_offset,\n'
    '                bytes(static_blob), n_bytes=len(static_blob),\n'
    '            )\n'
    '            self.txrx.write_memory(\n'
    '                x, y, base_addr + self.dyn_offset,\n'
    '                bytes(MAX_CORES * self.dyn_stride),\n'
    '                n_bytes=MAX_CORES * self.dyn_stride,\n'
    '            )\n'
    '            self.txrx.write_memory(\n'
    '                x, y, base_addr + self.out_offset,\n'
    '                bytes(MAX_CORES * self.out_stride),\n'
    '                n_bytes=MAX_CORES * self.out_stride,\n'
    '            )\n'
    '        self.static_upload_s = float(time.perf_counter() - t0)\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        self.txrx.execute_flood(\n'
    '            self.worker_core, str(self.aplx), self.custom_app_id, wait=False\n'
    '        )\n'
    '        time.sleep(0.5)\n'
    '        self.txrx.send_signal(self.custom_app_id, Signal.SYNC0)\n'
    '        self.launch_s = float(time.perf_counter() - t0)\n'
    '\n'
    '        self._wait_group0({STATUS_IDLE}, epoch=0, timeout=30.0)\n'
    '        self.setup_total_s = float(time.perf_counter() - t_all)\n'
    '\n'
    '    def _output_addr(self, x: int, y: int, p: int) -> int:\n'
    '        return self.bases[(x, y)] + self.out_offset + int(p) * self.out_stride\n'
    '\n'
    '    @staticmethod\n'
    '    def _parse_out(data: bytes) -> dict[str, Any]:\n'
    '        a = np.frombuffer(data, dtype="<i4", count=OUT_WORDS).copy()\n'
    '        return {\n'
    '            "magic": int(np.uint32(a[OUT_MAGIC])),\n'
    '            "group": int(a[OUT_GROUP]),\n'
    '            "status": int(np.uint32(a[OUT_STATUS])),\n'
    '            "error": int(np.uint32(a[OUT_ERROR])),\n'
    '            "epoch": int(np.uint32(a[OUT_EPOCH])),\n'
    '            "steps": int(a[OUT_STEPS]),\n'
    '            "send_fail": int(np.uint32(a[OUT_SEND_FAIL])),\n'
    '            "duplicate": int(np.uint32(a[OUT_DUPLICATE])),\n'
    '            "wrong_route": int(np.uint32(a[OUT_WRONG_ROUTE])),\n'
    '            "late_event": int(np.uint32(a[OUT_LATE_EVENT])),\n'
    '            "event_generated": int(np.uint32(a[OUT_EVENT_GENERATED])),\n'
    '            "event_sent": int(np.uint32(a[OUT_EVENT_SENT])),\n'
    '            "event_rx": int(np.uint32(a[OUT_EVENT_RX])),\n'
    '            "event_processed": int(np.uint32(a[OUT_EVENT_PROCESSED])),\n'
    '            "loopback_ignored": int(np.uint32(a[OUT_LOOPBACK_IGNORED])),\n'
    '            "synops": int(np.uint32(a[OUT_SYNOP])),\n'
    '            "max_rx_interval": int(np.uint32(a[OUT_MAX_RX_INTERVAL])),\n'
    '            "timer_slip_max": int(np.uint32(a[OUT_TIMER_SLIP_MAX])),\n'
    '            "queue_overflow": int(np.uint32(a[OUT_QUEUE_OVERFLOW])),\n'
    '            "max_queue": int(np.uint32(a[OUT_MAX_QUEUE])),\n'
    '            "final_x_q24": a[OUT_FINAL_X0:OUT_FINAL_X0 + PACK].astype(np.int64),\n'
    '            "local_events": a[OUT_LOCAL_EVENTS0:OUT_LOCAL_EVENTS0 + PACK].astype(np.int64),\n'
    '            "a35_tx": int(np.uint32(a[OUT_A35_TX])),\n'
    '            "a35_rx": int(np.uint32(a[OUT_A35_RX])),\n'
    '            "a35_missing": int(np.uint32(a[OUT_A35_MISSING])),\n'
    '            "a35_wrong": int(np.uint32(a[OUT_A35_WRONG])),\n'
    '            "axis_synops": int(np.uint32(a[OUT_AXIS_SYNOP])),\n'
    '            "a35_updates": int(np.uint32(a[OUT_A35_UPDATES])),\n'
    '            "axis_final_q24": a[OUT_AXIS_FINAL0:OUT_AXIS_FINAL0 + N_AXES].astype(np.int64),\n'
    '        }\n'
    '\n'
    '    def _read_group(self, gid: int) -> dict[str, Any]:\n'
    '        gid, x, y, p, _base, _mv = self.placements[int(gid)]\n'
    '        data = bytes(self.txrx.read_memory(\n'
    '            x, y, self._output_addr(x, y, p), OUT_BYTES\n'
    '        ))\n'
    '        r = self._parse_out(data)\n'
    '        if r["magic"] not in (0, MAGIC_OUT):\n'
    '            raise RuntimeError(f"group {gid} bad output magic 0x{r[\'magic\']:08x}")\n'
    '        return r\n'
    '\n'
    '    def _read_all_outputs(self) -> dict[int, dict[str, Any]]:\n'
    '        out: dict[int, dict[str, Any]] = {}\n'
    '        for (x, y), groups in sorted(self.by_chip.items()):\n'
    '            base = self.bases[(x, y)] + self.out_offset\n'
    '            blob = bytes(self.txrx.read_memory(\n'
    '                x, y, base, MAX_CORES * self.out_stride\n'
    '            ))\n'
    '            for gid, p, _logical_base in groups:\n'
    '                off = int(p) * self.out_stride\n'
    '                r = self._parse_out(blob[off:off + OUT_BYTES])\n'
    '                if r["magic"] != MAGIC_OUT:\n'
    '                    raise RuntimeError(\n'
    '                        f"group {gid}: bad output magic 0x{r[\'magic\']:08x}"\n'
    '                    )\n'
    '                if r["group"] != gid:\n'
    '                    raise RuntimeError(f"group mismatch expected {gid} got {r[\'group\']}")\n'
    '                out[gid] = r\n'
    '        return out\n'
    '\n'
    '    def _save_error_snapshot(self, tag: str):\n'
    '        try:\n'
    '            rows = self._read_all_outputs()\n'
    '            simple = {}\n'
    '            for gid, r in rows.items():\n'
    '                simple[str(gid)] = {\n'
    '                    k: (v.tolist() if isinstance(v, np.ndarray) else v)\n'
    '                    for k, v in r.items()\n'
    '                }\n'
    '            json_dump(self.output / f"ERROR_SNAPSHOT_{tag}.json", simple)\n'
    '        except Exception as exc:\n'
    '            (self.output / f"ERROR_SNAPSHOT_{tag}.txt").write_text(str(exc))\n'
    '\n'
    '    def _wait_group0(self, allowed: set[int], epoch: int, timeout: float):\n'
    '        t0 = time.perf_counter()\n'
    '        while True:\n'
    '            r = self._read_group(0)\n'
    '            if r["status"] == STATUS_ERROR:\n'
    '                self._save_error_snapshot(f"epoch{epoch}")\n'
    '                raise RuntimeError(f"group0 error: {r}")\n'
    '            if r["status"] in allowed and r["epoch"] == int(epoch):\n'
    '                return r\n'
    '            if time.perf_counter() - t0 > timeout:\n'
    '                self._save_error_snapshot(f"timeout_epoch{epoch}")\n'
    '                raise TimeoutError(f"group0 wait timeout; last={r}")\n'
    '            time.sleep(0.01)\n'
    '\n'
    '    def _verify_all(self, status: int, epoch: int, timeout: float = 15.0):\n'
    '        t0 = time.perf_counter()\n'
    '        last = None\n'
    '        while True:\n'
    '            rows = self._read_all_outputs()\n'
    '            bad = [\n'
    '                r for r in rows.values()\n'
    '                if r["status"] != status or r["epoch"] != epoch or r["error"] != 0\n'
    '            ]\n'
    '            if not bad:\n'
    '                return rows\n'
    '            if any(r["status"] == STATUS_ERROR or r["error"] != 0 for r in bad):\n'
    '                self._save_error_snapshot(f"verify_epoch{epoch}")\n'
    '                raise RuntimeError(f"packed-core errors, first={bad[:4]}")\n'
    '            last = bad[:4]\n'
    '            if time.perf_counter() - t0 > timeout:\n'
    '                self._save_error_snapshot(f"verify_timeout_epoch{epoch}")\n'
    '                raise TimeoutError(f"not all groups reached status {status}, first={last}")\n'
    '            time.sleep(0.02)\n'
    '\n'
    '    def _write_word_all_chips(self, word_index: int, value: int):\n'
    '        payload = struct.pack("<I", int(value) & 0xFFFFFFFF)\n'
    '        for (x, y), base in self.bases.items():\n'
    '            self.txrx.write_memory(\n'
    '                x, y, base + int(word_index) * 4, payload, n_bytes=4\n'
    '            )\n'
    '\n'
    '    def _write_word_group0_chip(self, word_index: int, value: int):\n'
    '        _gid, x, y, _p, _base, _mv = self.placements[0]\n'
    '        payload = struct.pack("<I", int(value) & 0xFFFFFFFF)\n'
    '        base_addr = self.bases[(int(x), int(y))]\n'
    '        self.txrx.write_memory(\n'
    '            int(x), int(y), base_addr + int(word_index) * 4, payload, n_bytes=4\n'
    '        )\n'
    '\n'
    '    def _make_dynamic_blobs(self, seed: int, command: np.ndarray, gamma: float):\n'
    '        init_q = qround_i32(initial_membranes(seed), Q_STATE)\n'
    '        ctrl_q = command_coefficients(command, self.axes, gamma)\n'
    '        blobs: dict[tuple[int, int], bytearray] = {\n'
    '            xy: bytearray(MAX_CORES * self.dyn_stride) for xy in self.by_chip\n'
    '        }\n'
    '        for gid, x, y, p, logical_base, _ in self.placements:\n'
    '            words = np.zeros(DYN_WORDS, dtype="<u4")\n'
    '            local_ids = np.arange(logical_base, logical_base + PACK, dtype=int)\n'
    '            vloc = init_q[local_ids, :].reshape(-1).astype(np.int32, copy=False)\n'
    '            words[DYN_INIT_V0:DYN_INIT_V0 + PACK*N_BASIS] = vloc.view(np.uint32)\n'
    '            for l, logical_id in enumerate(local_ids):\n'
    '                for b in range(N_BASIS):\n'
    '                    micro_id = b * N + int(logical_id)\n'
    '                    ss = mars_kiss_seed_tuple(seed, micro_id)\n'
    '                    pos = DYN_SEED0 + 4 * (l*N_BASIS + b)\n'
    '                    words[pos:pos + 4] = np.asarray(ss, dtype=np.uint32)\n'
    '            if logical_base >= NV:\n'
    '                for l, logical_id in enumerate(local_ids):\n'
    '                    h = int(logical_id - NV)\n'
    '                    for a in range(N_AXES):\n'
    '                        words[DYN_CTRL0 + l*N_AXES + a] = np.uint32(np.int32(ctrl_q[a, h]))\n'
    '            raw = words.tobytes()\n'
    '            off = int(p) * self.dyn_stride\n'
    '            blobs[(x, y)][off:off + len(raw)] = raw\n'
    '        return blobs\n'
    '\n'
    '    def run_image(self, seed: int, command: np.ndarray, gamma: float = GAMMA):\n'
    '        self._epoch += 1\n'
    '        epoch = self._epoch\n'
    '        timing: dict[str, float] = {}\n'
    '        t_total = time.perf_counter()\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        blobs = self._make_dynamic_blobs(seed, np.asarray(command, float), float(gamma))\n'
    '        timing["host_prepare_dynamic_s"] = float(time.perf_counter() - t0)\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        for (x, y), blob in blobs.items():\n'
    '            self.txrx.write_memory(\n'
    '                x, y, self.bases[(x, y)] + self.dyn_offset,\n'
    '                bytes(blob), n_bytes=len(blob),\n'
    '            )\n'
    '        timing["dynamic_upload_s"] = float(time.perf_counter() - t0)\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        self._write_word_all_chips(CFG_ARM_EPOCH, epoch)\n'
    '        self._wait_group0({STATUS_ARMED}, epoch, timeout=30.0)\n'
    '        self._verify_all(STATUS_ARMED, epoch, timeout=10.0)\n'
    '        timing["arm_s"] = float(time.perf_counter() - t0)\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        self._write_word_group0_chip(CFG_GO_EPOCH, epoch)\n'
    '        self._wait_group0({STATUS_IMAGE_DONE}, epoch, timeout=self.timeout_s)\n'
    '        time.sleep(0.02)\n'
    '        rows = self._verify_all(STATUS_IMAGE_DONE, epoch, timeout=15.0)\n'
    '        timing["board_generation_s"] = float(time.perf_counter() - t0)\n'
    '\n'
    '        t0 = time.perf_counter()\n'
    '        x = np.empty(N, dtype=np.float64)\n'
    '        local_events = np.zeros(N, dtype=np.int64)\n'
    '        for gid in range(N_GROUPS):\n'
    '            r = rows[gid]\n'
    '            base = gid * PACK\n'
    '            x[base:base + PACK] = r["final_x_q24"] / float(1 << Q_STATE)\n'
    '            local_events[base:base + PACK] = r["local_events"]\n'
    '        timing["decode_s"] = float(time.perf_counter() - t0)\n'
    '        timing["total_image_s"] = float(time.perf_counter() - t_total)\n'
    '\n'
    '        transport = {\n'
    '            "packed_cores": N_GROUPS,\n'
    '            "worker_chips": len(self.by_chip),\n'
    '            "send_fail_total": int(sum(r["send_fail"] for r in rows.values())),\n'
    '            "duplicate_total": int(sum(r["duplicate"] for r in rows.values())),\n'
    '            "wrong_route_total": int(sum(r["wrong_route"] for r in rows.values())),\n'
    '            "late_event_total": int(sum(r["late_event"] for r in rows.values())),\n'
    '            "source_events_generated": int(sum(r["event_generated"] for r in rows.values())),\n'
    '            "source_packets_sent": int(sum(r["event_sent"] for r in rows.values())),\n'
    '            "destination_packet_deliveries": int(sum(r["event_rx"] for r in rows.values())),\n'
    '            "destination_packets_processed": int(sum(r["event_processed"] for r in rows.values())),\n'
    '            "loopback_packets_ignored": int(sum(r["loopback_ignored"] for r in rows.values())),\n'
    '            "event_synops": int(sum(r["synops"] for r in rows.values())),\n'
    '            "max_rx_packets_between_local_ticks": int(max(r["max_rx_interval"] for r in rows.values())),\n'
    '            "max_timer_slip_ticks": int(max(r["timer_slip_max"] for r in rows.values())),\n'
    '            "queue_overflow_total": int(sum(r["queue_overflow"] for r in rows.values())),\n'
    '            "max_event_queue_depth": int(max(r["max_queue"] for r in rows.values())),\n'
    '            "workers_complete": int(sum(r["steps"] == self.generation_ms for r in rows.values())),\n'
    '            "mean_source_packets_per_model_ms": float(\n'
    '                sum(r["event_sent"] for r in rows.values()) / max(1, self.generation_ms - 1)\n'
    '            ),\n'
    '            "logical_events_generated": int(sum(r["event_generated"] for r in rows.values())),\n'
    '            "dense_v1_source_packets_equivalent": int(N * self.generation_ms),\n'
    '            "a35_control_packets_sent": int(sum(r["a35_tx"] for r in rows.values())),\n'
    '            "a35_control_deliveries": int(sum(r["a35_rx"] for r in rows.values())),\n'
    '            "a35_missing_total": int(sum(r["a35_missing"] for r in rows.values())),\n'
    '            "a35_wrong_total": int(sum(r["a35_wrong"] for r in rows.values())),\n'
    '            "axis_synops": int(sum(r["axis_synops"] for r in rows.values())),\n'
    '            "a35_updates_total": int(sum(r["a35_updates"] for r in rows.values())),\n'
    '            "axis_final": (rows[0]["axis_final_q24"] / float(1 << Q_STATE)).tolist(),\n'
    '        }\n'
    '        sent = transport["source_packets_sent"]\n'
    '        transport["source_packet_reduction_vs_dense_v1"] = (\n'
    '            float(N * self.generation_ms / sent) if sent else float("inf")\n'
    '        )\n'
    '        return x[:NV].copy(), x[NV:].copy(), local_events, timing, transport\n'
    '\n'
    '    def close(self):\n'
    '        if self._closed:\n'
    '            return\n'
    '        self._closed = True\n'
    '        try:\n'
    '            if self.txrx is not None and self.custom_app_id is not None:\n'
    '                try:\n'
    '                    self._write_word_all_chips(CFG_STOP_FLAG, 1)\n'
    '                    self.txrx.wait_for_cores_to_be_in_state(\n'
    '                        self.worker_core, self.custom_app_id,\n'
    '                        self.CPUState.FINISHED, timeout=30.0,\n'
    '                    )\n'
    '                except Exception:\n'
    '                    try:\n'
    '                        self.txrx.stop_application(self.custom_app_id)\n'
    '                    except Exception:\n'
    '                        pass\n'
    '                try:\n'
    '                    g0 = self.placements[0]\n'
    '                    cs = self.CoreSubsets()\n'
    '                    cs.add_processor(g0[1], g0[2], g0[3])\n'
    '                    txt = _read_iobuf(self.txrx, cs)\n'
    '                    (self.output / "group0_iobuf.txt").write_text(\n'
    '                        txt, encoding="utf-8", errors="replace"\n'
    '                    )\n'
    '                except Exception:\n'
    '                    pass\n'
    '        finally:\n'
    '            if self.sim_started:\n'
    '                try:\n'
    '                    self.sim.end()\n'
    '                except Exception:\n'
    '                    pass\n'
    '\n'
    '\n'
    '# =============================================================================\n'
    '# Reporting\n'
    '# =============================================================================\n'
    'def save_montage(samples: np.ndarray, path: Path):\n'
    '    try:\n'
    '        import matplotlib.pyplot as plt\n'
    '    except Exception:\n'
    '        return\n'
    '    n = len(samples)\n'
    '    if n == 0:\n'
    '        return\n'
    '    cols = min(5, n)\n'
    '    rows = int(math.ceil(n / cols))\n'
    '    fig, axs = plt.subplots(rows, cols, figsize=(2.2*cols, 2.2*rows), squeeze=False)\n'
    '    vmax = float(np.quantile(np.abs(samples), 0.98)) + 1e-12\n'
    '    for j, ax in enumerate(axs.flat):\n'
    '        ax.axis("off")\n'
    '        if j < n:\n'
    '            ax.imshow(samples[j].reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)\n'
    '            ax.set_title(str(j))\n'
    '    fig.tight_layout()\n'
    '    fig.savefig(path, dpi=160)\n'
    '    plt.close(fig)\n'
    '\n'
    '\n'
    'TRAIN_KERNEL_C = \'/*\\n * Phase-I V4 custom SpiNNaker-1 training kernel for Physicodynamic MSDL.\\n *\\n * CURRENT SCIENTIFIC MODEL ONLY:\\n *   - current full_pipeline.py A1 centered Izhikevich dynamics\\n *   - A4 target - prediction comparator current\\n *   - A2 local W/H update\\n *   - W/H start from exactly zero\\n *   - path-start weights are frozen for all 27 force samples of a path\\n *\\n * Training trajectories are immutable read-only input data placed in SDRAM by\\n * the host orchestrator.  Learning arithmetic is performed on the SpiNNaker\\n * cores.  Every logical coordinate is one worker core.\\n *\\n * IMPORTANT: The validated current Izh regime is subthreshold.  This kernel\\n * intentionally fails if the centered state leaves a conservative range,\\n * rather than silently implement a different spike/reset approximation.\\n */\\n#include <spin1_api.h>\\n#include <sark.h>\\n#include <stdint.h>\\n#include <stdbool.h>\\n\\n#define MAGIC_CFG       0x50315431u  /* P1T1 */\\n#define MAGIC_OUT       0x50314f31u  /* P1O1 */\\n#define CFG_VERSION     1u\\n#define SDRAM_TAG       1u\\n\\n#define N_WORKERS       1296u\\n#define NV              784u\\n#define NH              512u\\n#define N_MODES         3u\\n#define PATH_STATES     28u\\n#define FORCE_STEPS     27u\\n#define IZH_SUBSTEPS    80u\\n#define MAX_CORES       18u\\n#define SEEN_WORDS      41u\\n#define BCAST_SLOTS_DFLT 64u\\n#define WAIT_TIMEOUT_TICKS 1000u\\n\\n#define Q_PATH          20\\n#define Q_WEIGHT        24\\n#define Q_STATE         30\\n#define Q_READ          16\\n\\n/* Frozen integer mapping generated by phase1_fixedpoint_oracle.py */\\n#define IZH_Q_Q30       ((int32_t)-6461475)\\n#define IZH_A_Q30       ((int32_t)1073742)\\n#define IZH_B_Q30       ((int32_t)225822)\\n#define C004_Q30        ((int32_t)42949673)\\n#define GAIN_E_Q30      ((int32_t)10737)\\n#define GAIN_M_Q30      ((int32_t)1074)\\n#define CV_E_Q16        ((int32_t)29461671)\\n#define CUA_E_Q16       ((int32_t)53987929)\\n#define CV_M_Q16        ((int32_t)294616714)\\n#define CUA_M_Q16       ((int32_t)539879288)\\n#define INV_MU_DT_Q20   ((int32_t)10708440)\\n#define ALPHA015_Q20    ((int32_t)157286)\\n#define LR_Q24          ((int32_t)167772)\\n#define H_LR_Q24        ((int32_t)25166)\\n\\n/* Config word layout in tagged SDRAM. */\\n#define CFG_MAGIC               0u\\n#define CFG_VER                 1u\\n#define CFG_NWORKERS            2u\\n#define CFG_NV                  3u\\n#define CFG_NH                  4u\\n#define CFG_NCYCLES             5u\\n#define CFG_PATH_STATES         6u\\n#define CFG_FORCE_STEPS         7u\\n#define CFG_BASE_KEY            8u\\n#define CFG_PATH_OFFSET         9u\\n#define CFG_SCHED_OFFSET       10u\\n#define CFG_OUTPUT_OFFSET      11u\\n#define CFG_OUTPUT_STRIDE      12u\\n#define CFG_BCAST_SLOTS        13u\\n#define CFG_RESERVED14         14u\\n#define CFG_RESERVED15         15u\\n#define CFG_WORKER_BY_P        16u  /* 18 words: 16..33 */\\n#define CFG_WORDS              64u\\n#define CFG_BYTES              (CFG_WORDS * 4u)\\n\\n/* Output record layout; weights start at OUT_W0. */\\n#define OUT_MAGIC               0u\\n#define OUT_WORKER              1u\\n#define OUT_STATUS              2u\\n#define OUT_ERROR               3u\\n#define OUT_CYCLES_DONE         4u\\n#define OUT_MISSING_E           5u\\n#define OUT_MISSING_M           6u\\n#define OUT_SEND_FAIL           7u\\n#define OUT_EREAD0              8u\\n#define OUT_EREAD1              9u\\n#define OUT_EREAD2             10u\\n#define OUT_EARLY_RX           11u  /* current-step packets accepted during INTEGRATE */\\n#define OUT_WRONG_STEP         12u  /* packets rejected because step mismatched */\\n#define OUT_WRONG_PHASE        13u  /* packets rejected only after FINISHED */\\n#define OUT_PHASE_AT_EXIT      14u\\n#define OUT_FORCE_AT_EXIT      15u\\n#define OUT_W0                 16u\\n#define OUT_WORDS              (OUT_W0 + N_WORKERS)\\n#define OUT_BYTES              (OUT_WORDS * 4u)\\n\\n#define STATUS_INIT             1u\\n#define STATUS_RUNNING          2u\\n#define STATUS_DONE             3u\\n#define STATUS_ERROR            0xdead0001u\\n\\n#define ERR_NONE                0u\\n#define ERR_CONFIG              (1u << 0)\\n#define ERR_WORKER              (1u << 1)\\n#define ERR_STATE_RANGE         (1u << 2)\\n#define ERR_PACKET_TIMEOUT      (1u << 3)\\n#define ERR_SEND                (1u << 4)\\n#define ERR_VALUE_RANGE         (1u << 5)\\n#define ERR_OUTPUT              (1u << 6)\\n\\n#define PHASE_INTEGRATE         0u\\n#define PHASE_BROADCAST         1u\\n#define PHASE_WAIT              2u\\n#define PHASE_FINISHED          3u\\n\\n/* DTCM-resident per-worker state. */\\nstatic int32_t weights[N_WORKERS];       /* Q24; sparse by worker type */\\nstatic int64_t accum[N_WORKERS];         /* Q20 path accumulator */\\nstatic int32_t e_all[N_WORKERS];         /* Q20 */\\nstatic int32_t m_all[N_WORKERS];         /* Q20 */\\nstatic uint32_t seen_e[SEEN_WORDS];\\nstatic uint32_t seen_m[SEEN_WORDS];\\n\\nstatic int32_t e_dv[N_MODES], e_z[N_MODES];  /* Q30 */\\nstatic int32_t m_dv[N_MODES], m_z[N_MODES];  /* Q30 */\\n\\nstatic volatile uint32_t *cfg = 0;\\nstatic volatile int32_t *paths = 0;\\nstatic volatile uint32_t *schedule = 0;\\nstatic volatile int32_t *outrec = 0;\\n\\nstatic uint32_t worker_id = 0xffffffffu;\\nstatic uint32_t route_prefix = 0;\\nstatic uint32_t n_cycles = 0;\\nstatic uint32_t bcast_slots = BCAST_SLOTS_DFLT;\\n\\nstatic uint32_t cycle_idx = 0;\\nstatic uint32_t force_idx = 0;\\nstatic uint32_t substep_idx = 0;\\nstatic volatile uint32_t phase = PHASE_INTEGRATE;\\nstatic uint32_t slot_idx = 0;\\nstatic uint32_t wait_ticks = 0;\\nstatic volatile uint32_t count_e = 0;\\nstatic volatile uint32_t count_m = 0;\\nstatic uint32_t send_fail = 0;\\nstatic uint32_t error_flags = ERR_NONE;\\nstatic volatile uint32_t early_rx = 0;\\nstatic volatile uint32_t wrong_step = 0;\\nstatic volatile uint32_t wrong_phase = 0;\\nstatic uint32_t exit_phase_snapshot = PHASE_INTEGRATE;\\nstatic uint32_t exit_force_snapshot = 0;\\n\\nstatic int32_t input_e = 0;   /* Q20 */\\nstatic int32_t input_m = 0;   /* Q20 */\\nstatic int32_t e_self = 0;    /* Q20 */\\nstatic int32_t m_self = 0;    /* Q20 */\\n\\n/* ------------------------------------------------------------------------- */\\n/* Exact floor arithmetic matching Python\\\'s signed >> and // positive divisor */\\n/* ------------------------------------------------------------------------- */\\nstatic inline int64_t floor_rshift64(int64_t x, uint32_t q) {\\n    if (x >= 0) return x >> q;\\n    int64_t a = -x;\\n    int64_t d = ((int64_t)1 << q);\\n    return -((a + d - 1) >> q);\\n}\\n\\nstatic inline int64_t floor_div_pos64(int64_t x, int64_t d) {\\n    if (x >= 0) return x / d;\\n    int64_t a = -x;\\n    return -((a + d - 1) / d);\\n}\\n\\nstatic inline int32_t mul_shift32(int32_t a, int32_t b, uint32_t q) {\\n    return (int32_t) floor_rshift64((int64_t)a * (int64_t)b, q);\\n}\\n\\nstatic inline int32_t clamp_i64_to_i32(int64_t x) {\\n    if (x > 2147483647LL) return 2147483647;\\n    if (x < -2147483648LL) return (int32_t)0x80000000u;\\n    return (int32_t)x;\\n}\\n\\n/* ------------------------------------------------------------------------- */\\n/* Q20 path algebra                                                           */\\n/* ------------------------------------------------------------------------- */\\nstatic inline int32_t grad_q20(int32_t x) {\\n    int32_t x2 = mul_shift32(x, x, Q_PATH);\\n    int32_t x3 = mul_shift32(x2, x, Q_PATH);\\n    return (int32_t)((int64_t)2 * x + (int64_t)4 * x3);\\n}\\n\\nstatic inline int32_t q20_product(int32_t a, int32_t b) {\\n    return mul_shift32(a, b, Q_PATH);\\n}\\n\\n/* ------------------------------------------------------------------------- */\\n/* Centered current Izh model: one of the 80 RK2 substeps                     */\\n/* ------------------------------------------------------------------------- */\\nstatic inline bool izh_one(int32_t *dv, int32_t *z, int32_t inp, int32_t gain) {\\n    int32_t x = *dv;\\n    int32_t y = *z;\\n    int32_t x2 = mul_shift32(x, x, Q_STATE);\\n    int32_t k1v =\\n        mul_shift32(IZH_Q_Q30, x, Q_STATE) +\\n        mul_shift32(C004_Q30, x2, Q_STATE) -\\n        mul_shift32(IZH_A_Q30, y, Q_STATE) +\\n        mul_shift32(gain, inp, Q_PATH);\\n    int32_t k1z =\\n        mul_shift32(IZH_B_Q30, x, Q_STATE) -\\n        mul_shift32(IZH_A_Q30, y, Q_STATE);\\n\\n    int32_t xm = x + (int32_t)floor_rshift64(k1v, 1);\\n    int32_t ym = y + (int32_t)floor_rshift64(k1z, 1);\\n    int32_t xm2 = mul_shift32(xm, xm, Q_STATE);\\n    int32_t k2v =\\n        mul_shift32(IZH_Q_Q30, xm, Q_STATE) +\\n        mul_shift32(C004_Q30, xm2, Q_STATE) -\\n        mul_shift32(IZH_A_Q30, ym, Q_STATE) +\\n        mul_shift32(gain, inp, Q_PATH);\\n    int32_t k2z =\\n        mul_shift32(IZH_B_Q30, xm, Q_STATE) -\\n        mul_shift32(IZH_A_Q30, ym, Q_STATE);\\n\\n    x += k2v;\\n    y += k2z;\\n    *dv = x;\\n    *z = y;\\n\\n    /* Canonical model is explicitly subthreshold. 0.5 in Q30 is already\\n       hundreds of times larger than the validated centered trajectory. */\\n    if (x > (1 << 29) || x < -(1 << 29) ||\\n        y > (1 << 29) || y < -(1 << 29)) {\\n        return false;\\n    }\\n    return true;\\n}\\n\\nstatic inline int32_t readout_q20(int32_t dv, int32_t z,\\n                                  int32_t cv_q16, int32_t cua_q16) {\\n    int64_t a = floor_rshift64((int64_t)cv_q16 * dv, 26);\\n    int64_t b = floor_rshift64((int64_t)cua_q16 * z, 26);\\n    return clamp_i64_to_i32(a + b);\\n}\\n\\n/* ------------------------------------------------------------------------- */\\n/* Packet payload: type[31], force-step[30:26], signed-Q20 value[25:0].       */\\n/* Q20 range is +/-32, far outside the canonical E/M range.                  */\\n/* ------------------------------------------------------------------------- */\\nstatic inline bool value_fits_payload(int32_t v) {\\n    return (v >= -(1 << 25) && v < (1 << 25));\\n}\\n\\nstatic inline uint32_t encode_payload(uint32_t type, uint32_t step, int32_t v) {\\n    uint32_t raw = ((uint32_t)v) & 0x03ffffffu;\\n    return ((type & 1u) << 31) | ((step & 0x1fu) << 26) | raw;\\n}\\n\\nstatic inline int32_t decode_value(uint32_t payload) {\\n    uint32_t raw = payload & 0x03ffffffu;\\n    if (raw & 0x02000000u) raw |= 0xfc000000u;\\n    return (int32_t)raw;\\n}\\n\\nstatic inline bool seen_test_set(uint32_t *seen, uint32_t i) {\\n    uint32_t w = i >> 5;\\n    uint32_t b = 1u << (i & 31u);\\n    bool was = (seen[w] & b) != 0;\\n    seen[w] |= b;\\n    return was;\\n}\\n\\nstatic inline void clear_seen(void) {\\n    for (uint32_t i = 0; i < SEEN_WORDS; i++) {\\n        seen_e[i] = 0;\\n        seen_m[i] = 0;\\n    }\\n}\\n\\nstatic void packet_callback(uint key, uint payload) {\\n    /*\\n     * V4 transport rule: receive CURRENT-step packets during INTEGRATE too.\\n     * V3 listened only in BROADCAST/WAIT, so small barrier skew could make a\\n     * later worker discard valid packets from an earlier worker at the next\\n     * force step. This changes transport timing only, never Izh/plasticity math.\\n     */\\n    if (phase == PHASE_FINISHED) {\\n        wrong_phase++;\\n        return;\\n    }\\n    if ((key & 0xff000000u) != 0xa5000000u) return;\\n    uint32_t src = key & 0x0000ffffu;\\n    if (src >= N_WORKERS) return;\\n    uint32_t step = (payload >> 26) & 0x1fu;\\n    if (step != force_idx) {\\n        wrong_step++;\\n        return;\\n    }\\n    if (phase == PHASE_INTEGRATE) early_rx++;\\n    uint32_t type = payload >> 31;\\n    int32_t value = decode_value(payload);\\n    if (type == 0) {\\n        if (!seen_test_set(seen_e, src)) {\\n            e_all[src] = value;\\n            count_e++;\\n        }\\n    } else {\\n        if (!seen_test_set(seen_m, src)) {\\n            m_all[src] = value;\\n            count_m++;\\n        }\\n    }\\n}\\n\\nstatic bool send_packet_retry(uint32_t key, uint32_t payload) {\\n    for (uint32_t n = 0; n < 256; n++) {\\n        if (spin1_send_mc_packet(key, payload, WITH_PAYLOAD)) return true;\\n    }\\n    send_fail++;\\n    error_flags |= ERR_SEND;\\n    return false;\\n}\\n\\n/* ------------------------------------------------------------------------- */\\n/* Training mechanics                                                         */\\n/* ------------------------------------------------------------------------- */\\nstatic inline uint32_t path_index(uint32_t cyc, uint32_t state, uint32_t coord) {\\n    return ((cyc * PATH_STATES + state) * N_WORKERS) + coord;\\n}\\n\\nstatic void prepare_force_step(void) {\\n    uint32_t idx0 = path_index(cycle_idx, force_idx, worker_id);\\n    uint32_t idx1 = path_index(cycle_idx, force_idx + 1u, worker_id);\\n    int32_t x0 = paths[idx0];\\n    int32_t x1 = paths[idx1];\\n    int32_t dx = x1 - x0;\\n    int32_t gstar = mul_shift32(dx, INV_MU_DT_Q20, Q_PATH);\\n    int32_t y = gstar - grad_q20(x1);\\n\\n    /* Path-start weights remain unchanged until all 27 samples finish. */\\n    volatile int32_t *xrow = &paths[path_index(cycle_idx, force_idx + 1u, 0)];\\n    int64_t pacc = 0;\\n    for (uint32_t j = 0; j < N_WORKERS; j++) {\\n        pacc += (int64_t)weights[j] * (int64_t)xrow[j];  /* Q44 */\\n    }\\n    int32_t pred = clamp_i64_to_i32(floor_rshift64(pacc, Q_WEIGHT)); /* Q20 */\\n    input_e = x1;\\n    input_m = y - pred;\\n}\\n\\nstatic void begin_force_step(void) {\\n    /*\\n     * Open the receive window BEFORE the 80 Izh substeps. Do not clear it at\\n     * integration end; current-step packets received early are valid data.\\n     */\\n    clear_seen();\\n    count_e = 0;\\n    count_m = 0;\\n    slot_idx = 0;\\n    wait_ticks = 0;\\n    substep_idx = 0;\\n    prepare_force_step();\\n    phase = PHASE_INTEGRATE;\\n}\\n\\nstatic void accumulate_local_update(void) {\\n    if (worker_id < NV) {\\n        /* Visible worker owns row W[v,:]. Only hidden endpoints are physical. */\\n        int32_t ev = e_self;\\n        int32_t mv = m_self;\\n        for (uint32_t j = NV; j < N_WORKERS; j++) {\\n            int32_t p1 = q20_product(mv, e_all[j]);\\n            int32_t p2 = q20_product(ev, m_all[j]);\\n            int32_t term = p1 + q20_product(ALPHA015_Q20, p2);\\n            accum[j] += term;\\n        }\\n    } else {\\n        /* Hidden worker owns W^T[h,:] over visible sources and H[h,:]. */\\n        int32_t eh = e_self;\\n        int32_t mh = m_self;\\n        for (uint32_t j = 0; j < NV; j++) {\\n            int32_t p1 = q20_product(m_all[j], eh);\\n            int32_t p2 = q20_product(e_all[j], mh);\\n            int32_t term = p1 + q20_product(ALPHA015_Q20, p2);\\n            accum[j] += term;\\n        }\\n        for (uint32_t j = NV; j < N_WORKERS; j++) {\\n            int32_t p1 = q20_product(mh, e_all[j]);\\n            int32_t p2 = q20_product(eh, m_all[j]);\\n            /* Python oracle: (p1 + p2) >> 1 */\\n            accum[j] += floor_rshift64((int64_t)p1 + p2, 1);\\n        }\\n    }\\n}\\n\\nstatic void apply_path_update(void) {\\n    if (worker_id < NV) {\\n        for (uint32_t j = NV; j < N_WORKERS; j++) {\\n            int64_t a = accum[j] * (int64_t)LR_Q24; /* Q44 */\\n            int64_t d = floor_div_pos64(a, FORCE_STEPS);\\n            int32_t delta = (int32_t)floor_rshift64(d, Q_PATH); /* Q24 */\\n            weights[j] += delta;\\n            accum[j] = 0;\\n        }\\n    } else {\\n        for (uint32_t j = 0; j < NV; j++) {\\n            int64_t a = accum[j] * (int64_t)LR_Q24;\\n            int64_t d = floor_div_pos64(a, FORCE_STEPS);\\n            int32_t delta = (int32_t)floor_rshift64(d, Q_PATH);\\n            weights[j] += delta;\\n            accum[j] = 0;\\n        }\\n        for (uint32_t j = NV; j < N_WORKERS; j++) {\\n            if (j == worker_id) {\\n                weights[j] = 0;\\n                accum[j] = 0;\\n                continue;\\n            }\\n            int64_t a = accum[j] * (int64_t)H_LR_Q24;\\n            int64_t d = floor_div_pos64(a, FORCE_STEPS);\\n            int32_t delta = (int32_t)floor_rshift64(d, Q_PATH);\\n            weights[j] += delta;\\n            accum[j] = 0;\\n        }\\n    }\\n}\\n\\nstatic void write_final_output(void) {\\n    if (outrec == 0) return;\\n    outrec[OUT_MAGIC] = (int32_t)MAGIC_OUT;\\n    outrec[OUT_WORKER] = (int32_t)worker_id;\\n    outrec[OUT_ERROR] = (int32_t)error_flags;\\n    outrec[OUT_CYCLES_DONE] = (int32_t)cycle_idx;\\n    outrec[OUT_MISSING_E] = (int32_t)(N_WORKERS - count_e);\\n    outrec[OUT_MISSING_M] = (int32_t)(N_WORKERS - count_m);\\n    outrec[OUT_SEND_FAIL] = (int32_t)send_fail;\\n    outrec[OUT_EARLY_RX] = (int32_t)early_rx;\\n    outrec[OUT_WRONG_STEP] = (int32_t)wrong_step;\\n    outrec[OUT_WRONG_PHASE] = (int32_t)wrong_phase;\\n    outrec[OUT_PHASE_AT_EXIT] = (int32_t)exit_phase_snapshot;\\n    outrec[OUT_FORCE_AT_EXIT] = (int32_t)exit_force_snapshot;\\n    for (uint32_t k = 0; k < N_MODES; k++) {\\n        int32_t r = readout_q20(e_dv[k], e_z[k], CV_E_Q16, CUA_E_Q16);\\n        outrec[OUT_EREAD0 + k] = r;\\n    }\\n    for (uint32_t j = 0; j < N_WORKERS; j++) {\\n        outrec[OUT_W0 + j] = weights[j];\\n    }\\n    outrec[OUT_STATUS] = (error_flags == ERR_NONE) ? STATUS_DONE : STATUS_ERROR;\\n}\\n\\nstatic void fail_and_exit(uint32_t err) {\\n    error_flags |= err;\\n    exit_phase_snapshot = phase;\\n    exit_force_snapshot = force_idx;\\n    phase = PHASE_FINISHED;\\n    write_final_output();\\n    if (worker_id == 0) {\\n        io_printf(IO_BUF,\\n                  "PHASE1_TRAIN_FAIL error=0x%08x cycle=%u step=%u E=%u M=%u early=%u wrong_step=%u phase=%u\\\\n",\\n                  error_flags, cycle_idx, force_idx, count_e, count_m,\\n                  early_rx, wrong_step, phase);\\n    }\\n    spin1_exit(1);\\n}\\n\\nstatic void finish_training(void) {\\n    exit_phase_snapshot = phase;\\n    exit_force_snapshot = force_idx;\\n    phase = PHASE_FINISHED;\\n    cycle_idx = n_cycles;\\n    write_final_output();\\n    if (worker_id == 0) {\\n        io_printf(IO_BUF, "PHASE1_TRAIN_DONE cycles=%u send_fail=%u error=0x%08x\\\\n",\\n                  n_cycles, send_fail, error_flags);\\n    }\\n    spin1_exit(0);\\n}\\n\\nstatic void timer_callback(uint unused0, uint unused1) {\\n    (void)unused0; (void)unused1;\\n    if (phase == PHASE_FINISHED) return;\\n\\n    if (phase == PHASE_INTEGRATE) {\\n        uint32_t k = schedule[cycle_idx];\\n        if (k >= N_MODES) {\\n            fail_and_exit(ERR_CONFIG); return;\\n        }\\n        if (!izh_one(&e_dv[k], &e_z[k], input_e, GAIN_E_Q30) ||\\n            !izh_one(&m_dv[k], &m_z[k], input_m, GAIN_M_Q30)) {\\n            fail_and_exit(ERR_STATE_RANGE); return;\\n        }\\n        substep_idx++;\\n        if (substep_idx >= IZH_SUBSTEPS) {\\n            e_self = readout_q20(e_dv[k], e_z[k], CV_E_Q16, CUA_E_Q16);\\n            m_self = readout_q20(m_dv[k], m_z[k], CV_M_Q16, CUA_M_Q16);\\n            if (!value_fits_payload(e_self) || !value_fits_payload(m_self)) {\\n                fail_and_exit(ERR_VALUE_RANGE); return;\\n            }\\n            /* Preserve peers already received during INTEGRATE; mark self once. */\\n            if (!seen_test_set(seen_e, worker_id)) {\\n                e_all[worker_id] = e_self;\\n                count_e++;\\n            }\\n            if (!seen_test_set(seen_m, worker_id)) {\\n                m_all[worker_id] = m_self;\\n                count_m++;\\n            }\\n            slot_idx = 0;\\n            wait_ticks = 0;\\n            phase = PHASE_BROADCAST;\\n        }\\n        return;\\n    }\\n\\n    if (phase == PHASE_BROADCAST) {\\n        if ((worker_id % bcast_slots) == slot_idx) {\\n            uint32_t key = route_prefix | worker_id;\\n            send_packet_retry(key, encode_payload(0, force_idx, e_self));\\n            send_packet_retry(key, encode_payload(1, force_idx, m_self));\\n        }\\n        slot_idx++;\\n        if (slot_idx >= bcast_slots) {\\n            phase = PHASE_WAIT;\\n            wait_ticks = 0;\\n        }\\n        return;\\n    }\\n\\n    if (phase == PHASE_WAIT) {\\n        if (count_e >= N_WORKERS && count_m >= N_WORKERS) {\\n            accumulate_local_update();\\n            force_idx++;\\n            substep_idx = 0;\\n            if (force_idx >= FORCE_STEPS) {\\n                apply_path_update();\\n                cycle_idx++;\\n                force_idx = 0;\\n                if (cycle_idx >= n_cycles) {\\n                    finish_training(); return;\\n                }\\n            }\\n            begin_force_step();\\n            return;\\n        }\\n        wait_ticks++;\\n        if (wait_ticks > WAIT_TIMEOUT_TICKS) {\\n            fail_and_exit(ERR_PACKET_TIMEOUT); return;\\n        }\\n    }\\n}\\n\\nvoid c_main(void) {\\n    cfg = (volatile uint32_t *)sark_tag_ptr(SDRAM_TAG, sark_app_id());\\n    if (cfg == 0 || cfg[CFG_MAGIC] != MAGIC_CFG || cfg[CFG_VER] != CFG_VERSION ||\\n        cfg[CFG_NWORKERS] != N_WORKERS || cfg[CFG_NV] != NV || cfg[CFG_NH] != NH ||\\n        cfg[CFG_PATH_STATES] != PATH_STATES || cfg[CFG_FORCE_STEPS] != FORCE_STEPS) {\\n        error_flags |= ERR_CONFIG;\\n        io_printf(IO_BUF, "PHASE1_TRAIN_BAD_CONFIG p=%u\\\\n", spin1_get_core_id());\\n        spin1_exit(1); return;\\n    }\\n\\n    uint32_t p = spin1_get_core_id();\\n    if (p >= MAX_CORES) {\\n        error_flags |= ERR_WORKER;\\n        spin1_exit(1); return;\\n    }\\n    worker_id = cfg[CFG_WORKER_BY_P + p];\\n    if (worker_id >= N_WORKERS) {\\n        /* Core is not a Phase-I worker. */\\n        spin1_exit(0); return;\\n    }\\n\\n    n_cycles = cfg[CFG_NCYCLES];\\n    route_prefix = cfg[CFG_BASE_KEY];\\n    bcast_slots = cfg[CFG_BCAST_SLOTS];\\n    if (bcast_slots == 0) bcast_slots = BCAST_SLOTS_DFLT;\\n    paths = (volatile int32_t *)((uint8_t *)cfg + cfg[CFG_PATH_OFFSET]);\\n    schedule = (volatile uint32_t *)((uint8_t *)cfg + cfg[CFG_SCHED_OFFSET]);\\n    outrec = (volatile int32_t *)((uint8_t *)cfg + cfg[CFG_OUTPUT_OFFSET] +\\n                                 p * cfg[CFG_OUTPUT_STRIDE]);\\n\\n    for (uint32_t j = 0; j < N_WORKERS; j++) {\\n        weights[j] = 0;\\n        accum[j] = 0;\\n        e_all[j] = 0;\\n        m_all[j] = 0;\\n    }\\n    for (uint32_t k = 0; k < N_MODES; k++) {\\n        e_dv[k] = e_z[k] = m_dv[k] = m_z[k] = 0;\\n    }\\n    clear_seen();\\n    outrec[OUT_MAGIC] = (int32_t)MAGIC_OUT;\\n    outrec[OUT_WORKER] = (int32_t)worker_id;\\n    outrec[OUT_STATUS] = STATUS_RUNNING;\\n    outrec[OUT_ERROR] = 0;\\n\\n    cycle_idx = 0; force_idx = 0; substep_idx = 0;\\n    begin_force_step();\\n\\n    if (worker_id == 0) {\\n        io_printf(IO_BUF,\\n            "PHASE1_TRAIN_START workers=%u cycles=%u slots=%u route_prefix=0x%08x\\\\n",\\n            N_WORKERS, n_cycles, bcast_slots, route_prefix);\\n    }\\n\\n    spin1_set_timer_tick(1000);  /* 1 ms */\\n    spin1_callback_on(MCPL_PACKET_RECEIVED, packet_callback, -1);\\n    spin1_callback_on(TIMER_TICK, timer_callback, 2);\\n    spin1_start(SYNC_WAIT);\\n}\\n\'\n'
    '\n'
    '\n'
    '# =============================================================================\n'
    '# Phase-I from-zero training: validated V4 transport + unchanged learning kernel\n'
    '# =============================================================================\n'
    'TR_NMODES = 3\n'
    'TR_CFG_WORDS = 64\n'
    'TR_CFG_BYTES = TR_CFG_WORDS * 4\n'
    'TR_OUT_W0 = 16\n'
    'TR_OUT_WORDS = TR_OUT_W0 + N\n'
    'TR_OUT_BYTES = TR_OUT_WORDS * 4\n'
    'TR_MAGIC_CFG = 0x50315431\n'
    'TR_MAGIC_OUT = 0x50314F31\n'
    'TR_CFG_VERSION = 1\n'
    'TR_SDRAM_TAG = 1\n'
    'TR_BCAST_SLOTS_DEFAULT = 64\n'
    'TR_STATUS_DONE = 3\n'
    'TR_STATUS_ERROR = 0xDEAD0001\n'
    '\n'
    'TR_ROUTE_MAGIC = 0xA5000000\n'
    'TR_ROUTE_GROUP_MASK = 0xFFFF0000\n'
    'TR_MAX_ROOTS = 0x100\n'
    '\n'
    '# Fixed scientific training constants, identical to the accepted pipeline.\n'
    'TR_MU = 1.2240065\n'
    'TR_FORCE_DT = 80e-3\n'
    'TR_LR = 0.01\n'
    'TR_AF, TR_AS = 0.62, 0.92\n'
    'TR_WF, TR_WS = 0.70, 0.30\n'
    'TR_FORCE_DT_MS = 80.0\n'
    'TR_Q_PATH = 20\n'
    'TR_Q_WEIGHT = 24\n'
    'TR_Q_IZH_STATE = 30\n'
    'TR_Q_READ_COEFF = 16\n'
    '\n'
    '\n'
    'def train_route_group_key(root_index: int) -> int:\n'
    '    if not 0 <= int(root_index) < TR_MAX_ROOTS:\n'
    '        raise ValueError(f"root_index out of range: {root_index}")\n'
    '    return TR_ROUTE_MAGIC | (int(root_index) << 16)\n'
    '\n'
    '\n'
    'def train_build_machine_adjacency(machine):\n'
    '    coords = {(int(ch.x), int(ch.y)) for ch in machine.chips}\n'
    '    adjacency = {xy: [] for xy in coords}\n'
    '    for ch in machine.chips:\n'
    '        src = (int(ch.x), int(ch.y))\n'
    '        for link in ch.router.links:\n'
    '            dst = (int(link.destination_x), int(link.destination_y))\n'
    '            if dst in coords:\n'
    '                adjacency[src].append((dst, int(link.source_link_id)))\n'
    '    return adjacency\n'
    '\n'
    '\n'
    'def train_bfs_parent_tree(adjacency, root):\n'
    '    root = tuple(root)\n'
    '    parent = {root: None}\n'
    '    parent_link = {}\n'
    '    q = deque([root])\n'
    '    while q:\n'
    '        u = q.popleft()\n'
    '        for v, link_id in adjacency.get(u, ()):\n'
    '            if v in parent:\n'
    '                continue\n'
    '            parent[v] = u\n'
    '            parent_link[v] = int(link_id)\n'
    '            q.append(v)\n'
    '    return parent, parent_link\n'
    '\n'
    '\n'
    'def train_pruned_tree_to_targets(adjacency, root, targets):\n'
    '    root = tuple(root)\n'
    '    targets = {tuple(t) for t in targets}\n'
    '    parent, parent_link = train_bfs_parent_tree(adjacency, root)\n'
    '    missing = sorted(t for t in targets if t not in parent)\n'
    '    if missing:\n'
    '        raise RuntimeError(f"root {root} cannot reach worker chips {missing[:8]}")\n'
    '    required = {root}\n'
    '    for t in targets:\n'
    '        u = t\n'
    '        while u is not None and u not in required:\n'
    '            required.add(u)\n'
    '            u = parent[u]\n'
    '    child_links = defaultdict(list)\n'
    '    for node in required:\n'
    '        if node == root:\n'
    '            continue\n'
    '        p = parent[node]\n'
    '        child_links[p].append(parent_link[node])\n'
    '    return required, {k: sorted(set(v)) for k, v in child_links.items()}\n'
    '\n'
    '\n'
    'def train_build_explicit_broadcast_routes(machine, by_chip):\n'
    '    from spinn_machine import MulticastRoutingEntry, RoutingEntry\n'
    '    worker_chips = sorted(tuple(xy) for xy in by_chip)\n'
    '    if len(worker_chips) >= TR_MAX_ROOTS:\n'
    '        raise RuntimeError(f"too many worker chips: {len(worker_chips)}")\n'
    '    root_index = {xy: i for i, xy in enumerate(worker_chips)}\n'
    '    adjacency = train_build_machine_adjacency(machine)\n'
    '    routes_by_chip = defaultdict(list)\n'
    '    tree_stats = {}\n'
    '    total_worker_processors = sum(len(v) for v in by_chip.values())\n'
    '    for root in worker_chips:\n'
    '        ridx = root_index[root]\n'
    '        required, child_links = train_pruned_tree_to_targets(adjacency, root, worker_chips)\n'
    '        group_key = train_route_group_key(ridx)\n'
    '        delivered_processors = 0\n'
    '        for node in sorted(required):\n'
    '            processors = sorted(int(p) for _, p in by_chip.get(node, ()))\n'
    '            links = child_links.get(node, [])\n'
    '            if processors or links:\n'
    '                routes_by_chip[node].append(\n'
    '                    MulticastRoutingEntry(\n'
    '                        group_key, TR_ROUTE_GROUP_MASK,\n'
    '                        RoutingEntry(processor_ids=processors, link_ids=links)))\n'
    '                delivered_processors += len(processors)\n'
    '        if delivered_processors != total_worker_processors:\n'
    '            raise RuntimeError(\n'
    '                f"root {root} covers {delivered_processors}, expected {total_worker_processors}")\n'
    '        tree_stats[f"{root[0]},{root[1]}"] = {\n'
    '            "root_index": ridx,\n'
    '            "nodes": len(required),\n'
    '            "edges": sum(len(v) for v in child_links.values()),\n'
    '            "worker_processors_reached": delivered_processors,\n'
    '        }\n'
    '    return {\n'
    '        "routes_by_chip": dict(routes_by_chip),\n'
    '        "root_index": root_index,\n'
    '        "tree_stats": tree_stats,\n'
    '        "route_chips": sorted(routes_by_chip),\n'
    '        "total_entries": sum(len(v) for v in routes_by_chip.values()),\n'
    '        "max_entries_per_chip": max((len(v) for v in routes_by_chip.values()), default=0),\n'
    '    }\n'
    '\n'
    '\n'
    'def train_route_sig(e):\n'
    '    return (int(e.key), int(e.mask), int(e.spinnaker_route), bool(e.defaultable))\n'
    '\n'
    '\n'
    'def train_install_and_verify_routes(txrx, plan, custom_app_id: int):\n'
    '    stats = {}\n'
    '    for (x, y), routes in sorted(plan["routes_by_chip"].items()):\n'
    '        txrx.load_multicast_routes(int(x), int(y), list(routes), int(custom_app_id))\n'
    '        reread = list(txrx.get_multicast_routes(int(x), int(y), int(custom_app_id)))\n'
    '        a = sorted(train_route_sig(e) for e in routes)\n'
    '        b = sorted(train_route_sig(e) for e in reread)\n'
    '        stats[f"{x},{y}"] = {"planned": len(a), "reloaded": len(b), "equal": a == b}\n'
    '        if a != b:\n'
    '            raise RuntimeError(f"training route verification failed on chip {(x,y)}")\n'
    '    return stats\n'
    '\n'
    '\n'
    'def train_low_discrepancy_schedule(prior: np.ndarray, n: int) -> np.ndarray:\n'
    '    p = np.asarray(prior, dtype=np.float64)\n'
    '    c = np.zeros(3, dtype=np.int64)\n'
    '    out = []\n'
    '    for t in range(int(n)):\n'
    '        deficit = (t + 1) * p - c\n'
    '        k = int(np.argmax(deficit))\n'
    '        out.append(k)\n'
    '        c[k] += 1\n'
    '    return np.asarray(out, dtype=np.int32)\n'
    '\n'
    '\n'
    'def train_izh_constants() -> dict[str, int | float]:\n'
    '    rf = 1.0 / (-TR_FORCE_DT_MS / math.log(TR_AF))\n'
    '    rs = 1.0 / (-TR_FORCE_DT_MS / math.log(TR_AS))\n'
    '    S = -(rf + rs)\n'
    '    P = rf * rs\n'
    '    a = 1e-3\n'
    '    q = S + a\n'
    '    b = q + P / a\n'
    '    v0 = (q - 5.0) / 0.08\n'
    '    u0 = b * v0\n'
    '    I0 = -0.04 * v0 * v0 - 5.0 * v0 - 140.0 + u0\n'
    '    num_s = TR_WF * rf + TR_WS * rs\n'
    '    num_0 = rf * rs\n'
    '    def readout(g):\n'
    '        cv = num_s / g\n'
    '        cu = (num_0 / g - cv * a) / (a * b)\n'
    '        return cv, cu\n'
    '    cE = readout(1e-5)\n'
    '    cM = readout(1e-6)\n'
    '    s20, s24, s30, s16 = 1 << 20, 1 << 24, 1 << 30, 1 << 16\n'
    '    return {\n'
    '        "q_path": TR_Q_PATH,\n'
    '        "q_weight": TR_Q_WEIGHT,\n'
    '        "q_state": TR_Q_IZH_STATE,\n'
    '        "q_read_coeff": TR_Q_READ_COEFF,\n'
    '        "q_q30": int(round(q * s30)),\n'
    '        "a_q30": int(round(a * s30)),\n'
    '        "b_q30": int(round(b * s30)),\n'
    '        "c004_q30": int(round(0.04 * s30)),\n'
    '        "gain_e_q30": int(round(1e-5 * s30)),\n'
    '        "gain_m_q30": int(round(1e-6 * s30)),\n'
    '        "cv_e_q16": int(round(cE[0] * s16)),\n'
    '        "cua_e_q16": int(round(cE[1] * a * s16)),\n'
    '        "cv_m_q16": int(round(cM[0] * s16)),\n'
    '        "cua_m_q16": int(round(cM[1] * a * s16)),\n'
    '        "inv_mu_force_dt_q20": int(round((1.0 / (TR_MU * TR_FORCE_DT)) * s20)),\n'
    '        "alpha015_q20": int(round(0.15 * s20)),\n'
    '        "lr_q24": int(round(TR_LR * s24)),\n'
    '        "h_lr_q24": int(round(0.15 * TR_LR * s24)),\n'
    '        "v0": float(v0), "u0": float(u0), "I0": float(I0),\n'
    '        "a": float(a), "b": float(b), "q": float(q),\n'
    '        "cv_e": float(cE[0]), "cu_e": float(cE[1]),\n'
    '        "cv_m": float(cM[0]), "cu_m": float(cM[1]),\n'
    '    }\n'
    '\n'
    '\n'
    '\n'
    '# =============================================================================\n'
    '# Standard mnist.npz -> fresh forward/noising paths\n'
    '# =============================================================================\n'
    '# Forward/noising constants/functions are copied from the validated cloud\n'
    '# full_pipeline.py FreshPathProvider.  Only the image source is changed from\n'
    '# one fixed 3-row file to distinct examples drawn from standard MNIST train.\n'
    'FP_DT = 1e-3\n'
    'FP_T_IMAGE_EQ = 2.16\n'
    'FP_T_FORWARD = 2.16\n'
    'FP_FORCE_DT = 80e-3\n'
    'FP_FORCE_STRIDE = int(round(FP_FORCE_DT / FP_DT))\n'
    'FP_TAU_M = 7.5e-3\n'
    'FP_TAU_SYN = 50e-3\n'
    'FP_SIGMA_LOCAL = 1.27\n'
    'FP_SIGN = np.array([1, 1, 1, -1, -1, -1, 0], dtype=np.float32)\n'
    'FP_DECODER = np.array([\n'
    '    0.00970358172, 0.155770390, 0.0397680875,\n'
    '   -0.00970358172,-0.155770390,-0.0397680875, 0.0\n'
    '], dtype=np.float32)\n'
    'FP_BASE = np.array([\n'
    '    0.96216805, 1.04082352, 1.02707158,\n'
    '    0.96216805, 1.04082352, 1.02707158, 0.90\n'
    '], dtype=np.float32)\n'
    'FP_G7 = np.array([\n'
    '    0.17617594, 0.309189415, 0.00132405373,\n'
    '    0.17617594, 0.309189415, 0.00132405373, 0.0\n'
    '], dtype=np.float32)\n'
    'FP_M7 = np.array([\n'
    '    0.00157977546, 0.000176450817, 0.0722733284,\n'
    '    0.00157977546, 0.000176450817, 0.0722733284, 0.0\n'
    '], dtype=np.float32)\n'
    'FP_W_LOCAL = np.outer(FP_SIGN * FP_G7, FP_DECODER).astype(np.float32)\n'
    'FP_W_LOCAL[:6, 6] += -0.06\n'
    'FP_FIELD_GAIN = (FP_SIGN * FP_M7).astype(np.float32)\n'
    'FP_AM = np.float32(math.exp(-FP_DT / FP_TAU_M))\n'
    'FP_AQ = np.float32(math.exp(-FP_DT / FP_TAU_SYN))\n'
    'FP_SIGMA_STEP = np.float32(FP_SIGMA_LOCAL * math.sqrt(FP_DT))\n'
    'FP_REFRACTORY_SENTINEL = np.float32(-1e9)\n'
    'FP_HIDDEN_PIXEL_INDEX = np.arange(NH, dtype=np.int64) % NV\n'
    '\n'
    '# Fixed calibration inferred from the historical three-symbol data.  This is a\n'
    '# preprocessing convention, not a trainable quantity.\n'
    'MNIST_PHYSICAL_CENTER = np.float32(0.130925363)\n'
    'MNIST_PHYSICAL_TARGET_RMS = np.float32(1.744274831)\n'
    '\n'
    '\n'
    'def locate_mnist_npz(user_path: Path | None) -> Path:\n'
    '    candidates = []\n'
    '    if user_path is not None:\n'
    '        candidates.append(Path(user_path).expanduser())\n'
    '    here = Path(__file__).resolve().parent\n'
    '    candidates += [here / "mnist.npz", Path.cwd() / "mnist.npz", Path("mnist.npz")]\n'
    '    seen = set()\n'
    '    for p in candidates:\n'
    '        try:\n'
    '            q = p.resolve()\n'
    '        except Exception:\n'
    '            q = p\n'
    '        key = str(q)\n'
    '        if key in seen:\n'
    '            continue\n'
    '        seen.add(key)\n'
    '        if p.is_file():\n'
    '            return p.resolve()\n'
    '    raise FileNotFoundError(\n'
    '        "Could not find standard Keras-format mnist.npz. Put mnist.npz next to "\n'
    '        "this Python script, or pass --mnist /full/path/mnist.npz. Required keys: "\n'
    '        "x_train, y_train, x_test, y_test."\n'
    '    )\n'
    '\n'
    '\n'
    'def load_standard_mnist_npz(path: Path):\n'
    '    path = Path(path)\n'
    '    with np.load(path, allow_pickle=False) as d:\n'
    '        required = {"x_train", "y_train", "x_test", "y_test"}\n'
    '        missing = required - set(d.files)\n'
    '        if missing:\n'
    '            raise ValueError(f"{path} is not Keras-format MNIST; missing {sorted(missing)}")\n'
    '        x_train_raw = np.asarray(d["x_train"])\n'
    '        y_train = np.asarray(d["y_train"], dtype=np.int64)\n'
    '        x_test_raw = np.asarray(d["x_test"])\n'
    '        y_test = np.asarray(d["y_test"], dtype=np.int64)\n'
    '\n'
    '    if x_train_raw.shape[1:] != (28, 28) or x_test_raw.shape[1:] != (28, 28):\n'
    '        raise ValueError(\n'
    '            f"Expected x_train/x_test images of shape (*,28,28); got "\n'
    '            f"{x_train_raw.shape} and {x_test_raw.shape}"\n'
    '        )\n'
    '    if len(x_train_raw) != len(y_train) or len(x_test_raw) != len(y_test):\n'
    '        raise ValueError("MNIST image/label lengths do not match")\n'
    '\n'
    '    x_train = x_train_raw.astype(np.float32)\n'
    '    x_test = x_test_raw.astype(np.float32)\n'
    '    if float(x_train.max()) > 1.0 or float(x_test.max()) > 1.0:\n'
    '        x_train /= np.float32(255.0)\n'
    '        x_test /= np.float32(255.0)\n'
    '    x_train = np.clip(x_train, 0.0, 1.0).reshape(len(x_train), NV)\n'
    '    x_test = np.clip(x_test, 0.0, 1.0).reshape(len(x_test), NV)\n'
    '\n'
    '    tr_mask = np.isin(y_train, [0, 1, 2])\n'
    '    te_mask = np.isin(y_test, [0, 1, 2])\n'
    '    xtr012, ytr012 = x_train[tr_mask], y_train[tr_mask]\n'
    '    xte012, yte012 = x_test[te_mask], y_test[te_mask]\n'
    '    report = {\n'
    '        "path": str(path.resolve()),\n'
    '        "file_sha256": sha256_file(path),\n'
    '        "keras_keys": ["x_train", "y_train", "x_test", "y_test"],\n'
    '        "full_train_n": int(len(x_train)),\n'
    '        "full_test_n": int(len(x_test)),\n'
    '        "train_012_n": int(len(xtr012)),\n'
    '        "test_012_n": int(len(xte012)),\n'
    '        "train_012_counts": np.bincount(ytr012, minlength=3)[:3].astype(int).tolist(),\n'
    '        "test_012_counts": np.bincount(yte012, minlength=3)[:3].astype(int).tolist(),\n'
    '        "raw_pixel_range": [float(x_train.min()), float(x_train.max())],\n'
    '    }\n'
    '    return xtr012, ytr012, xte012, yte012, report\n'
    '\n'
    '\n'
    'def preprocess_mnist_image(x: np.ndarray, mode: str = "whitelam-rms") -> np.ndarray:\n'
    '    x = np.asarray(x, dtype=np.float32).reshape(NV)\n'
    '    if mode == "unit":\n'
    '        return x.copy()\n'
    '    if mode != "whitelam-rms":\n'
    '        raise ValueError(f"unknown pixel preprocessing mode {mode!r}")\n'
    '    d = x - MNIST_PHYSICAL_CENTER\n'
    '    rms = np.float32(np.sqrt(np.mean(d.astype(np.float64) ** 2)))\n'
    '    if not np.isfinite(rms) or float(rms) <= 1e-12:\n'
    '        raise ValueError("MNIST image has effectively zero RMS after fixed centering")\n'
    '    return (d * (MNIST_PHYSICAL_TARGET_RMS / rms)).astype(np.float32)\n'
    '\n'
    '\n'
    'def select_training_examples(xtr: np.ndarray, ytr: np.ndarray, schedule: np.ndarray,\n'
    '                             data_seed: int, preprocess: str):\n'
    '    rng = np.random.default_rng(int(data_seed))\n'
    '    pools = {}\n'
    '    cursor = np.zeros(3, dtype=np.int64)\n'
    '    reuse = np.zeros(3, dtype=np.int64)\n'
    '    selected = []\n'
    '    selected_raw = []\n'
    '    selected_indices = []\n'
    '    for k in range(3):\n'
    '        ids = np.flatnonzero(ytr == k)\n'
    '        if len(ids) == 0:\n'
    '            raise RuntimeError(f"MNIST training set has no class {k}")\n'
    '        pools[k] = rng.permutation(ids)\n'
    '\n'
    '    for cyc, kk in enumerate(np.asarray(schedule, dtype=int)):\n'
    '        k = int(kk)\n'
    '        if cursor[k] >= len(pools[k]):\n'
    '            pools[k] = rng.permutation(np.flatnonzero(ytr == k))\n'
    '            cursor[k] = 0\n'
    '            reuse[k] += 1\n'
    '        idx = int(pools[k][cursor[k]])\n'
    '        cursor[k] += 1\n'
    '        raw = np.asarray(xtr[idx], dtype=np.float32)\n'
    '        selected_raw.append(raw)\n'
    '        selected.append(preprocess_mnist_image(raw, preprocess))\n'
    '        selected_indices.append(idx)\n'
    '\n'
    '    selected = np.asarray(selected, dtype=np.float32)\n'
    '    selected_raw = np.asarray(selected_raw, dtype=np.float32)\n'
    '    selected_indices = np.asarray(selected_indices, dtype=np.int64)\n'
    '\n'
    "    # Training-only class centroids are used by the pre-existing generator's\n"
    '    # between-image mode readout.  Held-out x_test is not used by generation.\n'
    '    refs = []\n'
    '    for k in range(3):\n'
    '        q = np.flatnonzero(np.asarray(schedule, dtype=int) == k)\n'
    '        if len(q) == 0:\n'
    '            raise RuntimeError(f"no exposure for class {k}; generator readout needs all three modes")\n'
    '        refs.append(np.mean(selected[q], axis=0))\n'
    '    refs = np.asarray(refs, dtype=np.float64)\n'
    '    return selected, selected_raw, selected_indices, refs, reuse\n'
    '\n'
    '\n'
    'def _fp_micro_step_single(v: np.ndarray, q: np.ndarray, field: np.ndarray,\n'
    '                          rng: np.random.Generator):\n'
    '    refractory = v < -1e8\n'
    '    if np.any(refractory):\n'
    '        v[refractory] = 0.0\n'
    '    current = FP_BASE[None, :] + q @ FP_W_LOCAL.T + field[:, None] * FP_FIELD_GAIN[None, :]\n'
    '    v_next = current + (v - current) * FP_AM + FP_SIGMA_STEP * rng.standard_normal(\n'
    '        v.shape, dtype=np.float32)\n'
    '    active = ~refractory\n'
    '    v[active] = v_next[active]\n'
    '    spikes = (v >= 1.0) & active\n'
    '    v[spikes] = FP_REFRACTORY_SENTINEL\n'
    '    q *= FP_AQ\n'
    '    q += spikes.astype(np.float32)\n'
    '    return v, q\n'
    '\n'
    '\n'
    'def _fp_decode(q: np.ndarray) -> np.ndarray:\n'
    '    return q @ FP_DECODER\n'
    '\n'
    '\n'
    'def forward_noising_path_from_mnist_image(symbol: np.ndarray, seed: int) -> np.ndarray:\n'
    '    """Exact FreshPathProvider dynamics from validated full_pipeline.py."""\n'
    '    symbol = np.asarray(symbol, dtype=np.float32).reshape(NV)\n'
    '    rng = np.random.default_rng(int(seed))\n'
    '    v = rng.random((N, 7), dtype=np.float32)\n'
    '    q = np.full((N, 7), 0.5, dtype=np.float32)\n'
    '    image = np.concatenate([symbol, symbol[FP_HIDDEN_PIXEL_INDEX]]).astype(np.float32)\n'
    '    for _ in range(int(round(FP_T_IMAGE_EQ / FP_DT))):\n'
    '        v, q = _fp_micro_step_single(v, q, 2.0 * image, rng)\n'
    '    states = [_fp_decode(q).copy()]\n'
    '    n_steps = int(round(FP_T_FORWARD / FP_DT))\n'
    '    for step in range(n_steps):\n'
    '        amp = max(0.0, 1.0 - step / (0.75 * n_steps))\n'
    '        v, q = _fp_micro_step_single(v, q, 2.0 * amp * image, rng)\n'
    '        if (step + 1) % FP_FORCE_STRIDE == 0:\n'
    '            states.append(_fp_decode(q).copy())\n'
    '    out = np.asarray(states, dtype=np.float32)\n'
    '    if out.shape != (28, N):\n'
    '        raise RuntimeError(f"fresh path shape {out.shape}, expected {(28, N)}")\n'
    '    return out\n'
    '\n'
    '\n'
    'def _fresh_path_job(args):\n'
    '    cycle, digit, seed, image = args\n'
    '    st = forward_noising_path_from_mnist_image(image, seed)\n'
    '    return int(cycle), int(digit), int(seed), st\n'
    '\n'
    '\n'
    'def build_training_payload_from_mnist_npz(\n'
    '    mnist_path: Path,\n'
    '    prior: np.ndarray,\n'
    '    cycles: int,\n'
    '    outdir: Path,\n'
    '    seed_start: int = 20000,\n'
    '    data_seed: int = 20261011,\n'
    '    path_workers: int = 1,\n'
    '    pixel_preprocess: str = "whitelam-rms",\n'
    '):\n'
    '    """Use distinct standard-MNIST train images and generate every trajectory fresh."""\n'
    '    global MNIST012_SYMBOLS\n'
    '    xtr, ytr, xte, yte, mnist_report = load_standard_mnist_npz(mnist_path)\n'
    '    schedule = train_low_discrepancy_schedule(prior, cycles)\n'
    '    images, raw_images, selected_indices, refs, reuse = select_training_examples(\n'
    '        xtr, ytr, schedule, data_seed, pixel_preprocess)\n'
    '    MNIST012_SYMBOLS = refs.astype(np.float64, copy=True)\n'
    '\n'
    '    seeds = np.arange(int(seed_start), int(seed_start) + int(cycles), dtype=np.int64)\n'
    '    jobs = [\n'
    '        (cyc, int(schedule[cyc]), int(seeds[cyc]), images[cyc])\n'
    '        for cyc in range(int(cycles))\n'
    '    ]\n'
    '    paths = [None] * int(cycles)\n'
    '    t0 = time.perf_counter()\n'
    '    workers = max(1, int(path_workers))\n'
    '    if workers == 1:\n'
    '        for n, job in enumerate(jobs, 1):\n'
    '            cyc, digit, seed, st = _fresh_path_job(job)\n'
    '            paths[cyc] = st\n'
    '            print(\n'
    '                f"fresh path {n:03d}/{cycles} digit={digit} "\n'
    '                f"mnist_train_index={selected_indices[cyc]} seed={seed}", flush=True)\n'
    '    else:\n'
    '        import multiprocessing as mp\n'
    '        from concurrent.futures import ProcessPoolExecutor, as_completed\n'
    '        try:\n'
    '            ctx = mp.get_context("fork")\n'
    '            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:\n'
    '                futs = [ex.submit(_fresh_path_job, job) for job in jobs]\n'
    '                done = 0\n'
    '                for fut in as_completed(futs):\n'
    '                    cyc, digit, seed, st = fut.result()\n'
    '                    paths[cyc] = st\n'
    '                    done += 1\n'
    '                    print(\n'
    '                        f"fresh path {done:03d}/{cycles} digit={digit} "\n'
    '                        f"mnist_train_index={selected_indices[cyc]} seed={seed}", flush=True)\n'
    '        except Exception as exc:\n'
    '            print(f"Parallel fresh-path generation failed ({exc!r}); falling back to sequential.", flush=True)\n'
    '            paths = [None] * int(cycles)\n'
    '            for n, job in enumerate(jobs, 1):\n'
    '                cyc, digit, seed, st = _fresh_path_job(job)\n'
    '                paths[cyc] = st\n'
    '                print(\n'
    '                    f"fresh path {n:03d}/{cycles} digit={digit} "\n'
    '                    f"mnist_train_index={selected_indices[cyc]} seed={seed}", flush=True)\n'
    '\n'
    '    paths = np.asarray(paths, dtype=np.float32)\n'
    '    if paths.shape != (cycles, 28, N):\n'
    '        raise RuntimeError(f"fresh paths shape {paths.shape}, expected {(cycles, 28, N)}")\n'
    '    paths_q20 = np.rint(paths.astype(np.float64) * float(1 << TR_Q_PATH)).astype(np.int32)\n'
    '    constants = train_izh_constants()\n'
    '    prep_dir = outdir / "00_training_payload"\n'
    '    prep_dir.mkdir(parents=True, exist_ok=True)\n'
    '    payload = prep_dir / "phase1_training_payload.npz"\n'
    '    np.savez_compressed(\n'
    '        payload,\n'
    '        paths_q20=paths_q20,\n'
    '        schedule=schedule.astype(np.int16),\n'
    '        seeds=seeds.astype(np.int64),\n'
    '        reuse=reuse.astype(np.int16),\n'
    '        prior=np.asarray(prior, dtype=np.float64),\n'
    '        constants_json=np.array([json.dumps(constants, sort_keys=True)]),\n'
    '        source_mnist_file_sha256=np.array([mnist_report["file_sha256"]]),\n'
    '        selected_train_indices=selected_indices,\n'
    '        selected_train_labels=schedule.astype(np.int16),\n'
    '        pixel_preprocess=np.array([pixel_preprocess]),\n'
    '        source=np.array(["standard mnist.npz -> fresh validated 9072-LIF forward/noising dynamics"]),\n'
    '    )\n'
    '    # Audit/provenance only; never consumed as a training input.\n'
    '    np.savez_compressed(\n'
    '        prep_dir / "MNIST_SELECTION_AUDIT.npz",\n'
    '        selected_train_indices=selected_indices,\n'
    '        selected_train_labels=schedule.astype(np.int16),\n'
    '        selected_raw_01=raw_images,\n'
    '        selected_physical=images,\n'
    '        class_reference_centroids=refs.astype(np.float32),\n'
    '    )\n'
    '    report = {\n'
    '        "source": "standard Keras mnist.npz -> distinct x_train examples -> fresh forward/noising dynamics",\n'
    '        "mnist": mnist_report,\n'
    '        "pixel_preprocess": pixel_preprocess,\n'
    '        "physical_center": float(MNIST_PHYSICAL_CENTER),\n'
    '        "physical_target_rms": float(MNIST_PHYSICAL_TARGET_RMS),\n'
    '        "fresh_paths": True,\n'
    '        "external_path_cache_used": False,\n'
    '        "held_out_test_used_for_learning": False,\n'
    '        "data_seed": int(data_seed),\n'
    '        "path_seed_start": int(seed_start),\n'
    '        "path_workers_requested": int(path_workers),\n'
    '        "path_generation_wall_s": float(time.perf_counter() - t0),\n'
    '        "shape": list(paths_q20.shape),\n'
    '        "target_prior": np.asarray(prior, dtype=float).tolist(),\n'
    '        "schedule_counts": np.bincount(schedule, minlength=3).astype(int).tolist(),\n'
    '        "reuse_rounds_per_class": reuse.astype(int).tolist(),\n'
    '        "selected_train_indices": selected_indices.astype(int).tolist(),\n'
    '        "payload": str(payload),\n'
    '        "payload_sha256": sha256_file(payload),\n'
    '        "digits": [0, 1, 2],\n'
    '    }\n'
    '    json_dump(prep_dir / "PAYLOAD_REPORT.json", report)\n'
    '    return payload, report, (xte, yte)\n'
    '\n'
    'def train_load_payload(payload_path: Path):\n'
    '    d = np.load(payload_path, allow_pickle=True)\n'
    '    required = {"paths_q20", "schedule", "prior", "constants_json"}\n'
    '    missing = required - set(d.files)\n'
    '    if missing:\n'
    '        raise RuntimeError(f"payload missing {sorted(missing)}")\n'
    '    paths = np.asarray(d["paths_q20"], dtype=np.int32)\n'
    '    schedule = np.asarray(d["schedule"], dtype=np.int32)\n'
    '    prior = np.asarray(d["prior"], dtype=np.float64)\n'
    '    if paths.shape != (len(schedule), 28, N):\n'
    '        raise RuntimeError(f"unexpected paths shape {paths.shape}")\n'
    '    reuse = np.asarray(d["reuse"], dtype=np.int32) if "reuse" in d.files else np.zeros(3, dtype=np.int32)\n'
    '    return paths, schedule, prior, json.loads(str(d["constants_json"][0])), reuse\n'
    '\n'
    '\n'
    'def train_get_new_app_id(FecDataView):\n'
    '    try:\n'
    '        return FecDataView.get_new_id()\n'
    '    except TypeError:\n'
    '        return FecDataView().get_new_id()\n'
    '\n'
    '\n'
    'def train_read_iobuf(txrx, cores) -> str:\n'
    '    chunks = []\n'
    '    for buf in txrx.get_iobuf(cores):\n'
    '        text = getattr(buf, "iobuf", None)\n'
    '        chunks.append(str(text if text is not None else buf))\n'
    '    return "\\n".join(chunks)\n'
    '\n'
    '\n'
    'def train_carrier_imports():\n'
    '    try:\n'
    '        import pyNN.spiNNaker as sim\n'
    '        from spinn_front_end_common.data import FecDataView\n'
    '        from spynnaker.pyNN.extra_algorithms.splitter_components import SplitterPopulationVertexFixed\n'
    '        from spinn_machine import CoreSubsets\n'
    '        from spinnman.model.enums import CPUState\n'
    '        from spinnman.messages.scp.enums.signal import Signal\n'
    '    except Exception as exc:\n'
    '        raise RuntimeError("run training inside the working sPyNNaker environment") from exc\n'
    '    return sim, FecDataView, SplitterPopulationVertexFixed, CoreSubsets, CPUState, Signal\n'
    '\n'
    '\n'
    'def train_make_header(route_prefix: int, n_cycles: int, path_offset: int, schedule_offset: int,\n'
    '                      output_offset: int, output_stride: int, worker_by_p: dict[int,int],\n'
    '                      bcast_slots: int) -> bytes:\n'
    '    w = np.zeros(TR_CFG_WORDS, dtype="<u4")\n'
    '    w[0] = TR_MAGIC_CFG; w[1] = TR_CFG_VERSION; w[2] = N; w[3] = NV; w[4] = NH\n'
    '    w[5] = n_cycles; w[6] = 28; w[7] = 27; w[8] = int(route_prefix)\n'
    '    w[9] = path_offset; w[10] = schedule_offset; w[11] = output_offset\n'
    '    w[12] = output_stride; w[13] = int(bcast_slots)\n'
    '    w[16:16 + MAX_CORES] = np.uint32(0xFFFFFFFF)\n'
    '    for p, wid in worker_by_p.items():\n'
    '        w[16 + int(p)] = int(wid)\n'
    '    return w.tobytes()\n'
    '\n'
    '\n'
    'def train_parse_output_record(data: bytes):\n'
    '    a = np.frombuffer(data, dtype="<i4", count=TR_OUT_WORDS).copy()\n'
    '    if int(np.uint32(a[0])) != TR_MAGIC_OUT:\n'
    '        raise RuntimeError(f"bad training output magic 0x{int(np.uint32(a[0])):08x}")\n'
    '    return {\n'
    '        "worker": int(a[1]), "status": int(np.uint32(a[2])), "error": int(np.uint32(a[3])),\n'
    '        "cycles_done": int(a[4]), "missing_e": int(a[5]), "missing_m": int(a[6]),\n'
    '        "send_fail": int(a[7]), "Eread_q20": a[8:11].astype(np.int32),\n'
    '        "early_rx": int(np.uint32(a[11])), "wrong_step": int(np.uint32(a[12])),\n'
    '        "wrong_phase": int(np.uint32(a[13])), "phase_at_exit": int(np.uint32(a[14])),\n'
    '        "force_at_exit": int(np.uint32(a[15])), "weights_q24": a[TR_OUT_W0:].astype(np.int32),\n'
    '    }\n'
    '\n'
    '\n'
    'def train_build_aplx(outdir: Path, build_tree: Path | None) -> tuple[Path, Path]:\n'
    '    outdir.mkdir(parents=True, exist_ok=True)\n'
    '    kernel_c = outdir / "phase1_train_kernel_v4.c"\n'
    '    kernel_c.write_text(TRAIN_KERNEL_C, encoding="utf-8")\n'
    '    if sha256_file(kernel_c) != VALIDATED_TRAIN_KERNEL_SHA256:\n'
    '        raise RuntimeError("embedded training kernel SHA mismatch")\n'
    '    if build_tree is None:\n'
    '        build_tree = Path(os.environ.get("MSDL_GATE3_BUILD", "/home/jovyan/PGI/_gate3_build"))\n'
    '    if not build_tree.exists():\n'
    '        raise FileNotFoundError(f"custom APLX build tree not found: {build_tree}")\n'
    '    srcdir = build_tree / "src"; builddir = build_tree / "build"\n'
    '    srcdir.mkdir(parents=True, exist_ok=True)\n'
    '    target_src = srcdir / "msdl_gate3.c"\n'
    '    backup = target_src.read_bytes() if target_src.exists() else None\n'
    '    try:\n'
    '        target_src.write_bytes(kernel_c.read_bytes())\n'
    '        if builddir.exists(): shutil.rmtree(builddir)\n'
    '        builddir.mkdir(parents=True, exist_ok=True)\n'
    '        env = os.environ.copy()\n'
    '        env.setdefault("FEC_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/SpiNNFrontEndCommon/c_common/front_end_common_lib")\n'
    '        env.setdefault("SPINN_COMMON_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/spinn_common")\n'
    '        env.setdefault("SPINN_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/spinnaker_tools")\n'
    '        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")\n'
    '        p = subprocess.run(["make"], cwd=build_tree, env=env, text=True,\n'
    '                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n'
    '        (outdir / "aplx_build.log").write_text(p.stdout, encoding="utf-8")\n'
    '        if p.returncode != 0:\n'
    '            raise RuntimeError("training APLX build failed:\\n" + "\\n".join(p.stdout.splitlines()[-120:]))\n'
    '        candidates = list(build_tree.rglob("msdl_gate3.aplx"))\n'
    '        if not candidates: raise FileNotFoundError("training APLX not produced")\n'
    '        built = max(candidates, key=lambda x: x.stat().st_mtime)\n'
    '        dest = outdir / "phase1_train_kernel.aplx"\n'
    '        shutil.copy2(built, dest)\n'
    '        return dest, kernel_c\n'
    '    finally:\n'
    '        if backup is None:\n'
    '            try: target_src.unlink()\n'
    '            except FileNotFoundError: pass\n'
    '        else:\n'
    '            target_src.write_bytes(backup)\n'
    '\n'
    '\n'
    'def run_from_zero_training(payload_path: Path, outdir: Path, build_tree: Path | None,\n'
    '                           timeout_s: float, bcast_slots: int):\n'
    '    outdir.mkdir(parents=True, exist_ok=True)\n'
    '    paths, schedule, prior, constants, cache_reuse = train_load_payload(payload_path)\n'
    '    aplx, kernel_c = train_build_aplx(outdir, build_tree)\n'
    '    sim, FecDataView, SplitterPopulationVertexFixed, CoreSubsets, CPUState, Signal = train_carrier_imports()\n'
    '    sim_started = False; txrx = None; custom_app_id = None\n'
    '    report = {\n'
    '        "stage": "from-zero SpiNNaker training -> clean-generation bridge",\n'
    '        "payload": str(payload_path), "payload_sha256": sha256_file(payload_path),\n'
    '        "kernel_sha256": sha256_file(kernel_c), "aplx_sha256": sha256_file(aplx),\n'
    '        "n_workers": N, "cycles": int(len(schedule)), "prior": prior.tolist(),\n'
    '        "schedule_counts": np.bincount(schedule, minlength=3).astype(int).tolist(),\n'
    '        "weights_start_from_zero": True, "host_learning_updates": 0,\n'
    '        "bcast_slots": int(bcast_slots),\n'
    '    }\n'
    '    try:\n'
    '        sim.setup(timestep=1.0, min_delay=1.0, time_scale_factor=1)\n'
    '        sim_started = True\n'
    '        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 1)\n'
    '        carrier = sim.Population(\n'
    '            N,\n'
    '            sim.IF_curr_exp(cm=1.0, tau_m=20.0, tau_refrac=1.0, tau_syn_E=5.0, tau_syn_I=5.0,\n'
    '                            v_rest=0.0, v_reset=0.0, v_thresh=1.0, i_offset=0.0),\n'
    '            label="phase1_route_carrier",\n'
    '            additional_parameters={"splitter": SplitterPopulationVertexFixed()},\n'
    '        )\n'
    '        t0 = time.perf_counter(); sim.run(1.0)\n'
    '        report["carrier_map_load_run_s"] = float(time.perf_counter() - t0)\n'
    '        app_vertex = carrier._vertex\n'
    '        mverts = list(app_vertex.machine_vertices)\n'
    '        if len(mverts) != N: raise RuntimeError(f"expected {N} training machine vertices, got {len(mverts)}")\n'
    '        placements = []\n'
    '        for mv in mverts:\n'
    '            sl = mv.vertex_slice\n'
    '            if int(sl.n_atoms) != 1: raise RuntimeError(f"training carrier slice is not one atom: {sl}")\n'
    '            wid = int(sl.lo_atom); pl = FecDataView.get_placement_of_vertex(mv)\n'
    '            placements.append((wid, int(pl.x), int(pl.y), int(pl.p), mv))\n'
    '        placements.sort(key=lambda x: x[0])\n'
    '        if [x[0] for x in placements] != list(range(N)): raise RuntimeError("training worker IDs not 0..1295")\n'
    '        txrx = FecDataView.get_transceiver(); pynn_app_id = int(FecDataView.get_app_id())\n'
    '        by_chip = defaultdict(list); worker_core = CoreSubsets()\n'
    '        for wid,x,y,p,mv in placements:\n'
    '            by_chip[(x,y)].append((wid,p)); worker_core.add_processor(x,y,p)\n'
    '        report["actual_worker_chips"] = len(by_chip)\n'
    '        machine = FecDataView.get_machine()\n'
    '        route_plan = train_build_explicit_broadcast_routes(machine, by_chip)\n'
    '        report["route_entries_total"] = int(route_plan["total_entries"])\n'
    '        report["max_route_entries_per_chip"] = int(route_plan["max_entries_per_chip"])\n'
    '        if route_plan["max_entries_per_chip"] >= 1023: raise RuntimeError("training route plan exceeds router capacity")\n'
    '        txrx.stop_application(pynn_app_id); time.sleep(0.25)\n'
    '        custom_app_id = int(train_get_new_app_id(FecDataView)); report["custom_app_id"] = custom_app_id\n'
    '        report["route_install_verification"] = train_install_and_verify_routes(txrx, route_plan, custom_app_id)\n'
    '\n'
    '        paths_bytes = np.asarray(paths, dtype="<i4", order="C").tobytes()\n'
    '        sched_bytes = np.asarray(schedule, dtype="<u4", order="C").tobytes()\n'
    '        path_offset = TR_CFG_BYTES\n'
    '        schedule_offset = align_up(path_offset + len(paths_bytes), 256)\n'
    '        output_offset = align_up(schedule_offset + len(sched_bytes), 256)\n'
    '        output_stride = TR_OUT_BYTES\n'
    '        region_size = align_up(output_offset + MAX_CORES * output_stride, 256)\n'
    '        report["per_chip_region_bytes"] = int(region_size)\n'
    '        report["total_path_upload_bytes"] = int(len(paths_bytes) * len(by_chip))\n'
    '        bases = {}\n'
    '        tload = time.perf_counter()\n'
    '        for (x,y), workers in sorted(by_chip.items()):\n'
    '            base = int(txrx.malloc_sdram(x,y,region_size,custom_app_id,tag=TR_SDRAM_TAG)); bases[(x,y)] = base\n'
    '            worker_by_p = {p: wid for wid,p in workers}\n'
    '            route_prefix = train_route_group_key(int(route_plan["root_index"][(x,y)]))\n'
    '            header = train_make_header(route_prefix,len(schedule),path_offset,schedule_offset,\n'
    '                                       output_offset,output_stride,worker_by_p,bcast_slots)\n'
    '            txrx.write_memory(x,y,base,header,n_bytes=len(header))\n'
    '            txrx.write_memory(x,y,base+path_offset,paths_bytes,n_bytes=len(paths_bytes))\n'
    '            txrx.write_memory(x,y,base+schedule_offset,sched_bytes,n_bytes=len(sched_bytes))\n'
    '            txrx.write_memory(x,y,base+output_offset,bytes(MAX_CORES*output_stride),n_bytes=MAX_CORES*output_stride)\n'
    '        report["training_data_load_s"] = float(time.perf_counter()-tload)\n'
    '        trun = time.perf_counter()\n'
    '        txrx.execute_flood(worker_core, str(aplx), custom_app_id, wait=False)\n'
    '        time.sleep(0.5); txrx.send_signal(custom_app_id, Signal.SYNC0)\n'
    '        txrx.wait_for_cores_to_be_in_state(worker_core, custom_app_id, CPUState.FINISHED, timeout=float(timeout_s))\n'
    '        report["board_training_wall_s"] = float(time.perf_counter()-trun)\n'
    '\n'
    '        w0 = placements[0]; w0cores = CoreSubsets(); w0cores.add_processor(w0[1],w0[2],w0[3])\n'
    '        (outdir/"worker0_iobuf.txt").write_text(train_read_iobuf(txrx,w0cores),encoding="utf-8",errors="replace")\n'
    '        raw = {}\n'
    '        for wid,x,y,p,mv in placements:\n'
    '            addr = bases[(x,y)] + output_offset + p*output_stride\n'
    '            rec = train_parse_output_record(bytes(txrx.read_memory(x,y,addr,TR_OUT_BYTES)))\n'
    '            if rec["worker"] != wid: raise RuntimeError(f"training worker mismatch {wid} vs {rec[\'worker\']}")\n'
    '            raw[wid] = rec\n'
    '        errors = [{"worker":w,"status":r["status"],"error":r["error"],"cycles_done":r["cycles_done"],\n'
    '                   "missing_e":r["missing_e"],"missing_m":r["missing_m"],"send_fail":r["send_fail"],\n'
    '                   "early_rx":r["early_rx"],"wrong_step":r["wrong_step"],"wrong_phase":r["wrong_phase"]}\n'
    '                  for w,r in raw.items() if r["status"] != TR_STATUS_DONE or r["error"] != 0]\n'
    '        report["n_worker_errors"] = len(errors); report["worker_errors"] = errors[:50]\n'
    '        report["transport_diagnostics"] = {\n'
    '            "early_rx_total": int(sum(r["early_rx"] for r in raw.values())),\n'
    '            "wrong_step_total": int(sum(r["wrong_step"] for r in raw.values())),\n'
    '            "wrong_phase_total": int(sum(r["wrong_phase"] for r in raw.values())),\n'
    '        }\n'
    '        if errors: raise RuntimeError(f"{len(errors)} training workers reported failure")\n'
    '        Wq = np.zeros((NV,NH),dtype=np.int32); Wcopy = np.zeros_like(Wq)\n'
    '        Hq = np.zeros((NH,NH),dtype=np.int32); Ereadq = np.zeros((3,N),dtype=np.int32)\n'
    '        for v in range(NV):\n'
    '            rec=raw[v]; Wq[v,:]=rec["weights_q24"][NV:]; Ereadq[:,v]=rec["Eread_q20"]\n'
    '        for h in range(NH):\n'
    '            rec=raw[NV+h]; Wcopy[:,h]=rec["weights_q24"][:NV]; Hq[h,:]=rec["weights_q24"][NV:]; Ereadq[:,NV+h]=rec["Eread_q20"]\n'
    '        Wdelta=Wq.astype(np.int64)-Wcopy.astype(np.int64); Hasym=Hq.astype(np.int64)-Hq.T.astype(np.int64)\n'
    '        consistency = {\n'
    '            "W_duplicate_max_raw_abs": int(np.max(np.abs(Wdelta))),\n'
    '            "H_asymmetry_max_raw_abs": int(np.max(np.abs(Hasym))),\n'
    '            "H_diag_max_raw_abs": int(np.max(np.abs(np.diag(Hq).astype(np.int64)))),\n'
    '            "pass": bool(np.all(Wdelta==0) and np.all(Hasym==0) and np.all(np.diag(Hq)==0)),\n'
    '        }\n'
    '        report["internal_consistency"] = consistency\n'
    '        if not consistency["pass"]: raise RuntimeError("training internal consistency failed")\n'
    '        np.savez_compressed(outdir/"PHASE1_BOARD_RAW_Q.npz",W_q24=Wq,W_hidden_copy_q24=Wcopy,H_q24=Hq,\n'
    '                            Eread_q20=Ereadq,schedule=schedule.astype(np.int16),prior=prior)\n'
    '        exposure=np.bincount(schedule,minlength=3).astype(np.int32)\n'
    '        raw_checkpoint=outdir/"training_checkpoint_raw.npz"\n'
    '        np.savez_compressed(raw_checkpoint,W=Wq.astype(np.float64)/(1<<24),H=Hq.astype(np.float64)/(1<<24),\n'
    '                            Eread=Ereadq.astype(np.float64)/(1<<20),prior=prior,exposure=exposure,reuse=cache_reuse,\n'
    '                            schedule=schedule,target_prior=prior,train_cycles=np.array([len(schedule)],np.int32),\n'
    '                            phase1_source=np.array(["SpiNNaker V4 from-zero learning"]),\n'
    '                            phase1_payload_sha256=np.array([sha256_file(payload_path)]),\n'
    '                            phase1_kernel_sha256=np.array([sha256_file(kernel_c)]))\n'
    '        report["train_pass"] = True; report["raw_checkpoint"] = str(raw_checkpoint)\n'
    '        json_dump(outdir/"PHASE1_TRAIN_REPORT.json",report)\n'
    '        return {"Wq":Wq,"Hq":Hq,"Ereadq":Ereadq,"prior":prior,"exposure":exposure,"report":report}\n'
    '    except Exception as exc:\n'
    '        report["train_pass"] = False; report["exception"] = repr(exc); report["traceback"] = traceback.format_exc()\n'
    '        json_dump(outdir/"PHASE1_TRAIN_REPORT.json",report)\n'
    '        raise\n'
    '    finally:\n'
    '        if txrx is not None and custom_app_id is not None:\n'
    '            try: txrx.stop_application(custom_app_id)\n'
    '            except Exception: pass\n'
    '        if sim_started:\n'
    '            try: sim.end()\n'
    '            except Exception: pass\n'
    '\n'
    '\n'
    'def bridge_training_to_clean_checkpoint(train_result: dict, outdir: Path) -> Path:\n'
    '    W = train_result["Wq"].astype(np.float64) / float(1 << 24)\n'
    '    H = train_result["Hq"].astype(np.float64) / float(1 << 24)\n'
    '    prior = np.asarray(train_result["prior"], dtype=np.float64)\n'
    '    exposure = np.asarray(train_result["exposure"], dtype=np.int32)\n'
    '    checkpoint = outdir / "training_checkpoint_clean.npz"\n'
    '    np.savez_compressed(\n'
    '        checkpoint,\n'
    '        Wvh=W,\n'
    '        Whh=H,\n'
    '        bv=np.zeros(NV,dtype=np.float64),\n'
    '        bh=np.zeros(NH,dtype=np.float64),\n'
    '        symbols=MNIST012_SYMBOLS.astype(np.float64),\n'
    '        prior_learned=prior,\n'
    '        exposure_counts=exposure,\n'
    '        Eread=train_result["Ereadq"].astype(np.float64)/float(1<<20),\n'
    '        target_prior=prior,\n'
    '        source=np.array(["from-zero SpiNNaker training -> frozen clean generator"]),\n'
    '    )\n'
    '    return checkpoint\n'
    '\n'
    '\n'
    'def run_clean_generation_from_checkpoint(checkpoint: Path, outdir: Path, n_images: int,\n'
    '                                         seed_start: int, generation_ms: int,\n'
    '                                         timer_tick_us: int, build_tree: Path | None,\n'
    '                                         timeout_s: float):\n'
    '    outdir.mkdir(parents=True,exist_ok=True)\n'
    '    cp = load_checkpoint(checkpoint)\n'
    '    axes = EMBEDDED_BOARD_AXES.copy()\n'
    '    symbols = np.asarray(cp["symbols"],dtype=np.float64)\n'
    '    prior = np.asarray(cp["prior_learned"],dtype=np.float64); prior = prior/prior.sum()\n'
    '    pop = SlowHerdingPopulation(1e-4,1e-4,1e-4); C=np.eye(3,dtype=float)\n'
    '    trace=[]; samples=[]; board=None\n'
    '    report={\n'
    '        "stage":"frozen spinnaker_dynamics_clean generation after from-zero training",\n'
    '        "checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),\n'
    '        "target_prior":prior.tolist(),"n_images":int(n_images),"seed_start":int(seed_start),\n'
    '        "generation_ms":int(generation_ms),"timer_tick_us":int(timer_tick_us),\n'
    '        "generator_source_sha256":VALIDATED_CLEAN_SOURCE_SHA256,\n'
    '        "embedded_axes_array_sha256":EMBEDDED_AXES_ARRAY_SHA256,\n'
    '        "digits":[0,1,2],\n'
    '    }\n'
    '    try:\n'
    '        board=G3Board(cp,axes,outdir,generation_ms=generation_ms,timer_tick_us=timer_tick_us,\n'
    '                      build_tree=build_tree,timeout_s=timeout_s)\n'
    '        report["resident_setup"]={\n'
    '            "map_load_s":board.map_load_s,"route_install_s":board.route_install_s,\n'
    '            "static_upload_s":board.static_upload_s,"launch_s":board.launch_s,\n'
    '            "setup_total_s":board.setup_total_s,"worker_chips":len(board.by_chip),\n'
    '            "packed_worker_cores":len(board.placements),"runtime_global_state_reconstruction":False,\n'
    '            "runtime_dense_Kx":False,"runtime_dense_A35_projection":False,"runtime_global_step_barrier":False,\n'
    '        }\n'
    '        for j in range(int(n_images)):\n'
    '            seed=int(seed_start+j); score=np.asarray(pop.score(),dtype=float); command=np.asarray(local_command(C,score),dtype=float)\n'
    '            C_before=C.copy(); xv,xh,events,timing,transport=board.run_image(seed,command,gamma=GAMMA)\n'
    '            mode,signed,corr=classify_visible(xv,symbols); qval=float(abs(signed))\n'
    '            C=update_local_couplings(C,score,mode); pop.update(mode,prior)\n'
    '            row={"j":j,"seed":seed,"mode":int(mode),"Q":qval,"signed_corr":float(signed),\n'
    '                 "corr_all":corr.tolist(),"score_pre":score.tolist(),"command_pre":command.tolist(),\n'
    '                 "C_before":C_before.tolist(),"C_after":C.tolist(),"timing":timing,"transport":transport}\n'
    '            trace.append(row); samples.append(xv.copy())\n'
    '            json_dump(outdir/"generation_trace.json",trace)\n'
    '            if (j+1)%10==0 or j==0 or j+1==n_images:\n'
    '                counts=np.bincount([r["mode"] for r in trace],minlength=3); pgen=counts/len(trace)\n'
    '                print(f"generate {j+1:03d}/{n_images} mode={mode} Q={qval:.6f} counts={counts.tolist()} TV={tv_distance(pgen,prior):.4f} board={timing[\'board_generation_s\']:.3f}s",flush=True)\n'
    '        arr=np.asarray(samples,dtype=np.float32)\n'
    '        np.savez_compressed(outdir/"generated_samples.npz",visible=arr)\n'
    '        save_montage(arr,outdir/"montage.png")\n'
    '        q=np.asarray([r["Q"] for r in trace],float); modes=np.asarray([r["mode"] for r in trace],int)\n'
    '        counts=np.bincount(modes,minlength=3); pgen=counts/len(modes)\n'
    '        execution_ok=all(\n'
    '            r["transport"]["send_fail_total"]==0 and r["transport"]["duplicate_total"]==0 and\n'
    '            r["transport"]["wrong_route_total"]==0 and r["transport"]["late_event_total"]==0 and\n'
    '            r["transport"]["queue_overflow_total"]==0 and r["transport"]["a35_missing_total"]==0 and\n'
    '            r["transport"]["a35_wrong_total"]==0 and r["transport"]["workers_complete"]==N_GROUPS\n'
    '            for r in trace)\n'
    '        btimes=np.asarray([r["timing"]["board_generation_s"] for r in trace],float)\n'
    '        etimes=np.asarray([r["timing"]["end_to_end_image_s"] for r in trace],float)\n'
    '        report["result"]={\n'
    '            "n":int(len(modes)),"counts":counts.astype(int).tolist(),"P_gen":pgen.tolist(),\n'
    '            "P_target":prior.tolist(),"TV":tv_distance(pgen,prior),\n'
    '            "Q_mean":float(q.mean()),"Q_median":float(np.median(q)),"Q_min":float(q.min()),"Q_max":float(q.max()),\n'
    '            "board_generation_mean_s":float(btimes.mean()),"board_generation_median_s":float(np.median(btimes)),\n'
    '            "end_to_end_image_mean_s":float(etimes.mean()),"end_to_end_image_median_s":float(np.median(etimes)),\n'
    '            "execution_integrity_pass":bool(execution_ok),\n'
    '            "closest_exact_uniform_counts_for_100":"any permutation of [34,33,33]" if int(n_images)==100 else None,\n'
    '        }\n'
    '        report["controller_final"]={"score":np.asarray(pop.score(),float).tolist(),"C":C.tolist(),\n'
    '                                    "spikes_rh":int(pop.spikes_rh),"spikes_n":int(pop.spikes_n),"spikes_a":int(pop.spikes_a)}\n'
    '        json_dump(outdir/"GENERATION_SUMMARY.json",report)\n'
    '        return report\n'
    '    finally:\n'
    '        if board is not None: board.close()\n'
    '\n'
    '# =============================================================================\n'
    '# One-click experiment\n'
    '# =============================================================================\n'
    'def parse_exact_uniform_prior(s: str) -> np.ndarray:\n'
    "    vals=np.asarray([float(x.strip()) for x in s.split(',')],dtype=np.float64)\n"
    '    if vals.shape!=(3,) or np.any(vals<0) or vals.sum()<=0:\n'
    '        raise ValueError("--prior requires three nonnegative values")\n'
    '    vals=vals/vals.sum()\n'
    '    return vals\n'
    '\n'
    '\n'
    'def main():\n'
    '    ap = argparse.ArgumentParser(\n'
    '        description=(\n'
    '            "standard mnist.npz -> distinct raw MNIST examples -> fresh dynamical paths -> "\n'
    '            "exact-zero SpiNNaker MSDL training -> persistent dynamical generation"\n'
    '        )\n'
    '    )\n'
    '    ap.add_argument(\n'
    '        "--mnist", type=Path, default=None,\n'
    '        help="standard Keras mnist.npz; auto-searched next to this script if omitted"\n'
    '    )\n'
    '    ap.add_argument(\n'
    '        "--prior", default="0.3333333333333333,0.3333333333333333,0.3333333333333334"\n'
    '    )\n'
    '    ap.add_argument("--train-cycles", type=int, default=99,\n'
    '                    help="99 gives exactly 33/33/33 distinct training exposures")\n'
    '    ap.add_argument("--data-seed", type=int, default=20261011,\n'
    '                    help="deterministic selection/shuffle of x_train examples")\n'
    '    ap.add_argument("--path-seed-start", type=int, default=20000,\n'
    '                    help="FreshPathProvider stochastic-dynamics seed start")\n'
    '    ap.add_argument("--path-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)),\n'
    '                    help="parallel host workers for fresh forward/noising path generation")\n'
    '    ap.add_argument("--pixel-preprocess", choices=["whitelam-rms", "unit"], default="whitelam-rms",\n'
    '                    help="fixed input scaling; whitelam-rms preserves the validated physical range")\n'
    '    ap.add_argument("--n-generate", type=int, default=100)\n'
    '    ap.add_argument("--train-timeout-s", type=float, default=1800.0)\n'
    '    ap.add_argument("--generation-timeout-s", type=float, default=120.0)\n'
    '    ap.add_argument("--bcast-slots", type=int, default=64)\n'
    '    ap.add_argument("--generation-ms", type=int, default=4320)\n'
    '    ap.add_argument("--timer-tick-us", type=int, default=100)\n'
    '    ap.add_argument("--seed-start", type=int, default=93000, help="generation seed start")\n'
    '    ap.add_argument("--build-tree", type=Path, default=None)\n'
    '    ap.add_argument("--output", type=Path, default=Path("MNIST_NPZ_ZERO_TO_GEN_MNIST012"))\n'
    '    args = ap.parse_args()\n'
    '\n'
    '    args.output.mkdir(parents=True, exist_ok=True)\n'
    '    prior = parse_exact_uniform_prior(args.prior)\n'
    '    if args.train_cycles <= 0:\n'
    '        raise ValueError("train-cycles must be positive")\n'
    '    if np.any(prior <= 0):\n'
    '        raise ValueError("all three prior entries must be >0 because the generator readout has three modes")\n'
    '    if np.allclose(prior, np.ones(3)/3, atol=1e-12) and args.train_cycles % 3 != 0:\n'
    '        print(\n'
    '            f"NOTE: train-cycles={args.train_cycles} cannot give exactly equal discrete exposures; "\n'
    '            "default 99 is exact.", flush=True\n'
    '        )\n'
    '\n'
    '    mnist_path = locate_mnist_npz(args.mnist)\n'
    '    print(f"MNIST input: {mnist_path}", flush=True)\n'
    '\n'
    '    print("\\n=== STAGE 0: STANDARD MNIST.NPZ -> DISTINCT FRESH DYNAMICAL PATHS ===", flush=True)\n'
    '    payload, payload_report, heldout = build_training_payload_from_mnist_npz(\n'
    '        mnist_path=mnist_path,\n'
    '        prior=prior,\n'
    '        cycles=int(args.train_cycles),\n'
    '        outdir=args.output,\n'
    '        seed_start=int(args.path_seed_start),\n'
    '        data_seed=int(args.data_seed),\n'
    '        path_workers=int(args.path_workers),\n'
    '        pixel_preprocess=str(args.pixel_preprocess),\n'
    '    )\n'
    '    xte012, yte012 = heldout\n'
    '    print(\n'
    '        "Fresh-path PASS. x_test/y_test were loaded only as held-out data and were not "\n'
    '        f"used for learning. Payload: {payload}", flush=True\n'
    '    )\n'
    '\n'
    '    print("\\n=== STAGE 1: EXACT-ZERO SPINNAKER MSDL TRAINING ===", flush=True)\n'
    '    train_dir = args.output / "01_training"\n'
    '    train_result = run_from_zero_training(\n'
    '        payload, train_dir, args.build_tree, args.train_timeout_s, args.bcast_slots\n'
    '    )\n'
    '    bridge = bridge_training_to_clean_checkpoint(train_result, args.output)\n'
    '    print(f"Training PASS. Clean checkpoint: {bridge}", flush=True)\n'
    '\n'
    '    print("\\n=== STAGE 2: PERSISTENT DYNAMICAL GENERATION ===", flush=True)\n'
    '    gen_dir = args.output / "02_generation"\n'
    '    gen_report = run_clean_generation_from_checkpoint(\n'
    '        bridge, gen_dir, args.n_generate, args.seed_start,\n'
    '        args.generation_ms, args.timer_tick_us, args.build_tree,\n'
    '        args.generation_timeout_s\n'
    '    )\n'
    '\n'
    '    final = {\n'
    '        "experiment": "standard mnist.npz -> fresh dynamics -> exact-zero MSDL training -> dynamical generation",\n'
    '        "digits": [0, 1, 2],\n'
    '        "mnist_input": str(mnist_path),\n'
    '        "mnist_file_sha256": payload_report["mnist"]["file_sha256"],\n'
    '        "mnist_full_train_n": payload_report["mnist"]["full_train_n"],\n'
    '        "mnist_full_test_n": payload_report["mnist"]["full_test_n"],\n'
    '        "mnist_train_012_counts": payload_report["mnist"]["train_012_counts"],\n'
    '        "mnist_test_012_counts": payload_report["mnist"]["test_012_counts"],\n'
    '        "held_out_test_used_for_learning": False,\n'
    '        "held_out_test_012_n": int(len(xte012)),\n'
    '        "pixel_preprocess": str(args.pixel_preprocess),\n'
    '        "external_path_cache_used": False,\n'
    '        "fresh_forward_noising_paths_generated": True,\n'
    '        "distinct_training_examples_until_pool_exhaustion": True,\n'
    '        "selected_train_indices": payload_report["selected_train_indices"],\n'
    '        "data_seed": int(args.data_seed),\n'
    '        "path_seed_start": int(args.path_seed_start),\n'
    '        "target_prior": prior.tolist(),\n'
    '        "train_cycles": int(args.train_cycles),\n'
    '        "training_exposure_counts": train_result["exposure"].astype(int).tolist(),\n'
    '        "weights_start_from_zero": True,\n'
    '        "host_learning_updates": 0,\n'
    '        "training_pass": bool(train_result["report"].get("train_pass", False)),\n'
    '        "board_training_wall_s": train_result["report"].get("board_training_wall_s"),\n'
    '        "clean_checkpoint": str(bridge),\n'
    '        "clean_checkpoint_sha256": sha256_file(bridge),\n'
    '        "n_generated": int(args.n_generate),\n'
    '        "generation": gen_report["result"],\n'
    '        "scientific_boundary": (\n'
    '            "This run is a new many-example standard-MNIST experiment. The Phase-I V4 learning "\n'
    '            "kernel and persistent generation kernel are unchanged, but historical quality/distribution "\n'
    '            "numbers from the old three-symbol experiment must be re-measured rather than assumed."\n'
    '        ),\n'
    '    }\n'
    '    json_dump(args.output / "FINAL_SUMMARY.json", final)\n'
    '    print("\\n=== STANDARD-MNIST ZERO-TO-GENERATION COMPLETE ===", flush=True)\n'
    '    print(json.dumps(final, indent=2), flush=True)\n'
    '    print(f"Output: {args.output.resolve()}", flush=True)\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    main()\n'
)

_EMBEDDED_PHASE1_TRAINER_SOURCE = (
    '#!/usr/bin/env python3\n'
    '# -*- coding: utf-8 -*-\n'
    '"""\n'
    'Phase I V4 — on-board from-zero training + few-shot generation.\n'
    '\n'
    'This is deliberately a TWO-part proof inside one Phase-I package:\n'
    '\n'
    'A) TRAIN ON SPINNAKER\n'
    '   - immutable reference forward/noising trajectories are loaded as training data;\n'
    '   - W and H are initialised to EXACT ZERO on every worker core;\n'
    '   - current full_pipeline.py A1 Izh dynamics, A4 target-prediction comparator,\n'
    '     and A2 local W/H updates run inside a custom SpiNNaker APLX;\n'
    '   - host performs orchestration/data movement only, never a learning update;\n'
    '   - learned W/H/Eread are read ONCE after all 100 cycles.\n'
    '\n'
    'B) FEW-SHOT GENERATION ON SPINNAKER\n'
    '   - the learned board checkpoint is passed to the already validated 9072-LIF\n'
    '     generator mapper (spinnaker_s1f_fast.py);\n'
    '   - default is 5 images; efficiency is NOT the target of Phase I.\n'
    '\n'
    'Phase II, not this file, is the persistent map-once / continuous-generation\n'
    'architecture with the slow controller closed on board.\n'
    '\n'
    'The custom training app uses direct explicit multicast trees built from the\n'
    'detected SpiNNaker machine links. V4 keeps the receive window open during\n'
    'Izh integration so small inter-core barrier skew cannot discard next-step data.  A 1296-neuron one-neuron-per-core carrier\n'
    'is used only for legal worker placement.  One spanning tree is built per\n'
    'source worker chip.  The packet key keeps the global worker id in low 16 bits,\n'
    'while mask 0xFFFF0000 groups all workers on one source chip onto one tree.\n'
    '\n'
    'IMPORTANT\n'
    '---------\n'
    'This implementation intentionally uses the cached canonical trajectories as\n'
    'READ-ONLY TRAINING INPUT. It proves on-board parameter learning from zero. It\n'
    'does not claim that the forward/noising trajectories themselves were generated\n'
    'on board during this Phase-I run. That is an explicit, auditable boundary.\n'
    '"""\n'
    'from __future__ import annotations\n'
    '\n'
    'import argparse\n'
    'import hashlib\n'
    'import importlib.util\n'
    'import json\n'
    'import os\n'
    'import shutil\n'
    'import struct\n'
    'import subprocess\n'
    'import sys\n'
    'import time\n'
    'import traceback\n'
    'from collections import defaultdict\n'
    'from pathlib import Path\n'
    'from typing import Dict, Iterable, Tuple\n'
    '\n'
    'import numpy as np\n'
    '\n'
    'from phase1_custom_broadcast_routes import (\n'
    '    ROUTE_MAGIC, ROUTE_GROUP_MASK, build_explicit_broadcast_routes,\n'
    '    install_and_verify_explicit_routes, route_group_key,\n'
    ')\n'
    '\n'
    'N = 1296\n'
    'NV = 784\n'
    'NH = 512\n'
    'NMODES = 3\n'
    'MAX_CORES = 18\n'
    'CFG_WORDS = 64\n'
    'CFG_BYTES = CFG_WORDS * 4\n'
    'OUT_W0 = 16\n'
    'OUT_WORDS = OUT_W0 + N\n'
    'OUT_BYTES = OUT_WORDS * 4\n'
    'MAGIC_CFG = 0x50315431\n'
    'MAGIC_OUT = 0x50314F31\n'
    'CFG_VERSION = 1\n'
    'SDRAM_TAG = 1\n'
    'BCAST_SLOTS_DEFAULT = 64\n'
    '\n'
    'STATUS_DONE = 3\n'
    'STATUS_ERROR = 0xDEAD0001\n'
    '\n'
    '\n'
    'def sha256_file(path: Path) -> str:\n'
    '    h = hashlib.sha256()\n'
    '    with path.open("rb") as f:\n'
    '        for block in iter(lambda: f.read(1 << 20), b""):\n'
    '            h.update(block)\n'
    '    return h.hexdigest()\n'
    '\n'
    '\n'
    'def align_up(n: int, a: int = 256) -> int:\n'
    '    return ((int(n) + a - 1) // a) * a\n'
    '\n'
    '\n'
    'def json_dump(path: Path, obj) -> None:\n'
    '    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")\n'
    '\n'
    '\n'
    'def load_module(path: Path, name="phase1_mod"):\n'
    '    spec = importlib.util.spec_from_file_location(name, path)\n'
    '    if spec is None or spec.loader is None:\n'
    '        raise RuntimeError(f"Could not import {path}")\n'
    '    m = importlib.util.module_from_spec(spec)\n'
    '    spec.loader.exec_module(m)\n'
    '    return m\n'
    '\n'
    '\n'
    'def load_payload(payload_path: Path):\n'
    '    d = np.load(payload_path, allow_pickle=True)\n'
    '    required = {"paths_q20", "schedule", "prior", "constants_json"}\n'
    '    missing = required - set(d.files)\n'
    '    if missing:\n'
    '        raise RuntimeError(f"payload missing {sorted(missing)}")\n'
    '    paths = np.asarray(d["paths_q20"], dtype=np.int32)\n'
    '    schedule = np.asarray(d["schedule"], dtype=np.int32)\n'
    '    prior = np.asarray(d["prior"], dtype=np.float64)\n'
    '    if paths.shape != (len(schedule), 28, N):\n'
    '        raise RuntimeError(f"unexpected paths shape {paths.shape}")\n'
    '    reuse = np.asarray(d["reuse"], dtype=np.int32) if "reuse" in d.files else np.zeros(3, dtype=np.int32)\n'
    '    return paths, schedule, prior, json.loads(str(d["constants_json"][0])), reuse\n'
    '\n'
    '\n'
    'def build_aplx(kernel_c: Path, outdir: Path, build_tree: Path | None) -> Path:\n'
    '    if build_tree is None:\n'
    '        build_tree = Path(os.environ.get("MSDL_GATE3_BUILD", "/home/jovyan/PGI/_gate3_build"))\n'
    '    if not build_tree.exists():\n'
    '        raise FileNotFoundError(\n'
    '            f"Known custom-APLX build tree not found: {build_tree}\\n"\n'
    '            "Set --build-tree or MSDL_GATE3_BUILD to the build tree that previously "\n'
    '            "built msdl_gate3.aplx.")\n'
    '\n'
    '    srcdir = build_tree / "src"\n'
    '    builddir = build_tree / "build"\n'
    '    srcdir.mkdir(parents=True, exist_ok=True)\n'
    '    target_src = srcdir / "msdl_gate3.c"\n'
    '    backup = target_src.read_bytes() if target_src.exists() else None\n'
    '    try:\n'
    '        target_src.write_bytes(kernel_c.read_bytes())\n'
    '        if builddir.exists():\n'
    '            shutil.rmtree(builddir)\n'
    '        builddir.mkdir(parents=True, exist_ok=True)\n'
    '\n'
    '        env = os.environ.copy()\n'
    '        env.setdefault(\n'
    '            "FEC_INSTALL_DIR",\n'
    '            "/home/jovyan/sPyNNakerGit/SpiNNFrontEndCommon/c_common/front_end_common_lib")\n'
    '        env.setdefault("SPINN_COMMON_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/spinn_common")\n'
    '        env.setdefault("SPINN_INSTALL_DIR", "/home/jovyan/sPyNNakerGit/spinnaker_tools")\n'
    '        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")\n'
    '\n'
    '        p = subprocess.run(\n'
    '            ["make"], cwd=build_tree, env=env, text=True,\n'
    '            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n'
    '        (outdir / "aplx_build.log").write_text(p.stdout, encoding="utf-8")\n'
    '        if p.returncode != 0:\n'
    '            raise RuntimeError("APLX build failed:\\n" + "\\n".join(p.stdout.splitlines()[-100:]))\n'
    '        candidates = list(build_tree.rglob("msdl_gate3.aplx"))\n'
    '        if not candidates:\n'
    '            raise FileNotFoundError("make succeeded but msdl_gate3.aplx was not produced")\n'
    '        built = max(candidates, key=lambda x: x.stat().st_mtime)\n'
    '        dest = outdir / "phase1_train_kernel.aplx"\n'
    '        shutil.copy2(built, dest)\n'
    '        return dest\n'
    '    finally:\n'
    '        if backup is None:\n'
    '            try:\n'
    '                target_src.unlink()\n'
    '            except FileNotFoundError:\n'
    '                pass\n'
    '        else:\n'
    '            target_src.write_bytes(backup)\n'
    '\n'
    '\n'
    'def _get_new_app_id(FecDataView):\n'
    '    try:\n'
    '        return FecDataView.get_new_id()\n'
    '    except TypeError:\n'
    '        return FecDataView().get_new_id()\n'
    '\n'
    '\n'
    'def _read_iobuf(txrx, cores) -> str:\n'
    '    chunks = []\n'
    '    for buf in txrx.get_iobuf(cores):\n'
    '        text = getattr(buf, "iobuf", None)\n'
    '        if text is None:\n'
    '            text = str(buf)\n'
    '        chunks.append(str(text))\n'
    '    return "\\n".join(chunks)\n'
    '\n'
    '\n'
    'def carrier_and_routes():\n'
    '    """Import sPyNNaker lazily and construct route-only carrier graph."""\n'
    '    try:\n'
    '        import pyNN.spiNNaker as sim\n'
    '        from spinn_front_end_common.data import FecDataView\n'
    '        from spynnaker.pyNN.extra_algorithms.splitter_components import (\n'
    '            SplitterPopulationVertexFixed,\n'
    '        )\n'
    '        from spinn_machine import CoreSubsets\n'
    '        from spinnman.model.enums import CPUState\n'
    '        try:\n'
    '            from spinnman.messages.scp.enums.signal import Signal\n'
    '        except ImportError:\n'
    '            from spinnman.messages.scp.enums import Signal\n'
    '    except Exception as exc:\n'
    '        raise RuntimeError(\n'
    '            "Run Phase-I train stage inside the working sPyNNaker 7.4.x environment") from exc\n'
    '    return sim, FecDataView, SplitterPopulationVertexFixed, CoreSubsets, CPUState, Signal\n'
    '\n'
    '\n'
    '\n'
    'def _route_sig(e):\n'
    '    return (int(e.key), int(e.mask), int(e.spinnaker_route), bool(e.defaultable))\n'
    '\n'
    '\n'
    'def snapshot_loaded_routes(FecDataView, txrx, app_id: int):\n'
    '    """Capture actual COMPRESSED multicast routes currently loaded on machine."""\n'
    '    snap = {}\n'
    '    for table in FecDataView.get_uncompressed().routing_tables:\n'
    '        routes = list(txrx.get_multicast_routes(int(table.x), int(table.y), int(app_id)))\n'
    '        if routes:\n'
    '            snap[(int(table.x), int(table.y))] = routes\n'
    '    if not snap:\n'
    '        raise RuntimeError("No multicast routes captured from PyNN carrier")\n'
    '    return snap\n'
    '\n'
    '\n'
    'def reinstall_and_verify_routes(txrx, snapshot, custom_app_id: int):\n'
    '    stats = {}\n'
    '    for (x, y), routes in sorted(snapshot.items()):\n'
    '        txrx.load_multicast_routes(x, y, routes, int(custom_app_id))\n'
    '        reread = list(txrx.get_multicast_routes(x, y, int(custom_app_id)))\n'
    '        a = sorted(_route_sig(e) for e in routes)\n'
    '        b = sorted(_route_sig(e) for e in reread)\n'
    '        stats[f"{x},{y}"] = {\n'
    '            "captured": len(a), "reloaded": len(b),\n'
    '            "bitwise_route_signature_equal": a == b,\n'
    '        }\n'
    '        if a != b:\n'
    '            raise RuntimeError(f"Route re-install verification failed on chip {(x,y)}")\n'
    '    return stats\n'
    '\n'
    '\n'
    'def make_header(\n'
    '    route_prefix: int, n_cycles: int, path_offset: int, schedule_offset: int,\n'
    '    output_offset: int, output_stride: int, worker_by_p: Dict[int, int],\n'
    '    bcast_slots: int,\n'
    ') -> bytes:\n'
    '    w = np.zeros(CFG_WORDS, dtype="<u4")\n'
    '    w[0] = MAGIC_CFG; w[1] = CFG_VERSION; w[2] = N; w[3] = NV; w[4] = NH\n'
    '    w[5] = n_cycles; w[6] = 28; w[7] = 27; w[8] = int(route_prefix)\n'
    '    w[9] = path_offset; w[10] = schedule_offset; w[11] = output_offset\n'
    '    w[12] = output_stride; w[13] = int(bcast_slots)\n'
    '    w[16:16 + MAX_CORES] = np.uint32(0xFFFFFFFF)\n'
    '    for p, wid in worker_by_p.items():\n'
    '        if not (0 <= int(p) < MAX_CORES):\n'
    '            raise RuntimeError(f"unexpected processor id p={p}")\n'
    '        w[16 + int(p)] = int(wid)\n'
    '    return w.tobytes()\n'
    '\n'
    '\n'
    'def parse_output_record(data: bytes):\n'
    '    a = np.frombuffer(data, dtype="<i4", count=OUT_WORDS).copy()\n'
    '    if int(np.uint32(a[0])) != MAGIC_OUT:\n'
    '        raise RuntimeError(f"bad output magic 0x{int(np.uint32(a[0])):08x}")\n'
    '    return {\n'
    '        "worker": int(a[1]),\n'
    '        "status": int(np.uint32(a[2])),\n'
    '        "error": int(np.uint32(a[3])),\n'
    '        "cycles_done": int(a[4]),\n'
    '        "missing_e": int(a[5]),\n'
    '        "missing_m": int(a[6]),\n'
    '        "send_fail": int(a[7]),\n'
    '        "Eread_q20": a[8:11].astype(np.int32),\n'
    '        "early_rx": int(np.uint32(a[11])),\n'
    '        "wrong_step": int(np.uint32(a[12])),\n'
    '        "wrong_phase": int(np.uint32(a[13])),\n'
    '        "phase_at_exit": int(np.uint32(a[14])),\n'
    '        "force_at_exit": int(np.uint32(a[15])),\n'
    '        "weights_q24": a[OUT_W0:].astype(np.int32),\n'
    '    }\n'
    '\n'
    '\n'
    'def compare_oracle(Wq, Hq, Ereadq, oracle_path: Path | None):\n'
    '    if oracle_path is None or not oracle_path.exists():\n'
    '        return {"available": False}\n'
    '    d = np.load(oracle_path, allow_pickle=True)\n'
    '    oW = np.asarray(d["W_q24"], dtype=np.int32)\n'
    '    oH = np.asarray(d["H_q24"], dtype=np.int32)\n'
    '    oE = np.asarray(d["Eread_q20"], dtype=np.int32)\n'
    '    dW = Wq.astype(np.int64) - oW.astype(np.int64)\n'
    '    dH = Hq.astype(np.int64) - oH.astype(np.int64)\n'
    '    dE = Ereadq.astype(np.int64) - oE.astype(np.int64)\n'
    '    return {\n'
    '        "available": True,\n'
    '        "W_max_raw_abs": int(np.max(np.abs(dW))),\n'
    '        "H_max_raw_abs": int(np.max(np.abs(dH))),\n'
    '        "Eread_max_raw_abs": int(np.max(np.abs(dE))),\n'
    '        "bit_exact": bool(np.all(dW == 0) and np.all(dH == 0) and np.all(dE == 0)),\n'
    '    }\n'
    '\n'
    '\n'
    'def run_board_training(\n'
    '    payload_path: Path, kernel_c: Path, outdir: Path,\n'
    '    build_tree: Path | None, oracle_path: Path | None,\n'
    '    timeout_s: float = 1200.0, bcast_slots: int = BCAST_SLOTS_DEFAULT,\n'
    '):\n'
    '    outdir.mkdir(parents=True, exist_ok=True)\n'
    '    paths, schedule, prior, constants, cache_reuse = load_payload(payload_path)\n'
    '    aplx = build_aplx(kernel_c, outdir, build_tree)\n'
    '\n'
    '    sim, FecDataView, SplitterPopulationVertexFixed, CoreSubsets, CPUState, Signal = carrier_and_routes()\n'
    '    sim_started = False\n'
    '    txrx = None\n'
    '    custom_app_id = None\n'
    '    report = {\n'
    '        "phase": "Phase-I on-board from-zero training v3 explicit broadcast trees",\n'
    '        "payload": str(payload_path),\n'
    '        "payload_sha256": sha256_file(payload_path),\n'
    '        "kernel_c_sha256": sha256_file(kernel_c),\n'
    '        "aplx_sha256": sha256_file(aplx),\n'
    '        "n_workers": N,\n'
    '        "cycles": int(len(schedule)),\n'
    '        "prior": prior.tolist(),\n'
    '        "weights_start_from_zero": True,\n'
    '        "host_learning_updates": 0,\n'
    '        "bcast_slots": int(bcast_slots),\n'
    '        "explicit_route_handoff": False,\n'
    '        "explicit_custom_broadcast_trees": True,\n'
    '    }\n'
    '\n'
    '    try:\n'
    '        sim.setup(timestep=1.0, min_delay=1.0, time_scale_factor=1)\n'
    '        sim_started = True\n'
    '        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 1)\n'
    '        carrier = sim.Population(\n'
    '            N,\n'
    '            sim.IF_curr_exp(\n'
    '                cm=1.0, tau_m=20.0, tau_refrac=1.0,\n'
    '                tau_syn_E=5.0, tau_syn_I=5.0,\n'
    '                v_rest=0.0, v_reset=0.0, v_thresh=1.0, i_offset=0.0),\n'
    '            label="phase1_route_carrier",\n'
    '            additional_parameters={"splitter": SplitterPopulationVertexFixed()},\n'
    '        )\n'
    '        # Carrier is used only for placement; no PyNN multicast projection is required.\n'
    '        t_map0 = time.perf_counter()\n'
    '        sim.run(1.0)\n'
    '        report["carrier_map_load_run_s"] = float(time.perf_counter() - t_map0)\n'
    '\n'
    '        app_vertex = carrier._vertex  # intentional low-level integration point\n'
    '        mverts = list(app_vertex.machine_vertices)\n'
    '        if len(mverts) != N:\n'
    '            raise RuntimeError(\n'
    '                f"Phase-I requires one atom/core: got {len(mverts)} machine vertices for {N} atoms")\n'
    '\n'
    '        placements = []\n'
    '        for mv in mverts:\n'
    '            sl = mv.vertex_slice\n'
    '            if int(sl.n_atoms) != 1:\n'
    '                raise RuntimeError(f"carrier machine slice is not one atom: {sl}")\n'
    '            wid = int(sl.lo_atom)\n'
    '            pl = FecDataView.get_placement_of_vertex(mv)\n'
    '            placements.append((wid, int(pl.x), int(pl.y), int(pl.p), mv))\n'
    '        placements.sort(key=lambda x: x[0])\n'
    '        if [x[0] for x in placements] != list(range(N)):\n'
    '            raise RuntimeError("carrier worker IDs are not exactly 0..1295")\n'
    '\n'
    '        txrx = FecDataView.get_transceiver()\n'
    '        pynn_app_id = int(FecDataView.get_app_id())\n'
    '        report["pynn_app_id"] = pynn_app_id\n'
    '\n'
    '        by_chip = defaultdict(list)\n'
    '        worker_core = CoreSubsets()\n'
    '        for wid, x, y, p, mv in placements:\n'
    '            by_chip[(x, y)].append((wid, p))\n'
    '            worker_core.add_processor(x, y, p)\n'
    '\n'
    '        report["actual_worker_chips"] = int(len(by_chip))\n'
    '        report["actual_worker_cores"] = int(len(placements))\n'
    '        report["workers_per_chip"] = {\n'
    '            f"{x},{y}": len(v) for (x, y), v in sorted(by_chip.items())}\n'
    '\n'
    '        machine = FecDataView.get_machine()\n'
    '        t_route_plan = time.perf_counter()\n'
    '        route_plan = build_explicit_broadcast_routes(machine, by_chip)\n'
    '        report["routing_scheme"] = "one-spanning-tree-per-source-worker-chip"\n'
    '        report["route_magic"] = int(ROUTE_MAGIC)\n'
    '        report["route_group_mask"] = int(ROUTE_GROUP_MASK)\n'
    '        report["planned_route_chips"] = int(len(route_plan["route_chips"]))\n'
    '        report["planned_route_entries_total"] = int(route_plan["total_entries"])\n'
    '        report["max_route_entries_per_chip"] = int(route_plan["max_entries_per_chip"])\n'
    '        report["route_tree_stats"] = route_plan["tree_stats"]\n'
    '        report["route_plan_s"] = float(time.perf_counter() - t_route_plan)\n'
    '        if route_plan["max_entries_per_chip"] >= 1023:\n'
    '            raise RuntimeError("custom route plan exceeds SpiNNaker router capacity")\n'
    '\n'
    '        txrx.stop_application(pynn_app_id)\n'
    '        time.sleep(0.25)\n'
    '        custom_app_id = int(_get_new_app_id(FecDataView))\n'
    '        report["custom_app_id"] = custom_app_id\n'
    '        t_route_load = time.perf_counter()\n'
    '        report["route_install_verification"] = install_and_verify_explicit_routes(\n'
    '            txrx, route_plan, custom_app_id)\n'
    '        report["route_install_s"] = float(time.perf_counter() - t_route_load)\n'
    '\n'
    '        paths_bytes = np.asarray(paths, dtype="<i4", order="C").tobytes()\n'
    '        sched_bytes = np.asarray(schedule, dtype="<u4", order="C").tobytes()\n'
    '        path_offset = CFG_BYTES\n'
    '        schedule_offset = align_up(path_offset + len(paths_bytes), 256)\n'
    '        output_offset = align_up(schedule_offset + len(sched_bytes), 256)\n'
    '        output_stride = OUT_BYTES\n'
    '        region_size = align_up(output_offset + MAX_CORES * output_stride, 256)\n'
    '        report["per_chip_region_bytes"] = int(region_size)\n'
    '        report["replicated_path_bytes_per_chip"] = int(len(paths_bytes))\n'
    '        report["total_path_upload_bytes"] = int(len(paths_bytes) * len(by_chip))\n'
    '\n'
    '        bases = {}\n'
    '        t_load0 = time.perf_counter()\n'
    '        for (x, y), workers in sorted(by_chip.items()):\n'
    '            base = int(txrx.malloc_sdram(x, y, region_size, custom_app_id, tag=SDRAM_TAG))\n'
    '            bases[(x, y)] = base\n'
    '            worker_by_p = {p: wid for wid, p in workers}\n'
    '            root_idx = int(route_plan["root_index"][(x, y)])\n'
    '            route_prefix = int(route_group_key(root_idx))\n'
    '            header = make_header(\n'
    '                route_prefix, len(schedule), path_offset, schedule_offset,\n'
    '                output_offset, output_stride, worker_by_p, bcast_slots)\n'
    '            txrx.write_memory(x, y, base, header, n_bytes=len(header))\n'
    '            txrx.write_memory(x, y, base + path_offset, paths_bytes, n_bytes=len(paths_bytes))\n'
    '            txrx.write_memory(x, y, base + schedule_offset, sched_bytes, n_bytes=len(sched_bytes))\n'
    '            txrx.write_memory(\n'
    '                x, y, base + output_offset,\n'
    '                bytes(MAX_CORES * output_stride),\n'
    '                n_bytes=MAX_CORES * output_stride)\n'
    '        report["custom_training_data_load_s"] = float(time.perf_counter() - t_load0)\n'
    '\n'
    '        # Launch all 1296 worker instances of the same APLX.\n'
    '        t_run0 = time.perf_counter()\n'
    '        txrx.execute_flood(worker_core, str(aplx), custom_app_id, wait=False)\n'
    '        time.sleep(0.5)\n'
    '        txrx.send_signal(custom_app_id, Signal.SYNC0)\n'
    '        txrx.wait_for_cores_to_be_in_state(\n'
    '            worker_core, custom_app_id, CPUState.FINISHED,\n'
    '            timeout=float(timeout_s))\n'
    '        report["custom_training_wall_s"] = float(time.perf_counter() - t_run0)\n'
    '\n'
    '        # Collect one concise IOBUF (worker 0) and every output record.\n'
    '        w0 = placements[0]\n'
    '        w0cores = CoreSubsets(); w0cores.add_processor(w0[1], w0[2], w0[3])\n'
    '        iobuf = _read_iobuf(txrx, w0cores)\n'
    '        (outdir / "worker0_iobuf.txt").write_text(iobuf, encoding="utf-8", errors="replace")\n'
    '\n'
    '        raw = {}\n'
    '        statuses = []\n'
    '        for wid, x, y, p, mv in placements:\n'
    '            base = bases[(x, y)]\n'
    '            addr = base + output_offset + p * output_stride\n'
    '            rec = parse_output_record(bytes(txrx.read_memory(x, y, addr, OUT_BYTES)))\n'
    '            if rec["worker"] != wid:\n'
    '                raise RuntimeError(f"worker mismatch at {(x,y,p)}: expected {wid}, got {rec[\'worker\']}")\n'
    '            raw[wid] = rec\n'
    '            statuses.append(rec["status"])\n'
    '\n'
    '        errors = [\n'
    '            {"worker": w, "status": r["status"], "error": r["error"],\n'
    '             "cycles_done": r["cycles_done"], "missing_e": r["missing_e"],\n'
    '             "missing_m": r["missing_m"], "send_fail": r["send_fail"],\n'
    '             "early_rx": r["early_rx"], "wrong_step": r["wrong_step"],\n'
    '             "wrong_phase": r["wrong_phase"],\n'
    '             "phase_at_exit": r["phase_at_exit"],\n'
    '             "force_at_exit": r["force_at_exit"]}\n'
    '            for w, r in raw.items()\n'
    '            if r["status"] != STATUS_DONE or r["error"] != 0\n'
    '        ]\n'
    '        report["worker_errors"] = errors[:50]\n'
    '        report["n_worker_errors"] = int(len(errors))\n'
    '        report["transport_diagnostics"] = {\n'
    '            "early_rx_total": int(sum(r["early_rx"] for r in raw.values())),\n'
    '            "early_rx_max_per_worker": int(max(r["early_rx"] for r in raw.values())),\n'
    '            "wrong_step_total": int(sum(r["wrong_step"] for r in raw.values())),\n'
    '            "wrong_phase_total": int(sum(r["wrong_phase"] for r in raw.values())),\n'
    '        }\n'
    '        if errors:\n'
    '            raise RuntimeError(f"{len(errors)} Phase-I worker(s) reported failure; see report")\n'
    '\n'
    '        Wq = np.zeros((NV, NH), dtype=np.int32)\n'
    '        Wq_hidden_copy = np.zeros((NV, NH), dtype=np.int32)\n'
    '        Hq = np.zeros((NH, NH), dtype=np.int32)\n'
    '        Ereadq = np.zeros((NMODES, N), dtype=np.int32)\n'
    '        for v in range(NV):\n'
    '            rec = raw[v]\n'
    '            Wq[v, :] = rec["weights_q24"][NV:]\n'
    '            Ereadq[:, v] = rec["Eread_q20"]\n'
    '        for h in range(NH):\n'
    '            rec = raw[NV + h]\n'
    '            Wq_hidden_copy[:, h] = rec["weights_q24"][:NV]\n'
    '            Hq[h, :] = rec["weights_q24"][NV:]\n'
    '            Ereadq[:, NV + h] = rec["Eread_q20"]\n'
    '\n'
    '        W_copy_delta = Wq.astype(np.int64) - Wq_hidden_copy.astype(np.int64)\n'
    '        H_asym = Hq.astype(np.int64) - Hq.T.astype(np.int64)\n'
    '        report["internal_consistency"] = {\n'
    '            "W_duplicate_max_raw_abs": int(np.max(np.abs(W_copy_delta))),\n'
    '            "H_asymmetry_max_raw_abs": int(np.max(np.abs(H_asym))),\n'
    '            "H_diag_max_raw_abs": int(np.max(np.abs(np.diag(Hq).astype(np.int64)))),\n'
    '            "pass": bool(\n'
    '                np.all(W_copy_delta == 0) and np.all(H_asym == 0) and np.all(np.diag(Hq) == 0)),\n'
    '        }\n'
    '        oracle_cmp = compare_oracle(Wq, Hq, Ereadq, oracle_path)\n'
    '        report["fixedpoint_oracle_comparison"] = oracle_cmp\n'
    '\n'
    '        np.savez_compressed(\n'
    '            outdir / "PHASE1_BOARD_RAW_Q.npz",\n'
    '            W_q24=Wq, W_hidden_copy_q24=Wq_hidden_copy,\n'
    '            H_q24=Hq, Eread_q20=Ereadq,\n'
    '            schedule=schedule.astype(np.int16), prior=prior)\n'
    '\n'
    '        # Export canonical checkpoint schema. This is the only learned checkpoint\n'
    '        # handed to the generation stage.\n'
    '        exposure = np.bincount(schedule, minlength=3).astype(np.int32)\n'
    '        checkpoint = outdir / "training_checkpoint.npz"\n'
    '        np.savez_compressed(\n'
    '            checkpoint,\n'
    '            W=Wq.astype(np.float64) / (1 << 24),\n'
    '            H=Hq.astype(np.float64) / (1 << 24),\n'
    '            Eread=Ereadq.astype(np.float64) / (1 << 20),\n'
    '            prior=prior,\n'
    '            exposure=exposure,\n'
    '            reuse=cache_reuse.astype(np.int32),\n'
    '            schedule=schedule.astype(np.int32),\n'
    '            spikes=np.array([0, 0], dtype=np.int32),\n'
    '            target_prior=prior,\n'
    '            train_cycles=np.array([len(schedule)], dtype=np.int32),\n'
    '            phase1_source=np.array(["SpiNNaker custom APLX from-zero learning"]),\n'
    '            phase1_payload_sha256=np.array([sha256_file(payload_path)]),\n'
    '            phase1_kernel_sha256=np.array([sha256_file(kernel_c)]),\n'
    '        )\n'
    '        report["checkpoint"] = str(checkpoint)\n'
    '        report["checkpoint_sha256"] = sha256_file(checkpoint)\n'
    '        report["train_pass"] = bool(\n'
    '            report["internal_consistency"]["pass"] and\n'
    '            (not oracle_cmp.get("available") or oracle_cmp.get("bit_exact", False)))\n'
    '        json_dump(outdir / "PHASE1_TRAIN_REPORT.json", report)\n'
    '        print(json.dumps(report, indent=2), flush=True)\n'
    '        return checkpoint, report\n'
    '\n'
    '    except Exception as exc:\n'
    '        report["train_pass"] = False\n'
    '        report["exception"] = repr(exc)\n'
    '        report["traceback"] = traceback.format_exc()\n'
    '        json_dump(outdir / "PHASE1_TRAIN_REPORT.json", report)\n'
    '        raise\n'
    '    finally:\n'
    '        if txrx is not None and custom_app_id is not None:\n'
    '            try:\n'
    '                txrx.stop_application(custom_app_id)\n'
    '            except Exception:\n'
    '                pass\n'
    '        if sim_started:\n'
    '            try:\n'
    '                sim.end()\n'
    '            except Exception:\n'
    '                pass\n'
    '\n'
    '\n'
    'def run_generation(\n'
    '    checkpoint: Path, s1f_script: Path, full_pipeline: Path,\n'
    '    axes: Path | None, n_generate: int, outdir: Path, neurons_per_core: int,\n'
    '):\n'
    '    outdir.mkdir(parents=True, exist_ok=True)\n'
    '    cmd = [\n'
    '        sys.executable, str(s1f_script),\n'
    '        "--gate", "g3",\n'
    '        "--n-generate", str(int(n_generate)),\n'
    '        "--full-pipeline", str(full_pipeline),\n'
    '        "--checkpoint", str(checkpoint),\n'
    '        "--neurons-per-core", str(int(neurons_per_core)),\n'
    '        "--output", str(outdir),\n'
    '    ]\n'
    '    if axes is not None:\n'
    '        cmd += ["--axes", str(axes)]\n'
    '    print("RUN:", " ".join(cmd), flush=True)\n'
    '    p = subprocess.run(cmd, text=True)\n'
    '    if p.returncode != 0:\n'
    '        raise RuntimeError(f"few-shot generation failed with exit code {p.returncode}")\n'
    '\n'
    '\n'
    'def main():\n'
    '    ap = argparse.ArgumentParser(description="Phase-I on-board from-zero MSDL training")\n'
    '    ap.add_argument("--stage", choices=["preflight", "train", "generate", "all"], default="preflight")\n'
    '    ap.add_argument("--full-pipeline", type=Path, default=Path("full_pipeline.py"))\n'
    '    ap.add_argument("--paths", type=Path, default=Path("HADAPT_REFERENCE_PATHS.npz"))\n'
    '    ap.add_argument("--payload", type=Path, default=None)\n'
    '    ap.add_argument("--oracle", type=Path, default=None)\n'
    '    ap.add_argument("--reference-checkpoint", type=Path, default=None,\n'
    '                    help="optional floating cloud checkpoint for preflight numerical comparison only")\n'
    '    ap.add_argument("--kernel-c", type=Path, default=Path("phase1_train_kernel_v4.c"))\n'
    '    ap.add_argument("--build-tree", type=Path, default=None)\n'
    '    ap.add_argument("--s1f-script", type=Path, default=Path("spinnaker_s1f_fast.py"))\n'
    '    ap.add_argument("--axes", type=Path, default=Path("basin_axes_used.npz"))\n'
    '    ap.add_argument("--n-generate", type=int, default=5)\n'
    '    ap.add_argument("--neurons-per-core", type=int, default=16)\n'
    '    ap.add_argument("--timeout-s", type=float, default=1200.0)\n'
    '    ap.add_argument("--bcast-slots", type=int, default=BCAST_SLOTS_DEFAULT,\n'
    '                    help="broadcast serialization slots; use the smallest value that passed P0")\n'
    '    ap.add_argument("--output", type=Path, default=Path("PHASE1_RUN"))\n'
    '    args = ap.parse_args()\n'
    '    args.output.mkdir(parents=True, exist_ok=True)\n'
    '\n'
    '    prep_dir = args.output / "prep"\n'
    '    train_dir = args.output / "train"\n'
    '    gen_dir = args.output / "generation"\n'
    '\n'
    '    payload = args.payload\n'
    '    oracle = args.oracle\n'
    '    if args.stage in {"preflight", "all"} or payload is None:\n'
    '        prep_dir.mkdir(parents=True, exist_ok=True)\n'
    '        oracle_script = Path(__file__).with_name("phase1_fixedpoint_oracle.py")\n'
    '        cmd = [\n'
    '            sys.executable, str(oracle_script),\n'
    '            "--full-pipeline", str(args.full_pipeline),\n'
    '            "--paths", str(args.paths),\n'
    '            "--cycles", "100",\n'
    '            "--output", str(prep_dir),\n'
    '        ]\n'
    '        if args.reference_checkpoint is not None:\n'
    '            cmd += ["--reference-checkpoint", str(args.reference_checkpoint)]\n'
    '        print("RUN:", " ".join(cmd), flush=True)\n'
    '        subprocess.run(cmd, check=True)\n'
    '        payload = prep_dir / "phase1_training_payload.npz"\n'
    '        oracle = prep_dir / "VALIDATION_ONLY_phase1_fixedpoint_oracle.npz"\n'
    '        if args.stage == "preflight":\n'
    '            print("PHASE-I PREFLIGHT COMPLETE. No board contacted.", flush=True)\n'
    '            return\n'
    '\n'
    '    checkpoint = train_dir / "training_checkpoint.npz"\n'
    '    if args.stage in {"train", "all"}:\n'
    '        checkpoint, report = run_board_training(\n'
    '            payload, args.kernel_c, train_dir, args.build_tree, oracle,\n'
    '            timeout_s=args.timeout_s, bcast_slots=args.bcast_slots)\n'
    '        if not report.get("train_pass", False):\n'
    '            raise RuntimeError("Phase-I training gate did not pass")\n'
    '        if args.stage == "train":\n'
    '            return\n'
    '\n'
    '    if args.stage in {"generate", "all"}:\n'
    '        if not checkpoint.exists():\n'
    '            raise FileNotFoundError(\n'
    '                f"Board-trained checkpoint not found: {checkpoint}. Run --stage train first.")\n'
    '        axes = args.axes if args.axes is not None and args.axes.exists() else None\n'
    '        run_generation(\n'
    '            checkpoint, args.s1f_script, args.full_pipeline, axes,\n'
    '            args.n_generate, gen_dir, args.neurons_per_core)\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    main()\n'
)

_EMBEDDED_PHASE1_ROUTES_SOURCE = (
    '#!/usr/bin/env python3\n'
    'from __future__ import annotations\n'
    '\n'
    'from collections import defaultdict, deque\n'
    '\n'
    'ROUTE_MAGIC = 0xA5000000\n'
    'ROUTE_MAGIC_MASK = 0xFF000000\n'
    'ROUTE_GROUP_MASK = 0xFFFF0000\n'
    'MAX_ROOTS = 0x100\n'
    '\n'
    '\n'
    'def route_group_key(root_index: int) -> int:\n'
    '    if not 0 <= int(root_index) < MAX_ROOTS:\n'
    '        raise ValueError(f"root_index out of range: {root_index}")\n'
    '    return ROUTE_MAGIC | (int(root_index) << 16)\n'
    '\n'
    '\n'
    'def worker_packet_key(root_index: int, worker_id: int) -> int:\n'
    '    if not 0 <= int(worker_id) < 0x10000:\n'
    '        raise ValueError(f"worker_id out of range: {worker_id}")\n'
    '    return route_group_key(root_index) | int(worker_id)\n'
    '\n'
    '\n'
    'def build_machine_adjacency(machine):\n'
    '    coords = {(int(ch.x), int(ch.y)) for ch in machine.chips}\n'
    '    adjacency = {xy: [] for xy in coords}\n'
    '    for ch in machine.chips:\n'
    '        src = (int(ch.x), int(ch.y))\n'
    '        for link in ch.router.links:\n'
    '            dst = (int(link.destination_x), int(link.destination_y))\n'
    '            if dst in coords:\n'
    '                adjacency[src].append((dst, int(link.source_link_id)))\n'
    '    return adjacency\n'
    '\n'
    '\n'
    'def bfs_parent_tree(adjacency, root):\n'
    '    root = tuple(root)\n'
    '    parent = {root: None}\n'
    '    parent_link = {}\n'
    '    q = deque([root])\n'
    '    while q:\n'
    '        u = q.popleft()\n'
    '        for v, link_id in adjacency.get(u, ()):\n'
    '            if v in parent:\n'
    '                continue\n'
    '            parent[v] = u\n'
    '            parent_link[v] = int(link_id)\n'
    '            q.append(v)\n'
    '    return parent, parent_link\n'
    '\n'
    '\n'
    'def pruned_tree_to_targets(adjacency, root, targets):\n'
    '    root = tuple(root)\n'
    '    targets = {tuple(t) for t in targets}\n'
    '    parent, parent_link = bfs_parent_tree(adjacency, root)\n'
    '    missing = sorted(t for t in targets if t not in parent)\n'
    '    if missing:\n'
    '        raise RuntimeError(f"root {root} cannot reach worker chips {missing[:8]}" + ("..." if len(missing) > 8 else ""))\n'
    '\n'
    '    required = {root}\n'
    '    for t in targets:\n'
    '        u = t\n'
    '        while u is not None and u not in required:\n'
    '            required.add(u)\n'
    '            u = parent[u]\n'
    '        if u is None and root not in required:\n'
    '            raise RuntimeError(f"broken parent chain from {t} to {root}")\n'
    '\n'
    '    child_links = defaultdict(list)\n'
    '    for node in required:\n'
    '        if node == root:\n'
    '            continue\n'
    '        p = parent[node]\n'
    '        if p not in required:\n'
    '            raise RuntimeError("internal required-tree inconsistency")\n'
    '        child_links[p].append(parent_link[node])\n'
    '\n'
    '    return required, {k: sorted(set(v)) for k, v in child_links.items()}\n'
    '\n'
    '\n'
    'def build_explicit_broadcast_routes(machine, by_chip):\n'
    '    """\n'
    '    Build one multicast spanning tree per *source worker chip*.\n'
    '\n'
    '    Packet key layout:\n'
    '        0xA5 RR WWWW\n'
    '        RR   = source worker-chip index (0..255)\n'
    '        WWWW = global worker id (0..65535)\n'
    '\n'
    '    Router entries use mask 0xFFFF0000, so all workers on the same source chip\n'
    '    share one multicast tree, while the low 16 key bits still preserve the\n'
    '    global worker id for the receiving learning kernel.\n'
    '    """\n'
    '    from spinn_machine import MulticastRoutingEntry, RoutingEntry\n'
    '\n'
    '    worker_chips = sorted(tuple(xy) for xy in by_chip)\n'
    '    if len(worker_chips) >= MAX_ROOTS:\n'
    '        raise RuntimeError(f"too many worker chips for key format: {len(worker_chips)}")\n'
    '    root_index = {xy: i for i, xy in enumerate(worker_chips)}\n'
    '    adjacency = build_machine_adjacency(machine)\n'
    '\n'
    '    routes_by_chip = defaultdict(list)\n'
    '    tree_stats = {}\n'
    '    total_worker_processors = sum(len(v) for v in by_chip.values())\n'
    '\n'
    '    for root in worker_chips:\n'
    '        ridx = root_index[root]\n'
    '        required, child_links = pruned_tree_to_targets(adjacency, root, worker_chips)\n'
    '        group_key = route_group_key(ridx)\n'
    '        delivered_processors = 0\n'
    '        for node in sorted(required):\n'
    '            processors = sorted(int(p) for _, p in by_chip.get(node, ()))\n'
    '            links = child_links.get(node, [])\n'
    '            if processors or links:\n'
    '                routes_by_chip[node].append(\n'
    '                    MulticastRoutingEntry(\n'
    '                        group_key,\n'
    '                        ROUTE_GROUP_MASK,\n'
    '                        RoutingEntry(processor_ids=processors, link_ids=links)))\n'
    '                delivered_processors += len(processors)\n'
    '        if delivered_processors != total_worker_processors:\n'
    '            raise RuntimeError(\n'
    '                f"root {root} route covers {delivered_processors} worker processors; expected {total_worker_processors}")\n'
    '        n_edges = sum(len(v) for v in child_links.values())\n'
    '        if n_edges != len(required) - 1:\n'
    '            raise RuntimeError(f"root {root} tree is not a tree: nodes={len(required)} edges={n_edges}")\n'
    '        tree_stats[f"{root[0]},{root[1]}"] = {\n'
    '            "root_index": ridx,\n'
    '            "group_key": group_key,\n'
    '            "nodes": len(required),\n'
    '            "edges": n_edges,\n'
    '            "worker_chips_reached": len(worker_chips),\n'
    '            "worker_processors_reached": delivered_processors,\n'
    '        }\n'
    '\n'
    '    max_entries = max((len(v) for v in routes_by_chip.values()), default=0)\n'
    '    return {\n'
    '        "routes_by_chip": dict(routes_by_chip),\n'
    '        "root_index": root_index,\n'
    '        "tree_stats": tree_stats,\n'
    '        "worker_chips": worker_chips,\n'
    '        "route_chips": sorted(routes_by_chip),\n'
    '        "total_entries": sum(len(v) for v in routes_by_chip.values()),\n'
    '        "max_entries_per_chip": max_entries,\n'
    '    }\n'
    '\n'
    '\n'
    'def route_sig(e):\n'
    '    return (int(e.key), int(e.mask), int(e.spinnaker_route), bool(e.defaultable))\n'
    '\n'
    '\n'
    'def install_and_verify_explicit_routes(txrx, plan, custom_app_id: int):\n'
    '    stats = {}\n'
    '    for (x, y), routes in sorted(plan["routes_by_chip"].items()):\n'
    '        txrx.load_multicast_routes(int(x), int(y), list(routes), int(custom_app_id))\n'
    '        reread = list(txrx.get_multicast_routes(int(x), int(y), int(custom_app_id)))\n'
    '        a = sorted(route_sig(e) for e in routes)\n'
    '        b = sorted(route_sig(e) for e in reread)\n'
    '        stats[f"{x},{y}"] = {\n'
    '            "planned": len(a),\n'
    '            "reloaded": len(b),\n'
    '            "bitwise_route_signature_equal": a == b,\n'
    '        }\n'
    '        if a != b:\n'
    '            raise RuntimeError(f"explicit route verification failed on chip {(x, y)}")\n'
    '    return stats\n'
)

_EMBEDDED_PHASE1_KERNEL_SOURCE = (
    '/*\n'
    ' * Phase-I V4 custom SpiNNaker-1 training kernel for Physicodynamic MSDL.\n'
    ' *\n'
    ' * CURRENT SCIENTIFIC MODEL ONLY:\n'
    ' *   - current full_pipeline.py A1 centered Izhikevich dynamics\n'
    ' *   - A4 target - prediction comparator current\n'
    ' *   - A2 local W/H update\n'
    ' *   - W/H start from exactly zero\n'
    ' *   - path-start weights are frozen for all 27 force samples of a path\n'
    ' *\n'
    ' * Training trajectories are immutable read-only input data placed in SDRAM by\n'
    ' * the host orchestrator.  Learning arithmetic is performed on the SpiNNaker\n'
    ' * cores.  Every logical coordinate is one worker core.\n'
    ' *\n'
    ' * IMPORTANT: The validated current Izh regime is subthreshold.  This kernel\n'
    ' * intentionally fails if the centered state leaves a conservative range,\n'
    ' * rather than silently implement a different spike/reset approximation.\n'
    ' */\n'
    '#include <spin1_api.h>\n'
    '#include <sark.h>\n'
    '#include <stdint.h>\n'
    '#include <stdbool.h>\n'
    '\n'
    '#define MAGIC_CFG       0x50315431u  /* P1T1 */\n'
    '#define MAGIC_OUT       0x50314f31u  /* P1O1 */\n'
    '#define CFG_VERSION     1u\n'
    '#define SDRAM_TAG       1u\n'
    '\n'
    '#define N_WORKERS       1296u\n'
    '#define NV              784u\n'
    '#define NH              512u\n'
    '#define N_MODES         3u\n'
    '#define PATH_STATES     28u\n'
    '#define FORCE_STEPS     27u\n'
    '#define IZH_SUBSTEPS    80u\n'
    '#define MAX_CORES       18u\n'
    '#define SEEN_WORDS      41u\n'
    '#define BCAST_SLOTS_DFLT 64u\n'
    '#define WAIT_TIMEOUT_TICKS 1000u\n'
    '\n'
    '#define Q_PATH          20\n'
    '#define Q_WEIGHT        24\n'
    '#define Q_STATE         30\n'
    '#define Q_READ          16\n'
    '\n'
    '/* Frozen integer mapping generated by phase1_fixedpoint_oracle.py */\n'
    '#define IZH_Q_Q30       ((int32_t)-6461475)\n'
    '#define IZH_A_Q30       ((int32_t)1073742)\n'
    '#define IZH_B_Q30       ((int32_t)225822)\n'
    '#define C004_Q30        ((int32_t)42949673)\n'
    '#define GAIN_E_Q30      ((int32_t)10737)\n'
    '#define GAIN_M_Q30      ((int32_t)1074)\n'
    '#define CV_E_Q16        ((int32_t)29461671)\n'
    '#define CUA_E_Q16       ((int32_t)53987929)\n'
    '#define CV_M_Q16        ((int32_t)294616714)\n'
    '#define CUA_M_Q16       ((int32_t)539879288)\n'
    '#define INV_MU_DT_Q20   ((int32_t)10708440)\n'
    '#define ALPHA015_Q20    ((int32_t)157286)\n'
    '#define LR_Q24          ((int32_t)167772)\n'
    '#define H_LR_Q24        ((int32_t)25166)\n'
    '\n'
    '/* Config word layout in tagged SDRAM. */\n'
    '#define CFG_MAGIC               0u\n'
    '#define CFG_VER                 1u\n'
    '#define CFG_NWORKERS            2u\n'
    '#define CFG_NV                  3u\n'
    '#define CFG_NH                  4u\n'
    '#define CFG_NCYCLES             5u\n'
    '#define CFG_PATH_STATES         6u\n'
    '#define CFG_FORCE_STEPS         7u\n'
    '#define CFG_BASE_KEY            8u\n'
    '#define CFG_PATH_OFFSET         9u\n'
    '#define CFG_SCHED_OFFSET       10u\n'
    '#define CFG_OUTPUT_OFFSET      11u\n'
    '#define CFG_OUTPUT_STRIDE      12u\n'
    '#define CFG_BCAST_SLOTS        13u\n'
    '#define CFG_RESERVED14         14u\n'
    '#define CFG_RESERVED15         15u\n'
    '#define CFG_WORKER_BY_P        16u  /* 18 words: 16..33 */\n'
    '#define CFG_WORDS              64u\n'
    '#define CFG_BYTES              (CFG_WORDS * 4u)\n'
    '\n'
    '/* Output record layout; weights start at OUT_W0. */\n'
    '#define OUT_MAGIC               0u\n'
    '#define OUT_WORKER              1u\n'
    '#define OUT_STATUS              2u\n'
    '#define OUT_ERROR               3u\n'
    '#define OUT_CYCLES_DONE         4u\n'
    '#define OUT_MISSING_E           5u\n'
    '#define OUT_MISSING_M           6u\n'
    '#define OUT_SEND_FAIL           7u\n'
    '#define OUT_EREAD0              8u\n'
    '#define OUT_EREAD1              9u\n'
    '#define OUT_EREAD2             10u\n'
    '#define OUT_EARLY_RX           11u  /* current-step packets accepted during INTEGRATE */\n'
    '#define OUT_WRONG_STEP         12u  /* packets rejected because step mismatched */\n'
    '#define OUT_WRONG_PHASE        13u  /* packets rejected only after FINISHED */\n'
    '#define OUT_PHASE_AT_EXIT      14u\n'
    '#define OUT_FORCE_AT_EXIT      15u\n'
    '#define OUT_W0                 16u\n'
    '#define OUT_WORDS              (OUT_W0 + N_WORKERS)\n'
    '#define OUT_BYTES              (OUT_WORDS * 4u)\n'
    '\n'
    '#define STATUS_INIT             1u\n'
    '#define STATUS_RUNNING          2u\n'
    '#define STATUS_DONE             3u\n'
    '#define STATUS_ERROR            0xdead0001u\n'
    '\n'
    '#define ERR_NONE                0u\n'
    '#define ERR_CONFIG              (1u << 0)\n'
    '#define ERR_WORKER              (1u << 1)\n'
    '#define ERR_STATE_RANGE         (1u << 2)\n'
    '#define ERR_PACKET_TIMEOUT      (1u << 3)\n'
    '#define ERR_SEND                (1u << 4)\n'
    '#define ERR_VALUE_RANGE         (1u << 5)\n'
    '#define ERR_OUTPUT              (1u << 6)\n'
    '\n'
    '#define PHASE_INTEGRATE         0u\n'
    '#define PHASE_BROADCAST         1u\n'
    '#define PHASE_WAIT              2u\n'
    '#define PHASE_FINISHED          3u\n'
    '\n'
    '/* DTCM-resident per-worker state. */\n'
    'static int32_t weights[N_WORKERS];       /* Q24; sparse by worker type */\n'
    'static int64_t accum[N_WORKERS];         /* Q20 path accumulator */\n'
    'static int32_t e_all[N_WORKERS];         /* Q20 */\n'
    'static int32_t m_all[N_WORKERS];         /* Q20 */\n'
    'static uint32_t seen_e[SEEN_WORDS];\n'
    'static uint32_t seen_m[SEEN_WORDS];\n'
    '\n'
    'static int32_t e_dv[N_MODES], e_z[N_MODES];  /* Q30 */\n'
    'static int32_t m_dv[N_MODES], m_z[N_MODES];  /* Q30 */\n'
    '\n'
    'static volatile uint32_t *cfg = 0;\n'
    'static volatile int32_t *paths = 0;\n'
    'static volatile uint32_t *schedule = 0;\n'
    'static volatile int32_t *outrec = 0;\n'
    '\n'
    'static uint32_t worker_id = 0xffffffffu;\n'
    'static uint32_t route_prefix = 0;\n'
    'static uint32_t n_cycles = 0;\n'
    'static uint32_t bcast_slots = BCAST_SLOTS_DFLT;\n'
    '\n'
    'static uint32_t cycle_idx = 0;\n'
    'static uint32_t force_idx = 0;\n'
    'static uint32_t substep_idx = 0;\n'
    'static volatile uint32_t phase = PHASE_INTEGRATE;\n'
    'static uint32_t slot_idx = 0;\n'
    'static uint32_t wait_ticks = 0;\n'
    'static volatile uint32_t count_e = 0;\n'
    'static volatile uint32_t count_m = 0;\n'
    'static uint32_t send_fail = 0;\n'
    'static uint32_t error_flags = ERR_NONE;\n'
    'static volatile uint32_t early_rx = 0;\n'
    'static volatile uint32_t wrong_step = 0;\n'
    'static volatile uint32_t wrong_phase = 0;\n'
    'static uint32_t exit_phase_snapshot = PHASE_INTEGRATE;\n'
    'static uint32_t exit_force_snapshot = 0;\n'
    '\n'
    'static int32_t input_e = 0;   /* Q20 */\n'
    'static int32_t input_m = 0;   /* Q20 */\n'
    'static int32_t e_self = 0;    /* Q20 */\n'
    'static int32_t m_self = 0;    /* Q20 */\n'
    '\n'
    '/* ------------------------------------------------------------------------- */\n'
    "/* Exact floor arithmetic matching Python's signed >> and // positive divisor */\n"
    '/* ------------------------------------------------------------------------- */\n'
    'static inline int64_t floor_rshift64(int64_t x, uint32_t q) {\n'
    '    if (x >= 0) return x >> q;\n'
    '    int64_t a = -x;\n'
    '    int64_t d = ((int64_t)1 << q);\n'
    '    return -((a + d - 1) >> q);\n'
    '}\n'
    '\n'
    'static inline int64_t floor_div_pos64(int64_t x, int64_t d) {\n'
    '    if (x >= 0) return x / d;\n'
    '    int64_t a = -x;\n'
    '    return -((a + d - 1) / d);\n'
    '}\n'
    '\n'
    'static inline int32_t mul_shift32(int32_t a, int32_t b, uint32_t q) {\n'
    '    return (int32_t) floor_rshift64((int64_t)a * (int64_t)b, q);\n'
    '}\n'
    '\n'
    'static inline int32_t clamp_i64_to_i32(int64_t x) {\n'
    '    if (x > 2147483647LL) return 2147483647;\n'
    '    if (x < -2147483648LL) return (int32_t)0x80000000u;\n'
    '    return (int32_t)x;\n'
    '}\n'
    '\n'
    '/* ------------------------------------------------------------------------- */\n'
    '/* Q20 path algebra                                                           */\n'
    '/* ------------------------------------------------------------------------- */\n'
    'static inline int32_t grad_q20(int32_t x) {\n'
    '    int32_t x2 = mul_shift32(x, x, Q_PATH);\n'
    '    int32_t x3 = mul_shift32(x2, x, Q_PATH);\n'
    '    return (int32_t)((int64_t)2 * x + (int64_t)4 * x3);\n'
    '}\n'
    '\n'
    'static inline int32_t q20_product(int32_t a, int32_t b) {\n'
    '    return mul_shift32(a, b, Q_PATH);\n'
    '}\n'
    '\n'
    '/* ------------------------------------------------------------------------- */\n'
    '/* Centered current Izh model: one of the 80 RK2 substeps                     */\n'
    '/* ------------------------------------------------------------------------- */\n'
    'static inline bool izh_one(int32_t *dv, int32_t *z, int32_t inp, int32_t gain) {\n'
    '    int32_t x = *dv;\n'
    '    int32_t y = *z;\n'
    '    int32_t x2 = mul_shift32(x, x, Q_STATE);\n'
    '    int32_t k1v =\n'
    '        mul_shift32(IZH_Q_Q30, x, Q_STATE) +\n'
    '        mul_shift32(C004_Q30, x2, Q_STATE) -\n'
    '        mul_shift32(IZH_A_Q30, y, Q_STATE) +\n'
    '        mul_shift32(gain, inp, Q_PATH);\n'
    '    int32_t k1z =\n'
    '        mul_shift32(IZH_B_Q30, x, Q_STATE) -\n'
    '        mul_shift32(IZH_A_Q30, y, Q_STATE);\n'
    '\n'
    '    int32_t xm = x + (int32_t)floor_rshift64(k1v, 1);\n'
    '    int32_t ym = y + (int32_t)floor_rshift64(k1z, 1);\n'
    '    int32_t xm2 = mul_shift32(xm, xm, Q_STATE);\n'
    '    int32_t k2v =\n'
    '        mul_shift32(IZH_Q_Q30, xm, Q_STATE) +\n'
    '        mul_shift32(C004_Q30, xm2, Q_STATE) -\n'
    '        mul_shift32(IZH_A_Q30, ym, Q_STATE) +\n'
    '        mul_shift32(gain, inp, Q_PATH);\n'
    '    int32_t k2z =\n'
    '        mul_shift32(IZH_B_Q30, xm, Q_STATE) -\n'
    '        mul_shift32(IZH_A_Q30, ym, Q_STATE);\n'
    '\n'
    '    x += k2v;\n'
    '    y += k2z;\n'
    '    *dv = x;\n'
    '    *z = y;\n'
    '\n'
    '    /* Canonical model is explicitly subthreshold. 0.5 in Q30 is already\n'
    '       hundreds of times larger than the validated centered trajectory. */\n'
    '    if (x > (1 << 29) || x < -(1 << 29) ||\n'
    '        y > (1 << 29) || y < -(1 << 29)) {\n'
    '        return false;\n'
    '    }\n'
    '    return true;\n'
    '}\n'
    '\n'
    'static inline int32_t readout_q20(int32_t dv, int32_t z,\n'
    '                                  int32_t cv_q16, int32_t cua_q16) {\n'
    '    int64_t a = floor_rshift64((int64_t)cv_q16 * dv, 26);\n'
    '    int64_t b = floor_rshift64((int64_t)cua_q16 * z, 26);\n'
    '    return clamp_i64_to_i32(a + b);\n'
    '}\n'
    '\n'
    '/* ------------------------------------------------------------------------- */\n'
    '/* Packet payload: type[31], force-step[30:26], signed-Q20 value[25:0].       */\n'
    '/* Q20 range is +/-32, far outside the canonical E/M range.                  */\n'
    '/* ------------------------------------------------------------------------- */\n'
    'static inline bool value_fits_payload(int32_t v) {\n'
    '    return (v >= -(1 << 25) && v < (1 << 25));\n'
    '}\n'
    '\n'
    'static inline uint32_t encode_payload(uint32_t type, uint32_t step, int32_t v) {\n'
    '    uint32_t raw = ((uint32_t)v) & 0x03ffffffu;\n'
    '    return ((type & 1u) << 31) | ((step & 0x1fu) << 26) | raw;\n'
    '}\n'
    '\n'
    'static inline int32_t decode_value(uint32_t payload) {\n'
    '    uint32_t raw = payload & 0x03ffffffu;\n'
    '    if (raw & 0x02000000u) raw |= 0xfc000000u;\n'
    '    return (int32_t)raw;\n'
    '}\n'
    '\n'
    'static inline bool seen_test_set(uint32_t *seen, uint32_t i) {\n'
    '    uint32_t w = i >> 5;\n'
    '    uint32_t b = 1u << (i & 31u);\n'
    '    bool was = (seen[w] & b) != 0;\n'
    '    seen[w] |= b;\n'
    '    return was;\n'
    '}\n'
    '\n'
    'static inline void clear_seen(void) {\n'
    '    for (uint32_t i = 0; i < SEEN_WORDS; i++) {\n'
    '        seen_e[i] = 0;\n'
    '        seen_m[i] = 0;\n'
    '    }\n'
    '}\n'
    '\n'
    'static void packet_callback(uint key, uint payload) {\n'
    '    /*\n'
    '     * V4 transport rule: receive CURRENT-step packets during INTEGRATE too.\n'
    '     * V3 listened only in BROADCAST/WAIT, so small barrier skew could make a\n'
    '     * later worker discard valid packets from an earlier worker at the next\n'
    '     * force step. This changes transport timing only, never Izh/plasticity math.\n'
    '     */\n'
    '    if (phase == PHASE_FINISHED) {\n'
    '        wrong_phase++;\n'
    '        return;\n'
    '    }\n'
    '    if ((key & 0xff000000u) != 0xa5000000u) return;\n'
    '    uint32_t src = key & 0x0000ffffu;\n'
    '    if (src >= N_WORKERS) return;\n'
    '    uint32_t step = (payload >> 26) & 0x1fu;\n'
    '    if (step != force_idx) {\n'
    '        wrong_step++;\n'
    '        return;\n'
    '    }\n'
    '    if (phase == PHASE_INTEGRATE) early_rx++;\n'
    '    uint32_t type = payload >> 31;\n'
    '    int32_t value = decode_value(payload);\n'
    '    if (type == 0) {\n'
    '        if (!seen_test_set(seen_e, src)) {\n'
    '            e_all[src] = value;\n'
    '            count_e++;\n'
    '        }\n'
    '    } else {\n'
    '        if (!seen_test_set(seen_m, src)) {\n'
    '            m_all[src] = value;\n'
    '            count_m++;\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    'static bool send_packet_retry(uint32_t key, uint32_t payload) {\n'
    '    for (uint32_t n = 0; n < 256; n++) {\n'
    '        if (spin1_send_mc_packet(key, payload, WITH_PAYLOAD)) return true;\n'
    '    }\n'
    '    send_fail++;\n'
    '    error_flags |= ERR_SEND;\n'
    '    return false;\n'
    '}\n'
    '\n'
    '/* ------------------------------------------------------------------------- */\n'
    '/* Training mechanics                                                         */\n'
    '/* ------------------------------------------------------------------------- */\n'
    'static inline uint32_t path_index(uint32_t cyc, uint32_t state, uint32_t coord) {\n'
    '    return ((cyc * PATH_STATES + state) * N_WORKERS) + coord;\n'
    '}\n'
    '\n'
    'static void prepare_force_step(void) {\n'
    '    uint32_t idx0 = path_index(cycle_idx, force_idx, worker_id);\n'
    '    uint32_t idx1 = path_index(cycle_idx, force_idx + 1u, worker_id);\n'
    '    int32_t x0 = paths[idx0];\n'
    '    int32_t x1 = paths[idx1];\n'
    '    int32_t dx = x1 - x0;\n'
    '    int32_t gstar = mul_shift32(dx, INV_MU_DT_Q20, Q_PATH);\n'
    '    int32_t y = gstar - grad_q20(x1);\n'
    '\n'
    '    /* Path-start weights remain unchanged until all 27 samples finish. */\n'
    '    volatile int32_t *xrow = &paths[path_index(cycle_idx, force_idx + 1u, 0)];\n'
    '    int64_t pacc = 0;\n'
    '    for (uint32_t j = 0; j < N_WORKERS; j++) {\n'
    '        pacc += (int64_t)weights[j] * (int64_t)xrow[j];  /* Q44 */\n'
    '    }\n'
    '    int32_t pred = clamp_i64_to_i32(floor_rshift64(pacc, Q_WEIGHT)); /* Q20 */\n'
    '    input_e = x1;\n'
    '    input_m = y - pred;\n'
    '}\n'
    '\n'
    'static void begin_force_step(void) {\n'
    '    /*\n'
    '     * Open the receive window BEFORE the 80 Izh substeps. Do not clear it at\n'
    '     * integration end; current-step packets received early are valid data.\n'
    '     */\n'
    '    clear_seen();\n'
    '    count_e = 0;\n'
    '    count_m = 0;\n'
    '    slot_idx = 0;\n'
    '    wait_ticks = 0;\n'
    '    substep_idx = 0;\n'
    '    prepare_force_step();\n'
    '    phase = PHASE_INTEGRATE;\n'
    '}\n'
    '\n'
    'static void accumulate_local_update(void) {\n'
    '    if (worker_id < NV) {\n'
    '        /* Visible worker owns row W[v,:]. Only hidden endpoints are physical. */\n'
    '        int32_t ev = e_self;\n'
    '        int32_t mv = m_self;\n'
    '        for (uint32_t j = NV; j < N_WORKERS; j++) {\n'
    '            int32_t p1 = q20_product(mv, e_all[j]);\n'
    '            int32_t p2 = q20_product(ev, m_all[j]);\n'
    '            int32_t term = p1 + q20_product(ALPHA015_Q20, p2);\n'
    '            accum[j] += term;\n'
    '        }\n'
    '    } else {\n'
    '        /* Hidden worker owns W^T[h,:] over visible sources and H[h,:]. */\n'
    '        int32_t eh = e_self;\n'
    '        int32_t mh = m_self;\n'
    '        for (uint32_t j = 0; j < NV; j++) {\n'
    '            int32_t p1 = q20_product(m_all[j], eh);\n'
    '            int32_t p2 = q20_product(e_all[j], mh);\n'
    '            int32_t term = p1 + q20_product(ALPHA015_Q20, p2);\n'
    '            accum[j] += term;\n'
    '        }\n'
    '        for (uint32_t j = NV; j < N_WORKERS; j++) {\n'
    '            int32_t p1 = q20_product(mh, e_all[j]);\n'
    '            int32_t p2 = q20_product(eh, m_all[j]);\n'
    '            /* Python oracle: (p1 + p2) >> 1 */\n'
    '            accum[j] += floor_rshift64((int64_t)p1 + p2, 1);\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    'static void apply_path_update(void) {\n'
    '    if (worker_id < NV) {\n'
    '        for (uint32_t j = NV; j < N_WORKERS; j++) {\n'
    '            int64_t a = accum[j] * (int64_t)LR_Q24; /* Q44 */\n'
    '            int64_t d = floor_div_pos64(a, FORCE_STEPS);\n'
    '            int32_t delta = (int32_t)floor_rshift64(d, Q_PATH); /* Q24 */\n'
    '            weights[j] += delta;\n'
    '            accum[j] = 0;\n'
    '        }\n'
    '    } else {\n'
    '        for (uint32_t j = 0; j < NV; j++) {\n'
    '            int64_t a = accum[j] * (int64_t)LR_Q24;\n'
    '            int64_t d = floor_div_pos64(a, FORCE_STEPS);\n'
    '            int32_t delta = (int32_t)floor_rshift64(d, Q_PATH);\n'
    '            weights[j] += delta;\n'
    '            accum[j] = 0;\n'
    '        }\n'
    '        for (uint32_t j = NV; j < N_WORKERS; j++) {\n'
    '            if (j == worker_id) {\n'
    '                weights[j] = 0;\n'
    '                accum[j] = 0;\n'
    '                continue;\n'
    '            }\n'
    '            int64_t a = accum[j] * (int64_t)H_LR_Q24;\n'
    '            int64_t d = floor_div_pos64(a, FORCE_STEPS);\n'
    '            int32_t delta = (int32_t)floor_rshift64(d, Q_PATH);\n'
    '            weights[j] += delta;\n'
    '            accum[j] = 0;\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    'static void write_final_output(void) {\n'
    '    if (outrec == 0) return;\n'
    '    outrec[OUT_MAGIC] = (int32_t)MAGIC_OUT;\n'
    '    outrec[OUT_WORKER] = (int32_t)worker_id;\n'
    '    outrec[OUT_ERROR] = (int32_t)error_flags;\n'
    '    outrec[OUT_CYCLES_DONE] = (int32_t)cycle_idx;\n'
    '    outrec[OUT_MISSING_E] = (int32_t)(N_WORKERS - count_e);\n'
    '    outrec[OUT_MISSING_M] = (int32_t)(N_WORKERS - count_m);\n'
    '    outrec[OUT_SEND_FAIL] = (int32_t)send_fail;\n'
    '    outrec[OUT_EARLY_RX] = (int32_t)early_rx;\n'
    '    outrec[OUT_WRONG_STEP] = (int32_t)wrong_step;\n'
    '    outrec[OUT_WRONG_PHASE] = (int32_t)wrong_phase;\n'
    '    outrec[OUT_PHASE_AT_EXIT] = (int32_t)exit_phase_snapshot;\n'
    '    outrec[OUT_FORCE_AT_EXIT] = (int32_t)exit_force_snapshot;\n'
    '    for (uint32_t k = 0; k < N_MODES; k++) {\n'
    '        int32_t r = readout_q20(e_dv[k], e_z[k], CV_E_Q16, CUA_E_Q16);\n'
    '        outrec[OUT_EREAD0 + k] = r;\n'
    '    }\n'
    '    for (uint32_t j = 0; j < N_WORKERS; j++) {\n'
    '        outrec[OUT_W0 + j] = weights[j];\n'
    '    }\n'
    '    outrec[OUT_STATUS] = (error_flags == ERR_NONE) ? STATUS_DONE : STATUS_ERROR;\n'
    '}\n'
    '\n'
    'static void fail_and_exit(uint32_t err) {\n'
    '    error_flags |= err;\n'
    '    exit_phase_snapshot = phase;\n'
    '    exit_force_snapshot = force_idx;\n'
    '    phase = PHASE_FINISHED;\n'
    '    write_final_output();\n'
    '    if (worker_id == 0) {\n'
    '        io_printf(IO_BUF,\n'
    '                  "PHASE1_TRAIN_FAIL error=0x%08x cycle=%u step=%u E=%u M=%u early=%u wrong_step=%u phase=%u\\n",\n'
    '                  error_flags, cycle_idx, force_idx, count_e, count_m,\n'
    '                  early_rx, wrong_step, phase);\n'
    '    }\n'
    '    spin1_exit(1);\n'
    '}\n'
    '\n'
    'static void finish_training(void) {\n'
    '    exit_phase_snapshot = phase;\n'
    '    exit_force_snapshot = force_idx;\n'
    '    phase = PHASE_FINISHED;\n'
    '    cycle_idx = n_cycles;\n'
    '    write_final_output();\n'
    '    if (worker_id == 0) {\n'
    '        io_printf(IO_BUF, "PHASE1_TRAIN_DONE cycles=%u send_fail=%u error=0x%08x\\n",\n'
    '                  n_cycles, send_fail, error_flags);\n'
    '    }\n'
    '    spin1_exit(0);\n'
    '}\n'
    '\n'
    'static void timer_callback(uint unused0, uint unused1) {\n'
    '    (void)unused0; (void)unused1;\n'
    '    if (phase == PHASE_FINISHED) return;\n'
    '\n'
    '    if (phase == PHASE_INTEGRATE) {\n'
    '        uint32_t k = schedule[cycle_idx];\n'
    '        if (k >= N_MODES) {\n'
    '            fail_and_exit(ERR_CONFIG); return;\n'
    '        }\n'
    '        if (!izh_one(&e_dv[k], &e_z[k], input_e, GAIN_E_Q30) ||\n'
    '            !izh_one(&m_dv[k], &m_z[k], input_m, GAIN_M_Q30)) {\n'
    '            fail_and_exit(ERR_STATE_RANGE); return;\n'
    '        }\n'
    '        substep_idx++;\n'
    '        if (substep_idx >= IZH_SUBSTEPS) {\n'
    '            e_self = readout_q20(e_dv[k], e_z[k], CV_E_Q16, CUA_E_Q16);\n'
    '            m_self = readout_q20(m_dv[k], m_z[k], CV_M_Q16, CUA_M_Q16);\n'
    '            if (!value_fits_payload(e_self) || !value_fits_payload(m_self)) {\n'
    '                fail_and_exit(ERR_VALUE_RANGE); return;\n'
    '            }\n'
    '            /* Preserve peers already received during INTEGRATE; mark self once. */\n'
    '            if (!seen_test_set(seen_e, worker_id)) {\n'
    '                e_all[worker_id] = e_self;\n'
    '                count_e++;\n'
    '            }\n'
    '            if (!seen_test_set(seen_m, worker_id)) {\n'
    '                m_all[worker_id] = m_self;\n'
    '                count_m++;\n'
    '            }\n'
    '            slot_idx = 0;\n'
    '            wait_ticks = 0;\n'
    '            phase = PHASE_BROADCAST;\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (phase == PHASE_BROADCAST) {\n'
    '        if ((worker_id % bcast_slots) == slot_idx) {\n'
    '            uint32_t key = route_prefix | worker_id;\n'
    '            send_packet_retry(key, encode_payload(0, force_idx, e_self));\n'
    '            send_packet_retry(key, encode_payload(1, force_idx, m_self));\n'
    '        }\n'
    '        slot_idx++;\n'
    '        if (slot_idx >= bcast_slots) {\n'
    '            phase = PHASE_WAIT;\n'
    '            wait_ticks = 0;\n'
    '        }\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    if (phase == PHASE_WAIT) {\n'
    '        if (count_e >= N_WORKERS && count_m >= N_WORKERS) {\n'
    '            accumulate_local_update();\n'
    '            force_idx++;\n'
    '            substep_idx = 0;\n'
    '            if (force_idx >= FORCE_STEPS) {\n'
    '                apply_path_update();\n'
    '                cycle_idx++;\n'
    '                force_idx = 0;\n'
    '                if (cycle_idx >= n_cycles) {\n'
    '                    finish_training(); return;\n'
    '                }\n'
    '            }\n'
    '            begin_force_step();\n'
    '            return;\n'
    '        }\n'
    '        wait_ticks++;\n'
    '        if (wait_ticks > WAIT_TIMEOUT_TICKS) {\n'
    '            fail_and_exit(ERR_PACKET_TIMEOUT); return;\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    'void c_main(void) {\n'
    '    cfg = (volatile uint32_t *)sark_tag_ptr(SDRAM_TAG, sark_app_id());\n'
    '    if (cfg == 0 || cfg[CFG_MAGIC] != MAGIC_CFG || cfg[CFG_VER] != CFG_VERSION ||\n'
    '        cfg[CFG_NWORKERS] != N_WORKERS || cfg[CFG_NV] != NV || cfg[CFG_NH] != NH ||\n'
    '        cfg[CFG_PATH_STATES] != PATH_STATES || cfg[CFG_FORCE_STEPS] != FORCE_STEPS) {\n'
    '        error_flags |= ERR_CONFIG;\n'
    '        io_printf(IO_BUF, "PHASE1_TRAIN_BAD_CONFIG p=%u\\n", spin1_get_core_id());\n'
    '        spin1_exit(1); return;\n'
    '    }\n'
    '\n'
    '    uint32_t p = spin1_get_core_id();\n'
    '    if (p >= MAX_CORES) {\n'
    '        error_flags |= ERR_WORKER;\n'
    '        spin1_exit(1); return;\n'
    '    }\n'
    '    worker_id = cfg[CFG_WORKER_BY_P + p];\n'
    '    if (worker_id >= N_WORKERS) {\n'
    '        /* Core is not a Phase-I worker. */\n'
    '        spin1_exit(0); return;\n'
    '    }\n'
    '\n'
    '    n_cycles = cfg[CFG_NCYCLES];\n'
    '    route_prefix = cfg[CFG_BASE_KEY];\n'
    '    bcast_slots = cfg[CFG_BCAST_SLOTS];\n'
    '    if (bcast_slots == 0) bcast_slots = BCAST_SLOTS_DFLT;\n'
    '    paths = (volatile int32_t *)((uint8_t *)cfg + cfg[CFG_PATH_OFFSET]);\n'
    '    schedule = (volatile uint32_t *)((uint8_t *)cfg + cfg[CFG_SCHED_OFFSET]);\n'
    '    outrec = (volatile int32_t *)((uint8_t *)cfg + cfg[CFG_OUTPUT_OFFSET] +\n'
    '                                 p * cfg[CFG_OUTPUT_STRIDE]);\n'
    '\n'
    '    for (uint32_t j = 0; j < N_WORKERS; j++) {\n'
    '        weights[j] = 0;\n'
    '        accum[j] = 0;\n'
    '        e_all[j] = 0;\n'
    '        m_all[j] = 0;\n'
    '    }\n'
    '    for (uint32_t k = 0; k < N_MODES; k++) {\n'
    '        e_dv[k] = e_z[k] = m_dv[k] = m_z[k] = 0;\n'
    '    }\n'
    '    clear_seen();\n'
    '    outrec[OUT_MAGIC] = (int32_t)MAGIC_OUT;\n'
    '    outrec[OUT_WORKER] = (int32_t)worker_id;\n'
    '    outrec[OUT_STATUS] = STATUS_RUNNING;\n'
    '    outrec[OUT_ERROR] = 0;\n'
    '\n'
    '    cycle_idx = 0; force_idx = 0; substep_idx = 0;\n'
    '    begin_force_step();\n'
    '\n'
    '    if (worker_id == 0) {\n'
    '        io_printf(IO_BUF,\n'
    '            "PHASE1_TRAIN_START workers=%u cycles=%u slots=%u route_prefix=0x%08x\\n",\n'
    '            N_WORKERS, n_cycles, bcast_slots, route_prefix);\n'
    '    }\n'
    '\n'
    '    spin1_set_timer_tick(1000);  /* 1 ms */\n'
    '    spin1_callback_on(MCPL_PACKET_RECEIVED, packet_callback, -1);\n'
    '    spin1_callback_on(TIMER_TICK, timer_callback, 2);\n'
    '    spin1_start(SYNC_WAIT);\n'
    '}\n'
)


_EMBEDDED_EXPECTED_SHA256 = {
    "base": "7fadaf8ac68b29d8df948d1358d87f63449c12f1ffb575f1c5146701cb40c993",
    "trainer": "6c238bbe40ae84c2936e22d77b6082850dcb6b412a5181bf92a1579e6e13b77a",
    "routes": "63fe1aec976edd1851e25c3ecb60b4df0fedc9f8fa8bc3c72842c66b6447a635",
    "kernel": "8eacc79d7cf32c31fa0bc8cb4407de1a521f276d5d0b9fbfc0ed28692effaf75",
}


if __name__ == "__main__":
    main()

