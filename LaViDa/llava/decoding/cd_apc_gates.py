"""CD-APC / VCHD commit gates — reference implementation (Option A).

This module is the source of truth for the three-condition commit criterion.
It is intentionally model-free so it can be unit-tested without GPU access.
Integrate into MMaDA VCHD decoder by calling ``score_position`` / ``is_eligible``.

MC1 fix: "w/o g-gate" MUST set ``enable_g_gate=False`` (or ``g_min=-inf``),
NOT ``tau_g=0``. Setting ``tau_g=0`` still enforces ``g >= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import math


class GMinPolicy(str, Enum):
    """Asymmetric default matches Eq. (gmin) in the paper."""

    ASYMMETRIC = "asymmetric"  # +tau_g on flip, -tau_g on non-flip
    ALWAYS_POS = "always_pos"  # +tau_g always
    ALWAYS_NEG = "always_neg"  # -tau_g always
    BYPASS = "bypass"  # g_min = -inf (true w/o g-gate)


@dataclass(frozen=True)
class GateConfig:
    tau_u: float = 0.10  # tau_base
    tau_r: float = 0.90  # tau_contrast
    tau_g: float = 0.05
    alpha: float = 0.25
    beta: float = 0.10
    enable_u_gate: bool = True
    enable_r_gate: bool = True
    enable_g_gate: bool = True
    g_min_policy: GMinPolicy = GMinPolicy.ASYMMETRIC
    fallback_to_raw: bool = True  # paper: fallback commits raw top-1


@dataclass(frozen=True)
class PositionScore:
    y_raw: int
    y_cd: int
    u: float  # c_base = p_vis(y_cd)
    r: float  # c_ctr = m_apc * q(y_cd)
    g: float  # log p_vis(y_cd) - log p_abl(y_cd)
    m_apc: float
    flipped: bool
    g_min: float
    eligible: bool


def apc_mask(log_p_vis: Sequence[float], beta: float) -> Tuple[bool, ...]:
    """Absolute Probability Cone over log-probs: keep v with log p >= max + log beta."""
    max_lp = max(log_p_vis)
    thr = max_lp + math.log(beta)
    return tuple(lp >= thr for lp in log_p_vis)


def contrast_scores(
    log_p_vis: Sequence[float],
    log_p_abl: Sequence[float],
    in_apc: Sequence[bool],
    alpha: float,
) -> Tuple[float, ...]:
    """CFG-style contrast score; -inf outside APC."""
    out = []
    for i, keep in enumerate(in_apc):
        if not keep:
            out.append(float("-inf"))
        else:
            out.append((1.0 + alpha) * log_p_vis[i] - alpha * log_p_abl[i])
    return tuple(out)


def _softmax(xs: Sequence[float]) -> Tuple[float, ...]:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        n = len(xs)
        return tuple(1.0 / n for _ in xs)
    m = max(finite)
    exps = [math.exp(x - m) if math.isfinite(x) else 0.0 for x in xs]
    z = sum(exps)
    if z <= 0:
        n = len(xs)
        return tuple(1.0 / n for _ in xs)
    return tuple(e / z for e in exps)


def compute_g_min(
    flipped: bool,
    tau_g: float,
    policy: GMinPolicy,
    enable_g_gate: bool,
) -> float:
    if (not enable_g_gate) or policy == GMinPolicy.BYPASS:
        return float("-inf")  # true removal of g-gate (MC1-correct)
    if policy == GMinPolicy.ALWAYS_POS:
        return +tau_g
    if policy == GMinPolicy.ALWAYS_NEG:
        return -tau_g
    # asymmetric
    return +tau_g if flipped else -tau_g


def score_position(
    log_p_vis: Sequence[float],
    log_p_abl: Sequence[float],
    cfg: GateConfig,
) -> PositionScore:
    """Score one mask position under CD-APC / VCHD gates."""
    assert len(log_p_vis) == len(log_p_abl)
    y_raw = max(range(len(log_p_vis)), key=lambda i: log_p_vis[i])
    in_apc = apc_mask(log_p_vis, cfg.beta)
    s = contrast_scores(log_p_vis, log_p_abl, in_apc, cfg.alpha)
    y_cd = max(range(len(s)), key=lambda i: s[i])
    q = _softmax(s)
    # APC mass in probability space
    p_vis = _softmax(log_p_vis)
    m_apc = sum(p for p, keep in zip(p_vis, in_apc) if keep)
    u = p_vis[y_cd]
    r = m_apc * q[y_cd]
    g = log_p_vis[y_cd] - log_p_abl[y_cd]
    flipped = y_cd != y_raw
    g_min = compute_g_min(flipped, cfg.tau_g, cfg.g_min_policy, cfg.enable_g_gate)

    ok_u = (u >= cfg.tau_u) if cfg.enable_u_gate else True
    ok_r = (r >= cfg.tau_r) if cfg.enable_r_gate else True
    ok_g = g >= g_min  # with g_min=-inf this is always True
    eligible = ok_u and ok_r and ok_g

    return PositionScore(
        y_raw=y_raw,
        y_cd=y_cd,
        u=u,
        r=r,
        g=g,
        m_apc=m_apc,
        flipped=flipped,
        g_min=g_min,
        eligible=eligible,
    )


def readiness(u: float, r: float, tau_u: float, tau_r: float) -> float:
    """Joint readiness for fallback selection."""
    a = float("inf") if tau_u == 0 else u / tau_u
    b = float("inf") if tau_r == 0 else r / tau_r
    return min(a, b)


def select_commits(
    scores: Sequence[PositionScore],
    cfg: GateConfig,
    budget: int,
) -> Tuple[list, str]:
    """Return (indices_to_commit, reason)."""
    eligible = [i for i, s in enumerate(scores) if s.eligible]
    if eligible:
        eligible.sort(key=lambda i: (-scores[i].r, -scores[i].u, i))
        return eligible[:budget], "THRESHOLD"

    # Fallback: highest readiness; paper requires g > -tau_g when g-gate on
    candidates = []
    for i, s in enumerate(scores):
        if cfg.enable_g_gate and s.g <= -cfg.tau_g:
            continue
        candidates.append(i)
    if not candidates:
        candidates = list(range(len(scores)))
    best = max(
        candidates,
        key=lambda i: (
            readiness(scores[i].u, scores[i].r, cfg.tau_u, cfg.tau_r),
            -i,
        ),
    )
    return [best], "FALLBACK"


def ablation_config(name: str, base: Optional[GateConfig] = None) -> GateConfig:
    """Named ablations for Phase-3 (MC1-correct)."""
    base = base or GateConfig()
    table = {
        "full": GateConfig(**{**base.__dict__}),
        "wo_u": GateConfig(**{**base.__dict__, "enable_u_gate": False}),
        "wo_r": GateConfig(**{**base.__dict__, "enable_r_gate": False}),
        # CRITICAL: true w/o g-gate — NOT tau_g=0
        "wo_g": GateConfig(
            **{**base.__dict__, "enable_g_gate": False, "g_min_policy": GMinPolicy.BYPASS}
        ),
        "gmin_always_pos": GateConfig(
            **{**base.__dict__, "g_min_policy": GMinPolicy.ALWAYS_POS}
        ),
        "gmin_always_neg": GateConfig(
            **{**base.__dict__, "g_min_policy": GMinPolicy.ALWAYS_NEG}
        ),
        "vchd_2gate": GateConfig(
            **{**base.__dict__, "enable_g_gate": False, "g_min_policy": GMinPolicy.BYPASS}
        ),
    }
    if name not in table:
        raise KeyError(f"Unknown ablation {name!r}; choose from {sorted(table)}")
    return table[name]
