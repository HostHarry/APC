"""CPU-only unit tests for the defer_only module (v4 M1).

Verifies:
- veto: three variants + dispatcher + eff_conf <= base_conf invariant
- dispatcher.pick_transfer_defer_only:
    * x0 == argmax(base_logits) at every masked position (INVARIANT)
    * drop_logits=None => behavior identical to plain DCD (baseline)
    * causal_lambda=0 => behavior identical to plain DCD (baseline)
    * debug records populated when return_debug=True
- Routing via _pick_transfer_cv when cv_mode == 'defer_only'

Run:
    /home/user/anaconda3/envs/mmada/bin/python tests/test_defer_only.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.defer_only import veto as defer_veto  # noqa: E402
from models.defer_only.dispatcher import pick_transfer_defer_only  # noqa: E402
from models.mmada_decode import (  # noqa: E402
    MMaDADecodeConfig,
    _confidence_from_logits,
    _pick_transfer,
    _pick_transfer_cv,
    dcd_decode_text_cv_dual_cache,
    dcd_decode_text_dual_cache,
)


torch.manual_seed(0)


def _cfg(**kwargs) -> MMaDADecodeConfig:
    base = dict(
        mask_id=126336,
        decode_algo="threshold",
        decode_param=0.9,
        temperature=0.0,
        remasking="low_confidence",
        causal_lambda=0.5,
        causal_clip=4.0,
        cv_mode="defer_only",
        defer_veto_type="mult",
        defer_tau=0.0,
        defer_beta=1.0,
        defer_gain_type="logit",
    )
    base.update(kwargs)
    return MMaDADecodeConfig(**base)


def _tiny_step(vocab: int = 16, batch: int = 1, seq: int = 4, mask_id: int = 126336):
    """Build a tiny synthetic step for the dispatcher."""
    logits = torch.randn(batch, seq, vocab) * 0.5
    drop = torch.randn(batch, seq, vocab) * 0.5
    tokens = torch.zeros(batch, seq, dtype=torch.long)
    tokens[:, :2] = mask_id
    mask_index = tokens == mask_id
    return logits, drop, mask_index, tokens


# --------------------------------------------------------------------------
# 1. veto.compute_visual_gain
# --------------------------------------------------------------------------


def test_gain_zero_when_logits_equal():
    logits = torch.randn(2, 3, 5)
    x0 = torch.tensor([[0, 1, 2], [3, 4, 0]])
    for gt in ("logit", "logprob"):
        gain = defer_veto.compute_visual_gain(logits, logits, x0, causal_clip=4.0, gain_type=gt)
        assert torch.allclose(gain, torch.zeros_like(gain)), f"gain_type={gt}"


def test_gain_clip_respected():
    # Extreme logits => extreme raw gain; after clip it must fall in [-clip, +clip].
    base = torch.tensor([[[100.0, -100.0]]])  # log p(0) ~ 0, log p(1) ~ -200
    drop = torch.tensor([[[-100.0, 100.0]]])  # log p(0) ~ -200, log p(1) ~ 0
    x0 = torch.tensor([[0]])
    for gt in ("logit", "logprob"):
        gain = defer_veto.compute_visual_gain(base, drop, x0, causal_clip=4.0, gain_type=gt)
        assert gain.abs().max().item() <= 4.0 + 1e-6, f"gain_type={gt}"
        assert gain.item() > 3.9, f"gain_type={gt}"  # gain is at (positive) clip


def test_gain_logit_matches_v32_definition():
    """gain_type='logit' must equal ``base_logit[x] - drop_logit[x]`` exactly."""
    torch.manual_seed(0)
    base = torch.randn(1, 3, 8)
    drop = torch.randn(1, 3, 8)
    x0 = torch.tensor([[0, 4, 7]])
    gain = defer_veto.compute_visual_gain(
        base, drop, x0, causal_clip=100.0, gain_type="logit"
    )
    for pos in range(3):
        idx = x0[0, pos].item()
        expected = float(base[0, pos, idx] - drop[0, pos, idx])
        assert abs(gain[0, pos].item() - expected) < 1e-6


def test_gain_logprob_matches_cd_paper():
    """gain_type='logprob' must equal ``log_softmax(base)[x] - log_softmax(drop)[x]``."""
    torch.manual_seed(0)
    base = torch.randn(1, 3, 8)
    drop = torch.randn(1, 3, 8)
    x0 = torch.tensor([[0, 4, 7]])
    gain = defer_veto.compute_visual_gain(
        base, drop, x0, causal_clip=100.0, gain_type="logprob"
    )
    base_lp = F.log_softmax(base.to(torch.float64), dim=-1)
    drop_lp = F.log_softmax(drop.to(torch.float64), dim=-1)
    for pos in range(3):
        idx = x0[0, pos].item()
        expected = float(base_lp[0, pos, idx] - drop_lp[0, pos, idx])
        assert abs(gain[0, pos].item() - expected) < 1e-6


def test_gain_logit_and_logprob_differ_by_lse_diff():
    """Regression: ``logit_gain - logprob_gain == LSE(base) - LSE(drop)``.

    This is the mathematical identity that explains why v3.2 (which used the
    logit-based definition) saw gain spread over +/-1.5 while v4's initial
    log-prob-based definition collapsed to ~0 for shuffle drop.
    """
    torch.manual_seed(0)
    base = torch.randn(1, 5, 12)
    drop = torch.randn(1, 5, 12)
    x0 = torch.tensor([[0, 3, 7, 11, 1]])
    g_logit = defer_veto.compute_visual_gain(base, drop, x0, causal_clip=100.0, gain_type="logit")
    g_logp = defer_veto.compute_visual_gain(base, drop, x0, causal_clip=100.0, gain_type="logprob")
    lse_diff = (
        torch.logsumexp(base.to(torch.float64), dim=-1)
        - torch.logsumexp(drop.to(torch.float64), dim=-1)
    )
    assert torch.allclose(g_logit - g_logp, lse_diff, atol=1e-6)


def test_gain_unknown_type_raises():
    logits = torch.randn(1, 2, 3)
    x0 = torch.tensor([[0, 1]])
    try:
        defer_veto.compute_visual_gain(logits, logits, x0, causal_clip=1.0, gain_type="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown gain_type must raise")


# --------------------------------------------------------------------------
# 2. veto variants
# --------------------------------------------------------------------------


def test_hard_veto_semantics():
    base_conf = torch.tensor([[0.9, 0.9, 0.9, 0.9]], dtype=torch.float64)
    gain = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    eff = defer_veto.apply_veto_hard(base_conf, gain, tau=0.5)
    # gain < 0.5 => 0; gain >= 0.5 => keep base_conf
    expected = torch.tensor([[0.0, 0.0, 0.9, 0.9]], dtype=torch.float64)
    assert torch.allclose(eff, expected)


def test_mult_veto_monotonic():
    base_conf = torch.full((1, 5), 0.9)
    gain = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    eff = defer_veto.apply_veto_mult(base_conf, gain, tau=0.0, beta=1.0)
    # Strictly monotonic in gain.
    diffs = eff[0, 1:] - eff[0, :-1]
    assert (diffs > 0).all().item()


def test_min_veto_never_boosts():
    torch.manual_seed(42)
    base_conf = torch.rand(1, 100)
    gain = torch.randn(1, 100) * 5.0
    eff = defer_veto.apply_veto_min(base_conf, gain, beta=1.0)
    # eff <= base_conf pointwise (this is the whole point of "defer-only").
    assert (eff <= base_conf + 1e-6).all().item()


def test_veto_dispatch_matches_direct_calls():
    base_conf = torch.rand(1, 4)
    gain = torch.randn(1, 4)
    a = defer_veto.apply_veto(base_conf, gain, "hard", tau=0.1, beta=2.0)
    b = defer_veto.apply_veto_hard(base_conf, gain, tau=0.1)
    assert torch.allclose(a, b)

    a = defer_veto.apply_veto(base_conf, gain, "mult", tau=0.1, beta=2.0)
    b = defer_veto.apply_veto_mult(base_conf, gain, tau=0.1, beta=2.0)
    assert torch.allclose(a, b)

    a = defer_veto.apply_veto(base_conf, gain, "min", tau=999.0, beta=2.0)
    b = defer_veto.apply_veto_min(base_conf, gain, beta=2.0)
    assert torch.allclose(a, b)

    a = defer_veto.apply_veto(base_conf, gain, "soft", tau=0.1, beta=2.0)
    b = defer_veto.apply_veto_soft(base_conf, gain, tau=0.1, beta=2.0)
    assert torch.allclose(a, b)


def test_soft_veto_no_penalty_when_gain_geq_tau():
    """The fix for the sigmoid-zero-point bug: gain >= tau => eff == base_conf."""
    base_conf = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9]], dtype=torch.float64)
    # gains at, above, and below tau=0.
    gain = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]], dtype=torch.float64)
    eff = defer_veto.apply_veto_soft(base_conf, gain, tau=0.0, beta=1.0)
    # gain >= 0 (positions 2, 3, 4) => untouched
    assert torch.allclose(eff[0, 2:], base_conf[0, 2:])
    # gain < 0 (positions 0, 1) => smoothly reduced
    expected_neg1 = 0.9 * math.exp(-1.0 * 1.0)   # ~0.331
    expected_neg05 = 0.9 * math.exp(-1.0 * 0.5)  # ~0.546
    assert abs(eff[0, 0].item() - expected_neg1) < 1e-6
    assert abs(eff[0, 1].item() - expected_neg05) < 1e-6


def test_soft_veto_matches_hard_in_limit_beta_infty():
    """As beta -> infty, soft veto converges to hard veto."""
    torch.manual_seed(1)
    base_conf = torch.rand(1, 100, dtype=torch.float64)
    gain = torch.randn(1, 100, dtype=torch.float64)
    soft = defer_veto.apply_veto_soft(base_conf, gain, tau=0.0, beta=1000.0)
    hard = defer_veto.apply_veto_hard(base_conf, gain, tau=0.0)
    assert torch.allclose(soft, hard, atol=1e-6)


def test_soft_veto_never_boosts():
    """eff_conf <= base_conf pointwise (the defer-only invariant)."""
    torch.manual_seed(2)
    base_conf = torch.rand(1, 200, dtype=torch.float64)
    gain = torch.randn(1, 200, dtype=torch.float64) * 3.0
    for tau in (-1.0, 0.0, 0.5):
        for beta in (0.5, 1.0, 5.0):
            eff = defer_veto.apply_veto_soft(base_conf, gain, tau=tau, beta=beta)
            assert (eff <= base_conf + 1e-9).all().item(), (tau, beta)


def test_veto_dispatch_unknown_raises():
    try:
        defer_veto.apply_veto(torch.zeros(1, 1), torch.zeros(1, 1), "shrug", 0.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown veto type must raise")


# --------------------------------------------------------------------------
# 3. dispatcher invariants
# --------------------------------------------------------------------------


def test_pick_transfer_argmax_from_base_only():
    """INVARIANT: x0 at masked positions == argmax(base_logits) (temperature=0)."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=6)
    for veto_type in ("hard", "mult", "min", "soft"):
        for gt in ("logit", "logprob"):
            cfg = _cfg(defer_veto_type=veto_type, defer_gain_type=gt, temperature=0.0)
            x0, _ = pick_transfer_defer_only(logits, drop, cfg, mask_index, tokens)
            x0_base = torch.argmax(logits, dim=-1)
            for b in range(mask_index.shape[0]):
                for pos in mask_index[b].nonzero(as_tuple=True)[0].tolist():
                    assert x0[b, pos].item() == x0_base[b, pos].item(), (
                        f"defer_only broke argmax invariant for veto={veto_type} "
                        f"gain_type={gt} pos={pos}"
                    )


def test_pick_transfer_no_drop_equals_baseline():
    """drop_logits=None => defer_only == baseline DCD _pick_transfer."""
    logits, _, mask_index, tokens = _tiny_step(vocab=16, seq=5)
    cfg_defer = _cfg(temperature=0.0)
    cfg_base = _cfg(temperature=0.0, cv_mode="off")
    x0_d, tr_d = pick_transfer_defer_only(logits, None, cfg_defer, mask_index, tokens)
    x0_b, tr_b = _pick_transfer(logits, cfg_base, mask_index, tokens)
    assert torch.equal(x0_d, x0_b)
    assert torch.equal(tr_d, tr_b)


def test_pick_transfer_causal_lambda_zero_equals_baseline():
    """causal_lambda=0 => veto NOT applied even if drop_logits present."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=16, seq=5)
    cfg_defer = _cfg(causal_lambda=0.0, temperature=0.0)
    cfg_base = _cfg(causal_lambda=0.0, temperature=0.0, cv_mode="off")
    x0_d, tr_d = pick_transfer_defer_only(logits, drop, cfg_defer, mask_index, tokens)
    x0_b, tr_b = _pick_transfer(logits, cfg_base, mask_index, tokens)
    assert torch.equal(x0_d, x0_b)
    assert torch.equal(tr_d, tr_b)


def test_pick_transfer_debug_populated():
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=4)
    cfg = _cfg(temperature=0.0)
    cfg.return_debug = True
    records: list = []
    _ = pick_transfer_defer_only(
        logits, drop, cfg, mask_index, tokens,
        debug_records=records, step_idx=7,
    )
    assert len(records) > 0
    r = records[0]
    for k in (
        "defer_veto_type", "defer_tau", "defer_beta", "defer_gain_type",
        "defer_lambda",
        "defer_base_conf", "defer_eff_conf", "defer_gain",
        "defer_active",
        # v3.2-schema keys are also present so downstream analysis works
        "conf_source_used", "gate_active", "raw_conf",
    ):
        assert k in r, f"missing debug key {k!r}"
    assert r["step"] == 7
    assert r["defer_veto_type"] == "mult"
    assert r["defer_gain_type"] == "logit"
    assert r["defer_lambda"] == 0.5


# --------------------------------------------------------------------------
# 4. v5: causal_lambda as INTERVENTION STRENGTH (linear blend)
# --------------------------------------------------------------------------


def test_lambda_zero_matches_sanity_conf():
    """lambda=0 => eff_conf == base_conf regardless of gain / tau / veto_type."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=6)
    torch.manual_seed(11)

    for veto_type in ("hard", "soft", "mult", "min"):
        cfg0 = _cfg(temperature=0.0, causal_lambda=0.0,
                    defer_veto_type=veto_type, defer_tau=0.0)
        cfg0.return_debug = True
        recs0: list = []
        _ = pick_transfer_defer_only(logits, drop, cfg0, mask_index, tokens,
                                     debug_records=recs0, step_idx=0)
        # every emitted record must have eff_conf == base_conf.
        for r in recs0:
            assert abs(r["defer_eff_conf"] - r["defer_base_conf"]) < 1e-9, (
                f"lambda=0 leaked veto for veto_type={veto_type!r}: "
                f"base={r['defer_base_conf']}  eff={r['defer_eff_conf']}"
            )
            assert r["defer_active"] is False


def test_lambda_one_matches_pure_veto():
    """lambda=1 => eff_conf == veto(base_conf) (equivalent to v1-v4 defer_only)."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=6)
    torch.manual_seed(13)

    for veto_type in ("hard", "soft", "mult", "min"):
        cfg = _cfg(temperature=0.0, causal_lambda=1.0,
                   defer_veto_type=veto_type, defer_tau=0.0, defer_beta=1.0)
        cfg.return_debug = True
        recs: list = []
        _ = pick_transfer_defer_only(logits, drop, cfg, mask_index, tokens,
                                     debug_records=recs, step_idx=0)
        # Reconstruct veto directly and compare to recorded eff_conf.
        x0 = torch.argmax(logits, dim=-1)
        base_logit_at_x0 = torch.gather(logits, -1, x0.unsqueeze(-1)).squeeze(-1).to(torch.float64)
        base_lse = torch.logsumexp(logits, dim=-1).to(torch.float64)
        base_conf = torch.exp(base_logit_at_x0 - base_lse)
        gain = defer_veto.compute_visual_gain(logits, drop, x0, causal_clip=4.0, gain_type="logit")
        expected_eff = defer_veto.apply_veto(base_conf, gain, veto_type, tau=0.0, beta=1.0)

        for r in recs:
            pos, batch = r["position"], r["batch"]
            e_expected = float(expected_eff[batch, pos].item())
            assert abs(r["defer_eff_conf"] - e_expected) < 1e-6, (
                f"lambda=1 mismatch for {veto_type}: expected {e_expected}, got {r['defer_eff_conf']}"
            )


def test_lambda_half_is_linear_blend():
    """lambda=0.5 => eff_conf == 0.5 * base + 0.5 * veto_conf."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=6)
    torch.manual_seed(17)

    # Use hard veto so the effect is easy to isolate: some positions get 0, others base.
    cfg_half = _cfg(temperature=0.0, causal_lambda=0.5,
                    defer_veto_type="hard", defer_tau=0.0)
    cfg_full = _cfg(temperature=0.0, causal_lambda=1.0,
                    defer_veto_type="hard", defer_tau=0.0)
    cfg_none = _cfg(temperature=0.0, causal_lambda=0.0,
                    defer_veto_type="hard", defer_tau=0.0)
    for c in (cfg_half, cfg_full, cfg_none):
        c.return_debug = True

    recs_half, recs_full, recs_none = [], [], []
    pick_transfer_defer_only(logits, drop, cfg_half, mask_index, tokens, debug_records=recs_half, step_idx=0)
    pick_transfer_defer_only(logits, drop, cfg_full, mask_index, tokens, debug_records=recs_full, step_idx=0)
    pick_transfer_defer_only(logits, drop, cfg_none, mask_index, tokens, debug_records=recs_none, step_idx=0)

    # Index by (batch, position) for cross-comparison.
    def by_pos(rs):
        return {(r["batch"], r["position"]): r for r in rs}
    half = by_pos(recs_half)
    full = by_pos(recs_full)
    none = by_pos(recs_none)

    common_keys = set(half.keys()) & set(full.keys()) & set(none.keys())
    assert len(common_keys) > 0, "no common committed positions across lambda settings"
    for key in common_keys:
        b = none[key]["defer_base_conf"]
        v = full[key]["defer_eff_conf"]
        h = half[key]["defer_eff_conf"]
        expected = 0.5 * b + 0.5 * v
        assert abs(h - expected) < 1e-6, (
            f"blend broken at pos={key}: half={h}, expected 0.5*base({b}) + 0.5*veto({v}) = {expected}"
        )


def test_lambda_zero_and_sanity_equal_plain_dcd():
    """lambda=0 with drop_logits STILL matches plain DCD, byte-identical."""
    torch.manual_seed(19)
    logits, drop, mask_index, tokens = _tiny_step(vocab=16, seq=5)
    cfg_defer = _cfg(causal_lambda=0.0, temperature=0.0)
    cfg_base = _cfg(causal_lambda=0.0, temperature=0.0, cv_mode="off")
    x0_d, tr_d = pick_transfer_defer_only(logits, drop, cfg_defer, mask_index, tokens)
    x0_b, tr_b = _pick_transfer(logits, cfg_base, mask_index, tokens)
    assert torch.equal(x0_d, x0_b)
    assert torch.equal(tr_d, tr_b)


def test_lambda_zero_base_conf_bit_identical_to_plain_dcd():
    """STRICT parity: defer_only base_conf at lambda=0 must equal
    _confidence_from_logits (plain DCD) bit-for-bit.

    Prior to the E0-parity fix, defer_only used a memory-efficient
    ``logsumexp(logits) [in bfloat16] -> cast fp64`` path, which diverged
    from plain DCD's ``F.softmax(logits.to(fp64))`` by ~2^-7 in the last
    bits and could occasionally tip a token across threshold=0.9. This
    test regresses the fix: eff_conf recorded in the defer debug records
    must match ``_confidence_from_logits`` for the SAME argmax token.
    """
    torch.manual_seed(41)
    for _ in range(5):
        logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=6)
        cfg = _cfg(causal_lambda=0.0, temperature=0.0,
                   defer_veto_type="mult", defer_tau=0.0)
        cfg.return_debug = True
        recs: list = []
        pick_transfer_defer_only(
            logits, drop, cfg, mask_index, tokens,
            debug_records=recs, step_idx=0,
        )
        x0 = torch.argmax(logits, dim=-1)
        # Reference: plain DCD's confidence computation, verbatim.
        conf_ref = _confidence_from_logits(logits, x0, "low_confidence")
        for r in recs:
            b, pos = r["batch"], r["position"]
            e_defer = r["defer_eff_conf"]
            b_defer = r["defer_base_conf"]
            expected = float(conf_ref[b, pos].item())
            assert e_defer == expected, (
                f"conf drift at (batch={b}, pos={pos}): "
                f"defer={e_defer!r}, plain_dcd={expected!r}"
            )
            assert b_defer == expected  # base and eff match at lambda=0


def test_lambda_zero_stress_bit_identical_plain_dcd():
    """Stress: over many seeds and shapes, defer_only lambda=0 must produce
    the same (x0, transfer_index) as plain DCD.
    """
    for seed in (0, 1, 7, 17, 41, 101):
        torch.manual_seed(seed)
        for vocab, seq in ((16, 5), (32, 8), (64, 4), (16, 12)):
            logits = torch.randn(1, seq, vocab) * 1.5
            drop = torch.randn(1, seq, vocab) * 1.5
            tokens = torch.zeros(1, seq, dtype=torch.long)
            n_mask = max(2, seq // 2)
            tokens[:, :n_mask] = 126336
            mask_index = tokens == 126336
            cfg_d = _cfg(causal_lambda=0.0, temperature=0.0,
                         defer_veto_type="soft", defer_tau=-1.0)
            cfg_b = _cfg(causal_lambda=0.0, temperature=0.0, cv_mode="off")
            x0_d, tr_d = pick_transfer_defer_only(logits, drop, cfg_d, mask_index, tokens)
            x0_b, tr_b = _pick_transfer(logits, cfg_b, mask_index, tokens)
            assert torch.equal(x0_d, x0_b), f"x0 mismatch seed={seed} V={vocab} L={seq}"
            assert torch.equal(tr_d, tr_b), f"tr mismatch seed={seed} V={vocab} L={seq}"


def test_lambda_zero_with_temperature_bit_identical_plain_dcd():
    """Stress: same as above but with temperature > 0 (Gumbel active).

    Since both paths use the SAME ``add_gumbel_noise`` formula and consume
    RNG the same way (float64 rand_like on same shape), byte-identity must
    still hold. If this test fails, either the Gumbel implementations have
    diverged or an intermediate RNG consumer has been introduced.
    """
    for seed in (2, 5, 13, 29, 53):
        torch.manual_seed(seed)
        logits = torch.randn(1, 8, 32) * 1.5
        drop = torch.randn(1, 8, 32) * 1.5
        tokens = torch.zeros(1, 8, dtype=torch.long)
        tokens[:, :4] = 126336
        mask_index = tokens == 126336
        cfg_d = _cfg(causal_lambda=0.0, temperature=0.8,
                     defer_veto_type="hard", defer_tau=-3.0)
        cfg_b = _cfg(causal_lambda=0.0, temperature=0.8, cv_mode="off")
        # Reseed just before each call so Gumbel RNG state matches.
        torch.manual_seed(seed)
        x0_d, tr_d = pick_transfer_defer_only(logits, drop, cfg_d, mask_index, tokens)
        torch.manual_seed(seed)
        x0_b, tr_b = _pick_transfer(logits, cfg_b, mask_index, tokens)
        assert torch.equal(x0_d, x0_b), f"x0 mismatch (temp>0) seed={seed}"
        assert torch.equal(tr_d, tr_b), f"tr mismatch (temp>0) seed={seed}"


def test_lambda_intermediate_produces_active_defer():
    """lambda in (0,1) with gain < tau => defer_active must fire on those positions."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=6)
    torch.manual_seed(23)

    cfg = _cfg(temperature=0.0, causal_lambda=0.4,
               defer_veto_type="hard", defer_tau=1e9)  # tau huge => gain always < tau => full veto
    cfg.return_debug = True
    recs: list = []
    pick_transfer_defer_only(logits, drop, cfg, mask_index, tokens, debug_records=recs, step_idx=0)
    # With hard + tau=+inf, every committed record should have eff = 0.6 * base + 0.4 * 0 = 0.6 * base.
    for r in recs:
        b = r["defer_base_conf"]
        e = r["defer_eff_conf"]
        assert abs(e - 0.6 * b) < 1e-6, f"blend at lambda=0.4 wrong: base={b}, eff={e}, expected {0.6*b}"
        assert r["defer_active"] is True
        assert r["defer_lambda"] == 0.4
        assert r["conf_source_used"] == "defer_hard"


def test_pick_transfer_gain_type_config_honored():
    """Config field ``defer_gain_type`` must be threaded through to the debug record."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=32, seq=4)
    for gt in ("logit", "logprob"):
        cfg = _cfg(temperature=0.0, defer_gain_type=gt, defer_veto_type="soft")
        cfg.return_debug = True
        records: list = []
        _ = pick_transfer_defer_only(
            logits, drop, cfg, mask_index, tokens,
            debug_records=records, step_idx=0,
        )
        assert len(records) > 0
        assert records[0]["defer_gain_type"] == gt


def test_pick_transfer_soft_zero_beta_equals_base_conf():
    """soft veto with beta=0 => eff = base_conf (no penalty)."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=16, seq=5)
    cfg = _cfg(defer_veto_type="soft", defer_beta=0.0, defer_tau=0.0, temperature=0.0)
    cfg_base = _cfg(cv_mode="off", temperature=0.0)
    x0_d, tr_d = pick_transfer_defer_only(logits, drop, cfg, mask_index, tokens)
    x0_b, tr_b = _pick_transfer(logits, cfg_base, mask_index, tokens)
    assert torch.equal(x0_d, x0_b)
    assert torch.equal(tr_d, tr_b)


class _FakeBatchSensitiveOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        batch, seq = logits.shape[:2]
        cache = torch.zeros(batch, 1, seq, 1)
        self.past_key_values = [(cache, cache)]


class _FakeBatchSensitiveModel:
    """Fake model that changes logits if CV accidentally uses paired batch=2."""

    def __init__(self, vocab: int = 8):
        self.vocab = vocab
        self.batch_sizes: list[int] = []

    def __call__(self, tokens, **_kwargs):
        self.batch_sizes.append(int(tokens.shape[0]))
        batch, seq = tokens.shape
        logits = torch.full((batch, seq, self.vocab), -10.0)
        logits[..., 1] = 10.0
        if batch > 1:
            # Regression trap: the old lambda=0 CV dual-cache path still ran a
            # paired forward with [base, drop], which could perturb base logits.
            logits[0, ..., 2] = 11.0
        return _FakeBatchSensitiveOutput(logits)


def test_cv_dual_cache_lambda_zero_uses_single_forward_like_dcd():
    """Regression: lambda=0 CV dual-cache must not call paired_forward(batch=2)."""
    mask_id = 126336
    tokens = torch.tensor([[7, mask_id, mask_id, mask_id]])
    cfg = _cfg(
        causal_lambda=0.0,
        cache_type="dual",
        block_size=3,
        decode_param=0.9,
        cv_mode="defer_only",
        temperature=0.0,
    )

    base_model = _FakeBatchSensitiveModel()
    cv_model = _FakeBatchSensitiveModel()
    base_out = dcd_decode_text_dual_cache(base_model, tokens, 1, 4, cfg)
    cv_out = dcd_decode_text_cv_dual_cache(cv_model, tokens, 1, 4, cfg)

    assert torch.equal(cv_out, base_out)
    assert max(cv_model.batch_sizes) == 1


def test_pick_transfer_defers_when_gain_very_negative():
    """Hard veto with tau=0: negative-gain positions must have eff_conf=0
    and therefore not commit under threshold=0.9."""
    # Construct a case where base thinks token 0 is very confident (~1.0)
    # but drop_logits also strongly prefers token 0 with even higher confidence
    # => gain very negative => hard veto zeros it out.
    vocab = 4
    base = torch.zeros(1, 2, vocab)
    base[0, 0, 0] = 5.0     # base: token 0 with high confidence
    base[0, 1, 0] = 5.0
    drop = torch.zeros(1, 2, vocab)
    drop[0, 0, 0] = 10.0    # drop: even more confident about token 0
    drop[0, 1, 0] = 10.0
    tokens = torch.tensor([[126336, 126336]])
    mask_index = tokens == 126336
    cfg = _cfg(defer_veto_type="hard", defer_tau=0.0, causal_clip=4.0)
    _, tr = pick_transfer_defer_only(base, drop, cfg, mask_index, tokens)
    # gain = base_logp - drop_logp ~ (0 - 0)? No -- both are argmax so logp
    # is ~0 for both (softmax([5,0,0,0]) ~ [0.99, 0.003, 0.003, 0.003]).
    # drop softmax([10,0,0,0]) ~ [1.0, ~0, ~0, ~0]. So drop_lp > base_lp =>
    # gain < 0 => hard veto zeros eff_conf. threshold=0.9 has no candidate
    # => fallback picks top-1 (still commits 1 position per batch).
    # We only assert commit count is 1 (fallback), not 2.
    assert tr.sum().item() == 1


# --------------------------------------------------------------------------
# 4. Routing via _pick_transfer_cv
# --------------------------------------------------------------------------


def test_pick_transfer_cv_routes_defer_only():
    logits, drop, mask_index, tokens = _tiny_step(vocab=16, seq=4)
    cfg = _cfg(cv_mode="defer_only")
    x0, tr = _pick_transfer_cv(logits, drop, cfg, mask_index, tokens)
    # Must equal direct call.
    x0d, trd = pick_transfer_defer_only(logits, drop, cfg, mask_index, tokens)
    assert torch.equal(x0, x0d)
    assert torch.equal(tr, trd)


def test_v32_still_works_after_defer_added():
    """Regression: v3.2 cd_apc_v32 path still runs cleanly."""
    logits, drop, mask_index, tokens = _tiny_step(vocab=16, seq=4)
    cfg = MMaDADecodeConfig(
        mask_id=126336,
        decode_algo="threshold", decode_param=0.9,
        temperature=0.0, remasking="low_confidence",
        causal_lambda=0.5, cv_alpha=0.1,
        cv_mode="cd_apc_v32", cv_conf_source="min_base_blended",
        cv_gate_tau=0.0,
    )
    x0, tr = _pick_transfer_cv(logits, drop, cfg, mask_index, tokens)
    assert x0.shape == tokens.shape
    assert tr.dtype == torch.bool


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _run_all_tests():
    ns = dict(globals())
    tests = sorted(k for k in ns if k.startswith("test_") and callable(ns[k]))
    passed = 0
    failed = []
    for name in tests:
        try:
            ns[name]()
        except AssertionError as e:
            failed.append((name, f"AssertionError: {e}"))
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed}/{len(tests)} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all_tests())
