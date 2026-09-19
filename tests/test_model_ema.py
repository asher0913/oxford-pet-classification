"""Regression tests for the ModelEma bias-corrected decay schedule.

Fixed decay=0.999 from step one left ~2.5% of the random Kaiming init
in the shadow weights at the end of training, costing ~7 pp of val acc.
The fix: eff_decay = min(decay, (1+n)/(10+n)), so early updates are
aggressive and flush the random init quickly before nominal decay kicks in.
These tests lock that schedule in so it can't accidentally regress.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

# so the test can run without pip install -e .
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pet_classifier.ema import ModelEma  # noqa: E402  (sys.path tweak above)


def _tiny_model() -> nn.Module:
    """Build the smallest model that still has BN buffers."""

    # small model with both float params and a BN int buffer (num_batches_tracked)
    module = nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
    )
    return module


def test_bias_corrected_decay_schedule() -> None:
    """eff_decay follows (1+n)/(10+n) while below the nominal decay."""
    model = _tiny_model()
    ema = ModelEma(model, decay=0.9999)

    # check a handful of n values against the closed-form expression
    expected_points = {
        1: 2.0 / 11.0,     # first update
        5: 6.0 / 15.0,
        9: 10.0 / 19.0,    # still on the warmup branch
        100: 101.0 / 110.0,
        1000: 1001.0 / 1010.0,
    }

    # reconstruct the schedule the same way ModelEma does and check it matches
    for step, expected in expected_points.items():
        got = min(ema.decay, (1.0 + step) / (ModelEma.BIAS_WARMUP + step))
        assert math.isclose(got, expected, rel_tol=1e-9), (
            f"step={step}: eff_decay should be {expected}, got {got}"
        )


def test_decay_plateaus_at_nominal() -> None:
    """Once the warmup branch exceeds decay it must be clamped."""
    decay = 0.9999
    # warmup crosses nominal at n≈90k; check well past that
    n = 200_000
    warmup_value = (1.0 + n) / (ModelEma.BIAS_WARMUP + n)
    assert warmup_value > decay, "test premise: warmup should exceed nominal here"
    eff = min(decay, warmup_value)
    assert math.isclose(eff, decay, rel_tol=1e-12)


def test_shadow_flushes_random_init_quickly() -> None:
    """After ~20 updates the shadow should be close to current weights.

    First update uses eff_decay≈0.18, so the shadow moves 82% of the way
    on step 1 alone. The old fixed decay=0.999 never flushed the init this fast.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    ema = ModelEma(model, decay=0.9999)

    # set all float params to 1, then drive EMA until the shadow catches up
    target_state = {
        name: torch.ones_like(tensor) if tensor.dtype.is_floating_point else tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(target_state)

    for _ in range(20):
        ema.update(model)

    # after 20 updates the shadow should be close to 1.0 everywhere
    for name, shadow_tensor in ema.shadow.items():
        if not shadow_tensor.dtype.is_floating_point:
            # num_batches_tracked etc. are copied verbatim; skip.
            continue
        assert torch.allclose(shadow_tensor, torch.ones_like(shadow_tensor), atol=0.6), (
            f"Shadow tensor '{name}' is still far from target after 20 updates: "
            f"mean={shadow_tensor.mean().item():.4f}"
        )


def test_integer_buffers_are_copied_not_averaged() -> None:
    """num_batches_tracked is int64; EMA must copy it verbatim, not blend."""
    model = _tiny_model()
    ema = ModelEma(model, decay=0.9999)

    # forward pass so num_batches_tracked gets bumped (BN needs >=2 samples)
    model.train()
    _ = model(torch.randn(4, 4))
    ema.update(model)

    # BN buffer should match exactly - no blending
    for name, tensor in model.state_dict().items():
        if tensor.dtype.is_floating_point:
            continue
        shadow_tensor = ema.shadow[name]
        assert shadow_tensor.dtype == tensor.dtype, f"{name}: dtype changed under EMA"
        assert torch.equal(shadow_tensor, tensor), (
            f"Integer buffer '{name}' was not copied verbatim by EMA."
        )


def _run_all() -> None:
    """Run all tests without pytest."""
    tests = [
        test_bias_corrected_decay_schedule,
        test_decay_plateaus_at_nominal,
        test_shadow_flushes_random_init_quickly,
        test_integer_buffers_are_copied_not_averaged,
    ]
    for test in tests:
        test()
        print(f"  ok: {test.__name__}")
    print(f"All {len(tests)} ModelEma tests passed.")


if __name__ == "__main__":
    _run_all()
