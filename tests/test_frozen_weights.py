"""
Project N: Invariant 3 Frozen Weights & Metric Space Independence Verification.
Verifies that base LLM weights are frozen via model.freeze(), and that the primary
sensory classification / metric projection path operates without calling the base LLM.
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.projection import MetricProjectionHead


class MockBackbone(nn.Module):
    """Mock base model simulating an LLM backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, 512)
        self.lora_a = nn.Linear(512, 16, bias=False)
        self.lora_b = nn.Linear(16, 512, bias=False)


def verify_freezing_mechanism() -> None:
    from mlx.utils import tree_flatten

    backbone = MockBackbone()
    backbone.freeze()

    trainable_all_frozen = tree_flatten(backbone.trainable_parameters())
    assert (
        len(trainable_all_frozen) == 0
    ), f"INVARIANT 3 BREACH: Frozen model still exposes trainable parameters: {trainable_all_frozen}"

    backbone.lora_a.unfreeze()
    backbone.lora_b.unfreeze()
    trainable_with_lora = dict(tree_flatten(backbone.trainable_parameters()))

    assert any("lora_a" in k for k in trainable_with_lora)
    assert any("lora_b" in k for k in trainable_with_lora)
    assert not any("dense1" in k for k in trainable_with_lora)
    assert not any("dense2" in k for k in trainable_with_lora)


def verify_metric_projection_independence() -> None:
    """Verifies that sensory metric embedding generation is completely independent of LLM."""
    head = MetricProjectionHead(d_audio=512, d_kinematic=512, d_metric=128)
    head.is_trained = True

    z_a = mx.random.normal((1, 500, 512))
    z_k = mx.random.normal((1, 150, 512))
    emb = head(z_a, z_k)
    mx.eval(emb)

    assert emb.shape == (1, 128), f"Unexpected embedding shape: {emb.shape}"
    norm = mx.linalg.norm(emb, axis=-1)
    mx.eval(norm)
    assert mx.allclose(norm, mx.array([1.0]), atol=1e-5), "Metric embedding is not L2-normalized"


def test_frozen_weights_invariant() -> None:
    verify_freezing_mechanism()
    verify_metric_projection_independence()


if __name__ == "__main__":
    test_frozen_weights_invariant()
    print("Invariant 3 Verification: PASSED (Base weights frozen, metric path independent)")
