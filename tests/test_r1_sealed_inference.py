from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_r1_sealed_inference import (  # noqa: E402
    install_memory_bounded_refine_forward,
    install_memory_bounded_score_data,
    install_memory_bounded_score_forward,
)


class _Batch:
    def __init__(self) -> None:
        self.rgbAs = None
        self.values = None
        self.constant = None


class _Dataset:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def transform_batch(
        self, batch: _Batch, H_ori: int, W_ori: int, bound: int = 1
    ) -> _Batch:
        self.calls.append(int(batch.rgbAs.shape[0]))
        batch.values = batch.values * 2 + H_ori + W_ori + bound
        return batch


def test_score_data_chunking_preserves_order_and_values() -> None:
    scorer = type("Scorer", (), {})()
    scorer.dataset = _Dataset()
    install_memory_bounded_score_data(scorer, batch_size=3)
    batch = _Batch()
    batch.rgbAs = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    batch.values = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    batch.constant = "same"
    result = scorer.dataset.transform_batch(batch, H_ori=11, W_ori=13, bound=2)
    assert scorer.dataset.calls == [3, 3, 3, 1]
    assert torch.equal(
        result.values,
        torch.arange(10, dtype=torch.float32).reshape(10, 1) * 2 + 26,
    )
    assert result.constant == "same"


class _CrossAttention(torch.nn.Module):
    def forward(self, query: torch.Tensor, _key: torch.Tensor, _value: torch.Tensor):
        return query + query.mean(dim=1, keepdim=True), None


class _ScoreModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.att_cross = _CrossAttention()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        self.linear.weight.data.copy_(torch.tensor([[0.25, -0.5]]))

    def extract_feat(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack((a[:, 0] + b[:, 0], a[:, 1] - b[:, 1]), dim=1)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, length: int
    ) -> dict[str, torch.Tensor]:
        outer = a.shape[0] // length
        features = self.extract_feat(a, b).reshape(outer, length, -1)
        attended, _ = self.att_cross(features, features, features)
        return {"score_logit": self.linear(attended).reshape(outer, length)}


def test_score_feature_chunking_keeps_full_candidate_attention() -> None:
    torch.manual_seed(7)
    model = _ScoreModel()
    a = torch.randn(12, 2)
    b = torch.randn(12, 2)
    expected = model(a, b, 6)["score_logit"]
    scorer = type("Scorer", (), {"model": model})()
    install_memory_bounded_score_forward(scorer, batch_size=4)
    actual = scorer.model(a, b, L=6)["score_logit"]
    assert torch.equal(actual, expected)


class _RefineModel(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"rot": a - b, "trans": a + 2 * b}


def test_refine_chunking_preserves_outputs() -> None:
    torch.manual_seed(11)
    model = _RefineModel()
    a = torch.randn(11, 3)
    b = torch.randn(11, 3)
    expected = model(a, b)
    refiner = type("Refiner", (), {"model": model})()
    install_memory_bounded_refine_forward(refiner, batch_size=4)
    actual = refiner.model(a, b)
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key])
