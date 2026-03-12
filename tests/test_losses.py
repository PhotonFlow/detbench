"""Unit tests for detbench.losses."""

from __future__ import annotations

import pytest
import torch

from detbench.losses import (
    CrossEntropyLoss,
    FocalLoss,
    GeneralizedCrossEntropy,
    MeanAbsoluteErrorLoss,
    SymmetricCrossEntropy,
    get_loss,
    LOSS_REGISTRY,
)


NUM_CLASSES = 3
BATCH_SIZE = 8


@pytest.fixture
def sample_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return logits, targets


class TestIndividualLosses:
    def test_ce_returns_scalar(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, targets = sample_data
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_focal_returns_scalar(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, targets = sample_data
        loss = FocalLoss(gamma=2.0)(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gce_returns_scalar(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, targets = sample_data
        loss = GeneralizedCrossEntropy(q=0.7)(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_sce_returns_scalar(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, targets = sample_data
        loss = SymmetricCrossEntropy(num_classes=NUM_CLASSES)(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_mae_returns_scalar(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, targets = sample_data
        loss = MeanAbsoluteErrorLoss(num_classes=NUM_CLASSES)(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestLossFactory:
    def test_all_registry_entries(
        self, sample_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        logits, targets = sample_data
        for name in LOSS_REGISTRY:
            loss_fn = get_loss(name, NUM_CLASSES)
            loss = loss_fn(logits, targets)
            assert loss.ndim == 0, f"{name} did not return a scalar"
            assert torch.isfinite(loss), f"{name} returned non-finite"

    def test_unknown_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("NONEXISTENT", 3)

    def test_focal_less_than_ce_on_easy(self) -> None:
        # For a "confident" prediction, focal loss should be less than CE
        logits = torch.tensor([[10.0, -10.0, -10.0]])
        targets = torch.tensor([0])
        ce = CrossEntropyLoss()(logits, targets)
        fl = FocalLoss(gamma=2.0)(logits, targets)
        assert fl.item() <= ce.item()
