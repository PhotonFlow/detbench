"""Noise-robust loss functions for detection/classification benchmarks.

Implements a suite of loss functions commonly used in research on
learning with noisy labels.  Each loss is implemented as a standard
``torch.nn.Module`` with a consistent ``forward(logits, targets)`` API.

Included losses
---------------
- **CE** — Standard Cross-Entropy
- **FL** — Focal Loss (Lin et al., ICCV 2017)
- **GCE** — Generalised Cross-Entropy (Zhang & Sabuncu, NeurIPS 2018)
- **SCE** — Symmetric Cross-Entropy (Wang et al., ICCV 2019)
- **MAE** — Mean Absolute Error (Ghosh et al., AAAI 2017)
- **NCE+MAE** — Active/Passive Normalised CE + MAE (Ma et al., ICML 2020)
- **NCE+RCE** — Active/Passive Normalised CE + Reverse CE
- **ANL-FL** — Active/Passive Focal + Reverse CE

References
----------
.. [1] Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.
.. [2] Zhang & Sabuncu. "Generalised Cross Entropy Loss for Training
       Deep Neural Networks with Noisy Labels." NeurIPS 2018.
.. [3] Wang et al. "Symmetric Cross Entropy for Robust Learning with
       Noisy Labels." ICCV 2019.
.. [4] Ma et al. "Normalised Loss Functions: Unifying PPL with Active
       Passive Losses." ICML 2020.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., ICCV 2017).

    Down-weights well-classified examples to focus on hard negatives.

    Parameters
    ----------
    gamma : float
        Focusing parameter (γ = 2 is standard).
    """

    def __init__(self, gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class GeneralizedCrossEntropy(nn.Module):
    """GCE Loss (Zhang & Sabuncu, NeurIPS 2018).

    A noise-robust generalisation of CE parameterised by *q*.
    When q → 0, GCE → CE; when q → 1, GCE → MAE.

    Parameters
    ----------
    q : float
        Truncation parameter (0 < q ≤ 1).  Default 0.7.
    """

    def __init__(self, q: float = 0.7, **kwargs: Any) -> None:
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1)
        p_y = p[torch.arange(p.shape[0], device=p.device), targets]
        return ((1 - (p_y**self.q)) / self.q).mean()


class SymmetricCrossEntropy(nn.Module):
    """SCE Loss (Wang et al., ICCV 2019).

    Combines standard CE with reverse CE for noise robustness.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    alpha : float
        Weight for the CE term.
    beta : float
        Weight for the reverse CE term.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.1,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self._ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(F.softmax(logits, dim=1), min=1e-7, max=1.0)
        label_oh = torch.clamp(
            F.one_hot(targets, self.num_classes).float(), min=1e-4, max=1.0
        )
        rce = -1 * torch.sum(pred * torch.log(label_oh), dim=1)
        return self.alpha * self._ce(logits, targets) + self.beta * rce.mean()


class MeanAbsoluteErrorLoss(nn.Module):
    """MAE Loss / Unhinged Loss (Ghosh et al., AAAI 2017).

    Inherently noise-tolerant for symmetric label noise.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    """

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).float()
        return torch.abs(pred - one_hot).mean()


class NormalizedCrossEntropy(nn.Module):
    """Normalised CE (Ma et al., ICML 2020)."""

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).float()
        loss = -1 * torch.sum(one_hot * torch.log(pred + 1e-7), dim=1)
        return loss.mean() / 10.0  # Approximate normalisation


class ReverseCrossEntropy(nn.Module):
    """Reverse CE (Ma et al., ICML 2020)."""

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(F.softmax(logits, dim=1), 1e-7, 1.0)
        one_hot = F.one_hot(targets, self.num_classes).float()
        return (-1 * torch.sum(pred * torch.log(one_hot + 1e-8), dim=1)).mean()


class ActivePassiveLoss(nn.Module):
    """Composite Active-Passive loss (Ma et al., ICML 2020).

    Combines an *active* loss (e.g. NCE, FL) with a *passive* loss
    (e.g. MAE, RCE) via a weighted sum.

    Parameters
    ----------
    active : nn.Module
        Active loss component.
    passive : nn.Module
        Passive loss component.
    alpha : float
        Weight for the active component.
    beta : float
        Weight for the passive component.
    """

    def __init__(
        self,
        active: nn.Module,
        passive: nn.Module,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.active = active
        self.passive = passive
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.active(logits, targets) + self.beta * self.passive(
            logits, targets
        )


# ---------------------------------------------------------------
# Factory
# ---------------------------------------------------------------

LOSS_REGISTRY: dict[str, str] = {
    "CE": "Cross-Entropy",
    "FL": "Focal Loss",
    "GCE": "Generalised CE",
    "SCE": "Symmetric CE",
    "MAE": "Mean Absolute Error",
    "NCE-MAE": "Normalised CE + MAE",
    "NCE-RCE": "Normalised CE + Reverse CE",
    "NCE-AGCE": "Normalised CE + GCE",
    "ANL-FL": "Active Focal + Reverse CE",
}


def get_loss(name: str, num_classes: int) -> nn.Module:
    """Instantiate a loss function by short name.

    Parameters
    ----------
    name : str
        One of: ``CE``, ``FL``, ``GCE``, ``SCE``, ``MAE``,
        ``NCE-MAE``, ``NCE-RCE``, ``NCE-AGCE``, ``ANL-FL``.
    num_classes : int
        Number of target classes.

    Returns
    -------
    nn.Module
        The requested loss module.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    k: dict[str, Any] = {"num_classes": num_classes}

    simple: dict[str, type[nn.Module]] = {
        "CE": CrossEntropyLoss,
        "FL": FocalLoss,
        "GCE": GeneralizedCrossEntropy,
        "SCE": SymmetricCrossEntropy,
        "MAE": MeanAbsoluteErrorLoss,
    }
    if name in simple:
        return simple[name](**k)

    nce = NormalizedCrossEntropy(**k)
    rce = ReverseCrossEntropy(**k)
    composite: dict[str, nn.Module] = {
        "NCE-MAE": ActivePassiveLoss(nce, MeanAbsoluteErrorLoss(**k)),
        "NCE-RCE": ActivePassiveLoss(nce, rce),
        "NCE-AGCE": ActivePassiveLoss(nce, GeneralizedCrossEntropy(**k)),
        "ANL-FL": ActivePassiveLoss(FocalLoss(**k), rce),
    }
    if name in composite:
        return composite[name]

    msg = f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}"
    raise ValueError(msg)
