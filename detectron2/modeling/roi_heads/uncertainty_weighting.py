# Automatic multi-task loss weighting via homoscedastic uncertainty
#
# Based on:
#   Kendall, A., Gal, Y., & Cipolla, R. (2018).
#   "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR 2018.

# This module is intentionally self-contained so it can be imported by mask_head.py without any circular dependency

import torch
import torch.nn as nn

class UncertaintyWeightedLoss(nn.Module):
    """
    Learns a per-task log-variance scalar s_k = log(sigma_k^2) that is jointly optimised with the network weights during training

    For a CLASSIFICATION-type loss (e.g. BCE over a pixel grid):
        weighted = exp(-s) * L  +  0.5 * s

    For a REGRESSION-type loss (e.g. MAE over a scalar ratio):
        weighted = 0.5 * exp(-s) * L  +  0.5 * s

    The factor of 0.5 on the regression term comes from the Gaussian
    log-likelihood:  -log p = (1 / 2*sigma^2) * L + log(sigma).
    The classification term uses the Bernoulli formulation where sigma
    acts as a temperature:  -log p = (1 / sigma^2) * L + log(sigma).

    As s increases the effective task weight exp(-s) decreases, preventing
    any single task from dominating.  The +0.5*s regulariser prevents the
    model from achieving zero loss using a trivial solution s -> +inf

    Arguments
    ---------
    task_types : list[str]
        One entry per loss term, each either 'cls' or 'reg'
        The ordering must match the list of losses passed to forward()

    Notes
    -----
    * All log_var parameters are initialised to 0, i.e. sigma = 1, so the
      initial effective weight of every task is 1.0 (regression) or 1.0
      (classification). 
      Kendall et al. show the method is robust to this initialisation choice.
    * The parameters are registered as nn.Parameter so Detectron2's standard
      build_optimizer() call will automatically include them in the optimiser
      without any extra configuration.
    """

    def __init__(self, task_types: list):
        super().__init__()
        assert all(t in ("cls", "reg") for t in task_types), (
            "Each task_type must be 'cls' or 'reg'."
        )
        self.task_types = task_types

        # One learnable log(sigma^2) per task, initialised to 0.
        self.log_vars = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in task_types]
        )

    def forward(self, losses: list) -> torch.Tensor:
        """
        Parameters

        losses : list[torch.Tensor]
            Scalar loss tensors, one per task, in the same order as
            task_types supplied at construction

        Returns
        torch.Tensor
            Scalar — the sum of all uncertainty-weighted loss terms
        """
        assert len(losses) == len(self.task_types), (
            f"Expected {len(self.task_types)} losses, got {len(losses)}."
        )
        total = losses[0].new_zeros(1).squeeze()  # device-safe zero
        for loss, task_type, log_var in zip(losses, self.task_types, self.log_vars):
            # Hard-clamp s_k = log(sigma^2) before use.
            # Without this, the optimiser drives s_k -> -inf so that the
            # 0.5*s_k regulariser dominates (going deeply negative) while
            # exp(-s_k) -> +inf blows up gradients.  Clamping to [-5, 5]
            # keeps effective task weights in [exp(-5), exp(5)] ≈ [0.007, 148]
            # and limits each task's regulariser contribution to [-2.5, 2.5].
            s = torch.clamp(log_var, -5.0, 5.0)
            # Numerical shorthand: exp(-s) == 1/sigma^2
            precision = torch.exp(-s)
            if task_type == "cls":
                # (1/sigma^2) * L + log(sigma) == exp(-s)*L + 0.5*s
                total = total + precision * loss + 0.5 * s
            else:  # for regression tasks
                # (1/2*sigma^2) * L + log(sigma) == 0.5*exp(-s)*L + 0.5*s
                total = total + 0.5 * precision * loss + 0.5 * s
        return total

    def extra_repr(self) -> str:
        return f"task_types={self.task_types}"