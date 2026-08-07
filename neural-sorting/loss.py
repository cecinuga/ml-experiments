

import numpy as np

"""Loss function for disorder"""
def disorder(x: np.ndarray) -> float:
    assert len(x) > 0

    """Base case: a vector with lenth 1 has no disorder."""
    if len(x) == 1: return 0
    
    return sum([abs(x[i] - x[i-1]) for i in range(1, len(x))])

"""Tiny-Torch compliant implementation of loss function (must be moved on framework repo as soon as possible)"""

from typing import override
from thorcino.tensor import Tensor
from thorcino.autograd import Function
from thorcino.losses import Loss

class DisorderLossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:

        return Tensor(),

class DisorderLoss(Loss):
    grad_fn:type[Function] = DisorderLossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor()
        out._grad_fn = self.grad_fn(predictions, targets)
        return out