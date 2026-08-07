

import numpy as np

"""Loss function for disorder"""
def disorder(x: np.ndarray) -> float:
    assert len(x) > 0

    """Base case: a vector with lenth 1 has no disorder."""
    if len(x) == 1: return 0
    
    return np.sum(np.abs(np.diff(x))) 

def disorder_grad(x: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    if len(x) >= 1:
        s = np.sign(np.diff(x))
        g[:-1] -= s
        g[1:] += s
    return g

"""Tiny-Torch compliant implementation of loss function (must be moved on framework repo as soon as possible)"""

from typing import override
from thorcino.tensor import Tensor
from thorcino.autograd import Function
from thorcino.losses import Loss

class DisorderLossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        pred, targ = self.saved_tensors
        grad_pred = grad_output.data * disorder_grad(pred.data)
        grad_targ = -grad_output.data * disorder_grad(targ.data)

        return Tensor(grad_pred), Tensor(grad_targ),

class DisorderLoss(Loss):
    grad_fn:type[Function] = DisorderLossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pred_disorder = disorder(predictions)
        targ_disorder = disorder(targets)

        out = Tensor(pred_disorder - targ_disorder)
        out._grad_fn = self.grad_fn(predictions, targets)
        return out