
from typing import override
from thorcino.tensor import Tensor
from thorcino.autograd import Function, SubBackward, SumBackward
from thorcino.losses import Loss
import numpy as np

"""Loss function for disorder"""
def disorder_v1(x: np.ndarray) -> float:
    assert len(x) > 0

    """Base case: a vector with lenth 1 has no disorder."""
    if len(x) == 1: return 0
    
    return np.sum(np.abs(np.diff(x))) 

def perm_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[:, np.newaxis] == y[np.newaxis, :]

def cross_disorder(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cross_diff = perm_matrix(x, y)

"""Tiny-Torch compliant implementation of loss function (must be moved on framework repo as soon as possible)"""

class PermutationLossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        pred, targ = self.saved_tensors


        return Tensor(), Tensor(),

class PermutationLoss(Loss):
    grad_fn:type[Function] = PermutationLossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        

        out = Tensor()
        out._grad_fn = self.grad_fn(predictions, targets)
        return out
    

