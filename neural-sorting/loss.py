
from typing import override
from thorcino.tensor import Tensor
from thorcino.autograd import Function
from thorcino.losses import Loss
import numpy as np

"""Loss function for disorder"""
def disorder(x: np.ndarray) -> float:
    assert len(x) > 0

    """Base case: a vector with lenth 1 has no disorder."""
    if len(x) == 1: return 0
    
    return np.sum(np.abs(np.diff(x))) 

def disorder_grad(x: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    if len(x) > 1:
        s = np.sign(np.diff(x))
        g[:-1] -= s
        g[1:] += s

    return g

def disorder_test(row: int, length_upto: list[int], seed: float) -> None:
    rng = np.random.default_rng(seed)

    for length in range(2, length_upto):
        X = rng.standard_normal((row, length))
        loss_test_dataset = np.stack([X, np.sort(X)])
        loss_test_dataset = loss_test_dataset.swapaxes(0, 1)

        for i, (x, y) in enumerate(loss_test_dataset):
            random_disorder = round(disorder(x), 5)
            sorted_disorder = round(disorder(y), 5)

            try:
                assert sorted_disorder <= random_disorder
            except Exception as e:
                print(f'iter={i}, lenght={length}')
                print(f"random disorder={disorder(x)}")
                print(f"sorted disorder={disorder(y)}")
                print('-----------------------------------------')
                print(f"random={x}")
                print(f"sorted={y}")
                raise e



"""Loss function for disorder V2: without sum reduction"""

def disorder_v2(x: np.ndarray) -> np.ndarray:
    assert len(x) > 0

    """Base case: a vector with lenth 1 has no disorder."""
    if len(x) == 1: return 0

    return np.abs(np.diff(x))


def disorder_v2_grad(x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    if len(x) > 1:
        s = np.sign(np.diff(x))
        weighted = grad_output * s
        g[:-1] -= weighted
        g[1:] += weighted

    return g


"""Tiny-Torch compliant implementation of loss function (must be moved on framework repo as soon as possible)"""

class DisorderLossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        pred, targ = self.saved_tensors

        u = disorder(pred.data) - disorder(targ.data)
        s = np.sign(u)

        grad_pred = grad_output.data * s * disorder_grad(pred.data)
        grad_targ = -grad_output.data * s * disorder_grad(targ.data)

        return Tensor(grad_pred), Tensor(grad_targ),

class DisorderLoss(Loss):
    grad_fn:type[Function] = DisorderLossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pred_disorder = disorder(predictions)
        targ_disorder = disorder(targets)

        out = Tensor(np.abs(pred_disorder - targ_disorder))
        out._grad_fn = self.grad_fn(predictions, targets)
        return out
    
"""Disorder V2 Tiny-Torch classes"""

class DisorderV2LossBackward(Function):
    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        pred, targ = self.saved_tensors

        grad_pred = disorder_v2_grad(pred.data, grad_output.data)
        grad_targ = -disorder_v2_grad(targ.data, grad_output.data)

        return Tensor(grad_pred), Tensor(grad_targ)

class DisorderV2Loss(Loss):
    grad_fn:type[Function] = DisorderV2LossBackward

    @override
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pred_disorder = disorder_v2(predictions)
        targ_disorder = disorder_v2(targets)

        out = Tensor(pred_disorder - targ_disorder)
        out._grad_fn = self.grad_fn(predictions, targets)
        return out