from collections import defaultdict
import typing as tp

import torch
from torch import autograd
from torch.nn import Module
import torch.distributed as dist

from .distrib import average_metrics


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatedly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


class Balancer(Module):
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        ema_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(self, weights: tp.Dict[str, float], rescale_grads: bool = True, total_norm: float = 1.,
                 ema_decay: float = 0.999, per_batch_item: bool = True, epsilon: float = 1e-12,
                 monitor: bool = False):
        super(Balancer, self).__init__()
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

        # Register buffers for norms
        self.register_buffer('norms_buffer', torch.zeros(len(weights)))

    @property
    def metrics(self):
        return self._metrics

    def _sync_buffers(self, norms: tp.Dict[str, float]):
        # Gather norms into a single tensor
        norms_tensor = torch.tensor([norms[k] for k in self.weights.keys()], device=self.norms_buffer.device)
        
        # Perform a single all_reduce operation
        dist.all_reduce(norms_tensor, op=dist.ReduceOp.SUM)
        
        # Update the norms buffer
        self.norms_buffer.copy_(norms_tensor / dist.get_world_size())
        
        # Extract the norms back into a dictionary
        synced_norms = {k: v.item() for k, v in zip(self.weights.keys(), self.norms_buffer)}

        return synced_norms

    def compute_scaling_factors(self, norms: tp.Dict[str, float]) -> tp.Dict[str, float]:
        avg_norms = average_metrics(self.averager(norms), 1)

        # Sync buffers
        synced_norms = self._sync_buffers(avg_norms)

        total = sum(synced_norms.values())

        self._metrics = {}
        if self.monitor:
            for k, v in synced_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in synced_norms])
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        scaling_factors = {}
        for name, avg_norm in synced_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                scaling_factors[name] = scale
            else:
                scaling_factors[name] = self.weights[name]

        return scaling_factors

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor):
        norms = {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[name] = norm

        scaling_factors = self.compute_scaling_factors(norms)

        scaled_losses = {name: loss * scaling_factors[name] for name, loss in losses.items()}
        total_loss = sum(scaled_losses.values())
        return total_loss


def test():
    from torch.nn import functional as F
    x = torch.zeros(1, requires_grad=True)
    one = torch.ones_like(x)
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}

    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=False)
    balancer.backward(losses, x)
    assert torch.allclose(x.grad, torch.tensor(99.)), x.grad

    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}
    x.grad = None
    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=True)
    balancer.backward({'1': loss_1, '2': loss_2}, x)
    assert torch.allclose(x.grad, torch.tensor(0.)), x.grad


if __name__ == '__main__':
    test()
