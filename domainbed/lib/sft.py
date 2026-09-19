"""SFT losses from arXiv:2412.13573v2, Algorithms 1/2 and Eqs. (6)-(13).

The SAM model step stops the perturbation gradient. Refinement differentiates
the normalized perturbation w.r.t. soft labels, holding the *updated* model
weights fixed. Functional forwards avoid changing live weights or probe buffers.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@torch.no_grad()
def project_labels(logits, labels, alpha):
    """KL(q || softmax(logits)) projection with q[y] >= alpha*q[k].

    Algorithm 2, batched and evaluated in log space. Sorting other classes and
    prefix sums evaluate all candidate active sets in O(C log C), without a
    Python loop over samples/classes or products of tiny probabilities.
    """
    if not math.isfinite(alpha) or alpha < 1:
        raise ValueError("SFT alpha must be finite and >= 1")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("SFT projection expects nonempty [batch, classes>=2] logits")
    if (labels.shape != logits.shape[:1] or labels.dtype != torch.long
            or labels.device != logits.device):
        raise ValueError("SFT labels must be a matching int64 vector on the logits device")
    if not torch.isfinite(logits).all():
        raise ValueError("SFT projection requires finite logits")
    if (labels < 0).any() or (labels >= logits.shape[1]).any():
        raise ValueError("SFT labels are outside the class range")

    # Preserve double precision for numerical checks; promote low-precision input.
    work = logits if logits.dtype == torch.float64 else logits.float()
    log_p = F.log_softmax(work, dim=-1)
    true = log_p.gather(1, labels[:, None])
    log_alpha = math.log(alpha)
    others = log_p.scatter(1, labels[:, None], float("-inf"))
    ordered, indices = others.sort(dim=1, descending=True)
    ordered, indices = ordered[:, :-1], indices[:, :-1]
    candidates = ordered + log_alpha
    prefix = torch.cat((torch.zeros_like(true), candidates.cumsum(dim=1)), dim=1)
    count = torch.arange(logits.shape[1], device=logits.device, dtype=work.dtype)
    # Divide by alpha first so a large (but finite) alpha cannot overflow the
    # product alpha*log_p[y]. This is the same weighted geometric mean in log space.
    thresholds = (true + prefix / alpha) / (1 + count / alpha)
    # Once a constraint is inactive, all following (smaller) candidates are too.
    active = (candidates > thresholds[:, :-1]).long().cumprod(dim=1).bool()
    size = active.sum(dim=1, keepdim=True)
    threshold = thresholds.gather(1, size)
    # Center at q_y before inserting active probabilities. Subtracting log(alpha)
    # from a very negative threshold in float32 could otherwise round it away,
    # silently turning an alpha:1 constraint into a 1:1 ratio.
    other_log_q = torch.where(active, torch.full_like(ordered, -log_alpha), ordered - threshold)
    log_q = torch.zeros_like(log_p).scatter(1, indices, other_log_q)
    return F.softmax(log_q, dim=-1)


def projection_cross_entropy(logits, labels, alpha):
    # The projection is an alternating target, not part of the backward graph.
    return soft_cross_entropy(logits, project_labels(logits, labels, alpha))


def normalized_perturbation(parameters, gradients, rho):
    """One global L2 ball, with a differentiable finite zero-gradient limit."""
    present = [g for g in gradients if g is not None]
    if not present:
        return [torch.zeros_like(p) for p in parameters]
    norm = torch.stack([g.norm(p=2) for g in present]).norm(p=2)
    scale = rho / norm.clamp_min(1e-12)
    return [torch.zeros_like(p) if g is None else g * scale
            for p, g in zip(parameters, gradients)]


def paired_losses(model, x, targets, rho, *, create_graph=False,
                  detach_parameters=False, update_buffers=False):
    """Return CE(theta) and CE(theta+epsilon) using the same stochastic draw.

    Only the base forward may commit buffer updates (for the model's SAM step).
    Probes use detached parameter leaves and never accumulate live model grads.
    The perturbed forward replays CPU/CUDA RNG and then restores its caller's
    state, so each pair advances randomness as a single unperturbed forward.
    Module training flags are never changed. This also respects frozen BN.
    """
    parameters = {name: (p.detach().requires_grad_(p.requires_grad)
                          if detach_parameters else p)
                  for name, p in model.named_parameters()}
    trainable = {name: p for name, p in parameters.items() if p.requires_grad}
    if not trainable:
        raise ValueError("SFT needs trainable classifier-model parameters")
    initial_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    base_buffers = {name: b.clone() for name, b in initial_buffers.items()}
    cpu_rng = torch.get_rng_state()
    cuda_devices = [x.device.index] if x.is_cuda else []
    cuda_rng = torch.cuda.get_rng_state(x.device) if x.is_cuda else None

    logits = functional_call(model, {**parameters, **base_buffers}, (x,))
    base = soft_cross_entropy(logits, targets)
    if rho == 0:
        perturbed = base
    else:
        gradients = torch.autograd.grad(
            base, tuple(trainable.values()), create_graph=create_graph,
            retain_graph=create_graph, allow_unused=True)
        delta = normalized_perturbation(tuple(trainable.values()), gradients, rho)
        perturbed_parameters = dict(parameters)
        for (name, p), e in zip(trainable.items(), delta):
            perturbed_parameters[name] = p + (e if create_graph else e.detach())
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, x.device)
            perturbed_logits = functional_call(
                model, {**perturbed_parameters, **initial_buffers}, (x,))
        perturbed = soft_cross_entropy(perturbed_logits, targets)

    if update_buffers:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(base_buffers[name])
    return base, perturbed


def sharpness(model, x, targets, rho, *, create_graph=True):
    """Eq. (9) at a fixed theta, with gradients through epsilon and targets."""
    if rho == 0:
        return targets.sum() * 0.0
    base, perturbed = paired_losses(
        model, x, targets, rho, create_graph=create_graph, detach_parameters=True)
    value = perturbed - base
    return value if create_graph else value.detach()


class SFTInferenceModel(nn.Module):
    """Optimizer/refiner-free view selected only by SFT's optional SWAD hook.

    The view borrows the live network. AveragedModel is responsible for copying
    it; constructing the view must not copy weights or change training flags.
    """
    def __init__(self, network, training=True):
        super().__init__()
        self.network = network
        self.training = training

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)
