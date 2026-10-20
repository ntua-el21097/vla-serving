"""Action expert on bmm, so one batch can carry per-sample fine-tuned weights.

A deployed fleet shares the frozen VLM but fine-tunes the action expert per task.
A batch therefore holds an arbitrary multiset of task ids -- all the same task, all
different, or mixed -- known only when the requests arrive.

Every weight-bearing module in both experts is an nn.Linear (pi0's PiGemmaModel
subclasses HF GemmaModel; smolvla's lm_expert is a stock AutoModel), so swapping
just those for BmmLinear leaves attention, rope, masking and the prefix KV cache
bit-identical to lerobot.

BmmLinear is mathematically identical to nn.Linear -- same weights, same math, a
different layout for the batch dimension. It is a serving-time strategy, not an
architecture: fine-tuning still happens offline with stock nn.Linear, and stacking
K checkpoints into [K, out, in] is the whole conversion. lerobot's source is never
modified; this operates on the built policy.

The K experts live in ONE [K, out, in] parameter rather than K modules, so expert
choice is an offset into a single allocation computed from tensor data. One
compiled graph then serves every id combination, and nothing recaptures.
"""

import torch
import torch.nn as nn

# The module denoise_step actually runs. pi0's lm_head sits outside .model and is
# never reached during sampling, so it is not replicated.
EXPERT_PATHS = (
  "model.paligemma_with_expert.gemma_expert.model",   # pi0
  "model.vlm_with_expert.lm_expert",                  # smolvla
)


class ExpertRouter(nn.Module):
  """The batch's expert ids, in a buffer whose address never changes.

  set() writes in place, so a compiled graph -- and a captured cudagraph -- reads
  new values through the same pointer. Sized to the largest batch the server runs;
  forward() slices to its own B, which covers batch_max and crit_batch_max."""

  def __init__(self, max_batch, num_experts, device=None):
    super().__init__()
    self.num_experts = num_experts
    self.max_batch = max_batch
    self.register_buffer(
      "expert_ids", torch.zeros(max_batch, dtype=torch.long, device=device), persistent=False
    )
    # Staged on the host so set() is ONE host-blocking H2D. Writing the device
    # buffer piecewise instead issues device ops on the CURRENT stream, which the
    # forward -- running on its own stream -- is not ordered against: the pass
    # could read stale ids and silently use the wrong experts. Plain attribute,
    # so .to(device) leaves it on the host.
    self._staging = torch.zeros(max_batch, dtype=torch.long)

  def set(self, ids):
    """ids: one expert id per REAL request; padded rows inherit ids[0]."""
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1).cpu()
    n = ids.numel()
    if n == 0 or n > self.max_batch:
      raise ValueError(f"got {n} ids, need 1..{self.max_batch}")
    if int(ids.max()) >= self.num_experts or int(ids.min()) < 0:
      raise ValueError(f"expert ids out of range [0, {self.num_experts})")
    self._staging[:n] = ids
    # Upstream padding repeats observation 0, so padded rows take its expert.
    self._staging[n:] = ids[0]
    self.expert_ids.copy_(self._staging)


class BmmLinear(nn.Module):
  """nn.Linear whose weights are selected per sample by the router."""

  def __init__(self, base, num_experts, router, noise_std=0.02):
    super().__init__()
    out_f, in_f = base.weight.shape
    w = base.weight.detach().unsqueeze(0).repeat(num_experts, 1, 1)
    if noise_std:
      # Per-expert noise so a routing bug gives a visibly wrong result instead of
      # silently matching. Added one expert at a time: randn_like over the whole
      # stack would double peak memory during construction.
      scale = noise_std * base.weight.detach().float().std()
      for k in range(num_experts):
        w[k].add_(torch.randn_like(w[k]) * scale)
    self.weight = nn.Parameter(w, requires_grad=False)
    if base.bias is not None:
      self.bias = nn.Parameter(base.bias.detach().unsqueeze(0).repeat(num_experts, 1),
                               requires_grad=False)
    else:
      self.bias = None
    self.in_features, self.out_features, self.num_experts = in_f, out_f, num_experts
    # Plain list: keeps the shared router out of this module's parameter tree.
    self._router = [router]

  def forward(self, x):
    if x.dim() != 3:
      raise RuntimeError(f"BmmLinear expects [B, S, in], got {tuple(x.shape)}")
    ids = self._router[0].expert_ids[:x.shape[0]]
    w = self.weight[ids]                      # [B, out, in]
    y = torch.bmm(x, w.transpose(1, 2))       # [B, S, in] @ [B, in, out]
    if self.bias is not None:
      y = y + self.bias[ids].unsqueeze(1)
    return y

  def extra_repr(self):
    return f"in={self.in_features}, out={self.out_features}, experts={self.num_experts}"


def resolve_expert(policy):
  """Return (module, path) for the action expert that denoise_step runs."""
  for path in EXPERT_PATHS:
    obj = policy
    try:
      for part in path.split("."):
        obj = getattr(obj, part)
    except AttributeError:
      continue
    return obj, path
  raise AttributeError(f"no action expert at any of {EXPERT_PATHS}")


def make_hetero(policy, num_experts, max_batch, noise_std=0.02, verbose=True):
  """Swap the action expert's Linears for BmmLinear. Returns the ExpertRouter.

  RMSNorm weights stay shared: 38K parameters against the expert's 311M, so
  replicating them changes no timing.
  """
  expert, path = resolve_expert(policy)
  device = next(expert.parameters()).device
  router = ExpertRouter(max_batch, num_experts, device=device)

  replaced = base_params = 0
  for module in expert.modules():
    for name, child in list(module.named_children()):
      if isinstance(child, nn.Linear):
        setattr(module, name, BmmLinear(child, num_experts, router, noise_std))
        replaced += 1
        base_params += child.weight.numel() + (0 if child.bias is None else child.bias.numel())

  if verbose:
    dtype = next(expert.parameters()).dtype
    added = base_params * (num_experts - 1) * torch.empty(0, dtype=dtype).element_size()
    print(f"[hetero] {path}: {replaced} Linear -> BmmLinear | K={num_experts} "
          f"max_batch={max_batch} | expert {base_params / 1e6:.0f}M params, "
          f"+{added / 2**30:.2f} GiB {dtype}")
  return router

