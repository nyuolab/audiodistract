"""Contrastive causal head gating (CCHG), activation patching, head scaling and distractor attention mass.

Self-contained re-implementation of the parts of Nam et al.'s causal-head-gating library that the paper used,
plus the paper's contrastive objective. All interventions act on per-head attention outputs immediately before
``o_proj`` (``model.model.layers[i].self_attn.o_proj``), which Llama-3, Qwen2.5, Mistral and Gemma-2 expose as a
[batch, seq, num_heads * head_dim] tensor (head_dim is derived from that width, so Gemma-2's 256-dim heads work).

Differences from the original run code, kept deliberately:
* the training log records the loss terms at every update (the original reset its metric dict before logging,
  leaving NaN in all but the first row);
* the size-matched random patching control is drawn with a seeded generator, excludes the selected heads and is
  recorded (the original used the unseeded global RNG and could overlap the selected set);
* token alignment for patching is computed on the unpadded sequences (the original aligned right-padded batches,
  which left the answer suffix of the shorter item in each batch of two unaligned);
* the attention-mass query position is the final prompt token, i.e. the position that predicts the answer letter
  (the original used the token preceding "Answer:"), and the inserted span is located with tokenizer offsets.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------- head hooks
def head_modules(model) -> list:
    """The o_proj modules whose input is the concatenation of per-head attention outputs."""
    return [layer.self_attn.o_proj for layer in model.model.layers]


def head_grid(model) -> tuple:
    return len(model.model.layers), int(model.config.num_attention_heads)


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, s, d = x.shape
    return x.view(b, s, num_heads, d // num_heads)


class HeadGates:
    """Multiplicative per-(layer, head) gates applied to head outputs; gates may require grad."""

    def __init__(self, model):
        self.model = model
        self.num_layers, self.num_heads = head_grid(model)

    @contextmanager
    def apply(self, gates: torch.Tensor):
        """Scale head outputs by ``gates`` ([num_layers, num_heads]) for every forward pass inside the context."""
        if tuple(gates.shape) != (self.num_layers, self.num_heads):
            raise ValueError(f"gates must have shape {(self.num_layers, self.num_heads)}")
        handles = []

        def make(idx):
            def hook(_module, inputs):
                x = inputs[0]
                g = gates[idx].to(device=x.device, dtype=x.dtype).view(1, 1, self.num_heads, 1)
                return ((_split_heads(x, self.num_heads) * g).reshape(x.shape),)
            return hook

        for idx, mod in enumerate(head_modules(self.model)):
            handles.append(mod.register_forward_pre_hook(make(idx)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


def target_logprob(model, ids: torch.Tensor, masks: torch.Tensor, agg: str = "sum") -> torch.Tensor:
    """Log-probability of the target tokens (masks True) given their prefix; 'sum' or per-token 'mean'."""
    logits = model(ids[:, :-1]).logits
    m = masks[:, 1:].to(torch.float32)
    nll = F.cross_entropy(logits.float().transpose(1, 2), ids[:, 1:], reduction="none")
    logp = -(nll * m).sum(-1)
    return logp / m.sum(-1) if agg == "mean" else logp


def select_heads(gates: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Interference heads = gates below the threshold (bool [num_layers, num_heads])."""
    return gates < threshold


def scaled_gates(selected: torch.Tensor, scale: float) -> torch.Tensor:
    """Gate tensor that multiplies the selected heads by ``scale`` (0 = hard suppression) and leaves others at 1."""
    out = torch.ones(selected.shape, dtype=torch.float32)
    out[selected] = float(scale)
    return out


# ----------------------------------------------------------------------------- CCHG trainer
class ContrastiveMaskTrainer:
    """Learn sigmoid gates that close the clean-vs-distracted log-prob gap while staying near 1.

    Objective per update (batch of pairs i, heads h):
        loss = mean_i (logp_clean_i - logp_distracted_i)^2 + lambda * mean_h (1 - g_h),  g = sigmoid(logits),
    with logits zero-initialised (g = 0.5 at start, equal to the selection threshold), logp = per-token mean log-prob
    of the single-letter target, Adam with LinearLR from lr to lr_end over ``num_updates`` updates, ``grad_accum``
    micro-batches of ``batch_size`` pairs per update, gradient-norm clipping and logit clamping to +/-logit_clamp.
    Non-finite micro-batch losses are skipped (counted in the log). Selection = g < threshold.
    """

    def __init__(self, model, tensors: dict, pad_token_id: int, batch_size: int = 4, grad_accum: int = 4,
                 lr: float = 1e-2, lr_end: float = 1e-3, l1_weight: float = 5.0, grad_clip: float = 1.0,
                 logit_clamp: float = 6.0, seed: int = 0, autocast_dtype: torch.dtype = torch.float16):
        self.model = model
        self.gates = HeadGates(model)
        self.device = next(model.parameters()).device
        self.tensors = {k: v.to(self.device) for k, v in tensors.items()}
        self.pad_token_id = pad_token_id
        self.batch_size, self.grad_accum = batch_size, grad_accum
        self.lr, self.lr_end, self.l1_weight = lr, lr_end, l1_weight
        self.grad_clip, self.logit_clamp = grad_clip, logit_clamp
        self.autocast_dtype = autocast_dtype
        self.generator = torch.Generator().manual_seed(seed)
        self.logits = nn.Parameter(torch.zeros(self.gates.num_layers, self.gates.num_heads, device=self.device))
        for p in model.parameters():
            p.requires_grad_(False)

    def _batches(self) -> Iterator[dict]:
        n = self.tensors["item_id"].shape[0]
        while True:
            order = torch.randperm(n, generator=self.generator)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size].to(self.device)
                batch = {k: v[idx] for k, v in self.tensors.items()}
                for side in ("pos", "neg"):  # trim right padding independently per side (the target is the last real token;
                    # counting non-pad tokens would fail when the pad id also occurs inside the prompt, e.g. <|eot_id|> in chat templates)
                    length = int(batch[f"{side}_masks"].any(0).nonzero().max().item()) + 1
                    batch[f"{side}_ids"] = batch[f"{side}_ids"][:, :length]
                    batch[f"{side}_masks"] = batch[f"{side}_masks"][:, :length]
                yield batch

    def _loss(self, batch: dict):
        g = torch.sigmoid(self.logits)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
            with self.gates.apply(g):
                logp_pos = target_logprob(self.model, batch["pos_ids"], batch["pos_masks"], "mean").float()
                logp_neg = target_logprob(self.model, batch["neg_ids"], batch["neg_masks"], "mean").float()
        gap = logp_pos - logp_neg
        gap_loss = gap.pow(2).mean()
        l1 = self.l1_weight * (1.0 - g).mean()
        loss = (gap_loss + l1) / self.grad_accum
        return loss, {"logp_clean": logp_pos.mean().item(), "logp_distracted": logp_neg.mean().item(),
                      "gap": gap.mean().item(), "gap_loss": gap_loss.item(), "l1": l1.item()}

    def fit(self, num_updates: int = 500, show_pbar: bool = True) -> tuple:
        """Run ``num_updates`` optimiser updates; returns (gates [L, H] float32 cpu, log DataFrame, one row/update)."""
        optimizer = torch.optim.Adam([self.logits], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=self.lr_end / self.lr,
                                                      total_iters=num_updates)
        batches = self._batches()
        rows = []
        try:
            from tqdm.auto import trange
            updates = trange(num_updates, disable=not show_pbar)
        except ImportError:  # pragma: no cover
            updates = range(num_updates)
        for update in updates:
            optimizer.zero_grad()
            acc, skipped = {}, 0
            for _ in range(self.grad_accum):
                loss, metrics = self._loss(next(batches))
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                for k, v in metrics.items():
                    acc[k] = acc.get(k, 0.0) + v / self.grad_accum
            if self.logits.grad is not None:
                self.logits.grad.nan_to_num_(0.0)
                torch.nn.utils.clip_grad_norm_([self.logits], self.grad_clip)
            optimizer.step()
            with torch.no_grad():
                self.logits.clamp_(-self.logit_clamp, self.logit_clamp)
            scheduler.step()
            g = torch.sigmoid(self.logits.detach()).cpu()
            rows.append({"update": update + 1, "lr": optimizer.param_groups[0]["lr"], **acc,
                         "loss": acc.get("gap_loss", math.nan) + acc.get("l1", math.nan), "skipped_microbatches": skipped,
                         "gate_min": g.min().item(), "gate_mean": g.mean().item(), "gate_max": g.max().item(),
                         "n_selected": int((g < 0.5).sum().item())})
        return torch.sigmoid(self.logits.detach()).float().cpu(), pd.DataFrame.from_records(rows)


# ----------------------------------------------------------------------------- activation patching
def align_tokens(clean_ids: torch.Tensor, distract_ids: torch.Tensor, clean_lens: Optional[torch.Tensor] = None,
                 distract_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Map each distracted position to its clean counterpart by longest common prefix + longest common suffix.

    Returns LongTensor [B, L_distract] with the clean index, or -1 for positions without a counterpart (the inserted
    distractor tokens and padding). Lengths default to the full sequence length (pass real lengths for padded batches).
    """
    bsz, d_len = distract_ids.shape
    align = torch.full((bsz, d_len), -1, dtype=torch.long)
    for b in range(bsz):
        c = clean_ids[b, : int(clean_lens[b]) if clean_lens is not None else None].tolist()
        d = distract_ids[b, : int(distract_lens[b]) if distract_lens is not None else None].tolist()
        n = min(len(c), len(d))
        prefix = 0
        while prefix < n and c[prefix] == d[prefix]:
            prefix += 1
        suffix = 0
        while suffix < n - prefix and c[-1 - suffix] == d[-1 - suffix]:
            suffix += 1
        align[b, :prefix] = torch.arange(prefix)
        for i in range(1, suffix + 1):
            align[b, len(d) - i] = len(c) - i
    return align


class ActivationPatcher:
    """Cache clean head outputs, then overwrite selected heads at aligned positions of a distracted forward pass."""

    def __init__(self, model):
        self.model = model
        self.num_layers, self.num_heads = head_grid(model)
        self.cache = {}

    @contextmanager
    def capture(self):
        handles = []

        def make(idx):
            def hook(_module, inputs):
                self.cache[idx] = inputs[0].detach()
            return hook

        for idx, mod in enumerate(head_modules(self.model)):
            handles.append(mod.register_forward_pre_hook(make(idx)))
        try:
            yield self.cache
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def patch(self, head_mask: torch.Tensor, align: torch.Tensor):
        """head_mask: bool [L, H] heads to patch; align: [B, L_distract] clean position per distracted position."""
        handles = []

        def make(idx):
            def hook(_module, inputs):
                x = inputs[0]
                cached = self.cache[idx].to(x.device)
                a = align[:, : x.shape[1]].to(x.device)
                valid = ((a >= 0) & (a < cached.shape[1])).unsqueeze(-1).unsqueeze(-1)  # unaligned or beyond the cached clean pass
                gathered = torch.gather(cached, 1, a.clamp(min=0, max=cached.shape[1] - 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                hm = head_mask[idx].to(x.device).view(1, 1, self.num_heads, 1)
                out = torch.where(hm & valid, _split_heads(gathered, self.num_heads), _split_heads(x, self.num_heads))
                return (out.reshape(x.shape),)
            return hook

        for idx, mod in enumerate(head_modules(self.model)):
            handles.append(mod.register_forward_pre_hook(make(idx)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()
            self.cache = {}


@torch.no_grad()
def patched_target_logprob(model, clean_ids, distract_ids, distract_masks, head_mask, pad_token_id) -> torch.Tensor:
    """Target log-prob on the distracted prompt with the selected heads' outputs replaced by clean-prompt outputs."""
    device = next(model.parameters()).device
    align = align_tokens(clean_ids, distract_ids, (clean_ids != pad_token_id).sum(-1), (distract_ids != pad_token_id).sum(-1))
    patcher = ActivationPatcher(model)
    with patcher.capture():
        model(clean_ids.to(device))  # full clean sequence so every aligned position (including the final one) has a cached state
    with patcher.patch(head_mask, align):
        return target_logprob(model, distract_ids.to(device), distract_masks.to(device), "sum").cpu()


def random_head_set(selected: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Size-matched random control: same number of heads as ``selected``, drawn from the non-selected heads."""
    pool = torch.nonzero(~selected.flatten()).flatten()
    gen = torch.Generator().manual_seed(seed)
    pick = pool[torch.randperm(pool.numel(), generator=gen)[: int(selected.sum().item())]]
    out = torch.zeros(selected.numel(), dtype=torch.bool)
    out[pick] = True
    return out.view(selected.shape)


# ----------------------------------------------------------------------------- attention mass
def _char_span_to_tokens(tokenizer, prompt: str, start: int, end: int) -> tuple:
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > start and s < end]
    return (idx[0], idx[-1] + 1) if idx else (0, 0)


@torch.no_grad()
def distractor_attention_mass(model, tokenizer, prompt: str, sentence: str) -> Optional[torch.Tensor]:
    """Attention from the final prompt token to the inserted sentence, per (layer, head); None if not found.

    Requires an eager attention implementation (``attn_implementation="eager"``) so that attention weights exist.
    """
    start = prompt.find(sentence)
    if start < 0:
        return None
    from llm_distract.mechanism.data import bos_prefix
    bos = bos_prefix(tokenizer)
    t0, t1 = _char_span_to_tokens(tokenizer, prompt, start, start + len(sentence))
    t0, t1 = t0 + len(bos), t1 + len(bos)
    ids = tokenizer(prompt.rstrip(), add_special_tokens=False, return_tensors="pt")["input_ids"]
    if bos and ids[0, :1].tolist() != bos:
        ids = torch.cat([torch.tensor([bos], dtype=ids.dtype), ids], dim=1)
    elif bos:
        t0, t1 = t0 - len(bos), t1 - len(bos)  # BOS is already part of the tokenised prompt
    out = model(ids.to(next(model.parameters()).device), output_attentions=True)
    return torch.stack([a[0, :, -1, t0:t1].sum(-1).float().cpu() for a in out.attentions])
