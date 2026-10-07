"""Gate/hook arithmetic, token alignment, patching and the CCHG trainer on a tiny random model (CPU only)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from llm_distract.mechanism import chg  # noqa: E402

VOCAB, DIM, HEADS, LAYERS = 40, 16, 4, 2


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, DIM)
        self.o_proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.qkv(x)))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList([_Layer() for _ in range(LAYERS)])


class _Out:
    def __init__(self, logits):
        self.logits = logits


class TinyLM(nn.Module):
    """Minimal causal-LM stand-in exposing model.model.layers[i].self_attn.o_proj and config.num_attention_heads."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = _Inner()
        self.lm_head = nn.Linear(DIM, VOCAB)
        self.config = type("Cfg", (), {"num_attention_heads": HEADS, "num_hidden_layers": LAYERS})()

    def forward(self, ids):
        x = self.model.embed(ids)
        for layer in self.model.layers:
            x = x + layer.self_attn(x)
        return _Out(self.lm_head(x))


def test_gate_hook_scales_each_head():
    model = TinyLM()
    gates = torch.rand(LAYERS, HEADS)
    x = torch.randn(2, 5, DIM)
    layer0 = model.model.layers[0].self_attn
    with chg.HeadGates(model).apply(gates):
        gated = layer0(x)
    pre = torch.tanh(layer0.qkv(x)).view(2, 5, HEADS, DIM // HEADS) * gates[0].view(1, 1, HEADS, 1)
    expected = layer0.o_proj(pre.reshape(2, 5, DIM))
    assert torch.allclose(gated, expected, atol=1e-6)
    assert torch.allclose(layer0(x), layer0.o_proj(torch.tanh(layer0.qkv(x))))  # hooks removed afterwards


def test_scaled_gates_and_random_control():
    selected = torch.zeros(LAYERS, HEADS, dtype=torch.bool)
    selected[0, 1] = selected[1, 3] = True
    g = chg.scaled_gates(selected, 0.25)
    assert g[0, 1] == 0.25 and g[1, 3] == 0.25 and g.sum() == 0.25 * 2 + (LAYERS * HEADS - 2)
    rnd = chg.random_head_set(selected, seed=0)
    assert rnd.sum() == 2 and not (rnd & selected).any()
    assert torch.equal(rnd, chg.random_head_set(selected, seed=0))
    assert chg.select_heads(torch.tensor([[0.2, 0.5], [0.49, 0.9]])).tolist() == [[True, False], [True, False]]


def test_align_tokens_prefix_suffix_and_padding():
    clean = torch.tensor([[1, 2, 3, 7, 8, 0, 0]])
    dist = torch.tensor([[1, 2, 3, 4, 5, 7, 8]])
    a = chg.align_tokens(clean, dist, clean_lens=torch.tensor([5]), distract_lens=torch.tensor([7]))
    assert a.tolist() == [[0, 1, 2, -1, -1, 3, 4]]
    a2 = chg.align_tokens(dist, clean, clean_lens=torch.tensor([7]), distract_lens=torch.tensor([5]))
    assert a2.tolist() == [[0, 1, 2, 5, 6, -1, -1]]  # padding never aligned
    same = torch.tensor([[4, 4, 4]])
    assert chg.align_tokens(same, same).tolist() == [[0, 1, 2]]


def test_patching_replaces_selected_heads_at_aligned_positions():
    model = TinyLM()
    clean = torch.tensor([[1, 2, 3, 7, 8, 9]])
    dist = torch.tensor([[1, 2, 3, 4, 5, 7, 8, 9]])
    head_mask = torch.zeros(LAYERS, HEADS, dtype=torch.bool)
    head_mask[0, 2] = True
    align = chg.align_tokens(clean, dist)
    captured = {}
    patcher = chg.ActivationPatcher(model)
    with torch.no_grad(), patcher.capture():
        model(clean)
        clean_pre = patcher.cache[0].clone()
    hook = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(lambda m, i: captured.setdefault("x", i[0].clone()))
    with torch.no_grad(), patcher.patch(head_mask, align):
        model(dist)
    hook.remove()
    x = captured["x"].view(1, 8, HEADS, DIM // HEADS)
    c = clean_pre.view(1, 6, HEADS, DIM // HEADS)
    assert torch.allclose(x[0, [0, 1, 2], 2], c[0, [0, 1, 2], 2])  # prefix aligned
    assert torch.allclose(x[0, [5, 6, 7], 2], c[0, [3, 4, 5], 2])  # suffix aligned
    with torch.no_grad():
        raw = torch.tanh(model.model.layers[0].self_attn.qkv(model.model.embed(dist))).view(1, 8, HEADS, DIM // HEADS)
    assert torch.allclose(x[0, [3, 4], 2], raw[0, [3, 4], 2])  # inserted tokens untouched
    assert torch.allclose(x[0, :, 0], raw[0, :, 0])  # non-selected heads untouched


def test_contrastive_trainer_logs_every_update():
    model = TinyLM()
    n = 6
    ids = torch.randint(1, VOCAB, (n, 7))
    masks = torch.zeros(n, 7, dtype=torch.bool)
    masks[:, -1] = True
    tensors = {"pos_ids": ids, "pos_masks": masks, "neg_ids": ids.flip(1), "neg_masks": masks,
               "item_id": torch.arange(n)}
    trainer = chg.ContrastiveMaskTrainer(model, tensors, pad_token_id=0, batch_size=2, grad_accum=2, lr=0.05,
                                         lr_end=0.005, l1_weight=1.0, seed=0)
    gates, log = trainer.fit(num_updates=3, show_pbar=False)
    assert gates.shape == (LAYERS, HEADS) and ((gates > 0) & (gates < 1)).all()
    assert len(log) == 3 and log[["gap_loss", "l1", "loss", "logp_clean"]].notna().all().all()
    assert log["skipped_microbatches"].sum() == 0
    assert abs(log["lr"].iloc[-1] - 0.005) < 1e-9  # LinearLR reached lr_end after num_updates
