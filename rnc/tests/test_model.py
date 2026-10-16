"""Model correctness on the GPU: parameter count, shapes, causality, attention parity, gradients, init loss."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from rnc.config import ModelConfig
from rnc.model import RNCModel, analytic_parameter_count

ROOT = Path(__file__).resolve().parent.parent


def small_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=512, d_model=64, n_layers=2, n_heads=4, d_mlp=128, context_length=64)
    base.update(kw)
    return ModelConfig(**base)


def test_parameter_count_real_config_matches_analytic(real_cfg):
    m = RNCModel(real_cfg.model)
    n = m.num_parameters()
    assert n == analytic_parameter_count(real_cfg.model) == 91_246_848
    # independent hand calculation of the documented design
    d, L, V, M = 768, 12, 8192, 2048
    assert n == V * d + L * (4 * d * d + 3 * d * M + 2 * d + 2 * 64) + d


def test_parameter_count_untied_and_no_qknorm():
    for kw in (dict(tie_embeddings=False), dict(qk_norm=False), dict(tie_embeddings=False, qk_norm=False)):
        cfg = small_cfg(**kw)
        assert RNCModel(cfg).num_parameters() == analytic_parameter_count(cfg)


def test_tied_readout_uses_embedding(device):
    m = RNCModel(small_cfg()).to(device)
    assert m.lm_head is None
    x = torch.randint(0, 512, (1, 8), device=device)
    a, _ = m(x)
    with torch.no_grad():
        m.embed.weight[5] += 1.0
    b, _ = m(x)
    assert not torch.allclose(a, b)


def test_shapes_and_finite(device):
    m = RNCModel(small_cfg()).to(device)
    x = torch.randint(0, 512, (3, 64), device=device)
    y = torch.randint(0, 512, (3, 64), device=device)
    logits, loss = m(x, y)
    assert logits.shape == (3, 64, 512)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} received a zero gradient"


def test_context_length_enforced(device):
    m = RNCModel(small_cfg()).to(device)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 65, dtype=torch.long, device=device))


def test_initial_loss_near_uniform(device, real_cfg):
    """Shifted random targets -> loss ~ ln(V): the init neither cheats nor blows up."""
    torch.manual_seed(0)
    m = RNCModel(real_cfg.model).to(device)
    x = torch.randint(0, real_cfg.model.vocab_size, (4, 129), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(x[:, :-1], x[:, 1:])
    assert abs(float(loss.detach()) - math.log(real_cfg.model.vocab_size)) < 0.5, float(loss.detach())


@pytest.mark.parametrize("backend", ["reference", "math", "auto"])
def test_causality_perturbation(device, backend):
    torch.manual_seed(1)
    m = RNCModel(small_cfg()).to(device).eval()
    m.set_attention_backend(backend)
    x = torch.randint(0, 512, (2, 64), device=device)
    base, _ = m(x)
    for j in (1, 17, 40, 63):
        x2 = x.clone()
        x2[:, j] = (x2[:, j] + 7) % 512
        out, _ = m(x2)
        tol = 0.0 if backend == "reference" else 1e-4
        assert torch.allclose(out[:, :j], base[:, :j], atol=tol, rtol=0), f"future token leaked to the past ({backend}, j={j})"
        assert not torch.allclose(out[:, j], base[:, j]), "the changed position itself must change"


def test_causality_by_gradient(device):
    """d loss_at_i / d embedding_at_j must be exactly zero for every j > i (reference attention)."""
    m = RNCModel(small_cfg()).to(device)
    m.set_attention_backend("reference")
    x = torch.randint(0, 512, (1, 16), device=device)
    emb = m.embed(x).detach().requires_grad_(True)
    # run the model body manually on the embedding to measure input gradients
    from rnc.model import rope_cos_sin

    cos, sin = rope_cos_sin(torch.arange(16, device=device), m.cfg.head_dim, m.cfg.rope_base, device)
    h = emb
    for b in m.blocks:
        h = b(h, cos, sin)
    logits = torch.nn.functional.linear(m.norm_f(h), m.embed.weight)
    i = 6
    logits[0, i].sum().backward()
    g = emb.grad[0].abs().sum(-1)
    assert torch.all(g[i + 1:] == 0), g
    assert torch.all(g[: i + 1] > 0)


def test_attention_backends_agree(device):
    torch.manual_seed(2)
    m = RNCModel(small_cfg()).to(device).eval()
    x = torch.randint(0, 512, (2, 64), device=device)
    outs = {}
    for be in ("reference", "math", "auto"):
        m.set_attention_backend(be)
        outs[be], _ = m(x)
    assert torch.allclose(outs["math"], outs["reference"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(outs["auto"], outs["reference"], atol=1e-4, rtol=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m.set_attention_backend("auto")
        bf, _ = m(x)
    rel = (bf.float() - outs["reference"]).abs().max() / outs["reference"].abs().max()
    assert rel < 0.05, float(rel)


def test_parameter_groups_partition(device):
    m = RNCModel(small_cfg())
    groups = m.parameter_groups(0.1)
    all_params = {id(p) for p in m.parameters()}
    seen = [id(p) for g in groups for p in g["params"]]
    assert sorted(seen) == sorted(all_params)
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert id(m.embed.weight) in no_decay_ids
    for name, p in m.named_parameters():
        if p.dim() == 1:
            assert id(p) in no_decay_ids, name


def test_generate_shapes_and_eos(device):
    torch.manual_seed(3)
    m = RNCModel(small_cfg()).to(device)
    x = torch.randint(8, 512, (1, 5), device=device)
    g = torch.Generator(device=device).manual_seed(0)
    out = m.generate(x, 20, temperature=1.0, top_k=10, generator=g)
    assert out.shape[1] <= 25 and out.shape[1] > 5
    assert torch.equal(out[:, :5], x)
    out2 = m.generate(x, 100, temperature=1.0, top_k=0)  # must crop to context_length internally without error
    assert out2.shape[1] == 105
    greedy1 = m.generate(x, 10, temperature=0)
    greedy2 = m.generate(x, 10, temperature=0)
    assert torch.equal(greedy1, greedy2)


def test_no_hard_coded_answers_in_model_or_generation():
    """No English string literals in the compute path: nothing can be injected into outputs."""
    for fname in ("model.py", "generate.py"):
        tree = ast.parse((ROOT / "rnc" / fname).read_text(encoding="utf-8"))
        literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
        suspicious = [s for s in literals if " " in s and s not in docstrings and not s.startswith(("[", "{", "RNC", "you", "rnc", "---", "checkpoint", "tokenizer", "sequence", "CONFIG"))]
        suspicious = [s for s in suspicious if not any(k in s for k in ("must", "not", "differs", "exceeds", "found", "parameters", "Empty line"))]
        assert suspicious == [], suspicious
