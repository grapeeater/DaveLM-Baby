from __future__ import annotations

import torch

from src.baby_v010.residual_overwrite import IdentityResidualOverwrite, LocalSlotOverwrite, pointer_aux


def test_gate_zero_is_identity() -> None:
    module = IdentityResidualOverwrite(8, gate_bias=-80.0)
    hidden = torch.randn(2, 5, 8)
    out = module(hidden)
    assert torch.allclose(out, hidden, atol=1e-5)


def test_identity_read_replace_algebra() -> None:
    hidden = torch.randn(1, 5, 3)
    attn = torch.zeros(1, 5, 5)
    for t in range(5):
        attn[0, t, t] = 1.0
    attn[0, 4] = 0
    attn[0, 4, 1] = 1.0
    gate = torch.ones(1, 5, 1)
    read = torch.matmul(attn, hidden)
    out = hidden + gate * (read - hidden)
    assert torch.allclose(out[0, 4], hidden[0, 1])
    assert torch.allclose(out[0, 2], hidden[0, 2])


def test_pointer_aux_peaked_and_gate_one() -> None:
    attn = torch.zeros(1, 4, 4)
    attn[0, 3, 1] = 1.0
    gate = torch.ones(1, 4)
    ptr, g = pointer_aux(attn, gate, [(0, 3, 1)], use_gate=True)
    assert float(ptr) < 1e-6
    assert float(g) == 0.0


def test_zero_qk_pointer_grad_vanishes() -> None:
    module = IdentityResidualOverwrite(8, gate_bias=-4.0, qk_init="zero")
    hidden = torch.randn(1, 6, 8)
    module(hidden)
    ptr = -torch.log(module.last_attn[0, 5, 1] + 1e-8)
    ptr.backward()
    assert float(module.q.weight.grad.abs().sum()) == 0.0
    assert float(module.k.weight.grad.abs().sum()) == 0.0


def test_xavier_qk_pointer_grad_flows() -> None:
    torch.manual_seed(0)
    module = IdentityResidualOverwrite(8, gate_bias=-4.0, qk_init="xavier")
    hidden = torch.randn(1, 6, 8)
    module(hidden)
    ptr = -torch.log(module.last_attn[0, 5, 1] + 1e-8)
    ptr.backward()
    assert float(module.q.weight.grad.abs().sum()) > 0.0
    assert float(module.k.weight.grad.abs().sum()) > 0.0


def test_local_slot_score_grad_flows() -> None:
    torch.manual_seed(0)
    module = LocalSlotOverwrite(8, gate_bias=-4.0)
    hidden = torch.randn(1, 6, 8)
    module(hidden)
    ptr = -torch.log(module.last_attn[0, 5, 1] + 1e-8)
    ptr.backward()
    assert float(module.scorer.weight.grad.abs().sum()) > 0.0


def test_gen_only_masks_non_gen_and_consumes_index() -> None:
    torch.manual_seed(0)
    module = LocalSlotOverwrite(8, gate_bias=8.0, gen_only=True)
    hidden = torch.randn(2, 5, 8)
    module.gen_index = torch.tensor([4, 3])
    out = module(hidden)
    assert module.gen_index is None
    assert torch.allclose(out[0, :4], hidden[0, :4], atol=1e-5)
    assert torch.allclose(out[1, :3], hidden[1, :3], atol=1e-5)
    assert torch.allclose(out[1, 4], hidden[1, 4], atol=1e-5)
    assert not torch.allclose(out[0, 4], hidden[0, 4], atol=1e-4)
    assert not torch.allclose(out[1, 3], hidden[1, 3], atol=1e-4)


def test_gen_only_without_index_is_identity() -> None:
    module = LocalSlotOverwrite(8, gate_bias=8.0, gen_only=True)
    hidden = torch.randn(2, 5, 8)
    out = module(hidden)
    assert torch.allclose(out, hidden, atol=1e-5)
