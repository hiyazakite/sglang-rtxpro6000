import ast
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODELS = Path(__file__).resolve().parents[4] / "python/sglang/srt/models"


def _load_mtp_helpers():
    # The model module pulls in the whole runtime; the helpers are pure torch.
    source = (MODELS / "qwen4_exp_mtp.py").read_text()
    wanted = {
        "_E2M1_VALUES",
        "_NVFP4_GROUP_SIZE",
        "_NVFP4_MTP_EXPERT",
        "dequantize_nvfp4",
        "dequantize_nvfp4_mtp_experts",
    }
    namespace = dict(
        torch=torch, re=re, Dict=Dict, Iterable=Iterable, Iterator=Iterator, Tuple=Tuple
    )
    for node in ast.parse(source).body:
        name = (
            node.name
            if isinstance(node, ast.FunctionDef)
            else node.targets[0].id
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None
        )
        if name in wanted:
            exec(ast.get_source_segment(source, node), namespace)
    return namespace


HELPERS = _load_mtp_helpers()
dequantize_nvfp4 = HELPERS["dequantize_nvfp4"]
dequantize_nvfp4_mtp_experts = HELPERS["dequantize_nvfp4_mtp_experts"]

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _expected(codes: torch.Tensor, group_scale, global_scale) -> torch.Tensor:
    lut = torch.tensor(E2M1)
    values = torch.where(codes & 8 != 0, -lut[codes & 7], lut[codes & 7])
    scale = group_scale.float().repeat_interleave(16, dim=1) * global_scale
    return (values * scale).to(torch.bfloat16)


def _random_matrix(rows=4, cols=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, cols), generator=generator, dtype=torch.int32)
    group_scale = (
        torch.rand(rows, cols // 16, generator=generator) + 0.25
    ).to(torch.float8_e4m3fn)
    global_scale = torch.tensor([0.0123])
    return codes, group_scale, global_scale


def test_dequantize_nvfp4_matches_reference():
    codes, group_scale, global_scale = _random_matrix()
    actual = dequantize_nvfp4(_pack(codes), group_scale, global_scale)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, _expected(codes, group_scale, global_scale))


def test_dequantize_nvfp4_nibble_order_is_low_first():
    codes = torch.tensor([[2] + [0] * 15], dtype=torch.int32)  # 2 -> 1.0
    packed = _pack(codes)
    assert packed[0, 0].item() == 2
    out = dequantize_nvfp4(
        packed,
        torch.ones(1, 1, dtype=torch.float8_e4m3fn),
        torch.ones(1),
    )
    assert out[0, 0].item() == 1.0 and out[0, 1].item() == 0.0


def _expert_tensors(prefix, seed):
    codes, group_scale, global_scale = _random_matrix(seed=seed)
    return codes, {
        f"{prefix}.weight": _pack(codes),
        f"{prefix}.weight_scale": group_scale,
        f"{prefix}.weight_scale_2": global_scale,
    }


def test_mtp_experts_are_dequantized_in_any_arrival_order():
    ref = {}
    stream = []
    for i, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
        prefix = f"mtp.layers.0.mlp.experts.3.{proj}"
        codes, tensors = _expert_tensors(prefix, i)
        ref[prefix] = _expected(
            codes, tensors[f"{prefix}.weight_scale"], tensors[f"{prefix}.weight_scale_2"]
        )
        stream.extend(tensors.items())
    stream = stream[::-1]
    stream.insert(2, ("mtp.layers.0.mlp.gate.weight", torch.ones(2, 2)))
    out = dict(dequantize_nvfp4_mtp_experts(stream))
    assert set(out) == {f"{p}.weight" for p in ref} | {"mtp.layers.0.mlp.gate.weight"}
    for prefix, expected in ref.items():
        assert torch.equal(out[f"{prefix}.weight"], expected)


def test_mtp_bf16_experts_pass_through():
    weight = torch.zeros(2, 2, dtype=torch.bfloat16)
    stream = [
        ("mtp.layers.0.mlp.experts.gate_up_proj", weight),
        ("mtp.layers.0.mlp.experts.0.gate_proj.weight", weight),
    ]
    assert list(dequantize_nvfp4_mtp_experts(stream)) == stream


def test_mtp_incomplete_expert_is_rejected():
    _, tensors = _expert_tensors("mtp.layers.0.mlp.experts.0.up_proj", 0)
    tensors.pop("mtp.layers.0.mlp.experts.0.up_proj.weight_scale_2")
    with pytest.raises(ValueError, match="Incomplete NVFP4 MTP expert"):
        list(dequantize_nvfp4_mtp_experts(tensors.items()))


def _load_ple_helpers():
    source = (MODELS / "qwen4_exp.py").read_text()
    wanted = {"_ple_table_is_nvfp4", "_validate_ple_embedding_dtype"}
    namespace = {"Optional": Optional, "QuantizationConfig": object}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(source, node), namespace)
    return namespace


PLE_HELPERS = _load_ple_helpers()


class _Config:
    def __init__(self, ple_embedding_dtype=None):
        self.ple_embedding_dtype = ple_embedding_dtype


@pytest.mark.parametrize("requested", [None, "nvfp4"])
def test_ple_embedding_dtype_allows_nvfp4_table(requested):
    PLE_HELPERS["_validate_ple_embedding_dtype"](_Config(requested), True)


def test_ple_embedding_dtype_conflict_with_nvfp4_table_is_rejected():
    with pytest.raises(ValueError, match="conflicts with the NVFP4 PLE table"):
        PLE_HELPERS["_validate_ple_embedding_dtype"](_Config("float8_e4m3fn"), True)
    # A bf16/fp8 table ignores the flag, so it stays accepted.
    PLE_HELPERS["_validate_ple_embedding_dtype"](_Config("float8_e4m3fn"), False)


class _QuantConfig:
    def __init__(self, algo):
        self._algo = algo

    def _resolve_quant_algo(self, prefix):
        return self._algo


def test_ple_table_is_nvfp4_requires_prefix_and_algo():
    resolve = PLE_HELPERS["_ple_table_is_nvfp4"]
    assert resolve(_QuantConfig("NVFP4"), "model.layers.1.ple")
    assert not resolve(_QuantConfig("FP8"), "model.layers.1.ple")
    assert not resolve(_QuantConfig("NVFP4"), "")
    assert not resolve(None, "model.layers.1.ple")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_gather_kernel_matches_reference(tmp_path):
    import importlib.util

    source = (MODELS / "qwen4_exp.py").read_text()
    kernel_node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_gather_ple_nvfp4_from_pinned_kernel"
    )
    start = min(d.lineno for d in kernel_node.decorator_list)
    module_path = tmp_path / "ple_nvfp4_kernel.py"
    module_path.write_text(
        "import triton\nimport triton.language as tl\n\n"
        + "\n".join(source.splitlines()[start - 1 : kernel_node.end_lineno])
    )
    spec = importlib.util.spec_from_file_location("ple_nvfp4_kernel", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows, dim = 64, 160
    codes, group_scale, global_scale = _random_matrix(rows, dim, seed=7)
    packed = _pack(codes).pin_memory()
    scales = group_scale.pin_memory()
    expected = _expected(codes, group_scale, global_scale)
    ids = torch.tensor([0, 63, 5, 17, 64, 1000], device="cuda")
    out = torch.empty(ids.numel(), dim, dtype=torch.bfloat16, device="cuda")
    module._gather_ple_nvfp4_from_pinned_kernel[(ids.numel(),)](
        packed.data_ptr(),
        scales.data_ptr(),
        global_scale.cuda(),
        ids,
        out,
        embedding_dim=dim,
        tp_vocab_start=0,
        tp_vocab_end=rows,
        GROUP_SIZE=16,
        BLOCK_D=256,
    )
    torch.cuda.synchronize()
    assert torch.equal(out[:4].cpu(), expected[[0, 63, 5, 17]])
    assert bool((out[4:] == 0).all())

    # Non-zero TP window: the pinned table is tp-local, so ids are rebased.
    start, end = 16, 32
    ids = torch.tensor([16, 24, 31, 15, 32], device="cuda")
    out = torch.empty(ids.numel(), dim, dtype=torch.bfloat16, device="cuda")
    module._gather_ple_nvfp4_from_pinned_kernel[(ids.numel(),)](
        packed[start:end].contiguous().data_ptr(),
        scales[start:end].contiguous().data_ptr(),
        global_scale.cuda(),
        ids,
        out,
        embedding_dim=dim,
        tp_vocab_start=start,
        tp_vocab_end=end,
        GROUP_SIZE=16,
        BLOCK_D=256,
    )
    torch.cuda.synchronize()
    assert torch.equal(out[:3].cpu(), expected[[16, 24, 31]])
    assert bool((out[3:] == 0).all())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
