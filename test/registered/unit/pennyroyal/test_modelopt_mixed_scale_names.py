"""Which block-scale parameter an MXFP8 layer's checkpoint tensor lands in.

Mixed-precision (QAD) checkpoints hold MXFP8 and NVFP4 layers in one file, and
the two spell their block scales the same way in the checkpoint but register
them under different parameter names: ``Fp8LinearMethod`` registers MXFP8
scales as ``weight_scale_inv``, while the NVFP4 methods genuinely use
``weight_scale``. The loader renames wholesale for a pure MXFP8 checkpoint
(``ModelOptModelLoader``'s ``get_name() == "mxfp8"`` branch); for a mixed one it
has to ask per layer, which is what ``mxfp8_weight_scale_name`` answers.
"""

import pytest

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptMixedPrecisionConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The fusions the checkpoint is written with, shard -> fused name.
PACKED_MODULES = {
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_a", "in_proj_b"],
}

# Module prefix as the model builds it; the checkpoint lists the same layer
# under the text-decoder spelling.
MODULE_PREFIX = "model.layers.0.linear_attn"
CHECKPOINT_PREFIX = "model.language_model.layers.0.linear_attn"


def _config(quantized_layers):
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "quantized_layers": quantized_layers,
            "packed_modules_mapping": PACKED_MODULES,
        }
    )


def test_mxfp8_scale_lands_in_weight_scale_inv():
    config = _config({f"{MODULE_PREFIX}.out_proj": {"quant_algo": "MXFP8"}})
    assert (
        config.mxfp8_weight_scale_name(f"{MODULE_PREFIX}.out_proj.weight_scale")
        == f"{MODULE_PREFIX}.out_proj.weight_scale_inv"
    )


def test_mxfp8_scale_resolves_through_the_checkpoint_spelling():
    """The checkpoint names the layer `model.language_model.*` while the module
    lives at `model.layers.*`."""
    config = _config({f"{CHECKPOINT_PREFIX}.out_proj": {"quant_algo": "MXFP8"}})
    assert (
        config.mxfp8_weight_scale_name(f"{MODULE_PREFIX}.out_proj.weight_scale")
        == f"{MODULE_PREFIX}.out_proj.weight_scale_inv"
    )


def test_nvfp4_scale_keeps_its_name():
    config = _config({f"{MODULE_PREFIX}.out_proj": {"quant_algo": "NVFP4"}})
    name = f"{MODULE_PREFIX}.out_proj.weight_scale"
    assert config.mxfp8_weight_scale_name(name) == name


def test_uniformly_quantized_fused_layer_renames():
    """`in_proj_qkvz` is only ever a checkpoint name: `quantized_layers` lists
    its shards, so the algo comes from the shards agreeing."""
    config = _config(
        {
            f"{MODULE_PREFIX}.in_proj_qkv": {"quant_algo": "MXFP8"},
            f"{MODULE_PREFIX}.in_proj_z": {"quant_algo": "MXFP8"},
        }
    )
    assert (
        config.mxfp8_weight_scale_name(f"{MODULE_PREFIX}.in_proj_qkvz.weight_scale")
        == f"{MODULE_PREFIX}.in_proj_qkvz.weight_scale_inv"
    )


def test_shards_disagreeing_leaves_the_name_alone():
    """A fused name whose shards were quantized differently cannot be resolved;
    guessing a rename would trade a clear missing-parameter error for a load
    into the wrong parameter."""
    config = _config(
        {
            f"{MODULE_PREFIX}.in_proj_qkv": {"quant_algo": "MXFP8"},
            f"{MODULE_PREFIX}.in_proj_z": {"quant_algo": "NVFP4"},
        }
    )
    name = f"{MODULE_PREFIX}.in_proj_qkvz.weight_scale"
    assert config.mxfp8_weight_scale_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        f"{MODULE_PREFIX}.out_proj.weight_scale_2",
        f"{MODULE_PREFIX}.out_proj.input_scale",
        f"{MODULE_PREFIX}.out_proj.weight",
    ],
)
def test_names_that_are_not_block_scales_pass_through(name):
    config = _config({f"{MODULE_PREFIX}.out_proj": {"quant_algo": "MXFP8"}})
    assert config.mxfp8_weight_scale_name(name) == name


def test_unquantized_layer_passes_through():
    """A `.weight_scale` outside `quantized_layers` (MTP, vision tower) has no
    algo to resolve, so it is not renamed."""
    config = _config({f"{MODULE_PREFIX}.out_proj": {"quant_algo": "MXFP8"}})
    name = "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale"
    assert config.mxfp8_weight_scale_name(name) == name


def test_other_configs_do_not_expose_the_hook():
    """The loader applies the hook only when the config offers it, so a single-
    algo config must not accidentally opt in."""
    assert not hasattr(
        ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16),
        "mxfp8_weight_scale_name",
    )
