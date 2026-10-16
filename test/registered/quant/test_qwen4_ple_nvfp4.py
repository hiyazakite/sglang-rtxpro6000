#!/usr/bin/env python3
"""Model-level coverage for the NVFP4 PLE (per-layer embedding) table.

`test_nvfp4_embedding.py` pins the dequant arithmetic of
`ModelOptNvFp4EmbeddingMethod`; these tests pin the wiring around it: which
quantization config selects the quantized table, that the table's scales are
applied exactly once, and that the gathered rows still equal the rows a plain
dequantization produces.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GROUP_SIZE = 16
EMBED_DIM = 64
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
HEAD_DIM_PER_NGRAM = EMBED_DIM // NGRAM_HEADS

# Module prefix as the model builds it, checkpoint key as the NVFP4 export
# spells it. The loader relies on the `model.layers.` <-> `model.language_model.`
# cross-mapping between the two, so resolving through the checkpoint key is the
# case that has to work.
MODULE_PREFIX = "model.layers.0.ple.ple_embedding.ngram_embedding"
CHECKPOINT_KEY = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"

# Written out independently of the implementation: the E2M1 code points in
# magnitude order, so index == the 3-bit magnitude code.
_REFERENCE_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class _SingleRankGroup:
    """Stand-in for the TP group: world_size 1 keeps symmetric memory out."""

    world_size = 1


@contextmanager
def _single_rank_tp_group():
    """`VocabParallelEmbedding.forward` reaches for the live TP group even at
    tp_size 1, so hand it a single-rank one instead of standing up process
    groups."""
    with patch(
        "sglang.srt.layers.vocab_parallel_embedding.get_tp_group",
        return_value=_SingleRankGroup(),
    ):
        yield


def _text_config(**overrides) -> SimpleNamespace:
    config = dict(
        vocab_size=256,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=0,
        seed=1234,
        ple_embedding_dtype=None,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _quant_config(quant_algo: str = "NVFP4"):
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
            "group_size": GROUP_SIZE,
            "quantized_layers": {CHECKPOINT_KEY: {"quant_algo": quant_algo}},
            "packed_modules_mapping": {},
        }
    )


def _reference_rows(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: float
) -> torch.Tensor:
    """Dequantize each row of an NVFP4 table with a plain per-element loop."""
    rows, half = packed.shape
    hidden = half * 2
    out = torch.zeros(rows, hidden, dtype=torch.float32)
    for r in range(rows):
        for c in range(hidden):
            byte = int(packed[r, c // 2])
            code = (byte & 0x0F) if c % 2 == 0 else (byte >> 4)
            magnitude = _REFERENCE_E2M1[code & 0x7]
            value = -magnitude if code & 0x8 else magnitude
            out[r, c] = value * float(block_scale[r, c // GROUP_SIZE]) * global_scale
    return out


class TestQwen4PleNvFp4Table(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # VocabParallelEmbedding resolves the live TP rank/size while it builds,
        # so pin single-rank topology rather than standing up process groups.
        cls._tp_override = get_parallel().override(tp_size=1, tp_rank=0)
        cls._tp_override.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._tp_override.__exit__(None, None, None)

    def _build(self, quant_config=None, **config_overrides):
        layer = Qwen4ExpNGramEmbedding(
            _text_config(**config_overrides),
            embedding_dim=EMBED_DIM,
            ple_layer_index=0,
            quant_config=quant_config,
            prefix=MODULE_PREFIX,
        )
        if layer.nvfp4_table:
            self._fill_as_checkpoint(layer)
        return layer

    def _fill_as_checkpoint(self, layer):
        """Fill the packed table the way the NVFP4 export lays it out."""
        weight = layer.ngram_embedding.weight
        scale = layer.ngram_embedding.weight_scale
        generator = torch.Generator().manual_seed(0)
        weight.data.copy_(
            torch.randint(0, 256, weight.shape, dtype=torch.uint8, generator=generator)
        )
        # Keep the block scales in a range e4m3 represents exactly.
        scale.data.copy_(
            torch.randint(1, 8, scale.shape, dtype=torch.int32, generator=generator).to(
                torch.float8_e4m3fn
            )
        )
        layer.ngram_embedding.weight_scale_2.data.fill_(0.125)

    def test_nvfp4_entry_selects_the_quantized_table(self):
        layer = self._build(_quant_config("NVFP4"))
        partition = layer.ngram_embedding.num_embeddings_per_partition
        self.assertTrue(layer.nvfp4_table)
        self.assertEqual(layer.ngram_embedding.weight.dtype, torch.uint8)
        self.assertEqual(
            tuple(layer.ngram_embedding.weight.shape),
            (partition, HEAD_DIM_PER_NGRAM // 2),
        )
        self.assertEqual(
            tuple(layer.ngram_embedding.weight_scale.shape),
            (partition, HEAD_DIM_PER_NGRAM // GROUP_SIZE),
        )
        self.assertEqual(layer.ngram_embedding.weight_scale.dtype, torch.float8_e4m3fn)

    def test_other_algos_keep_the_dense_table(self):
        for quant_algo in ("MXFP8", "FP8"):
            with self.subTest(quant_algo=quant_algo):
                layer = self._build(_quant_config(quant_algo))
                self.assertFalse(layer.nvfp4_table)
                self.assertEqual(layer.ngram_embedding.weight.dtype, torch.bfloat16)

    def test_unquantized_config_keeps_the_dense_table(self):
        layer = self._build(None)
        self.assertFalse(layer.nvfp4_table)
        self.assertEqual(layer.ngram_embedding.weight.dtype, torch.bfloat16)

    def test_table_scale_is_not_applied_twice(self):
        """NVFP4 folds both scales into the gather; re-applying them on top of
        the gathered rows would scale the PLE contribution twice."""
        quantized = self._build(_quant_config("NVFP4"))
        embeddings = torch.randn(4, HEAD_DIM_PER_NGRAM)
        torch.testing.assert_close(quantized.apply_table_scale(embeddings), embeddings)

        dense = self._build(None)
        dense.ngram_embedding.weight_scale.fill_(2.0)
        torch.testing.assert_close(
            dense.apply_table_scale(embeddings), embeddings * 2.0
        )

    def test_gather_matches_reference_dequant(self):
        layer = self._build(_quant_config("NVFP4"))
        ids = torch.tensor([[0, 5, 5], [17, 3, 40]])
        with _single_rank_tp_group():
            got = layer.ngram_embedding(ids)

        expected = _reference_rows(
            layer.ngram_embedding.weight[ids.reshape(-1)],
            layer.ngram_embedding.weight_scale[ids.reshape(-1)].float(),
            float(layer.ngram_embedding.weight_scale_2),
        )

        self.assertEqual(tuple(got.shape), (2, 3, HEAD_DIM_PER_NGRAM))
        self.assertEqual(got.dtype, torch.bfloat16)
        torch.testing.assert_close(
            got.reshape(-1, HEAD_DIM_PER_NGRAM).float(),
            expected.to(torch.bfloat16).float(),
            rtol=0,
            atol=0,
        )

    def test_ple_embedding_dtype_override_is_rejected_for_nvfp4(self):
        with self.assertRaisesRegex(ValueError, "conflicts with the NVFP4 PLE"):
            self._build(_quant_config("NVFP4"), ple_embedding_dtype="float8_e4m3fn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
