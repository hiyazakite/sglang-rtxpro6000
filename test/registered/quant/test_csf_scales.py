#!/usr/bin/env python3
"""Coverage for the CSF decoder that unlocks mixed NVFP4 checkpoints.

`nvidia/Qwen3.8-Flash-Next-FP4-Local` stores every routed expert's E4M3 block
scale as two tensors (`*.nvfp4_csf_fixed` plus `*.nvfp4_csf_exceptions`, codec
`byte-window4-fixed-stream-u24-exceptions/1`) instead of a plain
`weight_scale`. These tests pin the decode arithmetic against an encoder
written from the codec description rather than the decoder, the stream
pairing, and that a snapshot whose shards sit in a subdirectory named by the
index is still found by `_prepare_weights`.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.csf_scales import (
    decode_csf_scale_streams,
    decode_nvfp4_csf_scale,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def encode_csf_scales(scales):
    """Reference encoder, independent of the decoder.

    One base byte per row (the row minimum, capped at 240) followed by the
    rows' 4-bit offsets, low nibble first; every byte outside the resulting
    0..15 window is listed as an exception carrying its replacement byte.
    """
    raw = scales.view(torch.uint8).to(torch.int64)
    rows, cols = raw.shape
    bases = raw.min(dim=1, keepdim=True).values.clamp(max=240)
    codes = raw - bases
    outlier = (codes < 0) | (codes > 15)
    codes = torch.where(outlier, 0, codes)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat(
        (
            bases.to(torch.uint8).reshape(rows // 16, 16),
            packed.reshape(rows // 16, -1),
        ),
        dim=1,
    )
    positions = outlier.reshape(-1).nonzero().reshape(-1)
    words = (raw.reshape(-1)[positions] << 24) | positions
    return fixed, words.to(torch.int32).view(torch.uint32)


class TestCsfScaleDecode(CustomTestCase):
    def test_decode_matches_source_scales(self):
        for shape in ((640, 160), (2560, 40)):
            with self.subTest(shape=shape):
                torch.manual_seed(0)
                raw = torch.randint(0x30, 0x40, shape, dtype=torch.uint8)
                # Bytes too far from the row's minimum for the 4-bit window.
                raw.view(-1)[torch.randperm(raw.numel())[:64]] = 0x7E
                fixed, exceptions = encode_csf_scales(raw.view(torch.float8_e4m3fn))
                self.assertGreaterEqual(exceptions.numel(), 64)

                decoded = decode_nvfp4_csf_scale(fixed, exceptions)

                self.assertEqual(decoded.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(decoded.shape), shape)
                self.assertTrue(torch.equal(decoded.view(torch.uint8), raw))

    def test_stream_pairs_are_replaced_by_the_scale(self):
        raw = torch.full((16, 4), 0x38, dtype=torch.uint8)
        fixed, exceptions = encode_csf_scales(raw.view(torch.float8_e4m3fn))
        weights = [
            ("e.weight", torch.zeros(1)),
            ("e.weight_scale.nvfp4_csf_exceptions", exceptions),
            ("e.weight_scale.nvfp4_csf_fixed", fixed),
        ]

        out = dict(decode_csf_scale_streams(weights))

        self.assertEqual(list(out), ["e.weight", "e.weight_scale"])
        self.assertTrue(torch.equal(out["e.weight_scale"].view(torch.uint8), raw))

    def test_unpaired_stream_is_rejected(self):
        fixed, _ = encode_csf_scales(
            torch.full((16, 4), 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn)
        )
        with self.assertRaisesRegex(ValueError, "without a partner"):
            list(decode_csf_scale_streams([("e.weight_scale.nvfp4_csf_fixed", fixed)]))


class TestSubdirectoryShardDiscovery(CustomTestCase):
    """The published checkpoint keeps its 21 shards under `tensors/`, where the
    loader's flat `*.safetensors` glob of the snapshot root finds nothing."""

    def _snapshot(self, root):
        shard = Path(root) / "tensors" / "model-00001-of-00021.safetensors"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"")
        (Path(root) / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"w": "tensors/model-00001-of-00021.safetensors"}}
            )
        )
        return shard

    def test_shards_listed_by_the_index_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = self._snapshot(tmp)
            loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
            with patch(
                "sglang.srt.model_loader.loader.get_server_args", return_value=None
            ):
                folder, files, use_safetensors = loader._prepare_weights(
                    tmp, None, False
                )

        self.assertEqual(folder, tmp)
        self.assertEqual(files, [str(shard)])
        self.assertTrue(use_safetensors)

    def test_flat_snapshot_still_resolves_without_an_index(self):
        """The fallback must not disturb the ordinary case: shards at the
        snapshot root are found by the glob, with no index involved."""
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "model-00001-of-00002.safetensors"
            shard.write_bytes(b"")
            loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
            with patch(
                "sglang.srt.model_loader.loader.get_server_args", return_value=None
            ):
                folder, files, use_safetensors = loader._prepare_weights(
                    tmp, None, False
                )

        self.assertEqual(folder, tmp)
        self.assertEqual(files, [str(shard)])
        self.assertTrue(use_safetensors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
