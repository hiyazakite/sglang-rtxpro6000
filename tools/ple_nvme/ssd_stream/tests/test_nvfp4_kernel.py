import pytest
import torch

pytest.importorskip("sglang")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def test_staged_nvfp4_rows_dequantize_to_bfloat16():
    from sglang_ssd_stream.backend import _copy_ple_staged_nvfp4_rows_kernel

    rows, columns, group = 5, 160, 16
    generator = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 16, (rows, columns), generator=generator, dtype=torch.int32)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    scales = (torch.rand(rows, columns // group, generator=generator) + 0.25).to(
        torch.float8_e4m3fn
    )
    global_scale = torch.tensor([0.02])
    row_nbytes = columns // 2 + columns // group
    staged = torch.cat((packed, scales.view(torch.uint8)), dim=1)
    assert staged.shape == (rows, row_nbytes)
    staged = staged.contiguous().pin_memory()

    lut = torch.tensor(E2M1)
    values = torch.where(codes & 8 != 0, -lut[codes & 7], lut[codes & 7])
    expected = (
        values * scales.float().repeat_interleave(group, dim=1) * global_scale
    ).to(torch.bfloat16)

    out = torch.empty(rows, columns, dtype=torch.bfloat16, device="cuda")
    _copy_ple_staged_nvfp4_rows_kernel[(rows,)](
        staged.data_ptr(),
        global_scale.cuda(),
        out,
        embedding_dim=columns,
        row_nbytes=row_nbytes,
        GROUP_SIZE=group,
        BLOCK_D=256,
    )
    torch.cuda.synchronize()

    assert torch.equal(out.cpu(), expected)
