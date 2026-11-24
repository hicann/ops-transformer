# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Run a dual-level MXFP4 grouped matmul and check both experts' outputs."""

import importlib
import os

import torch
import torch_npu


def main():
    ops = importlib.import_module(
        os.environ.get("CANN_TRANSFORMER_PACKAGE", "cann_ops_transformer")
    )
    m, k, n, experts = 32, 640, 128, 2
    level0_group_size, level1_group_size = 512, 32
    fp4_elements_per_byte = 2
    mx_scale_pair = 2
    level1_pairs = k // (mx_scale_pair * level1_group_size)
    level0_groups = (k + level0_group_size - 1) // level0_group_size
    group_rows = m // experts
    packed_fp4_ones = 0x22
    e8m0_one, e8m0_two = 127, 128
    weight_level1_scale = 2
    tail_level0_scale = 2
    bias_step = 16
    format_nz, format_nd = 29, 2
    device = "npu:0"

    x = torch.full(
        (m, k // fp4_elements_per_byte),
        packed_fp4_ones,
        dtype=torch.int8,
        device=device,
    )
    weight = torch_npu.npu_format_cast(
        torch.full(
            (experts, k, n // fp4_elements_per_byte),
            packed_fp4_ones,
            dtype=torch.int8,
            device=device,
        ),
        format_nz,
    )
    x_level0 = torch.ones((m, level0_groups), dtype=torch.float32, device=device)
    x_level0[:, 1] = tail_level0_scale
    weight_level0 = torch.ones(
        (experts, level0_groups, n), dtype=torch.float32, device=device
    )
    weight_level0[:, 1, :] = tail_level0_scale
    x_level1 = torch.full(
        (m, level1_pairs, mx_scale_pair), e8m0_one, dtype=torch.uint8, device=device
    )
    weight_level1 = torch_npu.npu_format_cast(
        torch.full(
            (experts, level1_pairs, n, mx_scale_pair),
            e8m0_two,
            dtype=torch.uint8,
            device=device,
        ),
        format_nd,
    )
    group_list = torch.tensor([group_rows] * experts, dtype=torch.int64, device=device)
    bias = torch.full((experts, n), bias_step, dtype=torch.float32, device=device)
    bias[1, :] = bias_step + bias_step

    (output,) = ops.dual_level_grouped_matmul(
        [x],
        [weight],
        [x_level0],
        [x_level1],
        [weight_level0],
        [weight_level1],
        level0_group_size=level0_group_size,
        level1_group_size=level1_group_size,
        group_list=group_list,
        group_list_type=1,
        bias=[bias],
        output_dtype=torch.bfloat16,
    )

    matmul = weight_level1_scale * (
        level0_group_size
        + (k - level0_group_size) * tail_level0_scale * tail_level0_scale
    )
    expected = torch.arange(1, experts + 1, dtype=torch.float32) * bias_step + matmul
    expected = (
        expected.repeat_interleave(group_rows)
        .unsqueeze(1)
        .expand(m, n)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print(f"DualLevelGroupedMatmul passed, BF16 shape {tuple(output.shape)}")


if __name__ == "__main__":
    main()
