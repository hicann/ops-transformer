#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Generate diverse TTK CSV test cases for quant_flash_mla_with_kvcache."""

import csv
import math
import sys
import os


def ceil_div(a, b):
    return (a + b - 1) // b


def gen_case(
    name,
    B,
    N_q,
    seqused_q,
    cache_seqlens,
    kv_layout_golden,
    layout_q,
    layout_out,
    mask_mode=0,
    enable_lse=False,
    block_size=128,
):
    """Generate one CSV row dict.

    kv_layout_golden: 'BnNBsD', 'BnBsND', 'NZ'
    layout_q: 'BNSD', 'BSND', 'TND'
    """
    N_kv = 1
    D = 512
    D_V = 512
    D_rope = 64
    QK_DIM = D + D_rope  # 576
    softmax_scale = 1.0 / math.sqrt(D)  # 1/sqrt(512)

    max_sq = max(seqused_q)
    T = sum(seqused_q)
    num_blocks = sum(ceil_div(cs, block_size) for cs in cache_seqlens)
    max_blocks_per_batch = ceil_div(max(cache_seqlens), block_size)

    # Op layout_kv name
    kv_layout_op = {"BnNBsD": "PA_BNBD", "BnBsND": "PA_BBND", "NZ": "PA_NZ"}[
        kv_layout_golden
    ]

    # Q shapes by layout
    if layout_q == "BNSD":
        q_shape = (B, N_q, max_sq, QK_DIM)
        q_descale_shape = (B, N_q, max_sq)
        cu_seqlens_q_shape = None
    elif layout_q == "BSND":
        q_shape = (B, max_sq, N_q, QK_DIM)
        q_descale_shape = (B, max_sq, N_q)
        cu_seqlens_q_shape = None
    elif layout_q == "TND":
        q_shape = (T, N_q, QK_DIM)
        q_descale_shape = (T, N_q)
        cu_seqlens_q_shape = (B + 1,)
    else:
        raise ValueError(f"Unsupported layout_q: {layout_q}")

    # K cache shapes by KV layout
    if kv_layout_golden == "BnNBsD":
        k_cache_shape = (num_blocks, N_kv, block_size, QK_DIM)
    elif kv_layout_golden == "BnBsND":
        k_cache_shape = (num_blocks, block_size, N_kv, QK_DIM)
    elif kv_layout_golden == "NZ":
        k_cache_shape = (num_blocks, N_kv, QK_DIM // 32, block_size, 32)
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout_golden}")

    k_descale_shape = (1,)
    block_table_shape = (B, max_blocks_per_batch)
    cache_seqlens_shape = (B,)
    seqused_q_shape = (B,)
    attn_mask_shape = (2048, 2048) if mask_mode == 3 else None
    metadata_shape = (2, 4096)

    # For TND, cu_seqlens_q needs values
    if layout_q == "TND":
        cu_seqlens_q_vals = [0]
        for s in seqused_q:
            cu_seqlens_q_vals.append(cu_seqlens_q_vals[-1] + s)
        cu_seqlens_q_values_attr = f"'cu_seqlens_q_values': {cu_seqlens_q_vals}"
    else:
        cu_seqlens_q_values_attr = ""

    tensor_shapes = (
        q_shape,
        k_cache_shape,
        q_descale_shape,
        k_descale_shape,
        block_table_shape,
        cache_seqlens_shape,
        cu_seqlens_q_shape,
        seqused_q_shape,
        attn_mask_shape,
        metadata_shape,
    )

    def fmt_shape(s):
        if s is None:
            return "None"
        return str(tuple(s))

    tensor_shapes_str = "(" + ",".join(fmt_shape(s) for s in tensor_shapes) + ")"

    dtypes = (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float32",
        "float32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int8",
        "int32",
    )
    dtypes_str = "(" + ",".join(f"'{d}'" for d in dtypes) + ")"

    attrs = {
        "B": B,
        "N_q": N_q,
        "N_kv": N_kv,
        "D": D,
        "D_V": D_V,
        "D_rope": D_rope,
        "quant_mode": 1,
        "softmax_scale": softmax_scale,
        "seqused_q_values": list(seqused_q),
        "cache_seqlens_values": list(cache_seqlens),
        "kv_cache_layout": kv_layout_golden,
        "block_size": block_size,
        "mask_mode": mask_mode,
        "enable_lse": enable_lse,
        "layout_q": layout_q,
        "layout_kv": kv_layout_op,
        "layout_out": layout_out,
        "return_softmax_lse": enable_lse,
        "graph_path": 0,
        "q_data_range": [-1.0, 1.0],
        "k_data_range": [-5.0, 5.0],
        "v_data_range": [-5.0, 5.0],
        "p_scale": 1.0,
        "input_layout": layout_q,
        "output_layout": layout_out,
        "max_seqlen_q": max_sq,
        "max_seqlen_kv": max(cache_seqlens),
        "head_dim_v": D_V,
    }
    if cu_seqlens_q_values_attr:
        attrs_str_parts = []
        for k, v in attrs.items():
            attrs_str_parts.append(f"'{k}': {v!r}")
        attrs_str_parts.append(cu_seqlens_q_values_attr)
        attrs_str = "{" + ", ".join(attrs_str_parts) + "}"
    else:
        attrs_str = "{" + ", ".join(f"'{k}': {v!r}" for k, v in attrs.items()) + "}"

    input_ranges = "((-1.0,1.0),(-5.0,5.0),(-5.0,5.0))"

    return {
        "testcase_name": name,
        "api_name": "torch.ops.cann_ops_transformer.quant_flash_mla",
        "tensor_view_shapes": tensor_shapes_str,
        "tensor_dtypes": dtypes_str,
        "tensor_formats": "",
        "attributes": attrs_str,
        "output_tensor_indexes": "",
        "golden_api": "",
        "input_data_ranges": input_ranges,
    }


CASES = [
    # --- BNSD layout, PA_BNBD (BnNBsD) ---
    ("bnsd_bnbd_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "BnNBsD", "BNSD", "BNSD"),
    ("bnsd_bnbd_b1_n6_sq8_skv1024", 1, 6, [8], [1024], "BnNBsD", "BNSD", "BNSD"),
    (
        "bnsd_bnbd_b2_n12_sq1_skv1024",
        2,
        12,
        [1, 1],
        [1024, 1024],
        "BnNBsD",
        "BNSD",
        "BNSD",
    ),
    ("bnsd_bnbd_b1_n96_sq1_skv4096", 1, 96, [1], [4096], "BnNBsD", "BNSD", "BNSD"),
    (
        "bnsd_bnbd_b4_n6_sq16_skv128",
        4,
        6,
        [16, 16, 16, 16],
        [128, 128, 128, 128],
        "BnNBsD",
        "BNSD",
        "BNSD",
    ),
    (
        "bnsd_bnbd_b1_n6_sq1_skv1024_lse",
        1,
        6,
        [1],
        [1024],
        "BnNBsD",
        "BNSD",
        "BNSD",
        0,
        True,
    ),
    (
        "bnsd_bnbd_b2_n24_sq8_skv512_causal",
        2,
        24,
        [8, 8],
        [512, 512],
        "BnNBsD",
        "BNSD",
        "BNSD",
        3,
    ),
    # --- BNSD layout, different output layouts ---
    ("bnsd_bnbd_b1_n6_sq1_skv1024_out_tnd", 1, 6, [1], [1024], "BnNBsD", "BNSD", "TND"),
    (
        "bnsd_bnbd_b1_n6_sq1_skv1024_out_bsnd",
        1,
        6,
        [1],
        [1024],
        "BnNBsD",
        "BNSD",
        "BSND",
    ),
    # --- BSND layout ---
    ("bsnd_bnbd_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "BnNBsD", "BSND", "BSND"),
    (
        "bsnd_bnbd_b2_n12_sq1_skv512",
        2,
        12,
        [1, 1],
        [512, 512],
        "BnNBsD",
        "BSND",
        "BSND",
    ),
    ("bsnd_bnbd_b1_n6_sq1_skv256", 1, 6, [1], [256], "BnNBsD", "BSND", "BSND"),
    (
        "bsnd_bnbd_b4_n6_sq1_skv128",
        4,
        6,
        [1, 1, 1, 1],
        [128, 128, 128, 128],
        "BnNBsD",
        "BSND",
        "BSND",
    ),
    # BSND + sq>1 cases (previously avoided due to metadata op cu_seqlens_q=None issue)
    ("bsnd_bnbd_b1_n6_sq4_skv512", 1, 6, [4], [512], "BnNBsD", "BSND", "BSND"),
    ("bsnd_bnbd_b1_n6_sq8_skv1024", 1, 6, [8], [1024], "BnNBsD", "BSND", "BSND"),
    ("bsnd_bnbd_b2_n6_sq4_skv256", 2, 6, [4, 4], [256, 256], "BnNBsD", "BSND", "BSND"),
    # --- TND layout ---
    ("tnd_bnbd_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "BnNBsD", "TND", "TND"),
    ("tnd_bnbd_b2_n12_sq4_skv512", 2, 12, [4, 4], [512, 512], "BnNBsD", "TND", "TND"),
    ("tnd_bnbd_b1_n96_sq16_skv1024", 1, 96, [16], [1024], "BnNBsD", "TND", "TND"),
    ("tnd_bnbd_b2_n6_sq1_skv256", 2, 6, [1, 1], [256, 256], "BnNBsD", "TND", "TND"),
    ("tnd_bnbd_b1_n6_sq4_skv256", 1, 6, [4], [256], "BnNBsD", "TND", "TND"),
    # --- PA_BBND (BnBsND) ---
    ("bnsd_bbnd_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "BnBsND", "BNSD", "BNSD"),
    (
        "bnsd_bbnd_b2_n12_sq8_skv1024",
        2,
        12,
        [8, 8],
        [1024, 1024],
        "BnBsND",
        "BNSD",
        "BNSD",
    ),
    ("tnd_bbnd_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "BnBsND", "TND", "TND"),
    ("bsnd_bbnd_b1_n6_sq1_skv256", 1, 6, [1], [256], "BnBsND", "BSND", "BSND"),
    ("bnsd_bbnd_b1_n6_sq8_skv512", 1, 6, [8], [512], "BnBsND", "BNSD", "BNSD"),
    # --- PA_NZ ---
    ("bnsd_nz_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "NZ", "BNSD", "BNSD"),
    ("bnsd_nz_b1_n96_sq1_skv4096", 1, 96, [1], [4096], "NZ", "BNSD", "BNSD"),
    ("bsnd_nz_b2_n6_sq1_skv512", 2, 6, [1, 1], [512, 512], "NZ", "BSND", "BSND"),
    ("bnsd_nz_b1_n6_sq8_skv256", 1, 6, [8], [256], "NZ", "BNSD", "BNSD"),
    # --- Mixed combos ---
    ("bnsd_bnbd_b1_n48_sq8_skv2048", 1, 48, [8], [2048], "BnNBsD", "BNSD", "BNSD"),
    ("bnsd_bnbd_b2_n6_sq1_skv128", 2, 6, [1, 1], [128, 128], "BnNBsD", "BNSD", "BNSD"),
    (
        "bnsd_bnbd_b1_n6_sq8_skv256_causal",
        1,
        6,
        [8],
        [256],
        "BnNBsD",
        "BNSD",
        "BNSD",
        3,
    ),
    ("tnd_bbnd_b1_n6_sq4_skv256", 1, 6, [4], [256], "BnBsND", "TND", "TND"),
    # --- Additional edge cases ---
    ("bnsd_bnbd_b1_n1_sq1_skv128", 1, 1, [1], [128], "BnNBsD", "BNSD", "BNSD"),
    ("bnsd_bnbd_b1_n6_sq1_skv256", 1, 6, [1], [256], "BnNBsD", "BNSD", "BNSD"),
    (
        "bnsd_bnbd_b2_n6_sq8_skv128_causal",
        2,
        6,
        [8, 8],
        [128, 128],
        "BnNBsD",
        "BNSD",
        "BNSD",
        3,
    ),
    ("bsnd_bbnd_b1_n6_sq1_skv1024", 1, 6, [1], [1024], "BnBsND", "BSND", "BSND"),
]


def main():
    out_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/z00954345/ops-transformer_mla_1007/attention/quant_flash_mla_with_kvcache/tests/qmla_mxfp8_min.csv"
    )

    rows = []
    for case in CASES:
        rows.append(gen_case(*case))

    fieldnames = [
        "testcase_name",
        "api_name",
        "tensor_view_shapes",
        "tensor_dtypes",
        "tensor_formats",
        "attributes",
        "output_tensor_indexes",
        "golden_api",
        "input_data_ranges",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} cases → {out_path}")
    for r in rows:
        print(f"  {r['testcase_name']}")


if __name__ == "__main__":
    main()
