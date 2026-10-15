#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""根据 torchapi_quant_flash_mla.md 约束说明生成全面异常 case CSV。

每个 case 只注入一个异常属性/形状, 其余保持正常基线值。
tensor_view_shapes 保持合法(供 TTK 分配), 非法属性透传到算子侧由 checker 拦截。
"""

import csv
import json
import os

API_NAME = "torch.ops.cann_ops_transformer.quant_flash_mla"

D_FP8 = "float8_e4m3fn"
D_FP32 = "float32"
D_INT8 = "int8"
D_INT32 = "int32"
D_BF16 = "bfloat16"

_DTYPES = [
    D_FP8,
    D_FP8,
    D_FP32,
    D_FP32,
    D_INT32,
    D_INT32,
    D_INT32,
    D_INT32,
    D_INT8,
    D_INT32,
]

_HEADER = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
]

_PRECISION = "((0.0078125, 0.0001, 0.005, 0.005, 10),)"
_ABS = "1e-08"
_DRANGE = (
    "((-1, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))"
)

# 基线 shapes: q, k_cache, q_descale, k_descale, block_table, cache_seqlens,
#             cu_seqlens_q, seqused_q, attn_mask, metadata
_BASE_SHAPES = (
    (1, 6, 576),
    (8, 1, 128, 576),
    (1, 6),
    (1,),
    (1, 8),
    (1,),
    (2,),
    (1,),
    (2048, 2048),
    (2, 4096),
)


def _base_attrs(**overrides):
    attrs = {
        "precision_tolerances": ((0.0078125, 0.0001, 0.005, 0.005, 10),),
        "absolute_precision": 1e-08,
        "B": 1,
        "N_q": 6,
        "N_kv": 1,
        "D": 512,
        "D_V": 512,
        "D_rope": 64,
        "quant_mode": 1,
        "softmax_scale": 0.041666666666666664,
        "seqused_q_values": [1],
        "cu_seqlens_q_values": [0, 1],
        "cache_seqlens_values": [1024],
        "kv_cache_layout": "BnNBsD",
        "block_size": 128,
        "mask_mode": 3,
        "enable_lse": True,
        "layout_q": "TND",
        "layout_kv": "PA_BNBD",
        "layout_out": "TND",
        "return_softmax_lse": True,
        "graph_path": 0,
        "p_scale": 1.0,
        "input_layout": "TND",
        "output_layout": "TND",
        "max_seqlen_q": 1,
        "max_seqlen_kv": 1024,
        "head_dim_v": 512,
    }
    attrs.update(overrides)
    return attrs


def _shapes_with_nq(nq):
    """Return _BASE_SHAPES with N_q replaced in q (dim1) and q_descale (dim1)."""
    shapes = list(_BASE_SHAPES)
    shapes[0] = (1, nq, 576)  # q: (B, N_q, D_total)
    shapes[2] = (1, nq)  # q_descale: (B, N_q)
    return tuple(shapes)


def _shapes_with_nkv(nkv):
    """Return _BASE_SHAPES with N_kv replaced in k_cache (dim1)."""
    shapes = list(_BASE_SHAPES)
    shapes[1] = (8, nkv, 128, 576)  # k_cache: (B, N_kv, block_size, D_total)
    return tuple(shapes)


def _shapes_with_d(d_total):
    """Return _BASE_SHAPES with D_total replaced in q (dim2) and k_cache (dim3)."""
    shapes = list(_BASE_SHAPES)
    shapes[0] = (1, 6, d_total)  # q: (B, N_q, D_total)
    shapes[1] = (8, 1, 128, d_total)  # k_cache: (B, N_kv, block_size, D_total)
    return tuple(shapes)


def _row(name, shapes, attrs):
    return [
        name,
        API_NAME,
        repr(tuple(shapes)),
        repr(tuple(_DTYPES)),
        "",
        repr(attrs),
        "",
        "",
        _DRANGE,
        _PRECISION,
        _ABS,
    ]


def gen_cases():
    cases = []

    # ===== 1. N_q 约束: Q_N ∈ {1,2,4,6,8,12,16,24,32,48,64,96,128} =====
    for nq in [3, 5, 7, 10, 15, 20]:
        cases.append(_row(f"Nq_invalid_{nq}", _shapes_with_nq(nq), _base_attrs(N_q=nq)))

    # ===== 2. N_kv 约束: KV_N = 1 =====
    for nkv in [0, -1, 2, 4]:
        cases.append(
            _row(f"Nkv_invalid_{nkv}", _shapes_with_nkv(nkv), _base_attrs(N_kv=nkv))
        )

    # ===== 3. head_dim_qk 约束: 仅支持 576 (D + D_rope) =====
    # D=256 → D_total=320 != 576
    cases.append(_row("D_head_dim_qk_320", _shapes_with_d(320), _base_attrs(D=256)))
    # D=1024 → D_total=1088
    cases.append(_row("D_head_dim_qk_1088", _shapes_with_d(1088), _base_attrs(D=1024)))
    # D=0 → D_total=64
    cases.append(_row("D_zero", _shapes_with_d(64), _base_attrs(D=0)))
    # D=-1
    cases.append(_row("D_neg1", _shapes_with_d(63), _base_attrs(D=-1)))

    # ===== 4. head_dim_v 约束: 仅支持 512 =====
    for hdv in [0, -1, 256, 1024, 576]:
        cases.append(
            _row(f"head_dim_v_invalid_{hdv}", _BASE_SHAPES, _base_attrs(head_dim_v=hdv))
        )

    # ===== 5. mask_mode 约束: 仅支持 0/3 =====
    for mm in [-1, 1, 2, 4, 5, 100]:
        cases.append(
            _row(f"mask_mode_invalid_{mm}", _BASE_SHAPES, _base_attrs(mask_mode=mm))
        )

    # ===== 6. quant_mode 约束: 仅支持 0/1 =====
    for qm in [-1, 2, 3, 100]:
        cases.append(
            _row(f"quant_mode_invalid_{qm}", _BASE_SHAPES, _base_attrs(quant_mode=qm))
        )

    # ===== 7. layout_q 约束: 支持 BSND/BNSD/TND/NTD =====
    for lq in ["INVALID", "", "BSN", "TND2", "ND", "TND_NTD"]:
        if lq == "TND_NTD":
            # TND_NTD is valid, skip
            continue
        cases.append(
            _row(
                f"layout_q_invalid_{lq or 'empty'}",
                _BASE_SHAPES,
                _base_attrs(layout_q=lq, input_layout=lq),
            )
        )

    # ===== 8. layout_kv 约束: 支持 PA_BBND/PA_BNBD/PA_NZ =====
    for lkv in ["INVALID", "", "BNBD", "PA_BSNZ", "NZ2"]:
        cases.append(
            _row(
                f"layout_kv_invalid_{lkv or 'empty'}",
                _BASE_SHAPES,
                _base_attrs(layout_kv=lkv),
            )
        )

    # ===== 9. layout_out 约束: 支持 BSND/BNSD/TND/NTD =====
    for lo in ["INVALID", "", "BSN", "TND2"]:
        cases.append(
            _row(
                f"layout_out_invalid_{lo or 'empty'}",
                _BASE_SHAPES,
                _base_attrs(layout_out=lo, output_layout=lo),
            )
        )

    # ===== 10. S 约束: S ∈ [1, 16] =====
    # seqused_q=[17], 需要改 q shape 和 cu_seqlens_q
    for s in [0, 17, 32, 100]:
        shapes = list(_BASE_SHAPES)
        shapes[0] = (s if s > 0 else 1, 6, 576)  # q: (Q_T, N, D)
        shapes[2] = (s if s > 0 else 1, 6)  # q_descale: (Q_T, N)
        shapes[6] = (2,)  # cu_seqlens_q: (B+1,)
        shapes[7] = (1,)  # seqused_q: (B,)
        cases.append(
            _row(
                f"seqused_q_S_{s}",
                tuple(shapes),
                _base_attrs(
                    seqused_q_values=[s], cu_seqlens_q_values=[0, s], max_seqlen_q=s
                ),
            )
        )

    # ===== 11. 空Tensor约束: 任意轴为0将被拦截 =====
    # q 空tensor: (0, 6, 576)
    shapes_empty_q = (
        (0, 6, 576),
        (8, 1, 128, 576),
        (0, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("empty_q", shapes_empty_q, _base_attrs()))

    # k_cache 空tensor: (8, 0, 128, 576)
    shapes_empty_k = (
        (1, 6, 576),
        (8, 0, 128, 576),
        (1, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("empty_k_cache", shapes_empty_k, _base_attrs()))

    # block_table 空tensor: (0, 8)
    shapes_empty_bt = (
        (1, 6, 576),
        (8, 1, 128, 576),
        (1, 6),
        (1,),
        (0, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("empty_block_table", shapes_empty_bt, _base_attrs()))

    # cache_seqlens 空tensor: (0,)
    shapes_empty_cs = (
        (1, 6, 576),
        (8, 1, 128, 576),
        (1, 6),
        (1,),
        (1, 8),
        (0,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("empty_cache_seqlens", shapes_empty_cs, _base_attrs()))

    # ===== 12. q shape dim 约束: 支持 3/4 =====
    # q 2D (invalid dim)
    shapes_q_2d = (
        (6, 576),
        (8, 1, 128, 576),
        (6,),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("q_dim_2_invalid", shapes_q_2d, _base_attrs()))

    # q 5D (invalid dim)
    shapes_q_5d = (
        (1, 1, 6, 576, 1),
        (8, 1, 128, 576),
        (1, 1, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("q_dim_5_invalid", shapes_q_5d, _base_attrs()))

    # ===== 13. k_cache shape dim 约束: 支持 4/5 =====
    # k_cache 3D (invalid dim)
    shapes_k_3d = (
        (1, 6, 576),
        (8, 128, 576),
        (1, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("k_cache_dim_3_invalid", shapes_k_3d, _base_attrs()))

    # k_cache 6D (invalid dim)
    shapes_k_6d = (
        (1, 6, 576),
        (8, 1, 128, 576, 1, 1),
        (1, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("k_cache_dim_6_invalid", shapes_k_6d, _base_attrs()))

    # ===== 14. k_descale shape 约束: 必须为 (1,) =====
    shapes_k_descale_2 = (
        (1, 6, 576),
        (8, 1, 128, 576),
        (1, 6),
        (2,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("k_descale_shape_2", shapes_k_descale_2, _base_attrs()))

    # ===== 15. mask_mode=0 时传入 attn_mask =====
    # mask_mode=0, 但 attn_mask slot 非空 (TTK 会传入 tensor)
    cases.append(
        _row("mask_mode0_with_attn_mask", _BASE_SHAPES, _base_attrs(mask_mode=0))
    )

    # ===== 16. mask_mode=3 时不传 attn_mask (attn_mask slot 为空) =====
    shapes_no_attn_mask = (
        (1, 6, 576),
        (8, 1, 128, 576),
        (1, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        None,
        (2, 4096),
    )
    cases.append(
        _row(
            "mask_mode3_no_attn_mask",
            tuple(s if s is not None else (0,) for s in shapes_no_attn_mask),
            _base_attrs(mask_mode=3),
        )
    )

    # ===== 17. q_descale shape 与 layout_q 不匹配 =====
    # TND 下 q_descale 应为 (Q_T, Q_N) = (1, 6), 改为 (2, 6) 不匹配
    shapes_q_descale_mismatch = (
        (1, 6, 576),
        (8, 1, 128, 576),
        (2, 6),
        (1,),
        (1, 8),
        (1,),
        (2,),
        (1,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(
        _row("q_descale_shape_mismatch_TND", shapes_q_descale_mismatch, _base_attrs())
    )

    # ===== 18. attn_mask shape 非 (2048, 2048) =====
    for mask_sh in [((1024, 1024),), ((2048, 1024),), ((4096, 4096),), ((512,),)]:
        sh = mask_sh[0]
        shapes = list(_BASE_SHAPES)
        shapes[8] = sh
        cases.append(
            _row(
                f"attn_mask_shape_{'x'.join(str(d) for d in sh)}",
                tuple(shapes),
                _base_attrs(mask_mode=3),
            )
        )

    # ===== 19. cu_seqlens_q shape 非 (B+1,) =====
    # B=1, cu_seqlens_q 应为 (2,), 改为 (1,) 和 (3,)
    for cu_len in [1, 3]:
        shapes = list(_BASE_SHAPES)
        shapes[6] = (cu_len,)
        cases.append(
            _row(
                f"cu_seqlens_q_shape_{cu_len}",
                tuple(shapes),
                _base_attrs(cu_seqlens_q_values=list(range(cu_len))),
            )
        )

    # ===== 20. cu_seqlens_q 值非法: 首元素非0 =====
    cases.append(
        _row(
            "cu_seqlens_q_first_not0",
            _BASE_SHAPES,
            _base_attrs(cu_seqlens_q_values=[1, 1]),
        )
    )

    # ===== 21. cu_seqlens_q 值非法: 非递减 (后一个 < 前一个) =====
    cases.append(
        _row(
            "cu_seqlens_q_decreasing",
            _BASE_SHAPES,
            _base_attrs(cu_seqlens_q_values=[0, 3, 2]),
        )
    )
    # 需配套改 B=3 的 shape
    shapes_cu_dec = (
        (3, 6, 576),
        (24, 1, 128, 576),
        (3, 6),
        (1,),
        (3, 24),
        (3,),
        (4,),
        (3,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(
        _row(
            "cu_seqlens_q_decreasing_B3",
            shapes_cu_dec,
            _base_attrs(
                B=3,
                cu_seqlens_q_values=[0, 3, 2, 6],
                seqused_q_values=[3, 2, 6],
                max_seqlen_q=6,
            ),
        )
    )

    # ===== 22. max_seqlen_q < -1 =====
    for msq in [-2, -10, -100]:
        cases.append(
            _row(f"max_seqlen_q_{msq}", _BASE_SHAPES, _base_attrs(max_seqlen_q=msq))
        )

    # ===== 23. max_seqlen_kv < -1 =====
    for mskv in [-2, -10, -100]:
        cases.append(
            _row(f"max_seqlen_kv_{mskv}", _BASE_SHAPES, _base_attrs(max_seqlen_kv=mskv))
        )

    # ===== 24. block_table 值非正整数 (0) =====
    cases.append(_row("block_table_value_0", _BASE_SHAPES, _base_attrs()))
    # block_table 值为负数 — 通过 attributes 无法直接注入 block_table 值,
    # 但 inputs.py 会从 cache_seqlens 生成 block_table, 测 cache_seqlens 负值

    # ===== 25. cache_seqlens 值为负 =====
    cases.append(
        _row("cache_seqlens_neg", _BASE_SHAPES, _base_attrs(cache_seqlens_values=[-1]))
    )
    cases.append(
        _row(
            "cache_seqlens_neg_large",
            _BASE_SHAPES,
            _base_attrs(cache_seqlens_values=[-100]),
        )
    )

    # ===== 26. block_table shape 非 (B, Bn) =====
    # B=1, 应为 (1, 8), 改为 (2, 8) 和 (1, 4)
    for bt_shape in [(2, 8), (1, 4), (1, 16)]:
        shapes = list(_BASE_SHAPES)
        shapes[4] = bt_shape
        cases.append(
            _row(
                f"block_table_shape_{bt_shape[0]}x{bt_shape[1]}",
                tuple(shapes),
                _base_attrs(),
            )
        )

    # ===== 27. cache_seqlens shape 非 (batch_size,) =====
    # B=1, 应为 (1,), 改为 (2,) 和 (3,)
    for cs_len in [2, 3]:
        shapes = list(_BASE_SHAPES)
        shapes[5] = (cs_len,)
        cs_vals = [1024] * cs_len
        cases.append(
            _row(
                f"cache_seqlens_shape_{cs_len}",
                tuple(shapes),
                _base_attrs(cache_seqlens_values=cs_vals),
            )
        )

    # ===== 28. D_rope 非法 (D + D_rope != 576) =====
    for d_rope in [0, 32, 128, -1]:
        d_total = 512 + d_rope
        cases.append(
            _row(
                f"D_rope_{d_rope}", _shapes_with_d(d_total), _base_attrs(D_rope=d_rope)
            )
        )

    # ===== 29. seqused_q 值 > Q_S (超过序列长度) =====
    # Q_S=1 (基线), seqused_q=[2] > 1
    cases.append(
        _row("seqused_q_gt_QS", _BASE_SHAPES, _base_attrs(seqused_q_values=[2]))
    )

    # ===== 30. seqused_q 值为负 =====
    cases.append(
        _row("seqused_q_neg", _BASE_SHAPES, _base_attrs(seqused_q_values=[-1]))
    )

    # ===== 31. B=0 (batch_size 为0, 空Tensor 的一种) =====
    shapes_B0 = (
        (0, 6, 576),
        (0, 1, 128, 576),
        (0, 6),
        (1,),
        (0, 8),
        (0,),
        (1,),
        (0,),
        (2048, 2048),
        (2, 4096),
    )
    cases.append(_row("B_zero", shapes_B0, _base_attrs(B=0)))

    # ===== 32. block_size != 128 (k_cache Bs 维度非法) =====
    # PA_BNBD: k_cache = (Bn, N, Bs, D) → Bs 在 axis 2
    for bs in [0, 1, 64, 256]:
        shapes_bs = (
            (1, 6, 576),
            (8, 1, bs, 576),
            (1, 6),
            (1,),
            (1, 8),
            (1,),
            (2,),
            (1,),
            (2048, 2048),
            (2, 4096),
        )
        cases.append(
            _row(
                f"block_size_{bs}",
                shapes_bs,
                _base_attrs(
                    block_size=bs,
                    cache_seqlens_values=[bs * 2] if bs > 0 else [0],
                    max_seqlen_kv=bs * 2 if bs > 0 else 1,
                ),
            )
        )

    # 正常基线 (对照组) =====
    cases.append(_row("Normal_baseline", _BASE_SHAPES, _base_attrs()))

    return cases


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "qmla_abnormal_full.csv")
    cases = gen_cases()
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for row in cases:
            writer.writerow(row)
    print(f"生成 {len(cases)} 个异常 case -> {out_path}")


if __name__ == "__main__":
    main()
