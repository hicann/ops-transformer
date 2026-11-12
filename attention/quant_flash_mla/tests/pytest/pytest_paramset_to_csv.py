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
"""把 qfmla pytest paramset 转成符合 ttk e2e 规范的 CSV。

输出契约与 tests/assets/excel_to_csv.py 保持一致:
  - api_name = torch.ops.cann_ops_transformer.quant_flash_mla
  - 11 列: testcase_name / api_name / tensor_view_shapes / tensor_dtypes /
    tensor_formats / attributes / output_tensor_indexes / golden_api /
    input_data_ranges / precision_tolerances / absolute_precision
  - 10 个 tensor slot:
      0 q  1 k_cache  2 q_descale  3 k_descale  4 block_table
      5 cache_seqlens  6 cu_seqlens_q  7 seqused_q  8 attn_mask  9 metadata
  - metadata 默认写 (-1, -1): 动态槽位，由 npu_preprocess 在主算子调用前生成并回填。
    写死 (2, N) 会被当成普通输入按 data range 随机填充，导致 kernel 读到非法调度表。

用法:
  python3 pytest_paramset_to_csv.py [--paramset quant_flash_mla_paramset_debug]
                                   [--output qfmla_pytest.csv]
                                   [--dry-run]
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

API_NAME = "torch.ops.cann_ops_transformer.quant_flash_mla"

ATTN_MASK_SHAPE = (2048, 2048)
METADATA_DEFAULT_SHAPE = (-1, -1)
DRANGE_PLACEHOLDER = (0, 1)
DRANGE_REAL = (-1, 1)

D_FP8 = "float8_e4m3fn"
D_FP32 = "float32"
D_INT8 = "int8"
D_INT32 = "int32"

_DTYPES = [
    D_FP8,  # 0  q
    D_FP8,  # 1  k_cache
    D_FP32,  # 2  q_descale
    D_FP32,  # 3  k_descale
    D_INT32,  # 4  block_table
    D_INT32,  # 5  cache_seqlens
    D_INT32,  # 6  cu_seqlens_q
    D_INT32,  # 7  seqused_q
    D_INT8,  # 8  attn_mask
    D_INT32,  # 9  metadata
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

_PRECISION_TOLERANCES = ((0.0078125, 0.0001, 0.005, 0.005, 10),)
_ABSOLUTE_PRECISION = 1e-8


def _ceil_div(a, b):
    return -(-int(a) // int(b))


def _pa_kv_shape(total_blocks, n_kv, head_dim, block_size, kv_layout):
    """PA K cache shape，与 golden bnsd_to_k_cache 一致。"""
    if kv_layout == "PA_BNBD" or kv_layout == "BnNBsD":
        return (total_blocks, n_kv, block_size, head_dim)
    if kv_layout == "PA_BBND" or kv_layout == "BnBsND":
        return (total_blocks, block_size, n_kv, head_dim)
    if kv_layout == "PA_NZ" or kv_layout == "NZ":
        # FP8 (1 byte): d0=32, d1=head_dim//32
        # BF16 (2 byte): d0=16, d1=head_dim//16
        d0 = 32  # FP8
        d1 = head_dim // d0
        return (total_blocks, n_kv, d1, block_size, d0)
    raise ValueError(f"unsupported PA kv_layout: {kv_layout}")


def _normalize_layout(layout_kv):
    """layout_kv → golden kv_cache_layout 名。"""
    mapping = {
        "PA_BNBD": "BnNBsD",
        "PA_BBND": "BnBsND",
        "PA_NZ": "NZ",
    }
    return mapping.get(layout_kv, layout_kv)


def _to_op_layout(layout_kv):
    """layout_kv → 算子接口期望的 PA_ 前缀格式。"""
    mapping = {
        "BnNBsD": "PA_BNBD",
        "BnBsND": "PA_BBND",
        "NZ": "PA_NZ",
    }
    return mapping.get(layout_kv, layout_kv)


def _require(case, key, *alias_keys):
    """读取必填字段 (支持别名), 缺配或显式为 None 时直接报错, 不静默补值。

    转换器只负责透传 case 声明的值; 默认值应由 paramset 的
    TEST_PARAMS_DEFAULTS 显式声明, 避免掩盖 paramset 漏配。
    """
    for name in (key, *alias_keys):
        if case.get(name) is not None:
            return case[name]
    names = " / ".join((key, *alias_keys))
    raise ValueError(f"[PYTEST2CSV] case '{case.get('name')}' must provide {names}")


def _derive_case(case):
    """返回 (shapes, data_ranges, attrs)。"""
    quant_mode = int(_require(case, "quant_mode"))

    B = int(case["B"])
    N_q = int(case["N_q"])
    N_kv = int(case["N_kv"])
    D = int(case["D"])  # 算子侧 QK 总 head dim
    D_V = int(_require(case, "D_V", "D_v"))
    D_total = D

    kv_layout_raw = str(_require(case, "kv_cache_layout"))
    kv_layout = _normalize_layout(kv_layout_raw)
    enable_pa = bool(_require(case, "enable_pa"))
    block_size = int(_require(case, "block_size"))
    mask_mode = int(_require(case, "mask_mode"))
    return_lse = bool(_require(case, "enable_lse"))

    p_scale = float(_require(case, "p_scale"))
    softmax_scale = float(_require(case, "softmax_scale", "scale_value"))

    cu_q = case.get("cu_seqlens_q")
    seq_q = case.get("seqused_q")
    cache_seqlens = case.get("cache_seqlens")

    # 推导 seqused_q
    if seq_q is not None:
        act_q = list(seq_q)
    elif cu_q is not None:
        act_q = [cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1)]
    else:
        act_q = None

    # cache_seqlens
    if cache_seqlens is not None:
        act_kv = list(cache_seqlens)
    else:
        act_kv = None

    max_sq = case.get("max_seqlen_q")
    if max_sq is None or int(max_sq) < 0:
        max_sq = max(act_q) if act_q else 0
    else:
        max_sq = int(max_sq)

    max_skv = case.get("max_seqlen_kv")
    if max_skv is None or int(max_skv) < 0:
        max_skv = max(act_kv) if act_kv else 0
    else:
        max_skv = int(max_skv)

    # q 行数必须与 cu_seqlens_q 的末端一致:
    #   TND  → T = sum(seqused_q) (变长拼接，非 padding)
    #   稠密 → T = B * max_seqlen_q (padded 步长)，cu 也用 padded 步长
    # 混用会让 golden 拿到的 q 行数与 dense 展开后的 B*S 对不上而拒算。
    layout_is_tnd = str(case.get("layout", "TND")).upper().startswith("TND")
    Q_S = max_sq if max_sq > 0 else (max(act_q) if act_q else 1)
    if layout_is_tnd and act_q:
        T_rows = sum(act_q)
    else:
        T_rows = B * Q_S

    # q_shape: TND (T, N_q, D_total)
    q_shape = (T_rows, N_q, D_total)
    q_descale_shape = (T_rows, N_q)

    # K cache
    if enable_pa and act_kv:
        blocks = sum(_ceil_div(a, block_size) for a in act_kv)
        max_blocks = max(_ceil_div(a, block_size) for a in act_kv)
        k_cache_shape = _pa_kv_shape(blocks, N_kv, D_total, block_size, kv_layout)
        block_table_shape = (B, max_blocks)
    else:
        k_cache_shape = (0,)
        block_table_shape = (0,)

    k_descale_shape = (1,)

    # cu_seqlens_q
    if cu_q is not None:
        cu_seqlens_q_shape = (len(cu_q),)
    else:
        if layout_is_tnd and act_q:
            # TND 的 cu 必须是 packed 前缀和，与 q 的 T=sum(seqused_q) 对齐；
            # 变长 case 若按 padded 步长生成会得到 cu[-1] > T，golden 会直接拒绝。
            cu_q = [0]
            for length in act_q:
                cu_q.append(cu_q[-1] + int(length))
        else:
            # 稠密布局不使用 cu，保持 padded 步长不变
            cu_q = [0] + [Q_S * (i + 1) for i in range(B)]
        cu_seqlens_q_shape = (len(cu_q),)

    # seqused_q
    if seq_q is not None:
        seqused_q_shape = (B,)
    elif act_q is not None:
        seqused_q_shape = (B,)
    else:
        seqused_q_shape = (B,)

    # cache_seqlens
    if cache_seqlens is not None:
        cache_seqlens_shape = (B,)
    else:
        cache_seqlens_shape = (B,)

    # attn_mask
    attn_mask_shape = ATTN_MASK_SHAPE if mask_mode != 0 else None

    # metadata
    metadata_shape = METADATA_DEFAULT_SHAPE

    shapes = [
        q_shape,  # 0 q
        k_cache_shape,  # 1 k_cache
        q_descale_shape,  # 2 q_descale
        k_descale_shape,  # 3 k_descale
        block_table_shape,  # 4 block_table
        cache_seqlens_shape,  # 5 cache_seqlens
        cu_seqlens_q_shape,  # 6 cu_seqlens_q
        seqused_q_shape,  # 7 seqused_q
        attn_mask_shape,  # 8 attn_mask
        metadata_shape,  # 9 metadata
    ]

    data_ranges = [
        DRANGE_REAL,  # q
        DRANGE_REAL,  # k_cache
        DRANGE_PLACEHOLDER,  # q_descale
        DRANGE_PLACEHOLDER,  # k_descale
        DRANGE_PLACEHOLDER,  # block_table
        DRANGE_PLACEHOLDER,  # cache_seqlens
        DRANGE_PLACEHOLDER,  # cu_seqlens_q
        DRANGE_PLACEHOLDER,  # seqused_q
        DRANGE_PLACEHOLDER,  # attn_mask
        DRANGE_PLACEHOLDER,  # metadata
    ]

    attrs = {
        "precision_tolerances": _PRECISION_TOLERANCES,
        "absolute_precision": _ABSOLUTE_PRECISION,
        "B": B,
        "N_q": N_q,
        "N_kv": N_kv,
        "D": D,
        "D_V": D_V,
        "quant_mode": quant_mode,
        "softmax_scale": softmax_scale,
        "seqused_q_values": list(seq_q)
        if seq_q is not None
        else (list(act_q) if act_q else None),
        "cu_seqlens_q_values": list(cu_q) if cu_q else None,
        "cache_seqlens_values": list(cache_seqlens) if cache_seqlens else None,
        "kv_cache_layout": kv_layout,
        "block_size": block_size,
        "mask_mode": mask_mode,
        "enable_lse": return_lse,
        "layout_q": "TND",
        "layout_kv": _to_op_layout(kv_layout_raw),
        "layout_out": "TND",
        "return_softmax_lse": return_lse,
        "graph_path": 0,
        "p_scale": p_scale,
        "input_layout": "TND",
        "output_layout": "TND",
        "max_seqlen_q": max_sq,
        "max_seqlen_kv": max_skv,
        "head_dim_v": D_V,
    }
    return shapes, data_ranges, attrs


def _case_dtypes(case):
    """HIF8(quant_mode=0) 的 q/k_cache 以 uint8 承载 hif8 编码(算子侧要求 uint8),
    其余 dtype 同 _DTYPES。"""
    dtypes = list(_DTYPES)
    if int(case.get("quant_mode", 1)) == 0:
        dtypes[0] = dtypes[1] = "uint8"
    return tuple(dtypes)


def case_to_csv_row(case):
    shapes, data_ranges, attrs = _derive_case(case)
    return [
        case["name"],
        API_NAME,
        repr(tuple(shapes)),
        repr(_case_dtypes(case)),
        "",
        repr(attrs),
        "",
        "",
        repr(tuple(data_ranges)),
        repr(_PRECISION_TOLERANCES),
        repr(_ABSOLUTE_PRECISION),
    ]


def _load_cases(paramset_name):
    paramset_mod = __import__(paramset_name)
    if getattr(paramset_mod, "CASES", None) is not None:
        cases = paramset_mod.CASES
    elif hasattr(paramset_mod, "TEST_PARAMS"):
        from quant_flash_mla_paramset_common import expand_paramset_to_cases

        cases = expand_paramset_to_cases(paramset_mod.TEST_PARAMS)
    else:
        raise ValueError(
            f"paramset module '{paramset_name}' has neither CASES nor TEST_PARAMS"
        )
    return cases, getattr(paramset_mod, "SKIP_CASES", set())


def main():
    parser = argparse.ArgumentParser(description="qfmla paramset -> TTK e2e CSV")
    parser.add_argument(
        "--paramset",
        default="quant_flash_mla_paramset_debug",
        help="paramset 模块名(不含.py)",
    )
    parser.add_argument("--output", default="qfmla_pytest.csv", help="输出 CSV 路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    args = parser.parse_args()

    cases, skip = _load_cases(args.paramset)
    out_path = (
        args.output if os.path.isabs(args.output) else os.path.join(_HERE, args.output)
    )
    written = 0
    skipped = 0
    rows = []
    for case in cases:
        if case["name"] in skip:
            skipped += 1
            continue
        try:
            rows.append(case_to_csv_row(case))
        except ValueError as exc:
            print(f"[pytest_paramset_to_csv] SKIP {case['name']}: {exc}")
            skipped += 1
            continue
        written += 1

    if args.dry_run:
        writer = csv.writer(sys.stdout)
        writer.writerow(_HEADER)
        for row in rows:
            writer.writerow(row)
        print(
            f"\n生成 {written} 个 case (跳过 {skipped} 个) (dry-run)", file=sys.stderr
        )
        return

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for row in rows:
            writer.writerow(row)
    print(f"生成 {written} 个 case (跳过 {skipped} 个) -> {out_path}")


if __name__ == "__main__":
    main()
