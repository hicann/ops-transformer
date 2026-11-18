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
"""把 qfa_mxfp8 pytest paramset 转成符合 ttk e2e 规范的 CSV。

输出契约与 tests/assets/excel_to_csv.py 保持一致:
  - api_name = torch.ops.cann_ops_transformer.quant_flash_attn (spec.py 注册名)
  - 11 列:
      testcase_name / api_name / tensor_view_shapes / tensor_dtypes /
      tensor_formats / attributes / output_tensor_indexes / golden_api /
      input_data_ranges / precision_tolerances / absolute_precision
  - tensor_view_shapes 共 18 个 slot, 与算子 schema 位置参数对齐:
      0  q                 1  k                 2  v
      3  q_descale         4  k_descale         5  v_descale
      6  block_table       7  p_scale           8  cu_seqlens_q
      9  cu_seqlens_kv     10 seqused_q         11 seqused_kv
      12 sinks              13 attn_mask         14 metadata(动态 -1,-1)
      15 v_tail             16 block_table_tail  17 seqused_v_tail
  - metadata 固定写 (-1,-1): ttk 视作动态槽位, 由 assets/impl/npu_preprocess.py
    在 H2D 后调用 quant_flash_attn_metadata 物化并回填主算子。

形状按 pytest golden (common/quant_flash_attn_golden.py) 的 generate_data +
prepare_npu_inputs 产出的算子入参布局纯 Python 推导, 无需 torch/NPU。

支持 layout: TND(varlen) / BNSD / BSND(含 decode 的 BN2SGD q_scale)。
BNSD/BSND 场景算子的 cu_seqlens 不入参, 序列长度由 seqused 或
batch_size + max_seqlen 推导 (对齐 ttk 上仅给 batch_size 的用例)。

仅支持 MXFP8 (quant_mode=1)。

用法:
  python3 pytest_paramset_to_csv.py [--paramset quant_flash_attn_paramset_func_rdv]
                                    [--output qfa_mxfp8.csv]
                                    [--dry-run]

命名与 excel_to_pytest_paramset.py 对齐: 两者构成 Excel → pytest paramset →
ttk CSV 的转换链。
"""

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

API_NAME = "torch.ops.cann_ops_transformer.quant_flash_attn"

# 与 golden / kernel 常量对齐
GROUP_SIZE = 32  # MXFP8 量化分组
VSCALE_PACK = GROUP_SIZE * 2  # V scale 偶奇行 packing 后每 block 覆盖的行数
ATTN_MASK_SHAPE = (2048, 2048)  # golden _build_causal_mask 固定形状
METADATA_DYNAMIC_SHAPE = (-1, -1)  # 动态槽位: ttk 传 None, npu_preprocess 物化
DRANGE_PLACEHOLDER = (0, 1)
DRANGE_REAL = (-1, 1)

D_FP8 = "float8_e4m3fn"
D_E8M0 = "float8_e8m0"
D_BF16 = "bfloat16"
D_FP32 = "float32"
D_INT8 = "int8"
D_INT32 = "int32"

# 每个 slot 的 dtype (None 槽位仍保留 dtype, 与 excel_to_csv 一致)
_DTYPES = [
    D_FP8,  # 0  q
    D_FP8,  # 1  k
    D_FP8,  # 2  v
    D_E8M0,  # 3  q_descale
    D_E8M0,  # 4  k_descale
    D_E8M0,  # 5  v_descale
    D_INT32,  # 6  block_table
    D_FP32,  # 7  p_scale
    D_INT32,  # 8  cu_seqlens_q
    D_INT32,  # 9  cu_seqlens_kv
    D_INT32,  # 10 seqused_q
    D_INT32,  # 11 seqused_kv
    D_FP32,  # 12 sinks
    D_INT8,  # 13 attn_mask
    D_INT32,  # 14 metadata
    D_BF16,  # 15 v_tail
    D_INT32,  # 16 block_table_tail
    D_INT32,  # 17 seqused_v_tail
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


def _derive_seqused(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _num_groups(head_dim):
    """Q/K per-token-group scale 组数, 与 get_mxfp8_per_token_group_quant_scale 一致:
    D 先按 64 对齐再按 32 分组 (D=72 → 4 组)。"""
    dim4_align = _ceil_div(head_dim, 64) * 64
    return _ceil_div(dim4_align, GROUP_SIZE)


def _vscale_tnd_T(act_list):
    """非 PA V scale 转 TND 后的第 0 维 Tv: 每 batch ceil(S/32) 行按偶奇打包成一半。"""
    total = 0
    for act in act_list:
        sg = _ceil_div(act, GROUP_SIZE)
        sg_padded = sg + (sg % 2)
        total += sg_padded // 2
    return total


def _vscale_bnsd_S_out(max_s):
    """非 PA V scale 在 BNSD/BSND 下 S 维偶奇打包后的行数。"""
    sg = _ceil_div(max_s, GROUP_SIZE)
    return (sg + (sg % 2)) // 2


def _pa_kv_shape(total_blocks, n_kv, head_dim, block_size, kv_layout):
    if kv_layout == "PA_BNBD":
        return (total_blocks, n_kv, block_size, head_dim)
    if kv_layout == "PA_BBND":
        return (total_blocks, block_size, n_kv, head_dim)
    if kv_layout == "PA_NZ":
        return (total_blocks, n_kv, head_dim // 32, block_size, 32)
    raise ValueError(f"unsupported PA kv_layout: {kv_layout}")


def _pa_kscale_shape(total_blocks, n_kv, num_groups, block_size, kv_layout):
    half = num_groups // 2
    if kv_layout == "PA_BNBD":
        return (total_blocks, n_kv, block_size, half, 2)
    if kv_layout == "PA_BBND":
        return (total_blocks, block_size, n_kv, half, 2)
    if kv_layout == "PA_NZ":
        return (total_blocks, n_kv, block_size // 16, half, 16, 2)
    raise ValueError(f"unsupported PA kv_layout: {kv_layout}")


def _pa_vscale_shape(total_blocks_v, n_kv, head_dim, pack_block_size, kv_layout):
    if kv_layout == "PA_BNBD":
        return (total_blocks_v, n_kv, pack_block_size, head_dim, 2)
    if kv_layout == "PA_BBND":
        return (total_blocks_v, pack_block_size, n_kv, head_dim, 2)
    if kv_layout == "PA_NZ":
        return (total_blocks_v, n_kv, head_dim // 16, pack_block_size, 16, 2)
    raise ValueError(f"unsupported PA kv_layout: {kv_layout}")


def _qscale_shape(total_q, n_q, n_kv, num_groups, q_scale_layout):
    """Q descale shape (q_scale_layout: TND / N2TGD)。"""
    if q_scale_layout == "TND":
        return (total_q, n_q, num_groups // 2, 2)
    if q_scale_layout == "N2TGD":
        if n_kv <= 0 or n_q % n_kv != 0:
            raise ValueError(
                f"N_q must be divisible by N_kv, got N_q={n_q}, N_kv={n_kv}"
            )
        return (n_kv, total_q, n_q // n_kv, num_groups // 2, 2)
    raise ValueError(f"unsupported q_scale_layout: {q_scale_layout}")


def _qscale_bnsd_shape(B, n_q, n_kv, max_s, num_groups, q_scale_layout):
    """非 PA BNSD/BSND 的 Q descale shape (BNSD / BSND / BN2SGD)。"""
    dg_half = num_groups // 2
    if q_scale_layout == "BNSD":
        return (B, n_q, max_s, dg_half, 2)
    if q_scale_layout == "BSND":
        return (B, max_s, n_q, dg_half, 2)
    if q_scale_layout == "BN2SGD":
        if n_kv <= 0 or n_q % n_kv != 0:
            raise ValueError(
                f"N_q must be divisible by N_kv, got N_q={n_q}, N_kv={n_kv}"
            )
        return (B, n_kv, max_s, n_q // n_kv, dg_half, 2)
    raise ValueError(f"unsupported q_scale_layout for BNSD/BSND: {q_scale_layout}")


def _derive_case(case):
    """返回 (shapes, data_ranges, attrs)，dtypes 用模块常量 _DTYPES。"""
    quant_mode = case.get("quant_mode", 1)
    quant_mode = 1 if quant_mode is None else int(quant_mode)
    if quant_mode != 1:
        raise ValueError(
            f"pytest_paramset_to_csv only supports MXFP8 (quant_mode=1), got {quant_mode}"
        )

    B = int(case["B"])
    N_q = int(case["N_q"])
    N_kv = int(case["N_kv"])
    D = int(case["D"])
    enable_pa = bool(case["enable_pa"])
    kv_layout = str(case["kv_cache_layout"])
    block_size = int(case.get("block_size") or 0)
    mask_mode = int(case.get("mask_mode") or 0)
    q_scale_layout = str(case.get("q_scale_layout") or "TND")
    enable_v_tail = bool(case.get("enable_v_tail", False))
    return_lse = bool(case.get("enable_lse", False))

    p_scale = case.get("p_scale")
    p_scale = 1.0 if p_scale is None else float(p_scale)
    softmax_scale = case.get("softmax_scale")
    softmax_scale = (
        (1.0 / math.sqrt(D)) if softmax_scale is None else float(softmax_scale)
    )

    cu_q = case.get("cu_seqlens_q")
    cu_kv = case.get("cu_seqlens_kv")
    seq_q = case.get("seqused_q")
    seq_kv = case.get("seqused_kv")

    # 非 TND (BNSD/BSND) 场景算子的 cu_seqlens 不入参 (见 golden _call_npu_fa_op),
    # 因此 CSV 侧同步置空; 序列长度改由 seqused 或 batch_size 推导。
    input_layout = str(case.get("input_layout") or "TND").upper()
    if input_layout not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"unsupported input_layout for csv export: {input_layout}")
    is_varlen = input_layout == "TND"
    cu_q = cu_q if is_varlen else None
    cu_kv = cu_kv if is_varlen else None

    batch_size = case.get("batch_size")
    batch_size = int(batch_size) if batch_size is not None else None

    max_sq = case.get("max_seqlen_q")
    max_skv = case.get("max_seqlen_kv")

    act_q = list(seq_q) if seq_q is not None else _derive_seqused(cu_q)
    act_kv = list(seq_kv) if seq_kv is not None else _derive_seqused(cu_kv)
    # BNSD/BSND decode: 只给 batch_size 属性, 每 batch 长度即为 max_seqlen
    if not act_q and batch_size and max_sq is not None and int(max_sq) > 0:
        act_q = [int(max_sq)] * batch_size
    if not act_kv and batch_size and max_skv is not None and int(max_skv) > 0:
        act_kv = [int(max_skv)] * batch_size
    if not act_q or not act_kv:
        raise ValueError(
            "need cu_seqlens_* / seqused_* / batch_size+max_seqlen_* "
            "to derive sequence lengths"
        )

    max_sq = max(act_q) if (max_sq is None or int(max_sq) < 0) else int(max_sq)
    max_skv = max(act_kv) if (max_skv is None or int(max_skv) < 0) else int(max_skv)

    total_q = cu_q[-1] if cu_q else sum(act_q)
    total_kv = cu_kv[-1] if cu_kv else sum(act_kv)
    num_groups = _num_groups(D)
    # 生成符合 tiling 校验的合法组合 (与 golden resolve_q_scale_layout 的约束一致):
    # layout_q=TND -> {TND,N2TGD}; layout_q=BNSD -> {BNSD}; layout_q=BSND -> {BSND,BN2SGD}
    if input_layout == "BNSD":
        q_scale_layout = "BNSD"
    elif input_layout == "BSND":
        q_scale_layout = "BN2SGD" if q_scale_layout in ("N2TGD", "BN2SGD") else "BSND"

    if kv_layout == "PA_NZ" and enable_v_tail and D % 16 != 0:
        raise ValueError(
            f"PA_NZ + D={D} + v_tail 结构性不成立: NZ 16 列分形要求 D%16==0"
        )

    if input_layout == "BNSD":
        q_shape = (B, N_q, max_sq, D)
        kv_shape = (B, N_kv, max_skv, D)
        q_descale_shape = _qscale_bnsd_shape(
            B, N_q, N_kv, max_sq, num_groups, q_scale_layout
        )
        k_descale_shape = (B, N_kv, max_skv, num_groups // 2, 2)
        v_descale_shape = (B, N_kv, _vscale_bnsd_S_out(max_skv), D, 2)
    elif input_layout == "BSND":
        q_shape = (B, max_sq, N_q, D)
        kv_shape = (B, max_skv, N_kv, D)
        q_descale_shape = _qscale_bnsd_shape(
            B, N_q, N_kv, max_sq, num_groups, q_scale_layout
        )
        k_descale_shape = (B, max_skv, N_kv, num_groups // 2, 2)
        v_descale_shape = (B, _vscale_bnsd_S_out(max_skv), N_kv, D, 2)
    else:
        q_shape = (total_q, N_q, D)
        kv_shape = (total_kv, N_kv, D)
        q_descale_shape = _qscale_shape(total_q, N_q, N_kv, num_groups, q_scale_layout)
        k_descale_shape = (total_kv, N_kv, num_groups // 2, 2)
        v_descale_shape = (_vscale_tnd_T(act_kv), N_kv, D, 2)

    if enable_pa:
        if not kv_layout.startswith("PA_"):
            raise ValueError(f"enable_pa=True but kv_cache_layout={kv_layout!r}")
        blocks = sum(_ceil_div(a, block_size) for a in act_kv)
        max_blocks = max(_ceil_div(a, block_size) for a in act_kv)
        k_shape = _pa_kv_shape(blocks, N_kv, D, block_size, kv_layout)
        v_shape = k_shape
        k_descale_shape = _pa_kscale_shape(
            blocks, N_kv, num_groups, block_size, kv_layout
        )
        vscale_pack_block = _ceil_div(block_size, VSCALE_PACK)
        vscale_blocks = sum(
            _ceil_div(_ceil_div(a, VSCALE_PACK), vscale_pack_block) for a in act_kv
        )
        v_descale_shape = _pa_vscale_shape(
            vscale_blocks, N_kv, D, vscale_pack_block, kv_layout
        )
        block_table_shape = (B, max_blocks)
        layout_kv = kv_layout
        block_size_attr = block_size
    else:
        k_shape = kv_shape
        v_shape = kv_shape
        block_table_shape = None
        layout_kv = input_layout
        block_size_attr = 0

    if enable_v_tail:
        if enable_pa:
            tail_capacity = block_size
            if kv_layout == "PA_BBND":
                v_tail_shape = (B, tail_capacity, N_kv, D)
            elif kv_layout == "PA_NZ":
                v_tail_shape = (B, N_kv, D // 16, tail_capacity, 16)
            else:
                v_tail_shape = (B, N_kv, tail_capacity, D)
            block_table_tail_shape = (B, max(1, _ceil_div(64, block_size)))
        else:
            v_tail_shape = (B, N_kv, 64, D)  # TND 尾窗固定容量 64 直存
            block_table_tail_shape = None
        seqused_v_tail_shape = (B,)
    else:
        v_tail_shape = (0,)
        block_table_tail_shape = (0,)
        seqused_v_tail_shape = (0,)

    shapes = [
        q_shape,
        k_shape,
        v_shape,
        q_descale_shape,
        k_descale_shape,
        v_descale_shape,
        block_table_shape,
        (1,),  # p_scale 标量
        (len(cu_q),) if cu_q else None,
        (len(cu_kv),) if cu_kv else None,
        (B,) if seq_q is not None else None,
        (B,) if seq_kv is not None else None,
        None,  # sinks
        ATTN_MASK_SHAPE if mask_mode != 0 else None,
        METADATA_DYNAMIC_SHAPE,
        v_tail_shape,
        block_table_tail_shape,
        seqused_v_tail_shape,
    ]

    data_ranges = [
        DRANGE_REAL,  # q
        DRANGE_REAL,  # k
        DRANGE_REAL,  # v
    ] + [DRANGE_PLACEHOLDER] * (len(shapes) - 3)

    attrs = {
        "precision_tolerances": _PRECISION_TOLERANCES,
        "absolute_precision": _ABSOLUTE_PRECISION,
        "p_scale_value": p_scale,
        "cu_seqlens_q_values": list(cu_q) if cu_q else None,
        "cu_seqlens_kv_values": list(cu_kv) if cu_kv else None,
        "seqused_q_values": list(seq_q) if seq_q is not None else None,
        "seqused_kv_values": list(seq_kv) if seq_kv is not None else None,
        "attn_mask_shape": ATTN_MASK_SHAPE if mask_mode != 0 else None,
        "attn_mask_dtype": D_INT8,
        "quant_mode": quant_mode,
        "softmax_scale": softmax_scale,
        "mask_mode": mask_mode,
        "win_left": -1,
        "win_right": -1,
        "enable_v_tail": 1 if enable_v_tail else None,
        "max_seqlen_q": int(case["max_seqlen_q"])
        if case.get("max_seqlen_q") is not None
        else -1,
        "max_seqlen_kv": int(case["max_seqlen_kv"])
        if case.get("max_seqlen_kv") is not None
        else -1,
        "layout_q": input_layout,
        "layout_q_descale": q_scale_layout,
        "layout_kv": layout_kv,
        "layout_out": input_layout,
        "return_softmax_lse": return_lse,
        "N_q": N_q,
        "N_kv": N_kv,
        "D": D,
        "head_dim_v": None,
        "batch_size": batch_size,
        "enable_pa": enable_pa,
        "kv_cache_layout": kv_layout,
        "q_scale_layout": q_scale_layout,
        "block_size": block_size_attr,
    }
    return shapes, data_ranges, attrs


def case_to_csv_row(case):
    """把一个 case dict 转成 CSV 行 (list, 由 csv.writer 负责转义)。"""
    shapes, data_ranges, attrs = _derive_case(case)
    return [
        case["name"],
        API_NAME,
        repr(tuple(shapes)),
        repr(tuple(_DTYPES)),
        "",  # tensor_formats 留空, 框架默认 ('ND',)
        repr(attrs),
        "",  # output_tensor_indexes 留空 (wrapper 输出是返回值)
        "",  # golden_api 留空
        repr(tuple(data_ranges)),
        repr(_PRECISION_TOLERANCES),
        repr(_ABSOLUTE_PRECISION),
    ]


def _load_cases(paramset_name):
    paramset_mod = __import__(paramset_name)
    if getattr(paramset_mod, "CASES", None) is not None:
        cases = paramset_mod.CASES
    elif hasattr(paramset_mod, "TEST_PARAMS"):
        from quant_flash_attn_paramset_common import expand_paramset_to_cases

        cases = expand_paramset_to_cases(paramset_mod.TEST_PARAMS)
    else:
        raise ValueError(
            f"paramset module '{paramset_name}' has neither CASES nor TEST_PARAMS"
        )
    return cases, getattr(paramset_mod, "SKIP_CASES", set())


def main():
    parser = argparse.ArgumentParser(description="qfa_mxfp8 paramset → TTK e2e CSV")
    parser.add_argument(
        "--paramset",
        default="quant_flash_attn_paramset_func_rdv",
        help="paramset 模块名(不含.py)",
    )
    parser.add_argument("--output", default="qfa_mxfp8.csv", help="输出 CSV 路径")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印生成结果, 不写文件"
    )
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

    print(f"生成 {written} 个 case (跳过 {skipped} 个) → {out_path}")


if __name__ == "__main__":
    main()
