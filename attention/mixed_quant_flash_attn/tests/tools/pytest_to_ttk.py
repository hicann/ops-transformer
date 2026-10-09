# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest_to_ttk.py — 将 pytest TestCases 字典文件转换为 ttk CSV 用例文件

================================================================================
功能说明
================================================================================
读取 pytest 用例文件（含 TestCases = {...}），解析每个用例字典，
生成 ttk E2E CSV 用例文件。

pytest 用例字段是列表包装（如 "B": [64]），本脚本取 [0] 解包。
支持字段: B, N1, N2, S1, S2, D, layout_q, layout_kv, layout_attn_out,
          block_size, q_dtype, quant_compute_mode, mask_mode, softmax_scale,
          max_seqlen_q, max_seqlen_kv, seqused_q, seqused_kv, cu_seqlens_q,
          win_left, win_right, return_softmax_lse, q_range, kv_range,
          attn_mask, sinks

跳过字段: *_override（非法 shape / dtype 边界用例，不适合 TTK CSV 模式）

CSV 中所有 shape 都是物理存储 shape（fp4 pack 后最后一维减半：PA_NZ →32，PA_BBND/PA_BNBD → D//2）。
block_size 用 pytest block_size 字段（各布局通用）。支持 PA_NZ / PA_BBND / PA_BNBD。

================================================================================
用法
================================================================================
    python3 pytest_to_ttk.py --pytest functional_stc.py --csv cases.csv
    python3 pytest_to_ttk.py --pytest functional_stc.py --csv cases.csv --prefix "stc_"
    python3 pytest_to_ttk.py --pytest functional_stc.py --csv cases.csv --skip-invalid
    python3 pytest_to_ttk.py --pytest functional_stc.py --csv cases.csv --dry-run

================================================================================
tensor 顺序（12 个，与算子签名一致）
================================================================================
  0  q             — (B, S1, N1, D) 或 (B, N1, S1, D) 或 (S1, N1, D) (TND)
  1  k             — PA 物理形状（PA_NZ / PA_BBND / PA_BNBD，fp4 pack 后减半）
  2  v             — PA 物理形状
  3  k_descale     — PA 物理形状
  4  v_descale     — PA 物理形状
  5  block_table   — (B, max_blocks_per_batch)，int32
  6  cu_seqlens_q  — None（非 TND）或 (B+1,)，int32
  7  seqused_q     — (B,) 或 None，int32
  8  seqused_kv    — (B,) 或 None，int32
  9  sinks         — None 或 (8,)，int32（暂不支持，固定 None）
  10 attn_mask     — (2048, 2048) 或 None，int8
  11 metadata      — (16384,)，int32（固定）
"""

import argparse
import ast
import math
import os
import sys


# ============================================================================
# 默认配置
# ============================================================================
DEFAULT_TTK_PREFIX = ""
DEFAULT_ATTN_MASK_SHAPE = (2048, 2048)
DEFAULT_METADATA_SHAPE = (16384,)
API_NAME = "mqfa_ttk_ops.call_npu"

# pytest dtype -> ttk CSV torch dtype 名
CSV_DTYPE_MAP = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp4_e2m1": "uint8",
    "fp4_e1m2": "uint8",
    "fp8_e8m0": "float8_e8m0fnu",
    "hifp4_scale": "float32",
    "fp32_e6m2e1e16": "float32",
    "int8": "int8",
    "int32": "int32",
    "fp32": "float32",
    "float32": "float32",
}

CSV_SCALE_DTYPE = {1: "float8_e8m0", 2: "float32"}
CSV_SCALE_RANGE = {1: (0.01, 10.0), 2: (125.0, 129.0)}

# 跳过的边界/异常用例字段（这些用例构造非法输入，TTK CSV 模式无法表达）
SKIP_FIELDS = {
    "q_shape_override",
    "k_descale_shape_override",
    "block_table_override",
    "kv_scale_dtype_override",
    "attn_mask_shape_override",
}


# ============================================================================
# shape 计算（与 xlsx_to_testcase.py 一致，对齐 doc 公式）
# ============================================================================
def gen_pa_shape(
    num_blocks, n, block_size, d, quant_compute_mode, layout_kv, is_key, dtype
):
    """
    按公式计算 PA 布局的物理存储 shape（对齐 pytest gen_data.generate_input_shape）。

    PA_NZ  kv(fp4 物理): (num_blocks, n, d//64, block_size, 32)
            scale(qcm=1) K: (num_blocks, n, block_size//16, d//64, 16, 2)  V: (…, d//16, block_size//64, 16, 2)
            scale(qcm=2) K: (num_blocks, n, block_size//16, d//64, 16)      V: (…, d//16, block_size//64, 16)
    PA_BBND kv(fp4 物理): (num_blocks, block_size, n, d//2)      # (blockNum, S, N, D) pack 后 D//2
            scale       K: (num_blocks, block_size, n, d//group)  V: (num_blocks, n, d, block_size//group)
    PA_BNBD kv(fp4 物理): (num_blocks, n, block_size, d//2)      # (blockNum, N, S, D) pack 后 D//2
            scale       K: (num_blocks, n, block_size, d//group)  V: (num_blocks, n, d, block_size//group)
    """
    kv = dtype in ("fp4_e2m1", "fp4_e1m2")
    group = 32 if quant_compute_mode == 1 else 64

    if layout_kv == "PA_NZ":
        if kv:
            return (num_blocks, n, d // 64, block_size, 32)
        if quant_compute_mode == 1:
            return (
                (num_blocks, n, block_size // 16, d // 64, 16, 2)
                if is_key
                else (num_blocks, n, d // 16, block_size // 64, 16, 2)
            )
        return (
            (num_blocks, n, block_size // 16, d // 64, 16)
            if is_key
            else (num_blocks, n, d // 16, block_size // 64, 16)
        )

    if layout_kv == "PA_BBND":  # (blockNum, S, N, D)
        if kv:
            return (num_blocks, block_size, n, d // 2)
        return (
            (num_blocks, block_size, n, d // group)
            if is_key
            else (num_blocks, n, d, block_size // group)
        )

    if layout_kv == "PA_BNBD":  # (blockNum, N, S, D)
        if kv:
            return (num_blocks, n, block_size, d // 2)
        return (
            (num_blocks, n, block_size, d // group)
            if is_key
            else (num_blocks, n, d, block_size // group)
        )

    raise ValueError(f"不支持的 layout_kv: {layout_kv}")


def calc_block_info(seqused_kv, s2, block_size):
    """num_blocks = sum(ceil(seqused_kv[i]/block_size)), max_blocks_per_batch = ceil(max/blocks_size)。"""
    if seqused_kv:
        num_blocks = sum(math.ceil(x / block_size) for x in seqused_kv)
        max_blocks_per_batch = math.ceil(max(seqused_kv) / block_size)
    else:
        num_blocks = math.ceil(s2 / block_size)
        max_blocks_per_batch = num_blocks
    return num_blocks, max_blocks_per_batch


def resolve_max_seqlen(val, seq_list, fallback):
    """-1 时自动取 max(seq_list) 或 fallback。"""
    if val is not None and val >= 0:
        return val
    if seq_list:
        return max(seq_list)
    return fallback


# ============================================================================
# pytest 用例解析
# ============================================================================
def load_pytest_cases(pytest_path):
    """加载 pytest 文件，返回 TestCases 字典。"""
    ns = {}
    with open(pytest_path, encoding="utf-8") as f:
        exec(f.read(), ns)
    if "TestCases" not in ns:
        raise ValueError(f"{pytest_path} 中未找到 TestCases 变量")
    return ns["TestCases"]


def _val(case, key, default=None):
    """取 pytest 用例字段值（列表包装 [0]，标量直接返回）。"""
    v = case.get(key, default)
    if v is None:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return v


def has_override_fields(case):
    """检测是否含边界/异常 override 字段。"""
    return bool(SKIP_FIELDS & set(case.keys()))


def parse_pytest_case(name, case):
    """将 pytest 用例字典解析为标准化用例字典。"""
    B = _val(case, "B", 1)
    N1 = _val(case, "N1")
    N2 = _val(case, "N2", N1)
    S1 = _val(case, "S1", 1)
    S2 = _val(case, "S2", S1)
    D = _val(case, "D", 128)
    block_size = _val(case, "block_size", 128)
    layout_q = _val(case, "layout_q", "BNSD")
    layout_kv = _val(case, "layout_kv", layout_q)
    layout_out = _val(case, "layout_attn_out", layout_q)
    q_dtype = _val(case, "q_dtype", "bf16")
    quant_compute_mode = _val(case, "quant_compute_mode", None) or _val(
        case, "quant_mode", 1
    )
    mask_mode = _val(case, "mask_mode", 0)
    softmax_scale = _val(case, "softmax_scale", 1.0 / (D**0.5))

    seqused_q = _val(case, "seqused_q", None)
    seqused_kv = _val(case, "seqused_kv", None)
    cu_seqlens_q = _val(case, "cu_seqlens_q", None)

    max_seqlen_q = _val(case, "max_seqlen_q", -1)
    max_seqlen_kv = _val(case, "max_seqlen_kv", -1)

    win_left = _val(case, "win_left", -1)
    win_right = _val(case, "win_right", -1)
    return_softmax_lse = _val(case, "return_softmax_lse", False)

    q_range = _val(case, "q_range", (-10.0, 10.0))
    kv_range = _val(case, "kv_range", (-10.0, 10.0))

    # seqused_q/seqused_kv 可能是 [list]，_val 已取 [0]
    if isinstance(seqused_q, list) and seqused_q and isinstance(seqused_q[0], list):
        seqused_q = seqused_q[0]
    if isinstance(seqused_kv, list) and seqused_kv and isinstance(seqused_kv[0], list):
        seqused_kv = seqused_kv[0]
    if (
        isinstance(cu_seqlens_q, list)
        and cu_seqlens_q
        and isinstance(cu_seqlens_q[0], list)
    ):
        cu_seqlens_q = cu_seqlens_q[0]

    return {
        "name": name,
        "B": B,
        "N1": N1,
        "N2": N2,
        "S1": S1,
        "S2": S2,
        "D": D,
        "block_size": block_size,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "q_dtype": q_dtype,
        "quant_compute_mode": quant_compute_mode,
        "mask_mode": mask_mode,
        "softmax_scale": softmax_scale,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "win_left": win_left,
        "win_right": win_right,
        "return_softmax_lse": return_softmax_lse,
        "q_range": q_range,
        "kv_range": kv_range,
    }


# ============================================================================
# CSV 行生成
# ============================================================================
def build_csv_row(c, prefix):
    """构建单行 CSV。PA_NZ 时 block_size 从 k/v shape 第4维确定（= block_size 字段）。"""
    B, N1, N2, S1, S2, D = c["B"], c["N1"], c["N2"], c["S1"], c["S2"], c["D"]
    bs = c["block_size"]
    qcm = c["quant_compute_mode"]
    layout_q = c["layout_q"]
    layout_kv = c["layout_kv"]
    seqused_q = c["seqused_q"]
    seqused_kv = c["seqused_kv"]

    # PA 布局（block_size 用 pytest block_size 字段，各布局通用）
    is_pa = "PA" in layout_kv
    if not is_pa:
        raise NotImplementedError(f"非 PA 布局暂不支持: {layout_kv}")

    # PA 布局: 算子 batch_size 从 seqused_q/seqused_kv 长度推导（doc L168），
    # pytest B 字段可能与 seqused 列表长度不一致（如 mxfp4_stc 用例 B=1 但 seqused 128 值）。
    # 用 seqused 长度覆盖 B，保证 q/block_table/seqused shape 一致。
    if seqused_q:
        B = len(seqused_q)
    elif seqused_kv:
        B = len(seqused_kv)

    num_blocks, max_blocks_per_batch = calc_block_info(seqused_kv, S2, bs)

    max_seqlen_q = resolve_max_seqlen(c["max_seqlen_q"], seqused_q, S1)
    max_seqlen_kv = resolve_max_seqlen(c["max_seqlen_kv"], seqused_kv, S2)

    # q shape
    if layout_q == "BNSD":
        q_shape = (B, N1, S1, D)
    elif layout_q == "BSND":
        q_shape = (B, S1, N1, D)
    elif layout_q == "TND":
        q_shape = (S1, N1, D)
    else:
        q_shape = (B, S1, N1, D)

    # k/v/descale shapes (按布局公式计算)
    kv_dtype = "fp4_e2m1" if qcm == 1 else "fp4_e1m2"
    scale_dtype = "fp8_e8m0" if qcm == 1 else "hifp4_scale"
    k_shape = gen_pa_shape(num_blocks, N2, bs, D, qcm, layout_kv, True, kv_dtype)
    v_shape = gen_pa_shape(num_blocks, N2, bs, D, qcm, layout_kv, False, kv_dtype)
    k_scale_shape = gen_pa_shape(
        num_blocks, N2, bs, D, qcm, layout_kv, True, scale_dtype
    )
    v_scale_shape = gen_pa_shape(
        num_blocks, N2, bs, D, qcm, layout_kv, False, scale_dtype
    )

    block_table_shape = (B, max_blocks_per_batch)

    # cu_seqlens_q: TND 时输出
    if layout_q == "TND":
        cu_seqlens_q_shape = (
            (len(c["cu_seqlens_q"]),) if c["cu_seqlens_q"] else (B + 1,)
        )
    else:
        cu_seqlens_q_shape = None

    # PA 布局算子 tiling 要求 seqused_kv 必传；seqused_q 在非 TND 时也建议传
    # （算子 seq_len_checker: "seqused_kv must be provided when PagedAttention is enabled"）
    # pytest 无 seqused_kv 时，仍输出 (B,) shape tensor（TTK 随机生成，customize_inputs 不覆盖），
    # 并在 attrs 写入 [S2]*B 作为默认值，让 override_tensors_from_attributes 填充。
    # 注意: seqused shape 维度取自 seqused 列表长度（pytest B 可能与列表长度解耦）。
    if is_pa:
        sq_len = len(seqused_q) if seqused_q else B
        skv_len = len(seqused_kv) if seqused_kv else B
        seqused_q_shape = (sq_len,)
        seqused_kv_shape = (skv_len,)
        if seqused_q is None:
            seqused_q = [S1] * sq_len
        if seqused_kv is None:
            seqused_kv = [S2] * skv_len
    else:
        seqused_q_shape = (len(seqused_q),) if seqused_q is not None else None
        seqused_kv_shape = (len(seqused_kv),) if seqused_kv is not None else None

    sinks_shape = None  # 暂不支持

    attn_mask_shape = DEFAULT_ATTN_MASK_SHAPE if c["mask_mode"] in (3, 4) else None
    metadata_shape = DEFAULT_METADATA_SHAPE

    view_shapes = [
        q_shape,
        k_shape,
        v_shape,
        k_scale_shape,
        v_scale_shape,
        block_table_shape,
        cu_seqlens_q_shape,
        seqused_q_shape,
        seqused_kv_shape,
        sinks_shape,
        attn_mask_shape,
        metadata_shape,
    ]

    q_csv_dtype = CSV_DTYPE_MAP.get(c["q_dtype"], "bfloat16")
    scale_csv_dtype = CSV_SCALE_DTYPE.get(qcm, "float8_e8m0fnu")
    dtypes = [
        q_csv_dtype,
        "uint8",
        "uint8",
        scale_csv_dtype,
        scale_csv_dtype,
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int8",
        "int32",
    ]

    attrs = {
        "quant_compute_mode": qcm,
        "batch_size": B,
        "num_heads_q": N1,
        "num_heads_kv": N2,
        "head_dim": D,
        "softmax_scale": c["softmax_scale"],
        "mask_mode": c["mask_mode"],
        "win_left": c["win_left"],
        "win_right": c["win_right"],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": c["layout_out"],
        "return_softmax_lse": c["return_softmax_lse"],
    }
    if seqused_q is not None:
        attrs["seqused_q"] = seqused_q
    if seqused_kv is not None:
        attrs["seqused_kv"] = seqused_kv
    if c["cu_seqlens_q"] is not None and layout_q == "TND":
        attrs["cu_seqlens_q"] = c["cu_seqlens_q"]

    q_range = c["q_range"] if c["q_range"] else (-10.0, 10.0)
    kv_range = c["kv_range"] if c["kv_range"] else (-10.0, 10.0)
    scale_range = CSV_SCALE_RANGE.get(qcm, (0.01, 10.0))
    block_table_range = (0, max(num_blocks - 1, 0))
    cu_seqlens_q_range = (0, S1) if cu_seqlens_q_shape is not None else (-1, -1)
    seqused_q_range = (0, S1) if seqused_q_shape is not None else (-1, -1)
    seqused_kv_range = (0, S2) if seqused_kv_shape is not None else (-1, -1)
    sinks_range = (0, 0)
    attn_mask_range = (0, 1) if attn_mask_shape is not None else (0, 0)
    metadata_range = (0, 0)

    ranges = [
        q_range,
        kv_range,
        kv_range,
        scale_range,
        scale_range,
        block_table_range,
        cu_seqlens_q_range,
        seqused_q_range,
        seqused_kv_range,
        sinks_range,
        attn_mask_range,
        metadata_range,
    ]

    testcase_name = prefix + c["name"]
    fields = [
        testcase_name,
        API_NAME,
        _fmt_shapes(view_shapes),
        _fmt_dtypes(dtypes),
        _fmt_storage_shapes(view_shapes),
        _fmt_attrs(attrs),
        _fmt_ranges(ranges),
    ]
    return ",".join(fields)


# ============================================================================
# 格式化函数
# ============================================================================
def _fmt_shape(s):
    if s is None:
        return "None"
    if len(s) == 1:
        return f"({s[0]},)"
    return "(" + ",".join(str(x) for x in s) + ")"


def _fmt_shapes(shapes):
    return '"(' + ",".join(_fmt_shape(s) for s in shapes) + ')"'


def _fmt_storage_shapes(view_shapes):
    parts = ["()" if s is None else _fmt_shape(s) for s in view_shapes]
    return '"(' + ",".join(parts) + ')"'


def _fmt_dtypes(dts):
    return "\"('" + "','".join(dts) + "')\""


def _fmt_ranges(rngs):
    parts = [f"({r[0]},{r[1]})" for r in rngs]
    return '"(' + ",".join(parts) + ')"'


def _fmt_attrs(a):
    items = []
    for k, v in a.items():
        if isinstance(v, str):
            items.append(f"'{k}':'{v}'")
        elif isinstance(v, bool):
            items.append(f"'{k}':{v}")
        elif isinstance(v, list):
            items.append(f"'{k}':{v}")
        else:
            items.append(f"'{k}':{v}")
    return '"{' + ",".join(items) + '}"'


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="将 pytest TestCases 文件转换为 ttk CSV 用例文件（纯文件转换，不依赖代码目录结构）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --pytest functional_stc.py --csv cases.csv
  %(prog)s --pytest functional_stc.py --csv cases.csv --prefix "stc_"
  %(prog)s --pytest functional_stc.py --csv cases.csv --skip-invalid
  %(prog)s --pytest functional_stc.py --csv cases.csv --dry-run
        """,
    )
    parser.add_argument(
        "--pytest", required=True, help="输入 pytest 用例文件路径 (含 TestCases 字典)"
    )
    parser.add_argument("--csv", required=True, help="输出 ttk CSV 文件路径")
    parser.add_argument(
        "--prefix",
        default=DEFAULT_TTK_PREFIX,
        help=f"ttk 用例名前缀 (默认: '{DEFAULT_TTK_PREFIX}')",
    )
    parser.add_argument("--dry-run", action="store_true", help="只解析不写文件")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="跳过含 override 字段的边界/异常用例",
    )
    args = parser.parse_args()

    print(f"输入 pytest: {args.pytest}")
    print(f"输出 ttk CSV: {args.csv}")
    print(f"用例名前缀:  '{args.prefix}'")
    print()

    if not os.path.exists(args.pytest):
        print(f"ERROR: pytest 文件不存在: {args.pytest}", file=sys.stderr)
        sys.exit(1)

    testcases = load_pytest_cases(args.pytest)
    print(f"解析到 {len(testcases)} 个用例")

    parsed = []
    skipped = []
    for name, case in testcases.items():
        if has_override_fields(case):
            skipped.append(name)
            if args.skip_invalid:
                continue
        try:
            parsed.append(parse_pytest_case(name, case))
        except Exception as e:
            print(f"  [警告] 用例 {name} 解析失败: {e}", file=sys.stderr)
            skipped.append(name)

    if skipped:
        print(f"跳过 {len(skipped)} 个边界/异常用例:")
        for n in skipped:
            print(f"  {n}")
    print()

    print(f"有效用例 {len(parsed)} 个:")
    for c in parsed:
        print(
            f"  {c['name']}: B={c['B']} N1={c['N1']} N2={c['N2']} S1={c['S1']} "
            f"S2={c['S2']} bs={c['block_size']} qcm={c['quant_compute_mode']} "
            f"layout_q={c['layout_q']}"
        )
    print()

    if args.dry_run:
        print("--dry-run 模式，不写文件")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    csv_lines = [
        "testcase_name,api_name,tensor_view_shapes,tensor_dtypes,"
        "tensor_storage_shapes,attributes,input_data_ranges"
    ]
    for c in parsed:
        csv_lines.append(build_csv_row(c, args.prefix))
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"[ttk] 已生成 {len(parsed)} 个用例 -> {args.csv}")


if __name__ == "__main__":
    main()
