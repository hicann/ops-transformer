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
xlsx_to_testcase.py — 将用例 Excel 表（如 B250_readline.xlsx）转换为
                      pytest 用例文件 + ttk CSV 用例文件

================================================================================
重要说明
================================================================================
xlsx 中的 PA shape 列（k_shape / v_shape / k_descale_shape / v_descale_shape）
是算子接收的 PA 物理布局，支持 PA_NZ / PA_BBND / PA_BNBD:
  - 第1维 block_num 是预留 padding 值，本脚本保留原值（block_table id 取模复用避免越界）
  - block_size 按布局从 xlsx shape 中定位（PA_NZ→第4维, PA_BBND→第2维, PA_BNBD→第3维），
    无 xlsx shape 时退回 blocksize 列
  - k/v 最后一维是逻辑 fp4 元素数，CSV 中写 pack 后物理值（PA_NZ: 64→32; PA_BBND/PA_BNBD: D=128→64）

CSV 中所有 shape 都是 **物理存储 shape**（pack 后），不是逻辑 shape：
  - fp4 (e2m1/e1m2) kv: 最后一维按 pack 减半（uint8）
  - descale dtype uint8/float32 与物理 shape 对齐

tensor 顺序（12 个，与算子签名一致）:
  0  q             — (B, S1, N1, D) 或 (B, N1, S1, D)，取决于 layout_q
  1  k             — PA 物理形状（PA_NZ/PA_BBND/PA_BNBD）
  2  v             — PA 物理形状
  3  k_descale     — PA 物理形状
  4  v_descale     — PA 物理形状
  5  block_table   — (B, max_blocks_per_batch)，int32
  6  cu_seqlens_q  — None（非 TND 场景）或 (B+1,)，int32
  7  seqused_q     — (B,) 或 None，int32
  8  seqused_kv    — (B,) 或 None，int32
  9  sinks         — None，int32（暂不支持）
  10 attn_mask     — (2048, 2048) 或 None，int8（mask_mode in [3,4] 时输出）
  11 metadata      — (16384,)，int32（固定）

空 tensor 规则：
  - view_shapes  用 None 表示空 tensor
  - storage_shapes 用 () 表示空 tensor（TTK 解析约定）

================================================================================
用法
==============================================================================
    # 指定输入输出文件（推荐，不依赖目录结构）
    python3 xlsx_to_testcase.py --xlsx /path/to/input.xlsx \
        --pytest-out /path/to/functional.py --ttk-out /path/to/cases.csv

    # 只生成其中一个
    python3 xlsx_to_testcase.py --xlsx /path/to/input.xlsx \
        --ttk-out /path/to/cases.csv --ttk-only
    python3 xlsx_to_testcase.py --xlsx /path/to/input.xlsx \
        --pytest-out /path/to/functional.py --pytest-only

    # 其他选项
    python3 xlsx_to_testcase.py --dry-run                # 只解析不写文件
    python3 xlsx_to_testcase.py --prefix "rd_"           # 自定义 ttk 用例名前缀

================================================================================
xlsx 列说明（按列名查找，不依赖列位置）
================================================================================
必需列:
  Testcase_Name   用例名称
  batch_size      B     — batch size
  num_heads_q     N1    — Q 头数
  num_heads_kv    N2    — KV 头数
  Q_S             S1    — Q 序列长度
  KV_S            S2    — KV 序列长度
  head_dim        D     — 头维度（必须 128）
  blocksize             — PA 分块大小
  quant_compute_mode            — 1=MXFP4, 2=HiFP4
  mask_mode             — 0/3/4 等
  layout_q              — BSND / BNSD / TND
  layout_kv             — PA_NZ / PA_BBND / PA_BNBD
  q_dtype               — BF16 / FP16
  softmax_scale         — 缩放因子
  max_seqlen_q          — Q 最大序列长度
  max_seqlen_kv         — KV 最大序列长度（-1 表示自动取 max(seqused_kv)）

可选列:
  seqused_q_value       — Q 实际序列长度列表
  seqused_kv_value      — KV 实际序列长度列表
  cu_seqlens_q_value    — TND 场景的 cu_seqlens_q
  win_left / win_right  — 窗口注意力参数
  return_softmax_lse    — True/False
  q_datarange           — Q 数据范围
"""

import argparse
import ast
import math
import os
import sys

try:
    import openpyxl
except ImportError:
    print("ERROR: 需要安装 openpyxl，请运行: pip install openpyxl", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# 默认路径配置
# ============================================================================
DEFAULT_XLSX = "/home/l00949320/code/check_prof/B250_readline.xlsx"
DEFAULT_TTK_PREFIX = ""
DEFAULT_ATTN_MASK_SHAPE = (2048, 2048)
DEFAULT_METADATA_SHAPE = (16384,)


# ============================================================================
# dtype 映射表
# ============================================================================
# xlsx dtype -> pytest gen_data 内部 dtype
DTYPE_MAP = {
    "BF16": "bf16",
    "FP16": "fp16",
    "FP4_E2M1": "fp4_e2m1",
    "HIFLOAT4": "fp4_e1m2",
    "FP8_E8M0": "fp8_e8m0",
    "hifloat4_scale": "hifp4_scale",
    "INT8": "int8",
    "INT32": "int32",
    "FP32": "fp32",
}

# xlsx dtype -> ttk CSV torch dtype 名
CSV_DTYPE_MAP = {
    "BF16": "bfloat16",
    "FP16": "float16",
    "FP4_E2M1": "uint8",
    "HIFLOAT4": "uint8",
    "FP8_E8M0": "float8_e8m0fnu",
    "hifloat4_scale": "float32",
    "INT8": "int8",
    "INT32": "int32",
    "FP32": "float32",
}

# quant_compute_mode -> kv descale dtype（CSV）
CSV_SCALE_DTYPE = {1: "float8_e8m0", 2: "float32"}

# quant_compute_mode -> kv descale 数据范围（与 inputs.py _gen_*_scale 默认 low/high 一致）
CSV_SCALE_RANGE = {1: (0.01, 10.0), 2: (125.0, 129.0)}


# ============================================================================
# 工具函数
# ============================================================================
def parse_shape(s):
    """解析 "64,1,8,128" -> (64, 1, 8, 128)。空值返回 None。"""
    if s is None:
        return None
    return tuple(int(x.strip()) for x in str(s).split(",") if x.strip())


def parse_datarange(s):
    """解析 "0.001,0.01" -> (0.001, 0.01)。空值返回 None。"""
    if s is None:
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(parts) != 2:
        return None
    return (float(parts[0]), float(parts[1]))


def parse_int_list(s):
    """解析整型列表，支持换行/逗号分隔。空值返回 None。"""
    if s is None:
        return None
    s = str(s).replace("\n", ",")
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    return vals if vals else None


def parse_int(s, default=None):
    """安全转 int，失败返回 default。"""
    if s is None:
        return default
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def gen_pa_shape(
    num_blocks, n, block_size, d, quant_compute_mode, layout_kv, is_key, dtype
):
    """
    按公式计算 PA 布局的物理 tensor 存储形状（与 pytest gen_data.generate_input_shape 一致）。

    参数:
      num_blocks  — PA 分块总数
      n           — KV 头数 N2
      block_size  — PA 分块大小 S
      d           — 头维度
      quant_compute_mode  — 1=MXFP4, 2=HiFP4
      layout_kv   — PA_NZ / PA_BBND / PA_BNBD
      is_key      — True=K, False=V
      dtype       — 'fp4_e2m1'/'fp4_e1m2' (kv) 或 scale dtype

    返回:
      tuple — 物理存储形状（fp4 kv 最后一维已按 pack 减半）
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

    if layout_kv == "PA_BBND":  # (block_num, S, N, D)
        if kv:
            return (num_blocks, block_size, n, d // 2)
        return (
            (num_blocks, block_size, n, d // group)
            if is_key
            else (num_blocks, n, d, block_size // group)
        )

    if layout_kv == "PA_BNBD":  # (block_num, N, S, D)
        if kv:
            return (num_blocks, n, block_size, d // 2)
        return (
            (num_blocks, n, block_size, d // group)
            if is_key
            else (num_blocks, n, d, block_size // group)
        )

    raise ValueError(f"不支持的 layout_kv: {layout_kv}")


def _resolve_pa_shape(
    xlsx_shape,
    num_blocks,
    n,
    block_size,
    d,
    quant_compute_mode,
    layout_kv,
    is_key,
    dtype,
    testcase_name="",
    tensor_name="",
):
    """从 xlsx PA shape 构建最终 shape，或按公式计算。

    xlsx_shape: xlsx 中的 PA shape tuple，第1维是 block_num（padding，可信），其余维度可信。
                k/v fp4 最后一维是逻辑元素数（PA_NZ 为 64，PA_BBND/PA_BNBD 为 D=128），
                物理存储 pack 后需减半（32 / 64）。
    无 xlsx_shape 时用 gen_pa_shape 按公式算。

    xlsx 第1维 block_num 与 seqused_kv 计算值不一致时打警告，但仍用 xlsx 值。
    """
    # block_num: 优先用 xlsx 第1维（padding 预留值），否则用计算值
    xlsx_block_num = (
        xlsx_shape[0] if (xlsx_shape and len(xlsx_shape) >= 1) else num_blocks
    )
    if xlsx_block_num != num_blocks:
        print(
            f"[WARN] {testcase_name} {tensor_name}: xlsx block_num={xlsx_block_num} "
            f"与 seqused_kv 计算值={num_blocks} 不一致，采用 xlsx 值 {xlsx_block_num} "
            f"(block_table id 将取模复用)"
        )

    # descale（非 fp4 kv）: 按公式重建，保证 N2 等维度正确（xlsx 个别行 N2 写错过，如 1 vs 8）。
    # 仅保留 xlsx 第1维 block_num padding。
    if dtype not in ("fp4_e2m1", "fp4_e1m2"):
        return gen_pa_shape(
            xlsx_block_num,
            n,
            block_size,
            d,
            quant_compute_mode,
            layout_kv,
            is_key,
            dtype,
        )

    # k/v fp4: 优先用 xlsx 其余维度，最后一维物理 pack 减半（64→32 或 128→64）
    if not xlsx_shape or len(xlsx_shape) < 2:
        return gen_pa_shape(
            xlsx_block_num,
            n,
            block_size,
            d,
            quant_compute_mode,
            layout_kv,
            is_key,
            dtype,
        )
    shape = list(xlsx_shape)
    shape[0] = xlsx_block_num
    shape[-1] = shape[-1] // 2
    return tuple(shape)


def calc_block_info(seqused_kv, s2, block_size):
    """
    计算 PA 布局下的 block_num 与 max_blocks_per_batch。
    与 pytest gen_data.py:424-426 一致。
    """
    if seqused_kv:
        num_blocks = sum(math.ceil(x / block_size) for x in seqused_kv)
        max_blocks_per_batch = math.ceil(max(seqused_kv) / block_size)
    else:
        num_blocks = math.ceil(s2 / block_size)
        max_blocks_per_batch = num_blocks
    return num_blocks, max_blocks_per_batch


def resolve_max_seqlen_kv(max_seqlen_kv, seqused_kv, s2):
    """max_seqlen_kv=-1 时自动取 max(seqused_kv) 或 s2。"""
    if max_seqlen_kv is not None and max_seqlen_kv >= 0:
        return max_seqlen_kv
    if seqused_kv:
        return max(seqused_kv)
    return s2


def resolve_max_seqlen_q(max_seqlen_q, seqused_q, s1):
    """max_seqlen_q=-1 时自动取 max(seqused_q) 或 s1。"""
    if max_seqlen_q is not None and max_seqlen_q >= 0:
        return max_seqlen_q
    if seqused_q:
        return max(seqused_q)
    return s1


# ============================================================================
# xlsx 读取
# ============================================================================
class XlsxReader:
    """读取 xlsx 并解析为用例列表。按列名查找，不依赖列位置。"""

    REQUIRED_COLS = [
        "Testcase_Name",
        "batch_size",
        "num_heads_q",
        "num_heads_kv",
        "Q_S",
        "KV_S",
        "head_dim",
        "blocksize",
        "quant_compute_mode",
        "mask_mode",
        "layout_q",
        "layout_kv",
        "q_dtype",
        "softmax_scale",
        "max_seqlen_q",
        "max_seqlen_kv",
    ]

    def __init__(self, xlsx_path):
        self.xlsx_path = xlsx_path
        self.wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        self.ws = self.wb[self.wb.sheetnames[0]]
        self.headers = [c.value for c in self.ws[1]]
        self._validate_columns()

    def _validate_columns(self):
        missing = [c for c in self.REQUIRED_COLS if c not in self.headers]
        if missing:
            raise ValueError(f"xlsx 缺少必需列: {missing}")

    def get(self, row_idx, col_name):
        if col_name not in self.headers:
            return None
        col_idx = self.headers.index(col_name) + 1
        return self.ws.cell(row=row_idx, column=col_idx).value

    def parse_cases(self):
        cases = []
        for row_idx in range(2, self.ws.max_row + 1):
            name = self.get(row_idx, "Testcase_Name")
            if name is None:
                continue

            qm = parse_int(self.get(row_idx, "quant_compute_mode"))
            layout_q = self.get(row_idx, "layout_q")
            layout_kv = self.get(row_idx, "layout_kv")
            seqused_q = parse_int_list(self.get(row_idx, "seqused_q_value"))
            seqused_kv = parse_int_list(self.get(row_idx, "seqused_kv_value"))

            case = {
                "name": name,
                "row": row_idx,
                "B": parse_int(self.get(row_idx, "batch_size")),
                "N1": parse_int(self.get(row_idx, "num_heads_q")),
                "N2": parse_int(self.get(row_idx, "num_heads_kv")),
                "S1": parse_int(self.get(row_idx, "Q_S")),
                "S2": parse_int(self.get(row_idx, "KV_S")),
                "D": parse_int(self.get(row_idx, "head_dim")),
                "block_size": parse_int(self.get(row_idx, "blocksize")),
                "quant_compute_mode": qm,
                "mask_mode": parse_int(str(self.get(row_idx, "mask_mode"))),
                "win_left": parse_int(self.get(row_idx, "win_left"), -1),
                "win_right": parse_int(self.get(row_idx, "win_right"), -1),
                "layout_q": layout_q,
                "layout_kv": layout_kv,
                "layout_out": self.get(row_idx, "layout_out") or layout_q,
                "q_dtype": DTYPE_MAP.get(self.get(row_idx, "q_dtype"), "bf16"),
                "softmax_scale": float(self.get(row_idx, "softmax_scale")),
                "max_seqlen_q": parse_int(self.get(row_idx, "max_seqlen_q"), -1),
                "max_seqlen_kv": parse_int(self.get(row_idx, "max_seqlen_kv"), -1),
                "return_softmax_lse": str(self.get(row_idx, "return_softmax_lse"))
                == "True",
                "q_datarange": parse_datarange(self.get(row_idx, "q_datarange")),
                "seqused_q": seqused_q,
                "seqused_kv": seqused_kv,
                "cu_seqlens_q": parse_int_list(self.get(row_idx, "cu_seqlens_q_value")),
                # xlsx PA shape 列（第1维 block_num 是 padding，其余维度可信）
                "xlsx_k_shape": parse_shape(self.get(row_idx, "k_shape")),
                "xlsx_v_shape": parse_shape(self.get(row_idx, "v_shape")),
                "xlsx_k_descale_shape": parse_shape(
                    self.get(row_idx, "k_descale_shape")
                ),
                "xlsx_v_descale_shape": parse_shape(
                    self.get(row_idx, "v_descale_shape")
                ),
            }

            self._validate_case(case)
            cases.append(case)
        return cases

    def _validate_case(self, c):
        errors = []
        for field in (
            "B",
            "N1",
            "N2",
            "S1",
            "S2",
            "D",
            "block_size",
            "quant_compute_mode",
        ):
            if c[field] is None:
                errors.append(f"{field} 为空")
        if c["D"] is not None and c["D"] != 128:
            errors.append(f"D={c['D']} 不等于 128")
        if c["quant_compute_mode"] is not None and c["quant_compute_mode"] not in (
            1,
            2,
        ):
            errors.append(f"quant_compute_mode={c['quant_compute_mode']} 不在 [1, 2]")
        if c["layout_kv"] is not None and not c["layout_kv"].startswith("PA_"):
            errors.append(f"layout_kv={c['layout_kv']} 非 PA 布局")
        # seqused_kv 与 B 的一致性
        if c["seqused_kv"] and c["B"] is not None and len(c["seqused_kv"]) != c["B"]:
            errors.append(f"seqused_kv 长度 {len(c['seqused_kv'])} ≠ B={c['B']}")
        if c["seqused_q"] and c["B"] is not None and len(c["seqused_q"]) != c["B"]:
            errors.append(f"seqused_q 长度 {len(c['seqused_q'])} ≠ B={c['B']}")
        if errors:
            print(
                f"  [警告] 用例 {c['name']} 校验问题: {', '.join(errors)}",
                file=sys.stderr,
            )


# ============================================================================
# pytest 文件生成
# ============================================================================
def gen_pytest(cases, out_path):
    """
    生成 pytest 用例文件（TestCases 字典格式），与 functional_*.py 一致。
    """
    lines = ["TestCases = {"]
    for c in cases:
        lines.append(f'    "{c["name"]}": {{')
        lines.append(f'        "B": [{c["B"]}],')
        lines.append(f'        "N1": [{c["N1"]}],')
        lines.append(f'        "N2": [{c["N2"]}],')
        lines.append(f'        "S1": [{c["S1"]}],')
        lines.append(f'        "S2": [{c["S2"]}],')
        lines.append(f'        "D": [{c["D"]}],')
        lines.append(f'        "layout_q": ["{c["layout_q"]}"],')
        lines.append(f'        "layout_kv": ["{c["layout_kv"]}"],')
        lines.append(f'        "layout_attn_out": ["{c["layout_out"]}"],')
        lines.append(f'        "block_size": [{c["block_size"]}],')
        # block_num: 优先用 xlsx k_shape 第1维（PA 预留值），gen_data.py 会 max(传入, 计算) 兜底
        xlsx_k = c.get("xlsx_k_shape")
        if xlsx_k and len(xlsx_k) >= 1 and xlsx_k[0] > 0:
            lines.append(f'        "block_num": [{xlsx_k[0]}],')
        lines.append(f'        "q_dtype": ["{c["q_dtype"]}"],')
        lines.append(f'        "quant_compute_mode": [{c["quant_compute_mode"]}],')
        lines.append(f'        "mask_mode": [{c["mask_mode"]}],')

        if c["seqused_q"] is not None:
            lines.append(f'        "seqused_q": {[c["seqused_q"]]!r},')
        if c["seqused_kv"] is not None:
            lines.append(f'        "seqused_kv": {[c["seqused_kv"]]!r},')
        if c["cu_seqlens_q"] is not None:
            lines.append(f'        "cu_seqlens_q": {[c["cu_seqlens_q"]]!r},')

        lines.append(f'        "max_seqlen_q": [{c["max_seqlen_q"]}],')
        lines.append(f'        "max_seqlen_kv": [{c["max_seqlen_kv"]}],')
        lines.append(f'        "softmax_scale": {c["softmax_scale"]!r},')

        if c["win_left"] != -1:
            lines.append(f'        "win_left": [{c["win_left"]}],')
        if c["win_right"] != -1:
            lines.append(f'        "win_right": [{c["win_right"]}],')
        if c["return_softmax_lse"]:
            lines.append('        "return_softmax_lse": [True],')

        # attn_mask: mask_mode in [3,4] 时 pytest 自动生成 (2048,2048) causal mask
        # q_range / kv_range: pytest 默认 (-10, 10)
        if c["q_datarange"]:
            lines.append(f'        "q_range": [{c["q_datarange"]!r}],')
        lines.append('        "kv_range": [(-10.0, 10.0)],')
        lines.append("    },")
    lines.append("}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(cases)


# ============================================================================
# ttk CSV 文件生成
# ============================================================================
API_NAME = "mqfa_ttk_ops.call_npu"


def gen_ttk(cases, out_path, prefix=DEFAULT_TTK_PREFIX):
    """生成 ttk CSV 用例文件。"""
    csv_lines = [
        "testcase_name,api_name,tensor_view_shapes,tensor_dtypes,"
        "tensor_storage_shapes,attributes,input_data_ranges"
    ]
    for c in cases:
        csv_lines.append(_build_csv_row(c, prefix))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    return len(cases)


def _build_csv_row(c, prefix):
    """构建单行 CSV。

    xlsx k/v/descale shape 列是 PA shape，第4维=block_size 可信，第1维=block_num 是 padding。
    本脚本用 xlsx k_shape 第4维覆盖 blocksize 列，计算真实 block_num 替换第1维，
    k/v 最后一维 64(逻辑)→32(物理 pack)。
    """
    B, N1, N2, S1, S2, D = c["B"], c["N1"], c["N2"], c["S1"], c["S2"], c["D"]
    qm = c["quant_compute_mode"]
    layout_q = c["layout_q"]
    layout_kv = c["layout_kv"]
    seqused_q = c["seqused_q"]
    seqused_kv = c["seqused_kv"]

    # --- block_size: 优先从 xlsx k_shape 取（按布局定位 S 维），否则用 blocksize 列 ---
    xlsx_k = c.get("xlsx_k_shape")
    if xlsx_k and len(xlsx_k) >= 1:
        if layout_kv == "PA_BBND" and len(xlsx_k) >= 2:
            bs = xlsx_k[1]
        elif layout_kv == "PA_BNBD" and len(xlsx_k) >= 3:
            bs = xlsx_k[2]
        elif layout_kv == "PA_NZ" and len(xlsx_k) >= 4:
            bs = xlsx_k[3]
        else:
            bs = c["block_size"]
    else:
        bs = c["block_size"]

    # --- 计算 num_blocks / max_blocks_per_batch ---
    num_blocks, max_blocks_per_batch = calc_block_info(seqused_kv, S2, bs)

    # --- 解析 max_seqlen ---
    max_seqlen_q = resolve_max_seqlen_q(c["max_seqlen_q"], seqused_q, S1)
    max_seqlen_kv = resolve_max_seqlen_kv(c["max_seqlen_kv"], seqused_kv, S2)

    # --- q shape ---
    if layout_q == "BNSD":
        q_shape = (B, N1, S1, D)
    elif layout_q == "BSND":
        q_shape = (B, S1, N1, D)
    elif layout_q == "TND":
        q_shape = (S1, N1, D)
    else:
        q_shape = (B, S1, N1, D)

    # --- k/v/descale shapes ---
    # 用 xlsx PA shape（k/v fp4 最后一维 pack 减半，descale 直接用物理 shape），
    # block_num 用 xlsx 值，不一致时警告。无 xlsx shape 时按公式算。
    kv_dtype = "fp4_e2m1" if qm == 1 else "fp4_e1m2"
    tc_name = c.get("name", "")
    k_shape = _resolve_pa_shape(
        c.get("xlsx_k_shape"),
        num_blocks,
        N2,
        bs,
        D,
        qm,
        layout_kv,
        True,
        kv_dtype,
        tc_name,
        "k",
    )
    v_shape = _resolve_pa_shape(
        c.get("xlsx_v_shape"),
        num_blocks,
        N2,
        bs,
        D,
        qm,
        layout_kv,
        False,
        kv_dtype,
        tc_name,
        "v",
    )
    scale_dtype = "fp8_e8m0" if qm == 1 else "hifp4_scale"
    k_scale_shape = _resolve_pa_shape(
        c.get("xlsx_k_descale_shape"),
        num_blocks,
        N2,
        bs,
        D,
        qm,
        layout_kv,
        True,
        scale_dtype,
        tc_name,
        "k_descale",
    )
    v_scale_shape = _resolve_pa_shape(
        c.get("xlsx_v_descale_shape"),
        num_blocks,
        N2,
        bs,
        D,
        qm,
        layout_kv,
        False,
        scale_dtype,
        tc_name,
        "v_descale",
    )

    # --- block_table ---
    block_table_shape = (B, max_blocks_per_batch)

    # --- cu_seqlens_q: 仅 TND 时输出 ---
    if layout_q == "TND":
        if c["cu_seqlens_q"] is not None:
            cu_seqlens_q_shape = (len(c["cu_seqlens_q"]),)
        else:
            # TND 但无 cu_seqlens_q → 用 (B+1,) 由 customize_inputs 填充
            cu_seqlens_q_shape = (B + 1,)
    else:
        cu_seqlens_q_shape = None

    # --- seqused_q / seqused_kv ---
    # 规则: xlsx 有值 → 输出 tensor shape + attrs 值（让 TTK override 填充）
    #       xlsx 无值 → tensor shape=None, attrs 无 key（算子内部用 max_seqlen 容错）
    seqused_q_shape = (B,) if seqused_q is not None else None
    seqused_kv_shape = (B,) if seqused_kv is not None else None

    # --- sinks: 暂不支持 ---
    sinks_shape = None

    # --- attn_mask: mask_mode in [3,4] 时输出固定 (2048, 2048) ---
    if c["mask_mode"] in (3, 4):
        attn_mask_shape = DEFAULT_ATTN_MASK_SHAPE
    else:
        attn_mask_shape = None

    # --- metadata: 固定 16384，不做预计算 ---
    metadata_shape = DEFAULT_METADATA_SHAPE

    # --- 组装 12 tensor ---
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

    # --- dtypes ---
    q_csv_dtype = CSV_DTYPE_MAP.get(c["q_dtype"].upper(), "bfloat16")
    scale_csv_dtype = CSV_SCALE_DTYPE.get(qm, "float8_e8m0fnu")
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

    # --- attributes ---
    attrs = {
        "batch_size": B,
        "num_heads_q": N1,
        "num_heads_kv": N2,
        "head_dim": D,
        "quant_compute_mode": qm,
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
    # seqused_q / seqused_kv: 有值时加入 attrs（让 override_tensors_from_attributes 填充）
    if seqused_q is not None:
        attrs["seqused_q"] = seqused_q
    if seqused_kv is not None:
        attrs["seqused_kv"] = seqused_kv
    if c["cu_seqlens_q"] is not None and layout_q == "TND":
        attrs["cu_seqlens_q"] = c["cu_seqlens_q"]

    # --- data ranges ---
    q_range = c["q_datarange"] if c["q_datarange"] else (-10.0, 10.0)
    kv_range = (-10.0, 10.0)
    scale_range = CSV_SCALE_RANGE.get(qm, (0.01, 10.0))
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

    # --- 格式化 ---
    testcase_name = prefix + c["name"]
    fields = [
        testcase_name,
        API_NAME,
        _fmt_shapes(view_shapes),
        _fmt_dtypes(dtypes),
        _fmt_storage_shapes(view_shapes),  # storage = view, None → ()
        _fmt_attrs(attrs),
        _fmt_ranges(ranges),
    ]
    return ",".join(fields)


def _fmt_shape(s):
    """格式化单个 shape: None -> 'None', (64,) -> '(64,)', (2,3) -> '(2,3)'。"""
    if s is None:
        return "None"
    if len(s) == 1:
        return f"({s[0]},)"
    return "(" + ",".join(str(x) for x in s) + ")"


def _fmt_shapes(shapes):
    """view_shapes: None 用 'None'。"""
    return '"(' + ",".join(_fmt_shape(s) for s in shapes) + ')"'


def _fmt_storage_shapes(view_shapes):
    """storage_shapes: view 为 None 的位置用 '()' 表示。"""
    parts = []
    for s in view_shapes:
        if s is None:
            parts.append("()")
        else:
            parts.append(_fmt_shape(s))
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
        description="将用例 xlsx 转换为 pytest 用例文件 + ttk CSV 用例文件（纯文件转换，不依赖代码目录结构）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --xlsx input.xlsx --pytest-out functional.py --ttk-out cases.csv
  %(prog)s --xlsx input.xlsx --ttk-out cases.csv --ttk-only
  %(prog)s --xlsx input.xlsx --pytest-out functional.py --pytest-only --prefix "rd_"
  %(prog)s --xlsx input.xlsx --pytest-out functional.py --ttk-out cases.csv
  %(prog)s --xlsx input.xlsx --dry-run
        """,
    )
    parser.add_argument("--xlsx", required=True, help="输入 xlsx 文件路径")
    parser.add_argument(
        "--pytest-out",
        default=None,
        help="输出的 pytest 用例文件路径 (不指定且非 --ttk-only 时报错)",
    )
    parser.add_argument(
        "--ttk-out",
        default=None,
        help="输出的 ttk CSV 文件路径 (不指定且非 --pytest-only 时报错)",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_TTK_PREFIX,
        help=f"ttk 用例名前缀 (默认: '{DEFAULT_TTK_PREFIX}')",
    )
    parser.add_argument("--dry-run", action="store_true", help="只解析不写文件")
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="只生成 pytest 文件 (需 --pytest-out)",
    )
    parser.add_argument(
        "--ttk-only", action="store_true", help="只生成 ttk CSV 文件 (需 --ttk-out)"
    )
    args = parser.parse_args()

    # 校验输出路径
    if not args.pytest_only and not args.pytest_out and not args.ttk_only:
        args.pytest_only = False
    if not args.ttk_only and not args.ttk_out and not args.pytest_only:
        args.ttk_only = False
    need_pytest = not args.ttk_only
    need_ttk = not args.pytest_only
    if need_pytest and not args.pytest_out:
        parser.error("生成 pytest 需要 --pytest-out (或用 --ttk-only 跳过)")
    if need_ttk and not args.ttk_out:
        parser.error("生成 ttk CSV 需要 --ttk-out (或用 --pytest-only 跳过)")

    print(f"输入 xlsx:       {args.xlsx}")
    if need_pytest:
        print(f"pytest 输出:     {args.pytest_out}")
    if need_ttk:
        print(f"ttk CSV 输出:    {args.ttk_out}")
    print(f"用例名前缀:      '{args.prefix}'")
    print()

    if not os.path.exists(args.xlsx):
        print(f"ERROR: xlsx 文件不存在: {args.xlsx}", file=sys.stderr)
        sys.exit(1)

    reader = XlsxReader(args.xlsx)
    cases = reader.parse_cases()
    print(f"解析到 {len(cases)} 个用例:")
    for c in cases:
        print(f"  {c['name']}")
        print(
            f"    B={c['B']} N1={c['N1']} N2={c['N2']} S1={c['S1']} S2={c['S2']} "
            f"D={c['D']} bs={c['block_size']} qm={c['quant_compute_mode']} "
            f"mask={c['mask_mode']} layout_q={c['layout_q']}"
        )
    print()

    if args.dry_run:
        print("--dry-run 模式，不写文件")
        return

    if need_pytest:
        os.makedirs(os.path.dirname(os.path.abspath(args.pytest_out)), exist_ok=True)
        n = gen_pytest(cases, args.pytest_out)
        print(f"[pytest] 已生成 {n} 个用例 -> {args.pytest_out}")

    if need_ttk:
        os.makedirs(os.path.dirname(os.path.abspath(args.ttk_out)), exist_ok=True)
        n = gen_ttk(cases, args.ttk_out, prefix=args.prefix)
        print(f"[ttk]    已生成 {n} 个用例 -> {args.ttk_out}")


if __name__ == "__main__":
    main()
