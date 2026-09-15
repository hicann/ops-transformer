#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.

# !
#  \file parse_matrix.py
#  \brief Workspace 段数据解析脚本。
#         解析异常 dump 产生的 workspace_seg*_type*_*.bin 文件，按指定数据类型和 shape 展示矩阵内容。
#
# 用法:
#   python parse_matrix.py <seg_bin> --dtype <type> --rows <M> --cols <K> [options]
#
# 数据类型:
#   fp16       IEEE FP16 (2B/element)
#   bf16       BFloat16 (2B/element)
#   fp8_e4m3   FP8 E4M3FN (1B/element)
#   fp8_e5m2   FP8 E5M2 (1B/element)
#   fp4_e2m1   FP4 E2M1 (4bit/element, packed 2 per byte)
#   e8m0       E8M0 scale (1B/element, power-of-2 exponent)
#   int8       INT8 (1B/element)
#   uint8      UINT8 (1B/element)
#   fp32       IEEE FP32 (4B/element)

import sys
import os
import struct
import argparse
import logging
import math

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

DEFAULT_PREVIEW_ROWS = 16

DTYPE_INFO = {
    "fp16": {"size": 2, "name": "FP16 (IEEE half)"},
    "bf16": {"size": 2, "name": "BF16 (bfloat16)"},
    "fp8_e4m3": {"size": 1, "name": "FP8 E4M3FN"},
    "fp8_e5m2": {"size": 1, "name": "FP8 E5M2"},
    "fp4_e2m1": {"size": 0, "name": "FP4 E2M1 (packed)"},
    "e8m0": {"size": 1, "name": "E8M0 (scale exponent)"},
    "int8": {"size": 1, "name": "INT8"},
    "uint8": {"size": 1, "name": "UINT8"},
    "fp32": {"size": 4, "name": "FP32 (IEEE single)"},
}


# ====== FP16 ======
def parse_fp16(data: bytes, offset: int, count: int):
    vals = []
    for i in range(count):
        raw = struct.unpack_from("<H", data, offset + i * 2)[0]
        vals.append(fp16_to_float(raw))
    return vals


def fp16_to_float(bits: int) -> float:
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        if mant == 0:
            return 0.0 if sign == 0 else -0.0
        val = (mant / 1024.0) * (2**-14)
    elif exp == 0x1F:
        if mant != 0:
            return float("nan")
        return float("-inf") if sign else float("inf")
    else:
        val = (1.0 + mant / 1024.0) * (2 ** (exp - 15))
    return -val if sign else val


# ====== BF16 ======
def parse_bf16(data: bytes, offset: int, count: int):
    vals = []
    for i in range(count):
        raw = struct.unpack_from("<H", data, offset + i * 2)[0]
        # BF16: truncate FP32 to upper 16 bits, so expand: raw << 16
        fp32_bits = raw << 16
        vals.append(struct.unpack("<f", struct.pack("<I", fp32_bits))[0])
    return vals


# ====== FP8 E4M3FN ======
def parse_fp8_e4m3(data: bytes, offset: int, count: int):
    vals = []
    for i in range(count):
        b = data[offset + i]
        vals.append(fp8_e4m3_to_float(b))
    return vals


def fp8_e4m3_to_float(b: int) -> float:
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0 and mant == 0:
        return 0.0 if sign == 0 else -0.0
    if exp == 0:
        val = (mant / 8.0) * (2**-6)  # subnormal, bias=7
    elif exp == 0xF and mant == 0x7:
        return float("nan")
    else:
        val = (1.0 + mant / 8.0) * (
            2 ** (exp - 7)
        )  # bias=7, exp=0xF & math<7 is finite
    return -val if sign else val


# ====== FP8 E5M2 ======
def parse_fp8_e5m2(data: bytes, offset: int, count: int):
    vals = []
    for i in range(count):
        b = data[offset + i]
        vals.append(fp8_e5m2_to_float(b))
    return vals


def fp8_e5m2_to_float(b: int) -> float:
    sign = (b >> 7) & 1
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    if exp == 0 and mant == 0:
        return 0.0 if sign == 0 else -0.0
    if exp == 0:
        val = (mant / 4.0) * (2**-14)  # subnormal, exponent = 1 - bias = -14
    elif exp == 0x1F:
        if mant != 0:
            return float("nan")
        return float("-inf") if sign else float("inf")
    else:
        val = (1.0 + mant / 4.0) * (2 ** (exp - 15))  # bias=15
    return -val if sign else val


# ====== FP4 E2M1 (packed, 2 elements per byte) ======
def parse_fp4_e2m1(data: bytes, offset: int, count: int):
    vals = []
    for i in range(count):
        byte_idx = i // 2
        if i % 2 == 0:
            nibble = (data[offset + byte_idx] >> 4) & 0xF  # high nibble first
        else:
            nibble = data[offset + byte_idx] & 0xF
        vals.append(fp4_e2m1_to_float(nibble))
    return vals


def fp4_e2m1_to_float(nibble: int) -> float:
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 0x3
    mant = nibble & 0x1
    if exp == 0 and mant == 0:
        return 0.0 if sign == 0 else -0.0
    if exp == 0:
        val = (mant / 2.0) * (2**0)  # subnormal, bias=1
    else:
        val = (1.0 + mant / 2.0) * (2 ** (exp - 1))  # bias=1
    return -val if sign else val


# ====== E8M0 (scale exponent, power of 2) ======
def parse_e8m0(data: bytes, offset: int, count: int):
    vals = []
    for i in range(count):
        b = data[offset + i]
        # E8M0: 8-bit exponent with bias 127, no mantissa/sign。
        # 0x00 表示 2^-127，0xFE 表示 2^127，0xFF 表示 NaN。
        vals.append(float("nan") if b == 0xFF else 2.0 ** (b - 127))
    return vals


# ====== simple types ======
def parse_int8(data: bytes, offset: int, count: int):
    return list(struct.unpack_from(f"<{count}b", data, offset))


def parse_uint8(data: bytes, offset: int, count: int):
    return list(struct.unpack_from(f"<{count}B", data, offset))


def parse_fp32(data: bytes, offset: int, count: int):
    return list(struct.unpack_from(f"<{count}f", data, offset))


PARSE_FUNCS = {
    "fp16": parse_fp16,
    "bf16": parse_bf16,
    "fp8_e4m3": parse_fp8_e4m3,
    "fp8_e5m2": parse_fp8_e5m2,
    "fp4_e2m1": parse_fp4_e2m1,
    "e8m0": parse_e8m0,
    "int8": parse_int8,
    "uint8": parse_uint8,
    "fp32": parse_fp32,
}


def get_element_size(dtype: str) -> float:
    if dtype == "fp4_e2m1":
        return 0.5
    return float(DTYPE_INFO[dtype]["size"])


def parse_element(data: bytes, dtype: str, element_index: int):
    """按逻辑元素下标读取一个值，避免把大文件全部展开为 Python 对象。"""
    if dtype == "fp4_e2m1":
        packed = data[element_index // 2]
        nibble = (packed >> 4) & 0xF if element_index % 2 == 0 else packed & 0xF
        return fp4_e2m1_to_float(nibble)
    byte_offset = element_index * DTYPE_INFO[dtype]["size"]
    return PARSE_FUNCS[dtype](data, byte_offset, 1)[0]


def format_value(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "  NaN"
        if math.isinf(v):
            return " +Inf" if v > 0 else " -Inf"
        if v == 0.0:
            return " 0.00"
        if abs(v) < 0.01 or abs(v) >= 10000:
            return f"{v:8.2e}"
        return f"{v:8.4f}"
    return f"{v:8d}"


def main():
    parser = argparse.ArgumentParser(
        description="Workspace 段数据解析脚本（支持 fp16/bf16/fp8/fp4/e8m0）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 解析 COMM_OUT 段 (fp8_e4m3, shape 57086x1536, 只看前 8 行)
  python parse_matrix.py workspace_seg1_type5_*.bin --dtype fp8_e4m3 --rows 57086 --cols 1536 --row-range 0 8

  # 看行 5:10 列 5:10 的数据（空格分隔）
  python parse_matrix.py seg.bin --dtype fp8_e4m3 --rows 57086 --cols 1536 --row-range 5 10 --col-range 5 10

  # 看指定行和列
  python parse_matrix.py seg.bin --dtype fp16 --rows 4096 --cols 1024 --row 0 100 200 --col 0 63 1023

  # 解析 scale 通信段 (e8m0, shape 57086x48, 48=24*2)
  python parse_matrix.py workspace_seg4_type11_*.bin --dtype e8m0 --rows 57086 --cols 48 --row-range 0 8

        """,
    )
    parser.add_argument("bin_file", help="workspace_seg*_type*_*.bin 文件路径")
    parser.add_argument(
        "--dtype", required=True, choices=list(DTYPE_INFO.keys()), help="数据类型"
    )
    parser.add_argument("--rows", type=int, required=True, help="矩阵行数 M")
    parser.add_argument("--cols", type=int, required=True, help="矩阵列数 K")
    parser.add_argument(
        "--row-range",
        metavar=("START", "END"),
        type=int,
        nargs=2,
        default=None,
        help="显示指定行范围 [start, end)，如 --row-range 5 10",
    )
    parser.add_argument(
        "--col-range",
        metavar=("START", "END"),
        type=int,
        nargs=2,
        default=None,
        help="显示指定列范围 [start, end)，如 --col-range 5 10",
    )
    parser.add_argument(
        "--row",
        type=int,
        nargs="+",
        default=None,
        help="只显示指定行号（可指定多个，如 --row 0 100 57085）",
    )
    parser.add_argument(
        "--col",
        type=int,
        nargs="+",
        default=None,
        help="只显示指定列号（可指定多个，如 --col 0 63 1535）",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="按列优先顺序解析（AlltoAll 通信结果布局）",
    )
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows 和 --cols 必须为正整数")

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()
    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    dtype = args.dtype
    elem_size = get_element_size(dtype)
    total_elems = args.rows * args.cols
    file_elems = int(len(data) / elem_size)
    logging.info(
        "数据类型: %s, 元素大小: %.1f B, 矩阵: %dx%d = %d 元素, 文件可容纳: %d 元素",
        DTYPE_INFO[dtype]["name"],
        elem_size,
        args.rows,
        args.cols,
        total_elems,
        file_elems,
    )

    if file_elems < total_elems:
        logging.warning(
            "文件元素数 %d < 矩阵元素数 %d, 只解析文件中存在的数据",
            file_elems,
            total_elems,
        )
        total_elems = file_elems

    # 确定要显示的列
    col_start = 0
    col_end = args.cols
    if args.col_range:
        col_start = max(args.col_range[0], 0)
        col_end = min(args.col_range[1], args.cols)

    # 确定要显示的行
    if args.row:
        display_rows = sorted(set(r for r in args.row if 0 <= r < args.rows))
    elif args.row_range:
        display_rows = list(
            range(max(args.row_range[0], 0), min(args.row_range[1], args.rows))
        )
    else:
        # 未指定行时固定预览前 16 行，不再通过额外命令行参数控制。
        display_rows = list(range(min(DEFAULT_PREVIEW_ROWS, args.rows)))

    if args.col:
        display_cols = sorted(set(c for c in args.col if 0 <= c < args.cols))
    elif args.col_range:
        display_cols = list(range(args.col_range[0], min(args.col_range[1], args.cols)))
    elif args.row or args.row_range:
        # 指定了行但没指定列时，显示全部列
        display_cols = list(range(args.cols))
    else:
        display_cols = list(range(col_start, col_end))

    # 只解析用户选择的行列。普通布局按 row-major 取数；--transpose
    # 表示文件按 column-major 存储，逻辑坐标 (r, c) 的下标为 c * rows + r。
    matrix = {}
    selected_values = []
    for row_index in display_rows:
        row = {}
        for col_index in display_cols:
            element_index = (
                (col_index * args.rows + row_index)
                if args.transpose
                else (row_index * args.cols + col_index)
            )
            if element_index >= total_elems:
                continue
            value = parse_element(data, dtype, element_index)
            row[col_index] = value
            selected_values.append(value)
        matrix[row_index] = row

    # 人类可读输出
    print()
    print("=" * 80)
    print(f"  Workspace Segment: {os.path.basename(args.bin_file)}")
    print(f"  Dtype: {DTYPE_INFO[dtype]['name']}, Shape: [{args.rows}, {args.cols}]")
    if args.row:
        print(f"  Rows: {display_rows}")
    elif args.row_range:
        print(f"  Rows: [{args.row_range[0]}, {args.row_range[1]})")
    else:
        print(f"  Rows: [0, {min(DEFAULT_PREVIEW_ROWS, args.rows)}) (default)")
    if args.col:
        print(f"  Cols: {display_cols}")
    elif args.col_range:
        print(f"  Cols: [{args.col_range[0]}, {args.col_range[1]})")
    else:
        print(f"  Cols: [{col_start}, {col_end})")
    print("=" * 80)
    print()

    # 判断是否使用紧凑模式（指定了行范围且列数较少时，用空格分隔）
    compact = (args.row_range is not None or args.row is not None) and len(
        display_cols
    ) <= 64

    if compact:
        # 紧凑模式：每个数据用空格分隔
        for r in display_rows:
            row = matrix[r]
            parts = []
            for c in display_cols:
                if c in row:
                    v = row[c]
                    if isinstance(v, float):
                        if math.isnan(v):
                            parts.append("NaN")
                        elif math.isinf(v):
                            parts.append("-Inf" if v < 0 else "Inf")
                        elif v == 0.0:
                            parts.append("0")
                        else:
                            parts.append(f"{v:g}")
                    else:
                        parts.append(str(v))
                else:
                    parts.append("N/A")
            print(f"row {r}: " + " ".join(parts))
    else:
        # 表格模式
        header = f"{'row':>6} |"
        for c in display_cols:
            header += f"{'c' + str(c):>8}"
        print(header)
        print("-" * len(header))

        for r in display_rows:
            row = matrix[r]
            line = f"{r:>6} |"
            for c in display_cols:
                if c in row:
                    line += format_value(row[c])
                else:
                    line += "      N/A"
            print(line)

    # 统计信息仅针对本次显示范围，避免大文件被展开为数千万个 Python 对象。
    print()
    print("--- 显示范围统计信息 ---")
    non_zero = [
        v
        for v in selected_values
        if v != 0 and not (isinstance(v, float) and math.isnan(v))
    ]
    if not selected_values:
        print("  显示范围内没有可解析元素")
    elif non_zero:
        if all(isinstance(v, (int, float)) for v in non_zero):
            print(f"  非零元素数: {len(non_zero)} / {len(selected_values)}")
            print(f"  最小值: {min(non_zero):.6e}")
            print(f"  最大值: {max(non_zero):.6e}")
            abs_vals = [abs(v) for v in non_zero]
            print(f"  绝对值最小: {min(abs_vals):.6e}")
            print(f"  绝对值最大: {max(abs_vals):.6e}")
            avg = sum(abs_vals) / len(abs_vals)
            print(f"  绝对值平均: {avg:.6e}")
        else:
            print(f"  非零元素数: {len(non_zero)} / {len(selected_values)}")
    else:
        print("  全部为零")


if __name__ == "__main__":
    main()
