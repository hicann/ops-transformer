#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import ast
import csv
import logging
import os
import sys
import types
from typing import List

import numpy
import torch

# ----- 纯 CPU 环境兼容：在 import golden 模块之前 stub 掉 torch_npu / cann_ops_transformer -----
# common/qfa_mxfp4_golden.py 顶部 import torch_npu 并
# from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn，
# 本机没有这些包，注入空 module 以便 cpu_mxfp4_golden 路径可用（NPU 调用路径不会被触发）。
if "torch_npu" not in sys.modules:
    _tn = types.ModuleType("torch_npu")
    _tn.npu = types.SimpleNamespace(
        set_device=lambda *_a, **_k: None,
        synchronize=lambda *_a, **_k: None,
    )
    sys.modules["torch_npu"] = _tn

if "cann_ops_transformer" not in sys.modules:
    _cot = types.ModuleType("cann_ops_transformer")
    _cot_ops = types.ModuleType("cann_ops_transformer.ops")
    _cot_ops.quant_flash_attn_metadata = lambda *a, **k: None
    _cot_ops.quant_flash_attn = lambda *a, **k: (None, None)
    _cot.ops = _cot_ops
    sys.modules["cann_ops_transformer"] = _cot
    sys.modules["cann_ops_transformer.ops"] = _cot_ops

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from common import qfa_mxfp4_golden as golden_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("gen_bins")


# torch dtype → torch dtype 名字符串（用于 __bin_inputs / __bin_golden 三元组的 dtype_name 字段）。
# bin 文件按 numpy raw 字节存储，加载端用 _load_bin_tensor(path, shape, dtype_name) 还原。
# experimental MXFP4: q/k/v 是 uint8 (2 个 fp4 打包成 1 byte), descale 是 uint8 (e8m0)。
_TORCH_DTYPE_NAME = {
    torch.float32: "float32",
    torch.int32: "int32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.uint8: "uint8",
    torch.int8: "int8",
}


def _torch_dtype_name(torch_dtype):
    """torch dtype → 名字符串（__bin_inputs/__bin_golden 的 dtype_name 字段）。"""
    if torch_dtype not in _TORCH_DTYPE_NAME:
        raise ValueError(f"unsupported torch dtype: {torch_dtype}")
    return _TORCH_DTYPE_NAME[torch_dtype]


def _torch_to_bin(tensor, path):
    """torch tensor → numpy raw .bin."""
    if tensor is None:
        raise ValueError(f"_torch_to_bin: tensor 为 None，无法保存到 {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = tensor.detach().cpu().contiguous()
    numpy_arr = (
        arr.numpy() if arr.dtype != torch.bfloat16 else arr.view(torch.uint8).numpy()
    )
    numpy_arr.tofile(path)


def _bin_input_descriptor(tensor, path):
    """构造 (path, shape, dtype_name) 三元组，供 inputs.py __bin_inputs 加载。"""
    return (path, tuple(tensor.shape), _torch_dtype_name(tensor.dtype))


def _parse_csv_row(row):
    """解析 csv 一行为 (case_name, shapes, dtypes, attrs)。"""
    case_name = row["testcase_name"]
    shapes = (
        ast.literal_eval(row["tensor_view_shapes"]) if row["tensor_view_shapes"] else ()
    )
    dtypes = ast.literal_eval(row["tensor_dtypes"]) if row["tensor_dtypes"] else ()
    attrs = ast.literal_eval(row["attributes"]) if row["attributes"] else {}
    return case_name, shapes, dtypes, attrs


def _apply_golden_globals(attrs):
    # 与 inputs.py _apply_golden_globals 字段映射一致
    _apply_golden_globals_map = {
        "B": "B",
        "N_q": "N_q",
        "N_kv": "N_kv",
        "G": "G",
        "D": "D",
        "V_D": "V_D",
        "Rope_D": "Rope_D",
        "input_layout": "INPUT_LAYOUT",
        "layout_q_descale": "LAYOUT_Q_DESCALE",
        "layout_kv": "LAYOUT_KV",
        "layout_out": "LAYOUT_OUT",
        "kv_storage_mode": "KV_STORAGE_MODE",
        "block_size": "BLOCK_SIZE",
        "q_dtype": "Q_DTYPE",
        "kv_dtype": "KV_DTYPE",
        "out_dtype": "OUT_DTYPE",
        "q_quant_mode": "Q_QUANT_MODE",
        "mask_mode": "SPARSE_MODE",
        "pre_tokens": "PRE_TOKENS",
        "next_tokens": "NEXT_TOKENS",
        "enable_mask": "ENABLE_MASK",
        "enable_lse": "ENABLE_LSE",
        "inner_precise": "INNER_PRECISE",
        "device_id": "DEVICE_ID",
        "graph_path": "GRAPH_PATH",
        "softmax_scale": "SOFTMAX_SCALE",
        "data_range_q": "DATA_RANGE_Q",
        "data_range_k": "DATA_RANGE_K",
        "data_range_v": "DATA_RANGE_V",
        "act_seq_lens_q": "ACT_SEQ_LENS_Q",
        "act_seq_lens_kv": "ACT_SEQ_LENS_KV",
        "max_seqlen_q": "MAX_SEQLEN_Q",
        "max_seqlen_kv": "MAX_SEQLEN_KV",
        "cu_seqlens_q": "CU_SEQLENS_Q",
        "cu_seqlens_kv": "CU_SEQLENS_KV",
        "block_table_shape": "BLOCK_TABLE_SHAPE",
        "block_table_dtype": "BLOCK_TABLE_DTYPE",
        "block_table_datarange": "BLOCK_TABLE_DATARANGE",
        "p_scale_value": "P_SCALE_VALUE",
        "p_scale_shape": "P_SCALE_SHAPE",
        "p_scale_dtype": "P_SCALE_DTYPE",
        "p_scale_datarange": "P_SCALE_DATARANGE",
        "sinks_shape": "SINKS_SHAPE",
        "sinks_dtype": "SINKS_DTYPE",
        "sinks_datarange": "SINKS_DATARANGE",
        "attn_mask_shape": "ATTN_MASK_SHAPE",
        "attn_mask_dtype": "ATTN_MASK_DTYPE",
        "attn_mask_datarange": "ATTN_MASK_DATARANGE",
        "q_descale_dtype": "Q_DESCALE_DTYPE",
        "k_descale_dtype": "K_DESCALE_DTYPE",
        "v_descale_dtype": "V_DESCALE_DTYPE",
        "seqused_q_dtype": "SEQUSED_Q_DTYPE",
        "seqused_kv_dtype": "SEQUSED_KV_DTYPE",
        "cu_seqlens_q_dtype": "CU_SEQLENS_Q_DTYPE",
        "cu_seqlens_kv_dtype": "CU_SEQLENS_KV_DTYPE",
        "softmax_lse_dtype": "SOFTMAX_LSE_DTYPE",
    }
    for attr_key, golden_key in _apply_golden_globals_map.items():
        if attr_key in attrs and attrs[attr_key] is not None:
            val = attrs[attr_key]
            # list 类型字段: 确保 list (csv 可能是 tuple)
            if isinstance(val, (list, tuple)):
                val = list(val)
            setattr(golden_mod, golden_key, val)


def generate_for_case(case_name, shapes, dtypes, attrs, output_dir):
    """为一个 case 生成 cpu 侧 bin 文件。"""
    logger.info("=" * 60)
    logger.info("Case: %s", case_name)
    logger.info(
        "attrs: B=%s N_q=%s N_kv=%s D=%s, enable_lse=%s",
        attrs.get("B"),
        attrs.get("N_q"),
        attrs.get("N_kv"),
        attrs.get("D"),
        attrs.get("enable_lse"),
    )

    # 1. 注入 golden 全局变量 (generate_data 依赖)
    _apply_golden_globals(attrs)

    # 2. 调 generate_data 拿 final-layout MXFP4 data_dict (确定性种子, 纯 CPU)
    data_dict = golden_mod.generate_data()

    # 3. 保存输入 bin: {case}_cpu_{idx}.bin
    # data_dict 顺序: q, k, v, q_descale, k_descale, v_descale, block_table, p_scale, sinks, attn_mask
    input_keys = [
        "q",
        "k",
        "v",
        "q_descale",
        "k_descale",
        "v_descale",
        "block_table",
        "p_scale",
        "sinks",
        "attn_mask",
    ]
    bin_input_paths = []
    for idx, key in enumerate(input_keys):
        tensor = data_dict.get(key)
        if tensor is None:
            continue
        path = os.path.join(output_dir, f"{case_name}_cpu_{idx}.bin")
        _torch_to_bin(tensor, path)
        bin_input_paths.append(
            (idx, path, tuple(tensor.shape), _torch_dtype_name(tensor.dtype))
        )
        logger.info(
            "  saved input %d (%s) → %s (shape=%s, dtype=%s)",
            idx,
            key,
            path,
            tuple(tensor.shape),
            _torch_dtype_name(tensor.dtype),
        )

    # 4. 调 cpu_mxfp4_golden 拿 golden 输出, 存 bin: {case}_golden_{i}.bin
    enable_lse = bool(attrs.get("enable_lse", 0))
    cpu_out, cpu_lse = golden_mod.cpu_mxfp4_golden(data_dict)

    golden_out_paths = []
    # golden slot 0: atten_out
    out_path_0 = os.path.join(output_dir, f"{case_name}_golden_0.bin")
    _torch_to_bin(cpu_out, out_path_0)
    golden_out_paths.append(out_path_0)
    logger.info(
        "  saved golden 0 (atten_out) → %s (shape=%s, %d bytes)",
        out_path_0,
        tuple(cpu_out.shape),
        os.path.getsize(out_path_0),
    )

    # golden slot 1: lse_out (enable_lse=True 时才有)
    if enable_lse and cpu_lse is not None:
        out_path_1 = os.path.join(output_dir, f"{case_name}_golden_1.bin")
        _torch_to_bin(cpu_lse, out_path_1)
        golden_out_paths.append(out_path_1)
        logger.info(
            "  saved golden 1 (lse_out) → %s (shape=%s, %d bytes)",
            out_path_1,
            tuple(cpu_lse.shape),
            os.path.getsize(out_path_1),
        )

    return {
        "case": case_name,
        "inputs": bin_input_paths,
        "golden": golden_out_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="预生成 cpu 侧 bin 文件 (MXFP4)")
    parser.add_argument(
        "--csv",
        default=os.path.join(_TEST_DIR, "qfa_mxfp4.csv"),
        help="输入 csv 路径（默认 qfa_mxfp4.csv）",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/qfa_mxfp4_bins", help="bin 文件输出目录"
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="只生成指定 case（逗号分隔 testcase_name），默认全部",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    wanted = set(args.cases.split(",")) if args.cases else None
    results = []
    for row in rows:
        case_name, shapes, dtypes, attrs = _parse_csv_row(row)
        if wanted and case_name not in wanted:
            continue
        try:
            res = generate_for_case(case_name, shapes, dtypes, attrs, args.output_dir)
            results.append(res)
        except Exception as e:
            logger.error("[FAIL] %s: %s", case_name, e)
            raise

    logger.info("=" * 60)
    logger.info(
        "Done. Generated bin files for %d cases in %s", len(results), args.output_dir
    )
    for r in results:
        logger.info(
            "  %s: %d inputs, %d golden", r["case"], len(r["inputs"]), len(r["golden"])
        )


if __name__ == "__main__":
    main()
