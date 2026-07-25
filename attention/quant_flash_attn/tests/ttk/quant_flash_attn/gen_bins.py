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
"""
gen_bins.py — 预生成 cpu 侧 bin 文件，供 cpu/npu 分跑工作流使用。

工作流：
  1. cpu 机器跑：python3 gen_bins.py --output-dir /tmp/qfa_bins
     生成 {case}_cpu_{idx}.bin（输入）和 {case}_golden_{idx}.bin（golden 输出）。
  2. npu 机器跑：csv attributes 中加 __bin_inputs / __bin_golden 指向 bin 路径，
     inputs.py 从 bin 加载输入，golden.py 从 bin 加载 golden，wrapper 只跑 npu 执行。

bin 格式：numpy raw .bin（tensor.numpy().tofile()），
        加载端用 numpy.fromfile(path, dtype=...).reshape(shape)（与 ttk load_numpy_data 一致）。

cpu 侧不需要 torch_npu / cann_ops_transformer，本脚本在纯 torch CPU 环境下可运行。
"""

import argparse
import ast
import csv
import logging
import os
import sys
import types
from typing import List, Tuple

import numpy
import torch

# ----- 纯 CPU 环境兼容：在 import golden 模块之前 stub 掉 torch_npu / cann_ops_transformer -----
# common/quant_flash_attn_golden.py 顶部 import torch_npu 并
# from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn，
# 本机没有这些包，注入空 module 以便 cpu_mxfp8_golden 路径可用（NPU 调用路径不会被触发）。
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

# torch 2.4 没有 torch.float8_e8m0fnu，cpu 侧 gen_bins 路径不调用 e8m0 打包函数
# （fp32_to_e8m0fnu 只在 NPU 入参预处理 prepare_npu_inputs 中调）。
# 这里给 torch.float8_e8m0fnu 一个别名，使 common/quant_flash_attn_golden.py 顶部
# 模块级引用（FP8_DTYPE / 函数定义）不报 AttributeError。占位 tensor 用此 dtype 不会被保存到 bin。
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_TEST_DIR, "assets")
_ASSETS_IMPL_DIR = os.path.join(_ASSETS_DIR, "impl")
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
for _p in (_ASSETS_DIR, _ASSETS_IMPL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import quant_flash_attn_golden as golden_mod  # noqa: E402


def _load_impl_module(stem):
    """与 spec.py 同样的方式加载 assets/impl/{stem}.py（无 __init__.py 的目录）。"""
    import importlib.util

    path = os.path.join(_ASSETS_IMPL_DIR, f"{stem}.py")
    spec = importlib.util.spec_from_file_location(
        f"qfa_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inputs_module = _load_impl_module("inputs")
golden_module = _load_impl_module("golden")

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("gen_bins")


# torch dtype → torch dtype 名字符串（用于 __bin_inputs / __bin_golden 三元组的 dtype_name 字段）。
# bin 文件按 numpy raw 字节存储，加载端用 _load_bin_tensor(path, shape, dtype_name) 还原。
# fp8_e4m3fn / fp8_e8m0fnu 在 bin 中以 uint8 字节存储，dtype_name 用 torch dtype 名，
# 加载端 _load_bin_tensor 据此 view 还原（与 ttk load_numpy_data 的 dtype 参数约定一致）。
_TORCH_DTYPE_NAME = {
    torch.float32: "float32",
    torch.int32: "int32",
    torch.float16: "float16",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e8m0fnu: "float8_e8m0fnu",
    torch.uint8: "uint8",
    torch.int8: "int8",
}


def _torch_dtype_name(torch_dtype):
    """torch dtype → 名字符串（__bin_inputs/__bin_golden 的 dtype_name 字段）。"""
    if torch_dtype not in _TORCH_DTYPE_NAME:
        raise ValueError(f"unsupported torch dtype: {torch_dtype}")
    return _TORCH_DTYPE_NAME[torch_dtype]


def _torch_to_bin(tensor, path):
    """torch tensor → numpy raw .bin（与 assets/impl/golden.py 的 _torch_to_bin 一致）。"""
    if tensor is None:
        raise ValueError(f"_torch_to_bin: tensor 为 None，无法保存到 {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = tensor.detach().cpu().contiguous()
    if arr.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        numpy_arr = arr.view(torch.uint8).numpy()
    else:
        numpy_arr = arr.numpy()
    numpy_arr.tofile(path)


def _bin_input_descriptor(tensor, path):
    """构造 (path, shape, dtype_name) 三元组，供 inputs.py __bin_inputs 加载。

    dtype_name 是 torch dtype 字符串名（如 'float8_e4m3fn'、'float32'），
    与 ttk load_numpy_data 的 dtype 参数约定一致。
    """
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


def _torch_dtype_from_str(name):
    """tensor_dtypes 列中的字符串名 → torch dtype。"""
    mapping = {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e8m0fnu": torch.float8_e8m0fnu,
        "float32": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype string: {name}")
    return mapping[name]


def _build_kwargs(attrs):
    """从 csv attributes 构造 inputs/golden 函数的 keyword 参数 dict。

    与 qfa_mxfp8_wrapper.py / assets/impl/golden.py 接收的 keyword 参数对齐。
    csv attributes 中的 layout_kv / layout_q_descale 已映射到 wrapper 签名的
    enable_pa / kv_cache_layout / q_scale_layout；block_size 在 csv 未显式给出，
    PA 模式用 128（与 redline.xlsx PA 用例一致）。
    """
    layout_kv = attrs.get("layout_kv", "TND")
    enable_pa = isinstance(layout_kv, str) and layout_kv.startswith("PA_")
    # PA 模式 block_size：csv 未显式传，cpu 侧 bin 生成用 128（与 redline PA 用例一致）。
    # 影响 block_table 形状，不影响 q/k/v/scale 的数值；npu 侧 F3 验证时以真实 block_size 为准。
    block_size = attrs.get("block_size") or (128 if enable_pa else 0)

    # B = len(cu_seqlens_q) - 1，与 wrapper 约定一致（cu_seqlens_q[0]=0, cu_seqlens_q[-1]=total）
    cu_seqlens_q = attrs.get("cu_seqlens_q") or [0, 0]
    B = max(1, len(cu_seqlens_q) - 1) if len(cu_seqlens_q) >= 2 else 1
    # csv 用 B 字段名（与 wrapper 签名一致），优先取 attrs["B"]，否则从 cu_seqlens 推
    if attrs.get("B") is not None:
        B = int(attrs["B"])

    return dict(
        B=B,
        N_q=attrs["N_q"],
        N_kv=attrs["N_kv"],
        D=attrs["D"],
        cu_seqlens_q=attrs.get("cu_seqlens_q"),
        cu_seqlens_kv=attrs.get("cu_seqlens_kv"),
        seqused_q=attrs.get("seqused_q"),
        seqused_kv=attrs.get("seqused_kv"),
        max_seqlen_q=attrs.get("max_seqlen_q", -1),
        max_seqlen_kv=attrs.get("max_seqlen_kv", -1),
        enable_pa=enable_pa,
        kv_cache_layout=layout_kv,
        block_size=block_size,
        mask_mode=attrs.get("mask_mode", 0),
        q_scale_layout=attrs.get("layout_q_descale", "TND"),
        quant_mode=attrs.get("quant_mode", 1),
        enable_lse=attrs.get("enable_lse", 0),
        graph_path=attrs.get("graph_path", 0),
        input_layout=attrs.get("layout_q", "TND"),
        is_contiguous=True,
        device_id=0,
        softmax_scale=attrs.get("softmax_scale"),
        p_scale_value=attrs.get("p_scale_value", 1.0),
        data_range_q=1.0,
        data_range_k=1.0,
        data_range_v=1.0,
    )


def _make_dummy_tensors(shapes, dtypes):
    """构造 placeholder tensor 列表，传给 generate_qfa_mxfp8_inputs 的位置参数。

    inputs.py 的 bin 分跑路径会忽略这些占位；非 bin 路径也只读 shape 不读值。
    但函数签名要求位置参数存在，这里用零张量填充。
    """
    tensors = []
    for shape, dtype_name in zip(shapes, dtypes):
        if shape is None:
            tensors.append(None)
            continue
        dt = _torch_dtype_from_str(dtype_name)
        tensors.append(
            torch.zeros(shape, dtype=dt if dt != torch.float8_e4m3fn else torch.uint8)
        )
    return tensors


def generate_for_case(case_name, shapes, dtypes, attrs, output_dir):
    """为一个 case 生成 cpu 侧 bin 文件。"""
    logger.info("=" * 60)
    logger.info("Case: %s", case_name)
    logger.info(
        "attrs: B=N_q=%s N_kv=%s D=%s, enable_lse=%s",
        attrs.get("N_q"),
        attrs.get("N_kv"),
        attrs.get("D"),
        attrs.get("enable_lse"),
    )

    # 1. 占位 tensor（inputs.py 签名要求位置参数，但生成路径不读它们的值）
    placeholder_tensors = _make_dummy_tensors(shapes, dtypes)

    # 2. 调用 inputs 生成函数 → 缓存到 golden_mod._cached_mxfp8_inputs
    kwargs = _build_kwargs(attrs)
    inputs_module.generate_qfa_mxfp8_inputs(
        *placeholder_tensors[:8],  # q,k,v,deq_q,deq_k,deq_v,p_scale,block_table
        **kwargs,
    )
    cached = getattr(golden_mod, "_cached_mxfp8_inputs", None)
    if cached is None:
        raise RuntimeError(
            f"case {case_name}: inputs 生成后 _cached_mxfp8_inputs 仍为空"
        )

    # 3. 保存输入 bin：{case}_cpu_{idx}.bin
    # cached 顺序：q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table
    bin_input_paths = []
    for idx, tensor in enumerate(cached):
        if tensor is None:
            continue
        path = os.path.join(output_dir, f"{case_name}_cpu_{idx}.bin")
        _torch_to_bin(tensor, path)
        bin_input_paths.append(
            (idx, path, tuple(tensor.shape), _torch_dtype_name(tensor.dtype))
        )
        logger.info(
            "  saved input %d → %s (shape=%s, dtype=%s)",
            idx,
            path,
            tuple(tensor.shape),
            _torch_dtype_name(tensor.dtype),
        )

    # 4. 调用 golden 函数 + __bin_golden_out 保存 golden 输出
    enable_lse = bool(attrs.get("enable_lse", 0))
    golden_out_paths = [
        os.path.join(output_dir, f"{case_name}_golden_{i}.bin")
        for i in range(2 if enable_lse else 1)
    ]
    golden_module.cpu_qfa_mxfp8(
        *placeholder_tensors[:8],
        **kwargs,
        __bin_golden_out=golden_out_paths,
    )
    for i, path in enumerate(golden_out_paths):
        if os.path.exists(path):
            size = os.path.getsize(path)
            logger.info("  saved golden %d → %s (%d bytes)", i, path, size)

    return {
        "case": case_name,
        "inputs": bin_input_paths,
        "golden": golden_out_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="预生成 cpu 侧 bin 文件")
    parser.add_argument(
        "--csv",
        default=os.path.join(_TEST_DIR, "qfa_mxfp8_excel.csv"),
        help="输入 csv 路径（默认 qfa_mxfp8_excel.csv）",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/qfa_bins", help="bin 文件输出目录"
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
