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

import logging
import os
import sys
from typing import List

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..", "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as golden_mod

# try:
#     from common import golden_cache
# except ImportError:
#     import golden_cache

logger = logging.getLogger(__name__)

__golden__ = {"e2e": {"qfa_mxfp8_wrapper.npu_qfa_mxfp8": "cpu_qfa_mxfp8"}}


def _apply_golden_globals(attrs):
    """把 case 属性注入 golden 模块全局变量。"""
    mapping = {
        "B": "B",
        "N_q": "N_q",
        "N_kv": "N_kv",
        "D": "D",
        "enable_pa": "ENABLE_PA",
        "kv_cache_layout": "KV_CACHE_LAYOUT",
        "block_size": "BLOCK_SIZE",
        "mask_mode": "SPARSE_MODE",
        "q_scale_layout": "Q_SCALE_LAYOUT",
        "enable_lse": "ENABLE_LSE",
        "input_layout": "INPUT_LAYOUT",
        "is_contiguous": "IS_CONTIGUOUS",
        "device_id": "DEVICE_ID",
        "graph_path": "GRAPH_PATH",
        "cu_seqlens_q": "CU_SEQLENS_Q",
        "cu_seqlens_kv": "CU_SEQLENS_KV",
        "seqused_q": "SEQUSED_Q",
        "seqused_kv": "SEQUSED_KV",
        "max_seqlen_q": "MAX_SEQLEN_Q",
        "max_seqlen_kv": "MAX_SEQLEN_KV",
        "softmax_scale": "SOFTMAX_SCALE",
        "seed_q": "SEED_Q",
        "seed_k": "SEED_K",
        "seed_v": "SEED_V",
        "data_range_q": "DATA_RANGE_Q",
        "data_range_k": "DATA_RANGE_K",
        "data_range_v": "DATA_RANGE_V",
    }
    for attr_key, golden_key in mapping.items():
        if attr_key in attrs:
            setattr(golden_mod, golden_key, attrs[attr_key])
    setattr(golden_mod, "FP8_DTYPE", torch.float8_e4m3fn)
    setattr(golden_mod, "QUANT_GROUP_SIZE", 32)
    p_scale = attrs.get("p_scale_value", 1.0)
    if isinstance(p_scale, torch.Tensor):
        p_scale = float(p_scale.item())
    setattr(golden_mod, "P_SCALE", float(p_scale))


def _cu_seqlens_to_actual(cu_seqlens):
    """cu_seqlens → actual_seq (差分还原)。

    cu_seqlens = [0, 3, 7, 10] → actual_seq = [3, 4, 3]
    """
    if cu_seqlens is None or len(cu_seqlens) < 2:
        return []
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _load_bin_tensor(path, shape, dtype_name):
    """从 numpy raw .bin 加载 tensor：numpy.fromfile + reshape + torch 还原。

    与 ttk load_numpy_data 兼容：fp8/e8m0 以 uint8 字节存储，加载后 view 为 torch fp8 dtype。
    dtype_name 是 torch dtype 的字符串名（如 'float8_e4m3fn'、'float32'、'int32'）。
    与 assets/impl/inputs.py 的 _load_bin_tensor 保持一致（bin 分跑共用格式）。
    """
    _mapping = {
        "float32": (numpy.float32, torch.float32),
        "int32": (numpy.int32, torch.int32),
        "float16": (numpy.float16, torch.float16),
        "uint8": (numpy.uint8, torch.uint8),
        "int8": (numpy.int8, torch.int8),
        "float8_e4m3fn": (numpy.uint8, torch.float8_e4m3fn),
        "float8_e8m0fnu": (numpy.uint8, torch.float8_e8m0fnu),
    }
    if dtype_name not in _mapping:
        raise ValueError(f"unsupported dtype_name: {dtype_name}")
    np_load_dtype, torch_dtype = _mapping[dtype_name]
    arr = numpy.fromfile(path, dtype=np_load_dtype).reshape(shape)
    if torch_dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return torch.from_numpy(arr).view(torch_dtype)
    return torch.from_numpy(arr).to(torch_dtype)


def _torch_to_bin(tensor, path):
    """torch tensor → numpy raw .bin 文件（ttk load_numpy_data 兼容格式）。"""
    if tensor is None:
        raise ValueError(f"_torch_to_bin: tensor 为 None，无法保存到 {path}")
    arr = tensor.detach().cpu().contiguous()
    if arr.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        numpy_arr = arr.view(torch.uint8).numpy()
    else:
        numpy_arr = arr.numpy()
    numpy_arr.tofile(path)


def cpu_qfa_mxfp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    p_scale: torch.Tensor,
    block_table: torch.Tensor,
    *,
    B: int,
    N_q: int,
    N_kv: int,
    D: int,
    cu_seqlens_q: List[int],
    cu_seqlens_kv: List[int],
    seqused_q: List[int],
    seqused_kv: List[int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    enable_pa: bool,
    kv_cache_layout: str,
    block_size: int,
    mask_mode: int,
    q_scale_layout: str,
    quant_mode: int = 1,
    enable_lse: bool = False,
    graph_path: int = 0,
    input_layout: str = "TND",
    is_contiguous: bool = True,
    device_id: int = 0,
    softmax_scale: float = None,
    data_range_q: float = 1.0,
    data_range_k: float = 1.0,
    data_range_v: float = 1.0,
    **kwargs,
):
    """CPU golden:从 golden_mod 缓存取真实 FP8 数据,调 cpu_mxfp8_golden 算参考输出。

    Bin 分跑支持：
    - `__bin_golden` 非空时（list of (path, shape, numpy_dtype)），直接从 bin 加载 golden
      输出，跳过 CPU 计算。用于 cpu/npu 分跑：npu 侧直接用 cpu 预生成的 golden bin。
    - `__bin_golden_out` 非空时（list of path），CPU 计算完后把 golden 输出按顺序写入
      bin 路径。用于 cpu 侧预生成 golden bin 供 npu 侧读取。
    """
    # csv precision_tolerances / absolute_precision 经 testcase.attributes → kwargs 传入,
    # 暂存到 golden_mod 供 compare 插件读取（ttk 不把 testcase 直接传给 custom compare）。
    golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")

    # ----- Bin 分跑路径：__bin_golden 非空时从 numpy .bin 加载 golden，跳过 CPU 计算 -----
    # __bin_golden 是 list of (path, shape, dtype_name) 三元组（与 __bin_inputs 同格式）。
    bin_golden = kwargs.get("__bin_golden")
    if bin_golden:
        loaded = []
        for path, shape, dtype_name in bin_golden:
            loaded.append(_load_bin_tensor(path, shape, dtype_name))
        logger.info(
            "[GOLDEN] loaded %d golden tensors from bin via __bin_golden", len(loaded)
        )
        return loaded

    # 从 golden_mod 取 customize_inputs 缓存的真实 FP8 数据
    cached = getattr(golden_mod, "_cached_mxfp8_inputs", None)
    if cached is not None:
        (
            q,
            k,
            v,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
        ) = cached[:8]

    # actual_seq (有效长度) 用于 CPU golden 的 attention mask 构建——必须与 NPU 侧
    # prepare_npu_inputs 里的 _actual_seq_q/kv() 取值逻辑一致：
    #   优先用 seqused (实际有效长度，可小于 padded 范围)；
    #   seqused 为 None 时才用 cu_seqlens 差分 (物理 TND 范围)。
    # 设计文档 (3.2.3): cu_seqlens 提供物理 token 起止，seqused 提供实际有效长度，
    # 两者可同时传入且 seqused <= cu_seqlens 差分。mask 必须基于 seqused 截断，
    # 否则 CPU 会把 padding 位置当有效 token 算 (causal delta 错误) 导致大面积 mismatch。
    actual_seq_q = (
        list(seqused_q)
        if (seqused_q is not None and len(seqused_q) > 0)
        else _cu_seqlens_to_actual(cu_seqlens_q)
    )
    actual_seq_kv = (
        list(seqused_kv)
        if (seqused_kv is not None and len(seqused_kv) > 0)
        else _cu_seqlens_to_actual(cu_seqlens_kv)
    )

    # cu_seqlens_q/kv 需要转为 list(传入时可能是 tuple 或其他类型)
    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv_list = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]

    _apply_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "cu_seqlens_q": cu_seqlens_q_list,
            "cu_seqlens_kv": cu_seqlens_kv_list,
            "seqused_q": list(seqused_q) if seqused_q is not None else None,
            "seqused_kv": list(seqused_kv) if seqused_kv is not None else None,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "enable_pa": enable_pa,
            "kv_cache_layout": kv_cache_layout,
            "block_size": block_size,
            "mask_mode": mask_mode,
            "q_scale_layout": q_scale_layout,
            "enable_lse": enable_lse,
            "input_layout": input_layout,
            "is_contiguous": is_contiguous,
            "device_id": device_id,
            "graph_path": graph_path,
            "p_scale_value": p_scale,
        }
    )

    cpu_out, cpu_lse = golden_mod.cpu_mxfp8_golden(
        q,
        k,
        v,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        actual_seq_q,
        actual_seq_kv,
        softmax_scale=softmax_scale,
    )

    compare_layout = "TND" if enable_pa else input_layout
    cpu_out_aligned = golden_mod.convert_q_bnsd_to_layout(
        cpu_out,
        actual_seq_q,
        compare_layout,
        cu_seqlens=cu_seqlens_q if enable_pa else None,
    )

    if enable_lse:
        cpu_lse_aligned = golden_mod.convert_q_bnsd_to_layout(
            cpu_lse,
            actual_seq_q,
            compare_layout,
            cu_seqlens=cu_seqlens_q if enable_pa else None,
        )
        # TND padding 位置填 inf 以匹配 NPU 行为：NPU 对超出实际序列长度的 Q 位置输出 inf LSE
        if enable_pa and cu_seqlens_q is not None:
            cpu_lse_aligned = golden_mod.fill_tnd_padding(
                cpu_lse_aligned,
                actual_seq_q,
                list(cu_seqlens_q),
                fill_value=float("inf"),
            )
        result = [cpu_out_aligned, cpu_lse_aligned]

    else:
        result = [cpu_out_aligned]

    # ----- Bin 保存路径：__bin_golden_out 非空时按顺序写入 bin 文件 -----
    bin_golden_out = kwargs.get("__bin_golden_out")
    if bin_golden_out:
        if len(bin_golden_out) != len(result):
            raise ValueError(
                f"__bin_golden_out 长度 {len(bin_golden_out)} 与 golden 输出数 {len(result)} 不一致"
            )
        for path, tensor in zip(bin_golden_out, result):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            _torch_to_bin(tensor, path)
        logger.info(
            "[GOLDEN] saved %d golden tensors to bin via __bin_golden_out", len(result)
        )

    return result
