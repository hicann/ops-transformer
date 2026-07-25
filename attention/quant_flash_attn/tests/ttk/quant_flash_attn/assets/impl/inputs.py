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
import math
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

logger = logging.getLogger(__name__)

__input__ = {"e2e": {"qfa_mxfp8_wrapper.npu_qfa_mxfp8": "generate_qfa_mxfp8_inputs"}}

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def get_cached_inputs():
    """wrapper 调用前从这里取 customize_inputs 生成的真实数据。
    存在 golden_mod 上,避免模块重复加载导致缓存丢失。
    """
    return getattr(golden_mod, "_cached_mxfp8_inputs", None)


# torch dtype 名 → (numpy 加载 dtype, torch 还原 dtype)。
# fp8/e8m0 在 numpy 中以 uint8 字节存储，加载后 view 还原为 torch fp8/e8m0 dtype。
# 调用端传 dtype 名字符串（与 ttk load_numpy_data 的 dtype 参数约定一致）。
_TORCH_DTYPE_FROM_NAME = {
    "float32": (numpy.float32, torch.float32),
    "int32": (numpy.int32, torch.int32),
    "float16": (numpy.float16, torch.float16),
    "uint8": (numpy.uint8, torch.uint8),
    "int8": (numpy.int8, torch.int8),
    "float8_e4m3fn": (numpy.uint8, torch.float8_e4m3fn),
    "float8_e8m0fnu": (numpy.uint8, torch.float8_e8m0fnu),
}


def _load_bin_tensor(path, shape, dtype_name):
    """从 numpy raw .bin 加载 tensor：numpy.fromfile + reshape + torch 还原。

    与 ttk load_numpy_data 兼容：fp8/e8m0 以 uint8 字节存储，加载后 view 为 torch fp8 dtype。
    dtype_name 是 torch dtype 的字符串名（如 'float8_e4m3fn'、'float32'、'int32'）。
    """
    if dtype_name not in _TORCH_DTYPE_FROM_NAME:
        raise ValueError(f"unsupported dtype_name: {dtype_name}")
    np_load_dtype, torch_dtype = _TORCH_DTYPE_FROM_NAME[dtype_name]
    arr = numpy.fromfile(path, dtype=np_load_dtype).reshape(shape)
    if torch_dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return torch.from_numpy(arr).view(torch_dtype)
    return torch.from_numpy(arr).to(torch_dtype)


def generate_qfa_mxfp8_inputs(
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
    """生成 BNSD BF16 Q/K/V → 量化 MXFP8 + fp32 scale + block_table,缓存到 golden_mod 上。

    bf16 张量值域由 csv input_data_ranges 控制（如 (-1,1)），再 mxfp8 量化到 fp8。
    actual_seq_q/kv 是布局转换（cu_seqlens 差分还原），不是 op 参数推导。

    Bin 分跑支持：csv attributes 中传 `__bin_inputs`（list of (path, shape, dtype_name)
    三元组）时，直接用 numpy.fromfile 加载每个 tensor，跳过 torch.rand 生成路径。
    与 cpu/npu 分跑工作流配合：cpu 跑 gen_bins.py 预生成 bin，npu 跑时通过 csv attributes
    指向 bin 路径。
    """
    # cu_seqlens → actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    # PA 模式下 cu_seqlens_kv 可能为空，用 seqused_kv 推导
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [
            cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            for i in range(len(cu_seqlens_kv) - 1)
        ]
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = [0]

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    fp8_dtype = torch.float8_e4m3fn
    group_size = 32

    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("FP8_DTYPE", fp8_dtype),
        ("QUANT_GROUP_SIZE", group_size),
    ]:
        setattr(golden_mod, gkey, gval)

    # ----- Bin 分跑路径：__bin_inputs 非空时从 numpy .bin 加载，跳过 torch.rand -----
    # __bin_inputs 是 list of (path, shape, dtype_name) 三元组；
    # dtype_name 为 torch dtype 字符串名（如 'float8_e4m3fn'、'float32'、'int32'），
    # 与 ttk load_numpy_data 的 dtype 参数约定一致。
    bin_inputs = kwargs.get("__bin_inputs")
    if bin_inputs:
        loaded = []
        for path, shape, dtype_name in bin_inputs:
            loaded.append(_load_bin_tensor(path, shape, dtype_name))
        # 顺序：q, k, v, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table
        # 与 generate_qfa_mxfp8_inputs 输出顺序一致
        golden_mod._cached_mxfp8_inputs = list(loaded)
        logger.info("[INPUTS] loaded %d tensors from bin via __bin_inputs", len(loaded))
        return  # 不调 torch.rand / torch.manual_seed

    # csv input_data_ranges 列由 ttk 作为 input_ranges kwarg 传入（list of (min,max) per tensor）
    # q/k/v 是前 3 个；csv 省略或 None → 用默认 (-1, 1)
    input_ranges = kwargs.get("input_ranges")

    def _range_of(idx, default=(-1.0, 1.0)):
        if input_ranges is None or idx >= len(input_ranges):
            return default
        r = input_ranges[idx]
        if r is None:
            return default
        return (float(r[0]), float(r[1]))

    rq_min, rq_max = _range_of(0)
    rk_min, rk_max = _range_of(1)
    rv_min, rv_max = _range_of(2)

    torch.manual_seed(_SEED_MAP["q"])
    q_bf16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    torch.manual_seed(_SEED_MAP["k"])
    k_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    torch.manual_seed(_SEED_MAP["v"])
    v_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )

    quant_scale_q = golden_mod.get_mxfp8_per_token_group_quant_scale(
        q_bf16, fp8_dtype, group_size
    )
    quant_scale_k = golden_mod.get_mxfp8_per_token_group_quant_scale(
        k_bf16, fp8_dtype, group_size
    )
    quant_scale_v = golden_mod.get_mxfp8_per_channel_group_quant_scale(
        v_bf16, fp8_dtype, group_size
    )

    dequant_scale_q = quant_scale_q
    dequant_scale_k = quant_scale_k
    dequant_scale_v = quant_scale_v

    fp8_max = 448.0
    q_fp8 = (
        golden_mod.mxfp8_per_token_group_quant(q_bf16, quant_scale_q, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    k_fp8 = (
        golden_mod.mxfp8_per_token_group_quant(k_bf16, quant_scale_k, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    v_fp8 = (
        golden_mod.mxfp8_per_channel_group_quant(v_bf16, quant_scale_v, group_size)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # p_scale 从 csv attributes.p_scale_value 读取 (默认 1.0 兼容无该 key 的旧 csv)。
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale_t = torch.tensor([float(p_scale_value)], dtype=torch.float32)

    block_table_t = None
    if enable_pa:
        block_num = sum(math.ceil(s / block_size) for s in actual_seq_kv)
        max_blocks = (
            max(math.ceil(s / block_size) for s in actual_seq_kv)
            if actual_seq_kv
            else 0
        )
        block_idx_list = torch.randperm(block_num, dtype=torch.int32)
        block_table_t = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                block_table_t[b, j] = block_idx_list[idx]
                idx += 1

    # 缓存到 golden_mod 上(不是本模块,避免 spec.py 重复加载导致缓存丢失)
    golden_mod._cached_mxfp8_inputs = [
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale_t,
        block_table_t,
    ]
    # 无返回值(符合 TTK 约定)
