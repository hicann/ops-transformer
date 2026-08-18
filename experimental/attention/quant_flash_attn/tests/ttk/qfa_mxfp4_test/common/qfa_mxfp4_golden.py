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

import gc
import logging
import math
import os
import sys

import torch
import torch.nn as nn
import torch_npu

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

from . import mx_quant_fp4_tool as _mx_tool
from . import flash_attention_cpu_golden as _cpu_golden_mod

mxfp4_quantize_pack_last = _mx_tool.mxfp4_quantize_pack_last
flash_attention_cpu_golden_varlen = _cpu_golden_mod.flash_attention_cpu_golden_varlen

try:
    from . import result_compare_method
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置区: wrapper 通过 _apply_golden_globals 注入 case 参数覆盖默认值
# GRAPH_PATH 优先从环境变量读, 支持 GRAPH_PATH=7 python3 -m ttk e2e ...
# ==============================================================================
import os as _os

GRAPH_PATH = int(_os.environ.get("GRAPH_PATH", "0"))

B = 1
N_q = 1  # query heads = n2 * g
N_kv = 1  # kv heads = n2
G = 1  # GQA group size = N_q // N_kv
D = 128  # qk_d
V_D = 128  # v_d, 默认等于 D
Rope_D = 0

ACT_SEQ_LENS_Q = [4]
ACT_SEQ_LENS_KV = [1024]
CU_SEQLENS_Q = [0, 4]
CU_SEQLENS_KV = [0, 1024]
MAX_SEQLEN_Q = 4
MAX_SEQLEN_KV = 1024

INPUT_LAYOUT = "BNSD"  # BNSD / BSND / TND
LAYOUT_Q_DESCALE = "BSND"  # q descale layout
LAYOUT_KV = "BNSD"  # kv layout (默认与 INPUT_LAYOUT 一致)
LAYOUT_OUT = "BNSD"  # attn out layout (默认与 INPUT_LAYOUT 一致)
KV_STORAGE_MODE = "continue"  # continue / pa_bbh / pa_bnbd / pa_nz
BLOCK_SIZE = 0  # PA 模式才需要

Q_DTYPE = "fp4_e2m1"
KV_DTYPE = "fp4_e2m1"
OUT_DTYPE = "bfloat16"

Q_QUANT_MODE = 3  # MXFP4 固定 3
SPARSE_MODE = 0
PRE_TOKENS = 2147483647
NEXT_TOKENS = 2147483647
ENABLE_MASK = False
ENABLE_LSE = False
INNER_PRECISE = 0

SOFTMAX_SCALE = None
DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0

DEVICE_ID = 0

# 可选 tensor 入参 (shape 为空 -> 传 None; 有 shape -> generate_data 按形状生成随机 tensor)
BLOCK_TABLE_SHAPE = []
BLOCK_TABLE_DTYPE = None
P_SCALE_VALUE = None  # 标量 p_scale (表格传值则用之, 否则按 shape 随机生成)
P_SCALE_SHAPE = []
P_SCALE_DTYPE = None
P_SCALE_DATARANGE = None
SINKS_SHAPE = []
SINKS_DTYPE = None
SINKS_DATARANGE = None
ATTN_MASK_SHAPE = []
ATTN_MASK_DTYPE = None
ATTN_MASK_DATARANGE = None

# dtype 透传 (表格不传 -> None, generate_data / NPU 调用时用默认值)
# descale 默认 uint8 (E8M0 scale); seqused/cu_seqlens 默认 int32; softmax_lse 默认 float32
Q_DESCALE_DTYPE = None
K_DESCALE_DTYPE = None
V_DESCALE_DTYPE = None
SEQUSED_Q_DTYPE = None
SEQUSED_KV_DTYPE = None
CU_SEQLENS_Q_DTYPE = None
CU_SEQLENS_KV_DTYPE = None
SOFTMAX_LSE_DTYPE = None

QUANT_MODE_MXFP4 = 3
FP4_DTYPE_UINT8 = torch.uint8  # 打包后 fp4 张量用 uint8 存
SCALE_DTYPE_UINT8 = torch.uint8  # e8m0 scale 用 uint8 存
QUANT_GROUP_SIZE = 32

E8M0_BIAS = 127
# e8m0 没有 0 值语义, 0 byte 在 NPU 侧会被解释为 NaN
E8M0_MIN_POSITIVE_EXP = -127
E8M0_MIN_POSITIVE_BYTE = E8M0_BIAS + E8M0_MIN_POSITIVE_EXP  # 0

SEED_Q = 54
SEED_K = 3
SEED_V = 4

# manual-data replay 优化 slot: wrapper 在 replay 时把 bin 加载的 6 个 packed
# tensor 注入这里, generate_data() 检测到非空就跳过 FP32 生成 + mxfp4 量化 +
# rearrange (大 case 几百秒), 直接用 bin 数据; 其余字段仍按原逻辑生成。
# 非 replay 场景保持空 dict, 走原路径, 零影响。
_REPLAY_QKV_OVERRIDE = {}

# 物理 S override: wrapper/inputs 从 CSV 分配的 q/k/v tensor shape 推导物理 S 后注入,
# generate_data 优先用这两个值作为物理 S (生成 query/key/value 的 S 维),
# 而不是 max(ACT_SEQ_LENS) / MAX_SEQLEN (后者只决定有效长度).
# 非 override 场景保持 None, 走原逻辑 (物理 S = 有效长度).
_PHYSICAL_S_Q_OVERRIDE = None
_PHYSICAL_S_KV_OVERRIDE = None


def _infer_physical_s_from_tensor(tensor, layout):
    """从 packed tensor shape 推导物理 S (BNSD 的 S 维).

    Args:
        tensor: packed q/k/v tensor (最后一维已 //2)
        layout: query_layout / kv_layout (BSND/BNSD/BSH/TND)

    Returns:
        物理 S (int), 无法推导时返回 None (如 TND 的 T = sum(act) != 物理 S)
    """
    if tensor is None:
        return None
    layout_upper = (layout or "").upper()
    try:
        if layout_upper in ("BSND", "BSH"):
            if tensor.dim() < 2:
                return None
            return int(tensor.shape[1])
        if layout_upper == "BNSD":
            if tensor.dim() < 3:
                return None
            return int(tensor.shape[2])
    except (IndexError, TypeError):
        return None
    # TND: (T, N, D/2), T = sum(act), 无法反推单 batch 物理 S
    return None


def _resolve_s_with_effective(
    act_seq_lens, max_seqlen, physical_s_override, b_size, tag
):
    """解析物理 S 和有效长度, 并做校验.

    优先级:
        有效长度: max(act_seq_lens) if act_seq_lens else max_seqlen
        物理 S: override > 有效长度

    校验:
        - act_seq_lens 和 max_seqlen 都未传 -> 报错
        - 物理 S < 有效长度 -> 报错

    Returns:
        (s_physical, s_effective, act_seq_effective)
    """
    if act_seq_lens:
        s_effective = max(act_seq_lens)
        act_seq_effective = list(act_seq_lens)
    elif max_seqlen is not None and max_seqlen >= 0:
        s_effective = int(max_seqlen)
        act_seq_effective = [s_effective] * b_size
    else:
        raise RuntimeError(
            f"{tag}: act_seq_lens 和 max_seqlen 都未传, 无法确定有效长度"
        )

    if physical_s_override is not None:
        s_physical = int(physical_s_override)
        if s_physical < s_effective:
            raise RuntimeError(
                f"{tag}: 物理 S ({s_physical}) < 有效长度 ({s_effective}), "
                f"tensor 的 S 维不足以容纳有效数据"
            )
    else:
        s_physical = s_effective

    return s_physical, s_effective, act_seq_effective


def _inject_physical_s_override(q_tensor, v_tensor, query_layout, kv_layout):
    """从 q/v packed tensor shape 推导物理 S 并注入 override.

    在 generate_data 之前调用; generate_data 内部 _resolve_s_with_effective
    会优先使用这两个值作为物理 S. 调用方负责在 generate_data 之后调
    _clear_physical_s_override 清空, 避免污染同进程后续 case.
    """
    global _PHYSICAL_S_Q_OVERRIDE, _PHYSICAL_S_KV_OVERRIDE
    _PHYSICAL_S_Q_OVERRIDE = _infer_physical_s_from_tensor(q_tensor, query_layout)
    _PHYSICAL_S_KV_OVERRIDE = _infer_physical_s_from_tensor(v_tensor, kv_layout)


def _clear_physical_s_override():
    """清空物理 S override (generate_data 之后调用)."""
    global _PHYSICAL_S_Q_OVERRIDE, _PHYSICAL_S_KV_OVERRIDE
    _PHYSICAL_S_Q_OVERRIDE = None
    _PHYSICAL_S_KV_OVERRIDE = None


_CSV_SHAPE_OVERRIDE_SLOTS = ("q", "k", "v", "q_descale", "k_descale", "v_descale")


def _adapt_tensor_to_shape(src, dst_shape):
    dst_shape = tuple(int(x) for x in dst_shape)
    src_flat = src.flatten()
    dst_numel = 1
    for d in dst_shape:
        dst_numel *= d
    if src_flat.numel() >= dst_numel:
        adapted = src_flat[:dst_numel]
    else:
        adapted = torch.zeros(dst_numel, dtype=src.dtype, device=src.device)
        adapted[: src_flat.numel()] = src_flat
    return adapted.reshape(dst_shape).to(src.dtype)


def _apply_csv_shape_override(data_dict, csv_tensors):
    for name in _CSV_SHAPE_OVERRIDE_SLOTS:
        csv_t = csv_tensors.get(name)
        if csv_t is None:
            continue
        golden_t = data_dict.get(name)
        if golden_t is None:
            continue
        if tuple(csv_t.shape) == tuple(golden_t.shape):
            continue
        logger.warning(
            "[GOLDEN] %s CSV shape %s != golden shape %s, 用 CSV shape 重构",
            name,
            tuple(csv_t.shape),
            tuple(golden_t.shape),
        )
        data_dict[name] = _adapt_tensor_to_shape(golden_t, csv_t.shape).contiguous()


def _get_npu_fa_kwargs():
    return dict(
        return_softmax_lse=ENABLE_LSE,
    )


# ==============================================================================
# 配置注入: wrapper 把 case attrs 转成模块全局变量
# ==============================================================================
_GOLDEN_GLOBALS_MAP = {
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
    "cu_seqlens_q": "CU_SEQLENS_Q",
    "cu_seqlens_kv": "CU_SEQLENS_KV",
    "max_seqlen_q": "MAX_SEQLEN_Q",
    "max_seqlen_kv": "MAX_SEQLEN_KV",
    # 可选 tensor 入参
    "block_table_shape": "BLOCK_TABLE_SHAPE",
    "block_table_dtype": "BLOCK_TABLE_DTYPE",
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
    # dtype 透传 (表格不传 -> None, 用默认值)
    "q_descale_dtype": "Q_DESCALE_DTYPE",
    "k_descale_dtype": "K_DESCALE_DTYPE",
    "v_descale_dtype": "V_DESCALE_DTYPE",
    "seqused_q_dtype": "SEQUSED_Q_DTYPE",
    "seqused_kv_dtype": "SEQUSED_KV_DTYPE",
    "cu_seqlens_q_dtype": "CU_SEQLENS_Q_DTYPE",
    "cu_seqlens_kv_dtype": "CU_SEQLENS_KV_DTYPE",
    "softmax_lse_dtype": "SOFTMAX_LSE_DTYPE",
}


def _apply_golden_globals(attrs):
    """把 case attributes 注入 golden 模块全局变量"""
    for attr_key, golden_key in _GOLDEN_GLOBALS_MAP.items():
        if attr_key in attrs:
            setattr(golden_mod_self, golden_key, attrs[attr_key])


# ==============================================================================
# Layout 工具: 复用 mxfp4 quant_flash_attn_golden.py 中同名函数
# ==============================================================================
def get_query_layout(input_layout):
    if input_layout in ("BSH", "BSH_BNSD", "BSH_NBSD"):
        return "BSH"
    elif input_layout in ("BSND", "BSND_BNSD", "BSND_NBSD"):
        return "BSND"
    elif input_layout in ("BNSD", "BNSD_BSND", "BNSD_NBSD"):
        return "BNSD"
    elif input_layout in ("TND", "TND_NTD"):
        return "TND"
    elif input_layout in ("NTD", "NTD_TND"):
        return "NTD"
    return None


def get_attn_out_layout(input_layout):
    if input_layout == "BSH":
        return "BSH"
    elif input_layout in ("BSND", "BNSD_BSND"):
        return "BSND"
    elif input_layout in ("BNSD", "BSH_BNSD", "BSND_BNSD"):
        return "BNSD"
    elif input_layout in ("BSH_NBSD", "BSND_NBSD", "BNSD_NBSD"):
        return "NBSD"
    elif input_layout in ("TND", "NTD_TND"):
        return "TND"
    elif input_layout in ("NTD", "TND_NTD"):
        return "NTD"
    return None


def get_softmax_lse_layout(input_layout):
    if input_layout in ("TND", "NTD_TND", "NTD", "TND_NTD"):
        return "TND"
    return "BNSD"


def bnsd_to_bsh(bnsd_tensor):
    return bnsd_tensor.permute(0, 2, 1, 3).flatten(start_dim=2)


def bnsd_to_bsnd(bnsd_tensor):
    return bnsd_tensor.permute(0, 2, 1, 3)


def bnsd_to_tnd(bnsd_tensor, b, act_seq_lens):
    if act_seq_lens is None:
        return bnsd_tensor.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)
    elif len(act_seq_lens) == 1:
        return (
            torch.narrow(bnsd_tensor, dim=2, start=0, length=act_seq_lens[0])
            .permute(0, 2, 1, 3)
            .flatten(start_dim=0, end_dim=1)
        )
    t = sum(act_seq_lens)
    tnd_tensor = torch.empty(
        t, bnsd_tensor.shape[1], bnsd_tensor.shape[3], dtype=bnsd_tensor.dtype
    )
    t_idx = 0
    for i in range(b):
        if act_seq_lens[i] > 0:
            tnd_tensor[t_idx : (t_idx + act_seq_lens[i]), :, :] = bnsd_tensor[
                i, :, 0 : act_seq_lens[i], :
            ].permute(1, 0, 2)
            t_idx = t_idx + act_seq_lens[i]
    return tnd_tensor


def rearrange_by_layout(bnsd_tensor, layout, b, act_seq_lens):
    if layout == "BNSD":
        return bnsd_tensor
    elif layout == "BSH":
        return bnsd_to_bsh(bnsd_tensor)
    elif layout == "BSND":
        return bnsd_to_bsnd(bnsd_tensor)
    elif layout == "TND":
        return bnsd_to_tnd(bnsd_tensor, b, act_seq_lens)
    return None


def update_act_seq_lens_for_tnd(layout, b, act_seq_lens):
    cu_seqlens = None
    # 空列表视为 None (TND 必须有 seqused, 空则不算 cu_seqlens)
    if act_seq_lens and layout in ("TND", "NTD"):
        cu_seqlens = [0] * (b + 1)
        seq_list = (
            act_seq_lens.tolist()
            if hasattr(act_seq_lens, "tolist")
            else list(act_seq_lens)
        )
        for i in range(b):
            cu_seqlens[i + 1] = cu_seqlens[i] + seq_list[i]
    return cu_seqlens


def transpose_kscale(k_scale, layout):
    if k_scale is None:
        raise ValueError("k_scale cannot be None")
    if not isinstance(k_scale, torch.Tensor):
        raise TypeError("k_scale must be torch.Tensor")
    if layout == "TND":
        if k_scale.dim() != 3:
            raise ValueError(
                f"TND layout requires 3D tensor [T,N,D], got dim={k_scale.dim()}"
            )
        T, N, D = k_scale.shape
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2")
        return k_scale.reshape(T, N, D // 2, 2)
    elif layout == "BSND":
        B, S, N, D = k_scale.shape
        return k_scale.contiguous().reshape(B, S, N, D // 2, 2)
    elif layout == "BNSD":
        B, N, S, D = k_scale.shape
        return k_scale.contiguous().reshape(B, N, S, D // 2, 2)
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def transpose_qscale(q_scale, layout, n2):
    """Q scale 重排: BNSD/TND/BSND -> N2TGD-like (n2,B/S,T,G,D//2,2)"""
    if q_scale is None:
        raise ValueError("q_scale cannot be None")
    if not isinstance(q_scale, torch.Tensor):
        raise TypeError("q_scale must be torch.Tensor")
    if layout == "TND":
        if q_scale.dim() != 3:
            raise ValueError(
                f"TND layout requires 3D tensor [T,N,D], got dim={q_scale.dim()}"
            )
        T, N, D = q_scale.shape
        if N % n2 != 0:
            raise ValueError(f"Total head N={N} must be divisible by n2={n2}")
        G = N // n2
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2")
        q = q_scale.reshape(T, n2, G, D)
        q = q.reshape(T, n2, G, D // 2, 2)
        return q.permute(1, 0, 2, 3, 4)
    elif layout == "BSND":
        if q_scale.dim() != 4:
            raise ValueError(
                f"BSND layout requires 4D tensor [B,S,N,D], got dim={q_scale.dim()}"
            )
        B, S1, N, D = q_scale.shape
        if N % n2 != 0:
            raise ValueError(f"Total head N={N} must be divisible by n2={n2}")
        G = N // n2
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2 for MXFP4 pack")
        tmp = q_scale.reshape(B, S1, n2, G, D)
        tmp = tmp.reshape(B, S1, n2, G, D // 2, 2)
        return tmp.permute(2, 0, 1, 3, 4, 5)
    elif layout == "BNSD":
        B, N, S1, D = q_scale.shape
        return q_scale.contiguous().reshape(B, N, S1, D // 2, 2)
    else:
        raise ValueError(
            f"Unsupported layout: {layout}, only support TND, BSND and BNSD"
        )


def get_dtype(data_type):
    """dtype 字符串 -> torch.dtype. 支持 float16/bfloat16/float32/int8/int32/uint8 及大小写变体."""
    if data_type is None:
        return None
    s = str(data_type).strip().lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    elif s in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif s in ("float32", "fp32", "float"):
        return torch.float32
    elif s in ("int8",):
        return torch.int8
    elif s in ("int32",):
        return torch.int32
    elif s in ("uint8", "float8_e8m0"):
        return torch.uint8
    return None


def _parse_datarange(val):
    if val is None:
        return (-1.0, 1.0)
    if isinstance(val, (int, float)):
        f = float(val)
        return (-f, f)
    s = str(val).strip()
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 1:
        f = float(parts[0])
        return (-f, f)
    if len(parts) >= 2:
        return (float(parts[0]), float(parts[1]))
    return (-1.0, 1.0)


def _gen_range_tensor(shape, lo, hi, seed):
    """randn(seed) -> clamp 到 [lo, hi] -> float32 tensor."""
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.float32)
    return t.clamp_(min=lo, max=hi)


def _gen_opt_tensor(shape, dtype_str, datarange_str, seed):
    if not shape:
        return None
    # 解析 dtype
    dt_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "int32": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    torch_dtype = dt_map.get((dtype_str or "").lower(), torch.float32)
    # 解析 datarange
    lo, hi = _parse_datarange(datarange_str)

    torch.manual_seed(seed)
    if torch_dtype.is_floating_point:
        t = torch.randn(shape, dtype=torch_dtype)
        t = t.clamp_(min=lo, max=hi)
    else:
        t = torch.randint(int(lo), int(hi) + 1, shape, dtype=torch_dtype)
    return t.contiguous()


def generate_data():
    n2 = N_kv
    g = G
    num_heads = n2 * g
    # 有效长度: act 传入时用 act (不看 max_seqlen), 否则用 max_seqlen (每 batch 等长);
    # 都没传则报错. 物理 S 优先用 override (从 q/k/v tensor shape 推导), 否则等于有效长度.
    # 校验: 物理 S >= 有效长度 (tensor S 维必须能容纳有效数据).
    s1, s1_effective, act_seq_q_eff = _resolve_s_with_effective(
        ACT_SEQ_LENS_Q, MAX_SEQLEN_Q, _PHYSICAL_S_Q_OVERRIDE, B, "Q"
    )
    s2, s2_effective, act_seq_kv_eff = _resolve_s_with_effective(
        ACT_SEQ_LENS_KV, MAX_SEQLEN_KV, _PHYSICAL_S_KV_OVERRIDE, B, "KV"
    )
    v_d = V_D
    qk_d = D

    query_layout = get_query_layout(INPUT_LAYOUT)
    # attn_out_layout: 优先用表格 LAYOUT_OUT, 否则从 INPUT_LAYOUT 推导
    attn_out_layout = (
        get_attn_out_layout(LAYOUT_OUT)
        if LAYOUT_OUT
        else get_attn_out_layout(INPUT_LAYOUT)
    )
    # kv_layout: 优先用表格 LAYOUT_KV, 否则等于 query_layout
    kv_layout = get_query_layout(LAYOUT_KV) if LAYOUT_KV else query_layout

    # manual-data replay 优化: wrapper 注入了预生成 bin tensor 时,
    # 跳过 FP32 生成 + mxfp4 量化 + rearrange (大 case 几百秒),
    # 直接用 bin 数据; 其余字段 (block_table/p_scale/sinks/attn_mask/
    # cu_seqlens/layouts/num_heads/softmax_scale/fp32_bnsd) 仍按原逻辑生成。
    # fp32_bnsd 在此路径下置 None (replay 不走 CPU golden, 用 bin 里的 golden)。
    override = _REPLAY_QKV_OVERRIDE
    if override:
        query_packed = override["q"]
        key_packed = override["k"]
        value_packed = override["v"]
        q_descale = override["q_descale"]
        k_descale = override["k_descale"]
        v_descale = override["v_descale"]
        fp32_bnsd = None
    else:
        # Q/K/V FP32 数据按 DATA_RANGE_Q/K/V 生成: randn 后 clamp 到 [lo, hi]
        # (DATA_RANGE_* 来自 wrapper 注入, 支持 float 对称区间或 'min,max' 字符串)
        q_lo, q_hi = _parse_datarange(DATA_RANGE_Q)
        k_lo, k_hi = _parse_datarange(DATA_RANGE_K)
        v_lo, v_hi = _parse_datarange(DATA_RANGE_V)
        query = _gen_range_tensor((B, num_heads, s1, qk_d), q_lo, q_hi, SEED_Q)
        key = _gen_range_tensor((B, n2, s2, qk_d), k_lo, k_hi, SEED_K)
        value = _gen_range_tensor((B, n2, s2, v_d), v_lo, v_hi, SEED_V)

        # 原始 FP32 BNSD Q/K/V 随返回 dict 传给 CPU golden (与原 pytest cpu_golden_qkv_mxfp4_flash_attn
        # 一致: golden 接收 FP32, 内部 _apply_input_quant 做量化反量化)
        fp32_bnsd = (query, key, value)

        # Q/K padding 区域 (act_seq 之外) FP32 置 0: 量化后 packed=0/scale=127,
        # 避免 NPU 按 actSeq 向上取整分块时 padding 脏数据污染有效区域.
        # V 的 padding 由后续 per-batch 量化处理 (packed 填 0, scale 填 127).
        for b_idx in range(B):
            sq = min(act_seq_q_eff[b_idx], s1)
            if sq < s1:
                query[b_idx, :, sq:, :] = 0
            sk = min(act_seq_kv_eff[b_idx], s2)
            if sk < s2:
                key[b_idx, :, sk:, :] = 0

        # Q/K: 沿 D 维量化, V: 沿 S 维量化
        query_packed, q_descale = mxfp4_quantize_pack_last(
            query, quant_axis=-1, mode="baseline"
        )
        key_packed, k_descale = mxfp4_quantize_pack_last(
            key, quant_axis=-1, mode="baseline"
        )

        # V 按 act_seq_lens_kv 分批量化: padding 区域 packed 填 0, descale 只覆盖有效区域.
        # v_descale 的 S 维基于 s2_effective (= max(act_kv)), 与 NPU kernel stride
        # (基于 maxSeqlenKv) 及 metadata 校验 (ceil(max(seqused_kv)/64)) 对齐.
        n_blocks_effective = (s2_effective + 31) // 32
        packed_value_list = []
        v_descale_list = []
        for b_idx in range(B):
            sk = min(act_seq_kv_eff[b_idx], s2)
            v_b = value[b_idx : b_idx + 1, :, :sk, :]
            v_b_packed, v_b_scale = mxfp4_quantize_pack_last(
                v_b, quant_axis=-2, mode="baseline"
            )
            packed_padded = torch.zeros((1, n2, s2, v_d // 2), dtype=v_b_packed.dtype)
            packed_padded[:, :, :sk, :] = v_b_packed
            packed_value_list.append(packed_padded)
            n_blocks_actual = v_b_scale.shape[2]
            scale_padded = torch.full(
                (1, n2, n_blocks_effective, v_d), 127, dtype=v_b_scale.dtype
            )
            scale_padded[:, :, :n_blocks_actual, :] = v_b_scale
            v_descale_list.append(scale_padded)
        value_packed = torch.cat(packed_value_list, dim=0)
        v_descale = torch.cat(v_descale_list, dim=0)

        # V scale: n_blocks 奇数时 pad 一个 block, 然后 reshape (Sg//2, D*2)
        if v_descale.shape[2] % 2 != 0:
            pad_block = torch.full(
                (v_descale.shape[0], v_descale.shape[1], 1, v_descale.shape[3]),
                127,
                dtype=v_descale.dtype,
            )
            v_descale = torch.cat([v_descale, pad_block], dim=2)
        v_descale = (
            v_descale.reshape(
                v_descale.shape[0],
                v_descale.shape[1],
                v_descale.shape[2] // 2,
                2,
                v_descale.shape[3],
            )
            .transpose(-1, -2)
            .reshape(
                v_descale.shape[0],
                v_descale.shape[1],
                v_descale.shape[2] // 2,
                v_descale.shape[3] * 2,
            )
        )

        # Layout 转换 (Q packed + scale 用 query_layout; K/V packed + scale 用 kv_layout)
        query_packed = rearrange_by_layout(query_packed, query_layout, B, act_seq_q_eff)
        key_packed = rearrange_by_layout(key_packed, kv_layout, B, act_seq_kv_eff)
        value_packed = rearrange_by_layout(value_packed, kv_layout, B, act_seq_kv_eff)

        q_descale = rearrange_by_layout(q_descale, query_layout, B, act_seq_q_eff)
        q_descale = transpose_qscale(q_descale, query_layout, n2)

        aligned_seq_lens_kv_k = [(x + 31) // 32 for x in act_seq_kv_eff]
        k_descale = rearrange_by_layout(k_descale, kv_layout, B, aligned_seq_lens_kv_k)
        k_descale = transpose_kscale(k_descale, kv_layout)

        aligned_seq_lens_kv_v = [(x + 63) // 64 for x in act_seq_kv_eff]
        v_descale = rearrange_by_layout(v_descale, kv_layout, B, aligned_seq_lens_kv_v)

        # V descale 最后再 view 出 (*shape, -1, 2) 末尾两维
        v_descale = v_descale.view(*v_descale.shape[:-1], -1, 2)

    # cu_seqlens (TND only): 表格传了就用表格的, 否则用有效长度自动推导
    cu_seqlens_q = (
        CU_SEQLENS_Q
        if CU_SEQLENS_Q
        else update_act_seq_lens_for_tnd(query_layout, B, act_seq_q_eff)
    )
    cu_seqlens_kv = (
        CU_SEQLENS_KV
        if CU_SEQLENS_KV
        else (
            update_act_seq_lens_for_tnd(query_layout, B, act_seq_kv_eff)
            if KV_STORAGE_MODE == "continue"
            else None
        )
    )

    # 可选 tensor 入参: 有 shape 则按形状生成随机 tensor, 无 shape 传 None
    # 默认 dtype: block_table=int32, p_scale=fp32, sinks=fp32, attn_mask=int8
    block_table_t = _gen_opt_tensor(
        BLOCK_TABLE_SHAPE, BLOCK_TABLE_DTYPE or "int32", None, seed=100
    )
    # p_scale: 表格传了标量值 (P_SCALE_VALUE) 则用之 (按 P_SCALE_DTYPE 生成标量 tensor);
    #         否则按 P_SCALE_SHAPE 随机生成 (P_SCALE_SHAPE 为空 -> None)
    if P_SCALE_VALUE is not None:
        p_scale_dt = get_dtype(P_SCALE_DTYPE) or torch.float32
        p_scale_t = torch.tensor([float(P_SCALE_VALUE)], dtype=p_scale_dt).reshape(
            [1] * len(P_SCALE_SHAPE or [1])
        )
        if P_SCALE_SHAPE:
            p_scale_t = p_scale_t.reshape(P_SCALE_SHAPE)
    else:
        p_scale_t = _gen_opt_tensor(
            P_SCALE_SHAPE, P_SCALE_DTYPE or "float32", P_SCALE_DATARANGE, seed=101
        )
    sinks_t = _gen_opt_tensor(
        SINKS_SHAPE, SINKS_DTYPE or "float32", SINKS_DATARANGE, seed=102
    )
    attn_mask_t = _gen_opt_tensor(
        ATTN_MASK_SHAPE, ATTN_MASK_DTYPE or "int8", ATTN_MASK_DATARANGE, seed=103
    )

    return dict(
        q=query_packed.contiguous(),
        k=key_packed.contiguous(),
        v=value_packed.contiguous(),
        q_descale=q_descale.contiguous(),
        k_descale=k_descale.contiguous(),
        v_descale=v_descale.contiguous(),
        block_table=block_table_t,
        p_scale=p_scale_t,
        sinks=sinks_t,
        attn_mask=attn_mask_t,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        act_seq_lens_q=list(ACT_SEQ_LENS_Q),
        act_seq_lens_kv=list(ACT_SEQ_LENS_KV),
        s1_physical=s1,
        s2_physical=s2,
        s1_effective=s1_effective,
        s2_effective=s2_effective,
        act_seq_q_eff=act_seq_q_eff,
        act_seq_kv_eff=act_seq_kv_eff,
        query_layout=query_layout,
        kv_layout=kv_layout,
        attn_out_layout=attn_out_layout,
        num_heads=num_heads,
        num_key_value_heads=n2,
        softmax_scale=SOFTMAX_SCALE
        if SOFTMAX_SCALE is not None
        else 1.0 / math.sqrt(qk_d),
        fp32_bnsd=fp32_bnsd,
    )


# ==============================================================================
# CPU Golden: 复用 flash_attention_cpu_golden_varlen, 与原 run_fia_eager 一致
# 输入: generate_data() 输出的 dict (其中 q/k/v 是 MXFP4 packed 字节, 这里
#       不做反量化, 直接喂给 cpu golden, 由其内部 mxfp4_qdq 处理)
#       FP32 BNSD Q/K/V 通过 dict["fp32_bnsd"] 传入 (确定性种子生成, 无跨用例缓存)
# 输出: attn_out 在 attn_out_layout 下, 已转好 layout
# ==============================================================================
def cpu_mxfp4_golden(data_dict):
    act_seq_q = data_dict["act_seq_lens_q"]
    act_seq_kv = data_dict["act_seq_lens_kv"]
    softmax_scale = data_dict["softmax_scale"]
    attn_out_layout = data_dict["attn_out_layout"]
    b = B
    n1 = data_dict["num_heads"]
    qk_d = D
    # 物理 S: 优先用 generate_data 解析的 s1_physical/s2_physical (从 q/k/v tensor shape 来),
    # 否则回退到有效长度 (act 或 max_seqlen).
    # cu_seqlens 基于物理 S (每 batch 物理 S 个 token, padding 区域补 0);
    # seq_used (act_seq_q_eff) 是有效长度, flash_attention_cpu_golden_varlen 只算有效区域.
    s1 = data_dict.get("s1_physical")
    s2 = data_dict.get("s2_physical")
    act_seq_q_eff = data_dict.get("act_seq_q_eff")
    act_seq_kv_eff = data_dict.get("act_seq_kv_eff")
    if s1 is None:
        s1 = max(act_seq_q) if act_seq_q else (MAX_SEQLEN_Q if MAX_SEQLEN_Q >= 0 else 0)
    if s2 is None:
        s2 = (
            max(act_seq_kv)
            if act_seq_kv
            else (MAX_SEQLEN_KV if MAX_SEQLEN_KV >= 0 else 0)
        )
    if act_seq_q_eff is None:
        act_seq_q_eff = list(act_seq_q) if act_seq_q else [s1] * b
    if act_seq_kv_eff is None:
        act_seq_kv_eff = list(act_seq_kv) if act_seq_kv else [s2] * b

    cu_seqlens_q = [i * s1 for i in range(b + 1)]
    cu_seqlens_kv = [i * s2 for i in range(b + 1)]

    # 取原始 FP32 BNSD Q/K/V (与原 pytest 一致, golden 接收 FP32, 内部做量化反量化)
    # FP32 由 generate_data 用确定性种子 (SEED_Q/K/V) 生成, 与 inputs.py 同源, 无需跨用例缓存
    fp32_bnsd = data_dict.get("fp32_bnsd")
    if fp32_bnsd is None:
        raise RuntimeError(
            "data_dict['fp32_bnsd'] is None, generate_data must include fp32_bnsd"
        )
    query_fp32, key_fp32, value_fp32 = fp32_bnsd

    # BNSD -> BSND -> flatten(B,N) -> [T, N, D]  (与原 pytest cpu_golden_qkv_mxfp4_flash_attn 一致)
    query_bsnd = (
        bnsd_to_bsnd(query_fp32).to(torch.float32).flatten(start_dim=0, end_dim=1)
    )
    key_bsnd = bnsd_to_bsnd(key_fp32).to(torch.float32).flatten(start_dim=0, end_dim=1)
    value_bsnd = (
        bnsd_to_bsnd(value_fp32).to(torch.float32).flatten(start_dim=0, end_dim=1)
    )

    block_q = 128
    block_kv = 4096
    attn_out = flash_attention_cpu_golden_varlen(
        query_bsnd,
        key_bsnd,
        value_bsnd,
        cu_seqlens_q,
        cu_seqlens_kv,
        act_seq_q_eff,
        act_seq_kv_eff,
        softmax_scale=softmax_scale,
        quantize=True,
        quantize_p=True,
        block_q=block_q,
        block_kv=block_kv,
        s_layout="DN",
        quantize_p_mode="blockwise_snap_local",
        s_dtype="fp16",
        v_quant_axis="seq_k",
    )

    # 输出 reshape 回 BNSD 再转 attn_out_layout (与原 pytest 一致)
    # s1 是物理 S, attn_out padding 区域为 0 (flash_attention_cpu_golden_varlen 只算有效区域)
    attn_out = attn_out.reshape(b, s1, n1, qk_d).permute(0, 2, 1, 3)  # -> BNSD
    attn_out = rearrange_by_layout(attn_out, attn_out_layout, b, act_seq_q_eff)

    return attn_out.to(get_dtype(OUT_DTYPE)), None


# ==============================================================================
# NPU 调用: GRAPH_PATH=0 eager / GRAPH_PATH=7 npugraph_ex
# ==============================================================================
def _to_npu(tensor):
    if tensor is None:
        return None
    return tensor.to("npu:%s" % int(DEVICE_ID))


def _call_npu_fa_op(data_dict):
    q = _to_npu(data_dict["q"])
    k = _to_npu(data_dict["k"])
    v = _to_npu(data_dict["v"])
    q_descale = _to_npu(data_dict["q_descale"])
    k_descale = _to_npu(data_dict["k_descale"])
    v_descale = _to_npu(data_dict["v_descale"])
    block_table = (
        _to_npu(data_dict.get("block_table"))
        if data_dict.get("block_table") is not None
        else None
    )
    p_scale = (
        _to_npu(data_dict.get("p_scale"))
        if data_dict.get("p_scale") is not None
        else None
    )
    sinks = (
        _to_npu(data_dict.get("sinks")) if data_dict.get("sinks") is not None else None
    )
    attn_mask = (
        _to_npu(data_dict.get("attn_mask"))
        if data_dict.get("attn_mask") is not None
        else None
    )

    cu_seqlens_q = data_dict["cu_seqlens_q"]
    cu_seqlens_kv = data_dict["cu_seqlens_kv"]
    seqused_q = data_dict["act_seq_lens_q"]
    seqused_kv = data_dict["act_seq_lens_kv"]

    cu_seqlens_q_t = (
        torch.tensor(
            cu_seqlens_q, dtype=get_dtype(CU_SEQLENS_Q_DTYPE) or torch.int32
        ).npu()
        if cu_seqlens_q is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(
            cu_seqlens_kv, dtype=get_dtype(CU_SEQLENS_KV_DTYPE) or torch.int32
        ).npu()
        if cu_seqlens_kv is not None
        else None
    )
    # seqused 为空 -> 传 None (算子内部走 max_seqlen + defaultVal 回退)
    seqused_q_t = (
        torch.tensor(seqused_q, dtype=get_dtype(SEQUSED_Q_DTYPE) or torch.int32).npu()
        if seqused_q
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=get_dtype(SEQUSED_KV_DTYPE) or torch.int32).npu()
        if seqused_kv
        else None
    )

    query_layout = data_dict["query_layout"]
    kv_layout = data_dict.get("kv_layout", query_layout)
    attn_out_layout = data_dict["attn_out_layout"]

    torch_npu.npu.set_device(int(DEVICE_ID))
    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=data_dict["num_heads"],
        num_heads_kv=data_dict["num_key_value_heads"],
        head_dim=D,
        quant_mode=Q_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
    )

    main_kwargs = dict(
        q=q,
        k=k,
        v=v,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        quant_mode=Q_QUANT_MODE,
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        sinks=sinks,
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=data_dict["softmax_scale"],
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
    )
    main_kwargs.update(_get_npu_fa_kwargs())
    npu_attn_out, npu_lse = quant_flash_attn(**main_kwargs)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    return npu_attn_out, npu_lse


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        num_heads_q,
        num_heads_kv,
        softmax_scale,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        p_scale=None,
        sinks=None,
        attn_mask=None,
    ):
        metadata = quant_flash_attn_metadata(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=D,
            quant_mode=Q_QUANT_MODE,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            v_descale=v_descale,
            batch_size=B,
            max_seqlen_q=MAX_SEQLEN_Q,
            max_seqlen_kv=MAX_SEQLEN_KV,
            mask_mode=SPARSE_MODE,
            win_left=PRE_TOKENS,
            win_right=NEXT_TOKENS,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )
        main_kwargs = dict(
            q=q,
            k=k,
            v=v,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            quant_mode=Q_QUANT_MODE,
            block_table=block_table,
            p_scale=p_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            mask_mode=SPARSE_MODE,
            win_left=PRE_TOKENS,
            win_right=NEXT_TOKENS,
            max_seqlen_q=MAX_SEQLEN_Q,
            max_seqlen_kv=MAX_SEQLEN_KV,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )
        main_kwargs.update(_get_npu_fa_kwargs())
        atten_out, lse_out = quant_flash_attn(**main_kwargs)
        return atten_out, lse_out


def _mxfp4_fa_torch_npu(data_dict):
    """GRAPH_PATH=7: 用 npugraph_ex backend 编译 Network, 绕过 get_npu_backend() bug"""
    # 预处理: list -> NPU tensor (必须在编译区域外完成)
    cu_seqlens_q_t = (
        torch.tensor(
            data_dict["cu_seqlens_q"],
            dtype=get_dtype(CU_SEQLENS_Q_DTYPE) or torch.int32,
        ).npu()
        if data_dict["cu_seqlens_q"] is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(
            data_dict["cu_seqlens_kv"],
            dtype=get_dtype(CU_SEQLENS_KV_DTYPE) or torch.int32,
        ).npu()
        if data_dict["cu_seqlens_kv"] is not None
        else None
    )
    # seqused 为空 -> 传 None (算子内部走 max_seqlen + defaultVal 回退)
    seqused_q_t = (
        torch.tensor(
            data_dict["act_seq_lens_q"], dtype=get_dtype(SEQUSED_Q_DTYPE) or torch.int32
        ).npu()
        if data_dict["act_seq_lens_q"]
        else None
    )
    seqused_kv_t = (
        torch.tensor(
            data_dict["act_seq_lens_kv"],
            dtype=get_dtype(SEQUSED_KV_DTYPE) or torch.int32,
        ).npu()
        if data_dict["act_seq_lens_kv"]
        else None
    )

    q = _to_npu(data_dict["q"])
    k = _to_npu(data_dict["k"])
    v = _to_npu(data_dict["v"])
    q_descale = _to_npu(data_dict["q_descale"])
    k_descale = _to_npu(data_dict["k_descale"])
    v_descale = _to_npu(data_dict["v_descale"])
    block_table = (
        _to_npu(data_dict.get("block_table"))
        if data_dict.get("block_table") is not None
        else None
    )
    p_scale = (
        _to_npu(data_dict.get("p_scale"))
        if data_dict.get("p_scale") is not None
        else None
    )
    sinks = (
        _to_npu(data_dict.get("sinks")) if data_dict.get("sinks") is not None else None
    )
    attn_mask = (
        _to_npu(data_dict.get("attn_mask"))
        if data_dict.get("attn_mask") is not None
        else None
    )

    npu_mode = Network().to("npu:%s" % int(DEVICE_ID))
    with torch.no_grad():
        torch.npu.synchronize()

        query_layout = data_dict["query_layout"]
        kv_layout = data_dict.get("kv_layout", query_layout)
        attn_out_layout = data_dict["attn_out_layout"]
        fa_args = (
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            block_table,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            data_dict["num_heads"],
            data_dict["num_key_value_heads"],
            data_dict["softmax_scale"],
            query_layout,
            LAYOUT_Q_DESCALE,
            kv_layout,
            attn_out_layout,
            p_scale,
            sinks,
            attn_mask,
        )

        logger.info("[NPU] 调用 aclgraph (npugraph_ex)...")
        npu_mode = torch.compile(
            npu_mode, fullgraph=False, backend="npugraph_ex", dynamic=False
        )
        for t in (
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            block_table,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            p_scale,
            sinks,
            attn_mask,
        ):
            if t is not None:
                torch._dynamo.mark_static(t)

        atten_out, lse_out = npu_mode(*fa_args)
        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach() if lse_out is not None else None
        torch.npu.synchronize()
        return atten_out, lse_out


def npu_mxfp4_fa(data_dict):
    """NPU 调用入口: GRAPH_PATH=0 eager, GRAPH_PATH=7 aclgraph"""
    logger.info("[NPU] GRAPH_PATH=%d, layout=%s", GRAPH_PATH, data_dict["query_layout"])
    if GRAPH_PATH == 0:
        return _call_npu_fa_op(data_dict)
    return _mxfp4_fa_torch_npu(data_dict)


def call_npu_metadata(data_dict):
    """只调 quant_flash_attn_metadata, 返回 [metadata] 对象.

    分离测试入口: 验证 metadata 算子独立正确性, 不调主算子。
    """
    v_descale = _to_npu(data_dict["v_descale"])

    cu_seqlens_q = data_dict["cu_seqlens_q"]
    cu_seqlens_kv = data_dict["cu_seqlens_kv"]
    seqused_q = data_dict["act_seq_lens_q"]
    seqused_kv = data_dict["act_seq_lens_kv"]

    cu_seqlens_q_t = (
        torch.tensor(
            cu_seqlens_q, dtype=get_dtype(CU_SEQLENS_Q_DTYPE) or torch.int32
        ).npu()
        if cu_seqlens_q is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(
            cu_seqlens_kv, dtype=get_dtype(CU_SEQLENS_KV_DTYPE) or torch.int32
        ).npu()
        if cu_seqlens_kv is not None
        else None
    )
    seqused_q_t = (
        torch.tensor(seqused_q, dtype=get_dtype(SEQUSED_Q_DTYPE) or torch.int32).npu()
        if seqused_q
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=get_dtype(SEQUSED_KV_DTYPE) or torch.int32).npu()
        if seqused_kv
        else None
    )

    query_layout = data_dict["query_layout"]
    kv_layout = data_dict.get("kv_layout", query_layout)
    attn_out_layout = data_dict["attn_out_layout"]

    torch_npu.npu.set_device(int(DEVICE_ID))
    torch.npu.synchronize()

    logger.info("[NPU_METADATA] 调用 quant_flash_attn_metadata")
    metadata = quant_flash_attn_metadata(
        num_heads_q=data_dict["num_heads"],
        num_heads_kv=data_dict["num_key_value_heads"],
        head_dim=D,
        quant_mode=Q_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
    )
    torch.npu.synchronize()
    return [metadata]


def call_npu_main(data_dict):
    """内部重建 metadata 后调 quant_flash_attn 主算子, 返回 (atten_out, lse_out).

    分离测试入口: 验证主算子 + metadata 完整链路 (metadata 在本函数内部重建, 不跨用例传递)。
    """
    q = _to_npu(data_dict["q"])
    k = _to_npu(data_dict["k"])
    v = _to_npu(data_dict["v"])
    q_descale = _to_npu(data_dict["q_descale"])
    k_descale = _to_npu(data_dict["k_descale"])
    v_descale = _to_npu(data_dict["v_descale"])
    block_table = (
        _to_npu(data_dict.get("block_table"))
        if data_dict.get("block_table") is not None
        else None
    )
    p_scale = (
        _to_npu(data_dict.get("p_scale"))
        if data_dict.get("p_scale") is not None
        else None
    )
    sinks = (
        _to_npu(data_dict.get("sinks")) if data_dict.get("sinks") is not None else None
    )
    attn_mask = (
        _to_npu(data_dict.get("attn_mask"))
        if data_dict.get("attn_mask") is not None
        else None
    )

    cu_seqlens_q = data_dict["cu_seqlens_q"]
    cu_seqlens_kv = data_dict["cu_seqlens_kv"]
    seqused_q = data_dict["act_seq_lens_q"]
    seqused_kv = data_dict["act_seq_lens_kv"]

    cu_seqlens_q_t = (
        torch.tensor(
            cu_seqlens_q, dtype=get_dtype(CU_SEQLENS_Q_DTYPE) or torch.int32
        ).npu()
        if cu_seqlens_q is not None
        else None
    )
    cu_seqlens_kv_t = (
        torch.tensor(
            cu_seqlens_kv, dtype=get_dtype(CU_SEQLENS_KV_DTYPE) or torch.int32
        ).npu()
        if cu_seqlens_kv is not None
        else None
    )
    seqused_q_t = (
        torch.tensor(seqused_q, dtype=get_dtype(SEQUSED_Q_DTYPE) or torch.int32).npu()
        if seqused_q
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=get_dtype(SEQUSED_KV_DTYPE) or torch.int32).npu()
        if seqused_kv
        else None
    )

    query_layout = data_dict["query_layout"]
    kv_layout = data_dict.get("kv_layout", query_layout)
    attn_out_layout = data_dict["attn_out_layout"]

    torch_npu.npu.set_device(int(DEVICE_ID))
    torch.npu.synchronize()

    logger.info("[NPU_MAIN] 重建 metadata (调 quant_flash_attn_metadata)")
    metadata = quant_flash_attn_metadata(
        num_heads_q=data_dict["num_heads"],
        num_heads_kv=data_dict["num_key_value_heads"],
        head_dim=D,
        quant_mode=Q_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
    )

    main_kwargs = dict(
        q=q,
        k=k,
        v=v,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        quant_mode=Q_QUANT_MODE,
        block_table=block_table,
        p_scale=p_scale,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        sinks=sinks,
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=data_dict["softmax_scale"],
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
    )
    main_kwargs.update(_get_npu_fa_kwargs())

    logger.info("[NPU_MAIN] 调用 quant_flash_attn 主算子")
    npu_attn_out, npu_lse = quant_flash_attn(**main_kwargs)
    torch.npu.synchronize()
    return npu_attn_out, npu_lse


# 自引用, 用于 _apply_golden_globals (避免循环 import)
golden_mod_self = sys.modules[__name__]
