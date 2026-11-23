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

import mx_quant_fp4_tool as _mx_tool
import flash_attention_cpu_golden as _cpu_golden_mod

mxfp4_quantize_pack_last = _mx_tool.mxfp4_quantize_pack_last
mxfp4_quantize_last = _mx_tool.mxfp4_quantize_last
flash_attention_cpu_golden_varlen = _cpu_golden_mod.flash_attention_cpu_golden_varlen

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
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
KV_STORAGE_MODE = "continue"  # continue / pa_bbnd / pa_bnbd / pa_nz
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
DATA_DIST = "randn"  # randn | uniform | skew, 见 _gen_range_tensor

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

METADATA_SHAPE = [4096]
METADATA_DTYPE = None
METADATA_DATARANGE = None

# 表格全量透传列 (供查看/比对):
# 输出/附加信息
CUSTOM_INFO = ""
ATTN_OUT_SHAPE = []
ATTN_OUT_DATARANGE = None
SOFTMAX_LSE_SHAPE = []
SOFTMAX_LSE_DATARANGE = None
# descale / block_table datarange
Q_DESCALE_DATARANGE = None
K_DESCALE_DATARANGE = None
V_DESCALE_DATARANGE = None
BLOCK_TABLE_DATARANGE = None
# cu_seqlens / seqused 附加列
CU_SEQLENS_Q_SHAPE = []
CU_SEQLENS_KV_SHAPE = []
CU_SEQLENS_Q_DATARANGE = None
CU_SEQLENS_KV_DATARANGE = None
SEQUSED_Q_SHAPE = []
SEQUSED_KV_SHAPE = []
SEQUSED_Q_DATARANGE = None
SEQUSED_KV_DATARANGE = None

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
# 逐行/逐 token 尺度扰动: 让 descale 非均匀(自然数据 descale 恒=125, 会掩蔽 V-scale 通路问题).
# None=关闭; "logu"=每行乘 2^U(-R,R); "mix"=一半行*2^R, 一半行*2^-R (极端动态范围)
PER_ROW_SCALE = None
PER_ROW_SCALE_R = 6

# 物理 S override: generate_data 优先用这两个值作为物理 S (生成 query/key/value
# 的 S 维), 而不是 max(ACT_SEQ_LENS) / MAX_SEQLEN (后者只决定有效长度).
_PHYSICAL_S_Q_OVERRIDE = None
_PHYSICAL_S_KV_OVERRIDE = None

# 可选: TND 非 PA 的 KV 物理打包长度(per-batch). None => 与 act_seq_kv_eff 相同.
# 用于构造 "packed(cu 区段) 比 seqused 有效长度更长" 的 padding 场景:
# 数据/k_descale 按 pack_lens 打包(物理行), 有效长度仍由 seqused 决定,
# v_descale 组数仍按 seqused 向上取整(与 kernel 寻址口径一致).
PACK_LENS_KV = None


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
        nd = len(tensor.shape)
        if layout_upper in ("PA_NZ", "PA_BNBD", "PA_BBND"):
            # PA 分页布局: 单 batch 物理 S = 每 batch 块数上限 × block_size
            # (块表形状 (B, max_blocks_per_batch); 各 batch 有效长度由 seqused_kv 给出)
            if BLOCK_TABLE_SHAPE and len(BLOCK_TABLE_SHAPE) == 2 and BLOCK_SIZE:
                return int(BLOCK_TABLE_SHAPE[1]) * int(BLOCK_SIZE)
            # 无块表信息时无法反推每 batch 块数, 交回调用方按有效长度处理
            return None
        if nd >= 5:
            # 兜底: 5 维及以上都是分页打包布局(如 PA_NZ 的 (总块数, N, D/64, bs, 64)),
            # 没有单 batch S 语义; 若按 BSND 取 shape[1] 会误得 N(常为 1)
            return None
        if layout_upper in ("BSND", "BSH"):
            if nd < 2:
                return None
            return int(tensor.shape[1])
        if layout_upper == "BNSD":
            if nd < 3:
                return None
            return int(tensor.shape[2])
    except (IndexError, TypeError):
        return None
    # TND: (T, N, D), T = sum(act), 无法反推单 batch 物理 S
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
# 配置注入
# ==============================================================================
_GOLDEN_GLOBALS_MAP = {
    "B": "B",
    "N_q": "N_q",
    "N_kv": "N_kv",
    "G": "G",
    "D": "D",
    "V_D": "V_D",
    "Rope_D": "Rope_D",
    "layout_q": "INPUT_LAYOUT",
    "layout_q_descale": "LAYOUT_Q_DESCALE",
    "layout_kv": "LAYOUT_KV",
    "layout_out": "LAYOUT_OUT",
    "kv_storage_mode": "KV_STORAGE_MODE",
    "block_size": "BLOCK_SIZE",
    "q_dtype": "Q_DTYPE",
    "kv_dtype": "KV_DTYPE",
    "out_dtype": "OUT_DTYPE",
    "quant_mode": "Q_QUANT_MODE",
    "mask_mode": "SPARSE_MODE",
    "win_left": "PRE_TOKENS",
    "win_right": "NEXT_TOKENS",
    "enable_mask": "ENABLE_MASK",
    "return_softmax_lse": "ENABLE_LSE",
    "inner_precise": "INNER_PRECISE",
    "device_id": "DEVICE_ID",
    "graph_path": "GRAPH_PATH",
    "softmax_scale": "SOFTMAX_SCALE",
    "data_range_q": "DATA_RANGE_Q",
    "data_range_k": "DATA_RANGE_K",
    "data_range_v": "DATA_RANGE_V",
    "seqused_q_values": "ACT_SEQ_LENS_Q",
    "seqused_kv_values": "ACT_SEQ_LENS_KV",
    "cu_seqlens_q_values": "CU_SEQLENS_Q",
    "cu_seqlens_kv_values": "CU_SEQLENS_KV",
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
    "metadata_shape": "METADATA_SHAPE",
    "metadata_dtype": "METADATA_DTYPE",
    "custom_info": "CUSTOM_INFO",
    "attn_out_shape": "ATTN_OUT_SHAPE",
    "attn_out_datarange": "ATTN_OUT_DATARANGE",
    "softmax_lse_shape": "SOFTMAX_LSE_SHAPE",
    "softmax_lse_datarange": "SOFTMAX_LSE_DATARANGE",
    "metadata_datarange": "METADATA_DATARANGE",
    "q_descale_datarange": "Q_DESCALE_DATARANGE",
    "k_descale_datarange": "K_DESCALE_DATARANGE",
    "v_descale_datarange": "V_DESCALE_DATARANGE",
    "block_table_datarange": "BLOCK_TABLE_DATARANGE",
    "cu_seqlens_q_shape": "CU_SEQLENS_Q_SHAPE",
    "cu_seqlens_kv_shape": "CU_SEQLENS_KV_SHAPE",
    "cu_seqlens_q_datarange": "CU_SEQLENS_Q_DATARANGE",
    "cu_seqlens_kv_datarange": "CU_SEQLENS_KV_DATARANGE",
    "seqused_q_shape": "SEQUSED_Q_SHAPE",
    "seqused_kv_shape": "SEQUSED_KV_SHAPE",
    "seqused_q_datarange": "SEQUSED_Q_DATARANGE",
    "seqused_kv_datarange": "SEQUSED_KV_DATARANGE",
}


def _apply_golden_globals(attrs):
    """把 case attributes 注入 golden 模块全局变量"""
    for attr_key, golden_key in _GOLDEN_GLOBALS_MAP.items():
        if attr_key in attrs:
            setattr(golden_mod_self, golden_key, attrs[attr_key])


def _apply_kwargs_globals(kwargs):
    """把额外 attrs (kwargs) 注入 golden 全局变量."""
    for attr_key, golden_key in _GOLDEN_GLOBALS_MAP.items():
        if attr_key in kwargs:
            setattr(golden_mod_self, golden_key, kwargs[attr_key])


# ==============================================================================
# Layout 工具
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


def transpose_qscale(q_scale, layout):
    """Q scale 重排: 保持 layout 物理排列, 末维 (D//2, 2) 两元素一组"""
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
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2")
        return q_scale.reshape(T, N, D // 2, 2)
    elif layout == "BSND":
        if q_scale.dim() != 4:
            raise ValueError(
                f"BSND layout requires 4D tensor [B,S,N,D], got dim={q_scale.dim()}"
            )
        B, S1, N, D = q_scale.shape
        if D % 2 != 0:
            raise ValueError(f"Feature dim D={D} must be divisible by 2 for MXFP4 pack")
        return q_scale.contiguous().reshape(B, S1, N, D // 2, 2)
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
    s = str(val).strip().strip("[]()")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 1:
        f = float(parts[0])
        return (-f, f)
    if len(parts) >= 2:
        return (float(parts[0]), float(parts[1]))
    return (-1.0, 1.0)


def _gen_range_tensor(shape, lo, hi, seed):
    """按 DATA_DIST 生成落在 [lo, hi] 内的 fp32 数据.

    - "randn"(默认): randn 后 clamp. 注意 randn 的 3σ≈3, 因此 |lo|,|hi| >= 3 时 clamp 不生效,
      实际幅值恒在 ±3 左右 —— 想要真正测大/小量级必须用 "uniform"/"skew".
    - "uniform": [lo, hi] 均匀填充, 幅值真正到达边界.
    - "skew": 三次方偏置, 数据挤在 lo 附近并少量到达 hi, 用于制造行内大动态范围.
    """
    torch.manual_seed(seed)
    if DATA_DIST == "uniform":
        u = torch.rand(shape, dtype=torch.float32)
        return u * (hi - lo) + lo
    if DATA_DIST == "skew":
        u = torch.rand(shape, dtype=torch.float32) ** 3
        return u * (hi - lo) + lo
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


def _cu_seqlens_to_actual(cu_seqlens):
    """cu_seqlens → 有效长度 (差分还原): [0, 3, 7, 10] → [3, 4, 3]"""
    if not cu_seqlens or len(cu_seqlens) < 2:
        return []
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def generate_data():
    global CU_SEQLENS_KV
    n2 = N_kv
    g = G
    num_heads = n2 * g
    # seqused 缺省时回退 cu_seqlens 差分
    act_seq_q_src = ACT_SEQ_LENS_Q or _cu_seqlens_to_actual(CU_SEQLENS_Q)
    act_seq_kv_src = ACT_SEQ_LENS_KV or _cu_seqlens_to_actual(CU_SEQLENS_KV)
    s1, s1_effective, act_seq_q_eff = _resolve_s_with_effective(
        act_seq_q_src, MAX_SEQLEN_Q, _PHYSICAL_S_Q_OVERRIDE, B, "Q"
    )
    s2, s2_effective, act_seq_kv_eff = _resolve_s_with_effective(
        act_seq_kv_src, MAX_SEQLEN_KV, _PHYSICAL_S_KV_OVERRIDE, B, "KV"
    )
    if PACK_LENS_KV:
        s2 = max(PACK_LENS_KV)
        # 打包长度即物理长度 => cu_seqlens_kv 必须按打包长度给出(否则 cu 与数据自相矛盾).
        # 这正是"cu 差分 > seqused"的合法场景: 物理打包 256, 有效长度仍由 seqused 决定.
        CU_SEQLENS_KV = [0] * (B + 1)
        for _i in range(B):
            CU_SEQLENS_KV[_i + 1] = CU_SEQLENS_KV[_i] + int(PACK_LENS_KV[_i])
    v_d = V_D
    qk_d = D

    query_layout = get_query_layout(INPUT_LAYOUT)
    # attn_out_layout: 优先用表格 LAYOUT_OUT, 否则从 INPUT_LAYOUT 推导
    attn_out_layout = (
        get_attn_out_layout(LAYOUT_OUT)
        if LAYOUT_OUT
        else get_attn_out_layout(INPUT_LAYOUT)
    )
    # kv_layout: 优先用表格 LAYOUT_KV, 否则等于 query_layout; PA 布局映射为算子侧字符串
    _kv_mode = (KV_STORAGE_MODE or "").lower()
    if _kv_mode in ("pa_bnbd", "pa_bbnd", "pa_nz"):
        kv_layout = {"pa_bnbd": "PA_BNBD", "pa_bbnd": "PA_BBND", "pa_nz": "PA_NZ"}[
            _kv_mode
        ]
        # layout_kv 允许三种写法: PA 专用名(PA_NZ)、通用族名(BNSD/BSND, 由存储模式决定实际布局)、空
        # 只有明确写了不兼容的族(TND/BSND 之外的连续布局)才判冲突
        _family = (LAYOUT_KV or "").upper()
        if _family and _family not in ("PA_NZ", "PA_BNBD", "PA_BBND", "BNSD", "BSND"):
            raise RuntimeError(
                f"layout_kv({LAYOUT_KV}) 与 kv_storage_mode({KV_STORAGE_MODE}) 冲突"
            )
    else:
        kv_layout = get_query_layout(LAYOUT_KV) if LAYOUT_KV else query_layout

    # Q/K/V FP32 数据按 DATA_RANGE_Q/K/V 生成: randn 后 clamp 到 [lo, hi]
    q_lo, q_hi = _parse_datarange(DATA_RANGE_Q)
    k_lo, k_hi = _parse_datarange(DATA_RANGE_K)
    v_lo, v_hi = _parse_datarange(DATA_RANGE_V)
    query = _gen_range_tensor((B, num_heads, s1, qk_d), q_lo, q_hi, SEED_Q)
    key = _gen_range_tensor((B, n2, s2, qk_d), k_lo, k_hi, SEED_K)
    value = _gen_range_tensor((B, n2, s2, v_d), v_lo, v_hi, SEED_V)

    if PER_ROW_SCALE:

        def _apply_scales(t, rows_axis=2, tag=0):
            shp = t.shape
            n_rows = shp[rows_axis]
            if PER_ROW_SCALE == "mix":
                half = n_rows // 2
                fac = torch.ones(n_rows, dtype=torch.float32)
                fac[:half] = 2.0**PER_ROW_SCALE_R
                fac[half:] = 2.0 ** (-PER_ROW_SCALE_R)
            else:  # logu
                # 必须用独立、固定种子的 generator: 原来直接吃全局 RNG, 同一 case 每次跑的行尺度
                # 都不同 -> 失败无法复现, pa/tnd 对照也失去意义(实测同配置两次 0% vs 63% bad).
                gen = torch.Generator().manual_seed(7919 + 13 * tag)
                fac = 2.0 ** torch.empty(n_rows).uniform_(
                    -PER_ROW_SCALE_R, PER_ROW_SCALE_R, generator=gen
                )
            view = [1] * len(shp)
            view[rows_axis] = n_rows
            return t * fac.view(view)

        query = _apply_scales(query, tag=0)
        key = _apply_scales(key, tag=1)
        value = _apply_scales(value, tag=2)

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

    query_codes, q_descale = mxfp4_quantize_last(query, quant_axis=-1, mode="baseline")
    key_codes, k_descale = mxfp4_quantize_last(key, quant_axis=-1, mode="baseline")

    # V 按 act_seq_lens_kv 分批量化: padding 区域编码填 0, descale 只覆盖有效区域.
    # v_scale 长度口径 = KV 物理打包长度(与数据/ k_descale 一致), 保证 descale 覆盖全部物理行.
    # 用 act_seq_kv_eff(有效长度) 会在 packed > seqused 时覆盖不全 -> V 缩放丢失.
    _pack_lens_for_v = PACK_LENS_KV if PACK_LENS_KV else act_seq_kv_eff
    v_scale_s = max(int(x) for x in _pack_lens_for_v) if _pack_lens_for_v else s2
    n_blocks_effective = (v_scale_s + 31) // 32
    value_codes_list = []
    v_descale_list = []
    for b_idx in range(B):
        sk = min(act_seq_kv_eff[b_idx], s2)
        v_b = value[b_idx : b_idx + 1, :, :sk, :]
        v_b_codes, v_b_scale = mxfp4_quantize_last(v_b, quant_axis=-2, mode="baseline")
        codes_padded = torch.zeros((1, n2, s2, v_d), dtype=v_b_codes.dtype)
        codes_padded[:, :, :sk, :] = v_b_codes
        value_codes_list.append(codes_padded)
        n_blocks_actual = v_b_scale.shape[2]
        scale_padded = torch.full(
            (1, n2, n_blocks_effective, v_d), 127, dtype=v_b_scale.dtype
        )
        scale_padded[:, :, :n_blocks_actual, :] = v_b_scale
        v_descale_list.append(scale_padded)
    value_codes = torch.cat(value_codes_list, dim=0)
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

    # Layout 转换 (Q codes + scale 用 query_layout; K/V codes + scale 用 kv_layout)
    is_pa = KV_STORAGE_MODE in ("pa_bnbd", "pa_bbnd", "pa_nz")
    # PA 时 kv 数据保持 BNSD 中间形态, 散布在 _pa_scatter 中完成
    eff_kv_layout = "BNSD" if is_pa else kv_layout

    query_codes = rearrange_by_layout(query_codes, query_layout, B, act_seq_q_eff)
    pack_lens_kv = PACK_LENS_KV if PACK_LENS_KV else act_seq_kv_eff
    key_codes = rearrange_by_layout(key_codes, eff_kv_layout, B, pack_lens_kv)
    value_codes = rearrange_by_layout(value_codes, eff_kv_layout, B, pack_lens_kv)

    q_descale = rearrange_by_layout(q_descale, query_layout, B, act_seq_q_eff)
    q_descale = transpose_qscale(q_descale, query_layout)
    if LAYOUT_Q_DESCALE == "N2TGD" and query_layout == "TND":
        # TND 统一 N2TGD: (T, N_kv*G, D/2, 2) -> (N_kv, T, G, D/2, 2)
        t_total = q_descale.shape[0]
        q_descale = (
            q_descale.reshape(t_total, n2, -1, *q_descale.shape[2:])
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )

    # k_descale 行 = token 行 (per-token scale), 与 K 数据同 narrow 语义
    k_descale = rearrange_by_layout(k_descale, eff_kv_layout, B, pack_lens_kv)
    k_descale = transpose_kscale(k_descale, eff_kv_layout)

    # v_descale 的拼接长度必须与数据(value_codes)/k_descale 同源(都用 pack_lens_kv),
    # 否则 packed > seqused 时 v_descale 只取前 seqused/64 组, 后半段 V 无 scale -> V 缩放丢失.
    aligned_seq_lens_kv_v = [(x + 63) // 64 for x in pack_lens_kv]
    v_descale = rearrange_by_layout(v_descale, eff_kv_layout, B, aligned_seq_lens_kv_v)

    v_descale = v_descale.view(*v_descale.shape[:-1], V_D, 2)

    block_table_t = None
    if is_pa:
        block_size = int(BLOCK_SIZE)
        if block_size <= 0:
            raise RuntimeError("PA 模式要求 block_size > 0")
        # 每 batch 需要的块数; 池大小留 2 个冗余块 (全 0), 未用槽位指向冗余块
        blocks_per_batch = [math.ceil(x / block_size) for x in act_seq_kv_eff]
        max_blocks = max(blocks_per_batch)
        spare_cnt = 2
        total_blocks = sum(blocks_per_batch) + spare_cnt
        torch.manual_seed(100)
        perm = torch.randperm(sum(blocks_per_batch), dtype=torch.int32)
        block_table_t = torch.empty((B, max_blocks), dtype=torch.int32)
        spare_ids = list(range(sum(blocks_per_batch), total_blocks))
        idx = 0
        for b_idx in range(B):
            row = []
            for j in range(max_blocks):
                if j < blocks_per_batch[b_idx]:
                    row.append(int(perm[idx]))
                    idx += 1
                else:
                    row.append(spare_ids[j % spare_cnt])
            block_table_t[b_idx] = torch.tensor(row, dtype=torch.int32)

        def scatter_pa(bnsd, tail_shape_fn, fill=0):
            """BNSD (B,N,S,...) → PA cache (totalBlocks, ...), 尾维按块切散布; fill 为 padding 填充值."""
            out = torch.full((total_blocks,) + tail_shape_fn(), fill, dtype=bnsd.dtype)
            src = bnsd  # (B, N, S, ...)
            for b_idx in range(B):
                for j in range(blocks_per_batch[b_idx]):
                    dst_blk = int(block_table_t[b_idx][j])
                    s0 = j * block_size
                    # 源按物理 S 截断 (padding 区随块拷入, 与 cache 语义一致)
                    s1 = min(s0 + block_size, src.shape[2])
                    out[dst_blk, :, : s1 - s0] = src[b_idx, :, s0:s1]
            return out

        def scatter_pa_v(v_bnsd, tail_shape_fn, fill=127):
            """v_descale BNSD (B,N,Sg,D,2) → PA cache, 行=64-token 组, 按组维散布."""
            out = torch.full(
                (total_blocks,) + tail_shape_fn(), fill, dtype=v_bnsd.dtype
            )
            for b_idx in range(B):
                for j in range(blocks_per_batch[b_idx]):
                    dst_blk = int(block_table_t[b_idx][j])
                    g0 = j * block_groups
                    g1 = min(g0 + block_groups, v_bnsd.shape[2])
                    out[dst_blk, :, : g1 - g0] = v_bnsd[b_idx, :, g0:g1]
            return out

        n_kv = n2
        d_scale = qk_d // 64  # per-token scale 组数
        block_groups = (
            block_size + 63
        ) // 64  # 块内 64-token 组数, v_descale 每组一行 (D, 2)

        if KV_STORAGE_MODE == "pa_bnbd":
            # (Bn, N, Bs, ...): 块内 (N, Bs, tail)
            key_codes = scatter_pa(key_codes, lambda: (n_kv, block_size, qk_d))
            value_codes = scatter_pa(value_codes, lambda: (n_kv, block_size, qk_d))
            # k_descale padding 行填 127 (e8m0 最小正数), 避免 NPU 读到 0 byte
            k_descale = scatter_pa(
                k_descale, lambda: (n_kv, block_size, d_scale, 2), fill=127
            )
            # v_descale: (Bn, N, Bs/64, D, 2), 每组一行 (行内双子组交错, 与 BNSD 打包一致)
            v_descale = scatter_pa_v(v_descale, lambda: (n_kv, block_groups, V_D, 2))
        elif KV_STORAGE_MODE == "pa_nz":
            # PA_NZ (fp4 物理分形, d0=32B): K/V codes (Bn, N, D/64, Bs, 64 逻辑元素→32B),
            # k_descale (Bn, N, Bs/16, 2, 16, 2), v_descale (Bn, N, D/16, Bs/64, 16, 2)
            def _pack_nibbles(codes):
                lo = codes[..., 0::2].to(torch.uint8)
                hi = codes[..., 1::2].to(torch.uint8)
                return ((lo & 0x0F) | ((hi & 0x0F) << 4)).contiguous()

            def scatter_pa_nz_kv(codes, fill=0):
                d1_cnt = qk_d // 64  # D/64 逻辑组 (每单元 64 逻辑元素 = 32B 整块)
                out = torch.full(
                    (total_blocks, n_kv, d1_cnt, block_size, 64),
                    fill,
                    dtype=codes.dtype,
                )
                for b_idx in range(B):
                    for j in range(blocks_per_batch[b_idx]):
                        dst_blk = int(block_table_t[b_idx][j])
                        s0 = j * block_size
                        s1 = min(s0 + block_size, codes.shape[2])
                        seg = codes[b_idx, :, s0:s1].reshape(n_kv, s1 - s0, d1_cnt, 64)
                        out[dst_blk, :, :, : s1 - s0] = seg.permute(
                            0, 2, 1, 3
                        )  # (N, d1, rows, 64)
                # 返回未打包 code (1B/值, 末维 64), 与 ttk CSV 的声明口径一致;
                # fp4 打包在调算子前由 _pack_fp4_codes 完成
                return out.to(torch.uint8)

            def scatter_pa_nz_ks(ds, fill=127):
                t16 = (block_size + 15) // 16
                # PA_NZ cache 中每个块固定占 t16*16 行; skv 不是 block_size 整数倍时末块行数不足,
                # 先沿 S 轴按 fill(=127, e8m0 的 2^0) 补齐, 否则 reshape 到 (N, t16, 16, 2, 2)
                # 会因元素数不匹配直接报 shape 错误(旧实现假设 skv % 16 == 0).
                blk_rows = t16 * 16
                need_rows = max(blocks_per_batch) * blk_rows if B else 0
                if ds.shape[2] < need_rows:
                    pad = torch.full(
                        (ds.shape[0], ds.shape[1], need_rows - ds.shape[2])
                        + tuple(ds.shape[3:]),
                        fill,
                        dtype=ds.dtype,
                    )
                    ds = torch.cat([ds, pad], dim=2)
                out = torch.full(
                    (total_blocks, n_kv, t16, 2, 16, 2), fill, dtype=ds.dtype
                )
                for b_idx in range(B):
                    for j in range(blocks_per_batch[b_idx]):
                        dst_blk = int(block_table_t[b_idx][j])
                        s0 = j * blk_rows
                        seg = ds[b_idx, :, s0 : s0 + blk_rows].reshape(
                            n_kv, t16, 16, 2, 2
                        )
                        out[dst_blk, :, :t16] = seg.permute(
                            0, 1, 3, 2, 4
                        )  # (N, t16, d3, lane, p)
                return out

            def scatter_pa_nz_vs(vs, fill=127):
                out = torch.full(
                    (total_blocks, n_kv, V_D // 16, block_groups, 16, 2),
                    fill,
                    dtype=vs.dtype,
                )
                for b_idx in range(B):
                    for j in range(blocks_per_batch[b_idx]):
                        dst_blk = int(block_table_t[b_idx][j])
                        g0 = j * block_groups
                        g1 = min(g0 + block_groups, vs.shape[2])
                        seg = vs[b_idx, :, g0:g1].reshape(
                            n_kv, g1 - g0, V_D // 16, 16, 2
                        )
                        out[dst_blk, :, :, : g1 - g0] = seg.permute(
                            0, 2, 1, 3, 4
                        )  # (N, d3, tg, 16, 2)
                return out

            key_codes = scatter_pa_nz_kv(key_codes)
            value_codes = scatter_pa_nz_kv(value_codes)
            k_descale = scatter_pa_nz_ks(k_descale)
            v_descale = scatter_pa_nz_vs(v_descale)
        else:  # pa_bbnd: (Bn, Bs, N, ...) 块内 (Bs, N, tail)

            def scatter_pa_bbnd(bnsd, tail_shape, fill=0):
                out = torch.full(
                    (total_blocks, block_size, n_kv) + tuple(tail_shape),
                    fill,
                    dtype=bnsd.dtype,
                )
                for b_idx in range(B):
                    for j in range(blocks_per_batch[b_idx]):
                        dst_blk = int(block_table_t[b_idx][j])
                        s0 = j * block_size
                        s1 = min(s0 + block_size, bnsd.shape[2])
                        src_slice = bnsd[b_idx, :, s0:s1]  # (N, rows, *tail)
                        out[dst_blk, : s1 - s0] = src_slice.permute(
                            1, 0, *range(2, src_slice.dim())
                        )  # (rows, N, *tail)
                return out

            key_codes = scatter_pa_bbnd(key_codes, (qk_d,))
            value_codes = scatter_pa_bbnd(value_codes, (qk_d,))
            k_descale = scatter_pa_bbnd(k_descale, (d_scale, 2), fill=127)

            # v_descale BBND: (Bn, Bs/64, N, D, 2), 组维在 N 前
            def scatter_pa_bbnd_v(v_bnsd):
                # v_bnsd: (B, N, 组数, D, 2) → (Bn, 组数, N, D, 2)
                out = torch.full(
                    (total_blocks, block_groups, n_kv, V_D, 2), 127, dtype=v_bnsd.dtype
                )
                for b_idx in range(B):
                    for j in range(blocks_per_batch[b_idx]):
                        dst_blk = int(block_table_t[b_idx][j])
                        g0 = j * block_groups
                        g1 = min(g0 + block_groups, v_bnsd.shape[2])
                        out[dst_blk, : g1 - g0] = v_bnsd[b_idx, :, g0:g1].permute(
                            1, 0, 2, 3
                        )  # (组,N,D,2)
                return out

            v_descale = scatter_pa_bbnd_v(v_descale)

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
    if is_pa:
        block_table_t = block_table_t.contiguous()
    else:
        block_table_t = _gen_opt_tensor(
            BLOCK_TABLE_SHAPE,
            BLOCK_TABLE_DTYPE or "int32",
            BLOCK_TABLE_DATARANGE,
            seed=100,
        )
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
        q=query_codes.contiguous(),
        k=key_codes.contiguous(),
        v=value_codes.contiguous(),
        q_descale=q_descale.contiguous(),
        k_descale=k_descale.contiguous(),
        v_descale=v_descale.contiguous(),
        block_table=block_table_t,
        p_scale=p_scale_t,
        sinks=sinks_t,
        attn_mask=attn_mask_t,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        # NPU 侧传原始 seqused (可空)
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

    fp32_bnsd = data_dict.get("fp32_bnsd")
    if fp32_bnsd is None:
        raise RuntimeError(
            "data_dict['fp32_bnsd'] is None, generate_data must include fp32_bnsd"
        )
    query_fp32, key_fp32, value_fp32 = fp32_bnsd

    query_bsnd = (
        bnsd_to_bsnd(query_fp32).to(torch.float32).flatten(start_dim=0, end_dim=1)
    )
    key_bsnd = bnsd_to_bsnd(key_fp32).to(torch.float32).flatten(start_dim=0, end_dim=1)
    value_bsnd = (
        bnsd_to_bsnd(value_fp32).to(torch.float32).flatten(start_dim=0, end_dim=1)
    )

    block_q = 128
    block_kv = 4096
    causal = (
        SPARSE_MODE == 3
    )  # mask_mode=3 (CAUSAL): golden 按 left-padding 语义 (causal_offset=sk-sq)
    if _os.environ.get("QFA_GOLDEN_BACKEND", "").lower() == "npu":
        import flash_attention_npu_golden as _npu_golden_mod

        logger.info("[GOLDEN] QFA_GOLDEN_BACKEND=npu, 用 npu_quant_matmul 拼接 golden")
        attn_out = _npu_golden_mod.flash_attention_npu_golden_varlen(
            query_bsnd,
            key_bsnd,
            value_bsnd,
            cu_seqlens_q,
            cu_seqlens_kv,
            act_seq_q_eff,
            act_seq_kv_eff,
            softmax_scale=softmax_scale,
            causal=causal,
            block_q=block_q,
            block_kv=256,  # 对齐 kernel s2BaseSize=256 的 online softmax 分块
            mx_block_size=32,
            mx_mode="baseline",
            s_dtype="fp16",
        )
    else:
        attn_out = flash_attention_cpu_golden_varlen(
            query_bsnd,
            key_bsnd,
            value_bsnd,
            cu_seqlens_q,
            cu_seqlens_kv,
            act_seq_q_eff,
            act_seq_kv_eff,
            softmax_scale=softmax_scale,
            causal=causal,
            quantize=True,
            quantize_p=True,
            block_q=block_q,
            block_kv=block_kv,
            s_layout="DN",
            quantize_p_mode="blockwise_snap_local",
            s_dtype="fp16",
            v_quant_axis="seq_k",
        )

    # s1 是物理 S, attn_out padding 区域为 0 (flash_attention_cpu_golden_varlen 只算有效区域)
    attn_out = attn_out.reshape(b, s1, n1, qk_d).permute(0, 2, 1, 3)  # -> BNSD
    attn_out = rearrange_by_layout(attn_out, attn_out_layout, b, act_seq_q_eff)

    return attn_out.to(get_dtype(OUT_DTYPE)), torch.empty(0, dtype=torch.float32)


# ==============================================================================
# NPU 调用: GRAPH_PATH=0 eager / GRAPH_PATH=7 npugraph_ex
# ==============================================================================
def _pack_fp4_codes(t):
    """未打包 fp4 code(1B/值) -> torch 打包 dtype(2 值/字节), 末维减半."""
    if t is None or "float4" in str(t.dtype):
        return t
    lo = t[..., 0::2].to(torch.uint8)
    hi = t[..., 1::2].to(torch.uint8)
    return (((lo & 0x0F) | ((hi & 0x0F) << 4)).contiguous()).view(
        torch.float4_e2m1fn_x2
    )


def _to_npu(tensor):
    if tensor is None:
        return None
    return tensor.to("npu:%s" % int(DEVICE_ID))


def _call_npu_fa_op(data_dict):
    q = _to_npu(data_dict["q"])
    k = _to_npu(data_dict["k"])
    v = _to_npu(data_dict["v"])
    if KV_STORAGE_MODE == "pa_nz":  # PA_NZ 的 k/v 以未打包 code 生成, 调算子前打包
        k, v = _pack_fp4_codes(k), _pack_fp4_codes(v)
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

    # TND: metadata 要求 batch_size=None (由 cu_seqlens 推导 B)
    meta_batch_size = None if query_layout == "TND" else B

    metadata = quant_flash_attn_metadata(
        num_heads_q=data_dict["num_heads"],
        num_heads_kv=data_dict["num_key_value_heads"],
        head_dim=D,
        quant_mode=Q_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q_t,
        cu_seqlens_kv=cu_seqlens_kv_t,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        batch_size=meta_batch_size,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        head_dim_v=V_D,
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
        is_grad_enabled=False,
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
            batch_size=None if layout_q == "TND" else B,
            max_seqlen_q=MAX_SEQLEN_Q,
            max_seqlen_kv=MAX_SEQLEN_KV,
            head_dim_v=V_D,
            mask_mode=SPARSE_MODE,
            win_left=PRE_TOKENS,
            win_right=NEXT_TOKENS,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
            is_grad_enabled=False,
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
        batch_size=None if query_layout == "TND" else B,
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
        head_dim_v=V_D,
        mask_mode=SPARSE_MODE,
        win_left=PRE_TOKENS,
        win_right=NEXT_TOKENS,
        layout_q=query_layout,
        layout_q_descale=LAYOUT_Q_DESCALE,
        layout_kv=kv_layout,
        layout_out=attn_out_layout,
        is_grad_enabled=False,
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
    if KV_STORAGE_MODE == "pa_nz":  # PA_NZ 的 k/v 以未打包 code 生成, 调算子前打包
        k, v = _pack_fp4_codes(k), _pack_fp4_codes(v)
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

    if METADATA_SHAPE is None:
        metadata = None
        logger.info("[NPU_MAIN] 表格未提供 metadata_shape, metadata 传 None")
    else:
        metadata_shape = tuple(METADATA_SHAPE) if METADATA_SHAPE else (4096,)
        metadata = torch.ones(
            metadata_shape,
            dtype=get_dtype(METADATA_DTYPE) or torch.int32,
            device="npu",
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


golden_mod_self = sys.modules[__name__]
