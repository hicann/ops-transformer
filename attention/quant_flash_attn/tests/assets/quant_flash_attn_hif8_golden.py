# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
HIF8 Flash Attention Golden

功能：生成 BNSD 数据 → per-tensor 量化 → CPU golden 计算 → TND layout 转换 → 精度对比
HIF8: Q/K/V 各只有一个 per-tensor FP32 scale (shape=(1,))
      descale 直接为 FP32 标量，无需 E8M0 转换
      仅支持 TND layout，不支持 PA
      输出固定 BF16
"""

import argparse
import logging
import math

import torch
from torch import nn

try:
    import torch_npu
except ImportError:
    torch_npu = None

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

try:
    from cann_ops_transformer.ops import quant_flash_attn, quant_flash_attn_metadata

    _HAS_NPU = True
except ImportError as e:
    logger.warning("Failed to import cann_ops_transformer.ops: %s", e)
    _HAS_NPU = False

try:
    from . import result_compare_method
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method

# ==============================================================================
# 配置区
# ==============================================================================
import os as _os

GRAPH_PATH = int(_os.environ.get("GRAPH_PATH", "0"))
B = 1
N_q = 1
N_kv = 1
D = 128
# head_dim_v: csv 显式传入则用 csv 值, 否则 None → 算子内默认取 head_dim
HEAD_DIM_V = None

ENABLE_ROPE = False
D_rope = 64

CU_SEQLENS_Q = [0, 4]
CU_SEQLENS_KV = [0, 1024]
SEQUSED_Q = [4]
SEQUSED_KV = [1024]
MAX_SEQLEN_Q = 4
MAX_SEQLEN_KV = 1024


def _derive_seqused(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _get_seqused_q():
    return SEQUSED_Q if SEQUSED_Q is not None else _derive_seqused(CU_SEQLENS_Q)


def _get_seqused_kv():
    return SEQUSED_KV if SEQUSED_KV is not None else _derive_seqused(CU_SEQLENS_KV)


def _actual_seq_q():
    """布局辅助：返回 actual_seq_q（csv SEQUSED_Q 优先，否则从 CU_SEQLENS_Q 差分还原）。"""
    return _get_seqused_q()


def _actual_seq_kv():
    """布局辅助：返回 actual_seq_kv（csv SEQUSED_KV 优先，否则从 CU_SEQLENS_KV 差分还原）。"""
    return _get_seqused_kv()


SPARSE_MODE = 3
HIF8_MAX = 32768.0

INPUT_LAYOUT = "TND"
Q_SCALE_LAYOUT = "TND"
P_SCALE = 1.0

SOFTMAX_SCALE = None

IS_CONTIGUOUS = True

ENABLE_LSE = False

ENABLE_PA = False
BLOCK_SIZE = 0
KV_CACHE_LAYOUT = "TND"
FP8_DTYPE = torch.float8_e4m3fn
HIF8_DTYPE = torch.uint8

SEED_Q = 54
SEED_K = 3
SEED_V = 4
SEED_QR = 8
SEED_KR = 9

DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0

DEVICE_ID = 0


def _get_npu_fa_kwargs():
    return {
        "return_softmax_lse": ENABLE_LSE,
    }


# ==============================================================================
# HIF8 per-tensor 量化
# scale = max_abs / HIF8_MAX, 每个张量一个标量
# 量化: fp32 → trans_float_tensor_to_hifuint8 → uint8 (HIF8 编码)
# 反量化: uint8 → trans_hifuint8_tensor_to_float → fp32, 再乘 scale
# ==============================================================================

try:
    from . import generate_hifloat8_data as hif8_codec
except ImportError:
    import generate_hifloat8_data as hif8_codec


def get_hif8_per_tensor_quant_scale(tensor, hif8_max=HIF8_MAX):
    max_abs = tensor.abs().max().item()
    if max_abs == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([max_abs / hif8_max], dtype=torch.float32)


def hif8_per_tensor_quant(tensor, scale):
    """FP32 → HIF8 uint8: 先除 scale, 再用 HIF8 编码"""
    scaled = (tensor / scale).contiguous()
    return hif8_codec.trans_float_tensor_to_hifuint8(
        scaled, round_mode="round", over_mode=True
    )


def hif8_per_tensor_dequant(tensor_uint8):
    """HIF8 uint8 → FP32: HIF8 解码"""
    return hif8_codec.trans_hifuint8_tensor_to_float(tensor_uint8, over_mode=True)


def hif8_cast_p(p_tensor):
    """模拟 NPU 的 P→HIF8 量化损失"""
    return hif8_codec.trans_hifuint8_tensor_to_float(
        hif8_codec.trans_float_tensor_to_hifuint8(
            p_tensor.contiguous(), round_mode="round", over_mode=True
        )
    )


def broadcast_kv(num_heads, num_kv_heads, kv_tensor):
    factor = num_heads // num_kv_heads
    B, _, S, D = kv_tensor.shape
    result = torch.zeros([B, num_heads, S, D], dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i : i + 1, :, :] = kv_tensor[:, i // factor : i // factor + 1, :, :]
    return result


# ==============================================================================
# Layout 转换函数 - Q/K/V
# ==============================================================================


def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    tensor = (
        tensor_bnsd
        if isinstance(tensor_bnsd, torch.Tensor)
        else torch.as_tensor(tensor_bnsd)
    )
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)

    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return (
            tensor[:, :, :max_org_s, :]
            .permute(0, 2, 1, 3)
            .reshape(B, max_org_s, N * D)
            .contiguous()
        )
    elif layout == "TND":
        if cu_seqlens is not None:
            T = cu_seqlens[-1]
            result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
            for b in range(B):
                act_s = seq_lens[b]
                offset = cu_seqlens[b]
                if act_s <= 0:
                    continue
                for n in range(N):
                    result[offset : offset + act_s, n, :] = tensor[b, n, :act_s, :]
            return result.contiguous()
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t : t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")


def convert_kv_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    return convert_q_bnsd_to_layout(
        tensor_bnsd, seq_lens, layout, cu_seqlens=cu_seqlens
    )


def fill_tnd_padding(tensor_tnd, seq_lens, cu_seqlens, fill_value=float("inf")):
    if cu_seqlens is None:
        return tensor_tnd
    B = len(seq_lens)
    for b in range(B):
        act_s = seq_lens[b]
        offset = cu_seqlens[b]
        cu_diff = cu_seqlens[b + 1] - cu_seqlens[b]
        if cu_diff > act_s:
            tensor_tnd[offset + act_s : offset + cu_diff] = fill_value
    return tensor_tnd


def convert_qk_rope_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    return convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout)


def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result


# ==============================================================================
# 数据生成
# ==============================================================================


def generate_data():
    """生成 BNSD FP16 Q/K/V 并做 HIF8 per-tensor 量化"""
    max_sq = MAX_SEQLEN_Q
    max_skv = MAX_SEQLEN_KV
    seqused_q = _get_seqused_q()
    seqused_kv = _get_seqused_kv()
    if max_sq is None or max_sq < 0:
        max_sq = max(seqused_q)
    if max_skv is None or max_skv < 0:
        max_skv = max(seqused_kv)

    logger.info("[INFO] max_sq=%d, max_skv=%d", max_sq, max_skv)

    torch.manual_seed(SEED_Q)
    q_fp16 = (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_Q

    torch.manual_seed(SEED_K)
    k_fp16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1
    ) * DATA_RANGE_K

    torch.manual_seed(SEED_V)
    v_fp16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1
    ) * DATA_RANGE_V

    qr_bf16 = None
    kr_bf16 = None
    if ENABLE_ROPE and D_rope > 0:
        torch.manual_seed(SEED_QR)
        qr_bf16 = torch.randn(B, N_q, max_sq, D_rope, dtype=torch.bfloat16)
        torch.manual_seed(SEED_KR)
        kr_bf16 = torch.randn(B, N_kv, max_skv, D_rope, dtype=torch.bfloat16)

    logger.info(
        "[INFO] q_fp16=%s, k_fp16=%s, v_fp16=%s",
        q_fp16.shape,
        k_fp16.shape,
        v_fp16.shape,
    )

    quant_scale_q = get_hif8_per_tensor_quant_scale(q_fp16.to(torch.float32))
    quant_scale_k = get_hif8_per_tensor_quant_scale(k_fp16.to(torch.float32))
    quant_scale_v = get_hif8_per_tensor_quant_scale(v_fp16.to(torch.float32))

    dequant_scale_q = quant_scale_q
    dequant_scale_k = quant_scale_k
    v_descale = quant_scale_v

    q_fp8 = hif8_per_tensor_quant(q_fp16.to(torch.float32), quant_scale_q)
    k_fp8 = hif8_per_tensor_quant(k_fp16.to(torch.float32), quant_scale_k)
    v_fp8 = hif8_per_tensor_quant(v_fp16.to(torch.float32), quant_scale_v)

    logger.info(
        "[INFO] HIF8 per-tensor: q_scale=%.6f, k_scale=%.6f, v_scale=%.6f",
        dequant_scale_q.item(),
        dequant_scale_k.item(),
        v_descale.item(),
    )

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    return (
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        v_descale,
        p_scale,
        qr_bf16,
        kr_bf16,
        None,
    )


# ==============================================================================
# CPU Golden
# ==============================================================================


def _build_attention_mask(b, Sq, Skv, actual_seq_q, actual_seq_kv, sparse_mode):
    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32)
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32)
    q_lens_acl = q_lens_t.view(b, 1, 1, 1)
    k_lens_acl = k_lens_t.view(b, 1, 1, 1)

    q_range = torch.arange(Sq).view(1, 1, -1, 1)
    k_range = torch.arange(Skv).view(1, 1, 1, -1)
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if sparse_mode == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        return causal_mask | q_padding_mask | k_padding_mask
    else:
        return q_padding_mask | k_padding_mask


def _compute_s_block(
    Qi, Kj, deq_scale_q, deq_scale_k, softmax_scale, Qri=None, Krj=None
):
    """HIF8: descale 是 per-tensor 标量，直接标量乘法"""
    S_ij = torch.matmul(Qi * deq_scale_q, (Kj * deq_scale_k).permute(0, 1, 3, 2))
    if Qri is not None and Krj is not None:
        S_ij += torch.matmul(
            Qri.to(torch.float32), Krj.to(torch.float32).permute(0, 1, 3, 2)
        )
    return S_ij * softmax_scale


def _online_softmax_update(S_ij, mask_j, mi, si, oi, ln_p_scale):
    S_ij = S_ij.masked_fill(mask_j, float("-inf"))

    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
    m_block_j = torch.max(mi, m_block_j)
    m_block_j_copy = m_block_j - ln_p_scale

    P_ij_raw = torch.exp(S_ij - m_block_j_copy)
    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
    P_ij_drop = hif8_cast_p(P_ij_raw)

    return m_block_j, s_block_j, P_ij_drop


def cpu_hif8_golden(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    actual_seq_q,
    actual_seq_kv,
    softmax_scale=None,
    qr_bf16=None,
    kr_bf16=None,
):
    """CPU Flash Attention golden with HIF8 per-tensor quant
    HIF8: S1=128, S2=256, 每轮 C1V1 处理 1 个 K block (256) + 1 个 V block (256)
    """
    EPSILON = 1e-20
    Q_BLOCK_SIZE = 128
    K_BLOCK_SIZE = 256
    V_BLOCK_SIZE = 256

    if actual_seq_q is None:
        actual_seq_q = _get_seqused_q()
    if actual_seq_kv is None:
        actual_seq_kv = _get_seqused_kv()

    q_tensor = hif8_per_tensor_dequant(q_fp8)
    k_tensor = hif8_per_tensor_dequant(k_fp8)
    v_tensor = hif8_per_tensor_dequant(v_fp8)

    if N_q != N_kv:
        logger.info("[INFO] GQA 广播")
        k_tensor = broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = broadcast_kv(N_q, N_kv, v_tensor)
        if kr_bf16 is not None:
            kr_bf16 = broadcast_kv(N_q, N_kv, kr_bf16)

    b, n, _s, d = q_tensor.shape
    d_total = d + (D_rope if ENABLE_ROPE else 0)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_total)
    dv = v_tensor.shape[-1]
    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]

    minValue = torch.tensor(-3.402823466e38, dtype=torch.float32)
    out = torch.zeros([b, n, Sq, dv], dtype=torch.float32)
    o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]
    o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[
        ..., None
    ]

    TILES_Q = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    TILES_KV = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

    mask_global = _build_attention_mask(
        b, Sq, Skv, actual_seq_q, actual_seq_kv, SPARSE_MODE
    )

    Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
    V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
    o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))

    Qr_BLOCKS = None
    Kr_BLOCKS = None
    if ENABLE_ROPE and qr_bf16 is not None:
        Qr_BLOCKS = list(torch.split(qr_bf16, Q_BLOCK_SIZE, dim=2))
        Kr_BLOCKS = list(torch.split(kr_bf16, K_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)

    logger.info(
        "[CPU Golden] TILES_Q=%d, TILES_KV=%d, Sq=%d, Skv=%d",
        TILES_Q,
        TILES_KV,
        Sq,
        Skv,
    )

    for i in range(TILES_Q):
        Qi = Q_BLOCKS[i]
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
        Qri = Qr_BLOCKS[i] if Qr_BLOCKS is not None else None

        for j in range(TILES_KV):
            oi, si, mi = o_BLOCKS[i], s_BLOCKS[i], m_BLOCKS[i]

            Kj = K_BLOCKS[j]
            Sk_start = j * K_BLOCK_SIZE
            Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
            Krj = Kr_BLOCKS[j] if Kr_BLOCKS is not None else None

            S_ij = _compute_s_block(
                Qi, Kj, dequant_scale_q, dequant_scale_k, softmax_scale, Qri, Krj
            )
            mask_j = mask_global[:, :, Sq_start:Sq_end, Sk_start:Sk_end]
            m_block_j, s_block_j, P_ij_drop = _online_softmax_update(
                S_ij, mask_j, mi, si, oi, ln_p_scale
            )

            Vj = V_BLOCKS[j]
            Vj_dequant = Vj * v_descale

            P_ij_Vj = torch.matmul(P_ij_drop, Vj_dequant[:, :, : Kj.shape[2], :])
            update_mul_si = torch.exp(mi - m_block_j)
            si_new = update_mul_si * si + s_block_j
            o_BLOCKS[i] = update_mul_si * oi + P_ij_Vj
            s_BLOCKS[i] = si_new
            m_BLOCKS[i] = m_block_j

    out = torch.cat(o_BLOCKS, dim=2)
    out_sum = torch.cat(s_BLOCKS, dim=2)
    out = out / (out_sum + EPSILON)

    o_max = torch.cat(m_BLOCKS, dim=2)
    all_masked = o_max <= minValue.item()
    lse = torch.where(
        all_masked,
        torch.full_like(o_max, float("inf")),
        o_max + torch.log(out_sum + EPSILON),
    )
    out = torch.where(all_masked, torch.zeros_like(out), out)
    logger.info("[CPU Golden] output=%s", out.shape)
    return out, lse


def _call_npu_fa_op(
    q,
    k,
    v,
    mask,
    cu_seqlens_q_t,
    cu_seqlens_kv_t,
    seqused_q_t,
    seqused_kv_t,
    max_seqlen_q,
    max_seqlen_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    block_size,
    sparse_mode,
    out_dtype,
):
    """调用 NPU 双算子 (QFA: quant_flash_attn_metadata + quant_flash_attn)"""
    if not _HAS_NPU:
        raise ImportError(
            "cann_ops_transformer.ops.quant_flash_attn is not available. "
            "Please check that cann_ops_transformer is installed and all .so are compiled."
        )

    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"

    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=q_n,
        num_heads_kv=kv_n,
        head_dim=q.shape[-1],
        quant_mode=0,
        cu_seqlens_q=cu_seqlens_q_t if is_tnd_q else None,
        cu_seqlens_kv=cu_seqlens_kv_t if is_tnd_kv else None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        head_dim_v=HEAD_DIM_V,
        mask_mode=sparse_mode,
        layout_q=layout_q,
        layout_q_descale=layout_q_descale,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=B if not is_tnd_q else None,
    )

    main_kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "q_descale": dequant_scale_q,
        "k_descale": dequant_scale_k,
        "v_descale": dequant_scale_v,
        "quant_mode": 0,
        "block_table": block_table,
        "p_scale": p_scale,
        "cu_seqlens_q": cu_seqlens_q_t if is_tnd_q else None,
        "cu_seqlens_kv": cu_seqlens_kv_t if is_tnd_kv else None,
        "seqused_q": seqused_q_t,
        "seqused_kv": seqused_kv_t,
        "attn_mask": mask,
        "metadata": metadata,
        "softmax_scale": softmax_scale,
        "mask_mode": sparse_mode,
        "layout_q": layout_q,
        "layout_q_descale": layout_q_descale,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
    }
    main_kwargs.update(_get_npu_fa_kwargs())
    atten_out, lse_out = quant_flash_attn(**main_kwargs)
    torch.npu.synchronize()
    return atten_out, lse_out


class Network(nn.Module):
    """aclgraph 编译目标: forward 只包含两个 torch.library op 调用。

    输入已预处理:
      - cu_seqlens_q/kv, seqused_q/kv: 由 hif8_fa_torch_npu 转好 NPU tensor
      - q/k/v/deq_*: 已转好 layout 的 NPU tensor
      - max_seqlen_q/kv: 已计算好的标量
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        block_table,
        q_n,
        kv_n,
        softmax_scale,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        block_size,
        sparse_mode,
        out_dtype,
        max_seqlen_q,
        max_seqlen_kv,
        batch_size,
    ):
        is_tnd_q = layout_q == "TND"
        is_tnd_kv = layout_kv == "TND"
        metadata = quant_flash_attn_metadata(
            num_heads_q=q_n,
            num_heads_kv=kv_n,
            head_dim=q.shape[-1],
            quant_mode=0,
            cu_seqlens_q=cu_seqlens_q if is_tnd_q else None,
            cu_seqlens_kv=cu_seqlens_kv if is_tnd_kv else None,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            head_dim_v=HEAD_DIM_V,
            mask_mode=sparse_mode,
            layout_q=layout_q,
            layout_q_descale=layout_q_descale,
            layout_kv=layout_kv,
            layout_out=layout_out,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=batch_size if not is_tnd_q else None,
        )
        main_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "q_descale": dequant_scale_q,
            "k_descale": dequant_scale_k,
            "v_descale": dequant_scale_v,
            "quant_mode": 0,
            "block_table": block_table,
            "p_scale": p_scale,
            "cu_seqlens_q": cu_seqlens_q if is_tnd_q else None,
            "cu_seqlens_kv": cu_seqlens_kv if is_tnd_kv else None,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "attn_mask": mask,
            "metadata": metadata,
            "softmax_scale": softmax_scale,
            "mask_mode": sparse_mode,
            "layout_q": layout_q,
            "layout_q_descale": layout_q_descale,
            "layout_kv": layout_kv,
            "layout_out": layout_out,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
        }
        main_kwargs.update(_get_npu_fa_kwargs())
        atten_out, lse_out = quant_flash_attn(**main_kwargs)
        return atten_out, lse_out


def _build_causal_mask():
    # sparse_mode=0 不需要 mask，其他模式需要上三角 causal mask
    if SPARSE_MODE == 0:
        return None
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


def prepare_npu_inputs(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    input_layout=None,
    cu_seqlens_q_t=None,
    cu_seqlens_kv_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """准备 NPU 侧入参

    返回字典的 key 与 _call_npu_fa_op 的形参名一一对应:
      q, k, v, mask,
      cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q, max_seqlen_kv,
      dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
      block_table, q_n, kv_n, softmax_scale,
      layout_q, layout_q_descale, layout_kv, layout_out, block_size, sparse_mode, out_dtype
    其中 cu_seqlens_q/kv、seqused_q/kv 为 python list (或 None), 由 _call_npu_fa_op 负责转 NPU tensor;
    其余 tensor 字段均为已就绪的 NPU tensor.
    """
    torch_npu.npu.set_device(int(DEVICE_ID))
    eff_layout = input_layout or INPUT_LAYOUT or "TND"

    softmax_scale = SOFTMAX_SCALE

    q_runtime_layout = Q_SCALE_LAYOUT if Q_SCALE_LAYOUT is not None else "TND"

    q_npu = q_fp8.contiguous().view(HIF8_DTYPE).npu()
    deq_q_npu = dequant_scale_q.npu()
    p_scale_npu = p_scale.npu()

    out_dtype = torch.float16
    mask_arg = _build_causal_mask()

    if ENABLE_PA:
        k_npu = k_fp8.contiguous().view(HIF8_DTYPE).npu()
        v_npu = v_fp8.contiguous().view(HIF8_DTYPE).npu()
        deq_k_npu = dequant_scale_k.npu()
        deq_v_npu = dequant_scale_v.npu()

        if not IS_CONTIGUOUS:
            kv_cache = torch.stack([k_fp8, v_fp8], dim=2)
            kv_cache = kv_cache.npu()
            k_npu = kv_cache[:, :, 0]
            v_npu = kv_cache[:, :, 1]
            logger.info(
                f"[NPU] key is_contiguous={k_npu.is_contiguous()}, value is_contiguous={v_npu.is_contiguous()}"
            )
            fake_kscale_tensor = torch.ones_like(dequant_scale_k)
            fake_vscale_tensor = torch.ones_like(dequant_scale_v)
            double_kscale = torch.stack([dequant_scale_k, fake_kscale_tensor], dim=2)
            double_vscale = torch.stack([dequant_scale_v, fake_vscale_tensor], dim=2)
            double_kscale = double_kscale.npu()
            double_vscale = double_vscale.npu()
            deq_k_npu = double_kscale[:, :, 0]
            deq_v_npu = double_vscale[:, :, 0]
            logger.info(
                f"[NPU] deq_k_scale is_contiguous={deq_k_npu.is_contiguous()}, deq_v_scale is_contiguous={deq_v_npu.is_contiguous()}"
            )

        logger.info("[NPU PA] kv_layout=%s", KV_CACHE_LAYOUT)
        logger.info("[NPU PA] k=%s, v=%s", k_npu.shape, v_npu.shape)
        logger.info("[NPU PA] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)

        block_table_npu = (
            block_table_torch.npu()
            if isinstance(block_table_torch, torch.Tensor)
            else torch.as_tensor(block_table_torch, dtype=torch.int32).npu()
        )

        _pa_layout_kv_map = {"BnNBsD": "PA_BNBD", "PA_NZ": "PA_NZ"}
        pa_layout_kv = _pa_layout_kv_map.get(KV_CACHE_LAYOUT, "PA_BNBD")

        logger.info("[NPU] prepare PA inputs done.")
        return dict(
            q=q_npu,
            k=k_npu,
            v=v_npu,
            mask=mask_arg,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            cu_seqlens_q_t=cu_seqlens_q_t,
            cu_seqlens_kv_t=cu_seqlens_kv_t,
            seqused_q_t=seqused_q_t,
            seqused_kv_t=seqused_kv_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            dequant_scale_q=deq_q_npu,
            dequant_scale_k=deq_k_npu,
            dequant_scale_v=deq_v_npu,
            p_scale=p_scale_npu,
            block_table=block_table_npu,
            q_n=N_q,
            kv_n=N_kv,
            softmax_scale=softmax_scale,
            layout_q="TND",
            layout_q_descale=q_runtime_layout,
            layout_kv=pa_layout_kv,
            layout_out="TND",
            block_size=BLOCK_SIZE,
            sparse_mode=SPARSE_MODE,
            out_dtype=out_dtype,
        )

    # 非 PA 模式
    k_npu = k_fp8.contiguous().view(HIF8_DTYPE).npu()
    v_npu = v_fp8.contiguous().view(HIF8_DTYPE).npu()
    deq_k_npu = dequant_scale_k.npu()
    deq_v_npu = dequant_scale_v.npu()
    logger.info("[NPU TND] k=%s, v=%s", k_npu.shape, v_npu.shape)
    logger.info("[NPU TND] deq_k=%s, deq_v=%s", deq_k_npu.shape, deq_v_npu.shape)

    logger.info("[NPU] prepare non-PA inputs done.")
    return dict(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        mask=mask_arg,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        cu_seqlens_q_t=cu_seqlens_q_t,
        cu_seqlens_kv_t=cu_seqlens_kv_t,
        seqused_q_t=seqused_q_t,
        seqused_kv_t=seqused_kv_t,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        dequant_scale_q=deq_q_npu,
        dequant_scale_k=deq_k_npu,
        dequant_scale_v=deq_v_npu,
        p_scale=p_scale_npu,
        block_table=None,
        q_n=N_q,
        kv_n=N_kv,
        softmax_scale=softmax_scale,
        layout_q=eff_layout,
        layout_q_descale=q_runtime_layout,
        layout_kv=eff_layout,
        layout_out=eff_layout,
        block_size=0,
        sparse_mode=SPARSE_MODE,
        out_dtype=out_dtype,
    )


def npu_hif8_fa(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    cu_seqlens_q_t=None,
    cu_seqlens_kv_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """调用 NPU 算子，支持 N2TGD layout

    正常用例入口：准备入参 → 调用算子 → 输出截断/LSE 处理。
    """
    global INPUT_LAYOUT
    if q_fp8.dim() == 3:
        inferred_layout = "TND"
    elif INPUT_LAYOUT and INPUT_LAYOUT != "TND":
        inferred_layout = INPUT_LAYOUT
    else:
        inferred_layout = "BSND"
    if INPUT_LAYOUT in (None, "TND") and inferred_layout != "TND":
        INPUT_LAYOUT = inferred_layout
        logger.info(
            "[NPU] npu_hif8_fa: 从q维度推断layout=%s, 已更新INPUT_LAYOUT",
            inferred_layout,
        )

    # cu_seqlens/seqused 的 _t 优先 (由 inputs.py in-place 写入 tensor slot)
    def _as_tensor(t, values):
        if t is not None and t.numel() > 0:
            return t.npu()
        if values is not None:
            return torch.tensor(values, dtype=torch.int32).npu()
        return None

    cu_seqlens_q_t = _as_tensor(cu_seqlens_q_t, cu_seqlens_q)
    cu_seqlens_kv_t = _as_tensor(cu_seqlens_kv_t, cu_seqlens_kv)
    seqused_q_t = _as_tensor(seqused_q_t, seqused_q)
    seqused_kv_t = _as_tensor(seqused_kv_t, seqused_kv)

    inputs = prepare_npu_inputs(
        q_fp8,
        k_fp8,
        v_fp8,
        dequant_scale_q,
        dequant_scale_k,
        dequant_scale_v,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q,
        max_seqlen_kv,
        block_table_torch=block_table_torch,
        input_layout=INPUT_LAYOUT,
        cu_seqlens_q_t=cu_seqlens_q_t,
        cu_seqlens_kv_t=cu_seqlens_kv_t,
        seqused_q_t=seqused_q_t,
        seqused_kv_t=seqused_kv_t,
    )

    logger.info(
        "[NPU] 调用 %s 模式 (GRAPH_PATH=%d)...",
        "PA" if ENABLE_PA else inputs["layout_q"],
        GRAPH_PATH,
    )
    # 通过 hif8_fa_torch_npu 路由，支持 GRAPH_PATH=7 (aclgraph)
    atten_out, lse_out = hif8_fa_torch_npu(**inputs)

    act_seqused_q = _get_seqused_q()
    npu_output = atten_out
    if npu_output.dim() == 3:
        T_actual = (
            cu_seqlens_q[-1]
            if cu_seqlens_q is not None
            else (
                sum(act_seqused_q) if act_seqused_q is not None else npu_output.shape[0]
            )
        )
        if npu_output.shape[0] > T_actual:
            npu_output = npu_output[:T_actual]
    logger.info("[NPU] output=%s", npu_output.shape)
    return npu_output, lse_out


def hif8_fa_torch_npu(
    q,
    k,
    v,
    mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    dequant_scale_q,
    dequant_scale_k,
    dequant_scale_v,
    p_scale,
    block_table,
    q_n,
    kv_n,
    softmax_scale,
    layout_q,
    layout_q_descale,
    layout_kv,
    layout_out,
    block_size,
    sparse_mode,
    out_dtype,
    cu_seqlens_q_t=None,
    cu_seqlens_kv_t=None,
    seqused_q_t=None,
    seqused_kv_t=None,
):
    """
    NPU 调用入口, 支持 GRAPH_PATH=0 (单算子) 和 GRAPH_PATH=7 (aclgraph)
    """

    if GRAPH_PATH == 0:
        logger.info("[NPU] 调用 QFA 单算子模式...")
        return _call_npu_fa_op(
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            max_seqlen_q,
            max_seqlen_kv,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout_q,
            layout_q_descale,
            layout_kv,
            layout_out,
            block_size,
            sparse_mode,
            out_dtype,
        )

    # GRAPH_PATH == 7: aclgraph
    # 预处理: list → NPU tensor (必须在编译区域之外完成)
    if cu_seqlens_q_t is None and cu_seqlens_q is not None:
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32).npu()
    if cu_seqlens_kv_t is None and cu_seqlens_kv is not None:
        cu_seqlens_kv_t = torch.tensor(cu_seqlens_kv, dtype=torch.int32).npu()
    if seqused_q_t is None and seqused_q is not None:
        seqused_q_t = torch.tensor(seqused_q, dtype=torch.int32).npu()
    if seqused_kv_t is None and seqused_kv is not None:
        seqused_kv_t = torch.tensor(seqused_kv, dtype=torch.int32).npu()

    npu_mode = Network().to(f"npu:{int(DEVICE_ID)}")
    with torch.no_grad():
        torch.npu.synchronize()

        # Network.forward 只调 torch.library op，Python 预处理结果通过参数传入
        fa_args = (
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
            q_n,
            kv_n,
            softmax_scale,
            layout_q,
            layout_q_descale,
            layout_kv,
            layout_out,
            block_size,
            sparse_mode,
            out_dtype,
            max_seqlen_q,
            max_seqlen_kv,
            q.shape[0] if layout_q != "TND" else None,
        )

        # aclgraph: 直接使用 npugraph_ex backend
        logger.info("[NPU] 调用 aclgraph (npugraph_ex)...")
        npu_backend = "npugraph_ex"
        npu_mode = torch.compile(
            npu_mode, fullgraph=False, backend=npu_backend, dynamic=False
        )
        # mark_static 标记所有 tensor 输入
        for t in (
            q,
            k,
            v,
            mask,
            cu_seqlens_q_t,
            cu_seqlens_kv_t,
            seqused_q_t,
            seqused_kv_t,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            block_table,
        ):
            if t is not None:
                torch._dynamo.mark_static(t)
        atten_out, lse_out = npu_mode(*fa_args)

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        torch.npu.synchronize()
        return atten_out, lse_out


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    try:
        from . import golden_cache
    except ImportError:
        import golden_cache

    _VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}

    parser = argparse.ArgumentParser(description="HIF8 Flash Attention Golden")
    parser.add_argument(
        "--mode",
        default="all",
        help="执行模式，支持逗号组合: all/gen/cpu/npu/compare. 例: --mode=npu,compare",
    )
    parser.add_argument(
        "--case-name", default="default", help="case 名称，用于 .pt 文件命名"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="缓存目录路径（默认 golden_cache/）"
    )
    args = parser.parse_args()

    raw_parts = {m.strip() for m in args.mode.split(",") if m.strip()}
    invalid = raw_parts - _VALID_MODES
    if invalid:
        parser.error(f"Invalid mode: {invalid}. Valid: {_VALID_MODES}")
    mode = {"gen", "cpu", "npu", "compare"} if "all" in raw_parts else raw_parts

    case_name = args.case_name
    cdir = args.cache_dir

    logger.info("=" * 60)
    logger.info("HIF8 Flash Attention Golden  [mode=%s, case=%s]", mode, case_name)
    logger.info("输出: 逐元素表格 + 统计汇总 (PctRlt 通过率)")
    logger.info("=" * 60)
    logger.info("场景: %s", "PA" if ENABLE_PA else "TND")
    logger.info("INPUT_LAYOUT=%s, Q_SCALE_LAYOUT=%s", INPUT_LAYOUT, Q_SCALE_LAYOUT)
    logger.info("KV_CACHE_LAYOUT=%s", KV_CACHE_LAYOUT)
    logger.info("B=%d, N_q=%d, N_kv=%d, D=%d", B, N_q, N_kv, D)
    logger.info(
        "ACTUAL_SEQ_Q=%s, ACTUAL_SEQ_KV=%s", _get_seqused_q(), _get_seqused_kv()
    )

    if "gen" in mode:
        logger.info("\n[Step 1] 数据生成")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            _,
            _,
            block_table_torch,
        ) = generate_data()
        golden_cache.save_input(
            case_name,
            golden_cache.build_input_dict(
                q_fp8,
                k_fp8,
                v_fp8,
                dequant_scale_q,
                dequant_scale_k,
                dequant_scale_v,
                p_scale,
                None,
                None,
                block_table_torch,
            ),
            cache_dir=cdir,
        )
    else:
        logger.info("\n[Step 1] 加载已保存的输入数据")
        (
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            _,
            _,
            block_table_torch,
        ) = golden_cache.load_input(case_name, cache_dir=cdir)

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        logger.info("\n[Done] 数据已保存，退出")
        sys.exit(0)

    if "cpu" in mode:
        logger.info("\n[Step 2] CPU Golden")
        cpu_out, cpu_lse = cpu_hif8_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            _get_seqused_q(),
            _get_seqused_kv(),
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        logger.info("\n[Done] CPU 输出已保存，退出")
        sys.exit(0)

    if "npu" in mode:
        logger.info("\n[Step 3] NPU 调用")
        atten_out, lse_out = npu_hif8_fa(
            q_fp8,
            k_fp8,
            v_fp8,
            dequant_scale_q,
            dequant_scale_k,
            dequant_scale_v,
            p_scale,
            CU_SEQLENS_Q,
            CU_SEQLENS_KV,
            SEQUSED_Q,
            SEQUSED_KV,
            MAX_SEQLEN_Q,
            MAX_SEQLEN_KV,
            block_table_torch,
        )
        golden_cache.save_npu_output(case_name, atten_out, lse_out, cache_dir=cdir)
    else:
        atten_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        logger.info("\n[Done] NPU 输出已保存，退出")
        sys.exit(0)

    logger.info("\n[Step 4] Atten OUT 精度对比")
    cpu_tnd_torch = convert_q_bnsd_to_layout(
        cpu_out, _get_seqused_q(), "TND", cu_seqlens=CU_SEQLENS_Q
    )
    result_compare_method.check_result(cpu_tnd_torch, atten_out)

    if ENABLE_LSE:
        logger.info("\n[Step 5] LSE 精度对比")
        cpu_lse_tnd_torch = convert_q_bnsd_to_layout(
            cpu_lse, _get_seqused_q(), "TND", cu_seqlens=CU_SEQLENS_Q
        )
        fill_tnd_padding(
            cpu_lse_tnd_torch, _get_seqused_q(), CU_SEQLENS_Q, fill_value=float("inf")
        )
        # NPU LSE 输出为 N-major 排布 (N, T): N 在外, T 在内
        # CPU golden 经 convert 后是 [T, N, 1] (T-major), 需转成 [N, T] 对齐
        cpu_lse_nt_torch = cpu_lse_tnd_torch.squeeze(-1).permute(1, 0).contiguous()
        result_compare_method.check_result(cpu_lse_nt_torch, lse_out)
