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
MxFP8 Softmax FP16 Golden（quant_mode=3 / BNSD / 非 PA）

功能：生成 BNSD fp8 数据（5D scale）→ CPU fp32 softmax golden → NPU 调用 → 精度对比
量化：
  - Q/K: 沿 D 每 64 元素一组 shared exponent（e8m0），descale [B,N,S,D//64,2]
  - V:   沿 S 每 64 元素一组 shared exponent（e8m0），descale [B,N,ceil(S/64),D,2]
         （任意 S 泛化：尾组不足 64 行按 0 补齐计组内 max，2026-09-24）
"""

import logging
import math

import torch

_HAS_NPU = None


def _ensure_npu_imports():
    global _HAS_NPU
    if _HAS_NPU is None:
        import torch_npu  # noqa: F401

        try:
            from cann_ops_transformer.ops import (
                quant_flash_attn_metadata,
                quant_flash_attn,
            )

            _HAS_NPU = True
        except ImportError:
            _HAS_NPU = False
    return _HAS_NPU


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局参数（test_runner.apply_params 会按 PARAM_MAP 覆盖这些变量）
# ==============================================================================
B = 1
N_q = 1
N_kv = 1
D = 128

CU_SEQLENS_Q = None
CU_SEQLENS_KV = None
SEQUSED_Q = [256]
SEQUSED_KV = [256]
MAX_SEQLEN_Q = -1
MAX_SEQLEN_KV = -1

ENABLE_PA = False
KV_CACHE_LAYOUT = None
BLOCK_SIZE = None
SPARSE_MODE = 0
Q_SCALE_LAYOUT = "BNSD"
P_SCALE = 1.0
SOFTMAX_SCALE = None

DATA_RANGE_Q = 1.0
DATA_RANGE_K = 1.0
DATA_RANGE_V = 1.0
DATA_RANGE_QR = 1.0
DATA_RANGE_KR = 1.0

# 数据偏置：value = offset + (rand*2-1)*range，上下限 = [offset-range, offset+range]。
# offset=1 且 range=0 → 恒为 1（全 1 调试用例：score 全等 → softmax 均匀 → 输出精确全 1）
DATA_OFFSET_Q = 0.0
DATA_OFFSET_K = 0.0
DATA_OFFSET_V = 0.0

ENABLE_LSE = False
DEVICE_ID = 0
IS_CONTIGUOUS = True

# ==============================================================================
# 常量
# ==============================================================================
QUANT_MODE = 3  # A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16
INPUT_LAYOUT = "BNSD"
FP8_DTYPE = torch.float8_e4m3fn
QUANT_GROUP_SIZE = 64  # Q/K 沿 D，V 沿 S，每 64 元素一组
E8M0_MIN_POSITIVE = 2 ** (-127)
SEED_Q = 54
SEED_K = 3
SEED_V = 4


def _get_seqused_q():
    return SEQUSED_Q


def _get_seqused_kv():
    return SEQUSED_KV


# ==============================================================================
# e8m0 编解码
# ==============================================================================
def fp32_to_e8m0fnu(tensor_fp32):
    """FP32 -> e8m0fnu：提取 IEEE754 biased exponent（8bit）。"""
    bits = tensor_fp32.float().view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    return biased_exp.view(torch.float8_e8m0fnu)


def e8m0_to_fp32(tensor_e8m0):
    """e8m0fnu -> FP32：2^(biased_exp - 127)。"""
    biased_exp = tensor_e8m0.view(torch.uint8).to(torch.int32)
    return torch.pow(2.0, biased_exp - 127).to(torch.float32)


def sanitize_e8m0_scale(scale, name="scale"):
    """e8m0 无 0 值语义，非有限/0 值替换为最小正数。"""
    result = torch.as_tensor(scale, dtype=torch.float32).clone()
    bad = ~torch.isfinite(result) | (result == 0)
    if int(bad.sum().item()):
        logger.info(
            "[WARN] %s: replace %d bad scale value(s)", name, int(bad.sum().item())
        )
        result[bad] = E8M0_MIN_POSITIVE
    return result


# ==============================================================================
# 量化 scale（descale 原始 fp32）
# ==============================================================================
def _get_qk_block_scale(tensor):
    """Q/K：沿 D 每 QUANT_GROUP_SIZE 元素一组 shared exponent -> [B,N,S,D//G]。"""
    B_, N_, S_, D_ = tensor.shape
    num_groups = D_ // QUANT_GROUP_SIZE
    grouped = tensor.reshape(B_, N_, S_, num_groups, QUANT_GROUP_SIZE)
    max_vals = torch.max(torch.abs(grouped), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8  # e4m3fn emax=8
    return 2**shared_exp  # [B,N,S,G]


def _get_v_block_scale(tensor):
    """V：沿 S 每 QUANT_GROUP_SIZE 元素一组 shared exponent（每组内每 D 一值）-> [B,N,ceil(S/G),D]。

    任意 S 泛化（2026-09-24）：组数由 S//G（整除约束）改为 ceil(S/G)——尾组不足 G 行时按 0
    补齐参与组内 abs-max（0 不抬升 max，尾组 scale 仅由真实行决定）。与 kernel 侧
    CopyVScaleGmToL1/InitVScaleBuffer 的 ceil 组语义及 GM 布局 [ceil(S/G), D, 2] 对齐；
    _quantize_v/cpu 参考的 repeat_interleave(G)+截断到 S 写法天然兼容尾组，无需改动。
    """
    B_, N_, S_, D_ = tensor.shape
    num_groups = (S_ + QUANT_GROUP_SIZE - 1) // QUANT_GROUP_SIZE
    pad_rows = num_groups * QUANT_GROUP_SIZE - S_
    if pad_rows:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))  # S 轴尾部补 0
    grouped = tensor.reshape(B_, N_, num_groups, QUANT_GROUP_SIZE, D_)
    max_vals = torch.max(torch.abs(grouped), dim=-2)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    return 2**shared_exp  # [B,N,G,D]


def _pack_scale_5d(scale):
    """scale [..., G] -> [..., G, 2]，尾轴 2 重复（L0 排布最小 2Byte）。"""
    return scale.unsqueeze(-1).expand(*scale.shape, 2).contiguous()


def _quantize_qk(tensor, scale):
    scale_exp = scale.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)[
        ..., : tensor.shape[-1]
    ]
    return (tensor / scale_exp).clamp(-448.0, 448.0).to(FP8_DTYPE)


def _quantize_v(tensor, scale):
    # scale [B,N,G,D] 沿 S 扩到 [B,N,S,D]
    scale_exp = scale.repeat_interleave(QUANT_GROUP_SIZE, dim=2)[
        :, :, : tensor.shape[2], :
    ]
    return (tensor / scale_exp).clamp(-448.0, 448.0).to(FP8_DTYPE)


# ==============================================================================
# 数据生成
# ==============================================================================
def generate_data():
    max_sq = (
        MAX_SEQLEN_Q
        if (MAX_SEQLEN_Q is not None and MAX_SEQLEN_Q > 0)
        else max(_get_seqused_q())
    )
    max_skv = (
        MAX_SEQLEN_KV
        if (MAX_SEQLEN_KV is not None and MAX_SEQLEN_KV > 0)
        else max(_get_seqused_kv())
    )

    torch.manual_seed(SEED_Q)
    q_fp16 = (
        DATA_OFFSET_Q
        + (torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_Q
    )
    torch.manual_seed(SEED_K)
    k_fp16 = (
        DATA_OFFSET_K
        + (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_K
    )
    torch.manual_seed(SEED_V)
    v_fp16 = (
        DATA_OFFSET_V
        + (torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1) * DATA_RANGE_V
    )

    q_scale = _get_qk_block_scale(q_fp16.float())
    k_scale = _get_qk_block_scale(k_fp16.float())
    v_scale = _get_v_block_scale(v_fp16.float())

    q_fp8 = _quantize_qk(q_fp16.float(), q_scale)
    k_fp8 = _quantize_qk(k_fp16.float(), k_scale)
    v_fp8 = _quantize_v(v_fp16.float(), v_scale)

    # descale: fp32 -> e8m0 -> 5D packing（尾轴 2 重复）
    deq_q = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(q_scale, "q_scale")))
    deq_k = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(k_scale, "k_scale")))
    deq_v = _pack_scale_5d(fp32_to_e8m0fnu(sanitize_e8m0_scale(v_scale, "v_scale")))

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    logger.info(
        "[DATA] q=%s k=%s v=%s",
        tuple(q_fp8.shape),
        tuple(k_fp8.shape),
        tuple(v_fp8.shape),
    )
    logger.info(
        "[DATA] q_descale=%s k_descale=%s v_descale=%s",
        tuple(deq_q.shape),
        tuple(deq_k.shape),
        tuple(deq_v.shape),
    )

    # 返回 (q, k, v, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table)
    return q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, None, None, None


# ==============================================================================
# CPU Golden（NPU fp16 softmax 管线仿真，反量化按 5D descale）
# ==============================================================================
def cpu_mxfp8_golden(
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
    if actual_seq_q is None:
        actual_seq_q = _get_seqused_q()
    if actual_seq_kv is None:
        actual_seq_kv = _get_seqused_kv()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    # softmaxScale 显式折算 fp32 标量（kernel constInfo_.scaleValue 的 fp32 通路，
    # 同一 double 值的同一次 double→float 舍入）
    softmax_scale_f32 = torch.tensor(softmax_scale, dtype=torch.float32)
    # FixpipeMm1 deqScalar 的 (1,8,10) 硬件格式（accompanying_quantization.md 表 2，
    # NPU 架构版本 3510/ascend950）：标量以 1 符号 + 8 指数 + 10 尾数位参与计算——
    # fp32 尾数低 13 位（bit0-12）被硬件丢弃。2026-09-23 经 L0C/mm1Res dump 反推并
    # 1024 样本验证：fp16(v · s_hw) 与 kernel mm1Res 逐位一致（0/1024），
    # 偏差全部来自此标量截断（s=0.0883883→s_hw=0.0883789，相对 -1.07e-4），
    # 乘积本身的 fp16 舍入是标准 RNE。
    softmax_scale_hw = (softmax_scale_f32.view(torch.int32) & -8192).view(torch.float32)

    # 反量化：descale 取尾轴第一个有效值 [..., 0]，e8m0 -> fp32，再扩到逐元素
    q_scale = e8m0_to_fp32(dequant_scale_q[..., 0]).repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[..., : q_fp8.shape[-1]]
    k_scale = e8m0_to_fp32(dequant_scale_k[..., 0]).repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )[..., : k_fp8.shape[-1]]
    v_scale = e8m0_to_fp32(v_descale[..., 0]).repeat_interleave(
        QUANT_GROUP_SIZE, dim=2
    )[:, :, : v_fp8.shape[2], :]

    q_t = q_fp8.float() * q_scale
    k_t = k_fp8.float() * k_scale
    v_t = v_fp8.float() * v_scale

    # GQA 广播
    if N_q != N_kv:
        g = N_q // N_kv
        k_t = k_t.repeat_interleave(g, dim=1)
        v_t = v_t.repeat_interleave(g, dim=1)

    # ===== NPU fp16 softmax 管线仿真 golden（2026-09-23，fp16 tensor 风格，
    #       镜像 assets/flash_attention_cpu_golden.py 的 s_dtype 模式）=====
    # 逐级建模 kernel 的 FP16 Softmax 方案（quant_mode=3，P 量化 e4m3）：
    #   ① mm1Res：fp32 matmul（Mmad fp32 累加仿真，L0C 已证精确）× deqScalar 的
    #      (1,8,10) 硬件格式截断标量（softmax_scale_hw）→ 标准 RNE fp16 舍入，
    #      此后 S/max/k/Sub/Exp 全程 fp16 tensor
    #   ② k 链：k_i = max(accMax, ceil(max(S16)·INV_LN2))——fp16 域 Muls+CEIL，
    #      subLoop(128 KV) 粒度、跨块累积（kernel softmaxMaxUB half 域同构）
    #   ③ P：x = S16 - k·LN2；P16 = exp(x)——fp16 域 Sub+Exp（kernel V1 ③段同构）
    #   ④ 量化：P8 = e4m3(P16)·pscale——pscale 经 e8m0 编码-解码往返（VfCalcPScale→
    #      L1 网格→Mmad e8m0 字面链），在 fp32 域施加（不再重量化）
    #   ⑤ 分子/分母均用 P8（kernel oneFill rowsum 同款）；累积/跨块因子在 fp32
    #      （kernel mm2Res/stage2Out/accRowsum fp32 域同构）
    # fp16 常量镜像 kernel vf_common_def 的 half 常量（同一 double 的 fp16 舍入）：
    #   LN2_half = half(0.6931471806)，INV_LN2_half = half(1.4426950409)
    LN2 = math.log(2.0)
    F16 = torch.float16
    LN2_F16 = torch.tensor(LN2, dtype=F16)
    INV_LN2_F16 = torch.tensor(1.0 / LN2, dtype=F16)
    Q_BLOCK = 128  # V0/V1 subBlock 粒度（各自独立 k 链）
    K_BLOCK = 256  # actSingleLoopS2Size 粒度（每块内 2 个 128-KV subLoop）
    # 外部 pScale（headroom 因子）：fp32 → half（kernel InitInput 同款舍入），
    # 在 exp 之后、fp8 量化之前乘入 P（kernel VfSoftmax 的 Muls(r0,r0,pScale) 同位）。
    # 分子分母同乘自消（attention 输出不变），但 fp8 量化网格因 pScale 偏移而不同；
    # LSE 经 rowsum 自然携带 log(pScale) 因子（kernel VfCalcLse 同构）
    pScale_h = torch.tensor(
        float(p_scale.item()) if p_scale is not None else 1.0, dtype=F16
    )

    b_, n_ = q_t.shape[0], q_t.shape[1]
    sq, skv = q_t.shape[2], k_t.shape[2]
    dv = v_t.shape[-1]
    out = torch.zeros([b_, n_, sq, dv], dtype=torch.float32)
    lse = torch.zeros([b_, n_, sq], dtype=torch.float32)
    # per-batch seqused（2026-09-24 泛化）：每批独立的 S1/S2 上界（≤ 张量满 shape）。
    # 逐 batch 循环：kv 循环上界 = actual_seq_kv[b]、q 循环上界 = actual_seq_q[b]；
    # 行 [sq_b, sq) 本批无效——kernel 不写（DEAL_ZERO 清零为 TODO），golden 置 0，
    # 对比侧（test_runner）对 NPU 输出同步清零后比较。全满用例（seqused==max）与
    # 原全量循环逐位一致
    for b in range(b_):
        sq_b = int(actual_seq_q[b])
        skv_b = int(actual_seq_kv[b])
        for q0 in range(0, sq_b, Q_BLOCK):
            q1 = min(q0 + Q_BLOCK, sq_b)
            qi = q_t[b : b + 1, :, q0:q1, :]
            m_acc = None  # accMax（fp16 整数 k；None = 首块未初始化）
            s_acc = torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
            o_acc = torch.zeros([1, n_, q1 - q0, dv], dtype=torch.float32)
            for kv0 in range(0, skv_b, K_BLOCK):
                kv1 = min(kv0 + K_BLOCK, skv_b)
                # ① mm1Res fp16：fp32 matmul（Mmad fp32 累加仿真，L0C 已证精确）→
                #   乘 deqScalar 的 (1,8,10) 硬件格式截断标量（fp64 域乘避免中间舍入）→
                #   标准 RNE fp16 舍入，保持 fp16 tensor
                s_blk = torch.matmul(
                    qi, k_t[b : b + 1, :, kv0:kv1, :].transpose(-1, -2)
                )
                s_blk = (s_blk.double() * softmax_scale_hw.double()).to(F16)
                m_blk_start = m_acc
                # ② subLoop 粒度 k 链（128 KV 一节，尾节短行只取有效行）
                sub_ps = []
                for s0 in range(kv0, kv1, 128):
                    s1_ = min(s0 + 128, kv1)
                    s_sub = s_blk[:, :, :, s0 - kv0 : s1_ - kv0]  # fp16
                    k_cur = torch.ceil(s_sub.amax(dim=-1, keepdim=True) * INV_LN2_F16)
                    k_i = k_cur if m_acc is None else torch.maximum(m_acc, k_cur)
                    m_acc = k_i
                    # ③ fp16 域 Sub+Exp
                    norm = k_i * LN2_F16  # Muls(k, LN2) fp16
                    x = s_sub - norm  # Sub fp16
                    p16 = torch.exp(x) * pScale_h  # Exp fp16 + Muls(pScale) fp16
                    sub_ps.append((p16, k_i, s0, s1_))
                k_final = m_acc
                # ④⑤ 本块分子/分母（累积 fp32）
                num_blk = torch.zeros([1, n_, q1 - q0, dv], dtype=torch.float32)
                den_blk = torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
                for p16, k_i, s0, s1_ in sub_ps:
                    # pscale e8m0 编码-解码往返（kernel VfCalcPScale→L1 网格→Mmad e8m0 全链字面仿真）：
                    # byte = SAT(Δk+127)（Maxs 负钳 0；Δk ≤ 0 恒 ≤ 127，上界不可达），
                    # 解码 = e8m0_to_fp32 = 2^(byte-127)——与文件头输入 descale 同款解码路径
                    ps_u8 = torch.clamp(
                        k_i.float() - k_final.float() + 127.0, min=0.0, max=255.0
                    ).to(torch.uint8)
                    ps = e8m0_to_fp32(ps_u8.view(torch.float8_e8m0fnu))
                    # kernel 字面 Cast 链（2026-09-23 对齐）：fp16 --h2iCast--> fp32（精确展宽）
                    #   --castTraitRint--> e4m3（RINT+SAT 量化）；pscale 随后在 Mmad e8m0 域施加
                    #   （fp32 乘、不再重量化）。subnormal 区与"先乘 ps 再量化"不等价——
                    #   kernel 量化时带 subLoop headroom（网格细一档），实测对 PctRlt 无影响但逐值更真。
                    p8 = p16.float().to(FP8_DTYPE).to(torch.float32) * ps
                    num_blk = num_blk + torch.matmul(p8, v_t[b : b + 1, :, s0:s1_, :])
                    den_blk = den_blk + p8.sum(dim=-1, keepdim=True)
                # 跨块累积（factor = 2^(K_blkStart - K_final)，kernel VfCalcRescaleFactor 同值；
                # 同款负指数饱和：ΔK < -127 时钳到 2^-127 而非下溢 0）
                factor = (
                    torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
                    if m_blk_start is None
                    else torch.exp2(
                        torch.clamp(m_blk_start.float() - k_final.float(), min=-127.0)
                    )
                )
                o_acc = o_acc * factor + num_blk
                s_acc = s_acc * factor + den_blk
            # 分母零保护（assets/flash_attention_cpu_golden.py 的 l_safe 同款）：
            # l ≤ 0 的退化行（全零 P）除以 1，避免 0/0 与 inf 污染
            s_safe = torch.where(s_acc > 0, s_acc, torch.ones_like(s_acc))
            out[b : b + 1, :, q0:q1, :] = o_acc / s_safe
            # 空批保护（2026-09-27 seq_used=0 泛化）：seqused_kv[b]=0 时 KV 循环不执行、
            # m_acc 恒 None——softmax 空集：输出已由分母保护置 0，lse = log(0) = -inf
            # （原实现 None.float() 直接 AttributeError，B/C 组 kv=0 用例全崩于此）
            if m_acc is None:
                lse[b : b + 1, :, q0:q1] = float("-inf")
            else:
                lse[b : b + 1, :, q0:q1] = (
                    m_acc.float() * LN2 + torch.log(s_safe)
                ).squeeze(-1)

    out = out.to(torch.bfloat16)
    logger.info("[CPU] out=%s", tuple(out.shape))
    return out, lse


# ==============================================================================
# NPU 调用
# ==============================================================================
def npu_mxfp8_fa(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q,
    max_seqlen_kv,
    block_table_torch=None,
    qr_bf16=None,
    kr_bf16=None,
):
    if not _ensure_npu_imports():
        raise ImportError(
            "cann_ops_transformer.ops.quant_flash_attn is not available. "
            "Please check that cann_ops_transformer is installed and all .so are compiled."
        )

    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    seqused_q_t = (
        torch.tensor(seqused_q, dtype=torch.int32).npu()
        if seqused_q is not None
        else None
    )
    seqused_kv_t = (
        torch.tensor(seqused_kv, dtype=torch.int32).npu()
        if seqused_kv is not None
        else None
    )

    q_npu = q_fp8.npu()
    k_npu = k_fp8.npu()
    v_npu = v_fp8.npu()
    deq_q_npu = dequant_scale_q.npu()
    deq_k_npu = dequant_scale_k.npu()
    deq_v_npu = v_descale.npu()
    p_scale_npu = p_scale.npu()

    ss = 1.0 / math.sqrt(D) if SOFTMAX_SCALE is None else SOFTMAX_SCALE

    torch.npu.synchronize()

    metadata = quant_flash_attn_metadata(
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        head_dim=D,
        quant_mode=QUANT_MODE,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        mask_mode=SPARSE_MODE,
        layout_q=INPUT_LAYOUT,
        layout_q_descale=Q_SCALE_LAYOUT,
        layout_kv=INPUT_LAYOUT,
        layout_out=INPUT_LAYOUT,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )

    atten_out, lse_out = quant_flash_attn(
        q=q_npu,
        k=k_npu,
        v=v_npu,
        q_descale=deq_q_npu,
        k_descale=deq_k_npu,
        v_descale=deq_v_npu,
        quant_mode=QUANT_MODE,
        block_table=None,
        p_scale=p_scale_npu,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=seqused_q_t,
        seqused_kv=seqused_kv_t,
        attn_mask=None,
        metadata=metadata,
        softmax_scale=ss,
        mask_mode=SPARSE_MODE,
        layout_q=INPUT_LAYOUT,
        layout_q_descale=Q_SCALE_LAYOUT,
        layout_kv=INPUT_LAYOUT,
        layout_out=INPUT_LAYOUT,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        return_softmax_lse=ENABLE_LSE,
    )
    torch.npu.synchronize()
    return atten_out, lse_out


# ==============================================================================
# layout 转换（BNSD 单模板下基本是原样返回）
# ==============================================================================
def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout, cu_seqlens=None):
    if layout == "BNSD":
        return tensor_bnsd
    raise ValueError(f"Unsupported layout: {layout}")


def fill_tnd_padding(tensor_tnd, seq_lens, cu_seqlens, fill_value=float("inf")):
    return tensor_tnd
