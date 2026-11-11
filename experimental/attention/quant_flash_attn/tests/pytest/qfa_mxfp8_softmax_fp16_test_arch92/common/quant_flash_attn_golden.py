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
import os
import re

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


# fp16 exp 表: assets/exp16_patch.txt = 正确舍入参考 + 差异条目(idx:bits, 4170 条),
# 重建出 int16[32768]; index = bits & 0x7FFF。缺表直接报错, 不回退到通用 exp。
_EXP16_LUT_BITS = None


def _exp16_assets():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "assets"


def _exp16_ref_bits(n):
    """正确舍入的 fp16 exp 参考: 前 n 个非正 half(bit pattern 0x8000 起)的 exp, 舍入到 half."""
    import math
    import struct

    out = []
    for i in range(n):
        x = struct.unpack("<e", struct.pack("<H", 0x8000 + i))[0]
        try:
            out.append(struct.unpack("<H", struct.pack("<e", math.exp(x)))[0])
        except OverflowError:
            out.append(0x7C00)
    return out


def _load_exp16_patch():
    """从 assets/exp16_patch.txt 重建整张表.

    格式: N <n> / SPARSE / <idx>:<bits> ...  (当前为 4170 条稀疏差异)
    """
    path = _exp16_assets() / "exp16_patch.txt"
    if not path.exists():
        return False
    try:
        import base64
        import struct

        lines = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")  # 跳过补丁头部的注释行
        ]
        n = int(lines[0].split()[1])  # 头部注释已过滤, lines[0] = "N <n>"
        bits = _exp16_ref_bits(n)
        if lines[1].split()[0] == "DELTA":
            delta = int(lines[1].split()[1])
            mask = base64.b64decode(lines[2].split()[1])
            for i in range(n):
                if (mask[i >> 3] >> (i & 7)) & 1:
                    bits[i] = (bits[i] + delta) & 0xFFFF
        else:  # SPARSE
            for ln in lines[2:]:
                idx, hw = ln.split(":")
                bits[int(idx)] = int(hw, 16)
        # NaN 区(31745..32767)永远不会被 golden 查到, 填规范 NaN 占位
        bits.extend([0x7E00] * (32768 - n))
        raw = struct.pack("<32768H", *bits)
        return torch.frombuffer(bytearray(raw), dtype=torch.int16)
    except Exception:  # 补丁损坏时退回 exp 算子, 不让用例挂死
        return False


def _load_exp16_lut():
    """从 assets/exp16_patch.txt 重建整张 fp16 exp 表(int16 位模式, 32768 项)."""
    global _EXP16_LUT_BITS
    if _EXP16_LUT_BITS is None:
        _EXP16_LUT_BITS = _load_exp16_patch()
    return _EXP16_LUT_BITS


# ---- fp16 exp: 只查表 --------------------------------------------------------
# 硬件 fp16 Exp 不是正确舍入函数(实测 4170/31745 个输入差 1 ulp, 双向), 与 CPU/NPU/aclnn
# 的任何通用 exp 都不同源; 因此 golden 侧必须按 bit pattern 查硬件 dump 表才能位级对齐.
def _exp16(x_half):
    """fp16 exp(位级对齐 kernel 的硬件 Exp): 按 bit pattern 查表."""
    lut = _load_exp16_lut()
    if lut is False:
        raise RuntimeError(
            "exp16 查表资产缺失或损坏: 需要 assets/exp16_patch.txt. "
            "硬件 fp16 Exp 与通用 exp 差 4170/31745, 不能用 torch.exp 替代; "
            "重建方式见文件头注释。"
        )
    bits = x_half.view(torch.int16)
    if bool((bits > 0).any()):  # 正 half 的 bit pattern 也 <= 0x7FFF, 会与负数共用索引
        raise RuntimeError(
            "exp16 查表只覆盖 x <= 0: 出现正数 x(通常是 p_scale > 1 时基值 max - ln(pScale) 抬高所致)。"
            "p_scale > 1 需要扩展正数区间的表, 或改用 p_scale <= 1 的用例。"
        )
    idx = bits.to(torch.int64) & 0x7FFF
    return lut[idx].view(torch.float16)


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

# metadata 文件目录: --golden-mode 里带 metadata 时保存, 带 npu_meta 时读取
METADATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "golden_cache", "metadata"
)


def configure_metadata_dir(directory=None):
    """配置 metadata 文件目录 (默认 common/golden_cache/metadata)。"""
    global METADATA_DIR
    if directory is None:
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "golden_cache", "metadata"
        )
    METADATA_DIR = directory


def _metadata_path(case_name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name or "default")).strip("._")
    return os.path.join(METADATA_DIR, (name or "default") + ".pt")


def save_metadata(case_name, metadata):
    """metadata 模式下把算子生成的 metadata 落盘, 供 npu_meta 模式复用。"""
    path = _metadata_path(case_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"case": str(case_name), "metadata": metadata.detach().cpu()}, path)
    logging.info("[metadata] saved %s (shape=%s)", path, tuple(metadata.shape))
    return path


def load_metadata(case_name):
    """npu_meta 模式: 读取预生成的 metadata (CPU int32)。"""
    path = _metadata_path(case_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"npu_meta mode requires a pre-generated metadata file, not found: {path}; "
            "run once with --golden-mode=gen,metadata on NPU first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else payload
    return metadata.to(dtype=torch.int32)


def _metadata_args(seqused_q_t, seqused_kv_t):
    """quant_flash_attn_metadata 的入参 (与 NPU 主算子调用处保持一致)。"""
    return dict(
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
        max_seqlen_q=MAX_SEQLEN_Q,
        max_seqlen_kv=MAX_SEQLEN_KV,
    )


def _metadata_args_from_config():
    seqused_q_t = (
        torch.tensor(SEQUSED_Q, dtype=torch.int32).npu()
        if SEQUSED_Q is not None
        else None
    )
    seqused_kv_t = (
        torch.tensor(SEQUSED_KV, dtype=torch.int32).npu()
        if SEQUSED_KV is not None
        else None
    )
    return _metadata_args(seqused_q_t, seqused_kv_t)


def generate_metadata(case_name):
    """只调用 metadata 算子并保存 (--golden-mode=gen,metadata 用)。"""
    if not _ensure_npu_imports():
        raise ImportError(
            "cann_ops_transformer.ops.quant_flash_attn_metadata is not available. "
            "Please check that cann_ops_transformer is installed and all .so are compiled."
        )
    from cann_ops_transformer.ops import quant_flash_attn_metadata

    metadata = quant_flash_attn_metadata(**_metadata_args_from_config())
    save_metadata(case_name, metadata)
    return metadata


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
    # FixpipeMm1 deqScalar 的 (1,8,10) 硬件格式（accompanying_quantization.md 表 2）：
    # 标量以 1 符号 + 8 指数 + 10 尾数位参与计算——
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
    #      L1 网格→Mmad e8m0 字面链），在 fp32 域施加（不做重量化）
    #   ⑤ 分子用 P8、分母用未量化 P16·pscale（kernel V1 在线 row_sum 同款）；
    #      累积/跨块因子在 fp32（kernel stage2Out/softmaxSum fp32 域同构）
    # fp16 常量镜像 kernel vf_common_def 的 half 常量（同一 double 的 fp16 舍入）：
    #   LN2_half = half(0.6931471806)，INV_LN2_half = half(1.4426950409)
    LN2 = math.log(2.0)
    F16 = torch.float16
    LN2_F16 = torch.tensor(LN2, dtype=F16)
    INV_LN2_F16 = torch.tensor(1.0 / LN2, dtype=F16)
    Q_BLOCK = 128  # S1 基本块(mBaseSize)
    BASE_BLOCK = 512  # S2 基本块(s2BaseSize): 每块内 2 个 256-KV subLoop
    SUBLOOP = 256  # s2SplitSize: V1 子环粒度
    MM2_KSEG = 128  # C2 每次 Mmad 的 K 段(mm2BaseK)
    # 外部 pScale（headroom 因子）：fp32 → half（kernel InitInput 同款舍入），
    # 在 exp 之后、fp8 量化之前乘入 P（kernel VfSoftmax 的 Muls(r0,r0,pScale) 同位）。
    # 分子分母同乘自消（attention 输出不变），但 fp8 量化网格因 pScale 偏移而不同；
    # LSE 经 rowsum 自然携带 log(pScale) 因子（kernel VfCalcLse 同构）
    pScale_h = torch.tensor(
        float(p_scale.item()) if p_scale is not None else 1.0, dtype=F16
    )

    # ln(pScale): kernel 在 fp32 域算 Ln 后 cast 成 half, 再 Sub(local_max, ln_p_scale)
    pScale_val = float(p_scale.item()) if p_scale is not None else 1.0
    ln_p_scale_f32 = torch.tensor(math.log(pScale_val), dtype=torch.float32)
    ln_p_scale_f16 = torch.tensor(
        math.log(pScale_val), dtype=F16
    )  # half 域(LSE 与 Sub 用)

    b_, n_ = q_t.shape[0], q_t.shape[1]
    sq, skv = q_t.shape[2], k_t.shape[2]
    dv = v_t.shape[-1]
    out = torch.zeros([b_, n_, sq, dv], dtype=torch.float32)
    lse = torch.zeros([b_, n_, sq], dtype=torch.float32)
    # per-batch seqused: 每批的 S1/S2 上界; 行 [sq_b, sq) 本批无效,
    # golden 置 0(kernel 侧 ClearOutput 预清零, 对比前同步清零)。
    for b in range(b_):
        sq_b = int(actual_seq_q[b])
        skv_b = int(actual_seq_kv[b])
        for q0 in range(0, sq_b, Q_BLOCK):
            q1 = min(q0 + Q_BLOCK, sq_b)
            qi = q_t[b : b + 1, :, q0:q1, :]
            m_acc = None  # accMax（fp16 整数 k；None = 首块未初始化）
            s_acc = torch.zeros([1, n_, q1 - q0, 1], dtype=torch.float32)
            o_acc = torch.zeros([1, n_, q1 - q0, dv], dtype=torch.float32)
            for kv0 in range(0, skv_b, BASE_BLOCK):
                kv1 = min(kv0 + BASE_BLOCK, skv_b)
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
                for s0 in range(kv0, kv1, SUBLOOP):
                    s1_ = min(s0 + SUBLOOP, kv1)
                    s_sub = s_blk[:, :, :, s0 - kv0 : s1_ - kv0]  # fp16
                    k_cur = torch.ceil(s_sub.amax(dim=-1, keepdim=True) * INV_LN2_F16)
                    k_i = k_cur if m_acc is None else torch.maximum(m_acc, k_cur)
                    m_acc = k_i
                    # ③ half 域 Sub+Exp（对齐 kernel fp16 VF 的 FusedExpSub: 输入/输出都是 FP16）
                    #    注意: 传给 exp 的 x 必须是 FP16(half - half), 不能先转 fp32, 否则舍入域不同.
                    norm = k_i * LN2_F16  # Muls(k, LN2) fp16
                    # pScale 折入基值: kernel 在 half 域做 Sub(local_max, ln_p_scale)
                    norm_ps = (norm - ln_p_scale_f16).half()  # fp16 - fp16 -> fp16
                    x = (s_sub - norm_ps).half()  # fp16 - fp16 -> fp16
                    p16 = _exp16(x)  # fp16 exp(优先查全表) -> fp16
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
                    # kernel 字面 Cast 链：fp16 --h2iCast--> fp32（精确展宽）
                    #   --castTraitRint--> e4m3（RINT+SAT 量化）；pscale 随后在 Mmad e8m0 域施加
                    #   （fp32 乘、不做重量化）。subnormal 区与"先乘 ps 再量化"不等价——
                    #   kernel 量化时带 subLoop headroom（网格细一档），实测对 PctRlt 无影响但逐值更真。
                    p8_code = p16.float().to(FP8_DTYPE)  # e4m3 码点(未乘 ps)
                    p8 = p8_code.to(torch.float32) * ps
                    for t0 in range(s0, s1_, MM2_KSEG):
                        t1_ = min(t0 + MM2_KSEG, s1_)
                        num_blk = num_blk + torch.matmul(
                            p8[..., t0 - s0 : t1_ - s0], v_t[b : b + 1, :, t0:t1_, :]
                        )
                    # 分母：kernel V1 在线 row_sum 累加的是未量化 P16, 子环间用 **半域 exp** 重缩放
                    # (VF 的 vreg_subloop_update = FusedExpSub(旧max, 新max)), 与分子用的 e8m0 字节
                    # (精确 2^Δk) 不同源, 因此这里要用查表 exp 才能位级对齐。
                    fac_h = _exp16(k_i * LN2_F16 - k_final * LN2_F16)
                    den_blk = den_blk + (p16.float() * fac_h.float()).sum(
                        dim=-1, keepdim=True
                    )
                # 跨块累积因子（与 kernel VF 实现逐位同构）：
                #   kernel: factor = Exp_hw( (k_old * LN2)_fp16 - (k_new * LN2)_fp16 )，全程 half 域，
                #           指数由硬件 fp16 exp 完成（即本 golden 查表的那条链）；
                #   精确 2^Δk（fp32）与 half 链每块差 ~0.03%*|Δk|，跨块累积后 O 有 1e-4 量级偏差。
                if m_blk_start is None:
                    factor_h = torch.zeros([1, n_, q1 - q0, 1], dtype=F16)
                else:
                    factor_h = _exp16((m_blk_start * LN2_F16) - (k_final * LN2_F16))
                factor = factor_h.float()
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
                lse[b : b + 1, :, q0:q1] = float(
                    "inf"
                )  # 与 kernel 的 FLOAT_INF(3e+99, fp32 即 +inf) 同值
            else:
                # LSE 的 max 项必须用 fp16 域的 k*LN2: kernel softmaxMaxBuf_ 存的是
                # half(k * half(LN2))(VF 里 Muls(local_max, k, static_cast<half>(LN2))),
                # 若这里用精确 ln2, 每个 k 会引入 k*(half(ln2)-ln2); k=1 时正好 0.0002122 的系统偏差。
                # pScale 已折入 P16(基值 max - ln(pScale)), 故 s_safe 含 pScale 因子;
                # LSE 需除掉它(kernel 侧 pScaleRec = 1/pScale)。
                lse[b : b + 1, :, q0:q1] = (
                    (m_acc * LN2_F16).float() + torch.log(s_safe) - math.log(pScale_val)
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
    case_name=None,
    metadata_source="metadata",
    save_to_file=False,
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

    if metadata_source == "npu_meta":
        # 仿真环境: metadata 算子(AI_CPU)不可用, 直接读 NPU 上预生成的文件
        metadata = load_metadata(case_name).npu()
    else:
        metadata = quant_flash_attn_metadata(
            **_metadata_args(seqused_q_t, seqused_kv_t)
        )
        if save_to_file and case_name:
            save_metadata(case_name, metadata)

    # Reuse device inputs and metadata for stable kernel-only profiling.
    # The default preserves the single-call functional test behavior.
    repeats = int(os.environ.get("QFA_PROFILE_REPEATS", "1"))
    if repeats < 1:
        raise ValueError("QFA_PROFILE_REPEATS must be positive")
    for _ in range(repeats):
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
