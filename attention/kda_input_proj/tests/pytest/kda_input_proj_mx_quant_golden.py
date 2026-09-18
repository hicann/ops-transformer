# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU golden for the KdaInputProj Stage1 DynamicMxQuant module (bf16 -> mxfp8_e4m3fn).

逐 bit 复刻 kernel（``op_kernel/arch35/kda_input_proj_mx_quant.h``）的 OCP / scaleAlg=0 路径，
而不是 ops-nn python golden 的 log2 语义 —— 两者在 absmax 恰为 2 的幂或极小值时会有差异。

kernel 的位运算流程（每 32 元素一个 MX block）：

1. ``maxExp = max(bf16_bits & 0x7f80)``：取块内最大的指数域（掩掉符号和尾数）。
2. ``maxExp = max(maxExp, 0x0400)``：截断到 e4m3 的 emax=8，保证 sharedExp 非负、
   倒数仍是规格化 bf16。
3. ``sharedExp = maxExp - 0x0400``，``scale_byte = sharedExp >> 7``（即 e8m0 编码，
   值为 ``2**(scale_byte - 127)``）。块内出现 Inf/NaN 时 ``scale_byte = 0xff``。
4. ``recip_bits = 0x7f00 - sharedExp``，按 bf16 解释即精确的 ``1 / scale``。
   ``sharedExp == 0``（全零或极小块）时 recip 置 0；Inf/NaN 块置 bf16 NaN(0x7f81)。
5. ``y = fp8_e4m3(float(bf16(x) * bf16(recip)))``，round-to-nearest-even + 饱和。
"""

import numpy as np
import torch

MX_BLOCK_SIZE = 32
BF16_EXP_MASK = 0x7F80  # bf16 指数域（含掩掉符号/尾数）
BF16_INF_NAN_EXP = 0x7F80  # 指数域全 1 -> Inf/NaN
FP8_E4M3_MAX_EXP = 0x0400  # e4m3 emax=8，编码成 bf16 指数域即 8 << 7
BF16_EXP_BIAS_BITS = 0x7F00  # 127 << 7，bf16 的 1.0
BF16_NAN_CUSTOM = 0x7F81
E8M0_NAN = 0xFF
BF16_SHR_NUM = 7
FP8_E4M3_MAX = 448.0


def _as_bf16_bits(x: torch.Tensor) -> np.ndarray:
    """bf16 张量 -> uint16 位模式。"""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"expect bfloat16 input, got {x.dtype}")
    return x.contiguous().view(torch.uint16).cpu().numpy().astype(np.uint16)


def _bits_to_bf16(bits: np.ndarray) -> torch.Tensor:
    """uint16 位模式 -> bf16 张量。"""
    return torch.from_numpy(np.ascontiguousarray(bits.astype(np.uint16))).view(
        torch.bfloat16
    )


def mx_quant_golden(x: torch.Tensor, block_size: int = MX_BLOCK_SIZE):
    """把 bf16 的 ``x[T, K]`` 量化成 mxfp8_e4m3fn。

    Args:
        x: bfloat16 张量，shape ``[T, K]``，K 必须是 ``block_size`` 的整数倍。
        block_size: MX 量化块大小，kernel 固定为 32。

    Returns:
        (quant_x, x_scale)

        - ``quant_x``: uint8 张量 ``[T, K]``，每字节是一个 float8_e4m3fn 的位模式。
        - ``x_scale``: uint8 张量 ``[T, ceil(num_block/2), 2]``，e8m0 位模式。
          块数为奇数时按 kernel 的 pack-2 约定补一列 0。
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [T, K], got {tuple(x.shape)}")
    t_size, k_size = x.shape
    if k_size % block_size != 0:
        raise ValueError(f"K={k_size} must be a multiple of block_size={block_size}")
    num_block = k_size // block_size

    bits = _as_bf16_bits(x).reshape(t_size, num_block, block_size)

    # 1) 块内最大指数域
    max_exp = (bits & BF16_EXP_MASK).max(axis=-1).astype(np.uint16)
    is_inf_nan = max_exp == BF16_INF_NAN_EXP

    # 2) 截断到 emax，避免 sharedExp 下溢
    max_exp_clamped = np.maximum(max_exp, np.uint16(FP8_E4M3_MAX_EXP)).astype(np.uint16)

    # 3) shared exponent 与 e8m0 scale 字节
    shared_exp = (max_exp_clamped - np.uint16(FP8_E4M3_MAX_EXP)).astype(np.uint16)
    scale_byte = (shared_exp >> BF16_SHR_NUM).astype(np.uint8)
    scale_byte = np.where(is_inf_nan, np.uint8(E8M0_NAN), scale_byte).astype(np.uint8)

    # 4) 倒数 scale（bf16 位模式），零块与 Inf/NaN 块特殊处理
    recip_bits = (np.uint16(BF16_EXP_BIAS_BITS) - shared_exp).astype(np.uint16)
    recip_bits = np.where(is_inf_nan, np.uint16(BF16_NAN_CUSTOM), recip_bits).astype(
        np.uint16
    )
    recip_bits = np.where(shared_exp == 0, np.uint16(0), recip_bits).astype(np.uint16)

    # 5) bf16 乘法 -> float -> fp8_e4m3fn（RNE + 饱和）
    recip = _bits_to_bf16(recip_bits).reshape(t_size, num_block, 1)
    scaled = (x.reshape(t_size, num_block, block_size).float() * recip.float()).to(
        torch.bfloat16
    )
    scaled_f32 = scaled.float()
    # SatMode::SAT：非 NaN 的溢出值钳到 ±448，NaN 保持 NaN
    finite = ~torch.isnan(scaled_f32)
    scaled_f32 = torch.where(
        finite, scaled_f32.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX), scaled_f32
    )
    quant_x = (
        scaled_f32.to(torch.float8_e4m3fn).view(torch.uint8).reshape(t_size, k_size)
    )

    # scale 按 pack-2 约定排布：奇数块补零列
    pad = num_block % 2
    if pad:
        scale_byte = np.concatenate(
            [scale_byte, np.zeros((t_size, 1), dtype=np.uint8)], axis=1
        )
    x_scale = torch.from_numpy(scale_byte.reshape(t_size, -1, 2).copy())

    return quant_x.cpu(), x_scale


def dequant_mx(
    quant_x: torch.Tensor, x_scale: torch.Tensor, block_size: int = MX_BLOCK_SIZE
) -> torch.Tensor:
    """把 ``(quant_x, x_scale)`` 反量化回 float32，用于端到端 matmul 参考。"""
    t_size, k_size = quant_x.shape
    num_block = k_size // block_size
    elems = (
        quant_x.view(torch.float8_e4m3fn).float().reshape(t_size, num_block, block_size)
    )
    scale_byte = x_scale.reshape(t_size, -1)[:, :num_block].to(torch.int32)
    scale = torch.pow(2.0, (scale_byte - 127).float()).reshape(t_size, num_block, 1)
    return (elems * scale).reshape(t_size, k_size)
