#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math

import torch

FP32_FRACTION_BITS = 23  # fp32尾数位数

HIF8_EXP_ZERO_THRESHOLD = -23  # 边界值
HIF8_EXP_DML_MIN = -22  # DML最小指数
HIF8_EXP_DML_MAX = -15  # DML最大指数
HIF8_EXP_D0 = 0  # D0指数值
HIF8_EXP_D1_BOUNDARY = 1  # D1指数值
HIF8_EXP_D2_MIN, HIF8_EXP_D2_MAX = 2, 3  # D2指数范围
HIF8_EXP_D3_MIN, HIF8_EXP_D3_MAX = 4, 7  # D3指数范围
HIF8_EXP_D4_MIN, HIF8_EXP_D4_MAX = 8, 15  # D4指数范围

HIF8_DOT_DML = 0  # DML: Denormal Low, 指数范围 -22 ~ -16, 0位尾数
HIF8_DOT_D0 = 1  # D0: 指数为0，3位尾数（最高精度）
HIF8_DOT_D1 = 2  # D1: 指数为±1，3位尾数
HIF8_DOT_D2 = 4  # D2: 指数为±2 ~ ±3，3位尾数
HIF8_DOT_D3 = 8  # D3: 指数为±4 ~ ±7，2位尾数
HIF8_DOT_D4 = 12  # D4: 指数为±8 ~ ±15，1位尾数（最低精度）
HIF8_DOT_INVALID = -1  # 无效状态

HIF8_FRAC_BITS_DML = 0  # DML档位尾数位数
HIF8_FRAC_BITS_D0 = 3  # D0档位尾数位数
HIF8_FRAC_BITS_D1 = 3  # D1档位尾数位数
HIF8_FRAC_BITS_D2 = 3  # D2档位尾数位数
HIF8_FRAC_BITS_D3 = 2  # D3档位尾数位数
HIF8_FRAC_BITS_D4 = 1  # D4档位尾数位数

HIF8_EXP_BITS_DML = 3  # DML档位指数位数
HIF8_EXP_BITS_D0 = 0  # D0档位指数位数
HIF8_EXP_BITS_D1 = 1  # D1档位指数位数
HIF8_EXP_BITS_D2 = 2  # D2档位指数位数
HIF8_EXP_BITS_D3 = 3  # D3档位指数位数
HIF8_EXP_BITS_D4 = 4  # D4档位指数位数

HIF8_ZERO = 0
HIF8_NAN = 128  # 0b10000000, NaN
HIF8_NEG_INF = 239  # 0b11101111, -inf
HIF8_NEG_MAX = 238  # 0b11101110, 负极大值
HIF8_POS_INF = 111  # 0b01101111, +inf
HIF8_POS_MAX = 110  # 0b01101110, 正极大值

HIF8_SIGN_MASK = 128  # 0b10000000, 符号位掩码
HIF8_DOT_MASK = 120  # 0b01110000, dot值掩码
HIF8_FRAC_MASK_3BIT = 7  # 0b00000111, 3位尾数掩码（D0/D1/D2）
HIF8_FRAC_MASK_2BIT = 3  # 0b00000011, 2位尾数掩码（D3）
HIF8_FRAC_MASK_1BIT = 1  # 0b00000001, 1位尾数掩码（D4）
HIF8_EXP_MASK_DML = 7  # 0b00000111, DML指数掩码（bit0-2）
HIF8_EXP_MASK_D4 = 30  # 0b00011110, D4指数掩码（bit1-4）
HIF8_EXP_MASK_D3 = 28  # 0b00011100, D3指数掩码（bit2-4）
HIF8_EXP_MASK_D2 = 24  # 0b00011000, D2指数掩码（bit3-4）
HIF8_EXP_SIGN_MASK_D1 = 8  # 0b00001000, D1指数掩码（bit3）

HIF8_DOT_BIT_SHIFT = 3  # Dot值在HiF8中的起始位置(bit3)
HIF8_DML_EXP_OFFSET = 23  # DML指数偏移值
HIF8_OVERFLOW_SCALE = 1.25  # 溢出阈值缩放因子
HIF8_MAX_FINITE_VALUE = 32768  # 最大有限值（非饱和模式下的边界值, 2^15

SSR_T14_MASK = 16383  # 0b0011 1111 1111 1111, 14位低位掩码
SSR_F14_OFFSET = 8192  # 0b0010 0000 0000 0000, F14偏移值
SSR_DML_SHIFT = 10  # SSR舍入移位值
SSR_RESERVED_BITS = 14  # SSR舍入保留位数
HYBRID_ROUND_EXP_THRESHOLD = 4  # 混合舍入的指数分界点


def fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        return True, 0
    hif8_value_tmp = fraction32_int >> (FP32_FRACTION_BITS - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1


def fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = SSR_T14_MASK
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        f14_values = (fraction32_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (FP32_FRACTION_BITS - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (FP32_FRACTION_BITS - hif8_bits_num))
        f14_values = f14_t14 >> (FP32_FRACTION_BITS - hif8_bits_num - SSR_RESERVED_BITS)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def get_hif8_fraction_bits_number(exponent):
    if exponent < HIF8_EXP_DML_MIN:
        return HIF8_DOT_INVALID, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if HIF8_EXP_DML_MIN <= exponent < HIF8_EXP_DML_MAX:
        return HIF8_DOT_DML, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if exponent == HIF8_EXP_D0:
        return HIF8_DOT_D0, HIF8_EXP_BITS_D0, HIF8_FRAC_BITS_D0
    if abs(exponent) == HIF8_EXP_D1_BOUNDARY:
        return HIF8_DOT_D1, HIF8_EXP_BITS_D1, HIF8_FRAC_BITS_D1
    if HIF8_EXP_D2_MIN <= abs(exponent) <= HIF8_EXP_D2_MAX:
        return HIF8_DOT_D2, HIF8_EXP_BITS_D2, HIF8_FRAC_BITS_D2
    if HIF8_EXP_D3_MIN <= abs(exponent) <= HIF8_EXP_D3_MAX:
        return HIF8_DOT_D3, HIF8_EXP_BITS_D3, HIF8_FRAC_BITS_D3
    if HIF8_EXP_D4_MIN <= abs(exponent) <= HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_FRAC_BITS_D4
    if exponent > HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_DOT_INVALID


def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    ec = 0
    over_value = HIF8_OVERFLOW_SCALE * pow(2.0, HIF8_EXP_D4_MAX + ec)
    if x < 0.0:
        sign = True
        sign_int_value = HIF8_SIGN_MASK
    if torch.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return HIF8_NEG_INF
            else:
                return HIF8_NEG_MAX
        else:
            if over_mode:
                return HIF8_POS_INF
            else:
                return HIF8_POS_MAX
    if torch.isnan(x):
        if over_mode:
            return HIF8_NAN
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(
        x_abs * pow(2, FP32_FRACTION_BITS) * pow(2, -exponent)
        - pow(2, FP32_FRACTION_BITS)
    )
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = (
        get_hif8_fraction_bits_number(exponent)
    )
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = fp32_ta_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = fp32_ssr_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    else:
        print("unknown round type")
        return 0

    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = (
            get_hif8_fraction_bits_number(exponent)
        )
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < HIF8_EXP_ZERO_THRESHOLD:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= HIF8_EXP_ZERO_THRESHOLD:
            return 0
        else:
            return sign_int_value + exponent + HIF8_DML_EXP_OFFSET
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = (
            sign_int_value
            + dot_int_value
            + sig_exp
            + exponent_int_value
            + hif8_frac_value
        )
    return hif8_int_value


def cvt_hifuint8_to_float32(x, over_mode=True):
    x = int(x)
    if x == HIF8_ZERO:
        return float(0)
    elif x == HIF8_NAN:
        if over_mode:
            return float("nan")
        else:
            return float(0)
    elif x == HIF8_NEG_INF:
        if over_mode:
            return -torch.inf
        else:
            return -HIF8_MAX_FINITE_VALUE
    elif x == HIF8_POS_INF:
        if over_mode:
            return torch.inf
        else:
            return HIF8_MAX_FINITE_VALUE
    else:
        if x >= HIF8_NAN:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & HIF8_DOT_MASK
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= HIF8_DOT_D4:
            exponent = x & HIF8_EXP_MASK_D4
            exponent_int = exponent >> 1
            if exponent_int >= 8:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 8

            fra_int = x & HIF8_FRAC_MASK_1BIT
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= HIF8_DOT_D3:
            exponent = x & HIF8_EXP_MASK_D3
            exponent_int = exponent >> 2
            if exponent_int >= 4:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 4

            fra_int = x & HIF8_FRAC_MASK_2BIT
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= HIF8_DOT_D2:
            exponent = x & HIF8_EXP_MASK_D2
            exponent_int = exponent >> 3
            if exponent_int >= 2:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 2

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= HIF8_DOT_D1:
            exponent = x & HIF8_EXP_SIGN_MASK_D1
            exponent_sign = exponent >> 3
            if exponent_sign >= 1:
                exponent_value = -1
            else:
                exponent_value = 1

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_D0:
            exponent_value = 0
            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_DML:
            m_value = 1
            exponent_value = (x & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET
        else:
            print("error, dot error")
            m_value = 0.0
            exponent_value = 0
        return sign * pow(2.0, exponent_value) * m_value


def trans_hifuint8_tensor_to_float_point(in_tensor):
    shape = in_tensor.shape

    flat = in_tensor.reshape(-1)

    out = torch.tensor(
        [cvt_hifuint8_to_float32(x) for x in flat],
        dtype=torch.float32,
        device=in_tensor.device,
    )

    return out.reshape(shape)


def trans_float_tensor_to_hifuint8_point(in_tensor, round_mode="round", over_mode=True):
    shape = in_tensor.shape

    flat = in_tensor.reshape(-1)

    out = torch.tensor(
        [cvt_float32_to_hifuint8(x, round_mode, over_mode) for x in flat],
        dtype=torch.uint8,
        device=in_tensor.device,
    )

    return out.reshape(shape)


def trans_hifuint8_tensor_to_float(in_tensor, over_mode=True):
    """
    将 HiF8 编码的 uint8 Tensor 批量转换为 float32 Tensor (支持 CPU/GPU 矢量化)
    """
    shape = in_tensor.shape
    x = in_tensor.reshape(-1).to(torch.int32)
    out = torch.zeros_like(x, dtype=torch.float32)

    # 1. 特殊值处理 (Masks)
    mask_zero = x == HIF8_ZERO
    mask_nan = x == HIF8_NAN
    mask_ninf = x == HIF8_NEG_INF
    mask_pinf = x == HIF8_POS_INF

    if over_mode:
        out = torch.where(mask_nan, torch.tensor(float("nan"), device=x.device), out)
        out = torch.where(mask_ninf, torch.tensor(-torch.inf, device=x.device), out)
        out = torch.where(mask_pinf, torch.tensor(torch.inf, device=x.device), out)
    else:
        out = torch.where(mask_nan, 0.0, out)
        out = torch.where(mask_ninf, float(-HIF8_MAX_FINITE_VALUE), out)
        out = torch.where(mask_pinf, float(HIF8_MAX_FINITE_VALUE), out)

    # 正常数值的 Mask (排除特殊值)
    mask_normal = ~(mask_zero | mask_nan | mask_ninf | mask_pinf)
    if not mask_normal.any():
        return out.reshape(shape)

    # 提取正常数值子集进行计算
    x_norm = x[mask_normal]

    # 符号位计算
    sign = torch.where(x_norm >= HIF8_NAN, -1.0, 1.0)

    # 提取 dot 档位
    dot_4_value = (x_norm & HIF8_DOT_MASK) >> 3

    # 初始化指数和尾数乘子
    exponent_value = torch.zeros_like(x_norm, dtype=torch.float32)
    m_value = torch.zeros_like(x_norm, dtype=torch.float32)

    # --- 档位 D4 ---
    m_d4 = dot_4_value >= HIF8_DOT_D4
    if m_d4.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D4) >> 1
        exponent_value = torch.where(
            m_d4,
            torch.where(exp_int >= 8, -exp_int, exp_int + 8).float(),
            exponent_value,
        )
        m_value = torch.where(m_d4, 1.0 + (x_norm & HIF8_FRAC_MASK_1BIT) * 0.5, m_value)

    # --- 档位 D3 ---
    m_d3 = (~m_d4) & (dot_4_value >= HIF8_DOT_D3)
    if m_d3.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D3) >> 2
        exponent_value = torch.where(
            m_d3,
            torch.where(exp_int >= 4, -exp_int, exp_int + 4).float(),
            exponent_value,
        )
        m_value = torch.where(
            m_d3, 1.0 + (x_norm & HIF8_FRAC_MASK_2BIT) * 0.25, m_value
        )

    # --- 档位 D2 ---
    m_d2 = (~(m_d4 | m_d3)) & (dot_4_value >= HIF8_DOT_D2)
    if m_d2.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D2) >> 3
        exponent_value = torch.where(
            m_d2,
            torch.where(exp_int >= 2, -exp_int, exp_int + 2).float(),
            exponent_value,
        )
        m_value = torch.where(
            m_d2, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value
        )

    # --- 档位 D1 ---
    m_d1 = (~(m_d4 | m_d3 | m_d2)) & (dot_4_value >= HIF8_DOT_D1)
    if m_d1.any():
        exp_sign = (x_norm & HIF8_EXP_SIGN_MASK_D1) >> 3
        exponent_value = torch.where(
            m_d1, torch.where(exp_sign >= 1, -1.0, 1.0), exponent_value
        )
        m_value = torch.where(
            m_d1, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value
        )

    # --- 档位 D0 ---
    m_d0 = dot_4_value == HIF8_DOT_D0
    if m_d0.any():
        exponent_value = torch.where(m_d0, 0.0, exponent_value)
        m_value = torch.where(
            m_d0, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value
        )

    # --- 档位 DML ---
    m_dml = dot_4_value == HIF8_DOT_DML
    if m_dml.any():
        exponent_value = torch.where(
            m_dml,
            ((x_norm & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET).float(),
            exponent_value,
        )
        m_value = torch.where(m_dml, 1.0, m_value)

    # 计算正常值结果并写回
    norm_res = sign * torch.pow(2.0, exponent_value) * m_value
    out[mask_normal] = norm_res

    return out.reshape(shape)


def trans_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    """
    通过向量操作，将 float32 Tensor 批量转换为 HiF8 编码的 uint8 Tensor
    """

    shape = in_tensor.shape
    x = in_tensor.reshape(-1).to(torch.float32)

    # 先用int32作为输出类型，避免出现赋值错误
    out = torch.zeros_like(x, dtype=torch.int32)

    # 1. 符号位与绝对值提取
    sign_mask = x < 0.0
    sign_int_value = torch.where(sign_mask, HIF8_SIGN_MASK, 0)
    x_abs = torch.abs(x)

    # 2. 溢出与边界条件判断 (Masks)
    over_value = HIF8_OVERFLOW_SCALE * (2.0**HIF8_EXP_D4_MAX)
    mask_inf_or_over = torch.isinf(x) | (x_abs >= over_value)
    mask_nan = torch.isnan(x)
    mask_zero = x_abs == 0.0

    # 处理特殊边界填值
    if over_mode:
        out = torch.where(
            mask_inf_or_over, torch.where(sign_mask, HIF8_NEG_INF, HIF8_POS_INF), out
        )
        out = torch.where(mask_nan, HIF8_NAN, out)
    else:
        out = torch.where(
            mask_inf_or_over, torch.where(sign_mask, HIF8_NEG_MAX, HIF8_POS_MAX), out
        )
        out = torch.where(mask_nan, 0, out)
    out = torch.where(mask_zero, 0, out)

    # 提取正常数字的 Mask
    mask_normal = ~(mask_inf_or_over | mask_nan | mask_zero)
    if not mask_normal.any():
        return out.reshape(shape).to(torch.uint8)

    x_norm = x_abs[mask_normal]
    sign_norm = sign_int_value[mask_normal]

    # 计算基本指数
    exponent = torch.floor(torch.log2(x_norm)).to(torch.int32)

    # 确定截断模式 (TA / SSR)
    if round_mode == "hybrid":
        cut_bit_is_ta = torch.abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD
    elif round_mode == "round":
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)
    elif round_mode == "storound":
        cut_bit_is_ta = torch.zeros_like(exponent, dtype=torch.bool)
    else:
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)

    # 计算 fraction_int
    fraction_int = (
        x_norm * (2.0**FP32_FRACTION_BITS) * torch.pow(2.0, -exponent.float())
        - (2.0**FP32_FRACTION_BITS)
    ).to(torch.int32)

    # 批量获取档位属性 (根据 exponent 映射)
    abs_exp = torch.abs(exponent)

    dot = torch.full_like(exponent, HIF8_DOT_INVALID)
    exp_bits = torch.zeros_like(exponent)
    frac_bits = torch.zeros_like(exponent)

    # 条件区间映射
    m1 = exponent < HIF8_EXP_DML_MIN
    dot = torch.where(m1, HIF8_DOT_INVALID, dot)
    exp_bits = torch.where(m1, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m1, HIF8_FRAC_BITS_DML, frac_bits)

    m2 = (~m1) & (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX)
    dot = torch.where(m2, HIF8_DOT_DML, dot)
    exp_bits = torch.where(m2, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m2, HIF8_FRAC_BITS_DML, frac_bits)

    m3 = exponent == HIF8_EXP_D0
    dot = torch.where(m3, HIF8_DOT_D0, dot)
    exp_bits = torch.where(m3, HIF8_EXP_BITS_D0, exp_bits)
    frac_bits = torch.where(m3, HIF8_FRAC_BITS_D0, frac_bits)

    m4 = abs_exp == HIF8_EXP_D1_BOUNDARY
    dot = torch.where(m4, HIF8_DOT_D1, dot)
    exp_bits = torch.where(m4, HIF8_EXP_BITS_D1, exp_bits)
    frac_bits = torch.where(m4, HIF8_FRAC_BITS_D1, frac_bits)

    m5 = (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX)
    dot = torch.where(m5, HIF8_DOT_D2, dot)
    exp_bits = torch.where(m5, HIF8_EXP_BITS_D2, exp_bits)
    frac_bits = torch.where(m5, HIF8_FRAC_BITS_D2, frac_bits)

    m6 = (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX)
    dot = torch.where(m6, HIF8_DOT_D3, dot)
    exp_bits = torch.where(m6, HIF8_EXP_BITS_D3, exp_bits)
    frac_bits = torch.where(m6, HIF8_FRAC_BITS_D3, frac_bits)

    m7 = (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX)
    dot = torch.where(m7, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m7, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m7, HIF8_FRAC_BITS_D4, frac_bits)

    m8 = exponent > HIF8_EXP_D4_MAX
    dot = torch.where(m8, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m8, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m8, HIF8_DOT_INVALID, frac_bits)

    # ------------------ TA 舍入分支 ------------------
    carry_ta = torch.zeros_like(exponent, dtype=torch.bool)
    frac_val_ta = torch.zeros_like(exponent)

    m_zero_thresh = exponent == HIF8_EXP_ZERO_THRESHOLD
    carry_ta = torch.where(m_zero_thresh, True, carry_ta)

    m_ta_norm = ~m_zero_thresh
    shift_bits = torch.clamp(FP32_FRACTION_BITS - (frac_bits + 1), min=0)
    hif8_val_tmp = fraction_int >> shift_bits

    pow_frac = torch.pow(2, frac_bits + 1) - 1
    m_carry = m_ta_norm & (hif8_val_tmp == pow_frac)
    carry_ta = torch.where(m_carry, True, carry_ta)

    m_odd = m_ta_norm & (~m_carry) & (hif8_val_tmp != 0) & (hif8_val_tmp % 2 == 1)
    frac_val_ta = torch.where(m_odd, (hif8_val_tmp + 1) >> 1, frac_val_ta)

    m_even = m_ta_norm & (~m_carry) & (hif8_val_tmp != 0) & (hif8_val_tmp % 2 == 0)
    frac_val_ta = torch.where(m_even, hif8_val_tmp >> 1, frac_val_ta)

    # ------------------ SSR 舍入分支 ------------------
    carry_ssr = torch.zeros_like(exponent, dtype=torch.bool)
    frac_val_ssr = torch.zeros_like(exponent)

    f14_v1 = (fraction_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
    t14_v1 = fraction_int & SSR_T14_MASK
    hif8_v1 = torch.zeros_like(fraction_int)

    s_bits = torch.clamp(FP32_FRACTION_BITS - frac_bits, min=0)
    hif8_v2 = fraction_int >> s_bits
    f14_t14 = fraction_int - (hif8_v2 << s_bits)
    s_bits_f14 = torch.clamp(FP32_FRACTION_BITS - frac_bits - SSR_RESERVED_BITS, min=0)
    f14_v2 = f14_t14 >> s_bits_f14
    t14_v2 = f14_t14 & SSR_T14_MASK

    f14_values = torch.where(m_zero_thresh, f14_v1, f14_v2)
    t14_values = torch.where(m_zero_thresh, t14_v1, t14_v2)
    hif8_value = torch.where(m_zero_thresh, hif8_v1, hif8_v2)

    m_ge = f14_values >= t14_values
    pow_frac_ssr = torch.pow(2, frac_bits) - 1
    m_ssr_carry = m_ge & (hif8_value == pow_frac_ssr)
    carry_ssr = torch.where(m_ssr_carry, True, carry_ssr)
    frac_val_ssr = torch.where(m_ge & (~m_ssr_carry), hif8_value + 1, frac_val_ssr)
    frac_val_ssr = torch.where(~m_ge, hif8_value, frac_val_ssr)

    # ------------------ 合并舍入结果 ------------------
    carry_exp_status = torch.where(cut_bit_is_ta, carry_ta, carry_ssr)
    hif8_frac_value = torch.where(cut_bit_is_ta, frac_val_ta, frac_val_ssr)

    exponent = torch.where(carry_exp_status, exponent + 1, exponent)
    abs_exp = torch.abs(exponent)

    dot = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_DOT_INVALID, dot),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_DOT_DML,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status, torch.where(exponent == HIF8_EXP_D0, HIF8_DOT_D0, dot), dot
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_DOT_D1, dot),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_DOT_D2,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_DOT_D3,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_DOT_D4,
            dot,
        ),
        dot,
    )

    frac_bits = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_FRAC_BITS_DML, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_FRAC_BITS_DML,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(exponent == HIF8_EXP_D0, HIF8_FRAC_BITS_D0, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_FRAC_BITS_D1, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_FRAC_BITS_D2,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_FRAC_BITS_D3,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_FRAC_BITS_D4,
            frac_bits,
        ),
        frac_bits,
    )

    exp_bits = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_EXP_BITS_DML, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_EXP_BITS_DML,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(exponent == HIF8_EXP_D0, HIF8_EXP_BITS_D0, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_EXP_BITS_D1, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_EXP_BITS_D2,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_EXP_BITS_D3,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_EXP_BITS_D4,
            exp_bits,
        ),
        exp_bits,
    )

    # ------------------ 组合输出编码 ------------------
    hif8_int_value = torch.zeros_like(exponent)
    sig_exp = torch.where(exponent < 0, 1, 0)

    # 分支 A: dot <= 0
    m_a = dot <= 0
    val_a = torch.where(
        exponent <= HIF8_EXP_ZERO_THRESHOLD,
        0,
        sign_norm + exponent + HIF8_DML_EXP_OFFSET,
    )
    hif8_int_value = torch.where(m_a, val_a, hif8_int_value)

    # 分支 B: dot == 1
    m_b = dot == 1
    val_b = sign_norm + (dot << HIF8_DOT_BIT_SHIFT) + hif8_frac_value
    hif8_int_value = torch.where(m_b, val_b, hif8_int_value)

    # 分支 C: dot > 1
    m_c = dot > 1
    abs_exponent = torch.abs(exponent)
    abs_exponent = abs_exponent - torch.pow(2, exp_bits - 1)
    exponent_int_value = abs_exponent << frac_bits
    sig_exp_shifted = sig_exp << (exp_bits - 1 + frac_bits)
    dot_int_value = dot << HIF8_DOT_BIT_SHIFT
    val_c = (
        sign_norm
        + dot_int_value
        + sig_exp_shifted
        + exponent_int_value
        + hif8_frac_value
    )
    hif8_int_value = torch.where(m_c, val_c, hif8_int_value)

    hif8_int_value = torch.where(exponent < HIF8_EXP_ZERO_THRESHOLD, 0, hif8_int_value)

    out[mask_normal] = hif8_int_value

    return out.reshape(shape).to(torch.uint8)
