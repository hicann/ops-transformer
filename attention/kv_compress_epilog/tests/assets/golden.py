#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Golden reference for KvCompressEpilog (KV Cache 压缩写入 epilogue).

对 bf16 激活 x（任意 N-D，末维 d；末 64 列为 rope 段、前 d-64 列为 nope 段）做
量化压缩，按 slot_mapping 散写（scatter）到 uint8 cache（4D
[blockNum, blockSize, 1, headDim]，num_slots = blockNum×blockSize）各行的前
kvCacheCol 字节；slot == -1 的 token 跳过；行内 [kvCacheCol, headDim) 与未被
引用的行保持 cache 原值。输出为 cache 的副本（输入不可变）。

quant_mode（int，缺省 1，非 0/1/2 抛 NotImplementedError）决定编码格式（arch22 A2/A3 上 1 承载 INT8）：
  0  nope 逐 64 元素组动态量化为 FP8 E4M3，每组 scale 以 bf16（2B 小端）存储；
     rope 段以 bf16 原始位（128B）原样写入。
  1  同 0，但每组 scale 以 E8M0 指数（1B = f32 偏置指数，bias 127）存储；
     arch22（A2/A3）上 1 承载 INT8 对称动态量化（d=512、quant_group_size=64、
     headDim>=608、608B 定长行，语义见下段 INT8）。
  2  rope 段乘 x_scale 后量化为 hifloat8（64B，NaN→0、clip ±32768 含端点）；
     nope 按 quant_group_size 元素一组量化为 FLOAT4_E2M1（2 nibble 打包 1 字节，
     偶数下标低半字节），每组 scale 以 bf16（2B 小端）存储。
  arch22（A2/A3）INT8 对称动态量化（由 quant_mode=1 承载；仅 d=512、quant_group_size=64、
     headDim>=608）：rope 段 bf16 位型直通 128B（NaN/Inf/-0 位型保留，禁数值
     往返）；nope 448 列按每 64 元素一组（7 组）以 f32 中间值（uint16 位型
     <<16 解释）计算——amax_g=max|x|，scale_g = 1.0（amax==0）否则 amax/127
     （f32 除法，非乘倒数），q = clip(RNE(x/scale), -128, 127) → int8
     （数学真值 RNE：_rint_exact_ratio 整数域精确计算；2026-11-05 由
     f32 除法+rint 修订，f32 商在 .5 边界可能与真值反向）；
     scale f32→bf16 单次舍入写出。行 608B 定长：
     rope 128B | nope 448B(int8) | scale 14B(7×bf16) | pad 18B(写 0)；
     round_scale/x_scale 不生效。

mode0/1 量化链（逐 64 元素组，组大小硬编码、quant_group_size 被忽略）：
  amax = max|grp|（f32）；s = max(amax, 1e-4) × f32(1/448)；
  round_scale 为真值判断（非 0 一律取整）：scale = 2^⌈log2(s)⌉（f32 位运算：
  偏置指数 + 尾数非零进位），否则 scale = s 原值；mode1 的 E8M0 字节一律取
  （可能已取整的）scale 的偏置指数（rs=0 时为 ⌊log2(s)⌋+127，与数据所用 scale
  不自洽——忠实复现 kernel 语义）；q = grp/scale（f32 除法）→ clip ±448 →
  e4m3fn（RNE）→ 字节 0xFF 改写 0x7F；mode0 scale 为 NaN 时写 0x7FFF 特例
  （仅 rs=0+NaN 可达）。

mode2 量化链（逐 quant_group_size 元素组）：
  scale = bf16(amax × f32(1/6))；amax==0 时 inv = bf16(0) 否则 bf16(6/amax)；
  xq = bf16(grp × inv)（f32 积精确、单次舍入）→ f32 clip ±6 → fp4（RNE）；
  amax=+Inf → nibble 整组覆盖 0、scale 保持 bf16(+Inf)=0x7F80；
  amax=NaN → 仅 scale 覆盖 0x7FFF、nibble 保持 fp4(NaN) 转换结果。

kvCacheCol（headDim 须 ≥ kvCacheCol，否则 ValueError）：
  mode0: (d-64) + 128 + ((d-64)//64)×2，向上对齐 32（填充字节写 0）
  mode1: (d-64) + 128 + ((d-64)//64)×1，不对齐（特例）
  mode2: 64 + (d-64)//2 + ((d-64)//quant_group_size)×2，向上对齐 32（写 0）
  mode1(arch22): 608（d=512、quant_group_size=64 专属，不对齐、定长）

精度：x 为 bf16（torch bf16 走 uint16 位视图保位；非 bf16 先 astype(f32) 再
astype(bf16) 数值转换）；mode0/1 全程 f32；mode2 scale/inv 各自 f32 计算后单次
舍入到 bf16。输出 dtype/shape 与输入 cache 相同（uint8）。面向字节级对比
（tolerance binary_equal）。

加载（于 ops-test-kit 仓库目录下执行，插件路径按仓库相对位置调整）：

    python3 -m ttk kernel -i ttk_kernel_kv_compress_epilog_st.csv \
        --plugin tests/assets/golden.py
"""

__spec__ = {
    "kv_compress_epilog": "KvCompressEpilogTestSpec",
}

import os
import warnings

import numpy as np

# 第三方 dtype 句柄懒加载（ml_dtypes 必需；en_dtypes 仅 mode2 需要），保证仅导入
# 本文件运行其它算子 golden 时不强制依赖。
BF16 = None
F8E4M3 = None
FP4E2M1 = None
HIF8 = None
HAS_EN = False
_DTYPES_READY = False

ROPE_SIZE = 64  # rope 段元素数（x 末 64 列）
ROPE_BYTES = 128  # mode0/1 rope 段字节数（64 × bf16）
GROUP = 64  # mode0/1 量化组大小（硬编码，与 quant_group_size 无关）
FP8_E4M3_MAX = np.float32(448.0)  # e4m3fn 最大有限值
COEFF = np.float32(1.0) / np.float32(448.0)  # 与 kernel 的 1.0f/448.0f 一致
EPS = np.float32(1e-4)  # amax 下限
HIF8_MAX = np.float32(32768.0)  # hifloat8 最大幅值
FP4_E2M1_MAX = np.float32(6.0)  # FLOAT4_E2M1 最大可表示幅值
INV6 = np.float32(1.0) / np.float32(6.0)  # 与 kernel 的 1.0f/6.0f 一致

# ---- INT8 对称量化（arch22 由 quant_mode=1 承载）专用常量 ----
INT8_D = 512  # x 尾轴长度（rope 64 + nope 448）
INT8_NOPE = INT8_D - ROPE_SIZE  # 448
INT8_GROUPS = INT8_NOPE // GROUP  # 7
INT8_ROPE_BYTES = ROPE_SIZE * 2  # 128（64 × bf16）
INT8_SCALE_OFF = INT8_ROPE_BYTES + INT8_NOPE  # 576（int8 1B/元素）
INT8_PAD_OFF = INT8_SCALE_OFF + INT8_GROUPS * 2  # 590
INT8_ROW_BYTES = 608  # rope 128B | nope 448B | scale 14B | pad 18B


def _ensure_dtypes():
    """首次调用 golden 时导入 ml_dtypes（必需）与 en_dtypes（mode2 的 fp4/hif8）。"""
    global BF16, F8E4M3, FP4E2M1, HIF8, HAS_EN, _DTYPES_READY
    if _DTYPES_READY:
        return
    try:
        import ml_dtypes
    except ImportError as e:
        raise RuntimeError(
            "ml_dtypes is needed for the kv_compress_epilog golden. "
            "Please install with `pip3 install ml-dtypes`"
        ) from e
    BF16 = ml_dtypes.bfloat16
    F8E4M3 = ml_dtypes.float8_e4m3fn
    try:
        import en_dtypes

        FP4E2M1 = en_dtypes.float4_e2m1
        HIF8 = en_dtypes.hifloat8
        HAS_EN = True
    except ImportError:
        HAS_EN = False
    _DTYPES_READY = True


_PLATFORM_ARCH22 = None


def _is_arch22_platform():
    """当前运行设备是否 arch22（DAV_2201: 910B/910_93）；结果缓存，查询失败按 False。

    quant_mode=1 的平台承载语义：arch22（A2/A3）上为 INT8 对称动态量化（608B
    定长行），其余平台为 FP8 e8m0（对齐下游 MixedQuantSparseFlashMla INT8 CSA
    的 quant_mode=1 用法）。
    """
    global _PLATFORM_ARCH22
    if _PLATFORM_ARCH22 is None:
        try:
            import torch_npu

            name = torch_npu.npu.get_device_name()
        except Exception:
            name = ""
        _PLATFORM_ARCH22 = name.startswith("Ascend910")
    return _PLATFORM_ARCH22


def _to_numpy(t):
    """ndarray/torch 张量统一为连续 numpy；torch bf16 走 uint16 位视图保位转换。"""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(t, torch.Tensor):
        tc = t.detach().cpu().contiguous()
        if tc.dtype == torch.bfloat16:
            return tc.view(torch.uint16).numpy().view(BF16)
        return tc.numpy()
    return np.ascontiguousarray(t)


def _f32_bits(s):
    return np.array([s], dtype=np.float32).view(np.uint32)[0]


def _f32_to_bf16_bits(x_f32):
    """f32 → bf16 位型（RNE 单次舍入，位运算实现，与 spec formula 一致）。

    su + 0x7FFF + lsb 后取高 16 位；对 bf16 可精确表示的值（低 16 位为 0）为
    恒等映射，故 NaN/Inf/-0 位型直通保留。输入输出均为位型数组。
    """
    su = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    sb = su + (np.uint64(0x7FFF) + ((su >> np.uint64(16)) & np.uint64(1)))
    return (sb >> np.uint64(16)).astype(np.uint16)


def _round_scale_pow2(s):
    """复现 kernel roundScale 分支的位运算：scale = 2^⌈log2(s)⌉（s>0 正常数）。

    bits = f32(s).view(u32)；尾数非零 → 偏置指数进位；scale = expo<<23。
    返回 (scale, expo)。s 为 NaN 时返回 (-0.0, 256)（expo<<23 溢出至符号位），
    与 kernel 位运算结果一致。
    """
    bits = _f32_bits(s)
    mant = int(bits & np.uint32(0x7FFFFF))
    expo = int(bits >> np.uint32(23))
    if mant != 0:
        expo += 1
    scale = np.array([expo << 23], dtype=np.uint32).view(np.float32)[0]
    return scale, expo


def _kv_cache_col(d, quant_mode, quant_group_size):
    """每行写出的字节数（与 host tiling DoOpTiling 一致）。"""
    if quant_mode == 1 and _is_arch22_platform():
        return INT8_ROW_BYTES  # d=512、quant_group_size=64 由 core 校验
    quant_col = d - ROPE_SIZE
    if quant_mode == 0:
        concat = quant_col + ROPE_BYTES + (quant_col // GROUP) * 2
        return (concat + 31) // 32 * 32
    if quant_mode == 1:
        return quant_col + ROPE_BYTES + (quant_col // GROUP) * 1
    # mode2: 组大小来自 quant_group_size
    concat = ROPE_SIZE + quant_col // 2 + (quant_col // quant_group_size) * 2
    return (concat + 31) // 32 * 32


def _encode_row(x_bf16_row, x_f32_row, quant_mode, round_scale):
    """mode0/1 行编码：[rope bf16 位 128B][nope fp8][scale][pad 0]。"""
    d = x_f32_row.shape[0]
    quant_col = d - ROPE_SIZE
    scale_col = quant_col // GROUP
    scale_bytes = 2 if quant_mode == 0 else 1
    concat = quant_col + ROPE_BYTES + scale_col * scale_bytes
    if quant_mode == 1:
        kv_col = concat  # mode1 不做 32B 对齐（特例）
    else:
        kv_col = (concat + 31) // 32 * 32
    row = np.zeros(kv_col, dtype=np.uint8)

    # rope 段：bf16 原始位拷贝（NaN/Inf/-0 位型原样保留，小端）
    row[:ROPE_BYTES] = (
        np.ascontiguousarray(x_bf16_row[quant_col:]).view(np.uint16).view(np.uint8)
    )

    for g in range(scale_col):
        grp = x_f32_row[g * GROUP : (g + 1) * GROUP]
        amax = np.max(np.abs(grp))
        s = np.maximum(amax, EPS) * COEFF
        if round_scale:  # 真值判断：非 0 一律走向上取整到 2 的幂
            scale, expo = _round_scale_pow2(s)
        else:
            scale = s
            expo = int(_f32_bits(s) >> np.uint32(23))
        q = grp / scale
        q = np.clip(q, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        data = q.astype(F8E4M3).view(np.uint8).copy()
        data[data == 0xFF] = 0x7F  # e4m3fn -NaN 统一改写 +NaN
        row[ROPE_BYTES + g * GROUP : ROPE_BYTES + (g + 1) * GROUP] = data

        off = ROPE_BYTES + quant_col + g * scale_bytes
        if quant_mode == 1:
            row[off] = np.uint8(expo)  # E8M0：scale 的 f32 偏置指数（bias 127）
        elif np.isnan(scale):
            row[off : off + 2] = (0xFF, 0x7F)  # NaN scale 特例 0x7FFF（仅 rs=0 可达）
        else:
            row[off : off + 2] = (
                np.array([scale], dtype=np.float32)
                .astype(BF16)
                .view(np.uint16)
                .view(np.uint8)
            )
    return row


def _encode_row_mode2(x_bf16_row, x_f32_row, group_size, x_scale):
    """mode2 行编码：[rope hifloat8 64B][nope fp4 打包][scale bf16][pad 0]。"""
    if not HAS_EN:
        raise RuntimeError(
            "kv_compress_epilog mode2 golden needs en_dtypes (float4_e2m1/hifloat8)"
        )
    d = x_f32_row.shape[0]
    quant_col = d - ROPE_SIZE
    n_group = quant_col // group_size
    concat = ROPE_SIZE + quant_col // 2 + n_group * 2
    kv_col = (concat + 31) // 32 * 32
    row = np.zeros(kv_col, dtype=np.uint8)

    # rope 段：y = f32(x_rope) × f32(x_scale)；NaN→0；clip ±32768（含端点）→ hif8
    y = x_f32_row[quant_col:] * np.float32(x_scale)
    y = np.where(np.isnan(y), np.float32(0.0), y)
    y = np.clip(y, -HIF8_MAX, HIF8_MAX)
    row[:ROPE_SIZE] = y.astype(HIF8).view(np.uint8)

    fp4_off = ROPE_SIZE
    scale_off = ROPE_SIZE + quant_col // 2
    for g in range(n_group):
        grp_f32 = x_f32_row[g * group_size : (g + 1) * group_size]
        amax = np.max(np.abs(grp_f32))
        scale_bf16 = np.array([amax * INV6], dtype=np.float32).astype(BF16)
        if amax == 0:
            inv = np.zeros(1, dtype=np.float32).astype(BF16)
        else:
            inv = np.array([np.float32(6.0) / amax], dtype=np.float32).astype(BF16)
        # xq = bf16 单次舍入乘积（bf16×bf16 的 f32 积精确 → 一次舍入 = 正确舍入积）
        xq = (grp_f32 * inv.astype(np.float32)).astype(BF16)
        xq = np.clip(xq.astype(np.float32), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        nib = xq.astype(FP4E2M1).view(np.uint8).reshape(-1) & np.uint8(0x0F)
        if np.isinf(amax) and amax > 0:
            nib[:] = 0  # amax=+Inf：nibble 整组覆盖为 FP4E2M1(0)，scale 保持 bf16(+Inf)
        packed = nib[0::2] | (nib[1::2] << np.uint8(4))  # 偶数下标低半字节
        row[fp4_off + g * (group_size // 2) : fp4_off + (g + 1) * (group_size // 2)] = (
            packed
        )
        off = scale_off + g * 2
        if np.isnan(amax):
            row[off : off + 2] = (0xFF, 0x7F)  # amax=NaN：仅 scale 覆盖 0x7FFF
        else:
            row[off : off + 2] = scale_bf16.view(np.uint16).view(np.uint8)
    return row


def _rint_exact_ratio(x_f32, s_f32):
    """精确数学真值 RNE：round(x/s)（round-half-to-even），整数域计算，无 f32 除法。

    x_f32/s_f32 为有限 f32（s>0，NaN 传播由调用方保护）；把两者拆为整数尾数
    m∈[2^23,2^24) 与指数 e（v=m·2^e），商 = xm/sm·2^(xe-se)。用 int64 位移把
    商化为精确整数除法 + 余数 RNE 比较，结果与数学真值逐位一致（不同于 IEEE
    f32 除法：f32 商在 .5 边界上可能与真值相差 <1ulp 而 RNE 方向相反）。
    量程保护：|商|≥2^9 由 clamp 覆盖；商<2^-24 必为 0。
    """
    xb = np.ascontiguousarray(x_f32, dtype=np.float32).view(np.uint32)
    sb = np.ascontiguousarray(s_f32, dtype=np.float32).view(np.uint32)
    xsign = (xb >> np.uint32(31)) & np.uint32(1)
    axb = xb & np.uint32(0x7FFFFFFF)
    xe = (axb >> np.uint32(23)).astype(np.int64)
    xm = (axb & np.uint32(0x7FFFFF)).astype(np.int64)
    sub_x = xe == 0
    xm = np.where(sub_x, xm, xm | np.int64(0x800000))
    xe = np.where(sub_x, np.int64(-126 - 23), xe - np.int64(127 + 23))
    se = (sb >> np.uint32(23)).astype(np.int64)
    sm = (sb & np.uint32(0x7FFFFF)).astype(np.int64)
    sub_s = se == 0
    sm = np.where(sub_s, sm, sm | np.int64(0x800000))
    se = np.where(sub_s, np.int64(-126 - 23), se - np.int64(127 + 23))
    d = xe - se  # 商 = xm/sm · 2^d
    # 量程裁剪到可精确整数化的区间：d>10 → 商≥512 直接饱和；d<-25 → 商<2^-24 → 0
    huge = d > 10
    tiny = d < -25
    dc = np.clip(d, -24, 10)
    num = xm << (dc + 24)  # ≤ 2^24 << 34 = 2^58，int64 安全
    den = sm << np.int64(24)  # ≤ 2^24 << 24 = 2^48
    q = num // den
    rem = num - q * den
    half = den // np.int64(2)
    up = (rem > half) | ((rem == half) & ((q & np.int64(1)) == np.int64(1)))  # RNE
    q = q + up.astype(np.int64)
    q = np.where(tiny, np.int64(0), q)
    q = np.where(huge, np.int64(1) << np.int64(40), q)  # 饱和由下游 clip 完成
    q = np.where(xsign.astype(bool), -q, q)
    return q


def _encode_row_int8(x_u16_row):
    """int8 行编码：rope 128B 位型直通 | nope 448B int8 | scale 14B | pad 18B(0)。

    x_u16_row 为 uint16 [512] bf16 位型；全程位视图保位（禁数值往返），f32
    中间值 = uint16 位型 <<16 解释为 float32（精确、无舍入）。NaN nope 时
    scale 传播为 NaN，q 的 NaN→int8 转换为契约未定义（仅要求不崩溃）。
    """
    row = np.zeros(INT8_ROW_BYTES, dtype=np.uint8)
    # rope 段：bf16 原始位小端直通（NaN/Inf/-0 位型原样保留）
    row[:INT8_ROPE_BYTES] = np.ascontiguousarray(
        x_u16_row[INT8_NOPE:], dtype="<u2"
    ).view(np.uint8)
    # nope 段：7 组 × 64 元素，f32 中间计算
    x_f32 = (
        (x_u16_row[:INT8_NOPE].astype(np.uint32) << np.uint32(16))
        .view(np.float32)
        .reshape(INT8_GROUPS, GROUP)
    )
    amax = np.max(np.abs(x_f32), axis=1)  # [7] f32，NaN 传播
    # 除法语义（非乘倒数）；amax==0 组 scale=1.0
    scale = np.where(amax == np.float32(0.0), np.float32(1.0), amax / np.float32(127.0))
    scale = np.ascontiguousarray(scale, dtype=np.float32)
    # 量化取整：数学真值 RNE（round(x/scale)，round-half-to-even），整数域
    # 精确计算（见 _rint_exact_ratio）。与 f32 除法+rint 的差异仅在商恰好落在
    # .5 边界的样本：f32 商的舍入可能偏离真值 <1ulp 使 RNE 方向相反；此处以
    # 真值为准（CP 决策：golden 语义由「f32 除法」修订为「数学真值 RNE」）。
    # NaN 输入：整数路径输出饱和值，契约未定义（SA-01 仅断言不崩溃）。
    q = _rint_exact_ratio(x_f32, np.broadcast_to(scale[:, None], x_f32.shape))
    q = np.clip(q, np.int64(-128), np.int64(127)).astype(np.int8)
    row[INT8_ROPE_BYTES:INT8_SCALE_OFF] = q.reshape(-1).view(np.uint8)
    # scale f32→bf16 单次舍入（RNE 位运算），小端写出
    row[INT8_SCALE_OFF:INT8_PAD_OFF] = np.ascontiguousarray(
        _f32_to_bf16_bits(scale), dtype="<u2"
    ).view(np.uint8)
    return row  # pad [590, 608) 保持 0


def _kv_compress_epilog_core_int8(cache, x, slot_mapping, quant_group_size=64):
    """int8 逐字节 golden：arch22 专用 int8 量化，608B 行按 slot 散写。

    slot<0 或 >=num_slots 的 token 均跳过（valid mask 语义）；
    行内 [608, headDim) 与未引用行保持 cache 原值。
    """
    if int(quant_group_size) != GROUP:
        raise ValueError(
            "kv_compress_epilog golden: quant_group_size must be 64 when "
            "quant_mode==1 (arch22), got %d" % int(quant_group_size)
        )
    out = _to_numpy(cache).copy()
    head_dim = out.shape[-1]
    if head_dim < INT8_ROW_BYTES:
        raise ValueError(
            "kv_compress_epilog golden: cache headDim(%d) must be >= 608 when "
            "quant_mode==1 (arch22)" % head_dim
        )
    x_np = _to_numpy(x)
    if x_np.dtype != BF16:
        x_np = x_np.astype(np.float32).astype(BF16)
    x_np = x_np.reshape(-1, x_np.shape[-1])  # [bs, d] bf16
    bs, d = x_np.shape
    if d != INT8_D:
        raise ValueError(
            "kv_compress_epilog golden: x last dim d(%d) must be 512 when "
            "quant_mode==1 (arch22)" % d
        )
    x_u16 = np.ascontiguousarray(x_np).view(np.uint16)  # bf16 位视图保位
    sm = _to_numpy(slot_mapping).reshape(-1).astype(np.int64)
    num_slots = out.size // head_dim
    out2d = out.reshape(-1, head_dim)
    for i in range(bs):
        slot = int(sm[i])
        if slot < 0 or slot >= num_slots:  # -1 及越界 slot 跳过（spec valid mask）
            continue
        out2d[slot, :INT8_ROW_BYTES] = _encode_row_int8(x_u16[i])
    return np.ascontiguousarray(out)


def _kv_compress_epilog_core(
    cache,
    x,
    slot_mapping,
    quant_group_size=64,
    quant_mode=1,
    round_scale=1,
    x_scale=1.0,
):
    quant_mode = int(quant_mode)
    if quant_mode not in (0, 1, 2):
        raise NotImplementedError(
            "kv_compress_epilog golden: quant_mode should be 0/1/2, got %d" % quant_mode
        )
    if quant_mode == 1 and _is_arch22_platform():
        return _kv_compress_epilog_core_int8(
            cache, x, slot_mapping, int(quant_group_size)
        )
    out = _to_numpy(cache).copy()
    head_dim = out.shape[-1]
    out2d = out.reshape(-1, head_dim)

    x_bf16 = _to_numpy(x)
    if x_bf16.dtype != BF16:
        x_bf16 = x_bf16.astype(np.float32).astype(BF16)
    x_bf16 = x_bf16.reshape(-1, x_bf16.shape[-1])  # [bs, d]
    x_f32 = x_bf16.astype(np.float32)  # 精确
    bs, d = x_f32.shape
    sm = _to_numpy(slot_mapping).reshape(-1).astype(np.int64)

    kv_cache_col = _kv_cache_col(d, quant_mode, int(quant_group_size))
    if kv_cache_col > head_dim:
        raise ValueError(
            "kv_compress_epilog golden: cache headDim(%d) must be >= kvCacheCol(%d)"
            % (head_dim, kv_cache_col)
        )

    for i in range(bs):
        slot = int(sm[i])
        if slot == -1:  # 仅 -1 跳过该 token
            continue
        if quant_mode == 2:
            enc = _encode_row_mode2(
                x_bf16[i], x_f32[i], int(quant_group_size), float(x_scale)
            )
        else:
            enc = _encode_row(x_bf16[i], x_f32[i], quant_mode, round_scale)
        out2d[slot, :kv_cache_col] = enc
    return np.ascontiguousarray(out)


# ---- int8 npz 定值注入机制（tests/st/data/<testcase_name>.npz）----
# ST runner --input-dir 的 golden 侧入口：注入目录由 set_input_dir() 或环境变量
# KV_COMPRESS_EPILOG_INPUT_DIR 指定，缺省回退到仓内 tests/st/data；
# 用例名来自 customize_inputs 的 kwargs（testcase_name/case_name 等）或环境变量
# KV_COMPRESS_EPILOG_CASE_NAME。npz 约定三个键：cache(uint8)、x_bits(uint16
# bf16 位型；亦兼容 x 存 f32 数值)、slot_mapping(int)。仅 arch22（quant_mode=1 承载 INT8）用例使用。
_INPUT_DIR_ENV = "KV_COMPRESS_EPILOG_INPUT_DIR"
_CASE_NAME_ENV = "KV_COMPRESS_EPILOG_CASE_NAME"
_DEFAULT_INPUT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        "st",
        "data",
    )
)
_INPUT_DIR_OVERRIDE = None
_CASE_NAME_KEYS = ("testcase_name", "case_name", "test_case", "case", "name")


def set_input_dir(path):
    """设置 npz 注入目录（对应 ST runner 的 --input-dir 参数；None 恢复缺省）。"""
    global _INPUT_DIR_OVERRIDE
    _INPUT_DIR_OVERRIDE = os.path.abspath(path) if path else None


def _resolve_input_dir():
    return _INPUT_DIR_OVERRIDE or os.environ.get(_INPUT_DIR_ENV) or _DEFAULT_INPUT_DIR


def _resolve_case_name(kwargs):
    for key in _CASE_NAME_KEYS:
        value = kwargs.get(key)
        if isinstance(value, str) and value:
            return value
    return os.environ.get(_CASE_NAME_ENV) or None


def _load_case_npz(case_name):
    """按用例名加载注入 npz；文件不存在返回 None（回退确定性 slot 生成）。"""
    if not case_name:
        return None
    path = os.path.join(_resolve_input_dir(), case_name + ".npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    cache_np = np.ascontiguousarray(data["cache"], dtype=np.uint8)
    if "x_bits" in data:
        x_bits = np.ascontiguousarray(data["x_bits"], dtype=np.uint16)
    else:  # 兼容 f32 数值存储：RNE 单次舍入到 bf16 位型
        x_bits = _f32_to_bf16_bits(np.ascontiguousarray(data["x"], dtype=np.float32))
    slot_np = np.ascontiguousarray(data["slot_mapping"], dtype=np.int64)
    return cache_np, x_bits, slot_np


def _inject_case_npz(cache, x, slot_mapping, bundle):
    """用 npz 定值原地覆盖 cache/x/slot_mapping（保持 ttk 传入张量身份不变）。

    shape 必须与 csv 声明一致，不一致即抛错（防注入静默退化，SA-09）。
    """
    cache_np, x_bits, slot_np = bundle
    if tuple(cache_np.shape) != tuple(cache.shape):
        raise ValueError(
            "kv_compress_epilog: injected cache shape %s != expected %s"
            % (tuple(cache_np.shape), tuple(cache.shape))
        )
    if x_bits.size != int(np.prod(tuple(x.shape))):
        raise ValueError(
            "kv_compress_epilog: injected x size %d != expected %d"
            % (x_bits.size, int(np.prod(tuple(x.shape))))
        )
    if slot_np.size != int(np.prod(tuple(slot_mapping.shape))):
        raise ValueError(
            "kv_compress_epilog: injected slot_mapping size %d != expected %d"
            % (slot_np.size, int(np.prod(tuple(slot_mapping.shape))))
        )
    if hasattr(slot_mapping, "copy_"):
        # aclnn 路径：TTK 传 torch 张量并原地消费；bf16 走 uint16 位拷贝保位
        import torch

        cache.copy_(
            torch.from_numpy(cache_np).to(device=cache.device, dtype=cache.dtype)
        )
        x.view(torch.uint16).copy_(
            torch.from_numpy(x_bits.reshape(tuple(x.shape))).to(device=x.device)
        )
        slot_mapping.copy_(
            torch.from_numpy(slot_np.reshape(tuple(slot_mapping.shape))).to(
                device=slot_mapping.device, dtype=slot_mapping.dtype
            )
        )
        return (cache, x, slot_mapping)

    cache[...] = cache_np
    if x.dtype == BF16:
        x_u16 = x.view(np.uint16)  # 同 itemsize 位视图，写穿透回原数组
        x_u16[...] = x_bits.reshape(x_u16.shape)
    else:  # 非 bf16 容器（INT8 路径不出现）：按数值写入
        x[...] = (
            x_bits.reshape(tuple(x.shape)).view(BF16).astype(np.float32).astype(x.dtype)
        )
    slot_mapping[...] = slot_np.reshape(tuple(slot_mapping.shape)).astype(
        slot_mapping.dtype
    )
    return (cache, x, slot_mapping)


class KvCompressEpilogTestSpec:
    """KvCompressEpilog 测试规范（kernel/geir 流程 numpy 入参，aclnn 亦可复用）。

    golden 参数：位置输入 cache, x, slot_mapping（顺序同 def.cpp），关键字属性
    quant_group_size/quant_mode/round_scale/x_scale（缺省同算子声明 64/1/True/1.0）。
    """

    @staticmethod
    def golden(
        cache,
        x,
        slot_mapping,
        quant_group_size=64,
        quant_mode=1,
        round_scale=True,
        x_scale=1.0,
        **kwargs,
    ):
        # kernel/geir 传 snake_case 属性；aclnn 跟随 C 接口命名
        quant_group_size = kwargs.pop("quantGroupSize", quant_group_size)
        quant_mode = kwargs.pop("quantMode", quant_mode)
        round_scale = kwargs.pop("roundScale", round_scale)
        x_scale = kwargs.pop("xScale", x_scale)
        _ensure_dtypes()
        out = _kv_compress_epilog_core(
            cache,
            x,
            slot_mapping,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale=round_scale,
            x_scale=x_scale,
        )
        return [np.ascontiguousarray(out)]

    @staticmethod
    def customize_inputs(cache, x, slot_mapping, **kwargs):
        """slot_mapping 取值约束强（-1 或 [0, num_slots)、不应重复），随机数据无法
        满足：改为确定性生成——均匀散布的唯一 slot 覆盖各 cache block，并按 bs
        规模混入 -1 跳过 token（bs>=2 最后一个、bs>=10 另加中间一个）。

        arch22（quant_mode=1）分支：若 <input_dir>/<用例名>.npz 存在
        （input_dir 见 set_input_dir()/KV_COMPRESS_EPILOG_INPUT_DIR，缺省
        tests/st/data；用例名见 kwargs 或 KV_COMPRESS_EPILOG_CASE_NAME），
        则以 npz 定值原地覆盖 cache/x/slot_mapping 并跳过确定性 slot 生成
        （SA-07/SA-08）；注入文件不存在时回退既有确定性逻辑；mode 0/1/2 行为不变。
        """
        _ensure_dtypes()  # customize_inputs 先于 golden 执行, BF16 等 dtype 句柄需先就绪
        quant_mode = kwargs.get("quant_mode", kwargs.get("quantMode", None))
        if quant_mode is None:
            quant_mode = 1
        if int(quant_mode) == 1 and _is_arch22_platform():
            bundle = _load_case_npz(_resolve_case_name(kwargs))
            if bundle is not None:
                return _inject_case_npz(cache, x, slot_mapping, bundle)

        bs = int(np.prod(tuple(x.shape[:-1])))
        num_slots = int(
            np.prod(tuple(cache.shape[:3]))
        )  # [blockNum, blockSize, 1, headDim]
        if bs > num_slots:
            raise ValueError(
                "kv_compress_epilog: bs (%d) must not exceed numSlots (%d)"
                % (bs, num_slots)
            )
        if hasattr(slot_mapping, "copy_"):
            # aclnn 路径：TTK 传 torch 张量并原地消费
            import torch

            slots = torch.arange(bs, dtype=torch.int64, device=slot_mapping.device)
            slots = (slots * num_slots // bs).to(slot_mapping.dtype)
            if bs >= 2:
                slots[-1] = -1
            if bs >= 10:
                slots[bs // 2] = -1
            slot_mapping.copy_(slots.reshape(slot_mapping.shape))
            return (cache, x, slot_mapping)

        slots = np.arange(bs, dtype=np.int64) * num_slots // bs
        if bs >= 2:
            slots[-1] = -1
        if bs >= 10:
            slots[bs // 2] = -1
        slot_mapping[...] = slots.reshape(slot_mapping.shape).astype(slot_mapping.dtype)
        return (cache, x, slot_mapping)

    tolerance = {
        # 验收口径变更: int8 量化输出 <= 1 LSB (ATK kvce_int8_lsb1 同义)。
        # ttk quant 标准: 整型输出逐元素 |diff| <= 1 (round(x/scale) 为阶跃函数,
        # 商跨 x.5 边界时不同实现的 1ulp 差异必然翻转一格, 见 ttk comparison/quant.py)。
        "uint8": {"standard": "quant"},
    }


def int8_selfcheck():
    """SA-09 注入生效负向校验 + int8 golden 位级属性自检（docs/TEST.md §6.2）。

    1) 断言 golden int8 输出的关键位级属性（rope 位型直通 / RNE 平局 / pad=0 /
       scale=bf16(1.0) / amax=0→scale=1.0 / skip-all equals_input）；
    2) 对期望输出做四类错注入变体——pad 非零、scale 少一组、rope 数值往返
       （NaN payload 规范化）、slot 错值——bitwise 比对必须检出差异，否则说明
       注入静默退化或比对无牙（SA-09 门槛防失真）；
    3) 校验 npz 注入链路真实生效（注入后 golden 输出与直接调用一致，且注入
       错值 npz 时输出必须变化）。
    返回 True；任一断言失败抛 AssertionError。用法：python3 golden.py
    """
    _ensure_dtypes()
    checks = []

    def _check(name, cond):
        checks.append((name, bool(cond)))

    # cache (1,4,1,608) 预填 0xFF；x (2,512)
    global _PLATFORM_ARCH22
    _PLATFORM_ARCH22 = True  # 无 NPU 环境自检：强制 arch22 分支（INT8 语义）
    cache = np.full((1, 4, 1, INT8_ROW_BYTES), 0xFF, dtype=np.uint8)
    x_bits = np.zeros((2, INT8_D), dtype=np.uint16)
    # token0 nope 组0 = [127.0, 2.5, 3.5, -2.5, -3.5, 0.5, 1.5, 0...]
    # amax=127 → scale=1.0 精确；其余组全 1.0
    tie = [127.0, 2.5, 3.5, -2.5, -3.5, 0.5, 1.5]
    x_bits[0, : len(tie)] = _f32_to_bf16_bits(np.array(tie, dtype=np.float32))
    x_bits[0, GROUP:INT8_NOPE] = _f32_to_bf16_bits(
        np.full(INT8_NOPE - GROUP, 1.0, dtype=np.float32)
    )
    # token0 rope：-0.0 / +Inf / -Inf / NaN(payload 0x7FC1) / 1.0 / 0...
    x_bits[0, INT8_NOPE:] = np.array(
        [0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x3F80] + [0] * 59, dtype=np.uint16
    )
    # token1：nope 全零（amax=0 → scale=1.0），rope 全 -0.0
    x_bits[1, INT8_NOPE:] = np.uint16(0x8000)
    slot = np.array([0, 2], dtype=np.int64)  # num_slots=4：写行 0、2
    x = x_bits.view(BF16)
    spec = KvCompressEpilogTestSpec
    out = spec.golden(cache, x, slot, quant_mode=1)[0]

    _check(
        "deterministic",
        np.array_equal(out, spec.golden(cache, x, slot, quant_mode=1)[0]),
    )
    rows = out.reshape(-1, INT8_ROW_BYTES)
    row0 = rows[0]
    _check(
        "rope_passthrough",
        np.array_equal(row0[:INT8_ROPE_BYTES].view("<u2"), x_bits[0, INT8_NOPE:]),
    )
    _check("pad18_zero", np.all(row0[INT8_PAD_OFF:INT8_ROW_BYTES] == 0))
    _check(
        "scale_g0_is_bf16_1",
        int(row0[INT8_SCALE_OFF : INT8_SCALE_OFF + 2].view("<u2")[0]) == 0x3F80,
    )
    q0 = row0[INT8_ROPE_BYTES : INT8_ROPE_BYTES + len(tie)].view(np.int8)
    _check("rne_tie_values", list(q0) == [127, 2, 4, -2, -4, 0, 2])
    row2 = rows[2]
    _check(
        "amax0_scale1_all",
        np.all(row2[INT8_SCALE_OFF:INT8_PAD_OFF].view("<u2") == 0x3F80),
    )
    _check(
        "amax0_q_zero",
        np.all(row2[INT8_ROPE_BYTES:INT8_SCALE_OFF].view(np.int8) == 0),
    )
    _check("rope_negzero_bits", np.all(row2[:INT8_ROPE_BYTES].view("<u2") == 0x8000))
    _check(
        "unreferenced_rows_keep", np.all(rows[1] == 0xFF) and np.all(rows[3] == 0xFF)
    )
    _check(
        "skip_all_equals_input",
        np.array_equal(
            spec.golden(cache, x, np.array([-1, -1], dtype=np.int64), quant_mode=1)[0],
            cache,
        ),
    )
    for name, kw in (
        ("reject_d_not_512", {"quant_mode": 1}),
        ("reject_group_not_64", {"quant_mode": 1, "quant_group_size": 128}),
        ("reject_headdim_lt_608", {"quant_mode": 1}),
    ):
        try:
            if name == "reject_d_not_512":
                spec.golden(cache, x[:, :256], slot, **kw)
            elif name == "reject_headdim_lt_608":
                spec.golden(cache[..., :600], x, slot, **kw)
            else:
                spec.golden(cache, x, slot, **kw)
            _check(name, False)
        except ValueError:
            _check(name, True)

    # SA-09 负向：四类错注入变体必须被 bitwise 检出
    neg = out.copy()
    neg.reshape(-1, INT8_ROW_BYTES)[0, INT8_PAD_OFF] = 0x01  # pad 非零
    _check("neg_pad_nonzero_detected", not np.array_equal(neg, out))
    neg = out.copy()
    neg.reshape(-1, INT8_ROW_BYTES)[0, INT8_SCALE_OFF + 12 : INT8_PAD_OFF] = 0
    _check(
        "neg_scale_missing_group_detected", not np.array_equal(neg, out)
    )  # 少第 7 组
    neg = out.copy()
    neg.reshape(-1, INT8_ROW_BYTES)[0, 6:8] = np.array([0x7FC0], dtype="<u2").view(
        np.uint8
    )
    _check(
        "neg_rope_roundtrip_detected", not np.array_equal(neg, out)
    )  # NaN payload 规范化
    out_bad = spec.golden(cache, x, np.array([-1, 2], dtype=np.int64), quant_mode=1)[
        0
    ]  # slot 错注入
    _check("neg_slot_injected_detected", not np.array_equal(out_bad, out))

    # npz 注入链路：注入生效 + 错值 npz 输出必须变化
    import shutil
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="kce_npz_")
    try:
        case = "wb_int8_selfcheck_demo"
        np.savez(
            os.path.join(tmpdir, case + ".npz"),
            cache=cache,
            x_bits=x_bits,
            slot_mapping=slot,
        )
        set_input_dir(tmpdir)
        c2, x2, s2 = spec.customize_inputs(
            cache.copy(), x.copy(), slot.copy(), testcase_name=case, quant_mode=1
        )
        _check(
            "npz_inject_applied",
            np.array_equal(np.asarray(s2).reshape(-1).astype(np.int64), slot),
        )
        _check(
            "npz_inject_bitexact",
            np.array_equal(spec.golden(c2, x2, s2, quant_mode=1)[0], out),
        )
        np.savez(  # 错值 npz：slot[0] 由 0 改为 -1
            os.path.join(tmpdir, case + ".npz"),
            cache=cache,
            x_bits=x_bits,
            slot_mapping=np.array([-1, 2], dtype=np.int64),
        )
        c3, x3, s3 = spec.customize_inputs(
            cache.copy(), x.copy(), slot.copy(), testcase_name=case, quant_mode=1
        )
        _check(
            "npz_negative_detected",
            not np.array_equal(spec.golden(c3, x3, s3, quant_mode=1)[0], out),
        )
        # 无注入文件时回退确定性 slot（bs=2 → [0, -1]），与注入值不同
        os.remove(os.path.join(tmpdir, case + ".npz"))
        _, _, s4 = spec.customize_inputs(
            cache.copy(), x.copy(), slot.copy(), testcase_name=case, quant_mode=1
        )
        _check(
            "fallback_deterministic_slot",
            np.array_equal(
                np.asarray(s4).reshape(-1).astype(np.int64), np.array([0, -1])
            ),
        )
    finally:
        set_input_dir(None)
        shutil.rmtree(tmpdir, ignore_errors=True)

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print("int8_selfcheck %-36s %s" % (name, "PASS" if ok else "FAIL"))
    if failed:
        raise AssertionError("int8_selfcheck failed: %s" % ", ".join(failed))
    print("int8_selfcheck all %d checks PASS" % len(checks))
    return True


if __name__ == "__main__":
    int8_selfcheck()
