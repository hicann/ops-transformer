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
功能正确性 func_rdv 用例集（1005 基础用例 × 2 LSE 孪生 = 2010 条）

维度矩阵（18 分区）：
  1.  基础冒烟
  2.  S1 边值扫描（分形/对齐边界 ±1，含 16 边界与尖尾）
  3.  S2 边值扫描（同上 + 任意尾块 + 大值多块）
  4.  S1×S2 全交叉（18×20，覆盖所有尾块形态）
  5.  GQA/MHA × S2（折叠头 × 多块累积）
  6.  B/bN2 行分布（1 行/核 → 满核/多行）
  7.  per-batch 变长（step/mix/两极）
  8.  seq_used=0 空批（首/尾/交替 × B 与双轴）
  9.  S2=1 单 key 家族
  10. 维度两两交叉（S1×G / S2×G / B×G / B×S1 / B×S2）
  11. pScale(headroom) < 1
  12. G × S1×S2 交叉（尾块 + 折叠头）
  13. S1×S2 细分尾块（±1 边界，G=1）
  14. B × S1 × S2 三维交叉
  15. pScale × S1 尾块
  16. 大 B × 大 S2 / 变长 × 大 S2
  17. 超长序列多块累积（S2=2049/4096，8~16 段累积链）+ 长 S1
  18. 原 func_rdv 用例集交叉点（历史组合，含 S1=384）

LSE 语义（每用例 _LSE 孪生）：正常行 k·ln2+ln(rowsum)；padding 行 0（ClearOutput 预清零）；
  kv=0 批有效行 +inf；q=0 批全行 0
seqused 语义：per-batch 列表（长度==B），张量 S1/S2 = max(seqused)
"""


# ===== seqused 模式助手 =====
def _sq(s, n):
    return [s] * n  # 全等长


def _sq_zf(s, n):
    return [0] + [s] * (n - 1)  # 首 batch=0


def _sq_zl(s, n):
    return [s] * (n - 1) + [0]  # 末 batch=0


def _sq_alt(s, n):
    return [s if i % 2 == 0 else 0 for i in range(n)]  # 交替 0


def _sq_step(n, base=1, step=64):
    return [i * step + base for i in range(n)]  # 递增不定长


def _sq_dec(n, hi=1024, step=64):
    return [max(1, hi - i * step) for i in range(n)]  # 递减不定长


def _sq_mix(s, n):
    return [s] * n if n <= 1 else [1] + [s] * (n - 2) + [1]  # 两头窄中间宽


def _sq_two(s, n):
    return [s if i % 3 else 1 for i in range(n)]  # 每 3 个一条窄行


_BASE = []


def _a(name, b, nq, sq_q, sq_kv, nkv=1, pscale=1.0):
    _BASE.append(
        {
            "name": name,
            "B": b,
            "N_q": nq,
            "N_kv": nkv,
            "D": 128,
            "cu_seqlens_q": None,
            "cu_seqlens_kv": None,
            "seqused_q": sq_q,
            "seqused_kv": sq_kv,
            "enable_pa": False,
            "kv_cache_layout": None,
            "block_size": None,
            "mask_mode": 0,
            "q_scale_layout": "BNSD",
            "p_scale": pscale,
            "enable_lse": False,
        }
    )


# ===== 1. 基础冒烟（bN2 单核：1 行 → 1 核）=====
_a("SMOKE_B1", 1, 1, [256], [256])
_a("SMOKE_B1_S2_1024", 1, 1, [256], [1024])
_a("SMOKE_B1_S1_512", 1, 1, [512], [512])

# ===== 2. S1 边值扫描（KVS 256；16/32/64/128 对齐边界 ±1 + 尖尾）=====
_S1_EDGES = [
    1,
    2,
    3,
    15,
    16,
    17,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    159,
    160,
    161,
    191,
    192,
    193,
    223,
    224,
    225,
    255,
    256,
    257,
    319,
    320,
    321,
    383,
    384,
    385,
    447,
    448,
    449,
    511,
    512,
]  # 44
for _s1 in _S1_EDGES:
    _a(f"S1_{_s1}", 1, 1, [_s1], [256])

# ===== 3. S2 边值扫描（QS 256；含任意尾块与大值多块）=====
_S2_EDGES = [
    1,
    2,
    3,
    15,
    16,
    17,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    95,
    96,
    97,
    100,
    127,
    128,
    129,
    159,
    160,
    191,
    192,
    193,
    223,
    224,
    255,
    256,
    257,
    319,
    320,
    383,
    384,
    385,
    447,
    448,
    511,
    512,
    556,
    767,
    768,
    769,
    832,
    1023,
    1024,
]  # 48
for _s2 in _S2_EDGES:
    _a(f"S2_{_s2}", 1, 1, [256], [_s2])

# ===== 4. S1×S2 全交叉（18×20 = 360）=====
_S1X = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 192, 255, 256, 320, 512]
_S2X = [
    1,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    100,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    1024,
]
for _s1 in _S1X:
    for _s2 in _S2X:
        _a(f"X_S1_{_s1}_S2_{_s2}", 1, 1, [_s1], [_s2])

# ===== 5. GQA/MHA × S2（折叠头 × 多块累积；G 含非 2 幂）=====
for _g in [2, 3, 4, 8, 16, 32]:
    for _s2 in [1, 64, 256, 1024]:
        _a(f"G{_g}_S2_{_s2}", 1, _g, [256], [_s2])
_a("MHA_Nq2Nkv2", 1, 2, [256], [1024], nkv=2)
_a("MHA_Nq8Nkv8", 1, 8, [256], [256], nkv=8)
_a("MHA_Nq4Nkv4_S2_1", 1, 4, [256], [1], nkv=4)

# ===== 6. B/bN2 行分布（行 = B×N_q 跨 32 核；B48 = 核内多行）=====
for _b in [2, 3, 4, 8, 16, 32]:
    _a(f"B_{_b}", _b, 1, _sq(256, _b), _sq(256, _b))
_a("B_48", 48, 1, _sq(256, 48), _sq(256, 48))

# ===== 7. per-batch 变长（无 0）=====
for _b in [2, 3, 8, 32]:
    _a(f"V_Qstep_B{_b}", _b, 1, _sq_step(_b), _sq(256, _b))
    _a(f"V_Kstep_B{_b}", _b, 1, _sq(256, _b), _sq_step(_b))
    _a(f"V_Qmix_B{_b}", _b, 1, _sq_mix(512, _b), _sq(256, _b))
    _a(f"V_Kmix_B{_b}", _b, 1, _sq(256, _b), _sq_mix(1024, _b))
    _a(f"V_Qdec_B{_b}", _b, 1, _sq_dec(_b), _sq(256, _b))
    _a(f"V_Ktwo_B{_b}", _b, 1, _sq(256, _b), _sq_two(1024, _b))
_a("V_B2_256x100_832x65", 2, 1, [256, 100], [832, 65])
_a("V_B3_triple", 3, 1, [192, 444, 1], [100, 257, 832])
_a("V_B4_cross", 4, 1, [1, 511, 128, 320], [1024, 65, 256, 1])

# ===== 8. seq_used=0 空批 =====
for _b in [2, 3, 8, 32]:
    _a(f"Z_Q0first_B{_b}", _b, 1, _sq_zf(256, _b), _sq(256, _b))
    _a(f"Z_K0first_B{_b}", _b, 1, _sq(256, _b), _sq_zf(256, _b))
    _a(f"Z_Q0last_B{_b}", _b, 1, _sq_zl(256, _b), _sq(256, _b))
    _a(f"Z_K0last_B{_b}", _b, 1, _sq(256, _b), _sq_zl(256, _b))
    _a(f"Z_Q0alt_B{_b}", _b, 1, _sq_alt(256, _b), _sq(256, _b))
    _a(f"Z_K0alt_B{_b}", _b, 1, _sq(256, _b), _sq_alt(256, _b))
_a("ZM_Q0K0_B2", 2, 1, [0, 256], [256, 0])
_a("ZM_B4_cross", 4, 1, [0, 256, 0, 128], [256, 0, 65, 832])
_a("ZM_B3_G8", 3, 8, [0, 256, 256], [256, 0, 1])

# ===== 9. S2=1 单 key 家族 =====
_a("K1_G4", 1, 4, [256], [1])
_a("K1_G16", 1, 16, [256], [1])
_a("K1_MHA2", 1, 2, [256], [1], nkv=2)
_a("K1_B2", 2, 1, [256, 256], [1, 1])
_a("K1_B3_mix", 3, 1, [256, 256, 256], [1, 256, 1])
_a("K1_B32", 32, 1, _sq(256, 32), _sq(1, 32))

# ===== 10. 维度两两交叉 =====
for _s1 in [1, 64, 128, 256, 512]:
    for _g in [2, 4, 8]:
        _a(f"C_S1_{_s1}_G_{_g}", 1, _g, [_s1], [256])
for _s2 in [1, 64, 256, 1024]:
    for _g in [2, 4, 8]:
        _a(f"C_S2_{_s2}_G_{_g}", 1, _g, [256], [_s2])
for _b in [2, 4, 8, 16]:
    for _g in [2, 4, 8]:
        _a(f"C_B_{_b}_G_{_g}", _b, _g, _sq(256, _b), _sq(256, _b))
for _b in [2, 4, 8]:
    for _s1 in [1, 64, 256, 512]:
        _a(f"C_B_{_b}_S1_{_s1}", _b, 1, _sq(_s1, _b), _sq(256, _b))
for _b in [2, 4, 8]:
    for _s2 in [1, 64, 256, 1024]:
        _a(f"C_B_{_b}_S2_{_s2}", _b, 1, _sq(256, _b), _sq(_s2, _b))
for _b in [2, 4]:
    for _s1 in [64, 256]:
        _a(f"C_B_{_b}_S1_{_s1}_0f", _b, 1, _sq_zf(_s1, _b), _sq(256, _b))
for _b in [2, 4]:
    for _s2 in [64, 256]:
        _a(f"C_B_{_b}_S2_{_s2}_0f", _b, 1, _sq(256, _b), _sq_zf(_s2, _b))

# ===== 11. pScale(headroom) < 1（只取 <=1：golden exp 表覆盖 x<=0）=====
for _ps in [0.9, 0.75, 0.5, 0.25]:
    for _s2 in [256, 1024]:
        _a(f"PS_{str(_ps).replace('.', '')}_S2_{_s2}", 1, 1, [256], [_s2], pscale=_ps)

# ===== 12. G × S1×S2 交叉（尾块 + 折叠头）=====
for _g in [2, 4]:
    for _s1 in [1, 17, 64, 129, 256, 320, 512, 65]:
        for _s2 in [1, 17, 65, 129, 257, 512, 1024, 33]:
            _a(f"G{_g}_X_{_s1}_{_s2}", 1, _g, [_s1], [_s2])

# ===== 13. S1×S2 细分尾块（±1 边界，G=1）=====
_S1T = [1, 16, 17, 63, 65, 127, 129, 255, 257]
_S2T = [1, 15, 16, 17, 63, 65, 127, 129, 255, 257, 511, 513]
for _s1 in _S1T:
    for _s2 in _S2T:
        _a(f"T_{_s1}_{_s2}", 1, 1, [_s1], [_s2])

# ===== 14. B × S1 × S2 三维交叉 =====
for _b in [2, 4, 8]:
    for _s1 in [1, 129, 512]:
        for _s2 in [1, 257, 1024]:
            _a(f"D_B{_b}_S1_{_s1}_S2_{_s2}", _b, 1, _sq(_s1, _b), _sq(_s2, _b))

# ===== 15. pScale × S1 尾块 =====
for _ps in [0.5, 0.25]:
    for _s1 in [1, 17, 65, 129, 257, 512]:
        _a(f"PS_{str(_ps).replace('.', '')}_S1_{_s1}", 1, 1, [_s1], [256], pscale=_ps)

# ===== 16. 大 B × 大 S2 / 变长 × 大 S2 =====
for _b in [32, 48]:
    for _s2 in [1, 257, 1024]:
        for _g in [1, 8]:
            _a(f"LB{_b}_S2_{_s2}_G{_g}", _b, _g, _sq(256, _b), _sq(_s2, _b))
for _b in [2, 4, 8]:
    _a(f"VKstep_B{_b}_S2max1024", _b, 1, _sq(256, _b), _sq_step(_b, 1, 256))
    _a(f"VQstep_B{_b}_S2_1024", _b, 1, _sq_step(_b, 1, 128), _sq(1024, _b))

# ===== 17. 超长序列多块累积（S2 > 2048，累积链 8~16 段）+ 长 S1 =====
for _s2 in [2049, 4096]:
    for _s1 in [1, 64, 128, 256, 512]:
        _a(f"LONG_S2_{_s2}_S1_{_s1}", 1, 1, [_s1], [_s2])
for _b in [2, 4]:
    for _s2 in [1024, 2049]:
        _a(f"LONG_B{_b}_S2_{_s2}", _b, 1, _sq(256, _b), _sq(_s2, _b))
for _g in [2, 8]:
    for _s2 in [257, 1024]:
        _a(f"LONG_G{_g}_S2_{_s2}", 1, _g, [256], [_s2])
for _s2 in [2049, 4096]:
    _a(f"LONG_PS05_S2_{_s2}", 1, 1, [256], [_s2], pscale=0.5)
for _s1 in [768, 1024]:
    _a(f"LONG_S1_{_s1}_S2_512", 1, 1, [_s1], [512])
for _s1 in [1023, 1025]:
    _a(f"LONG_S1_{_s1}_S2_1024", 1, 1, [_s1], [1024])
_a("LONG_B3_S2_4096", 3, 1, _sq(256, 3), _sq(4096, 3))
_a("LONG_G3_S2_4096", 1, 3, [256], [4096])
_a("LONG_Z_B2_S2_4096", 2, 1, [0, 256], [4096, 0])


# ===== 18. 原 func_rdv 用例集交叉点 =====
_a("ORIG_000", 1, 1, [256], [1025], nkv=1, pscale=1.0)
_a("ORIG_001", 1, 1, [256], [2047], nkv=1, pscale=1.0)
_a("ORIG_002", 1, 1, [256], [2048], nkv=1, pscale=1.0)
_a("ORIG_003", 1, 1, [1], [192], nkv=1, pscale=1.0)
_a("ORIG_004", 1, 1, [1], [384], nkv=1, pscale=1.0)
_a("ORIG_005", 1, 1, [1], [768], nkv=1, pscale=1.0)
_a("ORIG_006", 1, 1, [1], [2048], nkv=1, pscale=1.0)
_a("ORIG_007", 1, 1, [32], [192], nkv=1, pscale=1.0)
_a("ORIG_008", 1, 1, [32], [384], nkv=1, pscale=1.0)
_a("ORIG_009", 1, 1, [32], [768], nkv=1, pscale=1.0)
_a("ORIG_010", 1, 1, [32], [2048], nkv=1, pscale=1.0)
_a("ORIG_011", 1, 1, [32], [4096], nkv=1, pscale=1.0)
_a("ORIG_012", 1, 1, [64], [192], nkv=1, pscale=1.0)
_a("ORIG_013", 1, 1, [64], [384], nkv=1, pscale=1.0)
_a("ORIG_014", 1, 1, [64], [768], nkv=1, pscale=1.0)
_a("ORIG_015", 1, 1, [64], [2048], nkv=1, pscale=1.0)
_a("ORIG_016", 1, 1, [128], [192], nkv=1, pscale=1.0)
_a("ORIG_017", 1, 1, [128], [384], nkv=1, pscale=1.0)
_a("ORIG_018", 1, 1, [128], [768], nkv=1, pscale=1.0)
_a("ORIG_019", 1, 1, [128], [2048], nkv=1, pscale=1.0)
_a("ORIG_020", 1, 1, [192], [192], nkv=1, pscale=1.0)
_a("ORIG_021", 1, 1, [192], [384], nkv=1, pscale=1.0)
_a("ORIG_022", 1, 1, [192], [768], nkv=1, pscale=1.0)
_a("ORIG_023", 1, 1, [192], [2048], nkv=1, pscale=1.0)
_a("ORIG_024", 1, 1, [192], [4096], nkv=1, pscale=1.0)
_a("ORIG_025", 1, 1, [256], [2048], nkv=1, pscale=1.0)
_a("ORIG_026", 1, 1, [320], [192], nkv=1, pscale=1.0)
_a("ORIG_027", 1, 1, [320], [384], nkv=1, pscale=1.0)
_a("ORIG_028", 1, 1, [320], [768], nkv=1, pscale=1.0)
_a("ORIG_029", 1, 1, [320], [2048], nkv=1, pscale=1.0)
_a("ORIG_030", 1, 1, [320], [4096], nkv=1, pscale=1.0)
_a("ORIG_031", 1, 1, [384], [192], nkv=1, pscale=1.0)
_a("ORIG_032", 1, 1, [384], [384], nkv=1, pscale=1.0)
_a("ORIG_033", 1, 1, [384], [768], nkv=1, pscale=1.0)
_a("ORIG_034", 1, 1, [384], [2048], nkv=1, pscale=1.0)
_a("ORIG_035", 1, 1, [384], [4096], nkv=1, pscale=1.0)
_a("ORIG_036", 1, 1, [512], [192], nkv=1, pscale=1.0)
_a("ORIG_037", 1, 1, [512], [384], nkv=1, pscale=1.0)
_a("ORIG_038", 1, 1, [512], [768], nkv=1, pscale=1.0)
_a("ORIG_039", 1, 1, [512], [2048], nkv=1, pscale=1.0)
_a("ORIG_040", 1, 2, [256], [256], nkv=2, pscale=1.0)
_a("ORIG_041", 3, 1, [1, 256, 1], [256, 256, 256], nkv=1, pscale=1.0)
_a(
    "ORIG_042",
    8,
    1,
    [1, 256, 256, 256, 256, 256, 256, 1],
    [256, 256, 256, 256, 256, 256, 256, 256],
    nkv=1,
    pscale=1.0,
)
_a(
    "ORIG_043",
    8,
    1,
    [256, 256, 256, 256, 256, 256, 256, 256],
    [1, 256, 256, 256, 256, 256, 256, 1],
    nkv=1,
    pscale=1.0,
)
_a(
    "ORIG_044",
    32,
    1,
    [
        1,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        1,
    ],
    [
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
    ],
    nkv=1,
    pscale=1.0,
)
_a(
    "ORIG_045",
    32,
    1,
    [
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
    ],
    [
        1,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        1,
    ],
    nkv=1,
    pscale=1.0,
)
_a("ORIG_046", 1, 2, [256], [4096], nkv=1, pscale=1.0)
_a("ORIG_047", 1, 4, [256], [4096], nkv=1, pscale=1.0)
_a("ORIG_048", 1, 8, [256], [4096], nkv=1, pscale=1.0)
_a("ORIG_049", 2, 1, [256, 256], [2048, 2048], nkv=1, pscale=1.0)
_a("ORIG_050", 4, 1, [256, 256, 256, 256], [2048, 2048, 2048, 2048], nkv=1, pscale=1.0)
_a(
    "ORIG_051",
    8,
    1,
    [256, 256, 256, 256, 256, 256, 256, 256],
    [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
    nkv=1,
    pscale=1.0,
)


# ===== 18b. S1=384 交叉点（任意 S2）=====
_a("X384_S2_1", 1, 1, [384], [1])
_a("X384_S2_64", 1, 1, [384], [64])
_a("X384_S2_128", 1, 1, [384], [128])
_a("X384_S2_512", 1, 1, [384], [512])
_a("X384_S2_1024", 1, 1, [384], [1024])

# ===== LSE 展开：每用例孪生双版本（非LSE + _LSE），成对相邻 =====
CASES = []
for _c in _BASE:
    CASES.append(_c)
    _twin = dict(_c)
    _twin["name"] = _c["name"] + "_LSE"
    _twin["enable_lse"] = True
    CASES.append(_twin)

# ===== 自校验 =====
_names = [c["name"] for c in CASES]
assert len(_names) == len(set(_names)), "用例重名"
for _c in CASES:
    assert len(_c["seqused_q"]) == _c["B"] == len(_c["seqused_kv"]), _c["name"]
    assert max(_c["seqused_q"]) > 0 and max(_c["seqused_kv"]) > 0, (
        _c["name"] + " 全零批不可构造"
    )
    assert _c["N_q"] % _c["N_kv"] == 0, _c["name"]
assert 1900 <= len(CASES) <= 2100, f"用例总数 {len(CASES)} 不在目标区间"
