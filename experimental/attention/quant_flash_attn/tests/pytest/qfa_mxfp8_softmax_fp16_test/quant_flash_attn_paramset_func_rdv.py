#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
功能正确性 func_rdv 用例集（2026-09-27 重建：全维度矩阵，每用例 LSE/非LSE 双版本）

维度矩阵（10 分区，319 基础用例 × 2 = 638）：
  1. 基础冒烟：B1/G1/S1=S2=256 单核基线
  2. S1 边值扫描：分形边界两侧 ±1（32/64/96/128/160/192/224/256/320/384/448/512...），
     覆盖 V0/V1 部分列、actMSizeAlign32 尾、最小 n=32 分形
  3. S2 边值扫描：同上 + 任意值尾块（100/556/832）+ 大值多块（767..2049）
  4. S1×S2 交叉：关键边值 9×11（含 S2=4096 十六块累积链）
  5. GQA/MHA：G=2/3(非2幂)/4/8/16/32 折叠头 + MHA 多 KV 头（Nq=Nkv）
  6. B/bN2 行分布：B=2..32（1 行/核 → 恰满核）+ B48（核内多行）
  7. per-batch 变长（无 0）：step 递增 / mix 两头窄 + 三重变长组合
  8. seq_used=0 空批：q=0/kv=0 × 首尾/交替 × B + 双轴混合 + 非对齐交错
  9. S2=1 边界家族：单 key × GQA/MHA/多批/满核
  10. 交叉组合：S1×G / S2×G / B×G / B×S1 / B×S2（含首批 0 变体）

LSE 语义（每用例 _LSE 孪生）：正常行 k·ln2+ln(rowsum)；padding 行 0（ClearOutput 预清零）；
  kv=0 批有效行 -inf（DEAL_ZERO 写入，golden 同值）；q=0 批全行 0

seqused 语义：per-batch 列表（长度==B），张量 S1/S2 = max(seqused)；
  kernel 保证：DEAL_ZERO 行清零 + needInitOutput 全量预清零（BNSD 变长 padding 判定）
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


def _sq_step(n):
    return [i * 64 + 1 for i in range(n)]  # 递增不定长


def _sq_mix(s, n):
    return [s] * n if n <= 1 else [1] + [s] * (n - 2) + [1]  # 两头窄中间宽


_BASE = []


def _a(name, b, nq, sq_q, sq_kv, nkv=1):
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
            "p_scale": 1.0,
            "enable_lse": False,
        }
    )


# ===== 1. 基础冒烟（bN2 单核：1 行 → 1 核，S1/S2 各一个 base 块）=====
_a("SMOKE_B1", 1, 1, [256], [256])

# ===== 2. S1 边值扫描（KVS256 固定；分形/对齐边界两侧 ±1）=====
_S1_EDGES = [
    1,
    2,
    31,
    32,
    33,
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
    383,
    384,
    385,
    447,
    448,
    511,
    512,
]
for _s1 in _S1_EDGES:
    _a(f"S1_{_s1}", 1, 1, [_s1], [256])

# ===== 3. S2 边值扫描（QS256 固定；含任意值尾块与大值多块）=====
_S2_EDGES = [
    1,
    2,
    31,
    32,
    33,
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
    832,
    767,
    768,
    769,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
]
for _s2 in _S2_EDGES:
    _a(f"S2_{_s2}", 1, 1, [256], [_s2])

# ===== 4. S1×S2 关键边值交叉（含 S2=4096 十六块跨块累积链）=====
_S1X = [1, 32, 64, 128, 192, 256, 320, 384, 512]
_S2X = [1, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
for _s1 in _S1X:
    for _s2 in _S2X:
        _a(f"X_S1_{_s1}_S2_{_s2}", 1, 1, [_s1], [_s2])

# ===== 5. GQA/MHA 头维（G = N_q/N_kv；折叠头 realN2 = n2×G, realG=1）=====
for _g in [2, 3, 4, 8, 16, 32]:
    _a(f"G_{_g}", 1, _g, [256], [256])
_a("MHA_Nq2Nkv2", 1, 2, [256], [256], nkv=2)  # 多 KV 头各持 KV（区别于 GQA 共享）
_a("MHA_Nq8Nkv8", 1, 8, [256], [256], nkv=8)

# ===== 6. B/bN2 行分布（行 = B×N_q，跨 32 核切分；B48 = 核内多行）=====
for _b in [2, 3, 4, 8, 16, 32]:
    _a(f"B_{_b}", _b, 1, _sq(256, _b), _sq(256, _b))
_a("B_48", 48, 1, _sq(256, 48), _sq(256, 48))

# ===== 7. per-batch 变长（无 0 值；tensor S1/S2 = max(seqused)）=====
for _b in [2, 3, 8, 32]:
    _a(f"V_Qstep_B{_b}", _b, 1, _sq_step(_b), _sq(256, _b))
    _a(f"V_Kstep_B{_b}", _b, 1, _sq(256, _b), _sq_step(_b))
    _a(f"V_Qmix_B{_b}", _b, 1, _sq_mix(256, _b), _sq(256, _b))
    _a(f"V_Kmix_B{_b}", _b, 1, _sq(256, _b), _sq_mix(256, _b))
_a("V_B2_256x100_832x65", 2, 1, [256, 100], [832, 65])  # 满宽 × 非对齐尾
_a("V_B3_triple", 3, 1, [192, 444, 1], [100, 257, 832])  # 三重变长组合

# ===== 8. seq_used=0 空批（DEAL_ZERO：q=0 行全清 / kv=0 行清零；LSE 孪生验证 -inf/0 语义）=====
for _b in [2, 3, 8, 32]:
    _a(f"Z_Q0first_B{_b}", _b, 1, _sq_zf(256, _b), _sq(256, _b))
    _a(f"Z_K0first_B{_b}", _b, 1, _sq(256, _b), _sq_zf(256, _b))
    _a(f"Z_Q0last_B{_b}", _b, 1, _sq_zl(256, _b), _sq(256, _b))
    _a(f"Z_K0last_B{_b}", _b, 1, _sq(256, _b), _sq_zl(256, _b))
    _a(f"Z_Q0alt_B{_b}", _b, 1, _sq_alt(256, _b), _sq(256, _b))
    _a(f"Z_K0alt_B{_b}", _b, 1, _sq(256, _b), _sq_alt(256, _b))
_a("ZM_Q0K0_B2", 2, 1, [0, 256], [256, 0])  # 双轴：b0 q=0 / b1 kv=0
_a("ZM_B4_cross", 4, 1, [0, 256, 0, 128], [256, 0, 65, 832])  # 双轴交错 + 非对齐变长

# ===== 9. S2=1 边界家族（单 key：k=align64(1)=64 + P 零填充 + softmax 尾行单行链）=====
_a("K1_G4", 1, 4, [256], [1])
_a("K1_G16", 1, 16, [256], [1])
_a("K1_MHA2", 1, 2, [256], [1], nkv=2)
_a("K1_B2", 2, 1, [256, 256], [1, 1])
_a("K1_B3_mix", 3, 1, [256, 256, 256], [1, 256, 1])
_a("K1_B32", 32, 1, _sq(256, 32), _sq(1, 32))

# ===== 10. 交叉组合（维度两两叠加：变长/空批 × 头维/批维）=====
for _s1 in [1, 64, 128, 256, 512]:
    for _g in [2, 4, 8]:
        _a(f"C_S1_{_s1}_G_{_g}", 1, _g, [_s1], [256])
for _s2 in [1, 64, 256, 1024, 4096]:
    for _g in [2, 4, 8]:
        _a(f"C_S2_{_s2}_G_{_g}", 1, _g, [256], [_s2])
for _b in [2, 4, 8, 16]:
    for _g in [2, 4, 8]:
        _a(f"C_B_{_b}_G_{_g}", _b, _g, _sq(256, _b), _sq(256, _b))
for _b in [2, 4, 8]:
    for _s1 in [1, 64, 256, 512]:
        _a(f"C_B_{_b}_S1_{_s1}", _b, 1, _sq(_s1, _b), _sq(256, _b))
for _b in [2, 4]:
    for _s1 in [64, 256]:
        _a(f"C_B_{_b}_S1_{_s1}_0f", _b, 1, _sq_zf(_s1, _b), _sq(256, _b))
for _b in [2, 4, 8]:
    for _s2 in [1, 64, 256, 2048]:
        _a(f"C_B_{_b}_S2_{_s2}", _b, 1, _sq(256, _b), _sq(_s2, _b))
for _b in [2, 4]:
    for _s2 in [64, 256]:
        _a(f"C_B_{_b}_S2_{_s2}_0f", _b, 1, _sq(256, _b), _sq_zf(_s2, _b))

# ===== LSE 展开：每用例孪生双版本（非LSE + _LSE），成对相邻 =====
CASES = []
for _c in _BASE:
    CASES.append(_c)
    _twin = dict(_c)
    _twin["name"] = _c["name"] + "_LSE"
    _twin["enable_lse"] = True
    CASES.append(_twin)

# ===== 自校验：命名唯一 + seqused 长度 == B + 每对 LSE 孪生成对 =====
_names = [c["name"] for c in CASES]
assert len(_names) == len(set(_names)), "用例重名"
for _c in CASES:
    assert len(_c["seqused_q"]) == _c["B"] == len(_c["seqused_kv"]), _c["name"]
    assert max(_c["seqused_q"]) > 0 and max(_c["seqused_kv"]) > 0, (
        _c["name"] + " 全零批不可构造"
    )
    assert _c["N_q"] % _c["N_kv"] == 0, _c["name"]
