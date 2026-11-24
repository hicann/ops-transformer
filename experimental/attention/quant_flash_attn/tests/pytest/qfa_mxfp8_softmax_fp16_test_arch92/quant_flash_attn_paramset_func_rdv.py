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
功能正确性 func_rdv 用例集：316 个形状 × {BNSD, TND} × {非LSE, _LSE 孪生} = 1276 条（每种 layout 638 条）

形状维度按基本块取值（m=128, n=512, D=128, VSCALE 组=64）：
  S1 取 128 的 ±1 边界与倍数；S2 取 512 的 ±1 边界与倍数；并单独覆盖 64 组的尾组；
  GQA 覆盖非 2 幂的 G；B 覆盖单行/满核/多行；变长覆盖 step/dec/mix/zf/zl/alt；
  另含空批(seqused=0)、S2=1、pScale<1、超长多段累积链。

两种 layout 共用同一形状表，保证 BNSD 与 TND 的覆盖度逐条对齐；
TND 由 golden 用 seqused 累加出 cu_seqlens（算子要求该入参非空）。

mask_mode：causal(3) 只出现在 seqused_kv[i] >= seqused_q[i] 的形状上，其余用 0。 另含 PA_BNBD(分页 KV, block_size=512) 3 个形状 x 2 布局 x LSE 孪生 = 12 条。
"""


# ===== seqused 模式助手 =====


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


# ===== 形状表 =====
_SHAPES = []


def _s(name, b, nq, sq_q, sq_kv, nkv=1, pscale=1.0, mask=3):
    _SHAPES.append((name, b, nq, sq_q, sq_kv, nkv, pscale, mask))


def _u(name, b, nq, s1, s2, nkv=1, pscale=1.0, mask=3):
    _s(name, b, nq, [s1] * b, [s2] * b, nkv, pscale, mask)


# ---- 1. 基础冒烟（S1/S2 各取一个块与其 1/4）----
_u("SMOKE_S1_128_S2_128", 1, 1, 128, 128)
_u("SMOKE_S1_128_S2_512", 1, 1, 128, 512)
_u("SMOKE_S1_512_S2_512", 1, 1, 512, 512)
_u("SMOKE_S1_1_S2_1", 1, 1, 1, 1, mask=0)

# ---- 2. S1 边值（m=128 的 ±1 与倍数；S2 取 512 或按需放大以保持因果右对齐）----
for _s1 in (1, 64, 127, 128, 129, 255, 256, 257, 384, 512, 513):
    _s2 = 512 if _s1 <= 512 else 2048
    _u(f"S1E_S1_{_s1}_S2_{_s2}", 1, 1, _s1, _s2)

# ---- 3. S1 边值（S2=1024，长 K 下的 m 分块）----
for _s1 in (1, 127, 128, 129, 256, 512, 1024):
    _u(f"S1E_S1_{_s1}_S2_1024", 1, 1, _s1, 1024)

# ---- 4. S2 边值（S1=256，n=512 的 ±1 与倍数）----
for _s2 in (256, 257, 511, 512, 513, 640, 768, 1023, 1024, 1025):
    _u(f"S2E_S1_256_S2_{_s2}", 1, 1, 256, _s2)

# ---- 5. S2 边值（S1=128，单 m 块下的多 n 块累积）----
for _s2 in (128, 129, 255, 256, 511, 512, 1024):
    _u(f"S2E_S1_128_S2_{_s2}", 1, 1, 128, _s2)

# ---- 6. S1×S2 交叉（mask=0，含 QS>S2 的非因果形态）----
_CROSS_S1 = (1, 127, 128, 129, 256, 384, 512, 1024)
_CROSS_S2 = (1, 63, 64, 65, 255, 256, 511, 512, 513, 1024)
_CROSS_PAIRS = (
    (0, 0),
    (0, 4),
    (0, 8),
    (1, 3),
    (1, 7),
    (2, 0),
    (2, 2),
    (2, 6),
    (3, 1),
    (3, 5),
    (4, 4),
    (4, 8),
    (5, 3),
    (6, 6),
    (7, 1),
    (7, 9),
)
for _i, _j in _CROSS_PAIRS:
    _s1, _s2 = _CROSS_S1[_i], _CROSS_S2[_j]
    _u(f"X_S1_{_s1}_S2_{_s2}", 1, 1, _s1, _s2, mask=0)

# ---- 7. GQA/MHA（折叠头 × 多块；G 含非 2 幂）----
_GQA = ((2, 1), (4, 1), (8, 1), (16, 1), (4, 2), (8, 2), (3, 1), (6, 2), (2, 2), (8, 8))
for _k, (_nq, _nkv) in enumerate(_GQA):
    for _s2 in (512 if _k % 2 == 0 else 1024,):
        _u(
            f"GQA_Nq{_nq}_Nkv{_nkv}_S2_{_s2}",
            1,
            _nq,
            128,
            _s2,
            nkv=_nkv,
            mask=0 if _k % 2 else 3,
        )

# ---- 8. B/行分布（单行→满核→核内多行）----
for _b in (1, 3, 8, 32, 48):
    _u(f"B_{_b}_S1_128_S2_512", _b, 1, 128, 512)

# ---- 9. per-batch 变长 ----
_VARLEN = (
    ("STEP_B4", 4, _sq_step(4)),
    ("DEC_B4", 4, _sq_dec(4)),
    ("MIX_B8", 8, _sq_mix(1024, 8)),
)
for _name, _b, _lens in _VARLEN:
    _s(f"VL_{_name}", _b, 1, _lens, [max(2 * x, x) for x in _lens], mask=3)
    _s(f"VL_{_name}_M0", _b, 1, _lens, _lens, mask=0)

# ---- 10. seqused=0 空批 ----
for _b in (4, 8):
    _s(f"EMPTY_ZF_B{_b}", _b, 1, _sq_zf(512, _b), _sq_zf(512, _b), mask=0)
    if _b == 4:
        _s(f"EMPTY_ZL_B{_b}", _b, 1, _sq_zl(512, _b), _sq_zl(512, _b), mask=0)
        _s(f"EMPTY_ALT_B{_b}", _b, 1, _sq_alt(512, _b), _sq_alt(512, _b), mask=0)

# ---- 11. S2=1 单 key 家族 ----
for _s1 in (1, 128, 256, 512):
    _u(f"S2_ONE_S1_{_s1}", 1, 1, _s1, 1, mask=0)

# ---- 12. pScale(headroom) < 1 ----
for _ps in (0.5, 0.25, 0.0625):
    _u(f"PS_{_ps}_S1_128_S2_512".replace(".", ""), 1, 1, 128, 512, pscale=_ps)
_u("PS_05_S1_129", 1, 1, 129, 512, pscale=0.5)

# ---- 13. 超长多段累积链 ----
for _s1, _s2 in ((128, 2048), (128, 2049), (512, 4096), (128, 8192)):
    _u(f"LONG_S1_{_s1}_S2_{_s2}", 1, 1, _s1, _s2, mask=0)

# ---- 14. 64 组尾组（VSCALE 分组边界；S2 不足 64 时 S1 取 1 以保持因果右对齐）----
for _s2 in (63, 64, 65, 127, 128, 700, 1023):
    _u(f"VSCALE_S2_{_s2}", 1, 1, 64 if _s2 >= 64 else 1, _s2)

# ---- 15. 极端长 S1 ----
for _s1, _s2 in ((2048, 4096), (1024, 1025)):
    _u(f"LONGS1_{_s1}_{_s2}", 1, 1, _s1, _s2, mask=0)

# ---- 16. 确定性伪随机组合（覆盖非对齐的任意形状）----
_lcg = 12345
for _i in range(20):
    _lcg = (_lcg * 1103515245 + 12345) & 0x7FFFFFFF
    _s1 = 1 + (_lcg >> 8) % 2048
    _lcg = (_lcg * 1103515245 + 12345) & 0x7FFFFFFF
    _s2 = _s1 + (_lcg >> 8) % 4096
    _b = 1 + (_lcg >> 20) % 8
    _u(
        f"RND_{_i:02d}_S1_{_s1}_S2_{_s2}_B{_b}",
        _b,
        1,
        _s1,
        _s2,
        mask=0 if _i % 2 else 3,
    )

# ---- 17. 变长 × GQA ----
_s("VL_GQA_G4_STEP_B4", 4, 4, _sq_step(4), [x * 2 for x in _sq_step(4)], nkv=1, mask=3)
_s("VL_MHA_Nq2_B8", 8, 2, _sq_mix(512, 8), _sq_mix(1024, 8), nkv=2, mask=0)


# ---- 18. MHA（N_q == N_kv）× S2 ----
for _h in (2, 4, 8):
    for _s2 in (512, 1024):
        _u(f"MHA_N{_h}_S2_{_s2}", 1, _h, 128, _s2, nkv=_h, mask=0 if _h % 2 else 3)

# ---- 19. S1=384/512 历史交叉点 × 多档 S2 ----
for _s1 in (384, 512):
    for _s2 in (768, 2048):
        if _s2 < _s1:
            continue
        _u(f"X384_512_S1_{_s1}_S2_{_s2}", 1, 1, _s1, _s2)

# ---- 20. pScale × GQA ----
for _ps in (0.5,):
    for _nq, _nkv in ((4, 1), (8, 2)):
        _u(
            f"PS_GQA_{_ps}_Nq{_nq}_Nkv{_nkv}".replace(".", ""),
            1,
            _nq,
            128,
            1024,
            nkv=_nkv,
            pscale=_ps,
        )


# ---- 21. 变长二分诊断（定位“批间长度不同”触发的问题；命名带 DBG）----
_s("DBG_VL_B1_LONG", 1, 1, [65], [130], mask=3)  # 单批、长 65/130（无跨批）
_s("DBG_VL_B2_EQ_TINY", 2, 1, [2, 2], [4, 4], mask=3)  # 两批等长，极小
_s("DBG_VL_B2_HETERO_TINY", 2, 1, [1, 2], [2, 4], mask=3)  # 两批不等长，都在 64 内
_s("DBG_VL_B2_HETERO_REV", 2, 1, [2, 1], [4, 2], mask=3)  # 先长后短
_s("DBG_VL_B2_ONE_TINY", 2, 1, [1, 128], [2, 256], mask=3)  # 一批 1 行、一批满块
_s("DBG_VL_B2_FULL_TINY", 2, 1, [128, 1], [256, 2], mask=3)  # 先满块后 1 行
_s("DBG_VL_B2_CROSS64", 2, 1, [65, 129], [130, 258], mask=3)  # 两批都跨 64 组
_s("DBG_VL_B3_STEP", 3, 1, [1, 65, 129], [2, 130, 258], mask=3)  # 三批递增


# ===== 用例展开：形状 × layout × LSE 孪生 =====
_BASE = []


def _a(name, b, nq, sq_q, sq_kv, nkv=1, pscale=1.0, mask=3, layout="BNSD"):
    _BASE.append(
        {
            "name": name,
            "B": b,
            "N_q": nq,
            "N_kv": nkv,
            "D": 128,
            "cu_seqlens_q": None,
            "cu_seqlens_kv": None,
            "seqused_q": list(sq_q),
            "seqused_kv": list(sq_kv),
            "enable_pa": False,
            "kv_cache_layout": None,
            "block_size": None,
            "mask_mode": mask,
            "q_scale_layout": "BNSD" if layout == "BNSD" else "TND",
            "p_scale": pscale,
            "enable_lse": False,
            "input_layout": layout,
        }
    )


for _name, _b, _nq, _sq_q, _sq_kv, _nkv, _ps, _mask in _SHAPES:
    _a(_name, _b, _nq, _sq_q, _sq_kv, _nkv, _ps, _mask, "BNSD")
    _a(_name + "_TND", _b, _nq, _sq_q, _sq_kv, _nkv, _ps, _mask, "TND")

# ===== LSE 展开：每用例孪生双版本（非LSE + _LSE），成对相邻 =====
# ===== PA_BNBD(分页 KV, block_size=512) 功能用例 =====
# 与 sparse0/sparse3 正交: mask 只决定可见范围, PA 只决定 KV 物理寻址。
_PA_SHAPES = [
    ("PA_BNBD_B1_QN16_KVN2_QS2048_KVS2048_SP0", 1, 16, [2048], [2048], 2, 0),
    ("PA_BNBD_B1_QN16_KVN2_QS2048_KVS2048_SP3", 1, 16, [2048], [2048], 2, 3),
    ("PA_BNBD_B2_QN8_KVN1_QS4096_KVS4096_SP3", 2, 8, [4096, 2048], [4096, 4096], 1, 3),
    ("PA_BNBD_B1_QN16_KVN2_QS1024_KVS1024_SP0", 1, 16, [1024], [1024], 2, 0),
    ("PA_BNBD_B2_QN8_KVN1_QS2048_KVS8192_SP3", 2, 8, [2048, 1024], [8192, 8192], 1, 3),
    (
        "PA_BNBD_B4_QN4_KVN1_QS2048_KVS2048_SP0",
        4,
        4,
        [2048, 2048, 2048, 2048],
        [2048, 2048, 2048, 2048],
        1,
        0,
    ),
    (
        "PA_BNBD_B2_QN8_KVN1_QS4096_KVS_varlen_SP3",
        2,
        8,
        [4096, 2048],
        [4096, 2048],
        1,
        3,
    ),
    (
        "PA_BNBD_B3_QN8_KVN1_QS_varlen_KVS_varlen_SP3",
        3,
        8,
        [2048, 1024, 3072],
        [3600, 1024, 3200],
        1,
        3,
    ),
    ("PA_BNBD_B1_QN16_KVN2_QS8192_KVS2048_SP0", 1, 16, [8192], [2048], 2, 0, 512),
    ("PA_BNBD_B8_QN8_KVN1_QS2048_KVS2048_SP0", 8, 8, [2048] * 8, [2048] * 8, 1, 0, 512),
    ("PA_BNBD_B1_QN16_KVN2_QS2048_KVS16384_SP3", 1, 16, [2048], [16384], 2, 3, 512),
    ("PA_BNBD_BS1024_B1_QN8_KVN1_QS2048_KVS4096_SP3", 1, 8, [2048], [4096], 1, 3, 1024),
]
for _pi, _t in enumerate(_PA_SHAPES):
    _name, _b, _nq, _sq_q, _sq_kv, _nkv, _mask = _t[:7]
    _bs = _t[7] if len(_t) > 7 else 512
    for _layout in ("BNSD", "TND"):
        _BASE.append(
            {
                "name": _name + ("" if _layout == "BNSD" else "_TND"),
                "B": _b,
                "N_q": _nq,
                "N_kv": _nkv,
                "D": 128,
                "cu_seqlens_q": None,
                "cu_seqlens_kv": None,
                "seqused_q": list(_sq_q),
                "seqused_kv": list(_sq_kv),
                "enable_pa": True,
                "kv_cache_layout": "PA_BNBD",
                "block_size": _bs,
                "pa_shuffle_blocks": _pi % 6 != 0,
                "mask_mode": _mask,
                "q_scale_layout": "BNSD" if _layout == "BNSD" else "TND",
                "p_scale": 1.0,
                "enable_lse": False,
                "input_layout": _layout,
            }
        )

# 打散块表用例(与上面 PA 用例同配置, 保留独立命名便于定位)
_PA_SHUFFLE_SHAPES = [
    (
        "PA_BNBD_SHUF_B2_QN8_KVN1_QS4096_KVS4096_SP0",
        2,
        8,
        [4096, 2048],
        [4096, 4096],
        1,
        0,
        512,
    ),
    (
        "PA_BNBD_SHUF_B3_QN8_KVN1_QS_varlen_KVS_varlen_SP3",
        3,
        8,
        [2048, 1024, 3072],
        [3600, 1024, 3200],
        1,
        3,
        512,
    ),
    (
        "PA_BNBD_SHUF_BS1024_B1_QN8_KVN1_QS2048_KVS4096_SP3",
        1,
        8,
        [2048],
        [4096],
        1,
        3,
        1024,
    ),
]
for _t in _PA_SHUFFLE_SHAPES:
    _name, _b, _nq, _sq_q, _sq_kv, _nkv, _mask, _bs = _t
    for _layout in ("BNSD", "TND"):
        _BASE.append(
            {
                "name": _name + ("" if _layout == "BNSD" else "_TND"),
                "B": _b,
                "N_q": _nq,
                "N_kv": _nkv,
                "D": 128,
                "cu_seqlens_q": None,
                "cu_seqlens_kv": None,
                "seqused_q": list(_sq_q),
                "seqused_kv": list(_sq_kv),
                "enable_pa": True,
                "kv_cache_layout": "PA_BNBD",
                "block_size": _bs,
                "pa_shuffle_blocks": True,
                "mask_mode": _mask,
                "q_scale_layout": "BNSD" if _layout == "BNSD" else "TND",
                "p_scale": 1.0,
                "enable_lse": False,
                "input_layout": _layout,
            }
        )


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
    if _c["mask_mode"] == 3:
        # 因果右侧对齐：每个 batch 的可见 KV 不少于 query 行数
        assert all(k >= q for q, k in zip(_c["seqused_q"], _c["seqused_kv"])), (
            _c["name"] + " causal 形状非法"
        )
_n_bnsd = sum(1 for c in CASES if c["input_layout"] == "BNSD")
_n_tnd = sum(1 for c in CASES if c["input_layout"] == "TND")
assert _n_bnsd == _n_tnd, f"两布局用例数不等: {_n_bnsd} vs {_n_tnd}"
assert _n_bnsd >= 300, f"每布局用例数 {_n_bnsd} 过少"
assert 600 <= len(CASES) <= 680, f"用例总数 {len(CASES)} 不在 600~680"
