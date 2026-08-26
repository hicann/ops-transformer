#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""UndGenQkvRmsNormRopeCache torch 接口测试（NPU 上板）。

golden、用例集与输入构造都取自 ../tests/assets/golden.py（本算子唯一 golden
实现，同时是 TTK 的输入/golden 插件），覆盖 headDim=128、(Hq,Hk,Hv) ∈ {(8,1,1),(16,2,2)}、block_size=16~512、T=1~64K
（T 本身算子侧不设上限，case 集覆盖到 64K）。

前置条件：
  1. 自定义算子包已安装：bash build.sh --pkg --ops=und_gen_qkv_rms_norm_rope_cache --soc=ascend950
                        && ./build/cann-ops-transformer-custom_linux-*.run --quiet
  2. torch_extension 已安装：cd torch_extension && python3 -m build --wheel -n
                            && pip install dist/*.whl --force-reinstall --no-deps
  3. export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH

用法：
  python3 test_torch_und_gen_qkv_rms_norm_rope_cache.py             # 默认 case 集（T ≤ 10752）
  python3 test_torch_und_gen_qkv_rms_norm_rope_cache.py --full      # 追加 T=64K
  python3 test_torch_und_gen_qkv_rms_norm_rope_cache.py --whitebox  # 白盒集（见 WHITEBOX_CASES）
  python3 test_torch_und_gen_qkv_rms_norm_rope_cache.py --case t10752_typical_h8
  python3 test_torch_und_gen_qkv_rms_norm_rope_cache.py --cpu       # 只验 case 集与 golden，不碰 NPU
"""

import argparse
import os
import sys

import torch


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests")
)
from assets.golden import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_COMBOS,
    HEAD_DIM,
    gather_cache_rows,
    golden_dense,
    golden_und_gen_qkv_rms_norm_rope_cache,
    make_cos_sin_cache,
)

_NPU = None


def _npu_modules():
    """惰性导入：--cpu 模式不需要 NPU 工具链（导入 cann_ops_transformer 即触发 JIT）。"""
    global _NPU
    if _NPU is None:
        import cann_ops_transformer
        import torch_npu

        _NPU = (torch_npu, cann_ops_transformer)
    return _NPU


# bf16 输出的比对判据。ok = (bad == 0)，不容许任何一个元素超标。
#   rtol=2^-7  —— 相对分歧上限，约 1~2 个 bf16 ULP
#   atol=2^-13 —— 绝对下限，|ref| < 2^-6 时接管（约 3% 的元素）。RoPE 两项相消出的
#                 近零点相对误差可到 100%，但绝对残差极小，靠它兜底
RTOL = 2**-7
ATOL = 2**-13


# --------------------------------------------------------------------------- #
# 白盒 case 集
# --------------------------------------------------------------------------- #
# 取自 tests/whitebox 的白盒枚举结果（括号内为白盒 case id），与
# tests/st/arch35/ttk_*_st.csv 中 w_ 前缀的 7 行一一对应，两处改动要同步。
# 补的是默认集与泛化集都没有的三个轴：
#   1. KV Cache 容量贴着下界 —— build_case 默认恒多给 2 个 block，这里用 extra_blocks
#      把 Bn*Bs 压到 ceil(T/Bs)*Bs，w_t520_bs40_cap_exact 更是 Bn*Bs 恰好等于 T，
#      slot_mapping 退化成整片 cache 的一个排列，没有未命中行；
#   2. 非 2 的幂且较大的 block_size（39/33/127/431/178）—— 默认集只有 16/64/100/128/256/512；
#   3. mrope_section=[24,20,20]（Qwen3-VL Interleaved MRoPE 的真实配置），
#      轴映射截断点落在 l=60，默认集与泛化集的截断点都不在这个位置。
# 下面两个 case 集人工对齐成表格，formatter 会炸成每键一行（142 -> 512 行），故关掉
# fmt: off
# 典型 case 集
# --------------------------------------------------------------------------- #
# headDim 固定 128；(Hq,Hk,Hv) 取 (8,1,1) / (16,2,2)。
# T = und_len + gen_len 不设上限（算子侧只要求为正，真实上限是 KV Cache 容量）；
# 下面的 case 集覆盖到 64K，再大只受跑用例的机器内存限制。
# block_size 同样不设限，case 里用 block_size 键指定，缺省取 BLOCK_SIZE。
# slot_mode: "shuffled"（分页乱序，默认）/ "contiguous"（顺序占位）
# NOTE: 当前算子实现只支持 gen_qkv / gen_weights_q / gen_weights_k / cat_indices
#       全部提供的场景，因此所有 case 均 gen_len > 0 且 cat_indices 非 None。
#       golden 本身仍支持缺省退化路径（见 golden_dense），待算子放开后可直接加用例。
TYPICAL_CASES = (
    dict(name="t2_min",            heads=(8, 1, 1),  und_len=1,     gen_len=1,
         cat="shuffled", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t8_shuffle",        heads=(8, 1, 1),  und_len=5,     gen_len=3,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t56_cores",         heads=(16, 2, 2), und_len=40,    gen_len=16,
         cat="identity", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t96_mixed",         heads=(16, 2, 2), und_len=80,    gen_len=16,
         cat="identity", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t128_block_edge",   heads=(8, 1, 1),  und_len=127,   gen_len=1,
         cat="shuffled", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False),
    dict(name="t1024_decode",      heads=(8, 1, 1),  und_len=1,     gen_len=1023,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t4096_und_heavy",   heads=(16, 2, 2), und_len=4095,  gen_len=1,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t4096_plain_rope",  heads=(8, 1, 1),  und_len=4095,  gen_len=1,
         cat="identity", slot_mode="contiguous", mrope_section=[],           big=False),
    dict(name="t10752_typical_h8", heads=(8, 1, 1),  und_len=6652,  gen_len=4100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    dict(name="t10752_typical_h16", heads=(16, 2, 2), und_len=6652, gen_len=4100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False),
    # block_size 覆盖：算子对 Bs 无假设，这几条把 2 的幂与非 2 的幂、小块与大块都过一遍
    dict(name="t64_bs16",          heads=(8, 1, 1),  und_len=40,    gen_len=24,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=16),
    dict(name="t200_bs64",         heads=(16, 2, 2), und_len=150,   gen_len=50,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=64),
    dict(name="t300_bs100",        heads=(8, 1, 1),  und_len=200,   gen_len=100,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=100),
    dict(name="t512_bs256",        heads=(8, 1, 1),  und_len=256,   gen_len=256,
         cat="identity", slot_mode="contiguous", mrope_section=[16, 16, 16], big=False, block_size=256),
    dict(name="t1024_bs512",       heads=(16, 2, 2), und_len=512,   gen_len=512,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=False, block_size=512),
    dict(name="t65536_h8",         heads=(8, 1, 1),  und_len=32768, gen_len=32768,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=True),
    dict(name="t65536_h16",        heads=(16, 2, 2), und_len=32768, gen_len=32768,
         cat="shuffled", slot_mode="shuffled",   mrope_section=[16, 16, 16], big=True),
)


# --------------------------------------------------------------------------- #
# 泛化 case 集
# --------------------------------------------------------------------------- #
# TYPICAL_CASES 是稳定的回归基线，刻意把随机种子与标量参数钉死，方便逐次比对；
# 代价是若干维度只覆盖了单点：mrope_section 17 例里 16 例是同一个 [16,16,16]，
# seed / norm_eps / max_pos 更是全集一个值。下面这组专门补这些维度，
# T 都取小值（CPU golden 是逐 token 向量化实现，小 T 才跑得快），只求覆盖不求规模。
#
# 轴映射 α(l) 的规则是 `l%3==1 且 l<3*sec[1] -> H`、`l%3==2 且 l<3*sec[2] -> W`、其余 -> T，
# 只读 sec[1]/sec[2]，sec[0] 不参与计算。所以下面按「截断点落在哪」来选值，
# 而不是按三段和是否等于 D/2 来选。
#
# 新增维度都做过区分力自检（扰动 golden 看用例是否会 FAIL）：忽略 mrope_section -> 4.6~64.8%
# 元素不达标；忽略 cat_indices -> 96.6~99.8%；忽略 norm_eps 需要把输入幅度压到 x_scale=1e-2
# 才测得出（默认幅度下 eps 对 rms 的影响低于 bf16 判据，g_eps_tiny 更是原理上无区分力，
# 只验极小 eps + 小幅度输入下 rsqrt 不出 inf/nan）。
GENERALIZED_CASES = (
    # ---- mrope_section：轴映射的各种截断形态 ----
    dict(name="g_mrope_zeros",     heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 0, 0],   big=False),
    dict(name="g_mrope_t_only",    heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[64, 0, 0],  big=False),
    dict(name="g_mrope_no_t",      heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 21, 21], big=False),
    dict(name="g_mrope_w_only",    heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 0, 21],  big=False),
    dict(name="g_mrope_sum_full",  heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[32, 16, 16], big=False),
    dict(name="g_mrope_tiny_sec",  heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[10, 1, 2],  big=False),
    # sec[1] 大到 3*sec[1] 越过 D/2：所有 l%3==1 都归 H，覆盖「截断点在区间外」
    dict(name="g_mrope_h_all",     heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[0, 44, 20], big=False),

    # ---- cat_indices 的分布形态：决定 undMask 与 gamma 选择 ----
    dict(name="g_cat_all_und",     heads=(8, 1, 1),  und_len=50,  gen_len=14,
         cat="all_und", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_cat_all_gen",     heads=(8, 1, 1),  und_len=14,  gen_len=50,
         cat="all_gen", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_cat_reverse",     heads=(16, 2, 2), und_len=40,  gen_len=24,
         cat="reverse", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_cat_dup",         heads=(8, 1, 1),  und_len=30,  gen_len=34,
         cat="dup",     slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),

    # ---- slot_mapping 的分布形态 ----
    dict(name="g_slot_reverse",    heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="reverse", mrope_section=[16, 16, 16], big=False),
    dict(name="g_slot_high",       heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="high",    mrope_section=[16, 16, 16], big=False),

    # ---- tile / 分核边界：ubFactor 上限是 64，尾块与核数边界都过一遍 ----
    dict(name="g_t55_under_cores", heads=(8, 1, 1),  und_len=30,  gen_len=25,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_t57_over_cores",  heads=(8, 1, 1),  und_len=30,  gen_len=27,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_t65_tail1",       heads=(16, 2, 2), und_len=32,  gen_len=33,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),
    dict(name="g_t113_tail1",      heads=(8, 1, 1),  und_len=56,  gen_len=57,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False),

    # ---- 标量参数：eps 与 cos_sin_cache 行数的边界 ----
    # x_scale 压到 1e-2 让 mean(x^2) ~ 1e-4，eps 才成为 rms 的主导项；
    # 用默认幅度这两条对 eps 没有任何区分力（实测扰动 eps->0 时不达标元素为 0）
    dict(name="g_eps_large",       heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         norm_eps=1e-3, x_scale=1e-2),
    dict(name="g_eps_mid",         heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         norm_eps=1e-4, x_scale=1e-2),
    # NOTE: 这条对「实现是否用了 eps」没有区分力，也不可能有——eps=1e-8 相对 mean(x^2)~1e-4
    #       本就可忽略，"忽略 eps" 在数值上就是正确答案。它验的是另一件事：
    #       极小 eps + 小幅度输入下 rsqrt 不出 inf/nan。
    dict(name="g_eps_tiny",        heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         norm_eps=1e-8, x_scale=1e-2),
    # max_pos=1 时所有 position 只能取 0，cos/sin 退化成同一行；覆盖 cache 只有一行的边界
    dict(name="g_maxpos_1",        heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         max_pos=1),
    dict(name="g_maxpos_2",        heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         max_pos=2),

    # ---- 换随机输入：同一 shape 换 4 个种子，覆盖「不同数值分布」而非不同 shape ----
    dict(name="g_seed_101",        heads=(8, 1, 1),  und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=101),
    dict(name="g_seed_202",        heads=(16, 2, 2), und_len=33,  gen_len=31,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=202),
    dict(name="g_seed_303",        heads=(8, 1, 1),  und_len=200, gen_len=112,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=303),
    dict(name="g_seed_404",        heads=(16, 2, 2), und_len=200, gen_len=112,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False, seed=404),

    # ---- 组合：多个非默认维度同时偏离，防止「单维各自过、组合起来挂」 ----
    dict(name="g_combo_a",         heads=(16, 2, 2), und_len=61,  gen_len=52,
         cat="all_gen", slot_mode="high",    mrope_section=[0, 21, 21], big=False,
         block_size=37, seed=505, norm_eps=1e-4),
    dict(name="g_combo_b",         heads=(8, 1, 1),  und_len=99,  gen_len=1,
         cat="dup",     slot_mode="reverse", mrope_section=[10, 1, 2], big=False,
         block_size=7,  seed=606, max_pos=8),
)
# fmt: on


def build_case(spec, device="cpu", seed=7, max_pos=4096, init_cache="randn"):
    """按 case spec 造一套完整输入，返回可直接 ``**kwargs`` 调 golden 的 dict。

    额外键（下划线开头）是 case 元信息，调用前用 ``pop`` 去掉即可，
    ``run_case`` 已代为处理。
    """
    # case 可以逐条覆盖随机种子与标量参数：默认集把它们钉死是为了让回归基线可复现，
    # 泛化集则靠改这几项把「同一 shape 换一批随机输入 / 换一组标量」也纳入覆盖
    seed = int(spec.get("seed", seed))
    max_pos = int(spec.get("max_pos", max_pos))
    norm_eps = float(spec.get("norm_eps", 1e-6))
    # qkv 的整体幅度。默认 1.0 时 mean(x^2) ~ 1，eps 在 [1e-8, 1e-3] 内只改变 rms 约 5e-4 相对量，
    # 落在 bf16 判据（rtol 4e-3）之下 —— 也就是说默认幅度下**测不出 eps 用没用**。
    # 把幅度压到 1e-2 后 mean(x^2) ~ 1e-4，eps 才成为 rms 的主导项，用例才有区分力。
    x_scale = float(spec.get("x_scale", 1.0))
    torch.manual_seed(seed)
    hq, hk, hv = spec["heads"]
    assert (hq, hk, hv) in HEAD_COMBOS, f"{spec['name']}: 不在支持的 head 组合内"
    n = hq + hk + hv
    d = HEAD_DIM
    und_len, gen_len = spec["und_len"], spec["gen_len"]
    total = und_len + gen_len
    assert total >= 1, "T 必须为正"

    # KV Cache 预留：按 block_size 向上取整再多给 2 个 block，制造未命中位置
    # block_size 由 case 指定（默认 BLOCK_SIZE）：算子不限制 Bs，cache 展平后 slot 直接
    # 当行号用，Bs 只影响 [Bn, Bs, ...] 怎么切分同一片连续内存
    bs = int(spec.get("block_size", BLOCK_SIZE))
    assert bs >= 1, "block_size 必须为正"
    # extra_blocks 控制冗余块数：默认 2 留出未命中位置，取 0 则 Bn*Bs 贴着 ceil(T/Bs)*Bs，
    # 覆盖 CheckKvCacheValid 的 blockNum_*blockSize_ >= totalTokens_ 下界（slot 铺满整片 cache）
    extra_blocks = int(spec.get("extra_blocks", 2))
    assert extra_blocks >= 0, "extra_blocks 不能为负"
    num_blocks = (total + bs - 1) // bs + extra_blocks
    k_shape, v_shape = (num_blocks, bs, hk, d), (num_blocks, bs, hv, d)  # 连续 BBND
    init = torch.randn if init_cache == "randn" else torch.zeros

    # cat_indices 取值只要求落在 [0, T-1]，不要求是排列：算子按 src_t < und_len 逐 token
    # 选段与选 gamma，所以 und/gen 的**分布形态**才是被覆盖的维度
    cat_mode = spec["cat"]
    if cat_mode == "none":
        cat_indices = None
    elif cat_mode == "identity":
        cat_indices = torch.arange(total, dtype=torch.int64, device=device)
    elif cat_mode == "reverse":
        # 逆序：und/gen 的边界翻到另一头，und 段落在 tile 尾部
        cat_indices = torch.arange(total - 1, -1, -1, dtype=torch.int64, device=device)
    elif cat_mode == "all_und":
        # 全部指向 und 段：undMask 恒为全 1，只走 und 的 gamma
        cat_indices = torch.randint(
            0, und_len, (total,), dtype=torch.int64, device=device
        )
    elif cat_mode == "all_gen":
        # 全部指向 gen 段：undMask 恒为 0，只走 gen 的 gamma
        assert gen_len > 0, "all_gen 需要 gen_len > 0"
        cat_indices = torch.randint(
            und_len, total, (total,), dtype=torch.int64, device=device
        )
    elif cat_mode == "dup":
        # 允许重复源：同一个源 token 被多个输出位置引用（接口只约束值域，不约束唯一性）
        cat_indices = torch.randint(
            0, total, (total,), dtype=torch.int64, device=device
        )
    else:  # shuffled：und/gen 交错，模拟真实 cat_indices
        cat_indices = torch.randperm(total, device=device).to(torch.int64)
    assert cat_mode == "none" or gen_len > 0 or und_len == total

    # slot_mapping 取值必须唯一（重复 slot 多核写冲突，结果不确定），下面每种模式都保证这点
    slot_mode = spec["slot_mode"]
    capacity = num_blocks * bs
    if slot_mode == "contiguous":
        slot_mapping = torch.arange(total, dtype=torch.int64, device=device)
    elif slot_mode == "reverse":
        slot_mapping = torch.arange(total - 1, -1, -1, dtype=torch.int64, device=device)
    elif slot_mode == "high":
        # 贴着容量上界写，覆盖最大 slot = Bn*Bs-1 这个边界行
        slot_mapping = torch.arange(
            capacity - total, capacity, dtype=torch.int64, device=device
        )
    else:
        slot_mapping = torch.randperm(capacity, device=device)[:total].to(torch.int64)

    case = dict(
        und_qkv=(
            torch.randn(und_len, n, d, dtype=torch.bfloat16, device=device) * x_scale
        ),
        und_weights_q=torch.randn(d, dtype=torch.bfloat16, device=device),
        und_weights_k=torch.randn(d, dtype=torch.bfloat16, device=device),
        cos_sin_cache=make_cos_sin_cache(max_pos, d, device=device),
        k_cache=init(k_shape, dtype=torch.bfloat16, device=device),
        v_cache=init(v_shape, dtype=torch.bfloat16, device=device),
        slot_mapping=slot_mapping,
        positions=torch.randint(
            0, max_pos, (3, total), dtype=torch.int64, device=device
        ),
        gen_qkv=(
            (torch.randn(gen_len, n, d, dtype=torch.bfloat16, device=device) * x_scale)
            if gen_len > 0
            else None
        ),
        gen_weights_q=(
            torch.randn(d, dtype=torch.bfloat16, device=device) if gen_len > 0 else None
        ),
        gen_weights_k=(
            torch.randn(d, dtype=torch.bfloat16, device=device) if gen_len > 0 else None
        ),
        cat_indices=cat_indices,
        num_heads_q=hq,
        num_heads_k=hk,
        num_heads_v=hv,
        norm_eps=norm_eps,
        mrope_section=spec["mrope_section"],
    )
    case["_meta"] = dict(
        name=spec["name"],
        total=total,
        heads=(hq, hk, hv),
        num_blocks=num_blocks,
        block_size=bs,
        seed=seed,
        max_pos=max_pos,
        norm_eps=norm_eps,
        x_scale=x_scale,
        cat=cat_mode,
        slot_mode=slot_mode,
        mrope_section=spec["mrope_section"],
    )
    return case


# --------------------------------------------------------------------------- #


# 自检：仅验证 case 集构造与 golden 可跑通（纯 CPU，不涉及 NPU）
# --------------------------------------------------------------------------- #
def _self_check(include_big=False):
    print(
        f"golden case set (headDim={HEAD_DIM}, block_size={BLOCK_SIZE}, heads in "
        f"{{{HEAD_COMBOS[0]}, {HEAD_COMBOS[1]}}})"
    )
    for spec in TYPICAL_CASES:
        total = spec["und_len"] + spec["gen_len"]
        if spec["big"] and not include_big:
            print(f"  [SKIP] {spec['name']:<22} T={total}")
            continue
        case = build_case(spec)
        meta = case.pop("_meta")
        q, k_cache, v_cache = golden_und_gen_qkv_rms_norm_rope_cache(**case)
        assert q.shape == (total, meta["heads"][0], HEAD_DIM)
        assert q.dtype == torch.bfloat16
        print(
            f"  [OK]   {meta['name']:<22} T={total:<6} heads={meta['heads']} "
            f"q={tuple(q.shape)} k_cache={tuple(k_cache.shape)}"
        )
    print("golden self-check PASS")


WHITEBOX_CASES = (
    # T=38：每核 1 token（T < 56 核），容量只余 1 个槽位
    dict(
        name="w_t38_bs39_capfull",
        heads=(8, 1, 1),
        und_len=33,
        gen_len=5,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[16, 16, 16],
        big=False,
        block_size=39,
        extra_blocks=0,
    ),
    # T 等于 AIV 核数：每核 1 个 token 且余数为 0，多核切分的边界
    dict(
        name="w_t56_bs33_h16",
        heads=(16, 2, 2),
        und_len=40,
        gen_len=16,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[16, 24, 24],
        big=False,
        block_size=33,
        extra_blocks=0,
    ),
    # 质数 block_size + 容量只余 2 个槽位
    dict(
        name="w_t379_bs127_h16",
        heads=(16, 2, 2),
        und_len=334,
        gen_len=45,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[16, 24, 24],
        big=False,
        block_size=127,
        extra_blocks=0,
    ),
    # Bn*Bs == T：cache 被 slot_mapping 铺满，无未命中行
    dict(
        name="w_t520_bs40_cap_exact",
        heads=(8, 1, 1),
        und_len=479,
        gen_len=41,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[16, 24, 24],
        big=False,
        block_size=40,
        extra_blocks=0,
    ),
    # ubFactor 被 UB 容量夹住（N=20 时上界 10）+ 大质数 block_size
    dict(
        name="w_t1356_bs431_h16",
        heads=(16, 2, 2),
        und_len=1129,
        gen_len=227,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[16, 24, 24],
        big=False,
        block_size=431,
        extra_blocks=0,
    ),
    # ubFactor 被 UB 容量夹住（N=10 时上界 18）
    dict(
        name="w_t1696_bs178_h8",
        heads=(8, 1, 1),
        und_len=1503,
        gen_len=193,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[16, 24, 24],
        big=False,
        block_size=178,
        extra_blocks=0,
    ),
    # Qwen3-VL 4B TP1 图像 prefill 的真实配置：sec=[24,20,20]、max_pos 取到 32K
    dict(
        name="w_qwen3vl_t2096_sec242020",
        heads=(16, 2, 2),
        und_len=2048,
        gen_len=48,
        cat="shuffled",
        slot_mode="shuffled",
        mrope_section=[24, 20, 20],
        big=False,
        block_size=256,
        extra_blocks=7,
        max_pos=32768,
    ),
)


def _to_npu(t):
    return None if t is None else t.npu()


def compare(name, got, ref):
    """got/ref 均为 CPU 张量；ref 为 golden 的 float32 结果，按 bf16 判据比对。

    ref 先降到 bf16 再比：NPU 输出本就是 bf16，若拿它直接对 fp32 的 golden，会把最后
    那次舍入（<=0.5 ULP）也算成误差，容差得凭空多留一截。降完之后两边同样舍入、该项
    抵消，判据与 TTK ST（ttk_golden 同样返回 bf16）完全同口径。
    """
    got_f = got.float()
    ref_f = ref.to(torch.bfloat16).float()
    diff = (got_f - ref_f).abs()
    denom = ref_f.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    bad = (diff > (ATOL + RTOL * ref_f.abs())).sum().item()
    ok = bad == 0
    return (
        ok,
        f"{name}: max_abs={max_abs:.5f} max_rel={max_rel:.5f} 不达标元素={bad}/{ref_f.numel()}",
    )


def run_case(spec, verbose=True):
    case = build_case(spec)
    meta = case.pop("_meta")
    total, (hq, hk, hv) = meta["total"], meta["heads"]

    # ---- golden（CPU float32）----
    ref_q, ref_k, ref_v = golden_dense(
        case["und_qkv"],
        case["gen_qkv"],
        case["und_weights_q"],
        case["und_weights_k"],
        case["gen_weights_q"],
        case["gen_weights_k"],
        case["cos_sin_cache"],
        case["positions"],
        case["cat_indices"],
        hq,
        hk,
        hv,
        case["norm_eps"],
        case["mrope_section"],
    )

    # ---- NPU ----
    k_cache_in, v_cache_in = case["k_cache"], case["v_cache"]
    k_cache = k_cache_in.clone().npu()
    v_cache = v_cache_in.clone().npu()
    _, cot = _npu_modules()
    q = cot.und_gen_qkv_rms_norm_rope_cache(
        case["und_qkv"].npu(),
        case["und_weights_q"].npu(),
        case["und_weights_k"].npu(),
        case["cos_sin_cache"].npu(),
        k_cache,
        v_cache,
        case["slot_mapping"].npu(),
        case["positions"].npu(),
        _to_npu(case["gen_qkv"]),
        _to_npu(case["gen_weights_q"]),
        _to_npu(case["gen_weights_k"]),
        _to_npu(case["cat_indices"]),
        num_heads_q=hq,
        num_heads_k=hk,
        num_heads_v=hv,
        norm_eps=case["norm_eps"],
        mrope_section=case["mrope_section"] or [],
    )
    torch.npu.synchronize()

    # ---- 接口连通性（硬断言）----
    assert q.shape == (total, hq, HEAD_DIM), (
        f"q shape {tuple(q.shape)} != {(total, hq, HEAD_DIM)}"
    )
    assert q.dtype == torch.bfloat16, f"q dtype {q.dtype}"
    assert k_cache.shape == k_cache_in.shape and v_cache.shape == v_cache_in.shape
    assert k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16

    # k_cache/v_cache 未被 slot_mapping 命中的位置必须保持入参原值
    slot = case["slot_mapping"]
    untouched = torch.ones(meta["num_blocks"] * meta["block_size"], dtype=torch.bool)
    untouched[slot] = False
    k_out_rows = k_cache.cpu().reshape(-1, hk, HEAD_DIM)
    k_in_rows = k_cache_in.reshape(-1, hk, HEAD_DIM)
    untouched_ok = torch.equal(k_out_rows[untouched], k_in_rows[untouched])

    if verbose:
        # 只打印偏离默认的维度，默认集的输出保持原样不被噪声淹没
        extra = []
        if meta["mrope_section"] != [16, 16, 16]:
            extra.append(f"sec={meta['mrope_section']}")
        if meta["cat"] != "shuffled":
            extra.append(f"cat={meta['cat']}")
        if meta["slot_mode"] != "shuffled":
            extra.append(f"slot={meta['slot_mode']}")
        if meta["seed"] != 7:
            extra.append(f"seed={meta['seed']}")
        if meta["norm_eps"] != 1e-6:
            extra.append(f"eps={meta['norm_eps']:g}")
        if meta["max_pos"] != 4096:
            extra.append(f"max_pos={meta['max_pos']}")
        print(
            f"  [{meta['name']:<22}] 接口 OK  T={total:<6} heads=({hq},{hk},{hv}) bs={meta['block_size']:<4} "
            f"未命中槽保持原值={untouched_ok}  {' '.join(extra)}"
        )

    # ---- 数值精度（对 golden）----
    results = [
        compare("q", q.cpu(), ref_q),
        compare("k_cache", gather_cache_rows(k_cache.cpu(), slot), ref_k),
        compare("v_cache", gather_cache_rows(v_cache.cpu(), slot), ref_v),
    ]
    numeric_ok = all(ok for ok, _ in results) and untouched_ok
    if verbose:
        for _, msg in results:
            print(f"      {msg}")
        print(f"      -> 精度 {'PASS' if numeric_ok else 'FAIL'}")
    return numeric_ok


def main():
    parser = argparse.ArgumentParser(
        description="UndGenQkvRmsNormRopeCache torch 接口测试"
    )
    parser.add_argument("--full", action="store_true", help="包含 T=64K 的大 case")
    parser.add_argument("--case", default=None, help="只跑指定 case 名")
    parser.add_argument(
        "--generalized",
        action="store_true",
        help="只跑泛化集（mrope_section / cat / slot / seed / eps / max_pos 的覆盖补充）",
    )
    parser.add_argument(
        "--whitebox",
        action="store_true",
        help="只跑白盒集（KV Cache 容量下界 / 非 2 的幂 block_size / Qwen3-VL 轴切分）",
    )
    parser.add_argument(
        "--all-sets", action="store_true", help="典型集 + 泛化集 + 白盒集一起跑"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="只在 CPU 上验证 case 集构造与 golden 可跑通，不碰 NPU",
    )
    args = parser.parse_args()

    if args.cpu:
        _self_check(include_big=args.full)
        return 0

    torch_npu, _ = _npu_modules()
    print(
        f"device: {torch.npu.get_device_name(0)}, torch {torch.__version__}, "
        f"torch_npu {torch_npu.__version__}"
    )
    if args.generalized:
        pool = list(GENERALIZED_CASES)
    elif args.whitebox:
        pool = list(WHITEBOX_CASES)
    elif args.all_sets:
        pool = list(TYPICAL_CASES) + list(GENERALIZED_CASES) + list(WHITEBOX_CASES)
    else:
        pool = list(TYPICAL_CASES)
    specs = [s for s in pool if (args.case is None or s["name"] == args.case)]
    if args.case is None and not args.full:
        skipped = [s["name"] for s in specs if s["big"]]
        specs = [s for s in specs if not s["big"]]
        if skipped:
            print(f"跳过大 case（--full 开启）: {', '.join(skipped)}")
    assert specs, "没有匹配的 case"

    results = [run_case(s) for s in specs]
    print(f"\n接口连通性: {len(results)}/{len(results)} PASS")
    print(
        f"数值精度:   {sum(results)}/{len(results)} PASS"
        f"{'' if all(results) else '  <-- 预期全 PASS，出现 FAIL 即为回退'}"
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
