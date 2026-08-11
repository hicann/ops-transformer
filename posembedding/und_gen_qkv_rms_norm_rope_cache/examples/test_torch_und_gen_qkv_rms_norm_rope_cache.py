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
"""

import argparse
import os
import sys

import torch
import torch_npu


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests"))
from assets.golden import (  # noqa: E402
    GENERALIZED_CASES,
    HEAD_DIM,
    TYPICAL_CASES,
    build_case,
    gather_cache_rows,
    golden_dense,
)

import cann_ops_transformer  # noqa: E402

# bf16 输出的比对判据，与 tests/st/arch35 两个 TTK CSV 的 precision_tolerances /
# absolute_precision 取同一套值，口径统一：
#   rtol=2^-7 —— |golden|>1e-3 处的分歧上限，恰为 2 个 bf16 ULP
#   atol=2^-13 —— 覆盖 |golden|<=1e-3 的长尾（绝对差 <=7.6e-6）与 RoPE 两项相消出的近零点
# 判据本身不放松：下面 ok = (bad == 0)，不容许任何一个元素超标。
RTOL = 2 ** -7
ATOL = 2 ** -13


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
WHITEBOX_CASES = (
    # T=38：每核 1 token（T < 56 核），容量只余 1 个槽位
    dict(name="w_t38_bs39_capfull",     heads=(8, 1, 1),  und_len=33,   gen_len=5,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 16, 16], big=False,
         block_size=39,  extra_blocks=0),
    # T 等于 AIV 核数：每核 1 个 token 且余数为 0，多核切分的边界
    dict(name="w_t56_bs33_h16",         heads=(16, 2, 2), und_len=40,   gen_len=16,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 24, 24], big=False,
         block_size=33,  extra_blocks=0),
    # 质数 block_size + 容量只余 2 个槽位
    dict(name="w_t379_bs127_h16",       heads=(16, 2, 2), und_len=334,  gen_len=45,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 24, 24], big=False,
         block_size=127, extra_blocks=0),
    # Bn*Bs == T：cache 被 slot_mapping 铺满，无未命中行
    dict(name="w_t520_bs40_cap_exact",  heads=(8, 1, 1),  und_len=479,  gen_len=41,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 24, 24], big=False,
         block_size=40,  extra_blocks=0),
    # ubFactor 被 UB 容量夹住（N=20 时上界 10）+ 大质数 block_size
    dict(name="w_t1356_bs431_h16",      heads=(16, 2, 2), und_len=1129, gen_len=227,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 24, 24], big=False,
         block_size=431, extra_blocks=0),
    # ubFactor 被 UB 容量夹住（N=10 时上界 18）
    dict(name="w_t1696_bs178_h8",       heads=(8, 1, 1),  und_len=1503, gen_len=193,
         cat="shuffled", slot_mode="shuffled", mrope_section=[16, 24, 24], big=False,
         block_size=178, extra_blocks=0),
    # Qwen3-VL 4B TP1 图像 prefill 的真实配置：sec=[24,20,20]、max_pos 取到 32K
    dict(name="w_qwen3vl_t2096_sec242020", heads=(16, 2, 2), und_len=2048, gen_len=48,
         cat="shuffled", slot_mode="shuffled", mrope_section=[24, 20, 20], big=False,
         block_size=256, extra_blocks=7, max_pos=32768),
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
    return ok, f"{name}: max_abs={max_abs:.5f} max_rel={max_rel:.5f} 不达标元素={bad}/{ref_f.numel()}"


def run_case(spec, verbose=True):
    case = build_case(spec)
    meta = case.pop("_meta")
    total, (hq, hk, hv) = meta["total"], meta["heads"]

    # ---- golden（CPU float32）----
    ref_q, ref_k, ref_v = golden_dense(
        case["und_qkv"], case["gen_qkv"], case["und_weights_q"], case["und_weights_k"],
        case["gen_weights_q"], case["gen_weights_k"], case["cos_sin_cache"],
        case["positions"], case["cat_indices"], hq, hk, hv,
        case["norm_eps"], case["mrope_section"])

    # ---- NPU ----
    k_cache_in, v_cache_in = case["k_cache"], case["v_cache"]
    k_cache = k_cache_in.clone().npu()
    v_cache = v_cache_in.clone().npu()
    q = cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
        case["und_qkv"].npu(), case["und_weights_q"].npu(), case["und_weights_k"].npu(),
        case["cos_sin_cache"].npu(), k_cache, v_cache,
        case["slot_mapping"].npu(), case["positions"].npu(),
        _to_npu(case["gen_qkv"]), _to_npu(case["gen_weights_q"]), _to_npu(case["gen_weights_k"]),
        _to_npu(case["cat_indices"]),
        num_heads_q=hq, num_heads_k=hk, num_heads_v=hv,
        norm_eps=case["norm_eps"], mrope_section=case["mrope_section"] or [])
    torch.npu.synchronize()

    # ---- 接口连通性（硬断言）----
    assert q.shape == (total, hq, HEAD_DIM), f"q shape {tuple(q.shape)} != {(total, hq, HEAD_DIM)}"
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
        print(f"  [{meta['name']:<22}] 接口 OK  T={total:<6} heads=({hq},{hk},{hv}) bs={meta['block_size']:<4} "
              f"未命中槽保持原值={untouched_ok}  {' '.join(extra)}")

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
    parser = argparse.ArgumentParser(description="UndGenQkvRmsNormRopeCache torch 接口测试")
    parser.add_argument("--full", action="store_true", help="包含 T=64K 的大 case")
    parser.add_argument("--case", default=None, help="只跑指定 case 名")
    parser.add_argument("--generalized", action="store_true",
                        help="只跑泛化集（mrope_section / cat / slot / seed / eps / max_pos 的覆盖补充）")
    parser.add_argument("--whitebox", action="store_true",
                        help="只跑白盒集（KV Cache 容量下界 / 非 2 的幂 block_size / Qwen3-VL 轴切分）")
    parser.add_argument("--all-sets", action="store_true", help="典型集 + 泛化集 + 白盒集一起跑")
    args = parser.parse_args()

    print(f"device: {torch.npu.get_device_name(0)}, torch {torch.__version__}, "
          f"torch_npu {torch_npu.__version__}")
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
    print(f"数值精度:   {sum(results)}/{len(results)} PASS"
          f"{'' if all(results) else '  <-- 预期全 PASS，出现 FAIL 即为回退'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
