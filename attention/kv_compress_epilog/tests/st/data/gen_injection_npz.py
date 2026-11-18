# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成 kv_compress_epilog arch22 INT8 ST 用例的 npz 定值注入文件。

对应 tests/st/arch22/ttk_kernel_kv_compress_epilog_int8_st.csv 的 SA-07/SA-08/SA-09 用例：

- 正向注入：为本目录（tests/st/data/）生成 <testcase_name>.npz，
  键固定为 cache(uint8)、x_bits(uint16 bf16 位型)、slot_mapping(int64)；
  shape/dtype 从 tests/st/arch22/ 下 ST 用例 csv 解析，不硬编码；
- SA-09 负向变体：为 sa09_negative/ 子目录生成同名 npz（slot 故意取错值），
  供「注入生效负向校验」使用——注入生效时该变体 golden 输出必与正向不同；
- 覆盖 SA-08 清单 26 条（l0_rne_tie/rne_tie_even/rne_tie_odd/rne_tie_neg、
  l0_rope_nan_inf/rope_nan/rope_inf/rope_negzero、minmax_alternate、large_input、
  tiny_input、nope_nan_nocrash、broadcast_size1、group_scale_spread、rowtail_keep、
  q_sat_pos127/q_sat_neg127、slot_zero/skip_head/skip_mid_bs32/slot_last/
  allvalid_full/slot_shuffle、x_mixzero、amax0_partial、scale_bf16_round、
  pad18_zero、l0_pad_zero、l0_skip_all/skip_all_bs1/skip_all_bs8/skip_all_bs64），
  另加 SA-09 负向 4 条。

用法（在有 python+numpy 的目标机上执行；npz 为生成物不入库）：
  python3 gen_injection_npz.py            # 生成全部 npz 到本目录
  python3 gen_injection_npz.py --check    # 校验已生成 npz 与脚本再生内容逐键一致
"""

import argparse
import ast
import csv
import io
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTCASES_DIR = os.path.normpath(os.path.join(HERE, os.pardir))
ST_CSV = os.path.join(
    HERE, os.pardir, "arch22", "ttk_kernel_kv_compress_epilog_int8_st.csv"
)

D = 512
NOPE = 448
ROPE = 64
GROUP = 64

BF16_ONE = np.uint16(0x3F80)  # bf16(1.0)
BF16_NAN = np.uint16(0x7FC0)
BF16_INF = np.uint16(0x7F80)
BF16_NINF = np.uint16(0xFF80)
BF16_NEGZERO = np.uint16(0x8000)
BF16_MAX = np.uint16(0x7F7F)  # bf16 最大有限值 ≈3.389e38
BF16_NMAX = np.uint16(0xFF7F)


def _bits(vals):
    """f32 → bf16 位型（RNE 单次舍入，与 golden.py _f32_to_bf16_bits 同语义）。"""
    su = (
        np.ascontiguousarray(np.asarray(vals), dtype=np.float32)
        .view(np.uint32)
        .astype(np.uint64)
    )
    sb = su + (np.uint64(0x7FFF) + ((su >> np.uint64(16)) & np.uint64(1)))
    return (sb >> np.uint64(16)).astype(np.uint16)


def _load_case_meta():
    """从 l0/l1 设计源 csv 解析 {name: (cache_shape, x_shape, slot_shape)}。"""
    meta = {}
    for path in (ST_CSV,):
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                shapes = ast.literal_eval(row["input_shapes"])
                meta[row["testcase_name"]] = tuple(tuple(s) for s in shapes)
    return meta


def _base_x(shape):
    bits = np.full(shape, BF16_ONE, dtype=np.uint16)
    return bits


def _mark_tokens(bits):
    """rope[63] 置 bf16(1.0+t) 作 token 区分标记（rope 直通段，不影响量化）。"""
    for t in range(bits.shape[0]):
        bits[t, D - 1] = _bits([1.0 + t])[0]
    return bits


def _x_rne_tie(shape, ties):
    bits = _base_x(shape)
    for g in range(NOPE // GROUP):
        bits[:, g * GROUP] = _bits([127.0])[0]  # amax=127 → scale=1.0 精确
        for k, val in enumerate(ties):
            bits[:, g * GROUP + 1 + k] = _bits([val])[0]
    return _mark_tokens(bits)


def _x_rope_bits(shape, rope_bits):
    bits = _base_x(shape)
    for t in range(bits.shape[0]):
        bits[t, NOPE:] = np.array(rope_bits, dtype=np.uint16)
    return bits


def _build_x(name, shape):
    """按用例名构造 x 的 bf16 位型数组 [bs, 512]。"""
    if name == "wb_int8_l0_rne_tie":
        return _x_rne_tie(shape, [2.5, 3.5, -2.5, -3.5, 0.5, 1.5])
    if name == "wb_int8_rne_tie_even":
        return _x_rne_tie(shape, [2.5, 0.5])
    if name == "wb_int8_rne_tie_odd":
        return _x_rne_tie(shape, [3.5, 1.5])
    if name == "wb_int8_rne_tie_neg":
        return _x_rne_tie(shape, [-2.5, -3.5])
    if name == "wb_int8_l0_rope_nan_inf":
        return _x_rope_bits(
            shape, [BF16_NAN, BF16_INF, BF16_NINF, BF16_NEGZERO] + [BF16_ONE] * 60
        )
    if name == "wb_int8_rope_nan":
        return _x_rope_bits(shape, [0x7FC0, 0x7FE0, 0xFFC0] + [BF16_ONE] * 61)
    if name == "wb_int8_rope_inf":
        return _x_rope_bits(shape, [BF16_INF, BF16_NINF] + [BF16_ONE] * 62)
    if name == "wb_int8_rope_negzero":
        return _x_rope_bits(shape, [BF16_NEGZERO] * ROPE)
    if name == "wb_int8_minmax_alternate":
        bits = _base_x(shape)
        # token0：nope 段 ±max_bf16 交替；token1+：全 1.0（token 区分，SA-09 需要）
        bits[0, :NOPE] = np.where(
            (np.arange(NOPE) % 2) == 0, BF16_MAX, BF16_NMAX
        ).astype(np.uint16)
        return bits
    if name == "wb_int8_large_input":
        bits = np.full(shape, _bits([1e30])[0], dtype=np.uint16)
        return _mark_tokens(bits)
    if name == "wb_int8_tiny_input":
        bits = np.full(shape, _bits([1e-30])[0], dtype=np.uint16)
        return _mark_tokens(bits)
    if name == "wb_int8_nope_nan_nocrash":
        bits = _base_x(shape)
        bits[:, 0] = BF16_NAN  # nope 首元素 NaN：amax/scale 传播 NaN，仅断言不崩溃
        return bits
    if name == "wb_int8_broadcast_size1":
        bits = _base_x(shape)
        for g in range(NOPE // GROUP):
            bits[:, g * GROUP : (g + 1) * GROUP] = _bits([2.0**g])[0]
        return bits
    if name == "wb_int8_group_scale_spread":
        bits = _base_x(shape)
        for g in range(NOPE // GROUP):
            bits[:, g * GROUP : (g + 1) * GROUP] = _bits([10.0 ** (3 - g)])[0]
        return _mark_tokens(bits)
    if name == "wb_int8_q_sat_pos127":
        bits = _base_x(shape)
        bits[:, :NOPE] = _bits([127.0])[0]  # 组内 amax=127 → q 全 +127
        return _mark_tokens(bits)
    if name == "wb_int8_q_sat_neg127":
        bits = _base_x(shape)
        bits[:, :NOPE] = _bits([-127.0])[0]  # q 全 -127
        return _mark_tokens(bits)
    if name == "wb_int8_x_mixzero":
        bits = np.zeros(shape, dtype=np.uint16)
        bits[:, NOPE:] = BF16_NEGZERO  # nope +0.0 / rope -0.0
        return bits
    if name == "wb_int8_amax0_partial":
        bits = _base_x(shape)
        bits[:, :GROUP] = 0  # 组0 全零（scale0=1.0），其余组 1.0
        return _mark_tokens(bits)
    if name == "wb_int8_scale_bf16_round":
        bits = _base_x(shape)
        for g in range(NOPE // GROUP):
            bits[:, g * GROUP] = _bits([100.0])[0]  # amax=100 → 100/127 非 bf16 精确
        return _mark_tokens(bits)
    # pad/skip/slot 形态类用例：x 内容非关注项，全 1.0（token 标记保持可区分）
    return _mark_tokens(_base_x(shape))


def _build_slot(name, bs, num_slots):
    """按用例名构造 slot_mapping（int64）。"""
    if name in (
        "wb_int8_l0_skip_all",
        "wb_int8_skip_all_bs1",
        "wb_int8_skip_all_bs8",
        "wb_int8_skip_all_bs64",
    ):
        return np.full((bs,), -1, dtype=np.int64)
    if name == "wb_int8_skip_head":
        return np.array([-1] + list(range(1, bs)), dtype=np.int64)
    if name == "wb_int8_skip_mid_bs32":
        slot = np.arange(bs, dtype=np.int64)
        slot[bs // 2] = -1
        return slot
    if name == "wb_int8_slot_last":
        slot = np.full((bs,), -1, dtype=np.int64)
        slot[0] = num_slots - 1
        return slot
    if name == "wb_int8_allvalid_full":
        return np.arange(bs, dtype=np.int64)  # E6 deviation：全有效无跳过
    if name == "wb_int8_slot_shuffle":
        return np.random.default_rng(20261104).permutation(bs).astype(np.int64)
    if name == "wb_int8_slot_zero":
        return np.zeros((bs,), dtype=np.int64)
    if name == "wb_int8_rowtail_keep":
        slot = np.arange(bs, dtype=np.int64)
        slot[-1] = -1  # 保留末行为 0xFF sentinel 参照
        return slot
    return np.arange(bs, dtype=np.int64)


def _build_cache(name, shape):
    """sentinel 用例预填 0xFF；其余用确定性伪随机字节（种子取自用例名）。"""
    if name in ("wb_int8_pad18_zero", "wb_int8_l0_pad_zero", "wb_int8_rowtail_keep"):
        return np.full(shape, 0xFF, dtype=np.uint8)
    rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


# SA-09 负向变体：与正向同 cache/x，slot 故意取错值（注入生效时 golden 输出必变）
NEGATIVE_SLOT = {
    "wb_int8_l0_skip_all": lambda slot: np.where(np.arange(slot.size) == 0, 0, -1),
    "wb_int8_skip_head": lambda slot: np.where(np.arange(slot.size) == 0, 0, slot),
    "wb_int8_l0_rne_tie": lambda slot: slot[[1, 0, 2, 3]],
    "wb_int8_minmax_alternate": lambda slot: slot[[1, 0]],
}

INJECT_CASES = [
    "wb_int8_l0_rne_tie",
    "wb_int8_rne_tie_even",
    "wb_int8_rne_tie_odd",
    "wb_int8_rne_tie_neg",
    "wb_int8_l0_rope_nan_inf",
    "wb_int8_rope_nan",
    "wb_int8_rope_inf",
    "wb_int8_rope_negzero",
    "wb_int8_minmax_alternate",
    "wb_int8_large_input",
    "wb_int8_tiny_input",
    "wb_int8_nope_nan_nocrash",
    "wb_int8_broadcast_size1",
    "wb_int8_group_scale_spread",
    "wb_int8_rowtail_keep",
    "wb_int8_q_sat_pos127",
    "wb_int8_q_sat_neg127",
    "wb_int8_slot_zero",
    "wb_int8_skip_head",
    "wb_int8_skip_mid_bs32",
    "wb_int8_slot_last",
    "wb_int8_allvalid_full",
    "wb_int8_slot_shuffle",
    "wb_int8_x_mixzero",
    "wb_int8_amax0_partial",
    "wb_int8_scale_bf16_round",
    "wb_int8_pad18_zero",
    "wb_int8_l0_pad_zero",
    "wb_int8_l0_skip_all",
    "wb_int8_skip_all_bs1",
    "wb_int8_skip_all_bs8",
    "wb_int8_skip_all_bs64",
]


def _serialize(cache, x_bits, slot):
    buf = io.BytesIO()
    np.savez(buf, cache=cache, x_bits=x_bits, slot_mapping=slot)
    return buf.getvalue()


def _npz_matches(path, cache, x_bits, slot):
    """加载已生成 npz 并逐键比对（zip 内嵌时间戳，不能按文件字节流比对）。"""
    try:
        with np.load(path) as data:
            return (
                set(data.files) == {"cache", "x_bits", "slot_mapping"}
                and data["cache"].dtype == np.uint8
                and data["x_bits"].dtype == np.uint16
                and data["slot_mapping"].dtype == np.int64
                and np.array_equal(data["cache"], cache)
                and np.array_equal(data["x_bits"], x_bits)
                and np.array_equal(data["slot_mapping"], slot)
            )
    except Exception:
        return False


def _build_case(name, meta):
    if name not in meta:
        raise KeyError("case %s not found in l0/l1 design csv" % name)
    cache_shape, x_shape, slot_shape = meta[name]
    if len(x_shape) != 2 or x_shape[1] != D:
        raise ValueError("case %s x shape %s unexpected" % (name, x_shape))
    num_slots = int(np.prod(cache_shape[:3]))
    cache = _build_cache(name, cache_shape)
    x_bits = _build_x(name, x_shape)
    slot = _build_slot(name, x_shape[0], num_slots).reshape(slot_shape)
    return cache, x_bits, slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="不写入，校验已存在 npz 与脚本再生内容逐字节一致",
    )
    args = parser.parse_args()

    meta = _load_case_meta()
    neg_dir = os.path.join(HERE, "sa09_negative")
    failed = []
    written = 0
    for name in INJECT_CASES:
        if name not in meta:
            print("skip %s: not in %s" % (name, ST_CSV))
            continue
        cache, x_bits, slot = _build_case(name, meta)
        targets = [(os.path.join(HERE, name + ".npz"), cache, x_bits, slot)]
        if name in NEGATIVE_SLOT:
            neg_slot = NEGATIVE_SLOT[name](slot).astype(np.int64)
            targets.append(
                (os.path.join(neg_dir, name + ".npz"), cache, x_bits, neg_slot)
            )
        for path, c, xb, sl in targets:
            if args.check:
                if not os.path.isfile(path):
                    print("skip (not generated): %s" % path)
                    continue
                if not _npz_matches(path, c, xb, sl):
                    failed.append("%s: content mismatch" % path)
                continue
            payload = _serialize(c, xb, sl)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as file:
                file.write(payload)
            written += 1
    if args.check:
        if failed:
            print("check FAILED:")
            for item in failed:
                print("  " + item)
            return 1
        print(
            "check OK: %d inject cases + %d negative variants"
            % (len(INJECT_CASES), len(NEGATIVE_SLOT))
        )
        return 0
    print("generated %d npz files under %s (+ sa09_negative/)" % (written, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
