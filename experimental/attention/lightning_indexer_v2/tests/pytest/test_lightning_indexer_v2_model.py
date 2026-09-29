#!/usr/bin/python
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

"""LightningIndexerV2 模型场景测试套（BSND 两级 TopK candidate source）。

模型场景规格（BSND）：
    N1 ∈ {32, 16, 8, 4}，D=128，topK=512，candidate_topk_blocks=2048，
    candidate_block_size=8，S1/S2=4096（S2 为压缩后 K 长度），cmp_ratio=2，
    sparse_mode (mask_mode) ∈ {0, 3}；其余适当泛化：
    dtype FP16/BF16、B>1 变长（seqused_k）、PA_BBND key 变体、16K 压缩 K 满覆盖边界
    （numBlocks==candidate_topk_blocks）与 24K 块级真实筛选用例
    （numBlocks>candidate_topk_blocks，均 LIV2_MODEL_LARGE=1 开启）。

Per case:
  * sparse_indices vs golden（check_result；source 不得改变位置级 topk）
  * candidate_block_length 恒为空 (0,)（预留接口）
  * candidate_topk_indices vs golden（check_result_candidate；行级块号集合 + -1 槽数硬校验）

candidate source 仅支持 arch22（Ascend910B / Ascend910_93）；arch35（Ascend950）上
candidate-on 被 host 拒绝，本套件直接 skip。

Run（Ascend910B / 910_93 板 + 已编译 candidate 算子）:
    python3 -m pytest -rA -s test_lightning_indexer_v2_model.py -v -m ci
    LIV2_CASE_NAMES=LIV2_MODEL_N1_32_MODE3 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci
    LIV2_MODEL_LARGE=1 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci   # 含 16K 满覆盖边界 + 24K 真实筛选用例
"""

import os

import pytest

# ---- guarded imports: keep the module collectable on hosts without NPU / built op package ----
_GOLDEN_IMPORT_ERROR = None
try:
    import lightning_indexer_v2_golden as golden
    import result_compare_method as rc
except Exception as _exc:  # noqa: BLE001
    golden = None
    rc = None
    _GOLDEN_IMPORT_ERROR = _exc

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


# 27 base LIV2 param fields (order matches lightning_indexer_v2_golden.liv2_output_single).
BASE_FIELDS = [
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "output_idx_offset",
    "layout_q",
    "layout_k",
    "topk",
    "mask_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "cmp_ratio",
    "return_value",
    "max_seqlen_q",
]
# candidate attrs（design §5.3: param 尾部追加 candidate_topk_blocks / candidate_block_size）
CANDIDATE_FIELDS = ["candidate_topk_blocks", "candidate_block_size"]
ALL_FIELDS = BASE_FIELDS + CANDIDATE_FIELDS

# ---- 模型场景固定规格 ----
MODEL_TOPK = 512
MODEL_CAND_BLOCKS = 2048
MODEL_CAND_BLOCK_SIZE = 8
MODEL_HEAD_DIM = 128
MODEL_S1 = 4096
MODEL_S2 = 4096  # 压缩后 K 长度
MODEL_CMP_RATIO = 2


def _model_case(
    name,
    batch_size,
    n1,
    mask_mode,
    dtype,
    k_seq=MODEL_S2,
    seqused_k=None,
    cmp_residual_k=None,
    layout_k="BSND",
    pa_block_size=None,
    pa_block_num=None,
):
    """Build one model-scenario case dict（单一组合，不做笛卡尔展开）。"""
    params = {
        "batch_size": batch_size,
        "q_seq": MODEL_S1,
        "k_seq": k_seq,
        "q_t_size": 0,
        "k_t_size": 0,
        "q_head_num": n1,
        "k_head_num": 1,
        "head_dim": MODEL_HEAD_DIM,
        "block_size": pa_block_size,
        "block_num": pa_block_num,
        "qk_dtype": dtype,
        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
        "seqused_q": None,
        "seqused_k": seqused_k,
        "cmp_residual_k": cmp_residual_k,
        "output_idx_offset": None,
        "layout_q": "BSND",
        "layout_k": layout_k,
        "topk": MODEL_TOPK,
        "mask_mode": mask_mode,
        "query_datarange": [-2.0, 2.0],
        "key_datarange": [-2.0, 2.0],
        "weights_datarange": [-2.0, 2.0],
        "cmp_ratio": MODEL_CMP_RATIO,
        "return_value": 0,
        "max_seqlen_q": -1,
        "candidate_topk_blocks": MODEL_CAND_BLOCKS,
        "candidate_block_size": MODEL_CAND_BLOCK_SIZE,
    }
    missing = [f for f in ALL_FIELDS if f not in params]
    if missing:
        raise ValueError(f"case '{name}' missing fields: {missing}")
    return name, params


# ---- 用例矩阵：固定规格 N1 × sparse_mode，其余适当泛化 ----
# mask_mode=3（rightDownCausal，训练尾块 causal）：cmp_ratio=2 时 cmp_residual_k 必传（值 < cmp_ratio）
# mask_mode=0（no mask）：cmp_residual_k 必须为 None
_CASES = [
    # (a) 主链路 BSND×BSND FP16：N1 = 32 / 16 / 8 / 4
    _model_case("LIV2_MODEL_N1_32_MODE3", 1, 32, 3, "FP16", cmp_residual_k=[1]),
    _model_case("LIV2_MODEL_N1_16_MODE3", 2, 16, 3, "FP16", cmp_residual_k=[1, 1]),
    _model_case("LIV2_MODEL_N1_8_MODE3", 2, 8, 3, "FP16", cmp_residual_k=[1, 1]),
    _model_case("LIV2_MODEL_N1_4_MODE3", 4, 4, 3, "FP16", cmp_residual_k=[1, 1, 1, 1]),
    # (b) sparse_mode=0（no mask）：N1 泛化
    _model_case("LIV2_MODEL_N1_32_MODE0", 1, 32, 0, "FP16"),
    _model_case("LIV2_MODEL_N1_4_MODE0", 2, 4, 0, "FP16"),
    # (c) dtype 泛化：BF16
    _model_case("LIV2_MODEL_N1_32_MODE3_BF16", 1, 32, 3, "BF16", cmp_residual_k=[1]),
    _model_case("LIV2_MODEL_N1_8_MODE0_BF16", 1, 8, 0, "BF16"),
    # (d) 变长泛化：B>1 seqused_k 变长（2049 % candidate_block_size != 0，S2 非对齐）
    _model_case(
        "LIV2_MODEL_VARLEN_MODE3",
        2,
        8,
        3,
        "FP16",
        seqused_k=[4096, 2049],
        cmp_residual_k=[1, 1],
    ),
    # (e) key 侧泛化：BSND q × PA_BBND k（block_table 链路）
    _model_case(
        "LIV2_MODEL_PA_MODE3",
        1,
        8,
        3,
        "FP16",
        layout_k="PA_BBND",
        pa_block_size=128,
        pa_block_num=32,
        seqused_k=[4096],
        cmp_residual_k=[1],
    ),
]

# 重型用例（score 矩阵 fp32 ~1GB+/条、golden 数分钟/条）：LIV2_MODEL_LARGE=1 开启。
# 16K：numBlocks = 16384/8 = 2048 == candidate_topk_blocks，满覆盖边界（-1 槽为 0）。
# 24K：numBlocks = 24576/8 = 3072 > candidate_topk_blocks = 2048，每行真实筛掉 768~1024 个块，
#      候选块级 topk 排序/归并真正有筛选压力（默认 S2=4096 时 numBlocks=512 < 2048，全选退化）。
_LARGE_CASES = [
    _model_case(
        "LIV2_MODEL_S2_16K_N1_4_MODE3", 1, 4, 3, "FP16", k_seq=16384, cmp_residual_k=[1]
    ),
    _model_case(
        "LIV2_MODEL_S2_24K_N1_4_MODE3", 1, 4, 3, "FP16", k_seq=24576, cmp_residual_k=[1]
    ),
]

_CASES = _CASES + (
    _LARGE_CASES if os.environ.get("LIV2_MODEL_LARGE", "0") == "1" else []
)
_CASE_MAP = dict(_CASES)

requested_names = {
    name.strip()
    for name in os.environ.get("LIV2_CASE_NAMES", "").split(",")
    if name.strip()
}


def _build_cases():
    cases = []
    matched = set()
    for name, params in _CASES:
        if requested_names and name not in requested_names:
            continue
        matched.add(name)
        cases.append(pytest.param(dict(params), id=name))
    unknown = sorted(requested_names - matched)
    if unknown:
        raise ValueError(f"unknown model case(s): {unknown}")
    return cases


CASES = _build_cases()


def _device_arch():
    """arch22: Ascend910B / Ascend910_93; arch35: Ascend950; unknown: dev host."""
    if torch is None:
        return "unknown"
    try:
        props = torch.npu.get_device_properties()
        name = (getattr(props, "name", "") or "").lower()
    except Exception:  # noqa: BLE001
        return "unknown"
    if "ascend950" in name or "3510" in name:
        return "arch35"
    if "910b" in name or "910_93" in name or "2201" in name:
        return "arch22"
    return "unknown"


def _collect_skip_reason():
    if golden is None:
        return f"lightning_indexer_v2_golden import failed: {_GOLDEN_IMPORT_ERROR!r}"
    # 本套件为 candidate 专用：按 candidate 可用性 gate（candidate 随 experimental
    # 发行版以独立包名分发，可与现网接口不同源，不以现网算子可用性为前置）
    return golden.candidate_ops_unavailable_reason()


_SKIP_REASON = _collect_skip_reason()


@pytest.mark.ci
@pytest.mark.parametrize(
    "case", CASES if CASES else [pytest.param(None, id="no_cases")]
)
def test_liv2_model(case):
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    if case is None:
        pytest.skip("no model cases enabled (LIV2_MODEL_LARGE=1 enables the 16K case)")

    arch = _device_arch()
    if arch == "arch35":
        pytest.skip(
            "candidate source is arch22-only (Ascend910B / Ascend910_93); "
            "candidate-on is host-rejected on arch35 (Ascend950)"
        )

    test_data = tuple(case[f] for f in BASE_FIELDS)
    candidate_topk_blocks = int(case["candidate_topk_blocks"])
    candidate_block_size = int(case["candidate_block_size"])

    # Resolve the candidate op once; skip cleanly while the kernel/op_api is under development.
    op, op_name = golden.resolve_candidate_op()
    if op is None:
        pytest.skip(
            "candidate torch op not registered (looked for "
            f"{golden._CANDIDATE_OP_NAMES}); kernel/op_api still under development"
        )

    print(
        f"[case] B={case['batch_size']} N1={case['q_head_num']} S1={case['q_seq']} "
        f"S2={case['k_seq']} topk={case['topk']} candBlocks={candidate_topk_blocks} "
        f"bs={candidate_block_size} mask_mode={case['mask_mode']} "
        f"dtype={case['qk_dtype']} layout_k={case['layout_k']} arch={arch} op={op_name}"
    )

    out = golden.liv2_output_candidate_single(
        test_data, candidate_topk_blocks, candidate_block_size
    )

    # ---- (a) sparse_indices regression: source must not change the position-level topk ----
    sparse_result, sparse_pct = rc.check_result(
        out["cpu_sparse"],
        out["npu_sparse"],
        out["topk_value"],
        case["output_idx_offset"],
        test_data,
        out["cpu_topk_value"],
        out["npu_sparse_value"],
    )
    print(f"sparse_indices result={sparse_result} fulfill={sparse_pct}")
    assert sparse_result == "Pass", (
        "candidate source changed sparse_indices (must be bit-stable)"
    )

    # ---- (b) candidate_block_length is a reserved empty tensor ----
    blk_len = out["npu_candidate_block_length"]
    assert blk_len is not None and blk_len.numel() == 0, (
        f"candidate_block_length must be empty (0,), got numel="
        f"{0 if blk_len is None else blk_len.numel()}"
    )

    # ---- (c) candidate_topk_indices: row-level block-id set + -1 slot hard check ----
    cand_result, cand_pct = rc.check_result_candidate(
        out["cpu_candidate"],
        out["npu_candidate"],
        test_data,
        candidate_topk_blocks,
        candidate_block_size,
        expect_blk_score=out["cpu_blk"],
    )
    print(f"candidate_topk_indices result={cand_result} fulfill={cand_pct}")
    assert cand_result == "Pass", "candidate_topk_indices mismatch vs golden"
