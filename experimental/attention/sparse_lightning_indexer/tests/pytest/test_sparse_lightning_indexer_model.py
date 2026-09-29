#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SparseLightningIndexer（candidate consumer / candidate_mode==2）模型场景测试套（BSND）。

模型场景规格与 LIV2 侧 test_lightning_indexer_v2_model.py 一一对应：
    N1 ∈ {32, 16, 8, 4}，D=128，topK=512，candidate_topk_blocks（candBlocks）=2048，
    candidate_block_size=8，S1/S2=4096（S2 为压缩后 K 长度），cmp_ratio=2，
    sparse_mode (mask_mode) ∈ {0, 3}；其余适当泛化：
    dtype FP16/BF16、B>1 变长（seqused_k）、PA_BBND key 变体、16K 压缩 K 满覆盖边界
    （SLI_MODEL_LARGE=1 开启）。

候选来源固定 cand_spec="self"：候选由同 score 过参考 source 行为构造，
即真实 source→consumer 链路形态。

Per case:
  * sparse_indices vs consumer golden（check_result：多重集合门 → compare_topk_valid
    官方规则 → -1 槽硬校验；golden = NEG_HUGE 降级 leak 语义 → 原 topk 管线）

仅支持 ascend910b / ascend910_93（arch22）。

Run:
    python3 -m pytest -rA -s test_sparse_lightning_indexer_model.py -v -m ci
    SLI_CASE_NAMES=SLI_MODEL_N1_32_MODE3 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci
    SLI_MODEL_LARGE=1 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci   # 含 16K 边界用例
"""

import os

import pytest

# ---- guarded imports: keep the module collectable on hosts without NPU / built op package ----
_GOLDEN_IMPORT_ERROR = None
try:
    import sparse_lightning_indexer_golden as sli_golden
except Exception as _exc:  # noqa: BLE001
    sli_golden = None
    _GOLDEN_IMPORT_ERROR = _exc

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


# 27 base LIV2 param fields（顺序与 lightning_indexer_v2_golden.liv2_output_single 一致）
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

# ---- 模型场景固定规格（与 LIV2 模型套一致）----
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
    }
    missing = [f for f in BASE_FIELDS if f not in params]
    if missing:
        raise ValueError(f"case '{name}' missing fields: {missing}")
    return name, params


# ---- 用例矩阵：与 LIV2 模型套一一配对（cand_spec="self"）----
_CASES = [
    # (a) 主链路 BSND×BSND FP16：N1 = 32 / 16 / 8 / 4
    _model_case("SLI_MODEL_N1_32_MODE3", 1, 32, 3, "FP16", cmp_residual_k=[1]),
    _model_case("SLI_MODEL_N1_16_MODE3", 2, 16, 3, "FP16", cmp_residual_k=[1, 1]),
    _model_case("SLI_MODEL_N1_8_MODE3", 2, 8, 3, "FP16", cmp_residual_k=[1, 1]),
    _model_case("SLI_MODEL_N1_4_MODE3", 4, 4, 3, "FP16", cmp_residual_k=[1, 1, 1, 1]),
    # (b) sparse_mode=0（no mask）：N1 泛化
    _model_case("SLI_MODEL_N1_32_MODE0", 1, 32, 0, "FP16"),
    _model_case("SLI_MODEL_N1_4_MODE0", 2, 4, 0, "FP16"),
    # (c) dtype 泛化：BF16
    _model_case("SLI_MODEL_N1_32_MODE3_BF16", 1, 32, 3, "BF16", cmp_residual_k=[1]),
    _model_case("SLI_MODEL_N1_8_MODE0_BF16", 1, 8, 0, "BF16"),
    # (d) 变长泛化：B>1 seqused_k 变长（2049 % candidate_block_size != 0，S2 非对齐）
    _model_case(
        "SLI_MODEL_VARLEN_MODE3",
        2,
        8,
        3,
        "FP16",
        seqused_k=[4096, 2049],
        cmp_residual_k=[1, 1],
    ),
    # (e) key 侧泛化：BSND q × PA_BBND k（block_table 链路）
    _model_case(
        "SLI_MODEL_PA_MODE3",
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

# 16K 压缩 K 边界用例（numBlocks = 2048 == candBlocks，满覆盖边界）。
# 较重：SLI_MODEL_LARGE=1 开启。
_LARGE_CASES = [
    _model_case(
        "SLI_MODEL_S2_16K_N1_4_MODE3", 1, 4, 3, "FP16", k_seq=16384, cmp_residual_k=[1]
    ),
]

_CASES = _CASES + (
    _LARGE_CASES if os.environ.get("SLI_MODEL_LARGE", "0") == "1" else []
)
_CASE_MAP = dict(_CASES)

requested_names = {
    name.strip()
    for name in os.environ.get("SLI_CASE_NAMES", "").split(",")
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
        raise ValueError(f"unknown sparse model case(s): {unknown}")
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
    if sli_golden is None:
        return (
            f"sparse_lightning_indexer_golden import failed: {_GOLDEN_IMPORT_ERROR!r}"
        )
    # The pure golden helpers (numpy/torch only) must be importable; the heavy LIV2 golden infra
    # (cann_ops_transformer) is loaded lazily by the NPU driver and surfaces as SparseOpUnavailableError.
    if sli_golden.cand_golden is None or sli_golden.rc is None:
        return (
            "LIV2 pure golden helpers (liv2_candidate_golden / result_compare_method) import "
            f"failed: {sli_golden._LIV2_IMPORT_ERROR!r}"
        )
    return None


_SKIP_REASON = _collect_skip_reason()


@pytest.mark.ci
@pytest.mark.parametrize(
    "case", CASES if CASES else [pytest.param(None, id="no_cases")]
)
def test_sparse_lightning_indexer_model(case):
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    if case is None:
        pytest.skip("no model cases enabled (SLI_MODEL_LARGE=1 enables the 16K case)")

    arch = _device_arch()
    if arch == "arch35":
        pytest.skip(
            "sparse_lightning_indexer is arch22-only (Ascend910B / Ascend910_93)"
        )

    test_data = tuple(case[f] for f in BASE_FIELDS)
    cand_spec = "self"
    cand_blocks = MODEL_CAND_BLOCKS
    block_size = MODEL_CAND_BLOCK_SIZE

    print(
        f"[case] B={case['batch_size']} N1={case['q_head_num']} S1={case['q_seq']} "
        f"S2={case['k_seq']} topk={case['topk']} candBlocks={cand_blocks} "
        f"bs={block_size} mask_mode={case['mask_mode']} dtype={case['qk_dtype']} "
        f"layout_k={case['layout_k']} cand_spec={cand_spec} arch={arch}"
    )

    try:
        out = sli_golden.sparse_output_single(
            test_data, cand_spec, cand_blocks, block_size
        )
    except sli_golden.SparseOpUnavailableError as exc:
        pytest.skip(f"sparse torch op not available: {exc}")

    # ---- sparse_indices vs golden（compare_topk_valid 官方规则 + -1 槽硬校验）----
    rc = sli_golden.rc
    sparse_result, sparse_pct = rc.check_result(
        out["cpu_result"],
        out["npu_result"],
        out["cpu_degraded"],  # 边界仲裁用降级 score'（golden 口径）
        case["output_idx_offset"],
        test_data,
        None,  # return_value 恒 0，无 value 比对
        out["npu_values"],
    )
    print(f"sparse_indices result={sparse_result} fulfill={sparse_pct}")
    assert sparse_result == "Pass", "sparse_indices mismatch vs consumer golden"
