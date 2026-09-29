# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


PARAM_NAMES = (
    "batch_size",
    "query_length",
    "sequence_length",
    "cmp_ratio",
    "storage_mode",
    "mask_mode",
    "topk",
    "candidate_block_length",
    "output_idx_offset",
    "seed",
)


def normalize_qsli_params(params):
    if isinstance(params, dict):
        values = dict(params)
    else:
        if len(params) != len(PARAM_NAMES):
            raise ValueError(
                f"QSLI parameter count mismatch: got {len(params)}, "
                f"expected {len(PARAM_NAMES)}"
            )
        values = dict(zip(PARAM_NAMES, params))
    q_seq = values.get("q_seq", values.get("query_length"))
    k_seq = values.get("k_seq", values.get("sequence_length"))
    if q_seq is None or k_seq is None:
        raise ValueError("QSLI requires q_seq/k_seq (or query_length/sequence_length)")
    k_t_size = values.get("k_t_size")
    normalized = {
        "batch_size": int(values["batch_size"]),
        "query_length": int(q_seq),
        "sequence_length": int(k_seq),
        "q_t_size": int(values.get("q_t_size", int(values["batch_size"]) * int(q_seq))),
        "k_t_size": int(k_seq if k_t_size is None else k_t_size),
        "q_head_num": int(values.get("q_head_num", 32)),
        "k_head_num": int(values.get("k_head_num", 1)),
        "head_dim": int(values.get("head_dim", 128)),
        "block_size": int(values.get("block_size", 128)),
        "block_num": int(values.get("block_num", (int(k_seq) + 127) // 128)),
        "cmp_ratio": int(values.get("cmp_ratio", 1)),
        "storage_mode": str(
            values.get("storage_layout", values.get("storage_mode", "contiguous"))
        ),
        "mask_mode": int(values.get("sparse_mode", values.get("mask_mode", 3))),
        "topk": int(values.get("sparse_count", values.get("topk", 512))),
        "candidate_block_indices": values.get("candidate_block_indices", "AUTO"),
        "candidate_block_length": values.get("candidate_block_length"),
        "output_idx_offset": values.get("output_idx_offset", True),
        "seed": int(values.get("seed", 1)),
        "cu_seqlens_q": values.get("cu_seqlens_q", "AUTO"),
        "cu_seqlens_k": values.get("cu_seqlens_k"),
        "seqused_q": values.get("seqused_q", "AUTO"),
        "seqused_k": values.get("seqused_k", "AUTO"),
        "cmp_residual_k": values.get("cmp_residual_k"),
        "block_table": values.get("block_table", "AUTO"),
        "quant_mode": int(values.get("quant_mode", 1)),
        "candidate_block_size": int(values.get("candidate_block_size", 8)),
        "max_seqlen_q": int(values.get("max_seqlen_q", q_seq)),
        "layout_q": str(values.get("layout_query", values.get("layout_q", "TND"))),
        "layout_k": str(values.get("layout_key", values.get("layout_k", "PA_BBND"))),
        "return_value": bool(values.get("return_value", True)),
        "qk_dtype": str(values.get("qk_dtype", "FLOAT4_E2M1")),
        "weight_dtype": str(values.get("weight_dtype", "FP32")),
        "dequant_dtype": str(values.get("dequant_dtype", "FLOAT8_E8M0")),
        "actual_seq_dtype": str(values.get("actual_seq_dtype", "INT32")),
        "query_datarange": values.get("query_datarange", [-6, 6]),
        "key_datarange": values.get("key_datarange", [-6, 6]),
        "weights_datarange": values.get("weights_datarange", [0, 1]),
        "q_scale_datarange": values.get("q_scale_datarange", [0.25, 4]),
        "k_scale_datarange": values.get("k_scale_datarange", [0.25, 4]),
    }
    return normalized
