#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""CPU input generator and FP64 reference for pytest validation."""

import math
from typing import Any, Dict, Tuple

import torch

from common import BASE_SEED, CalculationContext, input_dtype


def _lengths(case: Dict[str, Any]) -> Tuple[list[int], list[int], list[int], list[int]]:
    if case["layout"] == "TND":
        q_cap = list(case["q_lengths"])
        k_cap = list(case["k_lengths"])
    else:
        q_cap = [case["S1"]] * case["B"]
        k_cap = [case["S2"]] * case["B"]
    q_len = list(q_cap)
    k_len = list(k_cap)
    if case.get("seqused", False):
        q_len = [
            max(1, value - (index % min(value, 4))) for index, value in enumerate(q_cap)
        ]
        k_len = [
            max(1, value - ((index + 1) % min(value, 5)))
            for index, value in enumerate(k_cap)
        ]
    return q_cap, k_cap, q_len, k_len


def _causal_length(
    q_index: int, q_length: int, k_length: int, ratio: int, residual: int
) -> int:
    original_k = k_length * ratio + (residual if ratio > 1 else 0)
    return min(k_length, max((original_k - q_length + q_index + 1) // ratio, 0))


def _row_shapes(case: Dict[str, Any], q_cap: list[int], k_cap: list[int]):
    if case["layout"] == "BSND":
        q_shape = (case["B"], case["S1"], case["N1"], case["D"])
        k_shape = (case["B"], case["S2"], case["N2"], case["D"])
        out_shape = (case["B"], case["S1"], case["N2"], case["K"])
        lse_shape = (case["B"], case["N2"], case["S1"], case["N1"] // case["N2"])
    else:
        t1, t2 = sum(q_cap), sum(k_cap)
        q_shape = (t1, case["N1"], case["D"])
        k_shape = (t2, case["N2"], case["D"])
        out_shape = (t1, case["N2"], case["K"])
        lse_shape = (case["N2"], t1, case["N1"] // case["N2"])
    return q_shape, k_shape, lse_shape, out_shape


def build_inputs(case: Dict[str, Any]) -> Dict[str, Any]:
    q_cap, k_cap, q_len, k_len = _lengths(case)
    q_shape, k_shape, lse_shape, out_shape = _row_shapes(case, q_cap, k_cap)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BASE_SEED + sum(ord(c) for c in case["name"]))
    dtype = input_dtype(case["dtype"])
    q = (torch.randn(q_shape, generator=generator, dtype=torch.float32) * 0.1).to(dtype)
    k = (torch.randn(k_shape, generator=generator, dtype=torch.float32) * 0.1).to(dtype)
    lse = torch.full(lse_shape, -torch.inf, dtype=torch.float32)
    sparse_indices = (
        torch.zeros(out_shape, dtype=torch.int32) if case["sparse"] else None
    )
    topk_shape = out_shape[:-1]
    topk_length = (
        torch.zeros(topk_shape, dtype=torch.int32)
        if case["sparse"] and case["mask_mode"] == 0
        else None
    )
    residual = torch.tensor(
        [
            (index + 1) % case["cmp_ratio"] if case["cmp_ratio"] > 1 else 0
            for index in range(case["B"])
        ],
        dtype=torch.int32,
    )

    cuq = [0]
    cuk = [0]
    for value in q_cap:
        cuq.append(cuq[-1] + value)
    for value in k_cap:
        cuk.append(cuk[-1] + value)

    def process_row(batch: int, q_rel: int, q_index: int, k_base: int):
        valid_k = k_len[batch]
        if case["mask_mode"] == 3:
            valid = _causal_length(
                q_rel, q_len[batch], valid_k, case["cmp_ratio"], int(residual[batch])
            )
        else:
            valid = valid_k
        if case["sparse"]:
            valid = min(valid, case["K"])
            row_indices = (
                torch.arange(case["K"], dtype=torch.int32) * 7 + batch + q_rel * 3
            ) % max(valid_k, 1)
            selected = row_indices[:valid].to(torch.int64)
            if case["layout"] == "BSND":
                sparse_indices[batch, q_rel, 0] = row_indices
                if topk_length is not None:
                    topk_length[batch, q_rel, 0] = valid
            else:
                sparse_indices[q_index, 0] = row_indices
                if topk_length is not None:
                    topk_length[q_index, 0] = valid
        else:
            selected = torch.arange(valid, dtype=torch.int64)
        if valid == 0:
            return
        q_row = q[batch, q_rel] if case["layout"] == "BSND" else q[q_index]
        if case["layout"] == "BSND":
            key_rows = k[batch, selected, 0]
        else:
            key_rows = k[k_base + selected, 0]
        scores = torch.matmul(
            q_row.float(), key_rows.float().transpose(0, 1)
        ) / math.sqrt(case["D"])
        row_lse = torch.logsumexp(scores, dim=-1)
        if case["layout"] == "BSND":
            lse[batch, 0, q_rel] = row_lse
        else:
            lse[0, q_index] = row_lse

    for batch in range(case["B"]):
        for q_rel in range(q_cap[batch]):
            if q_rel >= q_len[batch]:
                continue
            q_index = q_rel if case["layout"] == "BSND" else cuq[batch] + q_rel
            process_row(batch, q_rel, q_index, cuk[batch])

    return {
        "q": q,
        "k": k,
        "softmax_lse": lse,
        "sparse_indices": sparse_indices,
        "cu_seqlens_q": torch.tensor(cuq, dtype=torch.int32)
        if case["layout"] == "TND"
        else None,
        "cu_seqlens_k": torch.tensor(cuk, dtype=torch.int32)
        if case["layout"] == "TND"
        else None,
        "seqused_q": torch.tensor(q_len, dtype=torch.int32)
        if case.get("seqused", False)
        else None,
        "seqused_k": torch.tensor(k_len, dtype=torch.int32)
        if case.get("seqused", False)
        else None,
        "cmp_residual_k": residual
        if case["mask_mode"] == 3 and case["cmp_ratio"] > 1
        else None,
        "topk_length": topk_length,
        "metadata": torch.zeros(64, dtype=torch.int32),
        "softmax_scale": 1.0 / math.sqrt(case["D"]),
        "max_seqlen_k": max(k_cap)
        if case["layout"] == "TND" and not case["sparse"]
        else 0,
        "cmp_ratio": case["cmp_ratio"],
        "mask_mode": case["mask_mode"],
        "layout_q": case["layout"],
        "layout_k": case["layout"],
        "out_shape": out_shape,
    }


def cpu_impl(ctx: CalculationContext) -> torch.Tensor:
    case = ctx.input_case
    data = ctx.inputs if ctx.inputs is not None else build_inputs(case)
    q_cap, k_cap, q_len, k_len = _lengths(case)
    output = torch.zeros(data["out_shape"], dtype=torch.float32)
    residual = data["cmp_residual_k"]
    if residual is None:
        residual = torch.zeros(case["B"], dtype=torch.int32)
    cuq = (
        data["cu_seqlens_q"]
        if data["cu_seqlens_q"] is not None
        else torch.tensor([0] + [case["S1"] * (i + 1) for i in range(case["B"])])
    )
    cuk = (
        data["cu_seqlens_k"]
        if data["cu_seqlens_k"] is not None
        else torch.tensor([0] + [case["S2"] * (i + 1) for i in range(case["B"])])
    )
    for batch in range(case["B"]):
        for q_rel in range(q_cap[batch]):
            if q_rel >= q_len[batch]:
                continue
            q_index = q_rel if case["layout"] == "BSND" else int(cuq[batch]) + q_rel
            valid_k = k_len[batch]
            if case["mask_mode"] == 3:
                valid = _causal_length(
                    q_rel,
                    q_len[batch],
                    valid_k,
                    case["cmp_ratio"],
                    int(residual[batch]),
                )
            else:
                valid = valid_k
            if case["sparse"]:
                indices = (
                    data["sparse_indices"][batch, q_rel, 0]
                    if case["layout"] == "BSND"
                    else data["sparse_indices"][q_index, 0]
                )
                valid = min(valid, case["K"])
                selected = indices[:valid].to(torch.int64)
            else:
                selected = torch.arange(valid, dtype=torch.int64)
            if valid == 0:
                continue
            q_row = (
                data["q"][batch, q_rel]
                if case["layout"] == "BSND"
                else data["q"][q_index]
            )
            key_rows = (
                data["k"][batch, selected, 0]
                if case["layout"] == "BSND"
                else data["k"][int(cuk[batch]) + selected, 0]
            )
            scores = (
                torch.matmul(q_row.double(), key_rows.double().transpose(0, 1))
                * data["softmax_scale"]
            )
            row_lse = (
                data["softmax_lse"][batch, 0, q_rel]
                if case["layout"] == "BSND"
                else data["softmax_lse"][0, q_index]
            )
            probs = torch.exp(scores - row_lse.double()).mean(dim=0).float()
            if case["layout"] == "BSND":
                output[batch, q_rel, 0, :valid] = probs
            else:
                output[q_index, 0, :valid] = probs
    return output
