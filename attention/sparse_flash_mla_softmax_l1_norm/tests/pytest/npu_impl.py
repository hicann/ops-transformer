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

"""NPU invocation matching the operator's documented PyTorch API."""

import torch

from common import CalculationContext, DEVICE_ID


def _load_operator():
    # Import torch_npu first so PrivateUse1 is registered before the custom
    # extension is loaded.  The installed custom package exports the operator
    # from its top-level namespace.
    import torch_npu  # noqa: F401
    import cann_ops_transformer

    return cann_ops_transformer.sparse_flash_mla_softmax_l1_norm


def PTAtest(ctx: CalculationContext) -> torch.Tensor:
    op = _load_operator()
    data = ctx.inputs
    if data is None:
        raise RuntimeError("CPU input generation must run before PTAtest")
    device = torch.device(f"npu:{DEVICE_ID}")

    def to_device(value):
        return None if value is None else value.to(device)

    q = data["q"].to(device)
    k = data["k"].to(device)
    lse = data["softmax_lse"].to(device)
    metadata = data["metadata"].to(device)
    result = op(
        q,
        k,
        lse,
        sparse_indices=to_device(data["sparse_indices"]),
        cu_seqlens_q=to_device(data["cu_seqlens_q"]),
        cu_seqlens_k=to_device(data["cu_seqlens_k"]),
        seqused_q=to_device(data["seqused_q"]),
        seqused_k=to_device(data["seqused_k"]),
        cmp_residual_k=to_device(data["cmp_residual_k"]),
        topk_length=to_device(data["topk_length"]),
        metadata=metadata,
        softmax_scale=data["softmax_scale"],
        max_seqlen_k=data["max_seqlen_k"],
        cmp_ratio=data["cmp_ratio"],
        mask_mode=data["mask_mode"],
        layout_q=data["layout_q"],
        layout_k=data["layout_k"],
    )
    torch.npu.synchronize()
    return result.cpu()
