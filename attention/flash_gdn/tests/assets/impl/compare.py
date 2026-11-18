# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Return protocol checks shared by pytest and TTK; no numerical fallback."""

import torch
from gdn_ttk_ops import load_asset


def shapes(inputs, attributes):
    c = load_asset("inputs").configuration(inputs, attributes)
    nc = (c["storage"] + 63) // 64
    state = (c["batch"], c["hv"], c["dv"], c["dk"])
    chunks = (
        (nc + c["batch"],) + state[1:]
        if c["layout"] == "TND"
        else (c["batch"], nc) + state[1:]
    )
    return (tuple(inputs[2].shape), state, chunks)


def check(result, inputs, attributes, *, entry="flash_gdn_fwd"):
    expected_count = 16 if entry == "flash_gdn_fwd" else 3
    if len(result) != expected_count:
        raise AssertionError(f"{entry} must return {expected_count} slots")
    f = load_asset("inputs").effective(attributes)
    enabled = (True, f["output_final_state"], f["return_intermediate_states"])
    for i, (value, shape, on) in enumerate(
        zip(result[:3], shapes(inputs, attributes), enabled)
    ):
        if not on:
            if value is not None:
                raise AssertionError(f"disabled output slot {i} must be None")
            continue
        dtype = inputs[2].dtype if i == 0 else torch.float32
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != inputs[2].device
        ):
            raise AssertionError(
                f"slot {i}: expected {shape}/{dtype}/{inputs[2].device}"
            )
    if any(value is not None for value in result[3:]):
        raise AssertionError("inference training-cache slots must be None")


def pack(result):
    """TTK compares enabled tensors only, retaining their public order/dtype."""
    return [value for value in result[:3] if value is not None]
