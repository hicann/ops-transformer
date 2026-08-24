# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from typing import Optional, List

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class _MoeFinalizeRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeFinalizeRoutingOpBuilder, self).__init__(
            "moe_finalize_routing", category="moe"
        )

    def sources(self):
        return ["csrc/moe/moe_finalize_routing.cpp"]

    def include_paths(self):
        paths = super().include_paths()
        paths.append(
            os.path.abspath(
                os.path.join(
                    self._package_path,
                    "..",
                    "..",
                    "moe",
                    "moe_finalize_routing_v2",
                    "op_host",
                    "op_api",
                )
            )
        )
        return paths

    def schema(self) -> str:
        return (
            "moe_finalize_routing(Tensor expanded_x, "
            "Tensor expanded_row_idx, "
            "Tensor? x1, Tensor? x2, Tensor? bias, Tensor? scales, "
            "Tensor? expert_idx, Tensor? x, Tensor? alpha1, Tensor? alpha2, Tensor? v, "
            "int? drop_pad_mode=0, "
            "int[]? zero_expert_range=None, int[]? copy_expert_range=None, "
            "int[]? constant_expert_range=None, int? k=1) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_finalize_routing_meta(
            expanded_x,
            expanded_row_idx,
            x1=None,
            x2=None,
            bias=None,
            scales=None,
            expert_idx=None,
            x=None,
            alpha1=None,
            alpha2=None,
            v=None,
            drop_pad_mode=0,
            zero_expert_range=None,
            copy_expert_range=None,
            constant_expert_range=None,
            k=1,
        ):
            k_val = k if k is not None else 1
            mode = drop_pad_mode if drop_pad_mode is not None else 0
            dimm = expanded_row_idx.size(0)
            if scales is not None:
                dimm = scales.size(0)
            elif k_val > 0:
                dimm = dimm // k_val
            if mode == 1 or mode == 3:
                dimn = expanded_x.size(2)
            else:
                dimn = expanded_x.size(1)
            return expanded_x.new_empty((dimm, dimn))


_moe_finalize_routing_builder = _MoeFinalizeRoutingOpBuilder()
_moe_finalize_routing_builder._ensure_initialized()
_op_module = _moe_finalize_routing_builder.load()


@impl(get_as_library(), _moe_finalize_routing_builder.name, "PrivateUse1")
def _moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    x=None,
    alpha1=None,
    alpha2=None,
    v=None,
    drop_pad_mode=0,
    zero_expert_range=None,
    copy_expert_range=None,
    constant_expert_range=None,
    k=1,
):
    _op_module = _moe_finalize_routing_builder.load()
    return _op_module.moe_finalize_routing(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    )


def moe_finalize_routing(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    x: Optional[torch.Tensor] = None,
    alpha1: Optional[torch.Tensor] = None,
    alpha2: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    drop_pad_mode: Optional[int] = 0,
    zero_expert_range: Optional[List[int]] = None,
    copy_expert_range: Optional[List[int]] = None,
    constant_expert_range: Optional[List[int]] = None,
    k: Optional[int] = 1,
) -> torch.Tensor:
    return torch.ops.cann_ops_transformer.moe_finalize_routing(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
    )
