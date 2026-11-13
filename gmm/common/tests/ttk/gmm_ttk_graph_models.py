#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Importable TorchAir models for the GMM family TestSpecs."""

import torch


class GroupedMatmulGraph(torch.nn.Module):
    preserve_input_views = True

    def __init__(self):
        super().__init__()
        import torch_npu

        self._api = torch_npu.npu_grouped_matmul

    def forward(self, **kwargs):
        return self._api(**kwargs)


class InplaceGraph(torch.nn.Module):
    """Trace the documented x1 and MX scale transposes inside the graph."""

    def __init__(self):
        super().__init__()
        import torch_npu

        self._api = torch_npu.npu_add_quant_gmm_

    def forward(self, input, x1, x2, x2_scale, group_list, x1_scale=None, **kwargs):
        if kwargs.get("x1_scale_dtype") is not None:
            x1_scale = x1_scale.transpose(0, 1)
        return self._api(
            input,
            x1.transpose(0, 1),
            x2,
            x2_scale,
            group_list,
            x1_scale=x1_scale,
            **kwargs,
        )


class SwigluGraph(torch.nn.Module):
    preserve_input_views = True

    def __init__(self):
        super().__init__()
        import torch_npu

        self._api = torch_npu.npu_grouped_matmul_swiglu_quant_v2
        self._fp4_e2m1 = torch_npu.float4_e2m1fn_x2

    def forward(self, x, weight, weight_scale, x_scale, group_list, **kwargs):
        if (
            x.dtype == torch.float8_e4m3fn
            and kwargs.get("weight_dtype") == self._fp4_e2m1
        ):
            weight = [value.transpose(-1, -2) for value in weight]
            weight_scale = [value.transpose(-3, -2) for value in weight_scale]
        return self._api(x, weight, weight_scale, x_scale, group_list, **kwargs)


class RoutingGraph(torch.nn.Module):
    preserve_input_views = True

    def __init__(self):
        super().__init__()
        import torch_npu

        self._api = torch_npu.npu_grouped_matmul_finalize_routing
        self._fp4_e2m1 = torch_npu.float4_e2m1fn_x2

    def forward(self, x, w, group_list, *, scale=None, **kwargs):
        if x.dtype == torch.float8_e4m3fn and kwargs.get("w_dtype") == self._fp4_e2m1:
            w = w.transpose(-1, -2)
            scale = scale.transpose(-3, -2)
        return self._api(x, w, group_list, scale=scale, **kwargs)
