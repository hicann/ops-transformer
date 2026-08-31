#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""mhc_post 在 kernel/geir/aclnn/e2e 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 torch tensor 后用小算子拼接计算，结果转回 numpy；
ACLNN/E2E 的 golden 直接收到 torch.Tensor（已在设备上），无需转换。
MhcPost 无 torch 现成接口，golden 与 third_party 均按文档公式用 torch 小算子拼接：

    y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + sum_i(h_res[..., i, :].unsqueeze(-1) * x[..., i, :].unsqueeze(-2))
    nohres 变体：y = x + h_post.unsqueeze(-1) * h_out.unsqueeze(-2)

golden 统一在 float32 下计算后 cast 回 x 的 dtype；h_res 为可选输入（def.cpp OPTIONAL），
nohres 用例框架传 h_res=None。
"""

__spec__ = {
    "mhc_post": "MhcPostTestSpec",
    "aclnnMhcPost": "AclnnMhcPostTestSpec",
    "torch.ops.cann_ops_transformer.mhc_post": "TorchMhcPostTestSpec",
}

import numpy as np
import torch


def _mhc_post_compute(x_f32, h_res_f32, h_out_f32, h_post_f32):
    """Shared torch composition; all inputs are float32 tensors."""
    h_post_term = h_post_f32.unsqueeze(-1) * h_out_f32.unsqueeze(-2)
    if h_res_f32 is None:
        return x_f32 + h_post_term
    h_comb_term = h_post_term
    for i in range(h_res_f32.size(-2)):
        a = h_res_f32[..., i, :].unsqueeze(-1)
        b = x_f32[..., i, :].unsqueeze(-2)
        h_comb_term = torch.addcmul(h_comb_term, a, b, value=1.0)
    return h_comb_term


class MhcPostTestSpec:
    """MhcPost 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow mhc_post_def.cpp: x, h_res(optional), h_out, h_post;
    no attributes.
    """

    def golden(x, h_res=None, h_out=None, h_post=None, **kwargs):
        dtype = x.dtype
        x_f32 = torch.from_numpy(x.astype(np.float32))
        h_res_f32 = (
            None if h_res is None else torch.from_numpy(h_res.astype(np.float32))
        )
        h_out_f32 = torch.from_numpy(h_out.astype(np.float32))
        h_post_f32 = torch.from_numpy(h_post.astype(np.float32))
        y = _mhc_post_compute(x_f32, h_res_f32, h_out_f32, h_post_f32)
        return [y.numpy().astype(dtype)]

    class ThirdPartyImpl:
        # __init__ 只预提取输出 dtype；__call__ 只放纯计算（计入性能比对时间）
        def __init__(self, x, **kwargs):
            self.out_dtype = x.dtype

        def __call__(self, x, h_res=None, h_out=None, h_post=None, **kwargs):
            y = _mhc_post_compute(
                x.float(),
                None if h_res is None else h_res.float(),
                h_out.float(),
                h_post.float(),
            )
            return [y.to(self.out_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class AclnnMhcPostTestSpec:
    """MhcPost 测试规范（aclnn 流程，torch 入参，已在设备上）

    Parameters follow aclnnMhcPostGetWorkspaceSize (without workspaceSize &
    executor): x, hRes(optional), hOut, hPost, out.
    """

    def golden(x, hRes, hOut, hPost, out, **kwargs):
        y = _mhc_post_compute(
            x.float(),
            None if hRes is None else hRes.float(),
            hOut.float(),
            hPost.float(),
        )
        return [y.to(x.dtype)]

    class ThirdPartyImpl:
        def __init__(self, x, **kwargs):
            self.out_dtype = x.dtype

        def __call__(self, x, hRes, hOut, hPost, **kwargs):
            y = _mhc_post_compute(
                x.float(),
                None if hRes is None else hRes.float(),
                hOut.float(),
                hPost.float(),
            )
            return [y.to(self.out_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class TorchMhcPostTestSpec:
    """MhcPost 测试规范（e2e 流程，torch 入参，已在设备上）

    Parameters follow the torch dispatcher schema
    mhc_post(Tensor x, Tensor hRes, Tensor hOut, Tensor hPost) -> Tensor.
    """

    def golden(x, hRes, hOut, hPost, **kwargs):
        y = _mhc_post_compute(
            x.float(),
            None if hRes is None else hRes.float(),
            hOut.float(),
            hPost.float(),
        )
        return [y.to(x.dtype)]

    class ThirdPartyImpl:
        def __init__(self, x, **kwargs):
            self.out_dtype = x.dtype

        def __call__(self, x, hRes, hOut, hPost, **kwargs):
            y = _mhc_post_compute(
                x.float(),
                None if hRes is None else hRes.float(),
                hOut.float(),
                hPost.float(),
            )
            return [y.to(self.out_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
