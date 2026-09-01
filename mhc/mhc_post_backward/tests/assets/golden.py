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
"""mhc_post_backward 在 kernel/geir/aclnn/e2e 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 torch tensor 后用小算子拼接计算，结果转回 numpy；
ACLNN/E2E 的 golden 直接收到 torch.Tensor（已在设备上），无需转换。
MhcPostBackward 无 torch 现成接口，golden 按文档公式用 torch 小算子拼接：

    grad_x      = h_res @ grad_y
    grad_h_res  = x @ grad_y^T
    grad_h_out  = (grad_y * h_post.unsqueeze(-1)).sum(-2)
    grad_h_post = (grad_y * h_out.unsqueeze(-2)).sum(-1)

golden 统一在 float32 下计算后 cast 回输入 dtype；h_res 为可选输入（def.cpp REQUIRED），
nohres 用例传 h_res=None 或空 Tensor。
"""

__spec__ = {
    "mhc_post_backward": "MhcPostBackwardTestSpec",
    "aclnnMhcPostBackward": "AclnnMhcPostBackwardTestSpec",
    "torch.ops.cann_ops_transformer.mhc_post_backward": "TorchMhcPostBackwardTestSpec",
    "cann_ops_transformer.ops.mhc_post_backward": "TorchMhcPostBackwardTestSpec",
}

import numpy as np
import torch


def _mhc_post_backward_compute(grad_y_f32, x_f32, h_res_f32, h_out_f32, h_post_f32):
    """Shared torch composition; all inputs are float32 tensors."""
    if h_res_f32 is None or h_res_f32.numel() == 0:
        grad_x = grad_y_f32.clone()
    else:
        grad_x = torch.matmul(h_res_f32, grad_y_f32)

    grad_h_res = torch.matmul(x_f32, grad_y_f32.transpose(-1, -2))

    grad_h_out = (grad_y_f32 * h_post_f32.unsqueeze(-1)).sum(dim=-2)

    grad_h_post = (grad_y_f32 * h_out_f32.unsqueeze(-2)).sum(dim=-1)

    return grad_x, grad_h_res, grad_h_out, grad_h_post


def _backward_dtypes(grad_y):
    """Output dtypes for numpy path: grad_x/grad_h_out 与 grad_y 同 dtype，
    grad_h_res/grad_h_post 为 float32。grad_y 为 numpy.ndarray。"""
    out_dtype = grad_y.dtype
    f32 = np.float32
    return out_dtype, f32, out_dtype, f32


def _backward_dtypes_torch(grad_y):
    """Output dtypes for torch path. grad_y 为 torch.Tensor。"""
    out_dtype = grad_y.dtype
    f32 = torch.float32
    return out_dtype, f32, out_dtype, f32


class MhcPostBackwardTestSpec:
    """MhcPostBackward 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow mhc_post_backward_def.cpp: grad_y, x, h_res(optional),
    h_out, h_post; no attributes.
    """

    def golden(grad_y, x, h_res=None, h_out=None, h_post=None, **kwargs):
        grad_y_f32 = torch.from_numpy(grad_y.astype(np.float32))
        x_f32 = torch.from_numpy(x.astype(np.float32))
        h_res_f32 = (
            None if h_res is None else torch.from_numpy(h_res.astype(np.float32))
        )
        h_out_f32 = torch.from_numpy(h_out.astype(np.float32))
        h_post_f32 = torch.from_numpy(h_post.astype(np.float32))

        grad_x, grad_h_res, grad_h_out, grad_h_post = _mhc_post_backward_compute(
            grad_y_f32, x_f32, h_res_f32, h_out_f32, h_post_f32
        )
        dt_x, dt_hres, dt_hout, dt_hpost = _backward_dtypes(grad_y)
        return [
            grad_x.numpy().astype(dt_x),
            grad_h_res.numpy().astype(dt_hres),
            grad_h_out.numpy().astype(dt_hout),
            grad_h_post.numpy().astype(dt_hpost),
        ]

    class ThirdPartyImpl:
        # __init__ 只预提取输出 dtype；__call__ 只放纯计算（计入性能比对时间）
        def __init__(self, grad_y, **kwargs):
            self.out_dtype = grad_y.dtype

        def __call__(self, grad_y, x, h_res=None, h_out=None, h_post=None, **kwargs):
            grad_x, grad_h_res, grad_h_out, grad_h_post = _mhc_post_backward_compute(
                grad_y.float(),
                x.float(),
                None if h_res is None else h_res.float(),
                h_out.float(),
                h_post.float(),
            )
            dt_x, dt_hres, dt_hout, dt_hpost = _backward_dtypes_torch(grad_y)
            return [
                grad_x.to(dt_x),
                grad_h_res.to(dt_hres),
                grad_h_out.to(dt_hout),
                grad_h_post.to(dt_hpost),
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class AclnnMhcPostBackwardTestSpec:
    """MhcPostBackward 测试规范（aclnn 流程，torch 入参，已在设备上）

    Parameters follow aclnnMhcPostBackwardGetWorkspaceSize (without
    workspaceSize & executor): gradOutput, x, hRes(optional), hOut, hPost,
    gradX, gradHRes, gradHOut, gradHPost (outputs).
    """

    def golden(gradOutput, x, hRes, hOut, hPost, **kwargs):
        grad_x, grad_h_res, grad_h_out, grad_h_post = _mhc_post_backward_compute(
            gradOutput.float(),
            x.float(),
            None if hRes is None else hRes.float(),
            hOut.float(),
            hPost.float(),
        )
        dt_x, dt_hres, dt_hout, dt_hpost = _backward_dtypes_torch(gradOutput)
        return [
            grad_x.to(dt_x),
            grad_h_res.to(dt_hres),
            grad_h_out.to(dt_hout),
            grad_h_post.to(dt_hpost),
        ]

    class ThirdPartyImpl:
        def __init__(self, gradOutput, **kwargs):
            self.out_dtype = gradOutput.dtype

        def __call__(self, gradOutput, x, hRes, hOut, hPost, **kwargs):
            grad_x, grad_h_res, grad_h_out, grad_h_post = _mhc_post_backward_compute(
                gradOutput.float(),
                x.float(),
                None if hRes is None else hRes.float(),
                hOut.float(),
                hPost.float(),
            )
            dt_x, dt_hres, dt_hout, dt_hpost = _backward_dtypes_torch(gradOutput)
            return [
                grad_x.to(dt_x),
                grad_h_res.to(dt_hres),
                grad_h_out.to(dt_hout),
                grad_h_post.to(dt_hpost),
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class TorchMhcPostBackwardTestSpec:
    """MhcPostBackward 测试规范（e2e 流程，torch 入参，已在设备上）

    Parameters follow the torch dispatcher schema
    mhc_post_backward(Tensor gradOutput, Tensor x, Tensor? hRes, Tensor hOut,
    Tensor hPost) -> (Tensor, Tensor, Tensor, Tensor).
    """

    def golden(grad_output, x, h_res, h_out, h_post, **kwargs):
        grad_x, grad_h_res, grad_h_out, grad_h_post = _mhc_post_backward_compute(
            grad_output.float(),
            x.float(),
            None if h_res is None else h_res.float(),
            h_out.float(),
            h_post.float(),
        )
        dt_x, dt_hres, dt_hout, dt_hpost = _backward_dtypes_torch(grad_output)
        return [
            grad_x.to(dt_x),
            grad_h_res.to(dt_hres),
            grad_h_out.to(dt_hout),
            grad_h_post.to(dt_hpost),
        ]

    class ThirdPartyImpl:
        def __init__(self, grad_output, **kwargs):
            self.out_dtype = grad_output.dtype

        def __call__(self, grad_output, x, h_res, h_out, h_post, **kwargs):
            grad_x, grad_h_res, grad_h_out, grad_h_post = _mhc_post_backward_compute(
                grad_output.float(),
                x.float(),
                None if h_res is None else h_res.float(),
                h_out.float(),
                h_post.float(),
            )
            dt_x, dt_hres, dt_hout, dt_hpost = _backward_dtypes_torch(grad_output)
            return [
                grad_x.to(dt_x),
                grad_h_res.to(dt_hres),
                grad_h_out.to(dt_hout),
                grad_h_post.to(dt_hpost),
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
