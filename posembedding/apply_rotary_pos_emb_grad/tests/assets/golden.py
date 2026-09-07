#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""apply_rotary_pos_emb_grad 的 TTK 多路径 TestSpec golden。

RoPE half 模式反向公式 (D 末维对半 chunk, 下标 1/2 为前/后两半):
    grad_query = cat(cos_1*gq_1 + sin_2*gq_2, cos_2*gq_2 - sin_1*gq_1, dim=-1)
    grad_key   = cat(cos_1*gk_1 + sin_2*gk_2, cos_2*gk_2 - sin_1*gk_1, dim=-1)
    query_rotate = cat(-q_2, q_1, dim=-1); key_rotate = cat(-k_2, k_1, dim=-1)
    grad_cos = sum(gqe*query + gke*key, broadcast_dims)
    grad_sin = sum(gqe*query_rotate + gke*key_rotate, broadcast_dims)

broadcast_dims = cos 中为 1 而 query/key 中大于 1 的轴 (N 轴, 及 BSND/SBND 下可选的 B 轴),
q/k 两路分别归约以兼容 Nq != Nk; query/key 缺失时不计算 grad_cos/grad_sin
(golden 输出 None, 比对置 SUPPRESSED)。所有路径统一升 FP32 计算, 结果转回输入 dtype。
"""

__spec__ = {
    "apply_rotary_pos_emb_grad": "ApplyRotaryPosEmbGradKernelSpec",
    "aclnnApplyRotaryPosEmbGrad": "AclnnApplyRotaryPosEmbGradSpec",
    "cann_ops_transformer.apply_rotary_pos_emb_grad": "ApplyRotaryPosEmbGradE2ESpec",
    "torch.ops.cann_ops_transformer.apply_rotary_pos_emb_grad": "ApplyRotaryPosEmbGradE2ESpec",
}

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None


def _to_np(t):
    """Normalize numpy / torch (CPU or NPU) input to a numpy ndarray (or None)."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    import torch

    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        # torch CPU half/bfloat16 have no stable numpy() view; lift to fp32.
        if t.dtype in (torch.float16, torch.bfloat16):
            return t.float().numpy()
        return t.numpy()
    return np.asarray(t)


def _acl_output_target(dtype):
    """Map the input dtype to the output cast target."""
    dt = str(dtype)
    if "bfloat16" in dt:
        return "bfloat16"
    if "float16" in dt:
        return "float16"
    return None  # float32 -> keep fp32


def _cast_out(y, target):
    """Cast an fp32 result back to the input dtype (bf16 via ml_dtypes)."""
    if y is None:
        return None
    if target == "bfloat16":
        return y.astype(_bf16) if _bf16 is not None else y
    if target == "float16":
        return y.astype(np.float16)
    return y


def _get_broadcast_dims(small_shape, big_shape):
    """cos/sin 广播到 query/key 时需要 reduce 求和的维度索引 (N 维广播场景)。"""
    dims = []
    s_len = len(small_shape)
    b_len = len(big_shape)
    for i in range(1, b_len + 1):
        b_size = big_shape[-i]
        s_size = small_shape[-i] if i <= s_len else 1
        if s_size == 1 and b_size > 1:
            dims.append(b_len - i)
    return sorted(dims)


def _check_attrs(rotary_mode, layout):
    if rotary_mode != "half":
        raise ValueError(f"golden 仅支持 half 模式, 收到 rotary_mode={rotary_mode!r}")
    if int(layout) not in (1, 2, 4):
        raise ValueError(f"非法 layout={layout!r} (1=BSND, 2=SBND, 4=TND)")


def _ref_rope_grad_half(gqe, gke, cos, sin, query=None, key=None, target=None):
    """float32 CPU reference shared by all paths.

    gqe/gke: grad_query_embed / grad_key_embed. query/key 任一缺失时
    grad_cos/grad_sin 返回 None (比对置 SUPPRESSED)。
    """

    def _f32(a):
        a = np.asarray(a)
        return a if a.dtype == np.float32 else a.astype(np.float32)

    gqe, gke, cos, sin = _f32(gqe), _f32(gke), _f32(cos), _f32(sin)

    gq_1, gq_2 = np.split(gqe, 2, axis=-1)
    gk_1, gk_2 = np.split(gke, 2, axis=-1)
    c_1, c_2 = np.split(cos, 2, axis=-1)
    s_1, s_2 = np.split(sin, 2, axis=-1)

    grad_query = np.concatenate(
        [c_1 * gq_1 + s_2 * gq_2, c_2 * gq_2 - s_1 * gq_1], axis=-1
    )
    grad_key = np.concatenate(
        [c_1 * gk_1 + s_2 * gk_2, c_2 * gk_2 - s_1 * gk_1], axis=-1
    )

    grad_cos = grad_sin = None
    if query is not None and key is not None:
        query, key = _f32(query), _f32(key)
        q_1, q_2 = np.split(query, 2, axis=-1)
        k_1, k_2 = np.split(key, 2, axis=-1)
        query_rotate = np.concatenate([-q_2, q_1], axis=-1)
        key_rotate = np.concatenate([-k_2, k_1], axis=-1)

        # cos/sin 在 N (及可选 B) 维度广播, 分路归约以兼容 Nq != Nk 场景
        dims_q = _get_broadcast_dims(cos.shape, query.shape)
        dims_k = _get_broadcast_dims(cos.shape, key.shape)

        d_cos_q, d_sin_q = gqe * query, gqe * query_rotate
        d_cos_k, d_sin_k = gke * key, gke * key_rotate
        if dims_q:
            d_cos_q = d_cos_q.sum(axis=tuple(dims_q), keepdims=True)
            d_sin_q = d_sin_q.sum(axis=tuple(dims_q), keepdims=True)
        if dims_k:
            d_cos_k = d_cos_k.sum(axis=tuple(dims_k), keepdims=True)
            d_sin_k = d_sin_k.sum(axis=tuple(dims_k), keepdims=True)

        grad_cos = (d_cos_q + d_cos_k).reshape(cos.shape)
        grad_sin = (d_sin_q + d_sin_k).reshape(sin.shape)

    return [_cast_out(y, target) for y in (grad_query, grad_key, grad_cos, grad_sin)]


# ---- Kernel / GEIR（numpy.ndarray）-------------------------------------------------
# GEIR 复用 Kernel 的注册（op_name=apply_rotary_pos_emb_grad）；TTK 中 GEIR golden 也走此 golden。
class ApplyRotaryPosEmbGradKernelSpec:
    """apply_rotary_pos_emb_grad 的 Kernel / GEIR 流程 golden（输入为 numpy.ndarray）。
    def.cpp inputs : grad_query_embed, grad_key_embed, cos, sin, query, key
    def.cpp outputs: grad_query, grad_key, grad_cos, grad_sin
    """

    def golden(*input_arrays, **kwargs):
        gqe = _to_np(input_arrays[0])
        gke = _to_np(input_arrays[1])
        cos = _to_np(input_arrays[2])
        sin = _to_np(input_arrays[3])
        query = _to_np(input_arrays[4]) if len(input_arrays) > 4 else None
        key = _to_np(input_arrays[5]) if len(input_arrays) > 5 else None

        _check_attrs(kwargs.get("rotary_mode", "half"), kwargs.get("layout", 1))

        output_dtypes = kwargs.get("output_dtypes")
        if output_dtypes is not None and len(output_dtypes) > 0:
            target = _acl_output_target(output_dtypes[0])
        else:
            target = _acl_output_target(gqe.dtype)

        return _ref_rope_grad_half(gqe, gke, cos, sin, query, key, target=target)


# ---- ACLNN : aclnnApplyRotaryPosEmbGrad(gradQueryEmbed, gradKeyEmbed, cos, sin, queryOptional, keyOptional, rotaryModeOptional, layout, gradQueryOut, ...) ----
class AclnnApplyRotaryPosEmbGradSpec:
    """aclnnApplyRotaryPosEmbGrad 的 ACLNN 流程 golden（输入为 torch.Tensor）。
    header inputs : gradQueryEmbed, gradKeyEmbed, cos, sin, queryOptional, keyOptional
    header attrs  : rotaryModeOptional, layout
    header outputs: gradQueryOut, gradKeyOut, gradCosOut, gradSinOut (占位, golden 不消费)
    """

    def golden(
        gradQueryEmbed,
        gradKeyEmbed,
        cos,
        sin,
        queryOptional=None,
        keyOptional=None,
        rotaryModeOptional="half",
        layout=1,
        gradQueryOut=None,
        gradKeyOut=None,
        gradCosOut=None,
        gradSinOut=None,
        **kwargs,
    ):
        _check_attrs(rotaryModeOptional, layout)
        target = _acl_output_target(gradQueryEmbed.dtype)
        return _ref_rope_grad_half(
            _to_np(gradQueryEmbed),
            _to_np(gradKeyEmbed),
            _to_np(cos),
            _to_np(sin),
            _to_np(queryOptional),
            _to_np(keyOptional),
            target=target,
        )


# ---- E2E : cann_ops_transformer.apply_rotary_pos_emb_grad(grad_query_embed, ..., query=None, key=None, rotary_mode="half", layout=1) ----
class ApplyRotaryPosEmbGradE2ESpec:
    """cann_ops_transformer.apply_rotary_pos_emb_grad 的 E2E 流程 golden（输入为 torch.Tensor）。
    API inputs : grad_query_embed, grad_key_embed, cos, sin, query, key
    API outputs: grad_query, grad_key, grad_cos, grad_sin (returned, not passed in)
    """

    def golden(
        grad_query_embed,
        grad_key_embed,
        cos,
        sin,
        query=None,
        key=None,
        rotary_mode="half",
        layout=1,
        **kwargs,
    ):
        _check_attrs(rotary_mode, layout)
        target = _acl_output_target(grad_query_embed.dtype)
        return _ref_rope_grad_half(
            _to_np(grad_query_embed),
            _to_np(grad_key_embed),
            _to_np(cos),
            _to_np(sin),
            _to_np(query),
            _to_np(key),
            target=target,
        )
