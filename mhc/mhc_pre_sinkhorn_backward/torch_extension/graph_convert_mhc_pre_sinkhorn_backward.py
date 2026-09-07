# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE converter for the decomposed MhcPreSinkhorn backward graph.

try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import DataType, Tensor, TensorSpec

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    def _mhc_sinkhorn(h_res: Tensor, hc_eps: float, num_iters: int):
        return ge_op(
            op_type="MhcSinkhorn",
            inputs={"h_res": h_res},
            attrs={
                "eps": attr.Float(hc_eps),
                "num_iters": attr.Int(num_iters),
                "out_flag": attr.Int(1),
            },
            outputs=["y", "norm_out", "sum_out"],
            ir=IrDef("MhcSinkhorn")
            .input("h_res", "DT_FLOAT")
            .attr("eps", attr.Float(1e-6))
            .attr("num_iters", attr.Int(20))
            .attr("out_flag", attr.Int(0))
            .output("y", "DT_FLOAT")
            .output("norm_out", "DT_FLOAT")
            .output("sum_out", "DT_FLOAT"),
        )

    def _mhc_sinkhorn_backward(grad_h_res: Tensor, norm_out: Tensor, sum_out: Tensor):
        return ge_op(
            op_type="MhcSinkhornBackward",
            inputs={"grad_y": grad_h_res, "norm": norm_out, "sum": sum_out},
            outputs=["grad_input"],
            ir=IrDef("MhcSinkhornBackward")
            .input("grad_y", "DT_FLOAT")
            .input("norm", "DT_FLOAT")
            .input("sum", "DT_FLOAT")
            .output("grad_input", "DT_FLOAT"),
        )

    def _mhc_pre_backward(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        grad_hin: Tensor,
        grad_h_post: Tensor,
        grad_h_res: Tensor,
        inv_rms: Tensor,
        h_mix: Tensor,
        h_pre: Tensor,
        h_post: Tensor,
        hc_eps: float,
    ):
        return ge_op(
            op_type="MhcPreBackward",
            inputs={
                "x": x,
                "phi": phi,
                "alpha": alpha,
                "grad_h_in": grad_hin,
                "grad_h_post": grad_h_post,
                "grad_h_res": grad_h_res,
                "inv_rms": inv_rms,
                "h_mix": h_mix,
                "h_pre": h_pre,
                "h_post": h_post,
                "gamma": None,
                "grad_x_post": None,
            },
            attrs={"hc_eps": attr.Float(hc_eps)},
            outputs=["grad_x", "grad_phi", "grad_alpha", "grad_bias"],
            ir=IrDef("MhcPreBackward")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("grad_h_in", "DT_BF16, DT_FLOAT16")
            .input("grad_h_post", "DT_FLOAT")
            .input("grad_h_res", "DT_FLOAT")
            .input("inv_rms", "DT_FLOAT")
            .input("h_mix", "DT_FLOAT")
            .input("h_pre", "DT_FLOAT")
            .input("h_post", "DT_FLOAT")
            .optional_input("gamma", "DT_FLOAT")
            .optional_input("grad_x_post", "DT_BF16, DT_FLOAT16")
            .attr("hc_eps", attr.Float(1e-6))
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_phi", "DT_FLOAT")
            .output("grad_alpha", "DT_FLOAT")
            .output("grad_bias", "DT_FLOAT"),
        )

    def _gather_last_dim(x: Tensor, begin: int, end: int):
        return ge.GatherV2(
            x,
            ge.Const(list(range(begin, end)), dtype=DataType.DT_INT64),
            ge.Const(-1, dtype=DataType.DT_INT64),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_backward.default
    )
    def convert_mhc_pre_sinkhorn_backward(
        grad_hin: Tensor,
        grad_h_post: Tensor,
        grad_h_res: Tensor,
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        h_pre: Tensor,
        hc_before_norm: Tensor,
        inv_rms: Tensor,
        sum_out: Tensor,
        norm_out: Tensor,
        hc_eps: float,
        meta_outputs: Optional[TensorSpec] = None,
    ):
        del meta_outputs, norm_out
        n = int(h_pre.symsize[-1])
        num_iters = int(sum_out.symsize[0]) // 2

        normalized_mix = ge.Mul(hc_before_norm, inv_rms)
        h_post_projection = _gather_last_dim(normalized_mix, n, 2 * n)
        h_post_bias = ge.GatherV2(
            bias,
            ge.Const(list(range(n, 2 * n)), dtype=DataType.DT_INT64),
            ge.Const(0, dtype=DataType.DT_INT64),
        )
        alpha_post = ge.GatherV2(
            alpha,
            ge.Const(1, dtype=DataType.DT_INT64),
            ge.Const(0, dtype=DataType.DT_INT64),
        )
        h_post_logits = ge.Add(ge.Mul(h_post_projection, alpha_post), h_post_bias)
        h_post = ge.Mul(
            ge.Sigmoid(h_post_logits),
            ge.Const(2.0, dtype=DataType.DT_FLOAT),
        )

        h_res_projection = _gather_last_dim(normalized_mix, 2 * n, 2 * n + n * n)
        h_res_bias = ge.GatherV2(
            bias,
            ge.Const(list(range(2 * n, 2 * n + n * n)), dtype=DataType.DT_INT64),
            ge.Const(0, dtype=DataType.DT_INT64),
        )
        alpha_res = ge.GatherV2(
            alpha,
            ge.Const(2, dtype=DataType.DT_INT64),
            ge.Const(0, dtype=DataType.DT_INT64),
        )
        h_res_logits = ge.Add(ge.Mul(h_res_projection, alpha_res), h_res_bias)

        x_shape = ge.Shape(x, dtype=DataType.DT_INT64)
        prefix_shape = ge.GatherV2(
            x_shape,
            ge.Const(list(range(x.rank - 2)), dtype=DataType.DT_INT64),
            ge.Const(0, dtype=DataType.DT_INT64),
        )
        h_res_shape = ge.ConcatV2(
            [
                prefix_shape,
                ge.Const([n, n], dtype=DataType.DT_INT64),
            ],
            ge.Const(0, dtype=DataType.DT_INT64),
            N=2,
        )
        h_res_logits = ge.Reshape(h_res_logits, h_res_shape)
        grad_h_res = ge.Reshape(grad_h_res, h_res_shape)

        _, sinkhorn_norm_out, sinkhorn_sum_out = _mhc_sinkhorn(
            h_res_logits, hc_eps, num_iters
        )
        grad_h_res_before_sinkhorn = _mhc_sinkhorn_backward(
            grad_h_res, sinkhorn_norm_out, sinkhorn_sum_out
        )
        inv_rms_squeezed = ge.Squeeze(inv_rms, axis=[-1])
        return _mhc_pre_backward(
            x,
            phi,
            alpha,
            grad_hin,
            grad_h_post,
            grad_h_res_before_sinkhorn,
            inv_rms_squeezed,
            hc_before_norm,
            h_pre,
            h_post,
            hc_eps,
        )
