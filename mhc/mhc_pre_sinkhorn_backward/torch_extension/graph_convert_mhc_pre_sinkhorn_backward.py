# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    def MhcPreSinkhornBackward(
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
        *,
        hc_eps: float = 1e-6,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MhcPreSinkhornBackward)\n
        .INPUT(grad_hin, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(grad_h_post, TensorType({DT_FLOAT}))\n
        .INPUT(grad_h_res, TensorType({DT_FLOAT}))\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(phi, TensorType({DT_FLOAT}))\n
        .INPUT(alpha, TensorType({DT_FLOAT}))\n
        .INPUT(bias, TensorType({DT_FLOAT}))\n
        .INPUT(h_pre, TensorType({DT_FLOAT}))\n
        .INPUT(hc_before_norm, TensorType({DT_FLOAT}))\n
        .INPUT(inv_rms, TensorType({DT_FLOAT}))\n
        .INPUT(sum_out, TensorType({DT_FLOAT}))\n
        .INPUT(norm_out, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(grad_phi, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_alpha, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_bias, TensorType({DT_FLOAT}))\n
        .ATTR(hc_eps, Float, 1e-6)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "grad_hin": grad_hin,
            "grad_h_post": grad_h_post,
            "grad_h_res": grad_h_res,
            "x": x,
            "phi": phi,
            "alpha": alpha,
            "bias": bias,
            "h_pre": h_pre,
            "hc_before_norm": hc_before_norm,
            "inv_rms": inv_rms,
            "sum_out": sum_out,
            "norm_out": norm_out,
        }

        attrs = {
            "hc_eps": attr.Float(hc_eps),
        }

        outputs = [
            "grad_x",
            "grad_phi",
            "grad_alpha",
            "grad_bias",
        ]

        return ge_op(
            op_type="MhcPreSinkhornBackward",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPreSinkhornBackward")
            .input("grad_hin", "DT_BF16, DT_FLOAT16")
            .input("grad_h_post", "DT_FLOAT")
            .input("grad_h_res", "DT_FLOAT")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .input("h_pre", "DT_FLOAT")
            .input("hc_before_norm", "DT_FLOAT")
            .input("inv_rms", "DT_FLOAT")
            .input("sum_out", "DT_FLOAT")
            .input("norm_out", "DT_FLOAT")
            .attr("hc_eps", attr.Float(1e-6))
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_phi", "DT_FLOAT")
            .output("grad_alpha", "DT_FLOAT")
            .output("grad_bias", "DT_FLOAT"),
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
        meta_outputs: TensorSpec = None,
    ):
        return MhcPreSinkhornBackward(
            grad_hin=grad_hin,
            grad_h_post=grad_h_post,
            grad_h_res=grad_h_res,
            x=x,
            phi=phi,
            alpha=alpha,
            bias=bias,
            h_pre=h_pre,
            hc_before_norm=hc_before_norm,
            inv_rms=inv_rms,
            sum_out=sum_out,
            norm_out=norm_out,
            hc_eps=hc_eps,
        )

    # pre_mix/grad_pre 必须放在 * 之后(keyword-only)且标志列表保持 12 项:
    # torchair 的 auto_convert_to_tensor 用 signature.bind() 绑定实参(不应用
    # 默认值), 标志列表扩到 14 会使存量 12 实参调用触发长度断言; 且 flags 第二
    # 列表为 inputs_optional, 可选输入标为非 optional 时转换器传 None(无 premix
    # 的 V2 调用, 41 环境实测会触发 "can not be None as it is not optional")。
    # keyword-only 经 kwargs 传递不受上述校验影响, 函数体已有 None 处理。
    @auto_convert_to_tensor(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    def MhcPreSinkhornBackwardV2(
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
        *,
        pre_mix: Optional[Tensor] = None,
        grad_pre: Optional[Tensor] = None,
        hc_eps: float = 1e-6,
        enable_hf32: bool = True,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MhcPreSinkhornBackward)\n
        .INPUT(grad_hin, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(grad_h_post, TensorType({DT_FLOAT}))\n
        .INPUT(grad_h_res, TensorType({DT_FLOAT}))\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(phi, TensorType({DT_FLOAT}))\n
        .INPUT(alpha, TensorType({DT_FLOAT}))\n
        .INPUT(bias, TensorType({DT_FLOAT}))\n
        .INPUT(h_pre, TensorType({DT_FLOAT}))\n
        .INPUT(hc_before_norm, TensorType({DT_FLOAT}))\n
        .INPUT(inv_rms, TensorType({DT_FLOAT}))\n
        .INPUT(sum_out, TensorType({DT_FLOAT}))\n
        .INPUT(norm_out, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(pre_mix, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(grad_pre, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(grad_phi, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_alpha, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_bias, TensorType({DT_FLOAT}))\n
        .OPTIONAL_OUTPUT(grad_pre_mix, TensorType({DT_FLOAT}))\n
        .ATTR(hc_eps, Float, 1e-6)\n
        .ATTR(enable_hf32, Bool, true)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "grad_hin": grad_hin,
            "grad_h_post": grad_h_post,
            "grad_h_res": grad_h_res,
            "x": x,
            "phi": phi,
            "alpha": alpha,
            "bias": bias,
            "h_pre": h_pre,
            "hc_before_norm": hc_before_norm,
            "inv_rms": inv_rms,
            "sum_out": sum_out,
            "norm_out": norm_out,
        }
        if pre_mix is not None:
            inputs["pre_mix"] = pre_mix
        if grad_pre is not None:
            inputs["grad_pre"] = grad_pre

        attrs = {
            "hc_eps": attr.Float(hc_eps),
            "enable_hf32": attr.Bool(enable_hf32),
        }

        outputs = [
            "grad_x",
            "grad_phi",
            "grad_alpha",
            "grad_bias",
            "grad_pre_mix",
        ]

        return ge_op(
            op_type="MhcPreSinkhornBackward",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPreSinkhornBackward")
            .input("grad_hin", "DT_BF16, DT_FLOAT16")
            .input("grad_h_post", "DT_FLOAT")
            .input("grad_h_res", "DT_FLOAT")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .input("h_pre", "DT_FLOAT")
            .input("hc_before_norm", "DT_FLOAT")
            .input("inv_rms", "DT_FLOAT")
            .input("sum_out", "DT_FLOAT")
            .input("norm_out", "DT_FLOAT")
            .optional_input("pre_mix", "DT_FLOAT")
            .optional_input("grad_pre", "DT_FLOAT")
            .attr("hc_eps", attr.Float(1e-6))
            .attr("enable_hf32", attr.Bool(True))
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_phi", "DT_FLOAT")
            .output("grad_alpha", "DT_FLOAT")
            .output("grad_bias", "DT_FLOAT")
            .output("grad_pre_mix", "DT_FLOAT"),
        )

    # converter 形参名须与 schema 参数名(pre_mix/grad_pre/enable_hf32)一致: 图模式按名传 kwargs,
    # 改成 camelCase 会报 unexpected keyword argument。
    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_backward_v2.default
    )
    def convert_mhc_pre_sinkhorn_backward_v2(
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
        pre_mix: Optional[Tensor] = None,
        grad_pre: Optional[Tensor] = None,
        hc_eps: float = 1e-6,
        enable_hf32: bool = True,
        meta_outputs: TensorSpec = None,
    ):
        return MhcPreSinkhornBackwardV2(
            grad_hin=grad_hin,
            grad_h_post=grad_h_post,
            grad_h_res=grad_h_res,
            x=x,
            phi=phi,
            alpha=alpha,
            bias=bias,
            h_pre=h_pre,
            hc_before_norm=hc_before_norm,
            inv_rms=inv_rms,
            sum_out=sum_out,
            norm_out=norm_out,
            pre_mix=pre_mix,
            grad_pre=grad_pre,
            hc_eps=hc_eps,
            enable_hf32=enable_hf32,
        )
