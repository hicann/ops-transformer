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
    # premix 必须放在 * 之后(keyword-only)且标志列表保持 4 项: torchair 的
    # auto_convert_to_tensor 用 signature.bind() 绑定实参(不应用默认值), 若
    # premix 为第 5 个位置参数则存量 4 实参调用会触发
    # "The length of args can not be less than inputs_dynamic"; 且 flags 第二
    # 列表为 inputs_optional, premix 标记非 optional 时转换器传 premix=None
    # (无 premix 的图)会触发 "can not be None as it is not optional"。
    # keyword-only 经 kwargs 传递不受上述两处校验影响, 函数体已有 None 处理。
    @auto_convert_to_tensor([False, False, False, False], [False, False, False, False])
    def MhcPreSinkhorn(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        *,
        premix: Optional[Tensor] = None,
        hc_mult: int = 4,
        num_iters: int = 20,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        need_backward: bool = True,
        enable_hf32: bool = True,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MhcPreSinkhorn)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(phi, TensorType({DT_FLOAT}))\n
        .INPUT(alpha, TensorType({DT_FLOAT}))\n
        .INPUT(bias, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(premix, TensorType({DT_FLOAT}))\n
        .OUTPUT(hin, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(h_post, TensorType({DT_FLOAT}))\n
        .OUTPUT(h_res, TensorType({DT_FLOAT}))\n
        .OUTPUT(h_pre, TensorType({DT_FLOAT}))\n
        .OUTPUT(hc_before_norm, TensorType({DT_FLOAT}))\n
        .OUTPUT(inv_rms, TensorType({DT_FLOAT}))\n
        .OUTPUT(sum_out, TensorType({DT_FLOAT}))\n
        .OUTPUT(norm_out, TensorType({DT_FLOAT}))\n
        .ATTR(hc_mult, Int, 4)\n
        .ATTR(num_iters, Int, 20)\n
        .ATTR(hc_eps, Float, 1e-6)\n
        .ATTR(norm_eps, Float, 1e-6)\n
        .ATTR(need_backward, Bool, true)\n
        .ATTR(enable_hf32, Bool, true)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "phi": phi,
            "alpha": alpha,
            "bias": bias,
        }
        if premix is not None:
            inputs["premix"] = premix

        attrs = {
            "hc_mult": attr.Int(hc_mult),
            "num_iters": attr.Int(num_iters),
            "hc_eps": attr.Float(hc_eps),
            "norm_eps": attr.Float(norm_eps),
            "need_backward": attr.Bool(need_backward),
            "enable_hf32": attr.Bool(enable_hf32),
        }

        outputs = [
            "hin",
            "h_post",
            "h_res",
            "h_pre",
            "hc_before_norm",
            "inv_rms",
            "sum_out",
            "norm_out",
        ]

        return ge_op(
            op_type="MhcPreSinkhorn",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPreSinkhorn")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .optional_input("premix", "DT_FLOAT")
            .attr("hc_mult", attr.Int(4))
            .attr("num_iters", attr.Int(20))
            .attr("hc_eps", attr.Float(1e-6))
            .attr("norm_eps", attr.Float(1e-6))
            .attr("need_backward", attr.Bool(True))
            .attr("enable_hf32", attr.Bool(True))
            .output("hin", "DT_BF16, DT_FLOAT16")
            .output("h_post", "DT_FLOAT")
            .output("h_res", "DT_FLOAT")
            .output("h_pre", "DT_FLOAT")
            .output("hc_before_norm", "DT_FLOAT")
            .output("inv_rms", "DT_FLOAT")
            .output("sum_out", "DT_FLOAT")
            .output("norm_out", "DT_FLOAT"),
        )

    # 与 torch schema 对齐(不含 enable_hf32): converter 形参名须与 schema 参数名
    # (pre_mix)一致, 图模式按名传 kwargs, 形参名不一致会报 unexpected keyword
    # argument。enable_hf32 于构图时读全局开关 torch_npu.npu.aclnn.allow_hf32 并
    # 固化进 GE ATTR(与 eager 侧 wrapper/dispatch 同源)。
    def _get_allow_hf32():
        try:
            import torch_npu

            return bool(torch_npu.npu.aclnn.allow_hf32)
        except Exception:
            return True

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_sinkhorn.default
    )
    def convert_mhc_pre_sinkhorn(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        hc_mult: int,
        num_iters: int,
        hc_eps: float,
        norm_eps: float,
        out_flag: bool,
        pre_mix: Optional[Tensor] = None,
        meta_outputs: TensorSpec = None,
    ):
        return MhcPreSinkhorn(
            x=x,
            phi=phi,
            alpha=alpha,
            bias=bias,
            premix=pre_mix,
            hc_mult=hc_mult,
            num_iters=num_iters,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            need_backward=out_flag,
            enable_hf32=_get_allow_hf32(),
        )
