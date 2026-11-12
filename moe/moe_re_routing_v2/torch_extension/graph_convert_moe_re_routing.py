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
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
    from torchair.ge._ge_graph import (
        torch_dtype_value_to_ge_type,
        torch_dtype_value_to_ge_proto_type,
    )
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import (
        _ge_dtype_to_ge_proto_dtype,
        torch_dtype_value_to_ge_proto_type,
        torch_dtype_value_to_ge_type,
    )
    from torchair.ge import attr
    from typing import Optional

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    # torchair 旧版本无 DT_HIFLOAT4 枚举时跳过（hif4 仅 Ascend960 支持）
    _DT_HIFLOAT4 = getattr(DataType, "DT_HIFLOAT4", None)
    # torch_npu 暂无 hifloat4 dtype 枚举，以 g_toAclOffset(256) + ACL_HIFLOAT4(42) 数值承载
    _HIF4_DTYPE_VALUE = getattr(torch_npu, "hifloat4", 256 + 42)

    def _is_hif4_dtype(dtype: int) -> bool:
        return dtype == _HIF4_DTYPE_VALUE

    def _is_fp4_dtype(dtype: int) -> bool:
        # torchair 的 torch_dtype_value_to_ge_type 尚无 hifloat4(298) 映射，先短路判定，
        # 避免 Unsupported torch dtype 异常
        if _is_hif4_dtype(dtype):
            return True
        ge_type = torch_dtype_value_to_ge_type(dtype)
        return ge_type in (DataType.DT_FLOAT4_E2M1, DataType.DT_FLOAT4_E1M2)

    def _unpack_uint8_to_hif4(tokens: Tensor) -> Tensor:
        """把 uint8 打包的 hif4 还原成 4bit 逻辑张量（末维 ×2）。

        dtype 显式声明为 DT_HIFLOAT4，不依赖 torchair 对 hifloat4 枚举值的映射。
        """
        tokens_const = ge.Const([1] * (tokens.rank - 1) + [2])
        tokens_shape = ge.Shape(tokens)
        tokens_new_shape = ge.Mul(tokens_shape, tokens_const)
        tokens = ge.Bitcast(tokens, type=_DT_HIFLOAT4)
        tokens.desc.dtype = _ge_dtype_to_ge_proto_dtype(_DT_HIFLOAT4)
        return ge.Reshape(tokens, tokens_new_shape)

    def _unpack_uint8_to_float4(tokens: Tensor, tokens_dtype: int) -> Tensor:
        target_dtype = torch_dtype_value_to_ge_type(tokens_dtype)
        tokens_const = ge.Const([1] * (tokens.rank - 1) + [2])
        tokens_shape = ge.Shape(tokens)
        tokens_new_shape = ge.Mul(tokens_shape, tokens_const)
        tokens = ge.Bitcast(tokens, type=target_dtype)
        tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)
        tokens = ge.Reshape(tokens, tokens_new_shape)
        return tokens

    def _pack_int4_to_uint8(tensor: Tensor) -> Tensor:
        bit_shape = [1, 2]
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(tensor)
        y_shape_uint8 = ge.Div(y_shape_int4, div_x2)
        y_shape_int4_2bit = ge.ConcatV2(
            [y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
            concat_dim=0,
            N=2,
        )
        y = ge.Bitcast(ge.Reshape(tensor, y_shape_int4_2bit), type=DataType.DT_UINT8)
        permute_tokens = ge.Reshape(y, y_shape_uint8)
        return permute_tokens

    @auto_convert_to_tensor([False, False, False, False], [False, False, True, True])
    def MoeReRouting(
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        per_token_scales: Optional[Tensor],
        expert_topk_weight: Optional[Tensor],
        *,
        expert_token_num_type: int,
        idx_type: int,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeReRoutingV2)\n
        .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                   DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .INPUT(expert_token_num_per_rank, TensorType({DT_INT32, DT_INT64}))\n
        .OPTIONAL_INPUT(per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OPTIONAL_INPUT(expert_topk_weight, TensorType({DT_FLOAT}))\n
        .OUTPUT(permute_tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                            DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .OUTPUT(permute_per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OUTPUT(permute_token_idx, TensorType({DT_INT32}))\n
        .OUTPUT(expert_token_num, TensorType({DT_INT32, DT_INT64}))\n
        .OUTPUT(permute_topk_weight, TensorType({DT_FLOAT}))\n
        .ATTR(expert_token_num_type, Int, 1)\n
        .ATTR(idx_type, Int, 0)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "tokens": tokens,
            "expert_token_num_per_rank": expert_token_num_per_rank,
            "per_token_scales": per_token_scales,
            "expert_topk_weight": expert_topk_weight,
        }

        attrs = {
            "expert_token_num_type": attr.Int(expert_token_num_type),
            "idx_type": attr.Int(idx_type),
        }

        outputs = [
            "permute_tokens",
            "permute_per_token_scales",
            "permute_token_idx",
            "expert_token_num",
            "permute_topk_weight",
        ]

        return ge_op(
            op_type="MoeReRoutingV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeReRoutingV2")
            .input(
                "tokens",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, "
                "DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_HIFLOAT4",
            )
            .input("expert_token_num_per_rank", "DT_INT32, DT_INT64")
            .optional_input("per_token_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("expert_topk_weight", "DT_FLOAT")
            .attr("expert_token_num_type", attr.Int(1))
            .attr("idx_type", attr.Int(0))
            .output(
                "permute_tokens",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, "
                "DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_HIFLOAT4",
            )
            .output("permute_per_token_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
            .output("permute_token_idx", "DT_INT32")
            .output("expert_token_num", "DT_INT32, DT_INT64")
            .output("permute_topk_weight", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_re_routing.default
    )
    def convert_moe_re_routing(
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        *,
        per_token_scales: Tensor = None,
        expert_topk_weight: Tensor = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
        tokens_dtype: int = None,
        meta_outputs: TensorSpec = None,
    ):
        if tokens_dtype is not None:
            if _is_hif4_dtype(tokens_dtype):
                if _DT_HIFLOAT4 is None:
                    raise RuntimeError(
                        "tokens_dtype=hifloat4 requires a torchair version with "
                        "DT_HIFLOAT4 support, please upgrade torchair."
                    )
                if tokens.dtype == DataType.DT_UINT8:
                    tokens = _unpack_uint8_to_hif4(tokens)
                # 图内 DT_HIFLOAT4 直通时无需转换
            elif tokens.dtype == DataType.DT_UINT8 and _is_fp4_dtype(tokens_dtype):
                tokens = _unpack_uint8_to_float4(tokens, tokens_dtype)
            else:
                tokens = ge.Bitcast(
                    tokens, type=torch_dtype_value_to_ge_type(tokens_dtype)
                )
                tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)

        (
            permute_tokens,
            permute_per_token_scales,
            permute_token_idx,
            expert_token_num,
            permute_topk_weight,
        ) = MoeReRouting(
            tokens,
            expert_token_num_per_rank,
            per_token_scales,
            expert_topk_weight,
            expert_token_num_type=expert_token_num_type,
            idx_type=idx_type,
        )

        if tokens_dtype is not None:
            if _is_fp4_dtype(tokens_dtype):
                permute_tokens = _pack_int4_to_uint8(permute_tokens)
            else:
                permute_tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(
                    tokens_dtype
                )

        return (
            permute_tokens,
            permute_per_token_scales,
            permute_token_idx,
            expert_token_num,
            permute_topk_weight,
        )
