# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
    from torchair.ge._ge_graph import DataType, TensorType
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
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    )
    def MoeDistributeCombine(
        context: Tensor,
        expand_x: Tensor,
        expert_ids: Tensor,
        assist_info_for_combine: Tensor,
        ep_send_counts: Tensor,
        expert_scales: Tensor,
        tp_send_counts: Optional[Tensor],
        x_active_mask: Optional[Tensor],
        expand_scales: Optional[Tensor],
        shared_expert_x: Optional[Tensor],
        elastic_info: Optional[Tensor],
        ori_x: Optional[Tensor],
        const_expert_alpha_1: Optional[Tensor],
        const_expert_alpha_2: Optional[Tensor],
        const_expert_v: Optional[Tensor],
        performance_info: Optional[Tensor],
        *,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        expert_shard_type: int = 0,
        shared_expert_num: int = 1,
        shared_expert_rank_num: int = 0,
        global_bs: int = 0,
        out_dtype: int = 0,
        comm_quant_mode: int = 0,
        group_list_type: int = 0,
        comm_alg: str = "",
        zero_expert_num: int = 0,
        copy_expert_num: int = 0,
        const_expert_num: int = 0,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(MoeDistributeCombineV3)\n
        .INPUT(context, TensorType({DT_INT32}))\n
        .INPUT(expand_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))\n
        .INPUT(expert_ids, TensorType({DT_INT32}))\n
        .INPUT(assist_info_for_combine, TensorType({DT_INT32}))\n
        .INPUT(ep_send_counts, TensorType({DT_INT32}))\n
        .INPUT(expert_scales, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(tp_send_counts, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))\n
        .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(expand_scales, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(shared_expert_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))\n
        .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(ori_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))\n
        .OPTIONAL_INPUT(const_expert_alpha_1, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))\n
        .OPTIONAL_INPUT(const_expert_alpha_2, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))\n
        .OPTIONAL_INPUT(const_expert_v, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))\n
        .OPTIONAL_INPUT(performance_info, TensorType({DT_INT64}))\n
        .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .REQUIRED_ATTR(ep_world_size, Int)\n
        .REQUIRED_ATTR(ep_rank_id, Int)\n
        .REQUIRED_ATTR(moe_expert_num, Int)\n
        .REQUIRED_ATTR(ccl_buffer_size, Int)\n
        .ATTR(tp_world_size, Int, 0)\n
        .ATTR(tp_rank_id, Int, 0)\n
        .ATTR(expert_shard_type, Int, 0)\n
        .ATTR(shared_expert_num, Int, 1)\n
        .ATTR(shared_expert_rank_num, Int, 0)\n
        .ATTR(global_bs, Int, 0)\n
        .ATTR(out_dtype, Int, 0)\n
        .ATTR(comm_quant_mode, Int, 0)\n
        .ATTR(group_list_type, Int, 0)\n
        .ATTR(comm_alg, String, "")\n
        .ATTR(zero_expert_num, Int, 0)\n
        .ATTR(copy_expert_num, Int, 0)\n
        .ATTR(const_expert_num, Int, 0)\n
        """

        op = get_default_ge_graph().op.add()
        op.type = "MoeDistributeCombineV3"
        op.name = next_unique_name(node_name, "MoeDistributeCombineV3")

        # process dependices
        for dependency in dependencies:
            op.input.append(dependency.controller)

        # process inputs
        op.input.append(context.tensor)
        op.input_desc.add().CopyFrom(context.desc)
        op.input_desc[-1].name = "context"

        op.input.append(expand_x.tensor)
        op.input_desc.add().CopyFrom(expand_x.desc)
        op.input_desc[-1].name = "expand_x"

        op.input.append(expert_ids.tensor)
        op.input_desc.add().CopyFrom(expert_ids.desc)
        op.input_desc[-1].name = "expert_ids"

        op.input.append(assist_info_for_combine.tensor)
        op.input_desc.add().CopyFrom(assist_info_for_combine.desc)
        op.input_desc[-1].name = "assist_info_for_combine"

        op.input.append(ep_send_counts.tensor)
        op.input_desc.add().CopyFrom(ep_send_counts.desc)
        op.input_desc[-1].name = "ep_send_counts"

        op.input.append(expert_scales.tensor)
        op.input_desc.add().CopyFrom(expert_scales.desc)
        op.input_desc[-1].name = "expert_scales"

        if tp_send_counts is not None:
            op.input.append(tp_send_counts.tensor)
            op.input_desc.add().CopyFrom(tp_send_counts.desc)
            op.input_desc[-1].name = "tp_send_counts"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "tp_send_counts"

        if x_active_mask is not None:
            op.input.append(x_active_mask.tensor)
            op.input_desc.add().CopyFrom(x_active_mask.desc)
            op.input_desc[-1].name = "x_active_mask"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "x_active_mask"

        # In V2, three unused reserved parameters from V1 have been removed. However, the oprator
        # prototype on the canndev still retains the original parameters. Additionally, when the
        # input to the torch layer is V2, canndev internalyy selects either the V1 or V2 version
        # of aclnn based on the A2/A3 platform. Therefore, it is necessary to perform a placeholder
        # operation for the parameters that exist in the V1.
        op.input.append("")
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "activation_scale"
        op.input.append("")
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "weight_scale"
        op.input.append("")
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "group_list"

        if expand_scales is not None:
            op.input.append(expand_scales.tensor)
            op.input_desc.add().CopyFrom(expand_scales.desc)
            op.input_desc[-1].name = "expand_scales"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "expand_scales"

        if shared_expert_x is not None:
            op.input.append(shared_expert_x.tensor)
            op.input_desc.add().CopyFrom(shared_expert_x.desc)
            op.input_desc[-1].name = "shared_expert_x"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "shared_expert_x"

        if elastic_info is not None:
            op.input.append(elastic_info.tensor)
            op.input_desc.add().CopyFrom(elastic_info.desc)
            op.input_desc[-1].name = "elastic_info"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "elastic_info"

        if ori_x is not None:
            op.input.append(ori_x.tensor)
            op.input_desc.add().CopyFrom(ori_x.desc)
            op.input_desc[-1].name = "ori_x"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "ori_x"

        if const_expert_alpha_1 is not None:
            op.input.append(const_expert_alpha_1.tensor)
            op.input_desc.add().CopyFrom(const_expert_alpha_1.desc)
            op.input_desc[-1].name = "const_expert_alpha_1"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "const_expert_alpha_1"

        if const_expert_alpha_2 is not None:
            op.input.append(const_expert_alpha_2.tensor)
            op.input_desc.add().CopyFrom(const_expert_alpha_2.desc)
            op.input_desc[-1].name = "const_expert_alpha_2"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "const_expert_alpha_2"

        if const_expert_v is not None:
            op.input.append(const_expert_v.tensor)
            op.input_desc.add().CopyFrom(const_expert_v.desc)
            op.input_desc[-1].name = "const_expert_v"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "const_expert_v"

        if performance_info is not None:
            op.input.append(performance_info.tensor)
            op.input_desc.add().CopyFrom(performance_info.desc)
            op.input_desc[-1].name = "performance_info"
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = "performance_info"

        # process attrs
        op.attr["ep_world_size"].i = ep_world_size
        op.attr["ep_rank_id"].i = ep_rank_id
        op.attr["moe_expert_num"].i = moe_expert_num
        op.attr["ccl_buffer_size"].i = ccl_buffer_size
        op.attr["tp_world_size"].i = tp_world_size
        op.attr["tp_rank_id"].i = tp_rank_id
        op.attr["expert_shard_type"].i = expert_shard_type
        op.attr["shared_expert_num"].i = shared_expert_num
        op.attr["shared_expert_rank_num"].i = shared_expert_rank_num
        op.attr["global_bs"].i = global_bs
        op.attr["out_dtype"].i = out_dtype
        op.attr["comm_quant_mode"].i = comm_quant_mode
        op.attr["group_list_type"].i = group_list_type
        op.attr["comm_alg"].s = compat_as_bytes(comm_alg)
        op.attr["zero_expert_num"].i = zero_expert_num
        op.attr["copy_expert_num"].i = copy_expert_num
        op.attr["const_expert_num"].i = const_expert_num

        # process outputs
        output_index = 0
        op.output_desc.add().name = "x"
        x = Tensor(op, output_index)
        output_index += 1

        return x

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.npu_moe_distribute_combine.default
    )
    def convert_npu_moe_distribute_combine(
        context: Tensor,
        expand_x: Tensor,
        expert_ids: Tensor,
        assist_info_for_combine: Tensor,
        ep_send_counts: Tensor,
        expert_scales: Tensor,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        *,
        tp_send_counts: Tensor = None,
        x_active_mask: Tensor = None,
        expand_scales: Tensor = None,
        shared_expert_x: Tensor = None,
        elastic_info: Tensor = None,
        ori_x: Tensor = None,
        const_expert_alpha_1: Tensor = None,
        const_expert_alpha_2: Tensor = None,
        const_expert_v: Tensor = None,
        performance_info: Tensor = None,
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        expert_shard_type: int = 0,
        shared_expert_num: int = 1,
        shared_expert_rank_num: int = 0,
        global_bs: int = 0,
        comm_quant_mode: int = 0,
        comm_alg: str = "",
        zero_expert_num: int = 0,
        copy_expert_num: int = 0,
        const_expert_num: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        return MoeDistributeCombine(
            context=context,
            expand_x=expand_x,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            tp_send_counts=tp_send_counts,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
            shared_expert_x=shared_expert_x,
            elastic_info=elastic_info,
            ori_x=ori_x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            performance_info=performance_info,
            ep_world_size=ep_world_size,
            ep_rank_id=ep_rank_id,
            moe_expert_num=moe_expert_num,
            ccl_buffer_size=ccl_buffer_size,
            tp_world_size=tp_world_size,
            tp_rank_id=tp_rank_id,
            expert_shard_type=expert_shard_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            global_bs=global_bs,
            out_dtype=0,
            comm_quant_mode=comm_quant_mode,
            group_list_type=0,
            comm_alg=comm_alg,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num,
        )
