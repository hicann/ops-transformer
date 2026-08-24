# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MoeDistributeCombineOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDistributeCombineOpBuilder, self).__init__(
            "npu_moe_distribute_combine", category="mc2"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mc2/moe_distribute_combine.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "npu_moe_distribute_combine(Tensor context, Tensor expand_x, Tensor expert_ids, "
            "Tensor assist_info_for_combine, Tensor ep_send_counts, Tensor expert_scales, "
            "int ep_world_size, int ep_rank_id, int moe_expert_num, int ccl_buffer_size, "
            "*, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, "
            "Tensor? expand_scales=None, Tensor? shared_expert_x=None, Tensor? elastic_info=None, "
            "Tensor? ori_x=None, Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, "
            "Tensor? const_expert_v=None, Tensor? performance_info=None, int tp_world_size=0, "
            "int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, "
            'int global_bs=0, int comm_quant_mode=0, str comm_alg="", int zero_expert_num=0, '
            "int copy_expert_num=0, int const_expert_num=0) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def npu_moe_distribute_combine_meta(
            context,
            expand_x,
            expert_ids,
            assist_info_for_combine,
            ep_send_counts,
            expert_scales,
            ep_world_size,
            ep_rank_id,
            moe_expert_num,
            ccl_buffer_size,
            tp_send_counts=None,
            x_active_mask=None,
            expand_scales=None,
            shared_expert_x=None,
            elastic_info=None,
            ori_x=None,
            const_expert_alpha_1=None,
            const_expert_alpha_2=None,
            const_expert_v=None,
            performance_info=None,
            tp_world_size=0,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg="",
            zero_expert_num=0,
            copy_expert_num=0,
            const_expert_num=0,
        ):
            dim_tuple = (expert_ids.size(0), expand_x.size(1))
            return expand_x.new_empty(dim_tuple)


moe_distribute_combine_op_builder = MoeDistributeCombineOpBuilder()
moe_distribute_combine_op_builder._ensure_initialized()


@impl(get_as_library(), moe_distribute_combine_op_builder.name, "PrivateUse1")
def _npu_moe_distribute_combine(
    context,
    expand_x,
    expert_ids,
    assist_info_for_combine,
    ep_send_counts,
    expert_scales,
    ep_world_size,
    ep_rank_id,
    moe_expert_num,
    ccl_buffer_size,
    tp_send_counts=None,
    x_active_mask=None,
    expand_scales=None,
    shared_expert_x=None,
    elastic_info=None,
    ori_x=None,
    const_expert_alpha_1=None,
    const_expert_alpha_2=None,
    const_expert_v=None,
    performance_info=None,
    tp_world_size=0,
    tp_rank_id=0,
    expert_shard_type=0,
    shared_expert_num=1,
    shared_expert_rank_num=0,
    global_bs=0,
    comm_quant_mode=0,
    comm_alg="",
    zero_expert_num=0,
    copy_expert_num=0,
    const_expert_num=0,
):
    op_module = moe_distribute_combine_op_builder.load()
    return op_module.npu_moe_distribute_combine(
        context,
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_counts,
        expert_scales,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        ccl_buffer_size,
        tp_send_counts,
        x_active_mask,
        expand_scales,
        shared_expert_x,
        elastic_info,
        ori_x,
        const_expert_alpha_1,
        const_expert_alpha_2,
        const_expert_v,
        performance_info,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        comm_quant_mode,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
    )
