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
from torch_npu.utils._error_code import ErrCode, ops_error
from .comm_context import CommContextManager


def _empty_topk_weights_like(tensor):
    return tensor.new_empty((0,), dtype=torch.float32)


def _get_hccl_comm_name(group, rank_id):
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


class MoeDistributeBuffer:
    def __init__(self, group, ccl_buffer_size: int = 0, comm_alg: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self._ctx_manager = CommContextManager(
            self.group_name,
            self.world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            customCclBufferSize=ccl_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size

        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size

    @staticmethod
    def get_low_latency_ccl_buffer_size(
        world_size: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_moe_expert: int,
        topk: int,
        num_shared_expert: int = 0,
        num_shared_expert_ranks: int = 0,
        comm_alg: str = "",
    ) -> int:
        def inline_align(value, base):
            return (value + base - 1) // base * base

        torch._check(
            ((world_size >= 2) and (world_size <= 768)),  # world_size当前仅支持[2,768]
            lambda: (f"world_size only support in [2, 768], but got {world_size=}."),
        )
        torch._check(
            ((hidden >= 1024) and (hidden <= 8192)),  # hidden当前仅支持[1024,8192]
            lambda: (f"hidden only support in [1024, 8192], but got {hidden=}."),
        )
        # num_max_dispatch_tokens_per_rank当前仅支持[1,512]
        torch._check(
            (
                (num_max_dispatch_tokens_per_rank >= 1)
                and (num_max_dispatch_tokens_per_rank <= 512)
            ),
            lambda: (
                f"num_max_dispatch_tokens_per_rank only support in [1, 512], "
                f"but got {num_max_dispatch_tokens_per_rank=}."
            ),
        )
        torch._check(
            (
                (num_moe_expert >= 1) and (num_moe_expert <= 1024)
            ),  # num_moe_expert当前仅支持[1,1024]
            lambda: (
                f"num_moe_expert only support in [1, 1024], but got {num_moe_expert=}."
            ),
        )
        torch._check(
            ((topk >= 1) and (topk <= 16)),  # topk当前仅支持[1,16]
            lambda: (f"topk only support in [1, 16], but got {topk=}."),
        )
        torch._check(
            (
                (num_shared_expert >= 0) and (num_shared_expert <= 4)
            ),  # num_shared_expert当前仅支持[0,4]
            lambda: (
                f"num_shared_expert only support in [0, 4], but got {num_shared_expert=}."
            ),
        )
        torch._check(
            (world_size - num_shared_expert_ranks > 0),  # 至少存在一张moe专家卡
            lambda: (
                f"world_size - num_shared_expert_ranks must be greater than 0, "
                f"but got {world_size=} {num_shared_expert_ranks=}."
            ),
        )
        # local_moe_expert_num*world_size仅支持(0，2048]
        local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks)
        torch._check(
            (
                (local_moe_expert_num * world_size > 0)
                and (local_moe_expert_num * world_size <= 2048)
            ),
            lambda: (
                f"local_moe_expert_num * world_size only support in (0, 2048], "
                f"but got {world_size=} {num_shared_expert_ranks=}, "
                f"local_moe_expert_num = num_moe_expert // (world_size - num_shared_expert_ranks), "
                f"{local_moe_expert_num=}"
            ),
        )

        max_out_dtype_size = 2  # sizeof(int32)
        mb_conversion = 1024 * 1024
        ub_align = 32  # 32B
        scale_expand_index_buffer = 44  # scale 32B + 3 * 4 expand_idx
        full_mesh_data_align = 480
        win_addr_align = 512

        comm_alg_support_list = ["fullmesh_v1", "fullmesh_v2", ""]
        torch._check(
            comm_alg in comm_alg_support_list,
            lambda: (
                f"comm_alg only support {comm_alg_support_list=}, but got {comm_alg=}."
            ),
        )

        token_actual_len = (
            inline_align(hidden * max_out_dtype_size, ub_align)
            + scale_expand_index_buffer
        )
        if comm_alg == "fullmesh_v2":
            token_need_size_dispatch = (
                inline_align(token_actual_len, full_mesh_data_align)
                // full_mesh_data_align
                * win_addr_align
            )
        else:
            token_need_size_dispatch = inline_align(token_actual_len, win_addr_align)
        token_need_size_combine = inline_align(
            hidden * max_out_dtype_size, win_addr_align
        )
        minimum_buffer_size = (
            2
            * (
                (
                    num_max_dispatch_tokens_per_rank
                    * token_need_size_dispatch
                    * world_size
                    * local_moe_expert_num
                )
                + (
                    num_max_dispatch_tokens_per_rank
                    * token_need_size_combine
                    * (topk + num_shared_expert)
                )
            )
            + mb_conversion
        )
        ## hccl按照2*bufferSize大小开设
        ccl_buffer_size = (
            inline_align(
                inline_align(minimum_buffer_size, mb_conversion) // mb_conversion, 2
            )
            // 2
        )
        return ccl_buffer_size

    def update_ctx(self, new_group) -> bool:
        self.group = new_group
        self.rank_id = torch.distributed.get_rank(new_group)
        self.group_name = _get_hccl_comm_name(new_group, self.rank_id)
        new_world_size = torch.distributed.get_world_size(new_group)
        torch._check(
            new_world_size == self.world_size,
            lambda: (
                f"New world size should be the same as orginal world size, "
                f"but got {new_world_size=}, orginial={self.world_size}"
                f"{ops_error(ErrCode.VALUE)}."
            ),
        )
        self._ctx_manager.update_group(self.group_name, self.context)
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
        return True

    def low_latency_dispatch(
        self,
        x,
        topk_idx,
        num_experts: int,
        *,
        quant_mode=0,
        comm_alg="",
        x_smooth_scale=None,
        x_active_mask=None,
        topk_weights=None,
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
        elastic_info=None,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        expert_token_nums_type=1,
        num_max_dispatch_tokens_per_rank=0,
        y_dtype=None,
        x_dtype=None,
        x_smooth_scales_dtype=None,
    ):
        (
            expand_x,
            dynamic_scales,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = torch.ops.cann_ops_transformer.npu_moe_distribute_dispatch(
            context=self.context,
            x=x,
            expert_ids=topk_idx,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank_id,
            moe_expert_num=num_experts,
            ccl_buffer_size=self.ccl_buffer_size,
            scales=x_smooth_scale,
            x_active_mask=x_active_mask,
            expert_scales=topk_weights,
            elastic_info=elastic_info,
            performance_info=None,
            expert_shard_type=expert_shard_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            quant_mode=quant_mode,
            expert_token_nums_type=expert_token_nums_type,
            global_bs=num_max_dispatch_tokens_per_rank * self.world_size,
            comm_alg=comm_alg,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num,
            y_dtype=y_dtype,
            x_dtype=x_dtype,
            x_smooth_scales_dtype=x_smooth_scales_dtype,
        )
        return (
            expand_x,
            dynamic_scales,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            expand_scales,
        )

    def low_latency_combine(
        self,
        x,
        topk_idx,
        topk_weights,
        assist_info_for_combine,
        ep_send_counts,
        *,
        num_experts=0,
        comm_alg="",
        comm_quant_mode=0,
        x_active_mask=None,
        expand_scales=None,
        shared_expert_x=None,
        elastic_info=None,
        ori_x=None,
        const_expert_alpha_1=None,
        const_expert_alpha_2=None,
        const_expert_v=None,
        zero_expert_num=0,
        copy_expert_num=0,
        const_expert_num=0,
        expert_shared_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        num_max_dispatch_tokens_per_rank=0,
    ):
        if assist_info_for_combine is None:
            raise ValueError("assist_info_for_combine must be provided.")
        if ep_send_counts is None:
            raise ValueError("ep_send_counts must be provided.")
        topk_weights = (
            _empty_topk_weights_like(x) if topk_weights is None else topk_weights
        )
        return torch.ops.cann_ops_transformer.npu_moe_distribute_combine(
            context=self.context,
            expand_x=x,
            expert_ids=topk_idx,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=topk_weights,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank_id,
            moe_expert_num=num_experts,
            ccl_buffer_size=self.ccl_buffer_size,
            tp_send_counts=None,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
            shared_expert_x=shared_expert_x,
            elastic_info=elastic_info,
            ori_x=ori_x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            performance_info=None,
            expert_shard_type=expert_shared_type,
            shared_expert_num=shared_expert_num,
            shared_expert_rank_num=shared_expert_rank_num,
            copy_expert_num=copy_expert_num,
            zero_expert_num=zero_expert_num,
            const_expert_num=const_expert_num,
            comm_alg=comm_alg,
            comm_quant_mode=comm_quant_mode,
            global_bs=num_max_dispatch_tokens_per_rank * self.world_size,
        )
