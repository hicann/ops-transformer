# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from .comm_context import CommContextManager


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    293: torch.uint8,  # torch_npu.float8_e8m0 use torch.uint8
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class MoeDistributeDispatchOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDistributeDispatchOpBuilder, self).__init__(
            "npu_moe_distribute_dispatch"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/moe_distribute_dispatch.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return ("npu_moe_distribute_dispatch(Tensor context, Tensor x, Tensor expert_ids, " \
            "int ep_world_size, int ep_rank_id, int moe_expert_num, int ccl_buffer_size, " \
            "*, Tensor? scales=None, Tensor? x_active_mask=None, " \
            "Tensor? expert_scales=None, Tensor? elastic_info=None, Tensor? performance_info=None, "\
            "int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, " \
            "int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, " \
            "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0, " \
            "int? y_dtype=None, int? x_dtype=None, int? x_smooth_scales_dtype=None) " \
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        def has_tensor(tensor):
            return tensor is not None

        def get_dtype_from_enum(dtype):
            return TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP[dtype]

        def is_packed_float4(dtype):
            return dtype in (296, 297)

        def is_ascend950():
            try:
                return "Ascend950" in torch_npu.npu.get_device_name()
            except Exception:
                return False

        def get_dispatch_dynamic_scales_dtype(x, scales, x_smooth_scales_dtype, quant_mode):
            dynamic_scales_dtype = torch.float32
            if quant_mode == 0:
                if x.dtype != torch.bfloat16 and x.dtype != torch.float16 and scales is not None:
                    dynamic_scales_dtype = (
                        get_dtype_from_enum(x_smooth_scales_dtype)
                        if x_smooth_scales_dtype is not None else scales.dtype
                    )
            elif quant_mode == 4 or quant_mode == 5:
                dynamic_scales_dtype = torch.uint8  # float8_e8m0
            return dynamic_scales_dtype

        def get_dispatch_dynamic_shape(scales, quant_mode, a, h):
            shape = tuple([a])
            if quant_mode == 0 and scales is not None:
                if scales.dim() < 2:
                    raise RuntimeError(f"Expected scales to be at least 2-d, but got {scales.dim()}-d.")
                shape = tuple([a, scales.shape[1]])
            elif quant_mode == 2:
                shape = tuple([a])
            elif quant_mode == 3:
                shape = tuple([a, math.ceil(h / 128)])
            elif quant_mode == 4 or quant_mode == 5:
                shape = tuple([a, (math.ceil(h / 32) + 1) // 2 * 2])
            return shape

        def check_dispatch_meta_args(ep_world_size, ep_rank_id, shared_expert_num, shared_expert_rank_num,
                                     expert_token_nums_type):
            torch._check(
                (ep_rank_id >= 0) and (ep_rank_id < ep_world_size),
                lambda: (
                    f"ep_rank_id should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {ep_rank_id=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (shared_expert_rank_num >= 0)
                and (shared_expert_rank_num < ep_world_size),
                lambda: (
                    f"shared_expert_rank_num should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {shared_expert_rank_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            is_shared_default = (shared_expert_num == 1) and (
                shared_expert_rank_num == 0
            )
            is_no_shared = (shared_expert_num == 0) and (shared_expert_rank_num == 0)
            is_valid_shared = (
                (shared_expert_num > 0)
                and ((shared_expert_rank_num // shared_expert_num) > 0)
                and ((shared_expert_rank_num % shared_expert_num) == 0)
            )
            torch._check(
                is_shared_default or is_no_shared or is_valid_shared,
                lambda: (
                    f"shared expert setting invalid, "
                    f"got {shared_expert_num=}, {shared_expert_rank_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                expert_token_nums_type in [0, 1],
                lambda: "the expert_token_nums_type should be 0 or 1"
                + ops_error(ErrCode.VALUE),
            )
            return is_shared_default, is_no_shared

        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_distribute_dispatch_meta(context, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num,
                                             ccl_buffer_size, scales=None, x_active_mask=None, expert_scales=None,
                                             elastic_info=None, performance_info=None, tp_world_size=0, tp_rank_id=0,
                                             expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0,
                                             quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="",
                                             zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None,
                                             x_dtype=None, x_smooth_scales_dtype=None):
            is_shared_default, is_no_shared = check_dispatch_meta_args(
                ep_world_size, ep_rank_id, shared_expert_num, shared_expert_rank_num, expert_token_nums_type
            )

            bs = x.size(0)
            h = x.size(1)
            k = expert_ids.size(1)
            shared_front = expert_shard_type == 0
            out_dtype = torch.int8
            local_moe_expert_num = 1
            global_bs_real = 0
            if global_bs == 0:
                global_bs_real = bs * ep_world_size
            else:
                global_bs_real = global_bs
            a = 0
            if shared_front:
                if ep_rank_id < shared_expert_rank_num:
                    local_moe_expert_num = 1
                    max_bs = global_bs_real // ep_world_size
                    rank_num_per_shared_expert = (
                        shared_expert_rank_num // shared_expert_num
                    )
                    max_shared_group_num = (
                        ep_world_size + rank_num_per_shared_expert - 1
                    ) // rank_num_per_shared_expert
                    a = max_bs * max_shared_group_num
                else:
                    local_moe_expert_num = moe_expert_num // (
                        ep_world_size - shared_expert_rank_num
                    )
                    a = global_bs_real * min(local_moe_expert_num, k)
                if elastic_info is not None:
                    if (is_shared_default) or (is_no_shared):
                        local_moe_expert_num = max(
                            local_moe_expert_num,
                            moe_expert_num // (ep_world_size - shared_expert_rank_num),
                        )
                        a = global_bs_real * min(local_moe_expert_num, k)
                    else:
                        max_bs = global_bs_real // ep_world_size
                        rank_num_per_shared_expert = (
                            shared_expert_rank_num // shared_expert_num
                        )
                        max_shared_group_num = (
                            ep_world_size + rank_num_per_shared_expert - 1
                        ) // rank_num_per_shared_expert
                        a = max(
                            max_bs * max_shared_group_num,
                            global_bs_real
                            * min(
                                moe_expert_num
                                // (ep_world_size - shared_expert_rank_num),
                                k,
                            ),
                        )
                        local_moe_expert_num = max(
                            local_moe_expert_num,
                            moe_expert_num // (ep_world_size - shared_expert_rank_num),
                        )

            ep_recv_cnt_num = 0
            if tp_world_size == 2:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size
            else:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num
            if quant_mode == 0:
                out_dtype = x.dtype
            if y_dtype is not None:
                out_dtype = get_dtype_from_enum(y_dtype)

            assist_info_for_combine_shape = max(bs * k, a * 128)
            expand_x_shape = tuple([max(a, a * tp_world_size), h])
            if y_dtype is not None and is_packed_float4(y_dtype) and scales is None:
                torch._check((h % 2) == 0, lambda: "h must be divisible by 2 when y_dtype is packed float4.")
                expand_x_shape = tuple([max(a, a * tp_world_size), h // 2])
            expand_x = x.new_empty(expand_x_shape, dtype=out_dtype)
            dynamic_scales_dtype = get_dispatch_dynamic_scales_dtype(x, scales, x_smooth_scales_dtype, quant_mode)
            if is_ascend950():
                dynamic_scales_shape = get_dispatch_dynamic_shape(scales, quant_mode, max(a, a * tp_world_size), h)
                dynamic_scales = x.new_empty(dynamic_scales_shape, dtype=dynamic_scales_dtype)
            elif tp_world_size == 0:
                dynamic_scales = x.new_empty((a), dtype=dynamic_scales_dtype)
            else:
                dynamic_scales = x.new_empty(
                    (a * tp_world_size), dtype=dynamic_scales_dtype
                )
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int64)
            ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
            tp_recv_counts = x.new_empty((tp_world_size), dtype=torch.int32)
            expand_scales = x.new_empty((0), dtype=torch.float32)
            if has_tensor(expert_scales):
                if not is_ascend950():
                    expert_scales_recv_extra = global_bs_real * 2 * k * (ep_world_size // 8)
                    ep_recv_cnt_num = ep_world_size * local_moe_expert_num + expert_scales_recv_extra
                    ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
                    assist_info_for_combine_shape = max(assist_info_for_combine_shape, expert_scales_recv_extra)
                expand_scales = x.new_empty((a), dtype=torch.float32)
            expand_idx = x.new_empty(assist_info_for_combine_shape, dtype=torch.int32)
            return (
                expand_x,
                dynamic_scales,
                expand_idx,
                expert_token_nums,
                ep_recv_counts,
                tp_recv_counts,
                expand_scales,
            )


class MoeDistributeCombineOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDistributeCombineOpBuilder, self).__init__(
            "npu_moe_distribute_combine"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/moe_distribute_combine.cpp"]

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

        @impl(AS_LIBRARY, self.name, "Meta")
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


moe_distribute_dispatch_op_builder = MoeDistributeDispatchOpBuilder()
moe_distribute_combine_op_builder = MoeDistributeCombineOpBuilder()


def _empty_topk_weights_like(tensor):
    return tensor.new_empty((0,), dtype=torch.float32)


def _get_hccl_comm_name(group, rank_id):
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


@impl(AS_LIBRARY, moe_distribute_dispatch_op_builder.name, "PrivateUse1")
def _npu_moe_distribute_dispatch(context, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num,
                                 ccl_buffer_size, scales=None, x_active_mask=None, expert_scales=None,
                                 elastic_info=None, performance_info=None, tp_world_size=0, tp_rank_id=0,
                                 expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0,
                                 quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="",
                                 zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None,
                                 x_dtype=None, x_smooth_scales_dtype=None):
    op_module = moe_distribute_dispatch_op_builder.load()
    return op_module.npu_moe_distribute_dispatch(context, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num,
                                                 ccl_buffer_size, scales, x_active_mask, expert_scales, elastic_info,
                                                 performance_info, tp_world_size, tp_rank_id,
                                                 expert_shard_type, shared_expert_num, shared_expert_rank_num,
                                                 quant_mode, global_bs, expert_token_nums_type, comm_alg,
                                                 zero_expert_num, copy_expert_num, const_expert_num, y_dtype,
                                                 x_dtype, x_smooth_scales_dtype)


@impl(AS_LIBRARY, moe_distribute_combine_op_builder.name, "PrivateUse1")
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


class MoeDistributeBuffer:
    def __init__(self, group, ccl_buffer_size: int = 0, comm_alg: int = 0):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self._ctx_manager = CommContextManager(self.group_name, self.world_size, backend={
            "Ascend910B": "kfc",
            "Ascend910_93": "kfc",
            "Ascend950": "channel"
        },
        customCclBufferSize=ccl_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
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

    def low_latency_dispatch(self, x, topk_idx, num_experts: int, *,
                             quant_mode=0, comm_alg="", x_smooth_scale=None,
                             x_active_mask=None, topk_weights=None, zero_expert_num=0, copy_expert_num=0,
                             const_expert_num=0, elastic_info=None, expert_shard_type=0, shared_expert_num=1,
                             shared_expert_rank_num=0, expert_token_nums_type=1, num_max_dispatch_tokens_per_rank=0,
                             y_dtype=None, x_dtype=None, x_smooth_scales_dtype=None):
        (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales) \
            = torch.ops.cann_ops_transformer.npu_moe_distribute_dispatch(
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
                                             x_smooth_scales_dtype=x_smooth_scales_dtype)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, expand_scales

    def low_latency_combine(self, x, topk_idx, topk_weights, assist_info_for_combine, ep_send_counts, *,
                            num_experts=0, comm_alg="", comm_quant_mode=0, x_active_mask=None, expand_scales=None,
                            shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None,
                            const_expert_alpha_2=None, const_expert_v=None, zero_expert_num=0, copy_expert_num=0,
                            const_expert_num=0, expert_shared_type=0, shared_expert_num=1, shared_expert_rank_num=0,
                            num_max_dispatch_tokens_per_rank=0):
        if assist_info_for_combine is None:
            raise ValueError("assist_info_for_combine must be provided.")
        if ep_send_counts is None:
            raise ValueError("ep_send_counts must be provided.")
        topk_weights = _empty_topk_weights_like(x) if topk_weights is None else topk_weights
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
