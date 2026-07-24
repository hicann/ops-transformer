# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from .comm_context import CommContextManager


class _MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(_MegaMoeOpBuilder, self).__init__("npu_mega_moe")

    def sources(self):
        return ["ops/csrc/mega_moe.cpp"]

    def schema(self) -> str:
        return (
            "npu_mega_moe(Tensor context, Tensor x, Tensor topk_ids, Tensor topk_weights, "
            "Tensor[] weight1, Tensor[] weight2, int moe_expert_num, int ep_world_size, int ccl_buffer_size, *, "
            "Tensor[]? weight_scales1=None, Tensor[]? weight_scales2=None, "
            "Tensor[]? bias1=None, Tensor[]? bias2=None, "
            "Tensor? x_active_mask=None, "
            "Tensor[]? shared_weight1=None, Tensor[]? shared_weight2=None, "
            "Tensor[]? shared_weight_scales1=None, Tensor[]? shared_weight_scales2=None, "
            "Tensor[]? shared_bias1=None, Tensor[]? shared_bias2=None, "
            "int max_recv_token_num=0, "
            "int dispatch_quant_mode=0, int combine_quant_mode=0, "
            'str comm_alg="", int num_max_tokens_per_rank=0, str activation="swiglu", '
            "float? activation_clamp=None, "
            "int? dispatch_quant_out_dtype=None,  "
            "int? weight1_type=None, int? weight2_type=None, int? topo_type=None, "
            "int? rank_num_per_server=None, int topk_weights_type=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mega_moe_meta(
            context,
            x,
            topk_ids,
            topk_weights,
            weight1,
            weight2,
            moe_expert_num,
            ep_world_size,
            ccl_buffer_size,
            weight_scales1=None,
            weight_scales2=None,
            bias1=None,
            bias2=None,
            x_active_mask=None,
            shared_weight1=None,
            shared_weight2=None,
            shared_weight_scales1=None,
            shared_weight_scales2=None,
            shared_bias1=None,
            shared_bias2=None,
            max_recv_token_num=0,
            dispatch_quant_mode=0,
            combine_quant_mode=0,
            comm_alg="",
            num_max_tokens_per_rank=0,
            activation="swiglu",
            activation_clamp=None,
            dispatch_quant_out_dtype=None,
            weight1_type=None,
            weight2_type=None,
            topo_type=None,
            rank_num_per_server=None,
            topk_weights_type=0,
        ):
            torch._check(
                ep_world_size != 0,
                lambda: (f"ep_world_size should not be 0, {ops_error(ErrCode.VALUE)}."),
            )
            bs = x.size(0)
            h = x.size(1)
            local_moe_expert_num = moe_expert_num // ep_world_size
            y = x.new_empty(tuple([bs, h]), dtype=x.dtype)
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int32)
            return (y, expert_token_nums)


_mega_moe_op_builder = _MegaMoeOpBuilder()


@impl(AS_LIBRARY, _mega_moe_op_builder.name, "PrivateUse1")
def _npu_mega_moe(
    context,
    x,
    topk_ids,
    topk_weights,
    weight1,
    weight2,
    moe_expert_num,
    ep_world_size,
    ccl_buffer_size,
    weight_scales1=None,
    weight_scales2=None,
    bias1=None,
    bias2=None,
    x_active_mask=None,
    shared_weight1=None,
    shared_weight2=None,
    shared_weight_scales1=None,
    shared_weight_scales2=None,
    shared_bias1=None,
    shared_bias2=None,
    max_recv_token_num=0,
    dispatch_quant_mode=0,
    combine_quant_mode=0,
    comm_alg="",
    num_max_tokens_per_rank=0,
    activation="swiglu",
    activation_clamp=None,
    dispatch_quant_out_dtype=None,
    weight1_type=None,
    weight2_type=None,
    topo_type=None,
    rank_num_per_server=None,
    topk_weights_type=0,
):
    _op_module = _mega_moe_op_builder.load()
    return _op_module.npu_mega_moe(
        context,
        x,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        moe_expert_num,
        ep_world_size,
        ccl_buffer_size,
        weight_scales1,
        weight_scales2,
        bias1,
        bias2,
        x_active_mask,
        shared_weight1,
        shared_weight2,
        shared_weight_scales1,
        shared_weight_scales2,
        shared_bias1,
        shared_bias2,
        max_recv_token_num,
        dispatch_quant_mode,
        combine_quant_mode,
        comm_alg,
        num_max_tokens_per_rank,
        activation,
        activation_clamp,
        dispatch_quant_out_dtype,
        weight1_type,
        weight2_type,
        topo_type,
        rank_num_per_server,
        topk_weights_type,
    )


class SymmBuffer:
    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        max_recv_token_num: int = 0,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: Optional[torch.dtype] = None,
        combine_quant_mode: int = 0,
        comm_alg: str = "",
        topk_weights_type: int = 0,
    ):
        # Metadata
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False
        )
        self.ep_world_size = torch.distributed.get_world_size(group)
        required_ccl_buffer_size = _get_mega_moe_ccl_buffer_size(
            self.ep_world_size,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            max_recv_token_num,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
            combine_quant_mode,
            comm_alg,
        )
        self._ctx_manager = CommContextManager(
            self.group_name,
            self.ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            customCclBufferSize=required_ccl_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
        self.num_experts = num_experts
        self.max_recv_token_num = max_recv_token_num
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.dispatch_quant_mode = dispatch_quant_mode
        self.dispatch_quant_out_dtype = dispatch_quant_out_dtype
        self.combine_quant_mode = combine_quant_mode
        self.comm_alg = comm_alg
        self.topk_weights_type = topk_weights_type
        self.topo_type = self._ctx_manager.topo_type
        self.rank_num_per_server = self._ctx_manager.rank_num_per_server

    def destroy(self):
        self._ctx_manager.destroy()


_TORCH_DTYPE_TO_INT = {  # torch枚举
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
    torch.int8: 1,
}


def _dtype_to_int(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, int):
        return dtype
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_TO_INT:
            raise TypeError(f"Unsupported dispatch_quant_out_dtype: {dtype}.")
        return _TORCH_DTYPE_TO_INT[dtype]
    raise TypeError(
        f"dispatch_quant_out_dtype must be torch.dtype or int, got {type(dtype)}."
    )


def _get_mega_moe_ccl_buffer_size(
    ep_world_size: int,
    moe_expert_num: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: Optional[torch.dtype] = None,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
    topk_weights_type: int = 0,
) -> int:
    _op_module = _mega_moe_op_builder.load()
    quant_dtype_int = _dtype_to_int(dispatch_quant_out_dtype)  # 将torch.dtype转换为int
    return _op_module.get_mega_moe_ccl_buffer_size(
        ep_world_size,
        moe_expert_num,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        max_recv_token_num,
        dispatch_quant_mode,
        quant_dtype_int,
        combine_quant_mode,
        comm_alg,
        topk_weights_type,
    )


def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    *,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: Optional[torch.dtype] = None,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
    use_ccl_buffer: bool = True,
    topk_weights_type: int = 0,
) -> SymmBuffer:
    return SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        max_recv_token_num,
        dispatch_quant_mode,
        dispatch_quant_out_dtype,
        combine_quant_mode,
        comm_alg,
        topk_weights_type,
    )


def mega_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: List[torch.Tensor],
    l2_weights: List[torch.Tensor],
    sym_buffer: SymmBuffer,
    *,
    l1_weights_sf: Optional[List[torch.Tensor]] = None,
    l2_weights_sf: Optional[List[torch.Tensor]] = None,
    l1_bias: Optional[List[torch.Tensor]] = None,
    l2_bias: Optional[List[torch.Tensor]] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    weight1_type: Optional[int] = None,
    weight2_type: Optional[int] = None,
    shared_l1_weights: Optional[List[torch.Tensor]] = None,
    shared_l2_weights: Optional[List[torch.Tensor]] = None,
    shared_l1_weights_sf: Optional[List[torch.Tensor]] = None,
    shared_l2_weights_sf: Optional[List[torch.Tensor]] = None,
    shared_l1_bias: Optional[List[torch.Tensor]] = None,
    shared_l2_bias: Optional[List[torch.Tensor]] = None,
):
    return torch.ops.cann_ops_transformer.npu_mega_moe(
        sym_buffer.context,
        x,
        topk_ids,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer.num_experts,
        sym_buffer.ep_world_size,
        sym_buffer.ccl_buffer_size,
        weight_scales1=l1_weights_sf,
        weight_scales2=l2_weights_sf,
        bias1=l1_bias,
        bias2=l2_bias,
        x_active_mask=x_active_mask,
        shared_weight1=shared_l1_weights,
        shared_weight2=shared_l2_weights,
        shared_weight_scales1=shared_l1_weights_sf,
        shared_weight_scales2=shared_l2_weights_sf,
        shared_bias1=shared_l1_bias,
        shared_bias2=shared_l2_bias,
        max_recv_token_num=sym_buffer.max_recv_token_num,
        dispatch_quant_mode=sym_buffer.dispatch_quant_mode,
        combine_quant_mode=sym_buffer.combine_quant_mode,
        comm_alg=sym_buffer.comm_alg,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        activation=activation,
        activation_clamp=activation_clamp,
        dispatch_quant_out_dtype=sym_buffer.dispatch_quant_out_dtype,
        weight1_type=weight1_type,
        weight2_type=weight2_type,
        topo_type=sym_buffer.topo_type,
        rank_num_per_server=sym_buffer.rank_num_per_server,
        topk_weights_type=sym_buffer.topk_weights_type,
    )
