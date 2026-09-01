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
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommContextManager


class _AttentionToFfnOpBuilder(OpBuilder):
    def __init__(self):
        super(_AttentionToFfnOpBuilder, self).__init__(
            "npu_attention_to_ffn", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/attention_to_ffn.cpp"]

    def schema(self) -> str:
        return (
            "npu_attention_to_ffn(Tensor(a!) context, Tensor x, Tensor session_id, "
            "Tensor micro_batch_id, Tensor layer_id, Tensor expert_ids, Tensor expert_rank_table, "
            "str group, int world_size, "
            "int[] ffn_token_info_table_shape, int[] ffn_token_data_shape, int[] attn_token_info_table_shape, "
            "int moe_expert_num, int quant_mode, int sync_flag, int ffn_start_rank_id, int ccl_buffer_size, *, "
            "Tensor? scales=None, Tensor? active_mask=None) -> ()"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_attention_to_ffn_meta(
            context,
            x,
            session_id,
            micro_batch_id,
            layer_id,
            expert_ids,
            expert_rank_table,
            group,
            world_size,
            ffn_token_info_table_shape,
            ffn_token_data_shape,
            attn_token_info_table_shape,
            moe_expert_num,
            quant_mode,
            sync_flag,
            ffn_start_rank_id,
            ccl_buffer_size,
            scales=None,
            active_mask=None,
        ):
            torch._check(
                world_size > 0,
                lambda: (
                    f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                moe_expert_num > 0,
                lambda: (
                    f"moe_expert_num should be greater than 0, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                x.dim() == 3,
                lambda: (
                    f"x should be 3D, but got {x.dim()}D, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(ffn_token_info_table_shape) == 3,
                lambda: (
                    f"ffn_token_info_table_shape should have 3 elements, "
                    f"but got {len(ffn_token_info_table_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(ffn_token_data_shape) == 5,
                lambda: (
                    f"ffn_token_data_shape should have 5 elements, "
                    f"but got {len(ffn_token_data_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(attn_token_info_table_shape) == 3,
                lambda: (
                    f"attn_token_info_table_shape should have 3 elements, "
                    f"but got {len(attn_token_info_table_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )


_attention_to_ffn_op_builder = _AttentionToFfnOpBuilder()
_attention_to_ffn_op_builder._ensure_initialized()


@impl(get_as_library(), _attention_to_ffn_op_builder.name, "PrivateUse1")
def _npu_attention_to_ffn(
    context,
    x,
    session_id,
    micro_batch_id,
    layer_id,
    expert_ids,
    expert_rank_table,
    group,
    world_size,
    ffn_token_info_table_shape,
    ffn_token_data_shape,
    attn_token_info_table_shape,
    moe_expert_num,
    quant_mode,
    sync_flag,
    ffn_start_rank_id,
    ccl_buffer_size,
    scales=None,
    active_mask=None,
):
    _op_module = _attention_to_ffn_op_builder.load()
    _op_module.npu_attention_to_ffn(
        context,
        x,
        session_id,
        micro_batch_id,
        layer_id,
        expert_ids,
        expert_rank_table,
        scales,
        active_mask,
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        attn_token_info_table_shape,
        moe_expert_num,
        quant_mode,
        sync_flag,
        ffn_start_rank_id,
        ccl_buffer_size,
    )


def _get_hccl_comm_name(group, rank_id):
    """Get HCCL communicator name from a torch.distributed ProcessGroup."""
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


def _get_attention_to_ffn_ccl_buffer_size(
    ffn_token_info_table_shape: List[int],
    ffn_token_data_shape: List[int],
    quant_mode: int = 0,
) -> int:
    _op_module = _attention_to_ffn_op_builder.load()
    return _op_module.get_attention_to_ffn_ccl_buffer_size(
        ffn_token_info_table_shape, ffn_token_data_shape, quant_mode
    )


class AttentionToFfnBuffer:
    """Communication buffer for attention_to_ffn, wraps CommContextManager.

    Uses the HCCL Channel backend (HcclCommMemReg + HcclChannelGetRemoteMems)
    to allocate and exchange window memory, compatible with Ascend 950.
    """

    def __init__(
        self,
        group: "torch.distributed.ProcessGroup",
        world_size: int,
        ffn_token_info_table_shape: List[int],
        ffn_token_data_shape: List[int],
        quant_mode: int = 0,
    ):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size
        self.ffn_token_info_table_shape = list(ffn_token_info_table_shape)
        self.ffn_token_data_shape = list(ffn_token_data_shape)
        self.quant_mode = quant_mode

        required_buffer_size = _get_attention_to_ffn_ccl_buffer_size(
            ffn_token_info_table_shape, ffn_token_data_shape, quant_mode
        )
        self._ctx_manager = CommContextManager(
            self.group_name,
            self.ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            commAlg="urma",
            opName="attention_to_ffn",
            customCclBufferSize=required_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size

    def destroy(self):
        self._ctx_manager.destroy()

    def get_window_addr(self) -> int:
        """Get the local window memory device address from the context."""
        import ctypes

        if self.context is None:
            return 0

        host_tensor = self.context.cpu()
        context_ptr = host_tensor.data_ptr()
        if context_ptr == 0:
            return 0

        offset_epHcclBuffer = 2 * 4 + 8
        offset_target = offset_epHcclBuffer + self.rank_id * 8
        addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_target)
        return int(addr_bytes.value)


def get_buffer_for_attention_to_ffn(
    group: "torch.distributed.ProcessGroup",
    world_size: int,
    ffn_token_info_table_shape: List[int],
    ffn_token_data_shape: List[int],
    *,
    quant_mode: int = 0,
) -> AttentionToFfnBuffer:
    """Create a communication buffer for attention_to_ffn.

    This is a thin factory around ``AttentionToFfnBuffer`` that computes the
    required CCL buffer size, creates the communication context, and returns a
    buffer object holding all the information ``attention_to_ffn`` needs.

    Args:
        group (ProcessGroup): HCCL process group.
        world_size (int): Communication domain size.
        ffn_token_info_table_shape (List[int]): 3-element list.
        ffn_token_data_shape (List[int]): 5-element list
            [attnWorkerNum, microBatchNum, BS, K+shared, HS].
        quant_mode (int): Quantization mode. 0=none, 2=pertoken+INT8,
            3=mx+FP8_E5M2, 4=mx+FP8_E4M3, 5=mx+FP4_E2M1,
            6=mx_clip+FP8_E5M2, 7=mx_clip+FP8_E4M3. Default 0.

    Returns:
        AttentionToFfnBuffer: Buffer holding the communication context,
        CCL buffer size and the FFN-side token shapes.
    """
    return AttentionToFfnBuffer(
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        quant_mode=quant_mode,
    )


def attention_to_ffn(
    buffer: AttentionToFfnBuffer,
    x: torch.Tensor,
    session_id: torch.Tensor,
    micro_batch_id: torch.Tensor,
    layer_id: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_rank_table: torch.Tensor,
    attn_token_info_table_shape: List[int],
    moe_expert_num: int,
    *,
    sync_flag: int = 0,
    ffn_start_rank_id: int = 0,
    scales: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> None:
    """Send token data from Attention workers to FFN workers via HCCL window.

    Uses the CommContextManager-based context tensor to pass window memory
    addresses to the kernel, compatible with Ascend 950 (Channel backend).

    Args:
        buffer (AttentionToFfnBuffer): Communication buffer created by
            ``get_buffer_for_attention_to_ffn``. Holds the context tensor,
            group name, world size, FFN-side token shapes, quant mode and CCL
            buffer size.
        x (Tensor): Token data to send, shape (1, BS, H), dtype float16/bfloat16.
        session_id (Tensor): Session ID, shape (1,), dtype int32.
        micro_batch_id (Tensor): Micro batch ID, shape (1,), dtype int32.
        layer_id (Tensor): Layer ID, shape (1,), dtype int32.
        expert_ids (Tensor): Per-token topK expert indices, shape (1, BS, K), dtype int32.
        expert_rank_table (Tensor): Expert-to-rank mapping table,
            shape (1, expertNum, expRankTableM), dtype int32.
        attn_token_info_table_shape (List[int]): 3-element list.
        moe_expert_num (int): Number of MOE (routed) experts.
        sync_flag (int): Sync flag, 0=async, 1=sync. Default 0.
        ffn_start_rank_id (int): Starting rank ID of FFN workers. Default 0.
        scales (Tensor, optional): Smooth scales for quantization,
            shape (expertNum, expRankTableM, H), dtype float32.
        active_mask (Tensor, optional): Token activation mask,
            shape (1, BS), dtype bool.

    Returns:
        None. Data is sent to peer ranks via the HCCL window; no host-visible output.
    """
    torch.ops.cann_ops_transformer.npu_attention_to_ffn(
        buffer.context,
        x,
        session_id,
        micro_batch_id,
        layer_id,
        expert_ids,
        expert_rank_table,
        buffer.group_name,
        buffer.ep_world_size,
        buffer.ffn_token_info_table_shape,
        buffer.ffn_token_data_shape,
        attn_token_info_table_shape,
        moe_expert_num,
        buffer.quant_mode,
        sync_flag,
        ffn_start_rank_id,
        buffer.ccl_buffer_size,
        scales=scales,
        active_mask=active_mask,
    )
