# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numbers
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


def _is_integer_list(value) -> bool:
    """Return whether value is an integer list-like shape."""
    if isinstance(value, (str, bytes, bytearray, memoryview, dict, torch.Tensor)):
        return False
    if not hasattr(value, "__iter__") or not hasattr(value, "__len__"):
        return False
    return all(isinstance(item, numbers.Integral) for item in value)


def _validate_buffer_types(
    world_size,
    ffn_token_info_table_shape,
    ffn_token_data_shape,
    window_addr,
    window_size,
) -> None:
    """Validate buffer argument types before using their values."""
    if not isinstance(world_size, numbers.Integral):
        raise TypeError("world_size must be an integer.")
    if not _is_integer_list(ffn_token_info_table_shape):
        raise TypeError("ffn_token_info_table_shape must be a list of integers.")
    if not _is_integer_list(ffn_token_data_shape):
        raise TypeError("ffn_token_data_shape must be a list of integers.")
    if window_addr is not None and not isinstance(window_addr, numbers.Integral):
        raise TypeError("window_addr must be an integer when provided.")
    if window_size is not None and not isinstance(window_size, numbers.Integral):
        raise TypeError("window_size must be an integer when provided.")


_PAYLOAD_ALIGN = 512  # payload slot stride alignment (matches kernel WIN_ALIGN)
_STAGING_ENTRY_ALIGN = 32  # staging/flag entry alignment (matches kernel UB_ALIGN)
_LAYER_FLAG_BYTES = (
    8  # layer flag block (kernel TOKEN_INFO_TABLE_COPY_BLOCK_CNT * int32)
)
_NO_QUANT_MODE = 0
_BYTES_PER_FP16 = 2
_BYTES_PER_INT32 = 4
_BYTES_PER_QUANT_ELEM = (
    1  # INT8/FP8 = 1 byte, FP4-packed HS already accounts for packing
)
_INT64_MAX = (1 << 63) - 1
_MIN_QUANT_ATTN_WORKER_NUM = 2


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _calc_payload_slot_stride(ffn_token_data_shape: List[int], quant_mode: int) -> int:
    """Per-slot bytes of the persistent payload region: [staging region | local flags | layer flag].

    Mirrors the kernel's slot layout in op_kernel/attention_to_ffn_urma.h but bounds the
    per-token payload by HS * bytesPerElem (tiling enforces HS * elemSize >= the kernel's
    commuBytes) and the send-pair count by BS * (K + shared); the kernel asserts the context
    payloadSlotStride covers its needs and fails loudly otherwise.
    """
    micro_batch_num, bs, k_plus_shared, hs = ffn_token_data_shape[1:]
    torch._check(
        micro_batch_num > 0 and bs > 0 and k_plus_shared > 0 and hs > 0,
        lambda: (
            f"ffn_token_data_shape dims [microBatchNum, BS, K+shared, HS] should be positive, but got "
            f"[{micro_batch_num}, {bs}, {k_plus_shared}, {hs}], {ops_error(ErrCode.VALUE)}."
        ),
    )
    bytes_per_elem = (
        _BYTES_PER_FP16 if quant_mode == _NO_QUANT_MODE else _BYTES_PER_QUANT_ELEM
    )
    staging_stride = _align_up(hs * bytes_per_elem, _STAGING_ENTRY_ALIGN)
    max_send_pairs = bs * k_plus_shared
    staging_region = _align_up(max_send_pairs * staging_stride, _PAYLOAD_ALIGN)
    flag_region = _align_up(max_send_pairs * _BYTES_PER_INT32, _STAGING_ENTRY_ALIGN)
    return _align_up(staging_region + flag_region + _LAYER_FLAG_BYTES, _PAYLOAD_ALIGN)


class AttentionToFfnBuffer:
    """Communication buffer for attention_to_ffn, wraps CommContextManager.

    Uses the HCCL Channel backend (HcclCommMemReg + HcclChannelGetRemoteMems)
    to allocate and exchange window memory, compatible with Ascend 950.

    When ``window_addr`` and ``window_size`` are provided, the user memory at
    ``window_addr`` (``window_size`` bytes, the attention-side window on this
    card) is used as the local communication window instead of an internally
    allocated buffer: it is zeroed once, registered into the HCCL
    communication domain, and never freed by this class. The caller owns the
    memory and must keep it alive while the buffer is in use. The FFN-side CCL
    buffer size is always computed internally from the FFN-side token shapes
    and exposed as ``ccl_buffer_size`` (the size of the remote FFN windows the
    attention_to_ffn op writes to).

    The buffer also owns a persistent payload region (``payload_slot_stride`` bytes per
    slot, ``payload_slot_num`` slots, manager-allocated and slotted by microBatchId): the
    kernel's relay WQEs read the staged token rows and flags asynchronously after the
    kernel returns, so these regions must live in buffer-owned memory instead of the
    per-call workspace. Different microbatches use different slots, so consecutive
    launches overlap without any drain; a slot is reused only a full pipeline round
    later, after the remote side consumed the old round. Call ``destroy()`` (or drop the
    last reference) to synchronize the device and release everything.
    """

    def __init__(
        self,
        group: "torch.distributed.ProcessGroup",
        world_size: int,
        ffn_token_info_table_shape: List[int],
        ffn_token_data_shape: List[int],
        quant_mode: int = 0,
        window_addr: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        _validate_buffer_types(
            world_size,
            ffn_token_info_table_shape,
            ffn_token_data_shape,
            window_addr,
            window_size,
        )
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size
        self.ffn_token_info_table_shape = list(ffn_token_info_table_shape)
        self.ffn_token_data_shape = list(ffn_token_data_shape)
        self.quant_mode = quant_mode

        torch._check(
            (window_addr is None) == (window_size is None),
            lambda: (
                f"window_addr and window_size must be provided together, but got "
                f"window_addr={window_addr}, window_size={window_size}, {ops_error(ErrCode.VALUE)}."
            ),
        )
        if window_addr is not None:
            torch._check(
                window_addr > 0,
                lambda: (
                    f"window_addr should be a positive device address, but got {window_addr}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                window_size > 0,
                lambda: (
                    f"window_size should be greater than 0, but got {window_size}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

        # FFN-side CCL buffer size, computed internally from the FFN-side token
        self.ccl_buffer_size = _get_attention_to_ffn_ccl_buffer_size(
            ffn_token_info_table_shape, ffn_token_data_shape, quant_mode
        )
        if quant_mode != _NO_QUANT_MODE:
            torch._check(
                self.ffn_token_data_shape[0] >= _MIN_QUANT_ATTN_WORKER_NUM,
                lambda: (
                    f"ffn_token_data_shape[0] (attnWorkerNum) should be >= {_MIN_QUANT_ATTN_WORKER_NUM} in "
                    f"quant mode (quant_mode={quant_mode}), because the quantized data region must host the "
                    f"ffn_to_attention return data, but got {self.ffn_token_data_shape[0]}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
        # Local attention-side window: user-provided memory
        self.window_addr = window_addr
        self.window_size = (
            window_size if window_size is not None else self.ccl_buffer_size
        )

        # Persistent payload region (WQE-source regions, microBatchId slotted, allocated by the
        # CommContextManager): relay WQEs read these regions asynchronously after the kernel
        # returns, so they must live in buffer-owned memory instead of the per-call workspace
        # (which the framework may reuse once the op returns). Different microbatches use
        # different slots, so consecutive launches overlap without a drain; a slot is reused
        # only a full pipeline round later, after the remote side consumed the old round.
        self.payload_slot_num = ffn_token_data_shape[1]
        self.payload_slot_stride = _calc_payload_slot_stride(
            ffn_token_data_shape, quant_mode
        )
        torch._check(
            self.payload_slot_stride <= _INT64_MAX // self.payload_slot_num,
            lambda: (f"total payload size exceeds int64, {ops_error(ErrCode.VALUE)}."),
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
            customCclBufferSize=self.window_size,
            customDeviceBufferAddr=window_addr if window_addr is not None else 0,
            multiChannel=True,
            payloadBufferSize=self.payload_slot_stride * self.payload_slot_num,
            payloadSlotNum=self.payload_slot_num,
        )
        self.context = self._ctx_manager.create_context()
        self._device = self.context.device

    def destroy(self):
        """Release the buffer and every retained resource.

        Retires every in-flight WQE with a device synchronization first: the last round's
        WQEs may still read the persistent payload region after the final kernel returned,
        so the payload memory (and the communication context) must only be released afterwards.
        """
        if self.context is None:
            return
        torch.npu.synchronize(self._device)
        self._ctx_manager.destroy()
        self.context = None

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
    window_addr: Optional[int] = None,
    window_size: Optional[int] = None,
) -> AttentionToFfnBuffer:
    """Create a communication buffer for attention_to_ffn.

    This is a thin factory around ``AttentionToFfnBuffer`` that computes the
    required FFN-side CCL buffer size internally, creates the communication
    context, and returns a buffer object holding all the information
    ``attention_to_ffn`` needs.

    Args:
        group (ProcessGroup): HCCL process group.
        world_size (int): Communication domain size.
        ffn_token_info_table_shape (List[int]): 3-element list.
        ffn_token_data_shape (List[int]): 5-element list
            [attnWorkerNum, microBatchNum, BS, K+shared, HS].
        quant_mode (int): Quantization mode. 0=none, 2=pertoken+INT8,
            3=mx+FP8_E5M2, 4=mx+FP8_E4M3, 5=mx+FP4_E2M1,
            6=mx_clip+FP8_E5M2, 7=mx_clip+FP8_E4M3. Default 0.
        window_addr (int, optional): Device address of the user-provided local
            (attention-side) communication window memory on this card. Must be
            provided together with ``window_size``. The memory is zeroed once,
            registered into the HCCL communication domain and never freed by
            the framework; the caller must keep it alive while the buffer is
            in use. Only supported on the channel backend (Ascend950).
            Default None, meaning the framework allocates the window
            internally.
        window_size (int, optional): Size in bytes of the memory at
            ``window_addr``, i.e. the attention-side memory size. It must be
            no smaller than the attention-side window size required to receive
            the ffn_to_attention token data (computed from
            [microBatchNum, BS, expertNumPerToken] and
            [microBatchNum, BS, expertNumPerToken, HS], 2MB aligned). The
            FFN-side memory size is computed internally from
            ``ffn_token_info_table_shape``/``ffn_token_data_shape`` and
            ``quant_mode`` and exposed as ``ccl_buffer_size`` of the returned
            buffer. Default None, meaning the framework allocates the window
            internally.

    Returns:
        AttentionToFfnBuffer: Buffer holding the communication context,
        CCL buffer size and the FFN-side token shapes. The buffer also owns
        a persistent payload region (``payload_slot_stride`` bytes per slot,
        ``payload_slot_num`` = microBatchNum slots, slotted by microBatchId)
        that the kernel's async relay WQEs read after each launch returns;
        it is held until ``destroy()`` (which synchronizes the device first)
        and must not be released while sends are in flight.
    """
    return AttentionToFfnBuffer(
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        quant_mode=quant_mode,
        window_addr=window_addr,
        window_size=window_size,
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
            buffer size (the FFN-side window size, computed internally).
        x (Tensor): Token data to send, shape (1, BS, H), dtype float16/bfloat16.
        session_id (Tensor): Session ID, shape (1,), dtype int32.
        micro_batch_id (Tensor): Micro batch ID, shape (1,), dtype int32.
        layer_id (Tensor): Layer ID, shape (1,), dtype int32.
        expert_ids (Tensor): Per-token topK expert indices, shape (1, BS, K), dtype int32.
        expert_rank_table (Tensor): Expert-to-rank mapping table,
            shape (L, expertNum, expRankTableM), dtype int32.
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

    Note:
        The send is asynchronous: the kernel may return while the communication engine is
        still reading ``x`` (non-quant direct send reads the x GM rows through the relay
        WQEs). The caller is responsible for keeping the input tensors alive until the
        sends complete, e.g. by holding the references or synchronizing the device.
    """
    torch._check(
        buffer.context is not None,
        lambda: (
            "attention_to_ffn cannot be called after the buffer has been destroyed, "
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    torch._check(
        x.device == buffer._device,
        lambda: (
            "attention_to_ffn requires x and buffer to be on the same device, "
            f"but got x.device={x.device}, buffer.device={buffer._device}, "
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )

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
