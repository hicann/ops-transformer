# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


_ENGRAM_DTYPE_TO_INT = {
    torch.float16: 5,
    torch.float32: 6,
    torch.bfloat16: 15,
}
_ENGRAM_INT_TO_DTYPE = {v: k for k, v in _ENGRAM_DTYPE_TO_INT.items()}


class ElasticBufferOpBuilder(OpBuilder):
    """OpBuilder for ElasticBuffer operations"""

    def __init__(self):
        super(ElasticBufferOpBuilder, self).__init__("npu_elastic_buffer")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/elastic_buffer.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return [
            "engram_fetch(Tensor context, Tensor indices, int hidden_size, "
            "int num_entries, int dtype) -> Tensor",
            "engram_fetch_wait(Tensor context, Tensor fetched) -> Tensor",
        ]

    def register_meta(self):
        """Meta implementation for FakeTensor / torch.compile graph tracing."""

        @impl(AS_LIBRARY, "engram_fetch", "Meta")
        def engram_fetch_meta(context, indices, hidden_size, num_entries, dtype):
            return torch.empty(
                (indices.size(0), hidden_size),
                dtype=_ENGRAM_INT_TO_DTYPE[dtype],
                device="meta",
            )

        @impl(AS_LIBRARY, "engram_fetch_wait", "Meta")
        def engram_fetch_wait_meta(context, fetched):
            return torch.empty_like(fetched, device="meta")

    def extra_ldflags(self):
        """Extra link flags for HCCL and ACL libraries."""
        flags = super().extra_ldflags()
        flags.append("-L" + os.path.join(self._cann_path, "lib64"))
        flags.append("-lhcomm")
        flags.append("-lascendcl")
        return flags

    def include_paths(self):
        """Override include paths to ensure CANN headers are prioritized."""
        return [
            os.path.join(self._cann_path, "include"),
            os.path.join(self._torch_npu_path, "include"),
            os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
            os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
            os.path.join(self._package_path, "common/inc"),
        ]


_elastic_buffer_op_builder = ElasticBufferOpBuilder()


@impl(AS_LIBRARY, "engram_fetch", "PrivateUse1")
def engram_fetch(context, indices, hidden_size, num_entries, dtype):
    op_module = _elastic_buffer_op_builder.load()
    return op_module.ElasticBuffer.engram_fetch(
        context, indices, hidden_size, num_entries, dtype
    )


@impl(AS_LIBRARY, "engram_fetch_wait", "PrivateUse1")
def engram_fetch_wait(context, fetched):
    op_module = _elastic_buffer_op_builder.load()
    return op_module.ElasticBuffer.engram_fetch_wait(context, fetched)


def _inline_align(value: int, base: int) -> int:
    return (value + base - 1) // base * base


def _get_moe_ep_minimum_window_bytes(
    world_size: int,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
    topk: int,
) -> int:
    win_addr_align = 512
    ub_align = 32
    max_out_dtype_size = 2
    metadata_dtype_size = 4
    state_dtype_size = 4
    local_experts_num = num_experts // world_size

    dispatch_count_size = world_size * _inline_align(
        local_experts_num * state_dtype_size, win_addr_align
    )
    dispatch_notify_size = world_size * win_addr_align
    combine_state_size = (
        num_max_tokens_per_rank * topk * win_addr_align + world_size * win_addr_align
    )
    state_buffer_size = (
        dispatch_count_size + dispatch_notify_size * 2 + combine_state_size
    )

    metadata_bytes = _inline_align(topk * metadata_dtype_size, ub_align)
    hidden_align = _inline_align(hidden * max_out_dtype_size, ub_align)
    dispatch_per_slot_bytes = _inline_align(
        hidden_align + metadata_bytes * 2 + ub_align, win_addr_align
    )
    combine_per_slot_bytes = _inline_align(hidden_align + ub_align, win_addr_align)
    dispatch_buffer_size = (
        world_size * num_max_tokens_per_rank * dispatch_per_slot_bytes
    )
    combine_recv_buffer_size = num_max_tokens_per_rank * combine_per_slot_bytes * topk
    return state_buffer_size + dispatch_buffer_size * 2 + combine_recv_buffer_size


@dataclass
class EPHandle:
    dst_buffer_slot_idx: torch.Tensor
    recv_src_metadata: torch.Tensor
    num_recv_tokens_per_rank: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    topk_idx: torch.Tensor

    @property
    def num_recv_tokens(self) -> int:
        return int(self.num_recv_tokens_per_rank.sum().item())


@dataclass
class _DispatchArgs:
    x: torch.Tensor
    scales: Optional[torch.Tensor]
    topk_idx: torch.Tensor
    cached_dst_slot: Optional[torch.Tensor]
    cached_recv_src_metadata: Optional[torch.Tensor]
    num_experts: int
    num_max_tokens_per_rank: int
    expert_alignment: int
    do_cpu_sync: bool
    cached_recv_tokens: Optional[int]


class ElasticBuffer:
    """
    ElasticBuffer for distributed Engram storage management and MoE dispatch/combine operations.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        *,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: Optional[int] = None,
        hidden: Optional[int] = None,
        num_topk: Optional[int] = None,
    ):
        """
        Initialize the ElasticBuffer.

        Arguments:
            group: the distributed process group.
            num_cpu_bytes: the CPU buffer size in bytes (must be 2MB-aligned).
            num_max_tokens_per_rank: maximum MoE dispatch tokens per rank.
            hidden: hidden dimension for MoE dispatch/combine.
            num_topk: top-k value for MoE dispatch/combine.
        """
        buffer_alignment = 2 * 1024 * 1024
        torch._check((group is not None), lambda: ("group must not be None."))
        torch._check(
            (num_cpu_bytes >= 0 and num_cpu_bytes % buffer_alignment == 0),
            lambda: (
                f"num_cpu_bytes must be non-negative and 2MB-aligned, got {num_cpu_bytes=}, "
                f"which is not divisible by {buffer_alignment=}."
            ),
        )
        moe_args = (num_max_tokens_per_rank, hidden, num_topk)
        moe_arg_names = ("num_max_tokens_per_rank", "hidden", "num_topk")
        missing_args = [
            name for name, val in zip(moe_arg_names, moe_args) if val is None
        ]
        torch._check(
            len(missing_args) == 0 or len(missing_args) == len(moe_args),
            lambda: (
                "num_max_tokens_per_rank, hidden and num_topk "
                "must be specified together for MoE dispatch/combine, "
                f"missing: {', '.join(missing_args)}."
            ),
        )

        self._group = group
        self._num_cpu_bytes = num_cpu_bytes
        self._rank_id = dist.get_rank(self._group)
        self._ep_world_size = dist.get_world_size(self._group)

        backend = self._group._get_backend(torch.device("npu"))
        self._group_name = backend.get_hccl_comm_name(self._rank_id, init_comm=False)
        if not self._group_name:
            self._group_name = backend.get_hccl_comm_name(self._rank_id, init_comm=True)
        torch._check(
            self._group_name is not None and len(self._group_name) > 0,
            lambda: "HCCL comm name is empty, please check HCCL group initialization.",
        )
        _elastic_buffer_ops = _elastic_buffer_op_builder.load()
        self._runtime = _elastic_buffer_ops.ElasticBuffer(
            self._group_name, self._num_cpu_bytes
        )

        self._num_max_tokens_per_rank = num_max_tokens_per_rank
        self._hidden = hidden
        self._num_topk = num_topk
        self._host_pinned_counter = None
        self._engram_context_tensor = None
        self._engram_hidden_size = None
        self._engram_num_entries = None
        self._engram_dtype_int = None
        self._engram_fetch_in_progress = False

    @staticmethod
    def get_engram_storage_size_hint(
        num_entries: int, hidden: int, dtype: torch.dtype = torch.bfloat16
    ) -> int:
        """
        Get the minimum CPU buffer size required for Engram storage.
        The returned value is aligned to 2 MB.

        Arguments:
            num_entries: the number of entries in the Engram storage (must be non-negative).
            hidden: the hidden dimension of each entry (must be 128-aligned and positive).
            dtype: the data type, defaults to `torch.bfloat16`.

        Returns:
            num_cpu_bytes: the recommended CPU buffer size in bytes (2 MB-aligned).
        """
        torch._check(
            num_entries >= 0,
            lambda: f"num_entries must be non-negative, got {num_entries}",
        )
        torch._check(
            hidden > 0,
            lambda: f"hidden must be positive, got {hidden}",
        )
        torch._check(
            hidden % 128 == 0,
            lambda: f"hidden must be 128-aligned, got {hidden}",
        )
        torch._check(
            dtype in (torch.bfloat16, torch.float16, torch.float32),
            lambda: f"dtype must be bfloat16/float16/float32, got {dtype}",
        )
        _elastic_buffer_ops = _elastic_buffer_op_builder.load()
        return _elastic_buffer_ops.ElasticBuffer.get_engram_storage_size_hint(
            num_entries, hidden, dtype
        )

    @staticmethod
    def get_moe_ep_ccl_buffer_size(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        topk: int,
    ) -> int:
        torch._check(
            2 <= num_experts <= 2048,
            lambda: f"num_experts only support in [2, 2048], but got {num_experts=}.",
        )
        torch._check(
            1 <= topk <= 32,
            lambda: f"topk only support in [1, 32], but got {topk=}.",
        )

        mb_conversion = 1024 * 1024
        minimum_window_bytes = _get_moe_ep_minimum_window_bytes(
            world_size,
            num_max_tokens_per_rank,
            hidden,
            num_experts,
            topk,
        )
        return (
            _inline_align(
                _inline_align(minimum_window_bytes, mb_conversion) // mb_conversion,
                2,
            )
            // 2
        )

    def engram_write(self, storage: torch.Tensor) -> None:
        """
        Write data to the host pinned memory of ElasticBuffer.

        Arguments:
            storage: the CPU tensor to write (must be 2D, contiguous, dtype=bf16/fp16/fp32).

        Returns:
            None

        Note: barrier(with_device_sync=True) is called before and after write internally.
        """
        torch._check(
            storage.is_cpu,
            lambda: f"storage must be on CPU, got device: {storage.device}",
        )
        torch._check(
            storage.dim() == 2,
            lambda: f"storage must be 2D, got dimensions: {storage.dim()}",
        )
        torch._check(storage.is_contiguous(), lambda: "storage must be contiguous")
        torch._check(
            storage.dtype in (torch.bfloat16, torch.float16, torch.float32),
            lambda: f"storage dtype must be bfloat16/float16/float32, got: {storage.dtype}",
        )
        torch._check(
            storage.size(1) > 0,
            lambda: f"storage second dimension must be positive, got: {storage.size(1)}",
        )
        torch._check(
            storage.size(1) % 128 == 0,
            lambda: f"storage second dimension must be 128-aligned, got: {storage.size(1)}",
        )
        self._runtime.engram_write(storage)
        self._engram_context_tensor = self._runtime.get_context_tensor()
        self._engram_hidden_size = storage.size(1)
        self._engram_num_entries = storage.size(0)
        self._engram_dtype_int = _ENGRAM_DTYPE_TO_INT[storage.dtype]

    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Fetch Engram data from remote ranks via RDMA.

        Arguments:
            indices: the indices of entries to fetch (must be 1D NPU tensor with dtype=int32).

        Returns:
            wait_callable: a callable that returns the fetched tensor when invoked.
        """
        if indices.device.type != torch.device("npu").type:
            raise RuntimeError(f"indices must be on NPU, got device: {indices.device}")
        if indices.dim() != 1:
            raise RuntimeError(f"indices must be 1D, got dimensions: {indices.dim()}")
        if indices.dtype != torch.int32:
            raise RuntimeError(f"indices dtype must be int32, got: {indices.dtype}")
        if self._runtime is None:
            raise RuntimeError(
                "engram_fetch cannot be called after destroy, please create a new ElasticBuffer instance"
            )
        if self._engram_context_tensor is None:
            raise RuntimeError(
                "engram_fetch must be called after at least one engram_write"
            )
        if self._engram_fetch_in_progress:
            raise RuntimeError(
                "Cannot call engram_fetch while previous fetch callback is pending, "
                "please invoke the callback function returned by the previous engram_fetch first"
            )
        self._engram_fetch_in_progress = True
        fetched = torch.ops.cann_ops_transformer.engram_fetch(
            self._engram_context_tensor,
            indices,
            self._engram_hidden_size,
            self._engram_num_entries,
            self._engram_dtype_int,
        )
        context = self._engram_context_tensor

        def _wait():
            result = torch.ops.cann_ops_transformer.engram_fetch_wait(context, fetched)
            self._engram_fetch_in_progress = False
            return result

        return _wait

    def barrier(
        self, use_comm_stream: bool = True, with_cpu_sync: bool = False
    ) -> None:
        """
        Perform an NPU-level barrier across all ranks, optionally with CPU synchronization.

        Args:
            use_comm_stream: whether to dispatch the barrier on the dedicated comm stream
                (otherwise on the current compute stream).
            with_cpu_sync: whether to call `aclrtSynchronizeDevice` before and after the barrier
                to fully drain the device.
        """
        self._runtime.engram_barrier(use_comm_stream, with_cpu_sync)

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        EPHandle,
    ]:
        self._ensure_moe_config()
        torch._check(
            isinstance(x, torch.Tensor)
            or (
                isinstance(x, tuple)
                and len(x) == 2
                and all(isinstance(t, torch.Tensor) for t in x)
            ),
            lambda: f"x must be a tensor or a tuple of two tensors, but got {type(x).__name__}.",
        )
        torch._check(
            topk_idx is None or isinstance(topk_idx, torch.Tensor),
            lambda: f"topk_idx must be a tensor or None, but got {type(topk_idx).__name__}.",
        )
        torch._check(
            topk_weights is None or isinstance(topk_weights, torch.Tensor),
            lambda: f"topk_weights must be a tensor or None, but got {type(topk_weights).__name__}.",
        )
        torch._check(
            handle is None or isinstance(handle, EPHandle),
            lambda: f"handle must be an EPHandle or None, but got {type(handle).__name__}.",
        )
        if expert_alignment is not None:
            torch._check(
                isinstance(expert_alignment, int)
                and not isinstance(expert_alignment, bool),
                lambda: (
                    f"expert_alignment must be an integer, but got {type(expert_alignment).__name__}."
                ),
            )
            torch._check(
                expert_alignment == 1,
                lambda: (
                    f"expert_alignment only supports 1, but got {expert_alignment}."
                ),
            )
        if do_cpu_sync is not None:
            torch._check(
                isinstance(do_cpu_sync, bool),
                lambda: (
                    f"do_cpu_sync must be a boolean, but got {type(do_cpu_sync).__name__}."
                ),
            )
        args = self._prepare_dispatch_args(
            x,
            topk_idx,
            topk_weights,
            handle,
            num_experts,
            num_max_tokens_per_rank,
            expert_alignment,
            do_cpu_sync,
        )
        hp_addr = self._prepare_host_counter(args.do_cpu_sync)

        num_recv_per_rank, num_recv_per_expert, dst_slot = (
            self._runtime.moe_ep_dispatch(
                args.x,
                args.topk_idx,
                topk_weights,
                args.scales,
                args.cached_dst_slot,
                self._ep_world_size,
                self._rank_id,
                args.num_experts,
                args.num_max_tokens_per_rank,
                args.expert_alignment,
                args.do_cpu_sync,
                hp_addr,
            )
        )

        actual_a = self._get_dispatch_recv_count(args)
        recv_x, recv_src_meta, recv_topk_weights, recv_scales = (
            self._allocate_dispatch_outputs(args, actual_a, topk_weights)
        )

        recv_x, recv_src_meta, recv_topk_weights, recv_scales = (
            self._runtime.moe_ep_dispatch_epilogue(
                dst_slot,
                num_recv_per_rank,
                num_recv_per_expert,
                args.cached_recv_src_metadata,
                self._ep_world_size,
                self._rank_id,
                args.num_experts,
                args.num_max_tokens_per_rank,
                args.expert_alignment,
                recv_x,
                recv_src_meta,
                recv_topk_weights,
                recv_scales,
            )
        )

        recv_x = (recv_x, recv_scales) if recv_scales is not None else recv_x
        new_handle = self._make_dispatch_handle(
            args, dst_slot, recv_src_meta, num_recv_per_rank, num_recv_per_expert
        )
        return recv_x, None, recv_topk_weights, new_handle

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        *,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._ensure_moe_config()
        torch._check(
            isinstance(x, torch.Tensor),
            lambda: f"x must be a tensor, but got {type(x).__name__}.",
        )
        torch._check(
            isinstance(handle, EPHandle),
            lambda: f"handle must be an EPHandle, but got {type(handle).__name__}.",
        )
        torch._check(
            topk_weights is None or isinstance(topk_weights, torch.Tensor),
            lambda: f"topk_weights must be a tensor or None, but got {type(topk_weights).__name__}.",
        )
        torch._check(
            bias is None or isinstance(bias, (torch.Tensor, tuple)),
            lambda: f"bias must be a tensor, a tuple, or None, but got {type(bias).__name__}.",
        )
        bias_0, bias_1 = self._unpack_bias(bias)
        torch._check(
            ((bias_0 is None) and (bias_1 is None)),
            lambda: ("bias is not supported, please set bias to None."),
        )

        combined_x, combined_topk_weights = self._runtime.moe_ep_combine(
            x,
            handle.topk_idx,
            handle.recv_src_metadata,
            handle.num_recv_tokens_per_expert,
            topk_weights,
            bias_0,
            bias_1,
            self._ep_world_size,
            self._rank_id,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
        )
        return combined_x, combined_topk_weights

    def destroy(self) -> None:
        """
        Destroy the ElasticBuffer and free host pinned memory.

        Returns:
            None
        """
        if self._runtime is not None:
            self._runtime.destroy()
            self._runtime = None
        if self._host_pinned_counter is not None:
            del self._host_pinned_counter
        self._host_pinned_counter = None
        self._engram_context_tensor = None
        self._engram_hidden_size = None
        self._engram_num_entries = None
        self._engram_dtype_int = None
        self._engram_fetch_in_progress = False

    def _ensure_moe_config(self):
        moe_args = (self._num_max_tokens_per_rank, self._hidden, self._num_topk)
        moe_arg_names = ("num_max_tokens_per_rank", "hidden", "num_topk")
        missing_args = [
            name for name, val in zip(moe_arg_names, moe_args) if val is None
        ]
        torch._check(
            len(missing_args) == 0,
            lambda: (
                "num_max_tokens_per_rank, hidden and num_topk "
                "must be specified to use MoE dispatch/combine, "
                f"missing: {', '.join(missing_args)}."
            ),
        )

    def _prepare_dispatch_args(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        handle: Optional[EPHandle],
        num_experts: Optional[int],
        num_max_tokens_per_rank: Optional[int],
        expert_alignment: Optional[int],
        do_cpu_sync: Optional[bool],
    ) -> _DispatchArgs:
        x, scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            torch._check(
                (topk_idx is None), lambda: ("topk_idx is not supported when cached.")
            )
            torch._check(
                (topk_weights is None),
                lambda: ("topk_weights is not supported when cached."),
            )
            torch._check(
                ((do_cpu_sync is None) or (do_cpu_sync is False)),
                lambda: ("do_cpu_sync is not supported when cached."),
            )
            return _DispatchArgs(
                x,
                scales,
                handle.topk_idx,
                handle.dst_buffer_slot_idx,
                handle.recv_src_metadata,
                handle.num_experts,
                handle.num_max_tokens_per_rank,
                handle.expert_alignment,
                False,
                handle.recv_src_metadata.shape[0],
            )

        torch._check(
            (topk_idx is not None), lambda: ("topk_idx are required when no-cached.")
        )
        torch._check(
            (num_experts is not None),
            lambda: ("num_experts must be specified when no-cached."),
        )
        return _DispatchArgs(
            x,
            scales,
            topk_idx,
            None,
            None,
            num_experts,
            num_max_tokens_per_rank
            if num_max_tokens_per_rank is not None
            else self._num_max_tokens_per_rank,
            1 if expert_alignment is None else expert_alignment,
            True if do_cpu_sync is None else do_cpu_sync,
            None,
        )

    def _prepare_host_counter(self, do_cpu_sync: bool) -> int:
        if not do_cpu_sync:
            return 0
        if self._host_pinned_counter is None:
            _elastic_buffer_ops = _elastic_buffer_op_builder.load()
            self._host_pinned_counter = _elastic_buffer_ops.HostPinnedCounter()
        self._host_pinned_counter.reset()
        return self._host_pinned_counter.device_ptr()

    def _get_dispatch_recv_count(self, args: _DispatchArgs) -> int:
        if args.cached_recv_tokens is not None:
            return args.cached_recv_tokens
        if args.do_cpu_sync:
            return self._host_pinned_counter.spin_wait()
        return (
            self._ep_world_size
            * args.num_max_tokens_per_rank
            * min(self._num_topk, args.num_experts // self._ep_world_size)
        )

    def _allocate_dispatch_outputs(
        self, args: _DispatchArgs, actual_a: int, topk_weights: Optional[torch.Tensor]
    ):
        recv_x = torch.empty(
            (actual_a, self._hidden), dtype=args.x.dtype, device=args.x.device
        )
        recv_src_meta = torch.empty(
            (actual_a, 4), dtype=torch.int32, device=args.x.device
        )
        recv_topk_weights = (
            None
            if topk_weights is None
            else torch.empty((actual_a,), dtype=torch.float32, device=args.x.device)
        )
        recv_scales = (
            None
            if args.scales is None
            else torch.empty(
                (actual_a, args.scales.shape[1]),
                dtype=args.scales.dtype,
                device=args.x.device,
            )
        )
        return recv_x, recv_src_meta, recv_topk_weights, recv_scales

    def _make_dispatch_handle(
        self,
        args: _DispatchArgs,
        dst_slot: torch.Tensor,
        recv_src_meta: torch.Tensor,
        num_recv_per_rank: torch.Tensor,
        num_recv_per_expert: torch.Tensor,
    ) -> EPHandle:
        topk_idx = (
            args.topk_idx
            if args.cached_recv_tokens is not None
            else args.topk_idx.clone()
        )
        return EPHandle(
            dst_buffer_slot_idx=dst_slot,
            recv_src_metadata=recv_src_meta,
            num_recv_tokens_per_rank=num_recv_per_rank,
            num_recv_tokens_per_expert=num_recv_per_expert,
            num_experts=args.num_experts,
            expert_alignment=args.expert_alignment,
            num_max_tokens_per_rank=args.num_max_tokens_per_rank,
            topk_idx=topk_idx,
        )

    def _unpack_bias(self, bias):
        if bias is None:
            return None, None
        if isinstance(bias, torch.Tensor):
            return bias, None
        if isinstance(bias, tuple) and len(bias) == 2:
            return bias[0], bias[1]
        raise TypeError(f"unsupported bias type: {type(bias)}")
