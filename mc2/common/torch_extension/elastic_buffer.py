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
from torch import _check as _torch_check

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


_ENGRAM_DTYPE_TO_INT = {
    torch.float16: 5,
    torch.float32: 6,
    torch.bfloat16: 15,
}
_ENGRAM_INT_TO_DTYPE = {v: k for k, v in _ENGRAM_DTYPE_TO_INT.items()}


@dataclass
class EngramFetchCtx:
    """save-for-backward context for engram_fetch_grad.

    Attributes:
        perm: (T,) int32 — bucket permutation index; backward step 1 reorders grad by this.
        send_counts: (W*8,) int32 — number of indices sent to each rank (padded to 32b alignment);
            only the first W elements are valid, used as a2a send counts in backward.
        recv_counts: (W,) int32 — number of indices received from each rank;
            used as a2a recv counts in backward.
        recv_local_entry: (R_max,) int32 — received local entries;
            backward step 3 scatter-add aggregates by this.
        num_recv: (1,) int32 — actual number of received rows R (first num_recv are valid).
    """

    perm: torch.Tensor
    send_counts: torch.Tensor
    recv_counts: torch.Tensor
    recv_local_entry: torch.Tensor
    num_recv: torch.Tensor


class ElasticBufferOpBuilder(OpBuilder):
    """OpBuilder for ElasticBuffer operations"""

    def __init__(self):
        super(ElasticBufferOpBuilder, self).__init__(
            "npu_elastic_buffer", category="mc2"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/mc2/elastic_buffer.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return [
            "engram_fetch(Tensor context, Tensor indices, int hidden_size, "
            "int num_entries, int dtype) -> Tensor",
            "engram_fetch_train(Tensor context, Tensor indices, int hidden_size, "
            "int num_entries, int dtype, Tensor local_storage_addr, "
            "int num_max_tokens_per_rank, int comm_buffer_size, int rank_size) "
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
            "engram_fetch_wait(Tensor context, Tensor fetched) -> Tensor",
            "engram_fetch_grad(Tensor context, Tensor grad_fetched, Tensor perm, Tensor send_counts, "
            "Tensor recv_counts, Tensor recv_local_entry, Tensor num_recv, int num_entries, "
            "int comm_buffer_size, int num_max_tokens_per_rank, int rank_size) "
            "-> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """Meta implementation for FakeTensor / torch.compile graph tracing."""

        @impl(get_as_library(), "engram_fetch", "Meta")
        def engram_fetch_meta(context, indices, hidden_size, num_entries, dtype):
            return torch.empty(
                (indices.size(0), hidden_size),
                dtype=_ENGRAM_INT_TO_DTYPE[dtype],
                device="meta",
            )

        @impl(get_as_library(), "engram_fetch_train", "Meta")
        def engram_fetch_train_meta(*args):
            (
                _context,
                indices,
                hidden_size,
                _num_entries,
                dtype,
                _local_storage_addr,
                num_max_tokens_per_rank,
                _comm_buffer_size,
                rank_size,
            ) = args
            num_tokens = indices.size(0)
            max_r = num_max_tokens_per_rank * rank_size
            data_dtype = _ENGRAM_INT_TO_DTYPE[dtype]
            fetched = torch.empty(
                (num_tokens, hidden_size), dtype=data_dtype, device="meta"
            )
            perm = torch.empty((num_tokens,), dtype=torch.int32, device="meta")
            send_counts = torch.empty(
                (rank_size * 8,), dtype=torch.int32, device="meta"
            )
            recv_counts = torch.empty((rank_size,), dtype=torch.int32, device="meta")
            recv_local_entry = torch.empty((max_r,), dtype=torch.int32, device="meta")
            num_recv = torch.empty((1,), dtype=torch.int32, device="meta")
            fetch_result = (
                fetched,
                perm,
                send_counts,
                recv_counts,
                recv_local_entry,
                num_recv,
            )
            return fetch_result

        @impl(get_as_library(), "engram_fetch_wait", "Meta")
        def engram_fetch_wait_meta(context, fetched):
            return torch.empty_like(fetched, device="meta")

        @impl(get_as_library(), "engram_fetch_grad", "Meta")
        def engram_fetch_grad_meta(
            context,
            grad_fetched,
            perm,
            send_counts,
            recv_counts,
            recv_local_entry,
            num_recv,
            num_entries,
            comm_buffer_size,
            num_max_tokens_per_rank,
            rank_size,
        ):
            # Outputs are NOT narrowed; full maxR rows. Caller narrows by num_unique.
            num_max_tokens_per_rank = (
                1 if num_max_tokens_per_rank <= 0 else num_max_tokens_per_rank
            )
            max_r = num_max_tokens_per_rank * rank_size
            grad_unique = torch.empty(
                (max_r, grad_fetched.size(1)),
                dtype=grad_fetched.dtype,
                device="meta",
            )
            unique_local_entry = torch.empty((max_r,), dtype=torch.int32, device="meta")
            num_unique = torch.empty((1,), dtype=torch.int32, device="meta")
            return (grad_unique, unique_local_entry, num_unique)

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
            os.path.join(self._package_path, "common"),
        ]


_elastic_buffer_op_builder = ElasticBufferOpBuilder()
_elastic_buffer_op_builder._ensure_initialized()


@impl(get_as_library(), "engram_fetch", "PrivateUse1")
def engram_fetch(context, indices, hidden_size, num_entries, dtype):
    op_module = _elastic_buffer_op_builder.load()
    return op_module.ElasticBuffer.engram_fetch(
        context, indices, hidden_size, num_entries, dtype
    )


@impl(get_as_library(), "engram_fetch_train", "PrivateUse1")
def engram_fetch_train(*args):
    op_module = _elastic_buffer_op_builder.load()
    return op_module.ElasticBuffer.engram_fetch_train(*args)


@impl(get_as_library(), "engram_fetch_wait", "PrivateUse1")
def engram_fetch_wait(context, fetched):
    op_module = _elastic_buffer_op_builder.load()
    return op_module.ElasticBuffer.engram_fetch_wait(context, fetched)


@impl(get_as_library(), "engram_fetch_grad", "PrivateUse1")
def engram_fetch_grad(
    context,
    grad_fetched,
    perm,
    send_counts,
    recv_counts,
    recv_local_entry,
    num_recv,
    num_entries,
    comm_buffer_size,
    num_max_tokens_per_rank,
    rank_size,
):
    op_module = _elastic_buffer_op_builder.load()
    return op_module.ElasticBuffer.engram_fetch_grad_op(
        context,
        grad_fetched,
        perm,
        send_counts,
        recv_counts,
        recv_local_entry,
        num_recv,
        num_entries,
        comm_buffer_size,
        num_max_tokens_per_rank,
        rank_size,
    )


def _inline_align(value: int, base: int) -> int:
    return (value + base - 1) // base * base


@dataclass(frozen=True)
class _MoeEpWindowLayout:
    state_buffer_bytes: int
    dispatch_slot_bytes: int
    combine_slot_bytes: int
    scaleup_receive_buffer_bytes: int


def _get_moe_ep_window_layout(
    world_size: int,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_experts: int,
    topk: int,
) -> _MoeEpWindowLayout:
    win_addr_align = 512
    ub_align = 32
    notify_cnt_align = 15000
    max_out_dtype_size = 2
    metadata_dtype_size = 4
    state_dtype_size = 4
    dump_metadata_bytes = 64 * 1024
    per_core_diag_bytes = 100 * win_addr_align
    local_experts_num = num_experts // world_size

    dispatch_count_size = world_size * _inline_align(
        local_experts_num * state_dtype_size, win_addr_align
    )
    dispatch_notify_count = (
        _inline_align(num_max_tokens_per_rank, notify_cnt_align) // notify_cnt_align
    )
    dispatch_notify_size = (
        world_size * win_addr_align
        + world_size * dispatch_notify_count * win_addr_align
    )
    combine_state_size = (
        num_max_tokens_per_rank * topk * win_addr_align + world_size * win_addr_align
    )
    state_buffer_size = (
        dump_metadata_bytes
        + per_core_diag_bytes
        + dispatch_count_size
        + dispatch_notify_size
        + combine_state_size
    )

    metadata_bytes = _inline_align(topk * metadata_dtype_size, ub_align)
    hidden_align = _inline_align(hidden * max_out_dtype_size, ub_align)
    dispatch_per_slot_bytes = _inline_align(
        hidden_align + metadata_bytes * 2 + ub_align, win_addr_align
    )
    combine_per_slot_bytes = _inline_align(hidden_align + ub_align, win_addr_align)
    scaleup_receive_buffer_bytes = (
        world_size * num_max_tokens_per_rank * dispatch_per_slot_bytes
    )
    return _MoeEpWindowLayout(
        state_buffer_bytes=state_buffer_size,
        dispatch_slot_bytes=dispatch_per_slot_bytes,
        combine_slot_bytes=combine_per_slot_bytes,
        scaleup_receive_buffer_bytes=scaleup_receive_buffer_bytes,
    )


def _get_moe_ep_direct_window_bytes(
    layout: _MoeEpWindowLayout,
    num_max_tokens_per_rank: int,
    topk: int,
) -> int:
    combine_receive_buffer_bytes = (
        num_max_tokens_per_rank * layout.combine_slot_bytes * topk
    )
    return (
        layout.state_buffer_bytes
        + layout.scaleup_receive_buffer_bytes * 2
        + combine_receive_buffer_bytes
    )


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
    route_count: Optional[torch.Tensor] = None
    route_dst_scaleout: Optional[torch.Tensor] = None
    route_scaleout_slot: Optional[torch.Tensor] = None

    @property
    def num_recv_tokens(self) -> int:
        return int(self.num_recv_tokens_per_rank.sum().item())


@dataclass
class _DispatchArgs:
    x: torch.Tensor
    scales: Optional[torch.Tensor]
    topk_idx: torch.Tensor
    cached_dst_slot_idx: Optional[torch.Tensor]
    cached_route_count: Optional[torch.Tensor]
    cached_route_dst_scaleout: Optional[torch.Tensor]
    cached_route_scaleout_slot: Optional[torch.Tensor]
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
        with_grad: bool = False,
        explicitly_destroy: bool = False,
    ):
        """
        Initialize the ElasticBuffer.

        Arguments:
            group: the distributed process group.
            num_cpu_bytes: the CPU buffer size in bytes (must be 2MB-aligned).
            num_max_tokens_per_rank: maximum MoE dispatch tokens per rank.
            hidden: hidden dimension for MoE dispatch/combine.
            num_topk: top-k value for MoE dispatch/combine.
            with_grad: whether to enable training mode.
            explicitly_destroy: if True, the caller needs to explicitly
                invoke ``destroy`` to release resources; if False (default),
                resources are released automatically when the instance is
                garbage collected.
        """
        moe_args = (num_max_tokens_per_rank, hidden, num_topk)
        self._validate_init_args(group, num_cpu_bytes, moe_args, with_grad)

        self._group = group
        self._num_cpu_bytes = num_cpu_bytes
        self._with_grad = with_grad
        self._explicitly_destroy = explicitly_destroy
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
            self._group_name,
            self._num_cpu_bytes,
            num_max_tokens_per_rank if num_max_tokens_per_rank is not None else 0,
            self._with_grad,
            self._explicitly_destroy,
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
        self._local_storage_addr = None
        self._comm_buffer_size = 0
        self._rank_size = 0

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
        window_layout = _get_moe_ep_window_layout(
            world_size,
            num_max_tokens_per_rank,
            hidden,
            num_experts,
            topk,
        )
        direct_minimum_window_bytes = _get_moe_ep_direct_window_bytes(
            window_layout,
            num_max_tokens_per_rank,
            topk,
        )
        win_addr_align = 512
        state_dtype_size = 4

        def get_dispatch_buffer_size(rank_num_per_server):
            scaleout_rank_count = world_size // rank_num_per_server
            scaleout_per_slot_bytes = _inline_align(
                window_layout.dispatch_slot_bytes + topk * state_dtype_size,
                win_addr_align,
            )
            scaleout_recv_data_size = (
                scaleout_rank_count * num_max_tokens_per_rank * scaleout_per_slot_bytes
            )
            scaleout_recv_status_size = (
                scaleout_rank_count * num_max_tokens_per_rank * win_addr_align
            )
            payload_stash_size = num_max_tokens_per_rank * scaleout_per_slot_bytes
            return (
                scaleout_recv_data_size
                + window_layout.scaleup_receive_buffer_bytes
                + scaleout_recv_status_size
                + payload_stash_size
            )

        # Context topology is unavailable here, so reserve for the largest valid hybrid layout.
        rank_num_per_server_candidates = [
            rank_num_per_server
            for rank_num_per_server in range(1, world_size + 1)
            if world_size % rank_num_per_server == 0
        ]
        dispatch_buffer_size = max(
            get_dispatch_buffer_size(rank_num_per_server)
            for rank_num_per_server in rank_num_per_server_candidates
        )
        combine_buffer_size = (
            num_max_tokens_per_rank * window_layout.combine_slot_bytes * topk
        )

        hybrid_minimum_window_bytes = (
            window_layout.state_buffer_bytes
            + dispatch_buffer_size
            + combine_buffer_size
        )
        minimum_window_bytes = max(
            direct_minimum_window_bytes, hybrid_minimum_window_bytes
        )
        return (
            _inline_align(
                _inline_align(minimum_window_bytes, mb_conversion) // mb_conversion,
                2,
            )
            // 2
        )

    @staticmethod
    def _validate_init_args(group, num_cpu_bytes, moe_args, with_grad):
        buffer_alignment = 2 * 1024 * 1024
        _torch_check((group is not None), lambda: ("group must not be None."))
        _torch_check(
            isinstance(num_cpu_bytes, int) and not isinstance(num_cpu_bytes, bool),
            lambda: (
                f"num_cpu_bytes must be an int, got {type(num_cpu_bytes).__name__}: {num_cpu_bytes!r}."
            ),
        )
        _torch_check(
            (num_cpu_bytes >= 0 and num_cpu_bytes % buffer_alignment == 0),
            lambda: (
                f"num_cpu_bytes must be non-negative and 2MB-aligned, got {num_cpu_bytes=}, "
                f"which is not divisible by {buffer_alignment=}."
            ),
        )
        if with_grad:
            num_max_tokens_per_rank = moe_args[0]
            _torch_check(
                isinstance(num_max_tokens_per_rank, int)
                and not isinstance(num_max_tokens_per_rank, bool)
                and num_max_tokens_per_rank > 0,
                lambda: (
                    "num_max_tokens_per_rank must be a positive int when with_grad=True, "
                    f"got {type(num_max_tokens_per_rank).__name__}: {num_max_tokens_per_rank!r}."
                ),
            )
        else:
            moe_arg_names = ("num_max_tokens_per_rank", "hidden", "num_topk")
            missing_args = [
                name for name, val in zip(moe_arg_names, moe_args) if val is None
            ]
            _torch_check(
                len(missing_args) == 0 or len(missing_args) == len(moe_args),
                lambda: (
                    "num_max_tokens_per_rank, hidden and num_topk "
                    "must be specified together for MoE dispatch/combine, "
                    f"missing: {', '.join(missing_args)}."
                ),
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
        self._local_storage_addr = self._runtime.get_local_storage_addr()
        self._comm_buffer_size = self._runtime.get_comm_buffer_size()
        self._rank_size = self._runtime.get_rank_size()

    def engram_fetch(self, indices: torch.Tensor) -> Callable:
        """
        Fetch Engram data from remote ranks via RDMA.

        Arguments:
            indices: the indices of entries to fetch (must be 1D NPU tensor with dtype=int32).

        Returns:
            wait_callable: a callable that returns the fetched tensor when invoked.
            In training mode (with_grad=True), the callable returns a tuple of
            (fetched_tensor, EngramFetchCtx) for save-for-backward.
        """
        self._check_engram_fetch_ready(indices)
        self._engram_fetch_in_progress = True
        context = self._engram_context_tensor

        if self._with_grad:
            fetched, perm, send_counts, recv_counts, recv_local_entry, num_recv = (
                torch.ops.cann_ops_transformer.engram_fetch_train(
                    context,
                    indices,
                    self._engram_hidden_size,
                    self._engram_num_entries,
                    self._engram_dtype_int,
                    self._local_storage_addr,
                    self._num_max_tokens_per_rank,
                    self._comm_buffer_size,
                    self._rank_size,
                )
            )
            ctx = EngramFetchCtx(
                perm=perm,
                send_counts=send_counts,
                recv_counts=recv_counts,
                recv_local_entry=recv_local_entry,
                num_recv=num_recv,
            )

            def _wait_train():
                self._engram_fetch_in_progress = False
                return fetched, ctx

            return _wait_train

        fetched = torch.ops.cann_ops_transformer.engram_fetch(
            context,
            indices,
            self._engram_hidden_size,
            self._engram_num_entries,
            self._engram_dtype_int,
        )

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

    def engram_fetch_grad(
        self, grad_fetched: torch.Tensor, fetch_ctx: EngramFetchCtx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward: launch fused kernel aclnnEngramFetchGrad.

        Returns:
            (grad_unique, unique_local_entry):
                grad_unique (K, H) grad_fetched.type
                unique_local_entry (K,) int32 — 1D sparse index。
        """
        _torch_check(
            self._with_grad,
            lambda: "engram_fetch_grad requires ElasticBuffer to be initialized with with_grad=True",
        )
        _torch_check(
            grad_fetched.device.type == torch.device("npu").type,
            lambda: f"grad_fetched must be on NPU, got device: {grad_fetched.device}",
        )
        _torch_check(
            grad_fetched.dim() == 2,
            lambda: f"grad_fetched must be 2D, got dimensions: {grad_fetched.dim()}",
        )
        _torch_check(
            not self._engram_fetch_in_progress,
            lambda: (
                "engram_fetch_grad must be called after the callable returned by engram_fetch."
            ),
        )
        _torch_check(
            self._engram_context_tensor is not None,
            lambda: "engram_fetch_grad must be called after at least one engram_write",
        )
        expected_dtype = _ENGRAM_INT_TO_DTYPE[self._engram_dtype_int]
        _torch_check(
            grad_fetched.dtype == expected_dtype,
            lambda: (
                f"grad_fetched dtype must match storage dtype ({expected_dtype}), "
                f"got {grad_fetched.dtype}"
            ),
        )
        _torch_check(
            fetch_ctx.perm.dim() == 1,
            lambda: f"fetch_ctx.perm must be 1D, got dimensions: {fetch_ctx.perm.dim()}",
        )
        expected_num_tokens = fetch_ctx.perm.size(0)
        _torch_check(
            grad_fetched.size(0) == expected_num_tokens,
            lambda: (
                f"grad_fetched row count ({grad_fetched.size(0)}) must match the number of tokens "
                f"fetched in forward pass ({expected_num_tokens})"
            ),
        )
        _torch_check(
            grad_fetched.size(1) == self._engram_hidden_size,
            lambda: (
                f"grad_fetched hidden size ({grad_fetched.size(1)}) must match storage hidden size "
                f"({self._engram_hidden_size})"
            ),
        )
        _torch_check(
            grad_fetched.size(1) % 128 == 0,
            lambda: (
                f"grad_fetched hidden size must be 128-aligned, got {grad_fetched.size(1)}"
            ),
        )
        for _name, _tensor in (
            ("perm", fetch_ctx.perm),
            ("send_counts", fetch_ctx.send_counts),
            ("recv_counts", fetch_ctx.recv_counts),
            ("recv_local_entry", fetch_ctx.recv_local_entry),
            ("num_recv", fetch_ctx.num_recv),
        ):
            _torch_check(
                _tensor.dtype == torch.int32,
                lambda _n=_name, _t=_tensor: (
                    f"fetch_ctx.{_n} dtype must be int32, got {_t.dtype}"
                ),
            )
            _torch_check(
                _tensor.device.type == torch.device("npu").type,
                lambda _n=_name, _t=_tensor: (
                    f"fetch_ctx.{_n} must be on NPU, got device: {_t.device}"
                ),
            )
            _torch_check(
                _tensor.dim() == 1,
                lambda _n=_name, _t=_tensor: (
                    f"fetch_ctx.{_n} must be 1D, got dimensions: {_t.dim()}"
                ),
            )
        expected_send_counts_len = self._rank_size * 8
        _torch_check(
            fetch_ctx.send_counts.size(0) == expected_send_counts_len,
            lambda: (
                f"fetch_ctx.send_counts length ({fetch_ctx.send_counts.size(0)}) must equal "
                f"rank_size * 8 = {expected_send_counts_len}"
            ),
        )
        _torch_check(
            fetch_ctx.recv_counts.size(0) == self._rank_size,
            lambda: (
                f"fetch_ctx.recv_counts length ({fetch_ctx.recv_counts.size(0)}) must equal "
                f"rank_size ({self._rank_size})"
            ),
        )
        expected_recv_local_entry_len = self._num_max_tokens_per_rank * self._rank_size
        _torch_check(
            fetch_ctx.recv_local_entry.size(0) == expected_recv_local_entry_len,
            lambda: (
                f"fetch_ctx.recv_local_entry length ({fetch_ctx.recv_local_entry.size(0)}) must equal "
                f"num_max_tokens_per_rank * rank_size = {expected_recv_local_entry_len}"
            ),
        )
        _torch_check(
            fetch_ctx.num_recv.size(0) == 1,
            lambda: (
                f"fetch_ctx.num_recv length ({fetch_ctx.num_recv.size(0)}) must be 1"
            ),
        )
        grad_unique_full, unique_local_entry_full, num_unique = (
            torch.ops.cann_ops_transformer.engram_fetch_grad(
                self._engram_context_tensor,
                grad_fetched,
                fetch_ctx.perm,
                fetch_ctx.send_counts,
                fetch_ctx.recv_counts,
                fetch_ctx.recv_local_entry,
                fetch_ctx.num_recv,
                self._engram_num_entries,
                self._comm_buffer_size,
                self._num_max_tokens_per_rank,
                self._rank_size,
            )
        )
        # Narrow to actualK rows outside the graph (.item() syncs; not graph-safe).
        actual_k = int(num_unique.item())
        max_r = self._num_max_tokens_per_rank * self._rank_size
        if actual_k < max_r:
            grad_unique = grad_unique_full.narrow(0, 0, actual_k)
            unique_local_entry = unique_local_entry_full.narrow(0, 0, actual_k)
        else:
            grad_unique = grad_unique_full
            unique_local_entry = unique_local_entry_full
        return grad_unique, unique_local_entry

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
        torch._check(
            args.x.shape[1] == self._hidden,
            lambda: f"x hidden must equal configured hidden, got {args.x.shape[1]} and {self._hidden}",
        )
        torch._check(
            args.topk_idx.shape[1] == self._num_topk,
            lambda: (
                "topk_idx topk must equal configured num_topk, "
                f"got {args.topk_idx.shape[1]} and {self._num_topk}"
            ),
        )
        hp_addr = self._prepare_host_counter(args.do_cpu_sync)

        (
            num_recv_per_rank,
            num_recv_per_expert,
            dst_slot,
            route_count,
            route_dst_scaleout,
            route_scaleout_slot,
        ) = self._runtime.moe_ep_dispatch(
            args.x,
            args.topk_idx,
            topk_weights,
            args.scales,
            args.cached_dst_slot_idx,
            args.cached_route_count,
            args.cached_route_dst_scaleout,
            args.cached_route_scaleout_slot,
            self._ep_world_size,
            self._rank_id,
            args.num_experts,
            args.num_max_tokens_per_rank,
            args.expert_alignment,
            args.do_cpu_sync,
            hp_addr,
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
                recv_x,
                recv_src_meta,
                recv_topk_weights,
                recv_scales,
            )
        )

        recv_x = (recv_x, recv_scales) if recv_scales is not None else recv_x
        new_handle = self._make_dispatch_handle(
            args,
            dst_slot,
            recv_src_meta,
            num_recv_per_rank,
            num_recv_per_expert,
            route_count,
            route_dst_scaleout,
            route_scaleout_slot,
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

        num_tokens = handle.topk_idx.shape[0]
        hidden = x.shape[1]
        topk = handle.topk_idx.shape[1]
        combined_x = torch.empty((num_tokens, hidden), dtype=x.dtype, device=x.device)
        combined_topk_weights = (
            None
            if topk_weights is None
            else torch.empty((num_tokens, topk), dtype=torch.float32, device=x.device)
        )

        self._runtime.moe_ep_combine(
            x,
            handle.topk_idx,
            handle.recv_src_metadata,
            handle.num_recv_tokens_per_expert,
            topk_weights,
            self._ep_world_size,
            self._rank_id,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
        )

        combined_x, combined_topk_weights = self._runtime.moe_ep_combine_epilogue(
            handle.topk_idx,
            topk_weights,
            self._ep_world_size,
            self._rank_id,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
            combined_x,
            combined_topk_weights,
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
        self._local_storage_addr = None
        self._comm_buffer_size = 0
        self._rank_size = 0

    def _check_engram_fetch_ready(self, indices: torch.Tensor) -> None:
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
                handle.route_count,
                handle.route_dst_scaleout,
                handle.route_scaleout_slot,
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
            None,
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
        route_count: torch.Tensor,
        route_dst_scaleout: torch.Tensor,
        route_scaleout_slot: torch.Tensor,
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
            route_count=route_count,
            route_dst_scaleout=route_dst_scaleout,
            route_scaleout_slot=route_scaleout_slot,
        )

    def _unpack_bias(self, bias):
        if bias is None:
            return None, None
        if isinstance(bias, torch.Tensor):
            return bias, None
        if isinstance(bias, tuple) and len(bias) == 2:
            return bias[0], bias[1]
        raise TypeError(f"unsupported bias type: {type(bias)}")
