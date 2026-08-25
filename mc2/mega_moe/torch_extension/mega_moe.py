# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Dict, List, Optional

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommContextManager


_CLAMP_DEFAULT = 3.4028234663852886e38
_ALPHA_DEFAULT = 1.702
_BETA_DEFAULT = 1.0
_ALLOWED_ACTIVATION_KEYS = {
    "swiglu": frozenset(),
    "swiglustep": frozenset(),
    "swigluoai": frozenset({"alpha", "beta"}),
    "situglu": frozenset({"beta", "linear_beta"}),
}


def _normalize_activation_params(
    activation: str,
    activation_clamp: Optional[float] = None,
    activation_params: Optional[Dict[str, float]] = None,
) -> List[float]:
    if activation not in _ALLOWED_ACTIVATION_KEYS:
        raise ValueError(
            f"Unsupported activation: {activation!r}; expected one of {sorted(_ALLOWED_ACTIVATION_KEYS)}."
        )

    activation_params = activation_params or {}
    allowed = _ALLOWED_ACTIVATION_KEYS[activation]
    for key in activation_params:
        if key not in allowed:
            raise ValueError(
                f"Unknown activation param key {key!r} for activation {activation!r}; "
                f"allowed keys: {sorted(allowed)}."
            )

    # aclnn cannot register a zero-length ListFloat because its data pointer is null. Materialize
    # every default here while the named Torch arguments can still be encoded without ambiguity.
    if activation in ("swiglu", "swiglustep"):
        clamp = _CLAMP_DEFAULT if activation_clamp is None else float(activation_clamp)
        return [clamp]

    if activation == "swigluoai":
        clamp = _CLAMP_DEFAULT if activation_clamp is None else float(activation_clamp)
        alpha = float(activation_params.get("alpha", _ALPHA_DEFAULT))
        beta = float(activation_params.get("beta", _BETA_DEFAULT))
        return [clamp, alpha, beta]

    beta = float(activation_params.get("beta", _BETA_DEFAULT))
    values = [beta]
    if "linear_beta" in activation_params:
        values.append(float(activation_params["linear_beta"]))
    return values


class _MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(_MegaMoeOpBuilder, self).__init__("npu_mega_moe", category="mc2")

    def sources(self):
        return ["csrc/mc2/mega_moe.cpp"]

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
            "Tensor? mask_buffer=None, "
            "int max_recv_token_num=0, "
            "int dispatch_quant_mode=0, int combine_quant_mode=0, "
            'str comm_alg="", int num_max_tokens_per_rank=0, str activation="swiglu", '
            "float? activation_clamp=None, Dict(str, float)? activation_params=None, "
            "int? dispatch_quant_out_dtype=None,  "
            "int? weight1_type=None, int? weight2_type=None, int? topo_type=None, "
            "int? rank_num_per_server=None, int topk_weights_type=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
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
            mask_buffer=None,
            max_recv_token_num=0,
            dispatch_quant_mode=0,
            combine_quant_mode=0,
            comm_alg="",
            num_max_tokens_per_rank=0,
            activation="swiglu",
            activation_clamp=None,
            activation_params=None,
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
_mega_moe_op_builder._ensure_initialized()


@impl(get_as_library(), _mega_moe_op_builder.name, "PrivateUse1")
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
    mask_buffer=None,
    max_recv_token_num=0,
    dispatch_quant_mode=0,
    combine_quant_mode=0,
    comm_alg="",
    num_max_tokens_per_rank=0,
    activation="swiglu",
    activation_clamp=None,
    activation_params=None,
    dispatch_quant_out_dtype=None,
    weight1_type=None,
    weight2_type=None,
    topo_type=None,
    rank_num_per_server=None,
    topk_weights_type=0,
):
    activation_params_list = _normalize_activation_params(
        activation, activation_clamp, activation_params
    )
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
        mask_buffer,
        max_recv_token_num,
        dispatch_quant_mode,
        combine_quant_mode,
        comm_alg,
        num_max_tokens_per_rank,
        activation,
        activation_params_list,
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
            topk_weights_type,
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
        self.mask_buffer = None

    def _create_mask_buffer(self, ep_world_size: int) -> torch.Tensor:
        return torch.zeros(ep_world_size, dtype=torch.int32, device=self.context.device)

    def _check_mask_buffer_supported(self) -> None:
        if "Ascend910_93" not in torch.npu.get_device_name():
            raise RuntimeError("mask_buffer is supported on Atlas A3 only.")

    def destroy(self):
        self._ctx_manager.destroy()

    def query_mask_buffer(self, mask_status: torch.Tensor) -> None:
        """Copy the current rank mask to a caller-provided NPU tensor."""
        self._check_mask_buffer_supported()
        if self.mask_buffer is None:
            self.mask_buffer = self._create_mask_buffer(self.ep_world_size)
        if not isinstance(mask_status, torch.Tensor):
            raise TypeError(f"mask_status must be a Tensor, got {type(mask_status)}.")
        if mask_status.dtype != torch.int32:
            raise TypeError(
                f"mask_status dtype must be torch.int32, got {mask_status.dtype}."
            )
        if mask_status.device.type != "npu":
            raise ValueError(f"mask_status must be on NPU, got {mask_status.device}.")
        if mask_status.device != self.mask_buffer.device:
            raise ValueError(
                "mask_status must be on the same NPU device as mask_buffer."
            )
        if mask_status.shape != self.mask_buffer.shape:
            raise ValueError(
                "mask_status shape must match mask_buffer shape "
                f"{tuple(self.mask_buffer.shape)}, got {tuple(mask_status.shape)}."
            )
        if not mask_status.is_contiguous():
            raise ValueError("mask_status must be contiguous.")
        mask_status.copy_(self.mask_buffer, non_blocking=True)

    def update_mask_buffer(self, rank: int, masked: bool) -> None:
        self._check_mask_buffer_supported()
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank < self.ep_world_size
        ):
            raise IndexError(f"rank must be in [0, {self.ep_world_size}), got {rank}.")
        if not isinstance(masked, bool):
            raise TypeError(f"masked must be bool, got {type(masked)}.")

        if self.mask_buffer is None:
            self.mask_buffer = self._create_mask_buffer(self.ep_world_size)

        self.mask_buffer[rank] = int(masked)

    def clean_mask_buffer(self) -> None:
        """Set every rank mask entry to zero on the current NPU stream."""
        self._check_mask_buffer_supported()
        if self.mask_buffer is None:
            self.mask_buffer = self._create_mask_buffer(self.ep_world_size)
            return
        self.mask_buffer.zero_()

    def get_local_buffer_tensor(
        self,
        dtype: torch.dtype,
        size: Optional[torch.Size] = None,
        offset: int = 0,
    ) -> torch.Tensor:
        """Return a zero-copy tensor view of the local CCL buffer."""
        self._check_mask_buffer_supported()
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be torch.dtype, got {type(dtype)}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"offset must be a non-negative int, got {offset}.")

        tensor = self._ctx_manager.get_local_buffer_tensor(dtype, offset)
        if size is None:
            return tensor
        if not isinstance(size, torch.Size):
            raise TypeError(f"size must be torch.Size or None, got {type(size)}.")
        if tensor.numel() < size.numel():
            raise ValueError(
                f"requested size contains {size.numel()} elements, but only "
                f"{tensor.numel()} elements remain after offset {offset}."
            )
        return tensor[: size.numel()].view(size)

    def update_group(self, group) -> None:
        """Destroy the old links after a replacement context has been created."""
        self._check_mask_buffer_supported()
        rank_id = torch.distributed.get_rank(group)
        group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            rank_id, init_comm=True
        )
        ep_world_size = torch.distributed.get_world_size(group)
        required_ccl_buffer_size = _get_mega_moe_ccl_buffer_size(
            ep_world_size,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            self.max_recv_token_num,
            self.dispatch_quant_mode,
            self.dispatch_quant_out_dtype,
            self.combine_quant_mode,
            self.comm_alg,
            self.topk_weights_type,
        )
        new_manager = CommContextManager(
            group_name,
            ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            customCclBufferSize=required_ccl_buffer_size,
        )
        try:
            new_context = new_manager.create_context()
        except Exception:
            new_manager.destroy()
            raise

        old_manager = self._ctx_manager
        self.group = group
        self.rank_id = rank_id
        self.group_name = group_name
        self.ep_world_size = ep_world_size
        self._ctx_manager = new_manager
        self.context = new_context
        self.ccl_buffer_size = new_manager.ccl_buffer_size
        self.topo_type = new_manager.topo_type
        self.rank_num_per_server = new_manager.rank_num_per_server
        old_manager.destroy()


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
    activation_params: Optional[Dict[str, float]] = None,
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
        mask_buffer=sym_buffer.mask_buffer,
        max_recv_token_num=sym_buffer.max_recv_token_num,
        dispatch_quant_mode=sym_buffer.dispatch_quant_mode,
        combine_quant_mode=sym_buffer.combine_quant_mode,
        comm_alg=sym_buffer.comm_alg,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        activation=activation,
        activation_clamp=activation_clamp,
        activation_params=activation_params,
        dispatch_quant_out_dtype=sym_buffer.dispatch_quant_out_dtype,
        weight1_type=weight1_type,
        weight2_type=weight2_type,
        topo_type=sym_buffer.topo_type,
        rank_num_per_server=sym_buffer.rank_num_per_server,
        topk_weights_type=sym_buffer.topk_weights_type,
    )
