# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List, Tuple

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommChannelBuilderManager


_ACL_DTYPE_TO_TORCH_DTYPE = {
    5: torch.float16,
    6: torch.float32,
    15: torch.bfloat16,
}


class _AllToAllQuantMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super(_AllToAllQuantMatmulOpBuilder, self).__init__(
            "npu_all_to_all_quant_matmul", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/all_to_all_quant_matmul.cpp"]

    def schema(self) -> str:
        return (
            "npu_all_to_all_quant_matmul(Tensor context, Tensor x1, Tensor x2, int hccl_buffer_size, str group_name, int world_size, *, "
            "Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, "
            "int? x1_quant_mode=None, int? x2_quant_mode=None, "
            "int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, "
            "int? y_dtype=None, str? comm_mode=None, int? precision_mode=None"
            ") -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_all_to_all_quant_matmul_meta(
            context,
            x1,
            x2,
            hccl_buffer_size,
            group_name,
            world_size,
            bias=None,
            x1_scale=None,
            x2_scale=None,
            x1_quant_mode=None,
            x2_quant_mode=None,
            group_sizes=[],
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=None,
            x2_scale_dtype=None,
            y_dtype=None,
            comm_mode=None,
            precision_mode=None,
        ):
            torch._check(
                world_size > 0,
                lambda: f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}.",
            )
            torch._check(
                x1.dim() == 2,
                lambda: f"x1 should be 2D, but got {x1.dim()}D, {ops_error(ErrCode.VALUE)}.",
            )
            torch._check(
                x2.dim() == 2,
                lambda: f"x2 should be 2D, but got {x2.dim()}D, {ops_error(ErrCode.VALUE)}.",
            )
            bs = x1.size(0)
            h = x1.size(1)
            n = x2.size(1)
            local_bs = bs // world_size
            out_dtype = torch.float32
            if y_dtype is not None:
                out_dtype = _ACL_DTYPE_TO_TORCH_DTYPE.get(y_dtype, out_dtype)
            y = x1.new_empty(tuple([local_bs, n]), dtype=out_dtype)
            all2all_out = x1.new_empty(
                tuple([local_bs, h * world_size]), dtype=x1.dtype
            )
            return (y, all2all_out)


_all_to_all_quant_matmul_op_builder = _AllToAllQuantMatmulOpBuilder()
_all_to_all_quant_matmul_op_builder._ensure_initialized()


@impl(get_as_library(), _all_to_all_quant_matmul_op_builder.name, "PrivateUse1")
def _npu_all_to_all_quant_matmul(
    context,
    x1,
    x2,
    hccl_buffer_size,
    group_name,
    world_size,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    x1_quant_mode=None,
    x2_quant_mode=None,
    group_sizes=[],
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode=None,
    precision_mode=None,
):
    op_module = _all_to_all_quant_matmul_op_builder.load()
    result = op_module.npu_all_to_all_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        world_size,
        bias,
        x1_scale,
        x2_scale,
        x1_quant_mode,
        x2_quant_mode,
        group_sizes,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        y_dtype,
        comm_mode,
        precision_mode,
    )
    return result


def _prepare_comm_context(group, rank_id):
    world_size = torch.distributed.get_world_size(group)
    if world_size <= 0:
        world_size = torch.distributed.get_world_size()  # fallback to default group
    if world_size <= 0:
        raise ValueError(f"world_size should be greater than 0, got {world_size}.")
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
    channel_builder = CommChannelBuilderManager(group_name)
    context = channel_builder.create_context(group_name + "allto_all_matmul")
    hccl_buffer_size = channel_builder.get_hccl_buffer_size()
    return world_size, group_name, context, hccl_buffer_size


def all_to_all_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    group,
    *,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    x1_quant_mode: Optional[int] = None,
    x2_quant_mode: Optional[int] = None,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    y_dtype: Optional[
        int
    ] = None,  # fp32，未显式指定 BF16(15)/FP16(5) 时算子侧会校验失败
    comm_mode: Optional[str] = None,
    precision_mode: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AlltoAll + Matmul fused computation.

    Fuses AlltoAll communication, Permute, and Matmul into a single operator
    (communication first, then computation). Supports non-quantized, K-C
    quantized, K-C dynamic quantized, and MX-quantized scenarios.

    Args:
        x1 (Tensor): Left matrix, shape ``(BS, H)``, dtype float16/bfloat16/
            int4/float8_e4m3fn/float8_e5m2/float4_e2m1.
        x2 (Tensor): Right matrix, shape ``(H*world_size, N)``.
        group (ProcessGroup): torch.distributed process group for AlltoAll
            communication.
        bias (Tensor, optional): Bias added after matmul, shape ``(N,)``.
        x1_scale (Tensor, optional): Left matrix quantization scale.
        x2_scale (Tensor, optional): Right matrix quantization scale.
        x1_quant_mode (int): Left matrix quantization mode (0=none, 2=perchannel,
            3=pertoken, 6=mx, 7=pertoken-dynamic).
        x2_quant_mode (int): Right matrix quantization mode.
        group_sizes (List[int], optional): Quantization group sizes as
            ``[group_m, group_n, group_k]``.
        x1_dtype (int): x1 dtype enum for bitcast, -1 means use tensor dtype.
        x2_dtype (int): x2 dtype enum for bitcast, -1 means use tensor dtype.
        x1_scale_dtype (int): x1_scale dtype enum, -1 means use tensor dtype.
        x2_scale_dtype (int): x2_scale dtype enum, -1 means use tensor dtype.
        y_dtype (int): Output dtype enum, must be 5 (FP16) or 15 (BF16); the default
            fp32 is rejected by the operator.
        comm_mode (str): Communication engine, e.g. ``"ai_cpu"`` or ``"ccu"``.
        precision_mode (int): Precision mode, 0 means default.

    Returns:
        Tuple[Tensor, Tensor]: (y, all2all_out) where y has shape
        ``(BS/world_size, N)``. all2all_out is a reserved output and is not
        supported yet (returned as an empty tensor).
    """
    rank_id = torch.distributed.get_rank(group)
    world_size, group_name, context, hccl_buffer_size = _prepare_comm_context(
        group, rank_id
    )

    result = torch.ops.cann_ops_transformer.npu_all_to_all_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        world_size,
        bias=bias,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        x1_quant_mode=x1_quant_mode,
        x2_quant_mode=x2_quant_mode,
        group_sizes=group_sizes if group_sizes is not None else [],
        x1_dtype=x1_dtype,
        x2_dtype=x2_dtype,
        x1_scale_dtype=x1_scale_dtype,
        x2_scale_dtype=x2_scale_dtype,
        y_dtype=y_dtype,
        comm_mode=comm_mode,
        precision_mode=precision_mode,
    )
    return result
