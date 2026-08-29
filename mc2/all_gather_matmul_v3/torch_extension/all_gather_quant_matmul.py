# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
from typing import Optional, List, Tuple

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommChannelBuilderManager

_logger = logging.getLogger(__name__)

_ACL_DTYPE_TO_TORCH_DTYPE = {
    5: torch.float16,
    6: torch.float32,
    15: torch.bfloat16,
}


class _AllGatherQuantMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super(_AllGatherQuantMatmulOpBuilder, self).__init__(
            "npu_all_gather_quant_matmul", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/all_gather_quant_matmul.cpp"]

    def schema(self) -> str:
        return (
            "npu_all_gather_quant_matmul(Tensor context, Tensor x1, Tensor x2, "
            "int hccl_buffer_size, str group_name, int rank_size, *, "
            "Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, "
            "int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, "
            'int? y_dtype=None, str comm_mode="ai_cpu"'
            ") -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_all_gather_quant_matmul_meta(
            context,
            x1,
            x2,
            hccl_buffer_size,
            group_name,
            rank_size,
            bias=None,
            x1_scale=None,
            x2_scale=None,
            group_sizes=None,
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=None,
            x2_scale_dtype=None,
            y_dtype=None,
            comm_mode="ai_cpu",
        ):
            if rank_size <= 0:
                raise ValueError(
                    f"rank_size should be greater than 0, {ops_error(ErrCode.VALUE)}."
                )
            if x1.dim() != 2:
                raise ValueError(
                    f"x1 should be 2D, but got {x1.dim()}D, {ops_error(ErrCode.VALUE)}."
                )
            if x2.dim() != 2:
                raise ValueError(
                    f"x2 should be 2D, but got {x2.dim()}D, {ops_error(ErrCode.VALUE)}."
                )
            m_per_rank = x1.size(0)
            k = x1.size(1)
            n = x2.size(1)
            out_dtype = torch.bfloat16
            if y_dtype is not None:
                out_dtype = _ACL_DTYPE_TO_TORCH_DTYPE.get(y_dtype, out_dtype)
            y = x1.new_empty(tuple([m_per_rank * rank_size, n]), dtype=out_dtype)
            gather_out = x1.new_empty(
                tuple([m_per_rank * rank_size, k]), dtype=x1.dtype
            )
            amax_out = x1.new_empty(tuple([0]), dtype=torch.float32)
            return (y, gather_out, amax_out)


_all_gather_quant_matmul_op_builder = _AllGatherQuantMatmulOpBuilder()
_op_module = _all_gather_quant_matmul_op_builder.load()


@impl(get_as_library(), _all_gather_quant_matmul_op_builder.name, "PrivateUse1")
def _npu_all_gather_quant_matmul(
    context,
    x1,
    x2,
    hccl_buffer_size,
    group_name,
    rank_size,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode="ai_cpu",
):
    result = _op_module.npu_all_gather_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        rank_size,
        bias,
        x1_scale,
        x2_scale,
        group_sizes if group_sizes is not None else [],
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        y_dtype,
        comm_mode,
    )
    return result


def _check_params(
    x1,
    x2,
    x1_scale,
    x2_scale,
    x1_dtype,
    x2_dtype,
    x1_scale_dtype,
    x2_scale_dtype,
    comm_mode,
):
    """Validate input parameters."""
    if comm_mode != "urma":
        raise ValueError(
            f"comm_mode only supports 'urma', but got '{comm_mode}'. "
            f"Please pass comm_mode='urma' explicitly."
        )
    fp4_dtype_enums = (296,)
    if x1.dtype == torch.uint8 and x1_dtype not in fp4_dtype_enums:
        raise ValueError(
            f"x1 is uint8 tensor (likely fp4 packed) but x1_dtype={x1_dtype} "
            f"is not a valid fp4 enum. Please pass x1_dtype=296 (fp4_e2m1)."
        )
    if x2.dtype == torch.uint8 and x2_dtype not in fp4_dtype_enums:
        raise ValueError(
            f"x2 is uint8 tensor (likely fp4 packed) but x2_dtype={x2_dtype} "
            f"is not a valid fp4 enum. Please pass x2_dtype=296 (fp4_e2m1)."
        )
    fp8_e8m0_dtype_enum = 293
    if (
        x1_scale is not None
        and x1_scale.dtype == torch.uint8
        and x1_scale_dtype != fp8_e8m0_dtype_enum
    ):
        raise ValueError(
            f"x1_scale is uint8 tensor (fp8_e8m0 stored as uint8) but "
            f"x1_scale_dtype={x1_scale_dtype} is not 293 (fp8_e8m0). "
            f"Please pass x1_scale_dtype=293."
        )
    if (
        x2_scale is not None
        and x2_scale.dtype == torch.uint8
        and x2_scale_dtype != fp8_e8m0_dtype_enum
    ):
        raise ValueError(
            f"x2_scale is uint8 tensor (fp8_e8m0 stored as uint8) but "
            f"x2_scale_dtype={x2_scale_dtype} is not 293 (fp8_e8m0). "
            f"Please pass x2_scale_dtype=293."
        )


def _prepare_comm_context(group):
    """Create communication context, return context/buffer/rank info."""
    rank_id = torch.distributed.get_rank(group)
    rank_size = torch.distributed.get_world_size(group)
    if rank_size <= 0:
        rank_size = torch.distributed.get_rank_size()
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
    channel_builder = CommChannelBuilderManager(group_name)
    ctx_tag = group_name + "all_gather_matmul"
    context = channel_builder.create_context(ctx_tag)
    hccl_buffer_size = channel_builder.get_hccl_buffer_size()
    return context, hccl_buffer_size, rank_id, rank_size, group_name


def _check_hccl_buffer(x1, x1_scale, rank_size, hccl_buffer_size):
    """Check comm data + 2MB reserved <= hccl_buffer_size."""
    hccl_buffer_reserved_bytes = 2 * 1024 * 1024
    m_per_rank = x1.size(0)
    k = x1.size(1)
    x1_data_bytes = rank_size * m_per_rank * k * x1.element_size()
    scale_k_groups = (k + 63) // 64
    x1_scale_bytes = 0
    if x1_scale is not None:
        x1_scale_bytes = (
            rank_size * m_per_rank * scale_k_groups * 2 * x1_scale.element_size()
        )
    comm_data_bytes = x1_data_bytes + x1_scale_bytes
    need_bytes = comm_data_bytes + hccl_buffer_reserved_bytes
    _logger.info(
        "hccl_buffer_size=%s bytes (%.2f MB), comm_data_bytes=%s bytes (%.2f MB) "
        "= data %s + scale %s, need (with 2MB reserved)=%s",
        hccl_buffer_size,
        hccl_buffer_size / 1024 / 1024,
        comm_data_bytes,
        comm_data_bytes / 1024 / 1024,
        x1_data_bytes,
        x1_scale_bytes,
        need_bytes,
    )
    if need_bytes > hccl_buffer_size:
        raise ValueError(
            f"Communication data size + 2MB reserved ({need_bytes} bytes = "
            f"data {x1_data_bytes} + scale {x1_scale_bytes} + reserved "
            f"{hccl_buffer_reserved_bytes}) exceeds hccl_buffer_size "
            f"({hccl_buffer_size} bytes = {hccl_buffer_size / 1024 / 1024:.2f} MB). "
            f"Please increase HCCL_BUFFSIZE environment variable."
        )


def all_gather_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    group,
    *,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    y_dtype: Optional[int] = None,
    comm_mode: Optional[str] = "ai_cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AllGather + MX quantized Matmul fused computation (apace UDMA path).

    Each rank gathers A from all ranks, then multiplies with its local B:
    ``y = AllGather(x1) @ x2^T``.

    Args:
        x1 (Tensor): Left matrix (per-rank A), shape ``(M_per_rank, K)``, dtype
            float8_e4m3fn/float8_e5m2/float4_e2m1fn_x2.
        x2 (Tensor): Right matrix (transposed B), shape ``(N, K)``, dtype
            float8_e4m3fn/float8_e5m2/float4_e2m1fn_x2. FP4 requires x1 and x2
            both FP4.
        group (ProcessGroup): torch.distributed process group for AllGather
            communication.
        bias (Tensor, optional): Bias added after matmul, shape ``(N,)``, dtype float32.
        x1_scale (Tensor, optional): MX scale for x1, shape ``(M_per_rank, Ceil(K/64), 2)``,
            dtype float8_e8m0fnu. Required in MX scene.
        x2_scale (Tensor, optional): MX scale for x2, shape ``(N, Ceil(K/64), 2)``, dtype
            float8_e8m0fnu. Required in MX scene.
        group_sizes (List[int], optional): Quantization group sizes, MX only
            supports ``[1, 1, 32]``. None means auto-inferred from scale shapes.
        x1_dtype (int, optional): x1 dtype enum for bitcast, None means use tensor dtype.
        x2_dtype (int, optional): x2 dtype enum for bitcast, None means use tensor dtype.
        x1_scale_dtype (int, optional): x1_scale dtype enum for bitcast, None means use tensor dtype.
        x2_scale_dtype (int, optional): x2_scale dtype enum for bitcast, None means use tensor dtype.
        y_dtype (int, optional): Output dtype enum, None means default bfloat16.
            Only bfloat16/float16 are supported.
        comm_mode (str, optional): Communication engine. Only ``"urma"`` is supported;
            the default ``"ai_cpu"`` is rejected, so pass ``"urma"`` explicitly.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (y, gather_out, amax_out) where y has
            shape ``(M_per_rank * rank_size, N)``, gather_out and amax_out are
            empty tensor (not implemented yet). Users typically use
            ``y, _, _ = all_gather_quant_matmul(...)`` to ignore the latter two.
    """
    if group_sizes is None:
        group_sizes = []
    _check_params(
        x1,
        x2,
        x1_scale,
        x2_scale,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        comm_mode,
    )
    context, hccl_buffer_size, rank_id, rank_size, group_name = _prepare_comm_context(
        group
    )
    _check_hccl_buffer(x1, x1_scale, rank_size, hccl_buffer_size)
    _logger.info(
        "rank=%s, rank_size=%s, comm_mode=%s, calling npu_all_gather_quant_matmul ...",
        rank_id,
        rank_size,
        comm_mode,
    )
    result = torch.ops.cann_ops_transformer.npu_all_gather_quant_matmul(
        context,
        x1,
        x2,
        hccl_buffer_size,
        group_name,
        rank_size,
        bias=bias,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=group_sizes,
        x1_dtype=x1_dtype,
        x2_dtype=x2_dtype,
        x1_scale_dtype=x1_scale_dtype,
        x2_scale_dtype=x2_scale_dtype,
        y_dtype=y_dtype,
        comm_mode=comm_mode,
    )
    _logger.info("rank=%s, npu_all_gather_quant_matmul returned", rank_id)
    return result
