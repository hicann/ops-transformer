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
import os
from typing import Optional, Sequence

import torch
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl

_logger = logging.getLogger(__name__)


class _QuantMatmulReduceScatterOpBuilder(OpBuilder):
    def __init__(self):
        super(_QuantMatmulReduceScatterOpBuilder, self).__init__(
            "npu_quant_matmul_reduce_scatter", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/quant_matmul_reduce_scatter.cpp"]

    def include_paths(self):
        # apace 通信建链头文件（hccl_common.h/apace/*）随 CANN custom_transformer 包安装，
        # 与 mc2/common/torch_extension 的 CommChannelBuilderManagerOpBuilder 保持一致
        paths = super().include_paths()
        candidate_paths = [
            os.path.join(
                self._cann_path,
                "opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common",
            ),
            os.path.join(
                self._cann_path,
                "vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common",
            ),
            os.path.join(
                self._cann_path,
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common",
            ),
            os.path.join(
                self._cann_path,
                "opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/op_kernel",
            ),
            os.path.join(
                self._cann_path,
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/op_kernel",
            ),
        ]
        for path in candidate_paths:
            if os.path.isdir(path):
                paths.append(path)
        return paths

    def schema(self) -> str:
        return (
            "npu_quant_matmul_reduce_scatter(Tensor context, Tensor x1, Tensor x2, "
            "Tensor x1_scale, Tensor x2_scale, "
            "int hccl_buffer_size, str group_name, int rank_size, *, "
            "Tensor? bias=None, int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, int? y_dtype=None, "
            "str comm_mode=None"
            ") -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_quant_matmul_reduce_scatter_meta(
            context,
            x1,
            x2,
            x1_scale,
            x2_scale,
            hccl_buffer_size,
            group_name,
            rank_size,
            bias=None,
            group_sizes=None,
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=None,
            x2_scale_dtype=None,
            y_dtype=None,
            comm_mode=None,
        ):
            if hccl_buffer_size <= 0:
                raise ValueError("hccl_buffer_size should be greater than 0.")
            if rank_size <= 0:
                raise ValueError("rank_size should be greater than 0.")
            if bias is not None:
                raise ValueError("bias not supported in ubmem path")
            if comm_mode is not None and comm_mode != "ubmem":
                raise ValueError(
                    f"comm_mode only supports 'ubmem' in this experimental build, but got '{comm_mode}'."
                )
            if x1.dim() != 2:
                raise ValueError(f"x1 should be 2D, but got {x1.dim()}D.")
            if x2.dim() != 2 and x2.dim() != 4:
                raise ValueError(
                    f"x2 should be 2D (ND) or 4D (FRACTAL_NZ), but got {x2.dim()}D."
                )
            m = x1.size(0)
            if m % rank_size != 0:
                raise ValueError(
                    f"x1.size(0) should be divisible by rank_size, but got {m} % {rank_size} != 0."
                )
            x2_is_fp4 = x2_dtype == 296
            k = x1.size(1) * 2 if x2_is_fp4 else x1.size(1)
            kd = (k + 63) // 64
            s0, s1 = x2_scale.size(0), x2_scale.size(1)
            if s0 == kd and s1 != kd:
                n = s1  # 非转置 B：N=scale.dim1
            elif s1 == kd and s0 != kd:
                n = s0  # 转置 B：N=scale.dim0
            else:
                n = s1 if s0 == kd else s0  # K==N 等歧义，仍按 scale 取 N
            if y_dtype is None or y_dtype == 283:
                out_dtype = torch.bfloat16
            elif y_dtype == 257:
                out_dtype = torch.float16
            else:
                raise ValueError(
                    f"y_dtype only supports 283 (BF16) or 257 (FP16), but got {y_dtype}."
                )
            y = x1.new_empty(tuple([m // rank_size, n]), dtype=out_dtype)
            return y


_quant_matmul_reduce_scatter_op_builder = _QuantMatmulReduceScatterOpBuilder()
_quant_matmul_reduce_scatter_op_builder._ensure_initialized()


@impl(get_as_library(), _quant_matmul_reduce_scatter_op_builder.name, "PrivateUse1")
def _npu_quant_matmul_reduce_scatter(
    context,
    x1,
    x2,
    x1_scale,
    x2_scale,
    hccl_buffer_size,
    group_name,
    rank_size,
    bias=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode=None,
):
    op_module = _quant_matmul_reduce_scatter_op_builder.load()
    return op_module.npu_quant_matmul_reduce_scatter(
        context,
        x1,
        x2,
        x1_scale,
        x2_scale,
        hccl_buffer_size,
        group_name,
        rank_size,
        bias,
        group_sizes if group_sizes is not None else [],
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        y_dtype,
        comm_mode if comm_mode is not None else "ubmem",
    )


def _check_params(
    x1,
    x2,
    x1_scale,
    x2_scale,
    bias,
    x1_dtype,
    x2_dtype,
    x1_scale_dtype,
    x2_scale_dtype,
    group_sizes,
    comm_mode,
):
    """Validate input parameters (ubmem path only)."""
    if x1 is None:
        raise ValueError("x1 must not be None.")
    if x2 is None:
        raise ValueError("x2 must not be None.")
    if x1_scale is None:
        raise ValueError("x1_scale must not be None.")
    if x2_scale is None:
        raise ValueError("x2_scale must not be None.")
    if bias is not None:
        raise ValueError(
            "bias is not supported in the ubmem path of quant_matmul_reduce_scatter."
        )
    if group_sizes is not None and list(group_sizes) not in ([], [1, 1, 32]):
        raise ValueError(
            f"group_sizes only supports empty (auto-infer) or [1, 1, 32] (MX fixed group) "
            f"in the ubmem path, but got {group_sizes}."
        )
    if comm_mode is not None and comm_mode != "ubmem":
        raise ValueError(
            f"comm_mode only supports 'ubmem' in this experimental build, but got '{comm_mode}'."
        )
    if x1.dim() != 2:
        raise ValueError(f"x1 should be 2D, but got {x1.dim()}D.")
    if x2.dim() != 2 and x2.dim() != 4:
        raise ValueError(
            f"x2 should be 2D (ND) or 4D (FRACTAL_NZ), but got {x2.dim()}D."
        )
    if x1_dtype is not None and not isinstance(x1_dtype, (int, torch.dtype)):
        raise ValueError(
            f"x1_dtype must be an int (ACL dtype enum), a torch.dtype, or None, "
            f"but got {type(x1_dtype).__name__}."
        )
    if x2_dtype is not None and not isinstance(x2_dtype, (int, torch.dtype)):
        raise ValueError(
            f"x2_dtype must be an int (ACL dtype enum), a torch.dtype, or None, "
            f"but got {type(x2_dtype).__name__}."
        )
    if x1_scale_dtype is not None and not isinstance(
        x1_scale_dtype, (int, torch.dtype)
    ):
        raise ValueError(
            f"x1_scale_dtype must be an int (ACL dtype enum), a torch.dtype, or None, "
            f"but got {type(x1_scale_dtype).__name__}."
        )
    if x2_scale_dtype is not None and not isinstance(
        x2_scale_dtype, (int, torch.dtype)
    ):
        raise ValueError(
            f"x2_scale_dtype must be an int (ACL dtype enum), a torch.dtype, or None, "
            f"but got {type(x2_scale_dtype).__name__}."
        )
    # ACL dtype 枚举（GetAclDataType 的 +256 偏移口径，与 csrc 一致）：
    #   291 = FLOAT8_E5M2, 292 = FLOAT8_E4M3FN, 293 = FLOAT8_E8M0, 296 = FLOAT4_E2M1
    fp8_e4m3_dtype_enum = 292
    fp8_e5m2_dtype_enum = 291
    fp4_e2m1_dtype_enum = 296
    x_dtype_enum_list = (fp8_e4m3_dtype_enum, fp8_e5m2_dtype_enum, fp4_e2m1_dtype_enum)
    if x1.dtype == torch.uint8 and x1_dtype not in x_dtype_enum_list:
        raise ValueError(
            f"x1 is uint8 tensor (fp8 bitcast or fp4 packed 2 elems/byte) but x1_dtype={x1_dtype} "
            f"is not a valid enum (291=fp8_e5m2, 292=fp8_e4m3fn, 296=fp4_e2m1). "
            f"Please pass the matching x1_dtype."
        )
    if x2.dtype == torch.uint8 and x2_dtype not in x_dtype_enum_list:
        raise ValueError(
            f"x2 is uint8 tensor (fp8 bitcast or fp4 packed 2 elems/byte) but x2_dtype={x2_dtype} "
            f"is not a valid enum (291=fp8_e5m2, 292=fp8_e4m3fn, 296=fp4_e2m1). "
            f"Please pass the matching x2_dtype."
        )
    # fp4 双向拦截（与 def/tiling/csrc 对齐）：fp4 只能与 fp4 配对
    if (x1_dtype == fp4_e2m1_dtype_enum) != (x2_dtype == fp4_e2m1_dtype_enum):
        raise ValueError(
            f"fp4 requires both x1 and x2 to be fp4 (x1_dtype=296 and x2_dtype=296), "
            f"but got x1_dtype={x1_dtype}, x2_dtype={x2_dtype}."
        )
    fp8_e8m0_dtype_enum = 293
    if x1_scale.dtype == torch.uint8 and x1_scale_dtype != fp8_e8m0_dtype_enum:
        raise ValueError(
            f"x1_scale is uint8 tensor (fp8_e8m0 stored as uint8) but "
            f"x1_scale_dtype={x1_scale_dtype} is not 293 (fp8_e8m0). "
            f"Please pass x1_scale_dtype=293."
        )
    if x2_scale.dtype == torch.uint8 and x2_scale_dtype != fp8_e8m0_dtype_enum:
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
    op_module = _quant_matmul_reduce_scatter_op_builder.load()
    ctx_tag = group_name + "quant_matmul_reduce_scatter"
    context = op_module.create_ubmem_context(group_name, ctx_tag)
    hccl_buffer_size = op_module.get_hccl_buffer_size(group_name)
    return context, hccl_buffer_size, rank_id, rank_size, group_name


def _check_hccl_buffer(x1, x2, x2_scale, x2_dtype, rank_size, hccl_buffer_size):
    """Check comm window bytes <= hccl_buffer_size.

    窗口布局与 kernel 侧 apace/utils/apace_constant.h 保持一致：
    APACE_MTE_STATE_WIN_SIZE = 1MB（状态窗口）+ (rank_size - 1) * APACE_EP_RANK_OFFSET_STEP(1024B)
    + 本 rank 全量 matmul 输出 C（[M, N] bf16，2B/元素）。
    N 推导与 meta/csrc 一致（按 x2 布局/形状自动推断，不再依赖 transpose_x2）。
    """
    state_win_bytes = 1024 * 1024
    rank_offset_step_bytes = 1024
    m = x1.size(0)
    x2_is_fp4 = x2_dtype == 296
    k = x1.size(1) * 2 if x2_is_fp4 else x1.size(1)
    kd = (k + 63) // 64
    # 2D（ND / NZ 2D）/ NZ 4D：N 统一由 x2_scale 精确推导（不受 fp4 打包/.t() 视图干扰）
    s0, s1 = x2_scale.size(0), x2_scale.size(1)
    if s0 == kd and s1 != kd:
        n = s1  # 非转置 B：scale=[kd,n,2]
    elif s1 == kd and s0 != kd:
        n = s0  # 转置 B：scale=[n,kd,2]
    else:
        n = s1 if s0 == kd else s0  # K==N 等歧义
    comm_data_bytes = m * n * 2
    need_bytes = (
        state_win_bytes + (rank_size - 1) * rank_offset_step_bytes + comm_data_bytes
    )
    _logger.info(
        "hccl_buffer_size=%s bytes (%.2f MB), comm_data_bytes=%s bytes (%.2f MB), "
        "need (state win %s + rank offset %s + C %s)=%s",
        hccl_buffer_size,
        hccl_buffer_size / 1024 / 1024,
        comm_data_bytes,
        comm_data_bytes / 1024 / 1024,
        state_win_bytes,
        (rank_size - 1) * rank_offset_step_bytes,
        comm_data_bytes,
        need_bytes,
    )
    if need_bytes > hccl_buffer_size:
        raise ValueError(
            f"Communication buffer required {need_bytes} bytes (state win "
            f"{state_win_bytes} + rank offset {(rank_size - 1) * rank_offset_step_bytes} "
            f"+ matmul C {comm_data_bytes}) exceeds hccl_buffer_size "
            f"({hccl_buffer_size} bytes = {hccl_buffer_size / 1024 / 1024:.2f} MB). "
            f"Please increase HCCL_BUFFSIZE environment variable."
        )


def quant_matmul_reduce_scatter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x1_scale: torch.Tensor,
    x2_scale: torch.Tensor,
    group,
    *,
    bias: Optional[torch.Tensor] = None,
    group_sizes: Optional[Sequence[int]] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
    y_dtype: Optional[int] = None,
    comm_mode: Optional[str] = None,
) -> torch.Tensor:
    """QuantMatmul + ReduceScatter fused computation (apace UBMEM path).

    Each rank computes the full ``x1 @ x2^T`` first, then the AlltoAll epilogue
    scatters row blocks across ranks and ReduceAdd sums them:
    ``y = ReduceScatter(x1 @ x2^T)``.

    Args:
        x1 (Tensor): Left matrix (per-rank A), shape ``(M, K)``, dtype
            float8_e4m3fn / float8_e5m2, or uint8 packed fp4 (2 elems/byte,
            last dim ``K/2``, pass ``x1_dtype=296``). ``M`` must be divisible
            by ``rank_size``.
        x2 (Tensor): Right matrix, dtype float8_e4m3fn / float8_e5m2, or uint8
            packed fp4 (2 elems/byte). 方向（转置/非转置）由 x2 布局自动推断，
            不再需要 transpose_x2 参数：
            DN：B 物理存 [N,K]、以 ``.t()`` 视图（[K,N] view，stride[0]==1）传入；
            ND：连续 [N,K]（转置 B）或连续 [K,N]（非转置 B）；
            FRACTAL_NZ：2D 逻辑 shape（[N,K]=ZN 转置 / [K,N]=NZ 非转置）或 4D
            storage view（转置 ``(ceil(K/32), ceil(N/16), 16, 32)`` / 非转置
            ``(ceil(N/32), ceil(K/16), 16, 32)``）。
            fp4 only supports the ND path (fp8 supports ND/NZ), and fp4 must
            pair with fp4 on both x1 and x2.
        x1_scale (Tensor): MX scale for x1, shape ``(M, Ceil(K/64), 2)``, dtype
            float8_e8m0fnu (fp8/fp4 语义一致：32 元素一组，e8m0).
        x2_scale (Tensor): MX scale for x2, ND shape ``(N, Ceil(K/64), 2)`` /
            FRACTAL_NZ shape ``(Ceil(K/64), N, 2)``, dtype float8_e8m0fnu.
        group (ProcessGroup): torch.distributed process group for
            ReduceScatter communication.
        bias (Tensor, optional): Not supported in the ubmem path (must be None).
        group_sizes (List[int], optional): Quantization group sizes; MX only
            supports empty (auto-infer) or ``[1, 1, 32]``. Currently validated
            only and not used for computation.
        x1_dtype (int, optional): x1 dtype enum for bitcast, None means use tensor dtype.
            291=fp8_e5m2, 292=fp8_e4m3fn, 296=fp4_e2m1 (fp4 只能与 fp4 配对).
        x2_dtype (int, optional): x2 dtype enum for bitcast, None means use tensor dtype.
            291=fp8_e5m2, 292=fp8_e4m3fn, 296=fp4_e2m1 (fp4 只能与 fp4 配对).
        x1_scale_dtype (int, optional): x1_scale dtype enum for bitcast, None means use tensor dtype.
        x2_scale_dtype (int, optional): x2_scale dtype enum for bitcast, None means use tensor dtype.
        y_dtype (int, optional): y dtype enum (ACL+256): 283=BF16（默认）、257=FP16.
        comm_mode (str, optional): Communication mode; only ``"ubmem"`` is
            supported in this experimental build (default).

    Returns:
        Tensor: y with shape ``(M // rank_size, N)``, dtype BF16 or FP16 per ``y_dtype``.
    """
    _check_params(
        x1,
        x2,
        x1_scale,
        x2_scale,
        bias,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
        group_sizes,
        comm_mode,
    )
    context, hccl_buffer_size, rank_id, rank_size, group_name = _prepare_comm_context(
        group
    )
    _check_hccl_buffer(x1, x2, x2_scale, x2_dtype, rank_size, hccl_buffer_size)
    _logger.info(
        "rank=%s, rank_size=%s, calling npu_quant_matmul_reduce_scatter ...",
        rank_id,
        rank_size,
    )
    result = torch.ops.cann_ops_transformer.npu_quant_matmul_reduce_scatter(
        context,
        x1,
        x2,
        x1_scale,
        x2_scale,
        hccl_buffer_size,
        group_name,
        rank_size,
        bias=bias,
        group_sizes=group_sizes if group_sizes is not None else [],
        x1_dtype=x1_dtype,
        x2_dtype=x2_dtype,
        x1_scale_dtype=x1_scale_dtype,
        x2_scale_dtype=x2_scale_dtype,
        y_dtype=y_dtype,
        comm_mode=comm_mode,
    )
    _logger.info("rank=%s, npu_quant_matmul_reduce_scatter returned", rank_id)
    return result
