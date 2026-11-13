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
from typing import List, Optional, Tuple

import torch
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error

_UBMEM_CTX_CACHE = {}

UBMEM_CONTEXT_TAG = "all_gather_quant_matmul_ubmem"

_logger = logging.getLogger(__name__)

# dtype 枚举: <256 为 torch ScalarType 枚举, >=256 为 ACL 枚举 + 256
_DTYPE_ENUM_FP16 = 5
_DTYPE_ENUM_BF16 = 15
_DTYPE_ENUM_FP8_E5M2 = 23
_DTYPE_ENUM_FP8_E4M3FN = 24
_DTYPE_ENUM_FP4_E2M1 = 296  # ACL_FLOAT4_E2M1, 以 uint8 打包存储(2 元素/字节)

_ACL_DTYPE_TO_TORCH_DTYPE = {
    _DTYPE_ENUM_FP16: torch.float16,
    _DTYPE_ENUM_BF16: torch.bfloat16,
}


# 不包含算子逻辑，只声明需要哪个.cpp，加哪些include路径，注册什么op签名，然后父类OpBuilder框架才知道怎么编译，底层
# 走torch的cpp_extension JIT机制：调gcc、缓存.so等
class _AllGatherQuantMatmulOpBuilder(
    OpBuilder
):  # 继承 cann_ops_transformer的OpBuilder框架
    def __init__(self):
        super(_AllGatherQuantMatmulOpBuilder, self).__init__(
            "npu_all_gather_quant_matmul", category="mc2"
        )

    def sources(self):  # 编译的源文件
        return ["csrc/mc2/all_gather_quant_matmul.cpp"]

    # 已装CANN的kernel头
    def include_paths(self):
        paths = super().include_paths()
        candidate_paths = [
            os.path.join(
                self._cann_path,
                "opp/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/op_kernel",
            ),
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
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/op_kernel",
            ),
        ]
        for path in candidate_paths:
            if os.path.isdir(path):
                paths.append(path)
        return paths

    # torch.library的op签名
    def schema(self) -> str:
        return (
            "npu_all_gather_quant_matmul(Tensor context, Tensor x1, Tensor x2, "
            "int hccl_buffer_size, str group_name, int rank_size, *, "
            "Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None, "
            "int[] group_sizes=[], "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None, "
            "int? y_dtype=None, str comm_mode=None"
            ") -> (Tensor, Tensor, Tensor)"
        )

    # FakeTensorMode的shape推演
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
            comm_mode=None,
        ):
            if rank_size <= 0:
                raise ValueError("rank_size should be greater than 0.")
            if x1.dim() != 2:
                raise ValueError(f"x1 should be 2D, but got {x1.dim()}D.")
            if x2.dim() not in (2, 4):
                raise ValueError(
                    f"x2 should be 2D (ND/NZ logical) or 4D (FRACTAL_NZ storage), "
                    f"but got {x2.dim()}D."
                )
            if x1_scale is None:
                raise ValueError("x1_scale must not be None.")
            if x2_scale is None:
                raise ValueError("x2_scale must not be None.")
            if x1_scale.dim() != 3 or x2_scale.dim() != 3:
                raise ValueError(
                    f"x1_scale/x2_scale must be 3D, but got "
                    f"{x1_scale.dim()}D / {x2_scale.dim()}D."
                )
            m_per_rank = x1.size(0)
            _x2_is_fp4 = _resolve_dtype_enum(x2_dtype, x2.dtype) == _DTYPE_ENUM_FP4_E2M1
            # fp4 打包存储(2 元素/字节)时 x1 末维为 K/2, 方向判断用逻辑 K
            k = x1.size(1) * 2 if _x2_is_fp4 else x1.size(1)
            _kd = (k + 63) // 64
            _s0, _s1 = x2_scale.size(0), x2_scale.size(1)
            if _s0 == _kd and _s1 != _kd:
                n = _s1  # 非转置 B: scale=[kd,n,2]
            elif _s1 == _kd and _s0 != _kd:
                n = _s0  # 转置 B: scale=[n,kd,2]
            else:
                n = _s1 if _s0 == _kd else _s0  # K==N 等歧义
            torch._check(
                y_dtype is None or y_dtype in _ACL_DTYPE_TO_TORCH_DTYPE,
                lambda: f"y_dtype only supports 15 (BF16) / 5 (FP16) dtype enum, "
                f"but got {y_dtype}, {ops_error(ErrCode.VALUE)}.",
            )
            out_dtype = (
                _ACL_DTYPE_TO_TORCH_DTYPE[y_dtype]
                if y_dtype is not None
                else torch.bfloat16
            )
            y = x1.new_empty(tuple([m_per_rank * rank_size, n]), dtype=out_dtype)
            gather_out = x1.new_empty(
                tuple([m_per_rank * rank_size, k]), dtype=x1.dtype
            )
            amax_out = x1.new_empty(tuple([0]), dtype=torch.float32)
            return (y, gather_out, amax_out)


_all_gather_quant_matmul_op_builder = _AllGatherQuantMatmulOpBuilder()
_all_gather_quant_matmul_op_builder._ensure_initialized()


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
    comm_mode=None,
):
    op_module = (
        _all_gather_quant_matmul_op_builder.load()
    )  # 编译all_gather_quant_matmul.cpp产出.so，import成python模块
    result = op_module.npu_all_gather_quant_matmul(
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
        y_dtype if y_dtype is not None else 15,  # ACL BF16
        comm_mode if comm_mode is not None else "ubmem",
    )
    return result


def _resolve_quant_kind(tensor, dtype_enum, name):
    """返回输入量化类型(e4m3/e5m2/fp4), 并校验存储 dtype 与 dtype 枚举覆盖一致。"""
    # 挡字符串等, None=不指定也合法
    if dtype_enum is not None and not isinstance(dtype_enum, (int, torch.dtype)):
        raise ValueError(
            f"{name}_dtype must be an int (ACL dtype enum), a torch.dtype, or None, "
            f"but got {type(dtype_enum).__name__}."
        )
    if tensor.dtype == torch.uint8:
        # uint8 存储: 296 -> fp4 打包; 24/23 -> fp8 bitcast(如 e5m2 无法直接 format_cast)
        if dtype_enum is None:  # uint8时，枚举必须填，不能None
            raise ValueError(
                f"{name} is a uint8 tensor, please pass {name}_dtype "
                f"(296 fp4 / 24 e4m3 / 23 e5m2)."
            )
        resolved = _resolve_dtype_enum(dtype_enum, tensor.dtype)
        if resolved == _DTYPE_ENUM_FP4_E2M1:
            return "fp4"
        if resolved == _DTYPE_ENUM_FP8_E4M3FN:
            return "e4m3"
        if resolved == _DTYPE_ENUM_FP8_E5M2:
            return "e5m2"
        raise ValueError(
            f"{name}_dtype={dtype_enum} is not supported for uint8 storage; "
            f"expected 296 (fp4) / 24 (e4m3) / 23 (e5m2)."
        )
    elif tensor.dtype == torch.float8_e4m3fn:
        kind, expect_enum = "e4m3", _DTYPE_ENUM_FP8_E4M3FN
    elif tensor.dtype == torch.float8_e5m2:
        kind, expect_enum = "e5m2", _DTYPE_ENUM_FP8_E5M2
    else:
        raise ValueError(
            f"{name} only supports torch.float8_e4m3fn, torch.float8_e5m2 or "
            f"uint8 (bitcast, pass {name}_dtype), but got {tensor.dtype}."
        )
    if (
        dtype_enum is not None
        and _resolve_dtype_enum(dtype_enum, tensor.dtype) != expect_enum
    ):
        raise ValueError(
            f"{name}_dtype is inconsistent with the tensor dtype: expected "
            f"{expect_enum} for {tensor.dtype}, but got {dtype_enum}."
        )
    return kind


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
    """Validate input parameters (ubmem path only)."""
    if x1 is None:
        raise ValueError("x1 must not be None.")
    if x2 is None:
        raise ValueError("x2 must not be None.")
    if x1.dim() != 2:
        raise ValueError(f"x1 should be 2D, but got {x1.dim()}D.")
    if x2.dim() not in (2, 4):
        raise ValueError(
            f"x2 should be 2D (ND/NZ logical) or 4D (FRACTAL_NZ storage), "
            f"but got {x2.dim()}D."
        )
    if comm_mode is not None and comm_mode != "ubmem":
        raise ValueError(
            f"comm_mode only supports 'ubmem' in this experimental build, but got '{comm_mode}'."
        )
    x1_kind = _resolve_quant_kind(x1, x1_dtype, "x1")
    x2_kind = _resolve_quant_kind(x2, x2_dtype, "x2")
    if (x1_kind == "fp4") != (x2_kind == "fp4"):
        raise ValueError(
            "FP4 requires both x1 and x2 to be packed uint8 tensors with "
            "x1_dtype=296 / x2_dtype=296."
        )
    fp8_e8m0_dtype_enum = 293
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
    for name, scale, scale_dtype in (
        ("x1_scale", x1_scale, x1_scale_dtype),
        ("x2_scale", x2_scale, x2_scale_dtype),
    ):
        if (
            scale is not None
            and scale.dtype == torch.uint8
            and scale_dtype != fp8_e8m0_dtype_enum
        ):
            raise ValueError(
                f"{name} is uint8 tensor (fp8_e8m0 stored as uint8) but "
                f"{name}_dtype={scale_dtype} is not 293 (fp8_e8m0). Please pass {name}_dtype=293."
            )


_UBMEM_STATUS_REGION_BYTES = 4 * 1024
_UBMEM_WINDOW_ALIGN = 512


def _ubmem_window_bytes(x1, rank_size, x1_scale):
    """ubmem 窗口大小: 4KB status 区 + gather data 区 + scale 区, 全 512 对齐。

    data 字节数直接取 x1.size(1): fp8 系为 K, fp4 打包存储为 K/2;
    scale 字节数取 x1_scale 实际维度 [M, Ceil(K/64), 2]。
    """
    m = x1.size(0)
    data_bytes = rank_size * m * x1.size(1)
    data_bytes = (
        (data_bytes + _UBMEM_WINDOW_ALIGN - 1)
        // _UBMEM_WINDOW_ALIGN
        * _UBMEM_WINDOW_ALIGN
    )
    scale_bytes = (
        rank_size * m * x1_scale.size(1) * x1_scale.size(2)
        if x1_scale is not None
        else 0
    )
    total = _UBMEM_STATUS_REGION_BYTES + data_bytes + scale_bytes
    return (
        (total + _UBMEM_WINDOW_ALIGN - 1) // _UBMEM_WINDOW_ALIGN * _UBMEM_WINDOW_ALIGN
    )


_TORCH_DTYPE_TO_ENUM = {
    torch.float16: _DTYPE_ENUM_FP16,
    torch.bfloat16: _DTYPE_ENUM_BF16,
    torch.float8_e5m2: _DTYPE_ENUM_FP8_E5M2,
    torch.float8_e4m3fn: _DTYPE_ENUM_FP8_E4M3FN,
}


def _resolve_dtype_enum(dtype_enum, tensor_dtype, default=None):
    # 把调用方三种可能形态的dtype参数，统一为枚举
    if isinstance(dtype_enum, int):
        return dtype_enum
    if isinstance(dtype_enum, torch.dtype):
        return _TORCH_DTYPE_TO_ENUM.get(dtype_enum)
    if tensor_dtype is not None:
        return _TORCH_DTYPE_TO_ENUM.get(tensor_dtype)
    return default


def _check_hccl_buffer(window_bytes, hccl_buffer_size):
    """Check gather window bytes <= hccl_buffer_size（窗口落在各 rank 内置 HCCL buffer 内）."""
    if window_bytes > hccl_buffer_size:
        raise ValueError(
            f"ubmem gather window required {window_bytes} bytes "
            f"({window_bytes / 1024 / 1024:.2f} MB), exceeds hccl_buffer_size "
            f"({hccl_buffer_size} bytes = {hccl_buffer_size / 1024 / 1024:.2f} MB). "
            f"Please increase HCCL_BUFFSIZE environment variable."
        )


def _prepare_comm_context(group, window_bytes):
    """Create ubmem communication context (CreateDeviceContext way), return context/rank info."""
    rank_id = torch.distributed.get_rank(group)
    rank_size = torch.distributed.get_world_size(group)
    if rank_size <= 0:
        rank_size = torch.distributed.get_rank_size()
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
    cached = _UBMEM_CTX_CACHE.get(group_name)
    if cached is not None:
        context, buffer_size = cached
    else:
        # 加载那个so，op_module不是.so文件本身，而是.so被加载进程后形成的Python模块对象
        # 可以通过这个op_module调用cpp里面的函数
        op_module = _all_gather_quant_matmul_op_builder.load()
        context = op_module.create_ubmem_context(group_name, UBMEM_CONTEXT_TAG)
        buffer_size = op_module.get_hccl_buffer_size(group_name)
        _UBMEM_CTX_CACHE[group_name] = (context, buffer_size)
        _logger.info(
            "ubmem ctx created (group=%s): hccl_buffer_size=%s bytes (%.2f MB)",
            group_name,
            buffer_size,
            buffer_size / 1024 / 1024,
        )
    _check_hccl_buffer(window_bytes, buffer_size)
    return context, window_bytes, rank_id, rank_size, group_name


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
    comm_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AllGather + MX quantized Matmul fused computation (ubmem/MTE window path).

    Each rank gathers A from all ranks via the ubmem window, then multiplies
    with its local B: ``y = AllGather(x1) @ x2^T``.

    ubmem-only (experimental): x1/x2 may be float8_e4m3fn / float8_e5m2
    (any mix) or packed fp4 (uint8, x1_dtype=x2_dtype=296), y may be
    bfloat16/float16, bias is not supported, and ``comm_mode`` must be
    ``"ubmem"`` (default).

    Args:
        x1 (Tensor): Left matrix (per-rank A), shape ``(M_per_rank, K)``, dtype
            float8_e4m3fn / float8_e5m2, or ``(M_per_rank, K/2)`` uint8 for fp4.
        x2 (Tensor): Right matrix, dtype float8_e4m3fn / float8_e5m2, or
            ``(N, K/2)`` uint8 for fp4. 支持转置/非转置两种方向(按形状自动识别):
            转置 B ``[N,K]`` (ND 传 ``x2_nk.t()``, NZ 传 ``npu_format_cast(x2_nk, 29)``);
            非转置 B ``[K,N]`` (ND 传连续的 ``x2_kn``, NZ 传 ``npu_format_cast(x2_kn, 29)``)。
            NZ 可以是 2D 逻辑 shape 或 4D storage view; fp4 仅支持 ND。
        group (ProcessGroup): torch.distributed process group for AllGather
            communication.
        bias (Tensor, optional): Not supported in the ubmem path (must be None).
        x1_scale (Tensor, optional): MX scale for x1, shape ``(M_per_rank, Ceil(K/64), 2)``,
            dtype float8_e8m0fnu. Required in MX scene.
        x2_scale (Tensor, optional): MX scale for x2, shape ``(N, Ceil(K/64), 2)``, dtype
            float8_e8m0fnu. Required in MX scene.
        group_sizes (List[int], optional): Quantization group sizes, MX only
            supports ``[1, 1, 32]``. None means auto-inferred from scale shapes.
        x1_dtype (int, optional): x1 dtype enum for bitcast, 24 (e4m3) / 23 (e5m2) /
            296 (fp4, required for packed uint8); None means use tensor dtype.
        x2_dtype (int, optional): x2 dtype enum for bitcast, same values as x1_dtype.
        x1_scale_dtype (int, optional): x1_scale dtype enum for bitcast, None means use tensor dtype.
        x2_scale_dtype (int, optional): x2_scale dtype enum for bitcast, None means use tensor dtype.
        y_dtype (int, optional): Output dtype enum, None means default bfloat16.
            Supports bfloat16 (15) / float16 (5).

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (y, gather_out, amax_out) where y has
            shape ``(M_per_rank * rank_size, N)``, gather_out and amax_out are
            empty tensor (not implemented yet). Users typically use
            ``y, _, _ = all_gather_quant_matmul(...)`` to ignore the latter two.
    """
    if group_sizes is None:
        group_sizes = []
    if comm_mode is None:
        comm_mode = "ubmem"
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
    if bias is not None:
        raise ValueError("bias is not supported in the ubmem path.")
    if (
        y_dtype is not None
        and _resolve_dtype_enum(y_dtype, None, default=_DTYPE_ENUM_BF16)
        not in _ACL_DTYPE_TO_TORCH_DTYPE
    ):
        raise ValueError(
            f"y_dtype only supports 15 (BF16) / 5 (FP16) dtype enum, but got {y_dtype}."
        )
    rank_size = torch.distributed.get_world_size(group)
    if rank_size <= 0:
        rank_size = torch.distributed.get_rank_size()
    window_bytes = _ubmem_window_bytes(x1, rank_size, x1_scale)
    context, hccl_buffer_size, rank_id, rank_size, group_name = _prepare_comm_context(
        group, window_bytes
    )
    _logger.info(
        "rank=%s, rank_size=%s, comm_mode=ubmem, calling npu_all_gather_quant_matmul ...",
        rank_id,
        rank_size,
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
        y_dtype=y_dtype if y_dtype is not None else _DTYPE_ENUM_BF16,
        comm_mode=comm_mode,
    )
    _logger.info("rank=%s, npu_all_gather_quant_matmul returned", rank_id)
    return result
