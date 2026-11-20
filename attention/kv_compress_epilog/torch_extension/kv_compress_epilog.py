# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
from typing import Union
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class KvCompressQuantMode(IntEnum):
    """quant_mode 枚举：对外字符串经由本枚举映射为传给算子侧的 int。"""

    FP8_BF16 = 0  # "fp8_bf16"     per-group FP8(e4m3)，bf16 scale
    FP8_E8M0 = 1  # "fp8_e8m0"     per-group FP8(e4m3)，e8m0 scale（即 MX-FP8 微缩放）
    HIFLOAT8_FP4 = (
        2  # "hifloat8_fp4" rope hifloat8 静态 + nope per-group FLOAT4_E2M1 动态
    )
    INT8_BF16 = 1  # "int8_bf16"     arch22(A2/A3)承载：与FP8_E8M0同值1按平台分流；nope per-group(64) int8 对称动态量化，bf16 scale，rope 保留 bf16
    # 对外字符串取枚举名小写："nope_rope_fp8_bf16" / "nope_rope_fp4_bf16"
    # （全行量化覆盖 nope+rope 段，量化格式为 MXFP8 / packed MXFP4，x 为 bf16）。
    NOPE_ROPE_FP8_BF16 = (
        3  # 全行 MXFP8（E4M3FN/E5M2 cache，per-group amax scale，可选 2 的幂取整）
    )
    NOPE_ROPE_FP4_BF16 = (
        4  # 全行 packed MXFP4（E2M1 两元素一字节，共享指数 scale；主场景 group=16）
    )


DEFAULT_QUANT_MODE = "fp8_e8m0"


def _resolve_quant_mode(quant_mode: Union[str, int]) -> int:
    """对外 str quant_mode 与内部枚举匹配，返回传给算子侧的 int；兼容直接传 int/IntEnum。"""
    if isinstance(quant_mode, str):
        try:
            return int(KvCompressQuantMode[quant_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(n.lower() for n in KvCompressQuantMode.__members__)
            raise ValueError(
                f"quant_mode should be one of [{valid}], but got {quant_mode!r}"
            ) from exc
    try:
        return int(KvCompressQuantMode(quant_mode))
    except ValueError as exc:
        valid = ", ".join(m.name.lower() for m in KvCompressQuantMode)
        raise ValueError(
            f"quant_mode should be one of [{valid}], but got {quant_mode!r}"
        ) from exc


class KvCompressEpilogOpBuilder(OpBuilder):
    def __init__(self):
        super(KvCompressEpilogOpBuilder, self).__init__(
            "kv_compress_epilog", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/kv_compress_epilog.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "kv_compress_epilog(Tensor(a!) cache, Tensor x, Tensor slot_mapping, *, "
            'int quant_group_size=64, str quant_mode="fp8_e8m0", bool round_scale=True, '
            "float x_scale=1.0) -> ()"
        )

    def register_meta(self):
        """
        Registers the Meta implementation.
        """

        @impl(get_as_library(), self.name, "Meta")
        def kv_compress_epilog_meta(
            cache,
            x,
            slot_mapping,
            *,
            quant_group_size=64,
            quant_mode=DEFAULT_QUANT_MODE,
            round_scale=True,
            x_scale=1.0,
        ) -> None:
            # mode 3/4 早失败校验（与 csrc TORCH_CHECK 同步；legacy mode 0/1/2 维持 v1 行为不在 meta 拦截）
            mode = _resolve_quant_mode(quant_mode)
            if mode in (
                KvCompressQuantMode.NOPE_ROPE_FP8_BF16,
                KvCompressQuantMode.NOPE_ROPE_FP4_BF16,
            ):
                if mode == KvCompressQuantMode.NOPE_ROPE_FP8_BF16:
                    if quant_group_size != 32:
                        raise ValueError(
                            f"quant_group_size should be 32 when quant_mode is 3, but got {quant_group_size}"
                        )
                elif quant_group_size not in (16, 32):
                    raise ValueError(
                        f"quant_group_size should be 16 or 32 when quant_mode is 4, but got {quant_group_size}"
                    )
                tail_dim = x.shape[-1]
                if tail_dim <= 0:
                    raise ValueError(
                        f"The last dim (d) of x should be positive when quant_mode is 3/4, but got {tail_dim}"
                    )
                if tail_dim % quant_group_size != 0:
                    raise ValueError(
                        "The last dim (d) of x should be divisible by quant_group_size when quant_mode is 3/4, "
                        f"but got d {tail_dim} and group {quant_group_size}"
                    )
                if tail_dim % 128 != 0:
                    raise ValueError(
                        f"The last dim (d) of x should be a multiple of 128 when quant_mode is 3/4, but got {tail_dim}"
                    )
                if cache.dim() not in (2, 4):
                    raise ValueError(
                        f"cache should be 2D (flat) or 4D (paged) when quant_mode is 3/4, but got {cache.dim()}-dim"
                    )
                if mode == KvCompressQuantMode.NOPE_ROPE_FP8_BF16:
                    if cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                        raise ValueError(
                            "cache dtype should be float8_e4m3fn or float8_e5m2 when quant_mode==3, "
                            f"but got {cache.dtype}"
                        )
                elif cache.dtype is not torch.uint8:
                    raise ValueError(
                        f"cache dtype should be uint8 when quant_mode==4, but got {cache.dtype}"
                    )
                if x.dtype is not torch.bfloat16:
                    raise ValueError(
                        f"x dtype should be bfloat16 when quant_mode is 3/4, but got {x.dtype}"
                    )
                if slot_mapping.dtype not in (torch.int32, torch.int64):
                    raise ValueError(
                        "slot_mapping dtype should be int32 or int64 when quant_mode is 3/4, "
                        f"but got {slot_mapping.dtype}"
                    )
            return None


# Instantiate the builder
kv_compress_epilog_op_builder = KvCompressEpilogOpBuilder()
kv_compress_epilog_op_builder._ensure_initialized()


@impl(get_as_library(), kv_compress_epilog_op_builder.name, "PrivateUse1")
def _kv_compress_epilog(
    cache: torch.Tensor,
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    quant_group_size: int = 64,
    quant_mode: str = DEFAULT_QUANT_MODE,
    round_scale: bool = True,
    x_scale: float = 1.0,
) -> None:
    quant_mode_int = _resolve_quant_mode(quant_mode)
    op_module = kv_compress_epilog_op_builder.load()
    op_module.kv_compress_epilog(
        cache, x, slot_mapping, quant_group_size, quant_mode_int, round_scale, x_scale
    )


def kv_compress_epilog(
    cache: torch.Tensor,
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    quant_group_size: int = 64,
    quant_mode: str = DEFAULT_QUANT_MODE,
    round_scale: bool = True,
    x_scale: float = 1.0,
) -> None:
    return torch.ops.cann_ops_transformer.kv_compress_epilog(
        cache,
        x,
        slot_mapping,
        quant_group_size=quant_group_size,
        quant_mode=quant_mode,
        round_scale=round_scale,
        x_scale=x_scale,
    )
