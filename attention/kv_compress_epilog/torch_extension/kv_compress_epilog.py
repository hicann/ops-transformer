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


DEFAULT_QUANT_MODE = "fp8_e8m0"


def _resolve_quant_mode(quant_mode: Union[str, int]) -> int:
    """对外 str quant_mode 与内部枚举匹配，返回传给算子侧的 int；兼容直接传 int/IntEnum。"""
    if isinstance(quant_mode, str):
        try:
            return int(KvCompressQuantMode[quant_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in KvCompressQuantMode)
            raise ValueError(
                f"quant_mode should be one of [{valid}], but got {quant_mode!r}"
            ) from exc
    return int(KvCompressQuantMode(quant_mode))  # int/IntEnum：校验取值合法性


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
