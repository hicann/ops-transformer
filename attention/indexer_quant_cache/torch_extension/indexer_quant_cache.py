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


class IndexerQuantMode(IntEnum):
    """quant_mode 枚举：对外字符串经由本枚举映射为传给算子侧的 int。"""

    MXFP8 = 0  # "mxfp8"    MX-FP8：微缩放 FP8，每 32 元素一组，e8m0 scale
    FP8 = 1  # "fp8"      逐 token 动态 FP8(e4m3/e5m2)，整行一个 float32 scale
    HIFLOAT8 = 2  # "hifloat8" HiFloat8 静态量化
    MXFP4 = 3  # "mxfp4"    MX-FP4：微缩放 FP4，每 32 元素一组，e8m0 scale


DEFAULT_QUANT_MODE = "fp8"


def _resolve_quant_mode(quant_mode: Union[str, int]) -> int:
    """对外 str quant_mode 与内部枚举匹配，返回传给算子侧的 int；兼容直接传 int/IntEnum。"""
    if isinstance(quant_mode, str):
        try:
            return int(IndexerQuantMode[quant_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in IndexerQuantMode)
            raise ValueError(
                f"quant_mode should be one of [{valid}], but got {quant_mode!r}"
            ) from exc
    return int(IndexerQuantMode(quant_mode))  # int/IntEnum：校验取值合法性


class IndexerQuantCacheOpBuilder(OpBuilder):
    def __init__(self):
        super(IndexerQuantCacheOpBuilder, self).__init__(
            "indexer_quant_cache", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/indexer_quant_cache.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "indexer_quant_cache(Tensor(a!) cache, Tensor(b!) cache_scale, Tensor x, "
            'Tensor slot_mapping, *, str quant_mode="fp8", bool round_scale=True, '
            "float x_scale=1.0) -> ()"
        )

    def register_meta(self):
        """
        Registers the Meta implementation.
        """

        @impl(get_as_library(), self.name, "Meta")
        def indexer_quant_cache_meta(
            cache,
            cache_scale,
            x,
            slot_mapping,
            *,
            quant_mode=DEFAULT_QUANT_MODE,
            round_scale=True,
            x_scale=1.0,
        ) -> None:
            return None


# Instantiate the builder
indexer_quant_cache_op_builder = IndexerQuantCacheOpBuilder()
indexer_quant_cache_op_builder._ensure_initialized()


@impl(get_as_library(), indexer_quant_cache_op_builder.name, "PrivateUse1")
def _indexer_quant_cache(
    cache: torch.Tensor,
    cache_scale: torch.Tensor,
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    quant_mode: str = DEFAULT_QUANT_MODE,
    round_scale: bool = True,
    x_scale: float = 1.0,
) -> None:
    quant_mode_int = _resolve_quant_mode(quant_mode)
    op_module = indexer_quant_cache_op_builder.load()
    op_module.indexer_quant_cache(
        cache, cache_scale, x, slot_mapping, quant_mode_int, round_scale, x_scale
    )


def indexer_quant_cache(
    cache: torch.Tensor,
    cache_scale: torch.Tensor,
    x: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    quant_mode: str = DEFAULT_QUANT_MODE,
    round_scale: bool = True,
    x_scale: float = 1.0,
) -> None:
    return torch.ops.cann_ops_transformer.indexer_quant_cache(
        cache,
        cache_scale,
        x,
        slot_mapping,
        quant_mode=quant_mode,
        round_scale=round_scale,
        x_scale=x_scale,
    )
