# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class ScatterPaKvCacheWithKScaleOpBuilder(OpBuilder):
    """
    torch_npu适配：scatter_pa_kv_cache_with_k_scale算子

    功能：将FP8格式的key/value和对应的key_scale更新到KV cache中

    输入：
        key: [num_tokens, num_head, k_head_size] FP8格式
        value: [num_tokens, num_head, v_head_size] FP8格式
        key_cache: [num_blocks, num_head, block_size, k_head_size] FP8格式
        value_cache: [num_blocks, num_head, block_size, v_head_size] FP8格式
        slot_mapping: [num_tokens] int32/int64
        key_scale: [num_tokens, num_head] float32
        key_scale_cache: [num_blocks, num_head, block_size, 1] float32

    输出（inplace更新）：
        key_cache, value_cache, key_scale_cache
    """

    def __init__(self):
        super(ScatterPaKvCacheWithKScaleOpBuilder, self).__init__(
            "scatter_pa_kv_cache_with_k_scale", category="attention"
        )

    def sources(self):
        """C++源代码路径"""
        return ["csrc/attention/scatter_pa_kv_cache_with_k_scale.cpp"]

    def schema(self) -> str:
        """PyTorch算子签名"""
        return (
            "scatter_pa_kv_cache_with_k_scale("
            "Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, "
            "Tensor slot_mapping, Tensor key_scale, Tensor(c!) key_scale_cache, *, "
            "str cache_layout='BNBD') -> ()"
        )

    def register_meta(self):
        """
        Meta实现：Shape/Dtype推导

        原地更新key_cache、value_cache、key_scale_cache，无返回值
        """

        @impl(get_as_library(), self.name, "Meta")
        def scatter_pa_kv_cache_with_k_scale_meta(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            key_scale,
            key_scale_cache,
            *,
            cache_layout="BNBD",
        ) -> None:
            return None


# 实例化OpBuilder
scatter_pa_kv_cache_with_k_scale_op_builder = ScatterPaKvCacheWithKScaleOpBuilder()
scatter_pa_kv_cache_with_k_scale_op_builder._ensure_initialized()


@impl(get_as_library(), scatter_pa_kv_cache_with_k_scale_op_builder.name, "PrivateUse1")
def scatter_pa_kv_cache_with_k_scale(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    *,
    cache_layout: Optional[str] = "BNBD",
) -> None:
    """
    torch_npu实现：scatter_pa_kv_cache_with_k_scale

    Args:
        key: [num_tokens, num_head, k_head_size] FP8格式，需要更新的key
        value: [num_tokens, num_head, v_head_size] FP8格式，需要更新的value
        key_cache: [num_blocks, num_head, block_size, k_head_size] FP8格式，被原地更新的cache
        value_cache: [num_blocks, num_head, block_size, v_head_size] FP8格式，被原地更新的cache
        slot_mapping: [num_tokens] int32/int64，每个token在cache中的偏移
        key_scale: [num_tokens, num_head] float32，需要更新的scale
        key_scale_cache: [num_blocks, num_head, block_size, 1] float32，被原地更新的scale cache
        cache_layout: str，cache布局格式（默认'BNBD'）

    注意：
        1. 算子支持FP8_E5M2和FP8_E4M3FN两种dtype
        2. slot_mapping值必须在[0, num_blocks*block_size-1]范围内
        3. 超出范围的slot会被跳过（不更新对应cache）
        4. 原地更新接口，无返回值，key_cache/value_cache/key_scale_cache 调用后即被更新
    """
    op_module = scatter_pa_kv_cache_with_k_scale_op_builder.load()
    op_module.scatter_pa_kv_cache_with_k_scale(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
        cache_layout,
    )
