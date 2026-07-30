# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

DIM_QK = 128


class StemOamPrepPagedKvOpBuilder(OpBuilder):
    def __init__(self):
        super(StemOamPrepPagedKvOpBuilder, self).__init__("stem_oam_prep_paged_kv")

    def sources(self):
        return ["ops/csrc/stem_oam_prep_paged_kv.cpp"]

    def schema(self) -> str:
        return [
            "stem_oam_prep_paged_kv(Tensor k_cache, Tensor v_cache, Tensor kv_indices, "
            "int[] kv_seq_lens, Tensor k_scale_cache, Tensor v_scale, "
            "float lambda_mag, int cache_layout, int kv_block_size, "
            "int stem_block_size, int stem_stride) -> (Tensor, Tensor)"
        ]

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def stem_oam_prep_paged_kv_meta(
            k_cache,
            v_cache,
            kv_indices,
            kv_seq_lens,
            k_scale_cache,
            v_scale,
            lambda_mag,
            cache_layout,
            kv_block_size,
            stem_block_size,
            stem_stride,
        ):
            batch = kv_indices.shape[0]
            num_heads = k_cache.shape[2] if cache_layout == 0 else k_cache.shape[1]
            max_kv_len = max(kv_seq_lens)
            max_kb = (max_kv_len + stem_block_size - 1) // stem_block_size
            if max_kb < 1:
                max_kb = 1
            kflat_dim = stem_stride * DIM_QK
            k_flat = torch.empty(
                [batch, num_heads, max_kb, kflat_dim],
                dtype=torch.bfloat16,
                device="meta",
            )
            v_bias = torch.empty(
                [batch, num_heads, max_kb], dtype=torch.float32, device="meta"
            )
            return (k_flat, v_bias)


stem_oam_prep_paged_kv_op_builder = StemOamPrepPagedKvOpBuilder()


@impl(AS_LIBRARY, "stem_oam_prep_paged_kv", "PrivateUse1")
def stem_oam_prep_paged_kv(
    k_cache,
    v_cache,
    kv_indices,
    kv_seq_lens,
    k_scale_cache,
    v_scale,
    lambda_mag=0.3,
    cache_layout=0,
    kv_block_size=64,
    stem_block_size=128,
    stem_stride=16,
):
    op_module = stem_oam_prep_paged_kv_op_builder.load()
    return op_module.stem_oam_prep_paged_kv(
        k_cache,
        v_cache,
        kv_indices,
        kv_seq_lens,
        k_scale_cache,
        v_scale,
        lambda_mag,
        cache_layout,
        kv_block_size,
        stem_block_size,
        stem_stride,
    )
