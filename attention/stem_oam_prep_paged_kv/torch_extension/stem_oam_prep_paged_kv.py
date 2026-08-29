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
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

DIM_QK = 128


class StemOamPrepPagedKvOpBuilder(OpBuilder):
    def __init__(self):
        super(StemOamPrepPagedKvOpBuilder, self).__init__(
            "stem_oam_prep_paged_kv", category="attention"
        )

    def sources(self):
        return ["csrc/attention/stem_oam_prep_paged_kv.cpp"]

    def schema(self) -> str:
        return [
            "stem_oam_prep_paged_kv(Tensor k_cache, Tensor v_cache, Tensor kv_indices, "
            "SymInt[] kv_seq_lens, *, "
            "Tensor? k_scale_cache=None, Tensor? v_scale=None, "
            'float lambda_mag=0.3, str kv_layout="BNBD", '
            "int stem_block_size=128, int stem_stride=16) -> (Tensor, Tensor)"
        ]

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def stem_oam_prep_paged_kv_meta(
            k_cache,
            v_cache,
            kv_indices,
            kv_seq_lens,
            *,
            k_scale_cache=None,
            v_scale=None,
            lambda_mag=0.3,
            kv_layout="BNBD",
            stem_block_size=128,
            stem_stride=16,
        ):
            batch = kv_indices.shape[0]
            num_heads = k_cache.shape[2] if kv_layout == "BBND" else k_cache.shape[1]
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
stem_oam_prep_paged_kv_op_builder._ensure_initialized()


@impl(get_as_library(), "stem_oam_prep_paged_kv", "PrivateUse1")
def stem_oam_prep_paged_kv(
    k_cache,
    v_cache,
    kv_indices,
    kv_seq_lens,
    *,
    k_scale_cache=None,
    v_scale=None,
    lambda_mag=0.3,
    kv_layout="BNBD",
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
        kv_layout,
        stem_block_size,
        stem_stride,
    )
