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


class UndGenQkvRmsNormRopeCacheOpBuilder(OpBuilder):
    def __init__(self):
        super(UndGenQkvRmsNormRopeCacheOpBuilder, self).__init__(
            "und_gen_qkv_rms_norm_rope_cache", category="posembedding"
        )

    def sources(self):
        return ["csrc/posembedding/und_gen_qkv_rms_norm_rope_cache.cpp"]

    def schema(self) -> str:
        return [
            # k_cache/v_cache 由算子原地写入，必须带 (a!)/(b!) 别名标注，
            # 否则 functionalization / torch.compile 会把这个原地写当成无副作用而丢掉 cache 更新
            "und_gen_qkv_rms_norm_rope_cache(Tensor und_qkv, Tensor und_weights_q, Tensor und_weights_k, "
            "Tensor cos_sin_cache, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor slot_mapping, Tensor positions, "
            "Tensor? gen_qkv, Tensor? gen_weights_q, Tensor? gen_weights_k, Tensor? cat_indices, "
            "int num_heads_q, int num_heads_k, int num_heads_v, float norm_eps, int[] mrope_section) -> Tensor"
        ]

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def und_gen_qkv_rms_norm_rope_cache_meta(
            und_qkv,
            und_weights_q,
            und_weights_k,
            cos_sin_cache,
            k_cache,
            v_cache,
            slot_mapping,
            positions,
            gen_qkv,
            gen_weights_q,
            gen_weights_k,
            cat_indices,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            norm_eps,
            mrope_section,
        ):
            head_dim = und_qkv.shape[2]
            total = und_qkv.shape[0] + (0 if gen_qkv is None else gen_qkv.shape[0])
            return torch.empty(
                [total, num_heads_q, head_dim], dtype=und_qkv.dtype, device="meta"
            )


und_gen_qkv_rms_norm_rope_cache_op_builder = UndGenQkvRmsNormRopeCacheOpBuilder()
und_gen_qkv_rms_norm_rope_cache_op_builder._ensure_initialized()


@impl(get_as_library(), "und_gen_qkv_rms_norm_rope_cache", "PrivateUse1")
def und_gen_qkv_rms_norm_rope_cache(
    und_qkv,
    und_weights_q,
    und_weights_k,
    cos_sin_cache,
    k_cache,
    v_cache,
    slot_mapping,
    positions,
    gen_qkv=None,
    gen_weights_q=None,
    gen_weights_k=None,
    cat_indices=None,
    num_heads_q=8,
    num_heads_k=1,
    num_heads_v=1,
    norm_eps=1e-6,
    mrope_section=(),
):
    """und/gen QKV 融合 RMSNorm + MRoPE，并把 K/V 写入分页 KV Cache（k_cache/v_cache 原地更新）。"""
    op_module = und_gen_qkv_rms_norm_rope_cache_op_builder.load()
    return op_module.und_gen_qkv_rms_norm_rope_cache(
        und_qkv,
        und_weights_q,
        und_weights_k,
        cos_sin_cache,
        k_cache,
        v_cache,
        slot_mapping,
        positions,
        gen_qkv,
        gen_weights_q,
        gen_weights_k,
        cat_indices,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        norm_eps,
        list(mrope_section),
    )
