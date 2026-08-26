# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from typing import List, Optional

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache.default
    )
    def convert_und_gen_qkv_rms_norm_rope_cache(
        und_qkv: Tensor,
        und_weights_q: Tensor,
        und_weights_k: Tensor,
        cos_sin_cache: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
        positions: Tensor,
        gen_qkv: Optional[Tensor] = None,
        gen_weights_q: Optional[Tensor] = None,
        gen_weights_k: Optional[Tensor] = None,
        cat_indices: Optional[Tensor] = None,
        *,
        num_heads_q: int = 8,
        num_heads_k: int = 1,
        num_heads_v: int = 1,
        norm_eps: float = 1e-6,
        mrope_section: List[int] = (),
        meta_outputs: TensorSpec = None,
    ):
        return ge.UndGenQkvRmsNormRopeCache(
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
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            num_heads_v=num_heads_v,
            norm_eps=norm_eps,
            mrope_section=list(mrope_section),
        )
else:

    def convert_und_gen_qkv_rms_norm_rope_cache(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
