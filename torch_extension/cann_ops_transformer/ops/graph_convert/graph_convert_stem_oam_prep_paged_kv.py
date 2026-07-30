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
    from typing import List

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.stem_oam_prep_paged_kv.default
    )
    def converter_stem_oam_prep_paged_kv(
        k_cache: Tensor,
        v_cache: Tensor,
        kv_indices: Tensor,
        kv_seq_lens: List[int],
        k_scale_cache: Tensor,
        v_scale: Tensor,
        *,
        lambda_mag: float = 0.3,
        cache_layout: int = 0,
        kv_block_size: int = 64,
        stem_block_size: int = 128,
        stem_stride: int = 16,
        meta_outputs: TensorSpec = None,
    ):
        return ge.StemOamPrepPagedKv(
            k_cache,
            v_cache,
            kv_indices,
            kv_seq_lens,
            k_scale_cache,
            v_scale,
            lambda_mag=lambda_mag,
            cache_layout=cache_layout,
            kv_block_size=kv_block_size,
            stem_block_size=stem_block_size,
            stem_stride=stem_stride,
        )
else:

    def converter_stem_oam_prep_paged_kv(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
