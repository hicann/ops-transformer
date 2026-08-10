# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .sparse_flash_mla_grad import sparse_flash_mla_grad, sparse_flash_mla_grad_metadata
from .moe_finalize_routing import moe_finalize_routing
from .moe_token_permute import moe_token_permute
from .mega_moe import get_symm_buffer_for_mega_moe, mega_moe
from .deep_ep import MoeDistributeBuffer
from .flash_attn import flash_attn, flash_attn_metadata
from .quant_flash_attn import quant_flash_attn, quant_flash_attn_metadata
from .mixed_quant_sparse_flash_mla import (
    mixed_quant_sparse_flash_mla,
    mixed_quant_sparse_flash_mla_metadata,
)
from .quant_sparse_flash_mla import (
    quant_sparse_flash_mla,
    quant_sparse_flash_mla_metadata,
)
from .scatter_pa_kv_cache_with_k_scale import scatter_pa_kv_cache_with_k_scale
from .sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from .sparse_flash_mla_softmax_l1_norm import (
    sparse_flash_mla_softmax_l1_norm,
    sparse_flash_mla_softmax_l1_norm_metadata,
)
from .sparse_lightning_indexer_kl_loss_grad import (
    sparse_lightning_indexer_kl_loss_grad,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)
from .dense_lightning_indexer_kl_loss_grad import (
    dense_lightning_indexer_kl_loss_grad,
    dense_lightning_indexer_kl_loss_grad_metadata,
)
from .lightning_indexer import lightning_indexer, lightning_indexer_metadata
from .quant_lightning_indexer import (
    quant_lightning_indexer,
    quant_lightning_indexer_metadata,
)
from .mhc_post import mhc_post
from .mhc_pre_sinkhorn import mhc_pre_sinkhorn
from .mhc_pre_sinkhorn_backward import mhc_pre_sinkhorn_backward
from .kv_compress_epilog import kv_compress_epilog
from .indexer_quant_cache import indexer_quant_cache
from .compressor import compressor
from .quant_compressor import quant_compressor
from .inplace_partial_rotary_mul import inplace_partial_rotary_mul
from .inplace_partial_rotary_mul_backward import inplace_partial_rotary_mul_backward
from .stem_oam_prep_varlen_q import stem_oam_prep_varlen_q
from .causal_conv1d_fn import causal_conv1d_fn
from .causal_conv1d_update import causal_conv1d_update
from .stem_oam_prep_paged_kv import stem_oam_prep_paged_kv
from .qkv_rms_norm_rope_cache_with_k_scale import (
    qkv_rms_norm_rope_cache_with_k_scale,
    qkv_rms_norm_rope_cache_with_k_scale_,
)
from .elastic_buffer import ElasticBuffer, EPHandle
from .grouped_matmul_activation_quant import grouped_matmul_activation_quant
from .dense_lightning_indexer_softmax_lse import (
    dense_lightning_indexer_softmax_lse,
    dense_lightning_indexer_softmax_lse_metadata,
)
from .moe_init_routing import moe_init_routing
from .moe_re_routing import moe_re_routing
from . import graph_convert as _graph_convert
