# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import importlib
import logging

logger = logging.getLogger(__name__)


def _discover_ops_from_entry_points():
    ops = {}
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        eps = (
            eps.select(group="cann_ops_transformer.ops")
            if hasattr(eps, "select")
            else eps.get("cann_ops_transformer.ops", [])
        )
        for ep in eps:
            ops[ep.name] = ep.value
    except (ImportError, RuntimeError, AttributeError):
        pass
    return ops


def _discover_ops_from_dir():
    ops = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_name = __name__

    _skip = frozenset(("graph_convert",))
    for entry in sorted(os.listdir(current_dir)):
        if entry in _skip or entry.startswith((".", "_", "__")):
            continue
        entry_path = os.path.join(current_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        for sub in sorted(os.listdir(entry_path)):
            if sub in _skip or sub.startswith((".", "_", "__")):
                continue
            sub_path = os.path.join(entry_path, sub)
            sub_init = os.path.join(sub_path, "__init__.py")
            if os.path.isdir(sub_path) and os.path.isfile(sub_init) and sub not in ops:
                ops[sub] = "%s.%s.%s" % (package_name, entry, sub)

    return ops


def _load_op(name, target):
    if ":" in target:
        module_path, attr_name = target.split(":", 1)
        try:
            _mod = importlib.import_module(module_path)
            if hasattr(_mod, attr_name):
                globals()[name] = getattr(_mod, attr_name)
            elif hasattr(_mod, name):
                globals()[name] = getattr(_mod, name)
            else:
                globals()[name] = _mod
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Failed to load op '%s': %s", name, e)
        return
    try:
        _mod = importlib.import_module(target)
        if hasattr(_mod, name):
            globals()[name] = getattr(_mod, name)
        else:
            globals()[name] = _mod
        for _extra in getattr(_mod, "__all__", []):
            if _extra != name and not _extra.startswith("_"):
                globals()[_extra] = getattr(_mod, _extra)
        try:
            _gmod = importlib.import_module("%s.graph_convert_%s" % (target, name))
            _func = "convert_%s" % name
            if hasattr(_gmod, _func):
                globals()[_func] = getattr(_gmod, _func)
        except ImportError:
            pass
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.warning("Failed to load op '%s': %s", name, e)


_ep_ops = _discover_ops_from_entry_points()
_dir_ops = _discover_ops_from_dir()

_all_ops = dict(_dir_ops)
_all_ops.update(_ep_ops)

for _name, _target in _all_ops.items():
    _load_op(_name, _target)

try:
    del _discover_ops_from_entry_points, _discover_ops_from_dir, _load_op
    del _ep_ops, _dir_ops, _all_ops, _name, _target
except NameError:
    pass


import sys as _sys

_legacy_map = {
    "attention_to_ffn": "mc2.attention_to_ffn_v2",
    "causal_conv1d_fn": "mamba.causal_conv1d",
    "causal_conv1d_update": "mamba.causal_conv1d",
    "comm_context": "mc2.common",
    "compressor": "attention.compressor",
    "deep_ep": "mc2.moe_distribute_dispatch",
    "dense_lightning_indexer_kl_loss_grad": "attention.dense_lightning_indexer_kl_loss_grad",
    "dense_lightning_indexer_softmax_lse": "attention.dense_lightning_indexer_softmax_lse_v2",
    "elastic_buffer": "mc2.common",
    "flash_attn": "attention.flash_attn",
    "ffn_to_attention": "mc2.ffn_to_attention_v2",
    "grouped_matmul_activation_quant": "gmm.grouped_matmul_activation_quant",
    "indexer_quant_cache": "attention.indexer_quant_cache",
    "inplace_partial_rotary_mul": "posembedding.inplace_partial_rotary_mul",
    "inplace_partial_rotary_mul_backward": "posembedding.inplace_partial_rotary_mul_grad",
    "kv_compress_epilog": "attention.kv_compress_epilog",
    "kv_quant_sparse_flash_attention": "attention.kv_quant_sparse_flash_attention_v2",
    "lightning_indexer": "attention.lightning_indexer_v2",
    "lightning_indexer_kl_loss": "attention.lightning_indexer_kl_loss",
    "mega_moe": "mc2.mega_moe",
    "mhc_post": "mhc.mhc_post",
    "mhc_post_backward": "mhc.mhc_post_backward",
    "mhc_pre_sinkhorn": "mhc.mhc_pre_sinkhorn",
    "mhc_pre_sinkhorn_backward": "mhc.mhc_pre_sinkhorn_backward",
    "mixed_quant_sparse_flash_mla": "attention.mixed_quant_sparse_flash_mla",
    "moe_finalize_routing": "moe.moe_finalize_routing_v2",
    "moe_init_routing": "moe.moe_init_routing_v4",
    "moe_re_routing": "moe.moe_re_routing_v2",
    "moe_token_permute": "moe.moe_token_permute",
    "msa_index_score": "attention.msa_index_score",
    "qkv_rms_norm_rope_cache_with_k_scale": "posembedding.qkv_rms_norm_rope_cache_with_k_scale",
    "quant_compressor": "attention.quant_compressor",
    "quant_flash_attn": "attention.quant_flash_attn",
    "quant_lightning_indexer": "attention.quant_lightning_indexer_v2",
    "quant_sparse_flash_mla": "attention.quant_sparse_flash_mla",
    "scatter_pa_kv_cache_with_k_scale": "attention.scatter_pa_kv_cache_with_k_scale",
    "sparse_flash_mla": "attention.sparse_flash_mla",
    "sparse_flash_mla_grad": "attention.sparse_flash_mla_grad",
    "sparse_flash_mla_softmax_l1_norm": "attention.sparse_flash_mla_softmax_l1_norm",
    "sparse_lightning_indexer_kl_loss_grad": "attention.sparse_lightning_indexer_kl_loss_grad",
    "stem_oam_prep_paged_kv": "attention.stem_oam_prep_paged_kv",
    "stem_oam_prep_varlen_q": "attention.stem_oam_prep_varlen_q",
    "und_gen_qkv_rms_norm_rope_cache": "posembedding.und_gen_qkv_rms_norm_rope_cache",
    "fused_causal_conv1d": "attention.fused_causal_conv1d",
    "fused_causal_conv1d_": "attention.inplace_fused_causal_conv1d",
    "block_sparse_attention": "attention.block_sparse_attention",
    "apply_rotary_pos_emb": "posembedding.apply_rotary_pos_emb",
    "apply_rotary_pos_emb_grad": "posembedding.apply_rotary_pos_emb_grad",
}

if __name__ == "cann_ops_transformer.ops":
    for _old_name, _new_target in _legacy_map.items():
        try:
            _new_mod = importlib.import_module(
                "cann_ops_transformer.ops.%s" % _new_target
            )
            _sys.modules["cann_ops_transformer.ops.%s" % _old_name] = _new_mod
        except (ImportError, RuntimeError, AttributeError) as _e:
            logger.warning("Failed to register legacy module '%s': %s", _old_name, _e)


# Centralized operators not yet migrated to distributed structure
