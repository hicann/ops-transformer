# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
        ],
    )
    def QuantFlashMlaWithKvcache(
        q: Tensor,
        k_cache: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        block_table: Tensor,
        cache_seqlens: Tensor,
        quant_mode: int,
        cu_seqlens_q: Tensor = None,
        seqused_q: Tensor = None,
        attn_mask: Tensor = None,
        metadata: Tensor = None,
        head_dim_v: int = 512,
        softmax_scale: float = 1.0,
        mask_mode: int = 0,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_kv: str = "PA_BNBD",
        layout_out: str = "BSND",
        return_softmax_lse: bool = False,
    ):
        result = q.new_empty(q.size())
        return result

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache.default
    )
    def convert_quant_flash_mla_with_kvcache(
        q: Tensor,
        k_cache: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        block_table: Tensor,
        cache_seqlens: Tensor,
        quant_mode: int,
        cu_seqlens_q: Tensor = None,
        seqused_q: Tensor = None,
        attn_mask: Tensor = None,
        metadata: Tensor = None,
        head_dim_v: int = 512,
        softmax_scale: float = 1.0,
        mask_mode: int = 0,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_kv: str = "PA_BNBD",
        layout_out: str = "BSND",
        return_softmax_lse: bool = False,
    ):
        raise AssertionError("GE not supported!")
