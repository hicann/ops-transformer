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
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False] * 13, [False] * 13)
    def QuantCompressor(
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        ape: Tensor,
        x_descale: Optional[Tensor],
        wkv_descale: Optional[Tensor],
        wgate_descale: Optional[Tensor],
        state_block_table: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        seqused: Optional[Tensor],
        start_pos: Optional[Tensor],
        cmp_kv: Tensor,
        quant_mode: int = 1,
        cmp_ratio: int = 4,
        coff: int = 1,
        cache_mode: int = 1,
        state_cache_stride_dim0: int = 0,
    ):
        result = x.new_empty(x.size())
        return result

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.quant_compressor.default
    )
    def convert_quant_compressor(
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        ape: Tensor,
        quant_mode: int,
        cmp_ratio: int,
        *,
        x_descale: Tensor = None,
        wkv_descale: Tensor = None,
        wgate_descale: Tensor = None,
        state_block_table: Tensor = None,
        cu_seqlens: Tensor = None,
        seqused: Tensor = None,
        start_pos: Tensor = None,
        coff: int = 1,
        cache_mode: int = 1,
        state_cache_stride_dim0: int = 0,
    ):
        raise AssertionError(
            "quant_compressor is not supported in GE graph mode. Please use aclgraph mode instead."
        )
