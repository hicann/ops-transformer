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

# GE Converter for Graph Mode (torchair适配)
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

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.scatter_pa_kv_cache_with_k_scale.default
    )
    def converter_scatter_pa_kv_cache_with_k_scale(
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        key_scale: Tensor,
        key_scale_cache: Tensor,
        *,
        cache_layout: str = "BNBD",
        meta_outputs: TensorSpec = None,
    ):
        """
        torchair GE转换器：scatter_pa_kv_cache_with_k_scale

        将PyTorch算子转换为GE图引擎算子，用于图模式执行

        Args:
            key: GE Tensor，FP8格式的key输入
            value: GE Tensor，FP8格式的value输入
            key_cache: GE Tensor，FP8格式的key cache
            value_cache: GE Tensor，FP8格式的value cache
            slot_mapping: GE Tensor，slot偏移索引
            key_scale: GE Tensor，float32格式的key scale
            key_scale_cache: GE Tensor，float32格式的key scale cache
            cache_layout: str，cache布局格式
            meta_outputs: TensorSpec，输出的shape/dtype信息

        """
        return ge.ScatterPaKvCacheWithKScale(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            key_scale,
            key_scale_cache,
            cache_layout=cache_layout,
        )
