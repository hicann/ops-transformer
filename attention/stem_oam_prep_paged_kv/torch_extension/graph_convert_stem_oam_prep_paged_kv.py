# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from typing import List, Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec, Const, DataType
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import TensorType
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False],
        [False, False, False, False, True, True],
        inputs_tensor_type=[
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    )
    def StemOamPrepPagedKv(
        kCache: Tensor,
        vCache: Tensor,
        kvIndices: Tensor,
        kvSeqLens: Tensor,
        kScaleCache: Optional[Tensor] = None,
        vScale: Optional[Tensor] = None,
        *,
        lambdaMag: float = 0.3,
        kvLayout: str = "BNBD",
        stemBlockSize: int = 128,
        stemStride: int = 16,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(StemOamPrepPagedKv)\n
        .INPUT(kCache, TensorType({DT_FLOAT8_E4M3FN}))\n
        .INPUT(vCache, TensorType({DT_FLOAT8_E4M3FN}))\n
        .INPUT(kvIndices, TensorType({DT_INT32}))\n
        .INPUT(kvSeqLens, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(kScaleCache, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(vScale, TensorType({DT_FLOAT}))\n
        .OUTPUT(kFlat, TensorType({DT_BF16}))\n
        .OUTPUT(vBias, TensorType({DT_FLOAT}))\n
        .ATTR(lambdaMag, Float, 0.3)\n
        .ATTR(kvLayout, Str, "BNBD")\n
        .ATTR(stemBlockSize, Int, 128)\n
        .ATTR(stemStride, Int, 16)
        """
        if dependencies is None:
            dependencies = []

        if not isinstance(kvSeqLens, Tensor):
            kvSeqLens = Const(kvSeqLens, dtype=DataType.DT_INT32)

        inputs = {
            "k_cache": kCache,
            "v_cache": vCache,
            "kv_indices": kvIndices,
            "kv_seq_lens": kvSeqLens,
            "k_scale_cache": kScaleCache,
            "v_scale": vScale,
        }
        attrs = {
            "lambda_mag": attr.Float(lambdaMag),
            "kv_layout": attr.Str(kvLayout),
            "stem_block_size": attr.Int(stemBlockSize),
            "stem_stride": attr.Int(stemStride),
        }
        outputs = ["k_flat", "v_bias"]
        return ge_op(
            op_type="StemOamPrepPagedKv",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("StemOamPrepPagedKv")
            .input("k_cache", "DT_FLOAT8_E4M3FN")
            .input("v_cache", "DT_FLOAT8_E4M3FN")
            .input("kv_indices", "DT_INT32")
            .input("kv_seq_lens", "DT_INT32")
            .optional_input("k_scale_cache", "DT_FLOAT")
            .optional_input("v_scale", "DT_FLOAT")
            .attr("lambda_mag", attr.Float(0.3))
            .attr("kv_layout", attr.Str("BNBD"))
            .attr("stem_block_size", attr.Int(128))
            .attr("stem_stride", attr.Int(16))
            .output("k_flat", "DT_BF16")
            .output("v_bias", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.stem_oam_prep_paged_kv.default
    )
    def convert_stem_oam_prep_paged_kv(
        k_cache: Tensor,
        v_cache: Tensor,
        kv_indices: Tensor,
        kv_seq_lens: List[int],
        *,
        k_scale_cache: Optional[Tensor] = None,
        v_scale: Optional[Tensor] = None,
        lambda_mag: float = 0.3,
        kv_layout: str = "BNBD",
        stem_block_size: int = 128,
        stem_stride: int = 16,
        meta_outputs: TensorSpec = None,
    ):
        """GE converter for stem_oam_prep_paged_kv.

        Converts the PyTorch custom op to a GE graph engine op for graph mode
        execution. The int[] input (kv_seq_lens) is automatically
        converted to a const tensor via auto_convert_to_tensor.
        """
        return StemOamPrepPagedKv(
            kCache=k_cache,
            vCache=v_cache,
            kvIndices=kv_indices,
            kvSeqLens=kv_seq_lens,
            kScaleCache=k_scale_cache,
            vScale=v_scale,
            lambdaMag=lambda_mag,
            kvLayout=kv_layout,
            stemBlockSize=stem_block_size,
            stemStride=stem_stride,
        )
else:

    def convert_stem_oam_prep_paged_kv(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
