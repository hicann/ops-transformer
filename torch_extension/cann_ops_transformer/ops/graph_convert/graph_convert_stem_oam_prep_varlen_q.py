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
    from typing import List, Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec
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
        [False, False, False, False],
        [False, False, False, True],
        inputs_tensor_type=[
            None,
            TensorType.TT_INDEX_NUMBER,
            TensorType.TT_INDEX_NUMBER,
            None,
        ],
    )
    def StemOamPrepVarlenQ(
        q: Tensor,
        qSeqLens: Tensor,
        cuSeqLensQ: Tensor,
        qScale: Optional[Tensor] = None,
        *,
        stemBlockSize: int = 128,
        stemStride: int = 16,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(StemOamPrepVarlenQ)\n
        .INPUT(q, TensorType({DT_FLOAT8_E4M3FN}))\n
        .INPUT(qSeqLens, TensorType({DT_INT64}))\n
        .INPUT(cuSeqLensQ, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(qScale, TensorType({DT_FLOAT}))\n
        .OUTPUT(qFlat, TensorType({DT_BF16}))\n
        .ATTR(stemBlockSize, Int, 128)\n
        .ATTR(stemStride, Int, 16)
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "q": q,
            "qSeqLens": qSeqLens,
            "cuSeqLensQ": cuSeqLensQ,
            "qScale": qScale,
        }
        attrs = {
            "stemBlockSize": attr.Int(stemBlockSize),
            "stemStride": attr.Int(stemStride),
        }
        outputs = ["qFlat"]
        return ge_op(
            op_type="StemOamPrepVarlenQ",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("StemOamPrepVarlenQ")
            .input("q", "DT_FLOAT8_E4M3FN")
            .input("qSeqLens", "DT_INT64")
            .input("cuSeqLensQ", "DT_INT64")
            .optional_input("qScale", "DT_FLOAT")
            .attr("stemBlockSize", attr.Int(128))
            .attr("stemStride", attr.Int(16))
            .output("qFlat", "DT_BF16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.stem_oam_prep_varlen_q.default
    )
    def convert_stem_oam_prep_varlen_q(
        q: Tensor,
        q_seq_lens: List[int],
        cu_seq_lens_q: List[int],
        *,
        q_scale: Optional[Tensor] = None,
        stem_block_size: int = 128,
        stem_stride: int = 16,
        meta_outputs: TensorSpec = None,
    ):
        """GE converter for stem_oam_prep_varlen_q.

        Converts the PyTorch custom op to a GE graph engine op for graph mode
        execution. The int[] inputs (q_seq_lens, cu_seq_lens_q) are automatically
        converted to INT64 const tensors via auto_convert_to_tensor.
        """
        return StemOamPrepVarlenQ(
            q=q,
            qSeqLens=q_seq_lens,
            cuSeqLensQ=cu_seq_lens_q,
            qScale=q_scale,
            stemBlockSize=stem_block_size,
            stemStride=stem_stride,
        )
else:

    def convert_stem_oam_prep_varlen_q(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
