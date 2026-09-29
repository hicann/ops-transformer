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


class IndexerPrologueQwOpBuilder(OpBuilder):
    def __init__(self):
        super(IndexerPrologueQwOpBuilder, self).__init__(
            "ds41.indexer_prologue_qw", category="attention"
        )

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        """PyTorch operator signature."""
        return [
            (
                "ds41.indexer_prologue_qw(Tensor x, Tensor qr, Tensor wqb, Tensor ww, "
                "Tensor descale_qr, Tensor descale_wqb, Tensor rope_sin, Tensor rope_cos, "
                "*, float softmax_scale, Tensor? q=None, Tensor? descale_q=None, "
                "Tensor? w=None) -> (Tensor, Tensor, Tensor)"
            ),
        ]

    def register_meta(self):
        """Register the FakeTensor/Meta output implementation."""

        @impl(get_as_library(), self.name, "Meta")
        def indexer_prologue_qw_meta(
            x,
            qr,
            wqb,
            ww,
            descale_qr,
            descale_wqb,
            rope_sin,
            rope_cos,
            *,
            softmax_scale,
            q=None,
            descale_q=None,
            w=None,
        ):
            # T comes from ``qr``, not ``x``: the kernel lets ``x`` arrive already
            # grown to whole row tiles, so ``x.shape[0]`` can be larger than the
            # live row count and would give the outputs the padded height here.
            t = qr.shape[0]
            n_heads = ww.shape[0]
            d = wqb.shape[0] // n_heads
            groups = (d + 63) // 64
            q_out = torch.empty((t, n_heads, d // 2), dtype=torch.uint8, device="meta")
            descale_out = torch.empty(
                (t, n_heads, groups, 2), dtype=torch.uint8, device="meta"
            )
            w_out = torch.empty((t, n_heads), dtype=torch.float32, device="meta")
            return q_out, descale_out, w_out


indexer_prologue_qw_op_builder = IndexerPrologueQwOpBuilder()
indexer_prologue_qw_op_builder._ensure_initialized()

# Imported at module load, not inside the impl.  ``ops/__init__.py`` registers
# the packaged ``_native`` directory; dynamo traces a function-local import as a
# direct module load and skips that, so a capture-first ACLGraph path would
# miss every precompiled ``.so`` and either JIT (prefer) or raise (require).
from ops.indexer_prologue_qw import indexer_prologue_qw as dsl_indexer_prologue_qw


@impl(get_as_library(), indexer_prologue_qw_op_builder.name, "PrivateUse1")
def indexer_prologue_qw(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    rope_sin,
    rope_cos,
    *,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
):
    """Run the CANNBotDSL kernel through the Torch NPU dispatcher."""
    return dsl_indexer_prologue_qw(
        x,
        qr,
        wqb,
        ww,
        descale_qr,
        descale_wqb,
        rope_sin,
        rope_cos,
        softmax_scale=softmax_scale,
        q=q,
        descale_q=descale_q,
        w=w,
    )
