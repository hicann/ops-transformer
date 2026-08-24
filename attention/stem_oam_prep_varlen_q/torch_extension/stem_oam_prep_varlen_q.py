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
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class StemOamPrepVarlenQOpBuilder(OpBuilder):
    def __init__(self):
        super(StemOamPrepVarlenQOpBuilder, self).__init__(
            "stem_oam_prep_varlen_q", category="attention"
        )

    def sources(self):
        return ["csrc/attention/stem_oam_prep_varlen_q.cpp"]

    def schema(self) -> str:
        return (
            "stem_oam_prep_varlen_q(Tensor q, int[] q_seq_lens, int[] cu_seq_lens_q, *, "
            "Tensor? q_scale=None, "
            "int stem_block_size=128, int stem_stride=16) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def stem_oam_prep_varlen_q_meta(
            q,
            q_seq_lens,
            cu_seq_lens_q,
            *,
            q_scale=None,
            stem_block_size=128,
            stem_stride=16,
        ):
            qflat_dim = stem_stride * 128
            batch = len(q_seq_lens)
            h_q = q.size(1)
            max_q_len = max(q_seq_lens)
            max_q_padded = (
                (max_q_len + stem_block_size - 1) // stem_block_size
            ) * stem_block_size
            max_q_b = max_q_padded // stem_block_size
            return torch.empty(
                batch, h_q, max_q_b, qflat_dim, dtype=torch.bfloat16, device="meta"
            )


stem_oam_prep_varlen_q_op_builder = StemOamPrepVarlenQOpBuilder()
stem_oam_prep_varlen_q_op_builder._ensure_initialized()


@impl(get_as_library(), stem_oam_prep_varlen_q_op_builder.name, "PrivateUse1")
def stem_oam_prep_varlen_q(
    q, q_seq_lens, cu_seq_lens_q, *, q_scale=None, stem_block_size=128, stem_stride=16
):
    op_module = stem_oam_prep_varlen_q_op_builder.load()
    return op_module.stem_oam_prep_varlen_q(
        q, q_seq_lens, cu_seq_lens_q, q_scale, stem_block_size, stem_stride
    )
