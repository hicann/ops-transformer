# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch interface migrated from op-plugin/GroupMatmulNpuOpApi.cpp."""

import os
import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class GroupMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("gmm", category="gmm")

    def sources(self):
        return ["csrc/gmm/gmm_k_dim.cpp"]

    def include_paths(self):
        return [os.path.join(self.cann_path, "include"), *super().include_paths()]

    def schema(self):
        return "gmm(Tensor a, Tensor b, Tensor batch_sizes, bool trans_a=False, bool trans_b=False, Tensor(a!)? c=None, int aicore_num=-1, bool type_promotion=False) -> Tensor(a!)"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(
            a,
            b,
            batch_sizes,
            trans_a=False,
            trans_b=False,
            c=None,
            aicore_num=-1,
            type_promotion=False,
        ):
            if c is not None:
                return c
            return a.new_empty(
                (batch_sizes.numel(), a.shape[1 if trans_a else 0], b.shape[1]),
                dtype=torch.float32 if type_promotion else a.dtype,
            )


builder = GroupMatmulOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def gmm(
    a,
    b,
    batch_sizes,
    trans_a=False,
    trans_b=False,
    c=None,
    aicore_num=-1,
    type_promotion=False,
):
    return builder.load().gmm(
        a, b, batch_sizes, trans_a, trans_b, c, aicore_num, type_promotion
    )


# The wheel collector exports the directory name as its entry point.
gmm_k_dim = gmm
