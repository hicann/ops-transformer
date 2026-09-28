# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch interface migrated from op-plugin/GmmLocalExpNpuOpApi.cpp."""

import os
import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class GroupMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("local_exp_gmm", category="gmm")

    def sources(self):
        return ["csrc/gmm/gmm_local_exp.cpp"]

    def include_paths(self):
        return [os.path.join(self.cann_path, "include"), *super().include_paths()]

    def schema(self):
        return "local_exp_gmm(Tensor a, Tensor b, Tensor problemList, int expStartIdx, int expEndIdx, *, bool trans_a=False, bool trans_b=False, bool is_b_nz=False, bool type_promotion=False) -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def meta(
            a,
            b,
            problemList,
            expStartIdx,
            expEndIdx,
            *,
            trans_a=False,
            trans_b=False,
            is_b_nz=False,
            type_promotion=False,
        ):
            return a.new_empty(
                (a.shape[0], b.shape[1 if trans_b else 2]),
                dtype=torch.float32 if type_promotion else a.dtype,
            )


builder = GroupMatmulOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def local_exp_gmm(
    a,
    b,
    problemList,
    expStartIdx,
    expEndIdx,
    *,
    trans_a=False,
    trans_b=False,
    is_b_nz=False,
    type_promotion=False,
):
    return builder.load().local_exp_gmm(
        a,
        b,
        problemList,
        expStartIdx,
        expEndIdx,
        trans_a,
        trans_b,
        is_b_nz,
        type_promotion,
    )


# The wheel collector exports the directory name as its entry point.
gmm_local_exp = local_exp_gmm
