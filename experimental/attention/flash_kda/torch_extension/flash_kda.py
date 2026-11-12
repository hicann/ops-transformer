# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch dispatcher bindings for the Flash KDA DSL entry points."""

import torch

from cann_ops_transformer.op_builder import get_as_library


_library = get_as_library()
_forward_args = (
    "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor? metadata, "
    "Tensor? initial_state=None, Tensor? A_log=None, Tensor? dt_bias=None, "
    "Tensor? cu_seqlens=None, Tensor? seqused=None, *, "
    "float? scale=0.08838834764831845, bool output_final_state=False, "
    "bool return_intermediate_states=False, bool use_qk_l2norm_in_kernel=True, "
    "bool use_gate_in_kernel=True, bool use_beta_sigmoid_in_kernel=True, "
    'bool allow_neg_eigval=False, bool disable_recompute=False, float? lower_bound=None, str layout="TND"'
)
_library.define(f"flash_kda({_forward_args}) -> (Tensor, Tensor?, Tensor?)")
_forward_outputs = ", ".join(["Tensor"] + ["Tensor?"] * 15)
_library.define(f"flash_kda_fwd({_forward_args}) -> ({_forward_outputs})")
_library.define(
    "flash_kda_metadata(Tensor? cu_seqlens=None, Tensor? seqused=None, *, "
    "int batch_size, int nheads_qk, int nheads_v, int seqlens, int headdim_qk=128, "
    'int headdim_v=128, bool enable_grad=False, str layout="TND") -> Tensor'
)


def _flash_kda(*args, **kwargs):
    from .op_dsl.flash_kda import flash_kda

    return flash_kda(*args, **kwargs)


def _flash_kda_fwd(*args, **kwargs):
    from .op_dsl.flash_kda import flash_kda_fwd

    return flash_kda_fwd(*args, **kwargs)


def _metadata(*args, **kwargs):
    from .op_dsl.flash_kda_metadata import flash_kda_metadata

    return flash_kda_metadata(*args, **kwargs)


# The DSL entry points implement both eager execution and Meta output inference.
for _key in ("PrivateUse1", "Meta"):
    _library.impl("flash_kda", _flash_kda, _key)
    _library.impl("flash_kda_fwd", _flash_kda_fwd, _key)
    _library.impl("flash_kda_metadata", _metadata, _key)
# Dense layouts also support scalar-only metadata generation, with no tensor dispatch key.
_library.impl("flash_kda_metadata", _metadata, "CompositeExplicitAutograd")

flash_kda = torch.ops.cann_ops_transformer.flash_kda.default
flash_kda_fwd = torch.ops.cann_ops_transformer.flash_kda_fwd.default
flash_kda_metadata = torch.ops.cann_ops_transformer.flash_kda_metadata.default
