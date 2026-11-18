# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch dispatcher schemas for the inference-only GDN MVP.

Registration does not import the DSL compiler. Entry-module imports are
delayed until dispatch; the compute kernel is loaded only for NPU execution.
No CPU Golden is registered as an operator kernel.
"""

from importlib import import_module

import torch
from cann_ops_transformer.op_builder import get_as_library

_library = get_as_library()
_FORWARD_ARGS = (
    "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor metadata, "
    "Tensor? initial_state=None, Tensor? A_log=None, Tensor? dt_bias=None, "
    "Tensor? cu_seqlens=None, Tensor? seqused=None, *, "
    "float? scale=None, bool output_final_state=False, "
    "bool return_intermediate_states=False, bool use_qk_l2norm_in_kernel=False, "
    "bool use_gate_in_kernel=False, bool use_beta_sigmoid_in_kernel=False, "
    'bool allow_neg_eigval=False, bool disable_recompute=True, str layout="TND"'
)
_library.define(f"flash_gdn({_FORWARD_ARGS}) -> (Tensor, Tensor?, Tensor?)")
_library.define(
    f"flash_gdn_fwd({_FORWARD_ARGS}) -> ("
    + ", ".join(["Tensor"] + ["Tensor?"] * 15)
    + ")"
)
_library.define(
    "flash_gdn_metadata(Tensor? cu_seqlens=None, Tensor? seqused=None, *, "
    "int batch_size, int nheads_qk, int nheads_v, int seqlens, "
    'int headdim_qk=128, int headdim_v=128, bool enable_grad=False, str layout="TND") -> Tensor'
)


def _dsl_module(name):
    # Source checkout: op_dsl is a sibling of torch_extension. The wheel
    # collector flattens the adapter into the operator package instead.
    prefix = "..op_dsl" if __package__.endswith(".torch_extension") else ".op_dsl"
    return import_module(f"{prefix}.{name}", package=__package__)


def _route(*args, **kwargs):
    flash_gdn = _dsl_module("flash_gdn").flash_gdn

    return flash_gdn(*args, **kwargs)


def _forward(*args, **kwargs):
    flash_gdn_fwd = _dsl_module("flash_gdn").flash_gdn_fwd

    return flash_gdn_fwd(*args, **kwargs)


def _metadata(*args, **kwargs):
    flash_gdn_metadata = _dsl_module("flash_gdn").flash_gdn_metadata

    return flash_gdn_metadata(*args, **kwargs)


def _metadata_meta(*args, **kwargs):
    flash_gdn_metadata = _dsl_module("flash_gdn_metadata").flash_gdn_metadata

    return flash_gdn_metadata(*args, _meta_only=True, **kwargs)


def _guard(name):
    def implementation(*args, **kwargs):
        values = list(args) + list(kwargs.values())
        if torch.is_grad_enabled() and any(
            isinstance(x, torch.Tensor) and x.requires_grad for x in values
        ):
            raise NotImplementedError(
                "FlashGDN MVP is inference-only; flash_gdn_bwd is not implemented"
            )
        with torch._C._AutoDispatchBelowAutograd():
            return getattr(torch.ops.cann_ops_transformer, name).default(
                *args, **kwargs
            )

    return implementation


for _key in ("PrivateUse1", "Meta", "CPU"):
    _library.impl("flash_gdn", _route, _key)
    _library.impl("flash_gdn_fwd", _forward, _key)
_library.impl("flash_gdn", _guard("flash_gdn"), "Autograd")
_library.impl("flash_gdn_fwd", _guard("flash_gdn_fwd"), "Autograd")
_library.impl("flash_gdn_metadata", _metadata, "CompositeExplicitAutograd")
_library.impl("flash_gdn_metadata", _metadata_meta, "Meta")

flash_gdn = torch.ops.cann_ops_transformer.flash_gdn.default
flash_gdn_fwd = torch.ops.cann_ops_transformer.flash_gdn_fwd.default
flash_gdn_metadata = torch.ops.cann_ops_transformer.flash_gdn_metadata.default
