#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Input customization for standalone FlashAttnMetadata cases."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


_VECTOR_NAMES = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "seqused_q",
    "seqused_kv",
)


def load_metadata_protocol():
    name = "fa_ttk_metadata_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("metadata_protocol.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_sidecar_values(kwargs):
    metadata_input = load_metadata_protocol().load_metadata_inputs(
        "flash_attn", kwargs.get("testcase_name")
    )
    return {} if metadata_input is None else metadata_input


def copy_values(target, values, name):
    if target is None:
        return
    if values is None:
        raise ValueError(
            f"FlashAttnMetadata requires {name}_values for an active tensor"
        )
    source = np.asarray(values, dtype=np.int32)
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"FlashAttnMetadata {name} shape mismatch: "
            f"target={tuple(target.shape)}, values={tuple(source.shape)}"
        )
    if hasattr(target, "copy_"):
        import torch

        target.copy_(torch.as_tensor(source, dtype=target.dtype, device=target.device))
    else:
        np.copyto(target, source.astype(target.dtype, copy=False))


def generate_flash_attn_metadata_inputs(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    **kwargs,
):
    """Copy explicit descriptor vectors into the metadata API input tensors."""
    del num_heads_q, num_heads_kv, head_dim
    sidecar = load_sidecar_values(kwargs)
    tensors = (
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
    )
    for name, tensor in zip(_VECTOR_NAMES, tensors):
        values = sidecar[name] if name in sidecar else kwargs.get(f"{name}_values")
        copy_values(tensor, values, name)
