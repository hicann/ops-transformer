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

"""Input customization for stem_indexer TTK cases.

E2E customize_inputs contract: modify tensors in-place via x.copy_(value).
"""

import torch


def _make_float_tensor(shape, dtype, pattern, seed=17):
    if pattern == "zeros":
        return torch.zeros(shape, dtype=dtype)
    if pattern == "constant_positive":
        return torch.full(shape, 0.125, dtype=dtype)
    if pattern == "strictly_ascending_by_k_block":
        base = torch.arange(shape[-1], dtype=torch.float32).reshape(
            *([1] * (len(shape) - 1)), shape[-1]
        )
        return base.expand(shape).contiguous().to(dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    return data.to(dtype)


def customize_inputs(
    qflat,
    kflat,
    vbias,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens,
    metadata,
    *,
    causal=True,
    stem_block_size=128,
    stem_stride=16,
    alpha=1.0,
    initial_blocks=4,
    window_size=4,
    q_heads=0,
    kv_heads=0,
    batch_size=0,
    q_seq_lens_list=None,
    kv_seq_lens_list=None,
    num_prompt_tokens_list=None,
    special_setting="",
    **kwargs,
):
    """Fill input tensors in-place with proper test data for stem_indexer."""
    q_heads = int(q_heads) if q_heads else int(qflat.shape[1])
    kv_heads = int(kv_heads) if kv_heads else int(kflat.shape[1])
    batch_size = int(batch_size) if batch_size else int(qflat.shape[0])

    settings = {}
    if special_setting:
        for item in str(special_setting).split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                settings[k.strip()] = v.strip()

    qflat_pattern = settings.get("qflat", "random_uniform_seeded")
    kflat_pattern = settings.get("kflat", "random_uniform_seeded")
    vbias_pattern = settings.get("vbias", "random_uniform_seeded")

    qflat_data = _make_float_tensor(qflat.shape, qflat.dtype, qflat_pattern, seed=17)
    qflat.copy_(qflat_data)

    kflat_data = _make_float_tensor(kflat.shape, kflat.dtype, kflat_pattern, seed=23)
    kflat.copy_(kflat_data)

    vbias_data = _make_float_tensor(vbias.shape, vbias.dtype, vbias_pattern, seed=31)
    vbias.copy_(vbias_data)

    if q_seq_lens_list is not None:
        q_seq_lens.copy_(
            torch.tensor(q_seq_lens_list, dtype=torch.int32).reshape(q_seq_lens.shape)
        )
    if kv_seq_lens_list is not None:
        kv_seq_lens.copy_(
            torch.tensor(kv_seq_lens_list, dtype=torch.int32).reshape(kv_seq_lens.shape)
        )
    if num_prompt_tokens_list is not None:
        num_prompt_tokens.copy_(
            torch.tensor(num_prompt_tokens_list, dtype=torch.int32).reshape(
                num_prompt_tokens.shape
            )
        )

    metadata.copy_(torch.zeros(metadata.shape, dtype=torch.int32))
