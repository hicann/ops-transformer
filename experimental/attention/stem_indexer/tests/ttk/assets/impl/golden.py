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

"""CPU golden adapter for stem_indexer TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch

PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[3] / "pytest"
    module_path = pytest_dir / "stem_indexer_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"si_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return module


def _to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def cpu_stem_indexer(
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
    topk_score_precision=1,
    q_heads=0,
    kv_heads=0,
    batch_size=0,
    q_seq_lens_list=None,
    kv_seq_lens_list=None,
    num_prompt_tokens_list=None,
    special_setting="",
    **kwargs,
):
    """CPU reference implementation wrapping the existing pytest golden."""
    golden_module = load_pytest_golden_module()

    qflat_cpu = _to_cpu(qflat)
    kflat_cpu = _to_cpu(kflat)
    vbias_cpu = _to_cpu(vbias)
    q_seq_lens_cpu = _to_cpu(q_seq_lens)
    kv_seq_lens_cpu = _to_cpu(kv_seq_lens)
    num_prompt_tokens_cpu = _to_cpu(num_prompt_tokens)

    q_heads = int(q_heads) if q_heads else int(qflat_cpu.shape[1])
    kv_heads = int(kv_heads) if kv_heads else int(kflat_cpu.shape[1])
    batch_size = int(batch_size) if batch_size else int(qflat_cpu.shape[0])

    if q_seq_lens_list is None:
        q_seq_lens_list = q_seq_lens_cpu.tolist()
    if kv_seq_lens_list is None:
        kv_seq_lens_list = kv_seq_lens_cpu.tolist()
    if num_prompt_tokens_list is None:
        num_prompt_tokens_list = num_prompt_tokens_cpu.tolist()

    case = {
        "batch_size": batch_size,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "q_seq_lens": list(q_seq_lens_list),
        "kv_seq_lens": list(kv_seq_lens_list),
        "num_prompt_tokens": list(num_prompt_tokens_list),
        "causal": bool(causal),
        "alpha": float(alpha),
        "stem_block_size": int(stem_block_size),
        "stem_stride": int(stem_stride),
        "initial_blocks": int(initial_blocks),
        "window_size": int(window_size),
        "topk_score_precision": int(topk_score_precision),
        "qflat_dtype": "BF16",
        "kflat_dtype": "BF16",
        "vbias_dtype": "FP32",
        "special_setting": special_setting,
    }

    inputs = {
        "qflat": qflat_cpu,
        "kflat": kflat_cpu,
        "vbias": vbias_cpu,
        "q_seq_lens": torch.tensor(q_seq_lens_list, dtype=torch.int32),
        "kv_seq_lens": torch.tensor(kv_seq_lens_list, dtype=torch.int32),
        "num_prompt_tokens": torch.tensor(num_prompt_tokens_list, dtype=torch.int32),
        "metadata": _to_cpu(metadata) if metadata is not None else None,
    }

    sparse_indices, sparse_seq_len = golden_module.stem_indexer_golden(case, inputs)
    return [sparse_indices, sparse_seq_len]
