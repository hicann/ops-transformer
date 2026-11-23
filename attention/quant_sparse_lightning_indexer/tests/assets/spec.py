# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QSLI TTK assets using the operator-local pytest reference and tolerances."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


def load_test_module(stem, relative):
    name = f"qsli_ttk_{stem}"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / relative
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load QSLI test module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


GOLDEN = load_test_module(
    "golden", "../pytest/quant_sparse_lightning_indexer_golden.py"
)
INPUTS = load_test_module("inputs", "impl/inputs.py")
RELATIONS = load_test_module("relations", "impl/batch_consistency.py")
METADATA = load_test_module("npu_preprocess", "impl/npu_preprocess.py")
COMPARATOR = load_test_module("compare", "impl/compare.py").COMPARATOR
build_data = GOLDEN.build_data
CASE_DATA = {}
OPTIONAL_INPUTS = (
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "block_table",
    "output_idx_offset",
    "metadata",
)


def customize_inputs(*args, **kwargs):
    seed = int(np.random.randint(0, 2**32))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generate_inputs(*args, **kwargs)
    return None


def generate_inputs(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_block_indices,
    candidate_block_length,
    **kwargs,
):
    q, k, w = query, key, weights
    descale_q, descale_k = query_dequant_scale, key_dequant_scale
    kwargs.setdefault("layout_q", "BSND")
    kwargs.setdefault("layout_k", "BSND")
    packed = kwargs.get("layout_k", "BSND") == "PA_BBND" and quant_mode == 5
    INPUTS.generate(
        q, k, w, descale_q, descale_k, topk, quant_mode, kwargs, GOLDEN, RELATIONS
    )
    if packed:
        key, scale = GOLDEN.unpack_paged(GOLDEN.to_cpu_tensor(k), None, quant_mode)
    else:
        key, scale = k, descale_k
    if kwargs.get("layout_q", "BSND") == "BSND":
        prefix = [batch * q.shape[1] for batch in range(q.shape[0] + 1)]
    else:
        prefix = GOLDEN.list_value(kwargs, "cu_seqlens_q")
        if prefix is None:
            prefix = [0, q.shape[0]]
    used = GOLDEN.list_value(kwargs, "seqused_k")
    if used is None:
        if kwargs.get("layout_k", "BSND") == "BSND":
            used = [key.shape[1]] * key.shape[0]
        else:
            key_prefix = GOLDEN.list_value(kwargs, "cu_seqlens_k")
            if key_prefix is None:
                key_prefix = [0, key.shape[0]]
            used = [end - start for start, end in zip(key_prefix, key_prefix[1:])]
    candidate_block_indices = torch.as_tensor(candidate_block_indices)
    candidate_block_length = torch.as_tensor(candidate_block_length)
    candidate_block_indices.fill_(-1)
    indices = candidate_block_indices.view(-1, 1, candidate_block_indices.shape[-1])
    lengths = candidate_block_length.view(-1, 1)
    for batch, (start, end) in enumerate(zip(prefix, prefix[1:])):
        block_count = (int(used[batch]) + 7) // 8
        for row in range(start, end):
            count = min(2048, block_count)
            selected = torch.randperm(block_count)[:count].int()
            indices[row, 0, :count].copy_(selected)
            lengths[row, 0] = count
    return None


def golden(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_block_indices,
    candidate_block_length,
    testcase_name=None,
    **kwargs,
):
    q, k, w = query, key, weights
    descale_q, descale_k = query_dequant_scale, key_dequant_scale
    data = build_data(
        q,
        k,
        w,
        descale_q,
        descale_k,
        candidate_block_indices,
        candidate_block_length,
        topk,
        quant_mode,
        kwargs,
    )
    CASE_DATA[testcase_name] = data
    values = (
        data["cpu_topk_value"]
        if kwargs.get("return_value", 0)
        else torch.empty(0, dtype=torch.bfloat16)
    )
    return data["cpu_result"], values


def npu_preprocess(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    candidate_block_indices,
    candidate_block_length,
    metadata=None,
    candidate_block_size=8,
    **kwargs,
):
    q, k, w = query, key, weights
    descale_q, descale_k = query_dequant_scale, key_dequant_scale
    del w, candidate_block_indices
    # Keep native FP4 inputs at stable addresses for ACLGraph replay.
    for tensor in (query, key):
        if (
            torch.is_tensor(tensor)
            and tensor.device.type == "npu"
            and tensor.dtype == getattr(torch, "float4_e2m1fn_x2", None)
        ):
            torch._dynamo.mark_static_address(tensor)
    # Preserve strides and keep empty PA scales out of graph input copies.
    for tensor in (key, key_dequant_scale):
        if torch.is_tensor(tensor) and tensor.device.type == "npu":
            if not tensor.is_contiguous() or tensor.numel() == 0:
                torch._dynamo.mark_static_address(tensor)
    if metadata is not None and metadata.numel() and bool(metadata.any()):
        return None
    arguments = METADATA.build_metadata_arguments(
        q,
        k,
        topk,
        quant_mode,
        kwargs.get("layout_q", "BSND"),
        kwargs.get("layout_k", "BSND"),
        kwargs.get("mask_mode", 0),
        kwargs.get("cmp_ratio", 1),
        kwargs,
    )
    if kwargs.get("layout_k", "BSND") == "PA_BBND":
        arguments["num_heads_k"] = 1
    else:
        arguments["batch_size"] = None
    if kwargs.get("layout_q", "BSND") == "BSND":
        arguments["max_seqlen_q"] = q.shape[1]
    if kwargs.get("layout_k", "BSND") == "BSND":
        arguments["max_seqlen_k"] = k.shape[1]
    arguments.update(
        candidate_block_length=candidate_block_length,
        candidate_block_size=candidate_block_size,
    )
    generated = torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer_metadata(
        **arguments
    )
    if metadata is None:
        return {"metadata": generated}
    metadata.copy_(generated)
    return None


def compare(*outputs, compare_context=None, **_kwargs):
    if compare_context is None or len(outputs) != 4:
        raise ValueError("QSLI compare requires context and two NPU/two CPU outputs")
    data = CASE_DATA.pop(compare_context.testcase_name, None)
    if data is None:
        tensors = tuple(compare_context.input_tensors)
        attributes = dict(compare_context.attributes)
        attributes.update(zip(OPTIONAL_INPUTS, tensors[7:]))
        data = build_data(
            *tensors[:7],
            attributes.pop("topk"),
            attributes.pop("quant_mode"),
            attributes,
        )
    results = COMPARATOR.compare(*outputs, compare_data=data)
    actual = COMPARATOR.to_torch(outputs[0])
    expected = COMPARATOR.to_torch(outputs[2]).reshape_as(actual)
    for row, reference in zip(
        actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, actual.shape[-1])
    ):
        count = int((reference != -1).sum())
        assert (row[:count] != -1).all() and (row[count:] == -1).all(), (
            "invalid token prefix"
        )
        assert row[:count].unique().numel() == count, "duplicate QSLI indices"
    if not data["params"][-2]:
        values = COMPARATOR.to_torch(outputs[1])
        results.append(
            {
                "pass": values.numel() == 0,
                "precision": 100.0,
                "error_info": "return_value=0 requires empty values"
                if values.numel()
                else None,
            }
        )
    return results


class QuantSparseLightningIndexerSpec:
    customize_inputs = customize_inputs
    golden = golden
    compare = compare
    npu_preprocess = npu_preprocess


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_sparse_lightning_indexer": "QuantSparseLightningIndexerSpec",
}
