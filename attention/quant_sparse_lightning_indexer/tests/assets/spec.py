# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public QSLI TTK assets sharing the mainline QLI CPU math and tolerances."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


def load_qli_assets():
    name = "qsli_ttk_qli_assets"
    if name not in sys.modules:
        path = (
            Path(__file__).resolve().parents[3]
            / "quant_lightning_indexer_v2/tests/assets/spec.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


QLI = load_qli_assets()
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
    packed = kwargs.get("layout_k", "BSND") == "PA_BBND"
    if packed:
        pages, groups, width = k.shape
        if width != 544:
            raise ValueError("public PA QSLI requires packed K with width 544")
        key = torch.empty((pages, groups * 8, 1, 64), dtype=torch.uint8)
        scale = torch.empty((pages, groups * 8, 1, 2, 2), dtype=torch.uint8)
    else:
        key, scale = k, descale_k
    QLI.inputs_module.generate_qli_v2_inputs(
        q, key, w, descale_q, scale, topk, quant_mode, **kwargs
    )
    if packed:
        QLI.inputs_module.INPUT_ADAPTER.copy_tensor(
            k,
            torch.cat(
                (key.reshape(pages, groups, 512), scale.reshape(pages, groups, 32)), -1
            ),
            "key",
        )
    if kwargs.get("layout_q", "BSND") == "BSND":
        prefix = [batch * q.shape[1] for batch in range(q.shape[0] + 1)]
    else:
        prefix = QLI.inputs_module.INPUT_ADAPTER.list_value(kwargs, "cu_seqlens_q")
        if prefix is None:
            prefix = [0, q.shape[0]]
    used = QLI.inputs_module.INPUT_ADAPTER.list_value(kwargs, "seqused_k")
    if used is None:
        if kwargs.get("layout_k", "BSND") == "BSND":
            used = [key.shape[1]] * key.shape[0]
        else:
            key_prefix = QLI.inputs_module.INPUT_ADAPTER.list_value(
                kwargs, "cu_seqlens_k"
            )
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


def cpu_inputs(q, k, w, descale_q, descale_k, attributes):
    if attributes.get("layout_k", "TND") == "PA_BBND":
        pages, groups, width = k.shape
        if width != 544:
            raise ValueError("public PA QSLI requires packed K with width 544")
        descale_k = k[..., 512:].contiguous().reshape(pages, groups * 8, 1, 2, 2)
        k = k[..., :512].contiguous().reshape(pages, groups * 8, 1, 64)
    inputs = [q, k, w, descale_q, descale_k]
    inputs.extend(attributes.get(name) for name in OPTIONAL_INPUTS)
    if attributes.get("layout_q", "TND") == "TND" and inputs[5] is None:
        inputs[5] = torch.tensor([0, q.shape[0]], dtype=torch.int32)
    if attributes.get("layout_k", "TND") == "TND" and inputs[6] is None:
        inputs[6] = torch.tensor([0, k.shape[0]], dtype=torch.int32)
    return tuple(inputs)


def build_data(
    q,
    k,
    w,
    descale_q,
    descale_k,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    attributes,
):
    cpu_tensors = []
    for value in (
        q,
        k,
        w,
        descale_q,
        descale_k,
        candidate_block_indices,
        candidate_block_length,
    ):
        cpu_tensors.append(QLI.inputs_module.INPUT_ADAPTER.to_cpu_tensor(value))
    q, k, w, descale_q, descale_k, candidate_block_indices, candidate_block_length = (
        cpu_tensors
    )
    attributes = {"layout_q": "BSND", "layout_k": "BSND", **attributes}
    tensors = cpu_inputs(q, k, w, descale_q, descale_k, attributes)
    scalar_attributes = {
        name: value for name, value in attributes.items() if name not in OPTIONAL_INPUTS
    }
    context = SimpleNamespace(
        input_tensors=tensors,
        attributes=dict(scalar_attributes, topk=topk, quant_mode=quant_mode),
    )
    data = QLI.inputs_module.rebuild_qli_v2_compare_data(context)
    scores = data["scores"].clone()
    if attributes.get("layout_q", "TND") == "BSND":
        prefix = [batch * q.shape[1] for batch in range(q.shape[0] + 1)]
    else:
        prefix = tensors[5].tolist()
    shape = (*q.shape[:-2], 1, topk)
    indices = torch.full((prefix[-1], 1, topk), -1, dtype=torch.int32)
    values = torch.full((prefix[-1], 1, topk), -torch.inf, dtype=torch.bfloat16)
    candidate_indices = candidate_block_indices.reshape(
        -1, 1, candidate_block_indices.shape[-1]
    )
    candidate_lengths = candidate_block_length.reshape(-1, 1)
    offsets = tensors[11]
    for batch, (start, end) in enumerate(zip(prefix, prefix[1:])):
        for local, row in enumerate(range(start, end)):
            count = int(candidate_lengths[row, 0])
            blocks = candidate_indices[row, 0, :count].long()
            score = scores[batch, 0, local]
            allowed = torch.isin(torch.arange(score.numel()) // 8, blocks)
            score.masked_fill_(~allowed, -torch.inf)
            selected_count = min(topk, int(torch.isfinite(score).sum()))
            if selected_count:
                selected = score.topk(selected_count).indices
                offset = 0 if offsets is None else int(offsets.reshape(-1)[row])
                indices[row, 0, :selected_count] = selected.int() + offset
                values[row, 0, :selected_count] = score[selected].to(torch.bfloat16)
    data.update(
        topk_value=scores,
        cpu_result=indices.reshape(shape),
        cpu_topk_value=values.reshape(shape),
    )
    return data


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
    arguments = QLI.npu_preprocess_module.build_metadata_arguments(
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
    results = QLI.compare_module.COMPARATOR.compare(*outputs, compare_data=data)
    actual = QLI.compare_module.COMPARATOR.to_torch(outputs[0])
    expected = QLI.compare_module.COMPARATOR.to_torch(outputs[2]).reshape_as(actual)
    for row, reference in zip(
        actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, actual.shape[-1])
    ):
        count = int((reference != -1).sum())
        assert (row[:count] != -1).all() and (row[count:] == -1).all(), (
            "invalid token prefix"
        )
        assert row[:count].unique().numel() == count, "duplicate QSLI indices"
    if not data["params"][-2]:
        values = QLI.compare_module.COMPARATOR.to_torch(outputs[1])
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
