# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""QSLI input generation using the local pytest storage conventions."""

import math
from numbers import Real

import numpy as np
import torch


def random_tensor(template, data_range, quant_mode, kind, golden):
    lower, upper = map(float, data_range)
    shape = tuple(template.shape)
    if quant_mode == 5 and kind in ("query", "key"):
        codes = torch.nonzero(
            (golden.FP4_VALUES >= lower) & (golden.FP4_VALUES <= upper)
        ).flatten()
        if not codes.numel():
            raise ValueError("MXFP4 input range contains no representable value")
        selected = codes[torch.randint(codes.numel(), (*shape[:-1], shape[-1] * 2))].to(
            torch.uint8
        )
        return selected[..., ::2] | (selected[..., 1::2] << 4)
    if quant_mode == 5 and kind.endswith("scale"):
        if not (0 < lower <= upper and math.isfinite(upper)):
            raise ValueError("E8M0 range must be finite and positive")
        low = max(0, math.ceil(math.log2(lower)) + 127)
        high = min(254, math.floor(math.log2(upper)) + 127)
        if low > high:
            raise ValueError("E8M0 input range contains no representable value")
        return torch.randint(low, high + 1, shape, dtype=torch.uint8)
    values = np.random.uniform(lower, upper, shape)
    return torch.from_numpy(values).to(template.dtype)


def validate_packed_k_scale_range(semantic_ranges):
    scale_range = (
        semantic_ranges[4]
        if isinstance(semantic_ranges, (list, tuple)) and len(semantic_ranges) > 4
        else None
    )
    if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
        lower, upper = scale_range
        if all(isinstance(bound, Real) for bound in scale_range):
            if 0 < lower <= upper and math.isfinite(upper):
                return
    raise ValueError(
        "QSLI packed PA requires qli_input_ranges[4] to be a finite K scale "
        "range with 0 < min <= max"
    )


def generate(
    query,
    key,
    weights,
    query_scale,
    key_scale,
    topk,
    quant_mode,
    attributes,
    golden,
    relations,
):
    if str(attributes.get("expect_error", False)).lower() in ("true", "1", "yes", "on"):
        return
    if quant_mode not in (2, 5):
        raise ValueError("QSLI inputs support quant_mode 2 and 5")
    layout_q = attributes.get("layout_q", "BSND")
    layout_k = attributes.get("layout_k", "BSND")
    packed = layout_k == "PA_BBND" and quant_mode == 5
    q_template = golden.to_cpu_tensor(query)
    if packed:
        pages, groups, width = key.shape
        if width != 544:
            raise ValueError("QSLI packed PA key requires width 544")
        validate_packed_k_scale_range(attributes.get("qli_input_ranges"))
        k_template = torch.empty((pages, groups * 8, 1, 64), dtype=torch.uint8)
        ks_template = torch.empty((pages, groups * 8, 1, 2, 2), dtype=torch.uint8)
    else:
        k_template, ks_template = (
            golden.to_cpu_tensor(key),
            golden.to_cpu_tensor(key_scale),
        )
    defaults = (
        ((-6, 6), (-6, 6), (-1, 1), (1, 1), (1, 1))
        if quant_mode == 5
        else ((-20, 20), (-20, 20), (-1, 1), (1, 1), (1, 1))
    )
    supplied = (
        attributes.get("qli_input_ranges") or attributes.get("input_ranges") or ()
    )
    ranges = [
        supplied[i] if i < len(supplied) and supplied[i] is not None else default
        for i, default in enumerate(defaults)
    ]
    names = ("query", "key", "weights", "query_dequant_scale", "key_dequant_scale")
    templates = (
        q_template,
        k_template,
        golden.to_cpu_tensor(weights),
        golden.to_cpu_tensor(query_scale),
        ks_template,
    )
    with relations.CaseRandomContext(attributes):
        data = {
            name: random_tensor(template, span, quant_mode, name, golden)
            for name, template, span in zip(names, templates, ranges)
        }
        for name in (
            "cu_seqlens_q",
            "cu_seqlens_k",
            "seqused_q",
            "seqused_k",
            "cmp_residual_k",
            "output_idx_offset",
        ):
            target = attributes.get(name)
            if target is None:
                data[name] = None
                continue
            value = attributes.get(f"{name}_values")
            if value is None:
                value = golden.to_cpu_tensor(target).clone()
            data[name] = torch.as_tensor(value, dtype=torch.int32).reshape(target.shape)
        offset_range = attributes.get("output_idx_offset_range")
        if (
            data["output_idx_offset"] is not None
            and offset_range is not None
            and attributes.get("output_idx_offset_values") is None
        ):
            low, high = map(int, offset_range)
            data["output_idx_offset"] = torch.randint(
                low, high + 1, data["output_idx_offset"].shape, dtype=torch.int32
            )
        table = attributes.get("block_table")
        if table is not None:
            table_shape = tuple(table.shape)
            used = golden.list_value(data, "seqused_k")
            needed = [math.ceil(length / k_template.shape[1]) for length in used]
            if (
                sum(needed) > k_template.shape[0]
                or max(needed, default=0) > table_shape[1]
            ):
                raise ValueError(
                    "QSLI PA storage cannot hold the requested sequence lengths"
                )
            order = torch.randperm(k_template.shape[0])
            data["block_table"] = torch.full(table_shape, -1, dtype=torch.int32)
            position = 0
            for batch, count in enumerate(needed):
                data["block_table"][batch, :count] = order[
                    position : position + count
                ].int()
                position += count
        else:
            data["block_table"] = None
        actual = dict(
            attributes, **{name: data[name] for name in data if name not in names}
        )
        actual.update(layout_q=layout_q, layout_k=layout_k, qli_input_ranges=ranges)
        data["params"] = golden.compare_params(
            data["query"],
            data["key"],
            data["weights"],
            data["query_dequant_scale"],
            topk,
            quant_mode,
            actual,
        )
        relations.normalize_indexer_inputs(
            data, dict(actual, quant_mode=quant_mode), "QSLI", quantized=True
        )
    destinations = (query, key, weights, query_scale, key_scale)
    for name, dst in zip(names, destinations):
        if packed and name in ("key", "key_dequant_scale"):
            continue
        golden.copy_tensor(dst, data[name], name)
    if packed:
        raw = torch.cat(
            (
                data["key"].reshape(pages, groups, 512),
                data["key_dequant_scale"].reshape(pages, groups, 32),
            ),
            dim=-1,
        )
        golden.copy_tensor(key, raw, "key")
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
        "block_table",
        "output_idx_offset",
    ):
        golden.copy_tensor(attributes.get(name), data[name], name)
    metadata = attributes.get("metadata")
    if metadata is not None:
        if torch.is_tensor(metadata):
            metadata.zero_()
        else:
            metadata[...] = 0
