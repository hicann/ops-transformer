# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""将独立数值参考的判定结果适配为 TTK 比较记录和展示信息。"""

import importlib.util

import logging

import os

import sys

import numpy as np

import torch

_INPUT_NAMES = [
    "x",
    "wqa",
    "wqb",
    "wkva",
    "wkb",
    "norm_weight_qa",
    "norm_weight_kva",
    "kv_cache",
    "cache_index",
    "quant_scale_kv",
    "rope_sin",
    "rope_cos",
]

_REFERENCE_ATTRS = (
    "norm_eps",
    "qkv_quant_mode",
    "cache_mode",
    "qr_scale",
    "kvr_scale",
    "rope_mode",
    "return_qr",
)

TOLERANCE = {
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
}


def _load_sibling(stem):
    name = f"mla_prolog_v4_ttk_assets_impl_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{stem}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _as_torch(value):
    import mla_prolog_v4_ttk_ops as adapter

    tensor = adapter._transport().to_torch(value)
    return tensor.detach().cpu() if tensor is not None else None


def _cache_snapshot(value):
    """创建独立、连续的 CPU Cache 存储快照。

    设备布局由预处理保留；仅整理 CPU 快照，并保持各页 PA_BSND/PA_NZ 的载荷字节。
    """
    tensor = _as_torch(value)
    return tensor.contiguous().clone() if tensor is not None else None


def _mode_attr(name, value):
    import mla_prolog_v4_ttk_ops as adapter

    return adapter._mode_int(name, value)


_THREE_TO_TWO = {
    "cache": "cache",
    "cache_kv": "kv",
    "cache_kr": "kr",
}

_DISPLAY_SLOTS = ("q", "descale_q", "qr", "cache")


def _display_records(records):
    """将详细分段记录汇总到四个公开输出槽位。

    Comparison 列的百分比依次对应 q、descale_q、qr 和 kv_cache。
    KV、KR、cache_descale 及分支诊断继续参与内部判定，不增加展示槽位。
    """
    by_name = {}
    for record in records:
        by_name.setdefault(record["name"], record)

    groups = {
        "q": ("q",),
        "descale_q": ("descale_q",),
        "qr": ("qr",),
        "cache": ("cache",) if "cache" in by_name else ("kv", "kr", "cache_descale"),
    }

    slots = []
    for name in _DISPLAY_SLOTS:
        present = [by_name[src] for src in groups[name] if src in by_name]
        values = [v for item in present if (v := _record_precision(item)) is not None]
        passed = bool(present) and all(bool(item.get("pass")) for item in present)
        detail = None
        if present and not passed:
            failed = [item["name"] for item in present if not item.get("pass")]
            if not failed:
                detail = f"{name} acceptance rejected"
            elif failed == [name]:
                detail = f"{name} not accepted"
            else:
                detail = f"{name}: " + ", ".join(failed)
        slots.append(
            {
                "name": name,
                # 空槽位或已通过结构检查的槽位显示 100%，使其与 Golden 形状逐项对应。
                "precision": min(values) if values else (100.0 if passed else 0.0),
                "pass": passed,
                "error_info": detail,
            }
        )
    return slots


def _record_precision(record):
    """计算比较列显示的记录匹配百分比。"""
    if not isinstance(record, dict):
        return None
    value = record.get("precision")
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    value = record.get("fulfill_percent")
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    elements = record.get("elements")
    failed = record.get("failed_elements")
    if isinstance(elements, int) and isinstance(failed, int) and elements > 0:
        return round(100.0 * (elements - failed) / elements, 4)
    if "passed" in record:
        return 100.0 if record["passed"] else 0.0
    return None


_FLAT_PASS = [
    {"name": name, "pass": True, "precision": 100.0, "error_info": None}
    for name in ("q", "descale_q", "qr", "cache")
]

_LOGGED = False


def _reference_context(context):
    if context is None:
        raise ValueError("missing TTK compare_context; precision is unverified")
    ref = _load_sibling("pytest_reference")
    tensors = list(getattr(context, "input_tensors", None) or [])
    if len(tensors) < 9:
        raise ValueError("incomplete TTK comparison inputs")
    args = [_as_torch(t) for t in tensors]
    args += [None] * (len(_INPUT_NAMES) - len(args))
    args[7] = _cache_snapshot(args[7])
    raw = dict(getattr(context, "attributes", None) or {})
    import mla_prolog_v4_ttk_ops as adapter

    fields = dict(getattr(context, "csv_fields", None) or {})
    weight_format = adapter._resolve_weight_format(
        (args[1], args[2], args[3]),
        fields.get("tensor_formats"),
        raw.get("weight_format"),
    )
    args[1:4] = _load_sibling("golden").reference_weights(args[1:4], weight_format)
    attrs = {k: raw[k] for k in _REFERENCE_ATTRS if k in raw}
    return ref, args, attrs


def _unchanged_cache(ref, before, after, index, mode):
    old = (
        ref.cache_to_logical(before, mode)
        .contiguous()
        .view(torch.uint8)
        .reshape(before.shape[0] * before.shape[1], -1)
    )
    new = (
        ref.cache_to_logical(after, mode)
        .contiguous()
        .view(torch.uint8)
        .reshape(before.shape[0] * before.shape[1], -1)
    )
    untouched = torch.ones(old.shape[0], dtype=torch.bool)
    untouched[index[index >= 0]] = False
    if not torch.equal(old[untouched], new[untouched]):
        raise ValueError("Untouched Cache bytes changed")


def mla_prolog_v4_compare(
    q,
    descale_q,
    qr,
    cache,
    q_ref,
    descale_q_ref,
    qr_ref,
    cache_ref,
    *,
    compare_context=None,
    **kwargs,
):
    ref, args, attrs = _reference_context(compare_context)
    attrs = (
        dict(
            norm_eps=1e-5,
            qkv_quant_mode=0,
            cache_mode="PA_BSND",
            qr_scale=1.0,
            kvr_scale=1.0,
            rope_mode=1,
            return_qr=False,
        )
        | attrs
    )
    mode = _mode_attr("qkv_quant_mode", attrs["qkv_quant_mode"])
    attrs["qkv_quant_mode"] = mode
    cache_mode = attrs["cache_mode"]
    attrs["cache_mode"] = (ref.CacheMode.PA_BSND, ref.CacheMode.PA_NZ)[
        _mode_attr("cache_mode", cache_mode)
    ]
    attrs["rope_mode"] = _mode_attr("rope_mode", attrs["rope_mode"])
    inputs = dict(zip(_INPUT_NAMES, args))
    before = inputs["kv_cache"].clone()
    call_inputs = dict(inputs, kv_cache=before.clone())
    _, captured = ref.golden_with_reference(ref.golden, call_inputs, attrs)
    device = tuple(_as_torch(t) for t in (q, descale_q, qr))
    expected = tuple(_as_torch(t) for t in (q_ref, descale_q_ref, qr_ref))
    after, expected_cache = _cache_snapshot(cache), _cache_snapshot(cache_ref)
    for a, b in zip(device, expected):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("output shape/dtype differs from pytest Golden")
    direct = ref.compare_numeric_outputs(
        ref.golden, device, expected, after, expected_cache, inputs, attrs, captured
    )
    diagnostics = ref.compare_numeric_outputs(
        ref.golden,
        device,
        expected,
        after,
        expected_cache,
        inputs,
        attrs,
        captured,
        split_diagnostics=True,
    )
    fallback = None
    if ref.precision_status(direct) == "FAIL":
        fallback_inputs = dict(inputs, kv_cache=expected_cache)
        fallback, _ = ref.fallback_segments(
            ref.golden, fallback_inputs, attrs, device, after, expected, captured
        )
    verdict = ref.accept_output_comparisons(direct, fallback, diagnostics, mode)
    _unchanged_cache(ref, before, after, inputs["cache_index"], attrs["cache_mode"])
    logging.info("数值验收结果: %s", verdict)
    records = []
    for name, decision in verdict["outputs"].items():
        metric = direct[name]
        passed = decision["accepted_by"] in ("TWO_WAY", "THREE_WAY", "SPLIT_FALLBACK")
        precision = (
            100.0 * (1 - metric.get("failed_elements", 0) / metric["elements"])
            if metric.get("elements")
            else 100.0
        )
        records.append(
            dict(
                name=name,
                pass_=passed,
                precision=precision,
                error_info=None if passed else str(decision),
            )
        )
        records[-1]["pass"] = records[-1].pop("pass_")
    return _display_records(records)
