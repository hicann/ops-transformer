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
    "descale_x",
    "descale_wqa",
    "descale_wqb",
    "descale_wkva",
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
    "w_qr_quant_mode",
    "qkv_quant_mode",
    "cache_mode",
    "qr_scale",
    "kvr_scale",
    "rope_mode",
    "return_qr",
)

_MODE_NAMES = {
    "w_qr_quant_mode": {"MXFP8": 1, "HIF8": 2},
    "qkv_quant_mode": {
        "NO_QUANT": 0,
        "KV_FP8_PER_TOKEN_GROUP": 1,
        "Q_KV_FP8": 2,
        "Q_KV_HIF8": 3,
    },
    "cache_mode": {"PA_BSND": 0, "PA_NZ": 1},
    "rope_mode": {"NONE": 0, "INTERLEAVE_HALF": 1, "HALF": 2},
}

TOLERANCE = {
    "bfloat16": {"standard": "stat_rel_err"},
    "float32": {"standard": "stat_rel_err"},
    "float8_e4m3fn": {"standard": "stat_rel_err"},
    "float8_e5m2": {"standard": "stat_rel_err"},
}


def _load_sibling(stem):
    name = f"quant_mla_prolog_ttk_assets_impl_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{stem}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _as_torch(value):
    """将 TTK 的 NumPy 输出及 BF16、FP8、E8M0、uint8 载体适配为 Torch 张量。"""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    arr = np.asarray(value)
    name = str(arr.dtype)
    arr = np.ascontiguousarray(arr)
    if "bfloat16" in name:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    if "float8_e4m3fn" in name:
        return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)
    if name.startswith("float8_e5m2"):
        return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e5m2)
    if "hifloat8" in name:
        # 没有对应 Torch dtype 时保留字节载荷，遵循接口的 HiF8 载体契约。
        return torch.from_numpy(arr.view(np.uint8))
    if "e8m0" in name:
        native = getattr(torch, "float8_e8m0fnu", None)
        raw = torch.from_numpy(arr.view(np.uint8))
        return raw.view(native) if native is not None else raw
    return torch.from_numpy(arr)


def _cache_snapshot(value):
    """创建独立、连续的 CPU Cache 存储快照。

    设备布局由预处理保留；仅整理 CPU 快照，并保持各页 PA_BSND/PA_NZ 的载荷字节。
    """
    tensor = _as_torch(value)
    return tensor.contiguous().clone() if tensor is not None else None


def _mode_int(value, names):
    import quant_mla_ttk_ops as adapter

    mappings = _MODE_NAMES
    name = next(key for key, mapping in mappings.items() if mapping is names)
    return adapter._mode_int(name, value)


def _segment_precision(m):
    """返回单个分段的元素匹配百分比。

    TTK 直接展示 precision 字段，因此使用实际 matched_ratio，不将通过状态简化为 0/100。
    没有 matched_ratio 的合成验收汇总记录按判定结果显示。
    """
    ratio = m.get("matched_ratio")
    if isinstance(ratio, (int, float)) and m.get("elements"):
        # 显示四位小数，避免 0.9999984 的匹配率在 method10 失败时被显示为 100.00%。

        return round(float(ratio) * 100.0, 4)
    return 100.0 if m.get("passed", True) else 0.0


def _records(metrics):
    """将数值指标映射为 TTK 比较记录。

    MXFP8 验收路线可将失败分段的 gating 设为 False，保留原始 passed 和指标用于日志。
    整体结论由三方验收记录给出；TTK 必须遵循 gating，不能只使用 passed。
    """
    out = []
    for m in metrics:
        passed = bool(m.get("passed", True))
        gating = bool(m.get("gating", True))
        ok = passed or not gating
        out.append(
            {
                "name": m.get("name", "segment"),
                "pass": ok,
                "precision": _segment_precision(m),
                "error_info": None
                if ok
                else (
                    f"{m.get('name')}: {m.get('failed_elements')} failed / "
                    f"{m.get('elements')} elements, method={m.get('method')}"
                ),
                "metrics": m,
            }
        )
    return out


_TWO_WAY_ALIAS = {
    "Q NoPE/KV": "Q NoPE",
    "Q RoPE/KR": "Q RoPE",
    "Cache NoPE/KV": "Cache NoPE",
    "Cache RoPE/KR": "Cache RoPE",
    "descale_qr E8M0": "descale_qr",
}

_DISPLAY_SLOTS = ("q", "descale_q", "qr", "descale_qr", "kv_cache")


def _display_records(records, w_qr_quant_mode=1, return_qr=True):
    """将详细分段记录汇总到五个公开输出槽位。

    Comparison 百分比与 Golden 的五个形状逐项对应。
    Q NoPE/KV 等细分记录和验收记录保留在日志及内部判定中，不增加展示槽位。
    """
    by_name = {}
    for record in records:
        by_name.setdefault(record["name"], record)

    def _num(name):
        record = by_name.get(name)
        value = record.get("precision") if record else None
        return value if isinstance(value, (int, float)) else None

    def _slot(name, sources, acceptance):
        values = [(src, _num(src)) for src in sources if _num(src) is not None]
        decision = by_name.get(acceptance)
        if decision is not None:
            passed = bool(decision["pass"])
        else:
            passed = all(by_name[src]["pass"] for src, _ in values)
        # 空槽位或已通过结构检查的槽位显示 100%，保持与 Golden 形状逐项对应。

        precision = min(value for _, value in values) if values else 100.0
        detail = None
        if not passed:
            failed = [src for src, value in values if not by_name[src]["pass"]]
            detail = f"{name}: " + (
                ", ".join(failed) if failed else "acceptance rejected"
            )
        return {
            "name": name,
            "precision": precision,
            "pass": passed,
            "error_info": detail,
        }

    scale_name = "descale_qr E8M0" if w_qr_quant_mode == 1 else "descale_qr"
    if w_qr_quant_mode == 1 or return_qr:
        if scale_name not in by_name or scale_name + " acceptance" not in by_name:
            raise ValueError("missing descale_qr comparison/acceptance records")

    return [
        _slot("q", ("Q",), "Q acceptance"),
        _slot("descale_q", ("descale_q",), "descale_q acceptance"),
        _slot("qr", ("QR",), "QR acceptance"),
        (
            _slot("descale_qr", ("descale_qr E8M0",), "descale_qr E8M0 acceptance")
            if w_qr_quant_mode == 1
            else _slot("descale_qr", ("descale_qr",), "descale_qr acceptance")
            if return_qr
            else {
                "name": "descale_qr",
                "precision": 100.0,
                "pass": True,
                "error_info": None,
            }
        ),
        _slot(
            "kv_cache",
            ("Cache", "Cache KV", "Cache KR", "Cache descale"),
            "Cache acceptance",
        ),
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
    args[11] = _cache_snapshot(args[11])
    raw = dict(getattr(context, "attributes", None) or {})
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


def quant_mla_prolog_compare(
    q,
    descale_q,
    qr,
    descale_qr,
    cache,
    q_ref,
    descale_q_ref,
    qr_ref,
    descale_qr_ref,
    cache_ref,
    *,
    compare_context=None,
    **kwargs,
):
    ref, args, attrs = _reference_context(compare_context)
    attrs = (
        dict(
            norm_eps=1e-5,
            w_qr_quant_mode=1,
            qkv_quant_mode=0,
            cache_mode=0,
            qr_scale=1.0,
            kvr_scale=1.0,
            rope_mode=1,
            return_qr=False,
        )
        | attrs
    )
    for name, names in _MODE_NAMES.items():
        attrs[name] = _mode_int(attrs[name], names)
    raw = dict(getattr(compare_context, "attributes", None) or {})
    import quant_mla_ttk_ops as adapter

    fields = dict(getattr(compare_context, "csv_fields", None) or {})
    weight_format = adapter._resolve_weight_format(
        (args[1], args[2], args[3]),
        fields.get("tensor_formats"),
        raw.get("weight_format"),
    )
    for i in (6, 7, 8):
        args[i] = _load_sibling("golden").descale_logical(
            args[i], weight_format, args[i - 5].shape[0]
        )
    before = args[11].clone()
    call_args = list(args)
    call_args[11] = before.clone()
    _, observed = ref.golden_with_observations(
        ref.quant_mla_prolog_golden, call_args, attrs
    )
    actual = tuple(_as_torch(t) for t in (q, descale_q, qr, descale_qr))
    expected = tuple(
        _as_torch(t) for t in (q_ref, descale_q_ref, qr_ref, descale_qr_ref)
    )
    for a, b in zip(actual, expected):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("output shape/dtype differs from pytest Golden")
    actual_cache, reference_cache = _cache_snapshot(cache), _cache_snapshot(cache_ref)
    metrics = ref.compare_for_acceptance(
        actual, expected, actual_cache, reference_cache, args, attrs, observed
    )
    _unchanged_cache(ref, before, actual_cache, args[12], attrs["cache_mode"])
    records = _records(metrics)
    logging.info(
        "数值判定结果: %s",
        [
            (m.get("name"), m.get("passed"), m.get("gating", True), m.get("route"))
            for m in metrics
        ],
    )
    gating = [m for m in metrics if m.get("gating", True) and "passed" in m]
    if not gating:
        raise ValueError("pytest produced no verified acceptance records")
    display = _display_records(records, attrs["w_qr_quant_mode"], attrs["return_qr"])
    if any(not m["passed"] for m in gating) and all(r["pass"] for r in display):
        raise ValueError("unmapped pytest acceptance failure")
    return display
