# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mode-12 TestSpec adapter; share the pytest INT8 data and independent references."""

import inspect
from pathlib import Path
import sys

import torch

_ASSETS = str(Path(__file__).resolve().parents[1])
if _ASSETS not in sys.path:
    sys.path.insert(0, _ASSETS)
import quant_flash_attn_int8_golden as golden

_REFERENCES = {}


def _tensor(value):
    if getattr(getattr(value, "dtype", None), "name", None) in ("bfloat16", "uint16"):
        return torch.from_numpy(value.view("uint16")).view(torch.bfloat16)
    return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)


def _lengths(q, layout, cumulative):
    if layout != "TND":
        return [q.shape[2 if layout == "BNSD" else 1]] * q.shape[0]
    values = _tensor(cumulative).tolist()
    return [end - start for start, end in zip(values, values[1:])]


def cpu_golden(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    v_tail=None,
    block_table_tail=None,
    seqused_v_tail=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """Op-schema ordered CPU golden and ATK numerical benchmark, without NPU imports."""
    tensors = [_tensor(t) for t in (q, k, v, q_descale, k_descale)]
    qlens = _lengths(tensors[0], layout_q, cu_seqlens_q)
    klens = _lengths(tensors[1], layout_kv, cu_seqlens_kv)
    parts = [
        golden.unpack(t, layout, lengths)
        for t, layout, lengths in zip(
            tensors,
            (layout_q, layout_kv, layout_kv, layout_q, layout_kv),
            (qlens, klens, klens, qlens, klens),
        )
    ]
    vs = _tensor(v_descale)
    ps = float(_tensor(p_scale).reshape(-1)[0])
    options = dict(
        causal=int(mask_mode) == 3,
        win_left=win_left,
        win_right=win_right,
        softmax_scale=softmax_scale,
    )
    fp64, numerical, lses = [], [], []
    for batch in zip(*parts):
        fp64.append(
            (
                golden.attention_reference(
                    batch, vs, high_precision=True, p_scale=ps, **options
                ),
            )
        )
        numerical.append(
            (golden.attention_reference(batch, vs, p_scale=ps, **options).bfloat16(),)
        )
        if return_softmax_lse:
            lses.append(golden.lse_reference(batch, **options))
    result = [golden._pack(fp64, layout_out, 0)]
    benchmark = golden._pack(numerical, layout_out, 0)
    if return_softmax_lse:
        result.append(
            torch.cat(lses, 2).squeeze(0) if layout_out == "TND" else torch.cat(lses, 0)
        )
    _REFERENCES[kwargs.get("testcase_name")] = benchmark
    return result


def has_reference(testcase_name):
    return testcase_name in _REFERENCES


def clear_reference():
    _REFERENCES.clear()


def compare(*outputs, testcase_name=None):
    outputs = tuple(
        value for value in outputs if value is not None and _tensor(value).numel() != 0
    )
    count = len(outputs) // 2
    if count not in (1, 2) or len(outputs) != count * 2:
        return {
            "pass": False,
            "error_info": "INT8 expects attention[/LSE] actual and golden outputs",
        }
    actual, expected = outputs[:count], outputs[count:]
    try:
        golden.assert_atk_accuracy(
            _tensor(expected[0]), _tensor(actual[0]), _REFERENCES[testcase_name]
        )
        if count == 2:
            torch.testing.assert_close(
                _tensor(actual[1]).double(),
                _tensor(expected[1]).double(),
                rtol=5e-4,
                atol=3e-3,
            )
    except AssertionError as error:
        return {"pass": False, "error_info": str(error)}
    return {"pass": True, "error_info": None}


def _write(slot, value, name):
    source = torch.as_tensor(value)
    if slot is None or tuple(slot.shape) != tuple(source.shape):
        raise ValueError(
            f"INT8 {name} slot shape differs from generated data: {source.shape}"
        )
    if torch.is_tensor(slot):
        slot.copy_(source.to(dtype=slot.dtype, device=slot.device))
    else:
        slot[...] = source.numpy()


def customize_inputs(*args, **kwargs):
    bound = inspect.signature(cpu_golden).bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    attrs = arguments["kwargs"]
    q, k = arguments["q"], arguments["k"]
    layout, kv_layout = arguments["layout_q"], arguments["layout_kv"]
    qvalues = attrs.get("cu_seqlens_q_values", arguments["cu_seqlens_q"])
    kvalues = attrs.get("cu_seqlens_kv_values", arguments["cu_seqlens_kv"])
    qlens, klens = _lengths(q, layout, qvalues), _lengths(k, kv_layout, kvalues)
    heads, kv_heads = (
        q.shape[2 if layout == "BSND" else 1],
        k.shape[2 if kv_layout == "BSND" else 1],
    )
    batches, vs = golden._inputs(
        qlens, klens, heads, kv_heads, q.shape[-1], attrs.get("seed", 218)
    )
    for index, (name, tensor_layout) in enumerate(
        zip(
            ("q", "k", "v", "q_descale", "k_descale"),
            (layout, kv_layout, kv_layout, layout, kv_layout),
        )
    ):
        _write(arguments[name], golden._pack(batches, tensor_layout, index), name)
    _write(arguments["v_descale"], vs, "v_descale")
    _write(arguments["p_scale"], [attrs.get("p_scale_value", 120.0)], "p_scale")
    for name, values in (("cu_seqlens_q", qvalues), ("cu_seqlens_kv", kvalues)):
        if values is not None:
            _write(arguments[name], _tensor(values).to(torch.int32), name)
