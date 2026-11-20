# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""与 pytest 解耦的参数契约检查辅助函数。"""

from enum import Enum
import inspect

from unittest.mock import patch

from pathlib import Path

import json
import os

import pytest
import torch
import torch_npu


QKV_MODES = {
    "A16C16": 0,
    "A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE": 1,
    "A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR": 2,
    "A8C8_QKV_HIFLOAT8_PER_TENSOR": 3,
}


ROPE_MODES = {"NO_ROPE": 0, "ROPE_INTERLEAVE_HALF": 1, "ROPE_HALF": 2}


WEIGHT_MODES = {
    "A8W8_MXFP8": 1,
    "A8W8_HIFLOAT8_PER_TENSOR": 2,
    "A4W4_HIFLOAT4": 3,
}


def check_enum_contract(module):
    name = "quant_mla_prolog"
    operator = getattr(module, name)
    native = getattr(module, name + "_native")
    entries = [operator, native]
    expected = {
        "QkvQuantMode": QKV_MODES,
        "CacheMode": {"PA_BSND": 0, "PA_NZ": 1},
        "RopeMode": ROPE_MODES,
    }
    expected["WQrQuantMode"] = WEIGHT_MODES
    entries.append(module.quant_mla_prolog_precheck)
    assert operator.precheck is module.quant_mla_prolog_precheck
    attrs = dict(
        qkv_quant_mode=operator.QkvQuantMode.A16C16,
        cache_mode=operator.CacheMode.PA_BSND,
        rope_mode=operator.RopeMode.ROPE_INTERLEAVE_HALF,
    )
    attrs["w_qr_quant_mode"] = operator.WQrQuantMode.A8W8_MXFP8
    inputs = [None] * 13
    foreign = Enum("ForeignMode", {"VALUE": 1}).VALUE
    for entry in entries:
        for parameter in inspect.signature(entry).parameters.values():
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                assert parameter.default is inspect.Parameter.empty
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                assert parameter.default is not inspect.Parameter.empty
        for required in ("w_qr_quant_mode", "qkv_quant_mode"):
            with pytest.raises(TypeError, match=required):
                entry(
                    *inputs,
                    **{key: value for key, value in attrs.items() if key != required},
                )
        for enum_name, members in expected.items():
            enum = getattr(entry, enum_name)
            assert enum is getattr(operator, enum_name)
            assert {
                key: member.value for key, member in enum.__members__.items()
            } == members
        for attr in attrs:
            for bad in (0, 1, 2, 3, True, 1.0, "1", None, foreign):
                with pytest.raises(TypeError, match=attr):
                    entry(*inputs, **dict(attrs, **{attr: bad}))
    namespace = torch.ops.cannbotdsl_quant_mla_prolog
    with (
        patch.object(torch.compiler, "is_compiling", return_value=False),
        patch.object(getattr(namespace, name), "default") as dispatch,
    ):
        for rope in operator.RopeMode:
            result = operator(*inputs, **dict(attrs, rope_mode=rope))
            assert result is dispatch.return_value
            assert dispatch.call_args.kwargs["rope_mode"] == rope.value
            assert dispatch.call_args.kwargs["qkv_quant_mode"] == 0
            assert dispatch.call_args.kwargs["cache_mode"] == 0
        operator(*inputs, **{k: v for k, v in attrs.items() if k != "rope_mode"})
        assert dispatch.call_args.kwargs["rope_mode"] == 1
        operator(
            *inputs,
            qkv_quant_mode=attrs["qkv_quant_mode"],
            w_qr_quant_mode=attrs["w_qr_quant_mode"],
        )
        assert dispatch.call_args.kwargs["norm_eps"] == 1e-5
        assert dispatch.call_args.kwargs["cache_mode"] == 0
        operator(*inputs, attrs["w_qr_quant_mode"], attrs["qkv_quant_mode"])
        assert dispatch.call_args.kwargs["w_qr_quant_mode"] == 1
        assert dispatch.call_args.kwargs["qkv_quant_mode"] == 0
    for schema_name in ("quant_mla_prolog", "_quant_mla_prolog_graph"):
        arguments = {
            argument.name: argument
            for argument in getattr(namespace, schema_name).default._schema.arguments
        }
        for required in ("w_qr_quant_mode", "qkv_quant_mode"):
            assert not arguments[required].has_default_value()
        assert arguments["norm_eps"].default_value == 1e-5
        assert arguments["cache_mode"].default_value == 0
        for argument in arguments.values():
            assert argument.kwarg_only == argument.has_default_value()
    for entry in entries:
        with pytest.raises(NotImplementedError, match="A4W4_HIFLOAT4"):
            entry(
                *inputs,
                **dict(attrs, w_qr_quant_mode=operator.WQrQuantMode.A4W4_HIFLOAT4),
            )
    module.validate_modes(1, 0, 0, 2)
    with pytest.raises(ValueError, match="rope_mode"):
        module.validate_modes(1, 0, 0, 3)
    profiles = module.default_profiles()
    assert {p["rope_mode"] for p in profiles} == {0, 1, 2}
    with pytest.raises(ValueError, match="deployment mode"):
        module.program_configs(dict(profiles[0], rope_mode=3))


def pack_scale(scale):
    outputs, groups, pair = scale.shape
    return (
        scale.reshape(outputs // 16, 16, groups, pair).permute(0, 2, 1, 3).contiguous()
    )


def make_nz_weight_descale(packed, torch_npu, *, byte_carrier=False):
    rows, groups, block, pair = packed.shape
    logical = (
        packed.view(torch.uint8)
        .permute(1, 0, 2, 3)
        .contiguous()
        .reshape(groups, rows * block, pair)
    )
    if byte_carrier:
        return torch_npu.npu_format_cast(
            logical.npu(), 29, input_dtype=torch.float8_e8m0fnu
        )
    return torch_npu.npu_format_cast(logical.view(torch.float8_e8m0fnu).npu(), 29)


def public_attributes(operator, attrs, inputs=None):
    public = dict(
        attrs,
        w_qr_quant_mode=operator.WQrQuantMode(attrs["w_qr_quant_mode"]),
        qkv_quant_mode=operator.QkvQuantMode(attrs["qkv_quant_mode"]),
        cache_mode=operator.CacheMode(attrs["cache_mode"]),
        rope_mode=operator.RopeMode(attrs.get("rope_mode", 1)),
    )
    if inputs is not None:
        public.update(zip(("quant_scale_kv", "rope_sin", "rope_cos"), inputs[13:]))
    return public


def check_weight_descale_format_contract(module, torch_npu, device_args, attrs, nz):
    with patch.object(module, "_program") as program:
        for index in (6, 7, 8):
            bad = list(device_args)
            if nz:
                bad[index] = torch.empty(
                    device_args[index].shape,
                    dtype=torch.float8_e8m0fnu,
                    device=device_args[index].device,
                )
            else:
                packed = pack_scale(device_args[index].cpu().view(torch.uint8))
                bad[index] = make_nz_weight_descale(packed, torch_npu)
            with pytest.raises(ValueError, match="must match weight format"):
                module.quant_mla_prolog_native(
                    *bad[:13],
                    **public_attributes(module.quant_mla_prolog_native, attrs, bad),
                )
        program.assert_not_called()


def check_return_qr_contract(operator):
    inputs = [None] * 13
    attrs = public_attributes(
        operator, dict(w_qr_quant_mode=1, qkv_quant_mode=0, cache_mode=0)
    )
    with patch.object(
        torch.ops.cannbotdsl_quant_mla_prolog.quant_mla_prolog, "default"
    ) as dispatch:
        for value in (0, 1, -1, 0.0, 1.0, None, "true", [], torch.tensor(True)):
            with pytest.raises(TypeError, match="^return_qr must be bool$"):
                operator(*inputs, **attrs, return_qr=value)
        dispatch.assert_not_called()
        for value in (False, True):
            assert operator(*inputs, **attrs, return_qr=value) is dispatch.return_value
            assert dispatch.call_args.kwargs["return_qr"] is value
        operator(*inputs, **attrs)
        assert dispatch.call_args.kwargs["return_qr"] is False
