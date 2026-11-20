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


def check_enum_contract(module):
    name = "mla_prolog_v2"
    operator = getattr(module, name)
    native = getattr(module, name + "_native")
    entries = [operator, native]
    expected = {
        "QkvQuantMode": QKV_MODES,
        "CacheMode": {"PA_BSND": 0, "PA_NZ": 1},
        "RopeMode": ROPE_MODES,
    }
    attrs = dict(
        qkv_quant_mode=operator.QkvQuantMode.A16C16,
        cache_mode=operator.CacheMode.PA_BSND,
        rope_mode=operator.RopeMode.ROPE_INTERLEAVE_HALF,
    )
    inputs = [None] * 9
    foreign = Enum("ForeignMode", {"VALUE": 1}).VALUE
    for entry in entries:
        for parameter in inspect.signature(entry).parameters.values():
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                assert parameter.default is inspect.Parameter.empty
            elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                assert parameter.default is not inspect.Parameter.empty
        for required in ("qkv_quant_mode",):
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
    namespace = torch.ops.cannbot_dsl_mla_prolog_v4
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
        operator(*inputs, qkv_quant_mode=attrs["qkv_quant_mode"])
        assert dispatch.call_args.kwargs["norm_eps"] == 1e-5
        assert dispatch.call_args.kwargs["cache_mode"] == 0
        operator(*inputs, attrs["qkv_quant_mode"])
        assert dispatch.call_args.kwargs["qkv_quant_mode"] == 0
    for schema_name in ("mla_prolog_v2", "_mla_prolog_v2_graph"):
        arguments = {
            argument.name: argument
            for argument in getattr(namespace, schema_name).default._schema.arguments
        }
        for required in ("qkv_quant_mode",):
            assert not arguments[required].has_default_value()
        assert arguments["norm_eps"].default_value == 1e-5
        assert arguments["cache_mode"].default_value == 0
        for argument in arguments.values():
            assert argument.kwarg_only == argument.has_default_value()
    module.validate_schema_modes(0, 0, 2)
    with pytest.raises(ValueError, match="rope_mode"):
        module.validate_schema_modes(0, 0, 3)


def read_trace(path):
    payload = json.loads(path.read_text())
    events = payload if isinstance(payload, list) else payload["traceEvents"]
    assert events, "Empty trace cannot prove absence of copies"
    return events


def audit_trace(trace, execution):
    events = read_trace(trace)
    tasks = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("args", {}).get("Task Type")
    ]
    models = {
        event["args"].get("Model Id")
        for event in tasks
        if event["args"]["Task Type"] == "MODEL_EXECUTE"
    }
    assert models, "No real graph replay observed"
    children = [event for event in tasks if event["args"].get("Model Id") in models]
    kernels = [
        event
        for event in children
        if event["args"]["Task Type"]
        in ("AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV")
        or event["args"]["Task Type"].startswith("KERNEL_")
    ]
    expected = 3 if execution == "npugraph-ex" else 1
    assert len(kernels) == expected, (
        f"Expected {expected} graph kernels, got {len(kernels)}"
    )
    assert all("MlaPrologKernel" in event["name"] for event in kernels), kernels
    forbidden_device = [
        event
        for event in tasks
        if any(
            token in (event["name"] + " " + str(event["args"])).lower()
            for token in (
                "tensormove",
                "foreachcopy",
                "memcpy",
                "copyback",
                "copy_back",
            )
        )
    ]
    windows = [
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in events
        if event.get("name") == "MLAP_REPLAY" and event.get("ph") == "X"
    ]
    assert windows, "Replay scope is missing"
    forbidden_host = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("name") in ("aten::clone", "aten::copy_", "aten::_to_copy")
        and any(start <= float(event["ts"]) < end for start, end in windows)
    ]
    assert not forbidden_device and not forbidden_host, (
        forbidden_device,
        forbidden_host,
    )
    return dict(
        status="PASS",
        graph_models=sorted(models),
        graph_kernels=len(kernels),
        forbidden_device_moves=0,
        replay_host_copies=0,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
    )


def run_entry(actual, attrs, execution, trace_dir):
    # 预检查和编译在图捕获前完成，回放不得向 CPU 读取数值。
    result = op.mla_prolog_v2(**actual, **attrs)
    torch.npu.synchronize()
    if execution == "eager":
        assert trace_dir is None, "Trace audit requires aclgraph or npugraph-ex"
        return result, None, None
    if execution == "aclgraph":
        graph = torch.npu.NPUGraph()

        def capture_and_replay():
            with torch.npu.graph(graph):
                output = op.mla_prolog_v2(**actual, **attrs)
            with torch.profiler.record_function("MLAP_REPLAY"):
                graph.replay()
            return output

        def replay():
            graph.replay()
            return result
    elif execution == "npugraph-ex":
        torch.compiler.reset()
        compiled = torch.compile(
            op.mla_prolog_v2,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options={"force_eager": False, "clone_input": False},
        )

        def replay():
            return compiled(**actual, **attrs)

        def capture_and_replay():
            with torch.profiler.record_function("MLAP_CAPTURE"):
                output = replay()
            with torch.profiler.record_function("MLAP_REPLAY"):
                output = replay()
                output = replay()
            return output
    else:
        raise ValueError("MLA_PROLOG_EXECUTION must be eager, aclgraph or npugraph-ex")
    if trace_dir is None:
        result = capture_and_replay()
        torch.npu.synchronize()
        return result, replay, None
    trace_dir.mkdir(parents=True, exist_ok=False)
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            data_simplification=False,
        ),
    ) as profiler:
        result = capture_and_replay()
        torch.npu.synchronize()
        profiler.step()
    trace = trace_dir / "trace.json"
    profiler.export_chrome_trace(str(trace))
    return result, replay, audit_trace(trace, execution)


def check_zero_cache_scale(cpu, actual, attrs, outputs, before):
    """直接检查零乘数；普通 Golden 要求正 scale。

    Q、对应 descale 和 QR 应与已经验证的正 scale 调用一致。
    Cache 的零值在编码域检查，不执行除以 scale 的运算。
    """
    expected_q = [
        value.cpu().contiguous().view(torch.uint8).clone() for value in outputs
    ]
    actual["kv_cache"].copy_(before.to(actual["kv_cache"].device))
    zero_inputs = dict(
        actual, quant_scale_kv=torch.zeros_like(actual["quant_scale_kv"])
    )
    zero_outputs = op.mla_prolog_v2(**zero_inputs, **attrs)
    torch.npu.synchronize()
    assert all(
        torch.equal(value.cpu().contiguous().view(torch.uint8), expected)
        for value, expected in zip(zero_outputs, expected_q)
    ), "Cache scale changed Q, descale_q or QR"
    capacity = before.shape[0] * before.shape[1]
    layout = golden.CacheMode[attrs["cache_mode"].name]
    after = golden.cache_to_logical(actual["kv_cache"].cpu(), layout)
    old = golden.cache_to_logical(before, layout)
    indices = cpu["cache_index"]
    indices = indices[indices >= 0]
    rows = after.reshape(capacity, -1)
    if attrs["qkv_quant_mode"] == op.QkvQuantMode(3):
        assert bool((golden.decode_hif8(rows[indices]) == 0).all())
    else:
        assert bool((rows.float()[indices] == 0).all())
    untouched = torch.ones(capacity, dtype=torch.bool)
    untouched[indices] = False
    old_bytes = old.contiguous().view(torch.uint8).reshape(capacity, -1)
    new_bytes = after.contiguous().view(torch.uint8).reshape(capacity, -1)
    assert torch.equal(old_bytes[untouched], new_bytes[untouched])
