# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Identical measurement boundaries for the A5 and A6 adapters.

Kernel samples reuse metadata; only device main-kernel durations are comparable.
End-to-end samples regenerate metadata and include allocation, forward and sync.
Input generation/H2D and first compilation are outside both measurements.
"""

import os
import time


def options():
    timing = os.environ.get("KDA_PERF_TIMING", "kernel")
    template = os.environ.get("KDA_PERF_TEMPLATE", "auto")
    if timing not in ("kernel", "e2e"):
        raise ValueError("KDA_PERF_TIMING must be kernel or e2e")
    if template not in ("auto", "staged", "persistent"):
        raise ValueError("KDA_PERF_TEMPLATE must be auto, staged or persistent")
    return timing, template


def metadata(adapter, inputs, config, template):
    return adapter.produce(
        inputs, config, template=None if template == "auto" else template
    )


def sample(
    adapter,
    inputs,
    config,
    kwargs,
    *,
    timing,
    template,
    synchronize,
    clock=time.perf_counter_ns,
):
    """One completed call. The caller releases previous outputs before timing."""
    start = clock()
    if timing == "e2e":
        current = metadata(adapter, inputs, config, template)
        kwargs = adapter.call_kwargs(inputs, config, current)
    elif timing != "kernel":
        raise ValueError("timing must be kernel or e2e")
    out = adapter.dut_call(kwargs)
    synchronize()
    return out, (clock() - start) / 1000


def execute(adapter, inputs, config, cache):
    """TTK warmup and sampling; no monkeypatching of the DUT or its validation."""
    import torch

    timing, template = options()
    key = (
        inputs[0].device,
        tuple(x.data_ptr() for x in inputs),
        repr(config["kwargs"]),
        config["inference"],
        timing,
        template,
    )
    with adapter.mode(config):
        if key not in cache:
            cache.clear()
            md = metadata(adapter, inputs, config, template)
            kwargs = adapter.call_kwargs(inputs, config, md)
            torch.npu.synchronize()
            out = None
            for _ in range(config["test"]["warmup_count"]):
                out = None
                out, _ = sample(
                    adapter,
                    inputs,
                    config,
                    kwargs,
                    timing=timing,
                    template=template,
                    synchronize=torch.npu.synchronize,
                )
            main, producer, _ = adapter.dut_modules()
            schedule = producer.decode_metadata(md.detach().cpu().tolist())
            cache[key] = dict(kwargs=kwargs, count=0, samples=[])
            adapter.event(
                config,
                status="performance_prepared",
                timing=timing,
                requested_template=template,
                dut_path=main.__file__,
                template=schedule["template"],
                core_num=schedule["core_num"],
                dv_splits=schedule.get("dv_splits"),
                warmup_count=config["test"]["warmup_count"],
                sample_count=config["test"]["sample_count"],
            )
            return [out[0]]
        state = cache[key]
        # TTK --task-prof=false must be used for wall-clock E2E acceptance.
        # In kernel mode the external profiler supplies the device duration.
        out, elapsed_us = sample(
            adapter,
            inputs,
            config,
            state["kwargs"],
            timing=timing,
            template=template,
            synchronize=torch.npu.synchronize,
        )
        state["samples"].append(elapsed_us)
        state["count"] += 1
        if state["count"] == config["test"]["sample_count"]:
            adapter.event(
                config,
                status="performance_samples",
                timing=timing,
                **{
                    (
                        "e2e_us" if timing == "e2e" else "profiled_forward_wall_us"
                    ): state["samples"]
                },
                aicore_source="profiler main-kernel durations only; wall time is not kernel time",
            )
        return [out[0]]
