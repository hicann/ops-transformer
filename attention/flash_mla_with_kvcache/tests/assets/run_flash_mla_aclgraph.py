#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate and benchmark paged MLA eager and static-shape ACL graph execution.

Use system Python after sourcing CANN and the desired custom operator package.
No TTK or PYTHONPATH setup is needed. Metadata is prepared before capture/timing.
Event intervals include dispatch gaps; --profile-dir additionally measures
MLA kernel durations. Metadata is excluded. Replays use warm inputs/caches.
Online static kernel compilation is opt-in and distinct from graph capture.
Use --output result.csv to export timings and shapes for calculate_mfu_mbu.py;
other output suffixes produce JSON. Utilization is calculated separately.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import statistics
import tempfile


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def benchmark(torch, call, warmup, iters, repeats):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    durations = []
    for _ in range(repeats):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) * 1000 / iters)
    return durations


def profile_kernels(torch, call, name, directory, iters):
    from torch_npu.profiler import (
        ExportType,
        ProfilerActivity,
        ProfilerLevel,
        _ExperimentalConfig,
        profile,
        schedule,
        tensorboard_trace_handler,
    )

    directory.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=directory))
    config = _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1, export_type=ExportType.Text
    )
    torch.npu.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        experimental_config=config,
        schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(output)),
    ) as prof:
        call()
        torch.npu.synchronize()
        prof.step()
        for _ in range(iters):
            call()
        torch.npu.synchronize()
        prof.step()
    files = list(output.rglob("kernel_details.csv"))
    if len(files) != 1:
        raise RuntimeError(
            f"expected one kernel_details.csv under {output}, got {files}"
        )
    kernels = {}
    with files[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("Step Id", "").strip():
                continue
            kernel = row.get("Name", "")
            op_type = row.get("Type", "")
            identity = (kernel + op_type).lower().replace("_", "")
            if "flashmlawithkvcache" not in identity or "metadata" in identity:
                continue
            duration = float(row["Duration(us)"])
            entry = kernels.setdefault(kernel, dict(calls=0, total_us=0.0))
            entry["calls"] += 1
            entry["total_us"] += duration
    if not kernels or any(entry["calls"] % iters for entry in kernels.values()):
        raise RuntimeError(
            f"unexpected MLA kernel counts: {kernels}; inspect {files[0]}"
        )
    us = sum(entry["total_us"] for entry in kernels.values()) / iters
    if us <= 0:
        raise RuntimeError(f"invalid kernel duration: {kernels}")
    return us, kernels, str(files[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cube-cores", type=positive_int)
    parser.add_argument("--vector-cores", type=positive_int)
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--q-heads", type=int, choices=(64, 96), default=96)
    parser.add_argument("--q-len", type=positive_int, default=1)
    parser.add_argument("--kv-len", type=positive_int, default=8192)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--layout-kv", choices=("PA_BBND", "PA_NZ"), default="PA_BBND")
    parser.add_argument("--mask-mode", type=int, choices=(0, 3), default=0)
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--static-kernel-compile", action="store_true")
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.q_len > args.kv_len:
        parser.error("q-len must not exceed kv-len")

    import torch
    import torch_npu  # noqa: F401
    import cann_ops_transformer  # noqa: F401
    from impl.golden import cpu_flash_mla_with_kvcache
    from impl.compare import compare
    from impl.npu_preprocess import run as preprocess
    from impl.graph import FlashMlaWithKvcacheAclGraph

    torch.npu.set_device(args.device)
    props = torch.npu.get_device_properties(args.device)
    cube = args.cube_cores or props.cube_core_num
    vector = args.vector_cores or props.vector_core_num
    if cube > props.cube_core_num or vector > props.vector_core_num:
        parser.error(f"requested cores exceed device capacity: {props}")
    torch.npu.set_device_limit(args.device, cube_num=cube, vector_num=vector)
    limits = torch.npu.get_device_limit(args.device)
    if limits != {"cube_core_num": cube, "vector_core_num": vector}:
        raise RuntimeError(f"core limits were not applied: {limits}")
    print(f"device={props}, core_limits={limits}", flush=True)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    pages = (args.kv_len + 127) // 128
    q_cpu = (
        torch.empty(args.batch * args.q_len, args.q_heads, 576)
        .uniform_(-1, 1)
        .to(dtype)
    )
    kv_cpu = torch.empty(args.batch * pages, 128, 1, 576).uniform_(-1, 1).to(dtype)
    if args.layout_kv == "PA_NZ":
        kv_cpu = kv_cpu.reshape(-1, 128, 1, 36, 16).permute(0, 2, 3, 1, 4).contiguous()
    aux_cpu = dict(
        block_table=torch.arange(args.batch * pages, dtype=torch.int32).reshape(
            args.batch, pages
        ),
        cache_seqlens=torch.full((args.batch,), args.kv_len, dtype=torch.int32),
        cu_seqlens_q=torch.arange(args.batch + 1, dtype=torch.int32) * args.q_len,
        seqused_q=torch.full((args.batch,), args.q_len, dtype=torch.int32),
        attn_mask=torch.ones(2048, 2048, dtype=torch.int8).triu(1)
        if args.mask_mode
        else None,
        metadata=torch.empty(0, dtype=torch.int32),
    )
    attrs = dict(
        head_dim_v=512,
        softmax_scale=576**-0.5,
        mask_mode=args.mask_mode,
        max_seqlen_q=args.q_len,
        max_seqlen_kv=args.kv_len,
        layout_q="TND",
        layout_kv=args.layout_kv,
        layout_out="NTD",
        return_softmax_lse=args.return_lse,
    )
    with torch.inference_mode():
        golden = cpu_flash_mla_with_kvcache(q_cpu, kv_cpu, **aux_cpu, **attrs)
        device = f"npu:{args.device}"
        q, kv = q_cpu.to(device), kv_cpu.to(device)
        aux = {
            key: value.to(device) if value is not None else None
            for key, value in aux_cpu.items()
        }
        preprocess(q, kv, **aux, **attrs)
        torch.npu.synchronize()
        print(f"metadata shape={tuple(aux['metadata'].shape)}", flush=True)
        model = FlashMlaWithKvcacheAclGraph(**attrs).eval()
        torch._dynamo.config.ignore_logger_methods.add(logging.Logger.warning)
        compiled = torch.compile(
            model,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options={"static_kernel_compile": args.static_kernel_compile},
        )
        result = dict(
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            device=str(props),
            environment={
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "ASCEND_HOME_PATH": os.environ.get("ASCEND_HOME_PATH"),
                "ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            },
            core_limits=limits,
            metadata_shape=list(aux["metadata"].shape),
            timing="median NPU event interval per call; warm cache; metadata excluded",
            results={},
        )
        eager_cpu = None
        for name, runner in (("eager", model), ("aclgraph", compiled)):

            def call():
                return runner(q, kv, **aux)

            outputs = call()
            torch.npu.synchronize()
            actual = tuple(value.cpu() for value in outputs)
            for value in actual[: 2 if args.return_lse else 1]:
                if not torch.isfinite(value).all():
                    raise AssertionError(f"{name} returned nonfinite output")
            checks = compare(*actual, *golden)
            if not all(check["pass"] for check in checks):
                raise AssertionError(f"{name} golden comparison failed: {checks}")
            if eager_cpu is None:
                eager_cpu = actual
            else:
                for i in range(2 if args.return_lse else 1):
                    torch.testing.assert_close(actual[i], eager_cpu[i], rtol=0, atol=0)
            samples = benchmark(torch, call, args.warmup, args.iters, args.repeats)
            event_us = statistics.median(samples)
            us = event_us
            kernels, profile_csv = {}, None
            if args.profile_dir:
                us, kernels, profile_csv = profile_kernels(
                    torch, call, name, args.profile_dir, args.iters
                )
            if us <= 0:
                raise RuntimeError(f"invalid event duration: {samples}")
            row = dict(
                precision="PASS",
                event_median_us=event_us,
                duration_us=us,
                timing_source="kernel" if args.profile_dir else "event",
                kernels=kernels,
                profile_csv=profile_csv,
                samples_us=samples,
            )
            result["results"][name] = row
            print(
                f"{name}: PASS, {row['timing_source']}={us:.3f} us "
                f"(event={event_us:.3f} us)",
                flush=True,
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() == ".csv":
                case_attrs = dict(attrs)
                for key in ("cache_seqlens", "cu_seqlens_q", "seqused_q"):
                    case_attrs[f"{key}_values"] = aux_cpu[key].tolist()
                tensors = (q_cpu, kv_cpu, *aux_cpu.values())
                common = dict(
                    api_name="torch.ops.cann_ops_transformer.flash_mla",
                    tensor_view_shapes=repr(
                        tuple(
                            tuple(t.shape) if t is not None else None for t in tensors
                        )
                    ),
                    tensor_dtypes=repr(
                        tuple(
                            str(t.dtype).removeprefix("torch.")
                            if t is not None
                            else None
                            for t in tensors
                        )
                    ),
                    attributes=repr(case_attrs),
                )
                rows = [
                    dict(
                        common,
                        testcase_name=f"mla_{mode}",
                        precision_status=value["precision"],
                        duration_us=value["duration_us"],
                        timing_source=value["timing_source"],
                        event_median_us=value["event_median_us"],
                    )
                    for mode, value in result["results"].items()
                ]
                with args.output.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"results: {args.output}", flush=True)
    if args.static_kernel_compile:
        print(
            "Inspect static_kernel_compile_outputs compile logs: execution may fall back."
        )


if __name__ == "__main__":
    main()
