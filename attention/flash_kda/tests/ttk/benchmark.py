# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5/A6 paired-template benchmark, with separate kernel and unprofiled E2E passes."""

import argparse
import ast
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def read_cases(path, selected=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["is_enabled"].lower() == "true"
            and json.loads(ast.literal_eval(r["attributes"])["test_config"])["kind"]
            == "performance"
        ]
    if selected:
        missing = set(selected) - {r["testcase_name"] for r in rows}
        if missing:
            raise ValueError(f"unknown profiling cases: {sorted(missing)}")
        rows = [r for r in rows if r["testcase_name"] in selected]
    return rows


def kernel_samples(folder, template, expected_count):
    paths = list(Path(folder).rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one kernel_details.csv, found {len(paths)} in {folder}"
        )
    values, names = [], set()
    with paths[0].open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("Name", "")
            # A5 profiles are staged/persistent_flash_kda; A6 adds _fwd_arch92.
            if not any(f"{t}_flash_kda" in name for t in ("staged", "persistent")):
                continue
            if f"{template}_flash_kda" not in name:
                raise ValueError(f"unexpected main-kernel template: {name}")
            value = float(row["Duration(us)"])
            if not 0 < value < float("inf"):
                raise ValueError(f"invalid main-kernel duration: {value}")
            values.append(value)
            names.add(name)
    if len(values) != expected_count:
        raise ValueError(
            f"expected {expected_count} main launches, found {len(values)}"
        )
    return values, sorted(names), str(paths[0].resolve())


def prepare(adapter, row, torch):
    config, cpu = adapter.prepare(row)
    hashes = {}
    for name, value in zip(adapter.load_asset("inputs").INPUT_NAMES, cpu):
        if name != "metadata":
            hashes[name] = hashlib.sha256(
                value.contiguous().view(torch.uint8).numpy()
            ).hexdigest()
    return config, [value.npu() for value in cpu], hashes


def measure(adapter, config, inputs, template, folder, args, torch, torch_npu):
    common = adapter.load_asset("performance")
    main, producer, _ = adapter.dut_modules()
    with adapter.mode(config):
        md = common.metadata(adapter, inputs, config, template)
        schedule = producer.decode_metadata(md.detach().cpu().tolist())
        actual = schedule["template"]
        if template != "auto" and actual != template:
            raise ValueError("forced metadata template mismatch")
        kwargs = adapter.call_kwargs(inputs, config, md)
        out = None
        for _ in range(args.warmup):
            out = None
            out, _ = common.sample(
                adapter,
                inputs,
                config,
                kwargs,
                timing="kernel",
                template=template,
                synchronize=torch.npu.synchronize,
            )
        profiler = torch_npu.profiler
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(folder)),
        ) as trace:
            for _ in range(args.samples):
                out = None
                out, _ = common.sample(
                    adapter,
                    inputs,
                    config,
                    kwargs,
                    timing="kernel",
                    template=template,
                    synchronize=torch.npu.synchronize,
                )
                trace.step()
        values, names, export = kernel_samples(folder, actual, args.samples)
        # A separate, unprofiled pass. Each sample includes the producer and
        # allocations inside the forward, and finishes on the same NPU stream.
        for _ in range(args.warmup):
            out = None
            out, _ = common.sample(
                adapter,
                inputs,
                config,
                kwargs,
                timing="e2e",
                template=template,
                synchronize=torch.npu.synchronize,
            )
        e2e = []
        for _ in range(args.samples):
            out = None
            out, elapsed = common.sample(
                adapter,
                inputs,
                config,
                kwargs,
                timing="e2e",
                template=template,
                synchronize=torch.npu.synchronize,
            )
            e2e.append(elapsed)
        del out
    record = dict(
        case_id=config["id"],
        requested_template=template,
        template=actual,
        layout=config["layout"],
        inference=config["inference"],
        kwargs=config["kwargs"],
        dut_path=main.__file__,
        core_num=schedule["core_num"],
        dv_splits=schedule.get("dv_splits"),
        schedule=schedule,
        metadata_sha256=hashlib.sha256(md.detach().cpu().numpy()).hexdigest(),
        kernel_us=values,
        e2e_us=e2e,
        kernel_names=names,
        kernel_details=export,
    )
    if args.arch == "arch92":
        from dataclasses import asdict
        from types import SimpleNamespace

        short_group = (
            not config["inference"]
            and config["layout"] == "BNSD"
            and config["batch_size"] == 1
            and config["nheads_qk"] == config["nheads_v"] == producer.SHORT_TRAIN_HEADS
            and config["seqlens"] >= producer.SHORT_TRAIN_MIN_LENGTH
            and config["seqlens"] % producer.CHUNK_SIZE == 0
            and config["cu_seqlens_value"] is None
            and config["seqused_value"] is None
            and schedule["core_num"] == producer.MAX_AIC_CORES
        )
        features = producer._a6_route_features(
            SimpleNamespace(
                batch=config["batch_size"],
                storage_length=config["seqlens"],
                value_heads=config["nheads_v"],
                query_heads=config["nheads_qk"],
                core_num=schedule["core_num"],
                has_cu=1,
                workspace_slots=producer.WORKSPACE_SLOTS,
                cu_seqlens=[0]
                + list(torch.tensor(config["valid_lengths"]).cumsum(0).tolist()),
            ),
            producer.SHORT_TRAIN_CHUNK_GROUP
            if short_group
            else producer.MAX_CHUNK_GROUP,
        )
        producer.a6_routing.check_schedule(features, schedule)
        record["features"] = asdict(features)
        record["persistent_core_quantities"] = features.persistent_quantities()
        record["feature_version"] = producer.a6_routing.FEATURE_VERSION
        record["persistent_chunk_balance"] = sum(features.core_chunks) / (
            features.core_num * max(features.core_chunks)
        )
        record["persistent_resident_state"] = getattr(
            main.persistent, "PERSISTENT_RESIDENT_STATE", False
        )
    return record


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["case_id"], record["requested_template"]), []).append(
            record
        )
    return [
        dict(
            case_id=case,
            requested_template=template,
            rounds=len(items),
            kernel_us=statistics.median(statistics.mean(r["kernel_us"]) for r in items),
            e2e_us=statistics.median(statistics.mean(r["e2e_us"]) for r in items),
        )
        for (case, template), items in groups.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("arch35", "arch92"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sdk-tag", required=True, help="Installed wheel/source fingerprint"
    )
    parser.add_argument("--cases", nargs="+", help="Default: all 25 profiling cases")
    parser.add_argument(
        "--templates",
        nargs="+",
        choices=("auto", "staged", "persistent"),
        default=["auto"],
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    if root == output or root in output.parents or output.exists():
        parser.error("output must be a new directory outside the source checkout")
    if min(args.warmup, args.samples, args.rounds) < 1:
        parser.error("warmup, samples and rounds must be positive")
    if len(set(args.templates)) != len(args.templates):
        parser.error("templates must be distinct")
    os.environ["FLASH_KDA_ARCH"] = args.arch
    os.environ.setdefault("KDA_DUT", "installed")
    folder = Path(__file__).resolve().parent
    os.environ["KDA_SOURCE_ROOT"] = str(root)
    output.mkdir(parents=True)
    os.environ["KDA_EVIDENCE_DIR"] = str(output / "evidence")
    os.environ.pop("CANNBOTDSL_NATIVE_BINARY_REPORT", None)
    sys.path.insert(0, str(folder))
    import torch
    import torch_npu
    import cannbotdsl

    torch.npu.set_device(args.device)
    adapter = importlib.import_module("kda_ttk_ops").get_adapter()
    dut, _, _ = adapter.dut_modules()
    expected = Path(dut.__file__).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    )
    source_hash = hashlib.sha256()
    for path in sorted(expected.parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes().replace(b"\r\n", b"\n"))
    environment = dict(
        arch=args.arch,
        dut=adapter.runtime.selection()[0],
        public_path=adapter.runtime.public_api().__file__,
        interface_contract="68f2d95c780ea1c7e728363d376919d221798ffe",
        commit=revision,
        dirty=dirty,
        sdk_tag=args.sdk_tag,
        sdk_path=cannbotdsl.__file__,
        source_sha256=source_hash.hexdigest(),
        torch_version=torch.__version__,
        device=torch.npu.get_device_name(args.device),
        warmup=args.warmup,
        samples=args.samples,
        rounds=args.rounds,
        timing="kernel: profiler main launches; e2e: producer + forward + sync, no profiler",
    )
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    records = []
    for index, row in enumerate(read_cases(folder / "cases/ttk_cases.csv", args.cases)):
        config, inputs, hashes = prepare(adapter, row, torch)
        for repeat in range(args.rounds):
            templates = (
                args.templates if (index + repeat) % 2 == 0 else args.templates[::-1]
            )
            for template in templates:
                record = measure(
                    adapter,
                    config,
                    inputs,
                    template,
                    output / config["id"] / f"round{repeat}-{template}",
                    args,
                    torch,
                    torch_npu,
                )
                record.update(round=repeat, input_sha256=hashes)
                records.append(record)
                with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(
                    config["id"],
                    repeat,
                    template,
                    "kernel_us",
                    statistics.mean(record["kernel_us"]),
                    "e2e_us",
                    statistics.mean(record["e2e_us"]),
                    flush=True,
                )
        del inputs
    (output / "summary.json").write_text(
        json.dumps(summarize(records), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
