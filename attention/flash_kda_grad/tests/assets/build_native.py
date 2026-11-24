# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build one loadable FlashKDA Grad Native package with parallel AOT workers."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def compile_jobs(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "jobs must be auto or a positive integer"
        ) from error
    if count < 1:
        raise argparse.ArgumentTypeError("jobs must be auto or a positive integer")
    return count


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _cgroup_directories(controller: str) -> set[Path]:
    """Probe v1/v2 membership and ancestors, including namespace-mounted roots."""
    root = Path("/sys/fs/cgroup")
    bases = [root, root / controller]
    if controller == "cpu":
        bases.append(root / "cpu,cpuacct")
    result = set(bases)
    for line in _read_text(Path("/proc/self/cgroup")).splitlines():
        _, names, relative = line.split(":", 2)
        if names and controller not in names.split(","):
            continue
        for base in bases:
            current = base.joinpath(
                *(part for part in Path(relative).parts if part not in ("/", ".", ".."))
            )
            while current != base:
                result.add(current)
                current = current.parent
    return result


def native_build_resources() -> tuple[int, int | None]:
    """Bound CPU by affinity/quota and memory by host availability/cgroup headroom."""
    cpus = os.cpu_count() or 1
    try:
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for directory in _cgroup_directories("cpu"):
        limits = _read_text(directory / "cpu.max").split()
        if not limits:
            limits = [
                _read_text(directory / "cpu.cfs_quota_us"),
                _read_text(directory / "cpu.cfs_period_us"),
            ]
        try:
            quota, period = map(int, limits)
            if quota > 0 and period > 0:
                cpus = min(cpus, max(1, quota // period))
        except ValueError:
            pass
    available = None
    for line in _read_text(Path("/proc/meminfo")).splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    for directory in _cgroup_directories("memory"):
        for limit_name, usage_name in [
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ]:
            try:
                limit = int(_read_text(directory / limit_name))
                used = int(_read_text(directory / usage_name))
            except ValueError:
                continue
            headroom = max(0, limit - used)
            available = headroom if available is None else min(available, headroom)
    return max(1, cpus), available


def native_compile_jobs(requested: int | str) -> int:
    cpus, memory = native_build_resources()
    # Leave room for compiler subprocesses and other users. Unknown memory is
    # deliberately conservative; --jobs 1 always retains the original route.
    memory_jobs = (
        max(1, (memory - 2 * 1024**3) // (4 * 1024**3)) if memory is not None else 1
    )
    resource_limit = min(32, cpus, memory_jobs)
    count = (
        min(resource_limit, max(1, cpus // 4))
        if requested == "auto"
        else min(resource_limit, requested)
    )
    print(
        f"KDAG compile processes: {count} (requested={requested}, available CPUs={cpus}, memory limit={memory_jobs})",
        flush=True,
    )
    return count


def export_plan(source: Path):
    """Read the runtime export plan without importing Torch or initializing the DSL."""
    names = {"_all_export_variants", "_backward_variants", "_recompute_export_variants"}
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["_all_export_variants"]()


def compile_schedule(plan, jobs):
    weights = []
    for recompute, _, full, masked, _, staged, _, cube_causal in plan:
        cost = 4 if full and masked else 2 if masked else 1
        if full and masked and cube_causal:
            cost *= 2
        weights.append(cost * (2 if recompute else 1) + int(staged))
    schedule = [[] for _ in range(jobs)]
    loads = [0] * jobs
    for index in sorted(range(len(plan)), key=lambda i: (-weights[i], i)):
        worker = min(range(jobs), key=lambda i: (loads[i], i))
        schedule[worker].append(index)
        loads[worker] += weights[index]
    return schedule


def run_worker(job_path: Path):
    job = json.loads(job_path.read_text())
    temp = Path(job["tmp"])
    temp.mkdir()
    Path(os.environ["ASCEND_PROCESS_LOG_PATH"]).mkdir()
    os.environ["TMPDIR"] = str(temp)
    tempfile.tempdir = str(temp)
    from cannbotdsl.aot import build

    for shard in job["shards"]:
        os.environ["CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD"] = f"{shard}/{job['count']}"
        started = time.monotonic()
        build(
            collectors=Path(job["source"]).parent,
            pattern=Path(job["source"]).name,
            operators=["flash_kda_grad"],
            target=job["target"],
            output=Path(job["parts"]) / str(shard),
        )
        print(
            f"Variant {shard} completed in {time.monotonic() - started:.2f}s",
            flush=True,
        )


def merge_flash_kda_native(parts: list[Path], output: Path, expected: int) -> Path:
    """Verify every shard, then regenerate the SDK's standard package manifests."""
    header, index, files, sources = None, {}, {}, []
    requests = 0
    operator_path = "operators/flash_kda_grad/operator.json"
    for part in parts:
        manifest = json.loads((part / "manifest.json").read_text())
        current = {k: v for k, v in manifest.items() if k != "files"}
        if header is None:
            header = current
        if current != header or current["index"] != "index.json":
            raise ValueError("Incompatible KDAG Native shard manifest")
        if not {"index.json", operator_path} <= manifest["files"].keys():
            raise ValueError("Unlisted KDAG Native package inventory")
        for relative, digest in manifest["files"].items():
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (part / path).resolve().is_relative_to(part.resolve())
            ):
                raise ValueError("Invalid KDAG Native artifact path")
            artifact = part / path
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Corrupted KDAG Native artifact: {artifact}")
            if relative not in ("index.json", operator_path):
                if relative in files:
                    raise ValueError(f"Duplicate KDAG Native artifact: {relative}")
                files[relative] = artifact
        shard_index = json.loads((part / "index.json").read_text())
        task = json.loads((part / operator_path).read_text())
        if (
            task["name"] != "flash_kda_grad"
            or task["variants"] != shard_index
            or set(task["artifacts"]) != set(shard_index)
            or task["requests"] != len(shard_index)
        ):
            raise ValueError("Inconsistent KDAG Native shard inventory")
        for key, entry in shard_index.items():
            if key in index:
                raise ValueError(f"Duplicate KDAG Provider: {key}")
            if entry["descriptor"] not in manifest["files"]:
                raise ValueError("Unlisted KDAG Native Provider descriptor")
            descriptor = json.loads((part / entry["descriptor"]).read_text())
            if (
                descriptor["binary_key"] != key
                or descriptor["target"] != header["target"]
            ):
                raise ValueError("Inconsistent KDAG Native Provider identity")
            for relative in (
                entry["descriptor"],
                descriptor["binary"],
                descriptor["sidecar"],
            ):
                if relative not in manifest["files"]:
                    raise ValueError("Unlisted KDAG Native Provider artifact")
            index[key] = entry
        requests += task["requests"]
        for source in task["sources"]:
            if source not in sources:
                sources.append(source)
    if requests != expected or len(index) != expected:
        raise ValueError(
            f"Incomplete KDAG Native export: requests={requests}, providers={len(index)}; expected {expected}"
        )
    output.mkdir()
    for relative, source in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    index = dict(sorted(index.items()))
    (output / "index.json").write_text(json.dumps(index, sort_keys=True))
    (output / operator_path).write_text(
        json.dumps(
            dict(
                name="flash_kda_grad",
                requests=requests,
                artifacts=list(index),
                variants=index,
                sources=sources,
            ),
            sort_keys=True,
        )
    )
    manifest = dict(
        header,
        files={
            str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(output.rglob("*"))
            if f.is_file()
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return output


def build_flash_kda_parallel(
    source: Path, output: Path, work: Path, target: str, jobs: int, plan
) -> Path:
    schedule = compile_schedule(plan, jobs)
    worker_root = Path(tempfile.mkdtemp(prefix="native-workers-", dir=work))
    parts = worker_root / "shards"
    parts.mkdir()
    processes, logs = [], []
    try:
        for worker, shards in enumerate(schedule):
            job_file = worker_root / f"worker-{worker}.json"
            job_file.write_text(
                json.dumps(
                    dict(
                        source=str(source),
                        tmp=str(worker_root / f"tmp-{worker}"),
                        count=len(plan),
                        target=target,
                        parts=str(parts),
                        shards=shards,
                    )
                )
            )
            log = (worker_root / f"worker-{worker}.log").open("w")
            logs.append(log)
            environment = dict(
                os.environ,
                CANNBOTDSL_CACHE_DIR=str(work / "cache"),
                XDG_CACHE_HOME=str(work / "xdg-cache"),
                PYTHONDONTWRITEBYTECODE="1",
                TORCH_DEVICE_BACKEND_AUTOLOAD="0",
                ASCEND_PROCESS_LOG_PATH=str(worker_root / f"log-{worker}"),
                CANNBOT_BINDING_JOBS="2",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(job_file),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        pending = set(range(len(processes)))
        while pending:
            for worker in list(pending):
                code = processes[worker].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(
                        f"KDAG Native worker {worker} exited {code}; see {worker_root / f'worker-{worker}.log'}"
                    )
                pending.remove(worker)
            if pending:
                time.sleep(0.1)
    except BaseException:
        # Workers own process groups so compiler children are also reclaimed.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    finally:
        for log in logs:
            log.close()
    with tempfile.TemporaryDirectory(prefix=".native-", dir=output.parent) as stage:
        package = Path(stage) / "package"
        merge_flash_kda_native(
            [parts / str(i) for i in range(len(plan))], package, len(plan)
        )
        if output.exists():
            raise FileExistsError(output)
        package.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "op_dsl" / "flash_kda_grad.py",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--target", default="dav-3510")
    parser.add_argument("--jobs", type=compile_jobs, default="auto")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        run_worker(args.worker)
        return
    if args.output is None or args.work_dir is None:
        parser.error("--output and --work-dir are required")
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    work = args.work_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    plan = export_plan(source)
    if not plan:
        raise ValueError("No FlashKDA Grad export variants")
    jobs = min(len(plan), native_compile_jobs(args.jobs))
    started = time.monotonic()
    print(f"Building {len(plan)} variants with {jobs} workers", flush=True)
    build_flash_kda_parallel(source, output, work, args.target, jobs, plan)
    print(f"Published {output} in {time.monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
