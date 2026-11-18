# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only A6 route features and opt-in, measured cost profiles.

No A5 coefficients or unmeasured A6 coefficients are installed. Profiles are
reviewed source assets, not files loaded from an environment variable. Until
paired device measurements are available, automatic routing is unchanged.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import statistics


INT64_MAX = 2**63 - 1
FEATURE_VERSION = 2
STAGED_FEATURES = (
    "constant",
    "stage1_peak_chunks",
    "stage1_peak_rows",
    "rounds",
    "stage2_peak_tasks",
    "stage2_peak_chunks_dv1",
    "stage2_peak_chunks_dv2",
    "stage2_peak_chunks_dv4",
    "stage2_peak_chunks_dv8",
)
PERSISTENT_FEATURES = (
    "constant",
    "chunks",
    "rows",
    "head_tasks",
    "pair_handoffs",
    "state_transfers",
)


@dataclass(frozen=True)
class StagedRound:
    batch_indices: tuple[int, ...]
    chunk_start: int
    chunk_counts: tuple[int, ...]
    group_num: int
    dv_splits: int
    stage1_ranges: tuple[tuple[int, int], ...]
    stage2_ranges: tuple[tuple[int, int], ...]
    stage1_core_rows: tuple[int, ...]
    stage2_core_chunks: tuple[int, ...]


@dataclass(frozen=True)
class RouteFeatures:
    lengths: tuple[int, ...]
    query_heads: int
    value_heads: int
    core_num: int
    group_limit: int
    workspace_slots: int
    core_chunks: tuple[int, ...]
    core_rows: tuple[int, ...]
    rounds: int
    core_tasks: tuple[int, ...]
    core_groups: tuple[int, ...]
    persistent_chunk_group: int
    persistent_dv_base: int
    staged_rounds: tuple[StagedRound, ...]

    def persistent_quantities(self):
        # Each head starts a new group. Summing chunks before rounding would
        # undercount both handoffs and GM state transfers for short heads.
        return tuple(
            dict(
                constant=1,
                chunks=chunks,
                rows=rows,
                head_tasks=tasks,
                pair_handoffs=2 * groups,
                state_transfers=2 * (groups - tasks),
            )
            for chunks, rows, tasks, groups in zip(
                self.core_chunks,
                self.core_rows,
                self.core_tasks,
                self.core_groups,
            )
            if tasks
        )

    def quantities(self):
        total_chunks = sum(self.core_chunks)
        value_rows = sum(self.core_rows)
        quantities = {
            "constant": 1,
            "parallel_chunks": (total_chunks + self.core_num - 1) // self.core_num,
            "rounds": self.rounds,
            "query_rows": sum(self.lengths) * self.query_heads,
            "value_rows": value_rows,
            "peak_chunks": max(self.core_chunks),
            "peak_rows": max(self.core_rows),
            "head_tasks": len(self.lengths) * self.value_heads,
            "stage1_peak_chunks": sum(
                max(end - start for start, end in r.stage1_ranges)
                for r in self.staged_rounds
            ),
            "stage1_peak_rows": sum(
                max(r.stage1_core_rows) for r in self.staged_rounds
            ),
            "stage2_peak_tasks": sum(
                max(end - start for start, end in r.stage2_ranges)
                for r in self.staged_rounds
            ),
        }
        for dv in (1, 2, 4, 8):
            quantities[f"stage2_peak_chunks_dv{dv}"] = sum(
                max(r.stage2_core_chunks)
                for r in self.staged_rounds
                if r.dv_splits == dv
            )
        return quantities

    def signature(self):
        # Exact measured workload coverage for the first calibration round.
        return (
            self.lengths,
            self.query_heads,
            self.value_heads,
            self.group_limit,
            self.workspace_slots,
            self.persistent_chunk_group,
            self.persistent_dv_base,
        )


def _ranges(costs, core_num, scheduler):
    ranges, core_costs = [0] * 128, [0] * 64
    scheduler._partition_costs(costs, len(costs), core_num, ranges, core_costs)
    return tuple(zip(ranges[: core_num * 2 : 2], ranges[1 : core_num * 2 : 2])), max(
        core_costs
    )


def _staged_round(
    lengths, batches, chunk_start, counts, group, heads, cores, launch_dv, scheduler
):
    rows = [
        min(64, lengths[b] - (chunk_start + chunk) * 64)
        for b, count in zip(batches, counts)
        for _ in range(heads)
        for chunk in range(count)
    ]
    stage1_ranges, _ = _ranges(
        [scheduler._stage1_task_cost(row) for row in rows], cores, scheduler
    )
    candidates = []
    for index in range(4):
        dv = 1 << index
        if launch_dv and dv != launch_dv:
            continue  # Host Python only; not emitted into AICPU.
        chunks = [count for count in counts for _ in range(heads * dv)]
        costs = [scheduler._stage2_task_cost(index, count) for count in chunks]
        ranges, peak = _ranges(costs, cores, scheduler)
        # Same tie order as the producer; these costs select the existing DV,
        # they are NOT A6 latency coefficients for choosing a template.
        candidates.append(((peak, sum(costs), len(costs), dv), ranges, chunks))
    key, stage2_ranges, chunks = min(candidates, key=lambda candidate: candidate[0])
    return StagedRound(
        tuple(batches),
        chunk_start,
        tuple(counts),
        group,
        key[-1],
        stage1_ranges,
        stage2_ranges,
        tuple(sum(rows[start:end]) for start, end in stage1_ranges),
        tuple(sum(chunks[start:end]) for start, end in stage2_ranges),
    )


def build_features(
    lengths,
    query_heads,
    value_heads,
    core_num,
    group_limit,
    workspace_slots=861,
    *,
    scheduler,
    persistent_chunk_group,
    persistent_dv_base,
):
    """Simulate A6 whole-head LPT assignment and staged grouping, not execution."""
    lengths = tuple(lengths)
    if (
        not lengths
        or min(lengths) <= 0
        or not 1 <= query_heads <= value_heads <= 96
        or value_heads % query_heads
        or not 1 <= core_num <= 64
        or group_limit < 1
        or workspace_slots < value_heads
        or persistent_chunk_group < 1
        or persistent_dv_base not in (16, 32, 64, 128)
    ):
        raise ValueError("invalid A6 route workload")
    chunks = tuple((length + 63) // 64 for length in lengths)
    order = sorted(range(len(lengths)), key=lambda batch: (-chunks[batch], batch))
    core_chunks, core_rows = [0] * core_num, [0] * core_num
    core_tasks, core_groups = [0] * core_num, [0] * core_num
    for batch in order:
        for _ in range(value_heads):
            core = min(range(core_num), key=core_chunks.__getitem__)
            core_chunks[core] += chunks[batch]
            core_rows[core] += lengths[batch]
            core_tasks[core] += 1
            core_groups[core] += (
                chunks[batch] + persistent_chunk_group - 1
            ) // persistent_chunk_group
    # Match metadata's per-set group selection; workspace slots count head-chunks.
    per_set = workspace_slots // value_heads
    rounds = []
    launch_dv = 0
    for start in range(0, len(chunks), per_set):
        remaining = list(chunks[start : start + per_set])
        chunk_start = 0
        while max(remaining) > 0:
            low, high = 1, min(group_limit, max(remaining), per_set)
            while low < high:
                middle = (low + high + 1) // 2
                if sum(min(count, middle) for count in remaining) <= per_set:
                    low = middle
                else:
                    high = middle - 1
            active = [index for index, count in enumerate(remaining) if count]
            record = _staged_round(
                lengths,
                [start + index for index in active],
                chunk_start,
                [min(remaining[index], low) for index in active],
                low,
                value_heads,
                core_num,
                launch_dv,
                scheduler,
            )
            rounds.append(record)
            launch_dv = record.dv_splits
            remaining = [max(0, count - low) for count in remaining]
            chunk_start += low
    return RouteFeatures(
        lengths,
        query_heads,
        value_heads,
        core_num,
        group_limit,
        workspace_slots,
        tuple(core_chunks),
        tuple(core_rows),
        len(rounds),
        tuple(core_tasks),
        tuple(core_groups),
        persistent_chunk_group,
        persistent_dv_base,
        tuple(rounds),
    )


def check_schedule(features, decoded):
    """Reject calibration data if Host simulation differs from actual metadata."""
    if decoded["core_num"] != features.core_num:
        raise ValueError("route features disagree with metadata core count")
    if decoded["template"] == "persistent":
        for core, tasks in enumerate(
            decoded["chunk_tasks_by_core"][: features.core_num]
        ):
            chunks = [(task["sequence_length"] + 63) // 64 for task in tasks]
            actual = (
                sum(chunks),
                sum(task["sequence_length"] for task in tasks),
                len(tasks),
                sum(
                    (count + features.persistent_chunk_group - 1)
                    // features.persistent_chunk_group
                    for count in chunks
                ),
            )
            expected = (
                features.core_chunks[core],
                features.core_rows[core],
                features.core_tasks[core],
                features.core_groups[core],
            )
            if actual != expected:
                raise ValueError(f"route features disagree with persistent core {core}")
    else:
        rounds = decoded["stage12_rounds"]
        if len(rounds) != features.rounds:
            raise ValueError("route features disagree with staged round count")
        for index, (predicted, actual) in enumerate(
            zip(features.staged_rounds, rounds)
        ):
            if (
                predicted.batch_indices != tuple(actual["batch_idx"])
                or predicted.chunk_counts != tuple(actual["chunk_num_per_group"])
                or any(
                    start != predicted.chunk_start
                    for start in actual["chunk_start_per_group"]
                )
                or predicted.group_num != actual["group_num"]
                or any(dv != predicted.dv_splits for dv in actual["dv_splits_num"])
                or predicted.stage1_ranges
                != tuple(map(tuple, actual["stage1_core_ranges"][: features.core_num]))
                or predicted.stage2_ranges
                != tuple(map(tuple, actual["stage2_core_ranges"][: features.core_num]))
            ):
                raise ValueError(f"route features disagree with staged round {index}")


def _cost(coefficients, names, quantities):
    """Integer picosecond model; reject invalid/overflowed profiles safely."""
    if len(coefficients) != len(names):
        return None
    result = 0
    for coefficient, name in zip(coefficients, names):
        quantity = quantities[name]
        if (
            type(coefficient) is not int
            or coefficient < 0
            or quantity < 0
            or quantity > INT64_MAX
            or (quantity and coefficient > (INT64_MAX - result) // quantity)
        ):
            return None
        result += coefficient * quantity
    return result if result > 0 else None


@dataclass(frozen=True)
class RouteProfile:
    sdk_fingerprint: str
    kernel_fingerprint: str
    core_num: int
    layout: str
    workloads: frozenset
    staged_coefficients: tuple[int, ...]
    persistent_coefficients: tuple[int, ...]
    # Training/recompute modes share the public producer: activation requires
    # separate validation of all three modes, not an inference-only fit.
    validated_modes: tuple[str, ...]
    margin_ppm: int = 30_000
    feature_version: int = FEATURE_VERSION


PROFILES: tuple[RouteProfile, ...] = ()


def choose_template(features, layout, environment, fallback, profiles=None):
    """Return 1/2 only within a reviewed profile's measured coverage."""
    quantities = features.quantities()
    for profile in PROFILES if profiles is None else profiles:
        if (
            profile.feature_version != FEATURE_VERSION
            or environment != (profile.sdk_fingerprint, profile.kernel_fingerprint)
            or profile.core_num != features.core_num
            or profile.layout != layout
            or features.signature() not in profile.workloads
            or set(profile.validated_modes) != {"inference", "recompute", "save_caches"}
            or not 0 <= profile.margin_ppm < 1_000_000
        ):
            continue
        staged = _cost(profile.staged_coefficients, STAGED_FEATURES, quantities)
        core_costs = [
            _cost(profile.persistent_coefficients, PERSISTENT_FEATURES, quantities)
            for quantities in features.persistent_quantities()
        ]
        if staged is None or any(cost is None for cost in core_costs):
            continue
        # Max of per-core weighted costs, not a sum of independent maxima
        # belonging to different cores. Includes group boundaries on that chain.
        persistent = max(core_costs)
        required_gain = staged * profile.margin_ppm // 1_000_000
        return 2 if persistent < staged - required_gain else 1
    return fallback


@lru_cache(None)
def source_fingerprint(folder):
    """Hash executing kernel/scheduler source, excluding calibration coefficients."""
    folder = Path(folder)
    digest = hashlib.sha256()
    for name in (
        "flash_kda.py",
        "flash_kda_metadata.py",
        "flash_kda_staged.py",
        "flash_kda_persistent.py",
        "flash_kda_validation.py",
    ):
        digest.update(name.encode())
        digest.update((folder / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@lru_cache(None)
def sdk_fingerprint(folder):
    """Hash installed SDK Python/native files; version 0.8.2 alone is ambiguous."""
    folder = Path(folder)
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".so", ".pyd"}:
            digest.update(path.relative_to(folder).as_posix().encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def summarize_samples(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("kernel durations must be finite and positive")
    ordered = sorted(values)
    mean = sum(values) / len(values)
    return {
        "mean_us": mean,
        "p50_us": statistics.median(ordered),
        "p95_us": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "cv": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        / mean,
    }
