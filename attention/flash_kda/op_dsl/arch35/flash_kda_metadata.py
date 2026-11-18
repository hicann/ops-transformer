# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# 保持参数分组紧凑，与原有 FlashKDA 样例风格一致。

"""Host 选择计算模板，AICPU 生成 FlashKDA 前向调度元数据。"""

import os
import tempfile
import threading
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Optional

from cannbotdsl import get_platform_info
from cannbotdsl.aicpu import (
    I32,
    I64,
    U32,
    gm_in,
    gm_out,
    aicpu_kernel,
    current_raw_stream,
)
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
import torch

from .flash_kda_validation import (
    CHUNK_SIZE, SUPPORTED_HEAD_DIM, MAX_VALUE_HEADS, WORKSPACE_SLOTS, MAX_CHUNK_GROUP, HEADER_WORDS, INT32_MAX, STATUS_OK,
    SequenceSpec, require, validate_layout, validate_sizes, validate_length_specs,
    read_lengths, validate_schedule,
)

__all__ = [
    "HEADER_TEMPLATE",
    "HEADER_CHUNK_TASK_NUM",
    "TEMPLATE_AUTO",
    "TEMPLATE_STAGED",
    "TEMPLATE_PERSISTENT",
    "CHUNK_TASK_WORDS",
    "CHUNK_TASK_BATCH",
    "CHUNK_TASK_STORAGE_BATCH",
    "CHUNK_TASK_TOKEN_START",
    "CHUNK_TASK_LENGTH",
    "CHUNK_TASK_VALUE_HEAD",
    "CHUNK_TASK_NEXT",
    "ABI_VERSION",
    "FlashKDAFwdMetadataArgs",
    "HEADER_CORE_NUM",
    "HEADER_STAGE12_ROUND_NUM",
    "HEADER_STATUS",
    "HEADER_WORDS",
    "MAGIC",
    "MAX_AIC_CORES",
    "STATUS_BAD_CORE_NUM",
    "STATUS_BAD_CU_START",
    "STATUS_BAD_FLAGS",
    "STATUS_BAD_LENGTH",
    "STATUS_BAD_METADATA_CAPACITY",
    "STATUS_BAD_PACKED_END",
    "STATUS_BAD_PADDED_LENGTH",
    "STATUS_BAD_SHAPE",
    "STATUS_BAD_WORKSPACE_SLOTS",
    "STATUS_OK",
    "WORKSPACE_SLOTS",
    "clear_metadata_caches",
    "decode_metadata",
    "flash_kda_fwd_metadata",
    "flash_kda_fwd_metadata_kernel",
    "metadata_capacity_upper_bound",
    "round_record_words",
]


os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


MAGIC = 0x4B444132
ABI_VERSION = 4
MAX_AIC_CORES = 32
ROUND_FIXED_WORDS = 133
ROUND_WORDS_PER_BATCH = 9

HEADER_MAGIC = 0
HEADER_ABI_VERSION = 1
HEADER_STATUS = 2
HEADER_CORE_NUM = 3
HEADER_STAGE12_ROUND_NUM = 4
HEADER_BATCH = 5
HEADER_VALUE_HEADS = 6
HEADER_HAS_CU = 7
HEADER_WORKSPACE_SLOTS = 8
HEADER_TEMPLATE = 9
HEADER_CHUNK_TASK_NUM = 10

TEMPLATE_AUTO = 0
TEMPLATE_STAGED = 1
TEMPLATE_PERSISTENT = 2
CHUNK_TASK_WORDS = 6
CHUNK_TASK_BATCH = 0
CHUNK_TASK_STORAGE_BATCH = 1
CHUNK_TASK_TOKEN_START = 2
CHUNK_TASK_LENGTH = 3
CHUNK_TASK_VALUE_HEAD = 4
CHUNK_TASK_NEXT = 5


STATUS_BAD_CORE_NUM = 1
STATUS_BAD_WORKSPACE_SLOTS = 2
STATUS_BAD_FLAGS = 3
STATUS_BAD_SHAPE = 4
STATUS_BAD_CU_START = 5
STATUS_BAD_LENGTH = 6
STATUS_BAD_PADDED_LENGTH = 7
STATUS_BAD_PACKED_END = 8
STATUS_BAD_METADATA_CAPACITY = 9

# 单批次 32 头长序列训练的调优分组，保持已有调度行为。
SHORT_TRAIN_CHUNK_GROUP = 13
SHORT_TRAIN_MIN_LENGTH = 8192
SHORT_TRAIN_HEADS = 32

# 调度模型使用定点系数，先限制乘法范围以避免 INT64 溢出。
INT64_MAX = 2**63 - 1
ROUTE_COST_SCALE = 64
MAX_SCALED_QUANTITY = INT64_MAX // ROUTE_COST_SCALE
ROUTE_PARAMETER_WORDS = 16
ROUTE_SCHEMA_INDEX = 13
ROUTE_MARGIN_INDEX = 14
PARTS_PER_MILLION = 1_000_000
PICOSECONDS_PER_NANOSECOND = 1000
FULL_CHUNK_COST_NS = 4930
CALIBRATED_MIN_CHUNKS = 64
CALIBRATED_MAX_CHUNKS = 256
CALIBRATED_GROUP_LIMIT = 16

MAX_ACTIVE_BATCHES = WORKSPACE_SLOTS
MAX_STAGE1_TASKS = WORKSPACE_SLOTS
MAX_STAGE2_TASKS = WORKSPACE_SLOTS * 8


def _interpolate_stage1_task_cost(
    valid_row_num,
    left_row_num,
    left_cost,
    right_row_num,
    right_cost,
):
    # 拟合系数单位为皮秒，调度成本单位为纳秒。
    row_span = right_row_num - left_row_num
    numerator = left_cost * row_span + (valid_row_num - left_row_num) * (
        right_cost - left_cost
    )
    denominator = row_span * PICOSECONDS_PER_NANOSECOND
    return (numerator + denominator // 2) // denominator


def _stage1_task_cost(valid_row_num):
    cost = 0
    if valid_row_num > 0:
        if valid_row_num <= 16:
            cost = _interpolate_stage1_task_cost(valid_row_num, 1, 4892380, 16, 4920438)
        elif valid_row_num <= 32:
            cost = _interpolate_stage1_task_cost(valid_row_num, 16, 4920438, 32, 4970718)
        elif valid_row_num <= 48:
            cost = _interpolate_stage1_task_cost(valid_row_num, 32, 4970718, 48, 4992347)
        elif valid_row_num <= 63:
            cost = _interpolate_stage1_task_cost(valid_row_num, 48, 4992347, 63, 4952972)
        else:
            cost = FULL_CHUNK_COST_NS
    return cost


def _stage2_task_cost(candidate_idx, chunk_num):
    item_init = 352
    chunk_cost = 1839
    if candidate_idx == 1:
        item_init = 133
        chunk_cost = 1186
    elif candidate_idx == 2:
        item_init = 19
        chunk_cost = 884
    elif candidate_idx == 3:
        item_init = 65
        chunk_cost = 729
    return item_init + chunk_num * chunk_cost


def round_record_words(active_batch_num: int) -> int:
    """返回一条变长记录占用的字数。"""

    active_batch_num = int(active_batch_num)
    if active_batch_num <= 0:
        raise ValueError("active_batch_num must be positive")
    return ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH * active_batch_num


def metadata_capacity_upper_bound(
    batch: int,
    storage_length: int,
    *,
    is_packed: bool,
    value_heads: int = MAX_VALUE_HEADS,
) -> int:
    """返回覆盖所有合法轮次划分的安全容量上界。"""

    batch = int(batch)
    storage_length = int(storage_length)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if storage_length <= 0:
        raise ValueError("storage_length must be positive")
    storage_chunk_num = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if is_packed:
        chunk_entries = storage_chunk_num + batch - 1
    else:
        chunk_entries = batch * storage_chunk_num
    if not 1 <= int(value_heads) <= MAX_VALUE_HEADS:
        raise ValueError("value_heads must be in [1, 96]")
    # 每个可能的 chunk 条目预留一个目录字和一条单批次记录。
    group_capacity = (
        HEADER_WORDS
        + 1
        + (1 + ROUND_FIXED_WORDS + ROUND_WORDS_PER_BATCH) * chunk_entries
    )
    # chunk 调度器在任务记录后使用 B 个临时批次索引。
    chunk_capacity = (
        HEADER_WORDS
        + MAX_AIC_CORES
        + CHUNK_TASK_WORDS * batch * int(value_heads)
        + batch
    )
    return max(group_capacity, chunk_capacity)


class FlashKDAFwdMetadataArgs:
    cu_seqlens: gm_in(I32)
    metadata: gm_out(U32)
    metadata_word_capacity: U32
    batch: U32
    physical_batches: U32
    storage_length: U32
    value_heads: U32
    has_cu: U32
    is_packed: U32
    core_num: U32
    workspace_slots: U32
    template_kind: U32
    max_chunk_group: U32


def _select_group_num(
    remaining_chunk_num,
    active_batch_num,
    value_heads,
    workspace_slots,
    max_chunk_group,
):
    max_remaining_chunk_num = 0
    for active_idx in range(0, active_batch_num):
        if remaining_chunk_num[active_idx] > max_remaining_chunk_num:
            max_remaining_chunk_num = remaining_chunk_num[active_idx]

    selected_group_num = 1
    lower_group_num = 1
    upper_group_num = max_remaining_chunk_num
    if upper_group_num > max_chunk_group:
        upper_group_num = max_chunk_group
    workspace_group_limit = workspace_slots // value_heads
    if upper_group_num > workspace_group_limit:
        upper_group_num = workspace_group_limit
    # 槽位用量随组大小单调增长。选取最大可行分组，减少长序列跨越全局阶段屏障的次数。
    while lower_group_num <= upper_group_num:
        candidate_group_num = (lower_group_num + upper_group_num) // 2
        slot_num = 0
        for active_idx in range(0, active_batch_num):
            current_chunk_num = remaining_chunk_num[active_idx]
            if current_chunk_num > candidate_group_num:
                current_chunk_num = candidate_group_num
            slot_num += value_heads * current_chunk_num
        if slot_num <= workspace_slots:
            selected_group_num = candidate_group_num
            lower_group_num = candidate_group_num + 1
        else:
            upper_group_num = candidate_group_num - 1
    return selected_group_num


def _partition_costs(
    task_costs,
    task_num,
    core_num,
    core_ranges,
    core_costs,
):
    for core_idx in range(0, MAX_AIC_CORES):
        core_ranges[core_idx * 2] = 0
        core_ranges[core_idx * 2 + 1] = 0
        core_costs[core_idx] = 0

    if task_num > 0:
        active_core_num = task_num
        if active_core_num > core_num:
            active_core_num = core_num
        remaining_cost = 0
        for task_idx in range(0, task_num):
            remaining_cost += task_costs[task_idx]

        start = 0
        for core_idx in range(0, active_core_num):
            remaining_core_num = active_core_num - core_idx
            if remaining_core_num == 1:
                end = task_num
            else:
                max_end = task_num - (remaining_core_num - 1)
                end = start + 1
                segment_cost = task_costs[start]
                while end < max_end:
                    next_cost = segment_cost + task_costs[end]
                    next_error = abs(next_cost * remaining_core_num - remaining_cost)
                    current_error = abs(segment_cost * remaining_core_num - remaining_cost)
                    if next_error <= current_error:
                        segment_cost = next_cost
                        end += 1
                    else:
                        break

            segment_cost = 0
            for task_idx in range(start, end):
                segment_cost += task_costs[task_idx]
            core_ranges[core_idx * 2] = start
            core_ranges[core_idx * 2 + 1] = end
            core_costs[core_idx] = segment_cost
            remaining_cost -= segment_cost
            start = end


def _chunk_batch_precedes(cu_seqlens, left_batch, right_batch):
    left_length = cu_seqlens[left_batch + 1] - cu_seqlens[left_batch]
    right_length = cu_seqlens[right_batch + 1] - cu_seqlens[right_batch]
    left_chunks = (left_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    right_chunks = (right_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    precedes = 0
    if left_chunks > right_chunks:
        precedes = 1
    elif left_chunks == right_chunks and left_batch < right_batch:
        precedes = 1
    return precedes


def _chunk_heap_sift_down(cu_seqlens, metadata, heap_base, heap_num, root_idx):
    # 最大堆先按 chunk 数排序，再按原始批次索引排序。
    while root_idx * 2 + 1 < heap_num:
        child_idx = root_idx * 2 + 1
        if child_idx + 1 < heap_num:
            if (
                _chunk_batch_precedes(cu_seqlens, metadata[heap_base + child_idx + 1], metadata[heap_base + child_idx])
                == 1
            ):
                child_idx += 1
        root_batch = metadata[heap_base + root_idx]
        child_batch = metadata[heap_base + child_idx]
        if _chunk_batch_precedes(cu_seqlens, child_batch, root_batch) == 0:
            break
        metadata[heap_base + root_idx] = child_batch
        metadata[heap_base + child_idx] = root_batch
        root_idx = child_idx




def _training_group_limit(schedule_args, max_chunks):
    requested = schedule_args.max_chunk_group
    if schedule_args.calibration_enabled == 0:
        return requested
    if (schedule_args.layout_kind == 1 and schedule_args.core_num == MAX_AIC_CORES
        and schedule_args.batch == 1 and schedule_args.value_heads > 32
        and schedule_args.value_heads <= MAX_VALUE_HEADS and max_chunks >= CALIBRATED_MIN_CHUNKS and max_chunks <= CALIBRATED_MAX_CHUNKS):
        return min(requested, CALIBRATED_GROUP_LIMIT)
    return requested



def _route_add_cost(time, cost, quantity):
    if time < 0 or cost < 0 or quantity < 0:
        return -1
    if quantity > 0 and cost > (INT64_MAX - time) // quantity:
        return -1
    return time + cost * quantity


def _route_rounds(schedule_args, max_group, remaining_chunks):
    rounds = 0
    batch_start = 0
    batches_per_set = schedule_args.workspace_slots // schedule_args.value_heads
    while batch_start < schedule_args.batch:
        batch_count = min(batches_per_set, schedule_args.batch - batch_start)
        longest = 0
        for local_batch in range(batch_count):
            batch = batch_start + local_batch
            length = schedule_args.storage_length
            if schedule_args.has_cu == 1:
                length = schedule_args.cu_seqlens[batch + 1] - schedule_args.cu_seqlens[batch]
            chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
            remaining_chunks[local_batch] = chunks
            longest = max(longest, chunks)
        while longest > 0:
            low = 1
            high = min(max_group, min(longest, batches_per_set))
            while low < high:
                middle = (low + high + 1) // 2
                chunk_count = 0
                for local_batch in range(batch_count):
                    chunk_count += min(remaining_chunks[local_batch], middle)
                if chunk_count <= schedule_args.workspace_slots // schedule_args.value_heads:
                    low = middle
                else:
                    high = middle - 1
            longest = 0
            for local_batch in range(batch_count):
                chunks = max(0, remaining_chunks[local_batch] - low)
                remaining_chunks[local_batch] = chunks
                longest = max(longest, chunks)
            rounds += 1
        batch_start += batch_count
    return rounds


def _route_persistent_loads(schedule_args, uniform_chunks, core_chunks, core_rows):
    # 使用与执行调度相同的堆及按 chunk 数排序的 LPT 顺序。
    task_count = schedule_args.batch * schedule_args.value_heads
    heap_base = HEADER_WORDS + MAX_AIC_CORES + CHUNK_TASK_WORDS * task_count
    if heap_base + schedule_args.batch > schedule_args.metadata_word_capacity:
        return -1
    heap_count = schedule_args.batch
    if uniform_chunks == 0:
        for batch in range(schedule_args.batch):
            schedule_args.metadata[heap_base + batch] = batch
        root = heap_count // 2
        while root > 0:
            root -= 1
            _chunk_heap_sift_down(
                schedule_args.cu_seqlens, schedule_args.metadata, heap_base, heap_count, root
            )
    peak = 0
    for scheduled_batch in range(schedule_args.batch):
        batch = scheduled_batch
        if uniform_chunks == 0:
            batch = schedule_args.metadata[heap_base]
            heap_count -= 1
            if heap_count > 0:
                schedule_args.metadata[heap_base] = schedule_args.metadata[heap_base + heap_count]
                _chunk_heap_sift_down(
                    schedule_args.cu_seqlens, schedule_args.metadata, heap_base, heap_count, 0
                )
        length = schedule_args.storage_length
        if schedule_args.has_cu == 1:
            length = schedule_args.cu_seqlens[batch + 1] - schedule_args.cu_seqlens[batch]
        chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        for head in range(schedule_args.value_heads):
            selected_core = 0
            for core in range(1, schedule_args.core_num):
                if core_chunks[core] < core_chunks[selected_core]:
                    selected_core = core
            core_chunks[selected_core] += chunks
            core_rows[selected_core] += length
            peak = max(peak, core_chunks[selected_core])
    return peak


def _select_route_from_model(
    schedule_args, staged_peak, total_chunks, total_rows, query_rows,
    persistent_peak, core_rows, max_group, remaining_chunks,
):
    # 非负成本单位为飞秒，两种时间均乘以 64 后比较。
    if staged_peak < 1 or total_chunks < 1 or total_rows < 1 or query_rows < 1 or persistent_peak < 1:
        return TEMPLATE_STAGED
    parallel_chunks = (total_chunks - 1) // schedule_args.core_num + 1
    head_count = schedule_args.batch * schedule_args.value_heads
    rounds = _route_rounds(schedule_args, max_group, remaining_chunks)
    if (parallel_chunks > MAX_SCALED_QUANTITY or staged_peak > MAX_SCALED_QUANTITY
        or persistent_peak > MAX_SCALED_QUANTITY or rounds > MAX_SCALED_QUANTITY
        or head_count > MAX_SCALED_QUANTITY):
        return TEMPLATE_STAGED
    base = schedule_args.layout_kind * ROUTE_PARAMETER_WORDS
    schema = schedule_args.route_coefficients[base + ROUTE_SCHEMA_INDEX]
    if schema != 6 and schema != 7:
        return TEMPLATE_STAGED
    for index in range(ROUTE_SCHEMA_INDEX):
        if schedule_args.route_coefficients[base + index] < 0:
            return TEMPLATE_STAGED
    staged = 0
    for index in range(5):
        quantity = ROUTE_COST_SCALE
        if index == 1:
            quantity = parallel_chunks * ROUTE_COST_SCALE
        elif index == 2:
            quantity = staged_peak * ROUTE_COST_SCALE
        elif index == 3:
            quantity = rounds * ROUTE_COST_SCALE
        elif index == 4:
            quantity = query_rows
        staged = _route_add_cost(staged, schedule_args.route_coefficients[base + index], quantity)
        if staged < 0:
            return TEMPLATE_STAGED
    compute = _route_add_cost(0, schedule_args.route_coefficients[base + 5], parallel_chunks * ROUTE_COST_SCALE)
    memory = _route_add_cost(0, schedule_args.route_coefficients[base + 6], total_rows)
    if compute < 0 or memory < 0:
        return TEMPLATE_STAGED
    staged = _route_add_cost(staged, max(compute, memory), 1)
    if staged < 0:
        return TEMPLATE_STAGED
    persistent = 0
    for index in range(7, 11):
        quantity = ROUTE_COST_SCALE
        if index == 8:
            quantity = persistent_peak * ROUTE_COST_SCALE
        elif index == 9:
            quantity = head_count * ROUTE_COST_SCALE
        elif index == 10:
            quantity = query_rows
        persistent = _route_add_cost(persistent, schedule_args.route_coefficients[base + index], quantity)
        if persistent < 0:
            return TEMPLATE_STAGED
    previous_rows = 0
    for phase in range(schedule_args.core_num):
        next_rows = 0
        active_cores = 0
        for core in range(schedule_args.core_num):
            rows = core_rows[core]
            if rows > previous_rows:
                active_cores += 1
                if next_rows == 0 or rows < next_rows:
                    next_rows = rows
        if next_rows == 0:
            break
        memory = _route_add_cost(0, schedule_args.route_coefficients[base + 12], active_cores)
        if memory < 0:
            return TEMPLATE_STAGED
        cost = max(schedule_args.route_coefficients[base + 11], memory)
        persistent = _route_add_cost(persistent, cost, next_rows - previous_rows)
        if persistent < 0:
            return TEMPLATE_STAGED
        previous_rows = next_rows
    margin = schedule_args.route_coefficients[base + ROUTE_MARGIN_INDEX]
    if margin < 0 or margin >= PARTS_PER_MILLION:
        return TEMPLATE_STAGED
    required_gain = (staged // PARTS_PER_MILLION) * margin + (staged % PARTS_PER_MILLION) * margin // PARTS_PER_MILLION
    if persistent < staged - required_gain:
        return TEMPLATE_PERSISTENT
    return TEMPLATE_STAGED



def _select_template_on_host(schedule_args: SimpleNamespace) -> tuple[int, int]:
    """在 Host 执行原 AICPU 模板选择逻辑。

    schedule_args 包含序列长度前缀和、Q/K 与 V 头数、布局和核数、
    模板请求、成本模型系数、分组与工作区约束，以及 Host 临时存储。

    成本公式、定点保护、负载分配顺序和自动/强制模板语义保持不变。
    返回明确的模板编号和交给设备调度器的最终分组上限。
    """
    status = STATUS_OK
    route_core_costs = [0] * MAX_AIC_CORES
    route_lpt_costs = [0] * MAX_AIC_CORES
    route_valid_rows = [0] * MAX_AIC_CORES
    remaining_chunk_num = [0] * MAX_ACTIVE_BATCHES
    # 发布选路结果或任务前，检查所有实际序列长度。
    uniform_chunk_num = 1
    first_chunk_num = 0
    route_max_chunk_num = 0
    route_total_chunk_num = 0
    route_peak_chunk_num = 0
    route_total_rows = 0
    route_query_rows = 0
    evaluate_route = 0
    if schedule_args.template_kind == TEMPLATE_AUTO and schedule_args.calibration_enabled == 1:
        evaluate_route = 1
    for batch_idx in range(0, schedule_args.batch):
        valid_seq_len = schedule_args.storage_length
        if schedule_args.has_cu == 1:
            valid_seq_len = schedule_args.cu_seqlens[batch_idx + 1] - schedule_args.cu_seqlens[batch_idx]
        if valid_seq_len <= 0:
            status = STATUS_BAD_LENGTH
        elif schedule_args.is_packed == 0 and valid_seq_len > schedule_args.storage_length:
            status = STATUS_BAD_PADDED_LENGTH
        else:
            current_chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            if batch_idx == 0:
                first_chunk_num = current_chunk_num
            elif current_chunk_num != first_chunk_num:
                uniform_chunk_num = 0
            if current_chunk_num > route_max_chunk_num:
                route_max_chunk_num = current_chunk_num
            if evaluate_route == 1:
                if current_chunk_num > INT64_MAX // schedule_args.value_heads:
                    evaluate_route = 0
                else:
                    batch_chunk_num = schedule_args.value_heads * current_chunk_num
                    if route_total_chunk_num > INT64_MAX - batch_chunk_num:
                        evaluate_route = 0
                    else:
                        route_total_chunk_num += batch_chunk_num
                        route_total_rows = _route_add_cost(route_total_rows, schedule_args.value_heads, valid_seq_len)
                        route_query_rows = _route_add_cost(route_query_rows, schedule_args.query_heads, valid_seq_len)
                        if route_total_rows < 0 or route_query_rows < 0:
                            evaluate_route = 0
                        # 每个核的成本不超过已检查的总成本。
                        for value_head_idx in range(0, schedule_args.value_heads):
                            cyclic_core = (
                                batch_idx * schedule_args.value_heads + value_head_idx
                            ) % schedule_args.core_num
                            route_core_costs[cyclic_core] += current_chunk_num
    if status == STATUS_OK and schedule_args.is_packed == 1:
        if schedule_args.cu_seqlens[schedule_args.batch] != schedule_args.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        raise ValueError(f"flash_kda_fwd_metadata produced status {status}")
    selected_template = schedule_args.template_kind
    if selected_template == TEMPLATE_AUTO:
        selected_template = TEMPLATE_STAGED
        if evaluate_route == 1:
            for core_idx in range(0, schedule_args.core_num):
                if route_core_costs[core_idx] > route_peak_chunk_num:
                    route_peak_chunk_num = route_core_costs[core_idx]
            persistent_peak = _route_persistent_loads(
                schedule_args, uniform_chunk_num, route_lpt_costs, route_valid_rows
            )
            selected_template = _select_route_from_model(
                schedule_args, route_peak_chunk_num, route_total_chunk_num, route_total_rows,
                route_query_rows, persistent_peak, route_valid_rows,
                _training_group_limit(schedule_args, route_max_chunk_num), remaining_chunk_num,
            )
            # 短多序列训练保留直接 U 路径的缓存。
            if schedule_args.has_cu == 1 and schedule_args.batch > 1 and route_max_chunk_num <= 2:
                selected_template = TEMPLATE_PERSISTENT
    max_chunk_group = schedule_args.max_chunk_group
    if schedule_args.template_kind == TEMPLATE_AUTO:
        max_chunk_group = _training_group_limit(schedule_args, route_max_chunk_num)
    return selected_template, max_chunk_group


@aicpu_kernel
def flash_kda_fwd_metadata_kernel(metadata_args: FlashKDAFwdMetadataArgs):
    valid_seq_len_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    token_start_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    tail_row_num_by_batch = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    active_local_batch_idx = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    remaining_chunk_num = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_num_per_group = zeros(I64, MAX_ACTIVE_BATCHES)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_task_costs = zeros(I64, MAX_STAGE1_TASKS)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    stage1_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    stage2_task_costs = zeros(I64, MAX_STAGE2_TASKS)  # noqa: F821 - AICPU 编译器内建函数。
    candidate_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    candidate_stage2_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    selected_stage2_core_ranges = zeros(I64, MAX_AIC_CORES * 2)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_core_costs = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。
    chunk_core_tails = zeros(I64, MAX_AIC_CORES)  # noqa: F821 - AICPU 编译器内建函数。

    if metadata_args.metadata_word_capacity < HEADER_WORDS:
        # AICPU 将非零内核返回值视为任务异常（ACL 507005）。
        # 诊断码写入 metadata 头部，不作为内核返回值。
        return STATUS_OK

    metadata_args.metadata[HEADER_MAGIC] = MAGIC
    metadata_args.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    metadata_args.metadata[HEADER_STATUS] = STATUS_OK
    metadata_args.metadata[HEADER_CORE_NUM] = metadata_args.core_num
    metadata_args.metadata[HEADER_STAGE12_ROUND_NUM] = 0
    metadata_args.metadata[HEADER_BATCH] = metadata_args.batch
    metadata_args.metadata[HEADER_VALUE_HEADS] = metadata_args.value_heads
    metadata_args.metadata[HEADER_HAS_CU] = metadata_args.has_cu
    metadata_args.metadata[HEADER_WORKSPACE_SLOTS] = metadata_args.workspace_slots
    metadata_args.metadata[HEADER_TEMPLATE] = TEMPLATE_AUTO
    metadata_args.metadata[HEADER_CHUNK_TASK_NUM] = 0
    if metadata_args.metadata_word_capacity > HEADER_WORDS:
        metadata_args.metadata[HEADER_WORDS] = HEADER_WORDS + 1

    status = STATUS_OK
    if metadata_args.core_num < 1 or metadata_args.core_num > MAX_AIC_CORES:
        status = STATUS_BAD_CORE_NUM
    elif (
        metadata_args.template_kind != TEMPLATE_STAGED
        and metadata_args.template_kind != TEMPLATE_PERSISTENT
    ):
        status = STATUS_BAD_FLAGS
    elif metadata_args.workspace_slots != WORKSPACE_SLOTS:
        status = STATUS_BAD_WORKSPACE_SLOTS
    elif metadata_args.max_chunk_group < 1 or metadata_args.max_chunk_group > MAX_CHUNK_GROUP:
        status = STATUS_BAD_FLAGS
    elif not (metadata_args.has_cu == 0 or metadata_args.has_cu == 1):
        status = STATUS_BAD_FLAGS
    elif not (metadata_args.is_packed == 0 or metadata_args.is_packed == 1):
        status = STATUS_BAD_FLAGS
    elif metadata_args.is_packed == 1 and metadata_args.has_cu == 0:
        status = STATUS_BAD_FLAGS
    elif metadata_args.batch < 1 or metadata_args.storage_length < 1 or metadata_args.value_heads < 1:
        status = STATUS_BAD_SHAPE
    elif metadata_args.value_heads > MAX_VALUE_HEADS:
        status = STATUS_BAD_SHAPE
    elif metadata_args.is_packed == 1 and metadata_args.physical_batches != 1:
        status = STATUS_BAD_SHAPE
    elif metadata_args.is_packed == 0 and metadata_args.physical_batches != metadata_args.batch:
        status = STATUS_BAD_SHAPE
    if status != STATUS_OK:
        metadata_args.metadata[HEADER_STATUS] = status
        return STATUS_OK

    if metadata_args.has_cu == 1:
        if metadata_args.cu_seqlens[0] != 0:
            metadata_args.metadata[HEADER_STATUS] = STATUS_BAD_CU_START
            return STATUS_OK

    # 选路已在 Host 完成；仍在设备发布调度前检查所有实际序列长度。
    uniform_chunk_num = 1
    first_chunk_num = 0
    for batch_idx in range(0, metadata_args.batch):
        valid_seq_len = metadata_args.storage_length
        if metadata_args.has_cu == 1:
            valid_seq_len = metadata_args.cu_seqlens[batch_idx + 1] - metadata_args.cu_seqlens[batch_idx]
        if valid_seq_len <= 0:
            status = STATUS_BAD_LENGTH
        elif metadata_args.is_packed == 0 and valid_seq_len > metadata_args.storage_length:
            status = STATUS_BAD_PADDED_LENGTH
        else:
            current_chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            if batch_idx == 0:
                first_chunk_num = current_chunk_num
            elif current_chunk_num != first_chunk_num:
                uniform_chunk_num = 0
    if status == STATUS_OK and metadata_args.is_packed == 1:
        if metadata_args.cu_seqlens[metadata_args.batch] != metadata_args.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        metadata_args.metadata[HEADER_STATUS] = status
        return STATUS_OK
    selected_template = metadata_args.template_kind
    metadata_args.metadata[HEADER_TEMPLATE] = selected_template
    if selected_template == TEMPLATE_PERSISTENT:
        task_num = metadata_args.batch * metadata_args.value_heads
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        heap_base = used_words
        if used_words + metadata_args.batch > metadata_args.metadata_word_capacity:
            metadata_args.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
            return STATUS_OK
        for core_idx in range(0, MAX_AIC_CORES):
            metadata_args.metadata[HEADER_WORDS + core_idx] = 0
        # 等长序列保持原始批次和头顺序；变长批次采用 O(B log B) 的 LPT 调度。
        # 使用 B 个 GM 临时字，避免固定本地数组和反复扫描全部批次。
        heap_num = metadata_args.batch
        if uniform_chunk_num == 0:
            for batch_idx in range(0, metadata_args.batch):
                metadata_args.metadata[heap_base + batch_idx] = batch_idx
            heap_root = metadata_args.batch // 2
            while heap_root > 0:
                heap_root -= 1
                _chunk_heap_sift_down(metadata_args.cu_seqlens, metadata_args.metadata, heap_base, heap_num, heap_root)
        task_cursor = task_base
        for scheduled_batch_idx in range(0, metadata_args.batch):
            batch_idx = scheduled_batch_idx
            if uniform_chunk_num == 0:
                batch_idx = metadata_args.metadata[heap_base]
                heap_num -= 1
                if heap_num > 0:
                    metadata_args.metadata[heap_base] = metadata_args.metadata[heap_base + heap_num]
                    _chunk_heap_sift_down(metadata_args.cu_seqlens, metadata_args.metadata, heap_base, heap_num, 0)
            storage_batch_idx = batch_idx
            token_start = 0
            valid_seq_len = metadata_args.storage_length
            if metadata_args.has_cu == 1:
                token_start = metadata_args.cu_seqlens[batch_idx]
                valid_seq_len = metadata_args.cu_seqlens[batch_idx + 1] - token_start
            if metadata_args.is_packed == 1:
                storage_batch_idx = 0
            else:
                token_start = 0
            chunk_num = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            for value_head_idx in range(0, metadata_args.value_heads):
                selected_core = 0
                for core_idx in range(1, metadata_args.core_num):
                    if chunk_core_costs[core_idx] < chunk_core_costs[selected_core]:
                        selected_core = core_idx
                metadata_args.metadata[task_cursor + CHUNK_TASK_BATCH] = batch_idx
                metadata_args.metadata[task_cursor + CHUNK_TASK_STORAGE_BATCH] = storage_batch_idx
                metadata_args.metadata[task_cursor + CHUNK_TASK_TOKEN_START] = token_start
                metadata_args.metadata[task_cursor + CHUNK_TASK_LENGTH] = valid_seq_len
                metadata_args.metadata[task_cursor + CHUNK_TASK_VALUE_HEAD] = value_head_idx
                metadata_args.metadata[task_cursor + CHUNK_TASK_NEXT] = 0
                previous_task = chunk_core_tails[selected_core]
                if previous_task == 0:
                    metadata_args.metadata[HEADER_WORDS + selected_core] = task_cursor
                else:
                    metadata_args.metadata[previous_task + CHUNK_TASK_NEXT] = task_cursor
                chunk_core_tails[selected_core] = task_cursor
                chunk_core_costs[selected_core] += chunk_num
                task_cursor += CHUNK_TASK_WORDS
        metadata_args.metadata[HEADER_CHUNK_TASK_NUM] = task_num
        return STATUS_OK

    max_chunk_group = metadata_args.max_chunk_group
    batch_num_per_set = metadata_args.workspace_slots // metadata_args.value_heads
    stage12_round_num = 0
    total_active_batch_entries = 0
    batch_start = 0
    while batch_start < metadata_args.batch:
        batch_num_in_set = metadata_args.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = metadata_args.storage_length
            if metadata_args.has_cu == 1:
                token_start = metadata_args.cu_seqlens[batch_idx]
                token_end = metadata_args.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            if valid_seq_len <= 0:
                status = STATUS_BAD_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            elif metadata_args.is_packed == 0 and valid_seq_len > metadata_args.storage_length:
                status = STATUS_BAD_PADDED_LENGTH
                chunk_num_by_batch[local_batch_idx] = 0
                tail_row_num_by_batch[local_batch_idx] = 0
            else:
                chunk_num_by_batch[local_batch_idx] = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
                tail_row_num_by_batch[local_batch_idx] = ((valid_seq_len - 1) % CHUNK_SIZE) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break
            group_num = _select_group_num(
                remaining_chunk_num, active_batch_num, metadata_args.value_heads,
                metadata_args.workspace_slots, max_chunk_group,
            )
            stage12_round_num += 1
            total_active_batch_entries += active_batch_num
            chunk_start += group_num
        batch_start += batch_num_in_set

    if status == STATUS_OK and metadata_args.is_packed == 1:
        if metadata_args.cu_seqlens[metadata_args.batch] != metadata_args.storage_length:
            status = STATUS_BAD_PACKED_END
    if status != STATUS_OK:
        metadata_args.metadata[HEADER_STATUS] = status
        return STATUS_OK

    record_begin = HEADER_WORDS + stage12_round_num + 1
    used_words = (
        record_begin + ROUND_FIXED_WORDS * stage12_round_num
        + ROUND_WORDS_PER_BATCH * total_active_batch_entries
    )
    if used_words > metadata_args.metadata_word_capacity:
        metadata_args.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_OK

    metadata_args.metadata[HEADER_STAGE12_ROUND_NUM] = stage12_round_num
    record_cursor = record_begin
    global_round_idx = 0
    batch_start = 0
    while batch_start < metadata_args.batch:
        batch_num_in_set = metadata_args.batch - batch_start
        if batch_num_in_set > batch_num_per_set:
            batch_num_in_set = batch_num_per_set
        for local_batch_idx in range(0, batch_num_in_set):
            batch_idx = batch_start + local_batch_idx
            token_start = 0
            valid_seq_len = metadata_args.storage_length
            if metadata_args.has_cu == 1:
                token_start = metadata_args.cu_seqlens[batch_idx]
                token_end = metadata_args.cu_seqlens[batch_idx + 1]
                valid_seq_len = token_end - token_start
            valid_seq_len_by_batch[local_batch_idx] = valid_seq_len
            token_start_by_batch[local_batch_idx] = token_start
            chunk_num_by_batch[local_batch_idx] = (valid_seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            tail_row_num_by_batch[local_batch_idx] = ((valid_seq_len - 1) % CHUNK_SIZE) + 1

        chunk_start = 0
        while True:
            active_batch_num = 0
            for local_batch_idx in range(0, batch_num_in_set):
                if chunk_num_by_batch[local_batch_idx] > chunk_start:
                    active_local_batch_idx[active_batch_num] = local_batch_idx
                    remaining_chunk_num[active_batch_num] = (
                        chunk_num_by_batch[local_batch_idx] - chunk_start
                    )
                    active_batch_num += 1
            if active_batch_num == 0:
                break

            group_num = _select_group_num(
                remaining_chunk_num, active_batch_num, metadata_args.value_heads,
                metadata_args.workspace_slots, max_chunk_group,
            )
            for active_idx in range(0, active_batch_num):
                current_chunk_num = remaining_chunk_num[active_idx]
                if current_chunk_num > group_num:
                    current_chunk_num = group_num
                chunk_num_per_group[active_idx] = current_chunk_num

            metadata_args.metadata[HEADER_WORDS + global_round_idx] = record_cursor
            cursor = record_cursor
            metadata_args.metadata[cursor] = global_round_idx
            metadata_args.metadata[cursor + 1] = active_batch_num
            cursor += 2
            for active_idx in range(0, active_batch_num):
                metadata_args.metadata[cursor + active_idx] = (
                    batch_start + active_local_batch_idx[active_idx]
                )
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                logical_batch_idx = batch_start + active_local_batch_idx[active_idx]
                if metadata_args.is_packed == 1:
                    metadata_args.metadata[cursor + active_idx] = 0
                else:
                    metadata_args.metadata[cursor + active_idx] = logical_batch_idx
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                if metadata_args.is_packed == 1:
                    metadata_args.metadata[cursor + active_idx] = token_start_by_batch[
                        local_batch_idx
                    ]
                else:
                    metadata_args.metadata[cursor + active_idx] = 0
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                metadata_args.metadata[cursor + active_idx] = valid_seq_len_by_batch[
                    local_batch_idx
                ]
            cursor += active_batch_num
            metadata_args.metadata[cursor] = group_num
            cursor += 1
            for active_idx in range(0, active_batch_num):
                metadata_args.metadata[cursor + active_idx] = chunk_start
            cursor += active_batch_num
            for active_idx in range(0, active_batch_num):
                metadata_args.metadata[cursor + active_idx] = chunk_num_per_group[active_idx]
            cursor += active_batch_num

            stage1_task_num = 0
            metadata_args.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage1_task_num += metadata_args.value_heads * chunk_num_per_group[active_idx]
                metadata_args.metadata[cursor + active_idx + 1] = stage1_task_num
            cursor += active_batch_num + 1

            stage1_task_idx = 0
            for active_idx in range(0, active_batch_num):
                local_batch_idx = active_local_batch_idx[active_idx]
                for _ in range(0, metadata_args.value_heads):
                    for local_chunk_idx in range(0, chunk_num_per_group[active_idx]):
                        global_chunk_idx = chunk_start + local_chunk_idx
                        stage1_cost = FULL_CHUNK_COST_NS
                        if (
                            global_chunk_idx + 1 == chunk_num_by_batch[local_batch_idx]
                            and tail_row_num_by_batch[local_batch_idx] < CHUNK_SIZE
                        ):
                            stage1_cost = _stage1_task_cost(tail_row_num_by_batch[local_batch_idx])
                        stage1_task_costs[stage1_task_idx] = stage1_cost
                        stage1_task_idx += 1
            _partition_costs(
                stage1_task_costs, stage1_task_num, metadata_args.core_num,
                stage1_core_ranges, stage1_core_costs,
            )
            for range_idx in range(0, MAX_AIC_CORES * 2):
                metadata_args.metadata[cursor + range_idx] = stage1_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            selected_dv_splits_num = 1
            selected_max_core_cost = 0
            selected_total_cost = 0
            selected_task_num = 0
            selected_is_set = 0
            for candidate_idx in range(0, 4):
                candidate_dv_splits_num = 1 << candidate_idx
                candidate_task_num = 0
                candidate_total_cost = 0
                for active_idx in range(0, active_batch_num):
                    task_cost = _stage2_task_cost(candidate_idx, chunk_num_per_group[active_idx])
                    batch_task_num = metadata_args.value_heads * candidate_dv_splits_num
                    for _ in range(0, batch_task_num):
                        stage2_task_costs[candidate_task_num] = task_cost
                        candidate_task_num += 1
                        candidate_total_cost += task_cost
                _partition_costs(
                    stage2_task_costs, candidate_task_num, metadata_args.core_num, candidate_stage2_core_ranges,
                    candidate_stage2_core_costs,
                )
                candidate_max_core_cost = 0
                for core_idx in range(0, MAX_AIC_CORES):
                    if candidate_stage2_core_costs[core_idx] > candidate_max_core_cost:
                        candidate_max_core_cost = candidate_stage2_core_costs[core_idx]

                candidate_is_better = 0
                if selected_is_set == 0:
                    candidate_is_better = 1
                elif candidate_max_core_cost < selected_max_core_cost:
                    candidate_is_better = 1
                elif candidate_max_core_cost == selected_max_core_cost:
                    if candidate_total_cost < selected_total_cost:
                        candidate_is_better = 1
                    elif candidate_total_cost == selected_total_cost:
                        if candidate_task_num < selected_task_num:
                            candidate_is_better = 1
                        elif candidate_task_num == selected_task_num:
                            if candidate_dv_splits_num < selected_dv_splits_num:
                                candidate_is_better = 1
                if candidate_is_better == 1:
                    selected_is_set = 1
                    selected_dv_splits_num = candidate_dv_splits_num
                    selected_max_core_cost = candidate_max_core_cost
                    selected_total_cost = candidate_total_cost
                    selected_task_num = candidate_task_num
                    for range_idx in range(0, MAX_AIC_CORES * 2):
                        selected_stage2_core_ranges[range_idx] = (
                            candidate_stage2_core_ranges[range_idx]
                        )

            for active_idx in range(0, active_batch_num):
                metadata_args.metadata[cursor + active_idx] = selected_dv_splits_num
            cursor += active_batch_num
            stage2_task_num = 0
            metadata_args.metadata[cursor] = 0
            for active_idx in range(0, active_batch_num):
                stage2_task_num += metadata_args.value_heads * selected_dv_splits_num
                metadata_args.metadata[cursor + active_idx + 1] = stage2_task_num
            cursor += active_batch_num + 1
            for range_idx in range(0, MAX_AIC_CORES * 2):
                metadata_args.metadata[cursor + range_idx] = selected_stage2_core_ranges[range_idx]
            cursor += MAX_AIC_CORES * 2

            record_cursor = cursor
            global_round_idx += 1
            chunk_start += group_num
        batch_start += batch_num_in_set

    metadata_args.metadata[HEADER_WORDS + stage12_round_num] = record_cursor
    return STATUS_OK


def _take(words: list[int], cursor: int, count: int) -> tuple[list[int], int]:
    end = cursor + count
    if cursor < 0 or count < 0 or end > len(words):
        raise ValueError("metadata record extends past available words")
    return words[cursor:end], end


def _decode_ranges(flat: list[int]) -> list[tuple[int, int]]:
    if len(flat) != MAX_AIC_CORES * 2:
        raise ValueError("core range block must contain 64 words")
    return [(flat[index], flat[index + 1]) for index in range(0, len(flat), 2)]


def _validate_prefix(prefix: list[int], name: str) -> None:
    if not prefix or prefix[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(end < start for start, end in zip(prefix, prefix[1:])):
        raise ValueError(f"{name} must be nondecreasing")


def _validate_ranges(ranges: list[tuple[int, int]], total: int, name: str) -> None:
    cursor = 0
    saw_empty = False
    for start, end in ranges:
        if (start, end) == (0, 0):
            saw_empty = True
            continue
        if saw_empty:
            raise ValueError(f"{name} has work after an unused core")
        if start != cursor or end <= start or end > total:
            raise ValueError(f"{name} is not a contiguous partition")
        cursor = end
    if cursor != total:
        raise ValueError(f"{name} does not cover its task total")


def _decode_round(
    words: list[int],
    start: int,
    end: int,
    *,
    batch: int,
    value_heads: int,
) -> dict[str, object]:
    if start < 0 or end > len(words) or start + 2 > end:
        raise ValueError("metadata round record is truncated")
    # 将后续所有 _take 和标量访问限制在当前记录范围内。
    words = words[start:end]
    start, end = 0, end - start
    cursor = 0
    stage12_round_idx, active_batch_num = words[cursor : cursor + 2]
    cursor += 2
    if not 1 <= active_batch_num <= min(batch, WORKSPACE_SLOTS):
        raise ValueError("metadata active_batch_num outside batch/workspace bounds")
    if start + round_record_words(active_batch_num) != end:
        raise ValueError("metadata round record length does not match active batches")

    batch_idx, cursor = _take(words, cursor, active_batch_num)
    storage_batch_idx, cursor = _take(words, cursor, active_batch_num)
    token_start, cursor = _take(words, cursor, active_batch_num)
    valid_seq_len, cursor = _take(words, cursor, active_batch_num)
    group_num = words[cursor]
    cursor += 1
    chunk_start_per_group, cursor = _take(words, cursor, active_batch_num)
    chunk_num_per_group, cursor = _take(words, cursor, active_batch_num)
    stage1_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage1_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    dv_splits_num, cursor = _take(words, cursor, active_batch_num)
    stage2_task_prefix, cursor = _take(words, cursor, active_batch_num + 1)
    stage2_ranges_flat, cursor = _take(words, cursor, MAX_AIC_CORES * 2)
    if cursor != end:
        raise ValueError("metadata round decoder did not consume the full record")

    if group_num <= 0:
        raise ValueError("group_num must be positive")
    if any(index < 0 or index >= batch for index in batch_idx):
        raise ValueError("batch_idx is outside logical batch")
    if any(length <= 0 for length in valid_seq_len):
        raise ValueError("valid_seq_len must be positive")
    if any(count <= 0 for count in chunk_num_per_group):
        raise ValueError("chunk_num_per_group must be positive")
    if any(value not in (1, 2, 4, 8) for value in dv_splits_num):
        raise ValueError("dv_splits_num must be 1, 2, 4, or 8")

    _validate_prefix(stage1_task_prefix, "stage1_task_prefix")
    _validate_prefix(stage2_task_prefix, "stage2_task_prefix")
    for index, chunk_num in enumerate(chunk_num_per_group):
        if stage1_task_prefix[index + 1] - stage1_task_prefix[index] != (
            value_heads * chunk_num
        ):
            raise ValueError("stage1_task_prefix does not match chunk tasks")
        if stage2_task_prefix[index + 1] - stage2_task_prefix[index] != (
            value_heads * dv_splits_num[index]
        ):
            raise ValueError("stage2_task_prefix does not match DV tasks")
    if stage1_task_prefix[-1] > WORKSPACE_SLOTS:
        raise ValueError("stage1 task total exceeds workspace slots")
    if stage2_task_prefix[-1] > WORKSPACE_SLOTS * 8:
        raise ValueError("stage2 task total exceeds workspace slots")

    stage1_core_ranges = _decode_ranges(stage1_ranges_flat)
    stage2_core_ranges = _decode_ranges(stage2_ranges_flat)
    _validate_ranges(stage1_core_ranges, stage1_task_prefix[-1], "stage1_core_ranges")
    _validate_ranges(stage2_core_ranges, stage2_task_prefix[-1], "stage2_core_ranges")
    return {
        "stage12_round_idx": stage12_round_idx,
        "active_batch_num": active_batch_num,
        "batch_idx": batch_idx,
        "storage_batch_idx": storage_batch_idx,
        "token_start": token_start,
        "valid_seq_len": valid_seq_len,
        "group_num": group_num,
        "chunk_start_per_group": chunk_start_per_group,
        "chunk_num_per_group": chunk_num_per_group,
        "stage1_task_prefix": stage1_task_prefix,
        "stage1_core_ranges": stage1_core_ranges,
        "dv_splits_num": dv_splits_num,
        "stage2_task_prefix": stage2_task_prefix,
        "stage2_core_ranges": stage2_core_ranges,
    }


def decode_metadata(raw: Sequence[int]) -> dict[str, object]:
    """解码并检查 metadata，供测试和诊断使用。"""

    words = [int(value) for value in raw]
    if len(words) < HEADER_WORDS:
        raise ValueError("metadata is shorter than its header")
    if words[HEADER_MAGIC] != MAGIC:
        raise ValueError("metadata magic does not match")
    if words[HEADER_ABI_VERSION] != ABI_VERSION:
        raise ValueError("metadata ABI version does not match")

    status = words[HEADER_STATUS]
    header = {
        "magic": words[HEADER_MAGIC],
        "abi_version": words[HEADER_ABI_VERSION],
        "status": status,
        "core_num": words[HEADER_CORE_NUM],
        "stage12_round_num": words[HEADER_STAGE12_ROUND_NUM],
        "batch": words[HEADER_BATCH],
        "value_heads": words[HEADER_VALUE_HEADS],
        "has_cu": words[HEADER_HAS_CU],
        "workspace_slots": words[HEADER_WORKSPACE_SLOTS],
        "template": {TEMPLATE_STAGED: "staged", TEMPLATE_PERSISTENT: "persistent"}.get(words[HEADER_TEMPLATE], "unknown"),
    }
    if status != STATUS_OK:
        header["used_words"] = HEADER_WORDS
        header["stage12_rounds"] = []
        return header
    if not 1 <= header["core_num"] <= MAX_AIC_CORES:
        raise ValueError("metadata core_num must be in [1, 32]")
    if header["workspace_slots"] != WORKSPACE_SLOTS:
        raise ValueError("metadata workspace_slots must be 861")
    if header["batch"] <= 0:
        raise ValueError("successful metadata requires positive batch")
    if not 1 <= header["value_heads"] <= MAX_VALUE_HEADS:
        raise ValueError("metadata value_heads must be in [1, 96]")

    if header["has_cu"] not in (0, 1):
        raise ValueError("metadata has_cu must be 0 or 1")
    if header["template"] == "unknown":
        raise ValueError("metadata template is invalid")
    if header["template"] == "persistent":
        task_num = words[HEADER_CHUNK_TASK_NUM]
        task_base = HEADER_WORDS + MAX_AIC_CORES
        used_words = task_base + CHUNK_TASK_WORDS * task_num
        if (
            header["stage12_round_num"] != 0
            or task_num != header["batch"] * header["value_heads"]
        ):
            raise ValueError("chunk metadata task count does not match its shape")
        if used_words > len(words):
            raise ValueError("chunk metadata tasks are truncated")
        seen_offsets = set()
        seen_jobs = set()
        tasks_by_core = []
        for core_idx in range(MAX_AIC_CORES):
            cursor = words[HEADER_WORDS + core_idx]
            if core_idx >= header["core_num"] and cursor:
                raise ValueError("inactive core has chunk tasks")
            tasks = []
            while cursor:
                if cursor in seen_offsets:
                    raise ValueError("chunk task chain contains a cycle or duplicate")
                if (
                    cursor < task_base
                    or cursor + CHUNK_TASK_WORDS > used_words
                    or (cursor - task_base) % CHUNK_TASK_WORDS
                ):
                    raise ValueError("chunk task pointer is out of bounds or unaligned")
                seen_offsets.add(cursor)
                batch_idx = words[cursor + CHUNK_TASK_BATCH]
                head_idx = words[cursor + CHUNK_TASK_VALUE_HEAD]
                if (
                    not 0 <= batch_idx < header["batch"]
                    or not 0 <= head_idx < header["value_heads"]
                ):
                    raise ValueError("chunk task batch/head index is out of bounds")
                if (batch_idx, head_idx) in seen_jobs:
                    raise ValueError("chunk sequence/head task is duplicated")
                seen_jobs.add((batch_idx, head_idx))
                length = words[cursor + CHUNK_TASK_LENGTH]
                storage_batch = words[cursor + CHUNK_TASK_STORAGE_BATCH]
                token_start = words[cursor + CHUNK_TASK_TOKEN_START]
                if (
                    length <= 0
                    or token_start < 0
                    or storage_batch not in (0, batch_idx)
                ):
                    raise ValueError("chunk task has invalid sequence coordinates")
                tasks.append(
                    dict(
                        batch_idx=batch_idx, storage_batch_idx=storage_batch, token_start=token_start,
                        sequence_length=length, value_head_idx=head_idx,
                    ),
                )
                cursor = words[cursor + CHUNK_TASK_NEXT]
            tasks_by_core.append(tasks)
        if len(seen_jobs) != task_num:
            raise ValueError("chunk metadata does not cover every task")
        header.update(
            used_words=used_words, stage12_rounds=[], chunk_task_num=task_num, chunk_tasks_by_core=tasks_by_core,
        )
        return header

    if words[HEADER_CHUNK_TASK_NUM] != 0:
        raise ValueError("staged metadata must not contain persistent tasks")
    stage12_round_num = int(header["stage12_round_num"])
    if stage12_round_num < 1:
        raise ValueError("staged metadata round count must be positive")
    offset_begin = HEADER_WORDS
    offset_end = offset_begin + stage12_round_num + 1
    if offset_end > len(words):
        raise ValueError("metadata offset table is truncated")
    min_record_words = round_record_words(1)
    if offset_end + stage12_round_num * min_record_words > len(words):
        raise ValueError("metadata capacity is insufficient for round records")
    offsets = words[offset_begin:offset_end]
    if any(offset < 0 for offset in offsets):
        raise ValueError("metadata offset must be nonnegative")
    if offsets[0] < offset_end:
        raise ValueError("metadata first record offset overlaps the offset table")
    if offsets[-1] > len(words):
        raise ValueError("metadata final offset exceeds capacity")
    if any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("metadata offsets must be strictly increasing")

    header["used_words"] = offsets[-1]
    header["stage12_rounds"] = [
        _decode_round(
            words, offsets[index], offsets[index + 1], batch=int(header["batch"]),
            value_heads=int(header["value_heads"]),
        )
        for index in range(stage12_round_num)
    ]
    return header



_AICPU_COMPILED = None
_AICPU_LOCK = threading.RLock()
_AICPU_STREAMS = {}


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """返回当前 NPU 流上可安全使用全核屏障的启动核数。"""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return MAX_AIC_CORES
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("NPU runtime is unavailable")
    device_index = ref.device.index if ref is not None else npu.current_device()
    if device_index is None:
        device_index = npu.current_device()
    platform_info = get_platform_info(stream=npu.current_stream(device_index))
    try:
        cube_core_num = int(platform_info.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU {device_index} does not expose a valid cube_core_num") from exc
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports invalid cube_core_num={cube_core_num}")
    return min(cube_core_num, MAX_AIC_CORES)


def _get_aicpu_kernel():
    global _AICPU_COMPILED
    with _AICPU_LOCK:
        if _AICPU_COMPILED is None:
            _AICPU_COMPILED = compile_aicpu_kernel(
                flash_kda_fwd_metadata_kernel, workdir=tempfile.mkdtemp(prefix="flash_kda_metadata_"),
                launch_mode="interface", npu_arch="dav-3510",
            )
        return _AICPU_COMPILED


def flash_kda_fwd_metadata(
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    batch_size: int,
    nheads_qk: int,
    nheads_v: int,
    seqlens: int,
    headdim_qk: int = SUPPORTED_HEAD_DIM,
    headdim_v: int = SUPPORTED_HEAD_DIM,
    enable_grad: bool = False,
    layout: str = "TND",
) -> torch.Tensor:
    """根据显式存储维度在当前 NPU 上构建 metadata。"""
    return _build_flash_kda_fwd_metadata(
        batch_size, nheads_qk, nheads_v, cu_seqlens, seqused, seqlens,
        headdim_qk, headdim_v, enable_grad, layout,
    )


def _schedule_versions(metadata, cu_seqlens, seqused):
    # 推理张量没有版本计数器，捕获和重放期间必须保持调度数据不变。
    return tuple(None if t is None or t.is_inference() else t._version
                 for t in (metadata, cu_seqlens, seqused))


def _capture_schedule(metadata, spec, cu_seqlens, seqused, device, core_num):
    schedule = getattr(metadata, "_flash_kda_schedule", None)
    require(schedule is not None and schedule["owner"] == id(metadata),
            "capture requires the original tensor returned by flash_kda_metadata")
    require(metadata.device == device and schedule["spec"] == spec and schedule["core_num"] == core_num,
            "capture metadata does not match the input device, shape, layout or available cores")
    require(cu_seqlens is schedule["cu_seqlens"] and seqused is schedule["seqused"],
            "capture requires the length tensors used to generate metadata")
    require(_schedule_versions(metadata, cu_seqlens, seqused) == schedule["versions"],
            "metadata or lengths changed after metadata generation")
    return schedule["lengths"], schedule["template"]


def _build_flash_kda_fwd_metadata(
    batch_size: int,
    nheads_qk: int,
    nheads_v: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    seqlens: Optional[int] = None,
    headdim_qk: int = SUPPORTED_HEAD_DIM,
    headdim_v: int = SUPPORTED_HEAD_DIM,
    enable_grad: bool = False,
    layout: str = "TND",
    *,
    template: str = "auto",
) -> torch.Tensor:
    """根据标量存储维度和长度生成经过校验的 ABI4 metadata。

    seqlens 必须显式指定填充长度 S 或紧凑长度 T，有效长度不能确定填充存储容量。
    无需构造虚拟输入或状态张量。"""
    validate_layout(layout)
    require(isinstance(template, str) and template in ("auto", "staged", "persistent"),
            "template must be auto, staged, or persistent")
    template_id = {"auto": TEMPLATE_AUTO, "staged": TEMPLATE_STAGED, "persistent": TEMPLATE_PERSISTENT}[template]
    for name, value in (("batch_size", batch_size), ("nheads_qk", nheads_qk),
                        ("nheads_v", nheads_v), ("seqlens", seqlens),
                        ("headdim_qk", headdim_qk), ("headdim_v", headdim_v)):
        require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f"{name} must be a positive integer, got {value!r}")
        require(value <= INT32_MAX, f"{name} exceeds INT32 representable range")
    require(headdim_qk == headdim_v == SUPPORTED_HEAD_DIM, "headdim_qk/headdim_v must be 128")
    require(isinstance(enable_grad, bool), "enable_grad must be bool")
    validate_sizes(batch_size, nheads_qk, nheads_v, seqlens)
    spec = SequenceSpec(batch_size, nheads_qk, nheads_v, seqlens, layout)
    is_packed = layout == "TND"
    capacity = metadata_capacity_upper_bound(batch_size, seqlens, is_packed=is_packed, value_heads=nheads_v)
    require(capacity <= INT32_MAX, "metadata capacity exceeds INT32 representable range")
    # 查询运行时或读取数据前，检查张量规格。
    validate_length_specs(cu_seqlens, seqused, spec)
    if any(t is not None and t.is_meta for t in (cu_seqlens, seqused)):
        return torch.empty(capacity, dtype=torch.int32, device="meta")
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("NPU runtime is unavailable")
    device = torch.device("npu", npu.current_device())
    lengths = read_lengths(cu_seqlens, seqused, spec)
    core_num = _device_block_num()
    use_short_group = (enable_grad and layout == "BNSD" and batch_size == 1
                       and nheads_qk == nheads_v == SHORT_TRAIN_HEADS and seqlens >= SHORT_TRAIN_MIN_LENGTH
                       and seqlens % CHUNK_SIZE == 0 and cu_seqlens is None
                       and seqused is None and core_num == MAX_AIC_CORES)
    max_chunk_group = SHORT_TRAIN_CHUNK_GROUP if use_short_group else MAX_CHUNK_GROUP
    has_cu = cu_seqlens is not None or seqused is not None
    original_cu_seqlens = cu_seqlens
    if cu_seqlens is None and seqused is not None:
        prefix = [0]
        for size in lengths:
            prefix.append(prefix[-1] + size)
        cu_seqlens = torch.tensor(prefix, dtype=torch.int32, device=device)
    metadata = torch.empty(capacity, dtype=torch.int32, device=device)
    # 两个模板的舍入边界不同，同一输入在训推模式下选择相同模板。
    # 输出和缓存需求仍由调用方执行模式控制。
    route_parameters, calibration_enabled = _get_route_parameters(device)
    host_prefix = [0]
    for size in lengths:
        host_prefix.append(host_prefix[-1] + size)
    template_id, max_chunk_group = _select_template_on_host(SimpleNamespace(
        cu_seqlens=host_prefix, route_coefficients=route_parameters,
        calibration_enabled=calibration_enabled, layout_kind=0 if layout == "BNSD" else 1,
        # Only heap scratch is accessed by routing; retain its original word offsets
        # and capacity check without allocating the complete device metadata on Host.
        metadata={}, metadata_word_capacity=capacity, batch=batch_size,
        storage_length=seqlens, value_heads=nheads_v, query_heads=nheads_qk,
        has_cu=int(has_cu), is_packed=int(is_packed), core_num=core_num,
        workspace_slots=WORKSPACE_SLOTS, max_chunk_group=max_chunk_group,
        template_kind=template_id,
    ))
    device_id = device.index
    with torch.npu.device(device_id), _AICPU_LOCK:
        stream = torch.npu.current_stream(device_id)
        for tensor in (cu_seqlens, metadata):
            if tensor is not None:
                tensor.record_stream(stream)
        _AICPU_STREAMS[(device_id, stream.npu_stream)] = stream
        try:
            aicpu = _get_aicpu_kernel()
            aicpu.launch(
                current_raw_stream(device_id),
                cu_seqlens=0 if cu_seqlens is None else cu_seqlens.data_ptr(),
                metadata=metadata.data_ptr(), metadata_word_capacity=capacity,
                batch=batch_size, physical_batches=1 if is_packed else batch_size,
                storage_length=seqlens, value_heads=nheads_v, has_cu=int(has_cu),
                is_packed=int(is_packed), core_num=core_num, workspace_slots=WORKSPACE_SLOTS,
                max_chunk_group=max_chunk_group, template_kind=template_id,
            )
        except Exception:
            stream.synchronize()
            raise
    # 复制失败保留运行时异常；生产结果格式错误时，保留校验原因并报告生成失败。
    words = metadata.detach().cpu().tolist()
    if words[HEADER_STATUS] != STATUS_OK:
        raise ValueError(f"flash_kda_fwd_metadata produced status {words[HEADER_STATUS]}")
    try:
        decoded = decode_metadata(words)
        validate_schedule(decoded, spec, lengths, core_num, has_cu)
    except ValueError as exc:
        raise ValueError(f"flash_kda_fwd_metadata produced invalid metadata: {exc}") from exc
    # 主机端仅保留已校验的调度参数，完整调度表仍存放在设备端。
    metadata._flash_kda_schedule = {
        "owner": id(metadata),
        "spec": tuple(spec),
        "core_num": core_num,
        "cu_seqlens": original_cu_seqlens,
        "seqused": seqused,
        "versions": _schedule_versions(metadata, original_cu_seqlens, seqused),
        "lengths": tuple(lengths),
        "template": words[HEADER_TEMPLATE],
    }
    return metadata


def clear_metadata_caches():
    """等待已使用的流完成后，再释放 metadata 可执行对象。"""
    global _AICPU_COMPILED
    with _AICPU_LOCK:
        for (device_id, _), stream in _AICPU_STREAMS.items():
            with torch.npu.device(device_id):
                stream.synchronize()
        _AICPU_STREAMS.clear()
        if _AICPU_COMPILED is not None:
            close = getattr(_AICPU_COMPILED, "close", None)
            if callable(close):
                close()
        _AICPU_COMPILED = None


# 非负 PR/DT 重叠成本以飞秒计，同时记录格式版本和切换余量（ppm）。
_ROUTE_PARAMETERS = (
    0, 2711857704, 699286307, 16555661144, 28359460, 94263834, 94263834, 12013349865, 1045656019, 84358156, 12842477, 2527878117, 181898838, 6, 8441, 0,
    0, 692059912, 851406583, 21510815509, 73490570, 81791118, 166178779, 12029971961, 2552246171, 459139222, 45086212, 2303638937, 143977434, 6, 18992, 0,
)
_DT_ROUTE_PARAMETERS = (
    4135271144, 193981546, 366448654, 12881511041, 29417161, 4351059785, 158980371, 10652650460, 3206978745, 332321804, 0, 419242242, 29945874, 7, 0, 0,
    20317858430, 1160598799, 1018161115, 1015361233, 14389959, 933231823, 115749683, 11499084207, 4373896639, 413928806, 41555011, 0, 0, 7, 0, 0,
)


def _get_route_parameters(device) -> tuple[tuple[int, ...], bool]:
    """返回成本系数及标定模型启用标志，供 Host 模板选择使用。

    calibration_enabled 为 True，表示启用原有标定成本模型进行自动选路。
    """
    device = torch.device(device)
    with torch.npu.device(device):
        device_name = torch.npu.get_device_properties(device).name.upper()
    parameters = _DT_ROUTE_PARAMETERS if device_name.startswith("ASCEND950DT") else _ROUTE_PARAMETERS
    calibration_enabled = True
    return parameters, calibration_enabled


# Both public names share the same metadata contract and scheduler.
flash_kda_metadata = flash_kda_fwd_metadata
_build_flash_kda_metadata = _build_flash_kda_fwd_metadata
flash_kda_metadata_kernel = flash_kda_fwd_metadata_kernel
FlashKDAMetadataArgs = FlashKDAFwdMetadataArgs
__all__ += ["flash_kda_metadata", "flash_kda_metadata_kernel", "FlashKDAMetadataArgs"]
