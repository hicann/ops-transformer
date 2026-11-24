# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDAFwdArch92 的 Host 和 Meta 契约（设计第 4.6、4.9 节）。

静态校验不读取张量内容，也不分配设备缓冲区。
所有显式参数检查失败时均抛出 ValueError，在 python -O 下同样有效。"""

import math
from typing import NamedTuple

import torch

# Host layout discovery supplies declaration-only Torch symbols without graph APIs.
# Real Torch provides torch.library and keeps its original runtime registration.
if not hasattr(torch, "library"):

    class _HostLayoutNode:
        def __getattr__(self, name):
            return _HostLayoutNode()

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _HostLayoutNode()

    def _host_layout_torch_getattr(name):
        return _HostLayoutNode()

    torch.__getattr__ = _host_layout_torch_getattr

CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
MAX_VALUE_HEADS = 96
WORKSPACE_SLOTS = 861
MAX_CHUNK_GROUP = 512
HEADER_WORDS = 11  # ABI5 头部的 INT32 字数。
INT32_MAX = 2**31 - 1
MIN_GATE_BOUND = -5.0
STATUS_OK = 0

# ABI capacity is fixed by the format, not by the current stream core quota.
MAX_AIC_CORES = 64
ROUND_FIXED_WORDS = 5 + 4 * MAX_AIC_CORES
ROUND_WORDS_PER_BATCH = 9
CHUNK_TASK_WORDS = 6


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


class SequenceSpec(NamedTuple):
    batch: int
    n_qk: int
    n_v: int
    length: int
    layout: str


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_spec(name, value, *, rank, dtypes, shape=None):
    require(value is not None, f"{name} is required")
    require(isinstance(value, torch.Tensor), f"{name} must be Tensor")
    require(value.dim() == rank, f"{name} rank expected {rank}, got {value.dim()}")
    require(value.dtype in dtypes, f"{name} dtype expected {dtypes}, got {value.dtype}")
    if shape is not None:
        require(
            value.shape == shape,
            f"{name} shape expected {shape}, got {tuple(value.shape)}",
        )


def validate_contiguous(name, value):
    require(value.is_contiguous(), f"{name} must be contiguous")


def validate_layout(layout):
    require(isinstance(layout, str), "layout must be str")
    require(
        layout in ("BNSD", "BSND", "TND"),
        f"layout expected BNSD/BSND/TND, got {layout!r}",
    )


def validate_sizes(batch, n_qk, n_v, length):
    for name, value in (
        ("batch", batch),
        ("Nqk", n_qk),
        ("Nv", n_v),
        ("storage length", length),
    ):
        require(value > 0, f"{name} must be positive, got {value}")
    require(
        n_qk <= n_v <= MAX_VALUE_HEADS,
        f"heads must satisfy 1 <= Nqk <= Nv <= 96, got {n_qk}/{n_v}",
    )
    require(n_v % n_qk == 0, f"GQA requires Nv % Nqk == 0, got Nv={n_v}, Nqk={n_qk}")


def validate_length_specs(cu_seqlens, seqused, spec):
    # 长度参数按布局互斥，metadata 生成和前向入口共用此校验。
    if spec.layout == "TND":
        require(cu_seqlens is not None, "cu_seqlens is required for TND")
        require(seqused is None, "seqused must be None for TND; use cu_seqlens")
    else:
        require(
            cu_seqlens is None, "cu_seqlens must be None for BNSD/BSND; use seqused"
        )
        require(seqused is not None, "seqused is required for BNSD/BSND")
    for name, value, size in (
        ("cu_seqlens", cu_seqlens, spec.batch + 1),
        ("seqused", seqused, spec.batch),
    ):
        if value is not None:
            tensor_spec(name, value, rank=1, dtypes=(torch.int32,), shape=(size,))
            validate_contiguous(name, value)


def validate_inputs(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    A_log,
    dt_bias,
    *,
    metadata,
    cu_seqlens,
    seqused,
    scale,
    output_final_state,
    return_intermediate_states,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    disable_recompute,
    lower_bound,
    layout,
    real_execution=False,
    cu_seqlens_cpu=None,
):
    """在构造视图、类型转换、复制或分配前，执行 Python/Meta 共用检查。"""
    validate_layout(layout)
    require(
        cu_seqlens_cpu is None,
        "cu_seqlens_cpu must be None; host lengths are unsupported",
    )
    flags = dict(
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
    )
    for name, value in flags.items():
        require(isinstance(value, bool), f"{name} must be bool")
    require(
        scale is None
        or (isinstance(scale, (int, float)) and not isinstance(scale, bool)),
        "scale must be a number excluding bool, or None",
    )
    if scale is not None:
        require(math.isfinite(scale), f"scale must be finite, got {scale}")
    rank = 3 if layout == "TND" else 4
    inputs = dict(q=q, k=k, v=v, g=g, beta=beta, metadata=metadata)
    # 使用对象属性前，先检查所有必需对象。
    for name, value in inputs.items():
        require(value is not None, f"{name} is required")
        require(isinstance(value, torch.Tensor), f"{name} must be Tensor")
    for name, value in (("q", q), ("k", k), ("v", v)):
        tensor_spec(name, value, rank=rank, dtypes=(torch.bfloat16,))
    tensor_spec("g", g, rank=rank, dtypes=(torch.bfloat16, torch.float32))
    tensor_spec("beta", beta, rank=rank - 1, dtypes=(torch.bfloat16, torch.float32))
    if layout == "TND":
        require(cu_seqlens is not None, "cu_seqlens is required for TND")
        tensor_spec("cu_seqlens", cu_seqlens, rank=1, dtypes=(torch.int32,))
        batch, length, n_qk, n_v = (
            cu_seqlens.shape[0] - 1,
            q.shape[0],
            q.shape[1],
            v.shape[1],
        )
        expected_v = (length, n_v, SUPPORTED_HEAD_DIM)
    else:
        batch = q.shape[0]
        length = q.shape[2 if layout == "BNSD" else 1]
        n_qk = q.shape[1 if layout == "BNSD" else 2]
        n_v = v.shape[1 if layout == "BNSD" else 2]
        expected_v = (
            (batch, n_v, length, SUPPORTED_HEAD_DIM)
            if layout == "BNSD"
            else (batch, length, n_v, SUPPORTED_HEAD_DIM)
        )
    validate_sizes(batch, n_qk, n_v, length)
    for name, value in (("q", q), ("k", k), ("v", v), ("g", g)):
        require(
            value.shape[-1] == SUPPORTED_HEAD_DIM,
            f"{name} head dimension expected 128, got {value.shape[-1]}",
        )
    expected_g = (
        (length, n_v, SUPPORTED_HEAD_DIM)
        if layout == "TND"
        else (batch, length, n_v, SUPPORTED_HEAD_DIM)
    )
    for name, value, shape in (
        ("k", k, q.shape),
        ("v", v, expected_v),
        ("g", g, expected_g),
        ("beta", beta, expected_g[:-1]),
    ):
        require(
            value.shape == shape,
            f"{name} shape expected {tuple(shape)}, got {tuple(value.shape)}",
        )
    if initial_state is not None:
        tensor_spec(
            "initial_state",
            initial_state,
            rank=4,
            dtypes=(torch.float32,),
            shape=(batch, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
        )
        inputs["initial_state"] = initial_state
    require(
        not allow_neg_eigval or use_beta_sigmoid_in_kernel,
        "allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True",
    )
    if use_gate_in_kernel:
        require(A_log is not None, "A_log is required when use_gate_in_kernel=True")
        tensor_spec("A_log", A_log, rank=1, dtypes=(torch.float32,), shape=(n_v,))
        inputs["A_log"] = A_log
        if dt_bias is not None:
            flat_bias = isinstance(dt_bias, torch.Tensor) and dt_bias.dim() == 1
            tensor_spec(
                "dt_bias",
                dt_bias,
                rank=1 if flat_bias else 2,
                dtypes=(torch.float32,),
                shape=(n_v * SUPPORTED_HEAD_DIM,)
                if flat_bias
                else (n_v, SUPPORTED_HEAD_DIM),
            )
            inputs["dt_bias"] = dt_bias
        if lower_bound is not None:
            require(
                isinstance(lower_bound, (int, float))
                and not isinstance(lower_bound, bool),
                "lower_bound must be a number excluding bool",
            )
            require(
                math.isfinite(lower_bound) and MIN_GATE_BOUND <= lower_bound < 0,
                f"lower_bound must be finite and in [-5,0), got {lower_bound}",
            )
    tensor_spec("metadata", metadata, rank=1, dtypes=(torch.int32,))
    require(
        metadata.numel() >= HEADER_WORDS,
        f"metadata shorter than ABI5 header: expected >=11, got {metadata.numel()}",
    )
    for name, value in inputs.items():
        validate_contiguous(name, value)
    spec = SequenceSpec(batch, n_qk, n_v, length, layout)
    validate_length_specs(cu_seqlens, seqused, spec)
    if real_execution:
        require(
            q.device.type in ("npu", "privateuseone"), "execution device must be NPU"
        )
    return spec


def read_lengths(cu_seqlens, seqused, spec):
    """阻塞复制保证调用流上的写入可见；使用 Python 整数避免溢出。"""
    lengths = None
    if cu_seqlens is not None:
        values = cu_seqlens.detach().cpu().tolist()
        require(values[0] == 0, f"cu_seqlens[0] must be 0, got {values[0]}")
        lengths = [int(end) - int(start) for start, end in zip(values, values[1:])]
        for index, size in enumerate(lengths):
            require(
                size > 0, f"cu_seqlens length[{index}] must be positive, got {size}"
            )
            if spec.layout != "TND":
                require(
                    size <= spec.length,
                    f"cu_seqlens length[{index}] exceeds S={spec.length}: {size}",
                )
        if spec.layout == "TND":
            require(
                values[-1] == spec.length,
                f"cu_seqlens end expected {spec.length}, got {values[-1]}",
            )
    if seqused is not None:
        used = seqused.detach().cpu().tolist()
        for index, size in enumerate(used):
            require(
                0 < size <= spec.length,
                f"seqused[{index}] length out of bounds: {size}",
            )
            if lengths is not None:
                require(
                    size == lengths[index],
                    f"seqused[{index}] expected {lengths[index]} from cu_seqlens, got {size}",
                )
        lengths = used
    if lengths is None:
        lengths = [spec.length] * spec.batch
    require(sum(lengths) <= INT32_MAX, "sequence length prefix exceeds INT32 range")
    return lengths


def validate_schedule(decoded, spec, lengths, core_num, has_cu):
    """根据本次调用检查已解码的 ABI5 staged/persistent 记录。"""
    expected = {
        "status": STATUS_OK,
        "core_num": core_num,
        "batch": spec.batch,
        "value_heads": spec.n_v,
        "workspace_slots": WORKSPACE_SLOTS,
        "has_cu": int(has_cu),
    }
    for name, value in expected.items():
        require(
            decoded[name] == value,
            f"metadata {name} expected {value}, got {decoded[name]}",
        )
    next_chunk = [0] * spec.batch
    starts = [0] * spec.batch
    if spec.layout == "TND":
        for index in range(1, spec.batch):
            starts[index] = starts[index - 1] + lengths[index - 1]
    if decoded["template"] == "persistent":
        # 解码器已检查任务链边界、环、每个头的唯一性及 B*Nv 完整覆盖。
        # 此处进一步核对每项任务与本次调用是否一致。
        for core, tasks in enumerate(decoded["chunk_tasks_by_core"]):
            for task in tasks:
                b = task["batch_idx"]
                require(
                    core < core_num, "metadata persistent task uses unavailable core"
                )
                require(
                    task["storage_batch_idx"] == (0 if spec.layout == "TND" else b),
                    f"metadata persistent batch[{b}] storage batch mismatch",
                )
                require(
                    task["token_start"] == starts[b]
                    and task["sequence_length"] == lengths[b],
                    f"metadata persistent batch[{b}] token range mismatch",
                )
        return decoded
    for index, record in enumerate(decoded["stage12_rounds"]):
        prefix = f"metadata round[{index}]"
        require(record["stage12_round_idx"] == index, f"{prefix} index mismatch")
        require(
            len(set(record["batch_idx"])) == len(record["batch_idx"]),
            f"{prefix} repeats batch",
        )
        require(
            1
            <= record["group_num"]
            <= min(MAX_CHUNK_GROUP, WORKSPACE_SLOTS // spec.n_v),
            f"{prefix} group_num outside workspace bound",
        )
        for stage in (1, 2):
            require(
                all(
                    pair == (0, 0)
                    for pair in record[f"stage{stage}_core_ranges"][core_num:]
                ),
                f"{prefix} stage{stage} schedules unavailable cores",
            )
        for b, physical, start, length, chunk, count in zip(
            record["batch_idx"],
            record["storage_batch_idx"],
            record["token_start"],
            record["valid_seq_len"],
            record["chunk_start_per_group"],
            record["chunk_num_per_group"],
        ):
            require(0 <= b < spec.batch, f"{prefix} batch index out of bounds: {b}")
            require(
                physical == (0 if spec.layout == "TND" else b),
                f"{prefix} storage batch mismatch",
            )
            require(
                start == starts[b] and length == lengths[b],
                f"{prefix} batch[{b}] token range expected {starts[b]}/{lengths[b]}, got {start}/{length}",
            )
            require(
                chunk == next_chunk[b],
                f"{prefix} batch[{b}] chunk partition is not contiguous",
            )
            require(
                0 < count <= record["group_num"]
                and chunk + count <= (length + CHUNK_SIZE - 1) // CHUNK_SIZE,
                f"{prefix} batch[{b}] chunk range out of bounds",
            )
            next_chunk[b] += count
    for b, length in enumerate(lengths):
        require(
            next_chunk[b] == (length + CHUNK_SIZE - 1) // CHUNK_SIZE,
            f"metadata batch[{b}] chunk coverage incomplete",
        )
    return decoded


def validate_storage(name, value):
    """接受带步长、无重叠且底层存储容量充足的缓冲区。"""
    span = 1
    for stride, size in sorted(zip(value.stride(), value.shape)):
        if size > 1:
            require(stride >= span, f"{name} stride overlaps or is negative")
            span += (size - 1) * stride
    last = value.storage_offset() + sum(
        (size - 1) * stride for size, stride in zip(value.shape, value.stride())
    )
    require(
        value.storage_offset() >= 0
        and (last + 1) * value.element_size() <= value.untyped_storage().nbytes(),
        f"{name} accessible range exceeds storage capacity",
    )


def validate_buffer(name, value, shape, dtype, *, contiguous=True):
    tensor_spec(name, value, rank=len(shape), dtypes=(dtype,), shape=shape)
    if contiguous:
        validate_contiguous(name, value)
    validate_storage(name, value)


class OutputConditions(NamedTuple):
    final_state: bool
    chunk_states: bool
    matrices: bool
    caches: bool
    norm: bool
    beta: bool


def output_conditions(
    *,
    inference,
    output_final_state,
    return_intermediate_states,
    disable_recompute,
    use_qk_l2norm_in_kernel,
    use_beta_sigmoid_in_kernel,
):
    """返回设计第 4.9.4 节定义的各项独立输出条件。"""
    training = not inference
    return OutputConditions(
        final_state=output_final_state,
        chunk_states=(inference and return_intermediate_states)
        or (training and disable_recompute),
        matrices=training,
        caches=training and disable_recompute,
        norm=training and use_qk_l2norm_in_kernel,
        beta=training and use_beta_sigmoid_in_kernel,
    )
