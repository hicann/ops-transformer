# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Task metadata and all scheduling policies for fused FlashKDA backward.

The AICPU producer preserves the task-table ABI. Host policies select the
existing Native template and workspace; inline DSL policies map work to cores.
Both consumers share this module, including versioned plan/device-table caches.
"""

import os
import atexit
import heapq
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

from cannbotdsl import Constexpr, dtypes
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.arch import get_block_idx, get_block_num

from cannbotdsl.aicpu import gm_in, gm_out, I32, U32, aicpu_kernel, current_raw_stream
from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel, compile_aicpu_kernel


def tensor_version(tensor):
    """Inference tensors have no mutation counter and use the uncached path."""
    try:
        return tensor._version
    except RuntimeError:
        return None


class MetadataTensorCache:
    """Reuse read-only scheduling snapshots, guarded by input/output versions.

    Sequence-boundary tensors must not be changed through untracked raw-pointer
    writes. A different tensor object or a tracked mutation creates a new key.
    Weak input references prevent address reuse from hitting an unrelated entry.
    """

    def __init__(self, limit=8, byte_limit=8 * 1024 * 1024):
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.limit = limit
        self.byte_limit = byte_limit

    def get_or_create(self, key, offsets, factory):
        version = tensor_version(offsets) if offsets is not None else 0
        if version is None:
            # Uncached inference launches must also serialize with clear().
            with self.lock:
                return factory()
        key = (*key, id(offsets) if offsets is not None else None, version)
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                tensor, output_version, reference = entry
                if (reference is None or reference() is offsets) and tensor_version(
                    tensor
                ) == output_version:
                    self.entries.move_to_end(key)
                    return tensor
                del self.entries[key]
            tensor = factory()
            output_version = tensor_version(tensor)
            size = tensor.numel() * tensor.element_size()
            if output_version is not None and size <= self.byte_limit:
                self.entries[key] = (
                    tensor,
                    output_version,
                    weakref.ref(offsets) if offsets is not None else None,
                )
                while (
                    len(self.entries) > self.limit
                    or sum(
                        item[0].numel() * item[0].element_size()
                        for item in self.entries.values()
                    )
                    > self.byte_limit
                ):
                    self.entries.popitem(last=False)
            return tensor

    def clear(self):
        with self.lock:
            self.entries.clear()


CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
MAX_AIC_CORES = 32
STAGED_FULL_VALUE_ROWS = SUPPORTED_HEAD_DIM
STAGED_SPLIT_VALUE_ROWS = 16
MAGIC = 0x4B444142
ABI_VERSION = 1
HEADER_MAGIC = 0
HEADER_ABI_VERSION = 1
HEADER_STATUS = 2
HEADER_BATCH = 3
HEADER_VALUE_HEADS = 4
HEADER_STORAGE_LENGTH = 5
HEADER_CHUNK_COUNT = 6
HEADER_FLAGS = 7
HEADER_WORDS = 8
TASK_SEQUENCE = 0
TASK_COMPACT_TOKEN_START = 1
TASK_FIRST_CHUNK = 2
TASK_CHUNKS = 3
TASK_LENGTH = 4
TASK_VALUE_HEAD = 5
TASK_FLAT_CHUNK_START = 6
TASK_WORDS = 7
STATUS_OK = 0
STATUS_BAD_CU_START = 1
STATUS_BAD_LENGTH = 2
STATUS_BAD_PADDED_LENGTH = 3
STATUS_BAD_PACKED_END = 4
STATUS_BAD_METADATA_CAPACITY = 5
INT32_MAX = (1 << 31) - 1


def metadata_capacity(batch: int, value_heads: int) -> int:
    """Number of int32 words for one record per sequence and value head."""
    if batch < 1 or value_heads < 1:
        raise ValueError("batch and value_heads must be positive")
    capacity = HEADER_WORDS + batch * value_heads * TASK_WORDS
    if capacity > INT32_MAX:
        raise ValueError("metadata word capacity exceeds the int32 ABI")
    return capacity


class FlashKDAGradMetadataArgs:
    cu_seqlens: gm_in(I32)
    seqused: gm_in(I32)
    metadata: gm_out(U32)
    capacity: U32
    batch: U32
    value_heads: U32
    storage_length: U32
    has_cu: U32
    has_seqused: U32
    is_packed: U32


@aicpu_kernel
def flash_kda_grad_metadata_kernel(a: FlashKDAGradMetadataArgs):
    # Failed validation must not expose uninitialized task records to diagnostics.
    for word in range(0, a.capacity):
        a.metadata[word] = 0
    if a.capacity < HEADER_WORDS + a.batch * a.value_heads * TASK_WORDS:
        if a.capacity >= HEADER_WORDS:
            a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
        return STATUS_BAD_METADATA_CAPACITY
    a.metadata[HEADER_MAGIC] = MAGIC
    a.metadata[HEADER_ABI_VERSION] = ABI_VERSION
    a.metadata[HEADER_STATUS] = STATUS_OK
    a.metadata[HEADER_BATCH] = a.batch
    a.metadata[HEADER_VALUE_HEADS] = a.value_heads
    a.metadata[HEADER_STORAGE_LENGTH] = a.storage_length
    a.metadata[HEADER_CHUNK_COUNT] = 0
    a.metadata[HEADER_FLAGS] = a.has_cu + a.has_seqused + 2 * a.is_packed

    if a.has_cu == 1 and a.cu_seqlens[0] != 0:
        a.metadata[HEADER_STATUS] = STATUS_BAD_CU_START
        return STATUS_OK
    first_chunk = 0
    flat_chunk_start = 0
    token_offset = 0
    for sequence in range(0, a.batch):
        compact_token_start = sequence * a.storage_length
        length = a.storage_length
        if a.has_cu == 1:
            compact_token_start = a.cu_seqlens[sequence]
            token_end = a.cu_seqlens[sequence + 1]
            length = token_end - compact_token_start
        if a.has_seqused == 1:
            compact_token_start = token_offset
            length = a.seqused[sequence]
        if length < 0:
            a.metadata[HEADER_STATUS] = STATUS_BAD_LENGTH
            return STATUS_OK
        if a.is_packed == 0 and length > a.storage_length:
            a.metadata[HEADER_STATUS] = STATUS_BAD_PADDED_LENGTH
            return STATUS_OK
        if a.has_seqused == 1:
            if length > INT32_MAX - token_offset:
                a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
                return STATUS_OK
            token_offset += length
        chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        if first_chunk + chunks > INT32_MAX // a.value_heads:
            a.metadata[HEADER_STATUS] = STATUS_BAD_METADATA_CAPACITY
            return STATUS_OK
        for value_head in range(0, a.value_heads):
            record = HEADER_WORDS + (sequence * a.value_heads + value_head) * TASK_WORDS
            a.metadata[record + TASK_SEQUENCE] = sequence
            a.metadata[record + TASK_COMPACT_TOKEN_START] = compact_token_start
            a.metadata[record + TASK_FIRST_CHUNK] = first_chunk
            a.metadata[record + TASK_CHUNKS] = chunks
            a.metadata[record + TASK_LENGTH] = length
            a.metadata[record + TASK_VALUE_HEAD] = value_head
            a.metadata[record + TASK_FLAT_CHUNK_START] = flat_chunk_start
            flat_chunk_start += chunks
        first_chunk += chunks
    if a.is_packed == 1:
        if a.cu_seqlens[a.batch] != a.storage_length:
            a.metadata[HEADER_STATUS] = STATUS_BAD_PACKED_END
            return STATUS_OK
    a.metadata[HEADER_CHUNK_COUNT] = first_chunk
    return STATUS_OK


_COMPILED = None
_COMPILE_LOCK = threading.RLock()
_PENDING_EVENTS = []
_TENSOR_CACHE = MetadataTensorCache()
_HOST_INFO_LOCK = threading.RLock()
_READ_STREAMS = {}


@dataclass(frozen=True)
class MetadataHostInfo:
    header: tuple
    # One entry per sequence: token start, length, chunks and first chunk.
    sequences: tuple


def metadata_host_info(metadata):
    """Read a scheduling snapshot once; later consumers check its host fields.

    A private read stream waits only for a known producer's event. It must not
    synchronize unrelated work subsequently queued on the compute stream.
    Externally supplied tensors retain their original caller stream contract.
    """
    with _HOST_INFO_LOCK:
        version = tensor_version(metadata)
        cached = getattr(metadata, "_kda_metadata_host_info", None)
        if version is not None and cached is not None and cached[0] == version:
            return cached[1]
        ready = getattr(metadata, "_kda_metadata_ready_event", None)
        if (
            ready is None
            or version is None
            or version != getattr(metadata, "_kda_metadata_producer_version", None)
        ):
            words = metadata.cpu().tolist()
        else:
            with torch.npu.device(metadata.device):
                key = str(metadata.device)
                stream = _READ_STREAMS.get(key)
                if stream is None:
                    stream = torch.npu.Stream(device=metadata.device)
                    _READ_STREAMS[key] = stream
                with torch.npu.stream(stream):
                    stream.wait_event(ready)
                    metadata.record_stream(stream)
                    words = metadata.cpu().tolist()
        header = tuple(words[:HEADER_WORDS])
        if (
            len(header) != HEADER_WORDS
            or header[HEADER_MAGIC] != MAGIC
            or header[HEADER_ABI_VERSION] != ABI_VERSION
        ):
            raise ValueError("backward metadata ABI does not match")
        if header[HEADER_STATUS] != STATUS_OK:
            raise ValueError(f"backward metadata status {header[HEADER_STATUS]}")
        batch, heads = header[HEADER_BATCH], header[HEADER_VALUE_HEADS]
        if batch < 1 or heads < 1 or len(words) != metadata_capacity(batch, heads):
            raise ValueError("metadata task count is invalid")
        if header[HEADER_FLAGS] not in (0, 1, 3):
            raise ValueError("backward metadata input shape or layout does not match")
        sequences = []
        for sequence in range(batch):
            record = HEADER_WORDS + sequence * heads * TASK_WORDS
            sequences.append(
                (
                    words[record + TASK_COMPACT_TOKEN_START],
                    words[record + TASK_LENGTH],
                    words[record + TASK_CHUNKS],
                    words[record + TASK_FIRST_CHUNK],
                )
            )
        info = MetadataHostInfo(header, tuple(sequences))
        if version is not None:
            metadata._kda_metadata_host_info = (version, info)
        return info


@jit
def _owner_loop_bounds(gm_owner_schedule, groups):
    """A one-word view selects modulo; otherwise read this core's CSR range."""
    begin = dtypes.int64(get_block_idx())
    end = dtypes.int64(groups)
    step = dtypes.int64(get_block_num())
    if gm_owner_schedule.shape[0] > 1:
        begin = dtypes.int64(gm_owner_schedule[get_block_idx()])
        end = dtypes.int64(gm_owner_schedule[get_block_idx() + 1])
        step = dtypes.int64(1)
        if groups == 0:
            end = begin
    return begin, end, step


@jit
def _scheduled_owner(gm_owner_schedule, slot):
    owner = dtypes.int64(slot)
    if gm_owner_schedule.shape[0] > 1:
        owner = dtypes.int64(gm_owner_schedule[get_block_num() + 1 + slot])
    return owner


@jit
def _staged_chunk_start(metadata, record, heads_per_key_head, block_idx, block_num):
    """Identical chunk ownership in stages 1/3/4; GQA heads stay on one core."""
    phase = metadata[record + TASK_FIRST_CHUNK]
    if heads_per_key_head == 1:
        phase = phase + metadata[record + TASK_VALUE_HEAD]
    return (block_idx + block_num - phase % block_num) % block_num


@jit
def _staged_reverse_is_unsplit(task_count, block_num):
    return task_count >= block_num


@jit
def _staged_reverse_loop_bounds(task_count, value_parts: Constexpr[int]):
    return get_block_idx(), task_count * value_parts, get_block_num()


@jit
def _staged_reverse_owner(owner, value_parts: Constexpr[int]):
    return owner // value_parts, owner % value_parts


_OWNER_PLAN_CACHE_SIZE = 32
_OWNER_DEVICE_CACHE_SIZE = 16
_OWNER_PLAN_LOCK = threading.RLock()
_OWNER_DEVICE_PLANS = OrderedDict()
_OWNER_PENDING_UPLOADS = []


@dataclass(frozen=True)
class _PersistentOwnerPlan:
    """Immutable estimates plus optional CSR offsets and original owner IDs."""

    modulo_loads: tuple
    loads: tuple
    words: tuple = ()

    @property
    def strategy(self):
        return "persistent" if self.balanced else "staged"

    @property
    def balanced(self):
        return sum(self.loads) * 10 >= max(self.loads) * len(self.loads) * 9


def _modulo_owner_loads(weights, heads, cores):
    """Accumulate repeated head weights in O(B + C), including wraparound."""
    rounds, remainder = divmod(heads, cores)
    base = sum(weights) * rounds
    delta = [0] * (cores + 1)
    if remainder:
        for batch, weight in enumerate(weights):
            start = batch * remainder % cores
            end = start + remainder
            delta[start] += weight
            delta[min(end, cores)] -= weight
            if end > cores:
                delta[0] += weight
                delta[end - cores] -= weight
    loads = []
    for core in range(cores):
        base += delta[core]
        loads.append(base)
    return tuple(loads)


def _build_persistent_owner_plan(weights, heads, cores):
    modulo = _modulo_owner_loads(weights, heads, cores)
    original = _PersistentOwnerPlan(modulo, modulo)
    owners = len(weights) * heads
    # Empty sequences still transmit dht/clear padding; weights are positive.
    # Skip sorting when modulo is sufficient, each owner already has its own
    # core, or an indivisible owner rules out the 90% admission target.
    # Fewer owners than cores only skips planning; admission uses balance.
    # Planning has no owner-count ceiling.
    if (
        original.balanced
        or owners < cores
        or sum(modulo) * 10 < max(weights) * cores * 9
    ):
        return original
    order = sorted(range(len(weights)), key=lambda batch: (-weights[batch], batch))
    heap = [(0, 0, core) for core in range(cores)]
    owner_core = bytearray(owners)
    counts = [0] * cores
    loads = [0] * cores
    for batch in order:
        weight = weights[batch]
        for head in range(heads):
            load, count, core = heap[0]
            owner_core[batch * heads + head] = core
            counts[core] += 1
            loads[core] = load + weight
            heapq.heapreplace(heap, (loads[core], count + 1, core))
    # The greedy plan is admitted solely by the 90% estimated balance target.
    if sum(loads) * 10 < max(loads) * cores * 9:
        return original
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    cursor = offsets[:-1].copy()
    owner_ids = [0] * owners
    for batch in order:
        for head in range(heads):
            owner = batch * heads + head
            core = owner_core[owner]
            owner_ids[cursor[core]] = owner
            cursor[core] += 1
    return _PersistentOwnerPlan(modulo, tuple(loads), tuple(offsets + owner_ids))


_cached_persistent_owner_plan = lru_cache(maxsize=_OWNER_PLAN_CACHE_SIZE)(
    _build_persistent_owner_plan
)


def backward_schedule(metadata, heads, value_heads, core_num):
    """Reuse validated metadata snapshots; never cache by pointer/shape alone."""
    if (
        isinstance(core_num, bool)
        or not isinstance(core_num, int)
        or not 1 <= core_num <= MAX_AIC_CORES
    ):
        raise ValueError(f"core_num must be in [1, {MAX_AIC_CORES}]")
    if (
        isinstance(heads, bool)
        or isinstance(value_heads, bool)
        or not isinstance(heads, int)
        or not isinstance(value_heads, int)
        or heads < 1
        or value_heads < 1
        or value_heads % heads
    ):
        raise ValueError("value_heads must be a positive multiple of query heads")
    info = metadata_host_info(metadata)
    if info.header[HEADER_VALUE_HEADS] != value_heads:
        raise ValueError("schedule value heads do not match metadata")
    key = (heads, value_heads, core_num)
    with _OWNER_PLAN_LOCK:
        cached = getattr(metadata, "_kda_backward_owner_plan", None)
        if cached is not None and cached[0] is info and cached[1] == key:
            return cached[2]
        ratio = value_heads // heads
        weights = tuple(max(chunks, 1) * ratio for _, _, chunks, _ in info.sequences)
        plan = _cached_persistent_owner_plan(weights, heads, core_num)
        metadata._kda_backward_owner_plan = (info, key, plan)
        return plan


def owner_schedule_for_launch(metadata, heads, value_heads, core_num, staged):
    """Upload immutable CSR only for selected persistent plans; reuse across streams."""
    if staged:
        return metadata[:1]
    plan = backward_schedule(metadata, heads, value_heads, core_num)
    if not plan.words:
        # Shape, not contents, is the device-side modulo sentinel. No allocation
        # or host-to-device copy is needed, and metadata itself is not changed.
        return metadata[:1]
    if metadata.device.type not in {"npu", "privateuseone"}:
        return torch.tensor(plan.words, dtype=torch.int32, device=metadata.device)
    with torch.npu.device(metadata.device), _OWNER_PLAN_LOCK:
        stream = torch.npu.current_stream(metadata.device)
        key = (str(metadata.device), id(plan))
        entry = _OWNER_DEVICE_PLANS.get(key)
        if entry is None:
            source = torch.tensor(plan.words, dtype=torch.int32, pin_memory=True)
            table = source.to(metadata.device, non_blocking=True)
            ready = torch.npu.Event()
            ready.record(stream)
            # Keep the source until DMA completes, even if the LRU is evicted.
            _OWNER_PENDING_UPLOADS.append((metadata.device, ready, source))
            entry = (plan, table, ready)
            _OWNER_DEVICE_PLANS[key] = entry
            while len(_OWNER_DEVICE_PLANS) > _OWNER_DEVICE_CACHE_SIZE:
                _OWNER_DEVICE_PLANS.popitem(last=False)
        _OWNER_DEVICE_PLANS.move_to_end(key)
        _, table, ready = entry
        stream.wait_event(ready)
        table.record_stream(stream)
        _OWNER_PENDING_UPLOADS[:] = [
            item for item in _OWNER_PENDING_UPLOADS if not item[1].query()
        ]
        return table


def clear_schedule_caches():
    with _OWNER_PLAN_LOCK:
        for device, event, _ in _OWNER_PENDING_UPLOADS:
            with torch.npu.device(device):
                event.synchronize()
        _OWNER_PENDING_UPLOADS.clear()
        _OWNER_DEVICE_PLANS.clear()
        _cached_persistent_owner_plan.cache_clear()


atexit.register(clear_schedule_caches)


def backward_chunk_paths(metadata, query_heads, value_heads, block_size, staged=False):
    """Return fast-chunk eligibility and whether masked workspace is needed."""
    if staged:
        return False, True
    use_full_chunks = query_heads == value_heads and block_size > 1
    # Masked storage handles tails and unsupported full-chunk epilogues.
    needs_masked_chunks = not use_full_chunks or block_size != CHUNK_SIZE
    if use_full_chunks and block_size == CHUNK_SIZE:
        info = metadata_host_info(metadata)
        needs_masked_chunks = any(
            length == 0 or length % CHUNK_SIZE != 0
            for _, length, _, _ in info.sequences
        )
    return use_full_chunks, needs_masked_chunks


def backward_gradient_slots(metadata, value_heads, core_num, staged):
    """Persistent reuses per-core slots; staged retains every chunk and empty task."""
    if not staged:
        return core_num
    info = metadata_host_info(metadata)
    return sum(chunks + 1 for _, _, chunks, _ in info.sequences) * value_heads


def build_packaged_aicpu(output, target):
    """Build the shape-independent metadata binary for the operator wheel."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compiled = compile_aicpu_kernel(
        flash_kda_grad_metadata_kernel,
        workdir=str(output),
        launch_mode="interface",
        npu_arch=target,
    )
    binary = Path(compiled.so_path)
    manifest = dict(
        schema=1,
        target=target,
        name=flash_kda_grad_metadata_kernel.fn.__name__,
        launch_mode="interface",
        metadata_abi=ABI_VERSION,
        binary=binary.name,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Ship the linked binary and its manifest, not compiler intermediates.
    for path in output.iterdir():
        if path.name not in (binary.name, "manifest.json") and path.is_file():
            path.unlink()


def _load_packaged_kernel():
    mode = os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE", "prefer")
    if mode not in ("prefer", "require", "off"):
        raise ValueError(f"invalid CANNBOTDSL_NATIVE_BINARY_MODE: {mode}")
    root = Path(__file__).parent / "_aicpu" / "flash_kda_grad_metadata"
    if mode == "off":
        return None
    if not root.exists():
        if mode == "require":
            raise RuntimeError(
                "precompiled FlashKDA backward AICPU metadata is required but not installed"
            )
        return None
    manifest = json.loads((root / "manifest.json").read_text())
    expected_name = flash_kda_grad_metadata_kernel.fn.__name__
    if (
        manifest.get("schema") != 1
        or manifest.get("name") != expected_name
        or manifest.get("metadata_abi") != ABI_VERSION
        or manifest.get("launch_mode") != "interface"
        or manifest.get("target") not in ("dav-3510", "dav-2201")
        or manifest.get("source_sha256")
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise RuntimeError("incompatible precompiled FlashKDA backward AICPU metadata")
    target = os.environ.get("CANNBOTDSL_NPU_ARCH")
    if target and target != manifest["target"]:
        raise RuntimeError(
            "precompiled FlashKDA backward AICPU metadata target mismatch"
        )
    filename = manifest.get("binary")
    if not isinstance(filename, str) or filename != expected_name + ".so":
        raise RuntimeError(
            "invalid precompiled FlashKDA backward AICPU metadata binary name"
        )
    binary = root / filename
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest.get("binary_sha256"):
        raise RuntimeError(
            "precompiled FlashKDA backward AICPU metadata checksum mismatch"
        )
    return CompiledAicpuKernel(
        str(binary),
        flash_kda_grad_metadata_kernel.spec,
        name=expected_name,
        launch_mode="interface",
    )


def _get_compiled_kernel():
    global _COMPILED
    with _COMPILE_LOCK:
        if _COMPILED is None:
            _COMPILED = _load_packaged_kernel()
        if _COMPILED is None:
            workdir = tempfile.mkdtemp(prefix="flash_kda_grad_metadata_aicpu_")
            _COMPILED = compile_aicpu_kernel(
                flash_kda_grad_metadata_kernel, workdir=workdir, launch_mode="interface"
            )
        return _COMPILED


def _validated_metadata_launch_shape(
    cu_seqlens,
    seqused,
    batch_size,
    n_heads_qk,
    n_heads_v,
    seqlens,
    headdim_qk,
    headdim_v,
    layout,
):
    """Validate explicit attributes and lengths before allocating metadata."""
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError("layout must be TND, BNSD, or BSND")
    for name, value, minimum in (
        ("batch_size", batch_size, 1),
        ("n_heads_qk", n_heads_qk, 1),
        ("n_heads_v", n_heads_v, 1),
        ("seqlens", seqlens, 0),
        ("headdim_qk", headdim_qk, 1),
        ("headdim_v", headdim_v, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= INT32_MAX
        ):
            raise ValueError(
                f"{name} must be an int32 integer in [{minimum}, {INT32_MAX}]"
            )
    if n_heads_v % n_heads_qk:
        raise ValueError("n_heads_v must be divisible by n_heads_qk")
    if headdim_qk != SUPPORTED_HEAD_DIM or headdim_v != SUPPORTED_HEAD_DIM:
        raise ValueError(f"headdim_qk and headdim_v must be {SUPPORTED_HEAD_DIM}")
    if cu_seqlens is not None and seqused is not None:
        raise ValueError("cu_seqlens and seqused are mutually exclusive")
    if seqused is not None and layout == "TND":
        raise ValueError("seqused is only supported for BNSD/BSND; TND uses cu_seqlens")
    if cu_seqlens is not None and layout != "TND":
        raise ValueError("cu_seqlens is only supported for TND; BNSD/BSND use seqused")
    if layout == "TND" and cu_seqlens is None:
        raise ValueError("TND requires cu_seqlens")
    lengths = cu_seqlens if cu_seqlens is not None else seqused
    if lengths is not None:
        name = "cu_seqlens" if cu_seqlens is not None else "seqused"
        count = batch_size + 1 if cu_seqlens is not None else batch_size
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.shape != (count,)
            or lengths.dtype != torch.int32
            or lengths.device.type != "npu"
            or not lengths.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous int32 [{count}] Tensor on NPU"
            )
    capacity = metadata_capacity(batch_size, n_heads_v)
    if lengths is None and batch_size * seqlens > INT32_MAX:
        raise ValueError("sequence storage exceeds the int32 metadata ABI")
    device = (
        lengths.device if lengths is not None else f"npu:{torch.npu.current_device()}"
    )
    return capacity, device


def flash_kda_grad_metadata(
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    batch_size: int,
    n_heads_qk: int,
    n_heads_v: int,
    seqlens: int,
    headdim_qk: int = 128,
    headdim_v: int = 128,
    layout: str = "TND",
) -> torch.Tensor:
    """Build int32 task records from explicit dimensions and sequence lengths.

    seqlens is storage length S for BNSD/BSND and total tokens T for TND.
    TND requires cu_seqlens [B+1]; BNSD/BSND accept seqused [B], with None
    meaning every sequence uses S tokens. The two length inputs are exclusive.
    Output uses the length tensor's NPU, or the current NPU if both are None.
    Regenerate the snapshot after changing dimensions or sequence boundaries.
    """
    capacity, device = _validated_metadata_launch_shape(
        cu_seqlens,
        seqused,
        batch_size,
        n_heads_qk,
        n_heads_v,
        seqlens,
        headdim_qk,
        headdim_v,
        layout,
    )
    result = _TENSOR_CACHE.get_or_create(
        (
            str(device),
            layout,
            batch_size,
            n_heads_qk,
            n_heads_v,
            seqlens,
            headdim_qk,
            headdim_v,
            seqused is not None,
        ),
        seqused if seqused is not None else cu_seqlens,
        lambda: _launch_metadata(
            device,
            cu_seqlens,
            capacity,
            batch_size,
            n_heads_v,
            seqlens,
            layout == "TND",
            seqused=seqused,
        ),
    )
    wait_metadata_ready(result)
    return result


def _launch_metadata(
    device,
    cu_seqlens,
    capacity,
    batch,
    value_heads,
    storage_length,
    is_packed,
    *,
    seqused=None,
):
    """Submit the AICPU task and retain its allocator and executable lifetimes."""
    with torch.npu.device(device), _COMPILE_LOCK:
        device_index = torch.npu.current_device()
        stream = torch.npu.current_stream(device_index)
        metadata = torch.empty(capacity, dtype=torch.int32, device=device)
        # Raw-pointer launches do not themselves extend tensor allocator lifetimes.
        if cu_seqlens is not None:
            cu_seqlens.record_stream(stream)
        if seqused is not None:
            seqused.record_stream(stream)
        metadata.record_stream(stream)
        ready = torch.npu.Event()
        _PENDING_EVENTS[:] = [event for event in _PENDING_EVENTS if not event.query()]
        try:
            _get_compiled_kernel().launch(
                current_raw_stream(device_index),
                cu_seqlens=0 if cu_seqlens is None else cu_seqlens.data_ptr(),
                seqused=0 if seqused is None else seqused.data_ptr(),
                metadata=metadata.data_ptr(),
                capacity=capacity,
                batch=batch,
                value_heads=value_heads,
                storage_length=storage_length,
                has_cu=int(cu_seqlens is not None),
                has_seqused=int(seqused is not None),
                is_packed=int(is_packed),
            )
            ready.record(stream)
        except Exception:
            # No completion event is available: retire any submitted work
            # synchronously before a caller can clear the executable cache.
            stream.synchronize()
            raise
        _PENDING_EVENTS.append(ready)
        metadata._kda_metadata_ready_event = ready
        metadata._kda_metadata_producer_version = tensor_version(metadata)
        return metadata


def wait_metadata_ready(metadata):
    """Order the consumer stream after this module's AICPU producer, if present.

    Externally constructed metadata retains the caller's stream-dependency
    contract; copying a tensor does not transfer the producer's Python event.
    """
    with torch.npu.device(metadata.device):
        stream = torch.npu.current_stream(torch.npu.current_device())
        ready = getattr(metadata, "_kda_metadata_ready_event", None)
        if ready is not None:
            stream.wait_event(ready)
        metadata.record_stream(stream)


def clear_metadata_caches():
    """Wait for submitted AICPU tasks before releasing their executable."""
    global _COMPILED
    clear_schedule_caches()
    with _TENSOR_CACHE.lock, _COMPILE_LOCK:
        _TENSOR_CACHE.clear()
        for event in _PENDING_EVENTS:
            event.synchronize()
        _PENDING_EVENTS.clear()
        if _COMPILED is not None:
            close = getattr(_COMPILED, "close", None)
            if callable(close):
                close()
            _COMPILED = None


__all__ = [
    "flash_kda_grad_metadata",
    "metadata_capacity",
    "backward_schedule",
    "owner_schedule_for_launch",
    "backward_chunk_paths",
    "backward_gradient_slots",
    "clear_schedule_caches",
    "clear_metadata_caches",
    "HEADER_WORDS",
    "TASK_WORDS",
    "STATUS_OK",
]
