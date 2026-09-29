#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden and customize_inputs for aclnnFfnWorkerBatching.

This module combines:
  - inputs: real-device-address scheduleContext (HBM buffers + test data)
  - golden: CPU golden reference using cached test data
  - compare: custom comparison (kernel writes only actual_token_num elements)

Auxiliary buffers (token_data / token_info / flat ids) live in dedicated
aclrtMalloc allocations and the context carries absolute device addresses,
matching the production double-dereference path (no TEST_MAGIC relative
offsets). RECV consumes descriptors on device, so its TTK execution count is
pinned to 1 per case (TTK's default 3 would hang on the second run).
"""

import atexit
import ctypes
import json
import os
import struct
import numpy as np
import torch

__spec__ = {
    "aclnnFfnWorkerBatching": "AclnnFfnWorkerBatchingLegacyTestSpec",
    "aclnnFfnWorkerBatchingV2": "AclnnFfnWorkerBatchingTestSpec",
    "ffn_worker_batching": "FfnWorkerBatchingKernelTestSpec",
    "torch.ops.cann_ops_transformer.ffn_worker_batching": "TorchFfnWorkerBatchingTestSpec",
}

# ==================== Shared cache ====================

_DATA_CACHE = {}
_HBM_ALLOCS = []
_HBM_VIEWS = {}  # actual suballocation address -> payload size, for replay validation
_ACL_LIB = None
_MASK_VALUE = 1000000
# ScheduleContext ABI v1, defined by common/include/op_kernel/attention_ffn_schedule.h.
_SCHEDULE_CONTEXT_BYTES = 1024
_CONTEXT_ARCHIVE_VERSION = 1
_TOKEN_STRIDE_ALIGNMENT = 512
_HBM_ALLOCATION_ALIGNMENT = (
    512  # Allocation alignment and row stride are separate contracts.
)
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_NUM_OUTPUTS = 8
_OUT_Y = 0
_OUT_GROUP_LIST = 1
_OUT_SESSION_IDS = 2
_OUT_MICRO_BATCH_IDS = 3
_OUT_TOKEN_IDS = 4
_OUT_EXPERT_OFFSETS = 5
_OUT_SCALE = 6
_OUT_TOKEN_COUNT = 7
_TOKEN_PREFIX_OUTPUTS = (
    _OUT_Y,
    _OUT_SESSION_IDS,
    _OUT_MICRO_BATCH_IDS,
    _OUT_TOKEN_IDS,
    _OUT_EXPERT_OFFSETS,
    _OUT_SCALE,
)

# ---- ScheduleContext byte offsets ----
_OFF_SESSION_NUM = 0
_OFF_MICRO_BATCH_NUM = 4
_OFF_MICRO_BATCH_SIZE = 8
_OFF_SELECTED_EXPERT_NUM = 12
_OFF_EXPERT_NUM = 16
_OFF_ATTN_TO_FFN_TOKEN_SIZE = 20
_OFF_FFN_TO_ATTN_TOKEN_SIZE = 24
_OFF_SCHEDULE_MODE = 28
_OFF_RUN_FLAG = 128
_OFF_FFN_TOKEN_INFO_BUF = 384
_OFF_FFN_TOKEN_INFO_BUF_SIZE = 392
_OFF_FFN_TOKEN_DATA_BUF = 400
_OFF_FFN_TOKEN_DATA_BUF_SIZE = 408
_OFF_POLLING_INDEX = 416
_OFF_SESSION_IDS_BUF = 528
_OFF_SESSION_IDS_BUF_SIZE = 536
_OFF_MICRO_BATCH_IDS_BUF = 544
_OFF_MICRO_BATCH_IDS_BUF_SIZE = 552
_OFF_EXPERT_IDS_BUF = 560
_OFF_EXPERT_IDS_BUF_SIZE = 568
_OFF_OUT_NUM = 576

# ==================== ACL / HBM helpers ====================


def _load_acl():
    global _ACL_LIB
    if _ACL_LIB is not None:
        return _ACL_LIB
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMalloc.restype = ctypes.c_int32
    lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int32,
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int32
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
    ]
    lib.aclrtFree.restype = ctypes.c_int32
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtSetDevice.restype = ctypes.c_int32
    lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtGetDevice.restype = ctypes.c_int32
    lib.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    lib.aclrtSynchronizeDevice.restype = ctypes.c_int32
    lib.aclrtSynchronizeDevice.argtypes = []
    lib.aclrtMemFlush.restype = ctypes.c_int32
    lib.aclrtMemFlush.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.aclrtMemInvalidate.restype = ctypes.c_int32
    lib.aclrtMemInvalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _ACL_LIB = lib
    return lib


_HBM_BASE = None
_HBM_OFFSET = 0
_HBM_CAP = 0
_HBM_BLOCK_SIZE = 4 * 1024 * 1024


def _ensure_device(lib):
    # 沿用进程内已设置的当前设备（TTK/torch_npu 可能已初始化），否则落到 0 卡。
    dev = ctypes.c_int32(-1)
    if lib.aclrtGetDevice(ctypes.byref(dev)) == 0 and dev.value >= 0:
        return
    lib.aclrtSetDevice(0)


def _hbm_alloc(size):
    global _HBM_BASE, _HBM_OFFSET, _HBM_CAP
    lib = _load_acl()
    _ensure_device(lib)
    aligned_size = (size + (_HBM_ALLOCATION_ALIGNMENT - 1)) & ~(
        _HBM_ALLOCATION_ALIGNMENT - 1
    )
    if _HBM_BASE is None or _HBM_OFFSET + aligned_size > _HBM_CAP:
        # 超过默认块大小的缓冲（如 a4/h8192 的 token_data）整块独立分配
        cap = max(_HBM_BLOCK_SIZE, aligned_size)
        ptr = ctypes.c_void_p(0)
        ret = lib.aclrtMalloc(ctypes.byref(ptr), cap, 0)
        if ret != 0 or ptr.value is None:
            raise RuntimeError(f"aclrtMalloc failed: ret={ret}, size={cap}")
        _HBM_BASE = ptr.value
        _HBM_OFFSET = 0
        _HBM_CAP = cap
        _HBM_ALLOCS.append(_HBM_BASE)
    addr = _HBM_BASE + _HBM_OFFSET
    _HBM_OFFSET += aligned_size
    _HBM_VIEWS[addr] = size
    return addr


def _hbm_copy(host_data, hbm_ptr):
    lib = _load_acl()
    if isinstance(host_data, np.ndarray):
        buf = host_data.tobytes()
    elif isinstance(host_data, (bytes, bytearray)):
        buf = bytes(host_data)
    else:
        buf = host_data
    if not len(buf):
        return
    ret = lib.aclrtMemcpy(
        hbm_ptr, len(buf), ctypes.c_char_p(buf), len(buf), _ACL_MEMCPY_HOST_TO_DEVICE
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy H2D failed: ret={ret}")
    lib.aclrtMemInvalidate(ctypes.c_void_p(hbm_ptr), len(buf))


@atexit.register
def _cleanup_hbm():
    global _HBM_BASE, _HBM_OFFSET, _HBM_CAP
    lib = _ACL_LIB
    if lib is None:
        return
    for ptr in _HBM_ALLOCS:
        try:
            lib.aclrtFree(ctypes.c_void_p(ptr))
        except Exception:
            pass
    _HBM_ALLOCS.clear()
    _HBM_VIEWS.clear()
    _HBM_BASE = None
    _HBM_OFFSET = 0
    _HBM_CAP = 0


def _context_buffer_offsets(need_schedule):
    return (
        (_OFF_FFN_TOKEN_DATA_BUF, _OFF_FFN_TOKEN_INFO_BUF)
        if need_schedule
        else (
            _OFF_FFN_TOKEN_DATA_BUF,
            _OFF_SESSION_IDS_BUF,
            _OFF_MICRO_BATCH_IDS_BUF,
            _OFF_EXPERT_IDS_BUF,
        )
    )


def validate_live_context(header, need_schedule=0):
    """Reject a plain dump from another process before dereferencing its pointers."""
    header = np.asarray(header, dtype=np.int8).reshape(-1)
    if header.size < _SCHEDULE_CONTEXT_BYTES:
        raise ValueError("schedule_context needs at least 1024 bytes")
    for offset in _context_buffer_offsets(need_schedule):
        pointer, size = struct.unpack_from("<QQ", header, offset)
        if not size or _HBM_VIEWS.get(pointer) != size:
            raise ValueError(
                "raw context replay has no live secondary buffers; "
                "save with FFN_WB_SAVE_ARCHIVE and restore with FFN_WB_REPLAY_ARCHIVE"
            )


def save_context_archive(path, header, attributes):
    """Save payloads as well as metadata; archived addresses are deliberately zero.

    Only buffers owned by this TestSpec can be saved. This prevents accidentally
    reading stale addresses from an ordinary input dump. Synchronize before D2H
    so the archive represents completed input preparation, not queued writes.
    """
    header = (
        np.asarray(header, dtype=np.int8).reshape(-1)[:_SCHEDULE_CONTEXT_BYTES].copy()
    )
    mode = attributes["need_schedule"]
    validate_live_context(header, mode)
    lib = _load_acl()
    ret = lib.aclrtSynchronizeDevice()
    if ret:
        raise RuntimeError(f"archive synchronize failed: {ret}")
    records = {
        "version": np.array([_CONTEXT_ARCHIVE_VERSION], dtype=np.int64),
        "attributes": np.frombuffer(
            json.dumps(attributes, sort_keys=True).encode(), dtype=np.uint8
        ),
    }
    for offset in _context_buffer_offsets(mode):
        pointer, size = struct.unpack_from("<QQ", header, offset)
        payload = np.empty(size, dtype=np.uint8)
        ret = lib.aclrtMemcpy(
            payload.ctypes.data, size, pointer, size, _ACL_MEMCPY_DEVICE_TO_HOST
        )
        if ret:
            raise RuntimeError(f"archive D2H failed: {ret}")
        records[f"buffer_{offset}"] = payload
        struct.pack_into("<Q", header, offset, 0)
    records["header"] = header
    with open(path, "wb") as stream:
        np.savez(stream, **records)


def restore_context_archive(path, expected_attributes=None):
    """Allocate fresh HBM and relocate the header; never reuse saved addresses."""
    with np.load(path, allow_pickle=False) as archive:
        if archive["version"].tolist() != [_CONTEXT_ARCHIVE_VERSION]:
            raise ValueError("unsupported FFN context archive version")
        attributes = json.loads(archive["attributes"].tobytes())
        if expected_attributes is not None and attributes != expected_attributes:
            raise ValueError("FFN context archive attributes mismatch")
        header = archive["header"].copy()
        if header.dtype != np.int8 or header.shape != (_SCHEDULE_CONTEXT_BYTES,):
            raise ValueError("invalid FFN context archive header")
        # Validate every payload before starting any allocation/copy.
        offsets = _context_buffer_offsets(attributes["need_schedule"])
        for offset in offsets:
            pointer, size = struct.unpack_from("<QQ", header, offset)
            payload = archive[f"buffer_{offset}"]
            if (
                pointer != 0
                or size <= 0
                or payload.dtype != np.uint8
                or payload.shape != (size,)
            ):
                raise ValueError("invalid FFN context archive buffer")
        for offset in offsets:
            payload = archive[f"buffer_{offset}"]
            pointer = _hbm_alloc(payload.size)
            _hbm_copy(payload, pointer)
            struct.pack_into("<Q", header, offset, pointer)
    return header, attributes


_ORIG_RUN_TIME = None


def _pin_run_time(need_schedule):
    """RECV 在设备侧消费描述符（清 flag / ids 置 sentinel），真实地址缓冲不会随
    TTK 的重复执行刷新：第二次执行面对空表会永久等待。TTK 默认 run_time=3，
    这里把 RECV 用例钉到单次执行，NORM 恢复原值。依赖 TTK 逐用例
    “输入生成先于执行器读取 run_time”的时序（aclnn / kernel 两模式均如此）。"""
    global _ORIG_RUN_TIME
    try:
        from ttk.utilities.container_utils import get_global_storage

        storage = get_global_storage()
        if _ORIG_RUN_TIME is None:
            _ORIG_RUN_TIME = storage.run_time
        storage.run_time = 1 if need_schedule == 1 else _ORIG_RUN_TIME
    except Exception:
        pass


# ==================== Data generation ====================


def _seed_from_name(name):
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _pack_ctx(ctx_bytes, offset, fmt, value):
    struct.pack_into(fmt, ctx_bytes, offset, value)


def _payload_ranges(input_ranges, token_dtype):
    """Decode token/scale bounds; optional expert bounds are applied by the generator.

    TTK only exposes one input tensor here. Its first range is forwarded intact
    even for ACLNN (where it may be broadcast to output tensor placeholders).
    Temporary context bytes generated by TTK are replaced by customize_inputs.
    """
    if not input_ranges or input_ranges[0] is None:
        raise ValueError(
            "input_data_ranges must explicitly specify token and scale payloads"
        )
    values = input_ranges[0]
    if len(values) not in (2, 4, 6, 8):
        raise ValueError(
            "FFN input_data_ranges expects "
            "2/4/6/8 values: token bounds, scale bounds, expert bounds, token/scale modes"
        )

    def bounds(pair, label):
        if all(v is None for v in pair):
            return None
        if any(v is None for v in pair):
            raise ValueError(f"{label} range must specify both bounds or neither")
        low, high = map(float, pair)
        if np.isnan(low) and np.isnan(high):
            return low, high
        if (
            np.isnan([low, high]).any()
            or low > high
            or (not np.isfinite([low, high]).all() and low != high)
        ):
            raise ValueError(
                f"{label} range requires finite min <= max or identical special values, got {pair}"
            )
        return low, high

    token_range = bounds(values[:2], "token")
    scale_range = bounds(values[2:4], "scale") if len(values) >= 4 else None
    if token_dtype < 2 and scale_range is not None:
        raise ValueError(
            "FP16/BF16 tokens have no scale payload; specify only token bounds"
        )
    if token_range is None or (token_dtype >= 2 and scale_range is None):
        raise ValueError("token and existing scale payloads require explicit bounds")
    token_mode, scale_mode = values[6:8] if len(values) == 8 else (0, 0)
    if token_mode not in (0, 1) or scale_mode not in (0, 1):
        raise ValueError("payload modes must be 0 (numeric) or 1 (raw encoding)")
    if token_dtype < 3 and (token_mode or scale_mode):
        raise ValueError(
            "raw encoding mode is only supported for MX token/scale payloads"
        )
    return token_range, scale_range, token_mode, scale_mode


def _numeric_samples(rng, shape, bounds, dtype, label):
    """Generate numeric payloads using the target format's sampling contract.

    FP8 token bounds describe source values: sample in FP32, then convert to
    E4M3FN/E5M2 with the dtype's native rounding, including for constants.
    Device packing and Golden both consume these converted, cached tokens.
    Other formats retain their representable-value contract, especially E8M0
    scales, whose finite values are positive powers of two. BF16 is returned
    as exact FP32 values and converted by the existing packing path.
    """
    low, high = bounds
    if (
        label == "token"
        and dtype != "bfloat16"
        and np.dtype(dtype).name in ("float8_e4m3fn", "float8_e5m2")
    ):
        if low == high or (np.isnan(low) and np.isnan(high)):
            source = np.full(shape, low, dtype=np.float32)
        else:
            source = rng.uniform(float(low), float(high), shape).astype(np.float32)
        # No exact-value check or clipping after conversion: e.g. 127 becomes
        # 128. Overflow and special values follow the target dtype's cast rules.
        with np.errstate(over="ignore", invalid="ignore"):
            return source.astype(dtype)
    # Equal bounds specify one exact constant, including signed zero and specials.
    if low == high or (np.isnan(low) and np.isnan(high)):
        target = np.float32 if dtype == "bfloat16" else dtype
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.asarray(low).astype(target)
            numeric = float(value)
        matches = np.isnan(numeric) if np.isnan(low) else numeric == low
        if dtype == "bfloat16" and np.isfinite(low):
            bits = value.view(np.uint32).item()
            matches = matches and (bits & 0xFFFF) == 0
        if not matches or (low == 0 and np.signbit(numeric) != np.signbit(low)):
            raise ValueError(f"{label} constant {low} is not representable as {dtype}")
        return np.full(shape, value, dtype=target)
    if dtype == "bfloat16":
        values = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    elif np.dtype(dtype) == np.dtype(np.float32):
        limit = np.finfo(np.float32)
        low, high = max(low, float(limit.min)), min(high, float(limit.max))
        if low > high:
            raise ValueError(f"{label} range {bounds} contains no finite {dtype} value")
        lo, hi = np.float32(low), np.float32(high)
        if float(lo) < low:
            lo = np.nextafter(lo, np.float32(np.inf))
        if float(hi) > high:
            hi = np.nextafter(hi, np.float32(-np.inf))
        if lo > hi:
            raise ValueError(
                f"{label} range {bounds} contains no representable {dtype} value"
            )
        return np.clip(
            rng.uniform(float(lo), float(hi), shape).astype(np.float32), lo, hi
        )
    elif np.dtype(dtype) == np.dtype(np.int8):
        values = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    elif np.dtype(dtype) == np.dtype(np.float16):
        values = np.arange(65536, dtype=np.uint16).view(np.float16)
    else:
        # FP8 / E8M0 use one byte; unpacked FP4 uses its low four bits.
        count = 16 if "float4" in np.dtype(dtype).name else 256
        values = np.arange(count, dtype=np.uint8).view(dtype)
    # Encoding tables include signalling NaNs; they are deliberately filtered out.
    with np.errstate(invalid="ignore"):
        numeric = values.astype(np.float64)
    candidates = values[np.isfinite(numeric) & (numeric >= low) & (numeric <= high)]
    if candidates.size == 0:
        raise ValueError(
            f"{label} range {bounds} contains no finite representable {dtype} value"
        )
    return candidates[rng.randint(0, candidates.size, size=shape)]


def _payload_samples(rng, shape, bounds, dtype, label, mode):
    if mode == 0:
        return _numeric_samples(rng, shape, bounds, dtype, label)
    low, high = bounds
    maximum = 15 if "float4" in np.dtype(dtype).name else 255
    if (
        not np.isfinite([low, high]).all()
        or int(low) != low
        or int(high) != high
        or not 0 <= low <= high <= maximum
    ):
        raise ValueError(f"{label} encoding bounds must be integers in [0, {maximum}]")
    return rng.randint(int(low), int(high) + 1, size=shape).astype(np.uint8).view(dtype)


def _generate_test_data(
    testcase_name,
    A,
    M,
    BS,
    K,
    H,
    expertNum,
    tokenDtype,
    needSchedule,
    input_ranges=None,
):
    token_range, scale_range, token_mode, scale_mode = _payload_ranges(
        input_ranges, tokenDtype
    )
    rng = np.random.RandomState(_seed_from_name(testcase_name))
    out_num = (
        0
        if "outnum_zero" in testcase_name
        else (1 if "outnum_one" in testcase_name else A)
    )
    has_mask = "masked" in testcase_name
    single_expert = "single_expert" in testcase_name

    if needSchedule == 0:
        expert_ids = rng.randint(0, expertNum, size=(out_num, BS, K)).astype(np.int32)
    else:
        expert_ids = rng.randint(0, expertNum, size=(A, M, BS, K)).astype(np.int32)

    if "all_masked" in testcase_name:
        expert_ids[...] = _MASK_VALUE
    elif single_expert:
        expert_ids[...] = 0
    elif has_mask:
        mask = rng.random(expert_ids.shape) < 0.15
        expert_ids[mask] = _MASK_VALUE

    token_shape = (A, M, BS, K, H)
    token_scales = None
    if tokenDtype < 2:
        dtype = np.float16 if tokenDtype == 0 else "bfloat16"
        token_data = _numeric_samples(rng, token_shape, token_range, dtype, "token")
    elif tokenDtype == 2:
        token_data = _numeric_samples(rng, token_shape, token_range, np.int8, "token")
        token_scales = _numeric_samples(
            rng, (A, M, BS, K), scale_range, np.float32, "scale"
        )
    else:
        from ttk.utilities.dtypes import (
            numpy_float8_e5m2,
            numpy_float8_e4m3fn,
            numpy_float4_e2m1,
            numpy_float8_e8m0,
        )

        dtype = {3: numpy_float8_e5m2, 4: numpy_float8_e4m3fn, 5: numpy_float4_e2m1}[
            tokenDtype
        ]()
        token_data = _payload_samples(
            rng, token_shape, token_range, dtype, "token", token_mode
        )
        token_scales = _payload_samples(
            rng,
            (A, M, BS, K, (H + 31) // 32),
            scale_range,
            numpy_float8_e8m0(),
            "scale",
            scale_mode,
        )

    if input_ranges and input_ranges[0] is not None and len(input_ranges[0]) >= 6:
        lo, hi = input_ranges[0][4:6]
        if (lo is None) != (hi is None):
            raise ValueError("expert range requires both bounds or neither")
        if lo is not None:
            if (
                not np.isfinite([lo, hi]).all()
                or int(lo) != lo
                or int(hi) != hi
                or not 0 <= lo <= hi <= np.iinfo(np.int32).max
                or (hi >= expertNum and lo < _MASK_VALUE)
            ):
                raise ValueError(
                    "expert range must be valid IDs or mask IDs >= 1000000"
                )
            expert_ids[...] = rng.randint(int(lo), int(hi) + 1, size=expert_ids.shape)

    session_ids = np.arange(A, dtype=np.int32)
    if "reverse_sessions" in testcase_name:
        session_ids = session_ids[::-1].copy()
    micro_batch_ids = np.full(
        A, M - 1 if "micro_last" in testcase_name else 0, dtype=np.int32
    )

    return {
        "A": A,
        "M": M,
        "BS": BS,
        "K": K,
        "H": H,
        "expertNum": expertNum,
        "tokenDtype": tokenDtype,
        "needSchedule": needSchedule,
        "expert_ids": expert_ids,
        "token_data": token_data,
        "token_scales": token_scales,
        "session_ids_buf": session_ids,
        "micro_batch_ids_buf": micro_batch_ids,
        "out_num": out_num,
        "cur_micro_batch_id": 0,
    }


def _build_token_data_bytes(token_data, token_scales, tokenDtype, A, M, BS, K, H):
    if tokenDtype >= 3:
        from ttk.utilities.dtypes import pack_4bits

        raw_tokens = (
            pack_4bits(token_data).reshape(A, M, BS, K, H // 2)
            if tokenDtype == 5
            else token_data.view(np.uint8)
        )
        return np.concatenate((raw_tokens, token_scales.view(np.uint8)), axis=-1)
    if tokenDtype == 1:
        t = torch.from_numpy(np.ascontiguousarray(token_data, dtype=np.float32))
        t_bf16 = t.to(torch.bfloat16)
        return t_bf16.contiguous().view(torch.uint8).numpy()
    if tokenDtype != 2:
        return np.ascontiguousarray(token_data)
    per_token = H + 4
    buf = np.zeros((A, M, BS, K, per_token), dtype=np.uint8)
    buf[..., :H] = token_data.view(np.uint8)
    scale_bytes = (
        np.ascontiguousarray(token_scales, dtype=np.float32)
        .view(np.uint8)
        .reshape(A, M, BS, K, 4)
    )
    buf[..., H : H + 4] = scale_bytes
    return buf


def _build_token_info_buf(expert_ids, A, M, BS, K):
    F = 2 + BS * K
    token_info = np.zeros((A, M, F), dtype=np.int32)
    for a in range(A):
        for m in range(M):
            token_info[a, m, 0] = 1
            token_info[a, m, 1] = 0
            token_info[a, m, 2 : 2 + BS * K] = expert_ids[a, m].flatten()
    return token_info


# ==================== Golden helpers ====================

EXPERT_MASK_VALUE = 1000000
# ScheduleContext ABI v1, defined by common/include/op_kernel/attention_ffn_schedule.h.
_SCHEDULE_CONTEXT_BYTES = 1024
_CONTEXT_ARCHIVE_VERSION = 1
_TOKEN_STRIDE_ALIGNMENT = 512
_HBM_ALLOCATION_ALIGNMENT = (
    512  # Allocation alignment and row stride are separate contracts.
)
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_NUM_OUTPUTS = 8
_OUT_Y = 0
_OUT_GROUP_LIST = 1
_OUT_SESSION_IDS = 2
_OUT_MICRO_BATCH_IDS = 3
_OUT_TOKEN_IDS = 4
_OUT_EXPERT_OFFSETS = 5
_OUT_SCALE = 6
_OUT_TOKEN_COUNT = 7
_TOKEN_PREFIX_OUTPUTS = (
    _OUT_Y,
    _OUT_SESSION_IDS,
    _OUT_MICRO_BATCH_IDS,
    _OUT_TOKEN_IDS,
    _OUT_EXPERT_OFFSETS,
    _OUT_SCALE,
)


def _get_cached_data(testcase_name, **gen_kwargs):
    if testcase_name not in _DATA_CACHE:
        raise KeyError(
            f"No cached data for testcase '{testcase_name}'. "
            f"Available: {list(_DATA_CACHE.keys())}"
        )
    return _DATA_CACHE[testcase_name]


def _sort_expert_ids(expert_ids_flat):
    sort_key = np.where(
        expert_ids_flat < EXPERT_MASK_VALUE, expert_ids_flat, np.iinfo(np.int32).max
    ).astype(np.int64)
    sorted_order = np.argsort(sort_key, kind="stable")
    valid_count = int(np.sum(expert_ids_flat < EXPERT_MASK_VALUE))
    sorted_valid_ids = expert_ids_flat[sorted_order[:valid_count]]
    return sorted_order, valid_count, sorted_valid_ids


def _generate_group_list(sorted_expert_ids, valid_count, expert_num):
    group_list = np.zeros((expert_num, 2), dtype=np.int64)
    if valid_count == 0:
        return group_list
    # IDs are already sorted. Find run boundaries in bounded chunks rather
    # than visiting tens of millions of tokens in Python.
    boundaries = [np.array([0], dtype=np.int64)]
    for start in range(1, valid_count, 1 << 20):
        end = min(start + (1 << 20), valid_count)
        changes = np.flatnonzero(
            sorted_expert_ids[start:end] != sorted_expert_ids[start - 1 : end - 1]
        )
        boundaries.append(changes + start)
    boundaries.append(np.array([valid_count], dtype=np.int64))
    boundaries = np.concatenate(boundaries)
    count = min(len(boundaries) - 1, expert_num)
    group_list[:count, 0] = sorted_expert_ids[boundaries[:count]]
    group_list[:count, 1] = np.diff(boundaries)[:count]
    return group_list


def _to_torch(arr, dtype):
    if "float8" in arr.dtype.name or "float4" in arr.dtype.name:
        from ttk.utilities.dtypes import numpy_to_torch_tensor

        tensor = numpy_to_torch_tensor(np.ascontiguousarray(arr))
        return (
            tensor.reshape(*arr.shape[:-1], arr.shape[-1] // 2)
            if "float4" in arr.dtype.name
            else tensor
        )
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    return t


# ==================== TestSpec class ====================


class AclnnFfnWorkerBatchingTestSpec:
    @staticmethod
    def customize_inputs(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        syncFlag: bool,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        testcase_name = kwargs.get("testcase_name", "default")
        _pin_run_time(needSchedule)
        A = int(maxOutShape[0])
        BS = int(maxOutShape[1])
        K = int(maxOutShape[2])
        H = int(maxOutShape[3])
        M = (
            3
            if any(tag in testcase_name for tag in ("sync_skip_mb", "micro_last"))
            else 1
        )

        data = _generate_test_data(
            testcase_name,
            A,
            M,
            BS,
            K,
            H,
            expertNum,
            tokenDtype,
            needSchedule,
            input_ranges=kwargs.get("input_ranges"),
        )

        token_data = data["token_data"]
        token_scales = data["token_scales"]
        expert_ids = data["expert_ids"]
        session_ids = data["session_ids_buf"]
        micro_batch_ids = data["micro_batch_ids_buf"]
        out_num = data["out_num"]

        if tokenDtype in (0, 1):
            dtype_size = 2
        else:
            dtype_size = 1
        HS = H * dtype_size if tokenDtype != 2 else H + 4
        if tokenDtype >= 3:
            HS = (H // 2 if tokenDtype == 5 else H) + (H + 31) // 32

        td_arr = _build_token_data_bytes(
            token_data, token_scales, tokenDtype, A, M, BS, K, H
        )
        # Physical token stride follows the producer ABI. Keep scale immediately
        # after the payload and zero only the remaining 512-byte stride padding.
        payload = np.ascontiguousarray(td_arr).view(np.uint8).reshape(A, M, BS, K, HS)
        HS = (
            (HS + (_TOKEN_STRIDE_ALIGNMENT - 1))
            // _TOKEN_STRIDE_ALIGNMENT
            * _TOKEN_STRIDE_ALIGNMENT
        )
        padded = np.zeros((A, M, BS, K, HS), dtype=np.uint8)
        padded[..., : payload.shape[-1]] = payload
        td_bytes = padded.tobytes()

        # 真实地址协议：辅助数据放独立 HBM 缓冲，context 仅保留 1024B 头部并写入
        # 绝对设备地址（生产二级指针寻址路径）；不写 TEST_MAGIC，偏移 640 保持 0。
        td_ptr = _hbm_alloc(len(td_bytes))
        _hbm_copy(td_bytes, td_ptr)

        if needSchedule == 0:
            # out_num is the published prefix, not the allocation capacity.
            # Keep a live buffer even for zero published rows. Valid stale IDs
            # in the unused tail also expose a kernel that ignores out_num.
            expert_capacity = np.zeros((A, BS, K), dtype=np.int32)
            expert_capacity[:out_num] = expert_ids
            ei_bytes = expert_capacity.tobytes()
            si_bytes = np.ascontiguousarray(session_ids).tobytes()
            mi_bytes = np.ascontiguousarray(micro_batch_ids).tobytes()
            ei_ptr = _hbm_alloc(len(ei_bytes))
            _hbm_copy(ei_bytes, ei_ptr)
            si_ptr = _hbm_alloc(len(si_bytes))
            _hbm_copy(si_bytes, si_ptr)
            mi_ptr = _hbm_alloc(len(mi_bytes))
            _hbm_copy(mi_bytes, mi_ptr)
        else:
            token_info = _build_token_info_buf(expert_ids, A, M, BS, K)
            layer_ids = np.zeros((A, M), dtype=np.int32)
            if syncFlag and "async_layers" in testcase_name:
                per_layer = expertNum // layerNum
                valid = (expert_ids >= 0) & (expert_ids < _MASK_VALUE)
                expert_ids[valid] %= per_layer
                token_info[:, :, 2:] = expert_ids.reshape(A, M, BS * K)
                layer_ids[:] = (np.arange(A, dtype=np.int32)[::-1] % layerNum)[:, None]
                token_info[:, :, 1] = layer_ids
            if syncFlag and "layer_boundary" in testcase_name:
                value = (
                    -1
                    if "negative" in testcase_name
                    else (
                        np.iinfo(np.int32).max
                        if "int32max" in testcase_name
                        else (layerNum if "invalid" in testcase_name else layerNum - 1)
                    )
                )
                layer_ids[:] = value
                token_info[:, :, 1] = layer_ids
            data["layer_ids"] = layer_ids
            # 独立构造“已发布描述符”的集合，不把未就绪行的 expert_ids 预先 mask。
            # 这样错误读取未就绪行的 kernel 会产生额外 token，能够被 golden 检出。
            ready = np.ones((A, M), dtype=bool)
            if syncFlag and "sync_" in testcase_name:
                if "sync_partial" in testcase_name:
                    ready[1::2, :] = False
                elif "sync_last" in testcase_name or "sync_skip_mb" in testcase_name:
                    ready[:] = False
                    ready[-1, -1] = True
                elif "sync_masked" in testcase_name:
                    ready[1:, :] = False
                    token_info[0, :, 2:] = _MASK_VALUE
                    expert_ids[0, :, :, :] = _MASK_VALUE
            token_info[:, :, 0] = ready.astype(np.int32)
            data["ready"] = ready
            # 静态 ST 的入口 cursor 为 0，至少预置一个 ready；期望选中首个非空 micro batch。
            # 完全无 ready 的等待/延迟发布不能用静态 CSV 表达，由状态脚本单独验证。
            # RECV 单次执行（run_time 已钉到 1），描述符消费后不复位。
            data["cur_micro_batch_id"] = next(m for m in range(M) if ready[:, m].any())
            ti_bytes = token_info.tobytes()
            ti_ptr = _hbm_alloc(len(ti_bytes))
            _hbm_copy(ti_bytes, ti_ptr)

        total_size = 1024
        ctx = bytearray(total_size)

        _pack_ctx(ctx, _OFF_SESSION_NUM, "<I", A)
        _pack_ctx(ctx, _OFF_MICRO_BATCH_NUM, "<I", M)
        _pack_ctx(ctx, _OFF_MICRO_BATCH_SIZE, "<I", BS)
        _pack_ctx(ctx, _OFF_SELECTED_EXPERT_NUM, "<I", K)
        _pack_ctx(ctx, _OFF_EXPERT_NUM, "<I", expertNum)
        _pack_ctx(ctx, _OFF_ATTN_TO_FFN_TOKEN_SIZE, "<I", HS)
        _pack_ctx(ctx, _OFF_FFN_TO_ATTN_TOKEN_SIZE, "<I", HS)
        _pack_ctx(ctx, _OFF_SCHEDULE_MODE, "<i", 0)
        _pack_ctx(ctx, _OFF_RUN_FLAG, "<i", 1)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF, "<Q", td_ptr)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF_SIZE, "<Q", len(td_bytes))

        if needSchedule == 0:
            _pack_ctx(ctx, _OFF_SESSION_IDS_BUF, "<Q", si_ptr)
            _pack_ctx(ctx, _OFF_SESSION_IDS_BUF_SIZE, "<Q", len(si_bytes))
            _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF, "<Q", mi_ptr)
            _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF_SIZE, "<Q", len(mi_bytes))
            _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF, "<Q", ei_ptr)
            _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF_SIZE, "<Q", len(ei_bytes))
            _pack_ctx(ctx, _OFF_OUT_NUM, "<I", out_num)
        else:
            _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF, "<Q", ti_ptr)
            _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF_SIZE, "<Q", len(ti_bytes))
            _pack_ctx(ctx, _OFF_POLLING_INDEX, "<Q", 0)

        ctx_arr = np.array(ctx, dtype=np.int8)
        if isinstance(scheduleContext, np.ndarray):
            if scheduleContext.nbytes < total_size:
                raise ValueError(
                    f"scheduleContext requires {total_size} bytes, got {scheduleContext.nbytes}"
                )
            scheduleContext.reshape(-1)[:total_size] = ctx_arr
        else:
            cur_storage = scheduleContext.untyped_storage()
            if cur_storage.nbytes() < total_size:
                new_storage = torch.UntypedStorage(total_size)
                scheduleContext.set_(
                    new_storage,
                    scheduleContext.storage_offset(),
                    scheduleContext.shape,
                    scheduleContext.stride(),
                )
            scheduleContext.untyped_storage()[:total_size].copy_(
                torch.from_numpy(ctx_arr).untyped_storage()
            )

        data["HS"] = HS
        _DATA_CACHE[testcase_name] = data

    @staticmethod
    def golden(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        syncFlag: bool,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        testcase_name = kwargs.get("testcase_name", "default")
        A = int(maxOutShape[0])
        BS = int(maxOutShape[1])
        K = int(maxOutShape[2])
        H = int(maxOutShape[3])
        M = 1
        if tokenDtype in (0, 1):
            HS = H * 2
        elif tokenDtype >= 3:
            HS = (H // 2 if tokenDtype == 5 else H) + (H + 31) // 32
        else:
            HS = H + 4
        HS = (
            (HS + (_TOKEN_STRIDE_ALIGNMENT - 1))
            // _TOKEN_STRIDE_ALIGNMENT
            * _TOKEN_STRIDE_ALIGNMENT
        )
        data = _get_cached_data(
            testcase_name,
            A=A,
            M=M,
            BS=BS,
            K=K,
            H=H,
            expertNum=expertNum,
            tokenDtype=tokenDtype,
            needSchedule=needSchedule,
            HS=HS,
        )

        A = data["A"]
        M = data["M"]
        BS = data["BS"]
        K = data["K"]
        H = data["H"]
        token_data = data["token_data"]
        token_scales = data.get("token_scales")
        session_ids_buf = data["session_ids_buf"]
        micro_batch_ids_buf = data["micro_batch_ids_buf"]
        expert_ids = data["expert_ids"]

        Y = A * BS * K

        if tokenDtype == 0:
            out_torch_dtype = torch.float16
            out_np_dtype = np.float16
        elif tokenDtype == 1:
            out_torch_dtype = torch.bfloat16
            out_np_dtype = np.float32
        elif tokenDtype >= 3:
            out_torch_dtype = None
            out_np_dtype = token_data.dtype
        else:
            out_torch_dtype = torch.int8
            out_np_dtype = np.int8

        if needSchedule == 1:
            cur_mb = data.get("cur_micro_batch_id", 0)
            # 只在 CPU 参考副本上屏蔽未选中 session；设备输入仍保留它们的合法 ids。
            ids = expert_ids[:, cur_mb].copy()
            if syncFlag:
                per_layer = expertNum // layerNum
                layer_ids = data.get("layer_ids", np.zeros((A, M), np.int32))[:, cur_mb]
                valid = (
                    (ids >= 0)
                    & (ids < per_layer)
                    & (layer_ids[:, None, None] >= 0)
                    & (layer_ids[:, None, None] < layerNum)
                )
                ids = np.where(
                    valid,
                    ids.astype(np.int64) + layer_ids[:, None, None] * per_layer,
                    _MASK_VALUE,
                ).astype(np.int32)
                ids[~data.get("ready", np.ones((A, M), dtype=bool))[:, cur_mb]] = (
                    _MASK_VALUE
                )
            expert_ids_flat = ids.reshape(-1).astype(np.int32)
        else:
            expert_ids_flat = expert_ids.reshape(-1).astype(np.int32)
        sorted_order, valid_count, sorted_valid_ids = _sort_expert_ids(expert_ids_flat)

        y_out = np.zeros((Y, H), dtype=out_np_dtype)
        session_ids_out = np.zeros(Y, dtype=np.int32)
        micro_batch_ids_out = np.zeros(Y, dtype=np.int32)
        token_ids_out = np.zeros(Y, dtype=np.int32)
        expert_offsets_out = np.zeros(Y, dtype=np.int32)
        dynamic_scale_out = (
            np.zeros((Y, (H + 31) // 32), dtype=token_scales.dtype)
            if tokenDtype >= 3
            else np.ones(Y, dtype=np.float32)
        )

        bsk = BS * K
        cur_mb = data.get("cur_micro_batch_id", 0)

        # Bound the gathered payload to about 16 MiB, including wide-H cases,
        # and cap index temporaries independently for narrow-H cases.
        row_bytes = H * token_data.dtype.itemsize
        if token_scales is not None:
            row_bytes += token_scales.dtype.itemsize * (
                (H + 31) // 32 if tokenDtype >= 3 else 1
            )
        chunk_rows = max(1, min(1 << 18, (16 << 20) // max(1, row_bytes)))
        for start in range(0, valid_count, chunk_rows):
            end = min(start + chunk_rows, valid_count)
            gidx = sorted_order[start:end]
            a_idx = gidx // bsk
            rem = gidx % bsk
            bs_idx = rem // K
            k_idx = rem % K

            if needSchedule == 0:
                s_idx = session_ids_buf[a_idx]
                mb_idx = micro_batch_ids_buf[a_idx]
            else:
                s_idx = a_idx
                mb_idx = cur_mb

            session_ids_out[start:end] = s_idx
            micro_batch_ids_out[start:end] = mb_idx
            token_ids_out[start:end] = bs_idx
            expert_offsets_out[start:end] = k_idx
            y_out[start:end] = token_data[s_idx, mb_idx, bs_idx, k_idx, :]

            if tokenDtype >= 2 and token_scales is not None:
                dynamic_scale_out[start:end] = token_scales[
                    s_idx, mb_idx, bs_idx, k_idx
                ]

        group_list_out = _generate_group_list(sorted_valid_ids, valid_count, expertNum)
        actual_token_num_out = np.array([valid_count], dtype=np.int64)

        if kwargs.get("use_numpy", False):
            return [
                y_out,
                group_list_out,
                session_ids_out,
                micro_batch_ids_out,
                token_ids_out,
                expert_offsets_out,
                dynamic_scale_out,
                actual_token_num_out,
            ]

        return [
            _to_torch(y_out, out_torch_dtype),
            _to_torch(group_list_out, None),
            _to_torch(session_ids_out, None),
            _to_torch(micro_batch_ids_out, None),
            _to_torch(token_ids_out, None),
            _to_torch(expert_offsets_out, None),
            _to_torch(dynamic_scale_out, None),
            _to_torch(actual_token_num_out, None),
        ]

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        # 接口固定返回八个张量。先验证数量和元数据，不能通过截短到 min(size)
        # 来比较，否则空输出/缺行输出会因“没有发现不同元素”而被误判为通过。
        def failure(message):
            return {"pass": False, "precision": "0%", "error_info": message}

        if len(outputs) != 2 * _NUM_OUTPUTS:
            return [
                failure("expected 8 NPU outputs and 8 golden outputs")
                for _ in range(_NUM_OUTPUTS)
            ]

        def to_np(t):
            if isinstance(t, torch.Tensor):
                from ttk.utilities.dtypes import torch_to_numpy_tensor

                array = torch_to_numpy_tensor(t)
                if "float4" in str(t.dtype):
                    array = array.reshape(*t.shape[:-1], t.shape[-1] * 2)
                return array
            return np.asarray(t)

        npu_outputs = [to_np(t) for t in outputs[:_NUM_OUTPUTS]]
        golden_outputs = [to_np(t) for t in outputs[_NUM_OUTPUTS:]]
        actual_token_num = int(golden_outputs[_OUT_TOKEN_COUNT].flat[0])
        has_scale = golden_outputs[_OUT_Y].dtype.name not in ("float16", "bfloat16")
        results = []
        for idx, (npu, gold) in enumerate(zip(npu_outputs, golden_outputs)):
            if npu.shape != gold.shape or npu.dtype != gold.dtype:
                results.append(
                    failure(
                        f"metadata mismatch [idx={idx}]: {npu.shape}/{npu.dtype} "
                        f"!= {gold.shape}/{gold.dtype}"
                    )
                )
                continue
            # FP16/BF16 没有 scale 载荷，只有数值内容未定义；张量元数据仍须正确。
            if idx == _OUT_SCALE and not has_scale:
                results.append({"pass": True, "precision": "100%", "error_info": None})
                continue
            # y/索引/scale 只保证有效 token 前缀，容量尾部允许未写入。
            # group_list 则要求所有未使用行清零，尤其全 mask 场景不能直接 PASS。
            if idx in _TOKEN_PREFIX_OUTPUTS:
                if not 0 <= actual_token_num <= gold.shape[0]:
                    results.append(failure(f"invalid golden token count [idx={idx}]"))
                    continue
                npu = npu[:actual_token_num]
                gold = gold[:actual_token_num]
            # Payloads are copied verbatim: NaN payloads and signed zero matter.
            if idx in (_OUT_Y, _OUT_SCALE):
                npu = npu.view(np.uint8)
                gold = gold.view(np.uint8)
            diff = int(np.count_nonzero(npu != gold))
            total = gold.size
            passed = diff == 0
            results.append(
                {
                    "pass": passed,
                    "precision": f"{(total - diff) / total * 100}%"
                    if total
                    else "100%",
                    "error_info": None if passed else f"{diff} mismatches [idx={idx}]",
                }
            )
        return results


_TORCH_ORIG_WARMUP = None
_TORCH_GRAPH_REPLAY = {}
_TORCH_GRAPH_REPLAY_CHECKS = {}


def _ttk_assets_compat():
    """Load sibling adapters without depending on the caller's PYTHONPATH."""
    import importlib.util
    import pathlib
    import sys

    name = "_ffn_worker_batching_ttk_compat"
    if name not in sys.modules:
        path = pathlib.Path(__file__).with_name("ttk_compat.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def _configure_torch_execution(need_schedule):
    """Eager consumes once; ACL graph republishes descriptors before each call."""
    from ttk.utilities.container_utils import get_global_storage

    global _TORCH_ORIG_WARMUP
    storage = get_global_storage()
    if need_schedule == 1:
        ge_enabled = any(
            getattr(getattr(storage, name, None), "enabled", False)
            for name in ("cst_switches", "dyn_switches")
        )
        if ge_enabled:
            raise ValueError(
                "Torch RECV graph testing currently requires --aclgraph; GE is not adapted"
            )
        if getattr(storage, "aclgraph_enabled", False):
            from ttk.core_modules.framework_api import graph_execution

            if not getattr(graph_execution, "SUPPORTS_NPU_GRAPH_BEFORE_RUN", False):
                _ttk_assets_compat().install_graph()
            if _TORCH_ORIG_WARMUP is not None:
                storage.warmup = _TORCH_ORIG_WARMUP
            _pin_run_time(0)
            return
        if getattr(storage, "deterministic_level", 0):
            raise ValueError(
                "Torch RECV single-call testing does not support repeated determinism checks"
            )
        if _TORCH_ORIG_WARMUP is None:
            _TORCH_ORIG_WARMUP = storage.warmup
        storage.warmup = False
    elif _TORCH_ORIG_WARMUP is not None:
        storage.warmup = _TORCH_ORIG_WARMUP
    _pin_run_time(need_schedule)


class TorchFfnWorkerBatchingTestSpec:
    """TTK NORM, single-call eager RECV, and republished ACL graph RECV.

    Eager RECV consumes once. ACL graph restores the original descriptor bytes
    and context in place before every compiled call, preserving device pointers.
    """

    @staticmethod
    def npu_preprocess(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        mx_output_uint8=False,
        **kwargs,
    ):
        import logging
        import cann_ops_transformer

        logging.info("FFN Torch installed package: %s", cann_ops_transformer.__file__)
        # TTK manual input files contain only the context tensor. A fresh worker
        # must restore an explicit archive before invoking the pointer-bearing
        # operator; accepting the old process's raw addresses risks invalid DMA.
        # Eager reads run_time/warmup after this hook, including manual replay
        # where customize_inputs is bypassed.
        _configure_torch_execution(need_schedule)
        attributes = dict(
            expert_num=expert_num,
            max_out_shape=list(max_out_shape),
            token_dtype=token_dtype,
            need_schedule=need_schedule,
            layer_num=layer_num,
            sync_flag=sync_flag,
            mx_output_uint8=mx_output_uint8,
        )
        archive = os.getenv("FFN_WB_REPLAY_ARCHIVE")
        if archive:
            header, _ = restore_context_archive(archive, attributes)
            schedule_context.reshape(-1)[:_SCHEDULE_CONTEXT_BYTES].copy_(
                torch.from_numpy(header)
            )
        else:
            header = (
                schedule_context.reshape(-1)[:_SCHEDULE_CONTEXT_BYTES]
                .detach()
                .cpu()
                .numpy()
            )
            validate_live_context(header, need_schedule)
        archive = os.getenv("FFN_WB_SAVE_ARCHIVE")
        if archive:
            save_context_archive(archive, header, attributes)
        if need_schedule == 1:
            pointer, size = struct.unpack_from("<QQ", header, _OFF_FFN_TOKEN_INFO_BUF)
            descriptors = np.empty(size, dtype=np.uint8)
            lib = _load_acl()
            ret = lib.aclrtMemcpy(
                descriptors.ctypes.data, size, pointer, size, _ACL_MEMCPY_DEVICE_TO_HOST
            )
            if ret:
                raise RuntimeError(f"Torch RECV descriptor snapshot failed: {ret}")
            _TORCH_GRAPH_REPLAY[kwargs.get("testcase_name", "default")] = (
                schedule_context.data_ptr(),
                header.copy(),
                pointer,
                descriptors.tobytes(),
            )
            _TORCH_GRAPH_REPLAY_CHECKS[kwargs.get("testcase_name", "default")] = 0
            from ttk.utilities.container_utils import get_global_storage

            if getattr(get_global_storage(), "aclgraph_enabled", False):
                hook_args = {
                    **attributes,
                    "testcase_name": kwargs.get("testcase_name", "default"),
                }
                _ttk_assets_compat().remember_graph_state(
                    schedule_context,
                    lambda: TorchFfnWorkerBatchingTestSpec.npu_graph_before_run(
                        schedule_context, **hook_args
                    ),
                    lambda: TorchFfnWorkerBatchingTestSpec.npu_graph_after_run(
                        schedule_context, **hook_args
                    ),
                )

    @staticmethod
    def npu_graph_before_run(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag=False,
        mx_output_uint8=False,
        **kwargs,
    ):
        if need_schedule != 1:
            return
        key = kwargs.get("testcase_name", "default")
        if key not in _TORCH_GRAPH_REPLAY:
            raise RuntimeError("Torch RECV graph has no prepared descriptor snapshot")
        context_ptr, header, pointer, descriptors = _TORCH_GRAPH_REPLAY[key]
        if schedule_context.data_ptr() != context_ptr:
            raise RuntimeError(
                "Torch RECV graph context moved after snapshot preparation"
            )
        validate_live_context(header, need_schedule)
        # TTK synchronizes the preceding invocation before this hook. Restore
        # descriptors first, then the cursor/header, without reallocating HBM.
        _hbm_copy(descriptors, pointer)
        schedule_context.reshape(-1)[:_SCHEDULE_CONTEXT_BYTES].copy_(
            torch.from_numpy(header)
        )

    @staticmethod
    def npu_graph_after_run(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag=False,
        mx_output_uint8=False,
        **kwargs,
    ):
        if need_schedule != 1:
            return
        key = kwargs.get("testcase_name", "default")
        _, header, pointer, descriptors = _TORCH_GRAPH_REPLAY[key]
        current = np.empty(len(descriptors), dtype=np.uint8)
        ret = _load_acl().aclrtMemcpy(
            current.ctypes.data,
            current.size,
            pointer,
            current.size,
            _ACL_MEMCPY_DEVICE_TO_HOST,
        )
        if ret:
            raise RuntimeError(f"Torch RECV consumption check failed: {ret}")
        A, M, BS, K = struct.unpack_from("<4I", header, 0)
        original = np.frombuffer(descriptors, dtype=np.int32).reshape(A, M, 2 + BS * K)
        current = current.view(np.int32).reshape(original.shape)
        mb = next(m for m in range(M) if (original[:, m, 0] == 1).any())
        selected = original[:, mb, 0] == 1
        if np.any(current[selected, mb, 0] != 0):
            raise RuntimeError(
                "Torch RECV graph did not consume the republished ready descriptors"
            )
        _TORCH_GRAPH_REPLAY_CHECKS[key] += 1
        import logging

        logging.info(
            "Torch RECV graph consumption verified: call %d",
            _TORCH_GRAPH_REPLAY_CHECKS[key],
        )

    @staticmethod
    def customize_inputs(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        mx_output_uint8=False,
        **kwargs,
    ):
        _configure_torch_execution(need_schedule)
        # Only the 1024-byte header is part of this tensor; secondary buffers
        # have their own HBM allocations and are archived explicitly for replay.
        if schedule_context.numel() < _SCHEDULE_CONTEXT_BYTES:
            raise ValueError("schedule_context must include the 1024-byte header")
        AclnnFfnWorkerBatchingTestSpec.customize_inputs(
            schedule_context.numpy(),
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        # --no-prof only prepares inputs and never calls npu_preprocess. Save
        # here as well so TTK's documented prepare/replay workflow is complete.
        archive = os.getenv("FFN_WB_SAVE_ARCHIVE")
        if archive:
            attributes = dict(
                expert_num=expert_num,
                max_out_shape=list(max_out_shape),
                token_dtype=token_dtype,
                need_schedule=need_schedule,
                layer_num=layer_num,
                sync_flag=sync_flag,
                mx_output_uint8=mx_output_uint8,
            )
            save_context_archive(archive, schedule_context.numpy(), attributes)

    @staticmethod
    def golden(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        mx_output_uint8=False,
        **kwargs,
    ):
        outputs = AclnnFfnWorkerBatchingTestSpec.golden(
            schedule_context,
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        if mx_output_uint8 and token_dtype >= 3:
            outputs[6] = outputs[6].view(torch.uint8)
            if token_dtype == 5:
                outputs[0] = outputs[0].view(torch.uint8)
        return outputs

    compare = staticmethod(AclnnFfnWorkerBatchingTestSpec.compare)
    tolerance = AclnnFfnWorkerBatchingTestSpec.tolerance


class FfnWorkerBatchingKernelTestSpec:
    """Kernel 接口使用 snake_case 属性及 numpy 数据，复用 ACLNN 的输入协议和 golden。"""

    @staticmethod
    def customize_inputs(
        schedule_context,
        expert_num,
        max_out_shape,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        **kwargs,
    ):
        from ttk.utilities.container_utils import get_global_storage

        if getattr(get_global_storage(), "test_mode", None) == "geir":
            _ttk_assets_compat().install_geir(
                FfnWorkerBatchingKernelTestSpec.geir_prepare_inputs
            )
        tensor = torch.from_numpy(schedule_context.copy())
        AclnnFfnWorkerBatchingTestSpec.customize_inputs(
            tensor,
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        # 上下文只需 1024B 头部；CSV 声明的更大 shape 其尾部不再使用
        # （辅助数据走独立 HBM 真实地址缓冲）。
        raw = torch.empty(0, dtype=torch.int8).set_(tensor.untyped_storage())
        return [raw.numpy()]

    @staticmethod
    def geir_prepare_inputs(input_arrays, attributes, input_prefix):
        """Export live secondary buffers for relocation by the GEIR executable.

        Raw Python-process HBM pointers must never reach a different process.
        Keep golden's original context/cache intact and describe pointer fixups
        separately; the C++ runner patches a private copy before RunGraph.
        """
        header = np.asarray(input_arrays[0], dtype=np.int8).reshape(-1)
        mode = attributes.get("need_schedule", 0)
        validate_live_context(header, mode)
        lib = _load_acl()
        ret = lib.aclrtSynchronizeDevice()
        if ret:
            raise RuntimeError(f"GEIR synchronize failed: {ret}")
        relocations = []
        for offset in _context_buffer_offsets(mode):
            pointer, size = struct.unpack_from("<QQ", header, offset)
            payload = np.empty(size, dtype=np.uint8)
            ret = lib.aclrtMemcpy(
                payload.ctypes.data, size, pointer, size, _ACL_MEMCPY_DEVICE_TO_HOST
            )
            if ret:
                raise RuntimeError(f"GEIR payload D2H failed: {ret}")
            path = f"{input_prefix}_aux_{offset}.bin"
            payload.tofile(path)
            relocations.append(
                {"input_index": 0, "offset": offset, "size": size, "path": path}
            )
        return relocations

    @staticmethod
    def golden(
        schedule_context,
        expert_num,
        max_out_shape,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        **kwargs,
    ):
        from ttk.utilities.dtypes import torch_to_numpy_tensor

        outputs = AclnnFfnWorkerBatchingTestSpec.golden(
            schedule_context,
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        arrays = [torch_to_numpy_tensor(t) for t in outputs]
        if token_dtype == 5:
            arrays[0] = arrays[0].reshape(-1, max_out_shape[3])
        return arrays

    compare = staticmethod(AclnnFfnWorkerBatchingTestSpec.compare)
    tolerance = AclnnFfnWorkerBatchingTestSpec.tolerance


class AclnnFfnWorkerBatchingLegacyTestSpec(AclnnFfnWorkerBatchingTestSpec):
    """V1 has no syncFlag argument; keep its positional contract separate from V2."""

    @staticmethod
    def customize_inputs(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        return AclnnFfnWorkerBatchingTestSpec.customize_inputs(
            scheduleContext,
            expertNum,
            maxOutShape,
            tokenDtype,
            needSchedule,
            layerNum,
            False,
            y,
            groupList,
            sessionIds,
            microBatchIds,
            tokenIds,
            expertOffsets,
            dynamicScale,
            actualTokenNum,
            *args,
            **kwargs,
        )

    @staticmethod
    def golden(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        return AclnnFfnWorkerBatchingTestSpec.golden(
            scheduleContext,
            expertNum,
            maxOutShape,
            tokenDtype,
            needSchedule,
            layerNum,
            False,
            y,
            groupList,
            sessionIds,
            microBatchIds,
            tokenIds,
            expertOffsets,
            dynamicScale,
            actualTokenNum,
            *args,
            **kwargs,
        )


def _run_ttk_with_installed_package():
    """Run unmodified TTK after importing the active Python environment's wheel.

    TTK normally prioritizes the CANN namespace package. Preloading the actual
    installed package in the parent and workers avoids that competing copy.
    No source frontend or hand-written dispatcher schema is registered here.
    """
    import importlib
    import multiprocessing
    import pathlib
    import sys
    import sysconfig

    site = pathlib.Path(sysconfig.get_path("purelib")).resolve()
    expected = site / "cann_ops_transformer" / "__init__.py"
    if not expected.is_file():
        raise RuntimeError(
            f"Install the full cann_ops_transformer wheel in {sys.prefix} first"
        )
    if not (pathlib.Path.cwd() / "ttk" / "__init__.py").is_file():
        raise RuntimeError(
            "Run this command from the ops-test-kit repository directory"
        )
    sys.path.insert(0, str(pathlib.Path.cwd()))
    sys.path.insert(0, str(site))
    assets = str(pathlib.Path(__file__).resolve().parent)
    sys.path.insert(0, assets)
    # The server imports only a stdlib bootstrap. Workers import the wheel after
    # fork, because the complete package initializes NPU state during import.
    old_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(site),
            assets,
            *[p for p in old_paths if p and p not in (str(site), assets)],
        ]
    )
    package = importlib.import_module("cann_ops_transformer")
    actual = pathlib.Path(package.__file__).resolve()
    if actual != expected.resolve():
        raise RuntimeError(
            f"Expected installed package {expected}, but imported {actual}"
        )
    if not getattr(
        getattr(torch.ops.cann_ops_transformer, "ffn_worker_batching", None),
        "_schemas",
        None,
    ):
        raise RuntimeError(
            "Installed cann_ops_transformer did not register FFN; check competing entry points"
        )
    print(f"FFN TTK Python: {sys.executable}", flush=True)
    print(f"FFN TTK installed package: {actual}", flush=True)
    multiprocessing.set_forkserver_preload(["_ffn_ttk_worker_bootstrap"])
    from ttk.cli import main

    main()


if __name__ == "__main__":
    _run_ttk_with_installed_package()
