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
  - inputs: scheduleContext packing with HBM buffers + test data generation
  - golden: CPU golden reference using cached test data
  - compare: custom comparison (kernel writes only actual_token_num elements)
"""

import atexit
import ctypes
import struct
import numpy as np
import torch

__spec__ = {"aclnnFfnWorkerBatching": "AclnnFfnWorkerBatchingTestSpec"}

# ==================== Shared cache ====================

_DATA_CACHE = {}
_HBM_ALLOCS = []
_ACL_LIB = None
_MASK_VALUE = 1000000

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
_OFF_TEST_MAGIC = 640
_TEST_MAGIC = 0x54455354

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
_HBM_BLOCK_SIZE = 4 * 1024 * 1024


def _hbm_alloc(size):
    global _HBM_BASE, _HBM_OFFSET
    lib = _load_acl()
    lib.aclrtSetDevice(0)
    if _HBM_BASE is None or _HBM_OFFSET + size > _HBM_BLOCK_SIZE:
        ptr = ctypes.c_void_p(0)
        ret = lib.aclrtMalloc(ctypes.byref(ptr), _HBM_BLOCK_SIZE, 0)
        if ret != 0 or ptr.value is None:
            raise RuntimeError(f"aclrtMalloc failed: ret={ret}, size={_HBM_BLOCK_SIZE}")
        _HBM_BASE = ptr.value
        _HBM_OFFSET = 0
        _HBM_ALLOCS.append(_HBM_BASE)
    aligned_size = (size + 511) & ~511
    addr = _HBM_BASE + _HBM_OFFSET
    _HBM_OFFSET += aligned_size
    return addr


def _hbm_copy(host_data, hbm_ptr):
    lib = _load_acl()
    if isinstance(host_data, np.ndarray):
        buf = host_data.tobytes()
    elif isinstance(host_data, (bytes, bytearray)):
        buf = bytes(host_data)
    else:
        buf = host_data
    ret = lib.aclrtMemcpy(hbm_ptr, len(buf), ctypes.c_char_p(buf), len(buf), 1)
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy H2D failed: ret={ret}")
    lib.aclrtMemInvalidate(ctypes.c_void_p(hbm_ptr), len(buf))


@atexit.register
def _cleanup_hbm():
    global _HBM_BASE, _HBM_OFFSET
    lib = _ACL_LIB
    if lib is None:
        return
    for ptr in _HBM_ALLOCS:
        try:
            lib.aclrtFree(ctypes.c_void_p(ptr))
        except Exception:
            pass
    _HBM_ALLOCS.clear()
    _HBM_BASE = None
    _HBM_OFFSET = 0


# ==================== Data generation ====================


def _seed_from_name(name):
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _pack_ctx(ctx_bytes, offset, fmt, value):
    struct.pack_into(fmt, ctx_bytes, offset, value)


def _generate_test_data(
    testcase_name, A, M, BS, K, H, expertNum, tokenDtype, needSchedule
):
    rng = np.random.RandomState(_seed_from_name(testcase_name))
    out_num = A
    has_mask = "masked" in testcase_name
    single_expert = "single_expert" in testcase_name

    if needSchedule == 0:
        expert_ids = rng.randint(0, expertNum, size=(out_num, BS, K)).astype(np.int32)
    else:
        expert_ids = rng.randint(0, expertNum, size=(A, M, BS, K)).astype(np.int32)

    if single_expert:
        expert_ids[...] = 0
    elif has_mask:
        mask = rng.random(expert_ids.shape) < 0.15
        expert_ids[mask] = _MASK_VALUE

    if tokenDtype == 0:
        token_data = (rng.randn(A, M, BS, K, H) * 0.1).astype(np.float16)
        token_scales = None
    elif tokenDtype == 1:
        token_data = (rng.randn(A, M, BS, K, H) * 0.1).astype(np.float32)
        token_scales = None
    else:
        token_data = rng.randint(-128, 127, size=(A, M, BS, K, H)).astype(np.int8)
        token_scales = (rng.rand(A, M, BS, K) * 2 + 0.1).astype(np.float32)

    session_ids = np.arange(A, dtype=np.int32)
    micro_batch_ids = np.zeros(A, dtype=np.int32)

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


def _get_cached_data(testcase_name, **gen_kwargs):
    if testcase_name not in _DATA_CACHE:
        if gen_kwargs:
            try:
                data = _generate_test_data(
                    testcase_name,
                    gen_kwargs.get("A", 1),
                    gen_kwargs.get("M", 1),
                    gen_kwargs.get("BS", 1),
                    gen_kwargs.get("K", 1),
                    gen_kwargs.get("H", 1),
                    gen_kwargs.get("expertNum", 1),
                    gen_kwargs.get("tokenDtype", 0),
                    gen_kwargs.get("needSchedule", 0),
                )
                data["HS"] = gen_kwargs.get("HS", 1)
                _DATA_CACHE[testcase_name] = data
            except Exception:
                pass
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
    offset = 0
    cur_expert = int(sorted_expert_ids[0])
    token_count = 0
    for i in range(valid_count):
        eid = int(sorted_expert_ids[i])
        if eid != cur_expert:
            if offset < expert_num:
                group_list[offset, 0] = cur_expert
                group_list[offset, 1] = token_count
                offset += 1
            cur_expert = eid
            token_count = 1
        else:
            token_count += 1
    if offset < expert_num:
        group_list[offset, 0] = cur_expert
        group_list[offset, 1] = token_count
        offset += 1
    if offset < expert_num:
        group_list[offset, 0] = 0
        group_list[offset, 1] = 0
    return group_list


def _to_torch(arr, dtype):
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

        data = _generate_test_data(
            testcase_name, A, M, BS, K, H, expertNum, tokenDtype, needSchedule
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

        td_arr = _build_token_data_bytes(
            token_data, token_scales, tokenDtype, A, M, BS, K, H
        )
        td_bytes = td_arr.tobytes()

        if needSchedule == 0:
            ei_bytes = np.ascontiguousarray(expert_ids).tobytes()
            si_bytes = np.ascontiguousarray(session_ids).tobytes()
            mi_bytes = np.ascontiguousarray(micro_batch_ids).tobytes()
            aux_data = td_bytes + ei_bytes + si_bytes + mi_bytes
            td_offset = 1024
            ei_offset = td_offset + len(td_bytes)
            si_offset = ei_offset + len(ei_bytes)
            mi_offset = si_offset + len(si_bytes)
        else:
            ti_bytes = _build_token_info_buf(expert_ids, A, M, BS, K).tobytes()
            aux_data = td_bytes + ti_bytes
            td_offset = 1024
            ti_offset = td_offset + len(td_bytes)

        total_size = 1024 + len(aux_data)
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
        _pack_ctx(ctx, _OFF_TEST_MAGIC, "<I", _TEST_MAGIC)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF, "<Q", td_offset)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF_SIZE, "<Q", len(td_bytes))

        if needSchedule == 0:
            _pack_ctx(ctx, _OFF_SESSION_IDS_BUF, "<Q", si_offset)
            _pack_ctx(ctx, _OFF_SESSION_IDS_BUF_SIZE, "<Q", len(si_bytes))
            _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF, "<Q", mi_offset)
            _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF_SIZE, "<Q", len(mi_bytes))
            _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF, "<Q", ei_offset)
            _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF_SIZE, "<Q", len(ei_bytes))
            _pack_ctx(ctx, _OFF_OUT_NUM, "<I", out_num)
        else:
            _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF, "<Q", ti_offset)
            _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF_SIZE, "<Q", len(ti_bytes))
            _pack_ctx(ctx, _OFF_POLLING_INDEX, "<Q", 0)

        ctx[1024 : 1024 + len(aux_data)] = aux_data

        ctx_arr = np.array(ctx, dtype=np.int8)
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
        else:
            HS = H + 4
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
        else:
            out_torch_dtype = torch.int8
            out_np_dtype = np.int8

        expert_ids_flat = expert_ids.reshape(-1).astype(np.int32)
        sorted_order, valid_count, sorted_valid_ids = _sort_expert_ids(expert_ids_flat)

        y_out = np.zeros((Y, H), dtype=out_np_dtype)
        session_ids_out = np.zeros(Y, dtype=np.int32)
        micro_batch_ids_out = np.zeros(Y, dtype=np.int32)
        token_ids_out = np.zeros(Y, dtype=np.int32)
        expert_offsets_out = np.zeros(Y, dtype=np.int32)
        dynamic_scale_out = np.ones(Y, dtype=np.float32)

        bsk = BS * K
        cur_mb = data.get("cur_micro_batch_id", 0)

        for i in range(valid_count):
            gidx = int(sorted_order[i])
            a_idx = gidx // bsk
            rem = gidx % bsk
            bs_idx = rem // K
            k_idx = rem % K

            if needSchedule == 0:
                s_idx = int(session_ids_buf[a_idx])
                mb_idx = int(micro_batch_ids_buf[a_idx])
            else:
                s_idx = a_idx
                mb_idx = cur_mb

            session_ids_out[i] = s_idx
            micro_batch_ids_out[i] = mb_idx
            token_ids_out[i] = bs_idx
            expert_offsets_out[i] = k_idx
            y_out[i] = token_data[s_idx, mb_idx, bs_idx, k_idx, :]

            if tokenDtype == 2 and token_scales is not None:
                dynamic_scale_out[i] = token_scales[s_idx, mb_idx, bs_idx, k_idx]

        group_list_out = _generate_group_list(sorted_valid_ids, valid_count, expertNum)
        actual_token_num_out = np.array([valid_count], dtype=np.int64)

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
        num_outputs = len(outputs) // 2
        npu_outputs = outputs[:num_outputs]
        golden_outputs = outputs[num_outputs:]

        def to_np(t):
            if isinstance(t, torch.Tensor):
                if t.dtype == torch.bfloat16:
                    return t.to(torch.float32).numpy()
                return t.numpy()
            return np.asarray(t)

        actual_token_num = int(to_np(golden_outputs[7]).flat[0])
        results = []
        for idx in range(num_outputs):
            npu = to_np(npu_outputs[idx])
            gold = to_np(golden_outputs[idx])
            if idx == 1:
                npu_arr = npu.reshape(-1, 2)
                gold_arr = gold.reshape(-1, 2)
                valid = 0
                for row in gold_arr:
                    if row[0] == 0 and row[1] == 0:
                        break
                    valid += 1
                if valid == 0:
                    passed = True
                    precision = "100%"
                else:
                    match = int(np.sum(npu_arr[:valid] == gold_arr[:valid]))
                    total = valid * 2
                    passed = match == total
                    precision = f"{match / total * 100}%"
                results.append(
                    {
                        "pass": passed,
                        "precision": precision,
                        "error_info": None if passed else "group_list mismatch",
                    }
                )
            elif idx in (0, 2, 3, 4, 5):
                npu_flat = npu.flatten()
                gold_flat = gold.flatten()
                n = (
                    actual_token_num * (npu.shape[-1] if npu.ndim > 1 else 1)
                    if idx == 0
                    else actual_token_num
                )
                n = min(n, npu_flat.size)
                diff = int(np.sum(npu_flat[:n] != gold_flat[:n]))
                passed = diff == 0
                precision = f"{(n - diff) / n * 100}%" if n > 0 else "100%"
                results.append(
                    {
                        "pass": passed,
                        "precision": precision,
                        "error_info": None
                        if passed
                        else f"{diff} mismatches [idx={idx}]",
                    }
                )
            else:
                npu_flat = npu.flatten()
                gold_flat = gold.flatten()
                if npu_flat.size != gold_flat.size:
                    n = min(npu_flat.size, gold_flat.size)
                    diff = int(np.sum(npu_flat[:n] != gold_flat[:n]))
                    total = n
                else:
                    diff = int(np.sum(npu_flat != gold_flat))
                    total = npu_flat.size
                passed = diff == 0
                precision = f"{(total - diff) / total * 100}%" if total > 0 else "100%"
                results.append(
                    {
                        "pass": passed,
                        "precision": precision,
                        "error_info": None
                        if passed
                        else f"{diff} mismatches [idx={idx}]",
                    }
                )
        return results
