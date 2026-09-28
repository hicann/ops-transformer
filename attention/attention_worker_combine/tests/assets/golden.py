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

import atexit
import ctypes
import struct
import numpy as np
import ml_dtypes

__spec__ = {
    "attention_worker_combine": "AttentionWorkerCombineKernelTestSpec",
    "aclnnAttentionWorkerCombine": "AclnnAttentionWorkerCombineTestSpec",
    "torch_npu.npu_attention_worker_combine": "E2eAttentionWorkerCombineTestSpec",
}

# 按用例名保存同一份 CPU token/flags 及其 GM 地址，供各回调复用，比对结束后释放。
_DATA_CACHE = {}
# 首次使用时加载 ACL Runtime，后续复用动态库对象和 C 函数签名。
_ACL = None


def as_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    value = value.detach().cpu().contiguous()
    # BF16/E8M0 经整数视图传递原始比特，再由 ml_dtypes 解释，避免数值转换。
    if str(value.dtype) == "torch.bfloat16":
        import torch

        return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if str(value.dtype) == "torch.float8_e8m0fnu":
        import torch

        return value.view(torch.uint8).numpy().view(ml_dtypes.float8_e8m0fnu)
    return value.numpy()


def copy_into(destination, array):
    array = np.ascontiguousarray(array)
    if isinstance(destination, np.ndarray):
        if (
            destination.shape != array.shape
            or destination.dtype.itemsize != array.dtype.itemsize
        ):
            raise ValueError("Test tensor shape/itemsize does not match generated data")
        destination.view(np.uint8).reshape(-1)[:] = array.view(np.uint8).reshape(-1)
    else:
        import torch

        raw = torch.from_numpy(array.view(np.uint8).reshape(-1).copy())
        destination.view(torch.uint8).reshape(-1).copy_(raw)


def decode_fp8(raw, token_dtype):
    raw = np.asarray(raw, dtype=np.uint8)
    mantissa_bits, bias = (2, 15) if token_dtype == 2 else (3, 7)
    mantissa = (raw & ((1 << mantissa_bits) - 1)).astype(np.float32)
    exponent = ((raw & 127) >> mantissa_bits).astype(np.int32)
    # 正规数有隐含的前导 1；指数为 0 时按次正规数计算。
    normal = np.ldexp(1 + mantissa / (1 << mantissa_bits), exponent - bias)
    subnormal = np.ldexp(mantissa, 1 - bias - mantissa_bits)
    result = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    # E5M2 的全 1 指数表示 Inf/NaN；E4M3FN 仅绝对值编码 0x7f 表示 NaN。
    if token_dtype == 2:
        result = np.where(
            exponent == 31, np.where(mantissa == 0, np.inf, np.nan), result
        )
    else:
        result = np.where((raw & 127) == 127, np.nan, result)
    return np.where((raw & 128) != 0, -result, result).astype(np.float32)


def decode_fp4(packed, h):
    """每字节低/高半字节按 h 顺序展开；奇数 H 的最后高半字节不参与计算。"""
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    codes = codes[..., :h]
    # 使用独立查表解码，与 kernel 的指数/尾数算术解码互相校验。
    magnitudes = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    return np.where(codes & 8, -magnitudes[codes & 7], magnitudes[codes & 7])


def decode_e8m0(raw):
    raw = np.asarray(raw, dtype=np.uint8)

    # 编码 0..254 对应 2^(编码-127)，255 表示 NaN；编码 0 不是数值零。
    exponent = np.where(raw == 255, 127, raw).astype(np.int32) - 127
    return np.where(raw == 255, np.nan, np.ldexp(np.float32(1), exponent)).astype(
        np.float32
    )


def make_cpu_data(r, k, h, token_dtype, scheduled, repeatable_schedule=False):
    """生成一次 CPU token/flags，供 golden 和设备输入共同使用。"""
    if token_dtype not in (0, 1, 2, 3, 4) or scheduled not in (0, 1):
        raise ValueError("Unsupported token dtype or schedule mode")
    if not (r > 0 and 1 <= k <= 64 and h > 0):
        raise ValueError("Invalid dimensions")

    micro_batches = (
        2 if scheduled and repeatable_schedule else int(np.random.randint(1, 4))
    )
    current = int(np.random.randint(micro_batches))

    selected = (current + 1) % micro_batches if scheduled else current
    slots = k + 1
    dims = (micro_batches, r, slots, h)

    # 目标批次预先就绪，其他批次用哨兵值检查误清零；不模拟生产者异步写 flags。
    flags = np.full((micro_batches, r, slots), 0x13579, dtype=np.int32)
    flags[selected] = 1

    if token_dtype == 4:
        # 直接生成打包字节，覆盖 E2M1 的所有编码。每行独立补齐到字节。
        # 奇数 H 的无效高半字节也随机化，检查 kernel 是否误读尾部。
        tokens = np.random.randint(
            0, 256, (micro_batches, r, slots, (h + 1) // 2), dtype=np.uint8
        )
    elif token_dtype >= 2:
        # 按 FP8 编码随机生成有限值，排除 Inf/NaN，保留正负数和次正规数。
        tokens = np.random.randint(0, 256, dims, dtype=np.uint8)
        if token_dtype == 2:
            tokens[(tokens & 0x7C) == 0x7C] ^= 0x04
        else:
            tokens[(tokens & 0x7F) == 0x7F] ^= 0x01
    else:
        dtype = np.float16 if token_dtype == 0 else ml_dtypes.bfloat16
        tokens = (
            np.random.uniform(-10000.0, 10000.0, dims).astype(np.float32).astype(dtype)
        )

    if scheduled and repeatable_schedule:
        # TorchAir 在首次编译图时可能执行算子多次。E2E 使用内容相同且均已
        # 就绪的微批次，使每次内部执行得到同一结果；单次消费语义由 ACLNN/GEIR 校验。
        flags.fill(1)
        tokens[1:] = tokens[0]

    return {
        "tokens": tokens,
        "flags": flags,
        "hidden_size": h,
        "micro_batches": micro_batches,
        "current": current,
        "selected": selected,
        "scheduled": scheduled,
        "device_buffers": [],
        "device": None,
    }


def reference(data, scales, layer, token_dtype):
    # 使用与 NPU 相同的目标微批次；解码后统一为 [R, K+1, H] 的 FP32 数据。
    tokens = data["tokens"][data["selected"]]
    if token_dtype == 4:
        values = decode_fp4(tokens, data["hidden_size"])
    else:
        values = (
            decode_fp8(tokens, token_dtype)
            if token_dtype >= 2
            else tokens.astype(np.float32)
        )
    scales = as_numpy(scales)
    r, rows, h = values.shape
    result = np.zeros((r, h), dtype=np.float32)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if token_dtype >= 2:
            # 包括共享专家在内的 K+1 行都反量化；h//32 将每个元素映射到对应 scale。
            # 只索引有效 H，尾组不足 32 个也共用一个 scale，忽略补齐到偶数的 scale。
            expanded = decode_e8m0(scales.view(np.uint8))[..., np.arange(h) // 32]
            # 按专家顺序执行 FP32 乘法和累加，避免提前转 BF16 或改变归约顺序。
            for expert in range(rows):
                product = np.multiply(
                    values[:, expert], expanded[:, expert], dtype=np.float32
                )
                np.add(result, product, out=result)
        else:
            # 非量化前 K 行各乘一个标量权重，最后的共享专家直接累加。
            for expert in range(rows - 1):
                product = np.multiply(
                    values[:, expert], scales[:, expert, None], dtype=np.float32
                )
                np.add(result, product, out=result)
            np.add(result, values[:, -1], out=result)

    # 全部专家累加完成后只转换一次：FP16 输入输出 FP16，其余场景输出 BF16。
    y = result.astype(np.float16 if token_dtype == 0 else ml_dtypes.bfloat16)
    return y, as_numpy(layer).astype(np.int32) + 1


def build_context(data, token_address=0, flag_address=0):
    """只承载一份 ScheduleContext，数据另存；偏移对应 common_utils.h。"""
    context = np.zeros(1024, dtype=np.int8)
    struct.pack_into("<I", context, 4, data["micro_batches"])
    # 按小端布局写入：256/264 为 flags 地址/字节数，272/280 为 token 地址/字节数，
    # 288 为 uint32 微批次编号。初次构造地址为 0，设备内存准备后再填真实 GM 地址。
    struct.pack_into(
        "<QQQQI",
        context,
        256,
        flag_address,
        data["flags"].nbytes,
        token_address,
        data["tokens"].nbytes,
        data["current"],
    )
    return context


def output_like(array, template):
    if isinstance(template, np.ndarray):
        return np.asarray(array, dtype=template.dtype)
    import torch

    # ml_dtypes.bfloat16 先无损转 FP32，再转换为 torch 的 BF16。
    values = (
        array.astype(np.float32)
        if array.dtype == np.dtype(ml_dtypes.bfloat16)
        else array
    )
    return torch.from_numpy(np.ascontiguousarray(values)).to(template.dtype)


def acl_library():
    global _ACL
    if _ACL is None:
        lib = ctypes.CDLL("libascendcl.so")
        signatures = {
            "aclrtMalloc": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_int,
            ],
            "aclrtMemcpy": [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ],
            "aclrtFree": [ctypes.c_void_p],
            "aclrtGetDevice": [ctypes.POINTER(ctypes.c_int)],
            "aclrtSetDevice": [ctypes.c_int],
        }
        for name, args in signatures.items():
            getattr(lib, name).restype = ctypes.c_int
            getattr(lib, name).argtypes = args
        _ACL = lib
    return _ACL


def check_acl(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed: {status}")


def copy_to_device(array, data):
    """分配 GM 并复制真实数据；保存地址，直到比对结束才释放。"""
    array = np.ascontiguousarray(array)
    address = ctypes.c_void_p()
    lib = acl_library()
    check_acl(
        lib.aclrtMalloc(ctypes.byref(address), array.nbytes, 0), "allocate auxiliary GM"
    )
    data["device_buffers"].append(address.value)
    check_acl(
        lib.aclrtMemcpy(
            address, array.nbytes, ctypes.c_void_p(array.ctypes.data), array.nbytes, 1
        ),
        "copy CPU data to GM",
    )
    return address.value


def copy_to_device_address(array, address):
    """把 CPU 数据恢复到既有 GM 地址，供会消费 flags 的重复图执行使用。"""
    array = np.ascontiguousarray(array)
    check_acl(
        acl_library().aclrtMemcpy(
            ctypes.c_void_p(address),
            array.nbytes,
            ctypes.c_void_p(array.ctypes.data),
            array.nbytes,
            1,
        ),
        "restore auxiliary GM",
    )


def case_data(name):
    if name not in _DATA_CACHE:
        raise RuntimeError(
            f"CPU input cache missing for {name}; customize_inputs must run in this process"
        )
    return _DATA_CACHE[name]


def release_device_buffers(data):
    if data["device_buffers"]:
        lib = acl_library()
        check_acl(lib.aclrtSetDevice(data["device"]), "restore auxiliary device")
        while data["device_buffers"]:
            check_acl(
                lib.aclrtFree(ctypes.c_void_p(data["device_buffers"][-1])),
                "free auxiliary GM",
            )
            data["device_buffers"].pop()


def release_case(name):
    data = _DATA_CACHE.get(name)
    if data is None:
        return
    release_device_buffers(data)
    _DATA_CACHE.pop(name)


def prepare_auxiliary(schedule_context, data):
    """为本次 NPU 执行重新准备 token/flags，并把真实 GM 地址写入 context。"""
    device = ctypes.c_int(-1)
    check_acl(acl_library().aclrtGetDevice(ctypes.byref(device)), "get TTK device")
    data["device"] = device.value

    if data["device_buffers"]:
        # torch.compile 首次调用会完成编译并真实执行。后续执行必须恢复已被
        # kernel 清零的 flags，同时保持 ScheduleContext 内嵌的 GM 地址不变。
        token_address = data["token_address"]
        flag_address = data["flag_address"]
        copy_to_device_address(data["tokens"], token_address)
        copy_to_device_address(data["flags"], flag_address)
    else:
        token_address = copy_to_device(data["tokens"], data)
        flag_address = copy_to_device(data["flags"], data)
        data["token_address"] = token_address
        data["flag_address"] = flag_address

    # 同一个图输入 tensor 重复执行时只恢复其指向的 flags，避免再次原地写
    # context 触发 torch.compile 重新编译。Kernel 还会推进 micro_batch_id，
    # 因此通过 ACL 直接恢复这 4 字节；切换执行模式的 tensor 时再回填完整结构体。
    if data.get("context_object") is not schedule_context:
        copy_into(schedule_context, build_context(data, token_address, flag_address))
        data["context_object"] = schedule_context
    elif not isinstance(schedule_context, np.ndarray):
        current = np.asarray([data["current"]], dtype=np.uint32)
        copy_to_device_address(current, schedule_context.data_ptr() + 288)


def customize_case_inputs(
    schedule_context,
    expert_scales,
    hidden_size,
    token_dtype,
    need_schedule,
    testcase_name,
    repeatable_schedule=False,
):
    """根据公开输入 shape/属性构造一份 CPU 辅助数据，并初始化 context。"""
    release_case(testcase_name)
    r, scale_rows = expert_scales.shape[:2]
    # MXFP scale 包含共享专家，共 K+1 行；非量化 scale 只有 K 行。
    k = scale_rows - 1 if token_dtype >= 2 else scale_rows
    data = make_cpu_data(
        r,
        k,
        hidden_size,
        token_dtype,
        need_schedule,
        repeatable_schedule,
    )
    _DATA_CACHE[testcase_name] = data
    copy_into(schedule_context, build_context(data))


@atexit.register
def cleanup():
    # 正常情况 compare 立即释放；异常退出时兜底清理。
    for name in list(_DATA_CACHE):
        try:
            release_case(name)
        except RuntimeError:
            pass


def compare_values(actual, expected):
    a, b = as_numpy(actual), as_numpy(expected)
    if a.shape != b.shape:
        return {
            "pass": False,
            "precision": "FAIL",
            "error_info": f"shape mismatch {a.shape} != {b.shape}",
        }
    if a.dtype != b.dtype:
        return {
            "pass": False,
            "precision": "FAIL",
            "error_info": f"dtype mismatch {a.dtype} != {b.dtype}",
        }
    if a.dtype.kind == "f" or a.dtype == np.dtype(ml_dtypes.bfloat16):
        same_nan = np.isnan(a.astype(np.float32)) & np.isnan(b.astype(np.float32))
    else:
        same_nan = np.zeros(a.shape, dtype=bool)
    # 除两侧同为 NaN（允许 payload 不同）外，逐元素要求原始比特完全一致。
    equal_bits = (
        (
            a.reshape(-1).copy().view(np.uint8).reshape(a.size, a.dtype.itemsize)
            == b.reshape(-1).copy().view(np.uint8).reshape(b.size, b.dtype.itemsize)
        )
        .all(axis=1)
        .reshape(a.shape)
    )
    errors = int(np.count_nonzero(~(equal_bits | same_nan)))
    return {
        "pass": errors == 0,
        "precision": f"{a.size - errors}/{a.size}",
        "error_info": None
        if errors == 0
        else f"{errors} unequal elements (NaN payloads ignored)",
    }


class AclnnAttentionWorkerCombineTestSpec:
    @staticmethod
    def customize_inputs(
        scheduleContext,
        expertScales,
        layerId,
        hiddenSize,
        tokenDtype,
        needSchedule,
        yOut,
        nextLayerIdOut,
        **kwargs,
    ):
        """TTK 已创建 CPU tensor；这里填内容并缓存真实 token，不初始化设备。"""
        customize_case_inputs(
            scheduleContext,
            expertScales,
            hiddenSize,
            tokenDtype,
            needSchedule,
            kwargs["testcase_name"],
        )

    @staticmethod
    def npu_preprocess(
        scheduleContext,
        expertScales,
        layerId,
        hiddenSize,
        tokenDtype,
        needSchedule,
        yOut,
        nextLayerIdOut,
        **kwargs,
    ):
        """TTK 选好设备后调用：真实数据 H2D，再将 GM 地址填入 CPU context。"""
        name = kwargs["testcase_name"]
        data = case_data(name)
        try:
            prepare_auxiliary(scheduleContext, data)
        except Exception:
            release_case(name)
            raise

    @staticmethod
    def golden(
        scheduleContext,
        expertScales,
        layerId,
        hiddenSize,
        tokenDtype,
        needSchedule,
        yOut,
        nextLayerIdOut,
        **kwargs,
    ):
        data = case_data(kwargs["testcase_name"])
        # 从 CPU 缓存计算，不解引用 context 中的设备地址。
        y, next_layer = reference(data, expertScales, layerId, tokenDtype)
        # 只生成 ACLNN 原型声明的两个输出，不把 scheduleContext 的内部状态作为输出。
        return [
            output_like(y, yOut),
            output_like(next_layer, nextLayerIdOut),
        ]

    @staticmethod
    def compare(
        y,
        next_layer,
        expected_y,
        expected_next,
        *,
        compare_context,
        **kwargs,
    ):
        name = compare_context.testcase_name
        results = [
            compare_values(a, b)
            for a, b in zip((y, next_layer), (expected_y, expected_next))
        ]
        try:
            return results
        finally:
            release_case(name)

    tolerance = {
        dtype: {"standard": "binary_equal"}
        for dtype in ("float16", "bfloat16", "int32", "int8")
    }


class AttentionWorkerCombineKernelTestSpec:
    """Kernel/GEIR 模式使用的三输入标杆。"""

    @staticmethod
    def customize_inputs(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=0,
        need_schedule=0,
        **kwargs,
    ):
        name = kwargs["testcase_name"]
        customize_case_inputs(
            schedule_context,
            expert_scales,
            hidden_size,
            token_dtype,
            need_schedule,
            name,
        )

        case_data(name)["geir_context"] = schedule_context
        return schedule_context, expert_scales, layer_id

    @staticmethod
    def geir_prepare_inputs(input_arrays, attributes, input_prefix):
        """导出 CPU token/flags；由 GEIR runner 分配 GM 并回填两个指针。"""
        data = next(
            (
                item
                for item in _DATA_CACHE.values()
                if item.get("geir_context") is input_arrays[0]
            ),
            None,
        )
        if data is None:
            raise RuntimeError("GEIR context has no matching CPU token/flags")
        entries = []
        for offset, payload in ((256, data["flags"]), (272, data["tokens"])):
            path = f"{input_prefix}_aux_{offset}.bin"
            payload.tofile(path)
            entries.append(
                dict(input_index=0, offset=offset, size=payload.nbytes, path=path)
            )
        return entries

    @staticmethod
    def golden(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=0,
        need_schedule=0,
        **kwargs,
    ):
        data = case_data(kwargs["testcase_name"])
        return list(reference(data, expert_scales, layer_id, token_dtype))

    @staticmethod
    def compare(
        y,
        next_layer,
        expected_y,
        expected_next,
        *,
        compare_context,
        **kwargs,
    ):
        results = [
            compare_values(a, b)
            for a, b in zip((y, next_layer), (expected_y, expected_next))
        ]
        release_case(compare_context.testcase_name)
        return results

    tolerance = {
        dtype: {"standard": "binary_equal"}
        for dtype in ("float16", "bfloat16", "int32", "int8")
    }


class E2eAttentionWorkerCombineTestSpec:
    """torch eager、静态图和动态图共用的输入准备与 CPU 标杆。"""

    @staticmethod
    def customize_inputs(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=0,
        need_schedule=0,
        **kwargs,
    ):
        customize_case_inputs(
            schedule_context,
            expert_scales,
            hidden_size,
            token_dtype,
            need_schedule,
            kwargs["testcase_name"],
            repeatable_schedule=True,
        )

    @staticmethod
    def npu_preprocess(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=0,
        need_schedule=0,
        **kwargs,
    ):
        name = kwargs["testcase_name"]
        data = case_data(name)
        try:
            prepare_auxiliary(schedule_context, data)
        except Exception:
            release_case(name)
            raise

    @staticmethod
    def golden(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=0,
        need_schedule=0,
        **kwargs,
    ):
        data = case_data(kwargs["testcase_name"])
        return list(reference(data, expert_scales, layer_id, token_dtype))

    @staticmethod
    def compare(
        y,
        next_layer,
        expected_y,
        expected_next,
        *,
        compare_context,
        **kwargs,
    ):
        results = [
            compare_values(a, b)
            for a, b in zip((y, next_layer), (expected_y, expected_next))
        ]
        # 所有被测模式都已执行完毕；后续各模式的 compare 重复释放也是安全的。
        release_case(compare_context.testcase_name)
        return results

    tolerance = {
        dtype: {"standard": "binary_equal"}
        for dtype in ("float16", "bfloat16", "int32", "int8")
    }
