# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Exercise the installed aclnnKeyPool ABI; requires KEYPOOL_OPAPI_LIBRARY."""

import argparse
import ctypes as C
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    home = os.environ["ASCEND_HOME_PATH"]
    lib = os.path.join(home, "lib64")
    acl = C.CDLL(os.path.join(lib, "libascendcl.so"), mode=C.RTLD_GLOBAL)
    nnop = C.CDLL(os.path.join(lib, "libnnopbase.so"), mode=C.RTLD_GLOBAL)
    op = C.CDLL(os.environ["KEYPOOL_OPAPI_LIBRARY"], mode=C.RTLD_GLOBAL)
    acl.aclInit.argtypes = [C.c_char_p]
    acl.aclrtSetDevice.argtypes = [C.c_int32]
    acl.aclrtCreateStream.argtypes = [C.POINTER(C.c_void_p)]
    acl.aclrtDestroyStream.argtypes = [C.c_void_p]
    acl.aclrtMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_int]
    acl.aclrtFree.argtypes = [C.c_void_p]
    acl.aclrtMemcpy.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int]
    acl.aclrtSynchronizeStream.argtypes = [C.c_void_p]
    nnop.aclCreateTensor.restype = C.c_void_p
    nnop.aclCreateTensor.argtypes = [
        C.POINTER(C.c_int64),
        C.c_uint64,
        C.c_int32,
        C.POINTER(C.c_int64),
        C.c_int64,
        C.c_int32,
        C.POINTER(C.c_int64),
        C.c_uint64,
        C.c_void_p,
    ]
    nnop.aclDestroyTensor.argtypes = [C.c_void_p]
    gws = op.aclnnKeyPoolGetWorkspaceSize
    gws.restype = C.c_int32
    gws.argtypes = [C.c_void_p] * 13 + [
        C.c_int64,
        C.c_double,
        C.c_int64,
        C.c_int64,
        C.c_void_p,
        C.POINTER(C.c_uint64),
        C.POINTER(C.c_void_p),
    ]
    execute = op.aclnnKeyPool
    execute.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
    device = int(os.environ.get("ACL_DEVICE_ID", "7"))
    assert acl.aclInit(None) == 0
    assert acl.aclrtSetDevice(device) == 0
    stream = C.c_void_p()
    assert acl.aclrtCreateStream(C.byref(stream)) == 0
    resources = []
    failures = []

    def tensor(host, dtype, shape):
        dev = C.c_void_p()
        assert acl.aclrtMalloc(C.byref(dev), max(C.sizeof(host), 2), 0) == 0
        if C.sizeof(host):
            assert acl.aclrtMemcpy(dev, C.sizeof(host), host, C.sizeof(host), 1) == 0
        dims = (C.c_int64 * len(shape))(*shape)
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = shape[i + 1] * stride[i + 1]
        strides = (C.c_int64 * len(shape))(*stride)
        handle = nnop.aclCreateTensor(
            dims, len(shape), dtype, strides, 0, 2, dims, len(shape), dev
        )
        assert handle
        resources.append((handle, dev))
        return handle, dev, host

    for dtype, one in ((1, 0x3C00), (27, 0x3F80)):
        f16 = lambda n, bits=one: (C.c_uint16 * n)(*([bits] * n))
        f32 = lambda n: (C.c_float * n)()
        i32 = lambda values: (C.c_int32 * len(values))(*values)
        hidden = tensor(f16(2 * 4 * 1024), dtype, (2, 4, 1024))
        wk = tensor(f16(128 * 1024), dtype, (128, 1024))
        gate = tensor(f16(128 * 1024), dtype, (128, 1024))
        ape = tensor(f32(4 * 128), 0, (4, 128))
        cache = tensor(f32(3 * 4 * 256), 0, (3, 4, 256))
        table = tensor(i32([1, 2]), 3, (2, 1))
        start = tensor(i32([0, 0]), 3, (2,))
        out = tensor(f16(2 * 128, 0), dtype, (2, 1, 128))
        cos = tensor(f16(2 * 4 * 128, 0x3F40), 27, (2, 4, 128))
        sin = tensor(f16(2 * 4 * 128, 0x3F00), 27, (2, 4, 128))
        empty_cos = tensor(f16(0), 27, (0,))
        inputs = [x[0] for x in (hidden, wk, gate, ape, cache, table, start)]
        cases = [(0, None, None), (1, None, None)]
        cases += [
            (mode, None, None)
            for mode in (-1, 2, 7, 2**32, 2**32 + 1, 2**63 - 1, -(2**63))
        ]
        cases += [
            (mode, c, s)
            for mode in (0, 1)
            for c, s in (
                (cos[0], None),
                (None, sin[0]),
                (cos[0], sin[0]),
                (empty_cos[0], None),
            )
        ]
        baseline = None
        for mode, cos_ptr, sin_ptr in cases:
            expected = (
                0 if mode in (0, 1) and cos_ptr is None and sin_ptr is None else 561002
            )
            ws, executor = C.c_uint64(), C.c_void_p()
            rc = gws(
                *inputs,
                None,
                None,
                cos_ptr,
                sin_ptr,
                None,
                None,
                4,
                1e-6,
                mode,
                1024,
                out[0],
                C.byref(ws),
                C.byref(executor),
            )
            record = dict(
                dtype=dtype,
                mode=mode,
                cos=cos_ptr is not None,
                sin=sin_ptr is not None,
                rc=rc,
                expected=expected,
            )
            if rc != expected:
                failures.append(record)
            if rc == 0 and (expected == 0 or args.observe):
                assert (
                    acl.aclrtMemcpy(
                        cache[1], C.sizeof(cache[2]), cache[2], C.sizeof(cache[2]), 1
                    )
                    == 0
                )
                workspace = C.c_void_p()
                if ws.value:
                    assert acl.aclrtMalloc(C.byref(workspace), ws.value, 0) == 0
                record["execute_rc"] = execute(workspace, ws, executor, stream)
                assert record["execute_rc"] == 0
                assert acl.aclrtSynchronizeStream(stream) == 0
                actual = (C.c_uint16 * 256)()
                assert (
                    acl.aclrtMemcpy(
                        actual, C.sizeof(actual), out[1], C.sizeof(actual), 2
                    )
                    == 0
                )
                # Unit projection weights and hidden states yield K=1024;
                # identical tokens have identical pooled values.
                expected_bits = 0x6400 if dtype == 1 else 0x4480
                record["golden_equal"] = all(x == expected_bits for x in actual)
                assert record["golden_equal"]
                if baseline is None:
                    baseline = bytes(actual)
                record["baseline_equal"] = baseline == bytes(actual)
                assert record["baseline_equal"]
                if ws.value:
                    assert acl.aclrtFree(workspace) == 0
            print(json.dumps(record), flush=True)
    for handle, dev in reversed(resources):
        assert nnop.aclDestroyTensor(handle) == 0
        assert acl.aclrtFree(dev) == 0
    assert acl.aclrtDestroyStream(stream) == 0
    assert acl.aclrtResetDevice(device) == 0
    assert acl.aclFinalize() == 0
    print(json.dumps(dict(failed=len(failures))), flush=True)
    return 1 if failures and not args.observe else 0


if __name__ == "__main__":
    raise SystemExit(main())
