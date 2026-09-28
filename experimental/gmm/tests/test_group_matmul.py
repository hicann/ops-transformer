# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small NPU execution tests; reference is FP32 matmul on CPU.

Uses the source tests' group semantics and manually packed NZ layout.
Raises on numerical failure, checks GmmAdd aliasing and WithZero inactive rows.
"""

import importlib
import os

import torch
import torch_npu

PACKAGE = os.environ.get("CANN_OPS_TRANSFORMER_PACKAGE", "cann_ops_transformer")
importlib.import_module(PACKAGE)
ops = torch.ops.cann_ops_transformer


def check(label, actual, reference):
    result = actual.detach().cpu()
    expected = reference.to(result.dtype)
    tolerances = {
        torch.float16: (1e-3, 2e-3),
        torch.bfloat16: (1e-2, 2e-2),
        torch.float32: (1e-3, 1e-3),
    }
    rtol, atol = tolerances[result.dtype]
    assert torch.isfinite(result).all(), f"{label}: non-finite output"
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    error = (result.float() - expected.float()).abs().max().item()
    print(f"PASS {label} shape={tuple(result.shape)} max_abs={error:.8g}", flush=True)


def test_k_groups(device):
    sizes = [32, 0, 48, 16]
    total_k, m, n = sum(sizes), 64, 96
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        for trans_a in (False, True):
            a = (torch.randn((total_k, m) if trans_a else (m, total_k)) * 0.25).to(
                dtype
            )
            b = (torch.randn(total_k, n) * 0.25).to(dtype)
            ref = torch.zeros(len(sizes), m, n)
            start = 0
            for i, size in enumerate(sizes):
                end = start + size
                a_group = a[start:end].float().T if trans_a else a[:, start:end].float()
                ref[i] = a_group @ b[start:end].float()
                start = end
            a_npu, b_npu = a.to(device), b.to(device)
            groups = torch.tensor(sizes, dtype=torch.int64, device=device)
            for promoted in (False, True):
                output_dtype = torch.float32 if promoted else dtype
                label = f"{dtype} trans_a={trans_a} output={output_dtype}"
                y = ops.gmm(
                    a_npu, b_npu, groups, trans_a=trans_a, type_promotion=promoted
                )
                check("GmmKDim " + label, y, ref)
                assert torch.count_nonzero(y[1]).item() == 0, (
                    "empty K group must stay zero"
                )
                c_cpu = (torch.randn(len(sizes), m, n) * 0.25).to(output_dtype)
                c_npu = c_cpu.to(device)
                y = ops.gmm(
                    a_npu, b_npu, groups, trans_a=trans_a, c=c_npu, aicore_num=0
                )
                assert y.data_ptr() == c_npu.data_ptr(), (
                    "GmmAdd must return the input c storage"
                )
                check("GmmAdd " + label, y, ref + c_cpu.float())
                torch.testing.assert_close(y[1].cpu(), c_cpu[1], rtol=0, atol=0)
                count += 2
    return count


def test_local_experts(device):
    sizes = [16, 32, 0, 48, 16]
    start_exp, end_exp = 1, 4
    k, n = 64, 96
    begin, end = sum(sizes[:start_exp]), sum(sizes[:end_exp])
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        a = (torch.randn(sum(sizes), k) * 0.25).to(dtype)
        w = (torch.randn(end_exp - start_exp, n, k) * 0.25).to(dtype)
        ref = torch.zeros(sum(sizes), n)
        offset = begin
        for i in range(start_exp, end_exp):
            next_offset = offset + sizes[i]
            ref[offset:next_offset] = (
                a[offset:next_offset].float() @ w[i - start_exp].float().T
            )
            offset = next_offset
        a_npu = a.to(device)
        groups = torch.tensor(sizes, dtype=torch.int32, device=device)
        for layout in ("ND_transposed", "ND", "NZ_packed"):
            if layout == "NZ_packed":
                # Original source test packing, passed as a logical 3-D tensor.
                weight = (
                    w.reshape(w.shape[0], n, k // 16, 16)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape_as(w)
                )
            elif layout == "ND":
                weight = w.transpose(1, 2).contiguous()
            else:
                weight = w
            weight_npu = weight.to(device)
            for promoted in (False, True):
                for zero in (False, True):
                    func = ops.local_exp_gmm_with_zero if zero else ops.local_exp_gmm
                    label = f"{'GmmLocalExpWithZero' if zero else 'GmmLocalExp'} {dtype} {layout} fp32={promoted}"
                    y = func(
                        a_npu,
                        weight_npu,
                        groups,
                        start_exp,
                        end_exp,
                        trans_b=layout != "ND",
                        is_b_nz=layout == "NZ_packed",
                        type_promotion=promoted,
                    )
                    if zero:
                        check(label, y, ref)
                        assert torch.count_nonzero(y[:begin]).item() == 0
                        assert torch.count_nonzero(y[end:]).item() == 0
                    else:
                        # Ordinary LocalExp leaves inactive rows unspecified.
                        check(label, y[begin:end], ref[begin:end])
                    count += 1
    return count


def test_meta():
    a = torch.empty((64, 96), device="meta", dtype=torch.bfloat16)
    b = torch.empty((96, 80), device="meta", dtype=torch.bfloat16)
    groups = torch.empty(4, device="meta", dtype=torch.int32)
    y = ops.gmm(a, b, groups, type_promotion=True)
    assert y.shape == (4, 64, 80) and y.dtype == torch.float32
    c = torch.empty_like(y)
    assert ops.gmm(a, b, groups, c=c) is c
    for name in ("local_exp_gmm", "local_exp_gmm_with_zero"):
        x = torch.empty((112, 64), device="meta", dtype=torch.float16)
        w = torch.empty((3, 96, 64), device="meta", dtype=torch.float16)
        y = getattr(ops, name)(x, w, groups, 1, 4, trans_b=True)
        assert y.shape == (112, 96) and y.dtype == torch.float16
    print("PASS Meta shape/dtype/alias checks", flush=True)


if __name__ == "__main__":
    torch.manual_seed(20260924)
    torch.set_num_threads(4)
    device = torch.device(os.environ.get("NPU_DEVICE", "npu:11"))
    torch_npu.npu.set_device(device)
    torch.use_deterministic_algorithms(True)
    print(
        f"package={PACKAGE}, device={device}, torch={torch.__version__}, torch_npu={torch_npu.__version__}",
        flush=True,
    )
    test_meta()
    count = test_k_groups(device) + test_local_experts(device)
    torch.npu.synchronize()
    print(
        f"OVERALL PASS: {count} NPU cases; peak_allocated={torch.npu.max_memory_allocated(device) / 2**20:.2f} MiB",
        flush=True,
    )
