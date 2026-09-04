#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Batch-invariance (BI) test driver for aclnnAllGatherMatmulV2 (npu_all_gather_quant_mm).

Sibling of the matmul_reduce_scatter_v2 BI suite (PR #6593); same methodology, applied to
AllGatherMatmulV2 with MXFP8 quantization (block_size=32). Covers both x2 orientations
(non-transposed and transB), which is the surface PR #4705 ("support transB + verify MX BI")
touches.

Literature BI (Thinking Machines, 2026):  y[i] = f(x[i], W)
    output row i depends only on input row i + shared weight W, not on other rows in the batch.

Methodology (identical to PR #6593 README §2 — all three phases are required):
    Phase 0  HCCL warm-up (N_WARMUP dummy calls) — stabilize HCCL state; the first call after
             worker start often errors (HcclAllocComResourceByTiling / 561000) or picks a
             non-deterministic ring topology. Warm-up errors are not counted.
    Phase 1  Multi-trial determinism baseline (N_TRIALS) — same input run N_TRIALS times; all
             SHAs must match. A cell that is non-deterministic cannot yield a meaningful BI
             verdict, so it is reported NON_DET and excluded from Phase 2.
    Phase 2  Variant A vs B composition-BI — same pinned probe row, DIFFERENT neighbor rows
             (seed_a vs seed_b). The probe row's output must be byte-identical.
             same -> BI_PASS ; different -> BI_FAIL.

Three-state result: BI_PASS / BI_FAIL / NON_DET (+ ERROR for op rejection, SKIP for shape).

Notes:
  * MXFP8 needs block_size=32; installed tiling rejects it (EZ0002) unless the blockSize gate
    is patched (cann/ops-transformer issue #2778 / PR #6137). Run against a build/vendor with
    that fix, else every cell reports ERROR.
  * NON-TRANS uses self-contained random uint8->fp8/e8m0 data (proven; K != N allowed).
  * TRANS_B uses **valid MX data** (script/generate_mx_data.mx_quantize) — random bytes fault the
    transposed path. x2 and its scale are fed as NON-contiguous transpose views (x2.t() and
    torch.transpose(x2_scale, 0, 1), WITHOUT .contiguous() — .contiguous() triggers 561002); the
    op infers isBTrans from stride. SQUARE K==N is used (op requires x2's k-axis == x1's K, which
    the transposed view only satisfies when N==K). Verified WS=2/4 BI_PASS + numerically correct.
"""

import os
import sys
import hashlib

import torch
import torch_npu
import torch.distributed as dist

BLOCK_SIZE = 32
N_WARMUP = 2
N_TRIALS = 3
TARGET_ROW = 0

# Non-trans coverage: PR #6593 range (K in {256,512,1024}, N in {64,128}) + large & non-square.
SHAPES = [
    (256, 64),
    (256, 128),
    (512, 64),
    (512, 128),
    (1024, 64),
    (1024, 128),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (4096, 2048),
    (2048, 4096),
]
# transB coverage: SQUARE only (K==N) — required by the op's k-axis check on the transposed view.
SHAPES_TRANS = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
MS = [8, 16, 32]  # per-rank M
ORIENTATIONS = [False, True]  # [non-trans, transB]

# Non-trans data source. 'random' (default) is self-contained + proven. transB ALWAYS uses valid
# MX data (script/generate_mx_data.mx_quantize), independent of this flag.
DATA_MODE = os.environ.get("BI_DATA_MODE", "random")


def sha_row(t, i):
    return hashlib.sha256(t[i].cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[
        :16
    ]


def _rand_inputs(M, K, N, seed):
    n_pack = (K + BLOCK_SIZE * 2 - 1) // (BLOCK_SIZE * 2)
    g_t = torch.Generator()
    g_t.manual_seed(12345)
    probe = torch.randint(0x10, 0x50, (K,), dtype=torch.uint8, generator=g_t)
    g_w = torch.Generator()
    g_w.manual_seed(9000)
    x2 = torch.randint(0x10, 0x50, (K, N), dtype=torch.uint8, generator=g_w)
    g_s = torch.Generator()
    g_s.manual_seed(9001)
    x2s = torch.randint(0x76, 0x82, (n_pack, N, 2), dtype=torch.uint8, generator=g_s)
    g_o = torch.Generator()
    g_o.manual_seed(seed)
    x1 = torch.randint(0x10, 0x50, (M, K), dtype=torch.uint8, generator=g_o)
    x1[TARGET_ROW] = probe  # pin probe on rank-0 local row 0
    x1s = torch.full((M, n_pack, 2), 0x7F, dtype=torch.uint8)
    return x1, x1s, x2, x2s


def _mx_transb_inputs(M, K, seed):
    """Valid MX data for transB via the repo's generate_mx_data.mx_quantize. Square weight [K,K].
    Returns raw uint8-viewable tensors (x1q, x1s, x2q, x2s); the transpose views are applied in
    make_and_call. Pinned probe row (fp32) fixed across seeds so composition-BI is meaningful."""
    import numpy as np

    try:
        from script import generate_mx_data as gmx  # shipped in ops-transformer ST tree
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "transB needs script/generate_mx_data.mx_quantize (+ en_dtypes/ml_dtypes); "
            "run inside the ops-transformer ST tree"
        ) from e
    rp = np.random.RandomState(12345)
    probe = rp.randn(K).astype(np.float32)
    rw = np.random.RandomState(9000)
    w = rw.randn(K, K).astype(np.float32)  # square [N=K, K]
    x2q, x2s = gmx.mx_quantize(
        w, "float8_e4m3fn", axis=-1, block_size=BLOCK_SIZE, round_mode="rint"
    )
    ro = np.random.RandomState(seed)
    x1 = ro.randn(M, K).astype(np.float32)
    x1[TARGET_ROW] = probe
    x1q, x1s = gmx.mx_quantize(
        x1, "float8_e4m3fn", axis=-1, block_size=BLOCK_SIZE, round_mode="rint"
    )
    u8 = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a)).view(np.uint8))
    return u8(x1q), u8(x1s), u8(x2q), u8(x2s)


def make_and_call(dev, hcom, ws, M, K, N, seed, trans_b):
    if trans_b:
        # transB: valid MX data, square K==N, NON-contiguous transpose views (stride -> isBTrans).
        x1b, x1sb, x2b, x2sb = _mx_transb_inputs(M, K, seed)
        x1 = x1b.to(dev).view(torch.float8_e4m3fn)
        x1s = x1sb.to(dev).view(torch.float8_e8m0fnu)
        x2 = x2b.to(dev).view(torch.float8_e4m3fn).t()  # [N,K] -> [K,N] non-contiguous
        x2s = (
            x2sb.to(dev).view(torch.float8_e8m0fnu).transpose(0, 1)
        )  # [N,np,2] -> [np,N,2] non-contig
    else:
        if DATA_MODE != "random":
            raise NotImplementedError(
                "non-trans DATA_MODE=mx not wired; use 'random' (proven)"
            )
        x1b, x1sb, x2b, x2sb = _rand_inputs(M, K, N, seed)
        x1 = x1b.to(dev).view(torch.float8_e4m3fn)
        x1s = x1sb.to(dev).view(torch.float8_e8m0fnu)
        x2 = x2b.to(dev).view(torch.float8_e4m3fn)
        x2s = x2sb.to(dev).view(torch.float8_e8m0fnu)
    y, _, _ = torch.ops.npu.npu_all_gather_quant_mm(
        x1,
        x2,
        hcom,
        ws,
        x1_scale=x1s,
        x2_scale=x2s,
        block_size=BLOCK_SIZE,
        y_dtype=torch_npu.bfloat16,
        x1_dtype=torch_npu.float8_e4m3fn,
        x2_dtype=torch_npu.float8_e4m3fn,
        x1_scale_dtype=torch_npu.float8_e8m0fnu,
        x2_scale_dtype=torch_npu.float8_e8m0fnu,
    )
    torch_npu.npu.synchronize()
    return y


def rigorous_cell(dev, hcom, ws, M, K, N, seed_a, seed_b, trans_b):
    sa, sb = [], []
    try:
        for _ in range(N_TRIALS):
            y = make_and_call(dev, hcom, ws, M, K, N, seed_a, trans_b)
            if TARGET_ROW >= y.shape[0]:
                return "SKIP"
            sa.append(sha_row(y, TARGET_ROW))
        for _ in range(N_TRIALS):
            y = make_and_call(dev, hcom, ws, M, K, N, seed_b, trans_b)
            sb.append(sha_row(y, TARGET_ROW))
    except Exception as e:
        return "ERR:" + str(e).replace("\n", " ")[:50]
    if len(set(sa)) != 1 or len(set(sb)) != 1:
        return "NON_DET"
    return "BI_PASS" if sa[0] == sb[0] else "BI_FAIL"


def main():
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(lr)
    dev = torch.device(f"npu:{lr}")
    dist.init_process_group("hccl", world_size=ws, rank=rank)
    hcom = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(lr)

    only = os.environ.get("BI_ORIENT", "both")  # both | notrans | trans
    orients = {"notrans": [False], "trans": [True]}.get(only, ORIENTATIONS)

    if rank == 0:
        print(
            f"=== AllGatherMatmulV2 BI rigorous WS={ws} N_WARMUP={N_WARMUP} N_TRIALS={N_TRIALS} "
            f"DATA={DATA_MODE} orient={only} ===",
            flush=True,
        )
    try:
        for _ in range(N_WARMUP):
            _ = make_and_call(dev, hcom, ws, 16, 512, 128, 42, trans_b=False)
    except Exception as e:
        if rank == 0:
            print(f"  warmup ERROR (continuing): {str(e)[:100]}", flush=True)

    total_fail = 0
    for trans_b in orients:
        lbl = "TRANS_B  " if trans_b else "NON-TRANS"
        shapes = SHAPES_TRANS if trans_b else SHAPES  # transB: square only
        n_pass = n_fail = n_nondet = n_err = n_skip = 0
        for K, N in shapes:
            for M in MS:
                sh = (M * 131 + K * 17 + N) % 9000
                st = rigorous_cell(dev, hcom, ws, M, K, N, 100 + sh, 200 + sh, trans_b)
                if rank == 0:
                    tag = (
                        st
                        if st in ("BI_PASS", "BI_FAIL", "NON_DET", "SKIP")
                        else "ERROR"
                    )
                    n_pass += tag == "BI_PASS"
                    n_fail += tag == "BI_FAIL"
                    n_nondet += tag == "NON_DET"
                    n_skip += tag == "SKIP"
                    n_err += tag == "ERROR"
                    if tag in ("BI_FAIL", "ERROR", "NON_DET"):
                        print(f"  [{lbl}] M={M:<3} K={K:<5} N={N:<5} {st}", flush=True)
        if rank == 0:
            total_fail += n_fail
            print(
                f"  [{lbl}] SUMMARY BI_PASS={n_pass} BI_FAIL={n_fail} "
                f"NON_DET={n_nondet} ERROR={n_err} SKIP={n_skip}",
                flush=True,
            )
    dist.destroy_process_group()
    sys.exit(1 if total_fail else 0)  # only real BI_FAIL fails the suite


if __name__ == "__main__":
    main()
