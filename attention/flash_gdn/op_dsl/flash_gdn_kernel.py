# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Scalar-gate GDN forward: one class kernel with synchronized WU/state stages.

All mathematical preprocessing and recurrence runs in DSL kernels. Torch only
packs layouts, allocates buffers and constructs host scheduling records.
The current vector implementation is a functional chunk baseline, not a claim
of performance parity with KDA's tuned Cube/Vector implementation.
"""

import threading
import torch
import cannbotdsl
from cannbotdsl import (
    Buffer,
    Tensor,
    MemLoc,
    ProfileSpec,
    dtypes,
    host,
    jit,
    make_layout,
    mem_copy,
    onchip_rewind,
    range_constexpr,
)
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.control_flow import range as drange
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.sync import global_sync_all
from cannbotdsl.ops.reg import (
    update_mask,
    vload,
    vload_broadcast,
    vstore,
    vstore_first,
    vexp,
    vmul,
    vmuls,
    vmadd,
    vsub,
    vreduce_sum,
    vadd,
    vadds,
    vdiv,
    vsqrt,
    vabs,
    vmaxs,
    vdups,
    vdup,
)
from .flash_gdn_metadata import CHUNK_SIZE, HEADER_SIZE, HEAD_DIM

# Partition the Dv rows of each [Dv, Dk] state, not its tokens or heads.
FP32_VECTOR_LANES = 64
STATE_ROWS_PER_WORK = 32
STATE_WORK_ITEMS = HEAD_DIM // STATE_ROWS_PER_WORK
# Each input record contains g, beta, A_log, dt_bias, then reserved zeros.
TOKEN_PARAM_FIELDS = 4
TOKEN_PARAM_STRIDE = 8
# Cache slots: g_cumsum, effective beta, A_log, dt_bias, effective g,
# exp(sum of later effective g), and two reserved slots. Initially copy params.
TOKEN_CACHE_STRIDE = 8
# Scheduling records keep three fields followed by five reserved int32 words.
TASK_RECORD_STRIDE = 8  # token start, valid token count, value head
CHAIN_RECORD_STRIDE = 8  # first task, number of chunks, value head
MAX_LAUNCH_BLOCKS = 8
# FP32 vector registers have FP32_VECTOR_LANES lanes (256 bytes).
_LOCK = threading.Lock()
_COMPILED = None


def _softplus_stable(value, mask):
    # log1p(t) = 2*atanh(t/(2+t)), t=exp(-abs(x)) in [0,1].
    # Seven odd terms have truncation error < 1.05e-8 on that interval.
    # Unlike log(1+t), the formula never adds a tiny tail to 1 first.
    tail = vexp(vmuls(vabs(value, mask=mask), -1.0, mask=mask), mask=mask)
    ratio = vdiv(tail, vadds(tail, 2.0, mask=mask), mask=mask)
    square = vmul(ratio, ratio, mask=mask)
    polynomial = vdups(1.0 / 13.0, dtypes.float32)
    polynomial = vadds(vmul(polynomial, square, mask=mask), 1.0 / 11.0, mask=mask)
    polynomial = vadds(vmul(polynomial, square, mask=mask), 1.0 / 9.0, mask=mask)
    polynomial = vadds(vmul(polynomial, square, mask=mask), 1.0 / 7.0, mask=mask)
    polynomial = vadds(vmul(polynomial, square, mask=mask), 1.0 / 5.0, mask=mask)
    polynomial = vadds(vmul(polynomial, square, mask=mask), 1.0 / 3.0, mask=mask)
    polynomial = vadds(vmul(polynomial, square, mask=mask), 1.0, mask=mask)
    return vadd(
        vmaxs(value, 0.0, mask=mask),
        vmuls(vmul(ratio, polynomial, mask=mask), 2.0, mask=mask),
        mask=mask,
    )


@kernel(profile=ProfileSpec(name="flash_gdn"))
class FlashGdnKernel:
    """One profiled launch: chunk preparation, grid barrier, state/output."""

    def __init__(self):
        pass

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        params: Tensor,
        tasks: Tensor,
        chains: Tensor,
        w: Tensor,
        u: Tensor,
        qs: Tensor,
        ks: Tensor,
        a: Tensor,
        cs: Tensor,
        h0: Tensor,
        out: Tensor,
        ht: Tensor,
        states: Tensor,
        ntasks: dtypes.int64,
        nchains: dtypes.int64,
        hq: dtypes.int64,
        hv: dtypes.int64,
        norm: dtypes.int64,
        gate: dtypes.int64,
        sigmoid: dtypes.int64,
        negative: dtypes.int64,
        scale: dtypes.float32,
        save_states: dtypes.int64,
        save_final_state: dtypes.int64,
    ):
        self._chunks(
            q,
            k,
            v,
            params,
            tasks,
            w,
            u,
            qs,
            ks,
            a,
            cs,
            ntasks,
            hq,
            hv,
            norm,
            gate,
            sigmoid,
            negative,
            scale,
        )
        # Every chunk workspace must be globally visible before state reads.
        global_sync_all()
        # The first stage has completed; reclaim its dead on-chip buffers.
        onchip_rewind()
        self._state(
            w,
            u,
            qs,
            ks,
            a,
            cs,
            tasks,
            chains,
            h0,
            out,
            ht,
            states,
            nchains,
            hv,
            scale,
            save_states,
            save_final_state,
        )

    @jit
    def _chunks(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        params: Tensor,
        tasks: Tensor,
        wgm: Tensor,
        ugm: Tensor,
        qgm: Tensor,
        kgm: Tensor,
        agm: Tensor,
        cgm: Tensor,
        ntasks: dtypes.int64,
        hq: dtypes.int64,
        hv: dtypes.int64,
        norm: dtypes.int64,
        gate: dtypes.int64,
        sigmoid: dtypes.int64,
        negative: dtypes.int64,
        scale: dtypes.float32,
    ):
        qb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        kb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        vb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        wb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        ab = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * CHUNK_SIZE,)))
        cb = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * TOKEN_CACHE_STRIDE,))
        )
        # Scalar reduction scratch: eight slots, with the first two used for q/k.
        tmp = Buffer(MemLoc.UB, dtypes.float32, make_layout((8,)))
        for task in drange(get_block_idx(), ntasks, get_block_num()):
            start = tasks[task * TASK_RECORD_STRIDE]
            count = tasks[task * TASK_RECORD_STRIDE + 1]
            head = tasks[task * TASK_RECORD_STRIDE + 2]
            qhead = head // (hv // hq)
            with vf(mode="simd"):
                mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                zero = vdups(0.0, dtypes.float32)
                for tile in drange(CHUNK_SIZE * HEAD_DIM // FP32_VECTOR_LANES):
                    vstore(qb, tile * FP32_VECTOR_LANES, zero, mask)
                    vstore(kb, tile * FP32_VECTOR_LANES, zero, mask)
                    vstore(vb, tile * FP32_VECTOR_LANES, zero, mask)
                    vstore(wb, tile * FP32_VECTOR_LANES, zero, mask)
                    vstore(ub, tile * FP32_VECTOR_LANES, zero, mask)
                for tile in drange(CHUNK_SIZE * CHUNK_SIZE // FP32_VECTOR_LANES):
                    vstore(ab, tile * FP32_VECTOR_LANES, zero, mask)
                for tile in drange(
                    CHUNK_SIZE * TOKEN_CACHE_STRIDE // FP32_VECTOR_LANES
                ):
                    vstore(cb, tile * FP32_VECTOR_LANES, zero, mask)
            for i in drange(count):
                qo = ((start + i) * hq + qhead) * HEAD_DIM
                vo = ((start + i) * hv + head) * HEAD_DIM
                po = ((start + i) * hv + head) * TOKEN_PARAM_STRIDE
                mem_copy(
                    qb[slice(i * HEAD_DIM, (i + 1) * HEAD_DIM)],
                    q[slice(qo, qo + HEAD_DIM)],
                )
                mem_copy(
                    kb[slice(i * HEAD_DIM, (i + 1) * HEAD_DIM)],
                    k[slice(qo, qo + HEAD_DIM)],
                )
                mem_copy(
                    vb[slice(i * HEAD_DIM, (i + 1) * HEAD_DIM)],
                    v[slice(vo, vo + HEAD_DIM)],
                )
                mem_copy(
                    cb[slice(i * TOKEN_CACHE_STRIDE, (i + 1) * TOKEN_CACHE_STRIDE)],
                    params[slice(po, po + TOKEN_PARAM_STRIDE)],
                )
                if norm != 0:
                    # Broadcast reductions in registers; do not round-trip UB
                    # scratch within the fused normalization VF.
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        q0, q1 = (
                            vload(qb, i * HEAD_DIM),
                            vload(qb, i * HEAD_DIM + FP32_VECTOR_LANES),
                        )
                        k0, k1 = (
                            vload(kb, i * HEAD_DIM),
                            vload(kb, i * HEAD_DIM + FP32_VECTOR_LANES),
                        )
                        qs = vreduce_sum(
                            vmadd(q1, q1, vmul(q0, q0, mask=mask), mask=mask), mask=mask
                        )
                        ks = vreduce_sum(
                            vmadd(k1, k1, vmul(k0, k0, mask=mask), mask=mask), mask=mask
                        )
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        iq = vdiv(
                            vdups(1.0, dtypes.float32),
                            vsqrt(
                                vadds(vdup(qs, mask=mask), 1e-6, mask=mask),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                        ik = vdiv(
                            vdups(1.0, dtypes.float32),
                            vsqrt(
                                vadds(vdup(ks, mask=mask), 1e-6, mask=mask),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                        for part in range_constexpr(HEAD_DIM // FP32_VECTOR_LANES):
                            vstore(
                                qb,
                                i * HEAD_DIM + part * FP32_VECTOR_LANES,
                                vmul(
                                    vload(qb, i * HEAD_DIM + part * FP32_VECTOR_LANES),
                                    iq,
                                    mask=mask,
                                ),
                                mask,
                            )
                            vstore(
                                kb,
                                i * HEAD_DIM + part * FP32_VECTOR_LANES,
                                vmul(
                                    vload(kb, i * HEAD_DIM + part * FP32_VECTOR_LANES),
                                    ik,
                                    mask=mask,
                                ),
                                mask,
                            )
                if gate != 0:
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        gi = vload_broadcast(cb, i * TOKEN_CACHE_STRIDE)
                        x = vadd(
                            gi,
                            vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 3),
                            mask=mask,
                        )
                        soft = _softplus_stable(x, mask)
                        gi = vmuls(
                            vmul(
                                vexp(
                                    vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 2),
                                    mask=mask,
                                ),
                                soft,
                                mask=mask,
                            ),
                            -1.0,
                            mask=mask,
                        )
                        vstore_first(cb, i * TOKEN_CACHE_STRIDE, gi)
                if sigmoid != 0:
                    beta_factor = dtypes.float32(1.0)
                    if negative != 0:
                        beta_factor = dtypes.float32(2.0)
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        bi = vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 1)
                        # sigmoid(x) via exp(-softplus(-x)), avoiding overflow.
                        soft = _softplus_stable(vmuls(bi, -1.0, mask=mask), mask)
                        bi = vmuls(
                            vexp(vmuls(soft, -1.0, mask=mask), mask=mask),
                            beta_factor,
                            mask=mask,
                        )
                        vstore_first(cb, i * TOKEN_CACHE_STRIDE + 1, bi)
                # Preserve each effective g before accumulating its prefix. All scalar
                # feature/prefix decisions stay outside VF register-operation regions.
                if i > 0:
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        gi = vload_broadcast(cb, i * TOKEN_CACHE_STRIDE)
                        vstore_first(cb, i * TOKEN_CACHE_STRIDE + 4, gi)
                        gi = vadd(
                            gi,
                            vload_broadcast(cb, (i - 1) * TOKEN_CACHE_STRIDE),
                            mask=mask,
                        )
                        vstore_first(cb, i * TOKEN_CACHE_STRIDE, gi)
                else:
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        gi = vload_broadcast(cb, i * TOKEN_CACHE_STRIDE)
                        vstore_first(cb, i * TOKEN_CACHE_STRIDE + 4, gi)
            with vf(mode="simd"):
                mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                suffix = vdups(0.0, dtypes.float32)
                for reverse in drange(count):
                    i = count - 1 - reverse
                    # exp(sum(g[i+1:count])) for the state update. Never recover
                    # this small interval by subtracting two large FP32 prefixes.
                    vstore_first(
                        cb, i * TOKEN_CACHE_STRIDE + 5, vexp(suffix, mask=mask)
                    )
                    suffix = vadd(
                        suffix,
                        vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 4),
                        mask=mask,
                    )
            # Forward substitution for (I+L)w=beta*exp(gcum)*k and
            # (I+L)u=beta*v, with beta multiplying ROW i of L.
            for i in drange(count):
                with vf(mode="simd"):
                    mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                    interval = vdups(0.0, dtypes.float32)
                    for reverse in drange(i + 1):
                        j = i - reverse
                        # Temporarily use the attention row as an interval-decay
                        # table; the ascending substitution below consumes it.
                        vstore_first(ab, i * CHUNK_SIZE + j, vexp(interval, mask=mask))
                        interval = vadd(
                            interval,
                            vload_broadcast(cb, j * TOKEN_CACHE_STRIDE + 4),
                            mask=mask,
                        )
                    mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                    bi = vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 1)
                    eb = vmul(
                        bi,
                        vexp(vload_broadcast(cb, i * TOKEN_CACHE_STRIDE), mask=mask),
                        mask=mask,
                    )
                    for part in range_constexpr(HEAD_DIM // FP32_VECTOR_LANES):
                        vstore(
                            wb,
                            i * HEAD_DIM + part * FP32_VECTOR_LANES,
                            vmul(
                                vload(kb, i * HEAD_DIM + part * FP32_VECTOR_LANES),
                                eb,
                                mask=mask,
                            ),
                            mask,
                        )
                        vstore(
                            ub,
                            i * HEAD_DIM + part * FP32_VECTOR_LANES,
                            vmul(
                                vload(vb, i * HEAD_DIM + part * FP32_VECTOR_LANES),
                                bi,
                                mask=mask,
                            ),
                            mask,
                        )
                for j in drange(i + 1):
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        k0, k1 = (
                            vload(kb, j * HEAD_DIM),
                            vload(kb, j * HEAD_DIM + FP32_VECTOR_LANES),
                        )
                        qdot = vreduce_sum(
                            vmadd(
                                vload(qb, i * HEAD_DIM + FP32_VECTOR_LANES),
                                k1,
                                vmul(vload(qb, i * HEAD_DIM), k0, mask=mask),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                        kdot = vreduce_sum(
                            vmadd(
                                vload(kb, i * HEAD_DIM + FP32_VECTOR_LANES),
                                k1,
                                vmul(vload(kb, i * HEAD_DIM), k0, mask=mask),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                        decay = vload_broadcast(ab, i * CHUNK_SIZE + j)
                        vstore_first(
                            ab,
                            i * CHUNK_SIZE + j,
                            vmuls(vmul(qdot, decay, mask=mask), scale, mask=mask),
                        )
                        vstore_first(
                            tmp,
                            0,
                            vmul(
                                vmul(kdot, decay, mask=mask),
                                vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 1),
                                mask=mask,
                            ),
                        )
                    if j < i:
                        with vf(mode="simd"):
                            mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                            coeff = vload_broadcast(tmp, 0)
                            for part in range_constexpr(HEAD_DIM // FP32_VECTOR_LANES):
                                vstore(
                                    wb,
                                    i * HEAD_DIM + part * FP32_VECTOR_LANES,
                                    vsub(
                                        vload(
                                            wb, i * HEAD_DIM + part * FP32_VECTOR_LANES
                                        ),
                                        vmul(
                                            coeff,
                                            vload(
                                                wb,
                                                j * HEAD_DIM + part * FP32_VECTOR_LANES,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    ),
                                    mask,
                                )
                                vstore(
                                    ub,
                                    i * HEAD_DIM + part * FP32_VECTOR_LANES,
                                    vsub(
                                        vload(
                                            ub, i * HEAD_DIM + part * FP32_VECTOR_LANES
                                        ),
                                        vmul(
                                            coeff,
                                            vload(
                                                ub,
                                                j * HEAD_DIM + part * FP32_VECTOR_LANES,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    ),
                                    mask,
                                )
            mem_copy(
                wgm[
                    slice(
                        task * CHUNK_SIZE * HEAD_DIM, (task + 1) * CHUNK_SIZE * HEAD_DIM
                    )
                ],
                wb,
            )
            mem_copy(
                ugm[
                    slice(
                        task * CHUNK_SIZE * HEAD_DIM, (task + 1) * CHUNK_SIZE * HEAD_DIM
                    )
                ],
                ub,
            )
            mem_copy(
                qgm[
                    slice(
                        task * CHUNK_SIZE * HEAD_DIM, (task + 1) * CHUNK_SIZE * HEAD_DIM
                    )
                ],
                qb,
            )
            mem_copy(
                kgm[
                    slice(
                        task * CHUNK_SIZE * HEAD_DIM, (task + 1) * CHUNK_SIZE * HEAD_DIM
                    )
                ],
                kb,
            )
            mem_copy(
                agm[
                    slice(
                        task * CHUNK_SIZE * CHUNK_SIZE,
                        (task + 1) * CHUNK_SIZE * CHUNK_SIZE,
                    )
                ],
                ab,
            )
            mem_copy(
                cgm[
                    slice(
                        task * CHUNK_SIZE * TOKEN_CACHE_STRIDE,
                        (task + 1) * CHUNK_SIZE * TOKEN_CACHE_STRIDE,
                    )
                ],
                cb,
            )

    @jit
    def _state(
        self,
        wgm: Tensor,
        ugm: Tensor,
        qgm: Tensor,
        kgm: Tensor,
        agm: Tensor,
        cgm: Tensor,
        tasks: Tensor,
        chains: Tensor,
        h0: Tensor,
        output: Tensor,
        ht: Tensor,
        snapshots: Tensor,
        nchains: dtypes.int64,
        hv: dtypes.int64,
        scale: dtypes.float32,
        save_states: dtypes.int64,
        save_final_state: dtypes.int64,
    ):
        sb = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((STATE_ROWS_PER_WORK * HEAD_DIM,))
        )
        wb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        ub = Buffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((CHUNK_SIZE * HEAD_DIM + FP32_VECTOR_LANES,)),
        )
        qb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        kb = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * HEAD_DIM,)))
        ab = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * CHUNK_SIZE,)))
        cb = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE * TOKEN_CACHE_STRIDE,))
        )
        nb = Buffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((CHUNK_SIZE * STATE_ROWS_PER_WORK + FP32_VECTOR_LANES,)),
        )
        ob = Buffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((CHUNK_SIZE * STATE_ROWS_PER_WORK + FP32_VECTOR_LANES,)),
        )
        tmp = Buffer(MemLoc.UB, dtypes.float32, make_layout((FP32_VECTOR_LANES,)))
        for work in drange(
            get_block_idx(), nchains * STATE_WORK_ITEMS, get_block_num()
        ):
            chain = work // STATE_WORK_ITEMS
            rowbase = (work % STATE_WORK_ITEMS) * STATE_ROWS_PER_WORK
            first = chains[chain * CHAIN_RECORD_STRIDE]
            chunks = chains[chain * CHAIN_RECORD_STRIDE + 1]
            head = chains[chain * CHAIN_RECORD_STRIDE + 2]
            so = chain * HEAD_DIM * HEAD_DIM + rowbase * HEAD_DIM
            mem_copy(sb, h0[slice(so, so + STATE_ROWS_PER_WORK * HEAD_DIM)])
            with vf(mode="simd"):
                mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                zero = vdups(0.0, dtypes.float32)
                vstore(ub, CHUNK_SIZE * HEAD_DIM, zero, mask)
                vstore(nb, CHUNK_SIZE * STATE_ROWS_PER_WORK, zero, mask)
                vstore(ob, CHUNK_SIZE * STATE_ROWS_PER_WORK, zero, mask)
            for c in drange(chunks):
                task = first + c
                count = tasks[task * TASK_RECORD_STRIDE + 1]
                start = tasks[task * TASK_RECORD_STRIDE]
                if save_states != 0:
                    dest = task * HEAD_DIM * HEAD_DIM + rowbase * HEAD_DIM
                    mem_copy(
                        snapshots[slice(dest, dest + STATE_ROWS_PER_WORK * HEAD_DIM)],
                        sb,
                    )
                mem_copy(
                    wb,
                    wgm[
                        slice(
                            task * CHUNK_SIZE * HEAD_DIM,
                            (task + 1) * CHUNK_SIZE * HEAD_DIM,
                        )
                    ],
                )
                mem_copy(
                    ub[slice(0, CHUNK_SIZE * HEAD_DIM)],
                    ugm[
                        slice(
                            task * CHUNK_SIZE * HEAD_DIM,
                            (task + 1) * CHUNK_SIZE * HEAD_DIM,
                        )
                    ],
                )
                mem_copy(
                    qb,
                    qgm[
                        slice(
                            task * CHUNK_SIZE * HEAD_DIM,
                            (task + 1) * CHUNK_SIZE * HEAD_DIM,
                        )
                    ],
                )
                mem_copy(
                    kb,
                    kgm[
                        slice(
                            task * CHUNK_SIZE * HEAD_DIM,
                            (task + 1) * CHUNK_SIZE * HEAD_DIM,
                        )
                    ],
                )
                mem_copy(
                    ab,
                    agm[
                        slice(
                            task * CHUNK_SIZE * CHUNK_SIZE,
                            (task + 1) * CHUNK_SIZE * CHUNK_SIZE,
                        )
                    ],
                )
                mem_copy(
                    cb,
                    cgm[
                        slice(
                            task * CHUNK_SIZE * TOKEN_CACHE_STRIDE,
                            (task + 1) * CHUNK_SIZE * TOKEN_CACHE_STRIDE,
                        )
                    ],
                )
                for i in drange(count):
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        for row in drange(STATE_ROWS_PER_WORK):
                            s0, s1 = (
                                vload(sb, row * HEAD_DIM),
                                vload(sb, row * HEAD_DIM + FP32_VECTOR_LANES),
                            )
                            pred = vreduce_sum(
                                vmadd(
                                    s1,
                                    vload(wb, i * HEAD_DIM + FP32_VECTOR_LANES),
                                    vmul(s0, vload(wb, i * HEAD_DIM), mask=mask),
                                    mask=mask,
                                ),
                                mask=mask,
                            )
                            inter = vreduce_sum(
                                vmadd(
                                    s1,
                                    vload(qb, i * HEAD_DIM + FP32_VECTOR_LANES),
                                    vmul(s0, vload(qb, i * HEAD_DIM), mask=mask),
                                    mask=mask,
                                ),
                                mask=mask,
                            )
                            vstore_first(tmp, row, pred)
                            vstore_first(
                                ob,
                                i * STATE_ROWS_PER_WORK + row,
                                vmuls(
                                    vmul(
                                        inter,
                                        vexp(
                                            vload_broadcast(cb, i * TOKEN_CACHE_STRIDE),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    ),
                                    scale,
                                    mask=mask,
                                ),
                            )
                    # Row reductions land in UB scratch; retain the producer/consumer VF boundary.
                    with vf(mode="simd"):
                        mask, _ = update_mask(STATE_ROWS_PER_WORK, elem_bits=32)
                        vstore(
                            nb,
                            i * STATE_ROWS_PER_WORK,
                            vsub(
                                vload(ub, i * HEAD_DIM + rowbase),
                                vload(tmp, 0),
                                mask=mask,
                            ),
                            mask,
                        )
                # Fused Stage 3: scale*qg@h.T + causal Aqk@v_new.
                for i in drange(count):
                    with vf(mode="simd"):
                        mask, _ = update_mask(STATE_ROWS_PER_WORK, elem_bits=32)
                        acc = vload(ob, i * STATE_ROWS_PER_WORK)
                        for j in drange(i + 1):
                            acc = vmadd(
                                vload(nb, j * STATE_ROWS_PER_WORK),
                                vload_broadcast(ab, i * CHUNK_SIZE + j),
                                acc,
                                mask=mask,
                            )
                        vstore(ob, i * STATE_ROWS_PER_WORK, acc, mask)
                    dest = ((start + i) * hv + head) * HEAD_DIM + rowbase
                    mem_copy(
                        output[slice(dest, dest + STATE_ROWS_PER_WORK)],
                        ob[
                            slice(
                                i * STATE_ROWS_PER_WORK, (i + 1) * STATE_ROWS_PER_WORK
                            )
                        ],
                    )
                if c + 1 < chunks or save_final_state != 0:
                    with vf(mode="simd"):
                        mask, _ = update_mask(FP32_VECTOR_LANES, elem_bits=32)
                        last = vload_broadcast(cb, (count - 1) * TOKEN_CACHE_STRIDE)
                        decay = vexp(last, mask=mask)
                        for row in drange(STATE_ROWS_PER_WORK):
                            s0 = vmul(vload(sb, row * HEAD_DIM), decay, mask=mask)
                            s1 = vmul(
                                vload(sb, row * HEAD_DIM + FP32_VECTOR_LANES),
                                decay,
                                mask=mask,
                            )
                            for i in drange(count):
                                correction = vmul(
                                    vload_broadcast(nb, i * STATE_ROWS_PER_WORK + row),
                                    vload_broadcast(cb, i * TOKEN_CACHE_STRIDE + 5),
                                    mask=mask,
                                )
                                s0 = vmadd(
                                    vload(kb, i * HEAD_DIM), correction, s0, mask=mask
                                )
                                s1 = vmadd(
                                    vload(kb, i * HEAD_DIM + FP32_VECTOR_LANES),
                                    correction,
                                    s1,
                                    mask=mask,
                                )
                            vstore(sb, row * HEAD_DIM, s0, mask)
                            vstore(sb, row * HEAD_DIM + FP32_VECTOR_LANES, s1, mask)
            if save_final_state != 0:
                mem_copy(ht[slice(so, so + STATE_ROWS_PER_WORK * HEAD_DIM)], sb)


@host
def _entry(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    params: Tensor,
    tasks: Tensor,
    chains: Tensor,
    w: Tensor,
    u: Tensor,
    qs: Tensor,
    ks: Tensor,
    a: Tensor,
    cs: Tensor,
    h0: Tensor,
    out: Tensor,
    ht: Tensor,
    states: Tensor,
    ntasks: dtypes.int64,
    nchains: dtypes.int64,
    hq: dtypes.int64,
    hv: dtypes.int64,
    norm: dtypes.int64,
    gate: dtypes.int64,
    sigmoid: dtypes.int64,
    negative: dtypes.int64,
    scale: dtypes.float32,
    save_states: dtypes.int64,
    save_final_state: dtypes.int64,
    cores: dtypes.int64,
):
    op = FlashGdnKernel()
    op[cores](
        q,
        k,
        v,
        params,
        tasks,
        chains,
        w,
        u,
        qs,
        ks,
        a,
        cs,
        h0,
        out,
        ht,
        states,
        ntasks,
        nchains,
        hq,
        hv,
        norm,
        gate,
        sigmoid,
        negative,
        scale,
        save_states,
        save_final_state,
    )


def compile_kernel():
    specs = [
        cannbotdsl.TensorSpec(
            (cannbotdsl.Dim(f"ARG_{i}"),),
            dtypes.int32 if i in (4, 5) else dtypes.float32,
        )
        for i in range(16)
    ]
    return cannbotdsl.compile(
        _entry,
        *specs,
        # ntasks, nchains, hq, hv, norm, gate, sigmoid, negative.
        *([dtypes.int64] * 8),
        dtypes.float32,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


def _compiled():
    global _COMPILED
    with _LOCK:
        if _COMPILED is None:
            _COMPILED = compile_kernel()
    return _COMPILED


def run_device(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    scale,
    layout,
    header,
    A_log,
    dt_bias,
    norm,
    gate,
    sigmoid,
    negative,
    save_states,
    save_final_state,
):
    batch, hq, hv, length = header[2:6]
    # The fixed metadata prefix has HEADER_SIZE int32 words, then batch lengths.
    lengths = header[HEADER_SIZE : HEADER_SIZE + batch]

    def pack(t):
        if layout == "BNSD":
            t = t.transpose(1, 2)
        dims = 2 if t.ndim == q.ndim else 1
        return t.reshape(-1, *t.shape[-dims:]).float().contiguous()

    q32, k32, v32 = [pack(t) for t in (q, k, v)]
    gp, bp = [pack(t) for t in (g, beta)]
    al = (
        torch.zeros(hv, device=q.device, dtype=torch.float32)
        if A_log is None
        else A_log
    )
    bias = (
        torch.zeros(hv, device=q.device, dtype=torch.float32)
        if dt_bias is None
        else dt_bias
    )
    params = torch.nn.functional.pad(
        torch.stack((gp, bp, al.expand_as(gp), bias.expand_as(gp)), dim=-1),
        (0, TOKEN_PARAM_STRIDE - TOKEN_PARAM_FIELDS),
    ).contiguous()
    tasks, chains = [], []
    begin, prefix = 0, 0
    placements = []
    for b, used in enumerate(lengths):
        nc = (used + CHUNK_SIZE - 1) // CHUNK_SIZE
        for h in range(hv):
            chains.append([len(tasks), nc, h] + [0] * (CHAIN_RECORD_STRIDE - 3))
            for c in range(nc):
                tasks.append(
                    [begin + c * CHUNK_SIZE, min(CHUNK_SIZE, used - c * CHUNK_SIZE), h]
                    + [0] * (TASK_RECORD_STRIDE - 3)
                )
                placements.append((b, c, h, prefix + c))
        begin += used if layout == "TND" else length
        prefix += nc
    nt, nchain = len(tasks), len(chains)
    ti = torch.tensor(tasks, dtype=torch.int32, device=q.device)
    ch = torch.tensor(chains, dtype=torch.int32, device=q.device)

    def empty(*shape):
        return torch.empty(shape, device=q.device, dtype=torch.float32)

    # Four independent tile workspaces: w, u, normalized q, normalized k.
    w, u, qs, ks = [empty(nt, CHUNK_SIZE, HEAD_DIM) for _ in range(4)]
    aq, cs = (
        empty(nt, CHUNK_SIZE, CHUNK_SIZE),
        empty(nt, CHUNK_SIZE, TOKEN_CACHE_STRIDE),
    )
    h0 = (
        initial_state
        if initial_state is not None
        else torch.zeros(
            (batch, hv, HEAD_DIM, HEAD_DIM), device=q.device, dtype=torch.float32
        )
    )
    ht = empty(batch, hv, HEAD_DIM, HEAD_DIM) if save_final_state else empty(1)
    out = torch.zeros_like(v32)
    states = empty(nt, HEAD_DIM, HEAD_DIM) if save_states else empty(1)
    tensors = (q32, k32, v32, params, ti, ch, w, u, qs, ks, aq, cs, h0, out, ht, states)
    _compiled()(
        *[t.reshape(-1) for t in tensors],
        nt,
        nchain,
        hq,
        hv,
        int(norm),
        int(gate),
        int(sigmoid),
        int(negative),
        float(scale),
        int(save_states),
        int(save_final_state),
        min(MAX_LAUNCH_BLOCKS, nt),
    )
    if layout != "TND":
        out = out.reshape(batch, length, hv, HEAD_DIM)
        if layout == "BNSD":
            out = out.transpose(1, 2)
    public_states = None
    if save_states:
        capacity = (
            (length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch
            if layout == "TND"
            else (length + CHUNK_SIZE - 1) // CHUNK_SIZE
        )
        shape = (
            (capacity, hv, HEAD_DIM, HEAD_DIM)
            if layout == "TND"
            else (batch, capacity, hv, HEAD_DIM, HEAD_DIM)
        )
        public_states = torch.zeros(shape, device=q.device, dtype=torch.float32)
        for task, (b, c, h, packed) in enumerate(placements):
            if layout == "TND":
                public_states[packed, h].copy_(states[task])
            else:
                public_states[b, c, h].copy_(states[task])
    return (
        out.to(torch.bfloat16).contiguous(),
        ht if save_final_state else None,
        public_states,
    )
