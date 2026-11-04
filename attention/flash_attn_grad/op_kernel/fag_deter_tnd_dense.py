# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AscendC newDeter TND dense, non-swizzle coordinate schedule.

The linear assignment is core-major, ``ID = core * maxRound + round + 1``.
Batch areas use effective lengths; the returned index is a *synthetic* max-shape
coordinate, not an offset into packed GM. The caller decodes it and applies the
physical cu_seqlens offsets. No per-batch array or private workspace is needed.
"""

try:
    import pypto_pro.language as pl
except ImportError:  # CPU coordinate parity tests, without a device runtime.
    from types import SimpleNamespace

    class _Pl:
        DT_INT64 = 0
        min = staticmethod(min)
        max = staticmethod(max)
        range = staticmethod(range)

        @staticmethod
        def astype(value, _dtype):
            return int(value)

        @staticmethod
        def getval(tensor, index):
            return tensor[index]

        @staticmethod
        def struct(_name, **values):
            return SimpleNamespace(**values)

    pl = _Pl()

from fag_deter import _gcd


def tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk):
    """Return effective (Q, KV) token lengths, keeping cu offsets physical."""
    sq = pl.astype(pl.getval(cuq, batch + 1), pl.DT_INT64) - pl.astype(
        pl.getval(cuq, batch), pl.DT_INT64
    )
    sk = pl.astype(pl.getval(cuk, batch + 1), pl.DT_INT64) - pl.astype(
        pl.getval(cuk, batch), pl.DT_INT64
    )
    if ci.sequsedQOptional == 1:
        sq = pl.astype(pl.getval(usedq, batch), pl.DT_INT64)
    if ci.sequsedKvOptional == 1:
        sk = pl.astype(pl.getval(usedk, batch), pl.DT_INT64)
    return sq, sk


def tnd_permute_delta(delta1, m, n, g, phase_round):
    """AscendC CalTNDDenseIndex local permutation, shared by sparse phases.

    Input ``delta1`` and returned x/y are 1-based; returned head/g are 0-based.
    Folding g into m is essential: folding it into the batch/head axis changes
    both the gcd period and which cores share dK/dV. For g=1 this is precisely
    the IS_N_EQUAL specialization of the same AscendC function.
    """
    mm = pl.max(m, 1)
    mg = pl.max(m * g, 1)
    area = pl.max(m * n * g, 1)
    head = (delta1 - 1) // area
    delta = pl.astype((delta1 - 1) % area + 1, pl.DT_INT64)
    common = pl.max(_gcd(mg, phase_round), 1)
    t1 = pl.max(phase_round // common, 1)
    t2 = mg // common
    y = (delta - 1) // mg + 1
    # The last t1 columns deliberately retain their original coordinates.
    tail = (n - 1) % t1 + 1
    if t1 < n and y <= n - tail:
        adjustment = (y + t1 - 1) // t1
        period = t2 * phase_round
        shift = pl.astype(adjustment, pl.DT_INT64)
        # AscendC subtracts one period below. Narrow sparse virtual rectangles
        # can have adjustment > period: reduce only that out-of-domain shift,
        # otherwise one subtraction leaves the coordinate in the next group.
        if adjustment > period:
            shift = pl.astype(adjustment % period, pl.DT_INT64)
        delta = pl.astype(delta + shift, pl.DT_INT64)
        if delta > adjustment * period:
            delta = pl.astype(delta - period, pl.DT_INT64)
    xg = (delta - 1) % mg
    return head, xg // mm, xg % mm + 1, (delta - 1) // mg + 1


def make_tnd_deter_schedule(max_round, r0, r1, r2, core_divide, is_causal, r_whole):
    """Kernel-private plan and batch cursor; no tiling/workspace ABI fields.

    The cursor range is (begin, end] in the current phase's *unpermuted*
    one-based ID space. A phase change invalidates it, because causal/band
    phases do not have the same per-batch counts.
    """
    return pl.struct(
        "FagTndDeterSchedule",
        maxRound=pl.astype(max_round, pl.DT_INT64),
        r0=pl.astype(r0, pl.DT_INT64),
        r1=pl.astype(r1, pl.DT_INT64),
        r2=pl.astype(r2, pl.DT_INT64),
        coreDivide=pl.astype(core_divide, pl.DT_INT64),
        lineDeter=pl.astype(0, pl.DT_INT64),
        isCausal=pl.astype(is_causal, pl.DT_INT64),
        rWhole=pl.astype(r_whole, pl.DT_INT64),
        cursorPhase=pl.astype(-1, pl.DT_INT64),
        cursorBatch=pl.astype(0, pl.DT_INT64),
        cursorBegin=pl.astype(0, pl.DT_INT64),
        cursorEnd=pl.astype(0, pl.DT_INT64),
        cursorReady=pl.astype(0, pl.DT_INT64),
        cursorSq=pl.astype(0, pl.DT_INT64),
        cursorSk=pl.astype(0, pl.DT_INT64),
        cursorM=pl.astype(0, pl.DT_INT64),
        cursorN=pl.astype(0, pl.DT_INT64),
        cursorP=pl.astype(0, pl.DT_INT64),
        cursorQ=pl.astype(0, pl.DT_INT64),
        cursorMo=pl.astype(0, pl.DT_INT64),
        cursorNo=pl.astype(0, pl.DT_INT64),
        cursorVm=pl.astype(0, pl.DT_INT64),
        cursorVn=pl.astype(0, pl.DT_INT64),
        cursorArea=pl.astype(0, pl.DT_INT64),
    )


def reset_tnd_batch_cursor(plan, phase):
    """Reset once per phase, not once per physical task or empty round."""
    if plan.cursorPhase != phase:
        plan.cursorPhase = pl.astype(phase, pl.DT_INT64)
        plan.cursorBatch = pl.astype(0, pl.DT_INT64)
        plan.cursorBegin = pl.astype(0, pl.DT_INT64)
        plan.cursorEnd = pl.astype(0, pl.DT_INT64)
        plan.cursorReady = pl.astype(0, pl.DT_INT64)


def load_tnd_dense_batch(ci, plan, cuq, cuk, usedq, usedk):
    sq, sk = tnd_deter_lengths(ci, plan.cursorBatch, cuq, cuk, usedq, usedk)
    plan.cursorSq = pl.astype(sq, pl.DT_INT64)
    plan.cursorSk = pl.astype(sk, pl.DT_INT64)
    plan.cursorM = pl.astype((sq + 127) // 128, pl.DT_INT64)
    plan.cursorN = pl.astype((sk + 127) // 128, pl.DT_INT64)
    plan.cursorArea = pl.astype(plan.cursorM * plan.cursorN, pl.DT_INT64)


def seek_tnd_dense_batch(ci, ident, plan, cuq, cuk, usedq, usedk):
    """Locate an ID by moving the cached batch boundary in either direction.

    Monotone physical rounds cross each batch at most once. Reverse seeking
    also works for CPU random-access checks, without assuming synthetic tile
    indices follow the same order as the original IDs. Zero-area batches have
    empty ranges and are crossed without ever dividing by their dimensions.
    Call only for a valid physical round and bSize > 0.
    """
    reset_tnd_batch_cursor(plan, 0)
    copies = ci.n2Size * ci.gSize
    if plan.cursorReady == 0:
        load_tnd_dense_batch(ci, plan, cuq, cuk, usedq, usedk)
        plan.cursorEnd = pl.astype(plan.cursorArea * copies, pl.DT_INT64)
        plan.cursorReady = pl.astype(1, pl.DT_INT64)
    while ident > plan.cursorEnd and plan.cursorBatch + 1 < ci.bSize:
        plan.cursorBegin = pl.astype(plan.cursorEnd, pl.DT_INT64)
        plan.cursorBatch = pl.astype(plan.cursorBatch + 1, pl.DT_INT64)
        load_tnd_dense_batch(ci, plan, cuq, cuk, usedq, usedk)
        plan.cursorEnd = pl.astype(
            plan.cursorBegin + plan.cursorArea * copies, pl.DT_INT64
        )
    while ident <= plan.cursorBegin and plan.cursorBatch > 0:
        plan.cursorEnd = pl.astype(plan.cursorBegin, pl.DT_INT64)
        plan.cursorBatch = pl.astype(plan.cursorBatch - 1, pl.DT_INT64)
        load_tnd_dense_batch(ci, plan, cuq, cuk, usedq, usedk)
        plan.cursorBegin = pl.astype(
            plan.cursorEnd - plan.cursorArea * copies, pl.DT_INT64
        )


def init_tnd_dense_schedule(ci, k, cuq, cuk, usedq, usedk):
    """Mirror CalcleTNDDenseDeterParam, with an explicit all-zero fast exit."""
    area = pl.astype(0, pl.DT_INT64)
    max_m = pl.astype(0, pl.DT_INT64)
    max_n = pl.astype(0, pl.DT_INT64)
    for batch in pl.range(ci.bSize):
        sq, sk = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
        m = (sq + 127) // 128
        n = (sk + 127) // 128
        area = pl.astype(area + m * n, pl.DT_INT64)
        max_m = pl.astype(pl.max(max_m, m), pl.DT_INT64)
        max_n = pl.astype(pl.max(max_n, n), pl.DT_INT64)
    rounds = pl.astype(0, pl.DT_INT64)
    if area > 0 and k > 0:
        total = area * ci.n2Size * ci.gSize
        rounds = pl.astype(
            pl.max(pl.max((total + k - 1) // k, max_m * ci.gSize), max_n),
            pl.DT_INT64,
        )
    return make_tnd_deter_schedule(rounds, rounds, 0, 0, 0, 0, 0)


def tnd_dense_gidx(ci, c, k, r, plan, cuq, cuk, usedq, usedk):
    """Map 0-based (core, round) to a synthetic tile index, or -1.

    The batch cursor has the same prefix semantics as the original scan, but
    only reloads lengths/geometry on a batch transition. The core/round ID,
    permutation and returned synthetic max-shape coordinate are unchanged.
    """
    out = pl.astype(-1, pl.DT_INT64)
    ident = pl.astype(c * plan.maxRound + r + 1, pl.DT_INT64)
    if c >= 0 and c < k and r >= 0 and r < plan.maxRound and ci.bSize > 0:
        seek_tnd_dense_batch(ci, ident, plan, cuq, cuk, usedq, usedk)
        if ident > plan.cursorBegin and ident <= plan.cursorEnd:
            head, go, x, y = tnd_permute_delta(
                ident - plan.cursorBegin,
                plan.cursorM,
                plan.cursorN,
                ci.gSize,
                plan.maxRound,
            )
            out = pl.astype(
                ((plan.cursorBatch * ci.n2Size + head) * ci.gSize + go) * ci.s1oS2o
                + (y - 1) * ci.s1Outer
                + x
                - 1,
                pl.DT_INT64,
            )
    return out
