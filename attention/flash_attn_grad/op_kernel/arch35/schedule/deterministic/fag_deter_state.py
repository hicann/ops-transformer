# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-private deterministic schedule state and cached initialization.

Shared by legacy, fixed and TND decoders. Field order and the historical
FagTndDeterSchedule struct name are preserved for code generation stability.
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


from schedule.deterministic.fag_deter_formulas import _gcd, select_gqa_dense_round


def init_gqa_dense_plan(plan, k, m, n, b, g, dense_round):
    """Cache immutable GQA geometry in the kernel-private schedule."""
    if k > 0 and m > 0 and n > 0 and b > 0 and g > 1:
        selected_round = pl.astype(dense_round, pl.DT_INT64)
        if selected_round <= 0:
            selected_round = select_gqa_dense_round(k, m, n, b, g)
        active_k = pl.min(pl.min(k, b * g * m), b * n)
        nbg = pl.max(b * g, 1)
        common = pl.max(_gcd(nbg, selected_round), 1)
        plan.gqaDenseK = pl.astype(active_k, pl.DT_INT64)
        plan.gqaDenseRound = pl.astype(selected_round, pl.DT_INT64)
        plan.gqaDenseT1 = pl.astype(selected_round // common, pl.DT_INT64)
        plan.gqaDenseT2 = pl.astype(nbg // common, pl.DT_INT64)
    return plan


def make_tnd_deter_schedule(max_round, r0, r1, r2, core_divide, is_causal, r_whole):
    """Kernel-private plan and batch cursor; no tiling/workspace ABI fields.

    Legacy phases use (begin, end] in the unpermuted one-based ID space.
    Owned metadata phase 6 uses [begin, end) in zero-based task space.
    A phase change invalidates the cursor because per-batch counts differ.
    """
    return pl.struct(
        "FagTndDeterSchedule",
        metaOffset=pl.astype(0, pl.DT_INT64),
        normalized=pl.astype(0, pl.DT_INT64),
        normOffset=pl.astype(0, pl.DT_INT64),
        normFamily=pl.astype(0, pl.DT_INT64),
        strategy=pl.astype(0, pl.DT_INT64),
        protocolStatus=pl.astype(1, pl.DT_INT64),
        pairGroup=pl.astype(0, pl.DT_INT64),
        sparseType=pl.astype(0, pl.DT_INT64),
        maxRound=pl.astype(max_round, pl.DT_INT64),
        r0=pl.astype(r0, pl.DT_INT64),
        r1=pl.astype(r1, pl.DT_INT64),
        r2=pl.astype(r2, pl.DT_INT64),
        coreDivide=pl.astype(core_divide, pl.DT_INT64),
        lineDeter=pl.astype(0, pl.DT_INT64),
        isCausal=pl.astype(is_causal, pl.DT_INT64),
        rWhole=pl.astype(r_whole, pl.DT_INT64),
        # Immutable decoder parameters; private to one kernel invocation.
        gqaDenseK=pl.astype(0, pl.DT_INT64),
        gqaDenseRound=pl.astype(0, pl.DT_INT64),
        gqaDenseT1=pl.astype(0, pl.DT_INT64),
        gqaDenseT2=pl.astype(0, pl.DT_INT64),
        ownedCore=pl.astype(-1, pl.DT_INT64),
        ownedBegin=pl.astype(0, pl.DT_INT64),
        ownedEnd=pl.astype(0, pl.DT_INT64),
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
