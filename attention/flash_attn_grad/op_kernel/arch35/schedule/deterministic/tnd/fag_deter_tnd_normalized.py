# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed scheduling of effectively equal TND batches, with packed GM addressing.

Only the relevant MHA/GQA and mask branches compile in each capability family.
The metadata supplies both L2 decisions; the device uses its actual L2 size.
"""

from schedule.deterministic.tnd.fag_deter_tnd_dense import (
    pl,
    tnd_deter_lengths,
)
from schedule.deterministic.fag_deter_formulas import (
    cal_dense,
    cal_causal,
    cal_band,
    cal_gqa_band,
    cal_gqa_dense,
    cal_gqa_dense_precomputed,
    gqa_dense_parameters,
    _unpack_wxy,
    pack_fused,
    fused_valid_for_mask,
)
from schedule.deterministic.fag_deter_state import (
    init_gqa_dense_plan,
)
from schedule.deterministic.fag_deter_fixed import (
    cal_dense_swizzle,
    cal_causal_swizzle,
    cal_mha_band,
    cal_gqa_causal_plan,
)
from schedule.deterministic.fag_deter_baseline import (
    cal_source_mha_dense,
    cal_source_mha_causal,
    cal_source_mha_band,
    cal_source_gqa_band,
    source_baseline_valid,
)
from schedule.deterministic.fag_deter_band_owner import (
    cal_band_owner_plan_family,
)
from schedule.deterministic.fag_deter_band_plan import (
    cal_band_piecewise_plan,
    cal_band_overlap_plan,
)


def normalized_i64(metadata, base, field):
    low = pl.astype(pl.getval(metadata, base + field * 2), pl.DT_INT64)
    high = pl.astype(pl.getval(metadata, base + field * 2 + 1), pl.DT_INT64)
    if low < 0:
        low = low + 4294967296
    return high * 4294967296 + low


def normalized_fixed_ci(ci, metadata, base, plan_offset):
    sq = normalized_i64(metadata, base, 4)
    sk = normalized_i64(metadata, base, 5)
    return pl.struct(
        "FagNormalizedFixedInfo",
        bSize=ci.bSize,
        n2Size=ci.n2Size,
        gSize=ci.gSize,
        s1Size=sq,
        s2Size=sk,
        s1Outer=(sq + 127) // 128,
        s2Outer=(sk + 127) // 128,
        s1Token=normalized_i64(metadata, base, 13),
        s2Token=normalized_i64(metadata, base, 14),
        sequsedQOptional=0,
        sequsedKvOptional=0,
        deterFamily=normalized_i64(metadata, plan_offset, 0),
        deterSparseKind=normalized_i64(metadata, plan_offset, 1),
        deterMaxRound=normalized_i64(metadata, plan_offset, 2),
        deterDenseRound=normalized_i64(metadata, plan_offset, 3),
        deterCausalRoundsPerGroup=normalized_i64(metadata, plan_offset, 4),
        deterBandMode=normalized_i64(metadata, plan_offset, 5),
        deterM=normalized_i64(metadata, plan_offset, 6),
        deterN=normalized_i64(metadata, plan_offset, 7),
        deterP=normalized_i64(metadata, plan_offset, 8),
        deterQ=normalized_i64(metadata, plan_offset, 9),
        deterRowOffset=normalized_i64(metadata, plan_offset, 10),
        deterColOffset=normalized_i64(metadata, plan_offset, 11),
        deterBandPhaseCount=normalized_i64(metadata, plan_offset, 12),
        deterBandTailPhaseCount=normalized_i64(metadata, plan_offset, 13),
        deterBandK=normalized_i64(metadata, plan_offset, 14),
        deterBandB=normalized_i64(metadata, plan_offset, 15),
        deterBandG=normalized_i64(metadata, plan_offset, 16),
        deterBandGroupsPerChunk=normalized_i64(metadata, plan_offset, 17),
        deterBandFullRounds=normalized_i64(metadata, plan_offset, 18),
        deterBandTailRounds=normalized_i64(metadata, plan_offset, 19),
        deterBandTotalRounds=normalized_i64(metadata, plan_offset, 20),
        deterBandPhase0Kind=normalized_i64(metadata, plan_offset, 21),
        deterBandPhase0M=normalized_i64(metadata, plan_offset, 22),
        deterBandPhase0N=normalized_i64(metadata, plan_offset, 23),
        deterBandPhase0RowOffset=normalized_i64(metadata, plan_offset, 24),
        deterBandPhase0ColOffset=normalized_i64(metadata, plan_offset, 25),
        deterBandPhase0FullRounds=normalized_i64(metadata, plan_offset, 26),
        deterBandPhase0TailRounds=normalized_i64(metadata, plan_offset, 27),
        deterBandPhase1Kind=normalized_i64(metadata, plan_offset, 28),
        deterBandPhase1M=normalized_i64(metadata, plan_offset, 29),
        deterBandPhase1N=normalized_i64(metadata, plan_offset, 30),
        deterBandPhase1RowOffset=normalized_i64(metadata, plan_offset, 31),
        deterBandPhase1ColOffset=normalized_i64(metadata, plan_offset, 32),
        deterBandPhase1FullRounds=normalized_i64(metadata, plan_offset, 33),
        deterBandPhase1TailRounds=normalized_i64(metadata, plan_offset, 34),
        deterBandPhase2Kind=normalized_i64(metadata, plan_offset, 35),
        deterBandPhase2M=normalized_i64(metadata, plan_offset, 36),
        deterBandPhase2N=normalized_i64(metadata, plan_offset, 37),
        deterBandPhase2RowOffset=normalized_i64(metadata, plan_offset, 38),
        deterBandPhase2ColOffset=normalized_i64(metadata, plan_offset, 39),
        deterBandPhase2FullRounds=normalized_i64(metadata, plan_offset, 40),
        deterBandPhase2TailRounds=normalized_i64(metadata, plan_offset, 41),
        deterBandPhase3Kind=normalized_i64(metadata, plan_offset, 42),
        deterBandPhase3M=normalized_i64(metadata, plan_offset, 43),
        deterBandPhase3N=normalized_i64(metadata, plan_offset, 44),
        deterBandPhase3RowOffset=normalized_i64(metadata, plan_offset, 45),
        deterBandPhase3ColOffset=normalized_i64(metadata, plan_offset, 46),
        deterBandPhase3FullRounds=normalized_i64(metadata, plan_offset, 47),
        deterBandPhase3TailRounds=normalized_i64(metadata, plan_offset, 48),
        deterBandPhase4Kind=normalized_i64(metadata, plan_offset, 49),
        deterBandPhase4M=normalized_i64(metadata, plan_offset, 50),
        deterBandPhase4N=normalized_i64(metadata, plan_offset, 51),
        deterBandPhase4RowOffset=normalized_i64(metadata, plan_offset, 52),
        deterBandPhase4ColOffset=normalized_i64(metadata, plan_offset, 53),
        deterBandPhase4FullRounds=normalized_i64(metadata, plan_offset, 54),
        deterBandPhase4TailRounds=normalized_i64(metadata, plan_offset, 55),
        deterBandTailPhase0Kind=normalized_i64(metadata, plan_offset, 56),
        deterBandTailPhase0M=normalized_i64(metadata, plan_offset, 57),
        deterBandTailPhase0N=normalized_i64(metadata, plan_offset, 58),
        deterBandTailPhase0RowOffset=normalized_i64(metadata, plan_offset, 59),
        deterBandTailPhase0ColOffset=normalized_i64(metadata, plan_offset, 60),
        deterBandTailPhase0FullRounds=normalized_i64(metadata, plan_offset, 61),
        deterBandTailPhase0TailRounds=normalized_i64(metadata, plan_offset, 62),
        deterBandTailPhase1Kind=normalized_i64(metadata, plan_offset, 63),
        deterBandTailPhase1M=normalized_i64(metadata, plan_offset, 64),
        deterBandTailPhase1N=normalized_i64(metadata, plan_offset, 65),
        deterBandTailPhase1RowOffset=normalized_i64(metadata, plan_offset, 66),
        deterBandTailPhase1ColOffset=normalized_i64(metadata, plan_offset, 67),
        deterBandTailPhase1FullRounds=normalized_i64(metadata, plan_offset, 68),
        deterBandTailPhase1TailRounds=normalized_i64(metadata, plan_offset, 69),
        deterBandTailPhase2Kind=normalized_i64(metadata, plan_offset, 70),
        deterBandTailPhase2M=normalized_i64(metadata, plan_offset, 71),
        deterBandTailPhase2N=normalized_i64(metadata, plan_offset, 72),
        deterBandTailPhase2RowOffset=normalized_i64(metadata, plan_offset, 73),
        deterBandTailPhase2ColOffset=normalized_i64(metadata, plan_offset, 74),
        deterBandTailPhase2FullRounds=normalized_i64(metadata, plan_offset, 75),
        deterBandTailPhase2TailRounds=normalized_i64(metadata, plan_offset, 76),
        deterBandTailPhase3Kind=normalized_i64(metadata, plan_offset, 77),
        deterBandTailPhase3M=normalized_i64(metadata, plan_offset, 78),
        deterBandTailPhase3N=normalized_i64(metadata, plan_offset, 79),
        deterBandTailPhase3RowOffset=normalized_i64(metadata, plan_offset, 80),
        deterBandTailPhase3ColOffset=normalized_i64(metadata, plan_offset, 81),
        deterBandTailPhase3FullRounds=normalized_i64(metadata, plan_offset, 82),
        deterBandTailPhase3TailRounds=normalized_i64(metadata, plan_offset, 83),
        deterBandTailPhase4Kind=normalized_i64(metadata, plan_offset, 84),
        deterBandTailPhase4M=normalized_i64(metadata, plan_offset, 85),
        deterBandTailPhase4N=normalized_i64(metadata, plan_offset, 86),
        deterBandTailPhase4RowOffset=normalized_i64(metadata, plan_offset, 87),
        deterBandTailPhase4ColOffset=normalized_i64(metadata, plan_offset, 88),
        deterBandTailPhase4FullRounds=normalized_i64(metadata, plan_offset, 89),
        deterBandTailPhase4TailRounds=normalized_i64(metadata, plan_offset, 90),
    )


def normalized_plan_valid(ci, k, metadata, offset):
    family = normalized_i64(metadata, offset, 0)
    sparse = normalized_i64(metadata, offset, 1)
    rounds = normalized_i64(metadata, offset, 2)
    m = normalized_i64(metadata, offset, 6)
    n = normalized_i64(metadata, offset, 7)
    valid = pl.astype(0, pl.DT_INT64)
    if deter_family == 12:
        if sparse == 2 and (family == 21 or family == 1):
            valid = pl.astype(1, pl.DT_INT64)
    elif deter_family == 13:
        if mask_mode == 3:
            if (sparse == 3 and (family == 22 or family == 2)) or (
                sparse == 4 and (family == 23 or family == 3)
            ):
                valid = pl.astype(1, pl.DT_INT64)
        else:
            if (sparse == 2 and (family == 21 or family == 1)) or (
                sparse == 4 and (family == 23 or family == 3)
            ):
                valid = pl.astype(1, pl.DT_INT64)
    elif deter_family == 14:
        if sparse == 2 and family == 4:
            valid = pl.astype(1, pl.DT_INT64)
        if mask_mode == 4:
            if sparse == 4 and (family == 24 or (family >= 6 and family <= 11)):
                valid = pl.astype(1, pl.DT_INT64)
    elif deter_family == 15 or deter_family == 16 or deter_family == 19:
        if sparse == 3 and family == 5:
            valid = pl.astype(1, pl.DT_INT64)
    elif deter_family == 17 or deter_family == 18 or deter_family == 20:
        if (sparse == 3 and family == 5) or (
            sparse == 4 and (family == 24 or (family >= 6 and family <= 11))
        ):
            valid = pl.astype(1, pl.DT_INT64)
    if (
        rounds <= 0
        or rounds >= 4294967295
        or m <= 0
        or n <= 0
        or m > ci.s1Outer + ci.s2Outer + 2
        or n > 2 * (ci.s1Outer + ci.s2Outer) + 3
    ):
        valid = pl.astype(0, pl.DT_INT64)
    if (
        normalized_i64(metadata, offset, 4) < 0
        or normalized_i64(metadata, offset, 4) > rounds
        or normalized_i64(metadata, offset, 5) < 0
        or normalized_i64(metadata, offset, 5) > 4
        or normalized_i64(metadata, offset, 10) < 0
        or normalized_i64(metadata, offset, 11) < 0
    ):
        valid = pl.astype(0, pl.DT_INT64)
    if family == 4 and normalized_i64(metadata, offset, 3) <= 0:
        valid = pl.astype(0, pl.DT_INT64)
    phase_count = normalized_i64(metadata, offset, 12)
    tail_count = normalized_i64(metadata, offset, 13)
    if phase_count < 0 or phase_count > 5 or tail_count < 0 or tail_count > 5:
        valid = pl.astype(0, pl.DT_INT64)
    # Only these compile-time families can consume a GQA BAND phase plan.
    # Other families must not carry its runtime validation/phase traversal.
    if (
        (deter_family == 14 and mask_mode == 4)
        or deter_family == 17
        or deter_family == 18
        or deter_family == 20
    ):
        if family >= 6 and family <= 11:
            groups = normalized_i64(metadata, offset, 17)
            if (
                phase_count <= 0
                or groups <= 0
                or groups > ci.bSize * ci.n2Size
                or normalized_i64(metadata, offset, 14) != k
                or normalized_i64(metadata, offset, 15) != ci.bSize * ci.n2Size
                or normalized_i64(metadata, offset, 16) != ci.gSize
                or normalized_i64(metadata, offset, 18) <= 0
                or normalized_i64(metadata, offset, 20) != rounds
            ):
                valid = pl.astype(0, pl.DT_INT64)
            for side in pl.range(2):
                count = phase_count
                if side == 1:
                    count = tail_count
                for index in pl.range(5):
                    if index < count:
                        phase = 21 + side * 35 + index * 7
                        kind = normalized_i64(metadata, offset, phase)
                        if (
                            kind < 0
                            or kind == 10
                            or kind > 17
                            or normalized_i64(metadata, offset, phase + 1) <= 0
                            or normalized_i64(metadata, offset, phase + 2) <= 0
                            or normalized_i64(metadata, offset, phase + 5) < 0
                            or normalized_i64(metadata, offset, phase + 6) < 0
                        ):
                            valid = pl.astype(0, pl.DT_INT64)
    return valid


def apply_tnd_normalized_plan(ci, k, metadata, offset, cuq, cuk, usedq, usedk, plan):
    base = offset + 122 + 32 * ci.bSize
    legal = pl.astype(0, pl.DT_INT64)
    actual_equal = pl.astype(1, pl.DT_INT64)
    first_q, first_k = tnd_deter_lengths(ci, 0, cuq, cuk, usedq, usedk)
    for batch in pl.range(ci.bSize):
        actual_q, actual_k = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
        if actual_q != first_q or actual_k != first_k:
            actual_equal = pl.astype(0, pl.DT_INT64)
    previous_status = plan.protocolStatus
    plan.protocolStatus = pl.astype(9, pl.DT_INT64)
    if ci.metadataWordCount - base >= 396:
        if (
            normalized_i64(metadata, base, 0) == 0x464E5032
            and normalized_i64(metadata, base, 1) == 1
            and normalized_i64(metadata, base, 2) == 396
        ):
            if normalized_i64(metadata, base, 3) == 0 and actual_equal == 0:
                plan.protocolStatus = previous_status
            if normalized_i64(metadata, base, 3) == 1:
                legal = pl.astype(1, pl.DT_INT64)
                sq = normalized_i64(metadata, base, 4)
                sk = normalized_i64(metadata, base, 5)
                if (
                    sq <= 0
                    or sk <= 0
                    or sq > ci.s1Size
                    or sk > ci.s2Size
                    or normalized_i64(metadata, base, 6) < 0
                    or normalized_i64(metadata, base, 7) <= 0
                    or normalized_i64(metadata, base, 7) > k
                    or normalized_i64(metadata, base, 8) != ci.bSize
                    or normalized_i64(metadata, base, 9) != ci.n2Size
                    or normalized_i64(metadata, base, 10) != ci.gSize
                    or normalized_i64(metadata, base, 11) != k
                    or normalized_i64(metadata, base, 12) != mask_mode
                ):
                    legal = pl.astype(0, pl.DT_INT64)
                expected_pre = ci.s1Token - ci.s1Size + ci.s2Size + sq - sk
                expected_next = ci.s2Token + ci.s1Size - ci.s2Size - sq + sk
                if mask_mode == 0:
                    expected_pre = 2147483647
                    expected_next = 2147483647
                if (
                    normalized_i64(metadata, base, 13) != expected_pre
                    or normalized_i64(metadata, base, 14) != expected_next
                ):
                    legal = pl.astype(0, pl.DT_INT64)
                for batch in pl.range(ci.bSize):
                    actual_q, actual_k = tnd_deter_lengths(
                        ci, batch, cuq, cuk, usedq, usedk
                    )
                    if actual_q != sq or actual_k != sk:
                        legal = pl.astype(0, pl.DT_INT64)
                for slot in pl.range(2):
                    if (
                        normalized_plan_valid(ci, k, metadata, base + 32 + slot * 182)
                        == 0
                    ):
                        legal = pl.astype(0, pl.DT_INT64)
                chosen = base + 32
                if (
                    normalized_i64(metadata, base, 6) > ci.deterL2Bytes
                    and normalized_i64(metadata, base, 7) == k
                ):
                    chosen = chosen + 182
                if legal == 1:
                    plan.normalized = pl.astype(1, pl.DT_INT64)
                    plan.normOffset = chosen
                    plan.normFamily = normalized_i64(metadata, chosen, 0)
                    plan.maxRound = normalized_i64(metadata, chosen, 2)
                    plan.strategy = pl.astype(6, pl.DT_INT64)
                    plan.protocolStatus = pl.astype(0, pl.DT_INT64)
                    if deter_family == 14:
                        if plan.normFamily == 4:
                            plan = init_gqa_dense_plan(
                                plan,
                                k,
                                normalized_i64(metadata, chosen, 6),
                                normalized_i64(metadata, chosen, 7),
                                ci.bSize * ci.n2Size,
                                ci.gSize,
                                normalized_i64(metadata, chosen, 3),
                            )
                else:
                    plan.protocolStatus = pl.astype(9, pl.DT_INT64)
    return plan


def normalized_band_gidx_info(nci, k, j, r):
    family = nci.deterFamily
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(1, pl.DT_INT64)
    y = pl.astype(1, pl.DT_INT64)
    if family == 6:
        w, x, y = cal_band_piecewise_plan(nci, j, r)
    elif family >= 7 and family <= 10:
        w, x, y = cal_band_owner_plan_family(nci, j, r, family)
    elif family == 11:
        w, x, y = cal_band_overlap_plan(nci, j, r)
    else:
        w, x, y = cal_source_gqa_band(
            nci.deterBandK,
            nci.deterM,
            nci.deterN,
            nci.deterP,
            nci.deterQ,
            nci.bSize * nci.n2Size,
            nci.gSize,
            j,
            r,
        )
    return w, x, y


def tnd_normalized_gidx(ci, core, cores, physical_round, plan, metadata):
    base = plan.metaOffset + 122 + 32 * ci.bSize
    nci = normalized_fixed_ci(ci, metadata, base, plan.normOffset)
    j = core + 1
    r = physical_round + 1
    b = ci.bSize * ci.n2Size
    family = plan.normFamily
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(1, pl.DT_INT64)
    y = pl.astype(1, pl.DT_INT64)
    if deter_family == 12 or deter_family == 13:
        if deter_family == 12 or mask_mode == 4:
            if nci.deterSparseKind == 2:
                if family == 1:
                    w, x, y = cal_dense_swizzle(cores, nci.deterM, nci.deterN, b, j, r)
                else:
                    w, x, y = cal_source_mha_dense(
                        cores, nci.deterM, nci.deterN, b, j, r
                    )
        if deter_family == 13:
            if mask_mode == 3:
                if nci.deterSparseKind == 3:
                    if family == 2:
                        w, x, y = cal_causal_swizzle(
                            cores, nci.deterM, nci.deterN, b, j, r
                        )
                    else:
                        w, x, y = cal_source_mha_causal(
                            cores, nci.deterM, nci.deterN, b, j, r
                        )
            if nci.deterSparseKind == 4:
                if family == 3:
                    w, x, y = cal_mha_band(
                        cores,
                        nci.deterM,
                        nci.deterN,
                        nci.deterP,
                        nci.deterQ,
                        b,
                        j,
                        r,
                        nci.deterBandMode,
                    )
                else:
                    w, x, y = cal_source_mha_band(
                        cores, nci.deterM, nci.deterN, nci.deterP, nci.deterQ, b, j, r
                    )
    elif deter_family == 14:
        if nci.deterSparseKind == 2:
            active_k = plan.gqaDenseK
            selected_round = plan.gqaDenseRound
            t1 = plan.gqaDenseT1
            t2 = plan.gqaDenseT2
            if selected_round <= 0:
                active_k, selected_round, t1, t2 = gqa_dense_parameters(
                    cores, nci.deterM, nci.deterN, b, ci.gSize, nci.deterDenseRound
                )
            w, x, y = cal_gqa_dense_precomputed(
                active_k,
                nci.deterM,
                nci.deterN,
                b,
                j,
                r,
                ci.gSize,
                selected_round,
                t1,
                t2,
            )
        if mask_mode == 4:
            if nci.deterSparseKind == 4:
                w, x, y = normalized_band_gidx_info(nci, cores, j, r)
    elif deter_family == 15 or deter_family == 16 or deter_family == 19:
        w, x, y = cal_gqa_causal_plan(
            cores,
            nci.deterM,
            nci.deterN,
            b,
            j,
            r,
            ci.gSize,
            nci.deterCausalRoundsPerGroup,
        )
    elif deter_family == 17 or deter_family == 18 or deter_family == 20:
        if nci.deterSparseKind == 3:
            w, x, y = cal_gqa_causal_plan(
                cores,
                nci.deterM,
                nci.deterN,
                b,
                j,
                r,
                ci.gSize,
                nci.deterCausalRoundsPerGroup,
            )
        else:
            w, x, y = normalized_band_gidx_info(nci, cores, j, r)
    x = x + nci.deterRowOffset
    y = y + nci.deterColOffset
    valid = pl.astype(0, pl.DT_INT64)
    if family >= 21 and family <= 24:
        valid = source_baseline_valid(
            nci.s1Size, nci.s2Size, nci.s1Token, nci.s2Token, x - 1, y - 1
        )
    else:
        valid = fused_valid_for_mask(nci, x - 1, y - 1, mask_mode)
    if valid == 0:
        w = pl.astype(-1, pl.DT_INT64)
    return pack_fused(
        w, x, y, ci.bSize, ci.n2Size, ci.gSize, ci.s1Outer, ci.s2Outer, 0, 0
    )
