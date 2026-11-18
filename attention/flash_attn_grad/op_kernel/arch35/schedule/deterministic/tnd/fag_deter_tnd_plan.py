# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Versioned AICPU TND plans; no host readback and no per-round task table.

Coordinate formulas mirror AscendC deter.h production mode 2, line, owned BN2,
causal mirror P2, and causal column C4. A missing/invalid protocol takes the
existing closed-form schedule and exposes protocolStatus for trace validation.
"""

from schedule.deterministic.tnd.fag_deter_tnd_dense import (
    pl,
    init_tnd_dense_schedule,
    tnd_dense_gidx,
    tnd_deter_lengths,
)
from schedule.deterministic.fag_deter_state import (
    make_tnd_deter_schedule,
    reset_tnd_batch_cursor,
)
from schedule.deterministic.tnd.fag_deter_tnd_sparse import (
    init_tnd_sparse_schedule,
    tnd_sparse_gidx,
    tnd_sparse_band_shape,
)
from schedule.deterministic.tnd.fag_deter_tnd_normalized import (
    apply_tnd_normalized_plan,
    tnd_normalized_gidx,
)

TND_PLAN_MAGIC = 0x46545031
TND_PLAN_VERSION = 2
TND_PLAN_HEADER_WORDS = 16
TND_PLAN_BATCH_OFFSET = 122
TND_PLAN_BATCH_WORDS = 32
TND_PLAN_CUT_OFFSET = 48


def tnd_plan_i64(metadata, offset):
    low = pl.astype(pl.getval(metadata, offset), pl.DT_INT64)
    high = pl.astype(pl.getval(metadata, offset + 1), pl.DT_INT64)
    if low < 0:
        low = low + 4294967296
    return high * 4294967296 + low


def tnd_plan_scalar(metadata, base, field):
    return tnd_plan_i64(metadata, base + 16 + 2 * field)


def tnd_plan_batch(metadata, base, batch, field):
    return tnd_plan_i64(metadata, base + 122 + batch * 32 + field * 2)


def tnd_window_is_dense(ci, cuq, cuk, usedq, usedk):
    """Source SetSparseParams uses actual TND maxima and unshifted windows."""
    dense = pl.astype(1, pl.DT_INT64)
    pre = ci.s1Token - ci.s1Size + ci.s2Size
    nxt = ci.s2Token + ci.s1Size - ci.s2Size
    for batch in pl.range(ci.bSize):
        sq, sk = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
        if sq > pre or sk > nxt:
            dense = pl.astype(0, pl.DT_INT64)
    return dense


def tnd_plan_records_valid(
    ci,
    k,
    metadata,
    base,
    strategy,
    rounds,
    singles,
    sparse_type,
    cuq,
    cuk,
    usedq,
    usedk,
):
    """Validate only this compiled family's records once before optimized use."""
    valid = pl.astype(1, pl.DT_INT64)
    area = pl.astype(0, pl.DT_INT64)
    mode2_rounds = pl.astype(0, pl.DT_INT64)
    single_tasks = pl.astype(0, pl.DT_INT64)
    run_end = pl.astype(0, pl.DT_INT64)
    run_begin = pl.astype(0, pl.DT_INT64)
    line_end = singles
    run_m = pl.astype(0, pl.DT_INT64)
    run_n = pl.astype(0, pl.DT_INT64)
    run_p = pl.astype(0, pl.DT_INT64)
    run_q = pl.astype(0, pl.DT_INT64)
    if ci.bSize <= 0 or k <= 0 or k > 36:
        valid = pl.astype(0, pl.DT_INT64)
    for batch in pl.range(ci.bSize):
        sq, sk = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
        m = (sq + 127) // 128
        n = (sk + 127) // 128
        if (
            sq < 0
            or sk < 0
            or m > ci.s1Outer
            or n > ci.s2Outer
            or tnd_plan_batch(metadata, base, batch, 0) != sq
            or tnd_plan_batch(metadata, base, batch, 1) != sk
            or tnd_plan_batch(metadata, base, batch, 2) != m
            or tnd_plan_batch(metadata, base, batch, 3) != n
        ):
            valid = pl.astype(0, pl.DT_INT64)
        if deter_family == 12 or (
            deter_family == 13 and mask_mode == 4 and strategy == 1
        ):
            if (
                m <= 0
                or n <= 0
                or m < pl.min(k, n)
                or tnd_plan_batch(metadata, base, batch, 7) != mode2_rounds
            ):
                valid = pl.astype(0, pl.DT_INT64)
            mode2_rounds = mode2_rounds + ((n * ci.n2Size + k - 1) // k) * m
        elif deter_family >= 14:
            if m <= 0 or n <= 0 or tnd_plan_batch(metadata, base, batch, 6) != area:
                valid = pl.astype(0, pl.DT_INT64)
            if deter_family == 14:
                area = area + m * n
            elif deter_family == 15 or deter_family == 17:
                if sq != sk:
                    valid = pl.astype(0, pl.DT_INT64)
                area = area + m * (m + 1) // 2
            else:
                if sq != sk:
                    valid = pl.astype(0, pl.DT_INT64)
                area = area + (m + 1) * m
        else:
            p = pl.astype(0, pl.DT_INT64)
            q = pl.astype(0, pl.DT_INT64)
            shift = pl.astype(0, pl.DT_INT64)
            if mask_mode == 3:
                if sq != sk:
                    valid = pl.astype(0, pl.DT_INT64)
            else:
                m, n, p, q, mo, no, vm, vn, whole, bound = tnd_sparse_band_shape(
                    ci, sq, sk
                )
                shift = mo - no
                if sq <= 0 or sk <= 0:
                    m = pl.astype(0, pl.DT_INT64)
                    n = pl.astype(0, pl.DT_INT64)
                    p = pl.astype(0, pl.DT_INT64)
                    q = pl.astype(0, pl.DT_INT64)
                    shift = pl.astype(0, pl.DT_INT64)
            if (
                tnd_plan_batch(metadata, base, batch, 4) != p
                or tnd_plan_batch(metadata, base, batch, 5) != q
                or tnd_plan_batch(metadata, base, batch, 13) != m
                or tnd_plan_batch(metadata, base, batch, 14) != n
                or tnd_plan_batch(metadata, base, batch, 15) != shift
                or tnd_plan_batch(metadata, base, batch, 11) != single_tasks
            ):
                valid = pl.astype(0, pl.DT_INT64)
            if mask_mode == 3 and m == 1 and n == 1:
                single_tasks = single_tasks + ci.n2Size
            if tnd_plan_batch(metadata, base, batch, 12) != single_tasks:
                valid = pl.astype(0, pl.DT_INT64)
            if batch == run_end:
                run_begin = batch
                run_end = tnd_plan_batch(metadata, base, batch, 9)
                run_m = m
                run_n = n
                run_p = p
                run_q = q
                if run_end <= batch or run_end > ci.bSize:
                    valid = pl.astype(0, pl.DT_INT64)
                copies = pl.max(0, run_end - batch) * ci.n2Size
                run_rounds = pl.astype(0, pl.DT_INT64)
                if m > 0 and n > 0:
                    if mask_mode == 3:
                        if m != 1 or n != 1:
                            pairs = copies // 2
                            run_rounds = (((2 * n - m + 3) * pairs + k - 1) // k) * m
                            run_rounds = (
                                run_rounds + (copies % 2) * ((n + k - 1) // k) * m
                            )
                    elif p + q >= 2:
                        if p >= m:
                            run_rounds = ((n * copies + k - 1) // k) * m
                        elif p + q > m:
                            fold = pl.max(0, n - q - p + 1)
                            run_rounds = (((n - fold) * copies + k - 1) // k) * m
                        else:
                            l1 = q - 1
                            l2 = pl.min(n - q + 1, m + 2 - p - q)
                            l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
                            overlap = pl.astype(0, pl.DT_INT64)
                            if l3 != 0 and p <= l3:
                                overlap = pl.min(l3 - p + 1, l1)
                            width = (l1 + l2 + l3 - overlap) * copies + overlap
                            run_rounds = (p + q - 1) * ((width + k - 1) // k)
                line_end = line_end + run_rounds
            if (
                tnd_plan_batch(metadata, base, batch, 8) != run_begin
                or tnd_plan_batch(metadata, base, batch, 9) != run_end
                or tnd_plan_batch(metadata, base, batch, 10) != line_end
                or m != run_m
                or n != run_n
                or p != run_p
                or q != run_q
            ):
                valid = pl.astype(0, pl.DT_INT64)
    if (
        deter_family == 12 or (deter_family == 13 and mask_mode == 4 and strategy == 1)
    ) and (mode2_rounds != rounds or ci.gSize != 1):
        valid = pl.astype(0, pl.DT_INT64)
    if (
        deter_family == 13
        and strategy == 2
        and (
            line_end != rounds
            or (single_tasks + k - 1) // k != singles
            or run_end != ci.bSize
            or ci.gSize != 1
        )
    ):
        valid = pl.astype(0, pl.DT_INT64)
    if deter_family >= 14:
        scale = ci.n2Size * ci.gSize
        if deter_family == 16 or deter_family == 18:
            scale = (ci.n2Size // 2) * ci.gSize
        total = area * scale
        previous = tnd_plan_i64(metadata, base + 48)
        maximum = pl.astype(0, pl.DT_INT64)
        batch = pl.astype(0, pl.DT_INT64)
        if previous != 0:
            valid = pl.astype(0, pl.DT_INT64)
        for core in pl.range(k):
            cut = tnd_plan_i64(metadata, base + 48 + 2 * (core + 1))
            length = cut - previous
            if cut < previous or cut > total or length > rounds:
                valid = pl.astype(0, pl.DT_INT64)
            maximum = pl.max(maximum, length)
            while (
                batch + 1 < ci.bSize
                and cut >= tnd_plan_batch(metadata, base, batch + 1, 6) * scale
            ):
                batch = batch + 1
            m = tnd_plan_batch(metadata, base, batch, 2)
            n = tnd_plan_batch(metadata, base, batch, 3)
            local = cut - tnd_plan_batch(metadata, base, batch, 6) * scale
            unit = pl.max(1, m * n * ci.gSize)
            if deter_family == 15 or deter_family == 17:
                unit = pl.max(1, (m + 1) * ci.gSize)
            if deter_family == 16 or deter_family == 18:
                unit = pl.max(1, (m + 1) * ci.gSize)
            # The mirror's final odd centre column is half a paired unit.
            endpoint = pl.astype(0, pl.DT_INT64)
            if cut == total:
                endpoint = pl.astype(1, pl.DT_INT64)
            if local % unit != 0 and endpoint == 0:
                valid = pl.astype(0, pl.DT_INT64)
            previous = cut
        if previous != total or maximum != rounds:
            valid = pl.astype(0, pl.DT_INT64)
    return valid


def init_tnd_metadata_schedule(ci, k, metadata, cuq, cuk, usedq, usedk):
    """Validate the fixed header before any candidate/array reads.

    Status: 0 accepted, 1 no tail, 2 malformed header, 3 producer rejected,
    4 family mismatch, 5 reference mode-2 cache/full-core gate rejected,
    6 no optimized candidate, 7 invalid candidate scalars, 8 invalid records,
    9 malformed or inconsistent optional normalized plan; native TND fallback.
    """
    offset = pl.astype(0, pl.DT_INT64)
    status = pl.astype(1, pl.DT_INT64)
    strategy = pl.astype(0, pl.DT_INT64)
    rounds = pl.astype(0, pl.DT_INT64)
    singles = pl.astype(0, pl.DT_INT64)
    pair_group = pl.astype(0, pl.DT_INT64)
    sparse_type = pl.astype(0, pl.DT_INT64)
    dense_window = pl.astype(0, pl.DT_INT64)
    if mask_mode == 4:
        dense_window = tnd_window_is_dense(ci, cuq, cuk, usedq, usedk)
    normal_tail = pl.astype(0, pl.DT_INT64)
    expected_family = deter_family
    if deter_family == 17:
        expected_family = 15
    elif deter_family == 18:
        expected_family = 16
    elif deter_family == 19 or deter_family == 20:
        expected_family = 16
        if ci.n2Size == 1:
            expected_family = 15
    expected = 122 + 32 * ci.bSize
    if ci.metadataWordCount >= 8:
        fag_offset = pl.astype(pl.getval(metadata, 7), pl.DT_INT64)
        offset = fag_offset + 412
        if (
            fag_offset >= 16
            and offset <= ci.metadataWordCount
            and ci.metadataWordCount - offset >= 16
        ):
            status = pl.astype(2, pl.DT_INT64)
            version = pl.getval(metadata, offset + 1)
            wire_words = expected
            if version == 2:
                wire_words = expected + 396
            if (
                pl.getval(metadata, offset) == 0x46545031
                and (version == 1 or version == 2)
                and pl.getval(metadata, offset + 2) == wire_words
                and pl.getval(metadata, offset + 3) == 16
                and pl.getval(metadata, offset + 4) == k
                and pl.getval(metadata, offset + 5) == ci.bSize
                and pl.getval(metadata, offset + 9) == 16
                and pl.getval(metadata, offset + 10) == 16
                and pl.getval(metadata, offset + 11) == 37
                and wire_words <= ci.metadataWordCount - offset
            ):
                status = pl.astype(3, pl.DT_INT64)
                if pl.getval(metadata, offset + 8) == 0:
                    status = pl.astype(4, pl.DT_INT64)
                    if pl.getval(metadata, offset + 6) == expected_family:
                        if version == 2:
                            normal_tail = pl.astype(1, pl.DT_INT64)
                        strategy = pl.astype(
                            pl.getval(metadata, offset + 7), pl.DT_INT64
                        )
                        rounds = tnd_plan_scalar(metadata, offset, 0)
                        singles = tnd_plan_scalar(metadata, offset, 1)
                        pair_group = tnd_plan_scalar(metadata, offset, 6)
                        sparse_type = tnd_plan_scalar(metadata, offset, 9)
                        status = pl.astype(7, pl.DT_INT64)
                        legal = pl.astype(0, pl.DT_INT64)
                        if (
                            deter_family == 12
                            and strategy == 1
                            and ci.gSize == 1
                            and sparse_type == 0
                        ):
                            legal = pl.astype(1, pl.DT_INT64)
                        if deter_family == 13 and ci.gSize == 1:
                            if strategy == 2:
                                if (mask_mode == 3 and sparse_type == 1) or (
                                    mask_mode == 4
                                    and sparse_type == 2
                                    and dense_window == 0
                                ):
                                    legal = pl.astype(1, pl.DT_INT64)
                            if (
                                mask_mode == 4
                                and strategy == 1
                                and sparse_type == 0
                                and dense_window == 1
                            ):
                                legal = pl.astype(1, pl.DT_INT64)
                        if (
                            deter_family == 14
                            and strategy == 3
                            and ci.gSize > 1
                            and sparse_type == 0
                        ):
                            if mask_mode == 0 or (mask_mode == 4 and dense_window == 1):
                                legal = pl.astype(1, pl.DT_INT64)
                        if (
                            (deter_family == 15 or deter_family == 17)
                            and strategy == 4
                            and pair_group == 2
                            and ci.n2Size == 1
                            and ci.gSize >= 4
                            and ci.gSize % 4 == 0
                            and sparse_type == 1
                        ):
                            legal = pl.astype(1, pl.DT_INT64)
                        if (
                            (deter_family == 16 or deter_family == 18)
                            and strategy == 5
                            and pair_group == 4
                            and ci.n2Size >= 2
                            and ci.n2Size % 2 == 0
                            and ci.gSize >= 4
                            and ci.gSize % 4 == 0
                            and sparse_type == 1
                        ):
                            legal = pl.astype(1, pl.DT_INT64)
                        if (
                            legal == 1
                            and rounds > 0
                            and rounds < 4294967295
                            and singles >= 0
                            and singles <= rounds
                            and sparse_type >= 0
                            and sparse_type <= 2
                            and tnd_plan_scalar(metadata, offset, 11) == ci.n2Size
                            and tnd_plan_scalar(metadata, offset, 12) == ci.gSize
                            and tnd_plan_scalar(metadata, offset, 10) == mask_mode
                        ):
                            status = pl.astype(0, pl.DT_INT64)
                            if deter_family == 12 or (
                                deter_family == 13 and mask_mode == 4 and strategy == 1
                            ):
                                if (
                                    tnd_plan_scalar(metadata, offset, 7)
                                    <= ci.deterL2Bytes
                                    or tnd_plan_scalar(metadata, offset, 8) != k
                                ):
                                    status = pl.astype(5, pl.DT_INT64)
                        if strategy == 0:
                            status = pl.astype(6, pl.DT_INT64)
                        if status == 0:
                            if (
                                tnd_plan_records_valid(
                                    ci,
                                    k,
                                    metadata,
                                    offset,
                                    strategy,
                                    rounds,
                                    singles,
                                    sparse_type,
                                    cuq,
                                    cuk,
                                    usedq,
                                    usedk,
                                )
                                == 0
                            ):
                                status = pl.astype(8, pl.DT_INT64)
    plan = make_tnd_deter_schedule(rounds, singles, 0, 0, 0, 0, 0)
    if status != 0:
        if mask_mode == 0 or (mask_mode == 4 and dense_window == 1):
            plan = init_tnd_dense_schedule(ci, k, cuq, cuk, usedq, usedk)
        else:
            plan = init_tnd_sparse_schedule(ci, k, cuq, cuk, usedq, usedk)
        strategy = pl.astype(0, pl.DT_INT64)
        if mask_mode == 4:
            sparse_type = pl.astype(2, pl.DT_INT64)
            if dense_window == 1:
                sparse_type = pl.astype(0, pl.DT_INT64)
    plan.metaOffset = offset
    plan.strategy = strategy
    plan.protocolStatus = status
    plan.pairGroup = pair_group
    plan.sparseType = sparse_type
    if normal_tail == 1:
        plan = apply_tnd_normalized_plan(
            ci, k, metadata, offset, cuq, cuk, usedq, usedk, plan
        )
        if plan.protocolStatus == 9:
            if mask_mode == 0 or (mask_mode == 4 and dense_window == 1):
                plan = init_tnd_dense_schedule(ci, k, cuq, cuk, usedq, usedk)
            else:
                plan = init_tnd_sparse_schedule(ci, k, cuq, cuk, usedq, usedk)
            plan.metaOffset = offset
            plan.protocolStatus = pl.astype(9, pl.DT_INT64)
            plan.sparseType = sparse_type
    return plan


def tnd_plan_encode(ci, batch, head, group, x, y):
    """Zero-based logical coordinates to the existing synthetic max-shape ID."""
    return (
        ((batch * ci.n2Size + head) * ci.gSize + group) * ci.s1oS2o + y * ci.s1Outer + x
    )


def tnd_mode2_gidx(ci, c, k, r, plan, metadata):
    out = pl.astype(-1, pl.DT_INT64)
    batch = pl.astype(0, pl.DT_INT64)
    found = pl.astype(0, pl.DT_INT64)
    while batch < ci.bSize and found == 0:
        begin = tnd_plan_batch(metadata, plan.metaOffset, batch, 7)
        m = tnd_plan_batch(metadata, plan.metaOffset, batch, 2)
        n = tnd_plan_batch(metadata, plan.metaOffset, batch, 3)
        if m > 0 and n > 0:
            end = begin + ((n * ci.n2Size * ci.gSize + k - 1) // k) * m
            if r >= begin and r < end:
                delta = r - begin
                ident = (delta // m) * k + c
                if ident < n * ci.n2Size:
                    head = ident // n
                    y = ident % n
                    x = (y + delta) % m
                    out = pl.astype(
                        tnd_plan_encode(ci, batch, head, 0, x, y), pl.DT_INT64
                    )
                found = pl.astype(1, pl.DT_INT64)
        batch = batch + 1
    return out


def load_tnd_owned_batch(ci, plan, metadata):
    """Cache one accepted metadata batch in the owned task coordinate space."""
    m = tnd_plan_batch(metadata, plan.metaOffset, plan.cursorBatch, 2)
    n = tnd_plan_batch(metadata, plan.metaOffset, plan.cursorBatch, 3)
    prefix = tnd_plan_batch(metadata, plan.metaOffset, plan.cursorBatch, 6)
    scale = ci.n2Size * ci.gSize
    area = m * n
    if deter_family == 15 or deter_family == 17:
        area = m * (m + 1) // 2
    if deter_family == 16 or deter_family == 18:
        n = pl.min(m, n)
        area = (2 * m - n + 1) * n
        scale = (ci.n2Size // 2) * ci.gSize
    plan.cursorM = pl.astype(m, pl.DT_INT64)
    plan.cursorN = pl.astype(n, pl.DT_INT64)
    plan.cursorArea = pl.astype(area, pl.DT_INT64)
    plan.cursorBegin = pl.astype(prefix * scale, pl.DT_INT64)
    plan.cursorEnd = pl.astype(plan.cursorBegin + area * scale, pl.DT_INT64)


def seek_tnd_owned_batch(ci, task, plan, metadata):
    """Seek a zero-based task; phase 6 keeps owned and fallback cursors apart.

    Increasing physical rounds only cross each batch once. Reverse seeking
    preserves random-access decoder checks, including empty batch intervals.
    The caller guarantees a positive batch count before entering this helper.
    """
    reset_tnd_batch_cursor(plan, 6)
    if plan.cursorReady == 0:
        load_tnd_owned_batch(ci, plan, metadata)
        plan.cursorReady = pl.astype(1, pl.DT_INT64)
    while task >= plan.cursorEnd and plan.cursorBatch + 1 < ci.bSize:
        plan.cursorBatch = pl.astype(plan.cursorBatch + 1, pl.DT_INT64)
        load_tnd_owned_batch(ci, plan, metadata)
    while task < plan.cursorBegin and plan.cursorBatch > 0:
        plan.cursorBatch = pl.astype(plan.cursorBatch - 1, pl.DT_INT64)
        load_tnd_owned_batch(ci, plan, metadata)


def tnd_owned_gidx(ci, c, k, r, plan, metadata):
    out = pl.astype(-1, pl.DT_INT64)
    if plan.ownedCore != c:
        plan.ownedBegin = tnd_plan_i64(metadata, plan.metaOffset + 48 + 2 * c)
        plan.ownedEnd = tnd_plan_i64(metadata, plan.metaOffset + 48 + 2 * (c + 1))
        plan.ownedCore = pl.astype(c, pl.DT_INT64)
        plan.cursorPhase = pl.astype(-1, pl.DT_INT64)
    begin = plan.ownedBegin
    end = plan.ownedEnd
    if begin >= 0 and end >= begin and r < end - begin and ci.bSize > 0:
        task = begin + r
        seek_tnd_owned_batch(ci, task, plan, metadata)
        batch = plan.cursorBatch
        m = plan.cursorM
        n = plan.cursorN
        local = task - plan.cursorBegin
        if m > 0 and n > 0 and local >= 0 and task < plan.cursorEnd:
            head = pl.astype(0, pl.DT_INT64)
            group = pl.astype(0, pl.DT_INT64)
            x = pl.astype(0, pl.DT_INT64)
            y = pl.astype(0, pl.DT_INT64)
            if deter_family == 14:
                group_tasks = m * n * ci.gSize
                head = local // group_tasks
                rem = local % group_tasks
                y = rem // (ci.gSize * m)
                group = (rem % (ci.gSize * m)) // m
                x = rem % m
            elif deter_family == 15 or deter_family == 17:
                height = m + 1
                column_tasks = height * (ci.gSize // 2)
                column = local // column_tasks
                yy = pl.astype(m - column // 2, pl.DT_INT64)
                if column % 2 == 0:
                    yy = column // 2 + 1
                z = (r + yy) % column_tasks
                a = z // plan.pairGroup
                head_pair = plan.pairGroup * (a // height) + z % plan.pairGroup
                xx = a % height + 1
                if xx >= yy + 1:
                    group = 2 * head_pair
                    x = xx - 2
                    y = yy - 1
                else:
                    group = 2 * head_pair + 1
                    x = m - xx
                    y = m - yy
            else:
                height = 2 * m - n + 1
                column = local // (height * ci.gSize)
                pair = column // n
                yy = column % n + 1
                a = r // 4
                group = 4 * ((a // height) % (ci.gSize // 4)) + r % 4
                xx = (a + c + 1) % height + 1
                if xx >= yy + m - n + 1:
                    head = 2 * pair
                    x = xx - (m - n + 1) - 1
                    y = yy - 1
                else:
                    head = 2 * pair + 1
                    x = m - xx
                    y = n - yy
            out = pl.astype(tnd_plan_encode(ci, batch, head, group, x, y), pl.DT_INT64)
    return out


def tnd_line_dense(k, m, n, copies, c, r):
    valid = pl.astype(0, pl.DT_INT64)
    w = pl.astype(0, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if m > 0 and n > 0 and copies > 0 and r >= 0:
        ident = (r // m) * k + c
        if ident < n * copies:
            w = ident // n
            y = ident % n
            x = (y + r) % m
            valid = pl.astype(1, pl.DT_INT64)
    return valid, w, x, y


def tnd_line_gidx(ci, c, k, r, plan, metadata):
    out = pl.astype(-1, pl.DT_INT64)
    batch = pl.astype(0, pl.DT_INT64)
    found = pl.astype(0, pl.DT_INT64)
    if r < plan.r0:
        task = r * k + c
        while batch < ci.bSize and found == 0:
            begin = tnd_plan_batch(metadata, plan.metaOffset, batch, 11)
            end = tnd_plan_batch(metadata, plan.metaOffset, batch, 12)
            if task >= begin and task < end:
                out = pl.astype(
                    tnd_plan_encode(ci, batch, task - begin, 0, 0, 0), pl.DT_INT64
                )
                found = pl.astype(1, pl.DT_INT64)
            batch = batch + 1
    else:
        begin = plan.r0
        while batch < ci.bSize and found == 0:
            run_end = tnd_plan_batch(metadata, plan.metaOffset, batch, 9)
            end = tnd_plan_batch(metadata, plan.metaOffset, batch, 10)
            if r >= begin and r < end and run_end > batch and run_end <= ci.bSize:
                m = tnd_plan_batch(metadata, plan.metaOffset, batch, 13)
                n = tnd_plan_batch(metadata, plan.metaOffset, batch, 14)
                p = tnd_plan_batch(metadata, plan.metaOffset, batch, 4)
                q = tnd_plan_batch(metadata, plan.metaOffset, batch, 5)
                copies = (run_end - batch) * ci.n2Size
                local = r - begin
                valid = pl.astype(0, pl.DT_INT64)
                w = pl.astype(0, pl.DT_INT64)
                x = pl.astype(0, pl.DT_INT64)
                y = pl.astype(0, pl.DT_INT64)
                if mask_mode == 3:
                    vm = m
                    vn = 2 * n - m + 3
                    pairs = copies // 2
                    pair_rounds = ((vn * pairs + k - 1) // k) * vm
                    if local < pair_rounds:
                        valid, pair, x, y = tnd_line_dense(k, vm, vn, pairs, c, local)
                        w = 2 * pair
                        if y + 1 >= x + 1 + (n + 1) - m + 1:
                            y = 2 * (n + 1) - m - (y + 1) + 1
                            x = m - x - 1
                            w = 2 * pair + 1
                    else:
                        valid, unused, x, y = tnd_line_dense(
                            k, m, n, 1, c, local - pair_rounds
                        )
                        w = copies - 1
                    if y >= n or y > x + n - m + 1:
                        valid = pl.astype(0, pl.DT_INT64)
                elif p >= m:
                    valid, w, x, y = tnd_line_dense(k, m, n, copies, c, local)
                    if y > x + q - 1:
                        valid = pl.astype(0, pl.DT_INT64)
                elif p + q > m:
                    l1 = m - p
                    l2 = p + q - m
                    l3 = n - l1 - l2
                    fold = pl.max(0, l3 - p + 1)
                    valid, w, x, y = tnd_line_dense(k, m, n - fold, copies, c, local)
                    if fold > 0 and y < fold and x + 1 >= p + y + 1:
                        y = n - fold + y
                    elif y < l1 and x + 1 >= p + y + 1:
                        valid = pl.astype(0, pl.DT_INT64)
                    if y + 1 > l1 + l2 and x + 1 <= y + 1 - l1 - l2:
                        valid = pl.astype(0, pl.DT_INT64)
                else:
                    seg = p + q - 1
                    if seg > 0:
                        layer = local // seg
                        phase = local % seg + 1
                        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
                        ident = layer * k + c + 1
                        if l3 == 0 or n - m < 1:
                            w = (ident - 1) // n
                            y = (ident - 1) % n
                            x = y + phase - q
                            valid = pl.astype(1, pl.DT_INT64)
                        else:
                            x_abs = ident + phase - q
                            if x_abs >= 1:
                                w = (x_abs - 1) // m
                                x = (x_abs - 1) % m
                                y = x + q - phase
                                valid = pl.astype(1, pl.DT_INT64)
                if (
                    valid == 1
                    and w >= 0
                    and w < copies
                    and x >= 0
                    and x < m
                    and y >= 0
                    and y < n
                ):
                    target_batch = batch + w // ci.n2Size
                    raw_m = tnd_plan_batch(metadata, plan.metaOffset, target_batch, 2)
                    raw_n = tnd_plan_batch(metadata, plan.metaOffset, target_batch, 3)
                    if mask_mode == 3:
                        x = x + pl.max(0, raw_m - raw_n - 1)
                    else:
                        shift = tnd_plan_batch(
                            metadata, plan.metaOffset, target_batch, 15
                        )
                        x = x + pl.max(0, shift)
                        y = y + pl.max(0, -shift)
                    out = pl.astype(
                        tnd_plan_encode(ci, target_batch, w % ci.n2Size, 0, x, y),
                        pl.DT_INT64,
                    )
                found = pl.astype(1, pl.DT_INT64)
            begin = end
            # A corrupt run range must never produce an infinite decoder loop.
            batch = pl.max(batch + 1, run_end)
    return out


def tnd_source_valid_gidx(ci, index, cuq, cuk, usedq, usedk):
    """Apply AscendC IsValidDeterForTnd after decoding a physical round."""
    if mask_mode == 3 or mask_mode == 4:
        if index >= 0:
            batch = index // (ci.n2Size * ci.gSize * ci.s1oS2o)
            x = index % ci.s1Outer
            y = (index % ci.s1oS2o) // ci.s1Outer
            actual_q, actual_k = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
            pre = ci.s1Token - ci.s1Size + ci.s2Size + actual_q - actual_k
            nxt = ci.s2Token + ci.s1Size - ci.s2Size - actual_q + actual_k
            left = pl.max(x * 128 - pre, 0) // 64 * 64
            right = (pl.min((x + 1) * 128, ci.s1Size) + nxt + 63) // 64 * 64
            right = pl.min(right, actual_k)
            if y * 128 >= right or pl.min((y + 1) * 128, actual_k) <= left:
                index = pl.astype(-1, pl.DT_INT64)
    return index


def tnd_metadata_gidx(ci, c, k, r, plan, metadata, cuq, cuk, usedq, usedk):
    out = pl.astype(-1, pl.DT_INT64)
    if c >= 0 and c < k and r >= 0 and r < plan.maxRound:
        if plan.normalized == 1:
            out = tnd_normalized_gidx(ci, c, k, r, plan, metadata)
        elif plan.strategy == 0:
            if mask_mode == 0 or (mask_mode == 4 and plan.sparseType == 0):
                out = tnd_dense_gidx(ci, c, k, r, plan, cuq, cuk, usedq, usedk)
            else:
                out = tnd_sparse_gidx(ci, c, k, r, plan, cuq, cuk, usedq, usedk)
        elif deter_family == 12:
            out = tnd_mode2_gidx(ci, c, k, r, plan, metadata)
        elif deter_family == 13:
            # Full finite windows normalize to Dense only after actual lengths
            # are known. Keep precisely mode2 + line in this mask4 family.
            if mask_mode == 4 and plan.strategy == 1:
                out = tnd_mode2_gidx(ci, c, k, r, plan, metadata)
            else:
                out = tnd_line_gidx(ci, c, k, r, plan, metadata)
        else:
            out = tnd_owned_gidx(ci, c, k, r, plan, metadata)
        if plan.normalized == 0:
            out = tnd_source_valid_gidx(ci, out, cuq, cuk, usedq, usedk)
    return out
