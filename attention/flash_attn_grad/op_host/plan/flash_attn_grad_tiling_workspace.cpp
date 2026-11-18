/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_workspace.h"

#include <algorithm>
#include <limits>

namespace optiling {
namespace {
constexpr size_t FP32_SIZE = 4;
constexpr size_t FAG_WS_ALIGN_ROWS = 128;
// Kernel pointers and scalar offsets use signed int64 arithmetic.
constexpr size_t MAX_WORKSPACE_SIZE = static_cast<size_t>(std::numeric_limits<int64_t>::max());

bool CheckedAdd(size_t a, size_t b, size_t& result)
{
    if (a > MAX_WORKSPACE_SIZE || b > MAX_WORKSPACE_SIZE - a) {
        return false;
    }
    result = a + b;
    return true;
}

bool CheckedMul(size_t a, size_t b, size_t& result)
{
    if (a != 0 && b > MAX_WORKSPACE_SIZE / a) {
        return false;
    }
    result = a * b;
    return true;
}

size_t CeilDiv(size_t value, size_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

bool NzRows(const FagParsedInfo& info, int64_t totalTokens, int64_t maxSeq, size_t& rows)
{
    size_t tiles = 0;
    if (info.isTnd) {
        // Batch b starts at floor(cu[b]/128)+b. The per-head capacity is
        // ceil(T/128)+max(B-1,0), avoiding a B*maxSeq allocation or a host
        // readback of cu_seqlens. seqused never changes this physical layout.
        const size_t batchGaps = info.b > 0 ? static_cast<size_t>(info.b - 1) : 0;
        if (!CheckedAdd(CeilDiv(static_cast<size_t>(totalTokens), FAG_WS_ALIGN_ROWS), batchGaps, tiles)) {
            return false;
        }
    } else if (!CheckedMul(static_cast<size_t>(info.b), CeilDiv(static_cast<size_t>(maxSeq), FAG_WS_ALIGN_ROWS),
                           tiles)) {
        return false;
    }
    return CheckedMul(tiles, FAG_WS_ALIGN_ROWS, rows);
}

bool AppendSegment(size_t rows, size_t heads, size_t width, size_t& cursor, FagWorkspaceSegment& segment)
{
    size_t elems = 0;
    segment.offset = cursor;
    return CheckedMul(rows, heads, elems) && CheckedMul(elems, width, elems) &&
           CheckedMul(elems, FP32_SIZE, segment.size) && CheckedAdd(cursor, segment.size, cursor);
}
} // namespace

// ND: physical T*N*D fp32. NZ: each (batch, head, S128 tile) is independently
// stored as [cols/8, 128, 8] fp32, where the kernel's cols is ceil16(D).
// The host reserves the template-width upper bound (dAlign=dvAlign=128);
// the kernel computes its own contiguous segment offsets at cols<=128.
// These host segment offsets are not part of the kernel tiling ABI.
// Tail tiles retain all 128 rows. For TND,
// tileIndex=(cu[b]/128+b)*N + head*ceil(physicalSeq/128)+sTile;
// fixed layouts use ((b*N+head)*ceil(S/128)+sTile). External outputs stay ND.
// Q uses N1 heads; K/V use N2, so GQA query groups still accumulate into the
// same KV tile. Preclear covers physical storage, including seqused tails.
// Post scales dQ/dK and casts all three gradients; dV is not scaled.
// The trailing 128-row pad and dsink slots use the same offsets in the kernel.
bool BuildWorkspacePlan(const FagParsedInfo& info, const FagKernelPlan& kernel, FagWorkspacePlan& plan)
{
    plan = FagWorkspacePlan{};
    if (info.b < 0 || info.s1 < 0 || info.s2 < 0 || info.t1 < 0 || info.t2 < 0 || info.n1 < 0 || info.n2 < 0 ||
        info.d < 0 || info.dv < 0) {
        return false;
    }
    size_t qRows = static_cast<size_t>(info.t1);
    size_t kvRows = static_cast<size_t>(info.t2);
    size_t d = static_cast<size_t>(info.d);
    size_t dv = static_cast<size_t>(info.dv);
    if (kernel.nzOut != 0) {
        if (!NzRows(info, info.t1, info.s1, qRows) || !NzRows(info, info.t2, info.s2, kvRows)) {
            return false;
        }
        // Preserve the mainline template-width reservation. The actual kernel
        // packs all segments at ceil16(D)<=128, so this remains a safe upper
        // bound, including the trailing pad; no host offset is sent to it.
        d = static_cast<size_t>(kernel.dAlign);
        dv = static_cast<size_t>(kernel.dvAlign);
    }

    plan.libapiSize = info.libapiWorkspaceSize;
    size_t cursor = plan.libapiSize;
    if (!AppendSegment(qRows, static_cast<size_t>(info.n1), d, cursor, plan.dq) ||
        !AppendSegment(kvRows, static_cast<size_t>(info.n2), d, cursor, plan.dk) ||
        !AppendSegment(kvRows, static_cast<size_t>(info.n2), dv, cursor, plan.dv)) {
        return false;
    }

    size_t alignRowBytes = 0;
    if (!CheckedMul(FAG_WS_ALIGN_ROWS * FP32_SIZE, std::max(d, dv), alignRowBytes) ||
        !CheckedAdd(cursor, alignRowBytes, cursor) ||
        !AppendSegment(static_cast<size_t>(info.aivNum), static_cast<size_t>(info.n1), 1, cursor, plan.dsink)) {
        return false;
    }

    // sfmg 前移缓存：[N1*T1] fp32，token 主序。kernel 的 addptr 切分与这里
    // 对齐（dsink 段之后）。无条件预留：非前移档位不读，dq 段的 1/D，
    // 避免 host/kernel 判据漂移导致越界。
    if (!AppendSegment(static_cast<size_t>(info.n1), static_cast<size_t>(info.t1), 1, cursor, plan.sfmg)) {
        return false;
    }

    plan.totalSize = cursor;
    return true;
}

} // namespace optiling
