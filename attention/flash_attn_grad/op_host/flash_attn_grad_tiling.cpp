/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling.h"
#include <set>

#include "flash_attn_grad_tiling_check.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

namespace optiling {

static bool HasAttenMask(gert::TilingContext *context, int64_t maskMode)
{
    if (maskMode == 0 || (maskMode != 3 && maskMode != 4)) {
        return false;
    }
    auto attnMaskShape = context->GetOptionalInputShape(ATTN_MASK_INDEX);
    return (attnMaskShape != nullptr) && (attnMaskShape->GetStorageShape().GetDimNum() > 0);
}

static ge::graphStatus FlashAttnGradTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", aicNum, aivNum),
                return ge::GRAPH_PARAM_INVALID);

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGE(context->GetNodeName(), "SOC Version is not support.");
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto checkRet = FlashAttnGradCheck::CheckParams(context);
    OP_CHECK_IF(checkRet != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckParams failed."),
                return ge::GRAPH_PARAM_INVALID);

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    auto attrs = context->GetAttrs();
    std::string layoutQStr = "BSND";
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;
    int64_t maskMode = 0;
    float scaleValue = 0.0f;
    if (attrs != nullptr) {
        auto scalePtr = attrs->GetAttrPointer<float>(0);
        if (scalePtr != nullptr) {
            scaleValue = *scalePtr;
        }
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(1);
        if (maskModePtr != nullptr) {
            maskMode = *maskModePtr;
        }
        auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(4);
        if (maxSeqlenQPtr != nullptr) {
            maxSeqlenQ = *maxSeqlenQPtr;
        }
        auto maxSeqlenKvPtr = attrs->GetAttrPointer<int64_t>(5);
        if (maxSeqlenKvPtr != nullptr) {
            maxSeqlenKv = *maxSeqlenKvPtr;
        }
        auto layoutQPtr = attrs->GetAttrPointer<char>(6);
        if (layoutQPtr != nullptr) {
            layoutQStr = std::string(layoutQPtr);
        }
    }

    // tilingkey bit2 only distinguishes TND from non-TND; BNSD is a non-TND
    // layout and deliberately does NOT get a tilingkey bit of its own -- it is
    // handled at runtime through the view parameters below, so the tilingkey
    // combination count (and thus the operator binary size) stays unchanged.
    int64_t layout = (layoutQStr == "TND") ? 1 : 0;
    bool isBnsd = (layoutQStr == "BNSD");

    auto qShape = context->GetInputShape(Q_INDEX);
    auto kShape = context->GetInputShape(K_INDEX);
    auto vShape = context->GetInputShape(V_INDEX);

    int64_t bSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t dSize = 0;
    int64_t dvSize = 0;

    if (layout == 1) {
        s1Size = maxSeqlenQ > 0 ? maxSeqlenQ : qShape->GetStorageShape().GetDim(0);
        s2Size = maxSeqlenKv > 0 ? maxSeqlenKv : kShape->GetStorageShape().GetDim(0);
        n1Size = qShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(1);
        dSize = qShape->GetStorageShape().GetDim(2);
        dvSize = vShape->GetStorageShape().GetDim(2);
        auto cuSeqShape = context->GetOptionalInputShape(CU_SEQLENS_Q_INDEX);
        if (cuSeqShape != nullptr && cuSeqShape->GetStorageShape().GetDimNum() > 0) {
            bSize = cuSeqShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else if (isBnsd) {
        // BNSD [B,N,S,D]; matches the AscendC reference's derivation in
        // flash_attention_score_grad_infershape.cpp (the "BNSD" else branch).
        bSize = qShape->GetStorageShape().GetDim(0);
        n1Size = qShape->GetStorageShape().GetDim(1);
        s1Size = qShape->GetStorageShape().GetDim(2);
        dSize = qShape->GetStorageShape().GetDim(3);
        dvSize = vShape->GetStorageShape().GetDim(3);
        n2Size = kShape->GetStorageShape().GetDim(1);
        s2Size = kShape->GetStorageShape().GetDim(2);
    } else {
        bSize = qShape->GetStorageShape().GetDim(0);
        s1Size = qShape->GetStorageShape().GetDim(1);
        n1Size = qShape->GetStorageShape().GetDim(2);
        dSize = qShape->GetStorageShape().GetDim(3);
        dvSize = vShape->GetStorageShape().GetDim(3);
        s2Size = kShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(2);
    }

    // GQA: Q/dout/attn_out/dq carry N1 = N2*G heads while K/V/dk/dv carry N2.
    // G is derived by division, mirroring the AscendC reference
    // (flash_attention_score_grad_tiling_normal_regbase.cpp, e.g. the BNSD
    // branch: g = qShape.GetDim(1) / kShape.GetDim(1)). G == 1 is plain MHA.
    OP_CHECK_IF(n2Size <= 0 || n1Size <= 0,
                OP_LOGE(context->GetNodeName(), "head num must be positive, got N1=%ld, N2=%ld.", n1Size, n2Size),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(n1Size % n2Size != 0,
                OP_LOGE(context->GetNodeName(),
                        "head num of q (%ld) must be divisible by head num of k/v (%ld) for GQA.", n1Size, n2Size),
                return ge::GRAPH_PARAM_INVALID);
    int64_t gSize = n1Size / n2Size;

    // Decide the block-to-core mapping. Linear splitting spreads the live cores
    // across many heads: at S=8192 D=128 with 36 cores, core 0 works head 0
    // while core 35 works head 31, so 36 cores stream K/V for 32 heads at once
    // and the working set blows past L2 (128 MB). msprof shows mte2_ratio 0.72
    // on both cube and vector while mac *drops* to 0.733 -- HBM-bound, not
    // compute-bound. Swizzle hands out whole s2 columns strided across cores
    // instead, keeping the live cores inside one head.
    //
    // Mirrors the AscendC reference's isExceedL2Cache / isSplitByBlockIdx gate.
    // Count every tensor a head touches (~22 MB/head at S=8192): q+dout+attn_out
    // read fp16, dq written fp32, k+v read fp16, dk+dv written fp32. Counting
    // only K+V lands exactly on 128 MB at N=32 and wrongly reports "fits",
    // where swizzle actually wins by 34%.
    //
    // Swizzle's granularity is a whole column, so it is skipped when there are
    // too few columns to balance (N=1 has 64 columns for 36 cores: core 0 takes
    // 2 while core 35 takes 1, a 2x skew that costs 14%).
    int64_t swizzle = 0;
    if (layout != 1) {
        constexpr int64_t CUBE_BASE_M = 128;
        constexpr int64_t CUBE_BASE_N = 128;
        constexpr int64_t L2_CACHE_BYTES = 128L * 1024 * 1024;
        constexpr int64_t FP16_BYTES = 2;
        constexpr int64_t FP32_BYTES = 4;
        int64_t s1Outer = (s1Size + CUBE_BASE_M - 1) / CUBE_BASE_M;
        int64_t s2Outer = (s2Size + CUBE_BASE_N - 1) / CUBE_BASE_N;
        // Under GQA the block grid is b*n2*G*s1Outer*s2Outer and the fused batch
        // dimension is b*n2*G, so G simply multiplies the head count here (the
        // kernel walks g inside n2, see set_run_info).
        int64_t perHeadBlocks = s1Outer * s2Outer;
        int64_t fusedHeads = bSize * n2Size * gSize;
        int64_t totalBlocks = fusedHeads * perHeadBlocks;
        // Must use aicNum, not blockDim: blockDim is the mixed scheduling value
        // from CalcTschBlockDim, while the kernel's coreNum comes from
        // pl.get_block_num() which equals the physical cube core count (36).
        // Using blockDim here would miscount the live heads.
        if (perHeadBlocks > 0 && aicNum > 0) {
            int64_t cores = static_cast<int64_t>(aicNum);
            int64_t perCore = (totalBlocks + cores - 1) / cores;
            // How many distinct heads are live at once under linear splitting
            std::set<int64_t> liveHeads;
            for (int64_t c = 0; c < cores; ++c) {
                liveHeads.insert((c * perCore) / perHeadBlocks);
            }
            // Per fused (b,n2,g) unit: the Q-side tensors (q+dout+attn_out read
            // fp16, dq written fp32) are private to this g, while the KV-side
            // ones (k+v read fp16, dk+dv written fp32) are shared by all G of
            // them -- so charge the KV bytes once per G.
            int64_t perHeadBytes = 3 * s1Size * dSize * FP16_BYTES + s1Size * dSize * FP32_BYTES +
                                   (2 * s2Size * dSize * FP16_BYTES + 2 * s2Size * dSize * FP32_BYTES) / gSize;
            int64_t liveBytes = static_cast<int64_t>(liveHeads.size()) * perHeadBytes;
            // The gate depends on the *direction* of S1 vs S2, not just on the
            // working-set size. What swizzle buys is "one core walks a whole
            // column of s1Outer blocks", i.e. K/V loaded once and dK/dV
            // accumulated in resident L0C; what it costs is a coarser unit of
            // work (a column is s1Outer blocks). That trade scales with how
            // long a column is.
            //
            // 1) When S2 is the longer axis (s2Outer > s1Outer) swizzle never
            //    wins: the column is only s1Outer blocks so the reuse window is
            //    short, while the column count b*n*s2Outer is large enough that
            //    linear splitting already stays inside one head. Measured 0/5
            //    wins (36 cores, D=128, linear/swizzle us):
            //      N8  1024/8192 106MB 237.2/250.3   N16 1024/8192 212MB 491.9/492.1
            //      N16 2048/8192 232MB 848.2/858.4   N24 1024/8192 318MB 723.1/728.0
            //      N32  512/8192 404MB 629.3/632.1
            //    The last two are 318/404 MB working sets -- far past any
            //    threshold, yet still linear. Hence a separate direction test;
            //    no amount of threshold tuning recovers these.
            //
            // 2) When S1 dominates (s1Outer >= 2*s2Outer) the crossover comes
            //    earlier than in the symmetric case, so the threshold drops to
            //    1.25x L2. Measured boundary sits between 138 MB (linear) and
            //    172 MB (swizzle):
            //      N12 8192/512  129MB 194.9/245.9   N12 8192/1024 138MB 330.0/353.0
            //      N24 4096/512  138MB 201.2/201.6   N16 8192/512  172MB 277.4/263.7
            //      N16 8192/1024 184MB 509.3/474.3   N20 8192/1024 230MB 781.8/583.1
            //      N24 8192/1024 276MB 994.6/698.5   N32 8192/512  344MB 725.9/523.4
            //    1.1x-1.4x form one plateau (same 2 mispredictions); 1.25x is
            //    the midpoint.
            //
            // 3) The symmetric case keeps the original 2x L2, which was
            //    calibrated over N=8/10/12/16/24 and must not regress:
            //      N=8 176MB 1556/1761, N=10 220MB 1904/2007,
            //      N=12 264MB 2833/2677, N=16 352MB 4121/3550,
            //      N=24 528MB 6483/4982
            //    N=10 still prefers linear at 220 MB while the S1-heavy branch
            //    should already switch at 172 MB -- the two crossovers genuinely
            //    differ, so one global threshold cannot serve both.
            //
            // Over all 29 measured points: the old direction-agnostic 2x rule
            // mispredicts 7 (67.7% cumulative cost), this one mispredicts 2
            // (18.0%). Must stay identical to decide_swizzle() in the kernel.
            bool s1Heavy = (s1Outer >= 2 * s2Outer);
            // 1.25x expressed without floating point: 5*L2/4.
            int64_t threshold = s1Heavy ? (5 * L2_CACHE_BYTES / 4) : (2 * L2_CACHE_BYTES);
            if (s2Outer <= s1Outer && liveBytes > threshold && fusedHeads * s2Outer >= 2 * cores) {
                swizzle = 1;
            }
        }
    }

    FlashAttnGradTilingData *tilingData = context->GetTilingData<FlashAttnGradTilingData>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->n2 = n2Size;
    tilingData->d = dSize;
    tilingData->dv = dvSize;
    tilingData->scaleValue = scaleValue == 0.0f ? 1.0f / sqrt(static_cast<float>(dSize)) : scaleValue;

    // Collapse the layout difference into the 4D view parameters (see the field
    // comments in flash_attn_grad_tiling.h). The kernel then runs one pl.load
    // form for both layouts, picking the slice with plain arithmetic.
    // Two sets: the Q side has N1 = N2*G heads, the KV side has N2.
    tilingData->gSize = gSize;
    if (isBnsd) {
        tilingData->viewD0Q = bSize * n1Size;
        tilingData->viewD2Q = 1;
        tilingData->coefB0Q = n1Size;
        tilingData->viewD0KV = bSize * n2Size;
        tilingData->viewD2KV = 1;
        tilingData->coefB0KV = n2Size;
        tilingData->coefN0 = 1;
        tilingData->coefN2 = 0;
    } else {
        tilingData->viewD0Q = bSize;
        tilingData->viewD2Q = n1Size;
        tilingData->coefB0Q = 1;
        tilingData->viewD0KV = bSize;
        tilingData->viewD2KV = n2Size;
        tilingData->coefB0KV = 1;
        tilingData->coefN0 = 0;
        tilingData->coefN2 = 1;
    }

    bool hasAttenMask = HasAttenMask(context, maskMode);
    // D bucket. Unlike layout and GQA -- which only change index arithmetic and
    // so are handled with runtime numbers -- D sets the *width of every on-chip
    // tile*, and tile shapes are trace-time constants in pypto. It therefore has
    // to be a compile-time bucket, exactly as the AscendC reference makes it a
    // template parameter (DTemplateType::Aligned128 / Aligned192).
    //
    // Allocating everything at the maximum unconditionally is not merely slower,
    // it does not fit: L0C would need 288KB against a 256KB budget even with the
    // acc buffer reduced to single. Measured (us per D traversal, 96+96 split vs
    // 192 in one shot): MM2 0.466 vs 0.781, dQ/dK 0.738 vs 1.123.
    //
    // Dv stays a runtime value (tilingData->dv): it only narrows how many columns
    // of an already-allocated tile get filled, which needs no specialization.
    int64_t dAlign = (dSize <= 128) ? 128 : 192;
    OP_CHECK_IF(
        dSize > 192,
        OP_LOGE(context->GetNodeName(), "head dim of q (%ld) is not supported; only D <= 192 is implemented.", dSize),
        return ge::GRAPH_PARAM_INVALID);
    // Dv gets its own bucket for the same reason D does: the UB block holding
    // y/dy/prod/tmp is Dv wide, and the vector function that builds
    // softmaxGradFront walks it in trace-time-constant 128-element segments, so
    // the segment count is baked into the binary. Dispatching a Dv=192 shape
    // into the Dv=128 binary silently under-walks that buffer.
    // dvAlign <= dAlign always holds (CheckParams rejects Dv > D), which is
    // what keeps the added bit from doubling the combination count: the kernel's
    // is_valid prunes dv_align > d_align, so the total is 36, not 64.
    int64_t dvAlign = (dvSize <= 128) ? 128 : 192;
    // TilingKey: bit[1:0]=template(0=bn2gs1s2,1=bn2,2=bn2s2,3=bn2multiblk),
    //            bit2=layout(0=non-TND,1=TND), bit3=hasAttenMask,
    //            bit4=swizzle(0=linear split, 1=s2-column strided),
    //            bit5=d_align(0=128, 1=192), bit6=dv_align(0=128, 1=192)
    uint64_t tilingKey = static_cast<uint64_t>(0) | (static_cast<uint64_t>(layout) << 2) |
                         (static_cast<uint64_t>(hasAttenMask) << 3) | (static_cast<uint64_t>(swizzle) << 4) |
                         (static_cast<uint64_t>(dAlign == 192 ? 1 : 0) << 5) |
                         (static_cast<uint64_t>(dvAlign == 192 ? 1 : 0) << 6);
    context->SetTilingKey(tilingKey);

    // Workspace layout must match the kernel's make_ptr partitioning exactly:
    //   [0]          libapi reserve (system)
    //   [+libapi]    dq accumulator  B*S1*N1*D fp32   (N1 = N2*G)
    //   [+...]       dk accumulator  B*S2*N2*D fp32
    //   [+...]       dv accumulator  B*S2*N2*D fp32
    // dq follows q's head count (N1) while dk/dv follow k/v's (N2) -- under GQA
    // the G Q-heads of one KV head all atomicAdd into the same dk/dv slot.
    // The kernel accumulates dq/dk/dv in fp32 via atomicAdd, then its post
    // stage applies scale and casts down to the fp16/bf16 outputs. dv gets no
    // scale (matches qkvIdx < 2 in the AscendC reference).
    // dq/dk are D wide, dv is Dv wide (D and Dv may differ, e.g. 192 vs 128).
    constexpr size_t FP32_SIZE = 4;
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t dqElems = static_cast<size_t>(bSize) * s1Size * n1Size * dSize;
    size_t dkElems = static_cast<size_t>(bSize) * s2Size * n2Size * dSize;
    size_t dvElems = static_cast<size_t>(bSize) * s2Size * n2Size * dvSize;
    size_t workspaceSize = libapiSize + (dqElems + dkElems + dvElems) * FP32_SIZE;
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "FlashAttnGrad tiling: blockDim=%u, layoutStr=%s, layout=%ld, maskMode=%ld, hasAttenMask=%d, swizzle=%ld, "
            "B=%ld, S1=%ld, S2=%ld, N1=%ld, N2=%ld, G=%ld, D=%ld, Dv=%ld, viewQ=[%ld,S,%ld,D] b0=%ld, "
            "viewKV=[%ld,S,%ld,D] b0=%ld, coefN=(%ld,%ld), tilingKey=%llu, workspaceSize=%zu.",
            blockDim, layoutQStr.c_str(), layout, maskMode, hasAttenMask, swizzle, bSize, s1Size, s2Size, n1Size,
            n2Size, gSize, dSize, dvSize, tilingData->viewD0Q, tilingData->viewD2Q, tilingData->coefB0Q,
            tilingData->viewD0KV, tilingData->viewD2KV, tilingData->coefB0KV, tilingData->coefN0, tilingData->coefN2,
            tilingKey, workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFlashAttnGrad([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlashAttnGrad)
    .Tiling(FlashAttnGradTilingFunc)
    .TilingParse<FlashAttnGradCompileInfo>(TilingParseForFlashAttnGrad);

} // namespace optiling
