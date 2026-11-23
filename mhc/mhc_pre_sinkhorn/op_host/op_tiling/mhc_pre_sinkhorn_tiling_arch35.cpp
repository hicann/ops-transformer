/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_sinkhorn_tiling_arch35.cpp
 * \brief
 */

#include "op_common/op_host/util/op_const_def.h"
#include <sstream>
#include "mhc_pre_sinkhorn_tiling.h"
#include "util/shape_util.h"

using namespace ge;
namespace optiling {
namespace MhcPreSinkhornRegbase {
struct CoreRowTiling {
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
    int64_t usedCoreNum = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t tailRowFactorOfTailBlock = 0;
};

struct RowAndDTiling {
    int64_t rowFactor = 0;
    int64_t dLoop = 0;
    int64_t dFactor = 0;
    int64_t tailDFactor = 0;
};

struct PremixUbTiling {
    int64_t kUbSize = 0;
    int64_t mUbSize = 0;
    int64_t bufferPool0Size = 0;
    int64_t bufferPool1Size = 0;
    bool castSegments = false;
};
} // namespace MhcPreSinkhornRegbase

namespace {
using MhcPreSinkhornRegbase::CoreRowTiling;
using MhcPreSinkhornRegbase::RowAndDTiling;
using MhcPreSinkhornRegbase::PremixUbTiling;

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BF16_BYTES = 2;
constexpr int64_t CUBE_D_ALIGN = 16;
constexpr int64_t D_FACTOR_ALIGN = 32;
constexpr uint64_t M_L1_MAX_SIZE = 256;
// The fallback stage-1 search reduces the K window by 64 elements.
constexpr int64_t PREMIX_MIN_KUB_SIZE = 64;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t HALF_SPLIT = 2;
constexpr int64_t K_L1_ALIGN_SIZE = 128;
// Accepts any tail-axis(c) size that is 128 aligned within [1, 100000].
constexpr int64_t C_ALIGN_SIZE = 128;
constexpr int64_t C_MIN_VALUE = 1;
constexpr int64_t C_MAX_VALUE = 100000;
constexpr int64_t HCMULT_VALUE = 4;
constexpr int64_t NUM_ITERS_VALUE = 20;
constexpr size_t X_DIM_NUM_3 = 3;
constexpr size_t X_DIM_NUM_4 = 4;
constexpr int32_t BATCH_CONSISTENCY_LEVEL = 3; // batch一致性开关使能的确定性级别
constexpr uint32_t BATCH_MODE = 1;
constexpr uint64_t TILING_KEY_NO_GRAD_K_SPLIT = 1000;
constexpr uint64_t TILING_KEY_NO_GRAD_M_SPLIT = 1001;
constexpr uint64_t TILING_KEY_GRAD_K_SPLIT = 2000;
constexpr uint64_t TILING_KEY_GRAD_M_SPLIT = 2001;
constexpr uint64_t TILING_KEY_CV_ONE_NO_GRAD_K_SPLIT = 3000;
constexpr uint64_t TILING_KEY_CV_ONE_NO_GRAD_M_SPLIT = 3001;
constexpr uint64_t TILING_KEY_CV_ONE_GRAD_K_SPLIT = 4000;
constexpr uint64_t TILING_KEY_CV_ONE_GRAD_M_SPLIT = 4001;
constexpr uint64_t CV_RATIO_ONE = 1;
constexpr uint64_t CV_RATIO_TWO = 2;

struct PremixUbConfig {
    int64_t hcMix = 0;
    int64_t hcMult = 0;
    int64_t maxMRows = 0;
    int64_t maxGroupK = 0;
    int64_t d = 0;
    int64_t kL1Size = 0;
    int64_t initialKUbSize = 0;
    bool isKSplit = false;
    bool batchConsistency = false;
};

struct UbBufferConfig {
    int64_t hcMix = 0;
    int64_t hcMult = 0;
    int64_t hcMultAlign = 0;
    int64_t iterTimes = 0;
    int64_t kBlockNum = 0;
    int64_t vectorMSize = 0;
    bool withGradOut = false;
    bool isKSplit = false;
    bool isMpmd = false;
    bool hasPremix = false;
    bool premixStage1 = false;
};

bool CalcCoreRowTiling(int64_t bs, int64_t coreNum, CoreRowTiling& rowTiling)
{
    if (bs <= 0 || coreNum <= 0) {
        return false;
    }
    rowTiling.rowOfFormerBlock = CeilDiv(bs, coreNum);
    rowTiling.usedCoreNum = std::min(CeilDiv(bs, rowTiling.rowOfFormerBlock), coreNum);
    rowTiling.rowOfTailBlock = bs - (rowTiling.usedCoreNum - 1) * rowTiling.rowOfFormerBlock;
    return true;
}

void CompleteCoreRowTiling(CoreRowTiling& rowTiling, int64_t rowFactor)
{
    rowTiling.rowLoopOfFormerBlock = CeilDiv(rowTiling.rowOfFormerBlock, rowFactor);
    rowTiling.rowLoopOfTailBlock = CeilDiv(rowTiling.rowOfTailBlock, rowFactor);
    rowTiling.tailRowFactorOfFormerBlock =
        rowTiling.rowOfFormerBlock % rowFactor == 0 ? rowFactor : rowTiling.rowOfFormerBlock % rowFactor;
    rowTiling.tailRowFactorOfTailBlock =
        rowTiling.rowOfTailBlock % rowFactor == 0 ? rowFactor : rowTiling.rowOfTailBlock % rowFactor;
}

int64_t CalcUbBufferSize(const UbBufferConfig& config, int64_t rowFactor, int64_t dFactor)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t hcMixAlign = RoundUp(config.hcMix, floatAlign);
    const int64_t dAlign = RoundUp(dFactor, CUBE_D_ALIGN);

    int64_t totalSize = rowFactor * hcMixAlign * sizeof(float); // mixes
    if (!config.premixStage1) {
        totalSize += rowFactor * config.hcMult * dAlign * BF16_BYTES * DOUBLE_BUFFER; // x
        totalSize += rowFactor * dAlign * BF16_BYTES * DOUBLE_BUFFER;                 // y
        if (config.hasPremix) {
            totalSize += rowFactor * config.hcMultAlign * sizeof(float); // external weights
        }
    }
    if (config.hasPremix && !config.withGradOut) {
        // hPre remains an output regardless of which stage computes hIn.
        totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // hPre
    }
    totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // post
    if (!config.isMpmd) {
        totalSize += rowFactor * config.hcMult * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // combFrag
    }

    if (config.isKSplit) {
        totalSize += config.kBlockNum * rowFactor * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;         // mm
        totalSize += config.kBlockNum * RoundUp(rowFactor, floatAlign) * sizeof(float) * DOUBLE_BUFFER; // rms
        totalSize += config.hcMultAlign * sizeof(float) * 2;                                            // base0/base1
        if (!config.isMpmd) {
            totalSize += config.hcMult * config.hcMultAlign * sizeof(float); // base2
        }
    }

    if (config.withGradOut) {
        totalSize += 2 * config.iterTimes * rowFactor * config.hcMult * config.hcMultAlign * sizeof(float) *
                     DOUBLE_BUFFER;                                                                         // normOut
        totalSize += 2 * config.iterTimes * rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // sumOut
        totalSize += RoundUp(rowFactor, floatAlign) * sizeof(float) * DOUBLE_BUFFER;                        // invRmsOut
        const int64_t hcBeforeNormRows = config.isKSplit ? rowFactor : config.vectorMSize;
        totalSize += hcBeforeNormRows * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;  // hcBeforeNorm
        totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // hPre
    }
    return totalSize;
}

int64_t CalcMOnlyBufferPool1Size(uint64_t ubSize, int64_t vectorMSize, int64_t hcMix, int64_t hcMult,
                                 int64_t hcMultAlign)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t mmXBufSize = vectorMSize * RoundUp(hcMix, floatAlign) * sizeof(float);
    const int64_t rmsNormBufSize = RoundUp(vectorMSize, floatAlign) * sizeof(float);
    const int64_t baseSize = hcMultAlign * sizeof(float) * 2 + hcMult * hcMultAlign * sizeof(float);
    return DownAlign(static_cast<int64_t>(ubSize) - mmXBufSize - rmsNormBufSize - baseSize, BLOCK_SIZE);
}

int64_t CalcStage1TemporaryUbSize(int64_t kUbSize, int64_t mUbSize)
{
    const int64_t b16KAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(uint16_t));
    const int64_t floatKAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(float));
    int64_t totalSize = DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t); // xQue
    totalSize += mUbSize * (floatKAlign * sizeof(float) + BLOCK_SIZE);          // castBuf
    totalSize += DOUBLE_BUFFER * RoundUp(mUbSize, BLOCK_SIZE / sizeof(float)) * floatKAlign * sizeof(float);
    return totalSize; // nd2NzBuf
}

// Stage-1 UB footprint when hIn is fused with the invRms pass using the external preMix.
// hIn is accumulated for the whole AIV row range, so yAcc/preMix do not scale with mUbSize.
int64_t CalcMPremixPhase1UbSize(int64_t kUbSize, int64_t mUbSize, int64_t vectorMRows, int64_t hcMix)
{
    const int64_t b16KAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(uint16_t));
    const int64_t floatKAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(float));
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    int64_t totalSize = DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t);              // xQue
    totalSize += DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t);                     // yQue
    totalSize += mUbSize * (floatKAlign * sizeof(float) + BLOCK_SIZE);                       // cast
    totalSize += DOUBLE_BUFFER * RoundUp(mUbSize, floatAlign) * floatKAlign * sizeof(float); // nd2Nz
    totalSize += vectorMRows * RoundUp(hcMix, floatAlign) * sizeof(float);                   // preMix
    totalSize += vectorMRows * floatKAlign * sizeof(float);                                  // yAcc
    return totalSize;
}

int64_t CalcMPremixCastPhase1UbSize(int64_t kUbSize, int64_t mUbSize, int64_t hcMix, int64_t hcMult)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t b16KAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(uint16_t));
    const int64_t floatKAlign = RoundUp(kUbSize, floatAlign);
    int64_t totalSize = DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t);              // xQue
    totalSize += DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t);                     // yQue
    totalSize += hcMult * mUbSize * (floatKAlign * sizeof(float) + BLOCK_SIZE);              // cast slices
    totalSize += DOUBLE_BUFFER * RoundUp(mUbSize, floatAlign) * floatKAlign * sizeof(float); // nd2Nz
    totalSize += RoundUp(mUbSize, floatAlign) * RoundUp(hcMix, floatAlign) * sizeof(float);  // preMix
    return totalSize;
}

int64_t CalcMKPremixPhase1UbSize(int64_t kUbSize, int64_t mUbSize, int64_t maxMRows, int64_t maxGroupK, int64_t hcMix)
{
    const int64_t b16KAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(uint16_t));
    const int64_t floatKAlign = RoundUp(kUbSize, BLOCK_SIZE / sizeof(float));
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    int64_t totalSize = DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t);              // xQue
    totalSize += DOUBLE_BUFFER * mUbSize * b16KAlign * sizeof(uint16_t);                     // yQue
    totalSize += mUbSize * (floatKAlign * sizeof(float) + BLOCK_SIZE);                       // cast
    totalSize += DOUBLE_BUFFER * RoundUp(mUbSize, floatAlign) * floatKAlign * sizeof(float); // nd2Nz
    totalSize += maxMRows * RoundUp(hcMix, floatAlign) * sizeof(float);                      // preMix
    totalSize += maxMRows * floatKAlign * sizeof(float) * CeilDiv(maxGroupK, kUbSize);       // y accumulator
    return totalSize;
}

// Solves the stage-1 vector UB tiling for both ratios.
bool CalcCvUbTiling(int64_t kL1Size, int64_t mL1Size, uint64_t ubSize, uint64_t cvRatio, bool isKSplit, int64_t hcMix,
                    int64_t hcMult, int64_t hcMultAlign, PremixUbTiling& ubTiling)
{
    if (kL1Size <= 0 || mL1Size <= 0 || (cvRatio != CV_RATIO_ONE && cvRatio != CV_RATIO_TWO)) {
        return false;
    }
    if (cvRatio == CV_RATIO_TWO) {
        ubTiling.kUbSize = CeilDiv(kL1Size, static_cast<int64_t>(cvRatio));
        ubTiling.mUbSize = CeilDiv(mL1Size, static_cast<int64_t>(cvRatio));
        return true;
    }

    ubTiling.kUbSize = kL1Size;
    const int64_t vectorMSize = mL1Size;
    int64_t temporaryUbSize = 0;
    if (isKSplit) {
        const int64_t rmsQueSize = DOUBLE_BUFFER * RoundUp(vectorMSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
        temporaryUbSize = static_cast<int64_t>(ubSize) - rmsQueSize;
    } else {
        temporaryUbSize = CalcMOnlyBufferPool1Size(ubSize, vectorMSize, hcMix, hcMult, hcMultAlign);
    }
    if (temporaryUbSize <= 0 || CalcStage1TemporaryUbSize(ubTiling.kUbSize, 1) > temporaryUbSize) {
        return false;
    }

    int64_t left = 1;
    int64_t right = vectorMSize;
    while (left <= right) {
        const int64_t middle = left + (right - left) / 2;
        if (CalcStage1TemporaryUbSize(ubTiling.kUbSize, middle) <= temporaryUbSize) {
            ubTiling.mUbSize = middle;
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    return ubTiling.mUbSize > 0;
}

template <typename FitsInUb>
int64_t FindMaxRowFactor(int64_t maxRowFactor, const FitsInUb& fitsInUb)
{
    int64_t left = 1;
    int64_t right = maxRowFactor;
    int64_t result = 0;
    while (left <= right) {
        const int64_t middle = left + (right - left) / 2;
        if (fitsInUb(middle)) {
            result = middle;
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    return result;
}

template <typename FitsInUb>
int64_t FindFirstDFactorDivisor(int64_t d, const FitsInUb& fitsInUb)
{
    int64_t left = 2;
    int64_t right = d;
    while (left < right) {
        const int64_t middle = left + (right - left) / 2;
        if (fitsInUb(1, CeilDiv(d, middle))) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    return left;
}

template <typename FitsInUb>
bool CalcRowAndDTiling(int64_t d, int64_t maxRowFactor, const FitsInUb& fitsInUb, RowAndDTiling& result)
{
    if (d <= 0 || maxRowFactor <= 0 || !fitsInUb(1, 1)) {
        return false;
    }

    result.rowFactor = 1;
    result.dFactor = d;
    if (fitsInUb(1, d)) {
        result.rowFactor =
            FindMaxRowFactor(maxRowFactor, [&fitsInUb, d](int64_t rowFactor) { return fitsInUb(rowFactor, d); });
    } else {
        // fitsInUb(1, CeilDiv(d, base)) is monotonic in base, so binary search finds the first feasible base.
        result.dFactor = CeilDiv(d, FindFirstDFactorDivisor(d, fitsInUb));
        if (result.dFactor > D_FACTOR_ALIGN) {
            result.dFactor = DownAlign(result.dFactor, D_FACTOR_ALIGN);
        }
    }

    result.dLoop = CeilDiv(d, result.dFactor);
    result.tailDFactor = d % result.dFactor == 0 ? result.dFactor : d % result.dFactor;
    return result.rowFactor > 0 && result.dFactor > 0;
}

int64_t CalcPremixPhase1Size(const PremixUbConfig& config, int64_t mUbSize, int64_t kUbSize, bool castSegments)
{
    if (castSegments) {
        return CalcMPremixCastPhase1UbSize(kUbSize, mUbSize, config.hcMix, config.hcMult);
    }
    return config.isKSplit ?
               CalcMKPremixPhase1UbSize(kUbSize, mUbSize, config.maxMRows, config.maxGroupK, config.hcMix) :
               CalcMPremixPhase1UbSize(kUbSize, mUbSize, config.maxMRows, config.hcMix);
}

void TryPremixUbCandidate(const PremixUbConfig& config, int64_t availableSize, int64_t kUbSize, bool castSegments,
                          PremixUbTiling& ubTiling, int64_t& bestChunkCount)
{
    if (kUbSize <= 0 || kUbSize > config.kL1Size || (!config.isKSplit && config.d % kUbSize != 0)) {
        return;
    }
    const auto fitsInUb = [&](int64_t mUbSize) {
        return CalcPremixPhase1Size(config, mUbSize, kUbSize, castSegments) <= availableSize;
    };
    // Accumulators cover the full AIV row range, but the temporary x/cast/NZ
    // buffers can tile it. Keep full rows for four-cast and batch-consistent paths.
    const bool requireFullRows = castSegments || (!config.isKSplit && config.batchConsistency);
    const int64_t mUbSize = requireFullRows ? (fitsInUb(config.maxMRows) ? config.maxMRows : 0) :
                                              FindMaxRowFactor(config.maxMRows, fitsInUb);
    if (mUbSize == 0) {
        return;
    }
    const int64_t vectorBlocks = castSegments ? CeilDiv(kUbSize, REPEAT_SIZE / sizeof(float)) : 1;
    const int64_t chunkCount = CeilDiv(config.maxMRows, mUbSize) * CeilDiv(config.maxGroupK, kUbSize) * vectorBlocks;
    if (ubTiling.mUbSize == 0 || chunkCount < bestChunkCount ||
        (chunkCount == bestChunkCount && mUbSize > ubTiling.mUbSize) ||
        (chunkCount == bestChunkCount && mUbSize == ubTiling.mUbSize && kUbSize > ubTiling.kUbSize)) {
        bestChunkCount = chunkCount;
        ubTiling.kUbSize = kUbSize;
        ubTiling.mUbSize = mUbSize;
        ubTiling.castSegments = castSegments;
    }
}

void SelectPremixUbCandidates(const PremixUbConfig& config, int64_t availableSize, bool castSegments,
                              PremixUbTiling& ubTiling)
{
    const int64_t cubeAlign = static_cast<int64_t>(AscendC::BLOCK_CUBE);
    const int64_t fullKUb = RoundUp(config.maxGroupK, cubeAlign);
    const int64_t halfKUb = RoundUp(CeilDiv(config.maxGroupK, HALF_SPLIT), cubeAlign);
    int64_t bestChunkCount = 0;
    for (int64_t kUbSize : {fullKUb, halfKUb}) {
        TryPremixUbCandidate(config, availableSize, kUbSize, castSegments, ubTiling, bestChunkCount);
        // Batch consistency keeps the existing first-fit reduction partition.
        if (config.batchConsistency && ubTiling.mUbSize > 0) {
            break;
        }
    }
}

void SelectPremixUbFallback(const PremixUbConfig& config, int64_t availableSize, PremixUbTiling& ubTiling)
{
    const int64_t initialK = config.initialKUbSize > 0 ? config.initialKUbSize : config.kL1Size;
    for (int64_t kUbSize = std::min(initialK, config.kL1Size); kUbSize >= PREMIX_MIN_KUB_SIZE;
         kUbSize -= PREMIX_MIN_KUB_SIZE) {
        if (!config.isKSplit && config.d % kUbSize != 0) {
            continue;
        }
        const int64_t mUbSize = FindMaxRowFactor(config.maxMRows, [&](int64_t rows) {
            return CalcPremixPhase1Size(config, rows, kUbSize, false) <= availableSize;
        });
        if (mUbSize > 0) {
            ubTiling.kUbSize = kUbSize;
            ubTiling.mUbSize = mUbSize;
            return;
        }
    }
}

void SelectPremixPhase1Tiling(const PremixUbConfig& config, int64_t availableSize, PremixUbTiling& selected)
{
    if (!config.isKSplit && !config.batchConsistency) {
        SelectPremixUbCandidates(config, availableSize, true, selected);
    }
    if (selected.mUbSize == 0) {
        SelectPremixUbCandidates(config, availableSize, false, selected);
    }
    if (selected.mUbSize == 0) {
        SelectPremixUbFallback(config, availableSize, selected);
    }
}

void PrintFinalTilingData(gert::TilingContext* context, MhcPreSinkhornRegbaseTilingData& tilingData, uint64_t tilingKey,
                          uint64_t blockDim, size_t workspaceSize)
{
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling summary: tilingKey=%lu, blockDim=%lu, workspaceSize=%zu, "
            "tilingDataSize=%zu",
            tilingKey, blockDim, workspaceSize, tilingData.GetDataSize());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling shape: bs=%ld, hcMix=%ld, hcMult=%ld, d=%ld, hcMultAlign=%ld, "
            "iterTimes=%ld, hcEps=%g, normEps=%g",
            tilingData.get_bs(), tilingData.get_hcMix(), tilingData.get_hcMult(), tilingData.get_d(),
            tilingData.get_hcMultAlign(), tilingData.get_iterTimes(), tilingData.get_hcEps(), tilingData.get_normEps());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling core: rowOfFormerBlock=%ld, "
            "rowLoopOfFormerBlock=%ld, rowLoopOfTailBlock=%ld, stage2RowFactor=%ld, "
            "secondUsedCoreNum=%ld, rowInnerFactor=%ld",
            tilingData.get_rowOfFormerBlock(), tilingData.get_rowLoopOfFormerBlock(),
            tilingData.get_rowLoopOfTailBlock(), tilingData.get_stage2RowFactor(), tilingData.get_secondUsedCoreNum(),
            tilingData.get_rowInnerFactor());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling tail: tailRowFactorOfFormerBlock=%ld, "
            "tailRowFactorOfTailBlock=%ld, dLoop=%ld, dFactor=%ld, tailDFactor=%ld",
            tilingData.get_tailRowFactorOfFormerBlock(), tilingData.get_tailRowFactorOfTailBlock(),
            tilingData.get_dLoop(), tilingData.get_dFactor(), tilingData.get_tailDFactor());
    OP_LOGD(context->GetNodeName(),
            "MhcPreSinkhorn regbase final tiling split: k=%ld, cubeBlockDimM=%ld, cubeBlockDimK=%ld, "
            "kBlockFactor=%ld, multCoreSplitKSize=%ld, mL1Size=%ld, kL1Size=%ld, "
            "mUbSize=%ld, kUbSize=%ld, bufferPool0Size=%ld, bufferPool1Size=%ld",
            tilingData.get_k(), tilingData.get_cubeBlockDimM(), tilingData.get_cubeBlockDimK(),
            tilingData.get_kBlockFactor(), tilingData.get_multCoreSplitKSize(), tilingData.get_mL1Size(),
            tilingData.get_kL1Size(), tilingData.get_mUbSize(), tilingData.get_kUbSize(),
            tilingData.get_bufferPool0Size(), tilingData.get_bufferPool1Size());
}
} // namespace

MhcPreSinkhornTilingRegbase::MhcPreSinkhornTilingRegbase(gert::TilingContext* tilingContext)
    : context_(tilingContext)
{}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = context_->GetCompileInfo<MhcPreSinkhornCompileInfo>();
    if (platformInfo == nullptr) {
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        aicCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        sysWorkspaceSize_ = compileInfoPtr->sysWorkspaceSize;
        if (compileInfoPtr->cvRatio != 0) {
            cvRatio_ = compileInfoPtr->cvRatio;
        }
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
        npuArch_ = ascendcPlatform.GetCurNpuArch();
        OP_CHECK_IF(
            aicCoreNum_ == 0 || aivCoreNum_ == 0 || aivCoreNum_ % aicCoreNum_ != 0,
            OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Invalid CV core counts: aivCoreNum=%lu, aicCoreNum=%lu",
                                   aivCoreNum_, aicCoreNum_),
            return ge::GRAPH_FAILED);
        cvRatio_ = aivCoreNum_ / aicCoreNum_;
        if (compileInfoPtr != nullptr && compileInfoPtr->cvRatio != 0) {
            cvRatio_ = compileInfoPtr->cvRatio;
        }
    }
    OP_CHECK_IF(cvRatio_ != CV_RATIO_ONE && cvRatio_ != CV_RATIO_TWO,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Unsupported CV ratio: %lu", cvRatio_),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase platform: aivCoreNum=%lu, aicCoreNum=%lu, cvRatio=%lu, ubSize=%lu, "
            "sysWorkspaceSize=%lu",
            aivCoreNum_, aicCoreNum_, cvRatio_, ubSize_, sysWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(0);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(1);
    iterTimes_ = iterTimesAttr == nullptr ? 20 : *iterTimesAttr;

    auto epsAttr = attrs->GetAttrPointer<float>(2);
    hcEps_ = epsAttr == nullptr ? 1e-6 : *epsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(3);
    normEps_ = normEpsAttr == nullptr ? 1e-6 : *normEpsAttr;

    auto needGradAttr = attrs->GetAttrPointer<bool>(4);
    needGrad_ = needGradAttr == nullptr ? true : *needGradAttr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mult, d) or (bs, hc_mult, d)
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == X_DIM_NUM_3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == X_DIM_NUM_4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                 std::to_string(static_cast<int64_t>(xDimNum)).c_str(),
                                                 "The shape dim of input x must be 3 or 4");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(d_ < C_MIN_VALUE || d_ > C_MAX_VALUE || d_ % C_ALIGN_SIZE != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(d_).c_str(),
                    "The last dim c of input x must be an integer multiple of 128 in [1, 100000]."),
                return ge::GRAPH_FAILED);

    auto shapeHcFn = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHcFn);
    hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
    OP_CHECK_IF(shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "phi, x", Ops::Base::ToString(shapeHcFn->GetStorageShape()).c_str(),
                    ("The 1st dim of input phi must be equal to the last dim of input x times hc_mult, which is " +
                     std::to_string(d_ * hcMult_))
                        .c_str()),
                return ge::GRAPH_FAILED);

    auto shapeHcScale = context_->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHcScale);
    int64_t scaleFirstDim = shapeHcScale->GetStorageShape().GetDim(0);
    OP_CHECK_IF(scaleFirstDim != 3,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "alpha",
                                                          std::to_string(scaleFirstDim).c_str(),
                                                          "The size of input alpha must be 3"),
                return ge::GRAPH_FAILED);

    auto shapeHcBase = context_->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeHcBase);
    int64_t baseFirstDim = shapeHcBase->GetStorageShape().GetDim(0);
    OP_CHECK_IF(
        baseFirstDim != hcMix_,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "bias", std::to_string(baseFirstDim).c_str(),
            ("The size of input bias must be equal to hc_mult * hc_mult + 2 * hc_mult = " + std::to_string(hcMix_))
                .c_str()),
        return ge::GRAPH_FAILED);

    auto preMixShape = context_->GetOptionalInputShape(4);
    hasPremix_ = preMixShape != nullptr;
    if (hasPremix_) {
        const auto& shape = preMixShape->GetStorageShape();
        OP_CHECK_IF(
            shape.GetDimNum() != xDimNum - 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "preMix", std::to_string(shape.GetDimNum()).c_str(),
                ("The shape dim of input preMix must be one less than input x, which is " + std::to_string(xDimNum - 1))
                    .c_str()),
            return ge::GRAPH_FAILED);
        for (size_t i = 0; i + 1 < xDimNum; ++i) {
            OP_CHECK_IF(
                shape.GetDim(i) != xShape->GetStorageShape().GetDim(i),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "preMix, x", Ops::Base::ToString(shape).c_str(),
                    ("The dim " + std::to_string(i) + " of input preMix must be equal to the dim " + std::to_string(i) +
                     " of input x, which is " + std::to_string(xShape->GetStorageShape().GetDim(i)))
                        .c_str()),
                return ge::GRAPH_FAILED);
        }
        auto preMixDesc = context_->GetOptionalInputDesc(4);
        OP_CHECK_NULL_WITH_CONTEXT(context_, preMixDesc);
        OP_CHECK_IF(preMixDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "preMix",
                                                          Ops::Base::ToString(preMixDesc->GetDataType()).c_str(),
                                                          "The dtype of input preMix must be float32"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        hcMult_ != HCMULT_VALUE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "hc_mult", std::to_string(hcMult_).c_str(),
                                              "The value of attribute hc_mult must be 4"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        iterTimes_ != NUM_ITERS_VALUE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_iters", std::to_string(iterTimes_).c_str(),
                                              "The value of attribute num_iters must be 20"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase shape attrs: bs=%ld, hcMix=%ld, hcMult=%ld, d=%ld, iterTimes=%ld, "
            "needGrad=%d, hcEps=%g, normEps=%g",
            bs_, hcMix_, hcMult_, d_, iterTimes_, needGrad_ ? 1 : 0, hcEps_, normEps_);
    return ge::GRAPH_SUCCESS;
}

int64_t MhcPreSinkhornTilingRegbase::CalcPrePostCoreNum(bool withGradOut, bool isKSplit)
{
    int64_t prePostCoreNum = static_cast<int64_t>(aivCoreNum_);
    tilingData_.set_sinkhornCoreNum(0);
    tilingData_.set_sinkhornRowFactor(0);
    if (isKSplit && !withGradOut && bs_ >= 128 && aivCoreNum_ >= 4) {
        // MPMD follows A3: reserve a separate AIV group for Sinkhorn. The
        // group consumes compact [K, rows, 16] residual tiles from workspace.
        const int64_t sinkhornCoreNum = std::min(CeilDiv(bs_, 32), prePostCoreNum / 4);
        const int64_t kBlocks = tilingData_.get_cubeBlockDimK();
        const int64_t matrixElements = hcMult_ * hcMult_;
        const auto fitsSinkhorn = [&](int64_t rows) {
            const int64_t partialSize = kBlocks * (rows * matrixElements + RoundUp(rows, 8));
            const int64_t outputSize = rows * matrixElements;
            const int64_t biasSize = hcMult_ * RoundUp(hcMult_, 8);
            return (DOUBLE_BUFFER * (partialSize + outputSize) + biasSize) * sizeof(float) <= ubSize_;
        };
        const int64_t sinkhornRows = FindMaxRowFactor(64, fitsSinkhorn);
        if (sinkhornRows > 0) {
            tilingData_.set_sinkhornCoreNum(sinkhornCoreNum);
            tilingData_.set_sinkhornRowFactor(sinkhornRows);
            prePostCoreNum -= sinkhornCoreNum;
        }
    }
    return prePostCoreNum;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::InitUbTiling(bool isKSplit, PremixUbTiling& ubTiling)
{
    const int64_t hcMultAlign = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    OP_CHECK_IF(!CalcCvUbTiling(tilingData_.get_kL1Size(), tilingData_.get_mL1Size(), ubSize_, cvRatio_, isKSplit,
                                hcMix_, hcMult_, hcMultAlign, ubTiling),
                OP_LOGE_WITHOUT_REPORT(
                    context_->GetNodeName(),
                    "Invalid CV UB tiling: kL1Size=%ld, mL1Size=%ld, ubSize=%lu, cvRatio=%lu, isKSplit=%d",
                    tilingData_.get_kL1Size(), tilingData_.get_mL1Size(), ubSize_, cvRatio_, isKSplit ? 1 : 0),
                return ge::GRAPH_FAILED);
    const int64_t vectorMSize = CeilDiv(tilingData_.get_mL1Size(), static_cast<int64_t>(cvRatio_));
    ubTiling.bufferPool0Size = static_cast<int64_t>(ubSize_);
    ubTiling.bufferPool1Size =
        isKSplit ? 0 : CalcMOnlyBufferPool1Size(ubSize_, vectorMSize, hcMix_, hcMult_, hcMultAlign);
    OP_CHECK_IF(!isKSplit && ubTiling.bufferPool1Size < 0,
                OP_LOGE_WITHOUT_REPORT(
                    context_->GetNodeName(),
                    "M-split fixed buffers exceed UB: ubSize=%lu, bufferPool1Size=%ld, vectorMSize=%ld, hcMix=%ld",
                    ubSize_, ubTiling.bufferPool1Size, vectorMSize, hcMix_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::SelectPremixUbTiling(bool isKSplit, PremixUbTiling& ubTiling)
{
    PremixUbConfig config;
    config.hcMix = hcMix_;
    config.hcMult = hcMult_;
    config.maxMRows = CeilDiv(tilingData_.get_mL1Size(), static_cast<int64_t>(cvRatio_));
    config.d = d_;
    config.kL1Size = tilingData_.get_kL1Size();
    config.initialKUbSize = ubTiling.kUbSize;
    config.isKSplit = isKSplit;
    config.batchConsistency = context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL;
    const int64_t dPerCore = isKSplit ? tilingData_.get_multCoreSplitKSize() / hcMult_ : d_;
    config.maxGroupK = std::min(config.kL1Size, dPerCore);

    int64_t availableSize = isKSplit ? ubTiling.bufferPool0Size : ubTiling.bufferPool1Size;
    if (isKSplit) {
        availableSize -= DOUBLE_BUFFER * RoundUp(config.maxMRows, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    }
    PremixUbTiling selected;
    SelectPremixPhase1Tiling(config, availableSize, selected);
    OP_CHECK_IF(selected.mUbSize == 0,
                OP_LOGD(context_->GetNodeName(),
                        "UB is insufficient for premix stage-1 fusion: ubSize=%lu, available=%ld, "
                        "vectorMSize=%ld, mL1Size=%ld, kL1Size=%ld, hcMix=%ld",
                        ubSize_, availableSize, config.maxMRows, tilingData_.get_mL1Size(), config.kL1Size, hcMix_),
                return ge::GRAPH_FAILED);
    ubTiling.kUbSize = selected.kUbSize;
    ubTiling.mUbSize = selected.mUbSize;
    ubTiling.castSegments = selected.castSegments;
    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn premix stage-1 UB tiling: isKSplit=%d, kUbSize=%ld, mUbSize=%ld, available=%ld, "
            "castSegments=%d",
            isKSplit ? 1 : 0, ubTiling.kUbSize, ubTiling.mUbSize, availableSize, ubTiling.castSegments ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRowTiling(bool withGradOut, bool isKSplit,
                                                           const CoreRowTiling& coreRowTiling,
                                                           const PremixUbTiling& ubTiling, RowAndDTiling& rowAndDTiling)
{
    const int64_t vectorMSize = CeilDiv(tilingData_.get_mL1Size(), static_cast<int64_t>(cvRatio_));
    const int64_t availableUbSize = isKSplit ? ubTiling.bufferPool0Size : ubTiling.bufferPool1Size;
    UbBufferConfig bufferConfig;
    bufferConfig.hcMix = hcMix_;
    bufferConfig.hcMult = hcMult_;
    bufferConfig.hcMultAlign = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    bufferConfig.iterTimes = iterTimes_;
    bufferConfig.kBlockNum = tilingData_.get_cubeBlockDimK();
    bufferConfig.vectorMSize = vectorMSize;
    bufferConfig.withGradOut = withGradOut;
    bufferConfig.isKSplit = isKSplit;
    bufferConfig.isMpmd = tilingData_.get_sinkhornCoreNum() > 0;
    bufferConfig.hasPremix = hasPremix_;
    bufferConfig.premixStage1 = tilingData_.get_premixStage1() != 0;

    const auto fitsInUb = [&bufferConfig, availableUbSize](int64_t rowFactor, int64_t dFactor) {
        return CalcUbBufferSize(bufferConfig, rowFactor, dFactor) <= availableUbSize;
    };
    const int64_t maxRowFactor = isKSplit ? coreRowTiling.rowOfFormerBlock : vectorMSize;
    OP_CHECK_IF(!CalcRowAndDTiling(d_, maxRowFactor, fitsInUb, rowAndDTiling),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                                       "UB is insufficient for minimum tile: ubSize=%lu, availableUbSize=%ld, "
                                       "withGradOut=%d, isKSplit=%d",
                                       ubSize_, availableUbSize, withGradOut ? 1 : 0, isKSplit ? 1 : 0),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornTilingRegbase::SaveCommonTiling(bool withGradOut, bool isKSplit, const CoreRowTiling& coreRowTiling,
                                                   const RowAndDTiling& rowAndDTiling, const PremixUbTiling& ubTiling)
{
    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(RoundUp(hcMult_, BLOCK_SIZE / sizeof(float)));
    tilingData_.set_rowOfFormerBlock(coreRowTiling.rowOfFormerBlock);
    tilingData_.set_rowLoopOfFormerBlock(coreRowTiling.rowLoopOfFormerBlock);
    tilingData_.set_rowLoopOfTailBlock(coreRowTiling.rowLoopOfTailBlock);
    tilingData_.set_tailRowFactorOfFormerBlock(coreRowTiling.tailRowFactorOfFormerBlock);
    tilingData_.set_tailRowFactorOfTailBlock(coreRowTiling.tailRowFactorOfTailBlock);
    tilingData_.set_dLoop(rowAndDTiling.dLoop);
    tilingData_.set_dFactor(rowAndDTiling.dFactor);
    tilingData_.set_tailDFactor(rowAndDTiling.tailDFactor);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_hasPremix(hasPremix_ ? 1 : 0);
    tilingData_.set_premixCastSegments(ubTiling.castSegments ? 1 : 0);
    tilingData_.set_kUbSize(ubTiling.kUbSize);
    tilingData_.set_mUbSize(ubTiling.mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    if (isKSplit) {
        tilingData_.set_stage2RowFactor(rowAndDTiling.rowFactor);
        tilingData_.set_secondUsedCoreNum(coreRowTiling.usedCoreNum);
    } else {
        tilingData_.set_rowInnerFactor(rowAndDTiling.rowFactor);
        tilingData_.set_bufferPool0Size(ubTiling.bufferPool0Size);
        tilingData_.set_bufferPool1Size(ubTiling.bufferPool1Size);
    }

    const int64_t vectorMSize = CeilDiv(tilingData_.get_mL1Size(), static_cast<int64_t>(cvRatio_));
    const int64_t maxRowFactor = isKSplit ? coreRowTiling.rowOfFormerBlock : vectorMSize;
    const int64_t availableUbSize = isKSplit ? ubTiling.bufferPool0Size : ubTiling.bufferPool1Size;
    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase UB tiling: withGradOut=%d, isKSplit=%d, cvRatio=%lu, availableUbSize=%ld, "
            "kUbSize=%ld, mUbSize=%ld, vectorMSize=%ld, maxRowFactor=%ld, rowFactor=%ld, dFactor=%ld, dLoop=%ld, "
            "tailDFactor=%ld, premixStage1=%ld",
            withGradOut ? 1 : 0, isKSplit ? 1 : 0, cvRatio_, availableUbSize, ubTiling.kUbSize, ubTiling.mUbSize,
            vectorMSize, maxRowFactor, rowAndDTiling.rowFactor, rowAndDTiling.dFactor, rowAndDTiling.dLoop,
            rowAndDTiling.tailDFactor, tilingData_.get_premixStage1());
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseCommonTiling(GradOutMode gradOutMode, SplitMode splitMode)
{
    const bool withGradOut = gradOutMode == GradOutMode::ENABLED;
    const bool isKSplit = splitMode == SplitMode::K_SPLIT;
    tilingData_.set_premixStage1(hasPremix_ ? 1 : 0);
    const int64_t prePostCoreNum = CalcPrePostCoreNum(withGradOut, isKSplit);
    CoreRowTiling coreRowTiling;
    OP_CHECK_IF(!CalcCoreRowTiling(bs_, prePostCoreNum, coreRowTiling),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Invalid row tiling input: bs=%ld, aivCoreNum=%lu", bs_,
                                       aivCoreNum_),
                return ge::GRAPH_FAILED);
    PremixUbTiling ubTiling;
    if (InitUbTiling(isKSplit, ubTiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // For a single inference M tile, stage-2 hIn overlaps the independent
    // Sinkhorn group. Keep it off the stage-1 barrier's critical path.
    if (hasPremix_ && tilingData_.get_sinkhornCoreNum() > 0 && tilingData_.get_cubeBlockDimM() == 1 &&
        bs_ <= tilingData_.get_mL1Size()) {
        const int64_t vectorMSize = CeilDiv(tilingData_.get_mL1Size(), static_cast<int64_t>(cvRatio_));
        const int64_t rmsQueSize = DOUBLE_BUFFER * RoundUp(vectorMSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
        if (CalcStage1TemporaryUbSize(ubTiling.kUbSize, ubTiling.mUbSize) + rmsQueSize <= ubTiling.bufferPool0Size) {
            tilingData_.set_premixStage1(0);
        }
    }
    if (hasPremix_ && tilingData_.get_premixStage1() != 0 &&
        SelectPremixUbTiling(isKSplit, ubTiling) != ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(!isKSplit || withGradOut,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "UB is insufficient for premix stage-1 fusion."),
                    return ge::GRAPH_FAILED);
        // Retain inference support when fusion exceeds UB.
        tilingData_.set_premixStage1(0);
        OP_LOGD(context_->GetNodeName(), "MhcPreSinkhorn premix stage-2 UB fallback: bs=%ld, d=%ld, ubSize=%lu", bs_,
                d_, ubSize_);
    }
    RowAndDTiling rowAndDTiling;
    if (CalcRowTiling(withGradOut, isKSplit, coreRowTiling, ubTiling, rowAndDTiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    CompleteCoreRowTiling(coreRowTiling, rowAndDTiling.rowFactor);
    SaveCommonTiling(withGradOut, isKSplit, coreRowTiling, rowAndDTiling, ubTiling);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseOpTiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::DISABLED, SplitMode::M_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcRegbaseOpGradoutTiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::ENABLED, SplitMode::M_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcMKSplitCorePart2Tiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::DISABLED, SplitMode::K_SPLIT);
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcMKSplitCorePart2GradoutTiling()
{
    return CalcRegbaseCommonTiling(GradOutMode::ENABLED, SplitMode::K_SPLIT);
}

void MhcPreSinkhornTilingRegbase::SelectTilingKey(bool isKSplit)
{
    if (isKSplit) {
        if (cvRatio_ == CV_RATIO_ONE) {
            tilingKey_ = needGrad_ ? TILING_KEY_CV_ONE_GRAD_K_SPLIT : TILING_KEY_CV_ONE_NO_GRAD_K_SPLIT;
        } else {
            tilingKey_ = needGrad_ ? TILING_KEY_GRAD_K_SPLIT : TILING_KEY_NO_GRAD_K_SPLIT;
        }
    } else {
        if (cvRatio_ == CV_RATIO_ONE) {
            tilingKey_ = needGrad_ ? TILING_KEY_CV_ONE_GRAD_M_SPLIT : TILING_KEY_CV_ONE_NO_GRAD_M_SPLIT;
        } else {
            tilingKey_ = needGrad_ ? TILING_KEY_GRAD_M_SPLIT : TILING_KEY_NO_GRAD_M_SPLIT;
        }
    }
}

ge::graphStatus MhcPreSinkhornTilingRegbase::CalcOpTiling()
{
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    const uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t originalKDimNum = kDimNum;

    // 如果开启了batch一致性开关，则强制走M分核模板
    int32_t deterministicLevel = context_->GetDeterministicLevel();
    OP_LOGD(context_->GetNodeName(), "ops: MhcPreSinkhorn, deterministic_level=%d", deterministicLevel);
    if (deterministicLevel == BATCH_CONSISTENCY_LEVEL) {
        OP_LOGI(context_->GetNodeName(),
                "MhcPreSinkhorn batch consistency enabled: deterministicLevel=%d, force kDimNum from %lu to 1.",
                deterministicLevel, originalKDimNum);
        kDimNum = 1;
    }

    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
    uint64_t actualKBlockNum = CeilDiv(kSize, splitKSize);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(actualKBlockNum);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    // 开启batch一致性开关时mL1Size固定取M_L1_MAX_SIZE，使kL1Size不随实际bs_切分结果变化，保证在不同batch下切分一致
    uint64_t mSizeForKL1 = (deterministicLevel == BATCH_CONSISTENCY_LEVEL) ? M_L1_MAX_SIZE : tilingData_.get_mL1Size();
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / mSizeForKL1, K_L1_MAX_SIZE) / K_L1_ALIGN_SIZE * K_L1_ALIGN_SIZE);

    const bool isKSplit = kDimNum != 1;
    SelectTilingKey(isKSplit);
    OP_LOGD(context_->GetNodeName(),
            "MhcPreSinkhorn regbase tiling decision: tilingKey=%lu, needGrad=%d, cvRatio=%lu, deterministicLevel=%d, "
            "mDimNum=%lu, singleCoreM=%lu, originalKDimNum=%lu, finalKDimNum=%lu, actualKBlockNum=%lu, "
            "splitKSize=%lu, mL1Size=%ld, kL1Size=%ld",
            tilingKey_, needGrad_ ? 1 : 0, cvRatio_, deterministicLevel, mDimNum, singleCoreM, originalKDimNum, kDimNum,
            actualKBlockNum, splitKSize, tilingData_.get_mL1Size(), tilingData_.get_kL1Size());
    if (isKSplit) {
        return needGrad_ ? CalcMKSplitCorePart2GradoutTiling() : CalcMKSplitCorePart2Tiling();
    }
    return needGrad_ ? CalcRegbaseOpGradoutTiling() : CalcRegbaseOpTiling();
}

ge::graphStatus MhcPreSinkhornTilingRegbase::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::GetWorkspaceSize()
{
    workspaceSize_ = sysWorkspaceSize_;
    if (tilingKey_ == TILING_KEY_NO_GRAD_K_SPLIT || tilingKey_ == TILING_KEY_GRAD_K_SPLIT ||
        tilingKey_ == TILING_KEY_CV_ONE_NO_GRAD_K_SPLIT || tilingKey_ == TILING_KEY_CV_ONE_GRAD_K_SPLIT) {
        // K-split uses two float buffers after the system workspace.
        workspaceSize_ +=
            tilingData_.get_kBlockFactor() * tilingData_.get_bs() * tilingData_.get_hcMix() * sizeof(float) +
            tilingData_.get_kBlockFactor() * tilingData_.get_bs() * sizeof(float);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTilingRegbase::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    // 按实际参与计算的核数启动，避免多余核空转：
    // - M+K 分核模板(1000/2000)：stage1 需要 cubeBlockDimM * cubeBlockDimK 个 AIC 块；
    //    stage2 需要 secondUsedCoreNum 个 AIV，按 MIX 1 AIC:2 AIV 折算成块参与比较。
    uint64_t blockDim = aicCoreNum_;
    if (tilingKey_ == TILING_KEY_NO_GRAD_K_SPLIT || tilingKey_ == TILING_KEY_GRAD_K_SPLIT) {
        uint64_t stage1BlockNum = tilingData_.get_cubeBlockDimM() * tilingData_.get_cubeBlockDimK();
        uint64_t stage2BlockNum = CeilDiv(tilingData_.get_secondUsedCoreNum() + tilingData_.get_sinkhornCoreNum(), 2);
        blockDim = std::min(std::max(stage1BlockNum, stage2BlockNum), aicCoreNum_);
    }
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(BATCH_MODE);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    PrintFinalTilingData(context_, tilingData_, tilingKey_, blockDim, workspaceSize_);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
