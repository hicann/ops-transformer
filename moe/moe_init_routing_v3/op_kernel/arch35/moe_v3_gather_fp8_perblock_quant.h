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
 * \file moe_v3_gather_fp8_perblock_quant.h
 * \brief FP8 PerBlock量化Kernel（BlockSize=128，Scale dtype=FP32）
 */

#ifndef MOE_V3_GATHER_FP8_PERBLOCK_QUANT_H
#define MOE_V3_GATHER_FP8_PERBLOCK_QUANT_H

#include "moe_v3_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t FP8_PERBLOCK_BLOCK_SIZE = 128;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr uint32_t FP32_INF_VALUE = 0x7f800000;
constexpr uint32_t FP32_NAN_VALUE = 0x7fc00000;
constexpr uint16_t FP16_MAX_VALUE_MASK = 0x7fff;

template <typename T, typename U>
__simd_vf__ inline void ComputeVF(__ubuf__ T *xAddr, __ubuf__ float *scaleAddr, __ubuf__ U *yAddr,
                                  uint32_t totalElemNum, uint16_t blockNum, float fp8MaxValue)
{
    using namespace AscendC::MicroAPI;

    uint16_t vfNum = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    static constexpr DivSpecificMode mode = {MaskMergeMode::ZEROING, false};
    static constexpr CastTrait castTrait0 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                             RoundMode::CAST_RINT};
    static constexpr CastTrait castTrait32tofp8 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_RINT};

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1, vreg2, vreg3, maxReg0, maxReg1, yReg1;
        RegTensor<float> scaleReg0, scaleReg1, fp8MaxReg, yReg2, yReg3;
        RegTensor<U> outReg;
        RegTensor<uint32_t> infThresholdReg, nanIntReg;

        MaskReg preg0, yMaskReg, nanMask;
        MaskReg maskAll = CreateMask<uint16_t, MaskPattern::ALL>();

        Duplicate((RegTensor<uint16_t> &)vreg2, FP16_MAX_VALUE_MASK);
        Duplicate(fp8MaxReg, fp8MaxValue);
        Duplicate(infThresholdReg, FP32_INF_VALUE);
        Duplicate(nanIntReg, FP32_NAN_VALUE);

        // 检测边界情况（最后一个 block 可能不足 128 元素）
        uint16_t lastBlockElemNum = (totalElemNum > 0) ? ((totalElemNum - 1) % FP8_PERBLOCK_BLOCK_SIZE + 1) : 0;
        bool lastPairIsPartial = (lastBlockElemNum != FP8_PERBLOCK_BLOCK_SIZE);
        bool hasTail = (blockNum % 2 == 1);

        uint16_t fullPairs = blockNum / 2;
        uint16_t fastFullPairs = fullPairs;
        if (lastPairIsPartial && !hasTail && fullPairs > 0) {
            fastFullPairs = fullPairs - 1;
        }

        // === 主循环（完全无分支，处理所有完整的双 block 对）===
        for (uint16_t pairIdx = 0; pairIdx < fastFullPairs; pairIdx++) {
            uint32_t blockOffset = pairIdx * FP8_PERBLOCK_BLOCK_SIZE * 2;

            // --- block 0 ---
            LoadAlign(vreg0, xAddr + blockOffset);
            Duplicate(maxReg0, static_cast<T>(0));
            And((RegTensor<uint16_t> &)vreg3, (RegTensor<uint16_t> &)vreg0, (RegTensor<uint16_t> &)vreg2, maskAll);
            Max<T>(maxReg0, maxReg0, vreg3, maskAll);
            Reg::Reduce<Reg::ReduceType::MAX, uint16_t>((RegTensor<uint16_t> &)maxReg0, (RegTensor<uint16_t> &)maxReg0,
                                                        maskAll);
            Duplicate(maxReg0, maxReg0, maskAll);
            Cast<float, T, castTrait0>(scaleReg0, maxReg0, maskAll);
            Compare<uint32_t, CMPMODE::GE>(nanMask, (RegTensor<uint32_t> &)scaleReg0, infThresholdReg, maskAll);
            Div<float, &mode>(scaleReg0, scaleReg0, fp8MaxReg, maskAll);
            Select<uint32_t>((RegTensor<uint32_t> &)scaleReg0, nanIntReg, (RegTensor<uint32_t> &)scaleReg0, nanMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + pairIdx * 2, scaleReg0, maskAll);

            // --- block 1 ---
            LoadAlign(vreg1, xAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE);
            Duplicate(maxReg1, static_cast<T>(0));
            And((RegTensor<uint16_t> &)vreg3, (RegTensor<uint16_t> &)vreg1, (RegTensor<uint16_t> &)vreg2, maskAll);
            Max<T>(maxReg1, maxReg1, vreg3, maskAll);
            Reg::Reduce<Reg::ReduceType::MAX, uint16_t>((RegTensor<uint16_t> &)maxReg1, (RegTensor<uint16_t> &)maxReg1,
                                                        maskAll);
            Duplicate(maxReg1, maxReg1, maskAll);
            Cast<float, T, castTrait0>(scaleReg1, maxReg1, maskAll);
            Compare<uint32_t, CMPMODE::GE>(nanMask, (RegTensor<uint32_t> &)scaleReg1, infThresholdReg, maskAll);
            Div<float, &mode>(scaleReg1, scaleReg1, fp8MaxReg, maskAll);
            Select<uint32_t>((RegTensor<uint32_t> &)scaleReg1, nanIntReg, (RegTensor<uint32_t> &)scaleReg1, nanMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + pairIdx * 2 + 1, scaleReg1, maskAll);

            // --- 量化 block 0 ---
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + blockOffset);
            Cast<float, T, castTrait0>(yReg2, yReg1, maskAll);
            Div<float, &mode>(yReg3, yReg2, scaleReg0, maskAll);
            Cast<U, float, castTrait32tofp8>(outReg, yReg3, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset, outReg, maskAll);

            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + blockOffset + vfNum);
            Cast<float, T, castTrait0>(yReg2, yReg1, maskAll);
            Div<float, &mode>(yReg3, yReg2, scaleReg0, maskAll);
            Cast<U, float, castTrait32tofp8>(outReg, yReg3, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset + vfNum, outReg, maskAll);

            // --- 量化 block 1 ---
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE);
            Cast<float, T, castTrait0>(yReg2, yReg1, maskAll);
            Div<float, &mode>(yReg3, yReg2, scaleReg1, maskAll);
            Cast<U, float, castTrait32tofp8>(outReg, yReg3, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE, outReg, maskAll);

            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE + vfNum);
            Cast<float, T, castTrait0>(yReg2, yReg1, maskAll);
            Div<float, &mode>(yReg3, yReg2, scaleReg1, maskAll);
            Cast<U, float, castTrait32tofp8>(outReg, yReg3, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE + vfNum, outReg,
                                                     maskAll);
        }

        // === 处理包含 partial block 的最后一个 pair（在循环外）===
        if (lastPairIsPartial && !hasTail && fullPairs > 0) {
            uint16_t pairIdx = fullPairs - 1;
            uint32_t blockOffset = pairIdx * FP8_PERBLOCK_BLOCK_SIZE * 2;

            // --- block 0（完整）---
            LoadAlign(vreg0, xAddr + blockOffset);
            Duplicate(maxReg0, static_cast<T>(0));
            And((RegTensor<uint16_t> &)vreg3, (RegTensor<uint16_t> &)vreg0, (RegTensor<uint16_t> &)vreg2, maskAll);
            Max<T>(maxReg0, maxReg0, vreg3, maskAll);
            Reg::Reduce<Reg::ReduceType::MAX, uint16_t>((RegTensor<uint16_t> &)maxReg0, (RegTensor<uint16_t> &)maxReg0,
                                                        maskAll);
            Duplicate(maxReg0, maxReg0, maskAll);
            Cast<float, T, castTrait0>(scaleReg0, maxReg0, maskAll);
            Compare<uint32_t, CMPMODE::GE>(nanMask, (RegTensor<uint32_t> &)scaleReg0, infThresholdReg, maskAll);
            Div<float, &mode>(scaleReg0, scaleReg0, fp8MaxReg, maskAll);
            Select<uint32_t>((RegTensor<uint32_t> &)scaleReg0, nanIntReg, (RegTensor<uint32_t> &)scaleReg0, nanMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + pairIdx * 2, scaleReg0, maskAll);

            // --- block 1（partial，精确掩码）---
            LoadAlign(vreg1, xAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE);
            uint32_t partialInputNum = lastBlockElemNum;
            preg0 = UpdateMask<T>(partialInputNum);
            Duplicate(maxReg1, static_cast<T>(0));
            And((RegTensor<uint16_t> &)vreg3, (RegTensor<uint16_t> &)vreg1, (RegTensor<uint16_t> &)vreg2, preg0);
            Max<T, MaskMergeMode::MERGING>(maxReg1, maxReg1, vreg3, preg0);
            Reg::Reduce<Reg::ReduceType::MAX, uint16_t>((RegTensor<uint16_t> &)maxReg1, (RegTensor<uint16_t> &)maxReg1,
                                                        maskAll);
            Duplicate(maxReg1, maxReg1, maskAll);
            Cast<float, T, castTrait0>(scaleReg1, maxReg1, maskAll);
            Compare<uint32_t, CMPMODE::GE>(nanMask, (RegTensor<uint32_t> &)scaleReg1, infThresholdReg, maskAll);
            Div<float, &mode>(scaleReg1, scaleReg1, fp8MaxReg, maskAll);
            Select<uint32_t>((RegTensor<uint32_t> &)scaleReg1, nanIntReg, (RegTensor<uint32_t> &)scaleReg1, nanMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + pairIdx * 2 + 1, scaleReg1, maskAll);

            // --- 量化 block 0（完整）---
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + blockOffset);
            Cast<float, T, castTrait0>(yReg2, yReg1, maskAll);
            Div<float, &mode>(yReg3, yReg2, scaleReg0, maskAll);
            Cast<U, float, castTrait32tofp8>(outReg, yReg3, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset, outReg, maskAll);

            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + blockOffset + vfNum);
            Cast<float, T, castTrait0>(yReg2, yReg1, maskAll);
            Div<float, &mode>(yReg3, yReg2, scaleReg0, maskAll);
            Cast<U, float, castTrait32tofp8>(outReg, yReg3, maskAll);
            StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset + vfNum, outReg, maskAll);

            // --- 量化 block 1（partial，精确掩码）---
            uint32_t partialOutputNum = lastBlockElemNum;
            uint16_t outputVfLoop = (lastBlockElemNum + vfNum - 1) / vfNum;
            for (uint16_t i = 0; i < outputVfLoop; i++) {
                yMaskReg = UpdateMask<float>(partialOutputNum);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1,
                                                        xAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE + i * vfNum);
                Cast<float, T, castTrait0>(yReg2, yReg1, yMaskReg);
                Div<float, &mode>(yReg3, yReg2, scaleReg1, yMaskReg);
                Cast<U, float, castTrait32tofp8>(outReg, yReg3, yMaskReg);
                StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + blockOffset + FP8_PERBLOCK_BLOCK_SIZE + i * vfNum,
                                                         outReg, yMaskReg);
            }
        }

        // === tail 处理（单独的最后一个 block，当 blockNum 为奇数）===
        if (hasTail) {
            uint16_t lastBlockIdx = blockNum - 1;
            uint32_t lastBlockOffset = lastBlockIdx * FP8_PERBLOCK_BLOCK_SIZE;
            uint32_t blockElemNum = totalElemNum - lastBlockOffset;
            uint32_t inputNum = blockElemNum;
            uint32_t outputNum = blockElemNum;

            // 加载并计算 max（按 blockElemNum 限制有效范围）
            Duplicate(maxReg0, static_cast<T>(0));
            preg0 = UpdateMask<T>(inputNum);
            LoadAlign(vreg0, xAddr + lastBlockOffset);
            And((RegTensor<uint16_t> &)vreg3, (RegTensor<uint16_t> &)vreg0, (RegTensor<uint16_t> &)vreg2, preg0);
            Max<T, MaskMergeMode::MERGING>(maxReg0, maxReg0, vreg3, preg0);
            Reg::Reduce<Reg::ReduceType::MAX, uint16_t>((RegTensor<uint16_t> &)maxReg0, (RegTensor<uint16_t> &)maxReg0,
                                                        maskAll);
            Duplicate(maxReg0, maxReg0, maskAll);
            Cast<float, T, castTrait0>(scaleReg0, maxReg0, maskAll);
            Compare<uint32_t, CMPMODE::GE>(nanMask, (RegTensor<uint32_t> &)scaleReg0, infThresholdReg, maskAll);
            Div<float, &mode>(scaleReg0, scaleReg0, fp8MaxReg, maskAll);
            Select<uint32_t>((RegTensor<uint32_t> &)scaleReg0, nanIntReg, (RegTensor<uint32_t> &)scaleReg0, nanMask);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + lastBlockIdx, scaleReg0, maskAll);

            // 量化（按 blockElemNum 限制有效输出范围）
            uint16_t outputVfLoop = (blockElemNum + vfNum - 1) / vfNum;
            for (uint16_t i = 0; i < outputVfLoop; i++) {
                yMaskReg = UpdateMask<float>(outputNum);
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + lastBlockOffset + i * vfNum);
                Cast<float, T, castTrait0>(yReg2, yReg1, yMaskReg);
                Div<float, &mode>(yReg3, yReg2, scaleReg0, yMaskReg);
                Cast<U, float, castTrait32tofp8>(outReg, yReg3, yMaskReg);
                StoreAlign<U, StoreDist::DIST_PACK4_B32>(yAddr + lastBlockOffset + i * vfNum, outReg, yMaskReg);
            }
        }
    }
}

template <typename T, typename U>
class MoeGatherOutFP8PerBlockQuant {
public:
    __aicore__ inline MoeGatherOutFP8PerBlockQuant(){};

    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR sortedExpertIdxAddr,
                                GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR sortedExpertIdxAddr,
                                            const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void InitBuffer();

    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyExpandedXandFP8Quant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(int64_t loopCols, int64_t loopValidScaleCols, int64_t loopScaleCols);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe *pipe_;

    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> sortedRowIdxInQueue_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<float> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;

    int64_t validScaleCols_;
    int64_t scaleCols_;

    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopValidScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    float fp8MaxValue_;
};

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                                GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
                                                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                                const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                                TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ float *)expandedScaleAddr);

    if (rowIdxType_ == SCATTER) {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    } else {
        sortedRowIdxGm_.SetGlobalBuffer(
            (__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
            Align(perCoreRow_, sizeof(int32_t)));
    }

    InitBuffer();

    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fp8MaxValue_ = FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        fp8MaxValue_ = FP8_E5M2_MAX_VALUE;
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::InitKernelTiling(
    GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;

    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, FP8_PERBLOCK_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2);

    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        1);

    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    coreRows_ = perCoreRow_;
    int64_t originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;

    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    }

    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;

    perLoopScaleCols_ = perLoopCols_ / FP8_PERBLOCK_BLOCK_SIZE;
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::InitBuffer()
{
    int64_t inQueueBuffer = AlignBytes(perLoopCols_, sizeof(T));
    int64_t outQueueBuffer = AlignBytes(perLoopCols_, sizeof(U));
    // scale buffer至少分配一个VReg大小，避免Duplicate等向量指令写入时溢出
    int64_t scaleQueueBuffer = Max(static_cast<int64_t>(AlignBytes(perLoopScaleCols_, sizeof(float))),
                                   static_cast<int64_t>(AscendC::VECTOR_REG_WIDTH));

    pipe_->InitBuffer(inQueue_, GATHER_OUT_BUFFER_NUM, inQueueBuffer);
    pipe_->InitBuffer(outQueue_, GATHER_OUT_BUFFER_NUM, outQueueBuffer);
    pipe_->InitBuffer(scaleQueue_, GATHER_OUT_BUFFER_NUM, scaleQueueBuffer);
    pipe_->InitBuffer(sortedRowIdxInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopRows_, sizeof(int32_t)));
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXandFP8Quant(loop);
        }

        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        CopyExpandedXandFP8Quant(rowLoops_ - 1);
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::CopyExpandedXandFP8Quant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    for (int64_t index = 0; index < currentLoopRows_; index++) {
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;

        for (int64_t j = 0; j < colLoops_; j++) {
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            int64_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
            int64_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;

            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopValidScaleCols, loopScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad<T, PaddingMode::Compact>(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyParams,
                                         padParams);

    inQueue_.EnQue(inLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::Compute(int64_t loopCols, int64_t loopValidScaleCols,
                                                                   int64_t loopScaleCols)
{
    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    __ubuf__ T *xAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());

    LocalTensor<U> outLocal = outQueue_.AllocTensor<U>();
    __ubuf__ U *outAddr = reinterpret_cast<__ubuf__ U *>(outLocal.GetPhyAddr());

    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
    __ubuf__ float *scaleAddr = reinterpret_cast<__ubuf__ float *>(scaleLocal.GetPhyAddr());

    Duplicate(scaleLocal, 0.0f, Align(loopScaleCols, sizeof(float)));

    VF_CALL<ComputeVF<T, U>>(xAddr, scaleAddr, outAddr, static_cast<uint32_t>(loopCols),
                             static_cast<uint16_t>(loopValidScaleCols), fp8MaxValue_);

    inQueue_.FreeTensor(xLocal);
    outQueue_.EnQue(outLocal);
    scaleQueue_.EnQue(scaleLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                                   int64_t loopScaleCols)
{
    LocalTensor<uint8_t> outLocal = outQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);

    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], scaleLocal,
                       copyScaleParams);

    outQueue_.FreeTensor(outLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

} // namespace MoeInitRoutingV3

#endif // MOE_V3_GATHER_FP8_PERBLOCK_QUANT_H
