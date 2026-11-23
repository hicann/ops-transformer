/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file block_scheduler_matmul_basic.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor/layout.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_, int64_t FullLoadMode_ = 0, bool IsFp32_ = false, bool IsNdFormat_ = true>
class HcGradBlockSchedulerMatmulBasic {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        uint32_t mL1 = 0;
        uint32_t nL1 = 0;
        uint32_t kL1 = 0;
        uint32_t baseM = 0;
        uint32_t baseN = 0;
        uint32_t baseK = 0;
        uint32_t mTailCnt = 0;
        uint32_t nTailCnt = 0;
        uint32_t mBaseTailSplitCnt = 1;
        uint32_t nBaseTailSplitCnt = 1;
        uint32_t mTailMain = 1;
        uint32_t nTailMain = 1;
        uint8_t isHf32 = 0; // HF32 enable flag.
        uint32_t l2CacheMode = L2_CACHE_DEFAULT;
        uint32_t sliceM = 0;      // M axis for non-contiguous input.
        uint32_t srcNdStride = 1; // M-axis stride for non-contiguous input.
        uint32_t innerBatch = 1;  // Inner-axis batch size for non-contiguous transposed input.
        uint32_t windowLen = 4;
        uint32_t usedCoreNum = 0; // ASW sliding-window length selected by tiling.
    };

public:
    __aicore__ inline HcGradBlockSchedulerMatmulBasic(const ProblemShape& shape, const Params& params)
    {
        k_ = asc::te::get<MNK_K>(shape);
        batch_ = AscendC::Std::max(asc::te::get<MNK_B>(shape), 1L);
        innerBatch_ = params.innerBatch;
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        isHf32_ = params.isHf32;
        int64_t m = asc::te::get<MNK_M>(shape);
        int64_t n = asc::te::get<MNK_N>(shape);
        mBlockNums_ = CeilDiv(m, static_cast<int64_t>(params.mL1));
        nBlockNums_ = CeilDiv(n, static_cast<int64_t>(params.nL1));
        blockNum_ = params.usedCoreNum;
        if (blockNum_ <= 0) {
            return;
        }
        oriBlockIdx_ = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            oriBlockIdx_ /= AscendC::GetTaskRation();
        }
        perCoreBlockNums_ = CeilDiv(mBlockNums_ * nBlockNums_ * batch_, blockNum_);
        blockNums_ = mBlockNums_ * nBlockNums_;
        int64_t tailBlockNums = blockNums_ % blockNum_;
        mL1TailSplitCnt_ = params.mBaseTailSplitCnt;
        nL1TailSplitCnt_ = params.nBaseTailSplitCnt;
        mL1NormCnt_ = mBlockNums_ - mL1TailSplitCnt_;
        nL1NormCnt_ = nBlockNums_ - nL1TailSplitCnt_;
        tailL1M_ = m - mL1NormCnt_ * params.mL1;
        tailL1N_ = n - nL1NormCnt_ * params.nL1;
        mL1TailMain_ = mL1TailSplitCnt_ == 1 ? tailL1M_ : params.mTailMain;
        mL1TailLast_ = tailL1M_ - (mL1TailSplitCnt_ - 1) * mL1TailMain_;
        nL1TailMain_ = nL1TailSplitCnt_ == 1 ? tailL1N_ : params.nTailMain;
        nL1TailLast_ = tailL1N_ - (nL1TailSplitCnt_ - 1) * nL1TailMain_;
        sliceM_ = params.sliceM;
        srcNdStride_ = params.srcNdStride;
        isSlice_ = srcNdStride_ != 1 && sliceM_ != 0;

        if (batch_ == 1) {
            mTailCnt_ = params.mTailCnt;
            nTailCnt_ = params.nTailCnt;
            int64_t mTailSplit = CeilDiv(mL1TailLast_, mTailCnt_);
            int64_t nTailSplit = CeilDiv(nL1TailLast_, nTailCnt_);
            mTailCnt_ = CeilDiv(mL1TailLast_, mTailSplit);
            nTailCnt_ = CeilDiv(nL1TailLast_, nTailSplit);
            tailCnt_ = mTailCnt_ * nTailCnt_;
            blockNums_ += (tailCnt_ - 1) * tailBlockNums;
        }
        // Use the default window length of 4 when tiling supplies zero to avoid division by zero.
        uint32_t windowLen = params.windowLen == 0 ? 4 : params.windowLen;
        mainWindow_ = windowLen < mBlockNums_ ? windowLen : mBlockNums_;
        mainRow_ = mBlockNums_ / mainWindow_ - 1;
        tailWindow_ = mBlockNums_ - mainRow_ * mainWindow_;
    }

    /**
       Return the total tile count.
    */
    __aicore__ inline int64_t GetBlockNums()
    {
        return blockNums_ * batch_;
    }

    /**
       Return the required core count.
    */
    __aicore__ inline int64_t GetCoreNums()
    {
        int64_t tilingBlockNum = 0;
        if (blockNums_ * batch_ < blockNum_) {
            tilingBlockNum = blockNums_ * batch_;
        } else {
            tilingBlockNum = blockNum_;
        }
        return tilingBlockNum;
    }

    template <bool TransB_ = false, class BType_>
    __aicore__ inline BlockShape GetBlockShape(int64_t blockIdx)
    {
        UpdateMNBlockIdx(blockIdx);
        int64_t blkM = mL1_;
        int64_t blkN = nL1_;
        int64_t nAlignSize;
        if constexpr (TransB_) {
            nAlignSize = BLOCK_SIZE_16;
        } else {
            nAlignSize = BLOCK_SIZE_32 / sizeof(BType_);
        }
        if (nBlockIdx_ >= nL1NormCnt_) {
            blkN = nBlockIdx_ == (nBlockNums_ - 1) ? nL1TailLast_ : nL1TailMain_;
        }
        if (mBlockIdx_ >= mL1NormCnt_) {
            blkM = mBlockIdx_ == (mBlockNums_ - 1) ? mL1TailLast_ : mL1TailMain_;
        }

        if (blockIdx / blockNum_ != (perCoreBlockNums_ - 1) || tailCnt_ == 1) {
            // mL1, nL1, k, batch
            return {blkM, blkN, k_, batch_};
        }
        // SplitM and SplitN
        int64_t splitBlkM = CeilDiv(blkM, mTailCnt_);
        int64_t splitBlkN = CeilDiv(blkN, nTailCnt_);
        if (isSlice_) {
            splitBlkM = CeilAlign(splitBlkM, sliceM_);
        }
        if constexpr (!IsNdFormat_) {
            splitBlkN = CeilAlign(splitBlkN, nAlignSize);
            nTailCnt_ = CeilDiv(blkN, splitBlkN);
        }
        // must divide origin blockIdx
        int64_t mSplitIdx = (oriBlockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (oriBlockIdx_ % tailCnt_) / mTailCnt_;
        mSplitOffset_ = mSplitIdx * splitBlkM;
        nSplitOffset_ = nSplitIdx * splitBlkN;
        if (mSplitOffset_ >= blkM || nSplitOffset_ >= blkN) {
            return {0, 0, k_, batch_};
        }
        splitBlkM = AscendC::Std::min(blkM - mSplitOffset_, splitBlkM);
        splitBlkN = AscendC::Std::min(blkN - nSplitOffset_, splitBlkN);

        return {splitBlkM, splitBlkN, k_, batch_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx)
    {
        UpdateMNBlockIdx(blockIdx);
        int64_t batchIdx = 0;
        if (batch_ > 1) {
            batchIdx = blockIdx / blockNums_;
        }

        int64_t mOffset = mBlockIdx_ * mL1_ + mSplitOffset_;
        int64_t nOffset = nBlockIdx_ * nL1_ + nSplitOffset_;
        int64_t kOffset = 0; // This path keeps K unsplit.

        if (mBlockIdx_ > mL1NormCnt_) {
            mOffset = mL1NormCnt_ * mL1_ + (mBlockIdx_ - mL1NormCnt_) * mL1TailMain_ + mSplitOffset_;
        }
        if (nBlockIdx_ > nL1NormCnt_) {
            nOffset = nL1NormCnt_ * nL1_ + (nBlockIdx_ - nL1NormCnt_) * nL1TailMain_ + nSplitOffset_;
        }

        return {mOffset, nOffset, kOffset, batchIdx};
    }

private:
    __aicore__ inline void UpdateMNBlockIdx(int64_t tmpBlockIdx)
    {
        if (lastBlockIdx_ == tmpBlockIdx) {
            return;
        }
        lastBlockIdx_ = tmpBlockIdx;

        int64_t blockIdx = tmpBlockIdx % blockNums_;
        if (blockIdx / blockNum_ == (perCoreBlockNums_ - 1) && tailCnt_ > 1) {
            blockIdx = (perCoreBlockNums_ - 1) * blockNum_ + oriBlockIdx_ / tailCnt_;
        }
        int64_t rowIdx = blockIdx / nBlockNums_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mBlockIdx_ = rowIdx * mainWindow_ + blockIdx % mainWindow_;
            nBlockIdx_ = (blockIdx / mainWindow_) % nBlockNums_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = blockIdx - mainRow_ * mainWindow_ * nBlockNums_;
            mBlockIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nBlockIdx_ = (tailIndex / tailWindow_) % nBlockNums_;
        }
        if (rowIdx % 2 != 0) { // 2: mode 2 means even row, need reverse scan
            nBlockIdx_ = nBlockNums_ - 1 - nBlockIdx_;
        }
    }

private:
    static constexpr uint64_t BLOCK_SIZE_16 = 16UL;
    static constexpr uint64_t BLOCK_SIZE_32 = 32UL;
    static constexpr int64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr bool IS_ND_FORMAT = IsNdFormat_;
    static constexpr bool IS_FP32 = IsFp32_;

    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t oriBlockIdx_{0};
    int64_t perCoreBlockNums_{0};
    int64_t blockNum_{0};
    int64_t batch_{0};
    int64_t innerBatch_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1};
    int64_t blockNums_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mBlockIdx_{1};
    int64_t nBlockIdx_{1};
    int64_t lastBlockIdx_{-1};
    int64_t nSplitOffset_{0};
    int64_t mSplitOffset_{0};
    bool isSlice_{false};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};
    uint8_t isHf32_{0};
    int64_t sliceM_{1};
    int64_t srcNdStride_{1};
    int64_t mL1NormCnt_{0};
    int64_t mL1TailSplitCnt_{1};
    int64_t mL1TailMain_{0};
    int64_t mL1TailLast_{0};
    int64_t nL1NormCnt_{0};
    int64_t nL1TailSplitCnt_{1};
    int64_t nL1TailMain_{0};
    int64_t nL1TailLast_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
