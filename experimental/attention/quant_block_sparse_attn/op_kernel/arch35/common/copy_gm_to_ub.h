/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_gm_to_ub.h
 * \brief
 */
#ifndef COPY_GM_TO_UB_H
#define COPY_GM_TO_UB_H

#include "offset_calculator.h"

// GM->UB
// antiquant
// ----------------------------------------------CopyAntiquantGmToUb--------------------------------
template <typename T, QBSALayout GM_FORMAT>
class CopyAntiquantGmToUb {
public:
    __aicore__ inline void operator()(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                      const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo,
                                      const ConstInfo &constInfo)
    {
        // per per_head + per_token + PA
        ProcessPA(dstTensor, srcTensor, blockTableTensor, runInfo, constInfo);
    }

private:
    __aicore__ inline void ProcessPA(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                     const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo,
                                     const ConstInfo &constInfo)
    {
        int64_t byteOffset1 = 0;
        int64_t byteOffset2 = 0;
        if constexpr (GM_FORMAT == QBSALayout::PA_BNBD) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;

            // 当 runInfo.sparseBlkIdx2 >= 0 的情况时，runInfo.s2SparseBlk1RealSize = constInfo.kvSparseBlockSize
            dataCopyParams.blockLen = runInfo.s2SparseBlk1RealSize * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams{};
            if (runInfo.sparseBlkIdx1 >= 0) {
                GetPaKScaleOffset(byteOffset1, runInfo.sparseBlkIdx1, blockTableTensor, runInfo, constInfo);
                DataCopyPad(dstTensor, srcTensor[byteOffset1], dataCopyParams, dataCopyPadParams);
            }
            if (runInfo.sparseBlkIdx2 >= 0 && runInfo.s2SparseBlk2RealSize > 0) {
                GetPaKScaleOffset(byteOffset2, runInfo.sparseBlkIdx2, blockTableTensor, runInfo, constInfo);
                dataCopyParams.blockLen = runInfo.s2SparseBlk2RealSize * sizeof(float);
                DataCopyPad(dstTensor[constInfo.kvSparseBlockSize], srcTensor[byteOffset2], dataCopyParams,
                            dataCopyPadParams);
            }
        } else if constexpr (GM_FORMAT == QBSALayout::PA_BSND) {
            GetPaKScaleOffset(byteOffset1, runInfo.sparseBlkIdx1, blockTableTensor, runInfo, constInfo);
            GetPaKScaleOffset(byteOffset2, runInfo.sparseBlkIdx2, blockTableTensor, runInfo, constInfo);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 128;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = ((constInfo.n2Size - 1) * constInfo.combineDim) * sizeof(float);
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams{};
            DataCopyPad(dstTensor, srcTensor[byteOffset1], dataCopyParams, dataCopyPadParams);
            DataCopyPad(dstTensor[128], srcTensor[byteOffset2], dataCopyParams, dataCopyPadParams);
        }
    }
    __aicore__ inline void GetPaKScaleOffset(int64_t &byteOffset, const int64_t sparseBlkIdx,
                                             const GlobalTensor<int32_t> &blockTableTensor, const RunInfo &runInfo,
                                             const ConstInfo &constInfo)
    {
        int64_t blockTableOffset = runInfo.boIdx * constInfo.blockTableDim2 + sparseBlkIdx;
        int64_t physicalBlock = blockTableTensor.GetValue(blockTableOffset);
        byteOffset = physicalBlock * (constInfo.paBlockStride / sizeof(float)) +
                     runInfo.n2oIdx * constInfo.kvSparseBlockSize;
    }
};

// ----------------------------------------------CopyQueryScaleGmToUb--------------------------------
template <typename T, QBSALayout GM_FORMAT>
class CopyQueryScaleGmToUb {
public:
    __aicore__ inline void operator()(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                      const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        ProcessGS1(dstTensor, srcTensor, runInfo, constInfo);
    }

private:
    __aicore__ inline void ProcessGS1(LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor,
                                      const RunInfo &runInfo, const ConstInfo &constInfo)
    {
        if constexpr (GM_FORMAT == QBSALayout::TND) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 64;
            dataCopyParams.blockLen = sizeof(float);
            dataCopyParams.srcStride = (constInfo.n1Size - 1) * sizeof(float);
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams{};
            int64_t n1Idx = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
            int64_t qScaleOffset = runInfo.qBScalarOffset + runInfo.sOuterOffset * constInfo.n1Size + n1Idx;
            DataCopyPad(dstTensor, srcTensor[qScaleOffset], dataCopyParams, dataCopyPadParams);
        } else if constexpr (GM_FORMAT == QBSALayout::NTD) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = runInfo.halfS1RealSize * sizeof(float);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> dataCopyPadParams{};
            int64_t qScaleOffset = runInfo.qBScalarOffset + runInfo.sOuterOffset;
            DataCopyPad(dstTensor, srcTensor[qScaleOffset], dataCopyParams, dataCopyPadParams);
        }
    }
};
#endif
