/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_KERNEL_H
#define ADD_RMS_NORM_KERNEL_H
#include "mararn_rms_norm_base.h"
#include "add_rms_norm.h"
#include "add_rms_norm_split_d.h"
#include "add_rms_norm_multi_n.h"
#include "add_rms_norm_merge_n.h"
#include "add_rms_norm_single_n.h"

constexpr uint32_t ADD_RMS_NORM = 10;
constexpr uint32_t ADD_RMS_NORM_BF16 = 30;
constexpr uint32_t ADD_RMS_NORM_SPLIT_D = 11;
constexpr uint32_t ADD_RMS_NORM_SPLIT_BF16 = 31;
constexpr uint32_t ADD_RMS_NORM_MERGE_N = 12;
constexpr uint32_t ADD_RMS_NORM_MERGE_N_BF16 = 32;
constexpr uint32_t ADD_RMS_NORM_SINGLE_N = 13;
constexpr uint32_t ADD_RMS_NORM_SINGLE_N_BF16 = 33;
constexpr uint32_t ADD_RMS_NORM_MULTI_N = 14;

using namespace AscendC;
class AddRmsNormKernel {
public:
    __aicore__ inline AddRmsNormKernel(ArnGmAddrs *arnAddrs, TPipe *tPipe, uint32_t dataSize, uint32_t tileCnt,
                                       uint32_t tailCnt, Hccl<HCCL_SERVER_TYPE_AICPU> *hccl,
                                       AscendC::HcclHandle tileHandleId, AscendC::HcclHandle tailHandleId)
        : arnAddrs_(arnAddrs),
          tPipe_(tPipe),
          dataSize_(dataSize),
          tileCnt_(tileCnt),
          tailCnt_(tailCnt),
          hccl_(hccl),
          tileHandleId_(tileHandleId),
          tailHandleId_(tailHandleId)
    {}

    __aicore__ inline void ComputeAddRmsNorm(Mc2Tiling::AddRMSNormTilingData &addRMSNormTileTilingData,
                                             Mc2Tiling::AddRMSNormTilingData &addRMSNormTailTilingData,
                                             Mc2Tiling::AddRMSNormTilingeKeyData &addRmsNormTilingeKeyData,
                                             GM_ADDR rcvCntGM)
    {
        uint32_t lastCnt = 0;
        uint32_t addRmsNormCount = 1;
        GlobalTensor<int64_t> rcvCntGMTensor;
        rcvCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rcvCntGM));

        while (true) {
            if (GetBlockIdx() == 0) {
                const uint32_t curCnt = GetCurFinishedCnt();
                if (curCnt <= lastCnt) {
                    continue;
                }
                *rcvCntGM = curCnt;
                rcvCntGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rcvCntGM));
                DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                         AscendC::DcciDst::CACHELINE_OUT>(rcvCntGMTensor);
            }
            SyncAll();
            DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                rcvCntGMTensor);
            lastCnt = *rcvCntGM;
            if (lastCnt >= addRmsNormCount) {
                uint32_t rcvCntTile = lastCnt;
                if (tailCnt_ != 0U && lastCnt > tileCnt_) {
                    rcvCntTile = tileCnt_;
                }
                if (tailCnt_ == 0U || (tailCnt_ != 0U && addRmsNormCount <= tileCnt_)) {
                    ComputeAddRmsNormInner(addRMSNormTileTilingData, addRmsNormTilingeKeyData.ARNKeyTile,
                                           addRmsNormTilingeKeyData.ARNNumBlocksTile, rcvCntTile, addRmsNormCount);
                }

                if (tailCnt_ != 0U && addRmsNormCount > tileCnt_ && addRmsNormCount <= lastCnt) {
                    ComputeAddRmsNormInner(addRMSNormTailTilingData, addRmsNormTilingeKeyData.ARNKeyTail,
                                           addRmsNormTilingeKeyData.ARNNumBlocksTail, lastCnt, addRmsNormCount);
                }
            }
            if (lastCnt >= (tileCnt_ + tailCnt_)) {
                break;
            }
        }
    }

private:
#define INVOKE_ARN_OP_IMPL(templateClass, dType) \
    do { \
        templateClass<dType> op; \
        op.Init(arnAddrs_->gammaGM, rmsTilingData, tPipe_, numBlocks); \
        op.ComputeProcess(arnAddrs_->normOutGM, arnAddrs_->residualGM, arnAddrs_->yGM, rmsTilingData, addRmsNormCount, \
                          rcvCnt); \
    } while (0)

    __aicore__ inline void AddRmsNorm(Mc2Tiling::AddRMSNormTilingData &rmsTilingData, uint32_t keyTile,
                                      uint32_t numBlocks, uint32_t rcvCnt, uint32_t addRmsNormCount)
    {
        if (GetBlockIdx() >= numBlocks) {
            return;
        }

        tPipe_->Reset();
        if (keyTile == ADD_RMS_NORM) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNorm, half);
        } else if (keyTile == ADD_RMS_NORM_BF16) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNorm, bfloat16_t);
        } else if (keyTile == ADD_RMS_NORM_SPLIT_D) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSplitD, half);
        } else if (keyTile == ADD_RMS_NORM_SPLIT_BF16) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t);
        } else if (keyTile == ADD_RMS_NORM_MERGE_N) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormMergeN, half);
        } else if (keyTile == ADD_RMS_NORM_MERGE_N_BF16) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t);
        } else if (keyTile == ADD_RMS_NORM_SINGLE_N) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSingleN, half);
        } else if (keyTile == ADD_RMS_NORM_SINGLE_N_BF16) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t);
        } else if (keyTile == ADD_RMS_NORM_MULTI_N) {
            INVOKE_ARN_OP_IMPL(KernelAddRmsNormMultiN, half);
        }
    }

    __aicore__ inline void ComputeAddRmsNormInner(Mc2Tiling::AddRMSNormTilingData &addRMSNormTilingData,
                                                  uint32_t ARNKey, uint32_t ARNNumBlocks, uint32_t rcvCnt,
                                                  uint32_t &addRmsNormCount)
    {
        uint64_t offset = CalcShapeOffset(dataSize_, addRMSNormTilingData.num_row, addRMSNormTilingData.num_col);
        uint32_t cnt = rcvCnt - addRmsNormCount + 1;

        AddRmsNorm(addRMSNormTilingData, ARNKey, ARNNumBlocks, rcvCnt, addRmsNormCount);

        arnAddrs_->normOutGM += offset * cnt;
        arnAddrs_->yGM += offset * cnt;
        arnAddrs_->residualGM += offset * cnt;
        addRmsNormCount += cnt;
    }

    __aicore__ inline uint32_t GetCurFinishedCnt()
    {
        const int32_t tileCnt = hccl_->Query(tileHandleId_);
        if (tailCnt_ == 0U) {
            return tileCnt;
        }
        return tileCnt + hccl_->Query(tailHandleId_);
    }

    ArnGmAddrs *arnAddrs_;
    TPipe *tPipe_;
    uint32_t dataSize_;
    uint32_t tileCnt_;
    uint32_t tailCnt_;
    Hccl<HCCL_SERVER_TYPE_AICPU> *hccl_;
    AscendC::HcclHandle tileHandleId_, tailHandleId_;
};
#endif
