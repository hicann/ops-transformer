/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_grouped_matmul.h
 * \brief
 */

#ifndef MC2_QUANT_GROUPED_MATMUL_H
#define MC2_QUANT_GROUPED_MATMUL_H

#include "../a2av_gmm_utils.h"
#include "../common/a2av_common_tiling.h"
#include "kernel_operator.h"

#include "../../../../3rd/grouped_matmul/op_kernel/mc2_3rd_gqmm_cube_on_the_fly.h"

using namespace AscendC;

namespace MC2KernelTemplate {
constexpr uint64_t GROUP_LIST_INDEX = 0;

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
class QuantGroupedMatmul {
public:
    __aicore__ inline QuantGroupedMatmul() {}
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
                                GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                const GmmTilingDataType *gmmTilingData, TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag = false);
    __aicore__ inline void Process(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void Process(uint32_t expertIdx);
    __aicore__ inline void End();

protected:
    __aicore__ inline void UpdateAddr(uint32_t expertIdx, uint32_t expertTokenNum);
    __aicore__ inline GM_ADDR BuildPtrTable(GM_ADDR dataAddr, uint32_t slotIdx);

private:
    using biasType = float;

    GM_ADDR xGM_;
    GM_ADDR wGM_;
    GM_ADDR xScaleGM_;
    GM_ADDR weightScaleGM_;
    GM_ADDR yGM_;
    GM_ADDR groupListGm_;
    GM_ADDR cGroupOffsetTableGm_ = nullptr;
    GM_ADDR aGroupOffsetTableGm_ = nullptr;
    GM_ADDR xScaleGroupOffsetTableGm_ = nullptr;
    GM_ADDR workspaceGM_;
    GM_ADDR ptrTableBase_ = nullptr;
    GlobalTensor<xType> xGlobalBuffer_;
    GlobalTensor<wType> wGlobalBuffer_;
    GlobalTensor<scaleType> xScaleGlobalBuffer_;
    GlobalTensor<scaleType> wScaleGlobalBuffer_;
    GlobalTensor<yType> yGlobalBuffer_;
    GlobalTensor<int64_t> groupListGlobalBuffer_;
    const TilingDataType *tilingData_;
    TPipe *tPipe_;
    uint64_t expertTokenNum_[MAX_EXPERT_PER_RANK] = {0};
    uint64_t expertTokenOffset_ = 0;
    uint64_t expertNumInOneRank_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t h1_ = 0;
    uint64_t n1_ = 0;
    uint64_t bs_ = 0;
    uint64_t a_ = 0;
    const GmmTilingDataType *gmmTilingData_;
    TILING_TYPE *gmmArrayAddrIn_;
    GM_ADDR ttXScaleRepeatGm_ = nullptr;
    GM_ADDR ttWeightScaleRepeatGm_ = nullptr;
};

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void
QuantGroupedMatmul<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat, aTrans, bTrans, isLocal,
                   isA2avGmm>::Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
                                    GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                    const GmmTilingDataType *gmmTilingData, TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe,
                                    bool isA2avGmmFlag)
{
    if ASCEND_IS_AIV {
        return;
    }
    xGM_ = xGM;
    wGM_ = weightGM;
    xScaleGM_ = xScaleGM;
    weightScaleGM_ = weightScaleGM;
    yGM_ = yGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    workspaceGM_ = workspaceGM;
    gmmTilingData_ = gmmTilingData;
    gmmArrayAddrIn_ = gmmArrayAddrIn;

    expertNumInOneRank_ = tilingData_->taskTilingInfo.e;
    epWorldSize_ = tilingData_->taskTilingInfo.epWorldSize;
    h1_ = tilingData_->taskTilingInfo.H1;
    n1_ = tilingData_->taskTilingInfo.N1;
    bs_ = tilingData_->taskTilingInfo.BS;
    a_ = tilingData_->taskTilingInfo.A;

    if constexpr (!isLocal && !isA2avGmm) {
        uint64_t groupListSize = sizeof(int64_t) * expertNumInOneRank_ * epWorldSize_;
        uint64_t cGroupOffsetTableSize = sizeof(uint64_t) * expertNumInOneRank_ * epWorldSize_;
        groupListGm_ = workspaceGM_;
        cGroupOffsetTableGm_ = groupListGm_ + groupListSize;
        ptrTableBase_ = cGroupOffsetTableGm_ + cGroupOffsetTableSize;
    } else if constexpr (!isLocal && isA2avGmm) {
        uint64_t groupListSize = sizeof(int64_t) * expertNumInOneRank_ * epWorldSize_;
        uint64_t aGroupOffsetTableSize = sizeof(uint64_t) * expertNumInOneRank_ * epWorldSize_;
        uint64_t xScaleOffsetTableSize = sizeof(uint64_t) * expertNumInOneRank_ * epWorldSize_;
        groupListGm_ = workspaceGM_;
        aGroupOffsetTableGm_ = groupListGm_ + groupListSize;
        xScaleGroupOffsetTableGm_ = aGroupOffsetTableGm_ + aGroupOffsetTableSize;
        ptrTableBase_ = xScaleGroupOffsetTableGm_ + xScaleOffsetTableSize;
    } else {
        uint64_t groupListSize = sizeof(int64_t) * expertNumInOneRank_;
        groupListGm_ = workspaceGM_;
        ptrTableBase_ = groupListGm_ + groupListSize;
    }
    xGlobalBuffer_.SetGlobalBuffer((__gm__ xType *)this->xGM_);
    wGlobalBuffer_.SetGlobalBuffer((__gm__ wType *)this->wGM_);
    yGlobalBuffer_.SetGlobalBuffer((__gm__ yType *)this->yGM_);
    groupListGlobalBuffer_.SetGlobalBuffer((__gm__ int64_t *)groupListGm_);
    if constexpr (MX_QUANT_MODE) {
        xScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)xScaleGM);
        wScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)weightScaleGM);
    }
    if constexpr (PERTENSOR_QUANT_MODE) {
        uint32_t expertNumMax = tilingData_->taskTilingInfo.expertNum;
        ttWeightScaleRepeatGm_ = ptrTableBase_ + TENSOR_LIST_SIZE;
        ttXScaleRepeatGm_ = ttWeightScaleRepeatGm_ + sizeof(float) * expertNumMax;
    }

    const auto *opCnt =
        isA2avGmmFlag ? &tilingData_->taskTilingInfo.recvCnt[0] : &tilingData_->taskTilingInfo.sendCnt[0];
    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            expertTokenNum_[e] += static_cast<uint64_t>(opCnt[e + i * expertNumInOneRank_]);
        }
    }
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmul<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                          aTrans, bTrans, isLocal, isA2avGmm>::Process(uint32_t expertIdx)
{
    Process(expertIdx, 1);
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmul<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                          aTrans, bTrans, isLocal, isA2avGmm>::Process(uint32_t startExpertIdx,
                                                                                       uint32_t expertNum)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint64_t expertTokenNum = expertTokenNum_[startExpertIdx];
    for (int expertIdx = startExpertIdx + 1; expertIdx < startExpertIdx + expertNum; expertIdx++) {
        expertTokenNum += expertTokenNum_[expertIdx];
    }
    if (!isLocal && expertTokenNum == 0) {
        return;
    }

    __gm__ int64_t *groupListPtr = reinterpret_cast<__gm__ int64_t *>(groupListGm_);
    if constexpr (!isLocal && !isA2avGmm) {
        const auto *opCnt = &tilingData_->taskTilingInfo.sendCnt[0];
        __gm__ uint64_t *cGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(cGroupOffsetTableGm_);
        uint64_t rankTotalInBatch[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            for (uint32_t e = 0; e < expertNum; e++) {
                uint32_t absExpertIdx = startExpertIdx + e;
                uint64_t cnt = static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]);
                groupListPtr[GROUP_LIST_INDEX + e * epWorldSize_ + r] = static_cast<int64_t>(cnt);
                rankTotalInBatch[r] += cnt;
            }
        }
        uint64_t rankBaseElemOffset[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 1; r < epWorldSize_; r++) {
            rankBaseElemOffset[r] = rankBaseElemOffset[r - 1] + rankTotalInBatch[r - 1] * n1_;
        }
        uint64_t expertRunningElemOffset[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            expertRunningElemOffset[r] = rankBaseElemOffset[r];
        }
        for (uint32_t e = 0; e < expertNum; e++) {
            for (uint32_t r = 0; r < epWorldSize_; r++) {
                uint32_t groupIdx = e * epWorldSize_ + r;
                cGroupOffsetTable[groupIdx] = expertRunningElemOffset[r];
                uint32_t absExpertIdx = startExpertIdx + e;
                expertRunningElemOffset[r] +=
                    static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]) * n1_;
            }
        }
    } else if constexpr (!isLocal && isA2avGmm) {
        const auto *opCnt = &tilingData_->taskTilingInfo.recvCnt[0];
        __gm__ uint64_t *aGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(aGroupOffsetTableGm_);
        for (uint32_t e = 0; e < expertNum; e++) {
            for (uint32_t r = 0; r < epWorldSize_; r++) {
                uint32_t absExpertIdx = startExpertIdx + e;
                uint64_t cnt = static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]);
                groupListPtr[GROUP_LIST_INDEX + e * epWorldSize_ + r] = static_cast<int64_t>(cnt);
            }
        }
        uint64_t batchBaseOffset = expertTokenOffset_ * h1_ / PACK_FACTOR;
        uint64_t currentBatchRankSize[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            for (uint32_t e = 0; e < expertNum; e++) {
                uint32_t absExpertIdx = startExpertIdx + e;
                currentBatchRankSize[r] +=
                    static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]) * h1_ / PACK_FACTOR;
            }
        }
        uint64_t rankStartBase[MAX_EP_RANK_SIZE] = {0};
        rankStartBase[0] = batchBaseOffset;
        for (uint32_t r = 1; r < epWorldSize_; r++) {
            rankStartBase[r] = rankStartBase[r - 1] + currentBatchRankSize[r - 1];
        }
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            uint64_t aRunningElemOffset = rankStartBase[r];
            for (uint32_t e = 0; e < expertNum; e++) {
                uint32_t absExpertIdx = startExpertIdx + e;
                uint32_t groupIdx = e * epWorldSize_ + r;
                aGroupOffsetTable[groupIdx] = aRunningElemOffset;
                aRunningElemOffset +=
                    static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]) * h1_ / PACK_FACTOR;
            }
        }
        if constexpr (MX_QUANT_MODE) {
            __gm__ uint64_t *xScaleOffsetTable = reinterpret_cast<__gm__ uint64_t *>(xScaleGroupOffsetTableGm_);
            uint64_t scaleK = Mc2QuantUtils::MXFP_MULTI_BASE_SIZE *
                              Mc2QuantUtils::CeilDiv(h1_, static_cast<uint64_t>(Mc2QuantUtils::MXFP_DIVISOR_SIZE));
            uint64_t batchScaleBaseOffset = expertTokenOffset_ * scaleK;
            uint64_t currentBatchRankScaleSize[MAX_EP_RANK_SIZE] = {0};
            for (uint32_t r = 0; r < epWorldSize_; r++) {
                for (uint32_t e = 0; e < expertNum; e++) {
                    uint32_t absExpertIdx = startExpertIdx + e;
                    currentBatchRankScaleSize[r] +=
                        static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]) * scaleK;
                }
            }
            uint64_t rankScaleStartBase[MAX_EP_RANK_SIZE] = {0};
            rankScaleStartBase[0] = batchScaleBaseOffset;
            for (uint32_t r = 1; r < epWorldSize_; r++) {
                rankScaleStartBase[r] = rankScaleStartBase[r - 1] + currentBatchRankScaleSize[r - 1];
            }
            for (uint32_t r = 0; r < epWorldSize_; r++) {
                uint64_t scaleRunningOffset = rankScaleStartBase[r];
                for (uint32_t e = 0; e < expertNum; e++) {
                    uint32_t absExpertIdx = startExpertIdx + e;
                    uint32_t groupIdx = e * epWorldSize_ + r;
                    xScaleOffsetTable[groupIdx] = scaleRunningOffset;
                    scaleRunningOffset += static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]) * scaleK;
                }
            }
        }
    } else if (!isLocal) {
        for (uint32_t idx = 0; idx < expertNum; idx++) {
            groupListPtr[GROUP_LIST_INDEX + idx] = static_cast<int64_t>(expertTokenNum_[startExpertIdx + idx]);
        }
    } else {
        groupListPtr[GROUP_LIST_INDEX] = static_cast<int64_t>(bs_);
    }

    if constexpr (PERTENSOR_QUANT_MODE) {
        float xScaleValue = *((__gm__ float *)xScaleGM_);
        float wScaleValue = *((__gm__ float *)weightScaleGM_);
        __gm__ float *ttXScalePtr = reinterpret_cast<__gm__ float *>(ttXScaleRepeatGm_);
        __gm__ float *ttWeightScalePtr = reinterpret_cast<__gm__ float *>(ttWeightScaleRepeatGm_);
        for (uint32_t i = 0; i < expertNum; i++) {
            ttXScalePtr[i] = xScaleValue;
            ttWeightScalePtr[i] = wScaleValue;
        }
    }

    this->UpdateAddr(startExpertIdx, expertTokenNum);
    __gm__ uint8_t *xAddr = reinterpret_cast<__gm__ uint8_t *>(xGM_);
    __gm__ uint8_t *yAddr = reinterpret_cast<__gm__ uint8_t *>(yGM_);
    __gm__ uint8_t *wAddr = reinterpret_cast<__gm__ uint8_t *>(wGM_);
    GM_ADDR xPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(xAddr), 0);
    GM_ADDR wPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(wAddr), 1);
    GM_ADDR scaleBPtr;
    GM_ADDR xScalePtr;
    if constexpr (PERTENSOR_QUANT_MODE) {
        scaleBPtr = BuildPtrTable(ttWeightScaleRepeatGm_, 2);
        xScalePtr = ttXScaleRepeatGm_;
    } else if constexpr (MX_QUANT_MODE) {
        scaleBPtr = BuildPtrTable(weightScaleGM_, 2);
        xScalePtr = xScaleGM_;
    } else {
        scaleBPtr = nullptr;
        xScalePtr = nullptr;
    }
    GM_ADDR yPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(yAddr), 3);

    Mc2GroupedMatmulTilingData::GMMQuantParams localQuantParams = gmmTilingData_->gmmQuantParams;
    if constexpr (!isLocal && !isA2avGmm) {
        localQuantParams.groupNum = expertNum * static_cast<uint32_t>(epWorldSize_);
    } else if constexpr (!isLocal && isA2avGmm) {
        localQuantParams.groupNum = expertNum * static_cast<uint32_t>(epWorldSize_);
    } else {
        localQuantParams.groupNum = isLocal ? 1 : expertNum;
    }
    Mc2GroupedMatmul::Mc2GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmASWKernel;
    tPipe_->Reset();
    if constexpr (!isLocal && !isA2avGmm) {
        const __gm__ uint64_t *cGroupOffsetTablePtr = reinterpret_cast<const __gm__ uint64_t *>(cGroupOffsetTableGm_);
        gmmASWKernel.Init(xPtr, wPtr, nullptr, scaleBPtr, groupListGm_, xScalePtr, yPtr, workspaceGM_,
                          &localQuantParams, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_,
                          static_cast<uint32_t>(epWorldSize_), cGroupOffsetTablePtr);
    } else if constexpr (!isLocal && isA2avGmm) {
        const __gm__ uint64_t *aGroupOffsetTablePtr = reinterpret_cast<const __gm__ uint64_t *>(aGroupOffsetTableGm_);
        const __gm__ uint64_t *xScaleOffsetTablePtr = nullptr;
        if constexpr (MX_QUANT_MODE) {
            xScaleOffsetTablePtr = reinterpret_cast<const __gm__ uint64_t *>(xScaleGroupOffsetTableGm_);
        }
        gmmASWKernel.Init(xPtr, wPtr, nullptr, scaleBPtr, groupListGm_, xScalePtr, yPtr, workspaceGM_,
                          &localQuantParams, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_,
                          static_cast<uint32_t>(epWorldSize_), nullptr, aGroupOffsetTablePtr, xScaleOffsetTablePtr);
    } else {
        gmmASWKernel.Init(xPtr, wPtr, nullptr, scaleBPtr, groupListGm_, xScalePtr, yPtr, workspaceGM_,
                          &localQuantParams, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_);
    }
    gmmASWKernel.Process();
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmul<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                          aTrans, bTrans, isLocal, isA2avGmm>::End()
{
    if ASCEND_IS_AIV {
        return;
    }
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmul<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                          aTrans, bTrans, isLocal, isA2avGmm>::UpdateAddr(uint32_t startExpertIdx,
                                                                                          uint32_t expertTokenNum)
{
    if constexpr (!isLocal && isA2avGmm) {
        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr();
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(startExpertIdx * h1_ * n1_);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * n1_);
    } else {
        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * h1_);
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(startExpertIdx * h1_ * n1_);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * n1_);
    }

    if constexpr (MX_QUANT_MODE) {
        uint64_t scaleK = Mc2QuantUtils::MXFP_MULTI_BASE_SIZE *
                          Mc2QuantUtils::CeilDiv(h1_, static_cast<uint64_t>(Mc2QuantUtils::MXFP_DIVISOR_SIZE));
        if constexpr (!isLocal && isA2avGmm) {
            xScaleGM_ = (GM_ADDR)xScaleGlobalBuffer_.GetPhyAddr();
        } else {
            xScaleGM_ = (GM_ADDR)xScaleGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * scaleK);
        }
        weightScaleGM_ = (GM_ADDR)wScaleGlobalBuffer_.GetPhyAddr(startExpertIdx * n1_ * scaleK);
    }

    expertTokenOffset_ += expertTokenNum;
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline GM_ADDR QuantGroupedMatmul<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                             aTrans, bTrans, isLocal, isA2avGmm>::BuildPtrTable(GM_ADDR dataAddr,
                                                                                                uint32_t slotIdx)
{
    __gm__ uint64_t *slot =
        reinterpret_cast<__gm__ uint64_t *>(reinterpret_cast<__gm__ uint8_t *>(ptrTableBase_) + slotIdx * 16);
    slot[0] = sizeof(uint64_t);
    slot[1] = reinterpret_cast<uint64_t>(dataAddr);
    return reinterpret_cast<GM_ADDR>(slot);
}

} // namespace MC2KernelTemplate
#endif
// MC2_QUANT_GROUPED_MATMUL_H
