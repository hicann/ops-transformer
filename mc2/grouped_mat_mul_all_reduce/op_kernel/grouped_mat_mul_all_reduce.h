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
 * \file grouped_mat_mul_all_reduce.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MAT_MUL_ALL_REDUCE_H
#define ASCENDC_GROUPED_MAT_MUL_ALL_REDUCE_H

#ifdef __CCE_KT_TEST__
#include "../tests/ut/op_kernel/rac_server_stub.h"
#else
#include "rac_server.h"
#endif
#include "grouped_mat_mul_all_reduce_utils.h"

namespace GROUPED_MAT_MUL_ALL_REDUCE {

/*@brief store variables for core split configuration
 */
constexpr uint32_t MAX_TURN_NUM = 16;

struct MNConfig {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t mIdx;
    uint32_t nIdx;
    uint32_t numBlocksM;
    uint32_t numBlocksN;
    uint32_t singleM;
    uint32_t singleN;
    uint32_t splitM;
    uint32_t tileCnt;
};

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType>
class GMMProcess {
private:
    ComputeType &computeOp; // inernal computation operator
    const GMMAllReduceCoreTiling *__restrict tilingData;
    TPipe *pipe;
    uint32_t blockIdx;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t groupNum;
    uint32_t coreNum;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t ubBaseM;
    uint32_t ubBaseN;

public:
    /** @brief constructor */
    __aicore__ inline GMMProcess(ComputeType &computeOp_)
        : computeOp(computeOp_)
    {}

    __aicore__ inline void Init(const GMMAllReduceCoreTiling *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process(HcclServer &hcclServer);
};

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Init(const GMMAllReduceCoreTiling *__restrict tiling, TPipe *tPipe)
{
    blockIdx = GetBlockIdx();
    subBlockIdx = GetSubBlockIdx();
    coreIdx = blockIdx; // / GetTaskRation();
    pipe = tPipe;
    tilingData = tiling;
    groupNum = tilingData->baseParams.groupNum;
    baseM = tilingData->mmTilingData.baseM;
    baseN = tilingData->mmTilingData.baseN;
    ubBaseM = tilingData->baseParams.ubBaseM;
    ubBaseN = tilingData->baseParams.ubBaseN;
    coreNum = tilingData->baseParams.coreNum;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process(HcclServer &hcclServer)
{
    // aicore function
    auto &ubM = tilingData->baseParams.mList;
    auto &ubK = tilingData->baseParams.kList;
    auto &ubN = tilingData->baseParams.nList;
    MNConfig mnConfig;
    mnConfig.baseM = baseM;
    mnConfig.baseN = baseN;

    for (uint32_t groupIdx(0); groupIdx < groupNum; ++groupIdx) {
        mnConfig.m = ubM[groupIdx];
        mnConfig.k = ubK[groupIdx];
        mnConfig.n = ubN[groupIdx];
        ASCENDC_ASSERT(coreNum != 0, { KERNEL_LOG(KERNEL_ERROR, "coreNum should not be 0"); });
        ASCENDC_ASSERT(mnConfig.baseN != 0, { KERNEL_LOG(KERNEL_ERROR, "mnConfig.baseN should not be 0"); });

        uint32_t lambdaN = Ceil(mnConfig.n, coreNum * mnConfig.baseN);
        mnConfig.singleN = lambdaN * mnConfig.baseN;
        mnConfig.numBlocksN = Ceil(mnConfig.n, mnConfig.singleN);
        if (coreNum % mnConfig.numBlocksN == 0) {
            mnConfig.splitM = coreNum / mnConfig.numBlocksN;
        } else {
            mnConfig.splitM = 1;
        }
        uint32_t lambdaM = Ceil(mnConfig.m, MAX_TURN_NUM * mnConfig.splitM * mnConfig.baseM);
        mnConfig.singleM = lambdaM * mnConfig.baseM;
        mnConfig.numBlocksM = Ceil(mnConfig.m, mnConfig.singleM);
        mnConfig.tileCnt = Max<uint32_t>(1, mnConfig.m / mnConfig.singleM / mnConfig.splitM);
        uint32_t totalRun = Ceil(mnConfig.m, mnConfig.splitM * mnConfig.singleM);
        if (mnConfig.m < mnConfig.singleM) {
            mnConfig.singleM = mnConfig.m;
        }

        for (uint32_t i = 0; i < totalRun; ++i) {
            mnConfig.mIdx = i * mnConfig.splitM + coreIdx / mnConfig.numBlocksN;
            mnConfig.nIdx = coreIdx % mnConfig.numBlocksN;
            if (coreIdx < mnConfig.splitM * mnConfig.numBlocksN && mnConfig.mIdx < mnConfig.numBlocksM) {
                computeOp.MMCompute(groupIdx, mnConfig);
            }
#ifndef __CCE_KT_TEST__
            hcclServer.TurnNotifyRun(coreIdx, mnConfig.numBlocksN * mnConfig.splitM, i + 1);
#endif
        }
#ifndef __CCE_KT_TEST__
        if (coreIdx == 0) {
            hcclServer.TurnWait(totalRun);
        }
#endif
    }
}

/** @brief intenal computation class
 */
template <class mmType, bool sync = false>
class GMMCompute {
public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;

    /** @brief constructor */
    __aicore__ inline GMMCompute(typename mmType::MT &mm_)
        : mm(mm_)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace);

    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig &mnConfig);

private:
    typename mmType::MT &mm; // matmul operator
    bool hasBias = false;
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR biasTensorPtr;
    GM_ADDR yTensorPtr;
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<BiasT> biasGm;
    GlobalTensor<CT> yGm;
    GlobalTensor<CT> workspaceGm; // unused
};

template <class mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                      GM_ADDR workspace)
{
    xTensorPtr = x;
    weightTensorPtr = weight;
    biasTensorPtr = bias;
    yTensorPtr = y;
    if (bias != nullptr && GetTensorAddr<BiasT>(0, biasTensorPtr) != nullptr) {
        hasBias = true;
    }
    workspaceGm.SetGlobalBuffer((__gm__ CT *)workspace);
}

template <class mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::MMCompute(uint32_t groupIdx, MNConfig &mnConfig)
{
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.numBlocksN - 1) {
        curSingleN = mnConfig.n - mnConfig.nIdx * curSingleN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx >= mnConfig.numBlocksM - 1) {
        curSingleM = mnConfig.m - mnConfig.mIdx * curSingleM;
    }

    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    uint64_t wOffset = mnConfig.nIdx * mnConfig.singleN;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + mnConfig.nIdx * mnConfig.singleN;

    // Init global buffer
    xGm.SetGlobalBuffer(GetTensorAddr<AT>(groupIdx, xTensorPtr));
    weightGm.SetGlobalBuffer(GetTensorAddr<BT>(groupIdx, weightTensorPtr));
    yGm.SetGlobalBuffer(GetTensorAddr<CT>(groupIdx, yTensorPtr));

    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset]);
    mm.SetTensorB(weightGm[wOffset]);
    if (hasBias) {
        biasGm.SetGlobalBuffer(GetTensorAddr<BiasT>(groupIdx, biasTensorPtr));
        mm.SetBias(biasGm[mnConfig.nIdx * mnConfig.singleN]);
    }
    mm.template IterateAll<sync>(yGm[outOffset], false);
}
} // namespace GROUPED_MAT_MUL_ALL_REDUCE

#endif // ASCENDC_GROUPED_MAT_MUL_ALL_REDUCE_H
