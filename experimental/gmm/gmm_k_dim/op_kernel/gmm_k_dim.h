/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GMM_KDIM_H_
#define _GMM_KDIM_H_

#include "grouped_matmul_utils.h"

using namespace AscendC;

namespace ops_gmm_kdim {

constexpr uint32_t thresholdBlockNum = 8; // 8 is obtained by tests, indicating the threshold of basic block numbers
                                          // in both directions when assigning data blocks to cube cores when using
                                          // diagnal strategy

/*@brief store variables for core split configuration
 */
struct MNConfig {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t mIdx;
    uint32_t nIdx;
    uint32_t blockDimM;
    uint32_t blockDimN;
    uint32_t singleM;
    uint32_t singleN;
    // offset params
    uint32_t totalK = 0;
    uint32_t cuK = 0;
    uint32_t groupIdx = 0;
};

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType>
class GMMProcess {
private:
    __aicore__ inline void Process_();

    ComputeType &computeOp; // inernal computation operator
    const GmmKDimTilingData &tilingData;
    uint32_t coreIdx;
    MNConfig mnConfig;

public:
    /** @brief constructor */
    __aicore__ inline GMMProcess(ComputeType &computeOp_, const GmmKDimTilingData &tilingData)
        : computeOp(computeOp_),
          tilingData(tilingData)
    {}

    __aicore__ inline void Init();

    __aicore__ inline void Process();
};

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Init()
{
    coreIdx = GetBlockIdx();
    mnConfig.m = tilingData.gmmBaseParams.M;
    mnConfig.n = tilingData.gmmBaseParams.N;
    mnConfig.totalK = tilingData.gmmBaseParams.TotalK;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process()
{
    Process_();
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process_()
{
    uint32_t count = 0;

    uint32_t dimM = Ceil(mnConfig.m, tilingData.gmmBaseParams.singleM);
    uint32_t dimN = Ceil(mnConfig.n, tilingData.gmmBaseParams.singleN);
    mnConfig.blockDimM = dimM;
    mnConfig.blockDimN = dimN;
    mnConfig.singleM = tilingData.gmmBaseParams.singleM;
    mnConfig.singleN = tilingData.gmmBaseParams.singleN;

    for (uint32_t groupIdx(0); groupIdx < tilingData.gmmBaseParams.groupNum; ++groupIdx) {
        mnConfig.k = computeOp.CalcKForThisGroup(groupIdx);
        if (mnConfig.k == 0) {
            continue;
        }
        mnConfig.groupIdx = groupIdx;
        uint32_t curCount = count + dimM * dimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + tilingData.gmmBaseParams.usedCoreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * dimN;
        while (curBlock < curCount) {
            // TODO: 先沿用gmm_add经验，未来需要细粒度找寻对角线策略优势区间
            if ((dimM >= 12 && dimN >= 12) || (dimN > 6 && dimM / dimN >= 2) || (dimM > 6 && dimN / dimM >= 3)) {
                uint32_t relativeBlock = curBlock - count;
                uint32_t curThresholdM = relativeBlock >= gmm_utils::AlignDown(dimM * dimN, thresholdM_dimN) ?
                                             dimM % thresholdBlockNum :
                                             thresholdBlockNum;
                uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
                uint32_t curThresholdN = relativeBlock % thresholdM_dimN >=
                                                 gmm_utils::AlignDown(curThresholdM * dimN, curThresholdM_thresholdN) ?
                                             dimN % thresholdBlockNum :
                                             thresholdBlockNum;
                uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
                mnConfig.mIdx =
                    localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
                mnConfig.nIdx = (localRelativeBlock +
                                 localRelativeBlock / gmm_utils::LeastCommonMultiple(curThresholdM, curThresholdN)) %
                                    curThresholdN +
                                relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
            } else {
                mnConfig.mIdx = (curBlock - count) / dimN;
                mnConfig.nIdx = (curBlock - count) % dimN;
            }
            computeOp.MMCompute(groupIdx, mnConfig);
            curBlock += tilingData.gmmBaseParams.usedCoreNum;
        }
        count = curCount % tilingData.gmmBaseParams.usedCoreNum;

        // update cumulative k rows
        mnConfig.cuK += mnConfig.k;
    }
}

/** @brief intenal computation class
 */
template <typename mmType, typename AT, typename CT = AT, typename GT = int64_t, bool trans_a = false,
          bool sync = false>
class GMMCompute {
public:
    using BT = AT;
    // using CT = AT;

    /** @brief constructor */
    __aicore__ inline GMMCompute(mmType &mm_)
        : mm(mm_)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group, GM_ADDR y);

    __aicore__ inline void MMCompute(uint32_t groupIdx, const MNConfig &mnConfig);

    __aicore__ inline int32_t CalcKForThisGroup(int group_id)
    {
        return static_cast<int32_t>(groupGm.GetValue(group_id));
    }

private:
    mmType &mm; // matmul operator
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<GT> groupGm;
    GlobalTensor<CT> yGm;
};

template <typename mmType, typename AT, typename CT, typename GT, bool trans_a, bool sync>
__aicore__ inline void GMMCompute<mmType, AT, CT, GT, trans_a, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group,
                                                                           GM_ADDR y)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(weight));
    groupGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(group));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(y));
}

template <typename mmType, typename AT, typename CT, typename GT, bool trans_a, bool sync>
__aicore__ inline void GMMCompute<mmType, AT, CT, GT, trans_a, sync>::MMCompute(uint32_t groupIdx,
                                                                                const MNConfig &mnConfig)
{
    uint32_t curSingleN = mnConfig.singleN;
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curSingleN = mnConfig.n - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    uint32_t tailM = mnConfig.mIdx * curSingleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = mnConfig.m - tailM;
    }
    // calc offset for this basic block
    uint64_t xOffset;
    if (trans_a) {
        xOffset = static_cast<uint64_t>(mnConfig.cuK) * mnConfig.m + tailM;
    } else {
        xOffset = static_cast<uint64_t>(tailM) * mnConfig.totalK + mnConfig.cuK;
    }
    uint64_t wOffset = static_cast<uint64_t>(mnConfig.cuK) * mnConfig.n + tailN;
    uint64_t yOffset = static_cast<uint64_t>(mnConfig.groupIdx) * mnConfig.m * mnConfig.n +
                       static_cast<uint64_t>(tailM) * mnConfig.n + tailN;
    // @lg: 这里的single k，是每个专家对应的token数量
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], trans_a);
    mm.SetTensorB(weightGm[wOffset]);
    mm.template IterateAll<sync>(yGm[yOffset], 0);
}

} // namespace ops_gmm_kdim

#endif
