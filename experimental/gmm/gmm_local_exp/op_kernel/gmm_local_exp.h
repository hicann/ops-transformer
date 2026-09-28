/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GMM_LOCAL_EXP_H_
#define _GMM_LOCAL_EXP_H_

#include "grouped_matmul_utils.h"

using namespace AscendC;

namespace ops_gmm_local_exp {

constexpr uint32_t thresholdBlockNum = 8; // 8 is obtained by tests, indicating the threshold of basic block numbers
                                          // in both directions when assigning data blocks to cube cores when using
                                          // diagnal strategy
constexpr uint32_t thresholdDimM = 1;     // 1 is obtained by tests, indicating the threshold for distinguishing

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
    uint32_t cuM = 0;
    uint32_t xOffset = 0;
    uint32_t wOffset = 0;
    uint32_t yOffset = 0;
    uint32_t groupIdx = 0;
    uint32_t groupIdxLocal = 0;
};

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType>
class GMMProcess {
private:
    __aicore__ inline void Process_();

    ComputeType &computeOp; // inernal computation operator
    const GmmLocalExpTilingData &tilingData;
    uint32_t coreIdx;
    MNConfig mnConfig;

public:
    /** @brief constructor */
    __aicore__ inline GMMProcess(ComputeType &computeOp_, const GmmLocalExpTilingData &tilingData)
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
    mnConfig.k = tilingData.gmmBaseParams.K;
    mnConfig.n = tilingData.gmmBaseParams.N;
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
    const uint32_t tokenPerTile = 256;
    for (uint32_t groupIdx(0); groupIdx < tilingData.gmmBaseParams.expEndIdx; ++groupIdx) {
        mnConfig.m = computeOp.CalcMForThisGroup(groupIdx);

        if (groupIdx >= tilingData.gmmBaseParams.expStartIdx) {
            // set singlemn by perf results: https://km.sankuai.com/collabpage/2703179708
            const uint32_t mTileNum = Ceil(mnConfig.m, tokenPerTile);
            const uint32_t nTileNum = Ceil(mnConfig.n, tokenPerTile);
            if (mTileNum < 4) {
                mnConfig.singleM = 256;
            } else if (mTileNum < 8) {
                mnConfig.singleM = 128;
            } else {
                mnConfig.singleM = 256;
            }
            // reset singlemn by perf results: https://km.sankuai.com/collabpage/2704643930
            if ((mTileNum == 2 && nTileNum == 12) || (mTileNum == 3 && nTileNum == 8) ||
                (mTileNum == 2 && nTileNum >= 24)) {
                mnConfig.singleM = 128;
            }
            mnConfig.singleN = 256;

            uint32_t dimM = Ceil(mnConfig.m, mnConfig.singleM);
            uint32_t dimN = Ceil(mnConfig.n, mnConfig.singleN);
            mnConfig.blockDimM = dimM;
            mnConfig.blockDimN = dimN;
            mnConfig.groupIdx = groupIdx;
            mnConfig.groupIdxLocal = groupIdx - tilingData.gmmBaseParams.expStartIdx;
            uint32_t thresholdM_dimN = thresholdBlockNum * dimN;

            uint32_t curCount = count + dimM * dimN;
            uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + tilingData.gmmBaseParams.usedCoreNum;
            while (curBlock < curCount) {
                if (dimM <= thresholdDimM) {
                    mnConfig.mIdx = (curBlock - count) / dimN;
                    mnConfig.nIdx = (curBlock - count) % dimN;
                } else {
                    uint32_t relativeBlock = curBlock - count;
                    uint32_t curThresholdM = relativeBlock >= gmm_utils::AlignDown(dimM * dimN, thresholdM_dimN) ?
                                                 dimM % thresholdBlockNum :
                                                 thresholdBlockNum;
                    uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
                    uint32_t curThresholdN =
                        relativeBlock % thresholdM_dimN >=
                                gmm_utils::AlignDown(curThresholdM * dimN, curThresholdM_thresholdN) ?
                            dimN % thresholdBlockNum :
                            thresholdBlockNum;
                    uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
                    mnConfig.mIdx =
                        localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
                    mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / gmm_utils::LeastCommonMultiple(
                                                                                   curThresholdM, curThresholdN)) %
                                        curThresholdN +
                                    relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
                }
                computeOp.MMCompute(groupIdx, mnConfig);
                curBlock += tilingData.gmmBaseParams.usedCoreNum;
            }
            count = curCount % tilingData.gmmBaseParams.usedCoreNum;
        }
        // update cumulative m rows
        mnConfig.cuM += mnConfig.m;
    }
}

/** @brief intenal computation class
 */
template <typename mmType, typename AT, typename CT = AT, typename GT = int64_t, bool trans_b = false,
          bool sync = false, bool is_b_nz = false>
class GMMCompute {
public:
    using BT = AT;

    /** @brief constructor */
    __aicore__ inline GMMCompute(mmType &mm_)
        : mm(mm_)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR group, GM_ADDR y);

    __aicore__ inline void MMCompute(uint32_t groupIdx, const MNConfig &mnConfig);

    __aicore__ inline int32_t CalcMForThisGroup(int group_id)
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

template <typename mmType, typename AT, typename CT, typename GT, bool trans_b, bool sync, bool is_b_nz>
__aicore__ inline void GMMCompute<mmType, AT, CT, GT, trans_b, sync, is_b_nz>::Init(GM_ADDR x, GM_ADDR weight,
                                                                                    GM_ADDR group, GM_ADDR y)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(weight));
    groupGm.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(group));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(y));
}

template <typename mmType, typename AT, typename CT, typename GT, bool trans_b, bool sync, bool is_b_nz>
__aicore__ inline void GMMCompute<mmType, AT, CT, GT, trans_b, sync, is_b_nz>::MMCompute(uint32_t groupIdx,
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
    uint32_t xOffset = (mnConfig.cuM + tailM) * mnConfig.k;
    uint32_t wOffset;
    if (is_b_nz) {
        wOffset = mnConfig.groupIdxLocal * mnConfig.k * mnConfig.n + tailN * 16;
    } else {
        if (trans_b) {
            wOffset = mnConfig.groupIdxLocal * mnConfig.k * mnConfig.n + tailN * mnConfig.k;
        } else {
            wOffset = mnConfig.groupIdxLocal * mnConfig.k * mnConfig.n + tailN;
        }
    }
    uint64_t yOffset = (static_cast<uint64_t>(mnConfig.cuM) + tailM) * mnConfig.n + tailN;

    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset]);
    mm.SetTensorB(weightGm[wOffset], trans_b);
    mm.template IterateAll<sync>(yGm[yOffset], 0);
}

} // namespace ops_gmm_local_exp

#endif
