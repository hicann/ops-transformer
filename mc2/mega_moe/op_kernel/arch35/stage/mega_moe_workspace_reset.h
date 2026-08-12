/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_WORKSPACE_RESET_H
#define MEGA_MOE_WORKSPACE_RESET_H

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
struct ResetWorkspaceConfig {
    int32_t flagElementCount;
    int32_t resetBatchElementCount;
};

__aicore__ inline int64_t CalcResetFlagElementCount(const MegaMoeTilingData *tilingData)
{
    bool isA8W8 = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_GENERAL ||
                  tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ;
    bool useMteA8W8Wave = tilingData->topoType == TOPO_TYPE_MTE && isA8W8;
    bool useGroupSyncCounters =
        tilingData->topoType == TOPO_TYPE_URMA ||
        (tilingData->combineQuantMode != COMBINE_NO_QUANT && !useMteA8W8Wave);
    int64_t maxWavesPerExpert =
        Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(L1_TILE_M_256));
    int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
    int64_t activationFlagSlotsPerExpert =
        useMteA8W8Wave ? waveFlagSlotsPerExpert : static_cast<int64_t>(INT_CACHELINE);
    int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);

    int64_t flagElementCount =
        moeExpertCount *
        (activationFlagSlotsPerExpert + waveFlagSlotsPerExpert +
         static_cast<int64_t>(INT_CACHELINE) * tilingData->aicNum);
    if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        flagElementCount += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (useMteA8W8Wave) {
        flagElementCount += moeExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (useGroupSyncCounters) {
        flagElementCount += static_cast<int64_t>(tilingData->combineSyncSlotCountPerExpert) * moeExpertCount *
                            INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0) {
        int64_t tokenGroupCount =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs), static_cast<int64_t>(L1_TILE_M_256));
        flagElementCount += tokenGroupCount * static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE;
    }
    return flagElementCount;
}

// 清理连续 flag workspace；prefetch 路径额外清理独立分配的 GMM1 tile 状态区。
template <bool TopkWeightsPrefetch>
__aicore__ inline void ResetSyncStatus(const DispatchPrepareConfig &context,
                                       const Params &params,
                                       const ResetWorkspaceConfig &config,
                                       LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || config.resetBatchElementCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, params.workspaceInfo.flagActivationToGmm2Ptr, config.flagElementCount,
                            config.resetBatchElementCount, resetTensor);
    if constexpr (TopkWeightsPrefetch) {
        int32_t statusElementCount =
            (static_cast<int32_t>(context.common.moeExpertPerRank) *
                 static_cast<int32_t>(params.tilingData->maxTilesPerExpert) +
             1) *
            INT_CACHELINE;
        ResetWorkspaceRegion<1>(job, params.workspaceInfo.gmm1TileStatusPtr, statusElementCount,
                                config.resetBatchElementCount, resetTensor);
    }
}
#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_RESET_H
