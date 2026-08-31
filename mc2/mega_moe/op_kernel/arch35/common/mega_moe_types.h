/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TYPES_H
#define MEGA_MOE_TYPES_H

#include "lib/std/tuple.h"
#include "tensor_api/tensor.h"
#include "../mega_moe_tiling.h"
#include "mega_moe_constants.h"
#include "mega_moe_workspace.h"
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "adv_api/hcomm/hcomm.h"
#endif
#if __has_include("../../../common/mc2_moe_context.h")
#include "../../../common/mc2_moe_context.h"
#else
#include "../../../../common/op_kernel/mc2_moe_context.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

// GMM1/GMM2 逐专家遍历的公共状态。expertIdx 标识当前专家，globalTokenStartIndex 表示该专家
// 在本卡 MoE 专家紧凑 token 序列中的起始索引。
struct ExpertLoopState {
    ProblemShape problemShape;
    int64_t globalTokenStartIndex = 0;
    uint32_t expertIdx = 0U;
    bool expertCountTableReady = false;
};

// GMM1/GMM2 执行期间共同维护的流水状态；引用成员将更新直接回写到调用方持有的状态。
struct GmmRuntimeState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    uint16_t &pingpongIdx;
};

// 标识 MoE 专家序列中的二维 token 位置。
struct ExpertTokenPosition {
    uint32_t expertIdx = 0U;
    uint32_t tokenIndexInExpert = 0U;
};

using Mc2MoeContext = Mc2Aclnn::Mc2MoeContext;

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR gmm1OutGlobal;
    GM_ADDR gmm2OutGlobal;
    GM_ADDR metaInfoGlobal;
    __gm__ int32_t *activationToGmm2Flag;
    __gm__ int32_t *dispatchToGmm1Flag;
    __gm__ int32_t *gmm2CombineSyncCounter;
    __gm__ int32_t *gmmToEpilogueFlag;
    __gm__ int32_t *gmm1TileStatus;
    __gm__ int32_t *sharedExpertGmm2TileCounter;
};

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
struct CombineCommParams {
    uint32_t rankId;
    Hcomm<COMM_PROTOCOL_UBC_CTP> *hcomm;
    __gm__ Mc2MoeContext *mc2Context;
};
#endif

// 保存 TensorList 入口地址，供按 expert 布局解析当前专家权重。
struct ExpertWeightTensorListAddrs {
    GM_ADDR weight1 = nullptr;
    GM_ADDR weightScales1 = nullptr;
    GM_ADDR weight2 = nullptr;
    GM_ADDR weightScales2 = nullptr;
};

struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR sharedBGmAddr;
    GM_ADDR sharedBScaleGmAddr;
    GM_ADDR sharedB2GmAddr;
    GM_ADDR sharedB2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData *tilingData;
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
    CombineCommParams combineCommParams;
#endif
};

enum class AddrUpdateMode : int32_t {
    GMM1,
    GMM2
};

struct BlockJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

// Count/flag workspace 的物理分区；当前与 BlockJobContext 同值，但其编号由 workspace 生产者和消费者共同约定。
struct BlockWorkspaceContext {
    uint32_t blockIdx;
    uint32_t blockNum;
};

template <typename T>
struct PackedElementTraits {
    static constexpr uint32_t ELEMENTS_PER_BYTE = Std::IsSame<T, fp4x2_e2m1_t>::value ? 2U : 1U;
};

struct AivJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

struct MoeStageCommonConfig {
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t moeExpertPerRank;
    uint32_t sharedExpertNum;
    uint32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
};

// GMM1/GMM2 共用的执行方式：当前 block 的任务分工、矩阵模板模式和专家权重布局。
struct GmmExecutionConfig {
    BlockJobContext blockJob;
    int32_t groupedMatmulMode;
    bool isPerExpertWeightTensor;
};

// 各流水阶段在同步 workspace 中为每个专家预留的 slot 数量。
struct MoeSyncWorkspaceLayout {
    int32_t dispatchFlagSlotCountPerExpert;
    int32_t activationFlagSlotCountPerExpert;
    uint32_t gmm1TileStatusCountPerExpert;
    uint64_t combineSyncSlotCountPerExpert;
};

struct GroupSyncSlotLayout {
    uint32_t baseSlotCountPerGroup;
    uint32_t extraSlotGroupCount;
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TYPES_H
