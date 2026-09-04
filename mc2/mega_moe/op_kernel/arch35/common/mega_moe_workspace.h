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
 * \file mega_moe_workspace.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_H
#define MEGA_MOE_WORKSPACE_H

// peermem 窗口的布局与尺寸（含 HOST_DEVICE / GM_ADDR 宏与基础 include）。
#include "mega_moe_peermem.h"

namespace MegaMoeImpl {

constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60 * 1024LL; // 异常dump区
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;

constexpr int64_t INVALID_WORKSPACE_OFFSET = -1LL;

// 仅描述各 workspace 分区相对基址的字节偏移，不持有或构造任何地址。
struct WorkspaceLayout {
    int64_t dispatchRevDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dispatchRevScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t dispatchRevWeightsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t activationQuantDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t activationQuantScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t expertRevTokenNumsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t metaInfoOffset{INVALID_WORKSPACE_OFFSET};
    int64_t flagActivationToGmm2Offset{INVALID_WORKSPACE_OFFSET};
    int64_t flagDispatchToGmm1Offset{INVALID_WORKSPACE_OFFSET};
    int64_t flagSendCntCalToUpdParamsOffset{INVALID_WORKSPACE_OFFSET};
    int64_t flagGmmToEpilogueOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2ReadyOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2CombineSyncCounterOffset{INVALID_WORKSPACE_OFFSET};
    int64_t cumsumInfoOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm1MmadResOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm2MmadResOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertResultOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertGmm1OutOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertInputScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertActivationDataOffset{INVALID_WORKSPACE_OFFSET};
    int64_t sharedExpertActivationScaleOffset{INVALID_WORKSPACE_OFFSET};
    int64_t gmm1TileStatusOffset{INVALID_WORKSPACE_OFFSET}; // GMM1 tile 就绪状态位区（仅 prefetch 软同步分配）
    int64_t sharedExpertGmm2TileCounterOffset{INVALID_WORKSPACE_OFFSET};

    int64_t maskSlotOffset{INVALID_WORKSPACE_OFFSET};       // urma发送mask临时GM
    int64_t dispatchL1CommOffset{INVALID_WORKSPACE_OFFSET}; // dispatch L1 communication workspace
    int64_t dispatchCursorOffset{INVALID_WORKSPACE_OFFSET}; // dispatch cnt for each expert
    int64_t dispatchDoneOffset{INVALID_WORKSPACE_OFFSET};   // dispatch done
    int64_t dispatchL2CommOffset{INVALID_WORKSPACE_OFFSET}; // dispatch l2 communication workspace

    // 连续 flag 通知区（自 flagActivationToGmm2Offset 起）的 int32 元素总数。
    // 在分配处顺手记账，保证 ResetSyncStatus 的清零范围与分配范围恒同源。
    int64_t flagResetElementCount{0};
    // prefetch 软同步 GMM1 tile 状态区的 int32 元素总数（未分配时为 0）。
    int64_t gmm1TileStatusElementCount{0};

    int64_t workspaceSize{0};
    HOST_DEVICE explicit WorkspaceLayout(const MegaMoeTilingData *tilingData, uint32_t serverNum = 1U)
    {
        Initialize(tilingData, serverNum);
    }

private:
    HOST_DEVICE void Initialize(const MegaMoeTilingData *tilingData, uint32_t serverNum)
    {
        workspaceSize = 0;
        dispatchRevDataOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScaleOffset = workspaceSize;

        int64_t dispatchScaleElementsPerToken =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * dispatchScaleElementsPerToken, ALIGN_512);

        if (tilingData->topkWeightsPrefetch == 1 && tilingData->topoType == TOPO_TYPE_URMA) {
            dispatchRevWeightsOffset = workspaceSize;
            uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)),
                                                             static_cast<uint32_t>(ALIGN_32));
            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->maxOutputSize) * static_cast<int64_t>(weightAlignBytes), ALIGN_512);
        }

        activationQuantDataOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / ACTIVATION_N_HALF, ALIGN_512);

        activationQuantScaleOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / ACTIVATION_N_HALF / MXFP_SCALE_GROUP_NUM,
            ALIGN_512);

        expertRevTokenNumsOffset = workspaceSize;
        int64_t expertMajorPaddedCountBytes =
            static_cast<int64_t>(tilingData->moeExpertPerRank) * ALIGN_32 * tilingData->aicNum;
        int64_t blockMajorCountStride = static_cast<int64_t>(
            Ops::Base::CeilAlign(tilingData->moeExpertPerRank, static_cast<uint32_t>(INT_CACHELINE)));
        int64_t blockMajorCountBytes =
            blockMajorCountStride * static_cast<int64_t>(sizeof(int32_t)) * tilingData->aicNum;
        int64_t expertCountWorkspaceBytes =
            tilingData->topoType == TOPO_TYPE_MTE ? blockMajorCountBytes : expertMajorPaddedCountBytes;
        workspaceSize += Ops::Base::CeilAlign(expertCountWorkspaceBytes, ALIGN_512);

        metaInfoOffset = workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);

        // 以下三组 flag 仅由 MoE 专家使用；共享专家路径不使用这些 flag。
        const bool useMteWaveCombine = tilingData->topoType == TOPO_TYPE_MTE;
        int64_t maxWavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
        // MTE GMM2 按 256-token M group 消费 activation；URMA 只使用每个专家的首槽。
        int64_t activationFlagSlotsPerExpert =
            tilingData->topoType == TOPO_TYPE_MTE ? waveFlagSlotsPerExpert : static_cast<int64_t>(INT_CACHELINE);
        int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);

        // 所有 Scalar 通知都放在同一段连续 workspace 中。每个逻辑槽独占一个 64B cache line，
        // 因而 ResetFlagList 可以按任务分区一次清理完整区域。
        int64_t flagRegionBeginOffset = workspaceSize;
        flagActivationToGmm2Offset = workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * activationFlagSlotsPerExpert;
        flagDispatchToGmm1Offset = workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsOffset = workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * moeExpertCount * tilingData->aicNum;

        // 每个 AIC 的就绪序列与前面的 flag 连续存放，使 ResetFlagList 能用同一次 MTE reset 清理；
        // 每个序列独占一个 64B cache line。
        // W4 GMM1 activation 始终使用该序号；非量化 GMM2/Combine 也复用它做 tile 一对一通知。
        bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
        if (isW4Mode || (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode == COMBINE_NO_QUANT)) {
            flagGmmToEpilogueOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
        }

        if (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode != COMBINE_NO_QUANT) {
            // 量化 Combine 在完整专家结束后等待每个 AIC 的完成标记。
            gmm2ReadyOffset = workspaceSize;
            workspaceSize += SIZE_INT_32 * moeExpertCount * tilingData->aicNum * INT_CACHELINE;
        }

        if (tilingData->topoType == TOPO_TYPE_URMA) {
            gmm2CombineSyncCounterOffset = workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->combineSyncSlotCountPerExpert) * moeExpertCount *
                             INT_CACHELINE * SIZE_INT_32;
        }

        // Shared expert GMM2 tile counter: tile 级 flag counter, 每 shared expert 一组 slot。
        // 纳入连续 flag 区以便 ResetFlagList 一次性清零。
        if (tilingData->sharedExpertNum > 0) {
            sharedExpertGmm2TileCounterOffset = workspaceSize;
            int64_t tokenGroupCount =
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs), static_cast<int64_t>(L1_TILE_M_256));
            workspaceSize +=
                SIZE_INT_32 * tokenGroupCount * static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE;
        }
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;

        // A8W4 / Combine 量化路径的条件 workspace 分配。
        // 以下条件分配与 mega_moe.h 编译期守卫 (ENABLE_A8W4 / ENABLE_A4W4 / CombineQuantMode) 一致，
        // 由 TilingKey 保证同步。
        // W4 Wave-ahead Dispatch 与 layered A8W4 都会跨 Activation 保留 cumsum；Activation 会覆盖对应 UB，
        // 因此需要逐物理 block 的 GM 备份。A8W8 的 cumsum 常驻 AIV1 UB 并按 wave 连续推进，不需要该备份。
        cumsumInfoOffset = INVALID_WORKSPACE_OFFSET;
        gmm1MmadResOffset = INVALID_WORKSPACE_OFFSET;
        gmm2MmadResOffset = INVALID_WORKSPACE_OFFSET;
        bool activationOverwritesDispatchCumsum =
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            (tilingData->topoType == TOPO_TYPE_MTE && (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                                                       tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ));
        if (activationOverwritesDispatchCumsum) {
            // cumsumInfo：逐核备份 cumsum 状态，每核 moeExpertPerRank × epWorldSize 个 int32。
            cumsumInfoOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_INT_32 * tilingData->moeExpertPerRank *
                                                                       tilingData->epWorldSize),
                                                  ALIGN_32) *
                             tilingData->aicNum;
        }
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 || tilingData->topkWeightsPrefetch == 1) {
            // gmm1MmadRes：GMM1 matmul 输出，布局为 maxOutputSize × hiddenDim 个 BF16。
            gmm1MmadResOffset = workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
        // gmm2MmadRes：GMM2 matmul 输出，布局为 maxOutputSize × h 个 BF16；
        // A8W4 GMM1、A4W4 混合路径、A4W4_NZ 和 Combine 量化路径需要该区域。
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ ||
            tilingData->combineQuantMode != COMBINE_NO_QUANT || tilingData->topoType == TOPO_TYPE_URMA ||
            useMteWaveCombine) {
            if (useMteWaveCombine) {
                workspaceSize = Ops::Base::CeilAlign(workspaceSize, static_cast<int64_t>(ALIGN_512));
            }
            gmm2MmadResOffset = workspaceSize;
            // MTE wave 路径为每个 MoE token 行分配独立的 [H] BF16
            // 区域，生命周期覆盖整个 kernel。它不是 batch 或 ring slot，
            // 因此 AIC 推进游标前不需要等待 consumer ACK。
            int64_t gmm2OutputBytes = static_cast<int64_t>(SIZE_BF_16) *
                                      static_cast<int64_t>(tilingData->maxOutputSize) *
                                      static_cast<int64_t>(tilingData->h);
            workspaceSize += useMteWaveCombine ?
                                 Ops::Base::CeilAlign(gmm2OutputBytes, static_cast<int64_t>(ALIGN_512)) :
                                 gmm2OutputBytes;
        }

        // GMM1 tile 状态位区（仅 prefetch 路径分配，用于 AIC→AIV 软同步）。
        // 每个 tile 状态和末尾 allDone 状态都独占一条 64B cache line。
        gmm1TileStatusOffset = INVALID_WORKSPACE_OFFSET;
        if (tilingData->topkWeightsPrefetch == 1) {
            gmm1TileStatusOffset = workspaceSize;
            int64_t statusSlots =
                static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->maxTilesPerExpert + 1;
            gmm1TileStatusElementCount = statusSlots * INT_CACHELINE;
            int64_t statusBytes = SIZE_INT_32 * statusSlots * INT_CACHELINE;
            workspaceSize += Ops::Base::CeilAlign(statusBytes, ALIGN_512);
        }
        if (tilingData->topoType == TOPO_TYPE_URMA) {
            maskSlotOffset = workspaceSize;
            int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
            int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)),
                                                        static_cast<int64_t>(ALIGN_256)) /
                                   static_cast<int64_t>(sizeof(int32_t));
            int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, static_cast<int64_t>(ALIGN_32));
            int64_t maskSlotSize = maskAlignSize + static_cast<int64_t>(ALIGN_32); // mask + 32B count

            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->epWorldSize * maskSlotSize,
                static_cast<int64_t>(ALIGN_512));

            dispatchL1CommOffset = workspaceSize;
            int64_t serverWorkspaceBytes =
                static_cast<int64_t>(ALIGN_32) + static_cast<int64_t>(tilingData->bs) * ALIGN_32;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchCursorOffset = workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(static_cast<int64_t>(serverNum * SIZE_INT_32), static_cast<int64_t>(ALIGN_512));

            dispatchDoneOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->aicNum * SIZE_INT_32),
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchL2CommOffset = workspaceSize;
            int64_t flagSnapshotBytes = static_cast<int64_t>(tilingData->bs) * static_cast<int64_t>(sizeof(uint64_t));
            int64_t dispatchL2ScratchBytes = Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<int64_t>(ALIGN_512));
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * dispatchL2ScratchBytes;
        }

        // 共享专家 workspace buffer。
        if (tilingData->sharedExpertNum > 0) {
            // sharedExpertResult：共享专家 GMM2 输出 [sharedExpertNum × bs × h]。
            sharedExpertResultOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->h, ALIGN_512);
            // sharedExpertGmm1Out：共享专家 GMM1 输出 [sharedExpertNum × bs × hiddenDim]。
            sharedExpertGmm1OutOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim, ALIGN_512);
            // sharedExpertInputData：GMM1 输入数据 [bs × h]。
            sharedExpertInputDataOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->h, ALIGN_512);
            // sharedExpertInputScale：GMM1 输入 scale [bs × CeilDiv(h, 32) × 2]。
            sharedExpertInputScaleOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs *
                                                      Ops::Base::CeilDiv(static_cast<uint32_t>(tilingData->h),
                                                                         static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM)) *
                                                      MXFP_MULTI_BASE_SIZE,
                                                  ALIGN_512);
            // sharedExpertActivationData：SwiGLU 量化输出 [sharedExpertNum × bs × hiddenDim / 2] FP8。
            sharedExpertActivationDataOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim / ACTIVATION_N_HALF,
                ALIGN_512);
            // sharedExpertActivationScale：SwiGLU scale [sharedExpertNum × bs × hiddenDim / 2 / 32]。
            sharedExpertActivationScaleOffset = workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum *
                                                      tilingData->hiddenDim / ACTIVATION_N_HALF / MXFP_SCALE_GROUP_NUM,
                                                  ALIGN_512);
        }
    }
};

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr{nullptr};
    GM_ADDR dispatchRevScalePtr{nullptr};
    GM_ADDR dispatchRevWeightsPtr{nullptr};
    GM_ADDR activationQuantDataPtr{nullptr};
    GM_ADDR activationQuantScalePtr{nullptr};
    GM_ADDR expertRevTokenNumsPtr{nullptr};
    GM_ADDR metaInfoPtr{nullptr};
    GM_ADDR flagActivationToGmm2Ptr{nullptr};
    GM_ADDR flagDispatchToGmm1Ptr{nullptr};
    GM_ADDR flagSendCntCalToUpdParamsPtr{nullptr};
    GM_ADDR flagGmmToEpiloguePtr{nullptr};
    GM_ADDR gmm2ReadyPtr{nullptr};
    GM_ADDR gmm2CombineSyncCounterPtr{nullptr};
    GM_ADDR cumsumInfoPtr{nullptr};
    GM_ADDR gmm1MmadResPtr{nullptr};
    GM_ADDR gmm2MmadResPtr{nullptr};
    GM_ADDR sharedExpertResultPtr{nullptr};
    GM_ADDR sharedExpertGmm1OutPtr{nullptr};
    GM_ADDR sharedExpertInputDataPtr{nullptr};
    GM_ADDR sharedExpertInputScalePtr{nullptr};
    GM_ADDR sharedExpertActivationDataPtr{nullptr};
    GM_ADDR sharedExpertActivationScalePtr{nullptr};
    GM_ADDR gmm1TileStatusPtr{nullptr};
    GM_ADDR sharedExpertGmm2TileCounterPtr{nullptr};

    GM_ADDR maskSlotPtr{nullptr};
    GM_ADDR dispatchL1CommPtr{nullptr};
    GM_ADDR dispatchCursorPtr{nullptr};
    GM_ADDR dispatchDonePtr{nullptr};
    GM_ADDR dispatchL2CommPtr{nullptr};

    int64_t flagResetElementCount{0};
    int64_t gmm1TileStatusElementCount{0};

private:
    // 仅在取得真实 workspace 基址后，将有效偏移绑定为可访问地址。
    HOST_DEVICE static GM_ADDR ResolveWorkspaceAddress(GM_ADDR base, int64_t offset)
    {
        if (base == nullptr || offset < 0) {
            return nullptr;
        }
        return base + offset;
    }

public:
    HOST_DEVICE void Bind(GM_ADDR base, const WorkspaceLayout &layout)
    {
        dispatchRevDataPtr = ResolveWorkspaceAddress(base, layout.dispatchRevDataOffset);
        dispatchRevScalePtr = ResolveWorkspaceAddress(base, layout.dispatchRevScaleOffset);
        dispatchRevWeightsPtr = ResolveWorkspaceAddress(base, layout.dispatchRevWeightsOffset);
        activationQuantDataPtr = ResolveWorkspaceAddress(base, layout.activationQuantDataOffset);
        activationQuantScalePtr = ResolveWorkspaceAddress(base, layout.activationQuantScaleOffset);
        expertRevTokenNumsPtr = ResolveWorkspaceAddress(base, layout.expertRevTokenNumsOffset);
        metaInfoPtr = ResolveWorkspaceAddress(base, layout.metaInfoOffset);
        flagActivationToGmm2Ptr = ResolveWorkspaceAddress(base, layout.flagActivationToGmm2Offset);
        flagDispatchToGmm1Ptr = ResolveWorkspaceAddress(base, layout.flagDispatchToGmm1Offset);
        flagSendCntCalToUpdParamsPtr = ResolveWorkspaceAddress(base, layout.flagSendCntCalToUpdParamsOffset);
        flagGmmToEpiloguePtr = ResolveWorkspaceAddress(base, layout.flagGmmToEpilogueOffset);
        gmm2ReadyPtr = ResolveWorkspaceAddress(base, layout.gmm2ReadyOffset);
        gmm2CombineSyncCounterPtr = ResolveWorkspaceAddress(base, layout.gmm2CombineSyncCounterOffset);
        cumsumInfoPtr = ResolveWorkspaceAddress(base, layout.cumsumInfoOffset);
        gmm1MmadResPtr = ResolveWorkspaceAddress(base, layout.gmm1MmadResOffset);
        gmm2MmadResPtr = ResolveWorkspaceAddress(base, layout.gmm2MmadResOffset);
        sharedExpertResultPtr = ResolveWorkspaceAddress(base, layout.sharedExpertResultOffset);
        sharedExpertGmm1OutPtr = ResolveWorkspaceAddress(base, layout.sharedExpertGmm1OutOffset);
        sharedExpertInputDataPtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputDataOffset);
        sharedExpertInputScalePtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputScaleOffset);
        sharedExpertActivationDataPtr = ResolveWorkspaceAddress(base, layout.sharedExpertActivationDataOffset);
        sharedExpertActivationScalePtr = ResolveWorkspaceAddress(base, layout.sharedExpertActivationScaleOffset);
        gmm1TileStatusPtr = ResolveWorkspaceAddress(base, layout.gmm1TileStatusOffset);
        sharedExpertGmm2TileCounterPtr = ResolveWorkspaceAddress(base, layout.sharedExpertGmm2TileCounterOffset);
        maskSlotPtr = ResolveWorkspaceAddress(base, layout.maskSlotOffset);
        dispatchL1CommPtr = ResolveWorkspaceAddress(base, layout.dispatchL1CommOffset);
        dispatchCursorPtr = ResolveWorkspaceAddress(base, layout.dispatchCursorOffset);
        dispatchDonePtr = ResolveWorkspaceAddress(base, layout.dispatchDoneOffset);
        dispatchL2CommPtr = ResolveWorkspaceAddress(base, layout.dispatchL2CommOffset);
        flagResetElementCount = layout.flagResetElementCount;
        gmm1TileStatusElementCount = layout.gmm1TileStatusElementCount;
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_H
