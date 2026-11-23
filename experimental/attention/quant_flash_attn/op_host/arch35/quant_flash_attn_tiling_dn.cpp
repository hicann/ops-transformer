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
 * \file quant_flash_attn_tiling_dn.cpp
 * \brief
 */

#include "quant_flash_attn_tiling_dn.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"
#include "../../op_kernel/arch35/quant_flash_attn_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

void QuantFlashAttnTilingDn::InitTilingInfo(TilingInfo* tilingInfo)
{
    tilingInfo_ = static_cast<QuantFlashAttnTilingInfo*>(tilingInfo);
}

bool QuantFlashAttnTilingDn::IsCapable()
{
    if (tilingInfo_ == nullptr) {
        return false;
    }

    return true;
}

void QuantFlashAttnTilingDn::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(tilingInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus QuantFlashAttnTilingDn::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantFlashAttnTilingDn::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(tilingInfo_->opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.cvRatio = platformInfo_.aivNum / platformInfo_.aicNum;
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(tilingInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingDn::InitImplParam() {}

void QuantFlashAttnTilingDn::SplitPolicy()
{
    CalcNumBlocks(platformInfo_.aicNum);
}

void QuantFlashAttnTilingDn::GenTilingKey()
{
    uint8_t queryOutLayout = LAYOUT_ENUM_BSND;
    if (tilingInfo_->layoutQ == FiaLayout::BSND) {
        if (tilingInfo_->layoutOut == FiaLayout::BNSD) {
            queryOutLayout = LAYOUT_ENUM_BSND;
        }
    } else if (tilingInfo_->layoutQ == FiaLayout::BNSD) {
        if (tilingInfo_->layoutOut == FiaLayout::BNSD) {
            queryOutLayout = LAYOUT_ENUM_BNSD;
        } else if (tilingInfo_->layoutOut == FiaLayout::BSND) {
            queryOutLayout = LAYOUT_ENUM_BNSD_BSND;
        }
    } else if (tilingInfo_->layoutQ == FiaLayout::TND) {
        if (tilingInfo_->layoutOut == FiaLayout::TND) {
            queryOutLayout = LAYOUT_ENUM_TND;
        }
    }

    uint8_t kvStorageMode = KV_STORAGE_MODE_CONTINUE;
    if (tilingInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        if (tilingInfo_->layoutKV == FiaLayout::BSND || tilingInfo_->layoutKV == FiaLayout::BnBsND) {
            kvStorageMode = KV_STORAGE_MODE_PA_BBND; // PA_BBND 块内排布为 (Bs, N, D)
        } else if (tilingInfo_->layoutKV == FiaLayout::BNSD || tilingInfo_->layoutKV == FiaLayout::BnNBsD) {
            kvStorageMode = KV_STORAGE_MODE_PA_BNBD; // PA_BNBD 块内排布为 (N, Bs, D)
        } else if (tilingInfo_->layoutKV == FiaLayout::NZ) {
            kvStorageMode = KV_STORAGE_MODE_PA_NZ; // PA_NZ 块内排布为 (D/64, Bs, 64)
        }
    }

    bool hasMask = tilingInfo_->attnMaskFlag;

    // ND 判据: TND + 任意 KV 布局(PA BBND/BNBD/NZ 或 CONTINUE) + mask 0/3,
    // 且每个 batch 的 M 行 = S1*G <= 64(连续 KV 同规格已验证)
    constexpr int64_t ND_MAX_M_ROWS = 64;
    int64_t bNum = (tilingInfo_->bSize > 0) ? tilingInfo_->bSize : 1;
    int64_t mRows = (tilingInfo_->maxSeqLenQ >= 0) ? tilingInfo_->maxSeqLenQ : (tilingInfo_->s1Size + bNum - 1) / bNum;
    bool useNd = false;
    bool ndMaskOk = (tilingInfo_->maskMode == 0) || (tilingInfo_->maskMode == 3);
    if ((queryOutLayout == LAYOUT_ENUM_TND) && ndMaskOk && (mRows * tilingInfo_->gSize <= ND_MAX_M_ROWS)) {
        useNd = true;
    }

    const uint32_t tplHasMask = static_cast<uint32_t>(hasMask);
    tilingKey_ = GET_TPL_TILING_KEY(queryOutLayout, useNd ? Config_DN_ND : Config_DN_FIXED, QFA_MXFP4_DN, tplHasMask,
                                    kvStorageMode, 0);
}

void QuantFlashAttnTilingDn::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(tilingInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void QuantFlashAttnTilingDn::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    workspaceSize_ = sysWorkspaceSize;

    // FD partial workspace 三段: accumOut [slot][d=128][m=128] fp32 / sum、max 各 [slot][m=128] 1 个 fp32
    // 每 AIV 核 head/tail 双槽; 归约按 8 对齐读行可能越过槽尾, 末尾多留一个槽作读窗
    uint32_t mSize = 128;
    uint32_t dSize = static_cast<uint32_t>(tilingInfo_->vHeadDim);
    uint64_t slotNum = static_cast<uint64_t>(numBlocks_) * 2 + 1;
    uint64_t accumOutSize = slotNum * mSize * dSize * sizeof(float);
    uint64_t logSumExpSize = slotNum * mSize * sizeof(float); // sum 段; max 段同大
    workspaceSize_ += accumOutSize + logSumExpSize + logSumExpSize;
    tilingData_.baseTiling.flashAttnWorkspaceParams.accumOutSize = static_cast<uint32_t>(accumOutSize);
    tilingData_.baseTiling.flashAttnWorkspaceParams.logSumExpSize = static_cast<uint32_t>(logSumExpSize);
}

void QuantFlashAttnTilingDn::FillTiling()
{
    tilingData_.baseTiling.flashAttnBaseParams.bSize = tilingInfo_->bSize;
    tilingData_.baseTiling.flashAttnBaseParams.t1Size = tilingInfo_->queryTSize;
    tilingData_.baseTiling.flashAttnBaseParams.t2Size = tilingInfo_->keyTSize;
    // TND+N2TGD: S1G 合轴, M 轴 = S1*G; 非 TND 头轴摊平
    if (tilingInfo_->layoutQ == FiaLayout::TND && tilingInfo_->layoutQDescale == FiaLayout::N2TGD) {
        tilingData_.baseTiling.flashAttnBaseParams.n2Size = tilingInfo_->n1Size / tilingInfo_->gSize;
        tilingData_.baseTiling.flashAttnBaseParams.gSize = tilingInfo_->gSize;
    } else {
        tilingData_.baseTiling.flashAttnBaseParams.n2Size = tilingInfo_->n1Size;
        tilingData_.baseTiling.flashAttnBaseParams.gSize = 1;
    }
    tilingData_.baseTiling.flashAttnBaseParams.gRealSize = tilingInfo_->gSize;
    tilingData_.baseTiling.flashAttnBaseParams.s1Size = tilingInfo_->s1Size;
    tilingData_.baseTiling.flashAttnBaseParams.s2Size = tilingInfo_->s2Size;
    tilingData_.baseTiling.flashAttnBaseParams.dSize = tilingInfo_->qkHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.dSizeV = tilingInfo_->vHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.qCuSeqLensSize = tilingInfo_->qCuSeqLensSize;
    tilingData_.baseTiling.flashAttnBaseParams.kvCuSeqLensSize = tilingInfo_->kvCuSeqLensSize;
    tilingData_.baseTiling.flashAttnBaseParams.qSeqUsedSize = tilingInfo_->qSeqUsedSize;
    tilingData_.baseTiling.flashAttnBaseParams.kvSeqUsedSize = tilingInfo_->kvSeqUsedSize;
    /* seqUsed/cuSeq 为空时 kernel 侧每个 batch 的实际序列长度; -1 表示未提供, kernel 侧回退用 s1Size/s2Size */
    tilingData_.baseTiling.flashAttnBaseParams.maxSeqlenQ =
        (tilingInfo_->maxSeqLenQ >= 0) ? static_cast<int32_t>(tilingInfo_->maxSeqLenQ) : -1;
    tilingData_.baseTiling.flashAttnBaseParams.maxSeqlenKv =
        (tilingInfo_->maxSeqLenKv >= 0) ? static_cast<int32_t>(tilingInfo_->maxSeqLenKv) : -1;
    tilingData_.baseTiling.flashAttnBaseParams.scaleValue = tilingInfo_->softmaxScale;
    tilingData_.baseTiling.flashAttnBaseParams.isSoftMaxLseEnable = (tilingInfo_->returnSoftmaxLse > 0);
    tilingData_.baseTiling.flashAttnBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.flashAttnBaseParams.outputLayout = (uint32_t)(tilingInfo_->layoutOut);

    tilingData_.baseTiling.flashAttnAttenMaskParams.sparseMode = tilingInfo_->maskMode;
    tilingData_.baseTiling.flashAttnAttenMaskParams.winLefts = tilingInfo_->winLeft;
    tilingData_.baseTiling.flashAttnAttenMaskParams.winRights = tilingInfo_->winRight;
    tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS1Size = (tilingInfo_->maskMode != 0) ? 2048 : 0;
    tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS2Size = (tilingInfo_->maskMode != 0) ? 2048 : 0;
    tilingData_.baseTiling.flashAttnAttenMaskParams.isRowInvalid = true;

    static_assert(static_cast<uint32_t>(FiaLayout::BnBsND) == 23, "PA_LAYOUT_BBND mismatch");
    static_assert(static_cast<uint32_t>(FiaLayout::BnNBsD) == 12, "PA_LAYOUT_BNBD mismatch");
    static_assert(static_cast<uint32_t>(FiaLayout::NZ) == 3, "PA_LAYOUT_NZ mismatch");
    tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = (uint8_t)(tilingInfo_->layoutKV);
    tilingData_.baseTiling.flashAttnPageAttentionParams.blockSize = tilingInfo_->blockSize;
    tilingData_.baseTiling.flashAttnPageAttentionParams.maxBlockNumPerBatch = tilingInfo_->maxBlockNumPerBatch;

    int64_t outSize = tilingInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize =
        tilingInfo_->returnSoftmaxLse ? tilingInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.baseTiling.flashAttnEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.flashAttnEmptyTensorParams.needInit = false;
}

ge::graphStatus QuantFlashAttnTilingDn::SetTilingData(QuantFlashAttnTilingData& tilingData)
{
    QuantFlashAttnTilingData* tiling = context_->GetTilingData<QuantFlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(tilingInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_FIA(QuantFlashAttn, QuantFlashAttnTilingDn,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 210);
} // namespace optiling
