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
 * \file flash_mla_tiling.cpp
 * \brief FlashMla arch92 tiling implementation.
 */

#include "flash_mla_tiling_arch92.h"
#include "../flash_mla_tiling.h"
#include "../../op_kernel/utils/flash_mla_config.h"
#include <securec.h>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../flash_mla_tiling_utils.h"
#include "../../op_kernel/arch35/flash_mla_template_tiling_key.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

namespace optiling {
namespace flash_mla {

constexpr uint64_t PRE_LOAD_NUM_MLA_ARCH92 = 2ULL;

void FlashMlaTilingImplArch92::InitTilingInfo(TilingInfo* tilingInfo)
{
    faInfo_ = static_cast<FlashMlaTilingInfo*>(tilingInfo);
}

bool FlashMlaTilingImplArch92::IsCapable()
{
    return true;
}

void FlashMlaTilingImplArch92::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(faInfo_->opName, "FlashMla schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FlashMlaTilingImplArch92::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(faInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    const errno_t ret = memset_s(&tilingData_, sizeof(tilingData_), 0, sizeof(tilingData_));
    OP_CHECK_IF(ret != EOK, OP_LOGE(faInfo_->opName, "Failed to clear tiling data, error: %d.", ret),
                return ge::GRAPH_FAILED);

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    PrintAllTilingData();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaTilingImplArch92::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(faInfo_->opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(platformInfo_.aicNum == 0 || platformInfo_.aicNum > 64U || platformInfo_.aivNum != platformInfo_.aicNum,
                OP_LOGE(faInfo_->opName, "arch92 FlashMla requires positive AIC/AIV counts with CV 1:1"),
                return ge::GRAPH_FAILED);
    platformInfo_.cvRatio = FlashMlaConfig::ARCH92.cvRatio;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    OP_CHECK_IF(platformInfo_.ubSize < 327U * 1024U + 512U || platformInfo_.l1Size < 504U * 1024U ||
                    platformInfo_.l0aSize < 64U * 1024U || platformInfo_.l0bSize < 64U * 1024U ||
                    platformInfo_.l0cSize < 256U * 1024U,
                OP_LOGE(faInfo_->opName, "Insufficient local memory for arch92 FlashMla M64/S2=128"),
                return ge::GRAPH_FAILED);
    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(faInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void FlashMlaTilingImplArch92::InitImplParam()
{
    const gert::Tensor* cuSeqLenQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor* seqUsedQ = faInfo_->opParamInfo.sequsedQ.tensor;
    const gert::Tensor* cacheSeqlens = faInfo_->opParamInfo.cacheSeqlens.tensor;
    cuSeqLenQFlag_ = cuSeqLenQ != nullptr && cuSeqLenQ->GetShapeSize() > 0;
    seqUsedQFlag_ = seqUsedQ != nullptr && seqUsedQ->GetShapeSize() > 0;
    cacheSeqlensFlag_ = cacheSeqlens != nullptr && cacheSeqlens->GetShapeSize() > 0;
}

void FlashMlaTilingImplArch92::SplitPolicy()
{
    // MLA uses M64/S2=128; one AIV handles all 64 rows. The metadata producer
    // owns task partitioning, and the kernel executes its FA/FD sections.
    sOuterFactor_ = FlashMlaConfig::ARCH92.m;
    CalcNumBlocks(platformInfo_.aicNum);
}

void FlashMlaTilingImplArch92::UpdateTilingKeyConfig()
{
    tilingKeyInfo_.config = Config_S1Aligned96_S2Aligned112_DAligned576_DVAligned512;
}

void FlashMlaTilingImplArch92::UpdateTilingKeyLayout()
{
    if (faInfo_->qLayout == FlashMlaLayout::TND) {
        if (faInfo_->outLayout == FlashMlaLayout::NTD) {
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_NTD;
        } else {
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND;
        }
    } else if (faInfo_->qLayout == FlashMlaLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND;
    } else if (faInfo_->qLayout == FlashMlaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD;
    }
}

void FlashMlaTilingImplArch92::UpdateTilingKeyKvLayout()
{
    if (faInfo_->kvLayout == FlashMlaLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBND;
    } else if (faInfo_->kvLayout == FlashMlaLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else if (faInfo_->kvLayout == FlashMlaLayout::PA_NZ) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

void FlashMlaTilingImplArch92::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.hasAttenMask = faInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK);
}

void FlashMlaTilingImplArch92::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.config);

    OP_LOGI(faInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(faInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, kvLayoutType: %llu, hasAttenMask: %u, config: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.config);
}

void FlashMlaTilingImplArch92::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(faInfo_->opName, "FlashMla block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void FlashMlaTilingImplArch92::CalcWorkspaceSize()
{
    const uint64_t mSize = static_cast<uint64_t>(sOuterFactor_) * platformInfo_.cvRatio;
    const uint64_t dSize = faInfo_->vHeadDim;
    constexpr uint64_t lseSize = 8ULL;

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    const uint64_t faTmpAttenGmSize =
        static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH92 * mSize * dSize; // 每个核最多有2次写到workspace
    const uint64_t faTmpResLseGmSize = static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH92 * mSize * lseSize;
    workspaceSize_ += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
    tilingData_.baseTiling.flashMlaWorkspaceParams.accumOutSize = static_cast<uint32_t>(faTmpAttenGmSize);
    tilingData_.baseTiling.flashMlaWorkspaceParams.logSumExpSize = static_cast<uint32_t>(faTmpResLseGmSize);

    OP_LOGI(faInfo_->opName, "Workspaces: %lu", workspaceSize_);
}

void FlashMlaTilingImplArch92::FillTiling()
{
    ComputeTilingData();
    FillBaseTilingData();
}

void FlashMlaTilingImplArch92::ComputeTilingData()
{
    auto& maskParams = tilingData_.baseTiling.flashMlaAttenMaskParams;
    maskParams.sparseMode = static_cast<uint8_t>(faInfo_->maskMode);
    maskParams.winLeft = MASK_MODE_INT_MAX;
    maskParams.winRight = MASK_MODE_INT_MAX;
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        maskParams.winRight = 0;
    } else if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::BAND)) {
        maskParams.winLeft = static_cast<int32_t>(faInfo_->winLeft == -1 ? MASK_MODE_INT_MAX : faInfo_->winLeft);
        maskParams.winRight = static_cast<int32_t>(faInfo_->winRight == -1 ? MASK_MODE_INT_MAX : faInfo_->winRight);
    }
    tilingKeyInfo_.hasAttenMask = faInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK);
    if (tilingKeyInfo_.hasAttenMask) {
        const auto& maskShape = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape();
        maskParams.attenMaskBatch = 1;
        maskParams.attenMaskS1Size = maskShape.GetDim(0);
        maskParams.attenMaskS2Size = maskShape.GetDim(1);
    }
}

void FlashMlaTilingImplArch92::FillBaseTilingData()
{
    tilingData_.baseTiling.flashMlaBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.flashMlaBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.flashMlaBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.flashMlaBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.flashMlaBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.flashMlaBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.flashMlaBaseParams.dSize = faInfo_->qkNopeHeadDim;
    tilingData_.baseTiling.flashMlaBaseParams.dSizeV = faInfo_->vHeadDim;
    tilingData_.baseTiling.flashMlaBaseParams.dSizeRope = faInfo_->ropeHeadDim;
    tilingData_.baseTiling.flashMlaBaseParams.scaleValue = faInfo_->softmaxScale;
    tilingData_.baseTiling.flashMlaBaseParams.actualSeqLengthsQSize =
        cuSeqLenQFlag_ ? static_cast<uint32_t>(faInfo_->cuSeqlensQSize) : 0;
    tilingData_.baseTiling.flashMlaBaseParams.actualSeqLengthsKVSize =
        cacheSeqlensFlag_ ? static_cast<uint32_t>(faInfo_->cacheSeqlensSize) : 0;
    tilingData_.baseTiling.flashMlaBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.flashMlaBaseParams.coreNum = numBlocks_;
    // 小任务关闭 L2 缓存，避免 KV 流式读取造成缓存填充和驱逐。
    tilingData_.baseTiling.flashMlaBaseParams.l2CacheOffFlag = (faInfo_->gSize * faInfo_->s1Size) <= 128 ? 1U : 0U;

    tilingData_.baseTiling.flashMlaPageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = faInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    tilingData_.baseTiling.flashMlaPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;

    tilingData_.baseTiling.flashMlaEmptyTensorParams.needInit = CheckNeedInitOutput() ? 1 : 0;

    if (faInfo_->hasViewStride) {
        tilingData_.baseTiling.flashMlaBaseParams.keyStrides.bnStride = faInfo_->keyBnStride;
        tilingData_.baseTiling.flashMlaBaseParams.keyStrides.n2Stride = faInfo_->keyN2Stride;
    }
}

bool FlashMlaTilingImplArch92::CheckNeedInitOutput() const
{
    if (seqUsedQFlag_ || cacheSeqlensFlag_) {
        return true;
    }
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        return faInfo_->s1Size > faInfo_->s2Size;
    }
    return false;
}

ge::graphStatus FlashMlaTilingImplArch92::SetTilingData(FlashMlaTilingData& tilingData)
{
    FlashMlaTilingData* tiling = context_->GetTilingData<FlashMlaTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FlashMlaTilingImplArch92::PrintAllTilingData()
{
    const auto& base = tilingData_.baseTiling.flashMlaBaseParams;
    const auto& mask = tilingData_.baseTiling.flashMlaAttenMaskParams;
    const auto& page = tilingData_.baseTiling.flashMlaPageAttentionParams;
    const auto& workspace = tilingData_.baseTiling.flashMlaWorkspaceParams;
    const auto& output = tilingData_.baseTiling.flashMlaEmptyTensorParams;
    OP_LOGD(faInfo_->opName, "B:%u T:%u N2:%u G:%u S1:%u S2:%u D:%u DV:%u Rope:%u", base.bSize, base.t1Size,
            base.n2Size, base.gSize, base.s1Size, base.s2Size, base.dSize, base.dSizeV, base.dSizeRope);
    OP_LOGD(faInfo_->opName, "QSeqDims:%u KVSeqDims:%u scale:%f LSE:%u cores:%u L2Off:%u", base.actualSeqLengthsQSize,
            base.actualSeqLengthsKVSize, base.scaleValue, base.isSoftMaxLseEnable, base.coreNum, base.l2CacheOffFlag);
    OP_LOGD(faInfo_->opName, "maskMode:%u maskS1:%u maskS2:%u winLeft:%d winRight:%d", mask.sparseMode,
            mask.attenMaskS1Size, mask.attenMaskS2Size, mask.winLeft, mask.winRight);
    OP_LOGD(faInfo_->opName, "blockSize:%u maxBlocks:%u", page.blockSize, page.maxBlockNumPerBatch);
    OP_LOGD(faInfo_->opName, "accumOutSize:%u logSumExpSize:%u needInit:%u", workspace.accumOutSize,
            workspace.logSumExpSize, output.needInit);
}

} // namespace flash_mla

using flash_mla::FlashMlaTilingImplArch92;

REGISTER_TILING_TEMPLATE_FIA(FlashMla, FlashMlaTilingImplArch92,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_9201),
                                                   static_cast<int32_t>(NpuArch::DAV_9202)}),
                             1);
} // namespace optiling
