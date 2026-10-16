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
 * \file quant_flash_mla_tiling_arch35.cpp
 * \brief QuantFlashMla arch35 tiling implementation (MLA_FP8_E4M3_FULLQUANT)
 */

#include "quant_flash_mla_tiling_arch35.h"
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace quant_flash_mla {

// MLA FP8 全量化固定切分: SOUTER=32, SINNER=128
constexpr uint32_t MLA_SOUTER_32 = 32;
constexpr uint32_t MLA_SINNER_128 = 128;
// MLA preload次数
constexpr uint64_t MLA_PRE_LOAD_NUM = 3;
// MLA workspace: 每核最多2次写workspace
constexpr uint32_t MLA_WS_WRITE_NUM_PER_CORE = 2;
constexpr uint64_t MLA_CACHELINE_ALIGN_SIZE = 64;
constexpr uint64_t MLA_RES_LSE_NUM = 2; // ResLse有2份，sum和max

void QuantFlashMlaTilingFp8Impl::InitTilingInfo(TilingInfo* tilingInfo)
{
    qfmlaInfo_ = static_cast<QfmlaTilingInfo*>(tilingInfo);
}

bool QuantFlashMlaTilingFp8Impl::IsCapable()
{
    if (qfmlaInfo_ == nullptr) {
        return false;
    }
    if (qfmlaInfo_->quantMode != QfmlaQuantMode::MLA_FP8_E4M3_FULLQUANT) {
        return false;
    }
    if (qfmlaInfo_->qType != ge::DT_FLOAT8_E4M3FN) {
        return false;
    }
    return true;
}

void QuantFlashMlaTilingFp8Impl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(qfmlaInfo_->opName, "QuantFlashMla(MLA FP8 Fullquant) schedule mode: %u.",
            static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus QuantFlashMlaTilingFp8Impl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(qfmlaInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    InitImplParam();
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

ge::graphStatus QuantFlashMlaTilingFp8Impl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(qfmlaInfo_->opName, "The platformInfoPtr is null!"),
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
    OP_LOGI(qfmlaInfo_->opName, "MLA AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void QuantFlashMlaTilingFp8Impl::InitImplParam()
{
    // Online static compilation provides input shapes without runtime tensor data.
    const auto* cuSeqLenQShape = qfmlaInfo_->opParamInfo.cuSeqlensQ.shape;
    cuSeqLenQFlag_ = cuSeqLenQShape != nullptr && cuSeqLenQShape->GetStorageShape().GetShapeSize() > 0;
    const auto* seqUsedQShape = qfmlaInfo_->opParamInfo.sequsedQ.shape;
    seqUsedQFlag_ = seqUsedQShape != nullptr && seqUsedQShape->GetStorageShape().GetShapeSize() > 0;

    // MLA固定切分
    sOuterFactor_ = MLA_SOUTER_32;
    sInnerFactor_ = 256U;
    OP_LOGI(qfmlaInfo_->opName, "MLA Souter:%u SInner:%u", sOuterFactor_, sInnerFactor_);

    CalcNumBlocks(platformInfo_.aicNum);
}

void QuantFlashMlaTilingFp8Impl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(qfmlaInfo_->opName, "MLA QuantFlashMla block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void QuantFlashMlaTilingFp8Impl::UpdateTilingKeyLayout()
{
    // MLA 输入固定TND layout, 输出layout由layout_out属性决定（kernel侧为编译期模板参数）
    switch (qfmlaInfo_->layoutOut) {
        case QfmlaOutLayout::BSND:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_BSND;
            break;
        case QfmlaOutLayout::BNSD:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_BNSD;
            break;
        case QfmlaOutLayout::NTD:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_NTD;
            break;
        case QfmlaOutLayout::TND:
        default:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
            break;
    }
}

void QuantFlashMlaTilingFp8Impl::UpdateTilingKeyConfig()
{
    // MLA FP8 固定 config: S1Aligned64(M=64) S2Aligned256 DAligned576 DVAligned512
    tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned576_DVAligned512;
}

void QuantFlashMlaTilingFp8Impl::UpdateTilingKeyKvLayout()
{
    if (qfmlaInfo_->layoutKv == QfmlaKvLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBND;
    } else if (qfmlaInfo_->layoutKv == QfmlaKvLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else { // PA_NZ
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

void QuantFlashMlaTilingFp8Impl::UpdateTilingKeyQuantMode()
{
    tilingKeyInfo_.quantMode = QFMLA_MLA_FP8_E4M3_FULLQUANT;
}

void QuantFlashMlaTilingFp8Impl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyQuantMode();
    tilingKeyInfo_.hasAttenMask = qfmlaInfo_->attnMaskFlag;
    UpdateTilingKeyKvLayout();
    // MLA 分核由metadata算子完成，FD调度由kernel运行时读取metadata，tiling key固定false
    tilingKeyInfo_.isFd = true;
}

void QuantFlashMlaTilingFp8Impl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.quantMode,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd);

    OP_LOGI(qfmlaInfo_->opName, "MLA The tilingkey is %llu.", tilingKey_);
    OP_LOGI(qfmlaInfo_->opName,
            "MLA The tilingkey param is inOutLayoutType: %llu, config: %llu, quantMode: %llu, "
            "hasAttenMask: %u, kvLayoutType: %llu, isFd: %u.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd);
}

void QuantFlashMlaTilingFp8Impl::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    workspaceSize_ = sysWorkspaceSize;

    // MLA preload workspace: mSize * dVSize * fp32, 每核 PRE_LOAD_NUM 次
    constexpr uint32_t mSize = MLA_SOUTER_32 * 2; // M = sOuter * CV_RATIO, cvRatio=2
    constexpr uint32_t dVSize = 512;
    workspaceSize_ += platformInfo_.coreNum * MLA_PRE_LOAD_NUM * (mSize * dVSize) * sizeof(float);

    // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
    workspaceSize_ += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * MLA_CACHELINE_ALIGN_SIZE;

    // FD workspace（metadata调度可能产生FD任务，按最大情况预留）
    uint32_t faTmpAttenGmSize = numBlocks_ * MLA_WS_WRITE_NUM_PER_CORE * mSize * dVSize;
    uint32_t faTmpResLseGmSize = numBlocks_ * MLA_WS_WRITE_NUM_PER_CORE * mSize * 8;
    workspaceSize_ += (faTmpAttenGmSize + MLA_RES_LSE_NUM * faTmpResLseGmSize) * sizeof(float);
    tilingData_.workspaceParams.accumOutSize = faTmpAttenGmSize;
    tilingData_.workspaceParams.logSumExpSize = faTmpResLseGmSize;

    OP_LOGI(qfmlaInfo_->opName, "MLA Workspaces: %lu", workspaceSize_);
}

void QuantFlashMlaTilingFp8Impl::FillTiling()
{
    ComputeTilingData();
    SetQfmlaTilingData();
    PrintAllTilingData();
}

void QuantFlashMlaTilingFp8Impl::ComputeTilingData()
{
    tilingData_.attenMaskParams.maskMode = static_cast<uint8_t>(qfmlaInfo_->maskMode);

    if (qfmlaInfo_->attnMaskFlag) {
        tilingData_.attenMaskParams.attenMaskS1Size = static_cast<uint32_t>(qfmlaInfo_->attenMaskS1Size);
        tilingData_.attenMaskParams.attenMaskS2Size = static_cast<uint32_t>(qfmlaInfo_->attenMaskS2Size);
    } else {
        tilingData_.attenMaskParams.attenMaskS1Size = 0;
        tilingData_.attenMaskParams.attenMaskS2Size = 0;
    }

    // MLA 强制 PA 场景
    if (qfmlaInfo_->layoutKv == QfmlaKvLayout::PA_BBND) {
        tilingData_.pageAttentionParams.paLayoutType = 0;
    } else if (qfmlaInfo_->layoutKv == QfmlaKvLayout::PA_BNBD) {
        tilingData_.pageAttentionParams.paLayoutType = 1;
    } else { // PA_NZ
        tilingData_.pageAttentionParams.paLayoutType = 2;
    }
}

void QuantFlashMlaTilingFp8Impl::SetQfmlaTilingData()
{
    tilingData_.baseParams.bSize = static_cast<uint32_t>(qfmlaInfo_->bSize);
    tilingData_.baseParams.t1Size = static_cast<uint32_t>(qfmlaInfo_->qTSize);
    tilingData_.baseParams.t2Size = static_cast<uint32_t>(qfmlaInfo_->t2Size);
    tilingData_.baseParams.n1Size = static_cast<uint32_t>(qfmlaInfo_->n1Size);
    tilingData_.baseParams.n2Size = static_cast<uint32_t>(qfmlaInfo_->n2Size);
    tilingData_.baseParams.gSize = static_cast<uint32_t>(qfmlaInfo_->gSize);
    tilingData_.baseParams.s1Size = static_cast<uint32_t>(qfmlaInfo_->s1Size);
    tilingData_.baseParams.s2Size = static_cast<uint32_t>(qfmlaInfo_->s2Size);
    tilingData_.baseParams.dSize = static_cast<uint32_t>(qfmlaInfo_->headDimQk);
    tilingData_.baseParams.dSizeV = static_cast<uint32_t>(qfmlaInfo_->headDimV);
    tilingData_.baseParams.scaleValue = qfmlaInfo_->softmaxScale;
    // act_seq分裂: cu_seqlens_q（TND时必传）与seqused_q（可选）
    tilingData_.baseParams.cuSeqLensQSize = cuSeqLenQFlag_ ? static_cast<uint32_t>(qfmlaInfo_->bSize + 1) : 0;
    tilingData_.baseParams.seqUsedQSize = seqUsedQFlag_ ? static_cast<uint32_t>(qfmlaInfo_->bSize) : 0;
    tilingData_.baseParams.isSoftMaxLseEnable = qfmlaInfo_->returnSoftmaxLse;
    tilingData_.baseParams.isMetadataEnable = qfmlaInfo_->metadataFlag;
    tilingData_.baseParams.coreNum = numBlocks_;
    tilingData_.baseParams.outputLayout = static_cast<uint32_t>(qfmlaInfo_->layoutOut);

    // k_cache非连续stride: 0表示按shape推连续stride; V=K_nope复用, kernel侧valueStrides取keyStrides
    tilingData_.baseParams.keyStrides.bnStride =
        qfmlaInfo_->hasStride ? static_cast<uint64_t>(qfmlaInfo_->keyStrides->GetStride(0)) : 0ULL;
    tilingData_.baseParams.keyStrides.n2Stride =
        qfmlaInfo_->hasStride ? static_cast<uint64_t>(qfmlaInfo_->keyStrides->GetStride(1)) : 0ULL;

    tilingData_.pageAttentionParams.blockSize = static_cast<uint32_t>(qfmlaInfo_->blockSize);
    tilingData_.pageAttentionParams.maxBlockNumPerBatch = static_cast<uint32_t>(qfmlaInfo_->maxBlockNumPerBatch);

    int64_t outSize = qfmlaInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize =
        qfmlaInfo_->returnSoftmaxLse ? qfmlaInfo_->opParamInfo.softmaxLse.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.emptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.emptyTensorParams.totalOutputSize = static_cast<uint64_t>(std::max(outSize, static_cast<int64_t>(0)));
    tilingData_.emptyTensorParams.totalSoftMaxLseOutputSize =
        static_cast<uint64_t>(std::max(lseSize, static_cast<int64_t>(0)));
    tilingData_.emptyTensorParams.needInit = CheckNeedInitOutput();
}

bool QuantFlashMlaTilingFp8Impl::CheckNeedInitOutput() const
{
    // seqused_q存在时，可能存在无效行，需要init输出
    if (seqUsedQFlag_) {
        return true;
    }
    if (qfmlaInfo_->maskMode == QFMLA_MASK_MODE_NO_MASK) {
        return false;
    }
    if (qfmlaInfo_->maskMode == QFMLA_MASK_MODE_CAUSAL) {
        return qfmlaInfo_->s1Size > qfmlaInfo_->s2Size;
    }
    return false;
}

ge::graphStatus QuantFlashMlaTilingFp8Impl::SetTilingData(QuantFlashMlaTilingData& tilingData)
{
    QuantFlashMlaTilingData* tiling = context_->GetTilingData<QuantFlashMlaTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(qfmlaInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void QuantFlashMlaTilingFp8Impl::PrintAllTilingData()
{
    QuantFlashMlaBaseParams& params = tilingData_.baseParams;
    QuantFlashMlaAttenMaskParams& maskParams = tilingData_.attenMaskParams;
    QuantFlashMlaPageAttentionParams& paParams = tilingData_.pageAttentionParams;
    QuantFlashMlaWorkspaceParams& wsParams = tilingData_.workspaceParams;

    OP_LOGD(qfmlaInfo_->opName, "MLA bSize:%d", params.bSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA t1Size:%d", params.t1Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA t2Size:%d", params.t2Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA n1Size:%d", params.n1Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA n2Size:%d", params.n2Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA gSize:%d", params.gSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA s1Size:%d", params.s1Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA s2Size:%d", params.s2Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA dSize:%d", params.dSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA dSizeV:%d", params.dSizeV);
    OP_LOGD(qfmlaInfo_->opName, "MLA cuSeqLensQSize:%d", params.cuSeqLensQSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA seqUsedQSize:%d", params.seqUsedQSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA scaleValue:%f", params.scaleValue);
    OP_LOGD(qfmlaInfo_->opName, "MLA isSoftMaxLseEnable:%d", params.isSoftMaxLseEnable);
    OP_LOGD(qfmlaInfo_->opName, "MLA isMetadataEnable:%d", params.isMetadataEnable);
    OP_LOGD(qfmlaInfo_->opName, "MLA coreNum:%d", params.coreNum);
    OP_LOGD(qfmlaInfo_->opName, "MLA outputLayout:%d", params.outputLayout);

    OP_LOGD(qfmlaInfo_->opName, "MLA maskMode:%d", maskParams.maskMode);
    OP_LOGD(qfmlaInfo_->opName, "MLA attenMaskS1Size:%d", maskParams.attenMaskS1Size);
    OP_LOGD(qfmlaInfo_->opName, "MLA attenMaskS2Size:%d", maskParams.attenMaskS2Size);

    OP_LOGD(qfmlaInfo_->opName, "MLA paLayoutType:%d", paParams.paLayoutType);
    OP_LOGD(qfmlaInfo_->opName, "MLA blockSize:%d", paParams.blockSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA maxBlockNumPerBatch:%d", paParams.maxBlockNumPerBatch);

    OP_LOGD(qfmlaInfo_->opName, "MLA accumOutSize:%d", wsParams.accumOutSize);
    OP_LOGD(qfmlaInfo_->opName, "MLA logSumExpSize:%d", wsParams.logSumExpSize);

    OP_LOGD(qfmlaInfo_->opName, "MLA needInit:%d", tilingData_.emptyTensorParams.needInit);

    OP_LOGD(qfmlaInfo_->opName, "MLA tilingKey:%llu", tilingKey_);
}

} // namespace quant_flash_mla

using quant_flash_mla::QuantFlashMlaTilingFp8Impl;

REGISTER_TILING_TEMPLATE_FIA(QuantFlashMla, QuantFlashMlaTilingFp8Impl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 200);
} // namespace optiling
