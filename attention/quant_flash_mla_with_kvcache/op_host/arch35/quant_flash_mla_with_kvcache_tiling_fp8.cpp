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
 * \file quant_flash_mla_with_kvcache_tiling_fp8.cpp
 * \brief QuantFlashMlaWithKvcache arch35 tiling implementation (MLA_FP8_E4M3_FULLQUANT)
 */

#include "quant_flash_mla_with_kvcache_tiling_fp8.h"
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace quant_flash_mla_with_kvcache {

// MLA FP8 全量化固定切分: SOUTER=32, SINNER=128
constexpr uint32_t MLA_SOUTER_32 = 32;
constexpr uint32_t MLA_SINNER_128 = 128;
// MLA preload次数
constexpr uint64_t MLA_PRE_LOAD_NUM = 3;
// MLA workspace: 每核最多2次写workspace
constexpr uint32_t MLA_WS_WRITE_NUM_PER_CORE = 2;
constexpr uint64_t MLA_CACHELINE_ALIGN_SIZE = 64;
constexpr uint64_t MLA_RES_LSE_NUM = 2; // ResLse有2份，sum和max

void QuantFlashMlaWithKvcacheTilingFp8Impl::InitTilingInfo(TilingInfo* tilingInfo)
{
    qmlaInfo_ = static_cast<QmlaTilingInfo*>(tilingInfo);
}

bool QuantFlashMlaWithKvcacheTilingFp8Impl::IsCapable()
{
    if (qmlaInfo_ == nullptr) {
        return false;
    }
    if (qmlaInfo_->quantMode != QmlaQuantMode::MLA_FP8_E4M3_FULLQUANT) {
        return false;
    }
    if (qmlaInfo_->qType != ge::DT_FLOAT8_E4M3FN) {
        return false;
    }
    return true;
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(qmlaInfo_->opName, "QuantFlashMlaWithKvcache(MLA FP8 Fullquant) schedule mode: %u.",
            static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus QuantFlashMlaWithKvcacheTilingFp8Impl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(qmlaInfo_->opName, "Set plat memory info fail."),
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

ge::graphStatus QuantFlashMlaWithKvcacheTilingFp8Impl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(qmlaInfo_->opName, "The platformInfoPtr is null!"),
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
    OP_LOGI(qmlaInfo_->opName, "MLA AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::InitImplParam()
{
    // Online static compilation provides input shapes without runtime tensor data.
    const auto* cuSeqLenQShape = qmlaInfo_->opParamInfo.cuSeqlensQ.shape;
    cuSeqLenQFlag_ = cuSeqLenQShape != nullptr && cuSeqLenQShape->GetStorageShape().GetShapeSize() > 0;
    const auto* seqUsedQShape = qmlaInfo_->opParamInfo.sequsedQ.shape;
    seqUsedQFlag_ = seqUsedQShape != nullptr && seqUsedQShape->GetStorageShape().GetShapeSize() > 0;

    // MLA固定切分
    sOuterFactor_ = MLA_SOUTER_32;
    sInnerFactor_ = 256U;
    OP_LOGI(qmlaInfo_->opName, "MLA Souter:%u SInner:%u", sOuterFactor_, sInnerFactor_);

    CalcNumBlocks(platformInfo_.aicNum);
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(qmlaInfo_->opName, "MLA QuantFlashMlaWithKvcache block dim: %u aiv Num: %u aic Num: %u.", numBlocks_,
            aivNum, aicNum);
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::UpdateTilingKeyLayout()
{
    // MLA 输入固定TND layout, 输出layout由layout_out属性决定（kernel侧为编译期模板参数）
    switch (qmlaInfo_->layoutOut) {
        case QmlaOutLayout::BSND:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_BSND;
            break;
        case QmlaOutLayout::BNSD:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_BNSD;
            break;
        case QmlaOutLayout::NTD:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_NTD;
            break;
        case QmlaOutLayout::TND:
        default:
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
            break;
    }
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::UpdateTilingKeyConfig()
{
    // MLA FP8 固定 config: S1Aligned64(M=64) S2Aligned256 DAligned576 DVAligned512
    tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned576_DVAligned512;
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::UpdateTilingKeyKvLayout()
{
    if (qmlaInfo_->layoutKv == QmlaKvLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBND;
    } else if (qmlaInfo_->layoutKv == QmlaKvLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else { // PA_NZ
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::UpdateTilingKeyQuantMode()
{
    tilingKeyInfo_.quantMode = QMLA_MLA_FP8_E4M3_FULLQUANT;
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyQuantMode();
    tilingKeyInfo_.hasAttenMask = qmlaInfo_->attnMaskFlag;
    UpdateTilingKeyKvLayout();
    // MLA 分核由metadata算子完成，FD调度由kernel运行时读取metadata，tiling key固定false
    tilingKeyInfo_.isFd = true;
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.quantMode,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd);

    OP_LOGI(qmlaInfo_->opName, "MLA The tilingkey is %llu.", tilingKey_);
    OP_LOGI(qmlaInfo_->opName,
            "MLA The tilingkey param is inOutLayoutType: %llu, config: %llu, quantMode: %llu, "
            "hasAttenMask: %u, kvLayoutType: %llu, isFd: %u.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd);
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::CalcWorkspaceSize()
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

    OP_LOGI(qmlaInfo_->opName, "MLA Workspaces: %lu", workspaceSize_);
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::FillTiling()
{
    ComputeTilingData();
    SetQmlaTilingData();
    PrintAllTilingData();
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::ComputeTilingData()
{
    tilingData_.attenMaskParams.maskMode = static_cast<uint8_t>(qmlaInfo_->maskMode);

    if (qmlaInfo_->attnMaskFlag) {
        tilingData_.attenMaskParams.attenMaskS1Size = static_cast<uint32_t>(qmlaInfo_->attenMaskS1Size);
        tilingData_.attenMaskParams.attenMaskS2Size = static_cast<uint32_t>(qmlaInfo_->attenMaskS2Size);
    } else {
        tilingData_.attenMaskParams.attenMaskS1Size = 0;
        tilingData_.attenMaskParams.attenMaskS2Size = 0;
    }

    // MLA 强制 PA 场景
    if (qmlaInfo_->layoutKv == QmlaKvLayout::PA_BBND) {
        tilingData_.pageAttentionParams.paLayoutType = 0;
    } else if (qmlaInfo_->layoutKv == QmlaKvLayout::PA_BNBD) {
        tilingData_.pageAttentionParams.paLayoutType = 1;
    } else { // PA_NZ
        tilingData_.pageAttentionParams.paLayoutType = 2;
    }
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::SetQmlaTilingData()
{
    tilingData_.baseParams.bSize = static_cast<uint32_t>(qmlaInfo_->bSize);
    tilingData_.baseParams.t1Size = static_cast<uint32_t>(qmlaInfo_->qTSize);
    tilingData_.baseParams.t2Size = static_cast<uint32_t>(qmlaInfo_->t2Size);
    tilingData_.baseParams.n1Size = static_cast<uint32_t>(qmlaInfo_->n1Size);
    tilingData_.baseParams.n2Size = static_cast<uint32_t>(qmlaInfo_->n2Size);
    tilingData_.baseParams.gSize = static_cast<uint32_t>(qmlaInfo_->gSize);
    tilingData_.baseParams.s1Size = static_cast<uint32_t>(qmlaInfo_->s1Size);
    tilingData_.baseParams.s2Size = static_cast<uint32_t>(qmlaInfo_->s2Size);
    tilingData_.baseParams.dSize = static_cast<uint32_t>(qmlaInfo_->headDimQk);
    tilingData_.baseParams.dSizeV = static_cast<uint32_t>(qmlaInfo_->headDimV);
    tilingData_.baseParams.scaleValue = qmlaInfo_->softmaxScale;
    // act_seq分裂: cu_seqlens_q（TND时必传）与seqused_q（可选）
    tilingData_.baseParams.cuSeqLensQSize = cuSeqLenQFlag_ ? static_cast<uint32_t>(qmlaInfo_->bSize + 1) : 0;
    tilingData_.baseParams.seqUsedQSize = seqUsedQFlag_ ? static_cast<uint32_t>(qmlaInfo_->bSize) : 0;
    tilingData_.baseParams.isSoftMaxLseEnable = qmlaInfo_->returnSoftmaxLse;
    tilingData_.baseParams.isMetadataEnable = qmlaInfo_->metadataFlag;
    tilingData_.baseParams.coreNum = numBlocks_;
    tilingData_.baseParams.outputLayout = static_cast<uint32_t>(qmlaInfo_->layoutOut);

    // k_cache非连续stride: 0表示按shape推连续stride; V=K_nope复用, kernel侧valueStrides取keyStrides
    tilingData_.baseParams.keyStrides.bnStride =
        qmlaInfo_->hasStride ? static_cast<uint64_t>(qmlaInfo_->keyStrides->GetStride(0)) : 0ULL;
    tilingData_.baseParams.keyStrides.n2Stride =
        qmlaInfo_->hasStride ? static_cast<uint64_t>(qmlaInfo_->keyStrides->GetStride(1)) : 0ULL;

    tilingData_.pageAttentionParams.blockSize = static_cast<uint32_t>(qmlaInfo_->blockSize);
    tilingData_.pageAttentionParams.maxBlockNumPerBatch = static_cast<uint32_t>(qmlaInfo_->maxBlockNumPerBatch);

    int64_t outSize = qmlaInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize =
        qmlaInfo_->returnSoftmaxLse ? qmlaInfo_->opParamInfo.softmaxLse.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.emptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.emptyTensorParams.totalOutputSize = static_cast<uint64_t>(std::max(outSize, static_cast<int64_t>(0)));
    tilingData_.emptyTensorParams.totalSoftMaxLseOutputSize =
        static_cast<uint64_t>(std::max(lseSize, static_cast<int64_t>(0)));
    tilingData_.emptyTensorParams.needInit = CheckNeedInitOutput();
}

bool QuantFlashMlaWithKvcacheTilingFp8Impl::CheckNeedInitOutput() const
{
    // seqused_q存在时，可能存在无效行，需要init输出
    if (seqUsedQFlag_) {
        return true;
    }
    if (qmlaInfo_->maskMode == QMLA_MASK_MODE_NO_MASK) {
        return false;
    }
    if (qmlaInfo_->maskMode == QMLA_MASK_MODE_CAUSAL) {
        return qmlaInfo_->s1Size > qmlaInfo_->s2Size;
    }
    return false;
}

ge::graphStatus QuantFlashMlaWithKvcacheTilingFp8Impl::SetTilingData(QuantFlashMlaWithKvcacheTilingData& tilingData)
{
    QuantFlashMlaWithKvcacheTilingData* tiling = context_->GetTilingData<QuantFlashMlaWithKvcacheTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(qmlaInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void QuantFlashMlaWithKvcacheTilingFp8Impl::PrintAllTilingData()
{
    QuantFlashMlaWithKvcacheBaseParams& params = tilingData_.baseParams;
    QuantFlashMlaWithKvcacheAttenMaskParams& maskParams = tilingData_.attenMaskParams;
    QuantFlashMlaWithKvcachePageAttentionParams& paParams = tilingData_.pageAttentionParams;
    QuantFlashMlaWithKvcacheWorkspaceParams& wsParams = tilingData_.workspaceParams;

    OP_LOGD(qmlaInfo_->opName, "MLA bSize:%d", params.bSize);
    OP_LOGD(qmlaInfo_->opName, "MLA t1Size:%d", params.t1Size);
    OP_LOGD(qmlaInfo_->opName, "MLA t2Size:%d", params.t2Size);
    OP_LOGD(qmlaInfo_->opName, "MLA n1Size:%d", params.n1Size);
    OP_LOGD(qmlaInfo_->opName, "MLA n2Size:%d", params.n2Size);
    OP_LOGD(qmlaInfo_->opName, "MLA gSize:%d", params.gSize);
    OP_LOGD(qmlaInfo_->opName, "MLA s1Size:%d", params.s1Size);
    OP_LOGD(qmlaInfo_->opName, "MLA s2Size:%d", params.s2Size);
    OP_LOGD(qmlaInfo_->opName, "MLA dSize:%d", params.dSize);
    OP_LOGD(qmlaInfo_->opName, "MLA dSizeV:%d", params.dSizeV);
    OP_LOGD(qmlaInfo_->opName, "MLA cuSeqLensQSize:%d", params.cuSeqLensQSize);
    OP_LOGD(qmlaInfo_->opName, "MLA seqUsedQSize:%d", params.seqUsedQSize);
    OP_LOGD(qmlaInfo_->opName, "MLA scaleValue:%f", params.scaleValue);
    OP_LOGD(qmlaInfo_->opName, "MLA isSoftMaxLseEnable:%d", params.isSoftMaxLseEnable);
    OP_LOGD(qmlaInfo_->opName, "MLA isMetadataEnable:%d", params.isMetadataEnable);
    OP_LOGD(qmlaInfo_->opName, "MLA coreNum:%d", params.coreNum);
    OP_LOGD(qmlaInfo_->opName, "MLA outputLayout:%d", params.outputLayout);

    OP_LOGD(qmlaInfo_->opName, "MLA maskMode:%d", maskParams.maskMode);
    OP_LOGD(qmlaInfo_->opName, "MLA attenMaskS1Size:%d", maskParams.attenMaskS1Size);
    OP_LOGD(qmlaInfo_->opName, "MLA attenMaskS2Size:%d", maskParams.attenMaskS2Size);

    OP_LOGD(qmlaInfo_->opName, "MLA paLayoutType:%d", paParams.paLayoutType);
    OP_LOGD(qmlaInfo_->opName, "MLA blockSize:%d", paParams.blockSize);
    OP_LOGD(qmlaInfo_->opName, "MLA maxBlockNumPerBatch:%d", paParams.maxBlockNumPerBatch);

    OP_LOGD(qmlaInfo_->opName, "MLA accumOutSize:%d", wsParams.accumOutSize);
    OP_LOGD(qmlaInfo_->opName, "MLA logSumExpSize:%d", wsParams.logSumExpSize);

    OP_LOGD(qmlaInfo_->opName, "MLA needInit:%d", tilingData_.emptyTensorParams.needInit);

    OP_LOGD(qmlaInfo_->opName, "MLA tilingKey:%llu", tilingKey_);
}

} // namespace quant_flash_mla_with_kvcache

using quant_flash_mla_with_kvcache::QuantFlashMlaWithKvcacheTilingFp8Impl;

REGISTER_TILING_TEMPLATE_FIA(QuantFlashMlaWithKvcache, QuantFlashMlaWithKvcacheTilingFp8Impl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 200);
} // namespace optiling
