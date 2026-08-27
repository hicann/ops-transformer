/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_tiling_hif8.cpp
 * \brief QuantFlashAttn arch35 tiling implementation (A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32)
 */

#include "quant_flash_attn_tiling_hif8.h"
#include "../quant_flash_attn_tiling.h"
#include "../qfa_adjust_sinner_souter.h"
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../quant_flash_attn_tiling_utils.h"
#include "../../op_kernel/arch35/quant_flash_attn_template_tiling_key.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "../quant_flash_attn_tiling_constants.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace quant_flash_attn {
using namespace arch35QFA;

void QuantFlashAttnTilingHif8Impl::InitTilingInfo(TilingInfo *tilingInfo)
{
    qfaInfo_ = static_cast<QfaTilingInfo *>(tilingInfo);
}

bool QuantFlashAttnTilingHif8Impl::IsCapable()
{
    if (qfaInfo_ == nullptr) {
        return false;
    }
    if (qfaInfo_->quantMode != QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) {
        return false;
    }
    return true;
}

void QuantFlashAttnTilingHif8Impl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(qfaInfo_->opName, "QuantFlashAttn(HIF8) schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus QuantFlashAttnTilingHif8Impl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(qfaInfo_->opName, "Set plat memory info fail."),
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

ge::graphStatus QuantFlashAttnTilingHif8Impl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(qfaInfo_->opName, "The platformInfoPtr is null!"),
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
    OP_LOGI(qfaInfo_->opName, "HIF8 AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingHif8Impl::InitImplParam()
{
    const gert::Tensor *actSeqLenQ = qfaInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *actSeqLenKV = qfaInfo_->opParamInfo.cuSeqlensKv.tensor;
    uint32_t actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
    uint32_t actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
    cuSeqLenQFlag_ = !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int32_t>() == nullptr));
    cuSeqLenKVFlag_ =
        !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int32_t>() == nullptr));

    const gert::Tensor *seqUsedQ = qfaInfo_->opParamInfo.sequsedQ.tensor;
    const gert::Tensor *seqUsedKv = qfaInfo_->opParamInfo.sequsedKv.tensor;
    uint32_t seqUsedQDims = (seqUsedQ != nullptr) ? seqUsedQ->GetShapeSize() : 0;
    uint32_t seqUsedKvDims = (seqUsedKv != nullptr) ? seqUsedKv->GetShapeSize() : 0;
    seqUsedQFlag_ = !((seqUsedQDims == 0) || (seqUsedQ == nullptr) || (seqUsedQ->GetData<int32_t>() == nullptr));
    seqUsedKvFlag_ = !((seqUsedKvDims == 0) || (seqUsedKv == nullptr) || (seqUsedKv->GetData<int32_t>() == nullptr));
}

void QuantFlashAttnTilingHif8Impl::SplitPolicy()
{
    qfa_tiling_util::AdjustSinnerAndSouter(qfaInfo_->vHeadDim, qfaInfo_->maxSeqQ, qfaInfo_->maxSeqKv,
                                           static_cast<int32_t>(qfaInfo_->maskMode), qfaInfo_->winLeft,
                                           qfaInfo_->winRight, static_cast<uint32_t>(qfaInfo_->qLayout),
                                           static_cast<uint32_t>(qfaInfo_->quantMode), sOuterFactor_, sInnerFactor_);
    CalcNumBlocks(platformInfo_.aicNum);
}

void QuantFlashAttnTilingHif8Impl::UpdateTilingKeyConfig()
{
    tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
}

void QuantFlashAttnTilingHif8Impl::UpdateTilingKeyLayout()
{
    if (qfaInfo_->qLayout == QfaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BNSD;
    } else if (qfaInfo_->qLayout == QfaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
    } else {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND_BSND;
    }
}

void QuantFlashAttnTilingHif8Impl::UpdateTilingKeyKvLayout()
{
    tilingKeyInfo_.kvLayoutType = KvLayoutType_NO_PA;
}

void QuantFlashAttnTilingHif8Impl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyQuantMode();
    tilingKeyInfo_.hasAttenMask = (qfaInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK));
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.isFd = false;
}

void QuantFlashAttnTilingHif8Impl::UpdateTilingKeyQuantMode()
{
    tilingKeyInfo_.quantMode = static_cast<uint64_t>(QFA_HIF8_FP32);
}

void QuantFlashAttnTilingHif8Impl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.quantMode,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd);

    OP_LOGI(qfaInfo_->opName, "HIF8 The tilingkey is %llu.", tilingKey_);
    OP_LOGI(qfaInfo_->opName,
            "HIF8 The tilingkey param is inOutLayoutType: %llu, config: %llu, quantMode: %llu, "
            "hasAttenMask: %u, kvLayoutType: %llu, isFd: %u.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd);
}

void QuantFlashAttnTilingHif8Impl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(qfaInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(qfaInfo_->opName, "HIF8 QuantFlashAttn block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void QuantFlashAttnTilingHif8Impl::CalcWorkspaceSize()
{
    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    OP_LOGI(qfaInfo_->opName, "HIF8 Workspaces: %lu", workspaceSize_);
}

void QuantFlashAttnTilingHif8Impl::FillTiling()
{
    ComputeTilingData();
    SetQFATilingData();
    PrintAllTilingData();
}

void QuantFlashAttnTilingHif8Impl::ComputeTilingData()
{
    tilingData_.baseTiling.quantFlashAttnAttenMaskParams.sparseMode = qfaInfo_->maskMode;

    if (qfaInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK)) {
        uint64_t maskDimNum = qfaInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = qfaInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        uint64_t maskS2Size = qfaInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        tilingData_.baseTiling.quantFlashAttnAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.quantFlashAttnAttenMaskParams.attenMaskS2Size = maskS2Size;
    }
}

void QuantFlashAttnTilingHif8Impl::SetQFATilingData()
{
    tilingData_.baseTiling.quantFlashAttnBaseParams.bSize = qfaInfo_->bSize;
    tilingData_.baseTiling.quantFlashAttnBaseParams.t1Size = qfaInfo_->qTSize;
    tilingData_.baseTiling.quantFlashAttnBaseParams.t2Size = qfaInfo_->kTSize;
    tilingData_.baseTiling.quantFlashAttnBaseParams.n2Size = qfaInfo_->n2Size;
    tilingData_.baseTiling.quantFlashAttnBaseParams.gSize = qfaInfo_->gSize;
    tilingData_.baseTiling.quantFlashAttnBaseParams.s1Size = qfaInfo_->s1Size;
    tilingData_.baseTiling.quantFlashAttnBaseParams.s2Size = qfaInfo_->s2Size;
    tilingData_.baseTiling.quantFlashAttnBaseParams.dSize = qfaInfo_->qkHeadDim;
    tilingData_.baseTiling.quantFlashAttnBaseParams.dSizeV = qfaInfo_->vHeadDim;
    tilingData_.baseTiling.quantFlashAttnBaseParams.scaleValue = qfaInfo_->softmaxScale;
    tilingData_.baseTiling.quantFlashAttnBaseParams.cuSeqLensQSize =
        (qfaInfo_->qLayout == QfaLayout::TND && cuSeqLenQFlag_) ? qfaInfo_->bSize + 1 : 0;
    tilingData_.baseTiling.quantFlashAttnBaseParams.cuSeqLensKVSize =
        (qfaInfo_->kvLayout == QfaLayout::TND && cuSeqLenKVFlag_) ? qfaInfo_->bSize + 1 : 0;
    tilingData_.baseTiling.quantFlashAttnBaseParams.seqUsedQSize = seqUsedQFlag_ ? qfaInfo_->bSize : 0;
    tilingData_.baseTiling.quantFlashAttnBaseParams.seqUsedKvSize = seqUsedKvFlag_ ? qfaInfo_->bSize : 0;
    tilingData_.baseTiling.quantFlashAttnBaseParams.isKvContinuous = true;
    tilingData_.baseTiling.quantFlashAttnBaseParams.isSoftMaxLseEnable = qfaInfo_->softmaxLseFlag;
    tilingData_.baseTiling.quantFlashAttnBaseParams.iscuSeqLengthsNull =
        !(cuSeqLenQFlag_ && qfaInfo_->qLayout == QfaLayout::TND);
    tilingData_.baseTiling.quantFlashAttnBaseParams.iscuSeqLengthsKVNull =
        !(cuSeqLenKVFlag_ && qfaInfo_->kvLayout == QfaLayout::TND);
    tilingData_.baseTiling.quantFlashAttnBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.quantFlashAttnBaseParams.outputLayout = static_cast<uint32_t>(qfaInfo_->outLayout);
    tilingData_.baseTiling.quantFlashAttnBaseParams.needInitOutput = CheckNeedInitOutput();

    tilingData_.baseTiling.quantFlashAttnAttenMaskParams.winLefts = qfaInfo_->winLeft;
    tilingData_.baseTiling.quantFlashAttnAttenMaskParams.winRights = qfaInfo_->winRight;
    if (qfaInfo_->winLeft == -1) {
        tilingData_.baseTiling.quantFlashAttnAttenMaskParams.winLefts = MASK_MODE_INT_MAX;
    }
    if (qfaInfo_->winRight == -1) {
        tilingData_.baseTiling.quantFlashAttnAttenMaskParams.winRights = MASK_MODE_INT_MAX;
    }
    tilingData_.baseTiling.quantFlashAttnPageAttentionParams.blockSize = qfaInfo_->blockSize;
    tilingData_.baseTiling.quantFlashAttnPageAttentionParams.maxBlockNumPerBatch = 0;
}

bool QuantFlashAttnTilingHif8Impl::CheckNeedInitOutput() const
{
    if (seqUsedQFlag_ || seqUsedKvFlag_) {
        return true;
    }
    if (qfaInfo_->qLayout == QfaLayout::TND && qfaInfo_->kvLayout == QfaLayout::TND) {
        return qfaInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK);
    }
    if (qfaInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        return false;
    }
    if (qfaInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        return qfaInfo_->s1Size > qfaInfo_->s2Size;
    }
    if (qfaInfo_->maskMode == static_cast<int64_t>(MaskMode::SLIDING_WINDOW)) {
        if (qfaInfo_->winRight == -1) {
            return false;
        }
        return (qfaInfo_->s1Size - qfaInfo_->s2Size) > qfaInfo_->winRight;
    }
    return false;
}

ge::graphStatus QuantFlashAttnTilingHif8Impl::SetTilingData(QuantFlashAttnTilingData &tilingData)
{
    QuantFlashAttnTilingData *tiling = context_->GetTilingData<QuantFlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(qfaInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void QuantFlashAttnTilingHif8Impl::PrintAllTilingData()
{
    QuantFlashAttnQuantTilingArch35 &baseTiling = tilingData_.baseTiling;
    QuantFlashAttnBaseParams &params = baseTiling.quantFlashAttnBaseParams;
    QuantFlashAttnAttenMaskParams &maskParams = baseTiling.quantFlashAttnAttenMaskParams;
    QuantFlashAttnPageAttentionParams &paParams = baseTiling.quantFlashAttnPageAttentionParams;
    QuantFlashAttnWorkspaceParams &wsParams = baseTiling.quantFlashAttnWorkspaceParams;

    OP_LOGD(qfaInfo_->opName, "HIF8 bSize:%d", params.bSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 t1Size:%d", params.t1Size);
    OP_LOGD(qfaInfo_->opName, "HIF8 t2Size:%d", params.t2Size);
    OP_LOGD(qfaInfo_->opName, "HIF8 n2Size:%d", params.n2Size);
    OP_LOGD(qfaInfo_->opName, "HIF8 gSize:%d", params.gSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 s1Size:%d", params.s1Size);
    OP_LOGD(qfaInfo_->opName, "HIF8 s2Size:%d", params.s2Size);
    OP_LOGD(qfaInfo_->opName, "HIF8 dSize:%d", params.dSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 dSizeV:%d", params.dSizeV);
    OP_LOGD(qfaInfo_->opName, "HIF8 cuSeqLensQSize:%d", params.cuSeqLensQSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 cuSeqLensKVSize:%d", params.cuSeqLensKVSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 seqUsedQSize:%d", params.seqUsedQSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 seqUsedKvSize:%d", params.seqUsedKvSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 scaleValue:%f", params.scaleValue);
    OP_LOGD(qfaInfo_->opName, "HIF8 iscuSeqLengthsNull:%d", params.iscuSeqLengthsNull);
    OP_LOGD(qfaInfo_->opName, "HIF8 iscuSeqLengthsKVNull:%d", params.iscuSeqLengthsKVNull);
    OP_LOGD(qfaInfo_->opName, "HIF8 isKvContinuous:%d", params.isKvContinuous);
    OP_LOGD(qfaInfo_->opName, "HIF8 isSoftMaxLseEnable:%d", params.isSoftMaxLseEnable);
    OP_LOGD(qfaInfo_->opName, "HIF8 coreNum:%d", params.coreNum);
    OP_LOGD(qfaInfo_->opName, "HIF8 outputLayout:%d", params.outputLayout);
    OP_LOGD(qfaInfo_->opName, "HIF8 needInitOutput:%d", params.needInitOutput);

    OP_LOGD(qfaInfo_->opName, "HIF8 maskMode:%d", maskParams.sparseMode);
    OP_LOGD(qfaInfo_->opName, "HIF8 winLefts:%d", maskParams.winLefts);
    OP_LOGD(qfaInfo_->opName, "HIF8 winRights:%d", maskParams.winRights);
    OP_LOGD(qfaInfo_->opName, "HIF8 attenMaskS1Size:%d", maskParams.attenMaskS1Size);
    OP_LOGD(qfaInfo_->opName, "HIF8 attenMaskS2Size:%d", maskParams.attenMaskS2Size);

    OP_LOGD(qfaInfo_->opName, "HIF8 paLayoutType:%d", paParams.paLayoutType);
    OP_LOGD(qfaInfo_->opName, "HIF8 blockSize:%d", paParams.blockSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 maxBlockNumPerBatch:%d", paParams.maxBlockNumPerBatch);

    OP_LOGD(qfaInfo_->opName, "HIF8 accumOutSize:%d", wsParams.accumOutSize);
    OP_LOGD(qfaInfo_->opName, "HIF8 logSumExpSize:%d", wsParams.logSumExpSize);

    OP_LOGD(qfaInfo_->opName, "HIF8 tilingKey:%llu", tilingKey_);
}

} // namespace quant_flash_attn

using quant_flash_attn::QuantFlashAttnTilingHif8Impl;

REGISTER_TILING_TEMPLATE_FIA(QuantFlashAttn, QuantFlashAttnTilingHif8Impl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 3);
} // namespace optiling
