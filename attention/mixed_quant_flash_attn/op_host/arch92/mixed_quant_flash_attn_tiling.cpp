/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\file mixed_quant_flash_attn_tiling.cpp
 * \brief
 */

#include "mixed_quant_flash_attn_tiling_arch92.h"
#include "../mixed_quant_flash_attn_tiling.h"
#include "../mqfa_fa_adjust_sinner_souter.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../mixed_quant_flash_attn_tiling_utils.h"
#include "../../op_kernel/mixed_quant_flash_attn_template_tiling_key.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace mixed_quant_flash_attn {
constexpr uint64_t PRE_LOAD_NUM_GQA_ARCH35 = 3;

void MixedQuantFlashAttnTilingImpl::InitTilingInfo(TilingInfo* tilingInfo)
{
    faInfo_ = static_cast<FaTilingInfo*>(tilingInfo);
}

bool MixedQuantFlashAttnTilingImpl::IsCapable()
{
    return true;
}

void MixedQuantFlashAttnTilingImpl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(faInfo_->opName, "MixedQuantFlashAttn schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus MixedQuantFlashAttnTilingImpl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(faInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    InitImplParam();
    SplitPolicy();
    CalcScheduleMode();
    CalcWorkspaceSize();
    FillTiling();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MixedQuantFlashAttnTilingImpl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(faInfo_->opName, "The platformInfoPtr is null!"),
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
    OP_LOGI(faInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void MixedQuantFlashAttnTilingImpl::InitImplParam()
{
    const gert::Tensor* actSeqLenQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor* actSeqLenKV = faInfo_->opParamInfo.cuSeqlensKv.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
    actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
    cuSeqLenQFlag_ = !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int32_t>() == nullptr));
    cuSeqLenKVFlag_ =
        !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int32_t>() == nullptr));

    const gert::Tensor* seqUsedQ = faInfo_->opParamInfo.seqUsedQ.tensor;
    const gert::Tensor* seqUsedKv = faInfo_->opParamInfo.seqUsedKv.tensor;
    uint32_t seqUsedQDims = (seqUsedQ != nullptr) ? seqUsedQ->GetShapeSize() : 0;
    uint32_t seqUsedKvDims = (seqUsedKv != nullptr) ? seqUsedKv->GetShapeSize() : 0;
    seqUsedQFlag_ = !((seqUsedQDims == 0) || (seqUsedQ == nullptr) || (seqUsedQ->GetData<int32_t>() == nullptr));
    seqUsedKvFlag_ = !((seqUsedKvDims == 0) || (seqUsedKv == nullptr) || (seqUsedKv->GetData<int32_t>() == nullptr));
}

void MixedQuantFlashAttnTilingImpl::SplitPolicy()
{
    int64_t winLeft = faInfo_->winLeft;
    int64_t winRight = faInfo_->winRight;
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK) ||
        faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        winLeft = MASK_MODE_INT_MAX;
        winRight = MASK_MODE_INT_MAX;
    }
    fa_tiling_util::AdjustSinnerAndSouter(static_cast<uint32_t>(faInfo_->vHeadDim),
                                          static_cast<uint32_t>(faInfo_->gSize), faInfo_->maxSeqQ, faInfo_->maxSeqKv,
                                          static_cast<int32_t>(faInfo_->maskMode), winLeft, winRight,
                                          static_cast<uint32_t>(faInfo_->qLayout), sOuterFactor_, sInnerFactor_);
    CalcNumBlocks(platformInfo_.aicNum);
    flashDecodeFlag_ = true;
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyConfig()
{
    auto sOuter = sOuterFactor_ * platformInfo_.cvRatio;
    auto sInner = sInnerFactor_;
    auto dSize = faInfo_->qkHeadDim;
    auto dVsize = faInfo_->vHeadDim;
    if (dSize <= arch35FA::DSIZE_64)
        dSize = arch35FA::DSIZE_64;
    else if (dSize <= arch35FA::DSIZE_128)
        dSize = arch35FA::DSIZE_128;
    else if (dSize <= arch35FA::DSIZE_256)
        dSize = arch35FA::DSIZE_256;
    else if (dSize <= arch35FA::DSIZE_512)
        dSize = arch35FA::DSIZE_512;

    if (dVsize <= arch35FA::DSIZE_64)
        dVsize = arch35FA::DSIZE_64;
    else if (dVsize <= arch35FA::DSIZE_128)
        dVsize = arch35FA::DSIZE_128;
    else if (dVsize <= arch35FA::DSIZE_256)
        dVsize = arch35FA::DSIZE_256;
    else if (dVsize <= arch35FA::DSIZE_512)
        dVsize = arch35FA::DSIZE_512;
    if (sOuter == arch35FA::SOUTER_32 && sInner == arch35FA::SINNER_512 && dSize == arch35FA::DSIZE_128 &&
        dVsize == arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128;
    } else if (sOuter == arch35FA::SOUTER_48 && sInner == arch35FA::SINNER_512 && dSize == arch35FA::DSIZE_128 &&
               dVsize == arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned48_S2Aligned512_DAligned128_DVAligned128;
    } else if (sOuter == arch35FA::SOUTER_32 && sInner == arch35FA::SINNER_256 && dSize == arch35FA::DSIZE_128 &&
               dVsize == arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned32_S2Aligned256_DAligned128_DVAligned128;
    } else if (sOuter == arch35FA::SOUTER_64 && sInner == arch35FA::SINNER_256 && dSize == arch35FA::DSIZE_128 &&
               dVsize == arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128;
    } else if (sOuter == arch35FA::SOUTER_64 && sInner == arch35FA::SINNER_512 && dSize == arch35FA::DSIZE_128 &&
               dVsize == arch35FA::DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128; // qkvd不等长
    }
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyLayout()
{
    if (faInfo_->outLayout == FaLayout::BSND && faInfo_->qLayout == FaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BSND;
    } else if (faInfo_->outLayout == FaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD;
    } else if (faInfo_->outLayout == FaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND;
    } else {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND;
    }
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyKvLayout()
{
    if (faInfo_->kvLayout == FaLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBH;
    } else if (faInfo_->kvLayout == FaLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else if (faInfo_->kvLayout == FaLayout::PA_NZ) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

void MixedQuantFlashAttnTilingImpl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyKvLayout();
    UpdateTilingKeyConfig();
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;
    tilingKeyInfo_.quantComputeMode = static_cast<uint64_t>(faInfo_->quantComputeMode);
}
void MixedQuantFlashAttnTilingImpl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ =
        GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask,
                           tilingKeyInfo_.config, tilingKeyInfo_.quantComputeMode);
    OP_LOGI(faInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    fprintf(stderr,
            "[MQFA_HOST_TILING] tilingKey=%llu inOutLayout=%llu kvLayout=%llu hasMask=%llu config=%llu "
            "quantComputeMode=%llu\\n",
            tilingKey_, tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.config, tilingKeyInfo_.quantComputeMode);
    fflush(stderr);
    OP_LOGI(faInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, kvLayoutType: %llu, hasAttenMask: %llu, config: %llu, "
            "quantComputeMode: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.config,
            tilingKeyInfo_.quantComputeMode);
}

void MixedQuantFlashAttnTilingImpl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(faInfo_->opName, "MixedQuantFlashAttn block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
    fprintf(stderr, "[MQFA_HOST_TILING] numBlocks=%u aivNum=%u aicNum=%u cvRatio=%u\\n", numBlocks_, aivNum, aicNum,
            platformInfo_.cvRatio);
    fflush(stderr);
}

void MixedQuantFlashAttnTilingImpl::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    uint32_t mSize = sOuterFactor_ * platformInfo_.cvRatio;
    uint32_t dSize = faInfo_->vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= arch35FA::DSIZE_64) {
        dVBasicBlock = arch35FA::DSIZE_64;
    } else if (dSize <= arch35FA::DSIZE_128) {
        dVBasicBlock = arch35FA::DSIZE_128;
    } else if (dSize <= arch35FA::DSIZE_256) {
        dVBasicBlock = arch35FA::DSIZE_256;
    } else if (dSize <= arch35FA::DSIZE_512) {
        dVBasicBlock = arch35FA::DSIZE_512;
    }

    workspaceSize_ = sysWorkspaceSize;

    if (flashDecodeFlag_) {
        uint32_t faTmpAttenGmSize = platformInfo_.coreNum * 2 * mSize * dSize; // 每个核最多有2次写到workspace
        uint32_t fatmpResLseGmSize = platformInfo_.coreNum * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.mixedQuantFlashAttnWorkspaceParams.accumOutSize = faTmpAttenGmSize;
        tilingData_.baseTiling.mixedQuantFlashAttnWorkspaceParams.logSumExpSize = fatmpResLseGmSize;
    }

    OP_LOGI(faInfo_->opName, "Workspaces: %ld", workspaceSize_);
}

void MixedQuantFlashAttnTilingImpl::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
    PrintAllTilingData();
}

void MixedQuantFlashAttnTilingImpl::ComputeTilingData()
{
    tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.sparseMode = faInfo_->maskMode;
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;

    if (tilingKeyInfo_.hasAttenMask) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || faInfo_->s1Size == 1) {
            maskBatch = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(0);
        }
        maskS2Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.attenMaskS2Size = maskS2Size;
    }

    if (faInfo_->pageAttentionFlag) {
        if (faInfo_->kvLayout == FaLayout::PA_BBND) {
            tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.paLayoutType = 1;
        } else if (faInfo_->kvLayout == FaLayout::PA_BNBD) {
            tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.paLayoutType = 0;
        } else if (faInfo_->kvLayout == FaLayout::PA_NZ) {
            tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.paLayoutType = 2;
        }
    }

    tilingData_.baseTiling.mixedQuantFlashAttnQuantParams.quantComputeMode =
        static_cast<uint32_t>(faInfo_->quantComputeMode);
}

bool MixedQuantFlashAttnTilingImpl::CheckNeedInitOutput() const
{
    // varlen场景可能存在长度为0的batch, 对应行无计算, 必须清零
    if (seqUsedQFlag_ || seqUsedKvFlag_) {
        return true;
    }
    // TND变长: 各batch的s1/s2比不同, NO_MASK时每行至少算到actSeqLensKv, 有mask时可能存在空行
    if (faInfo_->qLayout == FaLayout::TND && faInfo_->kvLayout == FaLayout::TND) {
        return faInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK);
    }
    // 非TND固定shape: 按mask模式判断是否存在整行被mask掉的query块
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        return false;
    }
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        // 下三角: s1>s2时首个query块的窗口全在s2负侧
        return faInfo_->s1Size > faInfo_->s2Size;
    }
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::BAND)) {
        // winRight==-1表示右窗口无限大(被转为MASK_MODE_INT_MAX传给kernel), 每行必有有效S2块
        if (faInfo_->winRight == -1) {
            return false;
        }
        // 窗口右端=s2Size-s1Size+winRight, s1远大于s2时右端为负, 整行被mask
        return (faInfo_->s1Size - faInfo_->s2Size) > faInfo_->winRight;
    }
    return false;
}

void MixedQuantFlashAttnTilingImpl::SetFATilingData()
{
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.t2Size = faInfo_->kTSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.dSize = faInfo_->qkHeadDim;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.dSizeV = faInfo_->vHeadDim;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.scaleValue = faInfo_->softmaxScale;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.cuSeqLensQSize =
        faInfo_->qLayout == FaLayout::TND ? faInfo_->cuSeqLensQSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.cuSeqLensKVSize =
        faInfo_->kvLayout == FaLayout::TND ? faInfo_->bSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.seqUsedQSize = seqUsedQFlag_ ? faInfo_->seqUsedQSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.seqUsedKvSize = seqUsedKvFlag_ ? faInfo_->seqUsedKvSize : 0;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.isKvContinuous = true;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.iscuSeqLengthsNull = !cuSeqLenQFlag_;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.iscuSeqLengthsKVNull = !cuSeqLenKVFlag_;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.coreNum = numBlocks_;

    tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winLefts = faInfo_->winLeft;
    tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winRights = faInfo_->winRight;
    if (faInfo_->winLeft == -1) {
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winLefts = MASK_MODE_INT_MAX;
    }
    if (faInfo_->winRight == -1) {
        tilingData_.baseTiling.mixedQuantFlashAttnAttenMaskParams.winRights = MASK_MODE_INT_MAX;
    }
    tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    if (faInfo_->pageAttentionFlag) {
        maxBlockNumPerBatch = faInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    tilingData_.baseTiling.mixedQuantFlashAttnPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;
    tilingData_.baseTiling.mixedQuantFlashAttnBaseParams.needInitOutput = CheckNeedInitOutput();
}

ge::graphStatus MixedQuantFlashAttnTilingImpl::SetTilingData(MixedQuantFlashAttnTilingData& tilingData)
{
    MixedQuantFlashAttnTilingData* tiling = context_->GetTilingData<MixedQuantFlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void MixedQuantFlashAttnTilingImpl::PrintAllTilingData()
{
    MixedQuantFlashAttnTiling& baseTiling = tilingData_.baseTiling;
    MixedQuantFlashAttnBaseParams& mixedQuantFlashAttnBaseParams = baseTiling.mixedQuantFlashAttnBaseParams;
    MixedQuantFlashAttnAttenMaskParams& mixedQuantFlashAttnAttenMaskParams =
        baseTiling.mixedQuantFlashAttnAttenMaskParams;
    MixedQuantFlashAttnPageAttentionParams& mixedQuantFlashAttnPageAttentionParams =
        baseTiling.mixedQuantFlashAttnPageAttentionParams;
    MixedQuantFlashAttnWorkspaceParams& mixedQuantFlashAttnWorkspaceParams =
        baseTiling.mixedQuantFlashAttnWorkspaceParams;
    MixedQuantFlashAttnS1OuterSplitCoreParams& mixedQuantFlashAttnS1OuterSplitCoreParams =
        baseTiling.mixedQuantFlashAttnS1OuterSplitCoreParams;
    MixedQuantFlashAttnQuantParams& mixedQuantFlashAttnQuantParams = baseTiling.mixedQuantFlashAttnQuantParams;

    OP_LOGD(faInfo_->opName, "bSize:%d", mixedQuantFlashAttnBaseParams.bSize);
    OP_LOGD(faInfo_->opName, "t1Size:%d", mixedQuantFlashAttnBaseParams.t1Size);
    OP_LOGD(faInfo_->opName, "t2Size:%d", mixedQuantFlashAttnBaseParams.t2Size);
    OP_LOGD(faInfo_->opName, "n2Size:%d", mixedQuantFlashAttnBaseParams.n2Size);
    OP_LOGD(faInfo_->opName, "gSize:%d", mixedQuantFlashAttnBaseParams.gSize);
    OP_LOGD(faInfo_->opName, "s1Size:%d", mixedQuantFlashAttnBaseParams.s1Size);
    OP_LOGD(faInfo_->opName, "s2Size:%d", mixedQuantFlashAttnBaseParams.s2Size);
    OP_LOGD(faInfo_->opName, "dSize:%d", mixedQuantFlashAttnBaseParams.dSize);
    OP_LOGD(faInfo_->opName, "dSizeV:%d", mixedQuantFlashAttnBaseParams.dSizeV);
    OP_LOGD(faInfo_->opName, "dSizeRope:%d", mixedQuantFlashAttnBaseParams.dSizeRope);
    OP_LOGD(faInfo_->opName, "cuSeqLensQSize:%d", mixedQuantFlashAttnBaseParams.cuSeqLensQSize);
    OP_LOGD(faInfo_->opName, "cuSeqLensKVSize:%d", mixedQuantFlashAttnBaseParams.cuSeqLensKVSize);
    OP_LOGD(faInfo_->opName, "seqUsedQSize:%d", mixedQuantFlashAttnBaseParams.seqUsedQSize);
    OP_LOGD(faInfo_->opName, "seqUsedKvSize:%d", mixedQuantFlashAttnBaseParams.seqUsedKvSize);
    OP_LOGD(faInfo_->opName, "scaleValue:%f", mixedQuantFlashAttnBaseParams.scaleValue);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsNull:%d", mixedQuantFlashAttnBaseParams.iscuSeqLengthsNull);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsKVNull:%d", mixedQuantFlashAttnBaseParams.iscuSeqLengthsKVNull);
    OP_LOGD(faInfo_->opName, "isKvContinuous:%d", mixedQuantFlashAttnBaseParams.isKvContinuous);
    OP_LOGD(faInfo_->opName, "isSoftMaxLseEnable:%d", mixedQuantFlashAttnBaseParams.isSoftMaxLseEnable);
    OP_LOGD(faInfo_->opName, "coreNum:%d", mixedQuantFlashAttnBaseParams.coreNum);

    OP_LOGD(faInfo_->opName, "maskMode:%d", mixedQuantFlashAttnAttenMaskParams.sparseMode);
    OP_LOGD(faInfo_->opName, "winLefts:%d", mixedQuantFlashAttnAttenMaskParams.winLefts);
    OP_LOGD(faInfo_->opName, "winRights:%d", mixedQuantFlashAttnAttenMaskParams.winRights);
    OP_LOGD(faInfo_->opName, "attenMaskS1Size:%d", mixedQuantFlashAttnAttenMaskParams.attenMaskS1Size);
    OP_LOGD(faInfo_->opName, "attenMaskS2Size:%d", mixedQuantFlashAttnAttenMaskParams.attenMaskS2Size);

    OP_LOGD(faInfo_->opName, "paLayoutType:%d", mixedQuantFlashAttnPageAttentionParams.paLayoutType);
    OP_LOGD(faInfo_->opName, "blockSize:%d", mixedQuantFlashAttnPageAttentionParams.blockSize);
    OP_LOGD(faInfo_->opName, "maxBlockNumPerBatch:%d", mixedQuantFlashAttnPageAttentionParams.maxBlockNumPerBatch);

    OP_LOGD(faInfo_->opName, "accumOutSize:%d", mixedQuantFlashAttnWorkspaceParams.accumOutSize);
    OP_LOGD(faInfo_->opName, "logSumExpSize:%d", mixedQuantFlashAttnWorkspaceParams.logSumExpSize);

    OP_LOGD(faInfo_->opName, "totalSize:%d", mixedQuantFlashAttnS1OuterSplitCoreParams.totalSize);

    OP_LOGD(faInfo_->opName, "quantComputeMode:%d", mixedQuantFlashAttnQuantParams.quantComputeMode);

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(faInfo_->opName, "Tiling Data context_ GetCapacity: %lu.", cap);
}

} // namespace mixed_quant_flash_attn

using mixed_quant_flash_attn::MixedQuantFlashAttnTilingImpl;

REGISTER_TILING_TEMPLATE_FIA(MixedQuantFlashAttn, MixedQuantFlashAttnTilingImpl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_9201)}), 1);
} // namespace optiling
