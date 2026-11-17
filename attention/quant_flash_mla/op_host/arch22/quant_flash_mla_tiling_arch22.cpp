/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_flash_mla_tiling_arch22.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace quant_flash_mla {

namespace {
constexpr uint32_t IFA_BYTE_BLOCK = 32;
constexpr uint32_t IFA_BLOCK_TYPE_SIZE = 4;
constexpr uint32_t IFA_BALANCE_SG_BASIC_SIZE = 128;
constexpr uint32_t IFA_PRE_LOAD_NUM_MLA = 2;
constexpr uint32_t IFA_MLA_S_INNER = 512;
constexpr uint32_t IFA_MLA_HEAD_DIM = 576;
constexpr uint32_t IFA_G_MAX = 128;
constexpr uint32_t IFA_MM_PA_CALLBACK_SIZE = 64;
constexpr uint32_t IFA_QS_LIMIT_INT8 = 32;
} // namespace

REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataMla, IncreFlashAttentionTilingDataMla)

void QuantFlashMlaTilingInt8Impl::InitTilingInfo(TilingInfo* tilingInfo)
{
    qfmlaInfo_ = static_cast<QfmlaTilingInfo*>(tilingInfo);
}

bool QuantFlashMlaTilingInt8Impl::IsCapable()
{
    if (qfmlaInfo_ == nullptr) {
        return false;
    }
    return qfmlaInfo_->quantMode == QfmlaQuantMode::MLA_INT8_FULLQUANT && qfmlaInfo_->qType == ge::DT_INT8;
}

ge::graphStatus QuantFlashMlaTilingInt8Impl::SetPlatMemoryInfo()
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
    OP_LOGI(qfmlaInfo_->opName, "MLA int8 AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu",
            platformInfo_.aivNum, platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize,
            platformInfo_.l0cSize, platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void QuantFlashMlaTilingInt8Impl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(qfmlaInfo_->opName, "MLA int8 QuantFlashMla block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum,
            aicNum);
}

void QuantFlashMlaTilingInt8Impl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(qfmlaInfo_->opName, "QuantFlashMla(MLA int8 Fullquant) schedule mode: %u.",
            static_cast<uint32_t>(scheduleMode_));
}

void QuantFlashMlaTilingInt8Impl::FillTilingData()
{
    FillBaseParams();
    FillSplitKVParams();
    FillSingleCoreParams();
    PrintAllTilingData();
}

void QuantFlashMlaTilingInt8Impl::FillBaseParams()
{
    auto& baseParams = tilingData_.baseParams;
    baseParams.set_batchSize(static_cast<uint32_t>(qfmlaInfo_->bSize));
    const uint64_t seqSizeUpper =
        static_cast<uint64_t>(qfmlaInfo_->maxBlockNumPerBatch) * static_cast<uint64_t>(qfmlaInfo_->blockSize);
    baseParams.set_seqSize(static_cast<uint32_t>(seqSizeUpper));
    if (qfmlaInfo_->layoutQ == QfmlaLayout::TND) {
        qSeqSizeUpper_ = qfmlaInfo_->maxSeqQ > 0 ? static_cast<uint32_t>(qfmlaInfo_->maxSeqQ) : IFA_QS_LIMIT_INT8;
    } else {
        qSeqSizeUpper_ = static_cast<uint32_t>(qfmlaInfo_->s1Size);
    }
    baseParams.set_qSeqSize(qSeqSizeUpper_);
    baseParams.set_blockSize(static_cast<uint32_t>(qfmlaInfo_->blockSize));
    baseParams.set_maxBlockNumPerBatch(static_cast<uint32_t>(qfmlaInfo_->maxBlockNumPerBatch));
    baseParams.set_scaleValue(qfmlaInfo_->softmaxScale < 0.0f ? 1.0f / std::sqrt(static_cast<float>(IFA_MLA_HEAD_DIM)) :
                                                                qfmlaInfo_->softmaxScale);
    baseParams.set_nNumOfQInOneGroup(static_cast<uint32_t>(qfmlaInfo_->n1Size));
    baseParams.set_actualLenQDims(static_cast<uint32_t>(qfmlaInfo_->bSize));
    baseParams.set_actualLenDims(static_cast<uint32_t>(qfmlaInfo_->bSize));
    baseParams.set_antiquantMode(0);
    baseParams.set_attenMaskFlag(qfmlaInfo_->attnMaskFlag ? 1U : 0U);
    baseParams.set_attenMaskSize(qfmlaInfo_->attnMaskFlag ?
                                     static_cast<uint32_t>(qfmlaInfo_->attenMaskS1Size * qfmlaInfo_->attenMaskS2Size) :
                                     0U);
    baseParams.set_sparseMode(static_cast<uint32_t>(qfmlaInfo_->maskMode));
    baseParams.set_outputLayout(qfmlaInfo_->layoutQ == QfmlaLayout::TND ? IFA_LAYOUT_TND : IFA_LAYOUT_BSH_BSND);
    baseParams.set_softmaxLseFlag(0U);
    baseParams.set_tSeqSize(static_cast<uint32_t>(qfmlaInfo_->qTSize));
}

void QuantFlashMlaTilingInt8Impl::FillSplitKVParams()
{
    auto& splitKVParams = tilingData_.splitKVParams;
    const uint32_t kvSplitPartUpper = platformInfo_.aicNum;
    splitKVParams.set_s2(kvSplitPartUpper);

    const uint64_t maxActualseq = GetMaxActualseqUpperBound();
    uint64_t sInnerLoopSize = (maxActualseq + kvSplitPartUpper - 1U) / kvSplitPartUpper;
    if (qfmlaInfo_->blockSize > 0) {
        const uint64_t blockSize = static_cast<uint64_t>(qfmlaInfo_->blockSize);
        sInnerLoopSize = (sInnerLoopSize + blockSize - 1U) / blockSize * blockSize;
    }
    splitKVParams.set_sInnerLoopSize(static_cast<uint32_t>(sInnerLoopSize));

    const uint64_t headDimAlign = IFA_MLA_HEAD_DIM;
    const uint64_t balanceFDParamNums = static_cast<uint64_t>(kvSplitPartUpper) * 2ULL *
                                        static_cast<uint64_t>(qfmlaInfo_->n2Size) * IFA_BALANCE_SG_BASIC_SIZE;
    fdAccumOutSize_ = balanceFDParamNums * headDimAlign;
    fdLogSumExpSize_ = 2ULL * balanceFDParamNums * (IFA_BYTE_BLOCK / IFA_BLOCK_TYPE_SIZE);
    splitKVParams.set_accumOutSize(static_cast<uint32_t>(fdAccumOutSize_));
    splitKVParams.set_logSumExpSize(static_cast<uint32_t>(fdLogSumExpSize_));
}

void QuantFlashMlaTilingInt8Impl::FillSingleCoreParams()
{
    auto& singleCoreParams = tilingData_.increFlashAttentionSingleCoreParams;
    singleCoreParams.set_singleProcessSInnerSize(IFA_MLA_S_INNER);
    singleCoreParams.set_usedCoreNum(platformInfo_.aicNum);
    singleCoreParams.set_groupSplitSize(1);
    singleCoreParams.set_s1SplitSize(1);

    const uint64_t cubeMSize =
        std::min<uint64_t>(static_cast<uint64_t>(qSeqSizeUpper_) * static_cast<uint64_t>(qfmlaInfo_->n1Size),
                           static_cast<uint64_t>(IFA_G_MAX));
    auto& tensorSize = tilingData_.increFlashAttentionSingleCoreTensorSize;
    tensorSize.set_mmResUbSize(static_cast<uint32_t>(IFA_MLA_S_INNER * cubeMSize));
    tensorSize.set_bmm2ResUbSize(static_cast<uint32_t>(IFA_MLA_HEAD_DIM * cubeMSize));
}

uint64_t QuantFlashMlaTilingInt8Impl::GetMaxActualseqUpperBound() const
{
    if (qfmlaInfo_->maxSeqKv > 0) {
        return static_cast<uint64_t>(qfmlaInfo_->maxSeqKv);
    }
    return static_cast<uint64_t>(qfmlaInfo_->maxBlockNumPerBatch) * static_cast<uint64_t>(qfmlaInfo_->blockSize);
}

void QuantFlashMlaTilingInt8Impl::CalcWorkspaceSize()
{
    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    const uint64_t actCoreNum = platformInfo_.aicNum;
    const uint64_t mmResUbSize = static_cast<uint64_t>(IFA_MLA_S_INNER) * IFA_G_MAX;
    const uint64_t bmm2ResUbSize = static_cast<uint64_t>(IFA_MLA_HEAD_DIM) * IFA_G_MAX;
    workspaceSize_ += static_cast<uint64_t>(IFA_PRE_LOAD_NUM_MLA) * actCoreNum *
                      (mmResUbSize * (IFA_BLOCK_TYPE_SIZE + 2U /*vec1 fp16*/) +
                       bmm2ResUbSize * (IFA_BLOCK_TYPE_SIZE + IFA_BLOCK_TYPE_SIZE));
    workspaceSize_ += static_cast<uint64_t>(IFA_PRE_LOAD_NUM_MLA) * IFA_G_MAX * actCoreNum *
                      (IFA_BLOCK_TYPE_SIZE + IFA_BLOCK_TYPE_SIZE);

    workspaceSize_ += (fdAccumOutSize_ + fdLogSumExpSize_) * IFA_BLOCK_TYPE_SIZE;

    workspaceSize_ += actCoreNum * IFA_MM_PA_CALLBACK_SIZE * 2ULL;

    OP_LOGI(qfmlaInfo_->opName, "MLA int8 Workspaces: %lu", workspaceSize_);
}

void QuantFlashMlaTilingInt8Impl::GenTilingKey()
{
    const uint64_t inOutLayout =
        (qfmlaInfo_->layoutQ == QfmlaLayout::TND) ? QFMLA_ARCH22_INOUT_LAYOUT_TND : QFMLA_ARCH22_INOUT_LAYOUT_BSH;
    const uint64_t kvLayout = QFMLA_ARCH22_KV_LAYOUT_NZ;
    const uint64_t hasMask = qfmlaInfo_->attnMaskFlag ? 1ULL : 0ULL;
    tilingKey_ = (QFMLA_ARCH22_QUANT_MODE_INT8 << QFMLA_ARCH22_KEY_QUANT_MODE_SHIFT) |
                 (inOutLayout << QFMLA_ARCH22_KEY_IN_OUT_LAYOUT_SHIFT) |
                 (kvLayout << QFMLA_ARCH22_KEY_KV_LAYOUT_SHIFT) | (hasMask << QFMLA_ARCH22_KEY_HAS_ATTEN_MASK_SHIFT);
    OP_LOGI(qfmlaInfo_->opName,
            "MLA int8 tilingkey is %llu: quantMode: %llu, inOutLayout(BSH=0/TND=1): %llu, "
            "kvLayout(NZ=1): %llu, hasAttenMask: %llu.",
            tilingKey_, (tilingKey_ >> QFMLA_ARCH22_KEY_QUANT_MODE_SHIFT) & QFMLA_ARCH22_KEY_FIELD_MASK,
            (tilingKey_ >> QFMLA_ARCH22_KEY_IN_OUT_LAYOUT_SHIFT) & QFMLA_ARCH22_KEY_FIELD_MASK,
            (tilingKey_ >> QFMLA_ARCH22_KEY_KV_LAYOUT_SHIFT) & QFMLA_ARCH22_KEY_FIELD_MASK,
            (tilingKey_ >> QFMLA_ARCH22_KEY_HAS_ATTEN_MASK_SHIFT) & QFMLA_ARCH22_KEY_FIELD_MASK);
}

ge::graphStatus QuantFlashMlaTilingInt8Impl::SetTilingData(IncreFlashAttentionTilingDataMla& tilingData)
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(qfmlaInfo_->opName, "The raw tiling data is nullptr"),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void QuantFlashMlaTilingInt8Impl::PrintAllTilingData()
{
    IncreFlashAttentionBaseParamsMla& baseParams = tilingData_.baseParams;
    IncreFlashAttentionSplitKVParamsMla& splitKVParams = tilingData_.splitKVParams;
    IncreFlashAttentionSingleCoreParamsMla& singleCoreParams = tilingData_.increFlashAttentionSingleCoreParams;

    OP_LOGD(qfmlaInfo_->opName, "MLA int8 batchSize:%u seqSize:%u qSeqSize:%u blockSize:%u maxBlockNumPerBatch:%u",
            baseParams.get_batchSize(), baseParams.get_seqSize(), baseParams.get_qSeqSize(), baseParams.get_blockSize(),
            baseParams.get_maxBlockNumPerBatch());
    OP_LOGD(qfmlaInfo_->opName, "MLA int8 scaleValue:%f nNumOfQInOneGroup:%u actualLenQDims:%u actualLenDims:%u",
            baseParams.get_scaleValue(), baseParams.get_nNumOfQInOneGroup(), baseParams.get_actualLenQDims(),
            baseParams.get_actualLenDims());
    OP_LOGD(qfmlaInfo_->opName, "MLA int8 attenMaskFlag:%u attenMaskSize:%u sparseMode:%u outputLayout:%u tSeqSize:%u",
            baseParams.get_attenMaskFlag(), baseParams.get_attenMaskSize(), baseParams.get_sparseMode(),
            baseParams.get_outputLayout(), baseParams.get_tSeqSize());
    OP_LOGD(qfmlaInfo_->opName, "MLA int8 kvSplitPart(s2):%u sInnerLoopSize:%u accumOutSize:%u logSumExpSize:%u",
            splitKVParams.get_s2(), splitKVParams.get_sInnerLoopSize(), splitKVParams.get_accumOutSize(),
            splitKVParams.get_logSumExpSize());
    OP_LOGD(qfmlaInfo_->opName, "MLA int8 singleProcessSInnerSize:%u usedCoreNum:%u groupSplitSize:%u s1SplitSize:%u",
            singleCoreParams.get_singleProcessSInnerSize(), singleCoreParams.get_usedCoreNum(),
            singleCoreParams.get_groupSplitSize(), singleCoreParams.get_s1SplitSize());
    OP_LOGD(qfmlaInfo_->opName, "MLA int8 分核数组不由host计算（metadata AICPU 运行期下发）");
}

ge::graphStatus QuantFlashMlaTilingInt8Impl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(qfmlaInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    CalcNumBlocks(platformInfo_.aicNum);
    FillTilingData();
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

} // namespace quant_flash_mla

using quant_flash_mla::QuantFlashMlaTilingInt8Impl;

REGISTER_TILING_TEMPLATE_FIA(QuantFlashMla, QuantFlashMlaTilingInt8Impl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 200);
} // namespace optiling
