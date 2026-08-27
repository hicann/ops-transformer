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
 * \file fia_tiling_nonquant_mla.cpp
 * \brief MLA D512 non-quant tiling template (d=512, rope=64, v=512, no mask)
 */

#include "fia_tiling_nonquant_mla_arch35.h"
#include <map>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "../../../common/op_host/split_core_v2.h"
#include "../fused_infer_attention_score_tiling_utils.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "../../op_kernel/arch35/fia_arch35_template_tiling_key_enum.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
using namespace arch35FIA;
constexpr uint64_t PRE_LOAD_NUM_MLA_ARCH35 = 2;

void FiaTilingNonQuantMlaArch35::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

bool FiaTilingNonQuantMlaArch35::IsCapableBasicCheckMla()
{
    if (fiaInfo_ == nullptr) {
        return false;
    }
    // 不支持空Tensor
    if (fiaInfo_->emptyTensorFlag) {
        return false;
    }
    // 仅支持非量化
    if (fiaInfo_->quantMode != FiaQuantMode::NO_QUANT) {
        return false;
    }
    // 仅支持 MLA D512
    if (fiaInfo_->mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return false;
    }
    return true;
}

bool FiaTilingNonQuantMlaArch35::IsCapableFeatureCheckMla()
{                                                                // 以下场景路由至其他模板
    if (fiaInfo_->sysPrefixFlag ||                               // 使能公共前缀
        fiaInfo_->pseShiftFlag ||                                // 使能PSE
        fiaInfo_->enableAlibiPse ||                              // 使能alibi
        fiaInfo_->qPaddingSizeFlag ||                            // 使能Q 左padding
        fiaInfo_->kvPaddingSizeFlag ||                           // 使能KV 左padding
        fiaInfo_->isOutQuantEnable ||                            // 使能后量化
        fiaInfo_->learnableSinkFlag ||                           // 使能sink
        fiaInfo_->isQKVDDifferent ||                             // D不等长
        fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) { // 使能tensorlist
        return false;
    }
    return true;
}

bool FiaTilingNonQuantMlaArch35::IsCapableSparseLayoutCheckMla()
{
    if (fiaInfo_->sparseMode != SPARSE_MODE_NO_MASK && fiaInfo_->sparseMode != SPARSE_MODE_RIGHT_DOWN &&
        fiaInfo_->sparseMode != SPARSE_MODE_BAND) {
        return false;
    }
    if (fiaInfo_->sparseMode == SPARSE_MODE_NO_MASK && fiaInfo_->attenMaskFlag) {
        return false;
    }
    // MLA D512 维度约束
    if (fiaInfo_->qkHeadDim != DSIZE_512 || fiaInfo_->ropeHeadDim != MLA_ROPE_D_DIM_64 ||
        fiaInfo_->vHeadDim != DSIZE_512) {
        return false;
    }
    // N2 = 1
    if (fiaInfo_->n2Size != 1) {
        return false;
    }
    // N1 范围 [1, 128]
    if (fiaInfo_->n1Size < 1 || fiaInfo_->n1Size > NUM_128) {
        return false;
    }

    return true;
}

bool FiaTilingNonQuantMlaArch35::IsCapable()
{
    if (!IsCapableBasicCheckMla()) {
        return false;
    }

    if (!IsCapableFeatureCheckMla()) {
        return false;
    }

    if (!IsCapableSparseLayoutCheckMla()) {
        return false;
    }
    return true;
}

void FiaTilingNonQuantMlaArch35::CalcMaxWorkspaceSize()
{
    constexpr uint64_t mSize = 64;
    constexpr uint64_t dVSize = DSIZE_512;
    constexpr uint64_t lseSize = 8;

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    const uint64_t faTmpAttenGmSize =
        static_cast<uint64_t>(platformInfo_.aicNum) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * dVSize;
    const uint64_t faTmpResLseGmSize =
        static_cast<uint64_t>(platformInfo_.aicNum) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * lseSize;
    workspaceSize_ += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
}

void FiaTilingNonQuantMlaArch35::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(fiaInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FiaTilingNonQuantMlaArch35::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    if (fiaInfo_->isMaxWorkspace) {
        // tiling下沉场景，无法获取到actual_seq，分核结果未知，workspace设置成最大
        CalcMaxWorkspaceSize();
        // tiling下沉场景需传入有效Tilingkey，FE会严格校验tilingkey
        GenTilingKey();
        CalcNumBlocks(platformInfo_.aicNum);

        if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
            (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
            (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

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

ge::graphStatus FiaTilingNonQuantMlaArch35::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(fiaInfo_->opName, "The platformInfoPtr is null!"),
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
    OP_LOGI(fiaInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

int64_t FiaTilingNonQuantMlaArch35::GetActSeqLenMla(const gert::Tensor *tensor, uint32_t dims, FiaLayout layout,
                                                    uint32_t bIdx)
{
    if (dims == 1) {
        return tensor->GetData<int64_t>()[0];
    }
    if (layout != FiaLayout::TND && layout != FiaLayout::NTD) {
        return tensor->GetData<int64_t>()[bIdx];
    }
    if (bIdx == 0) {
        return tensor->GetData<int64_t>()[0];
    }
    return tensor->GetData<int64_t>()[bIdx] - tensor->GetData<int64_t>()[bIdx - 1];
}

void FiaTilingNonQuantMlaArch35::InitImplParam()
{
    const gert::Tensor *actSeqLenQ = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenKV = fiaInfo_->opParamInfo.actualSeqLengths.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
    if (fiaInfo_->isMaxWorkspace) {
        actualSeqLenQFlag_ = false;
        actualSeqLenKVFlag_ = false;
    } else {
        actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
        actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
        actualSeqLenQFlag_ =
            !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int64_t>() == nullptr));
        actualSeqLenKVFlag_ =
            !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int64_t>() == nullptr));
    }

    actualSharedPrefixLenFlag_ = false;

    for (uint32_t bIdx = 0; bIdx < fiaInfo_->bSize; bIdx++) {
        if (actualSeqLenQFlag_) {
            actualSeqLengthsQ_.push_back(GetActSeqLenMla(actSeqLenQ, actSeqLenQDims, fiaInfo_->qLayout, bIdx));
        } else {
            actualSeqLengthsQ_.push_back(fiaInfo_->s1Size);
        }

        if (actualSeqLenKVFlag_) {
            actualSeqLengthsKV_.push_back(GetActSeqLenMla(actSeqLenKV, actSeqLenKVDims, fiaInfo_->kvLayout, bIdx));
        } else if (fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) {
            actualSeqLengthsKV_.push_back(fiaInfo_->kvListSeqLens[bIdx]);
        } else {
            actualSeqLengthsKV_.push_back(fiaInfo_->s2Size);
        }
    }
}

void FiaTilingNonQuantMlaArch35::AdjustSinnerAndSouter()
{
    // MLA D512 固定切分大小
    sOuterFactor_ = SOUTER_32;
    sInnerFactor_ = SINNER_128;

    OP_LOGI(fiaInfo_->opName, "Souter:%u SInner:%u", sOuterFactor_, sInnerFactor_);
}

void FiaTilingNonQuantMlaArch35::CreateSplitInput(split_core_v2::BaseInfo &baseInfo,
                                                  split_core_v2::SplitParam &splitParam)
{
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = fiaInfo_->n2Size;
    baseInfo.gSize = fiaInfo_->gSize;
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.s2Size = fiaInfo_->s2Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenKvDims;
    baseInfo.preToken = fiaInfo_->preToken;
    baseInfo.nextToken = fiaInfo_->nextToken;
    baseInfo.isS1G = (fiaInfo_->qLayout == FiaLayout::BSH) || (fiaInfo_->qLayout == FiaLayout::BSND) ||
                     (fiaInfo_->qLayout == FiaLayout::TND);
    baseInfo.sparseMode = fiaInfo_->sparseMode;
    baseInfo.attenMaskFlag = fiaInfo_->attenMaskFlag;

    if (fiaInfo_->qLayout == FiaLayout::TND) {
        baseInfo.isAccumSeqS1 = true;
        baseInfo.isAccumSeqS2 = !fiaInfo_->pageAttentionFlag;
    } else {
        baseInfo.isAccumSeqS1 = false;
        baseInfo.isAccumSeqS2 = false;
    }
    const gert::Tensor *actSeqLenData = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenDataKV = fiaInfo_->opParamInfo.actualSeqLengths.tensor;
    if (actSeqLenData != nullptr) {
        baseInfo.actualSeqS1Size.reserve(baseInfo.bSize);
        const int64_t *s1Ptr = actSeqLenData->GetData<int64_t>();
        for (uint32_t i = 0; i < baseInfo.bSize; i++) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actSeqLenDataKV != nullptr) {
        baseInfo.actualSeqS2Size.reserve(baseInfo.bSize);
        const int64_t *s2Ptr = actSeqLenDataKV->GetData<int64_t>();
        for (uint32_t i = 0; i < baseInfo.bSize; i++) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = sOuterFactor_ * CV_RATIO;
    splitParam.s2BaseSize = sInnerFactor_;
    splitParam.gS1BaseSizeOfFd = 8;
    splitParam.streamK = true;
    splitParam.fdTolerance = 9;  // 实验经验值
    splitParam.fdLeastBlock = 0; // 实验经验值
}

void FiaTilingNonQuantMlaArch35::SetSplitOutput(const split_core_v2::FAMetaData &splitRes)
{
    // FA Metadata Generate
    for (size_t i = 0; i < NPU_AIC_CORE_NUM; ++i) {
        if (i >= splitRes.usedCoreNum) {
            tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_START_INDEX] = (i == 0) ? 0 : splitRes.bN2End[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_START_INDEX] = (i == 0) ? 0 : splitRes.mEnd[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_START_INDEX] = (i == 0) ? 0 : splitRes.s2End[i - 1];
        // FA END
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_END_INDEX] = splitRes.mEnd[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
        // FA idx
        tilingData_.fiaMetaData.FAMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }

    // FD Metadata Generate
    const split_core_v2::FlashDecodeResult &fdRes = splitRes.fdRes;
    for (size_t i = 0; i < NPU_AIV_CORE_NUM; ++i) {
        if (i >= fdRes.fdUsedVecNum) {
            tilingData_.fiaMetaData.FDMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        tilingData_.fiaMetaData.FDMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = fdRes.taskIdx[i];
        tilingData_.fiaMetaData.FDMetadata[i][FD_BN2_IDX_INDEX] = fdRes.fdBN2Idx[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_M_IDX_INDEX] = fdRes.fdMIdx[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_S2_SPLIT_NUM_INDEX] = fdRes.fdS2SplitNum[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_WORKSPACE_IDX_INDEX] = fdRes.fdWorkspaceIdx[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_M_START_INDEX] = fdRes.mStart[i];
        tilingData_.fiaMetaData.FDMetadata[i][FD_M_NUM_INDEX] = fdRes.mLen[i];
    }
}

void FiaTilingNonQuantMlaArch35::SplitPolicy()
{
    AdjustSinnerAndSouter(); // 确定tiling切块

    split_core_v2::BaseInfo baseInfo{};
    split_core_v2::SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam);

    // MLA D512: 不支持S1外切，直接走SplitCore
    split_core_v2::FAMetaData result{platformInfo_.aicNum, platformInfo_.cvRatio};
    split_core_v2::SplitCore(platformInfo_.aicNum, baseInfo, splitParam, result);
    SetSplitOutput(result);
    CalcNumBlocks(result.usedCoreNum);
    flashDecodeFlag_ = (result.fdRes.fdNum > 0);

    fiaInfo_->isExistRowInvalid =
        (fiaInfo_->needInit || IsExistRowInvalid(baseInfo) || IsActualSeqLengthsKVHasZero(baseInfo));
    // 行无效自动开启(对齐兜底模板FusedInferAttentionScoreTilingImpl): 存在mask行无效,
    // 或 sparse=0+带mask+padding 时自动开, 与innerPrecise无关
    isRowInvalidOpenAuto_ =
        IsExistRowInvalid(baseInfo) || (baseInfo.sparseMode == SPARSE_MODE_NO_MASK && baseInfo.attenMaskFlag &&
                                        (fiaInfo_->qPaddingSizeFlag || fiaInfo_->kvPaddingSizeFlag));
}

bool FiaTilingNonQuantMlaArch35::IsActualSeqLengthsKVHasZero(const split_core_v2::BaseInfo &baseInfo)
{
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s2Size = split_core_v2::GetS2SeqSize(bIdx, baseInfo);
        if (s2Size == 0) {
            return true;
        }
    }
    return false;
}

bool FiaTilingNonQuantMlaArch35::IsExistRowInvalid(const split_core_v2::BaseInfo &baseInfo)
{
    if (!baseInfo.attenMaskFlag) {
        return false;
    }

    auto mode = static_cast<split_core_v2::SparseMode>(baseInfo.sparseMode);
    if (mode == split_core_v2::SparseMode::LEFT_UP_CAUSAL) {
        return false;
    }

    if (mode == split_core_v2::SparseMode::ALL_MASK) {
        return true;
    }

    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        int32_t s1Size = split_core_v2::GetS1SeqSize(bIdx, baseInfo);
        int32_t s2Size = split_core_v2::GetS2SeqSize(bIdx, baseInfo);
        if ((s1Size == 0) || (s2Size == 0)) {
            // 空tensor认为不会有无效行
            continue;
        }

        int64_t safePreToken = baseInfo.preToken;
        int64_t safeNextToken = baseInfo.nextToken;
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;

        GetSafeActToken(mode, s1Size, s2Size, safePreToken, safeNextToken);
        if (mode == split_core_v2::SparseMode::BAND) {
            preTokenLeftUp = safePreToken;
            nextTokenLeftUp = s2Size - s1Size + safeNextToken;
        } else if (mode == split_core_v2::SparseMode::DEFAULT_MASK) {
            preTokenLeftUp = s2Size - s1Size + safePreToken;
            nextTokenLeftUp = safeNextToken;
        } else {
            preTokenLeftUp = 0;
            nextTokenLeftUp = s2Size - s1Size;
        }

        if ((preTokenLeftUp < 0) || (nextTokenLeftUp < 0)) {
            return true;
        }
    }
    return false;
}

void FiaTilingNonQuantMlaArch35::GetSafeActToken(split_core_v2::SparseMode mode, int64_t actSeqLensQ,
                                                 int64_t actSeqLensKv, int64_t &safePreToken, int64_t &safeNextToken)
{
    if (mode == split_core_v2::SparseMode::DEFAULT_MASK) {
        safePreToken = std::max(-actSeqLensKv, safePreToken);
        safePreToken = std::min(safePreToken, actSeqLensQ);
        safeNextToken = std::max(-actSeqLensQ, safeNextToken);
        safeNextToken = std::min(safeNextToken, actSeqLensKv);
    } else if (mode == split_core_v2::SparseMode::BAND) {
        safePreToken = std::max(-actSeqLensQ, safePreToken);
        safePreToken = std::min(safePreToken, actSeqLensKv);
        safeNextToken = std::max(-actSeqLensKv, safeNextToken);
        safeNextToken = std::min(safeNextToken, actSeqLensQ);
    }
}

void FiaTilingNonQuantMlaArch35::UpdateTilingKeyConfig()
{
    // MLA D512 唯一路径: sOuter=64, sInner=128, D=576, DV=512 → Config index 9
    tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512;
}

void FiaTilingNonQuantMlaArch35::UpdateTilingKeyLayout()
{
    if (fiaInfo_->qLayout == FiaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BNSD;
    } else if (fiaInfo_->qLayout == FiaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
    } else if (fiaInfo_->qLayout == FiaLayout::NTD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_NTD_NTD;
    } else if (fiaInfo_->qLayout == FiaLayout::BSH || fiaInfo_->qLayout == FiaLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    }
}

void FiaTilingNonQuantMlaArch35::UpdateTilingKeyPseMode()
{
    tilingKeyInfo_.pseMode = PSE_MODE_PSE_NONE_TYPE;
}

void FiaTilingNonQuantMlaArch35::UpdateTilingKeyQuantMode()
{
    tilingKeyInfo_.quantMode = NoQuantMode;
}

void FiaTilingNonQuantMlaArch35::UpdateTilingKeyHasRope()
{
    tilingKeyInfo_.hasRope = true;
}

void FiaTilingNonQuantMlaArch35::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyPseMode();
    UpdateTilingKeyQuantMode();
    tilingKeyInfo_.isFd = flashDecodeFlag_;
    tilingKeyInfo_.hasAttenMask = fiaInfo_->attenMaskFlag;
    UpdateTilingKeyHasRope();

    if (fiaInfo_->pageAttentionFlag) {
        if (tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType == 0) { // BNBD
            tilingKeyInfo_.kvLayoutType = 2;
        } else if (tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType == 1) { // BBH
            tilingKeyInfo_.kvLayoutType = 1;
        } else { // PA NZ
            tilingKeyInfo_.kvLayoutType = 3;
        }
    } else {
        tilingKeyInfo_.kvLayoutType = 0;
    }

    tilingKeyInfo_.emptyTensor = fiaInfo_->emptyTensorFlag;
    tilingKeyInfo_.enableKvPrefix = fiaInfo_->sysPrefixFlag;
    tilingKeyInfo_.isReconstructTemp = true;
}

void FiaTilingNonQuantMlaArch35::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode,
                                    tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope,
                                    tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd, tilingKeyInfo_.emptyTensor,
                                    tilingKeyInfo_.enableKvPrefix, false, tilingKeyInfo_.isReconstructTemp);

    OP_LOGI(fiaInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(fiaInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu, quantMode: %llu, "
            "hasAttenMask: %u, hasRope: %u, kvLayoutType: %llu, isFd: %u, emptyTensor: %u, "
            "enableKvPrefix: %u, isReconstructTemp: %u.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
            tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd,
            tilingKeyInfo_.emptyTensor, tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.isReconstructTemp);
}

void FiaTilingNonQuantMlaArch35::CalcNumBlocks(uint32_t usedAicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aivNum = usedAicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, usedAicNum, aivNum);
    OP_LOGI(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, usedAicNum);
}

void FiaTilingNonQuantMlaArch35::CalcWorkspaceSize()
{
    // Keep the workspace slot stride consistent with the MLA kernel's mBaseSize.
    constexpr uint64_t mSize = 64;
    constexpr uint64_t dSize = DSIZE_512;
    constexpr uint64_t lseSize = 8;

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    if (flashDecodeFlag_) {
        const uint64_t faTmpAttenGmSize =
            static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * dSize; // 每个核最多有2次写到workspace
        const uint64_t faTmpResLseGmSize =
            static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * lseSize;
        workspaceSize_ += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.fiaWorkspaceParams.accumOutSize = static_cast<uint32_t>(faTmpAttenGmSize);
        tilingData_.baseTiling.fiaWorkspaceParams.logSumExpSize = static_cast<uint32_t>(faTmpResLseGmSize);
    }

    OP_LOGI(fiaInfo_->opName, "Workspaces: %lu", workspaceSize_);
}

void FiaTilingNonQuantMlaArch35::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
}

void FiaTilingNonQuantMlaArch35::ComputeTilingData()
{
    SetAttenMaskTilingData();
    SetStartIdxTilingData();
    SetPageAttentionLayoutTilingData();
}

void FiaTilingNonQuantMlaArch35::SetAttenMaskTilingData()
{
    if (fiaInfo_->attenMaskFlag) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || fiaInfo_->s1Size == 1) {
            maskBatch = fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDim(0);
        }
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskBatch = maskBatch;
        maskS2Size = fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS2Size = maskS2Size;
    } else {
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS1Size = 0;
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS2Size = 0;
    }
    tilingData_.baseTiling.fiaAttenMaskParams.sparseMode = fiaInfo_->sparseMode;
}

void FiaTilingNonQuantMlaArch35::SetStartIdxTilingData()
{
    tilingData_.baseTiling.fiaPseParams.qStartIdx = 0;
    tilingData_.baseTiling.fiaPseParams.kvStartIdx = 0;
}

void FiaTilingNonQuantMlaArch35::SetPageAttentionLayoutTilingData()
{
    if (fiaInfo_->pageAttentionFlag) {
        uint32_t keyCacheDimNum = fiaInfo_->opParamInfo.key.shape->GetStorageShape().GetDimNum();
        if (keyCacheDimNum == 3) { // 3: BBH
            tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType = 1;
        } else if (keyCacheDimNum == 4) { // 4: BNBD
            tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType = 0;
        } else if (keyCacheDimNum == 5) { // 5: PA NZ
            tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType = 2;
        }
    }
}

void FiaTilingNonQuantMlaArch35::SetFATilingData()
{
    tilingData_.baseTiling.fiaBaseParams.bSize = fiaInfo_->bSize;
    tilingData_.baseTiling.fiaBaseParams.t1Size = fiaInfo_->qTSize;
    tilingData_.baseTiling.fiaBaseParams.t2Size = fiaInfo_->kTSize;
    tilingData_.baseTiling.fiaBaseParams.n2Size = fiaInfo_->n2Size;
    tilingData_.baseTiling.fiaBaseParams.gSize = fiaInfo_->gSize;
    tilingData_.baseTiling.fiaBaseParams.s1Size = fiaInfo_->s1Size;
    tilingData_.baseTiling.fiaBaseParams.s2Size = fiaInfo_->s2Size;
    tilingData_.baseTiling.fiaBaseParams.dSize = fiaInfo_->qkHeadDim;
    tilingData_.baseTiling.fiaBaseParams.dSizeV = fiaInfo_->vHeadDim;
    tilingData_.baseTiling.fiaBaseParams.dSizeRope = fiaInfo_->ropeHeadDim;
    tilingData_.baseTiling.fiaBaseParams.scaleValue = fiaInfo_->scaleValue;
    tilingData_.baseTiling.fiaBaseParams.actualSeqLengthsQSize = fiaInfo_->actualLenQDims;
    tilingData_.baseTiling.fiaBaseParams.actualSeqLengthsKVSize = fiaInfo_->actualLenKvDims;
    tilingData_.baseTiling.fiaBaseParams.isKvContinuous = fiaInfo_->kvStorageMode != KvStorageMode::TENSOR_LIST;
    tilingData_.baseTiling.fiaBaseParams.isSoftMaxLseEnable = fiaInfo_->softmaxLseFlag;
    // gSize*s1Size<=64 的单token场景: K/V数据量远超L2容量, 完全无法复用, 关闭L2 Cache避免无意义的缓存填充/驱逐开销
    tilingData_.baseTiling.fiaBaseParams.l2CacheOffFlag = (fiaInfo_->gSize * fiaInfo_->s1Size) <= 64 ? 1U : 0U;
    OP_LOGI(fiaInfo_->opName, "l2CacheOffFlag: %u, gSize: %u, s1Size: %u",
            tilingData_.baseTiling.fiaBaseParams.l2CacheOffFlag, fiaInfo_->gSize, fiaInfo_->s1Size);
    tilingData_.baseTiling.fiaBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.fiaBaseParams.outputLayout = static_cast<uint32_t>(fiaInfo_->outputLayout);

    tilingData_.baseTiling.fiaAttenMaskParams.preTokens = fiaInfo_->preToken;
    tilingData_.baseTiling.fiaAttenMaskParams.nextTokens = fiaInfo_->nextToken;
    tilingData_.baseTiling.fiaAttenMaskParams.isRowInvalidOpen =
        ((fiaInfo_->innerPrecise >> 1) & 1) || isRowInvalidOpenAuto_;
    tilingData_.baseTiling.fiaAttenMaskParams.isExistRowInvalid = fiaInfo_->isExistRowInvalid;

    tilingData_.baseTiling.fiaPseParams.pseS1Size = fiaInfo_->pseShiftS1;
    tilingData_.baseTiling.fiaPseParams.pseS2Size = fiaInfo_->pseShiftS2;
    tilingData_.baseTiling.fiaPseParams.pseShiftByBatch = fiaInfo_->pseShiftByBatch;

    tilingData_.baseTiling.fiaLeftPaddingParams.isQHasLeftPadding = fiaInfo_->qPaddingSizeFlag;
    tilingData_.baseTiling.fiaLeftPaddingParams.isKVHasLeftPadding = fiaInfo_->kvPaddingSizeFlag;
    tilingData_.baseTiling.fiaPageAttentionParams.blockSize = fiaInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    if (fiaInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        maxBlockNumPerBatch = fiaInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    tilingData_.baseTiling.fiaPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;

    tilingData_.baseTiling.fiaPostQuantParams.isPostQuantPerChnl = fiaInfo_->isOutQuantPerChnOut;
    tilingData_.baseTiling.fiaPostQuantParams.isPostQuantBF16 = fiaInfo_->isOutQuantTypeBf16;

    tilingData_.baseTiling.fiaSystemPrefixParams.prefixSeqInnerSize = fiaInfo_->systemPrefixMaxLen;
    tilingData_.baseTiling.fiaSystemPrefixParams.isActualSharedPrefixLenNull = !actualSharedPrefixLenFlag_;

    int64_t outSize = fiaInfo_->opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize =
        fiaInfo_->softmaxLseFlag ? fiaInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo_->isOutQuantEnable) {
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    }
    tilingData_.baseTiling.fiaEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.fiaEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.fiaEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.fiaEmptyTensorParams.needInit = fiaInfo_->needInit;

    if (fiaInfo_->hasViewStride) {
        tilingData_.baseTiling.fiaBaseParams.keyStrides.bnStride = fiaInfo_->keyBnStride;
        tilingData_.baseTiling.fiaBaseParams.keyStrides.n2Stride = fiaInfo_->keyN2Stride;
        tilingData_.baseTiling.fiaBaseParams.valueStrides.bnStride = fiaInfo_->valueBnStride;
        tilingData_.baseTiling.fiaBaseParams.valueStrides.n2Stride = fiaInfo_->valueN2Stride;
        if (fiaInfo_->kRopeStrides != nullptr) {
            tilingData_.baseTiling.fiaBaseParams.kRopeStrides.bnStride = fiaInfo_->kRopeBnStride;
            tilingData_.baseTiling.fiaBaseParams.kRopeStrides.n2Stride = fiaInfo_->kRopeN2Stride;
        }
    }
}

ge::graphStatus FiaTilingNonQuantMlaArch35::SetTilingData(FusedInferAttentionScoreTilingData &tilingData)
{
    FusedInferAttentionScoreTilingData *tiling = context_->GetTilingData<FusedInferAttentionScoreTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(fiaInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FiaTilingNonQuantMlaArch35::PrintAllTilingData()
{
    NoQuantTilingArch35 &baseTiling = tilingData_.baseTiling;
    FiaBaseParams &fiaBaseParams = baseTiling.fiaBaseParams;
    FiaAttenMaskParams &fiaAttenMaskParams = baseTiling.fiaAttenMaskParams;
    FiaPseParams &fiaPseParams = baseTiling.fiaPseParams;
    FiaSystemPrefixParams &fiaSystemPrefixParams = baseTiling.fiaSystemPrefixParams;
    FiaPageAttentionParams &fiaPageAttentionParams = baseTiling.fiaPageAttentionParams;
    FiaLeftPaddingParams &fiaLeftPaddingParams = baseTiling.fiaLeftPaddingParams;
    FiaPostQuantParams &fiaPostQuantParams = baseTiling.fiaPostQuantParams;
    FiaWorkspaceParams &fiaWorkspaceParams = baseTiling.fiaWorkspaceParams;
    FiaEmptyTensorParams &fiaEmptyTensorParams = baseTiling.fiaEmptyTensorParams;
    FiaMetaData &fiaMetaData = tilingData_.fiaMetaData;

    OP_LOGD(fiaInfo_->opName, "bSize:%d", fiaBaseParams.bSize);
    OP_LOGD(fiaInfo_->opName, "t1Size:%d", fiaBaseParams.t1Size);
    OP_LOGD(fiaInfo_->opName, "t2Size:%d", fiaBaseParams.t2Size);
    OP_LOGD(fiaInfo_->opName, "n2Size:%d", fiaBaseParams.n2Size);
    OP_LOGD(fiaInfo_->opName, "gSize:%d", fiaBaseParams.gSize);
    OP_LOGD(fiaInfo_->opName, "s1Size:%d", fiaBaseParams.s1Size);
    OP_LOGD(fiaInfo_->opName, "s2Size:%d", fiaBaseParams.s2Size);
    OP_LOGD(fiaInfo_->opName, "dSize:%d", fiaBaseParams.dSize);
    OP_LOGD(fiaInfo_->opName, "dSizeV:%d", fiaBaseParams.dSizeV);
    OP_LOGD(fiaInfo_->opName, "dSizeRope:%d", fiaBaseParams.dSizeRope);
    OP_LOGD(fiaInfo_->opName, "actualSeqLengthsQSize:%d", fiaBaseParams.actualSeqLengthsQSize);
    OP_LOGD(fiaInfo_->opName, "actualSeqLengthsKVSize:%d", fiaBaseParams.actualSeqLengthsKVSize);
    OP_LOGD(fiaInfo_->opName, "scaleValue:%f", fiaBaseParams.scaleValue);
    OP_LOGD(fiaInfo_->opName, "isKvContinuous:%d", fiaBaseParams.isKvContinuous);
    OP_LOGD(fiaInfo_->opName, "isSoftMaxLseEnable:%d", fiaBaseParams.isSoftMaxLseEnable);
    OP_LOGD(fiaInfo_->opName, "coreNum:%d", fiaBaseParams.coreNum);

    OP_LOGD(fiaInfo_->opName, "sparseMode:%d", fiaAttenMaskParams.sparseMode);
    OP_LOGD(fiaInfo_->opName, "preTokens:%d", fiaAttenMaskParams.preTokens);
    OP_LOGD(fiaInfo_->opName, "nextTokens:%d", fiaAttenMaskParams.nextTokens);
    OP_LOGD(fiaInfo_->opName, "attenMaskS1Size:%d", fiaAttenMaskParams.attenMaskS1Size);
    OP_LOGD(fiaInfo_->opName, "attenMaskS2Size:%d", fiaAttenMaskParams.attenMaskS2Size);
    OP_LOGD(fiaInfo_->opName, "isRowInvalidOpen:%d", fiaAttenMaskParams.isRowInvalidOpen);
    OP_LOGD(fiaInfo_->opName, "isExistRowInvalid:%d", fiaAttenMaskParams.isExistRowInvalid);

    OP_LOGD(fiaInfo_->opName, "pseS1Size:%d", fiaPseParams.pseS1Size);
    OP_LOGD(fiaInfo_->opName, "pseS2Size:%d", fiaPseParams.pseS2Size);
    OP_LOGD(fiaInfo_->opName, "qStartIdx:%d", fiaPseParams.qStartIdx);
    OP_LOGD(fiaInfo_->opName, "kvStartIdx:%d", fiaPseParams.kvStartIdx);

    OP_LOGD(fiaInfo_->opName, "isActualSharedPrefixLenNull:%d", fiaSystemPrefixParams.isActualSharedPrefixLenNull);
    OP_LOGD(fiaInfo_->opName, "prefixSeqInnerSize:%d", fiaSystemPrefixParams.prefixSeqInnerSize);

    if (fiaInfo_->pageAttentionFlag) {
        OP_LOGD(fiaInfo_->opName, "paLayoutType:%d", fiaPageAttentionParams.paLayoutType);
        OP_LOGD(fiaInfo_->opName, "blockSize:%d", fiaPageAttentionParams.blockSize);
        OP_LOGD(fiaInfo_->opName, "maxBlockNumPerBatch:%d", fiaPageAttentionParams.maxBlockNumPerBatch);
    }

    OP_LOGD(fiaInfo_->opName, "isQHasLeftPadding:%d", fiaLeftPaddingParams.isQHasLeftPadding);
    OP_LOGD(fiaInfo_->opName, "isKVHasLeftPadding:%d", fiaLeftPaddingParams.isKVHasLeftPadding);

    OP_LOGD(fiaInfo_->opName, "isPostQuantPerChnl:%d", fiaPostQuantParams.isPostQuantPerChnl);
    OP_LOGD(fiaInfo_->opName, "isPostQuantBF16:%d", fiaPostQuantParams.isPostQuantBF16);

    if (flashDecodeFlag_) {
        OP_LOGD(fiaInfo_->opName, "accumOutSize:%d", fiaWorkspaceParams.accumOutSize);
        OP_LOGD(fiaInfo_->opName, "logSumExpSize:%d", fiaWorkspaceParams.logSumExpSize);
    }

    OP_LOGD(fiaInfo_->opName, "singleCoreSize:%d", fiaEmptyTensorParams.singleCoreSize);
    OP_LOGD(fiaInfo_->opName, "needInit:%d", fiaEmptyTensorParams.needInit);
    OP_LOGD(fiaInfo_->opName, "totalOutputSize:%d", fiaEmptyTensorParams.totalOutputSize);
    OP_LOGD(fiaInfo_->opName, "totalSoftMaxLseOutputSize:%d", fiaEmptyTensorParams.totalSoftMaxLseOutputSize);

    for (int aicIdx = 0; aicIdx < platformInfo_.aicNum; ++aicIdx) {
        OP_LOGD(fiaInfo_->opName, "FAMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aicIdx, fiaMetaData.FAMetadata[aicIdx][0], fiaMetaData.FAMetadata[aicIdx][1],
                fiaMetaData.FAMetadata[aicIdx][2], fiaMetaData.FAMetadata[aicIdx][3], fiaMetaData.FAMetadata[aicIdx][4],
                fiaMetaData.FAMetadata[aicIdx][5], fiaMetaData.FAMetadata[aicIdx][6],
                fiaMetaData.FAMetadata[aicIdx][7]);
    }

    for (int aivIdx = 0; aivIdx < platformInfo_.aivNum; ++aivIdx) {
        OP_LOGD(fiaInfo_->opName, "FDMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aivIdx, fiaMetaData.FDMetadata[aivIdx][0], fiaMetaData.FDMetadata[aivIdx][1],
                fiaMetaData.FDMetadata[aivIdx][2], fiaMetaData.FDMetadata[aivIdx][3], fiaMetaData.FDMetadata[aivIdx][4],
                fiaMetaData.FDMetadata[aivIdx][5], fiaMetaData.FDMetadata[aivIdx][6],
                fiaMetaData.FDMetadata[aivIdx][7]);
    }

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(fiaInfo_->opName, "Tiling Data context GetCapacity: %lld.", cap);
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore, FiaTilingNonQuantMlaArch35,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}),
                             5); // 0xx-非量化, x0x-mla, 优先级 005
} // namespace optiling
