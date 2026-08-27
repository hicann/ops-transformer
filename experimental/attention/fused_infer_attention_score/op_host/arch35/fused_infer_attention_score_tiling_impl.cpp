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
 * \file fused_infer_attention_score_tiling_impl.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_impl.h"

#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "../fused_infer_attention_score_tiling_utils.h"
#include "../fused_infer_attention_score_tiling_constants.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
using namespace arch35FIA;
ge::graphStatus FusedInferAttentionScoreTilingImpl::SetPlatMemoryInfo(gert::TilingContext *context,
                                                                      const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(fiaInfo.opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(fiaInfo.opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetEmptyTensor(gert::TilingContext *context,
                                                                   const FiaTilingInfo &fiaInfo)
{
    auto &initOutputParams = faRunTilingAdapter_.initOutputParams;
    int64_t outSize = fiaInfo.opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = fiaInfo.softmaxLseFlag ? fiaInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo.isOutQuantEnable) {
        singleCoreSize = AlignUp(singleCoreSize, uint32_t(2));
    }
    initOutputParams.set_singleCoreSize(singleCoreSize);
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(1);

    OP_CHECK_IF(GenTilingKey(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "GenTilingKey fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetBlockDim(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "SetBlockDim fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspace(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "GetWorkspace fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetTilingData(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "SetTilingData fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FusedInferAttentionScoreTilingImpl::CheckTransposeLayout(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD" || layoutStr == "NTD" ||
        layoutStr == "NTD_TND" || layoutStr == "TND_NTD") {
        return true;
    }
    return false;
}

void FusedInferAttentionScoreTilingImpl::SetIsIFA(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    bool isTransposeLayout = layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD" ||
                             layoutStr == "NTD" || layoutStr == "NTD_TND";
    if (fiaInfo.s1Size == 1 && !fiaInfo.enableAlibiPse && !isTransposeLayout &&
        fiaInfo.fullQuantMode != FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
        isIFAFlag_ = true;
        return;
    }
}

void FusedInferAttentionScoreTilingImpl::SetGSMerge(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && fiaInfo.s1Size <= 16) {
        gsMergeFlag_ = true;
        return;
    }
    if (!fiaInfo.antiQuantFlag && fiaInfo.qkHeadDim > 256U && (fiaInfo.qkHeadDim % 64) != 0) {
        gsMergeFlag_ = false;
        return;
    }

    if (fiaInfo.s1Size * fiaInfo.gSize < 64) {
        bool isTransposeLayout = CheckTransposeLayout(fiaInfo);
        pfaMergeFlag_ =
            !(fiaInfo.pseShiftFlag || fiaInfo.enableAlibiPse || fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 ||
              fiaInfo.isOutQuantEnable || fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag ||
              fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST || fiaInfo.quantMode == FiaQuantMode::FULL_QUANT ||
              isTransposeLayout);
    }
    bool actualSeqLenUnequal = false;
    if (actualSeqLenQFlag_ && !fiaInfo.antiQuantFlag) {
        const gert::Tensor *actSeqLenQ = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
        uint32_t actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
        uint32_t actSeqLengthSize = std::min(actSeqLenQDims, fiaInfo.bSize);
        for (uint32_t i = 0; i < actSeqLengthSize; ++i) {
            int64_t actSeqTmp = actSeqLenQ->GetData<int64_t>()[i];
            if ((fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD) && i >= 1) {
                actSeqTmp -= actSeqLenQ->GetData<int64_t>()[i - 1];
            }
            // query act seq len padding情况下不支持合轴
            if (actSeqTmp < fiaInfo.s1Size) {
                actualSeqLenUnequal = true;
                pfaMergeFlag_ = false;
            }
        }
    }

    if (fiaInfo.antiQuantFlag) {
        if (fiaInfo.s1Size == 1 && !fiaInfo.enableAlibiPse) {
            gsMergeFlag_ = true;
        } else {
            if (fiaInfo.gSize * fiaInfo.s1Size <= 0 || fiaInfo.gSize * fiaInfo.s1Size > NUM_32) {
                gsMergeFlag_ = false;
                return;
            }
            std::string layoutStr(fiaInfo.opParamInfo.layOut);
            gsMergeFlag_ =
                !(actualSeqLenUnequal || fiaInfo.qPaddingSizeFlag || fiaInfo.attenMaskFlag || fiaInfo.pseShiftFlag ||
                  fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION || fiaInfo.enableAlibiPse ||
                  fiaInfo.isOutQuantEnable || fiaInfo.softmaxLseFlag || layoutStr == "BNSD_BSND");
        }
    } else {
        gsMergeFlag_ = pfaMergeFlag_;
    }
}

void FusedInferAttentionScoreTilingImpl::InitImplParam(const FiaTilingInfo &fiaInfo)
{
    const gert::Tensor *actSeqLenQ = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    const gert::Tensor *actSharedPrefixLen = fiaInfo.opParamInfo.actualSharedPrefixLen.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
    uint32_t actSharedPrefixLenDims = 0;
    if (fiaInfo.isMaxWorkspace) {
        actualSeqLenQFlag_ = false;
        actualSeqLenKVFlag_ = false;
    } else {
        actSeqLenQDims = (actSeqLenQ != nullptr) ? actSeqLenQ->GetShapeSize() : 0;
        actSeqLenKVDims = (actSeqLenKV != nullptr) ? actSeqLenKV->GetShapeSize() : 0;
        actSharedPrefixLenDims = (actSharedPrefixLen != nullptr) ? actSharedPrefixLen->GetShapeSize() : 0;
        actualSeqLenQFlag_ =
            !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int64_t>() == nullptr));
        actualSeqLenKVFlag_ =
            !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int64_t>() == nullptr));

        if (fiaInfo.antiQuantFlag && fiaInfo.qLayout == FiaLayout::TND) {
            actualSeqLenQFlag_ = true;
        }
    }

    SetGSMerge(fiaInfo);
    SetIsIFA(fiaInfo);
    if (!fiaInfo.antiQuantFlag) {
        gsMergeFlag_ = gsMergeFlag_ || isIFAFlag_;
    }
    nLoopTimes_ = (gsMergeFlag_) ? fiaInfo.n2Size : fiaInfo.n1Size;
    gsSize_ = (gsMergeFlag_) ? fiaInfo.gSize * fiaInfo.s1Size : fiaInfo.s1Size;

    OP_LOGI(fiaInfo.opName, "gsMergeFlag:%u nLoopTimes:%u gsSize:%u ", gsMergeFlag_, nLoopTimes_, gsSize_);

    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT && fiaInfo.s1Size > 1) {
        isPFAFlag_ = true;
    }
    headDimAlign_ = AlignUp(fiaInfo.qkHeadDim, NUM_32);

    if (!fiaInfo.sysPrefixFlag) {
        actualSharedPrefixLenFlag_ = false;
    } else {
        if (fiaInfo.antiQuantFlag) {
            actualSharedPrefixLenFlag_ = !((actSharedPrefixLenDims == 0) || (actSharedPrefixLen == nullptr) ||
                                           (actSharedPrefixLen->GetData<int64_t>() == nullptr));
        } else {
            actualSharedPrefixLenFlag_ =
                !fiaInfo.isMaxWorkspace && !((actSharedPrefixLenDims == 0) || (actSharedPrefixLen == nullptr) ||
                                             (actSharedPrefixLen->GetData<int64_t>() == nullptr));
        }
    }

    if (fiaInfo.antiQuantFlag && fiaInfo.s1Size == 1 && fiaInfo.qLayout != FiaLayout::TND) {
        actualSeqLenQFlag_ = false;
    }

    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        if (actualSeqLenQFlag_) {
            int64_t actSeqLenQData = 0;
            if (actSeqLenQDims == 1) {
                actSeqLenQData = actSeqLenQ->GetData<int64_t>()[0];
            } else {
                if (fiaInfo.qLayout != FiaLayout::TND && fiaInfo.qLayout != FiaLayout::NTD) {
                    actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
                } else {
                    if (bIdx == 0) {
                        actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
                    } else {
                        actSeqLenQData =
                            actSeqLenQ->GetData<int64_t>()[bIdx] - actSeqLenQ->GetData<int64_t>()[bIdx - 1];
                    }
                }
            }
            if (gsMergeFlag_) {
                actualSeqLengthsQ_.push_back(actSeqLenQData * fiaInfo.gSize);
            } else {
                actualSeqLengthsQ_.push_back(actSeqLenQData);
            }
        } else {
            actualSeqLengthsQ_.push_back(gsSize_);
        }

        if (actualSeqLenKVFlag_) {
            if (actSeqLenKVDims == 1) {
                actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[0]);
            } else {
                if (fiaInfo.kvLayout != FiaLayout::TND && fiaInfo.kvLayout != FiaLayout::NTD) {
                    actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
                } else {
                    if (bIdx == 0) {
                        actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
                    } else {
                        actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx] -
                                                      actSeqLenKV->GetData<int64_t>()[bIdx - 1]);
                    }
                }
            }
        } else if (fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
            actualSeqLengthsKV_.push_back(fiaInfo.kvListSeqLens[bIdx]);
        } else {
            actualSeqLengthsKV_.push_back(fiaInfo.s2Size);
        }
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::AdjustSinnerAndSouter(gert::TilingContext *context,
                                                                          const FiaTilingInfo &fiaInfo)
{
    uint32_t softmaxSOuterFactor = SOUTER_64;
    sOuterFactor_ = SOUTER_64;
    sInnerFactor_ = SINNER_128;
    OP_CHECK_IF(SetMM1TilingData(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "set bmm1 tiling data fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetMM2TilingData(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "set bmm2 tiling data fail."), return ge::GRAPH_FAILED);

    if (fiaInfo.vHeadDim <= DSIZE_128 && fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128) {
        bool checkDtype = fiaInfo.quantMode == FiaQuantMode::NO_QUANT;
        bool checkQueryAndValueS = fiaInfo.s1Size <= SOUTER_64 && fiaInfo.s2Size > SINNER_128;
        uint32_t sparseMode = fiaInfo.sparseMode;
        int32_t preTokens = fiaInfo.preToken;
        int32_t nextTokens = fiaInfo.nextToken;
        if (sparseMode == 0) {
            preTokens = (preTokens > 0) ? 0 : preTokens;
        } else if (sparseMode == 4) {
            nextTokens = (nextTokens > 0) ? 0 : nextTokens;
        }
        bool checkSparseMode = (sparseMode != 2 && preTokens + nextTokens > 128);
        if (checkDtype && checkQueryAndValueS && checkSparseMode && fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D128 &&
            fiaInfo.fullQuantMode != FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
            softmaxSOuterFactor = SOUTER_32;
        } else if (((fiaInfo.qLayout == FiaLayout::BSH) || (fiaInfo.qLayout == FiaLayout::BSND) ||
                    (fiaInfo.qLayout == FiaLayout::TND)) &&
                   pfaMergeFlag_) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        }
    } else if (fiaInfo.vHeadDim > DSIZE_128 && fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512 && fiaInfo.s1Size != 1) {
        if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_64;
        } else if (((fiaInfo.qLayout == FiaLayout::BSH) || (fiaInfo.qLayout == FiaLayout::BSND) ||
                    (fiaInfo.qLayout == FiaLayout::TND)) &&
                   pfaMergeFlag_ && fiaInfo.vHeadDim <= DSIZE_256) { // 256 : D size
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        } else {
            sOuterFactor_ = SOUTER_64;
            sInnerFactor_ = SINNER_128;
        }
        softmaxSOuterFactor = SOUTER_32;
    } else if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 ||
               (fiaInfo.s1Size == 1 && fiaInfo.vHeadDim > DSIZE_128)) { // IFA VD > 128
        if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT ||
            fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
            sOuterFactor_ = SOUTER_32;
        } else {
            sOuterFactor_ = SOUTER_64;
        }
        sInnerFactor_ = fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT &&
                                fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512 && fiaInfo.pseShiftFlag ?
                            SINNER_64 :
                            SINNER_128;
        softmaxSOuterFactor = SOUTER_64;
    }

    OP_LOGI(fiaInfo.opName, "Souter:%u SInner:%u softmaxSOuterFactor %u", sOuterFactor_, sInnerFactor_,
            softmaxSOuterFactor);

    return ge::GRAPH_SUCCESS;
}
void FusedInferAttentionScoreTilingImpl::GetPreNextTokensLeftUp(const FiaTilingInfo &fiaInfo, int64_t actualSeqLength,
                                                                int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                                                int64_t &nextTokensLeftUp)
{
    if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT && fiaInfo.vHeadDim == 512) {
            if (fiaInfo.qLayout == FiaLayout::BSND || fiaInfo.qLayout == FiaLayout::BSH ||
                fiaInfo.qLayout == FiaLayout::TND) {
                nextTokensLeftUp = actualSeqLengthKV * fiaInfo.gSize - actualSeqLength;
            } else { // BNSD场景下分核不做优化
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
        preTokensLeftUp = fiaInfo.preToken - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = fiaInfo.nextToken + actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = fiaInfo.preToken;
        nextTokensLeftUp = fiaInfo.nextToken;
    }
}

void FusedInferAttentionScoreTilingImpl::FixParamWithRowInvalid(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLength,
                                                                int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                                                int64_t &nextTokensLeftUp)
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
    int64_t preTokensError = 0;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        preTokensError = (actualSeqLength > actualSeqLengthKV * fiaInfo.gSize + preTokensLeftUp) ?
                             (actualSeqLength - actualSeqLengthKV * fiaInfo.gSize - preTokensLeftUp) :
                             0;
    } else {
        preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                             (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) :
                             0;
    }
    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t FusedInferAttentionScoreTilingImpl::SumOfArithmeticSeries(int64_t an, int64_t d)
{
    // 等差数列求和，an：等差数列第n项，d：等差数列公差
    if (d == 0) {
        return 0;
    }
    return (an > 0) ? (an % d + an) * (an / d + 1) / 2 : 0; // 2：等差数列求和公式分母
}

int64_t FusedInferAttentionScoreTilingImpl::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
                                                            int64_t sInner, int64_t sOuter, int64_t token)
{
    // 以nextToken视角计算完全被nextToken掩盖的基本块数
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t inDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= static_cast<int64_t>(NUM1)) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = inDivOut;
        smallSize = sOuter;
    }

    // nextToken与上边右边构成的大三角形
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockNums, tolerance);

    // nextToken与上边左边构成的左侧三角形，需要减去
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockLeftNums, tolerance);

    // nextToken与下边右边构成的下侧三角形，需要减去
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockDownNums, tolerance);

    // nextToken与下边左边构成的小三角形，需要加上
    int64_t innerCutBlockLeftDownNums = (-blockToken - blockSeqLength) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockLeftDownNums, tolerance);
    return blockNums;
}

int64_t FusedInferAttentionScoreTilingImpl::GetCalcBlockNumsOneHead(const FiaTilingInfo &fiaInfo,
                                                                    int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                                    uint32_t sOuterSize, uint32_t sInnerSize,
                                                                    int64_t preTokensLeftUp, int64_t nextTokensLeftUp)
{
    if (!fiaInfo.attenMaskFlag) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums =
            (actualSeqLengthKV + sInnerSize - 1) / sInnerSize + (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums =
            (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize) +
            (fiaInfo.systemPrefixLen + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
        int64_t outerBlockNums =
            (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // Must meet this condition : pretoken + nexttoken > 0
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
                                           static_cast<int64_t>(sOuterSize), nextTokensLeftUp);
        toCalcBlockNums -=
            GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
                            static_cast<int64_t>(sOuterSize), blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}

int64_t FusedInferAttentionScoreTilingImpl::GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                               int64_t innerBlockNums)
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }
    return sInnerBlockNums;
}

void FusedInferAttentionScoreTilingImpl::ComputeSplitNBSeq(const FiaTilingInfo &fiaInfo, const size_t maxCoreNums,
                                                           uint32_t sOuterSize, uint32_t sInnerSize,
                                                           double coreWeightTarget, uint32_t &curCore)
{
    std::vector<uint32_t> coreSposEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreSposStart(maxCoreNums, 0U);
    std::vector<uint32_t> coreSidEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreSidStart(maxCoreNums, 0U);
    std::vector<uint32_t> coreNidEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreNidStart(maxCoreNums, 0U);
    std::vector<int64_t> sparseStartIdx(maxCoreNums, 0L);
    std::vector<uint32_t> bnStartIdx(maxCoreNums, 0U);
    std::vector<int64_t> gS1StartIdx(maxCoreNums, 0L);

    int64_t curWeight = 0;
    curCore = 0;
    uint32_t tmpCoreNidEnd = 0;
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        for (uint32_t nIdx = 0; nIdx < nLoopTimes_; nIdx++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(fiaInfo, actualSeqLengthsQ_[bIdx],
                                   actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen, preTokensLeftUp,
                                   nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengthsQ_[bIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV_[bIdx];
            FixParamWithRowInvalid(fiaInfo, actualSeqLength, actualSeqLengthKV + fiaInfo.systemPrefixLen,
                                   preTokensLeftUp, nextTokensLeftUp);

            int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize +
                                     (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWeightTarget * double(curCore + 1)) - curWeight;
                int64_t sInnerIndexStart =
                    -(preTokensLeftUp > 0 ?
                          (preTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize) :
                          preTokensLeftUp / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEnd =
                    nextTokensLeftUp > 0 ?
                        (nextTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize) :
                        nextTokensLeftUp / static_cast<int64_t>(sInnerSize);

                // The number of innerBlock blocks in each outBlock row represents the calculation amount of each
                // outBlock row.
                int64_t sInnerBlockNums = GetSInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);

                if (sInnerBlockNums - dif > dif && !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    coreNidEnd[curCore] = tmpCoreNidEnd;
                    coreSidEnd[curCore] = tmpCoreSidEnd;
                    coreSposEnd[curCore] = tmpCoreSposEnd;
                    curCore += 1;
                    coreNidStart[curCore] = nIdx;
                    coreSidStart[curCore] = bIdx;
                    coreSposStart[curCore] = sOuterIndex;
                    bnStartIdx[curCore] = bIdx * nLoopTimes_ + nIdx;
                    gS1StartIdx[curCore] = sOuterIndex;
                }

                tmpCoreNidEnd = nIdx + 1;
                tmpCoreSidEnd = bIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWeight += sInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }

    coreNidEnd[curCore] = tmpCoreNidEnd;
    coreSidEnd[curCore] = tmpCoreSidEnd;
    coreSposEnd[curCore] = tmpCoreSposEnd;
    if (curCore + 1 < maxCoreNums) {
        bnStartIdx[curCore + 1] = fiaInfo.bSize * nLoopTimes_;
        gS1StartIdx[curCore + 1] = tmpCoreSposEnd;
    }

    PromptAttentionSeqParams *seqParams = &pfaTilingData_.promptAttentionSeqParams;
    seqParams->set_CoreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());
    faRunTilingAdapter_.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faRunTilingAdapter_.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

bool FusedInferAttentionScoreTilingImpl::CheckGQAPerblockNz(const FiaTilingInfo &fiaInfo)
{
    constexpr uint32_t optFp8VBlockSize = 512U; // 512 is V SInnerSize
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    gert::Shape valueAntiquantScaleTensorShape = valueAntiquantScaleTensor->GetStorageShape();
    if ((fiaInfo.qLayout == FiaLayout::BNSD && fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT &&
         valueAntiquantScaleTensorShape.GetDim(2) ==
             CeilDivision(fiaInfo.s2Size, static_cast<int64_t>(optFp8VBlockSize)))) {
        return true;
    }
    return false;
}

bool FusedInferAttentionScoreTilingImpl::CheckS1OutSplit(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.antiQuantFlag || fiaInfo.quantFlag || fiaInfo.isOutQuantEnable ||
        fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        return false;
    }

    if (fiaInfo.sysPrefixFlag || fiaInfo.kvPaddingSizeFlag || fiaInfo.qPaddingSizeFlag || dnFlag_ ||
        fiaInfo.learnableSinkFlag || fiaInfo.enableAlibiPse || gsMergeFlag_ || pfaMergeFlag_) {
        return false;
    }

    if (fiaInfo.sparseMode == SPARSE_MODE_BAND ||
        (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK && fiaInfo.attenMaskFlag)) {
        return false;
    }

    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        return false;
    }

    // 仅支持非量化，占用2B
    const int64_t dataTypeSize = 2U;
    int64_t bnSize = std::min(fiaInfo.bSize * fiaInfo.n2Size, platformInfo_.aicNum);

    // 当所需的L2cache资源的超过系统配置一半时，开启S1外切分核优化L2cache复用率，乘2是经验值，后续进行优化
    return bnSize * fiaInfo.s2Size * (fiaInfo.qkHeadDim + fiaInfo.vHeadDim) * dataTypeSize * 2 >= platformInfo_.l2Size;
}

void FusedInferAttentionScoreTilingImpl::SplitOutSeq(const FiaTilingInfo &fiaInfo)
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    uint32_t sOuterSize = sOuterFactor_ * CV_RATIO;
    int64_t totalSize = 0;
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengthsQ_[bIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(fiaInfo, actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        int64_t outerBlockNums =
            (actualSeqLengthsTmp + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        totalSize += outerBlockNums * fiaInfo.n1Size;
    }

    int64_t s1OuterSize = (fiaInfo.s1Size + sOuterSize - 1) / sOuterSize;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(curCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize()));
}

void FusedInferAttentionScoreTilingImpl::SplitNBSeq(const FiaTilingInfo &fiaInfo)
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    uint32_t sOuterSize = sOuterFactor_ * CV_RATIO;
    uint32_t sInnerSize = sInnerFactor_;
    int64_t totalBlockNumsOneHead = 0;
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    std::vector<uint32_t> sInnerLoopTimes(fiaInfo.bSize);
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengthsQ_[bIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(fiaInfo, actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        sInnerLoopTimes[bIdx] = (actualSeqLengthsKV_[bIdx] + sInnerSize - 1) / sInnerSize +
                                (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[bIdx]);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKV_[bIdx],
                                                         sOuterSize, sInnerSize, preTokensLeftUp, nextTokensLeftUp);
    }
    pfaTilingData_.promptAttentionSingleCoreParams.set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);

    double coreWeightTarget = (double(totalBlockNumsOneHead * nLoopTimes_) / double(curCoreNum));

    int64_t s1OuterSize = (gsSize_ + sOuterSize - 1) / sOuterSize;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    // The tiling structure element needs to have a length greater than or equal to the length specified
    // by TILING_DATA_FIELD_DEF_ARR. If the tiling structure definition specifies a length of 64,
    // the vector definition needs to compare its size with coreNum and take the larger value
    const size_t tilingElementArrayLen =
        (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    uint32_t curIndx = 0;

    ComputeSplitNBSeq(fiaInfo, tilingElementArrayLen, sOuterSize, sInnerSize, coreWeightTarget, curIndx);

    uint32_t actualCoreNums = (curIndx + 1) * CV_RATIO;
    pfaTilingData_.promptAttentionSingleCoreParams.set_actualCoreNums(actualCoreNums);
    int64_t sinnerBlocknum = (fiaInfo.s2Size + sInnerSize - 1) / sInnerSize;
    int64_t totalSize = (totalBlockNumsOneHead / sinnerBlocknum) * nLoopTimes_;
    int64_t aicUsedCoreNum = curIndx + 1;

    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(aicUsedCoreNum));
    OP_LOGI(fiaInfo.opName, "aicUsedCoreNum: %ld", aicUsedCoreNum);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, aicUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize()));
}

bool FusedInferAttentionScoreTilingImpl::CheckFlashDecode(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.s1Size == 1 && fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
        return false;
    }
    float flashDecodeBNRatio = 0.4F; // 0.4, 经验值
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        if (fiaInfo.sysPrefixFlag) {
            return false;
        }
        // 如果s2方向上最长还不超过两个sinnersize，不生效FD
        if (fiaInfo.maxActualseq < sInnerFactor_ * 2) {
            return false;
        }
    } else {
        if (fiaInfo.s1Size != 1 || fiaInfo.ropeMode != RopeMode::NO_ROPE) {
            return false;
        }
        if (fiaInfo.maxActualseq < NUM_256) {
            return false;
        }
    }
    uint64_t loopTimes = fiaInfo.bSize * fiaInfo.n2Size * (fiaInfo.gSize + sOuterFactor_ - 1) / sOuterFactor_;
    if (loopTimes < flashDecodeBNRatio * platformInfo_.aicNum && fiaInfo.gSize == 1) {
        return true;
    }
    if (loopTimes < flashDecodeBNRatio * platformInfo_.aicNum &&
        (fiaInfo.s2Size >= NUM_2048)) { // 2048, 在flash decode + gqa时的经验值
        return true;
    }
    return false;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SplitS2(const FiaTilingInfo &fiaInfo)
{
    uint64_t batchSize = fiaInfo.bSize;
    uint64_t headNumSize = fiaInfo.n1Size;
    uint32_t kvSplitLimit =
        sInnerFactor_ <= 256U ?
            256U :
            sInnerFactor_; // 256: 经验值 这里的门限值也需要调整，否则会分的很碎，每个核分到的还不到一个基本块
    uint64_t loopTimes = fiaInfo.bSize * fiaInfo.n2Size * (fiaInfo.gSize + sOuterFactor_ - 1) / sOuterFactor_;
    int64_t kvSplitPart = platformInfo_.aicNum / loopTimes;
    while (((fiaInfo.s2Size / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }

    faRunTilingAdapter_.inputParamsRegbase.set_kvSplitPart(kvSplitPart);
    faRunTilingAdapter_.inputParamsRegbase.set_accumOutSize(batchSize * headNumSize * kvSplitPart * headDimAlign_);
    faRunTilingAdapter_.inputParamsRegbase.set_logSumExpSize(batchSize * headNumSize * kvSplitPart *
                                                             (BYTE_BLOCK / sizeof(float)));

    return ge::GRAPH_SUCCESS;
}

// 伪量化
void FusedInferAttentionScoreTilingImpl::SetDequantBaseSize(const FiaTilingInfo &fiaInfo)
{
    sOuterFactor_ = NUM_16;
    if (fiaInfo.s1Size > 1 && gsMergeFlag_) { // pfa gs1合轴时 s1base=32
        sOuterFactor_ = NUM_32;
    }
    if (fiaInfo.qkHeadDim <= NUM_64) {
        sInnerFactor_ = NUM_1024;
        if (fiaInfo.pseShiftFlag || (fiaInfo.s1Size > 1 && gsMergeFlag_)) { // pfa gs1合轴 s1base=32 s2base=512 dbase=64
            sInnerFactor_ = NUM_512;
        }
    } else if (fiaInfo.qkHeadDim <= NUM_128) {
        sInnerFactor_ = NUM_512;
    } else if (fiaInfo.qkHeadDim <= NUM_256) {
        sInnerFactor_ = NUM_256;
    } else {
        sInnerFactor_ = NUM_128;
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::CalcInnerSize(const FiaTilingInfo &fiaInfo, uint32_t seqSize)
{
    /**
     * sInnerFactor：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * GQA场景-伪量化：vector比较重，尽量较少vector的循环次数,
     * 因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    SetDequantBaseSize(fiaInfo);
    sInnerFactorSize_ = sInnerFactor_;
    if (sInnerFactor_ > seqSize) {
        sInnerFactor_ = seqSize;
    }
    sInnerSizeAlign_ = AlignUp(sInnerFactor_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    return ge::GRAPH_SUCCESS;
}

int64_t FusedInferAttentionScoreTilingImpl::GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                                    int64_t innerBlockNums)
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }

    return sInnerBlockNums;
}

void FusedInferAttentionScoreTilingImpl::GetAntiQuantPreNextTokensLeftUp(const FiaTilingInfo &fiaInfo,
                                                                         int64_t actualSeqLength,
                                                                         int64_t actualSeqLengthKV,
                                                                         int64_t &preTokensLeftUp,
                                                                         int64_t &nextTokensLeftUp)
{
    preTokensLeftUp = fiaInfo.preToken;
    nextTokensLeftUp = fiaInfo.nextToken;
    if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
    } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
        preTokensLeftUp = fiaInfo.preToken - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = fiaInfo.nextToken + actualSeqLengthKV - actualSeqLength;
    }
}

void FusedInferAttentionScoreTilingImpl::FixAntiQuantParamWithRowInvalid(const FiaTilingInfo &fiaInfo,
                                                                         int64_t &actualSeqLength,
                                                                         int64_t actualSeqLengthKV,
                                                                         int64_t &preTokensLeftUp,
                                                                         int64_t &nextTokensLeftUp)
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;

    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                                 (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) :
                                 0;
    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
    if (nextTokensError != 0 || preTokensError != 0) {
        needInit_ = true;
    }
}

void FusedInferAttentionScoreTilingImpl::ComputeDequantSplitNBSeq(
    const FiaTilingInfo &fiaInfo, std::vector<int64_t> sOuterLoopTimes, std::vector<int64_t> sInnerLoopTimes,
    int64_t sInnerLoopTimesPrefix, double coreWeightTarget, uint32_t &curCore, const size_t tilingElementArrayLen)
{
    std::vector<int64_t> sparseStartIdx(tilingElementArrayLen, 0L);
    std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
    // Temporary algorithm to be optimized
    int64_t curWeight = 0;
    uint32_t tmpCoreNidEnd = 0; // actual seq为0时不分配核
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    int64_t actualSeqLengthsTmp = 0;
    int64_t actualSeqLengthsKVTmp = 0;
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        GetActualSeqLength(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp, bIdx);
        for (uint32_t headNum = 0; headNum < nLoopTimes_; headNum++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetAntiQuantPreNextTokensLeftUp(fiaInfo, actualSeqLengthsTmp,
                                            actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen, preTokensLeftUp,
                                            nextTokensLeftUp);
            FixAntiQuantParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp,
                                            actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen, preTokensLeftUp,
                                            nextTokensLeftUp);
            int64_t outerBlockNums = sOuterLoopTimes[bIdx];
            int64_t innerBlockNums = sInnerLoopTimes[bIdx];
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWeightTarget * double(curCore + 1)) - curWeight;
                // 非prefix部分计算，去除prefix影响
                int64_t preTokensNoPrefix = preTokensLeftUp + fiaInfo.systemPrefixLen;
                int64_t nextTokensNoPrefix = nextTokensLeftUp - fiaInfo.systemPrefixLen;
                int64_t sInnerIndexStart =
                    -(preTokensNoPrefix > 0 ? (preTokensNoPrefix + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                  static_cast<int64_t>(sInnerFactor_) :
                                              preTokensNoPrefix / static_cast<int64_t>(sInnerFactor_));
                int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ?
                                             (nextTokensNoPrefix + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                 static_cast<int64_t>(sInnerFactor_) :
                                             nextTokensNoPrefix / static_cast<int64_t>(sInnerFactor_);

                // prefix部分单独计算
                int64_t sInnerIndexStartPrefix =
                    -(preTokensLeftUp > 0 ? (preTokensLeftUp + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                static_cast<int64_t>(sInnerFactor_) :
                                            preTokensLeftUp / static_cast<int64_t>(sInnerFactor_));
                int64_t sInnerIndexEndPrefix = nextTokensLeftUp > 0 ?
                                                   (nextTokensLeftUp + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                       static_cast<int64_t>(sInnerFactor_) :
                                                   nextTokensLeftUp / static_cast<int64_t>(sInnerFactor_);

                // 当前这一行有多少基本块需要计算
                int64_t actualInnerBlockNums =
                    GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
                    GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, sInnerLoopTimesPrefix);
                if (actualInnerBlockNums - dif > dif &&
                    !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    curCore += 1;
                    bnStartIdx[curCore] = bIdx * nLoopTimes_ + headNum;
                    gS1StartIdx[curCore] = sOuterIndex;
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = bIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWeight += actualInnerBlockNums;
                preTokensLeftUp -= sOuterFactor_;
                nextTokensLeftUp += sOuterFactor_;
            }
        }
    }
    bnStartIdx[curCore + 1] = fiaInfo.bSize * nLoopTimes_;
    gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

    faRunTilingAdapter_.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faRunTilingAdapter_.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

void FusedInferAttentionScoreTilingImpl::GetActualSeqLength(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLengths,
                                                            int64_t &actualSeqLengthsKV, uint32_t bIdx)
{
    if (fiaInfo.isMaxWorkspace) {
        actualSeqLengths = fiaInfo.s1Size;
        actualSeqLengthsKV = fiaInfo.s2Size;
        return;
    }
    uint32_t nNumOfQInOneGroup = 1;
    nNumOfQInOneGroup = fiaInfo.gSize;
    if (fiaInfo.qLayout == FiaLayout::TND) {
        actualSeqLengths = bIdx == 0 ? fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
                                       fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx] -
                                           fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx - 1];
        if (gsMergeFlag_) {
            actualSeqLengths *= nNumOfQInOneGroup;
        }
        actualSeqLengthsKV = fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
        if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION && bIdx > 0) {
            actualSeqLengthsKV -= fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx - 1];
        }
    } else {
        if (fiaInfo.actualSeqLenFlag && fiaInfo.actualLenDims > 0 &&
            fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) { // kvLengths
            actualSeqLengthsKV = fiaInfo.actualLenDims == NUM1 ?
                                     fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[0] :
                                     fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
        } else {
            actualSeqLengthsKV =
                fiaInfo.kvListSeqLens.size() == NUM1 ? fiaInfo.kvListSeqLens[0] : fiaInfo.kvListSeqLens[bIdx];
        }
        if (actualSeqLengthsKV < fiaInfo.s2Size) {
            needInit_ = true;
        }
        if (actualSeqLenQFlag_) { // qLengths
            actualSeqLengths = fiaInfo.actualLenQDims == NUM1 ?
                                   fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
                                   fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx];
        } else {
            actualSeqLengths = fiaInfo.s1Size;
            if (gsMergeFlag_) {
                actualSeqLengths = fiaInfo.s1Size * nNumOfQInOneGroup;
            }
        }
    }
}

int64_t FusedInferAttentionScoreTilingImpl::GetAntiQuantCalcBlockNumsOneHead(
    const FiaTilingInfo &fiaInfo, int64_t outerBlockNums, int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp)
{
    if (!fiaInfo.attenMaskFlag) {
        return (innerBlockNums + sInnerLoopTimesPrefix) * outerBlockNums;
    } else {
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterFactor_);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerFactor_);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // 必须满足pretoken + nexttoken > 0，否则会减出小于0的块数，这里需要去除prefix影响
        toCalcBlockNums -=
            GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerFactor_),
                            static_cast<int64_t>(sOuterFactor_), nextTokensLeftUp - fiaInfo.systemPrefixLen);
        toCalcBlockNums -= GetCutBlockNums(
            blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerFactor_), static_cast<int64_t>(sOuterFactor_),
            blockSeqLengthKV - blockSeqLength + preTokensLeftUp + fiaInfo.systemPrefixLen);

        // prefix部分单独计算
        int64_t blockSharedPrefix = sInnerLoopTimesPrefix * static_cast<int64_t>(sInnerFactor_);
        toCalcBlockNums += sInnerLoopTimesPrefix * outerBlockNums;
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerFactor_),
                                           static_cast<int64_t>(sOuterFactor_), nextTokensLeftUp);
        toCalcBlockNums -=
            GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerFactor_),
                            static_cast<int64_t>(sOuterFactor_), blockSharedPrefix - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}

void FusedInferAttentionScoreTilingImpl::DequantCubeSplitBNSeq(const FiaTilingInfo &fiaInfo) // 伪量化只用Cube视角分核
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    int64_t totalBlockNumsOneHead = 0;
    std::vector<int64_t> sOuterLoopTimes(fiaInfo.bSize, 0U);
    std::vector<int64_t> sInnerLoopTimes(fiaInfo.bSize, 0U);
    int64_t multiSmaxsInnerLoopTimes = 0U;
    int64_t sInnerLoopTimesPrefix =
        (fiaInfo.systemPrefixLen + static_cast<int64_t>(sInnerFactor_) - 1) / static_cast<int64_t>(sInnerFactor_);
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = 0;
        int64_t actualSeqLengthsKVTmp = 0;
        GetActualSeqLength(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp, bIdx);
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetAntiQuantPreNextTokensLeftUp(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen,
                                        preTokensLeftUp, nextTokensLeftUp);
        FixAntiQuantParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen,
                                        preTokensLeftUp, nextTokensLeftUp);

        sOuterLoopTimes[bIdx] =
            (actualSeqLengthsTmp + static_cast<int64_t>(sOuterFactor_) - 1) / static_cast<int64_t>(sOuterFactor_);
        sInnerLoopTimes[bIdx] =
            (actualSeqLengthsKVTmp + static_cast<int64_t>(sInnerFactor_) - 1) / static_cast<int64_t>(sInnerFactor_);
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[bIdx] + sInnerLoopTimesPrefix);

        totalBlockNumsOneHead +=
            GetAntiQuantCalcBlockNumsOneHead(fiaInfo, sOuterLoopTimes[bIdx], sInnerLoopTimes[bIdx],
                                             sInnerLoopTimesPrefix, preTokensLeftUp, nextTokensLeftUp);
    }
    double coreWeightTarget = (double(totalBlockNumsOneHead * nLoopTimes_) / double(curCoreNum));

    int64_t s1OuterSize = (fiaInfo.s1Size + sOuterFactor_ - 1) / sOuterFactor_;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);
    const size_t tilingElementArrayLen =
        (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    uint32_t curIndx = 0;
    ComputeDequantSplitNBSeq(fiaInfo, sOuterLoopTimes, sInnerLoopTimes, sInnerLoopTimesPrefix, coreWeightTarget,
                             curIndx, tilingElementArrayLen);
    int64_t sinnerBlocknum = (fiaInfo.maxActualseq + sInnerFactor_ - 1) / sInnerFactor_;
    int64_t totalSize = (totalBlockNumsOneHead / sinnerBlocknum) * nLoopTimes_;
    int64_t actualUsedCoreNum = static_cast<int64_t>((curIndx + 1));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize()));
}

void FusedInferAttentionScoreTilingImpl::SplitDequant(const FiaTilingInfo &fiaInfo)
{
    CalcInnerSize(fiaInfo, fiaInfo.maxActualseq);
    DequantCubeSplitBNSeq(fiaInfo);
    if (!isPFAFlag_) {
        if (CheckFlashDecode(fiaInfo)) {
            flashDecodeFlag_ = true;
            SplitS2(fiaInfo);
        }
    }
}

bool FusedInferAttentionScoreTilingImpl::CheckEnableDN(const FiaTilingInfo &fiaInfo)
{
    constexpr uint32_t dLimitDN = DSIZE_128;
    constexpr uint32_t sOuterLimitDN = SOUTER_64;
    bool res = !fiaInfo.attenMaskFlag && !fiaInfo.pseShiftFlag && !fiaInfo.enableAlibiPse &&
               fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION && fiaInfo.ropeMode == RopeMode::NO_ROPE &&
               fiaInfo.qkHeadDim <= dLimitDN && fiaInfo.vHeadDim <= dLimitDN && !fiaInfo.sysPrefixFlag &&
               (fiaInfo.quantMode == FiaQuantMode::NO_QUANT ||
                fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) &&
               sOuterFactor_ * CV_RATIO > sOuterLimitDN;
    return res;
}

bool FusedInferAttentionScoreTilingImpl::CheckQKVActualSeqLengthsRight(const FiaTilingInfo &fiaInfo)
{
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        if ((actualSeqLengthsQ_[bIdx] % SOUTER_32 > 0) || (actualSeqLengthsKV_[bIdx] <= SINNER_128)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SplitPolicy(gert::TilingContext *context,
                                                                const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        SplitDequant(fiaInfo);
    } else {
        OP_CHECK_IF(AdjustSinnerAndSouter(context, fiaInfo) != ge::GRAPH_SUCCESS, // 确认souter/sinner
                    OP_LOGE(fiaInfo.opName, "Get sinner and souter fail!"), return ge::GRAPH_FAILED);

        dnFlag_ = CheckEnableDN(fiaInfo);
        isQKVActualSeqLengthsRightFlag_ = CheckQKVActualSeqLengthsRight(fiaInfo);

        if (dnFlag_ && isQKVActualSeqLengthsRightFlag_ && fiaInfo.qkHeadDim == fiaInfo.vHeadDim &&
            fiaInfo.qkHeadDim == DSIZE_64) {
            sInnerFactor_ = SINNER_256;
        }
        if (dnFlag_ && fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT &&
            fiaInfo.qkHeadDim == fiaInfo.vHeadDim && fiaInfo.qkHeadDim <= 128) {
            sInnerFactor_ = SINNER_256;
        }

        if (CheckGQAPerblockNz(fiaInfo)) {
            sOuterFactor_ = SOUTER_64;
            sInnerFactor_ = SINNER_512;
        }

        enableS1OutSplit = CheckS1OutSplit(fiaInfo);
        if (enableS1OutSplit) {
            SplitOutSeq(fiaInfo);
        } else {
            SplitNBSeq(fiaInfo);
            if (isIFAFlag_ && !pfaMergeFlag_ && CheckFlashDecode(fiaInfo)) {
                flashDecodeFlag_ = true;
                OP_LOGI(fiaInfo.opName, "FlashDecode is enable.");
                SplitS2(fiaInfo);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyConfig(const FiaTilingInfo &fiaInfo)
{
    auto sOuter = sOuterFactor_ * 2;
    auto sInner = sInnerFactor_;
    auto dSize = fiaInfo.qkHeadDim;
    auto dVsize = fiaInfo.vHeadDim;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        dSize = fiaInfo.qkHeadDim + fiaInfo.ropeHeadDim;
    }
    if (dSize <= DSIZE_64)
        dSize = DSIZE_64;
    else if (dSize <= DSIZE_128)
        dSize = DSIZE_128;
    else if (dSize <= DSIZE_256)
        dSize = DSIZE_256;
    else if (dSize <= DSIZE_512)
        dSize = DSIZE_512;
    else if (dSize <= DSIZE_576)
        dSize = DSIZE_576;

    if (dVsize <= DSIZE_64)
        dVsize = DSIZE_64;
    else if (dVsize <= DSIZE_128)
        dVsize = DSIZE_128;
    else if (dVsize <= DSIZE_256)
        dVsize = DSIZE_256;
    else if (dVsize <= DSIZE_512)
        dVsize = DSIZE_512;

    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) { // 伪量化场景
        sInner = sInnerFactorSize_;
        sOuter = sOuterFactor_;
        if (sInner == 1024 && sOuter == 16 && dSize <= DSIZE_64 && dVsize <= DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64;
        } else if (sInner == 512 && sOuter == 16 && dSize <= DSIZE_64 && dVsize <= DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64;
        } else if (sInner == 512 && sOuter == 16 && dSize <= DSIZE_128 && dVsize <= DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128;
        } else if (sInner == 256 && sOuter == 16 && dSize <= DSIZE_256 && dVsize <= DSIZE_256) {
            tilingKeyInfo_.config = Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256;
        } else if (sInner == 128 && sOuter == 16 && dSize <= DSIZE_512 && dVsize <= DSIZE_512) {
            tilingKeyInfo_.config = Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512;
        } else if (sInner == 512 && sOuter == 32 && dSize <= 64 &&
                   dVsize <= 64) { // 以下为PFA伪量化合轴场景 32:s1base 512:s2base 64:dbase
            tilingKeyInfo_.config = Config_S1Aligned32_S2Aligned512_DAligned64_DVAligned64;
        } else if (sInner == 512 && sOuter == 32 && dSize <= DSIZE_128 &&
                   dVsize <= DSIZE_128) { // 32:s1base 512:s2base 128:dbase
            tilingKeyInfo_.config = Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128;
        } else if (sInner == 256 && sOuter == 32 && dSize <= DSIZE_256 &&
                   dVsize <= DSIZE_256) { // 32:s1base 256:s2base 256:dbase
            tilingKeyInfo_.config = Config_S1Aligned32_S2Aligned256_DAligned256_DVAligned256;
        } else if (sInner == 128 && sOuter == 32 && dSize <= DSIZE_512 &&
                   dVsize <= DSIZE_512) { // 32:s1base 128:s2base 512:dbase
            tilingKeyInfo_.config = Config_S1Aligned32_S2Aligned128_DAligned512_DVAligned512;
        } else {
        }
    } else { // 非量化+全量化 场景
        if (sOuter == SOUTER_64 && sInner == SINNER_64 && dSize == DSIZE_256 && dVsize == DSIZE_256) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256;
        } else if (sOuter == SOUTER_64 && sInner == SINNER_64 && dSize == DSIZE_512 && dVsize == DSIZE_512) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512;
        } else if (sOuter == SOUTER_64 && sInner == SINNER_256 && dSize == DSIZE_64 && dVsize == DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64;
        } else if (sOuter == SOUTER_64 && sInner == SINNER_256 && dSize == DSIZE_128 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_64 && dVsize == DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_128 && dVsize == DSIZE_128) {
            if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
                tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128;
            } else {
                tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128;
            }
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_192 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_256 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_256 && dVsize == DSIZE_256) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_512 && dVsize == DSIZE_512) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_256 && dSize == DSIZE_64 && dVsize == DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64;
        } else if (sOuter == SOUTER_64 && sInner == SINNER_128 && dSize == DSIZE_576 && dVsize == DSIZE_512) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512;
        } else if (sOuter == SOUTER_64 && sInner == SINNER_256 && dSize == DSIZE_256 && dVsize == DSIZE_256) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_256 && dSize == DSIZE_128 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_128 && dVsize == DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64; // qkvd不等长
        } else if (sOuter == SOUTER_128 && sInner == SINNER_128 && dSize == DSIZE_64 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128; // qkvd不等长
        } else if (sOuter == SOUTER_64 && sInner == SINNER_256 && dSize == DSIZE_128 && dVsize == DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64; // qkvd不等长
        } else if (sOuter == SOUTER_64 && sInner == SINNER_256 && dSize == DSIZE_64 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128; // qkvd不等长
        } else if (sOuter == SOUTER_128 && sInner == SINNER_512 && dSize == DSIZE_64 && dVsize == DSIZE_64) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64;
        } else if (sOuter == SOUTER_128 && sInner == SINNER_512 && dSize == DSIZE_128 && dVsize == DSIZE_128) {
            tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128;
        } else {
        }
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyLayout(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.qLayout == FiaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BNSD;
    } else if (fiaInfo.qLayout == FiaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
    } else if (fiaInfo.qLayout == FiaLayout::NTD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_NTD_NTD;
    } else if (fiaInfo.qLayout == FiaLayout::BSH || fiaInfo.qLayout == FiaLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyPseMode(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.pseShiftFlag && !fiaInfo.enableAlibiPse) {
        tilingKeyInfo_.pseMode = PSE_MODE_PSE_NONE_TYPE;
    } else {
        tilingKeyInfo_.pseMode = *(fiaInfo.opParamInfo.pseType);
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyQuantMode(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.quantMode == FiaQuantMode::NO_QUANT) {
        tilingKeyInfo_.quantMode = NoQuantMode;
    }

    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        uint32_t antiQuantMode = fiaInfo.keyAntiquantMode;
        tilingKeyInfo_.quantMode = 0; // 默认处理，与重构前保持一致
        switch (antiQuantMode) {
            case 0:
                tilingKeyInfo_.quantMode = (fiaInfo.valueAntiquantMode == 1) ? AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN :
                                                                               AntiquantMode_PER_CHANNEL;
                break;
            case 1:
                tilingKeyInfo_.quantMode = AntiquantMode_PER_TOKEN;
                break;
            case 2:
                tilingKeyInfo_.quantMode =
                    AntiquantMode_PER_CHANNEL; // perTensorHead,使用同perChannel,通过perHeadFlag来区分
                break;
            case 3:
                tilingKeyInfo_.quantMode = AntiquantMode_PER_TOKEN; // perTokenHead,使用同perToken,通过perHeadFlag来区分
                break;
            case 4:
                tilingKeyInfo_.quantMode = AntiquantMode_PER_TOKEN_PAGE_ATTENTION;
                break;
            case 5:
                tilingKeyInfo_.quantMode = AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION;
                break;
            default:
                break;
        }
    }

    if (fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
            tilingKeyInfo_.quantMode = FULLQUANT_MODE_PER_TOKEN_HEAD;
        } else if (*fiaInfo.opParamInfo.keyAntiquantMode == 7 && *fiaInfo.opParamInfo.valueAntiquantMode == 7 &&
                   *fiaInfo.opParamInfo.queryQuantMode == 7) {
            tilingKeyInfo_.quantMode = PerBlock;
        } else {
            tilingKeyInfo_.quantMode = FullQuantMode;
        }
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyHasRope(const FiaTilingInfo &fiaInfo)
{
    tilingKeyInfo_.hasRope = false;
    if (fiaInfo.quantMode == FiaQuantMode::NO_QUANT) {
        if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) {
            tilingKeyInfo_.hasRope = true;
        }
    }
    if (fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
            tilingKeyInfo_.hasRope = true;
        }
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyMaskMode(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
        if (!fiaInfo.attenMaskFlag) {
            tilingKeyInfo_.maskMode = PFAMask_DISABLE_MASK;
        } else if (fiaInfo.sparseMode == 4) {
            tilingKeyInfo_.maskMode = PFAMask_ENABLE_MASK_BAND;
        } else {
            tilingKeyInfo_.maskMode = PFAMask_ENABLE_MASK_NO_BAND;
        }
    } else {
        tilingKeyInfo_.maskMode = 0;
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyMatmulMode(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.quantMode == FiaQuantMode::NO_QUANT || fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        tilingKeyInfo_.matmulMode = 0;
    }
    if (fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        if (tilingKeyInfo_.quantMode != FullQuantMode) {
            tilingKeyInfo_.matmulMode = 0;
        } else { // 当前仅int8 per-tensor 量化场景
            if (fiaInfo.qkHeadDim == 512) {
                if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
                    tilingKeyInfo_.matmulMode = PFAMatMulType_MM_PA_D512;
                } else {
                    tilingKeyInfo_.matmulMode = PFAMatMulType_MM_IFA_MLA;
                }
            } else if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
                tilingKeyInfo_.matmulMode = PFAMatMulType_MM_PA;
            } else {
                tilingKeyInfo_.matmulMode = PFAMatMulType_MM_PFA;
            }
        }
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::UpdateTilingKeyInfo(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.emptyTensorFlag) {
        tilingKeyInfo_.emptyTensor = fiaInfo.emptyTensorFlag;
    } else {
        UpdateTilingKeyLayout(fiaInfo);
        UpdateTilingKeyConfig(fiaInfo);
        UpdateTilingKeyPseMode(fiaInfo);
        UpdateTilingKeyQuantMode(fiaInfo);
        tilingKeyInfo_.isFd = flashDecodeFlag_;
        if (fiaInfo.sysPrefixFlag) {
            tilingKeyInfo_.isFd = false;
        }
        tilingKeyInfo_.hasAttenMask = fiaInfo.attenMaskFlag;
        if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
            tilingKeyInfo_.hasAttenMask = false;
        }
        UpdateTilingKeyHasRope(fiaInfo);
        tilingKeyInfo_.isPa = fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION;
        tilingKeyInfo_.emptyTensor = fiaInfo.emptyTensorFlag;
        UpdateTilingKeyMaskMode(fiaInfo);
        UpdateTilingKeyMatmulMode(fiaInfo);
        tilingKeyInfo_.enableKvPrefix = fiaInfo.sysPrefixFlag;
        tilingKeyInfo_.enableS1OutSplit = enableS1OutSplit;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::GenTilingKey(gert::TilingContext *context,
                                                                 const FiaTilingInfo &fiaInfo)
{
    UpdateTilingKeyInfo(fiaInfo);
    uint64_t genTilingkey = GET_TPL_TILING_KEY(
        tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
        tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.isPa, tilingKeyInfo_.isFd,
        tilingKeyInfo_.emptyTensor, tilingKeyInfo_.maskMode, tilingKeyInfo_.matmulMode, tilingKeyInfo_.enableKvPrefix,
        tilingKeyInfo_.enableS1OutSplit);
    context->SetTilingKey(genTilingkey);
    OP_LOGI(fiaInfo.opName, "The tilingkey is %llu.", genTilingkey);
    OP_LOGI(fiaInfo.opName,
            "The tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu, quantMode: %llu, "
            "hasAttenMask: %llu, hasRope: %llu, isPa: %llu, isFd: %llu, emptyTensor: %llu, PFAMask: %llu, "
            "pFAMatMulType: %llu, enableKvPrefix: %llu, enableS1OutSplit: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
            tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.isPa, tilingKeyInfo_.isFd,
            tilingKeyInfo_.emptyTensor, tilingKeyInfo_.maskMode, tilingKeyInfo_.matmulMode,
            tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetBlockDim(gert::TilingContext *context,
                                                                const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    blockDim_ = ascendcPlatform.CalcTschBlockDim(platformInfo_.aivNum, platformInfo_.aicNum, platformInfo_.aivNum);
    context->SetBlockDim(blockDim_);
    OP_LOGI(fiaInfo.opName, "blockDim: %d", blockDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetWorkspaceNormal(const FiaTilingInfo &fiaInfo,
                                                                       int64_t &curWorkspaceSize)
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    size_t accumOutSize = 0;
    size_t logSumExpSize = 0;
    if (fiaInfo.isMaxWorkspace) { // 计算maxWorkSpaceSize时默认开启FD且使用最大核数进行归约
        auto vHeadSize = AlignUp(fiaInfo.vHeadDim, NUM_64);
        accumOutSize = platformInfo_.aicNum * vHeadSize * sizeof(float);
        logSumExpSize = platformInfo_.aicNum * BYTE_BLOCK * CV_RATIO;
    } else if (flashDecodeFlag_) {
        auto batchSize = fiaInfo.bSize;
        auto headNumSize = nLoopTimes_;
        auto vHeadSize = fiaInfo.vHeadDim;
        uint32_t kvSplitPart = faRunTilingAdapter_.inputParamsRegbase.get_kvSplitPart();
        accumOutSize = batchSize * fiaInfo.gSize * headNumSize * kvSplitPart * vHeadSize * sizeof(float);
        logSumExpSize = batchSize * fiaInfo.gSize * headNumSize * kvSplitPart * BYTE_BLOCK * 2;
    }

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    uint32_t dSize = fiaInfo.vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= DSIZE_64) {
        dVBasicBlock = DSIZE_64;
    } else if (dSize <= DSIZE_128) {
        dVBasicBlock = DSIZE_128;
    } else if (dSize <= DSIZE_256) {
        dVBasicBlock = DSIZE_256;
    } else if (dSize <= DSIZE_512) {
        dVBasicBlock = DSIZE_512;
    }
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > DSIZE_256) {
        bmm2ResBlockSize = DSIZE_512;
    }
    uint32_t s1BasicBlock = sOuterFactor_ * CV_RATIO;
    uint32_t calcTypeSize = sizeof(float);
    if ((!dnFlag_ && dSize > DSIZE_128) || (dnFlag_ && dSize > DSIZE_192)) {
        bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
        if (dVBasicBlock > DSIZE_256) {
            vec2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
        }
    }
    curWorkspaceSize = (bmm2Bytes + vec2Bytes) * 3 * platformInfo_.coreNum + // 3: perload 2次 需要2+1
                       sysWorkspaceSize + accumOutSize + logSumExpSize;

    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        curWorkspaceSize += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetWorkspaceAntiQuant(const FiaTilingInfo &fiaInfo,
                                                                          int64_t &workspaceSize_)
{
    uint32_t cubeL1UbSize = (512 / 2) * 1024;  // david L1 512K,提供给两个Vec使用,单个vector占用256K
    uint32_t cubeL0CUbSize = (256 / 2) * 1024; // david L0C 256K,提供给两个Vec使用,单个vector占用128K

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;
    // L1
    workspaceSize_ += cubeL1UbSize * platformInfo_.coreNum;
    // L0C
    workspaceSize_ += cubeL0CUbSize * platformInfo_.coreNum;
    if (fiaInfo.isMaxWorkspace) { // 计算maxWorkSpaceSize时默认开启FD且使用最大核数进行归约
        uint32_t maxAccumOutSize = platformInfo_.aicNum * fiaInfo.qkHeadDim;
        uint32_t maxLogSumExpSize = platformInfo_.aicNum * (BYTE_BLOCK / sizeof(float));
        workspaceSize_ += (maxAccumOutSize + maxLogSumExpSize * 2) * sizeof(float); // 2 : sMax 和 sSum
    } else if (flashDecodeFlag_) {
        workspaceSize_ += (faRunTilingAdapter_.inputParamsRegbase.get_accumOutSize() +
                           faRunTilingAdapter_.inputParamsRegbase.get_logSumExpSize() * 2) *
                          sizeof(float); // 2 : sMax 和 sSum
    }

    workspaceSize_ += 100 * 1024 * 1024; // 100*1024*1024: extra workspace for dump in david

    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        workspaceSize_ += platformInfo_.coreNum * 64 * 2; // bmm1 bmm2 2份
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetWorkspacePTQuant(const FiaTilingInfo &fiaInfo,
                                                                        int64_t &curWorkspaceSize)
{
    uint64_t maxSpmSize = 0; // 待处理 tilingData.promptAttentionTensorSizeRect.get_spmTmpSize();
    int64_t mm1ResSize = sOuterFactor_ * CV_RATIO * sInnerFactor_;
    int64_t mm2ResSize = sOuterFactor_ * CV_RATIO * fiaInfo.vHeadDim;
    curWorkspaceSize = platformInfo_.defaultSysWorkspaceSize +
                       platformInfo_.coreNum * 2 * (maxSpmSize + mm1ResSize * 2 + mm2ResSize * 2);
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        curWorkspaceSize += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::GetWorkspace(gert::TilingContext *context,
                                                                 const FiaTilingInfo &fiaInfo)
{
    int64_t workspace = 0;
    if (fiaInfo.emptyTensorFlag) {
        workspace = platformInfo_.defaultSysWorkspaceSize;
    } else if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        OP_CHECK_IF(SetWorkspaceAntiQuant(fiaInfo, workspace) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Get workspace failed ."), return ge::GRAPH_FAILED);

    } else if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
        OP_CHECK_IF(SetWorkspacePTQuant(fiaInfo, workspace) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Get workspace failed ."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(SetWorkspaceNormal(fiaInfo, workspace) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Get workspace failed ."), return ge::GRAPH_FAILED);
    }

    OP_LOGI(fiaInfo.opName, "Workspaces: %ld", workspace);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = workspace;
    return ge::GRAPH_SUCCESS;
}

bool FusedInferAttentionScoreTilingImpl::EnableMTE2BmmPipe(const FiaTilingInfo &fiaInfo,
                                                           matmul_tiling::MatmulApiTiling &bmm,
                                                           TCubeTiling &bmmTilingData)
{
    if (fiaInfo.s1Size > 16) {
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t headSize = fiaInfo.qkHeadDim;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        headSize = fiaInfo.qkHeadDim + fiaInfo.ropeHeadDim;
    }
    if (headSize % baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor_);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor_);
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        baseN = 128; // 128
    }
    bool res = (bmm.SetFixSplit(baseM, baseN, baseK) == ge::GRAPH_SUCCESS);
    OP_CHECK_IF(!res, OP_LOGE(fiaInfo.opName, "set fix split fail"), return ge::GRAPH_FAILED);
    // check
    res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}
void FusedInferAttentionScoreTilingImpl::GetMatMulType(const FiaTilingInfo &fiaInfo,
                                                       matmul_tiling::DataType &mmInputType,
                                                       matmul_tiling::DataType &mmOutputType)
{
    if (fiaInfo.inputQType == ge::DT_FLOAT16 && fiaInfo.innerPrecise == 0) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (fiaInfo.inputQType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (fiaInfo.inputQType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_INT32;
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetMM1TilingData(gert::TilingContext *context,
                                                                     const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);

    matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_INT8;
    matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_INT32;
    GetMatMulType(fiaInfo, bmm1InputType, bmm1OutputType);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm1OutputType);

    auto qkHeadDim = fiaInfo.qkHeadDim;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        qkHeadDim = fiaInfo.qkHeadDim + fiaInfo.ropeHeadDim;
    }
    int32_t ret = bmm1.SetShape(sOuterFactor_, sInnerFactor_, qkHeadDim);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Bmm1 set shape space failed"),
                return ge::GRAPH_FAILED);
    if ((fiaInfo.qLayout == FiaLayout::BSH) || (fiaInfo.qLayout == FiaLayout::BSND) ||
        (fiaInfo.qLayout == FiaLayout::TND)) {
        if (fiaInfo.qLayout == FiaLayout::TND && fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION &&
            fiaInfo.kvLayout == FiaLayout::BnNBsD) {
            bmm1.SetOrgShape(gsSize_, fiaInfo.s2Size, qkHeadDim * nLoopTimes_, qkHeadDim);
        } else if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 || fiaInfo.s1Size == 1) {
            bmm1.SetOrgShape(gsSize_, fiaInfo.s2Size, qkHeadDim, fiaInfo.n2Size * qkHeadDim);
        } else {
            bmm1.SetOrgShape(gsSize_, fiaInfo.s2Size, qkHeadDim * nLoopTimes_, fiaInfo.n2Size * qkHeadDim);
        }
    } else if (fiaInfo.qLayout == FiaLayout::BNSD) {
        if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION &&
            fiaInfo.kvLayout == FiaLayout::BnBsH) { // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(gsSize_, fiaInfo.s2Size, qkHeadDim, fiaInfo.n2Size * qkHeadDim);
        } else {
            bmm1.SetOrgShape(gsSize_, fiaInfo.s2Size, qkHeadDim);
        }
    }
    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(platformInfo_.l1Size, platformInfo_.l0cSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Bmm1 set buffer space failed"),
                return ge::GRAPH_FAILED);
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT && fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        ret = bmm1.SetFixSplit(sOuterFactor_, SINNER_128);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor_, sInnerFactor_);
    }
    TCubeTiling &bmm1TilingData = pfaTilingData_.bmm1TilingDataRect;

    ret = bmm1.GetTiling(bmm1TilingData);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Bmm1 get tiling failed, ret:%d", ret),
                return ge::GRAPH_FAILED);
    uint32_t baseN = std::min(SINNER_128, sInnerFactor_);
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        baseN = SINNER_128;
    }
    if (ret != ge::GRAPH_SUCCESS) {
        ret = bmm1.SetFixSplit(sOuterFactor_, baseN, 64U); // check
        ret = bmm1.GetTiling(bmm1TilingData);
    }
    // check ret
    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(platformInfo_.l1Size);
    bmm1TilingData.set_shareL0CSize(platformInfo_.l0cSize);
    bmm1TilingData.set_shareUbSize(0);

    if ((bmm1TilingData.get_depthA1() == 1) && (bmm1TilingData.get_depthB1() == 1)) {
        bmm1TilingData.set_depthA1(DOUBLE_BUFFER_NUM);
        bmm1TilingData.set_depthB1(DOUBLE_BUFFER_NUM);
    }

    OP_CHECK_IF(!EnableMTE2BmmPipe(fiaInfo, bmm1, bmm1TilingData), OP_LOGE(fiaInfo.opName, "Bmm1 mte2 bmm pipe fail"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetMM2TilingData(gert::TilingContext *context,
                                                                     const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_INT8;
    matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_INT32;
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);

    int32_t ret = bmm2.SetShape(sOuterFactor_, fiaInfo.vHeadDim, sInnerFactor_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Bmm2 set shape failed"), return ge::GRAPH_FAILED);

    if ((fiaInfo.qLayout == FiaLayout::BSH) || (fiaInfo.qLayout == FiaLayout::BSND) ||
        (fiaInfo.qLayout == FiaLayout::TND)) {
        if (fiaInfo.qLayout == FiaLayout::TND && fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION &&
            fiaInfo.kvLayout == FiaLayout::BnNBsD) {
            bmm2.SetOrgShape(gsSize_, fiaInfo.vHeadDim, fiaInfo.s2Size);
        } else {
            bmm2.SetOrgShape(gsSize_, fiaInfo.vHeadDim * fiaInfo.n2Size, fiaInfo.s2Size);
        }
    } else if (fiaInfo.qLayout == FiaLayout::BNSD) {
        if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION && fiaInfo.kvLayout == FiaLayout::BnBsH) {
            bmm2.SetOrgShape(gsSize_, fiaInfo.vHeadDim * fiaInfo.n2Size, fiaInfo.s2Size);
        } else {
            bmm2.SetOrgShape(gsSize_, fiaInfo.qkHeadDim, fiaInfo.s2Size);
        }
    }
    bmm2.SetBias(false);
    bmm2.SetBufferSpace(platformInfo_.l1Size, platformInfo_.l0cSize);
    ret = bmm2.GetTiling(pfaTilingData_.bmm2TilingDataRect);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Bmm2 get tiling failed"), return ge::GRAPH_FAILED);

    pfaTilingData_.bmm2TilingDataRect.set_shareMode(0);
    pfaTilingData_.bmm2TilingDataRect.set_shareL1Size(platformInfo_.l1Size);
    pfaTilingData_.bmm2TilingDataRect.set_shareL0CSize(platformInfo_.l0cSize);
    pfaTilingData_.bmm2TilingDataRect.set_shareUbSize(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetFullQuantTilingData(const FiaTilingInfo &fiaInfo)
{
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    baseParams.set_batchSize(fiaInfo.bSize);
    baseParams.set_headNumSize(fiaInfo.n1Size);
    baseParams.set_seqSize(gsSize_);
    baseParams.set_headSize(fiaInfo.qkHeadDim);
    baseParams.set_scaleValue(fiaInfo.scaleValue);
    baseParams.set_preTokens(fiaInfo.preToken);
    baseParams.set_nextTokens(fiaInfo.nextToken);
    baseParams.set_blockSize(fiaInfo.blockSize);
    uint32_t blockTableDim2 = 0;
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        blockTableDim2 = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    baseParams.set_blockTableDim2(blockTableDim2);
    baseParams.set_PABlockNumSum(fiaInfo.totalBlockNum);
    baseParams.set_dimNumOfseq(fiaInfo.bSize);
    baseParams.set_seqInnerSize(fiaInfo.s2Size);
    baseParams.set_prefixSeqInnerSize(fiaInfo.systemPrefixLen);
    baseParams.set_usePseShift(fiaInfo.pseShiftFlag);
    baseParams.set_useMask(fiaInfo.attenMaskFlag);
    baseParams.set_headNumRatio(gsMergeFlag_ ? 1 : fiaInfo.gSize);
    baseParams.set_typeByteNum(BYTE_BLOCK);
    baseParams.set_attenMaskElemType(fiaInfo.opParamInfo.attenMask.desc != nullptr ?
                                         fiaInfo.opParamInfo.attenMask.desc->GetDataType() :
                                         fiaInfo.inputQType);
    baseParams.set_pseShiftTypeByteNum(BYTE_BLOCK / FLOAT16SIZE);
    baseParams.set_pseMaskMaxSize(0);
    baseParams.set_maskTypeByteNum(BYTE_BLOCK);
    baseParams.set_outputTypeByteNum(fiaInfo.isOutQuantEnable ? BYTE_BLOCK : BYTE_BLOCK / (FLOAT16SIZE / INT8SIZE));
    baseParams.set_softmaxTypeByteNum(BYTE_BLOCK / sizeof(float));
    baseParams.set_sparseMode(fiaInfo.sparseMode);
    baseParams.set_alignedHeadSize(AlignUp(fiaInfo.qkHeadDim, BYTE_BLOCK));
    baseParams.set_splitS2(0);
    baseParams.set_splitD(0);
    baseParams.set_pseShiftS1Size(fiaInfo.pseShiftS1);
    baseParams.set_pseShiftS2Size(fiaInfo.pseShiftS2);
    baseParams.set_isLayoutSH(0);
    baseParams.set_isActualSeqLengthsNull(!actualSeqLenQFlag_);
    baseParams.set_isActualSeqLengthsKVNull(!actualSeqLenKVFlag_);
    baseParams.set_actualSeqLengthsSize(fiaInfo.actualLenQDims);
    baseParams.set_actualSeqLengthsKVSize(fiaInfo.actualLenDims);
    baseParams.set_deqScaleFlag(1);  // 当前仅per-tensor全量化会使用
    baseParams.set_deqScale2Flag(1); // 当前仅per-tensor全量化会使用
    baseParams.set_isAntiPerchannel(0);
    baseParams.set_isRowInvalid((fiaInfo.innerPrecise >> 1) & 1);
    baseParams.set_softmaxOuterSize(sOuterFactor_);
    baseParams.set_isQuant2Perchannel(fiaInfo.isOutQuantPerChnOut);
    baseParams.set_isQuant2BF16(fiaInfo.isOutQuantTypeBf16);
    baseParams.set_isKvContinuous(fiaInfo.kvStorageMode != KvStorageMode::TENSOR_LIST);
    baseParams.set_fromFused(!fromPFA_);
    baseParams.set_isBSNDOut(fiaInfo.qLayout == FiaLayout::BNSD && fiaInfo.outLayout == FiaLayout::BSND);
    baseParams.set_isIFA(gsMergeFlag_);
    baseParams.set_isSoftMaxLseEnable(fiaInfo.softmaxLseFlag);
    baseParams.set_isActualSharedPrefixLenNull(!actualSharedPrefixLenFlag_);
    baseParams.set_isQHasLeftPadding(fiaInfo.qPaddingSizeFlag);
    baseParams.set_isKVHasLeftPadding(fiaInfo.kvPaddingSizeFlag);
    baseParams.set_keyAntiquantMode(fiaInfo.keyAntiquantMode);
    baseParams.set_valueAntiquantMode(fiaInfo.valueAntiquantMode);
    baseParams.set_hasKeyAntiquantOffset(0);
    baseParams.set_isMsd(0);
    baseParams.set_isQuant2FP16(0);
    baseParams.set_ropeHeadSize(fiaInfo.ropeHeadDim);
    baseParams.set_qkHeadSize(fiaInfo.qkHeadDim);
    baseParams.set_vHeadSize(fiaInfo.vHeadDim);
    baseParams.set_gOfMla(gsMergeFlag_ ? fiaInfo.gSize : 1);

    auto &initOutputParams = pfaTilingData_.promptAttentionInitOutputParams;
    int64_t outSize = fiaInfo.opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = fiaInfo.softmaxLseFlag ? fiaInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo.isOutQuantEnable) {
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    }
    initOutputParams.set_singleCoreSize(singleCoreSize);
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(fiaInfo.needInit || needInit_);
    initOutputParams.set_isOneN(0); // 默认值,当前未使用

    auto &singleCoreParams = pfaTilingData_.promptAttentionSingleCoreParams;
    singleCoreParams.set_singleProcessSInnerSize(sInnerFactor_);
    singleCoreParams.set_singleProcessSOuterSize(sOuterFactor_);
    singleCoreParams.set_pseShiftBatch(fiaInfo.pseShiftByBatch);
    singleCoreParams.set_kvAntiquantSInnerSize(0);
    return ge::GRAPH_SUCCESS;
}

bool FusedInferAttentionScoreTilingImpl::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType)
{
    static struct {
        ge::DataType a;
        matmul_tiling::DataType b;
    } typeTrans[] = {{ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
                     {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
                     {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
                     {ge::DT_INT4, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_INT8}};

    for (uint32_t i = 0; i < sizeof(typeTrans) / sizeof(typeTrans[0]); i++) {
        if (typeTrans[i].a == getype) {
            *mmType = typeTrans[i].b;
            return true;
        }
    }
    return false;
}

void FusedInferAttentionScoreTilingImpl::AdjustPABmm1Tiling(const FiaTilingInfo &fiaInfo, uint32_t &bmm1BaseN)
{
    if (bmm1BaseN < static_cast<uint32_t>(fiaInfo.blockSize)) {
        while (static_cast<uint32_t>(fiaInfo.blockSize) % bmm1BaseN != 0) {
            bmm1BaseN /=
                2; // 2:不断减半，确保1个base块不会跨block拷贝。已校验过blockSize 16/32对齐，因此bmm1BaseN最小值为16/32
        }
    } else if (bmm1BaseN > static_cast<uint32_t>(fiaInfo.blockSize)) {
        // nd2nz拷贝时ndnum>1场景性能较差，通过设置baseN <= blocksize避免
        uint32_t tmpBaseN = increGcd(bmm1BaseN, static_cast<uint32_t>(fiaInfo.blockSize));
        bmm1BaseN = tmpBaseN;
    }
    OP_LOGD(fiaInfo.opName, "Page attention is enabled, blockSize is %d, bmm1 baseN is adjusted to %u.",
            fiaInfo.blockSize, bmm1BaseN);
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::ComputeTilingData(const FiaTilingInfo &fiaInfo)
{
    // 处理不能直接从fiaInfo赋值到tiling data
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    auto &singleCoreParams = pfaTilingData_.promptAttentionSingleCoreParams;

    /*
     *  mask,sparse mode 相关tiling data
     */
    if (fiaInfo.attenMaskFlag) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = NUM_2048;
        uint64_t maskS2Size = NUM_2048;
        if (maskDimNum != 2 || fiaInfo.s1Size == 1) {
            maskBatch = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape().GetDim(0);
        }
        if (fiaInfo.antiQuantFlag && fiaInfo.s1Size == 1) {
            inputParams.set_attenMaskShapeType(1);
        } else {
            inputParams.set_attenMaskShapeType(maskBatch > 1 ? 1 : 2);
        }
        singleCoreParams.set_attenMaskBatch(maskBatch);
        maskS2Size = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        inputParams.set_attenMaskS1Size(maskS1Size);
        inputParams.set_attenMaskS2Size(maskS2Size);
        baseParams.set_maskQsSize(maskS1Size);
        baseParams.set_maskKVsSize(maskS2Size);
    } else {
        inputParams.set_attenMaskS1Size(0);
        inputParams.set_attenMaskS2Size(0);
        inputParams.set_attenMaskShapeType(0);
        baseParams.set_maskQsSize(0);
        baseParams.set_maskKVsSize(0);
    }

    static std::map<uint32_t, uint8_t> sparseToCompressModeMap = {{SPARSE_MODE_NO_MASK, 0},
                                                                  {SPARSE_MODE_ALL_MASK, 0},
                                                                  {SPARSE_MODE_LEFT_UP, 1},
                                                                  {SPARSE_MODE_RIGHT_DOWN, 2},
                                                                  {SPARSE_MODE_BAND, 3}};
    auto itr = sparseToCompressModeMap.find(fiaInfo.sparseMode);
    if (itr == sparseToCompressModeMap.end()) {
        inputParams.set_attenMaskCompressMode(0);
    } else {
        inputParams.set_attenMaskCompressMode(itr->second);
    }

    if ((fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT && !isPFAFlag_) ||
        fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
        uint8_t sparseType = 0;
        inputParams.set_sparseType(sparseType);
    } else {
        uint8_t sparseType = 0;
        if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK) {
            if (fiaInfo.preToken >= fiaInfo.s1Size && fiaInfo.nextToken == 0) {
                sparseType = 3;
            } else if (fiaInfo.preToken >= fiaInfo.s1Size && fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION &&
                       fiaInfo.nextToken >= fiaInfo.s2Size) {
                sparseType = 0;
            } else {
                sparseType = 4;
            }
        } else if (fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) {
            sparseType = 0;
        } else if (fiaInfo.sparseMode == SPARSE_MODE_LEFT_UP) {
            sparseType = 3;
        } else if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION && fiaInfo.s1Size == fiaInfo.s2Size) {
                sparseType = 3;
            } else {
                sparseType = 4;
            }
        } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
            sparseType = 4;
        }
        inputParams.set_sparseType(sparseType);
    }

    /*
     *  alibi 相关tiling data
     */
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    auto qStartIdxTensor = fiaInfo.opParamInfo.qStartIdx.tensor;
    auto kvStartIdxTensor = fiaInfo.opParamInfo.kvStartIdx.tensor;
    ;
    if (qStartIdxTensor != nullptr && qStartIdxTensor->GetShapeSize() >= 1) {
        const int64_t *value = qStartIdxTensor->GetData<int64_t>();
        if (value != nullptr) {
            qStartIdx = value[0];
        }
    }

    if (kvStartIdxTensor != nullptr && kvStartIdxTensor->GetShapeSize() >= 1) {
        const int64_t *value = kvStartIdxTensor->GetData<int64_t>();
        if (value != nullptr) {
            kvStartIdx = value[0];
        }
    }
    inputParams.set_qStartIdx(qStartIdx);
    inputParams.set_kvStartIdx(kvStartIdx);

    /*
     *  layout 相关tiling data
     */
    static std::map<FiaLayout, uint8_t> layoutStrToLayoutTypeMap = {
        {FiaLayout::BSH, 1}, {FiaLayout::TND, 4}, {FiaLayout::BSND, 1}, {FiaLayout::BNSD, 3}, {FiaLayout::NTD, 5},
    };
    auto iter = layoutStrToLayoutTypeMap.find(fiaInfo.qLayout);
    if (iter == layoutStrToLayoutTypeMap.end()) {
        inputParams.set_layoutType(0);
        baseParams.set_layoutType(0);
    } else {
        inputParams.set_layoutType(iter->second);
        baseParams.set_layoutType(iter->second);
    }

    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        if (fiaInfo.antiQuantFlag) {
            uint32_t keyCacheDimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
            if (keyCacheDimNum == 3) { // 3: BBH
                inputParams.set_paLayoutType(0);
                baseParams.set_PAlayoutType(0);
            } else if (keyCacheDimNum == 4) { // 4: BNBD
                inputParams.set_paLayoutType(1);
                baseParams.set_PAlayoutType(1);
            } else if (keyCacheDimNum == 5) { // 5: PA NZ
                inputParams.set_paLayoutType(2);
                baseParams.set_PAlayoutType(2);
            }
        } else {
            uint32_t keyCacheDimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
            if (keyCacheDimNum == 3) { // 3: BBH
                inputParams.set_paLayoutType(1);
                baseParams.set_PAlayoutType(1);
            } else if (keyCacheDimNum == 4) { // 4: BNBD
                inputParams.set_paLayoutType(0);
                baseParams.set_PAlayoutType(0);
            } else if (keyCacheDimNum == 5) { // 5: PA NZ
                inputParams.set_paLayoutType(2);
                baseParams.set_PAlayoutType(2);
            }
        }
    }

    const std::map<std::string, uint32_t> transposeLayoutMp = {{"BNSD_BSND", 1}, {"BSND_BNSD", 2}, {"BSH_BNSD", 3},
                                                               {"BNSD_NBSD", 4}, {"BSND_NBSD", 5}, {"BSH_NBSD", 6},
                                                               {"NTD_TND", 7},   {"TND_NTD", 8}};
    string layout(fiaInfo.opParamInfo.layOut);
    if (transposeLayoutMp.find(layout) != transposeLayoutMp.end()) {
        inputParams.set_transposeLayout(transposeLayoutMp.at(layout));
        baseParams.set_transposeLayout(transposeLayoutMp.at(layout));
    } else {
        inputParams.set_transposeLayout(0);
        baseParams.set_transposeLayout(0);
    }
    /*
     *  伪量化 相关tiling data
     */
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        auto scaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
        uint32_t expectShape = 1;
        uint32_t scaleDimNum = scaleTensor->GetStorageShape().GetDimNum();
        uint32_t scaleShape = scaleTensor->GetStorageShape().GetDim(0);
        if (scaleDimNum == 1 && scaleShape == expectShape) {
            inputParams.set_antiquantPerTensorFlag(1);
        } else {
            inputParams.set_antiquantPerTensorFlag(0);
        }
    } else {
        inputParams.set_antiquantPerTensorFlag(0);
    }

    // needInit
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    for (uint32_t i = 0; i < fiaInfo.bSize && fiaInfo.quantMode != FiaQuantMode::ANTI_QUANT; i++) {
        if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            preTokensPerbatch = SPARSE_MODE_INT_MAX;
            if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
                nextTokensPerbatch =
                    actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen - actualSeqLengthsQ_[i] / fiaInfo.gSize;
            } else {
                nextTokensPerbatch = actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen - actualSeqLengthsQ_[i];
            }
        } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
            preTokensPerbatch =
                fiaInfo.preToken - actualSeqLengthsKV_[i] - fiaInfo.systemPrefixLen + actualSeqLengthsQ_[i];
            nextTokensPerbatch =
                fiaInfo.nextToken + actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen - actualSeqLengthsQ_[i];
        } else {
            preTokensPerbatch = fiaInfo.preToken;
            nextTokensPerbatch = fiaInfo.nextToken;
        }
        if ((nextTokensPerbatch < 0) ||
            (actualSeqLengthsQ_[i] > (actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen + preTokensPerbatch))) {
            needInit_ = true;
        }
        OP_LOGI(fiaInfo.opName, "preTokensPerbatch[%u] is %ld, nextTokensPerbatch[%u] is %ld", i, preTokensPerbatch, i,
                nextTokensPerbatch);
        OP_LOGI(
            fiaInfo.opName,
            "actualSeqLengths[%u] is %ld, actualSeqLengthsKV[%u] is %ld, actualSharedPrefixLen is %ld, needInit is %u",
            i, actualSeqLengthsQ_[i], i, actualSeqLengthsKV_[i], fiaInfo.systemPrefixLen, needInit_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetFATilingData(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    inputParams.set_bSize(fiaInfo.bSize);
    inputParams.set_t1Size(fiaInfo.qTSize);
    inputParams.set_t2Size(fiaInfo.kTSize);
    inputParams.set_n2Size(fiaInfo.n2Size);
    inputParams.set_gSize(fiaInfo.gSize);
    inputParams.set_s1Size(fiaInfo.s1Size);
    inputParams.set_s2Size(fiaInfo.s2Size);
    inputParams.set_dSize(fiaInfo.qkHeadDim);
    inputParams.set_dSizeV(fiaInfo.vHeadDim);
    inputParams.set_dSizeRope(64);
    inputParams.set_scaleValue(fiaInfo.scaleValue);
    inputParams.set_preTokens(fiaInfo.preToken);
    inputParams.set_nextTokens(fiaInfo.nextToken);
    inputParams.set_pseShapeType(fiaInfo.enableAlibiPse ? 3 : 0);
    inputParams.set_pseS1Size(fiaInfo.pseShiftS1);
    inputParams.set_pseS2Size(fiaInfo.pseShiftS2);
    inputParams.set_pseBSize(fiaInfo.pseShiftByBatch);
    inputParams.set_implMode(0);
    inputParams.set_needDropMaskOp(0);
    inputParams.set_dropMaskOuter(0);
    inputParams.set_pseEncodeType(0);
    inputParams.set_remain(0);
    inputParams.set_rsv1(0);
    inputParams.set_seed(0);
    inputParams.set_offset(0);
    inputParams.set_keepProbUint8(0);
    inputParams.set_pseType(*fiaInfo.opParamInfo.pseType);
    inputParams.set_deqScaleFlag(0);  // per-tensor全量化场景才会用到
    inputParams.set_deqScale2Flag(0); // per-tensor全量化场景才会用到
    inputParams.set_actualSeqLengthsSize(fiaInfo.actualLenQDims);
    inputParams.set_actualSeqLengthsKVSize(fiaInfo.actualLenDims);
    inputParams.set_isKvContinuous(fiaInfo.kvStorageMode != KvStorageMode::TENSOR_LIST);
    inputParams.set_fromFused(!fromPFA_);
    inputParams.set_isBSNDOut(fiaInfo.qLayout == FiaLayout::BNSD && fiaInfo.outLayout == FiaLayout::BSND);
    if (fiaInfo.antiQuantFlag) {
        inputParams.set_isGqa(gsMergeFlag_);
    } else {
        inputParams.set_isGqa(gsMergeFlag_ && fiaInfo.ropeMode != RopeMode::ROPE_SPLIT);
    }
    inputParams.set_isSoftMaxLseEnable(fiaInfo.softmaxLseFlag);
    inputParams.set_isQHasLeftPadding(fiaInfo.qPaddingSizeFlag);
    inputParams.set_isKVHasLeftPadding(fiaInfo.kvPaddingSizeFlag);
    inputParams.set_ropeHeadSize(fiaInfo.ropeHeadDim);
    inputParams.set_prefixSeqInnerSize(fiaInfo.systemPrefixMaxLen);
    inputParams.set_headNumRatio(gsMergeFlag_ ? 1 : fiaInfo.gSize);
    inputParams.set_blockSize(fiaInfo.blockSize);
    uint32_t blockTableDim2 = 0;
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        blockTableDim2 = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    inputParams.set_blockTableDim2(blockTableDim2);
    inputParams.set_paBlockNumSum(fiaInfo.totalBlockNum);
    inputParams.set_isRowInvalid((fiaInfo.innerPrecise >> 1) & 1);
    inputParams.set_isPostQuantPerChnl(fiaInfo.isOutQuantPerChnOut);
    inputParams.set_isPostQuantBF16(fiaInfo.isOutQuantTypeBf16);
    inputParams.set_antiquantParaSeqSize(fiaInfo.antiqSeqSize);
    inputParams.set_antiquantPerHeadFlag(fiaInfo.keyAntiquantMode == 2 || fiaInfo.keyAntiquantMode == 3 ||
                                         fiaInfo.keyAntiquantMode == 5);
    inputParams.set_isActualSeqLengthsNull(!actualSeqLenQFlag_);
    inputParams.set_isActualSeqLengthsKVNull(!actualSeqLenKVFlag_);
    inputParams.set_isActualSharedPrefixLenNull(!actualSharedPrefixLenFlag_);
    inputParams.set_keepProb(0);          // 默认值, 未使用
    inputParams.set_bandIndex(0);         // 默认值，未使用
    inputParams.set_attenMaskDataType(1); // 默认值，未使用
    inputParams.set_alignedS2(0);         // 默认值, 未使用
    inputParams.set_s1SparseValidSize(0); // 默认值，未使用
    inputParams.set_s2SparseValidSize(0); // 默认值，未使用
    inputParams.set_pseAlibiBaseS1(0);    // 默认值，未使用
    inputParams.set_pseAlibiBaseS2(0);    // 默认值，未使用

    auto &initOutputParams = faRunTilingAdapter_.initOutputParams;
    int64_t outSize = fiaInfo.opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = fiaInfo.softmaxLseFlag ? fiaInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo.isOutQuantEnable) {
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    }
    if (fiaInfo.antiQuantFlag && !(fiaInfo.needInit || needInit_)) {
        initOutputParams.set_singleCoreSize(0);
    } else {
        initOutputParams.set_singleCoreSize(singleCoreSize);
    }
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(fiaInfo.needInit || needInit_);
    initOutputParams.set_isOneN(0); // 默认值,当前未使用

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetTilingData(gert::TilingContext *context,
                                                                  const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
        SetFullQuantTilingData(fiaInfo);
    } else if (!fiaInfo.emptyTensorFlag) {
        if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
            OP_LOGE(fiaInfo.opName, "This demo only supports FullQuant!");
        }
        SetFATilingData(fiaInfo);
    }

    int64_t cap = context->GetRawTilingData()->GetCapacity();
    OP_LOGI(fiaInfo.opName, "Tiling Data context GetCapacity: %lu.", cap);
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT) {
        PFAFullQuantTilingData *tiling = context_->GetTilingData<PFAFullQuantTilingData>();
        OP_CHECK_IF(tiling == nullptr, OP_LOGE(fiaInfo.opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
        tiling->MigrateFromLegacyFormat(pfaTilingData_);
    } else {
        PrintAllTilingData(fiaInfo);
        FlashAttentionScoreSimplifiedTilingData *tiling =
            context->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
        OP_CHECK_IF(tiling == nullptr, OP_LOGE(fiaInfo.opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
        *tiling = faRunTilingAdapter_;
    }
    return ge::GRAPH_SUCCESS;
}

void FusedInferAttentionScoreTilingImpl::PrintAllTilingData(const FiaTilingInfo &fiaInfo)
{
    OP_LOGD(fiaInfo.opName, "bSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_bSize());
    OP_LOGD(fiaInfo.opName, "t1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_t1Size());
    OP_LOGD(fiaInfo.opName, "t2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_t2Size());
    OP_LOGD(fiaInfo.opName, "n2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_n2Size());
    OP_LOGD(fiaInfo.opName, "gSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_gSize());
    OP_LOGD(fiaInfo.opName, "s1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_s1Size());
    OP_LOGD(fiaInfo.opName, "s2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_s2Size());
    OP_LOGD(fiaInfo.opName, "alignedS2:%d", faRunTilingAdapter_.inputParamsRegbase.get_alignedS2());
    OP_LOGD(fiaInfo.opName, "dSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSize());
    OP_LOGD(fiaInfo.opName, "dSizeV:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSizeV());
    OP_LOGD(fiaInfo.opName, "dSizeRope:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSizeRope());
    OP_LOGD(fiaInfo.opName, "keepProb:%d", faRunTilingAdapter_.inputParamsRegbase.get_keepProb());
    OP_LOGD(fiaInfo.opName, "scaleValue:%d", faRunTilingAdapter_.inputParamsRegbase.get_scaleValue());
    OP_LOGD(fiaInfo.opName, "preTokens:%d", faRunTilingAdapter_.inputParamsRegbase.get_preTokens());
    OP_LOGD(fiaInfo.opName, "nextTokens:%d", faRunTilingAdapter_.inputParamsRegbase.get_nextTokens());
    OP_LOGD(fiaInfo.opName, "pseS1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseS1Size());
    OP_LOGD(fiaInfo.opName, "pseS2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseS2Size());
    OP_LOGD(fiaInfo.opName, "pseBSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseBSize());
    OP_LOGD(fiaInfo.opName, "bandIndex:%d", faRunTilingAdapter_.inputParamsRegbase.get_bandIndex());
    OP_LOGD(fiaInfo.opName, "layoutType:%d", faRunTilingAdapter_.inputParamsRegbase.get_layoutType());
    OP_LOGD(fiaInfo.opName, "pseShapeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseShapeType());
    OP_LOGD(fiaInfo.opName, "attenMaskShapeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskShapeType());
    OP_LOGD(fiaInfo.opName, "attenMaskDataType:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskDataType());
    OP_LOGD(fiaInfo.opName, "attenMaskCompressMode:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_attenMaskCompressMode());
    OP_LOGD(fiaInfo.opName, "implMode:%d", faRunTilingAdapter_.inputParamsRegbase.get_implMode());
    OP_LOGD(fiaInfo.opName, "sparseType:%d", faRunTilingAdapter_.inputParamsRegbase.get_sparseType());
    OP_LOGD(fiaInfo.opName, "needDropMaskOp:%d", faRunTilingAdapter_.inputParamsRegbase.get_needDropMaskOp());
    OP_LOGD(fiaInfo.opName, "dropMaskOuter:%d", faRunTilingAdapter_.inputParamsRegbase.get_dropMaskOuter());
    OP_LOGD(fiaInfo.opName, "pseEncodeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseEncodeType());
    OP_LOGD(fiaInfo.opName, "remain:%d", faRunTilingAdapter_.inputParamsRegbase.get_remain());
    OP_LOGD(fiaInfo.opName, "attenMaskS2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskS2Size());
    OP_LOGD(fiaInfo.opName, "pseType:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseType());
    OP_LOGD(fiaInfo.opName, "rsv1:%d", faRunTilingAdapter_.inputParamsRegbase.get_rsv1());
    OP_LOGD(fiaInfo.opName, "qStartIdx:%d", faRunTilingAdapter_.inputParamsRegbase.get_qStartIdx());
    OP_LOGD(fiaInfo.opName, "kvStartIdx:%d", faRunTilingAdapter_.inputParamsRegbase.get_kvStartIdx());
    OP_LOGD(fiaInfo.opName, "s1SparseValidSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_s1SparseValidSize());
    OP_LOGD(fiaInfo.opName, "s2SparseValidSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_s2SparseValidSize());
    OP_LOGD(fiaInfo.opName, "seed:%d", faRunTilingAdapter_.inputParamsRegbase.get_seed());
    OP_LOGD(fiaInfo.opName, "offset:%d", faRunTilingAdapter_.inputParamsRegbase.get_offset());
    OP_LOGD(fiaInfo.opName, "keepProbUint8:%d", faRunTilingAdapter_.inputParamsRegbase.get_keepProbUint8());
    OP_LOGD(fiaInfo.opName, "pseAlibiBaseS1:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseAlibiBaseS1());
    OP_LOGD(fiaInfo.opName, "pseAlibiBaseS2:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseAlibiBaseS2());
    OP_LOGD(fiaInfo.opName, "deqScaleFlag:%d", faRunTilingAdapter_.inputParamsRegbase.get_deqScaleFlag());
    OP_LOGD(fiaInfo.opName, "deqScale2Flag:%d", faRunTilingAdapter_.inputParamsRegbase.get_deqScale2Flag());
    OP_LOGD(fiaInfo.opName, "isActualSeqLengthsNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSeqLengthsNull());
    OP_LOGD(fiaInfo.opName, "isActualSeqLengthsKVNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSeqLengthsKVNull());
    OP_LOGD(fiaInfo.opName, "actualSeqLengthsSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_actualSeqLengthsSize());
    OP_LOGD(fiaInfo.opName, "actualSeqLengthsKVSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_actualSeqLengthsKVSize());
    OP_LOGD(fiaInfo.opName, "isKvContinuous:%d", faRunTilingAdapter_.inputParamsRegbase.get_isKvContinuous());
    OP_LOGD(fiaInfo.opName, "fromFused:%d", faRunTilingAdapter_.inputParamsRegbase.get_fromFused());
    OP_LOGD(fiaInfo.opName, "isBSNDOut:%d", faRunTilingAdapter_.inputParamsRegbase.get_isBSNDOut());
    OP_LOGD(fiaInfo.opName, "transposeLayout:%d", faRunTilingAdapter_.inputParamsRegbase.get_transposeLayout());
    OP_LOGD(fiaInfo.opName, "isGqa:%d", faRunTilingAdapter_.inputParamsRegbase.get_isGqa());
    OP_LOGD(fiaInfo.opName, "isSoftMaxLseEnable:%d", faRunTilingAdapter_.inputParamsRegbase.get_isSoftMaxLseEnable());
    OP_LOGD(fiaInfo.opName, "isActualSharedPrefixLenNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSharedPrefixLenNull());
    OP_LOGD(fiaInfo.opName, "isQHasLeftPadding:%d", faRunTilingAdapter_.inputParamsRegbase.get_isQHasLeftPadding());
    OP_LOGD(fiaInfo.opName, "isKVHasLeftPadding:%d", faRunTilingAdapter_.inputParamsRegbase.get_isKVHasLeftPadding());
    OP_LOGD(fiaInfo.opName, "prefixSeqInnerSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_prefixSeqInnerSize());
    OP_LOGD(fiaInfo.opName, "headNumRatio:%d", faRunTilingAdapter_.inputParamsRegbase.get_headNumRatio());
    OP_LOGD(fiaInfo.opName, "blockSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_blockSize());
    OP_LOGD(fiaInfo.opName, "blockTableDim2:%d", faRunTilingAdapter_.inputParamsRegbase.get_blockTableDim2());
    OP_LOGD(fiaInfo.opName, "paBlockNumSum:%d", faRunTilingAdapter_.inputParamsRegbase.get_paBlockNumSum());
    OP_LOGD(fiaInfo.opName, "paLayoutType:%d", faRunTilingAdapter_.inputParamsRegbase.get_paLayoutType());
    OP_LOGD(fiaInfo.opName, "attenMaskS1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskS1Size());
    OP_LOGD(fiaInfo.opName, "isRowInvalid:%d", faRunTilingAdapter_.inputParamsRegbase.get_isRowInvalid());
    OP_LOGD(fiaInfo.opName, "kvSplitPart:%d", faRunTilingAdapter_.inputParamsRegbase.get_kvSplitPart());
    OP_LOGD(fiaInfo.opName, "accumOutSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_accumOutSize());
    OP_LOGD(fiaInfo.opName, "logSumExpSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_logSumExpSize());
    OP_LOGD(fiaInfo.opName, "isPostQuantPerChnl:%d", faRunTilingAdapter_.inputParamsRegbase.get_isPostQuantPerChnl());
    OP_LOGD(fiaInfo.opName, "isPostQuantBF16:%d", faRunTilingAdapter_.inputParamsRegbase.get_isPostQuantBF16());
    OP_LOGD(fiaInfo.opName, "antiquantPerTensorFlag:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_antiquantPerTensorFlag());
    OP_LOGD(fiaInfo.opName, "antiquantPerHeadFlag:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_antiquantPerHeadFlag());
    OP_LOGD(fiaInfo.opName, "antiquantParaSeqSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_antiquantParaSeqSize());

    OP_LOGD(fiaInfo.opName, "coreNum:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_coreNum());
    OP_LOGD(fiaInfo.opName, "totalSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_totalSize());
    OP_LOGD(fiaInfo.opName, "s1OuterSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_s1OuterSize());
    OP_LOGD(fiaInfo.opName, "splitFactorSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize());
    OP_LOGD(fiaInfo.opName, "splitFactorTailSize:%d",
            faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorTailSize());
    for (uint32_t i = 0; i < 48; i++) {
        OP_LOGD(fiaInfo.opName, "bnStartIdx[%d]:%d", i,
                faRunTilingAdapter_.multiCoreParamsRegbase.get_bnStartIdxPtr()[i]);
    }
    for (uint32_t i = 0; i < 48; i++) {
        OP_LOGD(fiaInfo.opName, "sparseStartIdx[%d]:%d", i,
                faRunTilingAdapter_.multiCoreParamsRegbase.get_sparseStartIdxPtr()[i]);
    }
    OP_LOGD(fiaInfo.opName, "firstFullLoadS1OuterIdx:%d",
            faRunTilingAdapter_.multiCoreParamsRegbase.get_firstFullLoadS1OuterIdx());
    OP_LOGD(fiaInfo.opName, "splitCoreMode:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_splitCoreMode());

    OP_LOGD(fiaInfo.opName, "singleCoreSize:%d", faRunTilingAdapter_.initOutputParams.get_singleCoreSize());
    OP_LOGD(fiaInfo.opName, "needInit:%d", faRunTilingAdapter_.initOutputParams.get_needInit());
    OP_LOGD(fiaInfo.opName, "isOneN:%d", faRunTilingAdapter_.initOutputParams.get_isOneN());
    OP_LOGD(fiaInfo.opName, "totalOutputSize:%d", faRunTilingAdapter_.initOutputParams.get_totalOutputSize());
    OP_LOGD(fiaInfo.opName, "totalSoftMaxLseOutputSize:%d",
            faRunTilingAdapter_.initOutputParams.get_totalSoftMaxLseOutputSize());
}
ge::graphStatus FusedInferAttentionScoreTilingImpl::DoOpTiling(gert::TilingContext *context,
                                                               const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(SetPlatMemoryInfo(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Set plat memory info fail."), return ge::GRAPH_FAILED);

    if (fiaInfo.emptyTensorFlag) {
        OP_CHECK_IF(SetEmptyTensor(context, fiaInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Set empty tensor fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    InitImplParam(fiaInfo);

    OP_CHECK_IF(SplitPolicy(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Execute split policy fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeTilingData(fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Compute tilingData fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GenTilingKey(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Generate tilingKey fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetBlockDim(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Set blockDim fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspace(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Get workspace fail."),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    OP_CHECK_IF(SetTilingData(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Set tiling data fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
