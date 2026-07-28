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
 * \file quant_block_sparse_attn_tiling.cpp
 * \brief QuantBlockSparseAttn tiling: three-stage pipeline (Parse -> Check -> Tiling).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <tiling/platform/platform_ascendc.h>
#include "../op_kernel/quant_block_sparse_attn_template_tiling_key.h"
#include "quant_block_sparse_attn_tiling.h"
#include "quant_block_sparse_attn_check.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 2U;
constexpr uint32_t BSA_LAYOUT_Q_NTD_VALUE = 5U;
constexpr uint32_t BSA_LAYOUT_KV_PA_BNSD_VALUE = 1U;
constexpr uint32_t BSA_LAYOUT_SPARSE_B_N_QB_KB_VALUE = 0U;
constexpr uint8_t BSA_LAYOUT_TYPE_TND = 2U;
constexpr uint8_t BSA_LAYOUT_TYPE_NTD = 5U;
constexpr uint8_t BSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t BSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t BSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL = 2U;
constexpr uint32_t BSA_COMBINE_ALIGN_BYTES = 32U;
constexpr uint32_t BSA_K_SCALE_BYTES = sizeof(float);

uint32_t GetAicCoreNum(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return BSA_MAX_CORE_NUM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    return aicNum == 0U ? BSA_MAX_CORE_NUM : std::min<uint32_t>(aicNum, BSA_MAX_CORE_NUM);
}
} // namespace

QuantBlockSparseAttnTiling::QuantBlockSparseAttnTiling(gert::TilingContext *context) : context_(context) {}

void QuantBlockSparseAttnTiling::FillPaParams()
{
    const auto &info = *tilingInfo_;
    auto &paParams = tilingData_.paParams;
    paParams.set_blockTableDim2(info.maxBlockNumPerBatch);
    paParams.set_paBlockNumSum(info.paBlockNumSum);
    paParams.set_paLayoutType(BSA_PA_LAYOUT_TYPE_BNBD);
    paParams.set_kvBlockSize(info.kvBlockSizeVal);
    paParams.set_qBlockSize(info.qBlockSizeVal);
    paParams.set_paBlockStride(info.paBlockStrideVal);
    paParams.set_attenMaskS1Size(static_cast<int32_t>(info.s1Size));
    paParams.set_isRowInvalid(1U);
}

void QuantBlockSparseAttnTiling::FillSparseParams()
{
    const auto &info = *tilingInfo_;
    auto &sparseParams = tilingData_.sparseParams;
    sparseParams.set_sparseSeqLenStride(info.qbMax);
    sparseParams.set_sparseIndicesStride(info.sparseCount);
}

void QuantBlockSparseAttnTiling::FillInputParams()
{
    const auto &info = *tilingInfo_;
    const uint8_t compatMaskMode = (info.maskModeVal == 3U) ? BSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : BSA_COMPAT_MASK_NONE;
    auto &inputParams = tilingData_.inputParamsRegbase;
    inputParams.set_bSize(info.bSize);
    inputParams.set_t1Size(info.qTokenNum);
    inputParams.set_t2Size(info.s2Size);
    inputParams.set_n2Size(info.n2Size);
    inputParams.set_gSize(info.gSize);
    inputParams.set_s1Size(info.s1Size);
    inputParams.set_s2Size(info.s2Size);
    inputParams.set_dSize(info.dSize);
    inputParams.set_dSizeV(info.dSizeV);
    inputParams.set_scaleValue(info.softmaxScaleVal);
    inputParams.set_preTokens(std::numeric_limits<int32_t>::max());
    inputParams.set_nextTokens(0);
    inputParams.set_bandIndex(0); // 当前为起始 idx
    uint8_t layoutTypeVal = BSA_LAYOUT_TYPE_TND;
    if (info.layoutQValue == BSA_LAYOUT_Q_NTD_VALUE) {
        layoutTypeVal = BSA_LAYOUT_TYPE_NTD;
    }
    inputParams.set_layoutType(layoutTypeVal);
    inputParams.set_attenMaskShapeType(2);
    inputParams.set_attenMaskCompressMode(compatMaskMode);
    inputParams.set_attenMaskS2Size(2048);
    inputParams.set_isSeqUsedQlenNull(info.opParamInfo.cuSeqlensQ.tensor == nullptr ? 1 : 0);
    inputParams.set_isSeqUsedKvlenNull(info.opParamInfo.cuSeqlensKV.tensor == nullptr ? 1 : 0);
    inputParams.set_seqUsedQlenSize(info.bSize);
    inputParams.set_seqUsedKvlenSize(info.bSize);
    inputParams.set_isKvContinuous(1); // 稀疏算子固定设为 1
    inputParams.set_fromFused(0);      // 融合算子标记，稀疏算子固定设为 0
    inputParams.set_isGqa(info.isGqa ? 1 : 0);
    inputParams.set_isSoftMaxLseEnable(info.returnSoftmaxLseVal ? 1 : 0);
}

void QuantBlockSparseAttnTiling::FillMultiCoreParams()
{
    const auto &info = *tilingInfo_;
    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    multiCoreParams.set_coreNum(static_cast<int32_t>(usedCoreNum_));
    multiCoreParams.set_s1OuterSize(info.qbMax);

    uint32_t bnStartIdx[BSA_CORE_SPLIT_NUM] = {};
    for (uint32_t boundaryIdx = 0U; boundaryIdx <= usedCoreNum_; ++boundaryIdx) {
        const uint64_t taskOffset = totalTaskNum_ * boundaryIdx / usedCoreNum_;
        bnStartIdx[boundaryIdx] = static_cast<uint32_t>(taskOffset / info.gS1OuterSize);
    }
    multiCoreParams.set_bnStartIdx(bnStartIdx);
}

void QuantBlockSparseAttnTiling::FillInitOutputParams()
{
    const auto &info = *tilingInfo_;
    auto &initOutputParams = tilingData_.initOutputParams;
    const int64_t totalOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size * BSA_D_SIZE;
    const int64_t totalSoftmaxLseSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size;
    initOutputParams.set_singleCoreSize((totalOutputSize + static_cast<int64_t>(usedCoreNum_) * 2 - 1) /
                                        (static_cast<int64_t>(usedCoreNum_) * 2));
    initOutputParams.set_needInit(1);
    initOutputParams.set_totalOutputSize(totalOutputSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(totalSoftmaxLseSize);
}

void QuantBlockSparseAttnTiling::CalcTilingKey()
{
    const auto &info = *tilingInfo_;
    tilingKey_ = GET_TPL_TILING_KEY(BSA_DTYPE_FP8_E4M3FN, info.layoutQValue, BSA_KV_LAYOUT_PA_BNSD, info.maskModeVal,
                                    info.returnSoftmaxLseVal ? 1U : 0U);
}

void QuantBlockSparseAttnTiling::CalcWorkspaceSize() { workspaceSize_ = 0; }

void QuantBlockSparseAttnTiling::PrintAllTilingData()
{
    auto &paParams = tilingData_.paParams;
    OP_LOGD(kOpName, "===== PaParams =====");
    OP_LOGD(kOpName, "blockTableDim2:%u", paParams.get_blockTableDim2());
    OP_LOGD(kOpName, "paBlockNumSum:%u", paParams.get_paBlockNumSum());
    OP_LOGD(kOpName, "paLayoutType:%u", paParams.get_paLayoutType());
    OP_LOGD(kOpName, "kvBlockSize:%u", paParams.get_kvBlockSize());
    OP_LOGD(kOpName, "qBlockSize:%u", paParams.get_qBlockSize());
    OP_LOGD(kOpName, "paBlockStride:%u", paParams.get_paBlockStride());
    OP_LOGD(kOpName, "attenMaskS1Size:%d", paParams.get_attenMaskS1Size());
    OP_LOGD(kOpName, "isRowInvalid:%u", paParams.get_isRowInvalid());

    auto &sparseParams = tilingData_.sparseParams;
    OP_LOGD(kOpName, "===== SparseParams =====");
    OP_LOGD(kOpName, "sparseSeqLenStride:%u", sparseParams.get_sparseSeqLenStride());
    OP_LOGD(kOpName, "sparseIndicesStride:%u", sparseParams.get_sparseIndicesStride());

    auto &inputParams = tilingData_.inputParamsRegbase;
    OP_LOGD(kOpName, "===== InputParamsRegbase =====");
    OP_LOGD(kOpName, "bSize:%ld", inputParams.get_bSize());
    OP_LOGD(kOpName, "t1Size:%ld", inputParams.get_t1Size());
    OP_LOGD(kOpName, "t2Size:%ld", inputParams.get_t2Size());
    OP_LOGD(kOpName, "n2Size:%ld", inputParams.get_n2Size());
    OP_LOGD(kOpName, "gSize:%ld", inputParams.get_gSize());
    OP_LOGD(kOpName, "s1Size:%ld", inputParams.get_s1Size());
    OP_LOGD(kOpName, "s2Size:%ld", inputParams.get_s2Size());
    OP_LOGD(kOpName, "dSize:%ld", inputParams.get_dSize());
    OP_LOGD(kOpName, "dSizeV:%ld", inputParams.get_dSizeV());
    OP_LOGD(kOpName, "scaleValue:%f", inputParams.get_scaleValue());
    OP_LOGD(kOpName, "preTokens:%ld", inputParams.get_preTokens());
    OP_LOGD(kOpName, "nextTokens:%ld", inputParams.get_nextTokens());
    OP_LOGD(kOpName, "layoutType:%u", inputParams.get_layoutType());
    OP_LOGD(kOpName, "attenMaskCompressMode:%u", inputParams.get_attenMaskCompressMode());
    OP_LOGD(kOpName, "attenMaskS2Size:%u", inputParams.get_attenMaskS2Size());
    OP_LOGD(kOpName, "isSeqUsedQlenNull:%u", inputParams.get_isSeqUsedQlenNull());
    OP_LOGD(kOpName, "isSeqUsedKvlenNull:%u", inputParams.get_isSeqUsedKvlenNull());
    OP_LOGD(kOpName, "isKvContinuous:%u", inputParams.get_isKvContinuous());
    OP_LOGD(kOpName, "isGqa:%u", inputParams.get_isGqa());
    OP_LOGD(kOpName, "isSoftMaxLseEnable:%u", inputParams.get_isSoftMaxLseEnable());

    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    OP_LOGD(kOpName, "===== MultiCoreParams =====");
    OP_LOGD(kOpName, "coreNum:%d", multiCoreParams.get_coreNum());
    OP_LOGD(kOpName, "s1OuterSize:%ld", multiCoreParams.get_s1OuterSize());
    for (uint32_t i = 0; i <= usedCoreNum_ && i < BSA_CORE_SPLIT_NUM; ++i) {
        OP_LOGD(kOpName, "bnStartIdx[%u]:%u", i, multiCoreParams.get_bnStartIdx()[i]);
    }

    auto &initOutputParams = tilingData_.initOutputParams;
    OP_LOGD(kOpName, "===== InitOutputParams =====");
    OP_LOGD(kOpName, "singleCoreSize:%u", initOutputParams.get_singleCoreSize());
    OP_LOGD(kOpName, "needInit:%u", initOutputParams.get_needInit());
    OP_LOGD(kOpName, "totalOutputSize:%ld", initOutputParams.get_totalOutputSize());
    OP_LOGD(kOpName, "totalSoftMaxLseOutputSize:%ld", initOutputParams.get_totalSoftMaxLseOutputSize());

    OP_LOGD(kOpName, "===== Summary =====");
    OP_LOGD(kOpName, "usedCoreNum:%u", usedCoreNum_);
    OP_LOGD(kOpName, "totalTaskNum:%lu", totalTaskNum_);
    OP_LOGD(kOpName, "tilingKey:%lu", tilingKey_);
    OP_LOGD(kOpName, "workspaceSize:%lu", workspaceSize_);
    OP_LOGD(kOpName, "tilingDataSize:%lu", tilingData_.GetDataSize());
    OP_LOGD(kOpName, "tilingDataCapacity:%lu", context_->GetRawTilingData()->GetCapacity());
}

ge::graphStatus QuantBlockSparseAttnTiling::DoOpTiling(QuantBlockSparseAttnTilingInfo *tilingInfo)
{
    if (tilingInfo == nullptr || context_ == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: tilingInfo=%p or context=%p is nullptr", tilingInfo, context_);
        return ge::GRAPH_FAILED;
    }
    tilingInfo_ = tilingInfo;

    const uint64_t bnCount = tilingInfo_->bSize * tilingInfo_->n1Size;
    totalTaskNum_ = bnCount * tilingInfo_->gS1OuterSize;
    if (totalTaskNum_ == 0U || totalTaskNum_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bSize * n1Size * gS1OuterSize", std::to_string(totalTaskNum_),
            "total task num must be in range [1, " + std::to_string(std::numeric_limits<uint32_t>::max()) + "]");
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxAicCoreNum = GetAicCoreNum(context_);
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(maxAicCoreNum), totalTaskNum_));

    FillPaParams();
    FillSparseParams();
    FillInputParams();
    FillMultiCoreParams();
    FillInitOutputParams();
    CalcTilingKey();
    CalcWorkspaceSize();
    PrintAllTilingData();

    auto rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: rawTilingData is nullptr");
        return ge::GRAPH_FAILED;
    }

    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingQuantBlockSparseAttn(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "Tiling: tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnTilingInfo tilingInfo;
    QuantBlockSparseAttnInfoParser parser(context);
    if (parser.Parse(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnCheck checker(tilingInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QuantBlockSparseAttnTiling tiling(context);
    return tiling.DoOpTiling(&tilingInfo);
}

} // namespace optiling

namespace ge {
graphStatus TilingPrepareQuantBlockSparseAttn(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantBlockSparseAttn", "TilingPrepare: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class QuantBlockSparseAttnTiling {
public:
    static ge::graphStatus Tiling(gert::TilingContext *context)
    {
        return optiling::TilingQuantBlockSparseAttn(context);
    }
};
} // namespace ops

namespace optiling {
IMPL_OP_OPTILING(QuantBlockSparseAttn).Tiling(TilingQuantBlockSparseAttn);
} // namespace optiling
