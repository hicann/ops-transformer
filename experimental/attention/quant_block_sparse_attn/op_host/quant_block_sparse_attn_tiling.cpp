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
#include <cstring>
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
constexpr uint8_t QBSA_PA_LAYOUT_TYPE_BNBD = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_NONE = 0U;
constexpr uint8_t QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL = 2U;
constexpr uint32_t QBSA_COMBINE_ALIGN_BYTES = 32U;
constexpr uint32_t QBSA_K_SCALE_BYTES = sizeof(float);
constexpr uint32_t QBSA_ATTEN_MASK_DEFAULT_BATCH = 1U;
constexpr uint32_t QBSA_ATTEN_MASK_DEFAULT_S1_SIZE = 2048U;
constexpr uint32_t QBSA_ATTEN_MASK_DEFAULT_S2_SIZE = 2048U;
constexpr uint32_t QBSA_MXFP8_VALUE_SCALE_LAST_DIM = 2U;

uint32_t GetAicCoreNum(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return QBSA_MAX_CORE_NUM;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    return aicNum == 0U ? QBSA_MAX_CORE_NUM : std::min<uint32_t>(aicNum, QBSA_MAX_CORE_NUM);
}

uint32_t CalcMxScaleDSize(uint32_t dSize)
{
    // 返回 MX scale 的 D/64 维度。
    return QBSACeilDiv(dSize, QBSA_MXFP8_SCALE_GROUP_SIZE);
}

QuantBlockSparseAttnMxStrideParams MakePaBnbdStride(uint32_t n2Size, uint32_t blockSize, uint32_t lastDim)
{
    // PA BNBD：[physicalBlock,N,blockSize,lastDim]。
    QuantBlockSparseAttnMxStrideParams stride;
    stride.n2Stride = static_cast<uint64_t>(blockSize) * lastDim;
    stride.bnStride = static_cast<uint64_t>(n2Size) * stride.n2Stride;
    return stride;
}

} // namespace

QuantBlockSparseAttnTiling::QuantBlockSparseAttnTiling(gert::TilingContext *context)
    : context_(context)
{}

void QuantBlockSparseAttnTiling::FillPaParams()
{
    const auto &info = *tilingInfo_;
    auto &paParams = tilingData_.paParams;
    paParams.set_blockTableDim2(info.maxBlockNumPerBatch);
    paParams.set_paBlockNumSum(info.paBlockNumSum);
    paParams.set_paLayoutType(QBSA_PA_LAYOUT_TYPE_BNBD);
    paParams.set_kvBlockSize(info.kvBlockSizeVal);
    paParams.set_qBlockSize(info.qBlockSizeVal);
    paParams.set_paBlockStride(info.paBlockStrideVal);
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
    const uint8_t compatMaskMode =
        (info.maskModeVal == QBSA_MASK_MODE_CAUSAL) ? QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : QBSA_COMPAT_MASK_NONE;
    auto &inputParams = tilingData_.inputParamsRegbase;
    inputParams.set_bSize(info.bSize);
    inputParams.set_t1Size(info.qTokenNum);
    inputParams.set_n2Size(info.n2Size);
    inputParams.set_gSize(info.gSize);
    inputParams.set_dSize(info.dSize);
    inputParams.set_dSizeV(info.dSizeV);
    inputParams.set_scaleValue(info.softmaxScaleVal);
    inputParams.set_preTokens(std::numeric_limits<int32_t>::max());
    inputParams.set_nextTokens(0);
    inputParams.set_bandIndex(0); // 当前为起始 idx
    uint8_t layoutTypeVal = QBSA_REGBASE_LAYOUT_TYPE_TND;
    if (info.layoutQValue == QBSA_LAYOUT_Q_NTD_VALUE) {
        layoutTypeVal = QBSA_REGBASE_LAYOUT_TYPE_NTD;
    }
    inputParams.set_layoutType(layoutTypeVal);
    inputParams.set_attenMaskCompressMode(compatMaskMode);
    inputParams.set_attenMaskS2Size(2048);
    inputParams.set_seqUsedQlenSize(info.bSize);
    inputParams.set_seqUsedKvlenSize(info.bSize);
    inputParams.set_isKvContinuous(1); // 稀疏算子固定设为 1
    inputParams.set_fromFused(0);      // 融合算子标记，稀疏算子固定设为 0
    inputParams.set_isGqa(info.isGqa ? 1 : 0);
    inputParams.set_isSoftMaxLseEnable(info.returnSoftmaxLseVal ? 1 : 0);
    uint32_t fp8PScaleShapeSize = 0U;
    if (info.opParamInfo.pScale.shape != nullptr) {
        const int64_t shapeSize = info.opParamInfo.pScale.shape->GetStorageShape().GetShapeSize();
        fp8PScaleShapeSize = (shapeSize > 0) ? static_cast<uint32_t>(shapeSize) : 0U;
    }
    inputParams.set_pScaleShapeSize(fp8PScaleShapeSize);
}

void QuantBlockSparseAttnTiling::FillMultiCoreParams()
{
    const auto &info = *tilingInfo_;
    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    multiCoreParams.set_coreNum(static_cast<int32_t>(usedCoreNum_));
    multiCoreParams.set_s1OuterSize(info.qbMax);

    uint32_t bnStartIdx[QBSA_CORE_SPLIT_NUM] = {};
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
    const int64_t totalOutputSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size * QBSA_D_SIZE;
    const int64_t totalSoftmaxLseSize = static_cast<int64_t>(info.qTokenNum) * info.n1Size;
    initOutputParams.set_singleCoreSize((totalOutputSize + static_cast<int64_t>(usedCoreNum_) * 2 - 1) /
                                        (static_cast<int64_t>(usedCoreNum_) * 2));
    initOutputParams.set_needInit(1);
    initOutputParams.set_totalOutputSize(totalOutputSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(totalSoftmaxLseSize);
}

void QuantBlockSparseAttnTiling::FillMxTilingData()
{
    // 填充独立 MX tiling payload。
    const auto &info = *tilingInfo_;
    const uint8_t compatMaskMode =
        (info.maskModeVal == QBSA_MASK_MODE_CAUSAL) ? QBSA_COMPAT_MASK_RIGHT_DOWN_CAUSAL : QBSA_COMPAT_MASK_NONE;
    const uint32_t dSize = info.dSize == 0U ? QBSA_D_SIZE : info.dSize;
    const uint32_t dSizeV = info.dSizeV == 0U ? QBSA_D_SIZE : info.dSizeV;
    const uint32_t queryScaleDSize = CalcMxScaleDSize(dSize);
    const uint32_t keyScaleDSize = CalcMxScaleDSize(dSize);
    // 每个 PA block 的 VScale 为 [N,ceil(blockSize/64),DV,2]。
    const uint32_t valueScaleBlockSize = QBSACeilDiv(info.paBlockSizeVal, QBSA_MXFP8_SCALE_GROUP_SIZE);
    const uint32_t valueScaleDSize = dSizeV * QBSA_MXFP8_VALUE_SCALE_LAST_DIM;
    const bool actualSeqQNull = info.opParamInfo.cuSeqlensQ.tensor == nullptr;
    const bool actualSeqKVNull = info.opParamInfo.seqUsedKV.tensor == nullptr;

    auto &attrParams = mxTilingData_.attrParams;
    attrParams.layoutQ = info.layoutQValue;
    attrParams.layoutKv = QBSA_REGBASE_KV_PA_BNSD;
    attrParams.layoutSparseIndices = QBSA_REGBASE_SPARSE_B_N_QB_KB;
    attrParams.quantMode = info.quantModeVal;
    attrParams.maskMode = info.maskModeVal;
    attrParams.returnSoftmaxLse = info.returnSoftmaxLseVal ? 1U : 0U;

    auto &baseParams = mxTilingData_.baseParams;
    baseParams.bSize = info.bSize;
    baseParams.t1Size = info.qTokenNum;
    baseParams.n2Size = info.n2Size;
    baseParams.gSize = info.gSize;
    baseParams.dSize = dSize;
    baseParams.dSizeV = dSizeV;
    baseParams.dSizeRope = 0U;
    baseParams.actualSeqLengthsQSize = actualSeqQNull ? 0U : info.bSize;
    baseParams.actualSeqLengthsKVSize = actualSeqKVNull ? 0U : info.bSize;
    baseParams.scaleValue = info.softmaxScaleVal;
    baseParams.isKvContinuous = 0U;
    baseParams.isActualSeqLengthsNull = actualSeqQNull ? 1U : 0U;
    baseParams.isActualSeqLengthsKVNull = actualSeqKVNull ? 1U : 0U;
    baseParams.coreNum = usedCoreNum_;
    baseParams.outputLayout = info.layoutQValue;
    // K/V 数据与 K/V scale 均按 PA BNBD 映射。
    baseParams.keyStrides = MakePaBnbdStride(info.n2Size, info.paBlockSizeVal, dSize);
    baseParams.valueStrides = MakePaBnbdStride(info.n2Size, info.paBlockSizeVal, dSizeV);
    baseParams.kScaleStrides =
        MakePaBnbdStride(info.n2Size, info.paBlockSizeVal, keyScaleDSize * QBSA_MXFP8_SCALE_LAST_DIM);
    baseParams.vScaleStrides = MakePaBnbdStride(info.n2Size, valueScaleBlockSize, valueScaleDSize);

    auto &attenMaskParams = mxTilingData_.attenMaskParams;
    attenMaskParams.sparseMode = static_cast<uint8_t>(info.maskModeVal);
    attenMaskParams.attenMaskDataType = 1U;
    attenMaskParams.attenMaskCompressMode = compatMaskMode;
    attenMaskParams.isRowInvalidOpen = 1U;
    attenMaskParams.preTokens = std::numeric_limits<int32_t>::max();
    attenMaskParams.nextTokens = 0;
    attenMaskParams.attenMaskBatch = QBSA_ATTEN_MASK_DEFAULT_BATCH;
    attenMaskParams.attenMaskS1Size = QBSA_ATTEN_MASK_DEFAULT_S1_SIZE;
    attenMaskParams.attenMaskS2Size = QBSA_ATTEN_MASK_DEFAULT_S2_SIZE;
    attenMaskParams.isExistRowInvalid = 1U;

    auto &pageAttentionParams = mxTilingData_.pageAttentionParams;
    pageAttentionParams.paLayoutType = QBSA_PA_LAYOUT_TYPE_BNBD;
    pageAttentionParams.blockSize = info.paBlockSizeVal;
    pageAttentionParams.maxBlockNumPerBatch = info.maxBlockNumPerBatch;
    pageAttentionParams.paBlockNumSum = info.paBlockNumSum;
    pageAttentionParams.paBlockStride = info.paBlockStrideVal;
    pageAttentionParams.qBlockSize = info.qBlockSizeVal;
    pageAttentionParams.kvBlockSize = info.kvBlockSizeVal;

    auto &sparseParams = mxTilingData_.sparseParams;
    sparseParams.gS1OuterSize = info.gS1OuterSize;
    sparseParams.sparseSeqLenStride = info.qbMax;
    sparseParams.sparseIndicesStride = info.sparseCount;
    sparseParams.maxQb = info.qbMax;
    sparseParams.maxKb = info.sparseCount;
    sparseParams.sparseCount = info.sparseCount;

    auto &workspaceParams = mxTilingData_.workspaceParams;
    workspaceParams.accumOutSize = 0U;
    workspaceParams.logSumExpSize = 0U;

    auto &scaleParams = mxTilingData_.scaleParams;
    // Q/K mode=6，V mode=8。
    scaleParams.scaleGroupSize = QBSA_MXFP8_SCALE_GROUP_SIZE;
    scaleParams.scaleLastDim = QBSA_MXFP8_SCALE_LAST_DIM;
    scaleParams.queryScaleDSize = queryScaleDSize;
    scaleParams.keyScaleDSize = keyScaleDSize;
    scaleParams.valueScaleBlockSize = valueScaleBlockSize;
    scaleParams.valueScaleDSize = valueScaleDSize;
    uint32_t pScaleShapeSize = 0U;
    uint8_t pScaleDtype = MX_PSCALE_DTYPE_E8M0;
    if (info.opParamInfo.pScale.shape != nullptr) {
        const int64_t shapeSize = info.opParamInfo.pScale.shape->GetStorageShape().GetShapeSize();
        pScaleShapeSize = (shapeSize > 0) ? static_cast<uint32_t>(shapeSize) : 0U;
        if (pScaleShapeSize > 0U && info.opParamInfo.pScale.desc != nullptr) {
            if (info.opParamInfo.pScale.desc->GetDataType() == ge::DT_FLOAT) {
                pScaleDtype = MX_PSCALE_DTYPE_FP32;
            }
        }
    }
    scaleParams.pScaleShapeSize = pScaleShapeSize;
    scaleParams.pScaleDtype = pScaleDtype;
    scaleParams.queryQuantMode = QBSA_MXFP8_PER_TOKEN_GROUP_MODE;
    scaleParams.keyAntiquantMode = QBSA_MXFP8_PER_TOKEN_GROUP_MODE;
    scaleParams.valueAntiquantMode = QBSA_MXFP8_PER_CHANNEL_GROUP_MODE;

    auto &emptyTensorParams = mxTilingData_.emptyTensorParams;
    const uint64_t totalOutputSize = static_cast<uint64_t>(info.qTokenNum) * info.n1Size * dSizeV;
    const uint64_t totalSoftmaxLseSize = static_cast<uint64_t>(info.qTokenNum) * info.n1Size;
    emptyTensorParams.totalOutputSize = totalOutputSize;
    emptyTensorParams.totalSoftMaxLseOutputSize = totalSoftmaxLseSize;
}

void QuantBlockSparseAttnTiling::CalcTilingKey()
{
    const auto &info = *tilingInfo_;
    if (info.quantModeVal == QBSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        // MX 使用独立 tiling data 和 S2=512 config。
        tilingKey_ = GET_TPL_TILING_KEY(QBSA_DTYPE_FP8_E4M3FN, info.layoutQValue, QBSA_KV_LAYOUT_PA_BNSD,
                                        info.maskModeVal, info.returnSoftmaxLseVal ? 1U : 0U,
                                        Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128, MXFullQuantMode);
    } else {
        tilingKey_ = GET_TPL_TILING_KEY(QBSA_DTYPE_FP8_E4M3FN, info.layoutQValue, QBSA_KV_LAYOUT_PA_BNSD,
                                        info.maskModeVal, info.returnSoftmaxLseVal ? 1U : 0U,
                                        Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128, FP8QuantMode);
    }
}

void QuantBlockSparseAttnTiling::CalcWorkspaceSize()
{
    workspaceSize_ = 0;
}

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
    OP_LOGD(kOpName, "isRowInvalid:%u", paParams.get_isRowInvalid());

    auto &sparseParams = tilingData_.sparseParams;
    OP_LOGD(kOpName, "===== SparseParams =====");
    OP_LOGD(kOpName, "sparseSeqLenStride:%u", sparseParams.get_sparseSeqLenStride());
    OP_LOGD(kOpName, "sparseIndicesStride:%u", sparseParams.get_sparseIndicesStride());

    auto &inputParams = tilingData_.inputParamsRegbase;
    OP_LOGD(kOpName, "===== InputParamsRegbase =====");
    OP_LOGD(kOpName, "bSize:%ld", inputParams.get_bSize());
    OP_LOGD(kOpName, "t1Size:%ld", inputParams.get_t1Size());
    OP_LOGD(kOpName, "n2Size:%ld", inputParams.get_n2Size());
    OP_LOGD(kOpName, "gSize:%ld", inputParams.get_gSize());
    OP_LOGD(kOpName, "dSize:%ld", inputParams.get_dSize());
    OP_LOGD(kOpName, "dSizeV:%ld", inputParams.get_dSizeV());
    OP_LOGD(kOpName, "scaleValue:%f", inputParams.get_scaleValue());
    OP_LOGD(kOpName, "preTokens:%ld", inputParams.get_preTokens());
    OP_LOGD(kOpName, "nextTokens:%ld", inputParams.get_nextTokens());
    OP_LOGD(kOpName, "layoutType:%u", inputParams.get_layoutType());
    OP_LOGD(kOpName, "attenMaskCompressMode:%u", inputParams.get_attenMaskCompressMode());
    OP_LOGD(kOpName, "attenMaskS2Size:%u", inputParams.get_attenMaskS2Size());
    OP_LOGD(kOpName, "isKvContinuous:%u", inputParams.get_isKvContinuous());
    OP_LOGD(kOpName, "isGqa:%u", inputParams.get_isGqa());
    OP_LOGD(kOpName, "isSoftMaxLseEnable:%u", inputParams.get_isSoftMaxLseEnable());

    auto &multiCoreParams = tilingData_.multiCoreParamsRegbase;
    OP_LOGD(kOpName, "===== MultiCoreParams =====");
    OP_LOGD(kOpName, "coreNum:%d", multiCoreParams.get_coreNum());
    OP_LOGD(kOpName, "s1OuterSize:%ld", multiCoreParams.get_s1OuterSize());
    for (uint32_t i = 0; i <= usedCoreNum_ && i < QBSA_CORE_SPLIT_NUM; ++i) {
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
    auto rawTilingData = context_->GetRawTilingData();
    OP_LOGD(kOpName, "tilingDataCapacity:%lu", rawTilingData == nullptr ? 0UL : rawTilingData->GetCapacity());
}

void QuantBlockSparseAttnTiling::PrintMxTilingData()
{
    const auto &attrParams = mxTilingData_.attrParams;
    OP_LOGD(kOpName, "===== MX AttrParams =====");
    OP_LOGD(kOpName, "layoutQ:%u", attrParams.layoutQ);
    OP_LOGD(kOpName, "layoutKv:%u", attrParams.layoutKv);
    OP_LOGD(kOpName, "layoutSparseIndices:%u", attrParams.layoutSparseIndices);
    OP_LOGD(kOpName, "quantMode:%u", attrParams.quantMode);
    OP_LOGD(kOpName, "maskMode:%u", attrParams.maskMode);
    OP_LOGD(kOpName, "returnSoftmaxLse:%u", attrParams.returnSoftmaxLse);

    const auto &baseParams = mxTilingData_.baseParams;
    OP_LOGD(kOpName, "===== MX BaseParams =====");
    OP_LOGD(kOpName, "bSize:%u", baseParams.bSize);
    OP_LOGD(kOpName, "t1Size:%u", baseParams.t1Size);
    OP_LOGD(kOpName, "n2Size:%u", baseParams.n2Size);
    OP_LOGD(kOpName, "gSize:%u", baseParams.gSize);
    OP_LOGD(kOpName, "dSize:%u", baseParams.dSize);
    OP_LOGD(kOpName, "dSizeV:%u", baseParams.dSizeV);
    OP_LOGD(kOpName, "actualSeqLengthsQSize:%u", baseParams.actualSeqLengthsQSize);
    OP_LOGD(kOpName, "actualSeqLengthsKVSize:%u", baseParams.actualSeqLengthsKVSize);
    OP_LOGD(kOpName, "scaleValue:%f", baseParams.scaleValue);
    OP_LOGD(kOpName, "isKvContinuous:%u", baseParams.isKvContinuous);
    OP_LOGD(kOpName, "coreNum:%u", baseParams.coreNum);
    OP_LOGD(kOpName, "keyStrides:%lu/%lu", baseParams.keyStrides.bnStride, baseParams.keyStrides.n2Stride);
    OP_LOGD(kOpName, "valueStrides:%lu/%lu", baseParams.valueStrides.bnStride, baseParams.valueStrides.n2Stride);
    OP_LOGD(kOpName, "kScaleStrides:%lu/%lu", baseParams.kScaleStrides.bnStride, baseParams.kScaleStrides.n2Stride);
    OP_LOGD(kOpName, "vScaleStrides:%lu/%lu", baseParams.vScaleStrides.bnStride, baseParams.vScaleStrides.n2Stride);

    const auto &attenMaskParams = mxTilingData_.attenMaskParams;
    OP_LOGD(kOpName, "===== MX AttenMaskParams =====");
    OP_LOGD(kOpName, "sparseMode:%u", attenMaskParams.sparseMode);
    OP_LOGD(kOpName, "attenMaskCompressMode:%u", attenMaskParams.attenMaskCompressMode);
    OP_LOGD(kOpName, "preTokens:%d", attenMaskParams.preTokens);
    OP_LOGD(kOpName, "nextTokens:%d", attenMaskParams.nextTokens);
    OP_LOGD(kOpName, "attenMaskS1Size:%u", attenMaskParams.attenMaskS1Size);
    OP_LOGD(kOpName, "attenMaskS2Size:%u", attenMaskParams.attenMaskS2Size);

    const auto &pageAttentionParams = mxTilingData_.pageAttentionParams;
    OP_LOGD(kOpName, "===== MX PageAttentionParams =====");
    OP_LOGD(kOpName, "paLayoutType:%u", pageAttentionParams.paLayoutType);
    OP_LOGD(kOpName, "blockSize:%u", pageAttentionParams.blockSize);
    OP_LOGD(kOpName, "maxBlockNumPerBatch:%u", pageAttentionParams.maxBlockNumPerBatch);
    OP_LOGD(kOpName, "paBlockNumSum:%u", pageAttentionParams.paBlockNumSum);
    OP_LOGD(kOpName, "paBlockStride:%u", pageAttentionParams.paBlockStride);
    OP_LOGD(kOpName, "qBlockSize:%u", pageAttentionParams.qBlockSize);
    OP_LOGD(kOpName, "kvBlockSize:%u", pageAttentionParams.kvBlockSize);

    const auto &sparseParams = mxTilingData_.sparseParams;
    OP_LOGD(kOpName, "===== MX SparseParams =====");
    OP_LOGD(kOpName, "gS1OuterSize:%u", sparseParams.gS1OuterSize);
    OP_LOGD(kOpName, "sparseSeqLenStride:%u", sparseParams.sparseSeqLenStride);
    OP_LOGD(kOpName, "sparseIndicesStride:%u", sparseParams.sparseIndicesStride);
    OP_LOGD(kOpName, "maxQb:%u", sparseParams.maxQb);
    OP_LOGD(kOpName, "maxKb:%u", sparseParams.maxKb);
    OP_LOGD(kOpName, "sparseCount:%u", sparseParams.sparseCount);

    const auto &scaleParams = mxTilingData_.scaleParams;
    OP_LOGD(kOpName, "===== MX ScaleParams =====");
    OP_LOGD(kOpName, "scaleGroupSize:%u", scaleParams.scaleGroupSize);
    OP_LOGD(kOpName, "scaleLastDim:%u", scaleParams.scaleLastDim);
    OP_LOGD(kOpName, "queryScaleDSize:%u", scaleParams.queryScaleDSize);
    OP_LOGD(kOpName, "keyScaleDSize:%u", scaleParams.keyScaleDSize);
    OP_LOGD(kOpName, "valueScaleBlockSize:%u", scaleParams.valueScaleBlockSize);
    OP_LOGD(kOpName, "valueScaleDSize:%u", scaleParams.valueScaleDSize);
    OP_LOGD(kOpName, "pScaleShapeSize:%u", scaleParams.pScaleShapeSize);
    OP_LOGD(kOpName, "queryQuantMode/keyAntiquantMode/valueAntiquantMode:%u/%u/%u", scaleParams.queryQuantMode,
            scaleParams.keyAntiquantMode, scaleParams.valueAntiquantMode);

    const auto &emptyTensorParams = mxTilingData_.emptyTensorParams;
    OP_LOGD(kOpName, "===== MX EmptyTensorParams =====");
    OP_LOGD(kOpName, "totalOutputSize:%lu", emptyTensorParams.totalOutputSize);
    OP_LOGD(kOpName, "totalSoftMaxLseOutputSize:%lu", emptyTensorParams.totalSoftMaxLseOutputSize);

    auto rawTilingData = context_->GetRawTilingData();
    OP_LOGD(kOpName, "===== MX Summary =====");
    OP_LOGD(kOpName, "usedCoreNum:%u", usedCoreNum_);
    OP_LOGD(kOpName, "totalTaskNum:%lu", totalTaskNum_);
    OP_LOGD(kOpName, "tilingKey:%lu", tilingKey_);
    OP_LOGD(kOpName, "workspaceSize:%lu", workspaceSize_);
    OP_LOGD(kOpName, "mxTilingDataSize:%lu", sizeof(QuantBlockSparseAttnMxTilingData));
    OP_LOGD(kOpName, "tilingDataCapacity:%lu", rawTilingData == nullptr ? 0UL : rawTilingData->GetCapacity());
}

ge::graphStatus QuantBlockSparseAttnTiling::SaveTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(kOpName, "DoOpTiling: rawTilingData is nullptr");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_->quantModeVal == QBSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        const auto dataSize = sizeof(QuantBlockSparseAttnMxTilingData);
        if (rawTilingData->GetCapacity() < dataSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "tilingDataCapacity", std::to_string(rawTilingData->GetCapacity()),
                "Must be at least " + std::to_string(dataSize) + " for MXFP8 full-quant tiling data");
            return ge::GRAPH_FAILED;
        }
        std::memcpy(rawTilingData->GetData(), &mxTilingData_, dataSize);
        rawTilingData->SetDataSize(dataSize);
        return ge::GRAPH_SUCCESS;
    }

    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
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
            "Total task num must be in range [1, " + std::to_string(std::numeric_limits<uint32_t>::max()) + "]");
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxAicCoreNum = GetAicCoreNum(context_);
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(maxAicCoreNum), totalTaskNum_));

    if (tilingInfo_->quantModeVal == QBSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        FillMxTilingData();
    } else {
        FillPaParams();
        FillSparseParams();
        FillInputParams();
        FillMultiCoreParams();
        FillInitOutputParams();
    }
    CalcTilingKey();
    CalcWorkspaceSize();
    if (tilingInfo_->quantModeVal == QBSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        PrintMxTilingData();
    } else {
        PrintAllTilingData();
    }

    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    return SaveTilingData();
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
