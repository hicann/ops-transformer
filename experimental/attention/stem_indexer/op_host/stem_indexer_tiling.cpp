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
 * \file stem_indexer_tiling.cpp
 * \brief
 */

#include "stem_indexer_tiling.h"

#include "../op_kernel/stem_indexer_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
bool StemIndexerInfoParser::IsFloatEqual(float lhs, float rhs) const
{
    return (lhs > rhs - ATTR_FLOAT_EPS) && (lhs < rhs + ATTR_FLOAT_EPS);
}

bool StemIndexerInfoParser::IsSupportedQHeadNum(uint32_t qHeadNum) const
{
    return qHeadNum == Q_HEAD_NUM_32 || qHeadNum == Q_HEAD_NUM_64;
}

bool StemIndexerInfoParser::IsSupportedKvHeadNum(uint32_t kvHeadNum) const
{
    return kvHeadNum == KV_HEAD_NUM_2 || kvHeadNum == KV_HEAD_NUM_4 || kvHeadNum == KV_HEAD_NUM_8;
}

ge::graphStatus StemIndexerInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("StemIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    // 当前仅适配 A5（ASCEND950），A2/A3 暂未适配
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
        OP_LOGE(opName_, "SOC Version[%d] is not support, only ASCEND950 is supported.",
                static_cast<int32_t>(socVersion_));
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workspace size is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(opName_, "raw tiling data is nullptr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void StemIndexerInfoParser::GetInputParaInfo()
{
    opParamInfo_.qflat.desc = context_->GetInputDesc(QFLAT_INDEX);
    opParamInfo_.qflat.shape = context_->GetInputShape(QFLAT_INDEX);
    opParamInfo_.kflat.desc = context_->GetInputDesc(KFLAT_INDEX);
    opParamInfo_.kflat.shape = context_->GetInputShape(KFLAT_INDEX);
    opParamInfo_.vbias.desc = context_->GetInputDesc(VBIAS_INDEX);
    opParamInfo_.vbias.shape = context_->GetInputShape(VBIAS_INDEX);
    opParamInfo_.qSeqLens.desc = context_->GetInputDesc(Q_SEQ_LENS_INDEX);
    opParamInfo_.qSeqLens.shape = context_->GetInputShape(Q_SEQ_LENS_INDEX);
    opParamInfo_.kvSeqLens.desc = context_->GetInputDesc(KV_SEQ_LENS_INDEX);
    opParamInfo_.kvSeqLens.shape = context_->GetInputShape(KV_SEQ_LENS_INDEX);
    opParamInfo_.numPromptTokens.desc = context_->GetInputDesc(NUM_PROMPT_TOKENS_INDEX);
    opParamInfo_.numPromptTokens.shape = context_->GetInputShape(NUM_PROMPT_TOKENS_INDEX);
    opParamInfo_.metadata.desc = context_->GetInputDesc(METADATA_INDEX);
    opParamInfo_.metadata.shape = context_->GetInputShape(METADATA_INDEX);
}

void StemIndexerInfoParser::GetOutputParaInfo()
{
    opParamInfo_.sparseIndicesOut.desc = context_->GetOutputDesc(SPARSE_INDICES_INDEX);
    opParamInfo_.sparseIndicesOut.shape = context_->GetOutputShape(SPARSE_INDICES_INDEX);
    opParamInfo_.sparseSeqLenOut.desc = context_->GetOutputDesc(SPARSE_SEQ_LEN_INDEX);
    opParamInfo_.sparseSeqLenOut.shape = context_->GetOutputShape(SPARSE_SEQ_LEN_INDEX);
}

ge::graphStatus StemIndexerInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    opParamInfo_.causal = attrs->GetAttrPointer<bool>(ATTR_CAUSAL_INDEX);
    opParamInfo_.stemBlockSize = attrs->GetAttrPointer<int64_t>(ATTR_STEM_BLOCK_SIZE_INDEX);
    opParamInfo_.stemStride = attrs->GetAttrPointer<int64_t>(ATTR_STEM_STRIDE_INDEX);
    opParamInfo_.alpha = attrs->GetAttrPointer<float>(ATTR_ALPHA_INDEX);
    opParamInfo_.initialBlocks = attrs->GetAttrPointer<int64_t>(ATTR_INITIAL_BLOCKS_INDEX);
    opParamInfo_.windowSize = attrs->GetAttrPointer<int64_t>(ATTR_WINDOW_SIZE_INDEX);
    opParamInfo_.kBlockNumRateMedium = attrs->GetAttrPointer<float>(ATTR_K_BLOCK_NUM_RATE_MEDIUM_INDEX);
    opParamInfo_.kBlockNumBiasMedium = attrs->GetAttrPointer<int64_t>(ATTR_K_BLOCK_NUM_BIAS_MEDIUM_INDEX);
    opParamInfo_.kBlockNumRateLarge = attrs->GetAttrPointer<float>(ATTR_K_BLOCK_NUM_RATE_LARGE_INDEX);
    opParamInfo_.kBlockNumBiasLarge = attrs->GetAttrPointer<int64_t>(ATTR_K_BLOCK_NUM_BIAS_LARGE_INDEX);
    opParamInfo_.topkScorePrecision = attrs->GetAttrPointer<int64_t>(ATTR_TOPK_SCORE_PRECISION_INDEX);

    if (CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(*opParamInfo_.stemBlockSize != STEM_BLOCK_SIZE_LIMIT,
                OP_LOGE(opName_, "stem_block_size only supports %u, but got %ld.", STEM_BLOCK_SIZE_LIMIT,
                        *opParamInfo_.stemBlockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *opParamInfo_.stemStride != STEM_STRIDE_LIMIT,
        OP_LOGE(opName_, "stem_stride only supports %u, but got %ld.", STEM_STRIDE_LIMIT, *opParamInfo_.stemStride),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.initialBlocks != INITIAL_BLOCKS_LIMIT,
                OP_LOGE(opName_, "initial_blocks only supports %u, but got %ld.", INITIAL_BLOCKS_LIMIT,
                        *opParamInfo_.initialBlocks),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *opParamInfo_.windowSize != WINDOW_SIZE_LIMIT,
        OP_LOGE(opName_, "window_size only supports %u, but got %ld.", WINDOW_SIZE_LIMIT, *opParamInfo_.windowSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((*opParamInfo_.alpha <= ALPHA_MIN) || (*opParamInfo_.alpha > ALPHA_MAX),
                OP_LOGE(opName_, "alpha should be in (%f, %f], but got %f.", ALPHA_MIN, ALPHA_MAX, *opParamInfo_.alpha),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsFloatEqual(*opParamInfo_.kBlockNumRateMedium, K_BLOCK_NUM_RATE_MEDIUM_LIMIT),
                OP_LOGE(opName_, "k_block_num_rate_medium only supports %f, but got %f.", K_BLOCK_NUM_RATE_MEDIUM_LIMIT,
                        *opParamInfo_.kBlockNumRateMedium),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.kBlockNumBiasMedium != K_BLOCK_NUM_BIAS_MEDIUM_LIMIT,
                OP_LOGE(opName_, "k_block_num_bias_medium only supports %u, but got %ld.",
                        K_BLOCK_NUM_BIAS_MEDIUM_LIMIT, *opParamInfo_.kBlockNumBiasMedium),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsFloatEqual(*opParamInfo_.kBlockNumRateLarge, K_BLOCK_NUM_RATE_LARGE_LIMIT),
                OP_LOGE(opName_, "k_block_num_rate_large only supports %f, but got %f.", K_BLOCK_NUM_RATE_LARGE_LIMIT,
                        *opParamInfo_.kBlockNumRateLarge),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.kBlockNumBiasLarge != K_BLOCK_NUM_BIAS_LARGE_LIMIT,
                OP_LOGE(opName_, "k_block_num_bias_large only supports %u, but got %ld.", K_BLOCK_NUM_BIAS_LARGE_LIMIT,
                        *opParamInfo_.kBlockNumBiasLarge),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.topkScorePrecision != TOPK_SCORE_PRECISION_UINT32 &&
                    *opParamInfo_.topkScorePrecision != TOPK_SCORE_PRECISION_UINT16,
                OP_LOGE(opName_, "topk_score_precision only supports 1(uint32) or 2(uint16), but got %ld.",
                        *opParamInfo_.topkScorePrecision),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.qflat.shape == nullptr, OP_LOGE(opName_, "shape of qflat is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qflat.desc == nullptr, OP_LOGE(opName_, "desc of qflat is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kflat.shape == nullptr, OP_LOGE(opName_, "shape of kflat is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kflat.desc == nullptr, OP_LOGE(opName_, "desc of kflat is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vbias.shape == nullptr, OP_LOGE(opName_, "shape of vbias is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vbias.desc == nullptr, OP_LOGE(opName_, "desc of vbias is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qSeqLens.shape == nullptr, OP_LOGE(opName_, "shape of q_seq_lens is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qSeqLens.desc == nullptr, OP_LOGE(opName_, "desc of q_seq_lens is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvSeqLens.shape == nullptr, OP_LOGE(opName_, "shape of kv_seq_lens is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvSeqLens.desc == nullptr, OP_LOGE(opName_, "desc of kv_seq_lens is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.numPromptTokens.shape == nullptr,
                OP_LOGE(opName_, "shape of num_prompt_tokens is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.numPromptTokens.desc == nullptr, OP_LOGE(opName_, "desc of num_prompt_tokens is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.shape == nullptr, OP_LOGE(opName_, "shape of metadata is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.desc == nullptr, OP_LOGE(opName_, "desc of metadata is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndicesOut.shape == nullptr, OP_LOGE(opName_, "shape of sparse_indices is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndicesOut.desc == nullptr, OP_LOGE(opName_, "desc of sparse_indices is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseSeqLenOut.shape == nullptr, OP_LOGE(opName_, "shape of sparse_seq_len is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseSeqLenOut.desc == nullptr, OP_LOGE(opName_, "desc of sparse_seq_len is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.causal == nullptr, OP_LOGE(opName_, "attr causal is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.stemBlockSize == nullptr, OP_LOGE(opName_, "attr stem_block_size is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.stemStride == nullptr, OP_LOGE(opName_, "attr stem_stride is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.alpha == nullptr, OP_LOGE(opName_, "attr alpha is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.initialBlocks == nullptr, OP_LOGE(opName_, "attr initial_blocks is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.windowSize == nullptr, OP_LOGE(opName_, "attr window_size is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumRateMedium == nullptr,
                OP_LOGE(opName_, "attr k_block_num_rate_medium is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumBiasMedium == nullptr,
                OP_LOGE(opName_, "attr k_block_num_bias_medium is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumRateLarge == nullptr, OP_LOGE(opName_, "attr k_block_num_rate_large is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumBiasLarge == nullptr, OP_LOGE(opName_, "attr k_block_num_bias_large is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.topkScorePrecision == nullptr, OP_LOGE(opName_, "attr topk_score_precision is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.qflat.desc->GetDataType();
    inputKType_ = opParamInfo_.kflat.desc->GetDataType();
    vbiasType_ = opParamInfo_.vbias.desc->GetDataType();
    outputType_ = opParamInfo_.sparseIndicesOut.desc->GetDataType();
    seqLenType_ = opParamInfo_.qSeqLens.desc->GetDataType();
    metadataType_ = opParamInfo_.metadata.desc->GetDataType();

    OP_CHECK_IF(inputQType_ != ge::DT_BF16 || inputKType_ != ge::DT_BF16,
                OP_LOGE(opName_, "qflat and kflat only support bfloat16."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputQType_ != inputKType_, OP_LOGE(opName_, "qflat and kflat dtype should be same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vbiasType_ != ge::DT_FLOAT, OP_LOGE(opName_, "vbias only support float32."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqLenType_ != ge::DT_INT32 || opParamInfo_.kvSeqLens.desc->GetDataType() != ge::DT_INT32 ||
                    opParamInfo_.numPromptTokens.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE(opName_, "q_seq_lens, kv_seq_lens and num_prompt_tokens only support int32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataType_ != ge::DT_INT32, OP_LOGE(opName_, "metadata only support int32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputType_ != ge::DT_INT32 || opParamInfo_.sparseSeqLenOut.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE(opName_, "sparse_indices and sparse_seq_len only support int32."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::CheckShapeDim()
{
    OP_CHECK_IF(opParamInfo_.qflat.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE(opName_, "qflat dim num should be 4."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kflat.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE(opName_, "kflat dim num should be 4."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vbias.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(opName_, "vbias dim num should be 3."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qSeqLens.shape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(opName_, "q_seq_lens dim num should be 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvSeqLens.shape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(opName_, "kv_seq_lens dim num should be 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.numPromptTokens.shape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(opName_, "num_prompt_tokens dim num should be 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.shape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(opName_, "metadata dim num should be 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndicesOut.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE(opName_, "sparse_indices dim num should be 4."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseSeqLenOut.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(opName_, "sparse_seq_len dim num should be 3."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::GetBaseShapeInfo()
{
    bSize_ = static_cast<uint32_t>(opParamInfo_.qflat.shape->GetStorageShape().GetDim(DIM_IDX_ZERO));
    qHeadNum_ = static_cast<uint32_t>(opParamInfo_.qflat.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    maxQb_ = static_cast<uint32_t>(opParamInfo_.qflat.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    headDim_ = static_cast<uint32_t>(opParamInfo_.qflat.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
    kvHeadNum_ = static_cast<uint32_t>(opParamInfo_.kflat.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    maxKb_ = static_cast<uint32_t>(opParamInfo_.kflat.shape->GetStorageShape().GetDim(DIM_IDX_TWO));

    OP_CHECK_IF(bSize_ == 0 || maxQb_ == 0 || maxKb_ == 0, OP_LOGE(opName_, "shape dims should be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedQHeadNum(qHeadNum_), OP_LOGE(opName_, "q_heads only support 32 or 64."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedKvHeadNum(kvHeadNum_), OP_LOGE(opName_, "kv_heads only support 2, 4 or 8."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qHeadNum_ % kvHeadNum_ != 0, OP_LOGE(opName_, "q_heads should be divisible by kv_heads."),
                return ge::GRAPH_FAILED);
    gSize_ = qHeadNum_ / kvHeadNum_;
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT, OP_LOGE(opName_, "qflat last dim only support %u.", HEAD_DIM_LIMIT),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::ValidateInputShapesMatch()
{
    const auto &qShape = opParamInfo_.qflat.shape->GetStorageShape();
    const auto &kShape = opParamInfo_.kflat.shape->GetStorageShape();
    const auto &vShape = opParamInfo_.vbias.shape->GetStorageShape();
    const auto &qSeqShape = opParamInfo_.qSeqLens.shape->GetStorageShape();
    const auto &kvSeqShape = opParamInfo_.kvSeqLens.shape->GetStorageShape();
    const auto &numPromptShape = opParamInfo_.numPromptTokens.shape->GetStorageShape();
    const auto &metadataShape = opParamInfo_.metadata.shape->GetStorageShape();
    const auto &sparseIndicesShape = opParamInfo_.sparseIndicesOut.shape->GetStorageShape();
    const auto &sparseSeqLenShape = opParamInfo_.sparseSeqLenOut.shape->GetStorageShape();

    OP_CHECK_IF(kShape.GetDim(DIM_IDX_ZERO) != bSize_, OP_LOGE(opName_, "qflat and kflat batch should be same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kShape.GetDim(DIM_IDX_THREE) != headDim_, OP_LOGE(opName_, "qflat and kflat last dim should match."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDim(DIM_IDX_ZERO) != bSize_ || vShape.GetDim(DIM_IDX_ONE) != kvHeadNum_ ||
                    vShape.GetDim(DIM_IDX_TWO) != maxKb_,
                OP_LOGE(opName_, "vbias shape should be [batch, kv_heads, max_Kb]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qSeqShape.GetDim(DIM_IDX_ZERO) != bSize_ || kvSeqShape.GetDim(DIM_IDX_ZERO) != bSize_ ||
                    numPromptShape.GetDim(DIM_IDX_ZERO) != bSize_,
                OP_LOGE(opName_, "seq length input shape should be [batch]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataShape.GetDim(DIM_IDX_ZERO) != METADATA_LIMIT,
                OP_LOGE(opName_, "metadata shape dim0 should be %u.", METADATA_LIMIT), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sparseIndicesShape.GetDim(DIM_IDX_ZERO) != bSize_ || sparseIndicesShape.GetDim(DIM_IDX_ONE) != qHeadNum_ ||
            sparseIndicesShape.GetDim(DIM_IDX_TWO) != maxQb_ || sparseIndicesShape.GetDim(DIM_IDX_THREE) != maxKb_,
        OP_LOGE(opName_, "sparse_indices shape should be [batch, q_heads, max_Qb, max_Kb]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(sparseSeqLenShape.GetDim(DIM_IDX_ZERO) != bSize_ ||
                    sparseSeqLenShape.GetDim(DIM_IDX_ONE) != qHeadNum_ ||
                    sparseSeqLenShape.GetDim(DIM_IDX_TWO) != maxQb_,
                OP_LOGE(opName_, "sparse_seq_len shape should be [batch, q_heads, max_Qb]."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void StemIndexerInfoParser::GenerateInfo(StemIndexerTilingInfo &stemInfo)
{
    stemInfo.opName = opName_;
    stemInfo.platformInfo = platformInfo_;
    stemInfo.opParamInfo = opParamInfo_;
    stemInfo.socVersion = socVersion_;
    stemInfo.bSize = bSize_;
    stemInfo.qHeadNum = qHeadNum_;
    stemInfo.kvHeadNum = kvHeadNum_;
    stemInfo.gSize = gSize_;
    stemInfo.maxQb = maxQb_;
    stemInfo.maxKb = maxKb_;
    stemInfo.headDim = headDim_;
    stemInfo.causal = *opParamInfo_.causal;
    stemInfo.stemBlockSize = static_cast<uint32_t>(*opParamInfo_.stemBlockSize);
    stemInfo.stemStride = static_cast<uint32_t>(*opParamInfo_.stemStride);
    stemInfo.alpha = *opParamInfo_.alpha;
    stemInfo.initialBlocks = static_cast<uint32_t>(*opParamInfo_.initialBlocks);
    stemInfo.windowSize = static_cast<uint32_t>(*opParamInfo_.windowSize);
    stemInfo.kBlockNumRateMedium = *opParamInfo_.kBlockNumRateMedium;
    stemInfo.kBlockNumBiasMedium = static_cast<uint32_t>(*opParamInfo_.kBlockNumBiasMedium);
    stemInfo.kBlockNumRateLarge = *opParamInfo_.kBlockNumRateLarge;
    stemInfo.kBlockNumBiasLarge = static_cast<uint32_t>(*opParamInfo_.kBlockNumBiasLarge);
    stemInfo.topkScorePrecision = static_cast<uint32_t>(*opParamInfo_.topkScorePrecision);
    uint32_t stemRepTokens = stemInfo.stemBlockSize / stemInfo.stemStride;
    stemInfo.rSquare = 1.0f / static_cast<float>(stemRepTokens * stemRepTokens);
    stemInfo.inputQType = inputQType_;
    stemInfo.inputKType = inputKType_;
    stemInfo.outputType = outputType_;
}

ge::graphStatus StemIndexerInfoParser::ParseAndCheck(StemIndexerTilingInfo &stemInfo)
{
    if (GetOpName() != ge::GRAPH_SUCCESS || GetNpuInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GetInputParaInfo();
    GetOutputParaInfo();
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || GetAndCheckAttrParaInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetAndCheckInOutDataType() != ge::GRAPH_SUCCESS || CheckShapeDim() != ge::GRAPH_SUCCESS ||
        GetBaseShapeInfo() != ge::GRAPH_SUCCESS || ValidateInputShapesMatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(stemInfo);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForStemIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerTiling::DoTiling(const StemIndexerTilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t FP32_BYTES = 4;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += STEM_M_BASE_SIZE * STEM_S2_BASE_SIZE * FP32_BYTES * DOUBLE_BUFFER * aicNum;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_qHeadNum(tilingInfo->qHeadNum);
    tilingData_.set_kvHeadNum(tilingInfo->kvHeadNum);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_maxQb(tilingInfo->maxQb);
    tilingData_.set_maxKb(tilingInfo->maxKb);
    tilingData_.set_headDim(tilingInfo->headDim);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_causal(static_cast<uint32_t>(tilingInfo->causal));
    tilingData_.set_stemBlockSize(tilingInfo->stemBlockSize);
    tilingData_.set_stemStride(tilingInfo->stemStride);
    tilingData_.set_initialBlocks(tilingInfo->initialBlocks);
    tilingData_.set_windowSize(tilingInfo->windowSize);
    tilingData_.set_mBaseSize(STEM_M_BASE_SIZE);
    tilingData_.set_s2BaseSize(STEM_S2_BASE_SIZE);
    tilingData_.set_rSquare(tilingInfo->rSquare);
    tilingData_.set_alpha(tilingInfo->alpha);
    tilingData_.set_kBlockNumRateMedium(tilingInfo->kBlockNumRateMedium);
    tilingData_.set_kBlockNumBiasMedium(tilingInfo->kBlockNumBiasMedium);
    tilingData_.set_kBlockNumRateLarge(tilingInfo->kBlockNumRateLarge);
    tilingData_.set_kBlockNumBiasLarge(tilingInfo->kBlockNumBiasLarge);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t causal = static_cast<uint32_t>(tilingInfo->causal);
    uint32_t topkScorePrecision = tilingInfo->topkScorePrecision;
    uint64_t tilingKey = GET_TPL_TILING_KEY(inputQType, inputKType, outputType, causal, topkScorePrecision);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForStemIndexer(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("StemIndexer", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    StemIndexerTilingInfo stemInfo;
    StemIndexerInfoParser stemInfoParser(context);
    if (stemInfoParser.ParseAndCheck(stemInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    StemIndexerTiling stemTiling(context);
    return stemTiling.DoTiling(&stemInfo);
}

IMPL_OP_OPTILING(StemIndexer)
    .Tiling(TilingForStemIndexer)
    .TilingParse<StemIndexerCompileInfo>(TilingPrepareForStemIndexer);
} // namespace optiling
