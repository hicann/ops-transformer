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

#include <cmath>
#include <string>

#include "../op_kernel/stem_indexer_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
__attribute__((noinline)) bool StemIndexerInfoParser::IsFloatEqual(float lhs, float rhs) const
{
    return (lhs > rhs - ATTR_FLOAT_EPS) && (lhs < rhs + ATTR_FLOAT_EPS);
}

__attribute__((noinline)) bool StemIndexerInfoParser::IsSupportedQHeadNum(uint32_t qHeadNum) const
{
    return qHeadNum == Q_HEAD_NUM_32 || qHeadNum == Q_HEAD_NUM_64;
}

__attribute__((noinline)) bool StemIndexerInfoParser::IsSupportedKvHeadNum(uint32_t kvHeadNum) const
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
    opParamInfo_.numPromptTokens.tensor = context_->GetOptionalInputTensor(NUM_PROMPT_TOKENS_INDEX);
    opParamInfo_.numPromptTokens.desc = context_->GetOptionalInputDesc(NUM_PROMPT_TOKENS_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
    useKvSeqLensAsNumPrompt_ = opParamInfo_.numPromptTokens.tensor == nullptr;
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

    OP_CHECK_IF(
        *opParamInfo_.stemBlockSize != STEM_BLOCK_SIZE_LIMIT,
        OP_LOGE_WITH_INVALID_ATTR(opName_, "stem_block_size", std::to_string(*opParamInfo_.stemBlockSize).c_str(),
                                  std::to_string(STEM_BLOCK_SIZE_LIMIT).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.stemStride != STEM_STRIDE_LIMIT,
                OP_LOGE_WITH_INVALID_ATTR(opName_, "stem_stride", std::to_string(*opParamInfo_.stemStride).c_str(),
                                          std::to_string(STEM_STRIDE_LIMIT).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *opParamInfo_.initialBlocks != INITIAL_BLOCKS_LIMIT,
        OP_LOGE_WITH_INVALID_ATTR(opName_, "initial_blocks", std::to_string(*opParamInfo_.initialBlocks).c_str(),
                                  std::to_string(INITIAL_BLOCKS_LIMIT).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.windowSize != WINDOW_SIZE_LIMIT,
                OP_LOGE_WITH_INVALID_ATTR(opName_, "window_size", std::to_string(*opParamInfo_.windowSize).c_str(),
                                          std::to_string(WINDOW_SIZE_LIMIT).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        std::isnan(*opParamInfo_.alpha) || (*opParamInfo_.alpha <= ALPHA_MIN) || (*opParamInfo_.alpha > ALPHA_MAX),
        OP_LOGE_WITH_INVALID_ATTR(opName_, "alpha", std::to_string(*opParamInfo_.alpha).c_str(),
                                  ("(" + std::to_string(ALPHA_MIN) + ", " + std::to_string(ALPHA_MAX) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsFloatEqual(*opParamInfo_.kBlockNumRateMedium, K_BLOCK_NUM_RATE_MEDIUM_LIMIT),
                OP_LOGE_WITH_INVALID_ATTR(opName_, "k_block_num_rate_medium",
                                          std::to_string(*opParamInfo_.kBlockNumRateMedium).c_str(),
                                          std::to_string(K_BLOCK_NUM_RATE_MEDIUM_LIMIT).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.kBlockNumBiasMedium != K_BLOCK_NUM_BIAS_MEDIUM_LIMIT,
                OP_LOGE_WITH_INVALID_ATTR(opName_, "k_block_num_bias_medium",
                                          std::to_string(*opParamInfo_.kBlockNumBiasMedium).c_str(),
                                          std::to_string(K_BLOCK_NUM_BIAS_MEDIUM_LIMIT).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsFloatEqual(*opParamInfo_.kBlockNumRateLarge, K_BLOCK_NUM_RATE_LARGE_LIMIT),
                OP_LOGE_WITH_INVALID_ATTR(opName_, "k_block_num_rate_large",
                                          std::to_string(*opParamInfo_.kBlockNumRateLarge).c_str(),
                                          std::to_string(K_BLOCK_NUM_RATE_LARGE_LIMIT).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.kBlockNumBiasLarge != K_BLOCK_NUM_BIAS_LARGE_LIMIT,
                OP_LOGE_WITH_INVALID_ATTR(opName_, "k_block_num_bias_large",
                                          std::to_string(*opParamInfo_.kBlockNumBiasLarge).c_str(),
                                          std::to_string(K_BLOCK_NUM_BIAS_LARGE_LIMIT).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.topkScorePrecision != TOPK_SCORE_PRECISION_UINT32 &&
                    *opParamInfo_.topkScorePrecision != TOPK_SCORE_PRECISION_UINT16,
                OP_LOGE_WITH_INVALID_ATTR(opName_, "topk_score_precision",
                                          std::to_string(*opParamInfo_.topkScorePrecision).c_str(),
                                          (std::to_string(TOPK_SCORE_PRECISION_UINT32) + "(uint32) or " +
                                           std::to_string(TOPK_SCORE_PRECISION_UINT16) + "(uint16)")
                                              .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.qflat.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor qflat"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qflat.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor qflat"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kflat.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor kflat"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kflat.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor kflat"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vbias.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor vbias"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vbias.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor vbias"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qSeqLens.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor q_seq_lens"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qSeqLens.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor q_seq_lens"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvSeqLens.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor kv_seq_lens"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvSeqLens.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor kv_seq_lens"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.numPromptTokens.tensor != nullptr && opParamInfo_.numPromptTokens.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor num_prompt_tokens"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr, OP_LOGE(opName_, "metadata is required but is null!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "metadata", "metadata desc cannot be empty"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndicesOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor sparse_indices"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndicesOut.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor sparse_indices"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseSeqLenOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of tensor sparse_seq_len"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseSeqLenOut.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of tensor sparse_seq_len"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.causal == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr causal"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.stemBlockSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr stem_block_size"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.stemStride == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr stem_stride"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.alpha == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr alpha"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.initialBlocks == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr initial_blocks"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.windowSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attr window_size"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumRateMedium == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "attr k_block_num_rate_medium"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumBiasMedium == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "attr k_block_num_bias_medium"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumRateLarge == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "attr k_block_num_rate_large"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kBlockNumBiasLarge == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "attr k_block_num_bias_large"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.topkScorePrecision == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(opName_, "attr topk_score_precision"), return ge::GRAPH_FAILED);
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
    const auto promptDesc = useKvSeqLensAsNumPrompt_ ? opParamInfo_.kvSeqLens.desc : opParamInfo_.numPromptTokens.desc;
    const ge::DataType numPromptType = promptDesc->GetDataType();

    OP_CHECK_IF(inputQType_ != ge::DT_BF16 || inputKType_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "qflat and kflat",
                    (Ops::Base::ToString(inputQType_) + " and " + Ops::Base::ToString(inputKType_)).c_str(),
                    "The dtypes of input qflat and kflat should be BF16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vbiasType_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(opName_, "vbias", Ops::Base::ToString(vbiasType_).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        seqLenType_ != ge::DT_INT32 || opParamInfo_.kvSeqLens.desc->GetDataType() != ge::DT_INT32 ||
            numPromptType != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "q_seq_lens, kv_seq_lens and num_prompt_tokens",
            (Ops::Base::ToString(seqLenType_) + ", " + Ops::Base::ToString(opParamInfo_.kvSeqLens.desc->GetDataType()) +
             " and " + Ops::Base::ToString(numPromptType))
                .c_str(),
            "The dtypes of input q_seq_lens, kv_seq_lens and num_prompt_tokens should be INT32"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(metadataType_ != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(opName_, "metadata", Ops::Base::ToString(metadataType_).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputType_ != ge::DT_INT32 || opParamInfo_.sparseSeqLenOut.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "sparse_indices and sparse_seq_len",
                    (Ops::Base::ToString(outputType_) + " and " +
                     Ops::Base::ToString(opParamInfo_.sparseSeqLenOut.desc->GetDataType()))
                        .c_str(),
                    "The dtypes of output sparse_indices and sparse_seq_len should be INT32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemIndexerInfoParser::CheckShapeDim()
{
    OP_CHECK_IF(opParamInfo_.qflat.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "qflat",
                    (std::to_string(opParamInfo_.qflat.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kflat.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "kflat",
                    (std::to_string(opParamInfo_.kflat.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.vbias.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "vbias",
                    (std::to_string(opParamInfo_.vbias.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.qSeqLens.shape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "q_seq_lens",
                    (std::to_string(opParamInfo_.qSeqLens.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kvSeqLens.shape->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "kv_seq_lens",
                    (std::to_string(opParamInfo_.kvSeqLens.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !useKvSeqLensAsNumPrompt_ && opParamInfo_.numPromptTokens.tensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName_, "num_prompt_tokens",
            (std::to_string(opParamInfo_.numPromptTokens.tensor->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.tensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "metadata",
                    (std::to_string(opParamInfo_.metadata.tensor->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.sparseIndicesOut.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName_, "sparse_indices",
            (std::to_string(opParamInfo_.sparseIndicesOut.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.sparseSeqLenOut.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName_, "sparse_seq_len",
            (std::to_string(opParamInfo_.sparseSeqLenOut.shape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
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

    OP_CHECK_IF(bSize_ == 0 || maxQb_ == 0 || maxKb_ == 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "qflat and kflat",
                    (Ops::Base::ToString(opParamInfo_.qflat.shape->GetStorageShape()) + " and " +
                     Ops::Base::ToString(opParamInfo_.kflat.shape->GetStorageShape()))
                        .c_str(),
                    "The batch, maxQb and maxKb dimensions should be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        bSize_ > BATCH_SIZE_LIMIT,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "qflat", Ops::Base::ToString(opParamInfo_.qflat.shape->GetStorageShape()).c_str(),
            "The batch dimension of input qflat should be less than or equal to " + std::to_string(BATCH_SIZE_LIMIT)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedQHeadNum(qHeadNum_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "qflat", Ops::Base::ToString(opParamInfo_.qflat.shape->GetStorageShape()).c_str(),
                    "The q_heads dimension of input qflat should be " + std::to_string(Q_HEAD_NUM_32) + " or " +
                        std::to_string(Q_HEAD_NUM_64)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedKvHeadNum(kvHeadNum_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "kflat", Ops::Base::ToString(opParamInfo_.kflat.shape->GetStorageShape()).c_str(),
                    "The kv_heads dimension of input kflat should be " + std::to_string(KV_HEAD_NUM_2) + ", " +
                        std::to_string(KV_HEAD_NUM_4) + " or " + std::to_string(KV_HEAD_NUM_8)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qHeadNum_ % kvHeadNum_ != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "qflat and kflat",
                    (Ops::Base::ToString(opParamInfo_.qflat.shape->GetStorageShape()) + " and " +
                     Ops::Base::ToString(opParamInfo_.kflat.shape->GetStorageShape()))
                        .c_str(),
                    "The q_heads dimension of input qflat should be divisible by the kv_heads dimension of input "
                    "kflat"),
                return ge::GRAPH_FAILED);
    gSize_ = qHeadNum_ / kvHeadNum_;
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "qflat", Ops::Base::ToString(opParamInfo_.qflat.shape->GetStorageShape()).c_str(),
                    "The last dimension of input qflat should be " + std::to_string(HEAD_DIM_LIMIT)),
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
    const auto &metadataShape = opParamInfo_.metadata.tensor->GetStorageShape();
    const auto &sparseIndicesShape = opParamInfo_.sparseIndicesOut.shape->GetStorageShape();
    const auto &sparseSeqLenShape = opParamInfo_.sparseSeqLenOut.shape->GetStorageShape();
    const std::string qkShapeStr = Ops::Base::ToString(qShape) + " and " + Ops::Base::ToString(kShape);

    OP_CHECK_IF(
        kShape.GetDim(DIM_IDX_ZERO) != bSize_,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "qflat and kflat", qkShapeStr.c_str(),
                                               "The batch dimension of input qflat should be the same as input kflat"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        kShape.GetDim(DIM_IDX_THREE) != headDim_,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "qflat and kflat", qkShapeStr.c_str(),
                                               "The last dimension of input qflat should be the same as input kflat"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDim(DIM_IDX_ZERO) != bSize_ || vShape.GetDim(DIM_IDX_ONE) != kvHeadNum_ ||
                    vShape.GetDim(DIM_IDX_TWO) != maxKb_,
                OP_LOGE_FOR_INVALID_SHAPE(opName_, "vbias", Ops::Base::ToString(vShape).c_str(),
                                          ("[" + std::to_string(bSize_) + "," + std::to_string(kvHeadNum_) + "," +
                                           std::to_string(maxKb_) + "]")
                                              .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qSeqShape.GetDim(DIM_IDX_ZERO) != bSize_ || kvSeqShape.GetDim(DIM_IDX_ZERO) != bSize_,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "q_seq_lens and kv_seq_lens",
            (Ops::Base::ToString(qSeqShape) + " and " + Ops::Base::ToString(kvSeqShape)).c_str(),
            ("The shapes of input q_seq_lens and kv_seq_lens should both be [" + std::to_string(bSize_) + "]").c_str()),
        return ge::GRAPH_FAILED);
    if (!useKvSeqLensAsNumPrompt_) {
        const auto &numPromptShape = opParamInfo_.numPromptTokens.tensor->GetStorageShape();
        OP_CHECK_IF(numPromptShape.GetDim(DIM_IDX_ZERO) != bSize_,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "num_prompt_tokens", Ops::Base::ToString(numPromptShape).c_str(),
                                              ("[" + std::to_string(bSize_) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    }
    uint64_t metadataCapacity = CalcStemIndexerMetadataCapacity(bSize_, kvHeadNum_);
    OP_CHECK_IF(metadataShape.GetDim(DIM_IDX_ZERO) < static_cast<int64_t>(metadataCapacity),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "metadata", Ops::Base::ToString(metadataShape).c_str(),
                    ("The first dimension of input metadata should be at least " + std::to_string(metadataCapacity) +
                     " when batch is " + std::to_string(bSize_) + " and kv_heads is " + std::to_string(kvHeadNum_))
                        .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sparseIndicesShape.GetDim(DIM_IDX_ZERO) != bSize_ || sparseIndicesShape.GetDim(DIM_IDX_ONE) != qHeadNum_ ||
            sparseIndicesShape.GetDim(DIM_IDX_TWO) != maxQb_ || sparseIndicesShape.GetDim(DIM_IDX_THREE) != maxKb_,
        OP_LOGE_FOR_INVALID_SHAPE(opName_, "sparse_indices", Ops::Base::ToString(sparseIndicesShape).c_str(),
                                  ("[" + std::to_string(bSize_) + "," + std::to_string(qHeadNum_) + "," +
                                   std::to_string(maxQb_) + "," + std::to_string(maxKb_) + "]")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sparseSeqLenShape.GetDim(DIM_IDX_ZERO) != bSize_ || sparseSeqLenShape.GetDim(DIM_IDX_ONE) != qHeadNum_ ||
            sparseSeqLenShape.GetDim(DIM_IDX_TWO) != maxQb_,
        OP_LOGE_FOR_INVALID_SHAPE(
            opName_, "sparse_seq_len", Ops::Base::ToString(sparseSeqLenShape).c_str(),
            ("[" + std::to_string(bSize_) + "," + std::to_string(qHeadNum_) + "," + std::to_string(maxQb_) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);
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
    stemInfo.useKvSeqLensAsNumPrompt = useKvSeqLensAsNumPrompt_;
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

    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
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
    tilingData_.set_useKvSeqLensAsNumPrompt(static_cast<uint32_t>(tilingInfo->useKvSeqLensAsNumPrompt));
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
