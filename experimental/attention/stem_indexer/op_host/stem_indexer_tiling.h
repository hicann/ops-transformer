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
 * \file stem_indexer_tiling.h
 * \brief
 */

#ifndef STEM_INDEXER_TILING_H
#define STEM_INDEXER_TILING_H

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QFLAT_INDEX = 0;
constexpr uint32_t KFLAT_INDEX = 1;
constexpr uint32_t VBIAS_INDEX = 2;
constexpr uint32_t Q_SEQ_LENS_INDEX = 3;
constexpr uint32_t KV_SEQ_LENS_INDEX = 4;
constexpr uint32_t NUM_PROMPT_TOKENS_INDEX = 5;
constexpr uint32_t METADATA_INDEX = 6;
constexpr uint32_t SPARSE_INDICES_INDEX = 0;
constexpr uint32_t SPARSE_SEQ_LEN_INDEX = 1;

constexpr uint32_t ATTR_CAUSAL_INDEX = 0;
constexpr uint32_t ATTR_STEM_BLOCK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_STEM_STRIDE_INDEX = 2;
constexpr uint32_t ATTR_ALPHA_INDEX = 3;
constexpr uint32_t ATTR_INITIAL_BLOCKS_INDEX = 4;
constexpr uint32_t ATTR_WINDOW_SIZE_INDEX = 5;
constexpr uint32_t ATTR_K_BLOCK_NUM_RATE_MEDIUM_INDEX = 6;
constexpr uint32_t ATTR_K_BLOCK_NUM_BIAS_MEDIUM_INDEX = 7;
constexpr uint32_t ATTR_K_BLOCK_NUM_RATE_LARGE_INDEX = 8;
constexpr uint32_t ATTR_K_BLOCK_NUM_BIAS_LARGE_INDEX = 9;
constexpr uint32_t ATTR_TOPK_SCORE_PRECISION_INDEX = 10;

constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t STEM_BLOCK_SIZE_LIMIT = 128;
constexpr uint32_t STEM_STRIDE_LIMIT = 16;
constexpr uint32_t HEAD_DIM_LIMIT = 2048;
constexpr uint32_t BATCH_SIZE_LIMIT = 65536U;
constexpr uint32_t INITIAL_BLOCKS_LIMIT = 4;
constexpr uint32_t WINDOW_SIZE_LIMIT = 4;
constexpr uint64_t STEM_INDEXER_METADATA_HEADER_ELEMS = 16U;
constexpr uint64_t STEM_INDEXER_METADATA_CORE_STRIDE = 16U;
constexpr uint64_t STEM_INDEXER_METADATA_AIC_CORE_NUM = 36U;
constexpr uint64_t STEM_INDEXER_METADATA_AIV_CORE_NUM = 72U;
constexpr uint64_t STEM_INDEXER_METADATA_ALIGN_ELEMS = 4096U;
constexpr uint32_t Q_HEAD_NUM_32 = 32;
constexpr uint32_t Q_HEAD_NUM_64 = 64;
constexpr uint32_t KV_HEAD_NUM_2 = 2;
constexpr uint32_t KV_HEAD_NUM_4 = 4;
constexpr uint32_t KV_HEAD_NUM_8 = 8;
constexpr uint32_t STEM_M_BASE_SIZE = 64;
constexpr uint32_t STEM_S2_BASE_SIZE = 256;
constexpr float ALPHA_MIN = 0.0f;
constexpr float ALPHA_MAX = 1.0f;
constexpr float ATTR_FLOAT_EPS = 0.000001f;
constexpr float K_BLOCK_NUM_RATE_MEDIUM_LIMIT = 0.2f;
constexpr float K_BLOCK_NUM_RATE_LARGE_LIMIT = 0.1f;
constexpr uint32_t K_BLOCK_NUM_BIAS_MEDIUM_LIMIT = 30U;
constexpr uint32_t K_BLOCK_NUM_BIAS_LARGE_LIMIT = 30U;
constexpr int64_t TOPK_SCORE_PRECISION_UINT32 = 1;
constexpr int64_t TOPK_SCORE_PRECISION_UINT16 = 2;

constexpr uint64_t CalcStemIndexerMetadataCapacity(uint64_t batchSize, uint64_t kvHeadNum)
{
    uint64_t maxSectionNum = batchSize * kvHeadNum;
    uint64_t requiredElems = STEM_INDEXER_METADATA_HEADER_ELEMS +
                             maxSectionNum * (STEM_INDEXER_METADATA_AIC_CORE_NUM + STEM_INDEXER_METADATA_AIV_CORE_NUM) *
                                 STEM_INDEXER_METADATA_CORE_STRIDE;
    return (requiredElems + STEM_INDEXER_METADATA_ALIGN_ELEMS - 1U) / STEM_INDEXER_METADATA_ALIGN_ELEMS *
           STEM_INDEXER_METADATA_ALIGN_ELEMS;
}

BEGIN_TILING_DATA_DEF(StemIndexerTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, maxQb)
TILING_DATA_FIELD_DEF(uint32_t, maxKb)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, causal)
TILING_DATA_FIELD_DEF(uint32_t, stemBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, stemStride)
TILING_DATA_FIELD_DEF(uint32_t, initialBlocks)
TILING_DATA_FIELD_DEF(uint32_t, windowSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
TILING_DATA_FIELD_DEF(float, rSquare)
TILING_DATA_FIELD_DEF(float, alpha)
TILING_DATA_FIELD_DEF(float, kBlockNumRateMedium)
TILING_DATA_FIELD_DEF(uint32_t, kBlockNumBiasMedium)
TILING_DATA_FIELD_DEF(float, kBlockNumRateLarge)
TILING_DATA_FIELD_DEF(uint32_t, kBlockNumBiasLarge)
TILING_DATA_FIELD_DEF(uint32_t, useKvSeqLensAsNumPrompt)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(StemIndexer, StemIndexerTilingData)

struct StemIndexerCompileInfo {};

struct StemIndexerParaInfo {
    TilingRequiredParaInfo qflat = {nullptr, nullptr};
    TilingRequiredParaInfo kflat = {nullptr, nullptr};
    TilingRequiredParaInfo vbias = {nullptr, nullptr};
    TilingRequiredParaInfo qSeqLens = {nullptr, nullptr};
    TilingRequiredParaInfo kvSeqLens = {nullptr, nullptr};
    TilingOptionalParaInfo numPromptTokens = {nullptr, nullptr};
    TilingOptionalParaInfo metadata = {nullptr, nullptr};
    TilingRequiredParaInfo sparseIndicesOut = {nullptr, nullptr};
    TilingRequiredParaInfo sparseSeqLenOut = {nullptr, nullptr};

    const bool *causal = nullptr;
    const int64_t *stemBlockSize = nullptr;
    const int64_t *stemStride = nullptr;
    const float *alpha = nullptr;
    const int64_t *initialBlocks = nullptr;
    const int64_t *windowSize = nullptr;
    const float *kBlockNumRateMedium = nullptr;
    const int64_t *kBlockNumBiasMedium = nullptr;
    const float *kBlockNumRateLarge = nullptr;
    const int64_t *kBlockNumBiasLarge = nullptr;
    const int64_t *topkScorePrecision = nullptr;
};

class StemIndexerTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    StemIndexerParaInfo opParamInfo;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
    uint32_t bSize = 0;
    uint32_t qHeadNum = 0;
    uint32_t kvHeadNum = 0;
    uint32_t gSize = 0;
    uint32_t maxQb = 0;
    uint32_t maxKb = 0;
    uint32_t headDim = 0;
    bool useKvSeqLensAsNumPrompt = false;
    bool causal = true;
    uint32_t stemBlockSize = STEM_BLOCK_SIZE_LIMIT;
    uint32_t stemStride = STEM_STRIDE_LIMIT;
    float alpha = 1.0f;
    uint32_t initialBlocks = INITIAL_BLOCKS_LIMIT;
    uint32_t windowSize = WINDOW_SIZE_LIMIT;
    float kBlockNumRateMedium = K_BLOCK_NUM_RATE_MEDIUM_LIMIT;
    uint32_t kBlockNumBiasMedium = K_BLOCK_NUM_BIAS_MEDIUM_LIMIT;
    float kBlockNumRateLarge = K_BLOCK_NUM_RATE_LARGE_LIMIT;
    uint32_t kBlockNumBiasLarge = K_BLOCK_NUM_BIAS_LARGE_LIMIT;
    uint32_t topkScorePrecision = TOPK_SCORE_PRECISION_UINT32;
    float rSquare = 1.0f / 64.0f;
    ge::DataType inputQType = ge::DT_BF16;
    ge::DataType inputKType = ge::DT_BF16;
    ge::DataType outputType = ge::DT_INT32;
};

class StemIndexerInfoParser {
public:
    explicit StemIndexerInfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~StemIndexerInfoParser() = default;

    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus ParseAndCheck(StemIndexerTilingInfo &stemInfo);
    void GenerateInfo(StemIndexerTilingInfo &stemInfo);

private:
    __attribute__((noinline)) bool IsFloatEqual(float lhs, float rhs) const;
    __attribute__((noinline)) bool IsSupportedQHeadNum(uint32_t qHeadNum) const;
    __attribute__((noinline)) bool IsSupportedKvHeadNum(uint32_t kvHeadNum) const;

private:
    gert::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    StemIndexerParaInfo opParamInfo_;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND950;
    uint32_t bSize_ = 0;
    uint32_t qHeadNum_ = 0;
    uint32_t kvHeadNum_ = 0;
    uint32_t gSize_ = 0;
    uint32_t maxQb_ = 0;
    uint32_t maxKb_ = 0;
    uint32_t headDim_ = 0;
    bool useKvSeqLensAsNumPrompt_ = false;
    ge::DataType inputQType_ = ge::DT_BF16;
    ge::DataType inputKType_ = ge::DT_BF16;
    ge::DataType vbiasType_ = ge::DT_FLOAT;
    ge::DataType outputType_ = ge::DT_INT32;
    ge::DataType seqLenType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
};

class StemIndexerTiling {
public:
    explicit StemIndexerTiling(gert::TilingContext *context)
        : context_(context)
    {}
    ge::graphStatus DoTiling(const StemIndexerTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    StemIndexerTilingData tilingData_;
};
} // namespace optiling

#endif // STEM_INDEXER_TILING_H
