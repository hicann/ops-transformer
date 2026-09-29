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
 * \file sparse_lightning_indexer_tiling.h
 * \brief cloned from lightning_indexer_v2 @Phase1, consumer-mode specialization
 */

#ifndef SPARSE_LIGHTNING_INDEXER_TILING_H_
#define SPARSE_LIGHTNING_INDEXER_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"

namespace optiling {
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index（0-10 与 LIV2 一致）
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 3;
constexpr uint32_t CU_SEQLENS_K_INDEX = 4;
constexpr uint32_t SEQUSED_Q_INDEX = 5;
constexpr uint32_t SEQUSED_K_INDEX = 6;
constexpr uint32_t CMP_RESIDUAL_K_INDEX = 7;
constexpr uint32_t BLOCK_TABLE_INDEX = 8;
constexpr uint32_t OUTPUT_IDX_OFFSET_INDEX = 9;
constexpr uint32_t METADATA_INDEX = 10;
constexpr uint32_t CANDIDATE_TOPK_INDICES_INPUT_INDEX = 11; // REQUIRED（C1）
constexpr uint32_t CANDIDATE_BLOCK_LENGTH_INPUT_INDEX = 12; // OPTIONAL（预留，仅空，C7）
// Outputs Index
constexpr uint32_t LIGHTNING_INDEXER = 0;
constexpr uint32_t LIGHTNING_VALUES = 1;
// Attributes Index
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 5;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;
constexpr uint32_t ATTR_CANDIDATE_BLOCK_SIZE_INDEX = 7; // 本算子无 candidate_topk_blocks 属性（N2）
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_2K = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t TOPK_MAX = 8192;
constexpr uint32_t TOPK_MULTIPLE = 1024;
constexpr uint32_t G_SIZE_LIMIT = 64;
constexpr uint32_t METADATA_LIMIT = 1024;
// ------------------candidate 两级TopK 常量（与 kernel common_arch22.h 保持一致）------------------
// candBlocks 由输入 shape 末维推导（无属性源，N2）：(0, 2048] 且 64 的倍数（与 source 输出宽度同规）
constexpr uint32_t CANDIDATE_TOPK_BLOCKS_FIX = 2048; // 块级 topk 上限（BASE_TOPK 排序/归并结构约束）
constexpr uint32_t CANDIDATE_TOPK_BLOCKS_MULTIPLE = 64;
constexpr uint32_t CANDIDATE_BLOCK_SIZE_DEFAULT = 8;
constexpr uint32_t CANDIDATE_BLOCK_SIZE_MIN = 2; // [MIN, MAX] 内 2 的幂
constexpr uint32_t CANDIDATE_BLOCK_SIZE_MAX = 64;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(SparseLITilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, topk)
TILING_DATA_FIELD_DEF(uint32_t, maxSeqlenQ)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(int64_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, batchSupperFlag)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
// ---- candidate (two-level topk, consumer-only) ----
TILING_DATA_FIELD_DEF(uint32_t, candidateTopkBlocks) // = candidate_topk_indices.shape[-1]，host 推导
TILING_DATA_FIELD_DEF(uint32_t, candidateBlockSize)  // [2,64] 2 的幂，默认 8
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseLightningIndexer, SparseLITilingData)

// -----------算子CompileInfo定义-------------------
struct SparseLICompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct SparseLIParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    TilingOptionalParaInfo cuSeqlensK = {nullptr, nullptr};
    TilingOptionalParaInfo sequsedQ = {nullptr, nullptr};
    TilingOptionalParaInfo sequsedK = {nullptr, nullptr};
    TilingOptionalParaInfo cmpResidualK = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingOptionalParaInfo outputIdxOffset = {nullptr, nullptr};
    TilingOptionalParaInfo metadata = {nullptr, nullptr};
    TilingRequiredParaInfo candidateTopkIndices = {nullptr, nullptr}; // REQUIRED 输入（idx 11）
    TilingOptionalParaInfo candidateBlockLength = {nullptr, nullptr}; // 预留，仅空（idx 12）
    TilingRequiredParaInfo attenOut = {nullptr, nullptr};
    TilingRequiredParaInfo valuesOut = {nullptr, nullptr};

    const char *layOut = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    // 【R15 修复 2026-09-23】GE .Int() 属性统一 int64_t* 读取（仓惯例，与 cmp_ratio 一致；
    // 原 int32_t* 为依赖小端截断的 type-punning）
    const int64_t *maskMode = nullptr;
    const int64_t *topk = nullptr;
    const int64_t *maxSeqlenQ = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const int64_t *cmpRatio = nullptr;
    const int64_t *returnValue = nullptr;
    const int64_t *candidateBlockSize = nullptr;
};

// -----------算子Tiling入参信息类---------------
class SparseLITilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    SparseLIParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t maskMode = 0;
    // Others Flag
    uint32_t topk = 0;
    int32_t maxSeqlenQ = -1;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int64_t cmpRatio = 1;
    uint32_t batchSupperFlag = 0;
    uint32_t returnValue = 0;
    uint32_t keyStride0 = 0;
    std::vector<uint32_t> keyStridesVec;
    // candidate (two-level topk, consumer)：candBlocks 由输入 shape 末维推导
    uint32_t candidateTopkBlocks = 0;
    uint32_t candidateBlockSize = CANDIDATE_BLOCK_SIZE_DEFAULT;

    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BBND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class SparseLIInfoParser {
public:
    explicit SparseLIInfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~SparseLIInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName) const;
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatchQbsnd();
    ge::graphStatus ValidateInputShapesMatchQtnd();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus GetAndCheckCandidateInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus CheckKeyContiguous() const;
    void GenerateInfo(SparseLITilingInfo &liInfo);
    ge::graphStatus ParseAndCheck(SparseLITilingInfo &liInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    SparseLIParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t batchSupperFlag_ = 0;
    // candidate (two-level topk, consumer)：candBlocks 由输入 shape 末维推导（C2/C3）
    uint32_t candBlocks_ = 0;
    // Layout
    DataLayout qLayout_ = DataLayout::BSND;
    DataLayout kLayout_ = DataLayout::PA_BBND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType valuesOutType_ = ge::DT_FLOAT16;
    std::vector<uint32_t> keyStridesVec_;
    std::vector<uint32_t> keyDequantScaleStridesVec_;
};

// ---------------算子Tiling类---------------
class SparseLightningIndexerTiling {
public:
    explicit SparseLightningIndexerTiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoTiling(SparseLITilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    SparseLITilingData tilingData_;
};
} // namespace optiling
#endif // SPARSE_LIGHTNING_INDEXER_TILING_H_
