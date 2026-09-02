/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_tiling.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_H_
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_H_

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
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHT_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
// Outputs Index
constexpr uint32_t SOFTMAX_MAX_INDEX = 0;
constexpr uint32_t SOFTMAX_SUM_INDEX = 1;
// Attributes Index
constexpr uint32_t ATTR_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKENS_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKENS_INDEX = 3;
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_ZERO = 0;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t N2_SIZE_LIMIT = 1;
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_MODE_LOWER = 3;

constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
constexpr uint32_t DOUBLE_BUFFER = 2;     // 双Buffer
constexpr uint32_t M_BASE_SIZE = 512;     // m轴基本块大小
constexpr uint32_t S2_BASE_SIZE = 512;    // S2轴基本块大小
constexpr uint32_t S1_BASE_SIZE = 8;      // S1轴基本块大小
constexpr uint32_t MAX_SEQ_LENGTH = 1024 * 1024;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(DenseLISoftmaxLseTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint64_t, s1Size)
TILING_DATA_FIELD_DEF(uint64_t, s2Size)
TILING_DATA_FIELD_DEF(uint64_t, storageS1Size)
TILING_DATA_FIELD_DEF(uint64_t, storageS2Size)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(DenseLightningIndexerSoftmaxLse, DenseLISoftmaxLseTilingData)

// -----------算子CompileInfo定义-------------------
struct DenseLISoftmaxLseCompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct DenseLISoftmaxLseParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    TilingRequiredParaInfo softmaxMaxOut = {nullptr, nullptr};
    TilingRequiredParaInfo softmaxSumOut = {nullptr, nullptr};

    const char *layOut = nullptr;
    const int32_t *sparseMode = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
};

// -----------算子Tiling入参信息类---------------
class DenseLISoftmaxLseTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    DenseLISoftmaxLseParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint64_t s1Size = 0;
    uint64_t s2Size = 0;
    uint64_t storageS1Size = 0;
    uint64_t storageS2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    bool returnValue = false;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType softmaxMaxOutType = ge::DT_FLOAT;
    ge::DataType softmaxSumOutType = ge::DT_FLOAT;
    // Layout
    DataLayout layout = DataLayout::BSND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class DenseLISoftmaxLseInfoParser {
public:
    explicit DenseLISoftmaxLseInfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~DenseLISoftmaxLseInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatchBsnd();
    ge::graphStatus ValidateInputShapesMatchTnd();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetS2Size();
    ge::graphStatus CheckSeqLenLimit();
    ge::graphStatus GetMaxSeqLenFromTensor(const gert::Tensor *tensor, uint64_t &maxSeqLen, bool isAccumSeq);
    ge::graphStatus GetStorageSeqSize();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    void GenerateInfo(DenseLISoftmaxLseTilingInfo &denseLISoftmaxTilingInfo);
    ge::graphStatus ParseAndCheck(DenseLISoftmaxLseTilingInfo &denseLISoftmaxTilingInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    DenseLISoftmaxLseParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint64_t s1Size_ = 0;
    uint64_t s2Size_ = 0;
    uint64_t storageS1Size_ = 0;
    uint64_t storageS2Size_ = 0;
    uint32_t headDim_ = 0;
    // Layout
    DataLayout layout_ = DataLayout::BSND;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType softmaxMaxOutType_ = ge::DT_FLOAT;
    ge::DataType softmaxSumOutType_ = ge::DT_FLOAT;
};

// ---------------算子Tiling类---------------
class DenseLightningIndexerSoftmaxLseTiling {
public:
    explicit DenseLightningIndexerSoftmaxLseTiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoTiling(DenseLISoftmaxLseTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    DenseLISoftmaxLseTilingData tilingData_;
};
} // namespace optiling

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_H_
