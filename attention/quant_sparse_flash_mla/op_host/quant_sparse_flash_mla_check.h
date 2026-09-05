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
 * \file quant_sparse_flash_mla_check.h
 * \brief
 */
#ifndef QUANT_SPARSE_FLASH_MLA_CHECK_H
#define QUANT_SPARSE_FLASH_MLA_CHECK_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"

namespace optiling {

const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
const std::string SINKS_NAME = "sinks";

const std::string QUERY_NAME = "query";
const std::string KEY_NAME = "key";
const std::string VALUE_NAME = "value";

const std::string ORI_KV_NAME = "ori_kv";
const std::string CMP_KV_NAME = "cmp_kv";
const std::string ORI_SPARSE_INDICES_NAME = "ori_sparse_indices";
const std::string CMP_SPARSE_INDICES_NAME = "cmp_sparse_indices";
const std::string ATTEN_OUT_NAME = "attention_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string CU_SEQLENS_ORI_KV_NAME = "cu_seqlens_ori_kv";
const std::string CU_SEQLENS_CMP_KV_NAME = "cu_seqlens_cmp_kv";
const std::string SEQUSED_Q_NAME = "seqused_q";
const std::string SEQUSED_ORI_KV_NAME = "seqused_ori_kv";
const std::string SEQUSED_CMP_KV_NAME = "seqused_cmp_kv";
const std::string CMP_RESIDUAL_KV_NAME = "cmp_residual_kv";
const std::string ORI_TOPK_LENGTH_NAME = "ori_topk_length";
const std::string CMP_TOPK_LENGTH_NAME = "cmp_topk_length";
const std::string METADATA_NAME = "metadata";

// // ------------------公共定义--------------------------
struct QSMLATilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QSMLATilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
    const gert::StorageShape *shape;
};

enum class QSMLALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

enum class QSMLAAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3, // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7  // block size
};

enum class QSMLATemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    HCA_TEMPLATE_MODE = 1,
    CSA_TEMPLATE_MODE = 2,
    ORI_SPARSE_TEMPLATE_MODE = 3,
    ORI_CMP_SPARSE_TEMPLATE_MODE = 4
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t Q_DESCALE = 3;
constexpr uint32_t ORI_KV_DESCALE = 4;
constexpr uint32_t CMP_KV_DESCALE = 5;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 6;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 7;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 8;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 9;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 10;
constexpr uint32_t CU_SEQLENS_ORI_KV_INDEX = 11;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 12;
constexpr uint32_t SEQUSED_Q_INDEX = 13;
constexpr uint32_t SEQUSED_ORI_KV_INDEX = 14;
constexpr uint32_t SEQUSED_CMP_KV_INDEX = 15;
constexpr uint32_t CMP_RESIDUAL_KV_INDEX = 16;
constexpr uint32_t ORI_TOPK_LENGTH_INDEX = 17;
constexpr uint32_t CMP_TOPK_LENGTH_INDEX = 18;
constexpr uint32_t SINKS_INDEX = 19;
constexpr uint32_t METADATA_INDEX = 20;
// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

// Attributes Index
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 0;
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 1;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 2;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 3;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 5;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 6;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 7;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 8;
constexpr uint32_t ATTR_TOPK_VALUE_MODE_INDEX = 9;
constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 10;

// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_IDX_FOUR = 4;

// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

const std::map<QSMLALayout, std::vector<QSMLAAxis>> QSMLA_LAYOUT_AXIS_MAP = {
    {QSMLALayout::BSND, {QSMLAAxis::B, QSMLAAxis::S, QSMLAAxis::N, QSMLAAxis::D}},
    {QSMLALayout::TND, {QSMLAAxis::T, QSMLAAxis::N, QSMLAAxis::D}},
    {QSMLALayout::PA_BBND, {QSMLAAxis::Bn, QSMLAAxis::Bs, QSMLAAxis::N, QSMLAAxis::D}},
};

const std::map<QSMLALayout, size_t> QSMLA_LAYOUT_DIM_MAP = {
    {QSMLALayout::BSND, DIM_NUM_FOUR},
    {QSMLALayout::TND, DIM_NUM_THREE},
    {QSMLALayout::PA_BBND, DIM_NUM_FOUR},
};

std::string QSMLALayoutToSerialString(QSMLALayout layout);

// -----------算子Tiling入参信息解析及Check类---------------

struct QSMLAParaInfo {
    QSMLATilingRequiredParaInfo q = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo qDescale = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriKvDescale = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpKvDescale = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriSparseIndices = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cuSeqLensOriKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cuSeqLensCmpKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo sequsedOriKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo sequsedCmpKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpResidualKv = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo oriTopkLength = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo cmpTopkLength = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo sinks = {nullptr, nullptr};
    QSMLATilingOptionalParaInfo metadata = {nullptr, nullptr};
    QSMLATilingRequiredParaInfo attnOut = {nullptr, nullptr};
    QSMLATilingRequiredParaInfo softmaxLse = {nullptr, nullptr};

    const int64_t *quantMode = nullptr;
    const float *softmaxScale = nullptr;
    const int64_t *oriKvStride = nullptr;
    const int64_t *cmpKvStride = nullptr;
    const int64_t *cmpRatio = nullptr;
    const uint32_t *oriMaskMode = nullptr;
    const uint32_t *cmpMaskMode = nullptr;
    const int64_t *oriWinLeft = nullptr;
    const int64_t *oriWinRight = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKv = nullptr;
    const int64_t *topkValueMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

// -----------算子Tiling入参信息类---------------
class QSMLATilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QSMLAParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t cmpS2Size = 0;
    uint32_t gSize = 0;
    uint32_t qkHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND时生效

    uint32_t maxActualseq = 0;
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;

    int64_t quantMode = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeVInput = 0;
    float softmaxScale = 0;
    int64_t oriKvStride = 0;
    int64_t cmpKvStride = 0;
    std::vector<int64_t> oriKvStrides;
    std::vector<int64_t> cmpKvStrides;
    gert::Shape oriKvStorageShape;
    gert::Shape cmpKvStorageShape;
    int64_t cmpRatio = 0;
    uint64_t oriMaskMode = 0;
    uint64_t cmpMaskMode = 0;
    int64_t topkValueMode = 0;
    int64_t oriWinLeft = 0;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 0;
    int64_t oriSparseBlockCount = 0;
    int64_t cmpSparseBlockCount = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;

    // PageAttention
    uint32_t blockTypeSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    int32_t oriBlockSize = 0;
    int32_t cmpBlockSize = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    // DType
    ge::DataType qType = ge::DT_FLOAT16;
    ge::DataType oriKvType = ge::DT_FLOAT16;
    ge::DataType cmpKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    // Layout
    QSMLALayout qLayout = QSMLALayout::BSND;
    QSMLALayout kvLayout = QSMLALayout::PA_BBND;
    QSMLALayout outLayout = QSMLALayout::BSND;

    bool returnSoftmaxLse = false;
};

class QSMLAInfoParser {
public:
    explicit QSMLAInfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~QSMLAInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, QSMLALayout &layout,
                                        const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();

    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    void SetQSMLAShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetDSizeQ();
    ge::graphStatus GetDSizeKV();
    ge::graphStatus GetKvstride();
    void GenerateInfo(QSMLATilingInfo &qsmlaInfo);
    ge::graphStatus Parse(QSMLATilingInfo &qsmlaInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    QSMLAParaInfo opParamInfo_;

    bool HasAxis(const QSMLAAxis &axis, const QSMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const QSMLAAxis &axis, const QSMLALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const QSMLAAxis &axis, const QSMLALayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t qkHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t oriSparseBlockCount_ = 0;
    int64_t cmpSparseBlockCount_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t dSizeQ_ = 0;
    uint32_t dSizeKV_ = 0;
    uint32_t oriKvStride_ = 0;
    uint32_t cmpKvStride_ = 0;
    std::vector<int64_t> oriKvStridesVec_;
    std::vector<int64_t> cmpKvStridesVec_;
    // Layout
    QSMLALayout qLayout_ = QSMLALayout::BSND;
    QSMLALayout outLayout_ = QSMLALayout::BSND;
    QSMLALayout kvLayout_ = QSMLALayout::PA_BBND;
    // PageAttention
    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape qShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape cmpKvShape_{};
    gert::Shape oriSparseIndicesShape_{};
    gert::Shape cmpSparseIndicesShape_{};
};

} // namespace optiling
#endif // QUANT_SPARSE_FLASH_MLA_CHECK_H
