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
 * \file mixed_quant_sparse_flash_mla_check.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CHECK_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CHECK_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
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
struct MQSMLATilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct MQSMLATilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
    const gert::StorageShape *shape;
};

enum class MQSMLALayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

enum class MQSMLAAxis : uint32_t {
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

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

struct QSMLATilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    // PA
    int64_t Bs = 1;
    int64_t Bn = 1;
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t Q_INDEX = 0;
constexpr uint32_t ORI_KV_INDEX = 1;
constexpr uint32_t CMP_KV_INDEX = 2;
constexpr uint32_t ORI_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t CU_SEQLENS_ORI_KV_INDEX = 8;
constexpr uint32_t CU_SEQLENS_CMP_KV_INDEX = 9;
constexpr uint32_t SEQUSED_Q_INDEX = 10;
constexpr uint32_t SEQUSED_ORI_KV_INDEX = 11;
constexpr uint32_t SEQUSED_CMP_KV_INDEX = 12;
constexpr uint32_t CMP_RESIDUAL_KV_INDEX = 13;
constexpr uint32_t ORI_TOPK_LENGTH_INDEX = 14;
constexpr uint32_t CMP_TOPK_LENGTH_INDEX = 15;
constexpr uint32_t SINKS_INDEX = 16;
constexpr uint32_t METADATA_INDEX = 17;
// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

// Attributes Index
constexpr uint32_t ATTR_QUANT_SCALE_INDEX = 0;
constexpr uint32_t ATTR_ROPE_HEAD_DIM_INDEX = 1;
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 2;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 3;
constexpr uint32_t ATTR_ORI_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_CMP_MASK_MODE_INDEX = 5;
constexpr uint32_t ATTR_ORI_WIN_LEFT_INDEX = 6;
constexpr uint32_t ATTR_ORI_WIN_RIGHT_INDEX = 7;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 8;
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 9;
constexpr uint32_t ATTR_TOPK_VALUE_MODE_INDEX = 10;
constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 11;

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

// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t KV_INPUT_DIM_LIMIT_QUANT_MODE_ONE = 608;
constexpr uint32_t KV_INPUT_DIM_LIMIT_QUANT_MODE_TWO = 584;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t MAX_BLOCK_SIZE = 1024;
constexpr uint32_t METADATA_SIZE = 1024;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t BYTE_BLOCK = 32;
const uint32_t QSFA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"},                   // dt_variant type
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},   // dt_variant type
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},             // dt_variant type
};

const std::map<MQSMLALayout, std::vector<MQSMLAAxis>> QSMLA_LAYOUT_AXIS_MAP = {
    {MQSMLALayout::BSND, {MQSMLAAxis::B, MQSMLAAxis::S, MQSMLAAxis::N, MQSMLAAxis::D}},
    {MQSMLALayout::TND, {MQSMLAAxis::T, MQSMLAAxis::N, MQSMLAAxis::D}},
    {MQSMLALayout::PA_BBND, {MQSMLAAxis::Bn, MQSMLAAxis::Bs, MQSMLAAxis::N, MQSMLAAxis::D}},
};

const std::map<MQSMLALayout, size_t> QSMLA_LAYOUT_DIM_MAP = {
    {MQSMLALayout::BSND, DIM_NUM_FOUR},
    {MQSMLALayout::TND, DIM_NUM_THREE},
    {MQSMLALayout::PA_BBND, DIM_NUM_FOUR},
};
std::vector<int64_t> ToVector(const gert::Shape &shape);
std::string ToStringRaw(const gert::Shape &shape);
std::string MQSMLADataTypeToSerialString(ge::DataType type);
std::string MQSMLALayoutToSerialString(MQSMLALayout layout);
std::string GetShapeStr(gert::Shape shape);

// -----------算子Tiling入参信息解析及Check类---------------

struct MQSMLAParaInfo {
    MQSMLATilingRequiredParaInfo q = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo oriKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cmpKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo oriSparseIndices = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cmpSparseIndices = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo oriBlockTable = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cmpBlockTable = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cuSeqLensOriKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cuSeqLensCmpKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo seqUsedQ = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo sequsedOriKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo sequsedCmpKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cmpResidualKv = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo oriTopkLength = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo cmpTopkLength = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo sinks = {nullptr, nullptr};
    MQSMLATilingOptionalParaInfo metadata = {nullptr, nullptr};
    MQSMLATilingRequiredParaInfo attnOut = {nullptr, nullptr};
    MQSMLATilingRequiredParaInfo softmaxLse = {nullptr, nullptr};

    const int64_t *quantMode = nullptr;
    const int64_t *tileSize = nullptr;
    const int64_t *ropeHeadDim = nullptr;
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
class MQSMLATilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    MQSMLAParaInfo opParamInfo;

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
    int64_t tileSize = 0;
    int64_t ropeHeadDim = 0;
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
    bool batchConsistency = false;

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
    MQSMLALayout qLayout = MQSMLALayout::BSND;
    MQSMLALayout kvLayout = MQSMLALayout::PA_BBND;
    MQSMLALayout outLayout = MQSMLALayout::BSND;

    bool returnSoftmaxLse = false;
};

class MQSMLAInfoParser {
public:
    explicit MQSMLAInfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~MQSMLAInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, MQSMLALayout &layout,
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
    void GenerateInfo(MQSMLATilingInfo &qsmlaInfo);
    ge::graphStatus Parse(MQSMLATilingInfo &qsmlaInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    MQSMLAParaInfo opParamInfo_;

    bool HasAxis(const MQSMLAAxis &axis, const MQSMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const MQSMLAAxis &axis, const MQSMLALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const MQSMLAAxis &axis, const MQSMLALayout &layout) const;
    static constexpr uint32_t invalidDimValue_ = std::numeric_limits<uint32_t>::min();

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
    bool batchConsistency_ = false;
    uint32_t dSizeQ_ = 0;
    uint32_t dSizeKV_ = 0;
    uint32_t oriKvStride_ = 0;
    uint32_t cmpKvStride_ = 0;
    std::vector<int64_t> oriKvStridesVec_;
    std::vector<int64_t> cmpKvStridesVec_;
    // Layout
    MQSMLALayout qLayout_ = MQSMLALayout::BSND;
    MQSMLALayout outLayout_ = MQSMLALayout::BSND;
    MQSMLALayout kvLayout_ = MQSMLALayout::PA_BBND;
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

class MQSMLATilingCheck {
public:
    explicit MQSMLATilingCheck(const MQSMLATilingInfo &qsmlaInfo)
        : qsmlaInfo_(qsmlaInfo) {};
    ~MQSMLATilingCheck() = default;
    virtual ge::graphStatus Process();

private:
    void Init();
    bool HasAxis(const MQSMLAAxis &axis, const MQSMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const MQSMLAAxis &axis, const MQSMLALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const MQSMLAAxis &axis, const MQSMLALayout &layout) const;
    static constexpr uint32_t invalidDimValue_ = std::numeric_limits<uint32_t>::min();

    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                              const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                               const std::string subName) const;
    template <typename T>
    void LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                               const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::vector<size_t> &expectDimNumList,
                                       const std::string &name) const;
    ge::graphStatus CheckShapeNumSupport(const gert::StorageShape *shape,
                                         const std::vector<int64_t> &expectShapeNumList, const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const MQSMLALayout &layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<MQSMLALayout> &expectLayoutList, const MQSMLALayout &actualLayout,
                               const std::string &name) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected, const QSMLATilingShapeCompareParam &param,
                                     const MQSMLALayout &layout) const;
    ge::graphStatus CompareShape(QSMLATilingShapeCompareParam &param, const gert::Shape &shape,
                                 const MQSMLALayout &layout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const MQSMLALayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaKey() const;
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaSparseBlockSize() const;
    ge::graphStatus CheckSingleParaOriSparseIndices() const;
    ge::graphStatus CheckSingleParaCmpSparseIndices() const;
    ge::graphStatus CheckSingleParaCmpResidualKv() const;
    ge::graphStatus CheckSingleParaBlockTable() const;
    ge::graphStatus CheckSingleParaCuSeqLensQ() const;
    ge::graphStatus CheckSingleParaSequsedKv() const;
    ge::graphStatus CheckSingleParaCuSeqLensOriKv() const;
    ge::graphStatus CheckSingleParaCuSeqLensCmpKv() const;
    ge::graphStatus CheckSingleParaSequsedQ() const;
    ge::graphStatus CheckSingleParaSequsedCmpKv() const;
    ge::graphStatus CheckSingleParaOriTopkLength() const;
    ge::graphStatus CheckSingleParaCmpTopkLength() const;
    ge::graphStatus CheckSingleParaSinks() const;
    ge::graphStatus CheckSingleParaMetadata() const;
    ge::graphStatus CheckSingleParaTopkValueMode() const;

    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckParaExistenceAntiquant() const;
    ge::graphStatus CheckCmpSparseIndicesExistence();
    ge::graphStatus CheckParaExistence();
    ge::graphStatus CheckCmpRatioExistence();
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, const MQSMLALayout &layout,
                                        const std::string &name) const;
    ge::graphStatus CheckSWAExistence();
    ge::graphStatus CheckHCAExistence();
    ge::graphStatus CheckCSAExistence();

    ge::graphStatus CheckKVShapeForBatchContinuous();
    uint32_t GetTypeSize(ge::DataType dtype) const;
    ge::graphStatus CheckKVShapeForPageAttention();
    ge::graphStatus CheckKVShape();
    ge::graphStatus CheckKV();
    ge::graphStatus CheckTopK();
    ge::graphStatus CheckTopkShape();
    ge::graphStatus CheckBlockTable() const;
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype, const ge::DataType &expectDtype,
                                          const std::string &name) const;

    ge::graphStatus CheckAttenOut();
    ge::graphStatus CheckAttenOutShape();
    ge::graphStatus CheckSoftmaxLse() const;
    ge::graphStatus CheckActualSeqLensQ();
    ge::graphStatus CheckActualSeqLensQShape();
    ge::graphStatus CheckActualSeqLensQDType();
    ge::graphStatus CheckActualSeqLens();
    ge::graphStatus CheckActualSeqLensDType();
    ge::graphStatus CheckActualSeqLensShape();
    ge::graphStatus CheckMultiParaConsistency();

    ge::graphStatus CheckConsistency() const;
    ge::graphStatus CheckLayoutQKvConsistency() const;
    ge::graphStatus CheckSparseIndicesShapeMatchQ() const;
    ge::graphStatus CheckTopkLengthConsistency() const;
    ge::graphStatus CheckN2Consistency() const;
    ge::graphStatus CheckBConsistency() const;
    ge::graphStatus CheckFeatureWinKV() const;
    ge::graphStatus CheckFeatureAntiquantShape() const;
    ge::graphStatus CheckFeatureAntiquantLayout() const;
    ge::graphStatus CheckFeatureAntiquantDtype() const;
    ge::graphStatus CheckFeatureAntiquantAttr() const;
    ge::graphStatus CheckFeatureAntiquantPa() const;
    ge::graphStatus CheckFeatureAntiquant() const;
    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckKvContiguous() const;
    void SetQSMLAShapeCompare();

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    MQSMLAParaInfo opParamInfo_;
    const MQSMLATilingInfo &qsmlaInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    int64_t ropeHeadDim_ = 0;
    uint32_t qTSize_ = 0;  // 仅TND时生效
    uint32_t kvTSize_ = 0; // 仅TND时生效
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t oriSparseBlockCount_ = 0;
    uint32_t cmpSparseBlockCount_ = 0;
    uint32_t sparseBlockSize_ = 0;
    uint32_t oriKvStride_ = 0;
    uint32_t cmpKvStride_ = 0;
    int64_t topkValueMode_ = 0;
    uint32_t oriBlockNum_ = 0;
    uint32_t cmpBlockNum_ = 0;

    uint32_t oriBlockSize_ = 0;
    uint32_t cmpBlockSize_ = 0;
    uint32_t oriBlockTable_ = 0;
    uint32_t cmpBlockTable_ = 0;
    int64_t quant_mode_ = 0;
    int64_t tileSize_ = 0;

    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;

    int64_t cmpRatio_ = 0;

    uint32_t dSize_ = qsmlaInfo_.dSize;
    uint32_t dSizeV_ = qsmlaInfo_.dSizeV;
    uint32_t dSizeVInput_ = qsmlaInfo_.dSizeVInput;

    uint32_t dSizeOriKvInput_ = 0;
    uint32_t dSizeCmpKvInput_ = 0;

    uint32_t oriMaskMode_ = 0;
    uint32_t cmpMaskMode_ = 0;

    MQSMLALayout qLayout_ = MQSMLALayout::BSND;
    MQSMLALayout outLayout_ = MQSMLALayout::BSND;
    MQSMLALayout kvLayout_ = MQSMLALayout::PA_BBND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    QSMLATemplateMode perfMode_;

    gert::Shape queryShapeCmp_{};
    gert::Shape keyShapeCmp_{};
    gert::Shape valueShapeCmp_{};
    gert::Shape topkShapeCmp_{};
    gert::Shape attenOutShapeCmp_{};
};
} // namespace optiling
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CHECK_H
