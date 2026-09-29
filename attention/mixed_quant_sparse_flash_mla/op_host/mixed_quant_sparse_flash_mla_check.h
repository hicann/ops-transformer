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
#include "../../sparse_flash_mla/op_host/common/smla_host_common_defs.h"

namespace optiling {

const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";

// // ------------------公共定义--------------------------
using MQSMLATilingRequiredParaInfo = SMLATilingRequiredParaInfo;
using MQSMLATilingOptionalParaInfo = SMLATilingOptionalParaInfo;
using MQSMLALayout = SMLALayout;
using MQSMLAAxis = SMLAAxis;
using QSMLATemplateMode = SMLATemplateMode;

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t MQ_Q_INDEX = 0;
constexpr uint32_t MQ_ORI_KV_INDEX = 1;
constexpr uint32_t MQ_CMP_KV_INDEX = 2;
constexpr uint32_t MQ_ORI_SPARSE_INDICES_INDEX = 3;
constexpr uint32_t MQ_CMP_SPARSE_INDICES_INDEX = 4;
constexpr uint32_t MQ_ORI_BLOCK_TABLE_INDEX = 5;
constexpr uint32_t MQ_CMP_BLOCK_TABLE_INDEX = 6;
constexpr uint32_t MQ_CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t MQ_CU_SEQLENS_ORI_KV_INDEX = 8;
constexpr uint32_t MQ_CU_SEQLENS_CMP_KV_INDEX = 9;
constexpr uint32_t MQ_SEQUSED_Q_INDEX = 10;
constexpr uint32_t MQ_SEQUSED_ORI_KV_INDEX = 11;
constexpr uint32_t MQ_SEQUSED_CMP_KV_INDEX = 12;
constexpr uint32_t MQ_CMP_RESIDUAL_KV_INDEX = 13;
constexpr uint32_t MQ_ORI_TOPK_LENGTH_INDEX = 14;
constexpr uint32_t MQ_CMP_TOPK_LENGTH_INDEX = 15;
constexpr uint32_t MQ_SINKS_INDEX = 16;
constexpr uint32_t MQ_METADATA_INDEX = 17;

// Attributes Index
constexpr uint32_t MQ_ATTR_QUANT_SCALE_INDEX = 0;
constexpr uint32_t MQ_ATTR_ROPE_HEAD_DIM_INDEX = 1;
constexpr uint32_t MQ_ATTR_SOFTMAX_SCALE_INDEX = 2;
constexpr uint32_t MQ_ATTR_CMP_RATIO_INDEX = 3;
constexpr uint32_t MQ_ATTR_ORI_MASK_MODE_INDEX = 4;
constexpr uint32_t MQ_ATTR_CMP_MASK_MODE_INDEX = 5;
constexpr uint32_t MQ_ATTR_ORI_WIN_LEFT_INDEX = 6;
constexpr uint32_t MQ_ATTR_ORI_WIN_RIGHT_INDEX = 7;
constexpr uint32_t MQ_ATTR_LAYOUT_Q_INDEX = 8;
constexpr uint32_t MQ_ATTR_LAYOUT_KV_INDEX = 9;
constexpr uint32_t MQ_ATTR_TOPK_VALUE_MODE_INDEX = 10;
constexpr uint32_t MQ_ATTR_RETURN_SOFTMAX_LSE_INDEX = 11;

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
std::string MQSMLALayoutToSerialString(MQSMLALayout layout);

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
    uint32_t actualLenDimsQ = 0;
    uint32_t actualLenDimsKV = 0;
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

    ge::graphStatus GetActualSeqLenSize(int64_t &size, const gert::Tensor *tensor, MQSMLALayout &layout,
                                        const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(int64_t &size);
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
    const char *mqsmlaOpName_;
    fe::PlatFormInfos *mqsmlaPlatformInfo_;
    MQSMLAParaInfo mqsmlaParams_;

    bool HasAxis(const MQSMLAAxis &axis, const MQSMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const MQSMLAAxis &axis, const MQSMLALayout &layout) const;
    int64_t GetAxisNum(const gert::Shape &shape, const MQSMLAAxis &axis, const MQSMLALayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    // BaseParams
    int64_t mqsmlaBatchSize_ = 0;
    int64_t mqsmlaQueryHeads_ = 0;
    int64_t mqsmlaKvHeads_ = 0;
    int64_t mqsmlaGroupSize_ = 0;
    int64_t mqsmlaQuerySeqSize_ = 0;
    int64_t mqsmlaKvSeqSize_ = 0;
    int64_t mqsmlaCmpKvSeqSize_ = 0;
    int64_t headDim_ = 0;
    int64_t mqsmlaQueryTokenSize_ = 0;
    int64_t mqsmlaQkHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t mqsmlaOriSparseBlockCount_ = 0;
    int64_t mqsmlaCmpSparseBlockCount_ = 0;
    int64_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    bool batchConsistency_ = false;
    int64_t mqsmlaQueryDim_ = 0;
    int64_t mqsmlaKvDim_ = 0;
    int64_t mqsmlaOriKvStride_ = 0;
    int64_t mqsmlaCmpKvStride_ = 0;
    uint32_t mqsmlaActualQueryLenDims_ = 0;
    uint32_t mqsmlaActualKvLenDims_ = 0;
    std::vector<int64_t> mqsmlaOriKvStrides_;
    std::vector<int64_t> mqsmlaCmpKvStrides_;
    // Layout
    MQSMLALayout mqsmlaQLayout_ = MQSMLALayout::BSND;
    MQSMLALayout mqsmlaOutputLayout_ = MQSMLALayout::BSND;
    MQSMLALayout mqsmlaKvLayout_ = MQSMLALayout::PA_BBND;
    // PageAttention
    uint32_t mqsmlaOriMaxBlocksPerBatch_ = 0;
    uint32_t mqsmlaCmpMaxBlocksPerBatch_ = 0;
    int64_t mqsmlaOriBlockSize_ = 0;
    int64_t mqsmlaCmpBlockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType mqsmlaQType_ = ge::DT_FLOAT16;
    ge::DataType mqsmlaOriKvType_ = ge::DT_FLOAT16;
    ge::DataType mqsmlaCmpKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType mqsmlaOutputType_ = ge::DT_FLOAT16;

    gert::Shape mqsmlaQShape_{};
    gert::Shape mqsmlaOriKvShape_{};
    gert::Shape mqsmlaCmpKvShape_{};
    gert::Shape mqsmlaOriSparseIndicesShape_{};
    gert::Shape mqsmlaCmpSparseIndicesShape_{};
};

} // namespace optiling
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CHECK_H
