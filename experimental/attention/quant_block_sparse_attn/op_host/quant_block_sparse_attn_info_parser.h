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
 * \file quant_block_sparse_attn_info_parser.h
 * \brief QuantBlockSparseAttn parameter parsing class.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_INFO_PARSER_H_
#define QUANT_BLOCK_SPARSE_ATTN_INFO_PARSER_H_

#include <string>

#include <exe_graph/runtime/tiling_context.h>

namespace optiling {

// Inputs Index
constexpr uint32_t QBSA_QUERY_INDEX = 0U;
constexpr uint32_t QBSA_KEY_INDEX = 1U;
constexpr uint32_t QBSA_VALUE_INDEX = 2U;
constexpr uint32_t QBSA_Q_DESCALE_INDEX = 3U;
constexpr uint32_t QBSA_K_DESCALE_INDEX = 4U;
constexpr uint32_t QBSA_V_DESCALE_INDEX = 5U;
constexpr uint32_t QBSA_P_SCALE_INDEX = 6U;
constexpr uint32_t QBSA_CU_SEQLENS_Q_INDEX = 7U;
constexpr uint32_t QBSA_CU_SEQLENS_KV_INDEX = 8U;
constexpr uint32_t QBSA_SEQUSED_Q_INDEX = 9U;
constexpr uint32_t QBSA_SEQUSED_KV_INDEX = 10U;
constexpr uint32_t QBSA_SPARSE_INDICES_INDEX = 11U;
constexpr uint32_t QBSA_SPARSE_SEQ_LEN_INDEX = 12U;
constexpr uint32_t QBSA_BLOCK_TABLE_INDEX = 13U;
constexpr uint32_t QBSA_ATTEN_MASK_INDEX = 14U;
constexpr uint32_t QBSA_METADATA_INDEX = 15U;

// Outputs Index
constexpr uint32_t QBSA_ATTENTION_OUT_INDEX = 0U;
constexpr uint32_t QBSA_SOFTMAX_LSE_INDEX = 1U;

// Attributes Index
constexpr uint32_t QBSA_SOFTMAX_SCALE_ATTR_INDEX = 0U;
constexpr uint32_t QBSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX = 1U;
constexpr uint32_t QBSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX = 2U;
constexpr uint32_t QBSA_LAYOUT_KV_ATTR_INDEX = 3U;
constexpr uint32_t QBSA_LAYOUT_Q_ATTR_INDEX = 4U;
constexpr uint32_t QBSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX = 5U;
constexpr uint32_t QBSA_LAYOUT_OUT_ATTR_INDEX = 6U;
constexpr uint32_t QBSA_QUANT_MODE_ATTR_INDEX = 7U;
constexpr uint32_t QBSA_MASK_MODE_ATTR_INDEX = 8U;
constexpr uint32_t QBSA_RETURN_SOFTMAX_LSE_ATTR_INDEX = 9U;

struct QBSARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::StorageShape *shape = nullptr;
    const gert::Stride *stride = nullptr;
};

struct QBSAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::Tensor *tensor = nullptr;
};

struct QuantBlockSparseAttnParaInfo {
    QBSARequiredParaInfo query;
    QBSARequiredParaInfo key;
    QBSARequiredParaInfo value;
    QBSARequiredParaInfo qDescale;
    QBSARequiredParaInfo kDescale;
    QBSARequiredParaInfo vDescale;
    QBSARequiredParaInfo pScale;
    QBSARequiredParaInfo sparseIndices;
    QBSARequiredParaInfo sparseSeqLen;
    QBSARequiredParaInfo attenMask;

    QBSAOptionalParaInfo cuSeqlensQ;
    QBSAOptionalParaInfo cuSeqlensKV;
    QBSAOptionalParaInfo seqUsedQ;
    QBSAOptionalParaInfo seqUsedKV;
    QBSAOptionalParaInfo blockTable;
    QBSAOptionalParaInfo metadata;

    QBSARequiredParaInfo attnOut;
    QBSARequiredParaInfo lseOut;

    const float *softmaxScale = nullptr;
    const int64_t *qBlockSize = nullptr;
    const int64_t *kvBlockSize = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKV = nullptr;
    const char *layoutSparseIndices = nullptr;
    const int64_t *quantMode = nullptr;
    const int64_t *maskMode = nullptr;
    const bool *returnSoftmaxLse = nullptr;
};

struct QuantBlockSparseAttnTilingInfo {
    QuantBlockSparseAttnParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t qbMax = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t qTokenNum = 0;

    uint32_t sparseCount = 0;
    uint32_t qBlockSizeVal = 0;
    uint32_t kvBlockSizeVal = 0;
    uint32_t paBlockSizeVal = 0;
    uint32_t paBlockStrideVal = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t paBlockNumSum = 0;
    uint32_t gS1OuterSize = 0;

    ge::DataType qDtype = ge::DT_UNDEFINED;
    ge::DataType kvDtype = ge::DT_UNDEFINED;

    float softmaxScaleVal = 1.0F;
    uint32_t maskModeVal = 0;
    uint32_t quantModeVal = 1;
    uint32_t layoutQValue = 0;
    std::string layoutQStr = "TND";
    std::string layoutKVStr = "PA_BNBD";
    std::string layoutSparseIndicesStr = "B_N_Qb_Kb";
    std::string layoutOutStr = "TND";
    bool returnSoftmaxLseVal = false;

    bool isGqa = false;
    bool hasViewStride = true;
};

class QuantBlockSparseAttnInfoParser {
public:
    explicit QuantBlockSparseAttnInfoParser(gert::TilingContext *context);
    ~QuantBlockSparseAttnInfoParser() = default;

    ge::graphStatus Parse(QuantBlockSparseAttnTilingInfo &tilingInfo);

private:
    ge::graphStatus ParseQuery(QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::Shape &queryShape,
                               const gert::Shape &sparseIndicesShape);
    ge::graphStatus ParseKeyValue(QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::Shape &keyShape,
                                  const gert::Shape &valueShape, const gert::Shape &kDescaleShape,
                                  const gert::Stride *keyStride);
    ge::graphStatus ParseSparseIndices(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                       const gert::Shape &sparseIndicesShape);
    ge::graphStatus ParseOptionalInputs(QuantBlockSparseAttnTilingInfo &tilingInfo);
    ge::graphStatus ParseAttributes(QuantBlockSparseAttnTilingInfo &tilingInfo, const gert::RuntimeAttrs *attrs);

    gert::TilingContext *context_;
};

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_INFO_PARSER_H_
