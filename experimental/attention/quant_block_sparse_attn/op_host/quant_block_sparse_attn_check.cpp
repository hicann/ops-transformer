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
 * \file quant_block_sparse_attn_check.cpp
 * \brief QuantBlockSparseAttn parameter validation implementation.
 */

#include <array>
#include <cstddef>
#include <string>

#include "quant_block_sparse_attn_check.h"
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr size_t DIM_NUM_5 = 5U;
constexpr size_t DIM_0 = 0U;
constexpr size_t DIM_1 = 1U;
constexpr size_t DIM_2 = 2U;
constexpr size_t DIM_3 = 3U;
constexpr int64_t BSA_ATTEN_MASK_DIM_VALUE = 2048;
constexpr int64_t BSA_MXFP8_P_SCALE_SIZE = 1;

bool IsMXFP8SparseBlockSizeSupported(uint32_t blockSize)
{
    return blockSize == BSA_MXFP8_SPARSE_BLOCK_SIZE_128 || blockSize == BSA_MXFP8_SPARSE_BLOCK_SIZE_64;
}

template <size_t N>
std::string ShapeToString(const std::array<int64_t, N> &shape)
{
    std::string result = "[";
    for (size_t i = 0U; i < N; ++i) {
        if (i != 0U) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

template <size_t N>
bool IsShapeEqual(const gert::Shape &actualShape, const std::array<int64_t, N> &expectedShape)
{
    if (actualShape.GetDimNum() != N) {
        return false;
    }
    for (size_t i = 0U; i < N; ++i) {
        if (actualShape.GetDim(i) != expectedShape[i]) {
            return false;
        }
    }
    return true;
}

template <size_t N>
ge::graphStatus CheckInputShape(const BSARequiredParaInfo &input, const char *inputName,
                                const std::array<int64_t, N> &expectedShape)
{
    if (input.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, inputName, "nullptr", "shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &actualShape = input.shape->GetStorageShape();
    if (!IsShapeEqual(actualShape, expectedShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, inputName, Ops::Base::ToString(actualShape),
            "must be " + ShapeToString(expectedShape) + " in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputDtype(const BSARequiredParaInfo &input, const char *inputName, ge::DataType expectedType,
                                const char *expectedTypeName)
{
    if (input.desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, inputName, "nullptr", "desc is nullptr");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType actualType = input.desc->GetDataType();
    if (actualType != expectedType) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, inputName, std::to_string(static_cast<int32_t>(actualType)),
                                  expectedTypeName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMXFP8FullQuantScaleInputs(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    if (opParamInfo.pScale.shape == nullptr || opParamInfo.kDescale.shape == nullptr ||
        opParamInfo.vDescale.shape == nullptr || opParamInfo.qDescale.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "quantScale1/keyAntiquantScale/valueAntiquantScale/dequantScaleQuery", "nullptr",
            "all must be passed in quant_mode=2 MXFP8 full-quant scenario");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckKVDtypeConsistency(const QuantBlockSparseAttnParaInfo &opParamInfo)
{
    if (opParamInfo.key.desc == nullptr || opParamInfo.value.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const ge::DataType keyType = opParamInfo.key.desc->GetDataType();
    const ge::DataType valueType = opParamInfo.value.desc->GetDataType();
    if (keyType != valueType) {
        OP_LOGE_FOR_INVALID_DTYPE(
            kOpName, "key/value",
            std::to_string(static_cast<int32_t>(keyType)) + "/" + std::to_string(static_cast<int32_t>(valueType)),
            "key and value must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckStrideDim(const gert::Stride *stride, const char *inputName, size_t dim, uint64_t expected)
{
    if (stride == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, inputName, "nullptr",
            "stride is nullptr. PA_BNSD segmented KV-cache requires 4D non-contiguous key/value/k_descale views");
        return ge::GRAPH_FAILED;
    }
    if (stride->GetDimNum() <= dim) {
        const std::string argName = std::string(inputName) + " stride";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, argName.c_str(), std::to_string(stride->GetDimNum()) + "D",
            "at least " + std::to_string(dim + 1U) + "D stride for PA_BNSD segmented KV-cache");
        return ge::GRAPH_FAILED;
    }
    const uint64_t actual = stride->GetStride(dim);
    if (actual != expected) {
        const std::string argName = std::string(inputName) + " stride[" + std::to_string(dim) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, argName.c_str(), std::to_string(actual),
                                              "must be " + std::to_string(expected));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

QuantBlockSparseAttnCheck::QuantBlockSparseAttnCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo)
    : tilingInfo_(tilingInfo)
{}

ge::graphStatus QuantBlockSparseAttnCheck::CheckAttrs() const
{
    if (tilingInfo_.layoutKVStr != "PA_BNSD" || tilingInfo_.layoutSparseIndicesStr != "B_N_Qb_Kb" ||
        (tilingInfo_.quantModeVal != BSA_QUANT_MODE_FP8 &&
         tilingInfo_.quantModeVal != BSA_QUANT_MODE_MXFP8_FULL_QUANT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "layout_kv/layout_sparse_indices/quant_mode",
            tilingInfo_.layoutKVStr + "/" + tilingInfo_.layoutSparseIndicesStr + "/" +
                std::to_string(tilingInfo_.quantModeVal),
            "layout_kv must be PA_BNSD, layout_sparse_indices must be B_N_Qb_Kb, quant_mode must be 1 or 2");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_FP8 && tilingInfo_.layoutOutStr != "TND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "layout_out", tilingInfo_.layoutOutStr,
                                              "must be TND in quant_mode=1 FP8 scenario");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        if (tilingInfo_.layoutQStr != "TND" || tilingInfo_.layoutOutStr != "TND") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "layout_q/layout_out",
                                                  tilingInfo_.layoutQStr + "/" + tilingInfo_.layoutOutStr,
                                                  "must both be TND in quant_mode=2 MXFP8 full-quant scenario");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(kOpName, "quant_mode=2 maps to queryQuantMode/keyAntiquantMode/valueAntiquantMode=%u/%u/%u",
                BSA_MXFP8_PER_TOKEN_GROUP_MODE, BSA_MXFP8_PER_TOKEN_GROUP_MODE, BSA_MXFP8_PER_CHANNEL_GROUP_MODE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckDtype() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (CheckKVDtypeConsistency(opParamInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "query", std::to_string(static_cast<int>(tilingInfo_.qDtype)),
                                  "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.kvDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "key/value", std::to_string(static_cast<int>(tilingInfo_.kvDtype)),
                                  "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.value.desc != nullptr && opParamInfo.value.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE(
            kOpName, "value", std::to_string(static_cast<int>(opParamInfo.value.desc->GetDataType())), "FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.sparseIndices.desc != nullptr && opParamInfo.sparseIndices.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "sparse_indices",
                                  std::to_string(static_cast<int>(opParamInfo.sparseIndices.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.sparseSeqLen.desc != nullptr && opParamInfo.sparseSeqLen.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "sparse_seq_len",
                                  std::to_string(static_cast<int>(opParamInfo.sparseSeqLen.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.attenMask.desc != nullptr && opParamInfo.attenMask.shape != nullptr &&
        opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() > 0 &&
        opParamInfo.attenMask.desc->GetDataType() != ge::DT_UINT8) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "atten_mask",
                                  std::to_string(static_cast<int>(opParamInfo.attenMask.desc->GetDataType())), "UINT8");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.metadata.desc != nullptr && opParamInfo.metadata.tensor != nullptr &&
        opParamInfo.metadata.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "metadata",
                                  std::to_string(static_cast<int>(opParamInfo.metadata.desc->GetDataType())), "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedQ.desc != nullptr && opParamInfo.seqUsedQ.tensor != nullptr &&
        opParamInfo.seqUsedQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "seqused_q",
                                  std::to_string(static_cast<int>(opParamInfo.seqUsedQ.desc->GetDataType())), "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedKV.desc != nullptr && opParamInfo.seqUsedKV.tensor != nullptr &&
        opParamInfo.seqUsedKV.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "seqused_kv",
                                  std::to_string(static_cast<int>(opParamInfo.seqUsedKV.desc->GetDataType())), "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.blockTable.desc != nullptr && opParamInfo.blockTable.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "block_table",
                                  std::to_string(static_cast<int>(opParamInfo.blockTable.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensQ.desc != nullptr && opParamInfo.cuSeqlensQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "cu_seqlens_q",
                                  std::to_string(static_cast<int>(opParamInfo.cuSeqlensQ.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensKV.desc != nullptr && opParamInfo.cuSeqlensKV.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "cu_seqlens_kv",
                                  std::to_string(static_cast<int>(opParamInfo.cuSeqlensKV.desc->GetDataType())),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.attnOut.desc != nullptr && opParamInfo.attnOut.desc->GetDataType() != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "attention_out",
                                  std::to_string(static_cast<int>(opParamInfo.attnOut.desc->GetDataType())),
                                  "BFLOAT16");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.lseOut.desc != nullptr && opParamInfo.lseOut.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "softmax_lse",
                                  std::to_string(static_cast<int>(opParamInfo.lseOut.desc->GetDataType())), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckFormat() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    const gert::CompileTimeTensorDesc *requiredDescs[] = {
        opParamInfo.query.desc,    opParamInfo.key.desc,           opParamInfo.value.desc,
        opParamInfo.qDescale.desc, opParamInfo.kDescale.desc,      opParamInfo.vDescale.desc,
        opParamInfo.pScale.desc,   opParamInfo.sparseIndices.desc, opParamInfo.sparseSeqLen.desc,
        opParamInfo.attnOut.desc,  opParamInfo.lseOut.desc,
    };
    const char *requiredNames[] = {"query",          "key",           "value",      "q_descale",
                                   "k_descale",      "v_descale",     "p_scale",    "sparse_indices",
                                   "sparse_seq_len", "attention_out", "softmax_lse"};
    for (size_t i = 0; i < sizeof(requiredDescs) / sizeof(requiredDescs[0]); ++i) {
        if (requiredDescs[i] != nullptr && requiredDescs[i]->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, requiredNames[i],
                                       Ops::Base::ToString(requiredDescs[i]->GetOriginFormat()).c_str(), "ND");
            return ge::GRAPH_FAILED;
        }
    }

    const gert::CompileTimeTensorDesc *optionalDescs[] = {
        opParamInfo.attenMask.desc,  opParamInfo.blockTable.desc,  opParamInfo.metadata.desc,
        opParamInfo.cuSeqlensQ.desc, opParamInfo.cuSeqlensKV.desc, opParamInfo.seqUsedQ.desc,
        opParamInfo.seqUsedKV.desc,
    };
    const char *optionalNames[] = {"atten_mask",    "block_table", "metadata",  "cu_seqlens_q",
                                   "cu_seqlens_kv", "seqused_q",   "seqused_kv"};
    for (size_t i = 0; i < sizeof(optionalDescs) / sizeof(optionalDescs[0]); ++i) {
        if (optionalDescs[i] != nullptr && optionalDescs[i]->GetOriginFormat() != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, optionalNames[i],
                                       Ops::Base::ToString(optionalDescs[i]->GetOriginFormat()).c_str(), "ND");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckScaleDtype() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_FP8) {
        if (CheckInputDtype(opParamInfo.qDescale, "dequantScaleQuery", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.kDescale, "keyAntiquantScale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.vDescale, "valueAntiquantScale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.pScale, "quantScale1", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        if (CheckInputDtype(opParamInfo.qDescale, "dequantScaleQuery", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.kDescale, "keyAntiquantScale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.vDescale, "valueAntiquantScale", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS ||
            CheckInputDtype(opParamInfo.pScale, "quantScale1", ge::DT_FLOAT8_E8M0, "FLOAT8_E8M0") !=
                ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckBlockSize() const
{
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        if (!IsMXFP8SparseBlockSizeSupported(tilingInfo_.qBlockSizeVal) ||
            !IsMXFP8SparseBlockSizeSupported(tilingInfo_.kvBlockSizeVal)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "q_block_size/kv_block_size",
                std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
                "must each be " + std::to_string(BSA_MXFP8_SPARSE_BLOCK_SIZE_128) + " or " +
                    std::to_string(BSA_MXFP8_SPARSE_BLOCK_SIZE_64) + " in quant_mode=2 MXFP8 full-quant scenario");
            return ge::GRAPH_FAILED;
        }
    } else if (tilingInfo_.qBlockSizeVal != BSA_BLOCK_SIZE || tilingInfo_.kvBlockSizeVal != BSA_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
            std::to_string(BSA_BLOCK_SIZE));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckExistence() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.blockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "block_table", "nullptr",
                                              "block_table is required for PA execution path");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensQ.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "cu_seqlens_q", "nullptr",
                                              "cu_seqlens_q is required for TND/NTD query layout "
                                              "with PA BNBD KV-cache");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedKV.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "seqused_kv", "nullptr",
                                              "seqused_kv is required for TND/NTD query layout "
                                              "with PA BNBD KV-cache");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckShapeConsistency() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    const gert::Shape &sparseIndicesShape = opParamInfo.sparseIndices.shape->GetStorageShape();
    const gert::Shape &sparseSeqLenShape = opParamInfo.sparseSeqLen.shape->GetStorageShape();
    if (sparseIndicesShape.GetDimNum() != DIM_NUM_4 || sparseSeqLenShape.GetDimNum() != DIM_NUM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "sparse_indices, sparse_seq_len",
            std::to_string(sparseIndicesShape.GetDimNum()) + "D, " + std::to_string(sparseSeqLenShape.GetDimNum()) +
                "D",
            "sparse_indices must be 4D and sparse_seq_len must be 3D");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.bSize == 0U || tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.gSize == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bSize/n1Size/n2Size/gSize",
            std::to_string(tilingInfo_.bSize) + "/" + std::to_string(tilingInfo_.n1Size) + "/" +
                std::to_string(tilingInfo_.n2Size) + "/" + std::to_string(tilingInfo_.gSize),
            "all dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "n1Size (query head num) ", std::to_string(tilingInfo_.n1Size),
            "must be divisible by n2Size (kv head num) " + std::to_string(tilingInfo_.n2Size));
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qBlockSizeVal == 0U || tilingInfo_.kvBlockSizeVal == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
            "block sizes must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qbMax == 0U || tilingInfo_.sparseCount == 0U || tilingInfo_.maxBlockNumPerBatch == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "qbMax/sparseCount/maxBlockNumPerBatch",
            std::to_string(tilingInfo_.qbMax) + "/" + std::to_string(tilingInfo_.sparseCount) + "/" +
                std::to_string(tilingInfo_.maxBlockNumPerBatch),
            "block counts and sparse count must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.dSize != BSA_D_SIZE || tilingInfo_.dSizeV != BSA_D_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "dSize/dSizeV", std::to_string(tilingInfo_.dSize) + "/" + std::to_string(tilingInfo_.dSizeV),
            "only dSize=128 and dSizeV=128 are currently supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckKeyValueShape() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.key.shape == nullptr || opParamInfo.value.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "key/value", "nullptr", "shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();

    if (keyShape.GetDimNum() != DIM_NUM_4 || valueShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "key/value",
            std::to_string(keyShape.GetDimNum()) + "D, " + std::to_string(valueShape.GetDimNum()) + "D",
            "must both be 4D PA BNBD [blockNum, kvHeadNum, blockSize, headDim]");
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(DIM_0) != keyShape.GetDim(DIM_0) || valueShape.GetDim(DIM_1) != keyShape.GetDim(DIM_1) ||
        valueShape.GetDim(DIM_2) != keyShape.GetDim(DIM_2)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "key/value",
                                              Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
                                              "value dim[0..2] must match key dim[0..2] for PA BNBD tensors");
        return ge::GRAPH_FAILED;
    }
    if (keyShape.GetDim(DIM_2) != static_cast<int64_t>(tilingInfo_.kvBlockSizeVal) ||
        valueShape.GetDim(DIM_2) != static_cast<int64_t>(tilingInfo_.kvBlockSizeVal)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "key/value", Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
            "dim[2] (blockSize) must be " + std::to_string(tilingInfo_.kvBlockSizeVal));
        return ge::GRAPH_FAILED;
    }
    if (keyShape.GetDim(DIM_3) != static_cast<int64_t>(tilingInfo_.dSize) ||
        valueShape.GetDim(DIM_3) != static_cast<int64_t>(tilingInfo_.dSizeV)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "key/value",
                                              Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
                                              "key dim[3] must match dSize and value dim[3] must match dSizeV");
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", DIM_0, tilingInfo_.paBlockStrideVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", DIM_1, tilingInfo_.kvBlockSizeVal * tilingInfo_.dSize) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", DIM_2, tilingInfo_.dSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", DIM_3, 1U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", DIM_0, tilingInfo_.paBlockStrideVal) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", DIM_1, tilingInfo_.kvBlockSizeVal * tilingInfo_.dSizeV) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", DIM_2, tilingInfo_.dSizeV) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", DIM_3, 1U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckQuantShape() const
{
    if (tilingInfo_.quantModeVal == BSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        return CheckMXFP8FullQuantShape();
    }

    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.qDescale.shape != nullptr) {
        const gert::Shape &qDescaleShape = opParamInfo.qDescale.shape->GetStorageShape();
        if (qDescaleShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "q_descale", std::to_string(qDescaleShape.GetDimNum()) + "D", "2D");
            return ge::GRAPH_FAILED;
        }
        const bool isTnd = tilingInfo_.layoutQValue == 2U;
        const int64_t expectedDim0 = isTnd ? tilingInfo_.qTokenNum : tilingInfo_.n1Size;
        const int64_t expectedDim1 = isTnd ? tilingInfo_.n1Size : tilingInfo_.qTokenNum;
        if (qDescaleShape.GetDim(DIM_0) != expectedDim0 || qDescaleShape.GetDim(DIM_1) != expectedDim1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "q_descale", Ops::Base::ToString(qDescaleShape),
                                                  "shape does not match query layout");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.kDescale.shape != nullptr) {
        const gert::Shape &kDescaleShape = opParamInfo.kDescale.shape->GetStorageShape();
        if (kDescaleShape.GetDimNum() == DIM_NUM_4) {
            if (kDescaleShape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.paBlockNumSum) ||
                kDescaleShape.GetDim(DIM_1) != static_cast<int64_t>(tilingInfo_.n2Size) ||
                kDescaleShape.GetDim(DIM_2) != static_cast<int64_t>(tilingInfo_.kvBlockSizeVal) ||
                kDescaleShape.GetDim(DIM_3) != 1) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "k_descale", Ops::Base::ToString(kDescaleShape),
                                                      "must be (" + std::to_string(tilingInfo_.paBlockNumSum) + ", " +
                                                          std::to_string(tilingInfo_.n2Size) + ", " +
                                                          std::to_string(tilingInfo_.kvBlockSizeVal) + ", 1)");
                return ge::GRAPH_FAILED;
            }
            if (tilingInfo_.paBlockStrideVal % sizeof(float) != 0U) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, "key stride[0]", std::to_string(tilingInfo_.paBlockStrideVal),
                    "must be divisible by sizeof(float) to align k_descale block stride");
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale", DIM_0,
                               tilingInfo_.paBlockStrideVal / sizeof(float)) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale", DIM_1, tilingInfo_.kvBlockSizeVal) !=
                ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale", DIM_2, 1U) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale", DIM_3, 1U) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "k_descale", std::to_string(kDescaleShape.GetDimNum()) + "D",
                "4D [blockNum, kvHeadNum, blockSize, 1]");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.vDescale.shape != nullptr) {
        const gert::Shape &vDescaleShape = opParamInfo.vDescale.shape->GetStorageShape();
        if (vDescaleShape.GetDimNum() != DIM_NUM_1 ||
            vDescaleShape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.n2Size)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "v_descale", Ops::Base::ToString(vDescaleShape),
                                                  "must be [n2Size]");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.pScale.shape != nullptr) {
        const gert::Shape &pScaleShape = opParamInfo.pScale.shape->GetStorageShape();
        if (pScaleShape.GetShapeSize() != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "p_scale", Ops::Base::ToString(pScaleShape),
                                                  "p_scale must be a scalar (shape size == 1)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckActualSeqLen() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.cuSeqlensQ.tensor != nullptr) {
        const auto *cuSeqlensQShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.cuSeqlensQ.tensor);
        const gert::Shape &shape = cuSeqlensQShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1 || shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize) + 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "cu_seqlens_q", Ops::Base::ToString(shape),
                                                  "must be 1D with dim[0] == B+1");
            return ge::GRAPH_FAILED;
        }
    }
    if (opParamInfo.cuSeqlensKV.tensor != nullptr) {
        const auto *cuSeqlensKVShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.cuSeqlensKV.tensor);
        const gert::Shape &shape = cuSeqlensKVShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1 || shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize) + 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "cu_seqlens_kv", Ops::Base::ToString(shape),
                                                  "must be 1D with dim[0] == B+1");
            return ge::GRAPH_FAILED;
        }
    }
    if (opParamInfo.seqUsedQ.tensor != nullptr) {
        const auto *seqUsedQShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.seqUsedQ.tensor);
        const gert::Shape &shape = seqUsedQShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1 || shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "seqused_q", Ops::Base::ToString(shape),
                                                  "must be 1D with dim[0] == B");
            return ge::GRAPH_FAILED;
        }
    }
    if (opParamInfo.seqUsedKV.tensor != nullptr) {
        const auto *seqUsedKVShape = reinterpret_cast<const gert::StorageShape *>(opParamInfo.seqUsedKV.tensor);
        const gert::Shape &shape = seqUsedKVShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1 || shape.GetDim(DIM_0) != static_cast<int64_t>(tilingInfo_.bSize)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "seqused_kv", Ops::Base::ToString(shape),
                                                  "must be 1D with dim[0] == B");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckAttenMask() const
{
    if (tilingInfo_.maskModeVal != BSA_MASK_MODE_NONE && tilingInfo_.maskModeVal != BSA_MASK_MODE_CAUSAL) {
        OP_LOGE_WITH_INVALID_ATTR(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal), "0 or 3");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.maskModeVal == BSA_MASK_MODE_CAUSAL) {
        const auto &opParamInfo = tilingInfo_.opParamInfo;
        if (opParamInfo.attenMask.shape == nullptr ||
            opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "atten_mask", "nullptr",
                                                  "atten_mask is required when mask_mode=3");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &attenMaskShape = opParamInfo.attenMask.shape->GetStorageShape();
        if (attenMaskShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "atten_mask", std::to_string(attenMaskShape.GetDimNum()) + "D", "2D");
            return ge::GRAPH_FAILED;
        }
        if (attenMaskShape.GetDim(DIM_0) != BSA_ATTEN_MASK_DIM_VALUE ||
            attenMaskShape.GetDim(DIM_1) != BSA_ATTEN_MASK_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "atten_mask", Ops::Base::ToString(attenMaskShape),
                                                  "must be (2048, 2048)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::CheckMXFP8FullQuantShape() const
{
    const auto &opParamInfo = tilingInfo_.opParamInfo;
    if (CheckMXFP8FullQuantScaleInputs(opParamInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();
    if (keyShape.GetDimNum() != DIM_NUM_4 || valueShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "key/value",
            std::to_string(keyShape.GetDimNum()) + "D, " + std::to_string(valueShape.GetDimNum()) + "D",
            "must both be 4D PA BNBD [blockNum, kvHeadNum, blockSize, headDim] in quant_mode=2 MXFP8 "
            "full-quant scenario");
        return ge::GRAPH_FAILED;
    }

    const int64_t scaleDSize = static_cast<int64_t>(BSACeilDiv(tilingInfo_.dSize, BSA_MXFP8_SCALE_GROUP_SIZE));
    const int64_t valueScaleBlockSize =
        static_cast<int64_t>(BSACeilDiv(tilingInfo_.kvBlockSizeVal, BSA_MXFP8_SCALE_GROUP_SIZE));
    const std::array<int64_t, DIM_NUM_1> pScaleShape = {BSA_MXFP8_P_SCALE_SIZE};
    const std::array<int64_t, DIM_NUM_4> queryAntiquantScaleShape = {static_cast<int64_t>(tilingInfo_.qTokenNum),
                                                                     static_cast<int64_t>(tilingInfo_.n1Size),
                                                                     scaleDSize, BSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_4> keyShapeExpect = {
        static_cast<int64_t>(tilingInfo_.paBlockNumSum), static_cast<int64_t>(tilingInfo_.n2Size),
        static_cast<int64_t>(tilingInfo_.kvBlockSizeVal), static_cast<int64_t>(tilingInfo_.dSize)};
    const std::array<int64_t, DIM_NUM_4> valueShapeExpect = {
        static_cast<int64_t>(tilingInfo_.paBlockNumSum), static_cast<int64_t>(tilingInfo_.n2Size),
        static_cast<int64_t>(tilingInfo_.kvBlockSizeVal), static_cast<int64_t>(tilingInfo_.dSizeV)};
    const std::array<int64_t, DIM_NUM_5> keyAntiquantScaleShape = {
        static_cast<int64_t>(tilingInfo_.paBlockNumSum), static_cast<int64_t>(tilingInfo_.n2Size),
        static_cast<int64_t>(tilingInfo_.kvBlockSizeVal), scaleDSize, BSA_MXFP8_SCALE_LAST_DIM};
    const std::array<int64_t, DIM_NUM_5> valueAntiquantScaleShape = {
        static_cast<int64_t>(tilingInfo_.paBlockNumSum), static_cast<int64_t>(tilingInfo_.n2Size), valueScaleBlockSize,
        static_cast<int64_t>(tilingInfo_.dSizeV), BSA_MXFP8_SCALE_LAST_DIM};

    if (CheckInputShape(opParamInfo.pScale, "quantScale1", pScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.qDescale, "dequantScaleQuery", queryAntiquantScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.key, "key", keyShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.value, "value", valueShapeExpect) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.kDescale, "keyAntiquantScale", keyAntiquantScaleShape) != ge::GRAPH_SUCCESS ||
        CheckInputShape(opParamInfo.vDescale, "valueAntiquantScale", valueAntiquantScaleShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnCheck::Process()
{
    if (CheckAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScaleDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckKeyValueShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckActualSeqLen() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttenMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
