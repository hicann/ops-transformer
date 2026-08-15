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
 * \file quant_block_sparse_attn_info_parser.cpp
 * \brief QuantBlockSparseAttn parameter parsing implementation.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "quant_block_sparse_attn_info_parser.h"
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace optiling {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
} // namespace

QuantBlockSparseAttnInfoParser::QuantBlockSparseAttnInfoParser(gert::TilingContext *context)
    : context_(context)
{}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseQuery(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                           const gert::Shape &queryShape,
                                                           const gert::Shape &sparseIndicesShape)
{
    const size_t queryDimNum = queryShape.GetDimNum();
    const std::string &layoutQ = tilingInfo.layoutQStr;

    QBSALayout qLayout = QBSALayout::TND;
    if (queryDimNum == DIM_NUM_3 && layoutQ == "TND") {
        qLayout = QBSALayout::TND;
    } else if (queryDimNum == DIM_NUM_3 && layoutQ == "NTD") {
        qLayout = QBSALayout::NTD;
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query",
                                                 std::to_string(queryDimNum) + "D with layout " + layoutQ,
                                                 "3D with layout TND or 3D with layout NTD");
        return ge::GRAPH_FAILED;
    }

    tilingInfo.layoutQValue = static_cast<uint32_t>(qLayout);

    if (!QBSAGetDimAsU32(sparseIndicesShape, QBSAGetSparseIndicesAxisIdx(QBSAAxis::B), tilingInfo.bSize) ||
        !QBSAGetDimAsU32(queryShape, QBSAGetAxisIdx(qLayout, QBSAAxis::T), tilingInfo.qTokenNum) ||
        !QBSAGetDimAsU32(queryShape, QBSAGetAxisIdx(qLayout, QBSAAxis::N), tilingInfo.n1Size) ||
        !QBSAGetDimAsU32(queryShape, QBSAGetAxisIdx(qLayout, QBSAAxis::D), tilingInfo.dSize)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query/sparse_indices",
                                                 std::to_string(queryDimNum) + "D with layout " + layoutQ,
                                                 "failed to get query/sparse_indices dimensions");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.dSize != QBSA_D_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "query head_dim (dSize)", std::to_string(tilingInfo.dSize),
                                  std::to_string(QBSA_D_SIZE));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseKeyValue(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                              const gert::Shape &keyShape,
                                                              const gert::Shape &valueShape,
                                                              const gert::Shape &kDescaleShape,
                                                              const gert::Stride *keyStride)
{
    if (keyShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyShape.GetDimNum()) + "D",
                                                 "4D PA BNBD [blockNum, kvHeadNum, blockSize, headDim]");
        return ge::GRAPH_FAILED;
    }

    if (!QBSAGetDimAsU32(keyShape, QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N), tilingInfo.n2Size) ||
        !QBSAGetDimAsU32(keyShape, QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM),
                         tilingInfo.paBlockNumSum)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyShape.GetDimNum()) + "D",
                                                 "failed to get n2Size/paBlockNumSum from key shape");
        return ge::GRAPH_FAILED;
    }

    uint64_t paBlockStride = 0U;
    if (context_->InputIsView(QBSA_KEY_INDEX) && keyStride != nullptr) {
        tilingInfo.hasViewStride = true;
        paBlockStride = keyStride->GetStride(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM));
        if (tilingInfo.quantModeVal == QBSA_QUANT_MODE_FP8 && valueShape.GetDimNum() == DIM_NUM_4 &&
            kDescaleShape.GetDimNum() == DIM_NUM_4) {
            const uint64_t keyBlockBytes =
                static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
                static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
                static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)));
            const uint64_t valueBlockBytes =
                static_cast<uint64_t>(valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
                static_cast<uint64_t>(valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
                static_cast<uint64_t>(valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)));
            const uint64_t kDescaleBlockBytes =
                static_cast<uint64_t>(kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
                static_cast<uint64_t>(kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
                static_cast<uint64_t>(kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM))) *
                sizeof(float);
            const uint64_t expectedPaBlockStride = keyBlockBytes + valueBlockBytes + kDescaleBlockBytes;
            if (paBlockStride != expectedPaBlockStride) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, "key stride[0]", std::to_string(paBlockStride),
                    "Must be equal to K/V/k_descale concatenated physical block size " +
                        std::to_string(expectedPaBlockStride));
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        tilingInfo.hasViewStride = false;
        paBlockStride =
            static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
            static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
            static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)));
        OP_LOGD(kOpName, "key is not a view, treat as contiguous, paBlockStride=%llu", paBlockStride);
    }
    if (paBlockStride == 0U || paBlockStride > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "key stride[0]", std::to_string(paBlockStride),
                                              "Must be in range (0, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.paBlockStrideVal = static_cast<uint32_t>(paBlockStride);
    if (!QBSAGetDimAsU32(keyShape, QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE),
                         tilingInfo.paBlockSizeVal)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyShape.GetDimNum()) + "D",
                                                 "failed to get paBlockSize from key shape dim[2]");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.n2Size == 0U || tilingInfo.n1Size % tilingInfo.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "n1Size (query head num)", std::to_string(tilingInfo.n1Size),
            "Must be divisible by n2Size (kv head num) " + std::to_string(tilingInfo.n2Size));
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gSize = tilingInfo.n1Size / tilingInfo.n2Size;
    tilingInfo.isGqa = (tilingInfo.gSize > 1U);

    tilingInfo.dSizeV = QBSA_D_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseSparseIndices(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                                   const gert::Shape &sparseIndicesShape)
{
    if (!QBSAGetDimAsU32(sparseIndicesShape, QBSAGetSparseIndicesAxisIdx(QBSAAxis::QB), tilingInfo.qbMax) ||
        !QBSAGetDimAsU32(sparseIndicesShape, QBSAGetSparseIndicesAxisIdx(QBSAAxis::KB), tilingInfo.sparseCount)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "sparse_indices",
                                                 std::to_string(sparseIndicesShape.GetDimNum()) + "D",
                                                 "failed to get max_Qb/max_Kb from sparse_indices shape");
        return ge::GRAPH_FAILED;
    }
    const uint64_t qSeqUpperBound = static_cast<uint64_t>(tilingInfo.qbMax) * tilingInfo.qBlockSizeVal;
    if (qSeqUpperBound > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "sparse_indices.shape[2] * sparse_q_block_size",
                                              std::to_string(qSeqUpperBound), "Must be in range [0, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseOptionalInputs(QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.cuSeqlensQ.desc = context_->GetOptionalInputDesc(QBSA_CU_SEQLENS_Q_INDEX);
    const gert::StorageShape *cuSeqlensQShape = context_->GetOptionalInputShape(QBSA_CU_SEQLENS_Q_INDEX);
    opParamInfo.cuSeqlensQ.tensor =
        (cuSeqlensQShape != nullptr && cuSeqlensQShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(cuSeqlensQShape) :
            nullptr;

    opParamInfo.cuSeqlensKV.desc = context_->GetOptionalInputDesc(QBSA_CU_SEQLENS_KV_INDEX);
    const gert::StorageShape *cuSeqlensKVShape = context_->GetOptionalInputShape(QBSA_CU_SEQLENS_KV_INDEX);
    opParamInfo.cuSeqlensKV.tensor =
        (cuSeqlensKVShape != nullptr && cuSeqlensKVShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(cuSeqlensKVShape) :
            nullptr;

    opParamInfo.seqUsedQ.desc = context_->GetOptionalInputDesc(QBSA_SEQUSED_Q_INDEX);
    const gert::StorageShape *seqUsedQShape = context_->GetOptionalInputShape(QBSA_SEQUSED_Q_INDEX);
    opParamInfo.seqUsedQ.tensor = (seqUsedQShape != nullptr && seqUsedQShape->GetStorageShape().GetShapeSize() > 0) ?
                                      reinterpret_cast<const gert::Tensor *>(seqUsedQShape) :
                                      nullptr;

    opParamInfo.seqUsedKV.desc = context_->GetOptionalInputDesc(QBSA_SEQUSED_KV_INDEX);
    const gert::StorageShape *seqUsedKVShape = context_->GetOptionalInputShape(QBSA_SEQUSED_KV_INDEX);
    opParamInfo.seqUsedKV.tensor = (seqUsedKVShape != nullptr && seqUsedKVShape->GetStorageShape().GetShapeSize() > 0) ?
                                       reinterpret_cast<const gert::Tensor *>(seqUsedKVShape) :
                                       nullptr;

    opParamInfo.blockTable.desc = context_->GetOptionalInputDesc(QBSA_BLOCK_TABLE_INDEX);
    const gert::StorageShape *blockTableStorageShape = context_->GetOptionalInputShape(QBSA_BLOCK_TABLE_INDEX);
    opParamInfo.blockTable.tensor =
        (blockTableStorageShape != nullptr && blockTableStorageShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(blockTableStorageShape) :
            nullptr;
    if (opParamInfo.blockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "block_table", "nullptr",
                                              "Block_table is required to derive max_block_num_per_batch");
        return ge::GRAPH_FAILED;
    }

    opParamInfo.metadata.desc = context_->GetOptionalInputDesc(QBSA_METADATA_INDEX);
    const gert::StorageShape *metadataShape = context_->GetOptionalInputShape(QBSA_METADATA_INDEX);
    opParamInfo.metadata.tensor = (metadataShape != nullptr && metadataShape->GetStorageShape().GetShapeSize() > 0) ?
                                      reinterpret_cast<const gert::Tensor *>(metadataShape) :
                                      nullptr;

    if (tilingInfo.quantModeVal == QBSA_QUANT_MODE_FP8) {
        const QBSAOptionalParaInfo *requiredInputs[] = {
            &opParamInfo.blockTable,
            &opParamInfo.cuSeqlensQ,
            &opParamInfo.seqUsedKV,
            &opParamInfo.metadata,
        };
        const char *requiredInputNames[] = {
            "block_table",
            "cu_seqlens_q",
            "seqused_kv",
            "metadata",
        };
        for (size_t i = 0U; i < sizeof(requiredInputs) / sizeof(requiredInputs[0]); ++i) {
            if (requiredInputs[i]->desc == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, requiredInputNames[i], "nullptr",
                    "The input tensor must be provided when quant_mode is 1.");
                return ge::GRAPH_FAILED;
            }
            if (requiredInputs[i]->tensor == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, requiredInputNames[i], "empty",
                    "The input tensor cannot be empty when quant_mode is 1.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    const gert::Shape &blockTableShape = blockTableStorageShape->GetStorageShape();
    uint32_t blockTableB = 0;
    if (blockTableShape.GetDimNum() != DIM_NUM_2 ||
        !QBSAGetDimAsU32(blockTableShape, QBSAGetBlockTableAxisIdx(QBSAAxis::B), blockTableB) ||
        !QBSAGetDimAsU32(blockTableShape, QBSAGetBlockTableAxisIdx(QBSAAxis::MAX_BLOCK_NUM),
                         tilingInfo.maxBlockNumPerBatch) ||
        blockTableB != tilingInfo.bSize) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "block_table", std::to_string(blockTableShape.GetDimNum()) + "D",
            "2D [B=" + std::to_string(tilingInfo.bSize) + ", maxBlockNumPerBatch]");
        return ge::GRAPH_FAILED;
    }
    const uint64_t kvSeqUpperBound =
        static_cast<uint64_t>(tilingInfo.maxBlockNumPerBatch) * tilingInfo.kvBlockSizeVal;
    if (kvSeqUpperBound > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "block_table.shape[1] * sparse_kv_block_size",
                                              std::to_string(kvSeqUpperBound), "Must be in range [0, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseAttributes(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                                const gert::RuntimeAttrs *attrs)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.qBlockSize = attrs->GetAttrPointer<int64_t>(QBSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo.kvBlockSize = attrs->GetAttrPointer<int64_t>(QBSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX);
    tilingInfo.qBlockSizeVal = QBSAGetPositiveAttr(attrs, QBSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX, QBSA_BLOCK_SIZE);
    tilingInfo.kvBlockSizeVal = QBSAGetPositiveAttr(attrs, QBSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX, QBSA_BLOCK_SIZE);

    opParamInfo.maskMode = attrs->GetAttrPointer<int64_t>(QBSA_MASK_MODE_ATTR_INDEX);
    opParamInfo.returnSoftmaxLse = attrs->GetAttrPointer<bool>(QBSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    opParamInfo.layoutQ = attrs->GetAttrPointer<char>(QBSA_LAYOUT_Q_ATTR_INDEX);
    opParamInfo.layoutKV = attrs->GetAttrPointer<char>(QBSA_LAYOUT_KV_ATTR_INDEX);
    opParamInfo.layoutSparseIndices = attrs->GetAttrPointer<char>(QBSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX);
    opParamInfo.quantMode = attrs->GetAttrPointer<int64_t>(QBSA_QUANT_MODE_ATTR_INDEX);

    if (opParamInfo.quantMode != nullptr && *opParamInfo.quantMode != static_cast<int64_t>(QBSA_QUANT_MODE_FP8) &&
        *opParamInfo.quantMode != static_cast<int64_t>(QBSA_QUANT_MODE_MXFP8_FULL_QUANT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quant_mode", std::to_string(*opParamInfo.quantMode),
                                              "Must be 1 or 2");
        return ge::GRAPH_FAILED;
    }

    tilingInfo.softmaxScaleVal = QBSAGetFloatAttr(attrs, QBSA_SOFTMAX_SCALE_ATTR_INDEX, 1.0F);
    tilingInfo.maskModeVal = QBSAGetUintAttr(attrs, QBSA_MASK_MODE_ATTR_INDEX, 0U);
    tilingInfo.quantModeVal = QBSAGetUintAttr(attrs, QBSA_QUANT_MODE_ATTR_INDEX, QBSA_QUANT_MODE_FP8);
    tilingInfo.layoutQStr = QBSAGetStringAttr(attrs, QBSA_LAYOUT_Q_ATTR_INDEX, "TND");
    tilingInfo.layoutKVStr = QBSAGetStringAttr(attrs, QBSA_LAYOUT_KV_ATTR_INDEX, "PA_BNBD");
    tilingInfo.layoutSparseIndicesStr = QBSAGetStringAttr(attrs, QBSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX, "B_N_Qb_Kb");
    tilingInfo.layoutOutStr = QBSAGetStringAttr(attrs, QBSA_LAYOUT_OUT_ATTR_INDEX, "TND");
    tilingInfo.returnSoftmaxLseVal = QBSAGetBoolAttr(attrs, QBSA_RETURN_SOFTMAX_LSE_ATTR_INDEX, false);

    opParamInfo.query.desc = context_->GetInputDesc(QBSA_QUERY_INDEX);
    tilingInfo.qDtype =
        (opParamInfo.query.desc != nullptr) ? opParamInfo.query.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.key.desc = context_->GetInputDesc(QBSA_KEY_INDEX);
    tilingInfo.kvDtype = (opParamInfo.key.desc != nullptr) ? opParamInfo.key.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.value.desc = context_->GetInputDesc(QBSA_VALUE_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::Parse(QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    if (context_ == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "tiling context", "nullptr", "Context is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "attrs", "nullptr", "Attrs is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.query.shape = context_->GetInputShape(QBSA_QUERY_INDEX);
    opParamInfo.key.shape = context_->GetInputShape(QBSA_KEY_INDEX);
    opParamInfo.key.stride = context_->GetInputStride(QBSA_KEY_INDEX);
    opParamInfo.value.shape = context_->GetInputShape(QBSA_VALUE_INDEX);
    opParamInfo.value.stride = context_->GetInputStride(QBSA_VALUE_INDEX);
    opParamInfo.qDescale.desc = context_->GetInputDesc(QBSA_Q_DESCALE_INDEX);
    opParamInfo.qDescale.shape = context_->GetInputShape(QBSA_Q_DESCALE_INDEX);
    opParamInfo.kDescale.desc = context_->GetInputDesc(QBSA_K_DESCALE_INDEX);
    opParamInfo.kDescale.shape = context_->GetInputShape(QBSA_K_DESCALE_INDEX);
    opParamInfo.kDescale.stride = context_->GetInputStride(QBSA_K_DESCALE_INDEX);
    opParamInfo.vDescale.desc = context_->GetInputDesc(QBSA_V_DESCALE_INDEX);
    opParamInfo.vDescale.shape = context_->GetInputShape(QBSA_V_DESCALE_INDEX);
    opParamInfo.pScale.desc = context_->GetOptionalInputDesc(QBSA_P_SCALE_INDEX);
    opParamInfo.pScale.shape = context_->GetOptionalInputShape(QBSA_P_SCALE_INDEX);
    opParamInfo.sparseIndices.desc = context_->GetInputDesc(QBSA_SPARSE_INDICES_INDEX);
    opParamInfo.sparseIndices.shape = context_->GetInputShape(QBSA_SPARSE_INDICES_INDEX);
    opParamInfo.sparseSeqLen.desc = context_->GetInputDesc(QBSA_SPARSE_SEQ_LEN_INDEX);
    opParamInfo.sparseSeqLen.shape = context_->GetInputShape(QBSA_SPARSE_SEQ_LEN_INDEX);
    opParamInfo.attenMask.desc = context_->GetOptionalInputDesc(QBSA_ATTEN_MASK_INDEX);
    opParamInfo.attenMask.shape = context_->GetOptionalInputShape(QBSA_ATTEN_MASK_INDEX);
    opParamInfo.attnOut.desc = context_->GetOutputDesc(QBSA_ATTENTION_OUT_INDEX);
    opParamInfo.attnOut.shape = context_->GetOutputShape(QBSA_ATTENTION_OUT_INDEX);
    opParamInfo.lseOut.desc = context_->GetOutputDesc(QBSA_SOFTMAX_LSE_INDEX);
    opParamInfo.lseOut.shape = context_->GetOutputShape(QBSA_SOFTMAX_LSE_INDEX);

    if (ParseAttributes(tilingInfo, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.quantModeVal == QBSA_QUANT_MODE_FP8) {
        const QBSARequiredParaInfo *requiredInputs[] = {
            &opParamInfo.query,
            &opParamInfo.key,
            &opParamInfo.value,
            &opParamInfo.qDescale,
            &opParamInfo.kDescale,
            &opParamInfo.vDescale,
            &opParamInfo.sparseIndices,
            &opParamInfo.sparseSeqLen,
        };
        const char *requiredInputNames[] = {
            "query",
            "key",
            "value",
            "q_descale",
            "k_descale",
            "v_descale",
            "sparse_indices",
            "sparse_seq_len",
        };
        for (size_t i = 0U; i < sizeof(requiredInputs) / sizeof(requiredInputs[0]); ++i) {
            if (requiredInputs[i]->desc == nullptr || requiredInputs[i]->shape == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, requiredInputNames[i], "nullptr",
                    "The input tensor must be provided when quant_mode is 1.");
                return ge::GRAPH_FAILED;
            }
            if (requiredInputs[i]->shape->GetStorageShape().GetShapeSize() <= 0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, requiredInputNames[i], "empty",
                    "The input tensor cannot be empty when quant_mode is 1.");
                return ge::GRAPH_FAILED;
            }
            const gert::Shape &inputShape = requiredInputs[i]->shape->GetStorageShape();
            for (size_t dim = 0U; dim < inputShape.GetDimNum(); ++dim) {
                if (inputShape.GetDim(dim) <= 0) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        kOpName, requiredInputNames[i], std::to_string(inputShape.GetDim(dim)),
                        "Every dimension must be greater than 0 in quant_mode=1");
                    return ge::GRAPH_FAILED;
                }
            }
        }
        if (opParamInfo.key.stride == nullptr || opParamInfo.value.stride == nullptr ||
            opParamInfo.key.stride->GetDimNum() == 0U || opParamInfo.value.stride->GetDimNum() == 0U) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "key/value stride[0]", "nullptr",
                "The stride values of key and value must be provided when quant_mode is 1.");
            return ge::GRAPH_FAILED;
        }
        if (opParamInfo.key.stride->GetStride(0U) != opParamInfo.value.stride->GetStride(0U)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "key/value stride[0]",
                std::to_string(opParamInfo.key.stride->GetStride(0U)) + "/" +
                    std::to_string(opParamInfo.value.stride->GetStride(0U)),
                "The first stride values of key and value must be equal when quant_mode is 1.");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.query.shape == nullptr || opParamInfo.key.shape == nullptr || opParamInfo.value.shape == nullptr ||
        opParamInfo.qDescale.shape == nullptr || opParamInfo.kDescale.shape == nullptr ||
        opParamInfo.vDescale.shape == nullptr ||
        opParamInfo.sparseIndices.shape == nullptr || opParamInfo.sparseSeqLen.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "required input shape", "nullptr",
                                              "Query/key/value/scale/sparse input shape must not be nullptr");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &queryShape = opParamInfo.query.shape->GetStorageShape();
    const gert::Shape &keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &valueShape = opParamInfo.value.shape->GetStorageShape();
    const gert::Shape &kDescaleShape = opParamInfo.kDescale.shape->GetStorageShape();
    const gert::Shape &sparseIndicesShape = opParamInfo.sparseIndices.shape->GetStorageShape();

    if (ParseQuery(tilingInfo, queryShape, sparseIndicesShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseKeyValue(tilingInfo, keyShape, valueShape, kDescaleShape, opParamInfo.key.stride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseSparseIndices(tilingInfo, sparseIndicesShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseOptionalInputs(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gS1OuterSize = tilingInfo.qbMax;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
