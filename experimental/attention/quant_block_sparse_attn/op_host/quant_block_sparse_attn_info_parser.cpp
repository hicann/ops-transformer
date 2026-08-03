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
constexpr size_t DIM_0 = 0U;
constexpr size_t DIM_TND_T = 0U;
constexpr size_t DIM_TND_N = 1U;
constexpr size_t DIM_TND_D = 2U;
constexpr size_t DIM_NTD_N = 0U;
constexpr size_t DIM_NTD_T = 1U;
constexpr size_t DIM_NTD_D = 2U;
constexpr size_t DIM_PA_N = 1U;
constexpr size_t DIM_PA_BLOCK_SIZE = 2U;
constexpr size_t DIM_BLOCK_TABLE_MAX = 1U;
constexpr size_t DIM_SPARSE_QB = 2U;
constexpr size_t DIM_SPARSE_COUNT = 3U;
constexpr size_t DIM_KV_HEAD_DIM = 3U;
constexpr uint32_t BSA_LAYOUT_Q_TND_VALUE = 2U;
constexpr uint32_t BSA_LAYOUT_Q_NTD_VALUE = 5U;
} // namespace

QuantBlockSparseAttnInfoParser::QuantBlockSparseAttnInfoParser(gert::TilingContext *context) : context_(context) {}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseQuery(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                           const gert::Shape &queryShape,
                                                           const gert::Shape &sparseIndicesShape)
{
    const size_t queryDimNum = queryShape.GetDimNum();
    const std::string &layoutQ = tilingInfo.layoutQStr;
    if (queryDimNum == DIM_NUM_3 && layoutQ == "TND") {
        tilingInfo.layoutQValue = BSA_LAYOUT_Q_TND_VALUE;
        if (!BSAGetDimAsU32(sparseIndicesShape, DIM_0, tilingInfo.bSize) ||
            !BSAGetDimAsU32(queryShape, DIM_TND_T, tilingInfo.qTokenNum) ||
            !BSAGetDimAsU32(queryShape, DIM_TND_N, tilingInfo.n1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_TND_D, tilingInfo.dSize)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query/sparse_indices",
                                                     std::to_string(queryDimNum) + "D",
                                                     "failed to get TND query/sparse_indices dimensions");
            return ge::GRAPH_FAILED;
        }
    } else if (queryDimNum == DIM_NUM_3 && layoutQ == "NTD") {
        tilingInfo.layoutQValue = BSA_LAYOUT_Q_NTD_VALUE;
        if (!BSAGetDimAsU32(sparseIndicesShape, DIM_0, tilingInfo.bSize) ||
            !BSAGetDimAsU32(queryShape, DIM_NTD_N, tilingInfo.n1Size) ||
            !BSAGetDimAsU32(queryShape, DIM_NTD_T, tilingInfo.qTokenNum) ||
            !BSAGetDimAsU32(queryShape, DIM_NTD_D, tilingInfo.dSize)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query/sparse_indices",
                                                     std::to_string(queryDimNum) + "D",
                                                     "failed to get NTD query/sparse_indices dimensions");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query",
                                                 std::to_string(queryDimNum) + "D with layout " + layoutQ,
                                                 "3D with layout TND or 3D with layout NTD");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.dSize != BSA_D_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "query head_dim (dSize)", std::to_string(tilingInfo.dSize),
                                  std::to_string(BSA_D_SIZE));
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

    if (!BSAGetDimAsU32(keyShape, DIM_PA_N, tilingInfo.n2Size) ||
        !BSAGetDimAsU32(keyShape, DIM_0, tilingInfo.paBlockNumSum)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyShape.GetDimNum()) + "D",
                                                 "failed to get n2Size/paBlockNumSum from key shape");
        return ge::GRAPH_FAILED;
    }

    uint64_t paBlockStride = 0U;
    if (keyStride == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "key stride", "nullptr",
            "PA_BNSD segmented KV-cache requires 4D non-contiguous key/value/k_descale views");
        return ge::GRAPH_FAILED;
    }

    paBlockStride = keyStride->GetStride(DIM_0);
    if (tilingInfo.quantModeVal == BSA_QUANT_MODE_FP8 && valueShape.GetDimNum() == DIM_NUM_4 &&
        kDescaleShape.GetDimNum() == DIM_NUM_4) {
        const uint64_t keyBlockBytes = static_cast<uint64_t>(keyShape.GetDim(DIM_PA_N)) *
                                       static_cast<uint64_t>(keyShape.GetDim(DIM_PA_BLOCK_SIZE)) *
                                       static_cast<uint64_t>(keyShape.GetDim(DIM_KV_HEAD_DIM));
        const uint64_t valueBlockBytes = static_cast<uint64_t>(valueShape.GetDim(DIM_PA_N)) *
                                         static_cast<uint64_t>(valueShape.GetDim(DIM_PA_BLOCK_SIZE)) *
                                         static_cast<uint64_t>(valueShape.GetDim(DIM_KV_HEAD_DIM));
        const uint64_t kDescaleBlockBytes = static_cast<uint64_t>(kDescaleShape.GetDim(DIM_PA_N)) *
                                            static_cast<uint64_t>(kDescaleShape.GetDim(DIM_PA_BLOCK_SIZE)) *
                                            static_cast<uint64_t>(kDescaleShape.GetDim(DIM_KV_HEAD_DIM)) *
                                            sizeof(float);
        const uint64_t expectedPaBlockStride = keyBlockBytes + valueBlockBytes + kDescaleBlockBytes;
        if (paBlockStride != expectedPaBlockStride) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "key stride[0]", std::to_string(paBlockStride),
                "must be equal to K/V/k_descale concatenated physical block size " +
                    std::to_string(expectedPaBlockStride));
            return ge::GRAPH_FAILED;
        }
    }
    if (paBlockStride == 0U || paBlockStride > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "key stride[0]", std::to_string(paBlockStride),
                                              "must be in range (0, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.paBlockStrideVal = static_cast<uint32_t>(paBlockStride);
    if (!BSAGetDimAsU32(keyShape, DIM_PA_BLOCK_SIZE, tilingInfo.paBlockSizeVal)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "key", std::to_string(keyShape.GetDimNum()) + "D",
                                                 "failed to get paBlockSize from key shape dim[2]");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo.n2Size == 0U || tilingInfo.n1Size % tilingInfo.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "n1Size (query head num)", std::to_string(tilingInfo.n1Size),
            "must be divisible by n2Size (kv head num) " + std::to_string(tilingInfo.n2Size));
        return ge::GRAPH_FAILED;
    }

    tilingInfo.gSize = tilingInfo.n1Size / tilingInfo.n2Size;
    tilingInfo.isGqa = (tilingInfo.gSize > 1U);

    tilingInfo.dSizeV = BSA_D_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseSparseIndices(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                                   const gert::Shape &sparseIndicesShape)
{
    if (!BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_QB, tilingInfo.qbMax) ||
        !BSAGetDimAsU32(sparseIndicesShape, DIM_SPARSE_COUNT, tilingInfo.sparseCount)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "sparse_indices",
                                                 std::to_string(sparseIndicesShape.GetDimNum()) + "D",
                                                 "failed to get max_Qb/max_Kb from sparse_indices shape");
        return ge::GRAPH_FAILED;
    }
    const uint64_t qSeqUpperBound = static_cast<uint64_t>(tilingInfo.qbMax) * tilingInfo.qBlockSizeVal;
    if (qSeqUpperBound > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "sparse_indices.shape[2] * sparse_q_block_size",
                                              std::to_string(qSeqUpperBound), "must be in range [0, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseOptionalInputs(QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.cuSeqlensQ.desc = context_->GetInputDesc(BSA_CU_SEQLENS_Q_INDEX);
    const gert::StorageShape *cuSeqlensQShape = context_->GetOptionalInputShape(BSA_CU_SEQLENS_Q_INDEX);
    opParamInfo.cuSeqlensQ.tensor =
        (cuSeqlensQShape != nullptr && cuSeqlensQShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(cuSeqlensQShape) :
            nullptr;

    opParamInfo.cuSeqlensKV.desc = context_->GetInputDesc(BSA_CU_SEQLENS_KV_INDEX);
    const gert::StorageShape *cuSeqlensKVShape = context_->GetOptionalInputShape(BSA_CU_SEQLENS_KV_INDEX);
    opParamInfo.cuSeqlensKV.tensor =
        (cuSeqlensKVShape != nullptr && cuSeqlensKVShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(cuSeqlensKVShape) :
            nullptr;

    opParamInfo.seqUsedQ.desc = context_->GetInputDesc(BSA_SEQUSED_Q_INDEX);
    const gert::StorageShape *seqUsedQShape = context_->GetOptionalInputShape(BSA_SEQUSED_Q_INDEX);
    opParamInfo.seqUsedQ.tensor = (seqUsedQShape != nullptr && seqUsedQShape->GetStorageShape().GetShapeSize() > 0) ?
                                      reinterpret_cast<const gert::Tensor *>(seqUsedQShape) :
                                      nullptr;

    opParamInfo.seqUsedKV.desc = context_->GetInputDesc(BSA_SEQUSED_KV_INDEX);
    const gert::StorageShape *seqUsedKVShape = context_->GetOptionalInputShape(BSA_SEQUSED_KV_INDEX);
    opParamInfo.seqUsedKV.tensor = (seqUsedKVShape != nullptr && seqUsedKVShape->GetStorageShape().GetShapeSize() > 0) ?
                                       reinterpret_cast<const gert::Tensor *>(seqUsedKVShape) :
                                       nullptr;

    opParamInfo.blockTable.desc = context_->GetInputDesc(BSA_BLOCK_TABLE_INDEX);
    const gert::StorageShape *blockTableStorageShape = context_->GetOptionalInputShape(BSA_BLOCK_TABLE_INDEX);
    opParamInfo.blockTable.tensor =
        (blockTableStorageShape != nullptr && blockTableStorageShape->GetStorageShape().GetShapeSize() > 0) ?
            reinterpret_cast<const gert::Tensor *>(blockTableStorageShape) :
            nullptr;
    if (opParamInfo.blockTable.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "block_table", "nullptr",
                                              "block_table is required to derive max_block_num_per_batch");
        return ge::GRAPH_FAILED;
    }

    opParamInfo.metadata.desc = context_->GetInputDesc(BSA_METADATA_INDEX);
    const gert::StorageShape *metadataShape = context_->GetOptionalInputShape(BSA_METADATA_INDEX);
    opParamInfo.metadata.tensor = (metadataShape != nullptr && metadataShape->GetStorageShape().GetShapeSize() > 0) ?
                                      reinterpret_cast<const gert::Tensor *>(metadataShape) :
                                      nullptr;

    const gert::Shape &blockTableShape = blockTableStorageShape->GetStorageShape();
    uint32_t blockTableB = 0;
    if (blockTableShape.GetDimNum() != DIM_NUM_2 || !BSAGetDimAsU32(blockTableShape, DIM_0, blockTableB) ||
        !BSAGetDimAsU32(blockTableShape, DIM_BLOCK_TABLE_MAX, tilingInfo.maxBlockNumPerBatch) ||
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
                                              std::to_string(kvSeqUpperBound), "must be in range [0, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::ParseAttributes(QuantBlockSparseAttnTilingInfo &tilingInfo,
                                                                const gert::RuntimeAttrs *attrs)
{
    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.qBlockSize = attrs->GetAttrPointer<int64_t>(BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo.kvBlockSize = attrs->GetAttrPointer<int64_t>(BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX);
    tilingInfo.qBlockSizeVal = BSAGetPositiveAttr(attrs, BSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);
    tilingInfo.kvBlockSizeVal = BSAGetPositiveAttr(attrs, BSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX, BSA_BLOCK_SIZE);

    opParamInfo.softmaxScale = attrs->GetAttrPointer<float>(BSA_SOFTMAX_SCALE_ATTR_INDEX);
    opParamInfo.maskMode = attrs->GetAttrPointer<int64_t>(BSA_MASK_MODE_ATTR_INDEX);
    opParamInfo.returnSoftmaxLse = attrs->GetAttrPointer<bool>(BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    opParamInfo.layoutQ = attrs->GetAttrPointer<char>(BSA_LAYOUT_Q_ATTR_INDEX);
    opParamInfo.layoutKV = attrs->GetAttrPointer<char>(BSA_LAYOUT_KV_ATTR_INDEX);
    opParamInfo.layoutSparseIndices = attrs->GetAttrPointer<char>(BSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX);
    opParamInfo.quantMode = attrs->GetAttrPointer<int64_t>(BSA_QUANT_MODE_ATTR_INDEX);

    tilingInfo.softmaxScaleVal = BSAGetFloatAttr(attrs, BSA_SOFTMAX_SCALE_ATTR_INDEX, 1.0F);
    tilingInfo.maskModeVal = BSAGetUintAttr(attrs, BSA_MASK_MODE_ATTR_INDEX, 0U);
    tilingInfo.quantModeVal = BSAGetUintAttr(attrs, BSA_QUANT_MODE_ATTR_INDEX, BSA_QUANT_MODE_FP8);
    tilingInfo.layoutQStr = BSAGetStringAttr(attrs, BSA_LAYOUT_Q_ATTR_INDEX, "TND");
    tilingInfo.layoutKVStr = BSAGetStringAttr(attrs, BSA_LAYOUT_KV_ATTR_INDEX, "PA_BNSD");
    tilingInfo.layoutSparseIndicesStr = BSAGetStringAttr(attrs, BSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX, "B_N_Qb_Kb");
    tilingInfo.layoutOutStr = BSAGetStringAttr(attrs, BSA_LAYOUT_OUT_ATTR_INDEX, "TND");
    tilingInfo.returnSoftmaxLseVal = BSAGetBoolAttr(attrs, BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX, false);

    opParamInfo.query.desc = context_->GetInputDesc(BSA_QUERY_INDEX);
    tilingInfo.qDtype =
        (opParamInfo.query.desc != nullptr) ? opParamInfo.query.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.key.desc = context_->GetInputDesc(BSA_KEY_INDEX);
    tilingInfo.kvDtype = (opParamInfo.key.desc != nullptr) ? opParamInfo.key.desc->GetDataType() : ge::DT_FLOAT8_E4M3FN;
    opParamInfo.value.desc = context_->GetInputDesc(BSA_VALUE_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantBlockSparseAttnInfoParser::Parse(QuantBlockSparseAttnTilingInfo &tilingInfo)
{
    if (context_ == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "tiling context", "nullptr", "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "attrs", "nullptr", "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto &opParamInfo = tilingInfo.opParamInfo;

    opParamInfo.query.shape = context_->GetInputShape(BSA_QUERY_INDEX);
    opParamInfo.key.shape = context_->GetInputShape(BSA_KEY_INDEX);
    opParamInfo.key.stride = context_->GetInputStride(BSA_KEY_INDEX);
    opParamInfo.value.shape = context_->GetInputShape(BSA_VALUE_INDEX);
    opParamInfo.value.stride = context_->GetInputStride(BSA_VALUE_INDEX);
    opParamInfo.qDescale.desc = context_->GetInputDesc(BSA_Q_DESCALE_INDEX);
    opParamInfo.qDescale.shape = context_->GetInputShape(BSA_Q_DESCALE_INDEX);
    opParamInfo.kDescale.desc = context_->GetInputDesc(BSA_K_DESCALE_INDEX);
    opParamInfo.kDescale.shape = context_->GetInputShape(BSA_K_DESCALE_INDEX);
    opParamInfo.kDescale.stride = context_->GetInputStride(BSA_K_DESCALE_INDEX);
    opParamInfo.vDescale.desc = context_->GetInputDesc(BSA_V_DESCALE_INDEX);
    opParamInfo.vDescale.shape = context_->GetInputShape(BSA_V_DESCALE_INDEX);
    opParamInfo.pScale.desc = context_->GetInputDesc(BSA_P_SCALE_INDEX);
    opParamInfo.pScale.shape = context_->GetInputShape(BSA_P_SCALE_INDEX);
    opParamInfo.sparseIndices.desc = context_->GetInputDesc(BSA_SPARSE_INDICES_INDEX);
    opParamInfo.sparseIndices.shape = context_->GetInputShape(BSA_SPARSE_INDICES_INDEX);
    opParamInfo.sparseSeqLen.desc = context_->GetInputDesc(BSA_SPARSE_SEQ_LEN_INDEX);
    opParamInfo.sparseSeqLen.shape = context_->GetInputShape(BSA_SPARSE_SEQ_LEN_INDEX);
    opParamInfo.attenMask.desc = context_->GetInputDesc(BSA_ATTEN_MASK_INDEX);
    opParamInfo.attenMask.shape = context_->GetOptionalInputShape(BSA_ATTEN_MASK_INDEX);
    opParamInfo.attnOut.desc = context_->GetOutputDesc(BSA_ATTENTION_OUT_INDEX);
    opParamInfo.attnOut.shape = context_->GetOutputShape(BSA_ATTENTION_OUT_INDEX);
    opParamInfo.lseOut.desc = context_->GetOutputDesc(BSA_SOFTMAX_LSE_INDEX);
    opParamInfo.lseOut.shape = context_->GetOutputShape(BSA_SOFTMAX_LSE_INDEX);

    if (ParseAttributes(tilingInfo, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (opParamInfo.query.shape == nullptr || opParamInfo.key.shape == nullptr || opParamInfo.value.shape == nullptr ||
        opParamInfo.qDescale.shape == nullptr || opParamInfo.kDescale.shape == nullptr ||
        opParamInfo.vDescale.shape == nullptr || opParamInfo.pScale.shape == nullptr ||
        opParamInfo.sparseIndices.shape == nullptr || opParamInfo.sparseSeqLen.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "required input shape", "nullptr",
                                              "query/key/value/scale/sparse input shape must not be nullptr");
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
