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
 * \file nsa_selected_attention_infer_tiling_check.cpp
 * \brief
 */

#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "nsa_selected_attention_infer_tiling.h"
#include "nsa_selected_attention_infer_tiling_base.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus NsaSelectTiling::CheckBaseInputsNull() const
{
    // Check base input tensors
    OP_CHECK_IF(
        context_->query.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->opName, "query", "0", "The query tensor must be nonempty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->query.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "query",
                                                         "The tiling context is missing the query descriptor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->key.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "key",
                                                         "The tiling context is missing the key descriptor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->value.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "value",
                                                         "The tiling context is missing the value shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->value.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "value",
                                                         "The tiling context is missing the value descriptor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->topkIndices.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "topk_indices",
                                                         "The tiling context is missing the topk_indices shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->topkIndices.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "topk_indices",
                                                         "The tiling context is missing the topk_indices descriptor"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        context_->actualKVSeqLengths.tensor == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "actual_kv_seq_lengths",
                                                 "The tiling context is missing the actual_kv_seq_lengths tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->actualQSeqLengths.tensor == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "actual_q_seq_lengths",
                                                 "The tiling context is missing the actual_q_seq_lengths tensor"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockTable.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "block_table",
                                                         "The tiling context is missing the block_table tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockTable.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "block_table",
                                                         "The tiling context is missing the block_table descriptor"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->attenOut.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "attention_out",
                                                         "The tiling context is missing the attention_out descriptor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->attenOut.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "attention_out",
                                                         "The tiling context is missing the attention_out shape"),
                return ge::GRAPH_FAILED);

    const std::pair<const void*, const char*> requiredAttrs[] = {
        {context_->layOut, "input_layout"},
        {context_->numHeads, "num_heads"},
        {context_->kvHeadNums, "num_key_value_heads"},
        {context_->selectedBlockSize, "selected_block_size"},
        {context_->selectedBlockCount, "selected_block_count"},
        {context_->blockSize, "block_size"},
        {context_->scaleValue, "scale_value"},
    };
    for (const auto& attr : requiredAttrs) {
        OP_CHECK_IF(attr.first == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context_->opName, attr.second,
                        std::string("The tiling context is missing the ") + attr.second + " attribute"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckInputFormatAndLimits() const
{
    OP_CHECK_IF(((inputQType_ != inputKvType_)),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->opName, "query, key",
                                                       ge::TypeUtils::DataTypeToSerialString(inputQType_) + ", " +
                                                           ge::TypeUtils::DataTypeToSerialString(inputKvType_),
                                                       "Query, key and value must use the same dtype"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "block_size", std::to_string(blockSize_),
                                              "For FLOAT16 or BFLOAT16 KV cache, block_size must be a multiple of 16"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (batchSize_ > 3072),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->opName, "query batch size", std::to_string(batchSize_),
                                                  "The batch size must not exceed 3072"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((blockSize_ != 64) && (blockSize_ != 128),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "block_size", std::to_string(blockSize_),
                                                      "Supported block sizes are 64 and 128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockSize_ % 16 != 0 || selectedBlockSize_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "selected_block_size",
                                                      std::to_string(selectedBlockSize_),
                                                      "The selected block size must be nonzero and a multiple of 16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockSize_ > 128,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "selected_block_size",
                                                      std::to_string(selectedBlockSize_),
                                                      "The selected block size must not exceed 128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDim_ != 192),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->opName, "query, key",
                                                       std::to_string(headDim_) + ", " + std::to_string(headDim_),
                                                       "Query and key head dimensions must be 192"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDim_ <= headDimV_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->opName, "query, value", std::to_string(headDim_) + ", " + std::to_string(headDimV_),
                    "The query head dimension must be greater than the value head dimension"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (numKvHeads_ > 256),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "num_key_value_heads", std::to_string(numKvHeads_),
                                              "The number of key/value heads must not exceed 256"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDimV_ != 128),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, "value", std::to_string(headDimV_),
                                                      "The value head dimension must be 128"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckQKOutShape()
{
    const gert::StorageShape* queryShape = context_->query.shape;
    const gert::StorageShape* keyShape = context_->key.shape;
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH") {
        OP_CHECK_IF((dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != DIM_BSH),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->opName, "query, key, attention_out",
                        std::to_string(dimOfQ) + ", " + std::to_string(dimOfK) + ", " + std::to_string(dimOfOut),
                        "BSH requires rank 3 for query, key and attention_out"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            queryShape->GetStorageShape().GetDim(2) / numHeads_ != keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->opName, "query, key",
                std::to_string(queryShape->GetStorageShape().GetDim(2) / numHeads_) + ", " +
                    std::to_string(keyShape->GetStorageShape().GetDim(2) / numKvHeads_),
                "BSH query and key head dimensions must match"),
            return ge::GRAPH_FAILED);
    } else if (inputLayoutStr == "TND") {
        return TNDCheckQKOutShape();
    } else {
        OP_CHECK_IF((dimOfQ != DIM_BNSD_OR_BNSD) || (dimOfK != DIM_BNSD_OR_BNSD) || (dimOfOut != DIM_BNSD_OR_BNSD),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->opName, "query, key, attention_out",
                        std::to_string(dimOfQ) + ", " + std::to_string(dimOfK) + ", " + std::to_string(dimOfOut),
                        "BSND requires rank 4 for query, key and attention_out"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->opName, "query, key",
                                                   std::to_string(queryShape->GetStorageShape().GetDim(3)) + ", " +
                                                       std::to_string(keyShape->GetStorageShape().GetDim(3)),
                                                   "BSND query and key head dimensions must match"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::TNDCheckQKOutShape() const
{
    const gert::StorageShape* queryShape = context_->query.shape;
    const gert::StorageShape* keyShape = context_->key.shape;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((dimOfQ != DIM_TND) || (dimOfK != DIM_TND) || (dimOfOut != DIM_TND),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->opName, "query, key, attention_out",
                    std::to_string(dimOfQ) + ", " + std::to_string(dimOfK) + ", " + std::to_string(dimOfOut),
                    "TND requires rank 3 for query, key and attention_out"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(2) != keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->opName, "query, key",
                    std::to_string(queryShape->GetStorageShape().GetDim(2)) + ", " +
                        std::to_string(keyShape->GetStorageShape().GetDim(2) / numKvHeads_),
                    "TND query head dimension must equal key hidden size divided by num_key_value_heads"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckUbSpace()
{
    if (!CalcUbBmm()) {
        return false;
    }
    return true;
}

} // namespace optiling
