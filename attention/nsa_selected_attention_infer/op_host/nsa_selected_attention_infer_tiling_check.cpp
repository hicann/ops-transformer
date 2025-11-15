/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

ge::graphStatus NsaSelectTiling::CheckPABlockSize() const
{
    OP_CHECK_IF(
        blockSize_ == 0,
        OP_LOGE(context_->opName, "When Page Attention is enabled, input attribute blocksize can not be 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && (blockSize_ % 16 != 0),
        OP_LOGE(context_->opName,
            "When Page Attention is enabled, "
            "if kv cache dtype is float16/bfloat16, input attr blocksize should be 16 aligned"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckBaseInputsNull() const {
    // Check base input tensors
    OP_CHECK_IF(context_->query.shape == nullptr, OP_LOGE(context_->opName, "Shape of tensor query is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->query.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE(context_->opName, "Tensor q is empty cause shapesize is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->query.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor query is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->key.shape == nullptr, OP_LOGE(context_->opName, "Shape of tensor k is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->key.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor k is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->value.shape == nullptr, OP_LOGE(context_->opName, "Shape of tensor value is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->value.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor value is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->topkIndices.shape == nullptr, OP_LOGE(context_->opName, "Shape of tensor topkIndices is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->topkIndices.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor topkIndices is nullptr"),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->actualKVSeqLengths.tensor == nullptr, OP_LOGE(context_->opName, "Tensor of tensor actualKVSeqLengths is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->actualKVSeqLengths.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor actualKVSeqLengths is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->actualQSeqLengths.tensor == nullptr, OP_LOGE(context_->opName, "Tensor of tensor actualQSeqLengths is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->actualQSeqLengths.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor actualQSeqLengths is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockTable.tensor == nullptr, OP_LOGE(context_->opName, "Tensor of tensor blockTable is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockTable.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor blockTable is nullptr"),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->attenOut.desc == nullptr, OP_LOGE(context_->opName, "Desc of tensor output is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->attenOut.shape == nullptr, OP_LOGE(context_->opName, "Shape of tensor output is nullptr"),
            return ge::GRAPH_FAILED);

    // Check base input attrs
    OP_CHECK_IF(context_->numHeads == nullptr, OP_LOGE(context_->opName, "attr numHeads is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->scaleValue == nullptr, OP_LOGE(context_->opName, "attr scaleValue is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kvHeadNums == nullptr, OP_LOGE(context_->opName, "attr kvHeadNums is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->layOut == nullptr, OP_LOGE(context_->opName, "attr layOut is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->blockSize == nullptr, OP_LOGE(context_->opName, "attr blockSize is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->selectedBlockSize == nullptr, OP_LOGE(context_->opName, "attr selectedBlockSize is nullptr"),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->selectedBlockCount == nullptr, OP_LOGE(context_->opName, "attr selectedBlockCount is nullptr"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus NsaSelectTiling::CheckInputFormatAndLimits() const
{
    OP_CHECK_IF(
        ((inputQType_ != inputKvType_)),
        OP_LOGE(context_->opName, "when input Q type %d is not equal to input KV type %d", inputQType_, inputKvType_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
        OP_LOGE(context_->opName, "blockSize=%ld, it need align to 16 when kv dtype is fp16/bf16.", blockSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (batchSize_ > 3072),
        OP_LOGE(context_->opName, "batch size:%u cannot be greater than 3072.", batchSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((blockSize_ != 64) && (blockSize_ != 128), OP_LOGE(context_->opName, "blockSize is not 64 or 128"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockSize_ % 16 != 0 || selectedBlockSize_ == 0, OP_LOGE(context_->opName, "selectedBlockSize %ld is not 16 aligned", selectedBlockSize_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockSize_ > 128 , OP_LOGE(context_->opName, "selectedBlockSize %ld should not be greater than 128", selectedBlockSize_), return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDim_ != 192),
        OP_LOGE(context_->opName, "QK Head dim must == 192, but got: %u.", headDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDim_ <= headDimV_),
        OP_LOGE(context_->opName, "headDim:%u cannot be smaller than headDimV:%u.", headDim_, headDimV_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((numKvHeads_ > 256),
        OP_LOGE(context_->opName, "numHead of key and value:%ld cannot be greater than 256.", numKvHeads_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((headDimV_ != 128),
        OP_LOGE(context_->opName, "V Head dim must == 128, but got: %u.", headDimV_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CheckQKOutShape()
{
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->key.shape;
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH") {
        OP_CHECK_IF(
            (dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != DIM_BSH),
            OP_LOGE("[NsaSelectInfer]",
                    "When input layout is BSH, the dimension should be 3, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu",
                    dimOfQ, dimOfK, dimOfOut),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((queryShape->GetStorageShape().GetDim(1) < 1) || (queryShape->GetStorageShape().GetDim(1) > 4 ),
                OP_LOGE("[NsaSelectInfer]", "When input layout is BSH, the 2nd dimOfQ should be be greater than or equal to 1 and  less than 4, the 2nd dimOfQ: %ld",
                            queryShape->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(queryShape->GetStorageShape().GetDim(2) / numHeads_ !=
                    keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
                OP_LOGE("[NsaSelectInfer]","When input layout is BSH,"
                            "the 3rd dimOfQ/numHeads(%ld) should be equal to the 3rd dimOfK/numKvHeads(%ld)",
                            queryShape->GetStorageShape().GetDim(2) / numHeads_,
                            keyShape->GetStorageShape().GetDim(2) / numKvHeads_),
                return ge::GRAPH_FAILED);
    } else if (inputLayoutStr == "TND") {
        return TNDCheckQKOutShape();
    } else {
        OP_CHECK_IF(
            (dimOfQ != DIM_BNSD_OR_BNSD) || (dimOfK != DIM_BNSD_OR_BNSD) || (dimOfOut != DIM_BNSD_OR_BNSD),
            OP_LOGE("[NsaSelectInfer]",
                    "When input layout is BNSD/BSND, the dim should be 4, 4th dimOfQ: %lu, 4th dimOfK: %lu, fourth dimOfOut: %lu",
                    dimOfQ, dimOfK, dimOfOut),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
            OP_LOGE(
                "[NsaSelectInfer]",
                "When input layout is BNSD/BSND, the 4th dimOfQ not be equal the 4th dimOfK, dimOfQ: %ld, dimOfK: %ld",
                queryShape->GetStorageShape().GetDim(3), keyShape->GetStorageShape().GetDim(3)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::TNDCheckQKOutShape() const
{
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->key.shape;
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (dimOfQ != DIM_TND) || (dimOfK != DIM_TND) || (dimOfOut != DIM_TND),
        OP_LOGE("[NsaSelectInfer]",
                "When input layout is TND, the dim should be 3, dimOfQ: %lu,  dimOfK: %lu,  dimOfOut: %lu",
                dimOfQ, dimOfK, dimOfOut),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        queryShape->GetStorageShape().GetDim(2) != keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
        OP_LOGE(
            "[NsaSelectInfer]",
            "When input layout is TND, the 3th dimOfQ should be equal the 3th dimOfK/numKvHeads, dimOfQ: %ld, dimOfK: %ld",
            queryShape->GetStorageShape().GetDim(2), keyShape->GetStorageShape().GetDim(2)),
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
 