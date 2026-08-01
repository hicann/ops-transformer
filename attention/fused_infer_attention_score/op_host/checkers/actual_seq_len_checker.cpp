/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file actual_seq_len_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "actual_seq_len_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// single para
ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQDim(const FiaTilingInfo &fiaInfo)
{
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验query的actualSeqLengths的维度
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t batchSize = fiaInfo.bSize;
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // query的layout为TND/NTD时，actualSeqLengthsQ的长度为query的batch值
        if (actualSeqLengthsQDimNum != batchSize) {
            std::string shapeStr = "actual_seq_lengths element nums: " + std::to_string(actualSeqLengthsQDimNum) +
                ", batchSize: " + std::to_string(batchSize);
            std::string reason = "The element nums of actual_seq_lengths(" +
                std::to_string(actualSeqLengthsQDimNum) +
                ") must be equal to the batchSize of query(" + std::to_string(batchSize) +
                ") when the layout of query is TND or NTD";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "actual_seq_lengths and query",
                shapeStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        // query为非TND/NTD，actualSeqLengthsQ的长度为1或大于等于query的batch值
        if (actualSeqLengthsQDimNum != DIM_NUM_1 && actualSeqLengthsQDimNum < batchSize) {
            std::string correctStr = "greater than or equal to the batchSize: " +
                std::to_string(batchSize) + " of query or equal to 1.";
            OP_LOGE_FOR_INVALID_LISTSIZE(fiaInfo.opName, "actual_seq_lengths",
                std::to_string(actualSeqLengthsQDimNum).c_str(), correctStr.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQData(const FiaTilingInfo &fiaInfo)
{
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验query的actualSeqLengthData的数值约束
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    uint32_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // query的layout为TND/NTD时，其值应递增，且为非负数
        for (uint32_t bIdx = 0; bIdx < batchSize; bIdx++) {
            if (actualSeqLengthsQTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx];
            // 其值应为递增
            if (bIdx != 0U) {
                int64_t lastSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[bIdx - 1];
                if (curSeqLengthData < lastSeqLengthData) {
                    std::string valueStr = "actual_seq_lengths[" + std::to_string(bIdx) + "]";
                    std::string reason = "actual_seq_lengths[" + std::to_string(bIdx) + "](" +
                        std::to_string(curSeqLengthData) + ") must be greater than or equal to actual_seq_lengths[" +
                        std::to_string(bIdx - 1U) + "](" + std::to_string(lastSeqLengthData) +
                        ") when the layout of query is TND or NTD";
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                        std::to_string(curSeqLengthData).c_str(), reason.c_str());
                    return ge::GRAPH_FAILED;
                }
            }
            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths[" + std::to_string(bIdx) + "]";
                std::string reason = "The value of actual_seq_lengths[" + std::to_string(bIdx) +
                    "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                    std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        // query的layout为非TND/NTD，其值应不大于Q_S，且为非负数
        int64_t sOfQuery = static_cast<int64_t>(fiaInfo.s1Size);
        uint32_t actualSeqLengthsSize = std::min(actualSeqLengthsQDimNum, batchSize);
        for (uint32_t i = 0; i < actualSeqLengthsSize; i++) {
            if (actualSeqLengthsQTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsQTensor->GetData<int64_t>()[i];
            // curSeqLengthData应不大于Q_S
            if (curSeqLengthData > sOfQuery) {
                std::string valueStr = "actual_seq_lengths[" + std::to_string(i) + "]";
                std::string reason = "actual_seq_lengths[" + std::to_string(i) + "](" +
                    std::to_string(curSeqLengthData) + ") must not be greater than Q_S(" +
                    std::to_string(sOfQuery) + ") when the layout of query is not TND or NTD";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                    std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }

            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths[" + std::to_string(i) + "]";
                std::string reason = "The value of actual_seq_lengths[" + std::to_string(i) +
                    "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                    std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvDim(const FiaTilingInfo &fiaInfo)
{
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验key/value的actualSeqLengths的维度
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    uint32_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // key/value的layout为TND/NTD时，actualSeqLengthsKv的长度为batchSize
        if (actualSeqLengthsKvDimNum != batchSize) {
            std::string shapeStr = "actual_seq_lengths_kv element nums: " + std::to_string(actualSeqLengthsKvDimNum) +
                ", batchSize: " + std::to_string(batchSize);
            std::string reason = "The element nums of actual_seq_lengths_kv(" +
                std::to_string(actualSeqLengthsKvDimNum) +
                ") must be equal to the batchSize of key and value(" + std::to_string(batchSize) +
                ") when the layout of key and value is TND or NTD";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "actual_seq_lengths_kv and key and value",
                shapeStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        // key/value的layout为非TND/NTD，actualSeqLengthsKv的长度为1或大于等于batchSize
        if (actualSeqLengthsKvDimNum != DIM_NUM_1 && actualSeqLengthsKvDimNum < batchSize) {
            std::string correctStr = "greater than or equal to the batchSize: " +
                std::to_string(batchSize) + " or equal to 1.";
            OP_LOGE_FOR_INVALID_LISTSIZE(fiaInfo.opName, "actual_seq_lengths_kv",
                std::to_string(actualSeqLengthsKvDimNum).c_str(), correctStr.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvData(const FiaTilingInfo &fiaInfo)
{
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验key/value的actualSeqLengthData的数值约束
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    int64_t batchSize = fiaInfo.bSize;
    FiaLayout qLayout = fiaInfo.qLayout;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        // key/value的layout为TND或NTD时，非page attention场景时，其值应递增，且为非负数
        for (int64_t bIdx = 0; bIdx < batchSize; bIdx++) {
            if ( actualSeqLengthsKvTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx];
            // 非page attention场景时，其值应为递增
            if (bIdx != 0) {
                int64_t lastSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[bIdx - 1];
                if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION && curSeqLengthData < lastSeqLengthData) {
                    std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(bIdx) + "]";
                    std::string reason = "actual_seq_lengths_kv[" + std::to_string(bIdx) + "](" +
                        std::to_string(curSeqLengthData) + ") must be greater than or equal to actual_seq_lengths_kv[" +
                        std::to_string(bIdx - 1) + "](" + std::to_string(lastSeqLengthData) +
                        ") when the layout of key and value is TND or NTD and page attention is not enabled";
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                        std::to_string(curSeqLengthData).c_str(), reason.c_str());
                    return ge::GRAPH_FAILED;
                }
            }
            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(bIdx) + "]";
                std::string reason = "The value of actual_seq_lengths_kv[" + std::to_string(bIdx) +
                    "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                    std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        // key/value的layout为非TND/NTD，其值应不大于KV_S，且为非负数
        int64_t sOfKeyValue = static_cast<int64_t>(fiaInfo.s2Size);
        int64_t actualSeqLengthsSize = std::min(static_cast<int64_t>(actualSeqLengthsKvDimNum), batchSize);
        for (int64_t i = 0; i < actualSeqLengthsSize; i++) {
            if ( actualSeqLengthsKvTensor->GetData<int64_t>() == nullptr) {
                return ge::GRAPH_SUCCESS;
            }
            int64_t curSeqLengthData = actualSeqLengthsKvTensor->GetData<int64_t>()[i];
            // curSeqLengthData应不大于KV_S
            if (curSeqLengthData > sOfKeyValue) {
                std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(i) + "]";
                std::string reason = "The value of actual_seq_lengths_kv[" + std::to_string(i) +
                    "] cannot be greater than KV_S: " +
                    std::to_string(sOfKeyValue);
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                    std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            // curSeqLengthData应为非负数
            if (curSeqLengthData < 0) {
                std::string valueStr = "actual_seq_lengths_kv[" + std::to_string(i) + "]";
                std::string reason = "The value of actual_seq_lengths_kv[" + std::to_string(i) +
                    "] cannot be less than 0";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                    std::to_string(curSeqLengthData).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenQTNDLastData(const FiaTilingInfo &fiaInfo)
{
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验query的输入为TND/NTD时，actualSeqLengthQ的最后一个元素与T相等
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    auto &queryShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    if (actualSeqLengthsQTensor == nullptr) {
        // 若不存在actualSeqLengthsQ，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsQDimNum = actualSeqLengthsQTensor->GetShapeSize();
    int64_t actualSeqLengthsQLastData = actualSeqLengthsQTensor->GetData<int64_t>()[actualSeqLengthsQDimNum - 1];
    if (fiaInfo.qLayout == FiaLayout::TND) {
        if (actualSeqLengthsQLastData != queryShape.GetDim(DIM_NUM_0)) {
            int64_t queryT = queryShape.GetDim(DIM_NUM_0);
            std::string valueStr = "actual_seq_lengths last element";
            std::string reason = "The last element of actual_seq_lengths(" +
                std::to_string(actualSeqLengthsQLastData) +
                ") must be equal to the T of query(" + std::to_string(queryT) +
                ") when the layout of query is TND";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                std::to_string(actualSeqLengthsQLastData).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo.qLayout == FiaLayout::NTD) {
        if (actualSeqLengthsQLastData != queryShape.GetDim(DIM_NUM_1)) {
            int64_t queryT = queryShape.GetDim(DIM_NUM_1);
            std::string valueStr = "actual_seq_lengths last element";
            std::string reason = "The last element of actual_seq_lengths(" +
                std::to_string(actualSeqLengthsQLastData) +
                ") must be equal to the T of query(" + std::to_string(queryT) +
                ") when the layout of query is NTD";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                std::to_string(actualSeqLengthsQLastData).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckActualSeqLenKvTNDLastData(const FiaTilingInfo &fiaInfo)
{
    // tiling下沉场景，则放弃后续校验
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    // 校验key/value的输入为TND/NTD时，actualSeqLengthsKv的最后一个元素与T相等
    auto &actualSeqLengthsKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    auto &keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    if (actualSeqLengthsKvTensor == nullptr) {
        // 若不存在actualSeqLengthsKv，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        // 若使能page attention，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t actualSeqLengthsKvDimNum = actualSeqLengthsKvTensor->GetShapeSize();
    int64_t actualSeqLengthsKvLastData = actualSeqLengthsKvTensor->GetData<int64_t>()[actualSeqLengthsKvDimNum - 1];
    if (fiaInfo.kvLayout == FiaLayout::TND) {
        if (actualSeqLengthsKvLastData != keyShape.GetDim(DIM_NUM_0)) {
            int64_t keyT = keyShape.GetDim(DIM_NUM_0);
            std::string valueStr = "actual_seq_lengths_kv last element";
            std::string reason = "The last element of actual_seq_lengths_kv(" +
                std::to_string(actualSeqLengthsKvLastData) +
                ") must be equal to the T of key and value(" + std::to_string(keyT) +
                ") when the layout of key and value is TND";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                std::to_string(actualSeqLengthsKvLastData).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo.kvLayout == FiaLayout::NTD) {
        if (actualSeqLengthsKvLastData != keyShape.GetDim(DIM_NUM_1)) {
            int64_t keyT = keyShape.GetDim(DIM_NUM_1);
            std::string valueStr = "actual_seq_lengths_kv last element";
            std::string reason = "The last element of actual_seq_lengths_kv(" +
                std::to_string(actualSeqLengthsKvLastData) +
                ") must be equal to the T of key and value(" + std::to_string(keyT) +
                ") when the layout of key and value is NTD";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, valueStr.c_str(),
                std::to_string(actualSeqLengthsKvLastData).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus ActualSeqLenChecker::CheckExistenceActualSeqLenQ(const FiaTilingInfo &fiaInfo)
{
    // 校验actualSeqLenQ的存在性
    FiaLayout qLayout = fiaInfo.qLayout;
    // query的Layout为TND/NTD时，actualSeqLengthsQ必须传入
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (qLayout == FiaLayout::TND || qLayout == FiaLayout::NTD) {
        OP_CHECK_IF(actualSeqLengthsQTensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsQ",
                    "actualSeqLengthsQ cannot be empty when the layout of query is TND or NTD"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckExistenceActualSeqLenKv(const FiaTilingInfo &fiaInfo)
{
    FiaLayout kvLayout = fiaInfo.kvLayout;
    // key、value的layout为TND/NTD时，actualSeqLengthsKv必须传入
    if (kvLayout == FiaLayout::TND || kvLayout == FiaLayout::NTD) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsKv",
                    "actualSeqLengthsKv cannot be empty when the layout of key and value is TND or NTD"),
            return ge::GRAPH_FAILED);
    }
    // PagedAttention场景下，必须传入actualSeqLengthsKv
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsKv",
                    "actualSeqLengthsKv cannot be empty when page attention is enabled"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus ActualSeqLenChecker::CheckFeatureIFAMLA(const FiaTilingInfo &fiaInfo)
{
    auto &actualSeqLengthsQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    FiaLayout qLayout = fiaInfo.qLayout;
    // IFAMLA全量化场景，仅query的layout为TND/NTD时支持传入actualSeqLengthsQ
    bool enableIFAMLA = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512);
    if (enableIFAMLA && actualSeqLengthsQTensor != nullptr) {
        OP_CHECK_IF((qLayout != FiaLayout::TND) && (qLayout != FiaLayout::NTD),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "actualSeqLengthsQ",
                    "actualSeqLengthsQ must be empty in IFA MLA and non-TND/NTD scenarios"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLenQDim(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenQData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvDim(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenQTNDLastData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLenKvTNDLastData(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceActualSeqLenQ(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckExistenceActualSeqLenKv(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (enableFullQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureIFAMLA(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ActualSeqLenChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
