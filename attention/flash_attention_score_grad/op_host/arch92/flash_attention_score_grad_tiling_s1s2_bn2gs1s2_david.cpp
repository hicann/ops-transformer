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
 * \file flash_attention_score_grad_tiling_s1s2_bn2gs1s2_david.cpp
 * \brief
 */

#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_david.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "err/ops_err.h"

using namespace optiling::fag;
namespace optiling {
namespace fag {

uint64_t sumS1 = 0;
int64_t sfmgSize = 0;

template <class T>
inline T CeilDivideBy(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

inline int64_t AbsCeil(int64_t num1, int64_t num2)
{
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (std::abs(num1) + std::abs(num2) - 1) / std::abs(num2);
    return isNegative ? -result : result;
}

template <class T>
inline std::vector<T> SliceVector(const std::vector<T>& arr, const int64_t step)
{
    if (step == 1) {
        return arr;
    }

    std::vector<T> result;
    for (auto it = arr.begin(); it < arr.end(); it += step) {
        result.push_back(*it);
    }
    return result;
}

std::pair<uint32_t, uint32_t> FlashAttentionScoreGradTilingUs1s2Bs2David::GetS1S2TemplateType()
{
    if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) {
        // 当前FP8的Per-Block量化方案要求s1和s2切块相同
        fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM256;
        fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM128;
        return std::make_pair((uint32_t)ConstAxisTemplateNum::NUM256, (uint32_t)ConstAxisTemplateNum::NUM128);
    } else {
        if (fBaseParams.queryType == ge::DT_FLOAT) {
            fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM64;
            fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM128;
            return std::make_pair((uint32_t)ConstAxisTemplateNum::NUM64, (uint32_t)ConstAxisTemplateNum::NUM128);
        } else {
            fBaseParams.s1TemplateType = ConstAxisTemplateNum::NUM256;
            fBaseParams.s2TemplateType = ConstAxisTemplateNum::NUM128;
            return std::make_pair((uint32_t)ConstAxisTemplateNum::NUM256, (uint32_t)ConstAxisTemplateNum::NUM128);
        }
    }
}

uint32_t FlashAttentionScoreGradTilingUs1s2Bs2David::GetDTemplateType()
{
    if (fBaseParams.hasRope) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM192;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM192);
    }
    if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM64)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM64;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM64);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM128)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM128;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM128);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM192)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM192;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM192);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM256)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM256;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM256);
    } else if (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM768)) {
        fBaseParams.dTemplateType = ConstAxisTemplateNum::NUM768;
        return static_cast<uint32_t>(ConstAxisTemplateNum::NUM768);
    }
    return static_cast<uint32_t>(ConstAxisTemplateNum::NUM768);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::SetQKVStartIdx()
{
    fBaseParams.qStartIdx = 0;
    fBaseParams.kvStartIdx = 0;
    auto qStartIdxTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::Q_START_IDX));
    if (qStartIdxTensor != nullptr) {
        auto& qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
        if (qStartIdxShape.GetDimNum() >= 1 && qStartIdxShape.GetDim(0) != 0) {
            /* Get Data from tensor. */
            const int64_t* value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                fBaseParams.qStartIdx = value[0];
                OP_LOGD(context_, "[%s]SetQKVStartIdx qStartIdx: %ld", "FlashAttentionScoreGradTilingS1s2Bn2gs1s2",
                        fBaseParams.qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::KV_START_IDX));
    if (kvStartIdxTensor != nullptr) {
        auto& kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
        if (kvStartIdxShape.GetDimNum() >= 1 && kvStartIdxShape.GetDim(0) != 0) {
            /* Get Data from tensor. */
            const int64_t* kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                fBaseParams.kvStartIdx = kvValue[0];
                OP_LOGD(context_, "[%s]SetQKVStartIdx kvStartIdx: %ld", "FlashAttentionScoreGradTilingS1s2Bn2gs1s2",
                        fBaseParams.kvStartIdx);
            }
        }
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::QuantScaleShapeValidCheck()
{
    auto deqScaleQShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_Q));
    auto deqScaleKShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_K));
    auto deqScaleVShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_V));
    auto deqScaleDyShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_DY));
    auto deqScaleOShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::D_SCALE_O));
    if (deqScaleQShape != nullptr && deqScaleKShape != nullptr && deqScaleVShape != nullptr &&
        deqScaleDyShape != nullptr && deqScaleOShape != nullptr) {
        OP_CHECK_IF(!(deqScaleQShape->GetStorageShape() == deqScaleDyShape->GetStorageShape() &&
                      deqScaleDyShape->GetStorageShape() == deqScaleOShape->GetStorageShape()),
                    OP_LOGE(context_, "deqScaleQShape, deqScaleDyShape and deqScaleOShape are not equal."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(deqScaleKShape->GetStorageShape() != deqScaleVShape->GetStorageShape(),
                    OP_LOGE(context_, "deqScaleKShape nd deqScaleVShape are not equal."), return ge::GRAPH_FAILED);

        int64_t deqScaleQDimNum = deqScaleQShape->GetStorageShape().GetDimNum();
        if (deqScaleQDimNum != 0) {
            OP_CHECK_IF(deqScaleQDimNum != DEQUANT_SCALE_SHAPE_DIM,
                        OP_LOGE(context_, "Invalid deqScaleQ dimNum [%ld], only support 4 dims.", deqScaleQDimNum),
                        return ge::GRAPH_FAILED);
            int64_t deqScaleQDim0 = deqScaleQShape->GetStorageShape().GetDim(INPUT_DIM_0);
            int64_t deqScaleQDim1 = deqScaleQShape->GetStorageShape().GetDim(INPUT_DIM_1);
            int64_t deqScaleQDim2 = deqScaleQShape->GetStorageShape().GetDim(INPUT_DIM_2);
            int64_t deqScaleQDim3 = deqScaleQShape->GetStorageShape().GetDim(INPUT_DIM_3);
            OP_CHECK_IF(
                deqScaleQDim0 != fBaseParams.b || deqScaleQDim1 != fBaseParams.n1 ||
                    deqScaleQDim2 != (fBaseParams.s1 + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE || deqScaleQDim3 != 1,
                OP_LOGE(context_, "Invalid deqScaleQ shape [%ld,%ld,%ld,%ld], only support [B,N1,ceil(S1/%u),1].",
                        deqScaleQDim0, deqScaleQDim1, deqScaleQDim2, deqScaleQDim3, QUANT_BLOCK_SIZE),
                return ge::GRAPH_FAILED);
        }

        int64_t deqScaleKDimNum = deqScaleKShape->GetStorageShape().GetDimNum();
        if (deqScaleKDimNum != 0) {
            OP_CHECK_IF(deqScaleKDimNum != DEQUANT_SCALE_SHAPE_DIM,
                        OP_LOGE(context_, "Invalid deqScaleK dimNum [%ld], only support 4 dims.", deqScaleKDimNum),
                        return ge::GRAPH_FAILED);
            int64_t deqScaleKDim0 = deqScaleKShape->GetStorageShape().GetDim(INPUT_DIM_0);
            int64_t deqScaleKDim1 = deqScaleKShape->GetStorageShape().GetDim(INPUT_DIM_1);
            int64_t deqScaleKDim2 = deqScaleKShape->GetStorageShape().GetDim(INPUT_DIM_2);
            int64_t deqScaleKDim3 = deqScaleKShape->GetStorageShape().GetDim(INPUT_DIM_3);
            OP_CHECK_IF(
                deqScaleKDim0 != fBaseParams.b || deqScaleKDim1 != fBaseParams.n2 ||
                    deqScaleKDim2 != (fBaseParams.s2 + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE || deqScaleKDim3 != 1,
                OP_LOGE(context_, "Invalid deqScaleK shape [%ld,%ld,%ld,%ld], only support [B,N2,ceil(S2/%u),1].",
                        deqScaleKDim0, deqScaleKDim1, deqScaleKDim2, deqScaleKDim3, QUANT_BLOCK_SIZE),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::QuantScaleDtypeValidCheck()
{
    auto deqScaleQInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_Q));
    auto deqScaleKInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_K));
    auto deqScaleVInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_V));
    auto deqScaleDyInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_DY));
    auto deqScaleOInput = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::D_SCALE_O));
    if (deqScaleQInput != nullptr && deqScaleKInput != nullptr && deqScaleVInput != nullptr &&
        deqScaleDyInput != nullptr && deqScaleOInput != nullptr) {
        auto deqScaleQDtype = deqScaleQInput->GetDataType();
        auto deqScaleKDtype = deqScaleKInput->GetDataType();
        auto deqScaleVDtype = deqScaleVInput->GetDataType();
        auto deqScaleDyDtype = deqScaleDyInput->GetDataType();
        auto deqScaleODtype = deqScaleOInput->GetDataType();
        OP_CHECK_IF(deqScaleQDtype != ge::DT_FLOAT || deqScaleKDtype != ge::DT_FLOAT ||
                        deqScaleVDtype != ge::DT_FLOAT || deqScaleDyDtype != ge::DT_FLOAT ||
                        deqScaleODtype != ge::DT_FLOAT,
                    OP_LOGE(context_,
                            "Invalid deqScaleDType [deqScaleQDtype:%s, deqScaleKDtype:%s, deqScaleVDtype:%s, "
                            "deqScaleDyDtype:%s, deqScaleODtype:%s], only support FLOAT32.",
                            ge::TypeUtils::DataTypeToSerialString(deqScaleQDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(deqScaleKDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(deqScaleVDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(deqScaleDyDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(deqScaleODtype).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetShapeAttrsInfo()
{
    fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
    const gert::StorageShape* queryShape = context_->GetInputShape(QUERY_IDX); // [B, N2, G, S1, D]
    const gert::StorageShape* keyShape = context_->GetInputShape(KEY_IDX);     // [B, N2, 1, S2, D]
    const gert::StorageShape* valueShape = context_->GetInputShape(VALUE_IDX); // [B, N2, 1, S2, D_V]

    int64_t headNum = *context_->GetAttrs()->GetAttrPointer<int>(HEAD_ATTR_IDX);
    const char* inputLayout = context_->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);

    // get rope
    auto queryRope = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::QUERY_ROPE_IDX));
    const gert::Shape* queryRopeShape = &queryRope->GetStorageShape();
    bool hasQueryRope = queryRope != nullptr && queryRopeShape->GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::KEY_ROPE_IDX));
    const gert::Shape* keyRopeShape = &keyRope->GetStorageShape();
    bool hasKeyRope = keyRope != nullptr && keyRopeShape->GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        OP_LOGE(context_, "query_rope and key_rope should be present or absent at the same time, check this.");
        return ge::GRAPH_PARAM_INVALID;
    }
    fBaseParams.hasRope = hasQueryRope && hasKeyRope;
    int64_t qRopeD = 0;
    int64_t kRopeD = 0;

    if (strcmp(inputLayout, "SBH") == 0) {
        OP_LOGD(context_, "inputLayout == SBH queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_S2BN2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.d =
            fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / headNum; // H=N*D
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_2) / fBaseParams.n2;                // H=N2*D1
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_0);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_2) / headNum : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_2) / fBaseParams.n2 : 0;
    } else if (strcmp(inputLayout, "BSH") == 0) {
        OP_LOGD(context_, "inputLayout == BSH queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.n2 = headNum / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d =
            fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / headNum; // H=N*D
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_2) / fBaseParams.n2;                // H=N2*D1
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_2) / headNum : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_2) / fBaseParams.n2 : 0;
    } else if (strcmp(inputLayout, "BNSD") == 0) {
        OP_LOGD(context_, "inputLayout == BNSD queryShape");
        fBaseParams.layoutType = INPUT_FORMAT_BN2GS2D;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.d = fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_3) : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_3) : 0;
        OP_LOGD(context_, "inputLayout == BNSD queryShape", "%ld, %ld, %ld, %ld,",
                queryShape->GetStorageShape().GetDim(INPUT_DIM_0), queryShape->GetStorageShape().GetDim(INPUT_DIM_1),
                queryShape->GetStorageShape().GetDim(INPUT_DIM_2), queryShape->GetStorageShape().GetDim(INPUT_DIM_3));
    } else if (strcmp(inputLayout, "TND") == 0) {
        OP_LOGD(context_, "inputLayout == TND");
        fBaseParams.layoutType = INPUT_FORMAT_TND;

        auto actualSeqQlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
        auto actualSeqKvlenTensor =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        if (actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr) {
            OP_LOGE("inputLayout = TND", "actualSeqQlenTensor or actualSeqKvlenTensor is nullptr");
            return ge::GRAPH_PARAM_INVALID;
        }

        const size_t seqQShapeSize = actualSeqQlenTensor->GetShapeSize();
        const size_t kvSeqShapeSize = actualSeqKvlenTensor->GetShapeSize();
        if (seqQShapeSize != kvSeqShapeSize) {
            OP_LOGE("inputLayout = TND", "actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor");
            return ge::GRAPH_PARAM_INVALID;
        }

        int64_t lastQLen = 0;
        int64_t lastKvLen = 0;
        fBaseParams.isS1S2Same = true;
        fBaseParams.isAllSame = true;
        const int64_t* qValue = actualSeqQlenTensor->GetData<int64_t>();
        const int64_t* kvValue = actualSeqKvlenTensor->GetData<int64_t>();
        bool isEOD = false;
        for (size_t i = 0; i < seqQShapeSize; i++) {
            if (i == 0) {
                fBaseParams.actualSeqQlen.push_back(qValue[i]);
                fBaseParams.actualSeqKvlen.push_back(kvValue[i]);
                if (qValue[0] == 0 || kvValue[0] == 0) {
                    fBaseParams.sValueZeroUnderTND = true;
                    tndBaseInfo.isSeqExistZero = true;
                }
                tndBaseInfo.isS1GreaterThanS2 = tndBaseInfo.isS1GreaterThanS2 && (qValue[0] >= kvValue[0]);
                tndBaseInfo.isS1LessThanS2 = tndBaseInfo.isS1LessThanS2 && (qValue[0] <= kvValue[0]);
            } else {
                lastQLen = fBaseParams.actualSeqQlen[i - 1];
                lastKvLen = fBaseParams.actualSeqKvlen[i - 1];
                auto qLen = qValue[i] - qValue[i - 1];
                auto kvLen = kvValue[i] - kvValue[i - 1];
                fBaseParams.actualSeqQlen.push_back(qLen < 0 ? 0 : qLen);
                fBaseParams.actualSeqKvlen.push_back(kvLen < 0 ? 0 : kvLen);
                if (qLen < 0 || kvLen < 0) {
                    isEOD = true;
                }
                tndBaseInfo.isSeqExistZero = (tndBaseInfo.isSeqExistZero || (qLen == 0 || kvLen == 0));
                if (isEOD && (qValue[i] == 0 || kvValue[i] == 0)) {
                    ++fBaseParams.tailZeroCount;
                    fBaseParams.sValueZeroUnderTND = true;
                } else if (isEOD && (qValue[i] != 0 || kvValue[i] != 0)) {
                    std::string valuesMsg = "{" + std::to_string(qValue[i]) + ", " + std::to_string(kvValue[i]) + "}";
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "FlashAttentionScoreGrad", "actualSeqQlenOptional, actualSeqKvlenOptional", valuesMsg.c_str(),
                        "When inputLayout is TND EOD, the values of last several actualSeqQlenOptional and "
                        "actualSeqKvlenOptional must be 0");
                    return ge::GRAPH_PARAM_INVALID;
                }
                fBaseParams.isAllSame = (kvValue[i] - kvValue[i - 1] == lastKvLen) &&
                                        (qValue[i] - qValue[i - 1] == lastQLen) && fBaseParams.isAllSame;
                tndBaseInfo.isS1GreaterThanS2 = tndBaseInfo.isS1GreaterThanS2 && (qLen >= kvLen);
                tndBaseInfo.isS1LessThanS2 = tndBaseInfo.isS1LessThanS2 && (qLen <= kvLen);
            }
            fBaseParams.isS1S2Same =
                fBaseParams.actualSeqQlen[i] == fBaseParams.actualSeqKvlen[i] && fBaseParams.isS1S2Same;
            fBaseParams.sumS1S2Product += fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i];
            sumS1 += fBaseParams.actualSeqQlen[i];
        }

        fBaseParams.b = seqQShapeSize;
        fBaseParams.t1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.t2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.s1 = *std::max_element(fBaseParams.actualSeqQlen.begin(), fBaseParams.actualSeqQlen.end());
        fBaseParams.s2 = *std::max_element(fBaseParams.actualSeqKvlen.begin(), fBaseParams.actualSeqKvlen.end());
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_2);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_2) : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_2) : 0;
    } else {
        OP_LOGD(context_, "inputLayout == BSND queryShape");
        // inputLayout = "BSND"
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d = fBaseParams.hasRope ? ROPE_D_192 : queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.d1 = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        qRopeD = fBaseParams.hasRope ? queryRopeShape->GetDim(INPUT_DIM_3) : 0;
        kRopeD = fBaseParams.hasRope ? keyRopeShape->GetDim(INPUT_DIM_3) : 0;
    }

    // check rope
    if (fBaseParams.hasRope) {
        if (qRopeD != kRopeD || qRopeD != ROPE_D_64) {
            OP_LOGE(context_, "query_rope and key_rope only support 64D, check this.");
            return ge::GRAPH_PARAM_INVALID;
        }
    }

    fBaseParams.n1 = fBaseParams.n2 * fBaseParams.g;
    auto quantScaleShapeCheckRet = QuantScaleShapeValidCheck();
    if (quantScaleShapeCheckRet != ge::GRAPH_SUCCESS) {
        return quantScaleShapeCheckRet;
    }
    auto quantScaleDtypeCheckRet = QuantScaleDtypeValidCheck();
    if (quantScaleDtypeCheckRet != ge::GRAPH_SUCCESS) {
        return quantScaleDtypeCheckRet;
    }

    if (strcmp(inputLayout, "TND") == 0) {
        fBaseParams.qSize = static_cast<uint64_t>(fBaseParams.t1) * fBaseParams.n2 * fBaseParams.g * fBaseParams.d;
        fBaseParams.kSize = static_cast<uint64_t>(fBaseParams.t2) * fBaseParams.n2 * 1 * fBaseParams.d;
        fBaseParams.vSize = static_cast<uint64_t>(fBaseParams.t2) * fBaseParams.n2 * 1 * fBaseParams.d1;
        fBaseParams.dropMaskSize = static_cast<uint64_t>(fBaseParams.n2) * fBaseParams.g * fBaseParams.sumS1S2Product;

        sfmgSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * sumS1 * 8; // 8 is size of S1
    } else {
        fBaseParams.qSize =
            static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d;
        fBaseParams.kSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d;
        fBaseParams.vSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * 1 * fBaseParams.s2 * fBaseParams.d1;
        fBaseParams.dropMaskSize =
            static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s2 * fBaseParams.s1;

        sfmgSize = static_cast<uint64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 *
                   8; // 8 is size of S1
    }

    // process tnd softmax layout
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN)) {
        const char* tndMaxSumLayout =
            context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN));
        if (strcmp(inputLayout, "TND") == 0 && strcmp(tndMaxSumLayout, "same_as_input") == 0) {
            fBaseParams.tndMaxSumLayout = 1;
        }
        OP_LOGD("TND Max Sum Layout", "FAG tndMaxSumLayout = %d, tndMaxSumLayout = %s.", fBaseParams.tndMaxSumLayout,
                tndMaxSumLayout);
    }

    // mBaseParams is used for matmal tiling module
    auto queryType = context_->GetInputDesc(0)->GetDataType();
    fBaseParams.queryType = queryType;
    fBaseParams.calTypeSize = FP32_BYTES;

    fBaseParams.scaleValue = *(context_->GetAttrs()->GetAttrPointer<float>(0));
    fBaseParams.keepProb = *(context_->GetAttrs()->GetAttrPointer<float>(1));
    fBaseParams.dropoutIsDivisibleBy8 = 1;

    if (fBaseParams.keepProb < 1) {
        if (strcmp(inputLayout, "TND") == 0) {
            for (int64_t i = 0; i < fBaseParams.b; i++)
                if (fBaseParams.actualSeqKvlen[i] % BIT_NUMS != 0) {
                    fBaseParams.dropoutIsDivisibleBy8 = 0;
                    break;
                }
        } else {
            if (fBaseParams.s2 % BIT_NUMS != 0) {
                fBaseParams.dropoutIsDivisibleBy8 = 0;
            }
        }
    }

    // dropout mask
    bool hasDrop = fBaseParams.keepProb < 1;
    fBaseParams.keepProbUint8 = static_cast<int64_t>(fBaseParams.keepProb * UINT8_MAX);
    if (context_->GetAttrs()->GetAttrNum() > SEED_ATTR_IDX) {
        fBaseParams.seed = *(context_->GetAttrs()->GetAttrPointer<int>(SEED_ATTR_IDX));
    }
    if (context_->GetAttrs()->GetAttrNum() > OFFSET_ATTR_IDX) {
        fBaseParams.offset = *(context_->GetAttrs()->GetAttrPointer<int>(OFFSET_ATTR_IDX));
    }
    DetermineMode();
    fBaseParams.outDtype = fBaseParams.inputDtype;
    if (context_->GetAttrs()->GetAttrNum() > OUTDTYPE_ATTR_IDX &&
        (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN)) {
        int64_t outDType = *(context_->GetAttrs()->GetAttrPointer<int>(OUTDTYPE_ATTR_IDX));
        if (outDType == 0) {
            fBaseParams.outDtype = DtypeEnum::FLOAT16_PRECISION;
        } else if (outDType == 1) {
            fBaseParams.outDtype = DtypeEnum::BFLOAT16;
        } else {
            OP_LOGE("GetOutDType", "outDType value is not valid, got %ld, try setting it to 0(fp16) or 1(bf16)",
                    outDType);
            return ge::GRAPH_FAILED;
        }
    }

    auto dropMask = context_->GetOptionalInputDesc(DROP_MASK_IDX);
    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_IDX);
    if (dropMask != nullptr && dropMaskShape != nullptr && dropMaskShape->GetStorageShape().GetDimNum() != 0) {
        if (!hasDrop) {
            OP_LOGE(context_, "DropMask parameter is invalid, please check keepProb value.");
            return ge::GRAPH_FAILED;
        }
        auto dropMaskDType = dropMask->GetDataType();
        OP_CHECK_IF(
            dropMaskDType != ge::DT_UINT8,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "FAG invalid dropMask dtype[%s], only support uint8.",
                                        ge::TypeUtils::DataTypeToSerialString(dropMaskDType).c_str()),
            return ge::GRAPH_FAILED);
        uint64_t dropMaskDim = dropMaskShape->GetStorageShape().GetDimNum();
        uint64_t dropMaskShapeSize = 1;
        for (uint64_t i = 0; i < dropMaskDim; ++i) {
            uint64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
            dropMaskShapeSize *= dimValue;
        }
        auto shapeSize = AlignUp(fBaseParams.dropMaskSize, static_cast<uint64_t>(BIT_NUMS)) / BIT_NUMS;
        if (dropMaskShapeSize < shapeSize) {
            OP_LOGE(context_, "FAG input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld.",
                    shapeSize, dropMaskShapeSize);
            return ge::GRAPH_FAILED;
        }
        fBaseParams.dropMaskOuter = static_cast<uint8_t>(true);
    } else {
        fBaseParams.dropMaskOuter = static_cast<uint8_t>(false);
    }

    // token_info
    fBaseParams.s1Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(PRE_TOKEN_ATTR_IDX));
    fBaseParams.s2Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(NEXT_TOKEN_ATTR_IDX));

    auto ret = ProcessSparseModeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }
    ret = ProcessTokensInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }
    SetQKVStartIdx();
    ret = ProcessPseInfo(inputLayout);
    if (ret != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ret;
    }

    // sink
    ret = ProcessSinkInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    fBaseParams.isSparse = SetSparseParams();
    OP_LOGD("Sparse FLAG", "FAG Us1s2Bbn2gs1s2 sparse mode = %u, sparse %s.", fBaseParams.sparseMode,
            fBaseParams.isSparse ? "enable" : "disable");

    if (fBaseParams.isSparse == false && fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        OP_LOGE(context_, "Sparse capability must be supported under prefix compress mode, pls check input params");
        return ge::GRAPH_FAILED;
    }

    if (CheckAttenMaskShape() != ge::GRAPH_SUCCESS) {
        PrintShapeInfo();
        return ge::GRAPH_FAILED;
    }

    if (CheckSoftmaxMaxSumTndShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetSplitAxis();

    return (strcmp(inputLayout, "TND") == 0) ?
               CheckTndShapeValid(context_, fBaseParams.t1, fBaseParams.n1, fBaseParams.d) :
               CheckShapeValid(context_, fBaseParams.b, fBaseParams.n1, fBaseParams.s1, fBaseParams.d);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::SetSplitAxis()
{
    fBaseParams.isBn2 =
        (fBaseParams.s1 <= BN2_MAX_S && fBaseParams.s2 <= BN2_MAX_S) && fBaseParams.d <= BN2_MAX_D &&
        fBaseParams.n1 == fBaseParams.n2 && (fBaseParams.queryType != ge::DT_FLOAT) &&
        (fBaseParams.d == fBaseParams.d1) &&
        !(fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) &&
        (fBaseParams.tailZeroCount == 0);
    bool bnLimit = ((fBaseParams.b * fBaseParams.n1) >= BN2_MULTIBLK_BN_64);
    bool bnSparseLimit = bnLimit && (fBaseParams.layoutType != INPUT_FORMAT_TND) &&
                         (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::PREFIX)) &&
                         (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS));
    fBaseParams.isBn2MultiBlk =
        bnSparseLimit && (fBaseParams.s1 > BN2_MAX_S || fBaseParams.s2 > BN2_MAX_S) &&
        (fBaseParams.s1 <= BN2_MULTIBLK_SEQ_MAX && fBaseParams.s2 <= BN2_MULTIBLK_SEQ_MAX) &&
        (fBaseParams.n1 == fBaseParams.n2) && fBaseParams.d <= BN2_MAX_D && (fBaseParams.queryType != ge::DT_FLOAT) &&
        (fBaseParams.d == fBaseParams.d1) &&
        !(fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
          fBaseParams.queryType == ge::DT_HIFLOAT8) &&
        !fBaseParams.hasRope;
    fBaseParams.isBn2 = fBaseParams.isBn2MultiBlk ? true : fBaseParams.isBn2; // 多基本块场景是原始bn2的子集
    if (fBaseParams.isBn2) {
        fBaseParams.isDeterministic = false;
        if (!fBaseParams.isBn2MultiBlk && fBaseParams.layoutType == INPUT_FORMAT_TND && fBaseParams.d > ALIGN128) {
            // bn2模板不支持的场景
            fBaseParams.isBn2 = false;
            fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
        } else if (fBaseParams.isBn2MultiBlk && fBaseParams.dropMaskOuter) {
            // bn2MutliBlk模板不支持的场景
            fBaseParams.isBn2 = false;
            fBaseParams.isBn2MultiBlk = false;
            fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
        }
    }
    fBaseParams.splitAxis = fBaseParams.isBn2 ? SplitAxisEnum::BN2 : SplitAxisEnum::BN2GS1S2;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetPlatformInfo()
{
    uint32_t coreNum = CORE_INIT_NUM; // 40 is init core num

    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile_info is null"), return ge::GRAPH_FAILED);

        socVersion = compileInfoPtr->socVersion;
        npuArch = compileInfoPtr->npuArch;
        fBaseParams.coreNum = compileInfoPtr->aivNum;
        fBaseParams.aicNum = compileInfoPtr->aicNum;
        fBaseParams.ubSize = compileInfoPtr->ubSize;
        fBaseParams.l1Size = compileInfoPtr->l1Size;
        fBaseParams.l0aSize = compileInfoPtr->l0aSize;
        fBaseParams.l0cSize = compileInfoPtr->l0cSize;
        fBaseParams.l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        socVersion = ascendcPlatform.GetSocVersion();
        npuArch = ascendcPlatform.GetCurNpuArch();
        coreNum = ascendcPlatform.GetCoreNumAiv();
        fBaseParams.coreNum = coreNum;
        fBaseParams.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, fBaseParams.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, fBaseParams.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, fBaseParams.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, fBaseParams.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, fBaseParams.l2CacheSize);
    }

    OP_CHECK_IF((fBaseParams.coreNum == 0) || (fBaseParams.aicNum == 0),
                OP_LOGE(context_, "num of coreNum(aivNum) is %ld, num of aicNum is %ld.", fBaseParams.coreNum,
                        fBaseParams.aicNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fBaseParams.ubSize <= 0, OP_LOGE(context_, "ubSize is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingUnpaddedAttensionDavid::IsCapable()
{
    auto actualSeqQLenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_Q_LEN));
    OP_LOGD(context_, "coreNum is %lu", fBaseParams.coreNum);
    if ((npuArch == NpuArch::DAV_9201 || npuArch == NpuArch::DAV_9202) && actualSeqQLenTensor != nullptr &&
        actualSeqQLenTensor->GetShapeSize() != 0) {
        OP_LOGD(context_, "FlashAttentionScoreGradTilingUnpaddedAttensionRegbase hit");
        return true;
    }
    return false;
};

bool FlashAttentionScoreGradTilingUs1s2Bs2David::IsCapable()
{
    // 基础模板 全部支持
    if (npuArch == NpuArch::DAV_9201 || npuArch == NpuArch::DAV_9202) {
        OP_LOGD(context_, "FlashAttentionScoreGradTilingUs1s2Bs2Regbase hit");
        return true;
    }
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::DoOpTiling()
{
    DoSplit();
    auto ret = DoSparse();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 分核优化，对于超出l2 cache的case优先多个核处理BN下的S1S2
    bool isExceedL2Cache = CheckExceedL2Cache();
    bool isLargeInvalidBlk = CheckIsLargeInvalidBlk(fBaseParams);
    fBaseParams.enableSwizzle = (isExceedL2Cache || isLargeInvalidBlk) && fBaseParams.blockOuter == fBaseParams.aicNum;

    // 非tnd场景下 仅支持非确定性计算+BN2GS1S2模板 支持swizzle优化
    fBaseParams.isSwizzle = fBaseParams.enableSwizzle && (fBaseParams.layoutType != INPUT_FORMAT_TND) &&
                            fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 &&
                            fBaseParams.sparseType != static_cast<uint8_t>(SparseType::UNSUPPORTED) &&
                            (fBaseParams.queryType != ge::DT_FLOAT) && !fBaseParams.isDeterministic;
    // tnd场景下 仅支持非确定性计算+BN2GS1S2模板 支持swizzle优化
    tndBaseInfo.isTndSwizzle =
        fBaseParams.enableSwizzle && fBaseParams.layoutType == INPUT_FORMAT_TND &&
        fBaseParams.splitAxis == SplitAxisEnum::BN2GS1S2 && fBaseParams.b < TND_SWIZZLE_PREFIX_NUM &&
        fBaseParams.sparseType != static_cast<uint8_t>(SparseType::UNSUPPORTED) && !tndBaseInfo.isSeqExistZero &&
        fBaseParams.tailZeroCount == 0 && !fBaseParams.isDeterministic && (fBaseParams.queryType != ge::DT_FLOAT) &&
        tndBaseInfo.validS2ColumnEligible;
    if (tndBaseInfo.isTndSwizzle) {
        fBaseParams.blockOuter = tndBaseInfo.validS2ActiveCoreNum;
        fBaseParams.blockFactor = static_cast<int64_t>(tndBaseInfo.validS2MaxCoreTileNum);
        fBaseParams.maxValidBBLen = static_cast<int64_t>(tndBaseInfo.validS2MaxCoreTileNum);
    }
    OP_LOGI(context_, "isExceedL2Cache=[%d], sparseType=[%d], enableSwizzle=[%d], isTndSwizzle = [%d].",
            static_cast<int>(isExceedL2Cache), static_cast<int>(fBaseParams.sparseType),
            static_cast<int>(fBaseParams.enableSwizzle), tndBaseInfo.isTndSwizzle);

    ret = InitTilingData();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    DoPreTiling();
    DoPostTiling();
    DoPreSfmgTiling();
    DetermineMode();
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::DoSplit()
{
    fBaseParams.s1CvRatio = S1CV_RATIO_DEFAULT;
    fBaseParams.s2CvRatio = S2CV_RATIO_DEFAULT;
    std::tuple<uint32_t, uint32_t, uint32_t> bestSplitRes = FuzzyForBestSplit();
    int64_t s1Inner = std::get<0>(bestSplitRes);
    int64_t s1CvInner =
        s1Inner * fBaseParams.s1CvRatio > fBaseParams.s1 ? fBaseParams.s1 : s1Inner * fBaseParams.s1CvRatio;
    int64_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;
    int64_t s1TailTmp = fBaseParams.s1 % s1Inner;
    int64_t s1CvTailTmp = fBaseParams.s1 % s1CvInner;
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    fBaseParams.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1CvInner = s1CvInner;
    fBaseParams.s1Outer = s1Outer;

    int64_t s2Inner = std::get<1>(bestSplitRes);
    int64_t cvS2Inner =
        s2Inner * fBaseParams.s2CvRatio > fBaseParams.s2 ? fBaseParams.s2 : s2Inner * fBaseParams.s2CvRatio;
    int64_t s2Outer = (fBaseParams.s2 + cvS2Inner - 1) / cvS2Inner;
    int64_t s2TailTmp = fBaseParams.s2 % s2Inner;
    int64_t s2CvTailTmp = fBaseParams.s2 % cvS2Inner;
    fBaseParams.s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    fBaseParams.s2CvTail = s2CvTailTmp == 0 ? cvS2Inner : s2CvTailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.cvS2Inner = cvS2Inner;
    fBaseParams.s2Inner = s2Inner;

    OP_LOGI("DoSplit", "s1Inner = %d s1Outer = %d s2Inner = %d s2Outer = %d", s1Inner, s1Outer, s2Inner, s2Outer);
    uint32_t sfmgdInner = std::get<2>(bestSplitRes);
    fBaseParams.sfmgdInner = sfmgdInner;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::DoSparse()
{
    fBaseParams.sparseType = GetSparseType(); // 非确定性计算下获取sparseType
    fBaseParams.deterSparseType = GetDeterSparseTilingKey();
    CalcleDeterParam();
    if (fBaseParams.splitAxis == SplitAxisEnum::BN2 && fBaseParams.isBn2MultiBlk) {
        // Bn2MultiBlk模板有自己的分核
        bool earlyReturn = true;
        bool res = DoBn2MultiBlkSparse();
        // 当BN2多基本块场景，上方函数判断遇到无效行、列后，需要走S1S2模板，性能达到最优
        OP_LOGD("DoBn2MultiBlkSparse", "fBaseParams.isInvalidCol %d, fBaseParams.isInvalidRow %d",
                fBaseParams.isInvalidCol, fBaseParams.isInvalidRow);
        if ((fBaseParams.isInvalidCol || fBaseParams.isInvalidRow)) {
            fBaseParams.isBn2 = false;
            fBaseParams.isBn2MultiBlk = false;
            fBaseParams.isDeterministic = (context_->GetDeterministic() == 1);
            fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
            earlyReturn = false;
        }
        if (earlyReturn) {
            return res;
        }
    }
    if (fBaseParams.isSparse) {
        if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
            GetSparseUnpadBlockInfo();
        } else {
            if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
                fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
                GetSparsePrefixBlockInfo();
            } else {
                GetSparseBlockInfo();
            }
        }
    } else {
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        // block split
        int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
        int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
        int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

        OP_LOGI("DoSparse", "fusedOuter = %d blockFactor = %d blockOuter = %d", fusedOuter, blockFactor, blockOuter);

        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;
        fBaseParams.maxValidBBLen = blockFactor;

        for (int64_t i = 0; i < blockOuter; i++) {
            blockStarts[i] = blockFactor * i;
            blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
            OP_LOGI("DoSparse", "split core, coreid = %d blockStarts = %d blockEnds = %d", i, blockStarts[i],
                    blockEnds[i]);
        }
        for (uint32_t i = blockOuter; i < CORE_LIST_NUM; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }

        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
    }
    // each bit init 1
    std::fill(std::begin(fBaseParams.dqIsNeedDeter), std::end(fBaseParams.dqIsNeedDeter), (uint64_t)-1);
    std::fill(std::begin(fBaseParams.dkDvIsNeedDeter), std::end(fBaseParams.dkDvIsNeedDeter), (uint64_t)-1);
    if (fBaseParams.isDeterministic) {
        GetIsDeterArr();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::DoBn2MultiBlkSparse()
{
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        return GetBlockInfoOfTNDForBn2();
    } else if (fBaseParams.isSparse) {
        return GetSparseBlockInfoBn2();
    } else {
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        // block split, Core partitioning by BN
        int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g;
        int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
        int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
        blockFactor *= (fBaseParams.s1Outer * fBaseParams.s2Outer);
        fusedOuter *= (fBaseParams.s1Outer * fBaseParams.s2Outer);

        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;
        fBaseParams.maxValidBBLen = blockFactor;

        for (int64_t i = 0; i < blockOuter; i++) {
            blockStarts[i] = blockFactor * i;
            blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
            OP_LOGD("DoBn2MultiBlkSparse", "Normally partition, blockStarts[%ld] = %ld, blockEnds[%ld] = %ld", i,
                    blockStarts[i], i, blockEnds[i]);
        }
        for (uint32_t i = static_cast<uint32_t>(blockOuter); i < CORE_LIST_NUM; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }

        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetBlockInfoOfTNDForBn2()
{
    // 二维数组，第一维是batch，第二维的id0存储不乘N的基本块数，id1存每个batch乘N的基本块总数
    std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
    // 二维数组，第一维是batch + 2，第一维的倒数两维存下界和总基本块数(包含n2g)，第二维的最后一维存每个batch的基本块数
    std::vector<std::vector<float>> acturalBlockInfo(fBaseParams.b + NUM_THREE,
                                                     std::vector<float>(fBaseParams.s2Outer + 1));
    FillBlockInfoLoadBalanceForBn2(totalBlockInfo, acturalBlockInfo);

    float maxBlockNumPerCore = BinarySearchMaxBlockNumPerCore(acturalBlockInfo);

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];

    if (!CaclePerCoreBlockInfoBn2(totalBlockInfo, acturalBlockInfo, maxBlockNumPerCore, blockStarts, blockEnds)) {
        return ge::GRAPH_FAILED;
    }

    for (int64_t c = 0; c < fBaseParams.blockOuter; c++) {
        OP_LOGD("GetBlockInfoOfTNDForBn2", "blockNum[%ld], blockStarts = %ld , blockEnds = %ld", c, blockStarts[c],
                blockEnds[c]);
    }

    for (uint32_t c = fBaseParams.blockOuter; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

float FlashAttentionScoreGradTilingUs1s2Bs2David::BinarySearchMaxBlockNumPerCore(
    const std::vector<std::vector<float>>& acturalBlockInfo)
{
    float left = acturalBlockInfo[fBaseParams.b + 1][0];
    float right = acturalBlockInfo[fBaseParams.b][0];
    float mid = 0;
    while (left < right - 1) {
        mid = (left + right) / NUM_TWO;
        if (IsPossible(acturalBlockInfo, mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return right;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::IsPossible(const std::vector<std::vector<float>>& acturalBlockInfo,
                                                            const float possibleMax)
{
    float currentSum = 0;
    uint64_t needCoreNum = 1;
    int64_t n2g = fBaseParams.n2 * fBaseParams.g;
    int64_t bn2g = fBaseParams.b * n2g;
    if (fBaseParams.isBn2MultiBlk) {
        for (int64_t i = 0; i < bn2g; i++) {
            int64_t b = i / n2g;
            float blkNumPerBN = acturalBlockInfo[b][fBaseParams.s2Outer];
            if (currentSum + blkNumPerBN > possibleMax) {
                needCoreNum += 1;
                currentSum = blkNumPerBN;
            } else {
                currentSum += blkNumPerBN;
            }
            if (needCoreNum > fBaseParams.aicNum) {
                return false;
            }
        }
    } else {
        for (int64_t i = 0; i < bn2g; i++) {
            int64_t b = i / n2g;
            for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
                float num = acturalBlockInfo[b][j];
                if (currentSum + num > possibleMax) {
                    needCoreNum += 1;
                    currentSum = num;
                } else {
                    currentSum += num;
                }
                if (needCoreNum > fBaseParams.aicNum) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CaclePerCoreBlockInfoBn2(
    const std::vector<std::vector<int64_t>>& totalBlockInfo, const std::vector<std::vector<float>>& acturalBlockInfo,
    const float maxBlockNumPerCore, int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM])
{
    float currentSum = 0;
    uint64_t coreIdx = 0;
    int64_t n2g = fBaseParams.n2 * fBaseParams.g;
    int64_t bn2g = fBaseParams.b * n2g;
    for (int64_t i = 0; i < bn2g; i++) {
        int64_t b = i / n2g;
        int64_t n = i % n2g;
        float num = acturalBlockInfo[b][fBaseParams.s2Outer];
        if (coreIdx >= fBaseParams.aicNum) {
            OP_LOGD("CaclePerCoreBlockInfoBn2", " Not support BN2_MULTIBLK.");
            return false;
        } else if (currentSum + num > maxBlockNumPerCore) {
            OP_LOGD("CaclePerCoreBlockInfoBn2", " blockIdx = %ld: acturalBlock = %f", coreIdx, currentSum);
            int64_t preBatchBlockNum = b == 0 ? 0 : totalBlockInfo[b - 1][1];
            int64_t preNGBlockNum = n * totalBlockInfo[b][0];

            blockEnds[coreIdx] = preBatchBlockNum + preNGBlockNum;
            blockStarts[coreIdx + 1] = blockEnds[coreIdx];
            coreIdx += 1;
            currentSum = num;
        } else {
            currentSum += num;
        }
    }
    OP_LOGD("CaclePerCoreBlockInfoBn2", " blockIdx = %ld: acturalBlock = %f", coreIdx, currentSum);

    blockStarts[0] = 0;
    blockEnds[coreIdx] = totalBlockInfo[fBaseParams.b - 1][1];

    fBaseParams.blockOuter = coreIdx + 1;
    return true;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::FillBlockInfoLoadBalanceForBn2(
    std::vector<std::vector<int64_t>>& totalBlockInfo, std::vector<std::vector<float>>& acturalBlockInfo)
{
    acturalBlockInfo[fBaseParams.b][0] = 0;     // 存全部累积基本块: bn2g * acutalblocks1s2
    acturalBlockInfo[fBaseParams.b + 1][0] = 0; // 存最大的acutalblocks1s2，用于下界
    OP_LOGD("FillBlockInfoLoadBalanceForBn2", "SparseMode %u, find band index %u", fBaseParams.sparseMode,
            fBaseParams.bandIdx);
    float batchTotalValidBlk;
    std::vector<bool> invalidS1Array;
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];
        batchTotalValidBlk = 0;

        auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;
        invalidS1Array.assign(actualS1Outer, false);
        // 针对S2为0的场景，pre中增加initGm为0的操作
        if ((actualS2Outer == 0) != (actualS1Outer == 0)) {
            fBaseParams.isInvalidCol = (actualS1Outer == 0);
            fBaseParams.isInvalidRow = (actualS2Outer == 0);
        }

        // 对unpad场景的token值做二次校正
        // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(i, actualCalcS1Token, actualCalcS2Token);

        OP_LOGD("FillBlockInfoLoadBalanceForBn2",
                " b idx = %ld: actualS1Len = %ld, actualS2Len = %ld, actualCalcS1Token = %ld, actualCalcS2Token = %ld",
                i, actualS1Len, actualS2Len, actualCalcS1Token, actualCalcS2Token);

        // unpad 场景下s2Outer是按照最大的s2计算得到的
        for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
            if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                acturalBlockInfo[i][j] = 0;
            } else {
                int64_t leftIntersectionPoint = std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, 0L);
                int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                          actualS2Len - fBaseParams.cvS2Inner * j :
                                          fBaseParams.cvS2Inner;

                float acturalS1Begin =
                    leftIntersectionPoint > int64_t(actualS1Len) ? actualS1Len : leftIntersectionPoint;
                float acturalS1End = static_cast<float>(std::min(
                    int64_t(actualS1Len), std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, 0L)));
                float acturalS1Num = acturalS1Begin > acturalS1End ? 0 : acturalS1End - acturalS1Begin;
                // float acturalS2Num = static_cast<float>(cvBlockTail);
                acturalBlockInfo[i][j] = acturalS1Num / static_cast<float>(fBaseParams.s1CvInner);
                batchTotalValidBlk += acturalBlockInfo[i][j];
                acturalBlockInfo[fBaseParams.b][0] += acturalBlockInfo[i][j] * fBaseParams.n2 * fBaseParams.g;

                if (acturalS1Begin >= acturalS1End) {
                    fBaseParams.isInvalidCol = true;
                }
                // check invalid row or col block for BN2
                for (size_t k = 0; k < invalidS1Array.size(); k++) {
                    if (k >= acturalS1Begin && k < acturalS1End) {
                        invalidS1Array[k] = true;
                    }
                }
            }
        }

        // BN2场景下检查是否无效基本块行，用于清零GM
        for (size_t j = 0; j < invalidS1Array.size(); j++) {
            if (!invalidS1Array[j]) {
                fBaseParams.isInvalidRow = true;
                break;
            }
        }

        if (i == 0) {
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
        acturalBlockInfo[i][fBaseParams.s2Outer] = batchTotalValidBlk;
        OP_LOGD("FillBlockInfoLoadBalanceForBn2", " batchid = %ld: acturalBlock = %f", i,
                acturalBlockInfo[i][fBaseParams.s2Outer]);
        acturalBlockInfo[fBaseParams.b + 1][0] = acturalBlockInfo[fBaseParams.b + 1][0] < batchTotalValidBlk ?
                                                     batchTotalValidBlk :
                                                     acturalBlockInfo[fBaseParams.b + 1][0]; // 逐轮迭代，得到贪心的下界
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetSparseBlockInfoBn2()
{
    // [s2OuterIdx][begin, end, length]
    int64_t(*parseInfo)[ARRAY_LENGTH] = new int64_t[fBaseParams.s2Outer][ARRAY_LENGTH];
    GetParseS1S2OuterInfo(parseInfo);
    int64_t s1s2oCount = parseInfo[fBaseParams.s2Outer - 1][LENGTH_IDX];

    // block split
    int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g;
    int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
    int64_t bnTail = fusedOuter % blockFactor;
    OP_LOGD("Sparse", "GetSparseBlockInfoBn2, bnNum = %ld: bnFactor = %ld, bnTail = %ld", fusedOuter, blockFactor,
            bnTail);

    fusedOuter *= s1s2oCount;
    blockFactor *= s1s2oCount;
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = fBaseParams.blockFactor;

    int64_t bIdx = 0;
    int64_t bTail = 0;
    int64_t n2Idx = 0;
    int64_t n2Tail = 0;
    int64_t gIdx = 0;
    int64_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s2oIdx = 0;

    int64_t n2gs1s2o = fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    int64_t gs1s2o = fBaseParams.g * s1s2oCount;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] =
        fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
    for (int64_t c = 1; c < blockOuter; c++) {
        // cal indx for total bngs1os2o(sparse)
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        bIdx = currentIdx / n2gs1s2o;
        bTail = currentIdx % n2gs1s2o;
        n2Idx = bTail / gs1s2o;
        n2Tail = bTail % gs1s2o;
        gIdx = n2Tail / s1s2oCount;
        gTail = n2Tail % s1s2oCount;

        OP_LOGD("Sparse",
                "GetSparseBlockInfoBn2, currentIdx = %ld: bIdx = %ld, bTail = %ld, n2Idx = %ld, n2Tail = %ld, gIdx = "
                "%ld, gTail = %ld",
                currentIdx, bIdx, bTail, n2Idx, n2Tail, gIdx, gTail);
        GetCommonS1S2OuterIndex(parseInfo, gTail, s1oIdx, s2oIdx);

        // total indx in bngs1os2o (range is [))
        blockStarts[c] = (((bIdx * fBaseParams.n2 + n2Idx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2oIdx) *
                             fBaseParams.s1Outer +
                         s1oIdx + 1;
        blockEnds[c - 1] = blockStarts[c];
        OP_LOGD("Sparse", "GetSparseBlockInfoBn2, blockStarts[%ld] = %ld:", c, blockStarts[c]);
    }
    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    // free tensor
    delete[] parseInfo;
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CheckExceedL2Cache()
{
    std::array<int64_t, CORE_LIST_NUM> dqOffset;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffset;
    std::array<int64_t, CORE_LIST_NUM> loopIdx;
    std::set<int> dqOffsetSet;
    std::set<int> dkDvOffsetSet;
    uint64_t usedl2CacheSize = 0;
    int64_t calcNum = 0;
    int32_t inputSize = FP16_BYTES;
    std::fill(std::begin(loopIdx), std::end(loopIdx), static_cast<int64_t>(0));

    if (fBaseParams.queryType == ge::DT_FLOAT) {
        inputSize = FP32_BYTES;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        inputSize = FP16_BYTES;
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN ||
               fBaseParams.queryType == ge::DT_HIFLOAT8) {
        inputSize = 1;
    }

    bool isExceed = false;
    while (calcNum < fBaseParams.maxValidBBLen) {
        for (uint16_t cBlockIdx = 0; cBlockIdx < fBaseParams.blockOuter; cBlockIdx++) {
            while (!IsValid(fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx]) &&
                   (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] < fBaseParams.blockEnds[cBlockIdx])) {
                loopIdx[cBlockIdx]++;
            }
            if (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] >= fBaseParams.blockEnds[cBlockIdx]) {
                dqOffset[cBlockIdx] = OUTINDEX;
                dkDvOffset[cBlockIdx] = OUTINDEX;
                continue;
            }
            int64_t validBlockIdx = fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx];
            GetOffset(dqOffset[cBlockIdx], dkDvOffset[cBlockIdx], validBlockIdx);
            loopIdx[cBlockIdx]++;

            if (dqOffsetSet.find(dqOffset[cBlockIdx]) == dqOffsetSet.end()) {
                dqOffsetSet.insert(dqOffset[cBlockIdx]);
                // qSize + dxSize + dqSize(btyes) + ySize，3 means q, dx and y
                usedl2CacheSize += (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d * inputSize * NUM_THREE +
                                    fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d * FP32_BYTES);
            }
            if (dkDvOffsetSet.find(dkDvOffset[cBlockIdx]) == dkDvOffsetSet.end()) {
                dkDvOffsetSet.insert(dkDvOffset[cBlockIdx]);
                // kSize + vSize + dkSize + dvSize(btyes)，2 means k/dk and v/dv
                usedl2CacheSize += (fBaseParams.s2Inner * fBaseParams.d * inputSize * NUM_TWO +
                                    fBaseParams.s2Inner * fBaseParams.d * FP32_BYTES * NUM_TWO);
            }
            if (usedl2CacheSize > fBaseParams.l2CacheSize) {
                isExceed = true;
                break;
            }
        }
        if (isExceed) {
            break;
        }
        calcNum++;
    }
    if (!isExceed) {
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
            if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBN2GS1S2)) {
                usedl2CacheSize += (fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.s2);
            } else if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBS1S2)) {
                usedl2CacheSize += (fBaseParams.b * fBaseParams.s1 * fBaseParams.s2);
            } else if (fBaseParams.attenMaskShapeType == static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2)) {
                usedl2CacheSize += (fBaseParams.s1 * fBaseParams.s2);
            }
        } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                   fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
            usedl2CacheSize += COMPRESS_ATTEN_MASK_SIZE;
        }
        isExceed = usedl2CacheSize > fBaseParams.l2CacheSize;
    }
    return isExceed;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CheckIsLargeInvalidBlk(FuzzyBaseInfoParamsDavid& fBaseParams)
{
    if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) &&
        (fBaseParams.s1Outer >= 0 && fBaseParams.s2Outer >= 0) && (fBaseParams.s1Outer < fBaseParams.s2Outer) &&
        (fBaseParams.d <= static_cast<uint32_t>(ConstAxisTemplateNum::NUM256))) {
        return (fBaseParams.s2Outer - fBaseParams.s1Outer) * fBaseParams.s1Outer >= LARGE_INVALID_NUM;
    }
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::InitTilingData()
{
    if (IsNewDeter()) {
        FagTilingWithTemplateT* tilingData = this->context_->GetTilingData<FagTilingWithTemplateT>();
        if (tilingData == nullptr) {
            OP_LOGE("InitTilingData", "InitTilingData failed.");
            return ge::GRAPH_FAILED;
        }
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams;
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams;
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList;
        preTilingData_ = &tilingData->preTilingData;
        postTilingData_ = &tilingData->postTilingData;
        preSfmgTilingData_ = &tilingData->preSfmgTilingData;
        deterParam_ = &tilingData->deterParam;
    } else {
        FagTilingWithTemplateF* tilingData = this->context_->GetTilingData<FagTilingWithTemplateF>();
        if (tilingData == nullptr) {
            OP_LOGE("InitTilingData", "InitTilingData failed.");
            return ge::GRAPH_FAILED;
        }
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams;
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams;
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList;
        preTilingData_ = &tilingData->preTilingData;
        postTilingData_ = &tilingData->postTilingData;
        preSfmgTilingData_ = &tilingData->preSfmgTilingData;
        if (tndBaseInfo.isTndSwizzle) {
            tndSwizzleParam_ = &tilingData->tndSwizzleParam;
        }
    }

    if (s1s2BNGS1S2BaseParams_ == nullptr || s1s2BNGS1S2SplitCoreParams_ == nullptr ||
        s1s2BNGS1S2BlockNumList_ == nullptr || preTilingData_ == nullptr || postTilingData_ == nullptr) {
        OP_LOGE("InitTilingData", "InitTilingData failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::IsValid(int64_t blockIdx)
{
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        int64_t resbaseIdx = blockIdx;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
            int64_t s1OuterTmp = (actualS1Len + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (actualS2Len + fBaseParams.s2Inner * S2CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s2Inner * S2CV_RATIO_DEFAULT);
            int64_t totalBaseIdx = fBaseParams.n2 * fBaseParams.g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
                int64_t s2oDimIdx = gDimTail / s1OuterTmp;
                int64_t s1oDimIdx = gDimTail % s1OuterTmp;
                int64_t s2IdxLeft = s2oDimIdx * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT;
                int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT, actualS2Len);
                if (fBaseParams.attenMaskOptional != EMPTY_TENSOR) {
                    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
                        int64_t s2IgnoredEndLen =
                            actualS1Len -
                            static_cast<int64_t>(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1));
                        int64_t s2EndLen = 0;
                        if (actualS2Len > s2IgnoredEndLen) {
                            s2EndLen = actualS2Len - s2IgnoredEndLen;
                        }
                        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
                            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
                            s2EndLen = std::min(std::max(s2EndLen, static_cast<int64_t>(fBaseParams.prefixN[bIdx])),
                                                actualS2Len);
                        }
                        bool isValid = s2IdxLeft < s2EndLen;
                        return isValid;
                    }
                    int64_t actualCalcS1Token = fBaseParams.s1Token;
                    int64_t actualCalcS2Token = fBaseParams.s2Token;
                    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
                    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
                        bIdx != fBaseParams.bandIdx) {
                        actualCalcS1Token = static_cast<int64_t>(INT32_MAX) + actualS1Len - actualS2Len;
                        actualCalcS2Token = static_cast<int64_t>(0) - actualS1Len + actualS2Len;
                    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
                               bIdx != fBaseParams.bandIdx) {
                        actualCalcS1Token = INT32_MAX;
                        actualCalcS2Token = 0;
                    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                               fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
                               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
                                bIdx == fBaseParams.bandIdx) ||
                               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
                                bIdx == fBaseParams.bandIdx)) {
                        actualCalcS1Token = fBaseParams.s1Token + actualS1Len - actualS2Len;
                        actualCalcS2Token = fBaseParams.s2Token - actualS1Len + actualS2Len;
                    }
                    int64_t s2SparseLeft =
                        std::max(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * s1oDimIdx - actualCalcS1Token,
                                 static_cast<int64_t>(0));
                    s2SparseLeft = s2SparseLeft >> 6 << 6; // 6 is for 64 align
                    int64_t s2SparseRight =
                        AlignTo(std::min(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), fBaseParams.s1) +
                                    actualCalcS2Token,
                                static_cast<int64_t>(64));
                    s2SparseRight = std::min(s2SparseRight, actualS2Len);
                    bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                    return isValid;
                } else {
                    return true;
                }
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        return false;
    } else {
        int64_t gDimTail = blockIdx % (fBaseParams.s1Outer * fBaseParams.s2Outer);
        int64_t s2oDimIdx = gDimTail / fBaseParams.s1Outer;
        int64_t s1oDimIdx = gDimTail % fBaseParams.s1Outer;
        int64_t s2IdxLeft = s2oDimIdx * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT;
        int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fBaseParams.s2Inner * S2CV_RATIO_DEFAULT, fBaseParams.s2);
        if (fBaseParams.attenMaskOptional != EMPTY_TENSOR) {
            if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
                fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
                int64_t s2IgnoredEndLen =
                    static_cast<int64_t>(fBaseParams.s1) -
                    static_cast<int64_t>(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1));
                int64_t s2EndLen = static_cast<int64_t>(fBaseParams.s2) > s2IgnoredEndLen ?
                                       static_cast<int64_t>(fBaseParams.s2) - s2IgnoredEndLen :
                                       0;
                if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
                    fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
                    int64_t curBIdx =
                        blockIdx / (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
                    s2EndLen = std::min(std::max(s2EndLen, static_cast<int64_t>(fBaseParams.prefixN[curBIdx])),
                                        static_cast<int64_t>(fBaseParams.s2));
                }
                bool isValid = s2IdxLeft < s2EndLen;
                return isValid;
            } else {
                int64_t s2SparseLeft =
                    std::max(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * s1oDimIdx - fBaseParams.s1Token,
                             static_cast<int64_t>(0));
                s2SparseLeft = s2SparseLeft >> 6 << 6; // 6 is for 64 align
                int64_t s2SparseRight =
                    AlignTo(std::min(fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), fBaseParams.s1) +
                                fBaseParams.s2Token,
                            static_cast<int64_t>(64));
                s2SparseRight = std::min(s2SparseRight, fBaseParams.s2);
                bool isValid = s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
                return isValid;
            }
        }
        return true;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::GetOffset(int64_t& currentDqOffset, int64_t& currentDkDvOffset,
                                                           int64_t blockIdx)
{
    int64_t boIdx = 0;
    int64_t bDimTail = 0;
    int64_t n2oIdx = 0;
    int64_t n2DimTail = 0;
    int64_t goIdx = 0;
    int64_t gDimTail = 0;
    int64_t s2oIdx = 0;
    int64_t s1oIdx = 0;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t s2Offset = 0;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        int64_t resbaseIdx = blockIdx;
        for (int64_t bIdx = 0; bIdx < fBaseParams.b; bIdx++) {
            int64_t actualS1Len = fBaseParams.actualSeqQlen[bIdx];
            int64_t actualS2Len = fBaseParams.actualSeqKvlen[bIdx];
            int64_t s1OuterTmp = (actualS1Len + fBaseParams.s1Inner * S1CV_RATIO_DEFAULT - 1) /
                                 (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT);
            int64_t s2OuterTmp = (actualS2Len + fBaseParams.s2Inner - 1) / fBaseParams.s2Inner;
            int64_t totalBaseIdx = fBaseParams.n2 * fBaseParams.g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                boIdx = bIdx;
                bDimTail = resbaseIdx;
                n2oIdx = bDimTail / (fBaseParams.g * s1OuterTmp * s2OuterTmp);
                n2DimTail = bDimTail % (fBaseParams.g * s1OuterTmp * s2OuterTmp);
                goIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                s2oIdx = gDimTail / s1OuterTmp;
                s1oIdx = gDimTail % s1OuterTmp;
                break;
            } else {
                resbaseIdx -= totalBaseIdx;
            }
        }
        // caculate dq offset
        for (int64_t bIdx = 0; bIdx < boIdx; bIdx++) {
            bOffset += fBaseParams.actualSeqQlen[bIdx] * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        }
        s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
        n2Offset = n2oIdx * fBaseParams.g * fBaseParams.d;
        gOffset = goIdx * fBaseParams.d;
        currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
        // caculate dk dv offset
        bOffset = 0;
        for (int64_t bIdx = 0; bIdx < boIdx; bIdx++) {
            bOffset += fBaseParams.actualSeqKvlen[bIdx] * (fBaseParams.n2 * fBaseParams.d);
        }
        s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.n2 * fBaseParams.d;
        n2Offset = n2oIdx * fBaseParams.d;
        currentDkDvOffset = bOffset + n2Offset + s2Offset;
    } else {
        boIdx = blockIdx / (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        bDimTail = blockIdx % (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        n2oIdx = bDimTail / (fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        n2DimTail = bDimTail % (fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer);
        goIdx = n2DimTail / (fBaseParams.s1Outer * fBaseParams.s2Outer);
        gDimTail = n2DimTail % (fBaseParams.s1Outer * fBaseParams.s2Outer);
        s2oIdx = gDimTail / fBaseParams.s1Outer;
        s1oIdx = gDimTail % fBaseParams.s1Outer;
        // caculate dq offset
        if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            gOffset = goIdx * (fBaseParams.s1 * fBaseParams.d);
            s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
            s1Offset = s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT *
                       (fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
            gOffset = goIdx * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.g * fBaseParams.s1 * fBaseParams.d);
            s1Offset =
                s1oIdx * fBaseParams.s1Inner * S1CV_RATIO_DEFAULT * (fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.g * fBaseParams.d);
            gOffset = goIdx * fBaseParams.d;
        }
        currentDqOffset = bOffset + n2Offset + gOffset + s1Offset;
        // caculate dk dv offset
        if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
            n2Offset = n2oIdx * (fBaseParams.s2 * fBaseParams.d);
            s2Offset = s2oIdx * fBaseParams.s2Inner * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) {
            s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.b * fBaseParams.n2 * fBaseParams.d);
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.d);
            n2Offset = n2oIdx * fBaseParams.d;
        } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) {
            bOffset = boIdx * (fBaseParams.n2 * fBaseParams.s2 * fBaseParams.d);
            s2Offset = s2oIdx * fBaseParams.s2Inner * (fBaseParams.n2 * fBaseParams.d);
            n2Offset = n2oIdx * fBaseParams.d;
        }
        currentDkDvOffset = bOffset + n2Offset + s2Offset;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::JudgeIsNeedDeter(std::array<int64_t, CORE_LIST_NUM>& dqOffset,
                                                                  std::array<int64_t, CORE_LIST_NUM>& dkDvOffset,
                                                                  std::array<int64_t, CORE_LIST_NUM>& dqOffsetpre,
                                                                  std::array<int64_t, CORE_LIST_NUM>& dkDvOffsetpre,
                                                                  int64_t calcNum, bool& noNeedDeter,
                                                                  bool& dqNeedDeterpre, bool& dkDvNeedDeterpre)
{
    bool dqNeedDeter = false;
    bool dkDvNeedDeter = false;
    for (uint16_t i = 0; i < fBaseParams.blockOuter - 1; i++) {
        for (uint16_t j = i + 1; j < fBaseParams.blockOuter; j++) {
            if (!dqNeedDeter && dqOffset[i] == dqOffset[j] && dqOffset[i] != OUTINDEX) {
                dqNeedDeter = true;
            }
            if (!dkDvNeedDeter && dkDvOffset[i] == dkDvOffset[j] && dkDvOffset[i] != OUTINDEX) {
                dkDvNeedDeter = true;
            }
        }
    }

    if (calcNum != 0 && ((!dqNeedDeter && dqNeedDeterpre) || (!dkDvNeedDeter && dkDvNeedDeterpre))) {
        for (uint16_t i = 0; i < fBaseParams.blockOuter; i++) {
            for (uint16_t j = 0; j < fBaseParams.blockOuter; j++) {
                if (!dqNeedDeter && dqNeedDeterpre && dqOffset[i] == dqOffsetpre[j] && dqOffset[i] != OUTINDEX) {
                    dqNeedDeter = true;
                }
                if (!dkDvNeedDeter && dkDvNeedDeterpre && dkDvOffset[i] == dkDvOffsetpre[j] &&
                    dkDvOffset[i] != OUTINDEX) {
                    dkDvNeedDeter = true;
                }
            }
        }
    }

    dqNeedDeterpre = dqNeedDeter;
    dkDvNeedDeterpre = dkDvNeedDeter;

    for (uint16_t i = 0; i < fBaseParams.blockOuter; i++) {
        dqOffsetpre[i] = dqOffset[i];
        dkDvOffsetpre[i] = dkDvOffset[i];
    }

    noNeedDeter = noNeedDeter && !dqNeedDeter && !dkDvNeedDeter;
    // caculate index and position
    int64_t index = calcNum / 64;
    int64_t bitPosition = calcNum % 64;
    if (index >= 0 && index < 32) { // 32 is first half
        if (dqNeedDeter) {
            fBaseParams.dqIsNeedDeter[index] |= (1ULL << bitPosition);
        } else {
            fBaseParams.dqIsNeedDeter[index] &= ~(1ULL << bitPosition);
        }
        if (dkDvNeedDeter) {
            fBaseParams.dkDvIsNeedDeter[index] |= (1ULL << bitPosition);
        } else {
            fBaseParams.dkDvIsNeedDeter[index] &= ~(1ULL << bitPosition);
        }
    } else {
        OP_LOGI("JudgeIsNeedDeter", "calcNum = %d out of bounds", calcNum);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::GetIsDeterArr()
{
    std::array<int64_t, CORE_LIST_NUM> dqOffset;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffset;
    std::array<int64_t, CORE_LIST_NUM> dqOffsetpre;
    std::array<int64_t, CORE_LIST_NUM> dkDvOffsetpre;
    std::array<int64_t, CORE_LIST_NUM> loopIdx;
    bool dqNeedDeterpre = false;
    bool dkDvNeedDeterpre = false;
    int64_t calcNum = 0;
    std::fill(std::begin(loopIdx), std::end(loopIdx), (int64_t)0);
    while (calcNum < fBaseParams.maxValidBBLen) {
        for (uint16_t cBlockIdx = 0; cBlockIdx < fBaseParams.blockOuter; cBlockIdx++) {
            while (!IsValid(fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx]) &&
                   (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] < fBaseParams.blockEnds[cBlockIdx])) {
                loopIdx[cBlockIdx]++;
            }
            if (fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx] >= fBaseParams.blockEnds[cBlockIdx]) {
                dqOffset[cBlockIdx] = OUTINDEX;
                dkDvOffset[cBlockIdx] = OUTINDEX;
                continue;
            }
            int64_t validBlockIdx = fBaseParams.blockStarts[cBlockIdx] + loopIdx[cBlockIdx];
            GetOffset(dqOffset[cBlockIdx], dkDvOffset[cBlockIdx], validBlockIdx);
            loopIdx[cBlockIdx]++;
        }
        JudgeIsNeedDeter(dqOffset, dkDvOffset, dqOffsetpre, dkDvOffsetpre, calcNum, fBaseParams.noNeedDeter,
                         dqNeedDeterpre, dkDvNeedDeterpre);
        calcNum++;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::DoPreTiling()
{
    uint32_t inputBufferLen = PRE_BUFFER_SIZE; // x / 8 + 2 * x + 32 = fBaseParams.ubSize
    uint64_t singleUBProcessNum = CAST_BUFFER_LEN / 2;

    uint64_t maskSize = AlignTo(fBaseParams.dropMaskSize, static_cast<uint64_t>(BOOL_BLOCK_NUMS));
    uint64_t singleCoreNum = AlignTo(CeilDivideBy(maskSize, static_cast<uint64_t>(fBaseParams.blockOuter)),
                                     static_cast<uint64_t>(BOOL_BLOCK_NUMS));
    uint32_t maskUsedCoreNum = fBaseParams.blockOuter * CV_CORE_RATIO;

    uint64_t tailCoreNum = maskSize - (maskUsedCoreNum - 1) * singleCoreNum;

    uint32_t singleCoreUBLoop = static_cast<uint32_t>(CeilDivideBy(singleCoreNum, singleUBProcessNum));
    uint32_t tailCoreUBLoop = static_cast<uint32_t>(CeilDivideBy(tailCoreNum, singleUBProcessNum));

    uint32_t singleCoreUBLastLoopNum =
        static_cast<uint32_t>(singleCoreNum - (singleCoreUBLoop - 1) * singleUBProcessNum);
    uint32_t tailCoreUBLastLoopNum = static_cast<uint32_t>(tailCoreNum - (tailCoreUBLoop - 1) * singleUBProcessNum);

    preTilingData_->set_maskCoreNum(maskUsedCoreNum);
    preTilingData_->set_castBufferLen(CAST_BUFFER_LEN);
    preTilingData_->set_outputBufferLen(OUTPUT_BUFFER_LEN);
    preTilingData_->set_inputBufferLen(inputBufferLen);
    preTilingData_->set_singleUBProcessNum(static_cast<uint32_t>(singleUBProcessNum));
    preTilingData_->set_maskSingleCoreNum(singleCoreNum); // size == num
    preTilingData_->set_maskSingleCoreLoop(singleCoreUBLoop);
    preTilingData_->set_maskLastLoopNum(singleCoreUBLastLoopNum);
    preTilingData_->set_maskTailCoreLoop(tailCoreUBLoop);
    preTilingData_->set_maskTailCoreLastLoopNum(tailCoreUBLastLoopNum);

    uint32_t qPreBlockFactor = (static_cast<uint32_t>(fBaseParams.qSize) + maskUsedCoreNum - 1) / maskUsedCoreNum;
    uint32_t qPreBlockTotal = (static_cast<uint32_t>(fBaseParams.qSize) + qPreBlockFactor - 1) / qPreBlockFactor;
    uint32_t qPreTailNumTmp = static_cast<uint32_t>(fBaseParams.qSize) % qPreBlockFactor;
    uint32_t qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

    uint32_t kPreBlockFactor = (static_cast<uint32_t>(fBaseParams.kSize) + maskUsedCoreNum - 1) / maskUsedCoreNum;
    uint32_t kPreBlockTotal = (static_cast<uint32_t>(fBaseParams.kSize) + kPreBlockFactor - 1) / kPreBlockFactor;
    uint32_t kPreTailNumTmp = static_cast<uint32_t>(fBaseParams.kSize) % kPreBlockFactor;
    uint32_t kPreTailNum = kPreTailNumTmp == 0 ? kPreBlockFactor : kPreTailNumTmp;

    uint32_t vPreBlockFactor = (static_cast<uint32_t>(fBaseParams.vSize) + maskUsedCoreNum - 1) / maskUsedCoreNum;
    uint32_t vPreBlockTotal = (static_cast<uint32_t>(fBaseParams.vSize) + vPreBlockFactor - 1) / vPreBlockFactor;
    uint32_t vPreTailNumTmp = static_cast<uint32_t>(fBaseParams.vSize) % vPreBlockFactor;
    uint32_t vPreTailNum = vPreTailNumTmp == 0 ? vPreBlockFactor : vPreTailNumTmp;

    if (fBaseParams.sinkOptional == NORMAL_TENSOR) {
        fBaseParams.s1SinkOuter = fBaseParams.s1Outer * CV_CORE_RATIO;
        fBaseParams.s2SinkOuter = fBaseParams.s2Outer;
        fBaseParams.sinkSize =
            fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1SinkOuter * fBaseParams.s2SinkOuter;
        uint64_t sinkWorkSpaceSize = (fBaseParams.sinkSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        uint64_t sinkPreBlockFactor = (sinkWorkSpaceSize + maskUsedCoreNum - 1) / maskUsedCoreNum;
        uint64_t sinkPreBlockTotal = (sinkWorkSpaceSize + sinkPreBlockFactor - 1) / sinkPreBlockFactor;
        uint64_t sinkPreTailNumTmp = sinkWorkSpaceSize % sinkPreBlockFactor;
        uint64_t sinkPreTailNum =
            sinkPreTailNumTmp == static_cast<uint64_t>(0) ? sinkPreBlockFactor : sinkPreTailNumTmp;
        preTilingData_->set_sinkPreBlockFactor(sinkPreBlockFactor);
        preTilingData_->set_sinkPreBlockTotal(sinkPreBlockTotal);
        preTilingData_->set_sinkPreBlockTail(sinkPreTailNum);
        OP_LOGI(
            context_,
            "FAG sinkOptional, fBaseParams.s1SinkOuter is %ld, fBaseParams.s2SinkOuter = %ld, fBaseParams.sinkSize = "
            "%ld, maskUsedCoreNum = %ld, sinkPreBlockFactor = %ld, sinkPreBlockTotal = %ld, sinkPreTailNum = %ld.",
            fBaseParams.s1SinkOuter, fBaseParams.s2SinkOuter, fBaseParams.sinkSize, maskUsedCoreNum, sinkPreBlockFactor,
            sinkPreBlockTotal, sinkPreTailNum);
    }

    uint64_t maskPreBlockTotal = fBaseParams.dropMaskSize;
    preTilingData_->set_qPreBlockFactor(qPreBlockFactor);
    preTilingData_->set_qPreBlockTotal(qPreBlockTotal);
    preTilingData_->set_qPreBlockTail(qPreTailNum);
    preTilingData_->set_kPreBlockFactor(kPreBlockFactor);
    preTilingData_->set_kPreBlockTotal(kPreBlockTotal);
    preTilingData_->set_kPreBlockTail(kPreTailNum);
    preTilingData_->set_vPreBlockFactor(vPreBlockFactor);
    preTilingData_->set_vPreBlockTotal(vPreBlockTotal);
    preTilingData_->set_vPreBlockTail(vPreTailNum);
    preTilingData_->set_dropoutIsDivisibleBy8(fBaseParams.dropoutIsDivisibleBy8);
    preTilingData_->set_maskPreBlockTotal(maskPreBlockTotal);
    preTilingData_->set_sValueZeroUnderTND(fBaseParams.sValueZeroUnderTND);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::DoPostTiling()
{
    uint32_t postUbBaseSize = fBaseParams.hasRope ? ROPE_POST_BASE * FP16_BYTES : DAVID_POST_BASE * FP16_BYTES;
    uint32_t dAlign16 = AlignData(fBaseParams.d, 16);
    uint32_t qPostBaseNum = fBaseParams.hasRope ? ROPE_POST_BASE : DAVID_POST_BASE / (dAlign16) * (dAlign16);
    uint32_t qPostBlockTotal = static_cast<uint32_t>(fBaseParams.qSize);
    uint32_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    uint32_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    uint32_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    uint32_t qPostBlockFactor =
        (qPostBlockOuterTotal + fBaseParams.blockOuter * CV_CORE_RATIO - 1) / (fBaseParams.blockOuter * CV_CORE_RATIO);

    uint32_t kPostBaseNum = postUbBaseSize / FP16_BYTES / (dAlign16) * (dAlign16);
    uint32_t kPostBlockTotal = static_cast<uint32_t>(fBaseParams.kSize);
    uint32_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    uint32_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    uint32_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    uint32_t kPostBlockFactor =
        (kPostBlockOuterTotal + fBaseParams.blockOuter * CV_CORE_RATIO - 1) / (fBaseParams.blockOuter * CV_CORE_RATIO);

    uint32_t vPostBaseNum = postUbBaseSize / FP16_BYTES;
    uint32_t vPostBlockTotal = static_cast<uint32_t>(fBaseParams.vSize);
    uint32_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    uint32_t vPostTailNum = vPostTailNumTmp == 0 ? vPostBaseNum : vPostTailNumTmp;
    uint32_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - 1) / vPostBaseNum;
    uint32_t vPostBlockFactor =
        (vPostBlockOuterTotal + fBaseParams.blockOuter * CV_CORE_RATIO - 1) / (fBaseParams.blockOuter * CV_CORE_RATIO);

    if (fBaseParams.sinkOptional == NORMAL_TENSOR) {
        uint64_t sinkPostBaseNum = postUbBaseSize / FP16_BYTES;
        uint64_t sinkReduceAxis =
            (fBaseParams.b - fBaseParams.tailZeroCount) * fBaseParams.s1SinkOuter * fBaseParams.s2SinkOuter;
        uint64_t sinkPostTailNumTmp = sinkReduceAxis % sinkPostBaseNum;
        uint64_t sinkPostTailNum =
            sinkPostTailNumTmp == static_cast<uint64_t>(0) ? sinkPostBaseNum : sinkPostTailNumTmp;
        uint64_t sinkPostBlockTotal = fBaseParams.n1;
        uint64_t sinkPostBlockFactor =
            (fBaseParams.n1 + fBaseParams.blockOuter * CV_CORE_RATIO - 1) / (fBaseParams.blockOuter * CV_CORE_RATIO);
        postTilingData_->set_sinkReduceAxis(sinkReduceAxis);
        postTilingData_->set_sinkPostBlockTotal(sinkPostBlockTotal);
        postTilingData_->set_sinkPostBlockFactor(sinkPostBlockFactor);
        postTilingData_->set_sinkPostTailNum(sinkPostTailNum);
    }

    postTilingData_->set_postUbBaseSize(postUbBaseSize);
    postTilingData_->set_qPostBlockFactor(qPostBlockFactor);
    postTilingData_->set_qPostBlockTotal(qPostBlockTotal);
    postTilingData_->set_qPostBaseNum(qPostBaseNum);
    postTilingData_->set_qPostTailNum(qPostTailNum);

    postTilingData_->set_kPostBlockFactor(kPostBlockFactor);
    postTilingData_->set_kPostBlockTotal(kPostBlockTotal);
    postTilingData_->set_kPostBaseNum(kPostBaseNum);
    postTilingData_->set_kPostTailNum(kPostTailNum);

    postTilingData_->set_vPostBlockFactor(vPostBlockFactor);
    postTilingData_->set_vPostBlockTotal(vPostBlockTotal);
    postTilingData_->set_vPostBaseNum(vPostBaseNum);
    postTilingData_->set_vPostTailNum(vPostTailNum);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::DoPreSfmgTiling()
{
    bool isDNotEqual = fBaseParams.d != fBaseParams.d1;
    uint32_t value_dAlign =
        (fBaseParams.sfmgdInner == static_cast<uint32_t>(ConstAxisTemplateNum::NUM192) && isDNotEqual) ?
            static_cast<uint32_t>(ConstAxisTemplateNum::NUM256) :
            fBaseParams.sfmgdInner;

    int64_t normalAxisSize = 0;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        normalAxisSize = fBaseParams.t1 * fBaseParams.n2 * fBaseParams.g;
    } else {
        normalAxisSize = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1;
    }

    int32_t inputSize = FP16_BYTES;
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        inputSize = FP32_BYTES;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        inputSize = FP16_BYTES;
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) {
        inputSize = 1;
    }
    uint32_t availUbSize = fBaseParams.ubSize - UB_RESERVE_SPACE;
    uint32_t inputBufferLen =
        availUbSize / (value_dAlign * inputSize * 4 + 2 * 8 * FP32_BYTES) * value_dAlign * inputSize;
    uint32_t outputBufferLen = availUbSize / (value_dAlign * inputSize * 4 + 2 * 8 * FP32_BYTES) * 8 * FP32_BYTES;

    // 计算单核的计算量
    int64_t usedCoreNum = fBaseParams.blockOuter * CV_CORE_RATIO;
    int64_t normalCoreSize = CeilCommon(normalAxisSize, usedCoreNum);
    usedCoreNum = CeilCommon(normalAxisSize, normalCoreSize);
    int64_t tailCoreSize = normalAxisSize - (usedCoreNum - 1) * normalCoreSize;

    // 计算单loop的计算量及loop次数
    /// NOTE: FP8 Per-Block量化方案中，每核在s1方向上切分以 128 对齐（由于UB内存设置，dim=256时，nBurst取128）
    int64_t singleLoopNBurstNum =
        (fBaseParams.queryType == ge::DT_FLOAT8_E5M2 || fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) ?
            128 :
            inputBufferLen / inputSize / value_dAlign;
    int64_t normalCoreLoopTimes = CeilCommon(normalCoreSize, singleLoopNBurstNum);
    int64_t normalCoreLastLoopNBurstNum = normalCoreSize - (normalCoreLoopTimes - 1) * singleLoopNBurstNum;
    int64_t tailCoreLoopTimes = CeilCommon(tailCoreSize, singleLoopNBurstNum);
    int64_t tailCoreLastLoopNBurstNum = tailCoreSize - (tailCoreLoopTimes - 1) * singleLoopNBurstNum;

    OP_LOGI("DoPreSfmgTiling",
            "DoPreSfmgTiling, usedCoreNum = %d, ubsize = %d, value_dAlign = %d,"
            "normalAxisSize = %d, reals1percore = %d, inputBufferLen is %d, outputBufferLen is %d."
            "singleLoopNBurstNum = %d, normalCoreLoopTimes is %d, normalCoreLastLoopNBurstNum is %d."
            "tailCoreLoopTimes = %d, tailCoreLastLoopNBurstNum is %d.",
            usedCoreNum, availUbSize, value_dAlign, normalAxisSize, normalCoreSize, inputBufferLen, outputBufferLen,
            singleLoopNBurstNum, normalCoreLoopTimes, normalCoreLastLoopNBurstNum, tailCoreLoopTimes,
            tailCoreLastLoopNBurstNum);
    preSfmgTilingData_->set_usedCoreNum(usedCoreNum);
    preSfmgTilingData_->set_inputBufferLen(inputBufferLen);
    preSfmgTilingData_->set_outputBufferLen(outputBufferLen);

    preSfmgTilingData_->set_normalAxisSize(normalAxisSize);
    preSfmgTilingData_->set_singleLoopNBurstNum(singleLoopNBurstNum);
    preSfmgTilingData_->set_normalCoreLoopTimes(normalCoreLoopTimes);
    preSfmgTilingData_->set_tailCoreLoopTimes(tailCoreLoopTimes);
    preSfmgTilingData_->set_normalCoreLastLoopNBurstNum(normalCoreLastLoopNBurstNum);
    preSfmgTilingData_->set_tailCoreLastLoopNBurstNum(tailCoreLastLoopNBurstNum);
    preSfmgTilingData_->set_normalCoreNBurstNums(normalCoreSize);
    preSfmgTilingData_->set_tailCoreNBurstNums(tailCoreSize);
    preSfmgTilingData_->set_layout(fBaseParams.layoutType);
    preSfmgTilingData_->set_value_d(fBaseParams.d1);
    preSfmgTilingData_->set_b(fBaseParams.b - fBaseParams.tailZeroCount);
    preSfmgTilingData_->set_n2(fBaseParams.n2);
    preSfmgTilingData_->set_g(fBaseParams.g);
    preSfmgTilingData_->set_s1(fBaseParams.s1);
    preSfmgTilingData_->set_d(fBaseParams.d);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::DetermineMode()
{
    // 当前fp16都走高精度
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        fBaseParams.inputDtype = DtypeEnum::FLOAT32;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        fBaseParams.inputDtype = DtypeEnum::BFLOAT16;
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E5M2) {
        fBaseParams.inputDtype = (optiling::DtypeEnum)4; // 4 is DtypeEnum::FLOAT8_E5M2
    } else if (fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN) {
        fBaseParams.inputDtype = (optiling::DtypeEnum)5; // 5 is DtypeEnum::FLOAT8_E4M3
    } else {
        fBaseParams.inputDtype = DtypeEnum::FLOAT16_PRECISION;
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceSize = 0;
    workspaceSize = RESERVED_WORKSPACE_SIZE;
    int64_t qSize =
        ((fBaseParams.b * fBaseParams.n2 * fBaseParams.g - 1) * fBaseParams.s1 + AlignTo(fBaseParams.s1, ALIGN256)) *
        fBaseParams.d;
    int64_t kSize =
        ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) * fBaseParams.d;
    int64_t vSize =
        ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) * fBaseParams.d1;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        qSize = (AlignTo(fBaseParams.t1 * fBaseParams.n1, ALIGN256)) * fBaseParams.d;
        kSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) * fBaseParams.d;
        vSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) * fBaseParams.d1;
    }

    if (fBaseParams.isBn2) {
        if (fBaseParams.isBn2MultiBlk) {
            postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
            // matmal3 dq
            workspaceSize += CORE_LIST_NUM * AlignTo(fBaseParams.s1, ALIGN256) * fBaseParams.sfmgdInner * FP32_BYTES;
            postTilingData_->set_dkWorkSpaceOffset(workspaceSize);
            // matmal4 dk
            workspaceSize += CORE_LIST_NUM * AlignTo(fBaseParams.s2, ALIGN128) * fBaseParams.sfmgdInner * FP32_BYTES;
            // matmal5 dv
            postTilingData_->set_dvWorkSpaceOffset(workspaceSize);
            workspaceSize += CORE_LIST_NUM * AlignTo(fBaseParams.s2, ALIGN128) * fBaseParams.sfmgdInner * FP32_BYTES;
        } else {
            workspaceSize +=
                (fBaseParams.blockOuter * fBaseParams.s1Inner * fBaseParams.sfmgdInner * FP32_BYTES * 2 + GM_ALIGN) /
                GM_ALIGN * GM_ALIGN; // DQ
            workspaceSize += (fBaseParams.blockOuter * fBaseParams.s2Inner * fBaseParams.sfmgdInner * FP32_BYTES *
                                  2 // 2 is double size of Dinner
                              + GM_ALIGN) /
                             GM_ALIGN * GM_ALIGN; // DK
        }
    } else {
        if (fBaseParams.queryType != ge::DT_FLOAT) {
            postTilingData_->set_dqWorkSpaceOffset(workspaceSize);
            int64_t qSize = ((fBaseParams.b * fBaseParams.n2 * fBaseParams.g - 1) * fBaseParams.s1 +
                             AlignTo(fBaseParams.s1, ALIGN256)) *
                            fBaseParams.d;
            if (fBaseParams.tailZeroCount > 0) {
                qSize = (AlignTo(fBaseParams.t1 * fBaseParams.n1, ALIGN256)) * fBaseParams.d;
            }
            int64_t kSize =
                ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) *
                fBaseParams.d;
            int64_t vSize =
                ((fBaseParams.b * fBaseParams.n2 - 1) * fBaseParams.s2 + AlignTo(fBaseParams.s2, ALIGN128)) *
                fBaseParams.d1;
            if (fBaseParams.tailZeroCount > 0) {
                kSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) * fBaseParams.d;
                vSize = (AlignTo(fBaseParams.t2 * fBaseParams.n2, ALIGN128)) * fBaseParams.d1;
            }

            // matmal3 q
            workspaceSize = (workspaceSize + static_cast<size_t>(qSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
            postTilingData_->set_dkWorkSpaceOffset(workspaceSize);
            // matmal3 k
            workspaceSize = (workspaceSize + static_cast<size_t>(kSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
            postTilingData_->set_dvWorkSpaceOffset(workspaceSize);
            // matmal3 v
            workspaceSize = (workspaceSize + static_cast<size_t>(vSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        }
    }

    // mask bool workspace size
    if (fBaseParams.dropoutIsDivisibleBy8 == 0) {
        postTilingData_->set_dropMaskGmOffset(workspaceSize);
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.dropMaskSize) + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    sfmgSize =
        ((fBaseParams.b * fBaseParams.n2 * fBaseParams.g - 1) * fBaseParams.s1 + AlignTo(fBaseParams.s1, ALIGN128)) *
        BIT_NUMS;
    postTilingData_->set_sfmgWorkSpaceOffset(workspaceSize);
    workspaceSize = (workspaceSize + static_cast<size_t>(sfmgSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    if (fBaseParams.sinkOptional == NORMAL_TENSOR) {
        postTilingData_->set_dsinkWorkSpaceOffset(workspaceSize);
        OP_LOGI(context_, "FAG sinkOptional, sink baseoffset = %ld, sink workspaceSize = %ld.", workspaceSize,
                static_cast<size_t>(fBaseParams.sinkSize) * FP32_BYTES);
        // dsink sum data size
        workspaceSize =
            (workspaceSize + static_cast<size_t>(fBaseParams.sinkSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }

    if (fBaseParams.isDeterministic) {
        postTilingData_->set_deterGmOffset(workspaceSize);
        workspaceSize += (fBaseParams.s1Inner * S1CV_RATIO_DEFAULT + 2 * fBaseParams.s2Inner) * fBaseParams.sfmgdInner *
                         fBaseParams.aicNum * FP32_BYTES * 2; // 2 is double size of aiv
        postTilingData_->set_deterWorkSpaceOffset(workspaceSize);
        workspaceSize += fBaseParams.maxValidBBLen * fBaseParams.aicNum * INT64_BLOCK_NUM * NUM_THREE * INT64_BYTES;
    }

    workspaceSize += WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t FlashAttentionScoreGradTilingUs1s2Bs2David::GetTilingKey() const
{
    auto attenMaskCfg = fBaseParams.attenMaskOptional == EMPTY_TENSOR ? OptionEnum::DISABLE : OptionEnum::ENABLE;
    LayoutEnum inputLayout = LayoutEnum::BSND;
    if (fBaseParams.layoutType == INPUT_FORMAT_BN2GS2D) {        // 2
        inputLayout = LayoutEnum::BNSD;                          // 2
        inputLayout = LayoutEnum::BSND;                          // 0
    } else if (fBaseParams.layoutType == INPUT_FORMAT_S2BN2GD) { // 1
        inputLayout = LayoutEnum::SBND;                          // 1
        inputLayout = LayoutEnum::BSND;                          // 0
    } else if (fBaseParams.layoutType == INPUT_FORMAT_BS2N2GD) { // 0
        inputLayout = LayoutEnum::BSND;                          // 0
    } else if (fBaseParams.layoutType == INPUT_FORMAT_TND) {     // 3
        inputLayout = LayoutEnum::TND;                           // 3
    } else {
        OP_LOGE("GetTilingKey", "The layout type is not support!");
    }
    auto pseValue = fBaseParams.pseOptional == NORMAL_TENSOR ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto dropValue = fBaseParams.keepProb < 1 ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto isDavidPlatformValue = 1; // v121此位用2
    auto isTnd = (inputLayout == LayoutEnum::TND);
    auto reserved = 0;
    SplitAxisEnum splitAxis = fBaseParams.splitAxis;
    auto IsSwizzle = fBaseParams.isSwizzle || tndBaseInfo.isTndSwizzle;
    if (fBaseParams.isBn2) {
        splitAxis = SplitAxisEnum::BN2;
    }
    int dNoEqual = (fBaseParams.d1 != fBaseParams.d) || fBaseParams.hasRope;
    auto deterSparseType = (splitAxis == SplitAxisEnum::BN2) ? static_cast<uint8_t>(DeterSparseType::NO_DETER) :
                                                               static_cast<uint8_t>(fBaseParams.deterSparseType);
    bool isDeterNEqual = deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_OLD) &&
                         deterSparseType != static_cast<uint32_t>(DeterSparseType::NO_DETER) && fBaseParams.g == 1;

    OP_LOGI(context_,
            "splitAxis[%d], fBaseParams.inputDtype[%d], isTnd[%d], dropValue[%d], pseValue[%d], attenMaskCfg[%d], "
            "fBaseParams.s1TemplateType[%d], fBaseParams.s2TemplateType[%d], fBaseParams.dTemplateType[%d], "
            "fBaseParams.isDeterministic[%d], dNoEqual[%d], fBaseParams.outDtype[%d],isDavidPlatformValue[%d]",
            splitAxis, fBaseParams.inputDtype, isTnd, dropValue, pseValue, attenMaskCfg, fBaseParams.s1TemplateType,
            fBaseParams.s2TemplateType, (uint32_t)(fBaseParams.dTemplateType), fBaseParams.isDeterministic, dNoEqual,
            fBaseParams.outDtype, isDavidPlatformValue);
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        0, static_cast<uint8_t>(splitAxis), static_cast<uint8_t>(fBaseParams.inputDtype), static_cast<uint8_t>(isTnd),
        static_cast<uint8_t>(dropValue), static_cast<uint8_t>(pseValue), static_cast<uint8_t>(attenMaskCfg),
        static_cast<uint16_t>(fBaseParams.s1TemplateType), static_cast<uint16_t>(fBaseParams.s2TemplateType),
        static_cast<uint16_t>(fBaseParams.dTemplateType), static_cast<uint8_t>(deterSparseType),
        static_cast<uint8_t>(isDeterNEqual), static_cast<uint8_t>(fBaseParams.isBn2MultiBlk),
        static_cast<uint8_t>(dNoEqual), static_cast<uint8_t>(fBaseParams.outDtype), static_cast<uint8_t>(IsSwizzle),
        static_cast<uint8_t>(fBaseParams.hasRope), static_cast<uint8_t>(isDavidPlatformValue));
    OP_LOGI(context_, "FAGTiling S1s2Bn2gs1s2 DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::SetSparseParams()
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        return SetPrefixSparseParams();
    }

    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        OP_LOGD("SetSparseParams ", " in the TND scenario,isSparse is true by default");
        return true;
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
        fBaseParams.attenMaskOptional == EMPTY_TENSOR) {
        OP_LOGD("SetSparseParams ", " in the ALL_MASK or attenMask is none scenario,isSparse is false");
        return false;
    }

    // 兼容老版本，未配置sparseMode或配置sparseMode为0的处理
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
        if (int64_t(fBaseParams.s1) > fBaseParams.s1Token ||
            int64_t(fBaseParams.s2) > fBaseParams.s2Token) { // band场景，包含causal
            OP_LOGD("SetSparseParams ", " in the NONE_MASK  and token is band scenario,isSparse is true ");
            return true;
        } else {
            OP_LOGD("SetSparseParams ", " in the NONE_MASK  and token is not band scenario,isSparse is false");
            return false;
        }
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        OP_LOGD("SetSparseParams ", " in the LEFT_UP_CAUSAL  or RIGHT_DOWN_CAUSAL scenario,isSparse is true");
        return true;
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
        (int64_t(fBaseParams.s1) > fBaseParams.s1Token || int64_t(fBaseParams.s2) > fBaseParams.s2Token)) {
        OP_LOGD("SetSparseParams ", " in the BAND  and token is band scenario,isSparse is true ");
        return true;
    }

    OP_LOGD("SetSparseParams ", " no scenario is hit, isSparse is false ");
    return false;
}

// sink
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessSinkInfo()
{
    auto sinkShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINK_IDX));
    if (sinkShape == nullptr || sinkShape->GetStorageShape().GetDimNum() == 0) {
        OP_LOGD(context_, "ProcessSinkInfo, sinkShape is null : %d", sinkShape == nullptr);
        fBaseParams.sinkOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!(fBaseParams.queryType == ge::DT_FLOAT16 || fBaseParams.queryType == ge::DT_BF16),
                OP_LOGE(context_, "FAG sink, dtype only supports fp16 or bf16."), return ge::GRAPH_FAILED);
    fBaseParams.sinkOptional = NORMAL_TENSOR;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessPseNormal(const char* inputLayout)
{
    auto pseShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::PSE_SHIFT));
    auto pseShapeDim = pseShape->GetStorageShape().GetDimNum();
    if (pseShapeDim != PSE_NORMAL_SHAPE_DIM) {
        OP_LOGE(context_, "The shape of pse is not 4 dimensions, got %ld", pseShapeDim);
        return ge::GRAPH_PARAM_INVALID;
    }

    auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
    auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
    auto dim2 = pseShape->GetStorageShape().GetDim(DIM_2);
    auto dim3 = pseShape->GetStorageShape().GetDim(DIM_3);

    bool isBN1S = (dim0 == fBaseParams.b && dim1 == fBaseParams.n1 && dim2 == 1 && dim3 == fBaseParams.s2);
    bool isBNSS = (dim0 == fBaseParams.b && dim1 == fBaseParams.n1 && dim2 == fBaseParams.s1 && dim3 == fBaseParams.s2);
    bool is1NSS = (dim0 == 1 && dim1 == fBaseParams.n1 && dim2 == fBaseParams.s1 && dim3 == fBaseParams.s2);
    bool isAlibiPse = (dim1 == fBaseParams.n1 && dim2 == MAX_BASIC_BLOCK_SIZE && dim3 == fBaseParams.s2);
    bool isPse = (fBaseParams.s1 == fBaseParams.s2 && fBaseParams.s1 >= MAX_BASIC_BLOCK_SIZE &&
                  int64_t(fBaseParams.s1) <= fBaseParams.s1Token && fBaseParams.s2Token == 0);
    bool isTnd = strcmp(inputLayout, "TND") == 0;
    bool isTndPse = isTnd && int64_t(fBaseParams.s1) <= fBaseParams.s1Token && fBaseParams.s2Token == 0;
    bool isAlibi1NHS = isPse && isAlibiPse && (dim0 == 1);
    bool isAlibiBNHS = isPse && isAlibiPse && (dim0 == fBaseParams.b);
    bool isTndAlibiPse1NHS = isTndPse && isAlibiPse && (dim0 == 1);
    bool isTndAlibiPseBNHS = isTndPse && isAlibiPse && (dim0 == fBaseParams.b);

    if (isTndAlibiPse1NHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NHS);
    } else if (isTndAlibiPseBNHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNHS);
    } else if (isBN1S && !isTnd) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BN1S);
    } else if (isBNSS && !isTnd) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNSS);
    } else if (is1NSS && !isTnd) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NSS);
    } else if (isAlibi1NHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NHS);
    } else if (isAlibiBNHS) {
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNHS);
    } else {
        OP_LOGE(context_, "The shape of pse[%ld,%ld,%ld,%ld] is invalid or tocken[%ld,%ld] not casual", dim0, dim1,
                dim2, dim3, fBaseParams.s1Token, fBaseParams.s2Token);
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessPseInfo(const char* inputLayout)
{
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::PSETYPE)) {
        fBaseParams.pseType =
            *(context_->GetAttrs()->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndex::PSETYPE))); // 8
        if (fBaseParams.pseType >= static_cast<uint32_t>(PseType::PSE_INVALID_TYPE)) {
            OP_LOGE(context_, "FAG pseType %d is invalid", fBaseParams.pseType);
            return ge::GRAPH_FAILED;
        }
    }

    auto pseShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::PSE_SHIFT));
    if (pseShape == nullptr || pseShape->GetStorageShape().GetDimNum() == 0) {
        fBaseParams.pseOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }

    fBaseParams.pseOptional = NORMAL_TENSOR;
    auto pse = context_->GetOptionalInputDesc(static_cast<size_t>(InputIndex::PSE_SHIFT));
    if (fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_OUTER_MUL_ADD_TYPE) ||
        fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_OUTER_ADD_MUL_TYPE)) {
        OP_CHECK_IF(pse->GetDataType() != context_->GetInputDesc(static_cast<size_t>(InputIndex::QUERY))->GetDataType(),
                    OP_LOGE(context_, "FAG invalid pse dtype[%s], should be same with query's dtype",
                            ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(pse->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context_, "FAG invalid pse dtype[%s], should be ge::DT_FLOAT",
                            ge::TypeUtils::DataTypeToSerialString(pse->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto pseShapeDim = pseShape->GetStorageShape().GetDimNum();
    bool isTnd = strcmp(inputLayout, "TND") == 0;
    if (fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
        fBaseParams.pseType == static_cast<uint32_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
        auto ret = ProcessInnerPseInfo(pseShapeDim);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    } else if (pseShapeDim == PSE_DIM_NUM_1 && isTnd) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        bool isTndPseBN1S = isTnd && (dim0 == fBaseParams.t2 * fBaseParams.n1);
        bool isTndPseBNSS = isTnd && (dim0 == fBaseParams.sumS1S2Product * fBaseParams.n1);
        if (isTndPseBN1S) {
            fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BN1S);
        } else if (isTndPseBNSS) {
            fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNSS);
        } else {
            OP_LOGE(context_, "pse outer mode, tnd, unsupported pse shape");
            return ge::GRAPH_FAILED;
        }
    } else {
        auto ret = ProcessPseNormal(inputLayout);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    SetPseLayout();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessSparseModeInfo()
{
    // 新增SPARSE_MODE属性，上库兼容处理
    auto attrs = context_->GetAttrs();
    fBaseParams.sparseMode = static_cast<uint32_t>(SparseMode::NO_MASK);
    if (attrs->GetAttrNum() > static_cast<size_t>(AttrIndex::SPARSE_MODE)) {
        fBaseParams.sparseMode = *(attrs->GetAttrPointer<int>(static_cast<size_t>(AttrIndex::SPARSE_MODE))); // 7
        OP_LOGI("ProcessSparseModeInfo", "fBaseParams.sparseMode = %d.", fBaseParams.sparseMode);
    }

    if (SupportTrans2BS2N2GD()) {
        fBaseParams.layoutType = INPUT_FORMAT_BS2N2GD;
        OP_LOGD("inputLayout = TND, but all s1 s2 same, inputLayout set BSND");
    }
    if (!(fBaseParams.layoutType == INPUT_FORMAT_TND ? CheckVarLenSparseModeValue() : CheckSparseModeValue())) {
        return ge::GRAPH_FAILED;
    }
    fBaseParams.attenMaskCompressMode = 0;
    auto attenMaskShape = context_->GetOptionalInputShape(INPUT_IDX_ATTEN_MASK);
    if (attenMaskShape == nullptr || attenMaskShape->GetStorageShape().GetDimNum() == 0) {
        fBaseParams.attenMaskOptional = EMPTY_TENSOR;
        return ge::GRAPH_SUCCESS;
    }
    fBaseParams.attenMaskOptional = NORMAL_TENSOR;
    auto storageShape = attenMaskShape->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    auto ret = SetAttenMaskShapeType(attenMaskShape, dimNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    fBaseParams.attenMaskS2Size = storageShape.GetDim(dimNum - LAST_AXIS_IDX);
    fBaseParams.attenMaskS1Size = storageShape.GetDim(dimNum - SEC_LAST_AXIS_IDX);

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::BAND_EQUAL_S_MODE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::PREFIX_COMPRESS_MODE);
        fBaseParams.attenMaskS2Size = ATTEN_MASK_COMPRESS_LIMIT;
        fBaseParams.attenMaskS1Size = PREFIX_COMPRESS_S1_SIZE;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::RIGHT_DOWN_CASUAL_BAND_MODE);
        fBaseParams.attenMaskS2Size = ATTEN_MASK_COMPRESS_LIMIT;
        fBaseParams.attenMaskS1Size = ATTEN_MASK_COMPRESS_LIMIT;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        fBaseParams.attenMaskCompressMode = static_cast<uint32_t>(AttenMaskCompressMode::BAND_LEFT_UP_CASUAL_MODE);
        fBaseParams.attenMaskS2Size = ATTEN_MASK_COMPRESS_LIMIT;
        fBaseParams.attenMaskS1Size = ATTEN_MASK_COMPRESS_LIMIT;
    }

    auto attenMask = context_->GetOptionalInputDesc(INPUT_IDX_ATTEN_MASK);
    if (attenMask != nullptr) {
        if (attenMask->GetDataType() == fBaseParams.queryType) {
            fBaseParams.attenMaskDtype = static_cast<uint32_t>(AttenDataType::ATTEN_MASK_TYPE_SAME);
        } else {
            fBaseParams.attenMaskDtype = static_cast<uint32_t>(AttenDataType::ATTEN_MASK_TYPE_U8_BOOL);
        }
    }

    fBaseParams.bandIdx = FindBandIdx();
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::PrintShapeInfo()
{
    OP_LOGI(context_, "FAG s1s2_bn2gs1s2 with shape b[%u] n2[%u] g[%u] s1[%u] s2[%u] d[%u] preToken[%d] nextToken[%d]!",
            fBaseParams.b, fBaseParams.n2, fBaseParams.g, fBaseParams.s1, fBaseParams.s2, fBaseParams.d,
            fBaseParams.s1Token, fBaseParams.s2Token);
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CheckSparseModeValue()
{
    if (fBaseParams.sparseMode > static_cast<int64_t>(static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS))) {
        OP_LOGE("CheckSparseModeValue", "Not support sparse mode %d.", fBaseParams.sparseMode);
        return false;
    }
    return true;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CheckVarLenSparseModeValue()
{
    if (fBaseParams.sparseMode == static_cast<int64_t>(static_cast<uint32_t>(SparseMode::PREFIX)) ||
        fBaseParams.sparseMode > static_cast<int64_t>(static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL))) {
        OP_LOGE("CheckVarLenSparseModeValue", "Var len not support sparse mode %d.", fBaseParams.sparseMode);
        return false;
    }
    return true;
}

std::tuple<uint32_t, uint32_t, uint32_t> FlashAttentionScoreGradTilingUs1s2Bs2David::FuzzyForBestSplit()
{
    auto s1s2TemplateSize = GetS1S2TemplateType();
    uint32_t s1Inner = s1s2TemplateSize.first;
    uint32_t s2Inner = s1s2TemplateSize.second;
    uint32_t dInner = GetDTemplateType();
    return std::tie(s1Inner, s2Inner, dInner);
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner)
{
    return true;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::PostTiling()
{
    SaveToTilingData();
    auto blockdim = 0;
    if (fBaseParams.isDeterministic) {
        blockdim = fBaseParams.aicNum;
    } else {
        blockdim =
            CalcTschBlockDim(s1s2BNGS1S2SplitCoreParams_->get_blockOuter(), fBaseParams.aicNum, fBaseParams.coreNum);
    }
    OP_CHECK_IF(blockdim == 0,
                OP_LOGE("FlashAttentionScoreGradTilingUs1s2Bs2David", "blockdim is 0, aicNum is %ld, aivNum is %ld.",
                        fBaseParams.aicNum, fBaseParams.coreNum),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);

    // 使用SyncAll，需要设置为batch mode模式，所有核同时启动，否则在多流方式下执行可能会卡死
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2 || fBaseParams.layoutType != INPUT_FORMAT_TND) {
        context_->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH])
{
    std::vector<bool> invalidS1Array(fBaseParams.s1Outer, false);
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        int64_t leftIntersectionPoint = std::max(0L, int64_t(fBaseParams.cvS2Inner * i) - fBaseParams.s2Token);
        if (leftIntersectionPoint >= int64_t(fBaseParams.s1)) {
            parseInfo[i][BEGIN_IDX] = (fBaseParams.s1 + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        } else {
            parseInfo[i][BEGIN_IDX] = leftIntersectionPoint / fBaseParams.s1CvInner;
        }
        int64_t cvBlockTail = i == fBaseParams.s2Outer - 1 ? fBaseParams.s2CvTail : fBaseParams.cvS2Inner;
        parseInfo[i][END_IDX] =
            int64_t(std::min(std::max(0L, int64_t(fBaseParams.cvS2Inner * i + cvBlockTail) + fBaseParams.s1Token),
                             int64_t(fBaseParams.s1)) +
                    fBaseParams.s1CvInner - 1) /
            fBaseParams.s1CvInner;
        int64_t tmpSize =
            (parseInfo[i][END_IDX] > parseInfo[i][BEGIN_IDX]) ? parseInfo[i][END_IDX] - parseInfo[i][BEGIN_IDX] : 0;
        if (i == 0) {
            parseInfo[i][LENGTH_IDX] = tmpSize;
        } else {
            parseInfo[i][LENGTH_IDX] = parseInfo[i - 1][LENGTH_IDX] + tmpSize;
        }
        if (parseInfo[i][BEGIN_IDX] >= parseInfo[i][END_IDX]) {
            fBaseParams.isInvalidCol = true;
        }
        // check invalid row or col block for BN2
        for (int64_t j = 0; j < static_cast<int64_t>(invalidS1Array.size()); j++) {
            if (j >= parseInfo[i][BEGIN_IDX] && j < parseInfo[i][END_IDX]) {
                invalidS1Array[j] = true;
            }
        }
        OP_LOGD("Sparse", " idx = %d: Begin = %d, End = %d, Length = %d, total_Length = %d", i, parseInfo[i][0],
                parseInfo[i][1], tmpSize, parseInfo[i][LENGTH_IDX]);
    }
    for (size_t j = 0; j < invalidS1Array.size(); j++) {
        if (!invalidS1Array[j]) {
            fBaseParams.isInvalidRow = true;
            break;
        }
    }
}

// 以下场景对外部输入token屏蔽，重新设置token值并做校验
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessTokensInfo()
{
    OP_LOGD("ProcessTokensInfo", " Before correction ,the value of s1Token = %ld and the value of s2Token %ld.",
            fBaseParams.s1Token, fBaseParams.s2Token);

    // 自动校正left和right causal的token值，token信息仅用于sparse分核计算
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = 0;
    }

    // 对pad场景做校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL))
    // 时，token以右下角为基准，需要校正
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
         fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND))) {
        fBaseParams.s1Token = fBaseParams.s1Token + fBaseParams.s1 - fBaseParams.s2;
        fBaseParams.s2Token = fBaseParams.s2Token - fBaseParams.s1 + fBaseParams.s2;
    }

    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
        fBaseParams.attenMaskOptional == EMPTY_TENSOR) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = INT32_MAX;
    }

    OP_LOGD("ProcessTokensInfo", " the corrected s1Token = %ld, s2Token %ld.", fBaseParams.s1Token,
            fBaseParams.s2Token);

    // 1  2  3  5  6  不校验
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) {
        return ge::GRAPH_SUCCESS;
    }

    // 校验pad场景token是否合法
    if (fBaseParams.layoutType != INPUT_FORMAT_TND &&
        (-fBaseParams.s1Token > int64_t(fBaseParams.s2) || -fBaseParams.s2Token > int64_t(fBaseParams.s1) ||
         (fBaseParams.s1Token + fBaseParams.s2Token) < 0)) {
        OP_LOGE(
            "ProcessTokensInfo",
            "pre_token and next_token is invalid in the pad scene, got s1 %ld, s2 %ld,  pre_token %ld, next_token %ld",
            fBaseParams.s1, fBaseParams.s2, fBaseParams.s1Token, fBaseParams.s2Token);
        return ge::GRAPH_FAILED;
    }

    // 校验unpad场景token是否合法   0  4  7  8
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        auto ret = CheckUnpadTokensInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetSparseBlockInfo()
{
    // [s2OuterIdx][begin, end, length]
    int64_t(*parseInfo)[ARRAY_LENGTH] = new int64_t[fBaseParams.s2Outer][ARRAY_LENGTH];
    GetParseS1S2OuterInfo(parseInfo);
    int64_t s1s2oCount = parseInfo[fBaseParams.s2Outer - 1][LENGTH_IDX];

    // block split
    int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
    int64_t blockTailTmp = fusedOuter % blockFactor;
    int64_t blockTail = blockTailTmp == 0 ? blockFactor : blockTailTmp;
    OP_LOGD("Sparse", "Sparse parseInfo fusedOuter = %d: blockFactor = %d, blockTail = %d", fusedOuter, blockFactor,
            blockTail);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = fBaseParams.blockFactor;

    int64_t bIdx = 0;
    int64_t bTail = 0;
    int64_t n2Idx = 0;
    int64_t n2Tail = 0;
    int64_t gIdx = 0;
    int64_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s2oIdx = 0;

    int64_t n2gs1s2o = fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    int64_t gs1s2o = fBaseParams.g * s1s2oCount;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] =
        fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
    for (int64_t c = 1; c < blockOuter; c++) {
        // cal indx for total bngs1os2o(sparse)
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        bIdx = currentIdx / n2gs1s2o;
        bTail = currentIdx % n2gs1s2o;
        n2Idx = bTail / gs1s2o;
        n2Tail = bTail % gs1s2o;
        gIdx = n2Tail / s1s2oCount;
        gTail = n2Tail % s1s2oCount;

        OP_LOGD(
            "Sparse",
            "Sparse parseInfo currentIdx = %d: bIdx = %d, bTail = %d, n2Idx = %d, n2Tail = %d, gIdx = %d, gTail = %d",
            currentIdx, bIdx, bTail, n2Idx, n2Tail, gIdx, gTail);
        GetCommonS1S2OuterIndex(parseInfo, gTail, s1oIdx, s2oIdx);

        // total indx in bngs1os2o (range is [))
        blockStarts[c] = (((bIdx * fBaseParams.n2 + n2Idx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2oIdx) *
                             fBaseParams.s1Outer +
                         s1oIdx + 1;
        blockEnds[c - 1] = blockStarts[c];
        OP_LOGD("Sparse", "blockStarts[c] = %d:", blockStarts[c]);
    }
    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    // free tensor
    delete[] parseInfo;
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::GetCommS1S2OuterInfo(
    const int64_t prefixN, std::vector<std::pair<int64_t, int64_t>>& s1ValidIdx)
{
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        int64_t s1Start = 0;
        int64_t cvS2Idx = i * fBaseParams.cvS2Inner;
        if (cvS2Idx >= prefixN) {
            int64_t deltaS1S2 = static_cast<int64_t>(fBaseParams.s1) - static_cast<int64_t>(fBaseParams.s2);
            s1Start = std::min(static_cast<int64_t>(cvS2Idx) + deltaS1S2, static_cast<int64_t>(fBaseParams.s1));
        }

        s1ValidIdx[i].first = (static_cast<int64_t>(AlignData(fBaseParams.s1, fBaseParams.s1CvInner)) - s1Start +
                               static_cast<int64_t>(fBaseParams.s1CvInner) - 1) /
                              static_cast<int64_t>(fBaseParams.s1CvInner);
        if (i == 0) {
            s1ValidIdx[i].second = s1ValidIdx[i].first;
        } else {
            s1ValidIdx[i].second = s1ValidIdx[i - 1].second + s1ValidIdx[i].first;
        }
    }
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::CheckPrefixNExist(
    const int64_t bIdx, const int64_t prefixN, std::vector<std::vector<std::pair<int64_t, int64_t>>>& s1ValidIdx)
{
    for (int64_t i = 0; i < bIdx; ++i) {
        if (fBaseParams.prefixN[i] == prefixN) {
            OP_LOGD("Sparse", "prefixN of bIdx[%u] and bIdx[%u] is same as %u", i, bIdx, prefixN);
            s1ValidIdx[bIdx].assign(s1ValidIdx[i].begin(), s1ValidIdx[i].end());
            return true;
        }
    }
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetSparsePrefixBlockInfo()
{
    std::vector<std::vector<std::pair<int64_t, int64_t>>> s1ValidIdx(
        fBaseParams.b, std::vector<std::pair<int64_t, int64_t>>(fBaseParams.s2Outer, {0, 0}));
    uint64_t totalValidBaseBlock = 0; // include nRation, baseN * nRation
    int32_t comBIdx = -1;
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; ++bIdx) {
        int64_t prefixN = fBaseParams.prefixN[bIdx];
        if (CheckPrefixNExist(bIdx, prefixN, s1ValidIdx)) {
            totalValidBaseBlock += s1ValidIdx[bIdx][fBaseParams.s2Outer - 1].second;
            continue;
        }

        if (fBaseParams.s1 <= fBaseParams.s2 - prefixN) {
            if (comBIdx != -1) {
                s1ValidIdx[bIdx].assign(s1ValidIdx[comBIdx].begin(), s1ValidIdx[comBIdx].end());
                totalValidBaseBlock += s1ValidIdx[bIdx][fBaseParams.s2Outer - 1].second;
                continue;
            }
            comBIdx = bIdx;
        }

        GetCommS1S2OuterInfo(prefixN, s1ValidIdx[bIdx]);
        totalValidBaseBlock += s1ValidIdx[bIdx][fBaseParams.s2Outer - 1].second;
    }

    totalValidBaseBlock *= fBaseParams.n2 * fBaseParams.g;
    int64_t blockFactor =
        (totalValidBaseBlock + fBaseParams.aicNum - 1) / fBaseParams.aicNum;    // 每个核处理的最多数据个数
    int64_t blockOuter = (totalValidBaseBlock + blockFactor - 1) / blockFactor; // 实际使用的核数

    OP_LOGD("Sparse", "Sparse parseInfo totalValidBaseBlock = %lu: blockFactor = %u, blockOuter = %u",
            totalValidBaseBlock, blockFactor, blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = blockFactor;
    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] =
        fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;

    uint32_t coreNum = 0;
    int64_t tmepBlock = 0;
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; ++bIdx) {
        for (int64_t nIdx = 0; nIdx < fBaseParams.n2; ++nIdx) {
            SetSparsePrefixBlockInterval(bIdx, nIdx, s1ValidIdx, blockStarts, blockEnds, coreNum, tmepBlock);
        }
    }

    for (uint32_t coreIdx = blockOuter; coreIdx < CORE_LIST_NUM; ++coreIdx) {
        blockStarts[coreIdx] = 0;
        blockEnds[coreIdx] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

int64_t FlashAttentionScoreGradTilingUs1s2Bs2David::FindBandIdx()
{
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND)) {
        for (int64_t i = fBaseParams.b - 1; i >= 0; i--) {
            if (fBaseParams.actualSeqQlen[i] != 0) {
                return i;
            }
        }
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        for (int64_t i = 0; i < fBaseParams.b; i++) {
            if (fBaseParams.actualSeqQlen[i] != 0) {
                return i;
            }
        }
    }
    return 0;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::FillBlockInfo(
    std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo,
    std::vector<std::vector<int64_t>>& totalBlockInfo)
{
    OP_LOGD("FillBlockInfo", " Starting load balancing calculation in TND scenario");
    OP_LOGD("FillBlockInfo", "SparseMode %u, find band index %u", fBaseParams.sparseMode, fBaseParams.bandIdx);

    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];

        auto actualS1Outer = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        CalValidUnpadBlockInfo(i, calculatedBlockInfo);

        if (i == 0) {
            calculatedBlockInfo[0][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[0][fBaseParams.s2Outer - 1][SUM_S1S2];
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[i][fBaseParams.s2Outer - 1][SUM_S1S2] +
                calculatedBlockInfo[i - 1][0][SUM_ALL];
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
        OP_LOGD("FillBlockInfo", "Up to b idx = %d , a total of %d blocks that need to be calculated", i,
                calculatedBlockInfo[i][0][SUM_ALL]);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::BuildTndBn2gs1s2SwizzleParam(
    const std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo)
{
    if (fBaseParams.b >= TND_SWIZZLE_PREFIX_NUM) {
        return;
    }
    const uint32_t activeCoreNum = static_cast<uint32_t>(fBaseParams.aicNum);
    std::vector<uint64_t> columnWeights;
    uint64_t totalTileNum = 0;

    tndBaseInfo.tndValidBlockPrefixSum[0] = 0;
    tndBaseInfo.tndSwizzleS1S2PrefixSum[0] = 0;
    tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[0] = 0;
    for (int64_t bIdx = 0; bIdx < fBaseParams.b; ++bIdx) {
        const int64_t actualS2Outer = CeilDivideBy(fBaseParams.actualSeqKvlen[bIdx], fBaseParams.cvS2Inner);
        uint64_t validS2Count = 0;
        std::vector<uint64_t> perN2ColumnWeights;
        perN2ColumnWeights.reserve(static_cast<size_t>(actualS2Outer));

        for (int64_t s2Idx = 0; s2Idx < actualS2Outer; ++s2Idx) {
            const int64_t begin = calculatedBlockInfo[bIdx][s2Idx][BEGIN_IDX];
            const int64_t end = calculatedBlockInfo[bIdx][s2Idx][END_IDX];
            const int64_t validS1 = end > begin ? end - begin : 0;
            if (validS1 == 0) {
                continue;
            }
            ++validS2Count;
            perN2ColumnWeights.push_back(static_cast<uint64_t>(fBaseParams.g) * validS1);
        }

        for (int64_t n2Idx = 0; n2Idx < fBaseParams.n2; ++n2Idx) {
            for (uint64_t weight : perN2ColumnWeights) {
                columnWeights.push_back(weight);
                totalTileNum += weight;
            }
        }

        tndBaseInfo.tndValidBlockPrefixSum[bIdx + 1] =
            tndBaseInfo.tndValidBlockPrefixSum[bIdx] + static_cast<uint64_t>(fBaseParams.n2) * validS2Count;
        tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx + 1] =
            tndBaseInfo.tndSwizzleS1S2PrefixSum[bIdx] +
            fBaseParams.actualSeqQlen[bIdx] * fBaseParams.actualSeqKvlen[bIdx];
        tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[bIdx + 1] =
            tndBaseInfo.tndSwizzleS1S2AlignPrefixSum[bIdx] +
            fBaseParams.actualSeqQlen[bIdx] *
                AlignTo(fBaseParams.actualSeqKvlen[bIdx], static_cast<int64_t>(ConstAxisTemplateNum::NUM16));
    }

    const uint64_t taskNum = static_cast<uint64_t>(columnWeights.size());
    std::vector<uint64_t> simpleCoreWork(activeCoreNum, 0);
    for (uint64_t taskRank = 0; taskRank < taskNum; ++taskRank) {
        simpleCoreWork[taskRank % activeCoreNum] += columnWeights[taskRank];
    }
    const uint64_t simpleMaxCoreTileNum = *std::max_element(simpleCoreWork.begin(), simpleCoreWork.end());

    const bool isCausal = fBaseParams.sparseType == static_cast<uint8_t>(SparseType::CASUAL);
    uint64_t serpentineMaxCoreTileNum = simpleMaxCoreTileNum;
    if (isCausal) {
        std::vector<uint64_t> serpentineCoreWork(activeCoreNum, 0);
        uint64_t waveIdx = 0;
        for (uint64_t base = 0; base < taskNum; base += activeCoreNum, ++waveIdx) {
            const uint64_t waveWidth = std::min<uint64_t>(activeCoreNum, taskNum - base);
            for (uint64_t coreIdx = 0; coreIdx < waveWidth; ++coreIdx) {
                const uint64_t taskRank = (waveIdx & 1U) == 0 ? base + coreIdx : base + waveWidth - 1 - coreIdx;
                serpentineCoreWork[coreIdx] += columnWeights[taskRank];
            }
        }
        serpentineMaxCoreTileNum = *std::max_element(serpentineCoreWork.begin(), serpentineCoreWork.end());
    }

    const bool serpentineStrictlyBetter = isCausal && serpentineMaxCoreTileNum < simpleMaxCoreTileNum;
    const uint64_t reducedMaxCoreTileNum =
        serpentineStrictlyBetter ? simpleMaxCoreTileNum - serpentineMaxCoreTileNum : 0;
    const long double scaledBenefit =
        static_cast<long double>(reducedMaxCoreTileNum) * TND_VALID_S2_COLUMN_SERPENTINE_BENEFIT_DENOMINATOR;
    const long double requiredBenefit =
        static_cast<long double>(simpleMaxCoreTileNum) * TND_VALID_S2_COLUMN_SERPENTINE_BENEFIT_NUMERATOR;
    const bool useSerpentine =
        serpentineStrictlyBetter && scaledBenefit >= requiredBenefit && taskNum >= 2 * activeCoreNum;
    const uint64_t chosenMaxCoreTileNum = useSerpentine ? serpentineMaxCoreTileNum : simpleMaxCoreTileNum;
    const uint32_t chosenMode = useSerpentine ? TND_VALID_S2_COLUMN_SERPENTINE : TND_VALID_S2_COLUMN_SIMPLE_CYCLIC;

    const long double scaledMax =
        static_cast<long double>(chosenMaxCoreTileNum) * activeCoreNum * TND_VALID_S2_COLUMN_IMBALANCE_DENOMINATOR;
    const long double scaledTotal = static_cast<long double>(totalTileNum) * TND_VALID_S2_COLUMN_IMBALANCE_NUMERATOR;
    const bool workloadBalanced = totalTileNum > 0 && scaledMax <= scaledTotal;

    tndBaseInfo.totalValidS2TaskNum = taskNum;
    tndBaseInfo.totalValidS2TileNum = totalTileNum;
    tndBaseInfo.validS2MaxCoreTileNum = chosenMaxCoreTileNum;
    tndBaseInfo.validS2ActiveCoreNum = activeCoreNum;
    tndBaseInfo.columnScheduleMode = chosenMode;
    tndBaseInfo.validS2ColumnEligible = taskNum >= activeCoreNum && workloadBalanced;
    if (isCausal) {
        if (!useSerpentine) {
            tndBaseInfo.validS2ColumnEligible = false;
        }
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::GetSparseUnpadBlockInfo()
{
    std::vector<std::vector<std::vector<int64_t>>> calculatedBlockInfo(
        fBaseParams.b,
        std::vector<std::vector<int64_t>>(fBaseParams.s2Outer, std::vector<int64_t>(CALCULATED_BLOCK_DIMENSION)));
    std::vector<std::vector<int64_t>> totalBlockInfo(fBaseParams.b, std::vector<int64_t>(TOTAL_BLOCK_DIMENSION));
    FillBlockInfo(calculatedBlockInfo, totalBlockInfo);
    BuildTndBn2gs1s2SwizzleParam(calculatedBlockInfo);

    // block split
    int64_t fusedOuter = calculatedBlockInfo[fBaseParams.b - 1][0][SUM_ALL];
    int64_t blockFactor = (fusedOuter + fBaseParams.aicNum - 1) / fBaseParams.aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    OP_LOGD("GetSparseUnpadBlockInfo", " fusedOuter = %u: blockFactor = %u, blockOuter = %u", fusedOuter, blockFactor,
            blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;
    fBaseParams.maxValidBBLen = blockFactor;

    int64_t bIdx = 0;
    int64_t bTail = 0;
    int64_t n2Idx = 0;
    int64_t n2Tail = 0;
    int64_t gIdx = 0;
    int64_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s1oTail = 0;
    int64_t s2oIdx = 0;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] = totalBlockInfo[fBaseParams.b - 1][1];

    int64_t s1OuterTmp = 0;

    OP_LOGD("GetSparseUnpadBlockInfo", "Load balancing calculation results in TND scenario:");
    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        for (int64_t b = 0; b < fBaseParams.b; b++) {
            if (calculatedBlockInfo[b][0][SUM_ALL] > currentIdx) {
                bIdx = b;
                auto s1os2o = calculatedBlockInfo[b][fBaseParams.s2Outer - 1][SUM_S1S2];
                auto gs1os2o = s1os2o * fBaseParams.g;
                bTail = (b == 0) ? currentIdx : currentIdx - calculatedBlockInfo[b - 1][0][SUM_ALL];
                n2Idx = bTail / gs1os2o;
                n2Tail = bTail % gs1os2o;
                gIdx = n2Tail / s1os2o;
                gTail = n2Tail % s1os2o;

                GetUnpadS1S2OuterIndex(s1oIdx, s2oIdx, gTail, b, calculatedBlockInfo);
                s1OuterTmp = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
                break;
            }
        }
        if (bIdx == 0) {
            blockStarts[c] = (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
        } else {
            blockStarts[c] = totalBlockInfo[bIdx - 1][1] + (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] +
                             s2oIdx * s1OuterTmp + s1oIdx;
        }

        blockEnds[c - 1] = blockStarts[c];
    }

    for (int64_t c = 0; c < blockOuter; c++) {
        OP_LOGD("GetSparseUnpadBlockInfo", "blockNum[%d], blockStarts = %d , blockEnds = %d ", c, blockStarts[c],
                blockEnds[c]);
    }

    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::CheckAttenMaskShape()
{
    // check atten_mask shape when enable atten_mask_compress
    if (fBaseParams.attenMaskCompressMode == 0) {
        bool invalid =
            fBaseParams.attenMaskOptional != EMPTY_TENSOR && fBaseParams.layoutType != INPUT_FORMAT_TND &&
            (static_cast<int64_t>(fBaseParams.attenMaskS1Size) * static_cast<int64_t>(fBaseParams.attenMaskS2Size) <
             static_cast<int64_t>(fBaseParams.s1) * static_cast<int64_t>(fBaseParams.s2));
        OP_LOGI("CheckAttenMaskShape",
                "fBaseParams.attenMaskOptional = %d, fBaseParams.layoutType = %d."
                "attenMaskS1Size * attenMaskS2Size = %d, s1 * s2 = %d.",
                fBaseParams.attenMaskOptional, fBaseParams.layoutType,
                (int64_t)fBaseParams.attenMaskS1Size * (int64_t)fBaseParams.attenMaskS2Size,
                (int64_t)fBaseParams.s1 * (int64_t)fBaseParams.s2);
        if (invalid) {
            OP_LOGE("CheckAttenMaskShape", "atten mask shape [%u,%u] is invalid.", fBaseParams.attenMaskS1Size,
                    fBaseParams.attenMaskS2Size);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskCompressMode == static_cast<uint32_t>(AttenMaskCompressMode::PREFIX_COMPRESS_MODE)) {
        if (fBaseParams.attenMaskS1Size != PREFIX_COMPRESS_S1_SIZE ||
            fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
            OP_LOGE("Atten Mask Compress",
                    "atten mask shape for prefix compress mode is invalid, try setting it to [3072, 2048].");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fBaseParams.attenMaskS1Size != fBaseParams.attenMaskS2Size) {
        OP_LOGE("Atten Mask Compress", "atten mask shape is not square.");
        return ge::GRAPH_FAILED;
    }

    if (fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
        OP_LOGE("Atten Mask Compress", "atten mask shape is invalid, try setting it to [2048, 2048].");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::CheckSoftmaxMaxSumTndShape()
{
    if (context_->GetAttrs()->GetAttrNum() > static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN)) {
        // read 13th attr softmax_out_layout
        const char* softmaxInLayout =
            context_->GetAttrs()->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::TND_SOFTMAX_IN));
        if (strcmp(softmaxInLayout, "same_as_input") != 0) {
            // check whether softmax_out_layout is TND
            return ge::GRAPH_SUCCESS;
        }
    } else {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    auto softmaxMaxShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape != nullptr) {
        auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
        if (softmaxMaxShapeDim != DIM_3) { // softmax TND only support 3 dimensions
            OP_LOGE(context_, "The shape of softmaxMax is invalid, got %lu dimensions", softmaxMaxShapeDim);
            return ge::GRAPH_FAILED;
        }
        auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(DIM_0); // 0:t1
        auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(DIM_1); // 1:n1
        auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(DIM_2); // 2:8
        OP_CHECK_IF((dim0 != fBaseParams.t1 || dim1 != fBaseParams.n1 || dim2 != BIT_NUMS),
                    OP_LOGE(context_, "The shape of softmaxMax is invalid, got (%ld,%ld,%ld), should be (%ld,%ld,%u)",
                            dim0, dim1, dim2, fBaseParams.t1, fBaseParams.n1, BIT_NUMS),
                    return ge::GRAPH_FAILED);
    }
    if (softmaxSumShape != nullptr) {
        auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
        if (softmaxSumShapeDim != DIM_3) { // softmax TND only support 3 dimensions
            OP_LOGE(context_, "The shape of softmaxSum is invalid, got %lu dimensions", softmaxSumShapeDim);
            return ge::GRAPH_FAILED;
        }
        auto dim0 = softmaxSumShape->GetStorageShape().GetDim(DIM_0); // 0:t1
        auto dim1 = softmaxSumShape->GetStorageShape().GetDim(DIM_1); // 1:n1
        auto dim2 = softmaxSumShape->GetStorageShape().GetDim(DIM_2); // 2:8
        OP_CHECK_IF((dim0 != fBaseParams.t1 || dim1 != fBaseParams.n1 || dim2 != BIT_NUMS),
                    OP_LOGE(context_, "The shape of softmaxSum is invalid, got (%ld,%ld,%ld), should be (%ld,%ld,%u)",
                            dim0, dim1, dim2, fBaseParams.t1, fBaseParams.n1, BIT_NUMS),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

int64_t FlashAttentionScoreGradTilingUs1s2Bs2David::GetTotalPerBatchNum(FuzzyBaseInfoParamsDavid& fBaseParams,
                                                                        uint8_t sparseType)
{
    int64_t totalPerBatchNum = 0;
    if (sparseType == static_cast<uint8_t>(SparseType::DENSE)) {
        totalPerBatchNum = fBaseParams.s1Outer * fBaseParams.s2Outer;
    } else {
        int64_t m = fBaseParams.s1Outer;
        int64_t n = fBaseParams.s2Outer;
        for (int64_t i = 0; i < m; ++i) {
            int64_t s2SparseLeft = std::max(static_cast<int64_t>(fBaseParams.s1TemplateType) * i - fBaseParams.s1Token,
                                            static_cast<int64_t>(0));
            int64_t s2SparseRight =
                std::min(std::min(static_cast<int64_t>(fBaseParams.s1TemplateType) * (i + 1), fBaseParams.s1) +
                             fBaseParams.s2Token,
                         fBaseParams.s2);
            if (s2SparseRight <= s2SparseLeft) {
                continue;
            }
            int64_t jBegin = s2SparseLeft / static_cast<int64_t>(fBaseParams.s2TemplateType);
            int64_t jEnd = (s2SparseRight - 1) / static_cast<int64_t>(fBaseParams.s2TemplateType);
            if (jBegin < 0) {
                jBegin = 0;
            }
            if (jEnd >= n) {
                jEnd = n - 1;
            }
            if (jEnd >= jBegin) {
                totalPerBatchNum += (jEnd - jBegin + 1);
            }
        }
    }
    return totalPerBatchNum;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::SaveToTilingData()
{
    s1s2BNGS1S2BaseParams_->set_coreNum(fBaseParams.coreNum);

    // set tilingdata baseinfo
    s1s2BNGS1S2BaseParams_->set_b(fBaseParams.b - fBaseParams.tailZeroCount);
    s1s2BNGS1S2BaseParams_->set_n2(fBaseParams.n2);
    s1s2BNGS1S2BaseParams_->set_g(fBaseParams.g);
    s1s2BNGS1S2BaseParams_->set_s1(fBaseParams.s1);
    s1s2BNGS1S2BaseParams_->set_d(fBaseParams.d);
    s1s2BNGS1S2BaseParams_->set_d1(fBaseParams.d1);
    s1s2BNGS1S2BaseParams_->set_s2(fBaseParams.s2);

    s1s2BNGS1S2BaseParams_->set_pseOptional(fBaseParams.pseOptional);
    s1s2BNGS1S2BaseParams_->set_pseType(fBaseParams.pseType);
    s1s2BNGS1S2BaseParams_->set_pseShapeType(fBaseParams.pseShapeType);
    s1s2BNGS1S2BaseParams_->set_pseLayoutType(fBaseParams.pseLayoutType);
    s1s2BNGS1S2BaseParams_->set_pseDtype(fBaseParams.pseDtype);
    s1s2BNGS1S2BaseParams_->set_attenMaskOptional(fBaseParams.attenMaskOptional);
    s1s2BNGS1S2BaseParams_->set_attenMaskShapeType(fBaseParams.attenMaskShapeType);
    s1s2BNGS1S2BaseParams_->set_attenMaskDtype(fBaseParams.attenMaskDtype);
    s1s2BNGS1S2BaseParams_->set_layout(fBaseParams.layoutType);
    s1s2BNGS1S2BaseParams_->set_tndMaxSumLayout(fBaseParams.tndMaxSumLayout);
    s1s2BNGS1S2BaseParams_->set_scaleValue(fBaseParams.scaleValue);
    s1s2BNGS1S2BaseParams_->set_keepProb(fBaseParams.keepProb);
    s1s2BNGS1S2BaseParams_->set_keepProbUint8(fBaseParams.keepProbUint8);

    // sink
    s1s2BNGS1S2BaseParams_->set_sinkOptional(fBaseParams.sinkOptional);
    s1s2BNGS1S2BaseParams_->set_s1SinkOuter(fBaseParams.s1SinkOuter);
    s1s2BNGS1S2BaseParams_->set_s2SinkOuter(fBaseParams.s2SinkOuter);

    // fBaseParams.s1Token int64_t类型   s1s2BNGS1S2BaseParams_->s1Token  int32_t类型 防止溢出
    s1s2BNGS1S2BaseParams_->set_s1Token(fBaseParams.s1Token > INT32_MAX ? INT32_MAX : fBaseParams.s1Token);
    s1s2BNGS1S2BaseParams_->set_s2Token(fBaseParams.s2Token > INT32_MAX ? INT32_MAX : fBaseParams.s2Token);

    s1s2BNGS1S2BaseParams_->set_sparseMode(fBaseParams.sparseMode);
    s1s2BNGS1S2BaseParams_->set_attenMaskS2Size(fBaseParams.attenMaskS2Size);
    s1s2BNGS1S2BaseParams_->set_attenMaskCompressMode(fBaseParams.attenMaskCompressMode);
    s1s2BNGS1S2BaseParams_->set_seed(fBaseParams.seed);
    s1s2BNGS1S2BaseParams_->set_offset(fBaseParams.offset);
    s1s2BNGS1S2BaseParams_->set_qStartIdx(fBaseParams.qStartIdx);
    s1s2BNGS1S2BaseParams_->set_kvStartIdx(fBaseParams.kvStartIdx);
    s1s2BNGS1S2BaseParams_->set_dropMaskOuter(fBaseParams.dropMaskOuter);
    OP_LOGI(context_, "Determine whether to swizzle, get fBaseParams.isSwizzle=[%d]",
            static_cast<int>(fBaseParams.isSwizzle));
    if (fBaseParams.isSwizzle) {
        s1s2BNGS1S2BaseParams_->set_totalPerBatchNum(GetTotalPerBatchNum(fBaseParams, fBaseParams.sparseType));
    }
    s1s2BNGS1S2BaseParams_->set_sparseType(fBaseParams.sparseType);

    // s1/s2 split
    s1s2BNGS1S2SplitCoreParams_->set_s1Outer(fBaseParams.s1Outer);
    s1s2BNGS1S2SplitCoreParams_->set_s1Inner(fBaseParams.s1Inner);
    s1s2BNGS1S2SplitCoreParams_->set_s1CvInner(fBaseParams.s1CvInner);
    s1s2BNGS1S2SplitCoreParams_->set_s1Tail(fBaseParams.s1Tail);
    s1s2BNGS1S2SplitCoreParams_->set_s1CvTail(fBaseParams.s1CvTail);
    s1s2BNGS1S2SplitCoreParams_->set_s2Outer(fBaseParams.s2Outer);
    s1s2BNGS1S2SplitCoreParams_->set_s2Inner(fBaseParams.s2Inner);
    s1s2BNGS1S2SplitCoreParams_->set_s2Tail(fBaseParams.s2Tail);

    s1s2BNGS1S2SplitCoreParams_->set_bandIdx(fBaseParams.bandIdx);
    s1s2BNGS1S2BlockNumList_->set_blockStarts(fBaseParams.blockStarts);
    s1s2BNGS1S2BlockNumList_->set_blockEnds(fBaseParams.blockEnds);
    s1s2BNGS1S2SplitCoreParams_->set_blockOuter(fBaseParams.blockOuter);
    s1s2BNGS1S2SplitCoreParams_->set_maxValidBBLen(fBaseParams.maxValidBBLen);
    s1s2BNGS1S2SplitCoreParams_->set_deterMaxRound(fBaseParams.deterMaxRound);
    s1s2BNGS1S2SplitCoreParams_->set_noNeedDeter(fBaseParams.noNeedDeter);
    if ((fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND) ||
         fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_DENSE)) &&
        fBaseParams.layoutType == INPUT_FORMAT_TND) {
        s1s2BNGS1S2SplitCoreParams_->set_dqIsNeedDeter(fBaseParams.startNeedSyncRound);
        s1s2BNGS1S2SplitCoreParams_->set_dkDvIsNeedDeter(fBaseParams.endNeedSyncRound);
    } else {
        s1s2BNGS1S2SplitCoreParams_->set_dqIsNeedDeter(fBaseParams.dqIsNeedDeter);
        s1s2BNGS1S2SplitCoreParams_->set_dkDvIsNeedDeter(fBaseParams.dkDvIsNeedDeter);
    }
    if (IsNewDeter() && deterParam_ != nullptr) {
        deterParam_->set_coreDivide(fBaseParams.coreDivide);
        deterParam_->set_deterPrefixStep(fBaseParams.deterPrefixStep);
        deterParam_->set_deterPrefix(fBaseParams.deterPrefix);
        deterParam_->set_deterPrefixAlign(fBaseParams.deterPrefixAlign);
        deterParam_->set_deterPrefix0(fBaseParams.deterPrefix0);
        deterParam_->set_deterPrefix1(fBaseParams.deterPrefix1);
        deterParam_->set_deterPrefix2(fBaseParams.deterPrefix2);
    }
    if (tndBaseInfo.isTndSwizzle && tndSwizzleParam_ != nullptr) {
        tndSwizzleParam_->set_tndS2BlockPrefixSum(tndBaseInfo.tndS2BlockPrefixSum);
        tndSwizzleParam_->set_tndSwizzleS1S2PrefixSum(tndBaseInfo.tndSwizzleS1S2PrefixSum);
        tndSwizzleParam_->set_tndSwizzleS1S2AlignPrefixSum(tndBaseInfo.tndSwizzleS1S2AlignPrefixSum);
        tndSwizzleParam_->set_tndValidBlockPrefixSum(tndBaseInfo.tndValidBlockPrefixSum);
        tndSwizzleParam_->set_totalValidS2TaskNum(tndBaseInfo.totalValidS2TaskNum);
        tndSwizzleParam_->set_activeCoreNum(tndBaseInfo.validS2ActiveCoreNum);
        tndSwizzleParam_->set_columnScheduleMode(tndBaseInfo.columnScheduleMode);
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDBandDeterParam()
{
    if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        return;
    }
    int64_t N11 = fBaseParams.n1 / fBaseParams.aicNum;
    int64_t N12 = fBaseParams.n1 % fBaseParams.aicNum;
    int64_t mnMax = 0;

    DeterPrefixData deterPrefixData;
    CalcleTNDBandDeterPrefix(deterPrefixData, N11, mnMax);

    if (N11 > 0 && fBaseParams.g == 1) {
        int64_t R0 = deterPrefixData.prefix0.back();
        deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix0.push_back(R0);
    }

    // 将最大轮次append在了prefix的最后，需要的时候可以直接取用，形式更简洁
    // prefix0的是乘了N1的结果，也可以不乘，乘了后二分查找不用额外申请空间
    int64_t R1 =
        fBaseParams.g == 1 ?
            (N12 > 0 ?
                 std::max(CeilDivideBy(deterPrefixData.prefix1.back() * N12, static_cast<int64_t>(fBaseParams.aicNum)),
                          mnMax) :
                 0) :
            std::max(
                CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.n1, static_cast<int64_t>(fBaseParams.aicNum)),
                mnMax);
    std::vector<int64_t> slicePrefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix1.push_back(R1);
    slicePrefix1.push_back(R1);

    fBaseParams.deterMaxRound += deterPrefixData.prefix0.back();
    fBaseParams.deterMaxRound += slicePrefix1.back();
    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
    std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
    std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
              fBaseParams.deterPrefixAlign);
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
    std::copy(slicePrefix1.begin(), slicePrefix1.end(), fBaseParams.deterPrefix1);

    CalcleTNDBandBns2DeterParam(deterPrefixData);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::SetCoreRoundInfo(TndBandDeterRoundInfo& tndBandDeterRoundInfo,
                                                                  uint64_t round, int64_t batchId)
{
    if (batchId != tndBandDeterRoundInfo.lastBatchId && tndBandDeterRoundInfo.lastBatchId != 0 &&
        tndBandDeterRoundInfo.coreLastBatchStartRound == 0) {
        tndBandDeterRoundInfo.coreLastBatchStartRound = tndBandDeterRoundInfo.lastValidRound;
    }
    if (batchId != tndBandDeterRoundInfo.lastBatchId) {
        tndBandDeterRoundInfo.coreFirstBatchLastRound = round;
    }
    tndBandDeterRoundInfo.lastValidRound = round;
    tndBandDeterRoundInfo.lastBatchId = batchId;
}

uint32_t FlashAttentionScoreGradTilingUs1s2Bs2David::GetDeterSparseTilingKey()
{
    if (!fBaseParams.isDeterministic) {
        return static_cast<uint32_t>(DeterSparseType::NO_DETER);
    }
    OP_LOGD("GetDeterSparseTilingKey", "==========================");
    OP_LOGD("GetDeterSparseTilingKey",
            "====fBaseParams.sparseMode: %d, fBaseParams.s1Token: %d, fBaseParams.s2Token: %d, fBaseParams.s1: %d, "
            "fBaseParams.s2: %d=============",
            fBaseParams.sparseMode, fBaseParams.s1Token, fBaseParams.s2Token, fBaseParams.s1, fBaseParams.s2);

    if (!fBaseParams.isSparse || (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK)) ||
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
         fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token >= fBaseParams.s2)) {
        OP_LOGD("GetDeterSparseTilingKey", "DETER_DENSE");
        return static_cast<uint32_t>(DeterSparseType::DETER_DENSE);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                fBaseParams.s1Token >= fBaseParams.s1 &&
                (fBaseParams.s2Token > NEGATIVE_128 && fBaseParams.s2Token <= 0)) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                fBaseParams.isS1S2Same)) {
        OP_LOGD("GetDeterSparseTilingKey", "DETER_CAUSAL");
        return static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
               // RIGHT_DOWN_CAUSAL场景和Band类似，直接走Band分支
               fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
               fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
        OP_LOGD("GetDeterSparseTilingKey", "DETER_BAND");
        return static_cast<uint32_t>(DeterSparseType::DETER_BAND);
    }
    return static_cast<uint32_t>(DeterSparseType::DETER_OLD);
}

uint8_t FlashAttentionScoreGradTilingUs1s2Bs2David::GetSparseType()
{
    // DENSE: 1）非sparse；2）ALL_MASK；3）NO_MASK & preToken>=Sq & nextToken>=Skv
    bool denseCondition = !fBaseParams.isSparse ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::ALL_MASK)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token >= fBaseParams.s2);

    bool casualCondition = false;
    bool bandCondition = false;
    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        casualCondition = (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0);
        if (fBaseParams.sparseMode != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            casualCondition = casualCondition && tndBaseInfo.isS1GreaterThanS2;
        } else {
            casualCondition = casualCondition && tndBaseInfo.isS1LessThanS2;
            if (!fBaseParams.isS1S2Same) {
                casualCondition = casualCondition && (fBaseParams.s1 % fBaseParams.s1CvInner == 0);
            }
        }

        bandCondition = fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND);
        bandCondition = bandCondition && fBaseParams.s1Token >= 0 && fBaseParams.s2Token >= 0;
    } else {
        // CASUAL: 1）LEFT_UP_CASUAL；2）RIGHT_DOWN_CASUAL；3）NO_MASK & preToken>=Sq & nextToken=0；4）BAND &
        // preToken>=Sq & nextToken=0
        casualCondition = (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) &&
                           fBaseParams.s1 <= fBaseParams.s2) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
                           fBaseParams.s1 <= fBaseParams.s2) ||
                          (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) &&
                           fBaseParams.s1Token >= fBaseParams.s1 && fBaseParams.s2Token == 0);

        // BAND: 1）NO_MASK剩余场景；2）BAND剩余场景；3）LEFT_UP_CAUSAL剩余场景；4）RIGHT_DOWN_CAUSAL剩余场景
        bandCondition = fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL);
    }
    OP_LOGI(
        "GetSparseType",
        "denseCondition = %d, casualCondition = %d, bandCondition = %d, isS1GreaterThanS2 = %d, isS1LessThanS2 = %d",
        denseCondition, casualCondition, bandCondition, tndBaseInfo.isS1GreaterThanS2, tndBaseInfo.isS1LessThanS2);

    if (denseCondition) {
        return static_cast<uint8_t>(SparseType::DENSE);
    } else if (casualCondition) {
        return static_cast<uint8_t>(SparseType::CASUAL);
    } else if (bandCondition) {
        return static_cast<uint8_t>(SparseType::BAND);
    } else {
        // 超L2优化暂不支持的sparse场景
        return static_cast<uint8_t>(SparseType::UNSUPPORTED);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleDeterParam()
{
    if (!fBaseParams.isDeterministic ||
        fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_OLD)) {
        return;
    }

    // EOD场景确定性计算需要使用真实的b
    int64_t oriBsize = fBaseParams.b;
    fBaseParams.b -= fBaseParams.tailZeroCount;

    uint8_t deterTilingSplitMode =
        (fBaseParams.s1Inner == fBaseParams.s2Inner ? 0 : (fBaseParams.s1Inner > fBaseParams.s2Inner ? 2 : 1));
    int64_t s1Outer{fBaseParams.s1Outer}, s2Outer{fBaseParams.s2Outer}, s1Inner{fBaseParams.s1Inner},
        s2Inner{fBaseParams.s2Inner};
    bool needChangeSplitItemMode2 =
        (deterTilingSplitMode == 2) &&
        (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE));
    bool needChangeSplitItemMode1 =
        (deterTilingSplitMode == 1) &&
        (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE));
    // 若是256 * 128或64 * 128切分，则
    if (needChangeSplitItemMode2) {
        fBaseParams.s2Inner = fBaseParams.s2Inner * 2;
        fBaseParams.s2Outer = CeilDivideBy(s2Outer, static_cast<int64_t>(2));
    } else if (needChangeSplitItemMode1) {
        fBaseParams.s1Inner = fBaseParams.s1Inner * 2;
        fBaseParams.s1Outer = CeilDivideBy(s1Outer, static_cast<int64_t>(2));
    }

    OP_LOGD("CalcleDeterParam",
            "fBaseParams.s1Outer: %ld, fBaseParams.s2Outer: %ld, fBaseParams.s1Inner: %ld, fBaseParams.s2Inner: %ld, "
            "needChangeSplitItemMode2: %d, needChangeSplitItemMode1: %d",
            fBaseParams.s1Outer, fBaseParams.s2Outer, fBaseParams.s1Inner, fBaseParams.s2Inner,
            needChangeSplitItemMode2, needChangeSplitItemMode1);

    if (fBaseParams.layoutType == INPUT_FORMAT_TND) {
        CalcleTNDDeterParam();
    } else if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL)) {
        CalcleCausalDeterParam();
    } else if (fBaseParams.deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        CalcleBandDeterParam();
    }
    // 还原
    if (needChangeSplitItemMode1 || needChangeSplitItemMode2) {
        fBaseParams.s1Outer = s1Outer;
        fBaseParams.s2Outer = s2Outer;
        fBaseParams.s1Inner = s1Inner;
        fBaseParams.s2Inner = s2Inner;
        OP_LOGD("CalcleDeterParam", "fBaseParams.deterMaxRound1: %ld", fBaseParams.deterMaxRound);
        fBaseParams.deterMaxRound *= 2;
        OP_LOGD("CalcleDeterParam", "fBaseParams.deterMaxRound2: %ld", fBaseParams.deterMaxRound);
    }

    // 还原eod
    fBaseParams.b = oriBsize;
    return;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleBandDeterParam()
{
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    int64_t actualCalcS1Token{fBaseParams.s1Token}, actualCalcS2Token{fBaseParams.s2Token};
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
        actualCalcS1Token = actualCalcS1Token + fBaseParams.s1 - fBaseParams.s2;
        actualCalcS2Token = actualCalcS2Token - fBaseParams.s1 + fBaseParams.s2;
    }
    int64_t p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
    int64_t q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;
    q = q > n ? n : q;
    p = p > m ? m : p;

    // 负数场景变换
    if (q < 0) {
        m = m + q;
        p = p + q;
        q = 1;
    } else if (p < 0) {
        n = n + p;
        q = p + q;
        p = 1;
    }

    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t L1, L2, L3, n_seg;
    if (p + q > m) {
        L1 = m - p;
        L2 = p + q - m;
        L3 = std::min(m - 1, n - q);
        n_seg = L1 + L2 + L3;
    } else {
        L1 = q - 1;
        L2 = std::min(n - q + 1, m + NUM_TWO - p - q);
        L3 = std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO));
        if (L3 == 0) {
            m = p + q + L2 - NUM_TWO;
        }
        n_seg = L1 + L2 + L3;
    }
    int64_t r1 = (m * n_seg - (m - p) * (m - p + 1) / NUM_TWO - (n_seg - q) * (n_seg - q + 1) / NUM_TWO) * b1;
    int64_t r2 = 0;
    if (b2 > 0) {
        if (p + q > m) {
            r2 = std::max(m * CeilDivideBy((n * b2), std::min(k, b2 * m)), n);
        } else {
            r2 = std::max(CeilDivideBy((n * b2), k) * (p + q - 1), n);
        }
    }
    fBaseParams.deterMaxRound = r1 + r2;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::TransTilingSplitMode(uint8_t mode, int64_t& m, int64_t& n)
{
    if (mode == 0) {
        return;
    } else if (mode == 1) {
        return;
    } else {
        n = CeilDivideBy(n, static_cast<int64_t>(2));
        return;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleCausalDeterParam()
{
    // uint8_t deterTilingSplitMode = (fBaseParams.s1Inner == fBaseParams.s2Inner ? 0 : (fBaseParams.s1Inner >
    // fBaseParams.s2Inner ? 2 : 1));
    int64_t m{fBaseParams.s1Outer}, n{fBaseParams.s2Outer}, k{static_cast<int64_t>(fBaseParams.aicNum)},
        b{fBaseParams.b * fBaseParams.n2};
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && m > n) {
        int64_t skipM = (fBaseParams.s1 - fBaseParams.s2) / (fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        m -= skipM;
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK) ||
               (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) && n > m)) {
        n = CeilDivideBy(fBaseParams.s1, fBaseParams.s2Inner * fBaseParams.s2CvRatio);
    } else if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) && m < n) {
        fBaseParams.deterSparseType = static_cast<uint32_t>(DeterSparseType::DETER_BAND);
        return;
    }

    int64_t bTail = b % k;
    int64_t rUpper = b / k * (n * m - n * (n - 1) / MULT_BASE);
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t t1 = n / (MULT_BASE * k);
    int64_t n1 = t * k;

    if (fBaseParams.g != 1) {
        rUpper += (MULT_BASE * m - n1 + 1) * t * (bTail / MULT_BASE);
    } else {
        rUpper += bTail * (n1 * m - n1 * (n1 - 1) / MULT_BASE) / k;
    }
    if (bTail % MULT_BASE == 1) {
        if ((t % MULT_BASE) == 1) {
            int64_t m1 = m - t1 * MULT_BASE * k;
            if (ell == 0) {
                int64_t rm3 = (fBaseParams.g != 1) ? (m + m1 + 1) * t1 : 0;
                rUpper += m1 + rm3;
            } else {
                int64_t rm3 = (fBaseParams.g != 1) ? (m + m1 + 1) * t1 : 0;
                rUpper += std::max(m1, MULT_BASE * m1 - MULT_BASE * k + 1) + rm3;
            }
            bTail = bTail - 1;
        } else {
            rUpper += (NUM_TWO * m - n1 + 1) * t / MULT_BASE;
        }
    }

    int64_t ell1, L;
    if (ell % MULT_BASE == 0) {
        ell1 = ell / MULT_BASE;
        L = MULT_BASE * (m - n) + ell + 1;
    } else {
        ell1 = ell / MULT_BASE + 1;
        L = MULT_BASE * (m - n) + ell;
    }
    rUpper += CeilDivideBy(ell1 * bTail, k) * L;
    rUpper *= fBaseParams.g;
    fBaseParams.deterMaxRound = rUpper;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDDeterParam()
{
    if (fBaseParams.layoutType != INPUT_FORMAT_TND) {
        return;
    }

    // 如果b小于deterPrefixThreshold，传完整的前缀和，否则按步长切分传部分
    if (fBaseParams.b > fBaseParams.deterPrefixThreshold) {
        fBaseParams.deterPrefixStep = CeilDivideBy(fBaseParams.b, fBaseParams.deterPrefixThreshold);
    }

    std::fill(std::begin(fBaseParams.deterPrefix0), std::end(fBaseParams.deterPrefix0), static_cast<int64_t>(0));
    std::fill(std::begin(fBaseParams.deterPrefix1), std::end(fBaseParams.deterPrefix1), static_cast<int64_t>(0));
    std::fill(std::begin(fBaseParams.deterPrefix2), std::end(fBaseParams.deterPrefix2), static_cast<int64_t>(0));

    CalcleTNDDenseDeterParam();
    CalcleTNDCausalDeterParam();
    CalcleTNDBandDeterParam();
    OP_LOGD("CalcleTNDDeterParam", "TND deterMaxRound is %ld.", fBaseParams.deterMaxRound);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDDenseDeterParam()
{
    if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_DENSE)) {
        return;
    }
    DeterPrefixData deterPrefixData;
    int64_t s1Max = 0;
    int64_t s2Max = 0;
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Outer = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        int64_t actualS2Outer =
            CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
        deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                              fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
        deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + actualS1Outer * actualS2Outer);
        deterPrefixData.deterPrefixAlign.push_back(
            deterPrefixData.deterPrefixAlign.back() +
            fBaseParams.actualSeqQlen[i] *
                AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));
        s1Max = actualS1Outer > s1Max ? actualS1Outer : s1Max;
        s2Max = actualS2Outer > s2Max ? actualS2Outer : s2Max;
        deterPrefixData.mNewList.push_back(actualS1Outer);
        deterPrefixData.nNewList.push_back(actualS2Outer);
    }
    int64_t totalArea = deterPrefixData.prefix1.back() * fBaseParams.n1;
    fBaseParams.deterMaxRound =
        std::max({CeilDivideBy(totalArea, static_cast<int64_t>(fBaseParams.aicNum)), s1Max * fBaseParams.g, s2Max});

    deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
    std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
    std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
              fBaseParams.deterPrefixAlign);
    deterPrefixData.prefix1.push_back(fBaseParams.deterMaxRound);
    CalcleTNDDenseBns2DeterParam(deterPrefixData);
    return;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDDenseBns2DeterParam(DeterPrefixData& deterPrefixData)
{
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2) {
        return;
    }

    // 最多允许coreNum列分给不同的核fBaseParams.deterMaxRound
    if (!SupportTNDBns2(deterPrefixData, fBaseParams.deterMaxRound)) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        return;
    }

    // BNS2分核按顺序分核，存在前后两核收尾分同一列的情况，计算可能分开的列
    std::vector<std::pair<uint64_t, uint64_t>> syncRounds, syncRoundRanges;
    std::fill(std::begin(fBaseParams.startNeedSyncRound), std::end(fBaseParams.startNeedSyncRound),
              static_cast<uint64_t>(0));
    std::fill(std::begin(fBaseParams.endNeedSyncRound), std::end(fBaseParams.endNeedSyncRound),
              static_cast<uint64_t>(0));
    std::fill(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              static_cast<int64_t>(-1));
    CalcleTNDDenseDeterSplitDkOffset(deterPrefixData, syncRounds, syncRoundRanges);
    std::copy(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              std::begin(fBaseParams.deterPrefix2));

    CalcleTNDDeterSyncRounds(syncRounds, syncRoundRanges);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDDenseDeterSplitDkOffset(
    DeterPrefixData& deterPrefixData, std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
    std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges)
{
    int64_t precoreLastBatchStartRound = 0;
    for (uint32_t coreId = 0; coreId < CORE_LIST_NUM; coreId++) {
        if (fBaseParams.deterMaxRound == 0 || coreId > fBaseParams.aicNum - 1) {
            continue;
        }
        auto actualSeqKvlenTensor =
            context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
        const int64_t* kvValue = actualSeqKvlenTensor->GetData<int64_t>();

        TndBandDeterRoundInfo tndBandDeterRoundInfo;
        for (int64_t round = fBaseParams.deterMaxRound; round > 0; round--) {
            auto oriCoordinateInfo = CalTNDDenseIndex<static_cast<uint32_t>(DeterSparseType::DETER_DENSE)>(
                deterPrefixData, coreId + 1, round, fBaseParams.n1);
            int64_t w, x, y;
            std::tie(w, x, y) = oriCoordinateInfo;

            int64_t batchId = CeilDivideBy(w, fBaseParams.n1);
            if (w == -1 || batchId > fBaseParams.b) {
                continue;
            }
            int64_t m = deterPrefixData.mNewList[batchId - 1];

            if (fBaseParams.separateDkOffset[coreId] == -1 && x < m && round == fBaseParams.deterMaxRound) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
        if (coreId != 0) {
            uint64_t startSyncRound = precoreLastBatchStartRound;
            uint64_t endSyncRound = tndBandDeterRoundInfo.coreFirstBatchLastRound;
            if (startSyncRound > endSyncRound) {
                syncRounds.push_back(std::make_pair(startSyncRound, endSyncRound));
            } else {
                syncRoundRanges.push_back(std::make_pair(startSyncRound, endSyncRound));
            }
        }
        if (coreId == 0) {
            precoreLastBatchStartRound = 1; // 如果在0-1核上涉及切BN轴，那么0核的起点就是第一个roundid = 1
        } else {
            precoreLastBatchStartRound = tndBandDeterRoundInfo.coreLastBatchStartRound;
        }
    }
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::SupportTNDBns2(DeterPrefixData& deterPrefixData, int64_t round)
{
    for (int64_t b = 0; b < fBaseParams.b; b++) {
        int64_t m = deterPrefixData.mNewList[b];
        int64_t n = deterPrefixData.nNewList[b];
        if ((round / Gcd(m, round)) >= n) {
            continue;
        }
        return false;
    }
    return true;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDBandBns2DeterParam(DeterPrefixData& deterPrefixData)
{
    if (fBaseParams.splitAxis != SplitAxisEnum::BN2S2) {
        return;
    }

    // 最多允许coreNum列分给不同的核
    if (!SupportTNDBns2(deterPrefixData, deterPrefixData.prefix1.back())) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
        OP_LOGD("CalcleTNDBandBns2DeterParam", "Not support BNS2, change to BN2GS1S2.");
        return;
    }

    // BNS2分核按顺序分核，存在前后两核收尾分同一列的情况，计算可能分开的列
    std::vector<std::pair<uint64_t, uint64_t>> syncRounds, syncRoundRanges;
    std::fill(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              static_cast<int64_t>(-1));
    std::fill(std::begin(fBaseParams.startNeedSyncRound), std::end(fBaseParams.startNeedSyncRound),
              static_cast<uint64_t>(0));
    std::fill(std::begin(fBaseParams.endNeedSyncRound), std::end(fBaseParams.endNeedSyncRound),
              static_cast<uint64_t>(0));

    CalcleTNDBandDeterSplitDkOffset(deterPrefixData, syncRounds, syncRoundRanges);
    std::copy(std::begin(fBaseParams.separateDkOffset), std::end(fBaseParams.separateDkOffset),
              std::begin(fBaseParams.deterPrefix2));

    CalcleTNDDeterSyncRounds(syncRounds, syncRoundRanges);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDCausalDeterPrefix(DeterPrefixData& deterPrefixData,
                                                                            int64_t& m0Max, int64_t& m1Max,
                                                                            int64_t& m2Max)
{
    int64_t N12 = fBaseParams.g == 1 ? fBaseParams.n2 % fBaseParams.aicNum % NUM_TWO : fBaseParams.n2 % NUM_TWO;
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Outer = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        int64_t actualS2Outer =
            CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
        deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                              fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
        deterPrefixData.deterPrefixAlign.push_back(
            deterPrefixData.deterPrefixAlign.back() +
            fBaseParams.actualSeqQlen[i] *
                AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));

        // left_up_causal场景下，如果m<n, 需要将n裁剪为m
        if (actualS1Outer < actualS2Outer) {
            actualS2Outer = actualS1Outer;
        }

        m0Max = std::max(m0Max, fBaseParams.g * (NUM_TWO * actualS1Outer - actualS2Outer + 1));
        deterPrefixData.prefix0.push_back(deterPrefixData.prefix0.back() +
                                          (NUM_TWO * actualS1Outer - actualS2Outer + 1) * actualS2Outer);

        if (N12 > 0) {
            deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() +
                                              (actualS1Outer - (actualS2Outer + 1) / NUM_TWO + 1) *
                                                  (actualS2Outer / NUM_TWO));
            if (fBaseParams.g == 1 || (actualS2Outer >= NUM_TWO && fBaseParams.g != 1)) {
                m1Max = std::max(m1Max, fBaseParams.g * (actualS1Outer - (actualS2Outer + 1) / NUM_TWO + 1));
            }

            deterPrefixData.prefix2.push_back(deterPrefixData.prefix2.back() +
                                              (actualS1Outer - actualS2Outer / NUM_TWO) *
                                                  ((actualS2Outer + 1) / NUM_TWO));
            m2Max = std::max(m2Max, fBaseParams.g * (actualS1Outer - actualS2Outer / NUM_TWO));
        }
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDCausalDeterParam()
{
    if (fBaseParams.deterSparseType != static_cast<uint32_t>(DeterSparseType::DETER_CAUSAL)) {
        return;
    }
    fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    int64_t m0Max{0}, m1Max{0}, m2Max{0};
    DeterPrefixData deterPrefixData;
    CalcleTNDCausalDeterPrefix(deterPrefixData, m0Max, m1Max, m2Max);
    OP_LOGD("CalcleTNDCausalDeterParam", "m0Max: %ld, m1Max: %ld, m2Max: %ld, fBaseParams.g: %ld!", m0Max, m1Max, m2Max,
            fBaseParams.g);

    deterPrefixData.deterPrefix = SliceVector(deterPrefixData.deterPrefix, fBaseParams.deterPrefixStep);
    deterPrefixData.deterPrefixAlign = SliceVector(deterPrefixData.deterPrefixAlign, fBaseParams.deterPrefixStep);
    std::copy(deterPrefixData.deterPrefix.begin(), deterPrefixData.deterPrefix.end(), fBaseParams.deterPrefix);
    std::copy(deterPrefixData.deterPrefixAlign.begin(), deterPrefixData.deterPrefixAlign.end(),
              fBaseParams.deterPrefixAlign);

    if (fBaseParams.g == 1) {
        CalcleTNDCausalDeterParamNormal(deterPrefixData, m0Max, m1Max, m2Max);
    } else {
        CalcleTNDCausalDeterParamGQA(deterPrefixData, m0Max, m1Max, m2Max);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDCausalDeterParamNormal(DeterPrefixData& deterPrefixData,
                                                                                 const int64_t m0Max,
                                                                                 const int64_t m1Max,
                                                                                 const int64_t m2Max)
{
    int64_t N11 = fBaseParams.n2 % fBaseParams.aicNum / NUM_TWO;
    int64_t N12 = fBaseParams.n2 % fBaseParams.aicNum % NUM_TWO;
    int64_t prefix0Max1 = deterPrefixData.prefix0.back() / NUM_TWO * (fBaseParams.n2 / fBaseParams.aicNum);
    int64_t prefix0Max2 = std::max(
        CeilDivideBy(deterPrefixData.prefix0.back() * N11 * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
        m0Max);
    OP_LOGD("CalcleTNDCausalDeterParam", "N11: %ld, N12: %ld, prefix0Max1: %ld, prefix0Max2: %ld!", N11, N12,
            prefix0Max1, prefix0Max2);
    deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix0.push_back(prefix0Max1);
    fBaseParams.deterMaxRound += prefix0Max1;
    OP_LOGD("CalcleTNDCausalDeterParam", "fBaseParams.deterMaxRound: %ld!", fBaseParams.deterMaxRound);
    if (N11 > 0) {
        deterPrefixData.prefix0.push_back(prefix0Max2);
        fBaseParams.deterMaxRound += prefix0Max2;
        OP_LOGD("CalcleTNDCausalDeterParam", "fBaseParams.deterMaxRound: %ld!", fBaseParams.deterMaxRound);
    } else {
        deterPrefixData.prefix0.push_back(0);
    }
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);

    if (N12 > 0) {
        int64_t r1 = std::max(
            CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m1Max);
        int64_t r2 = std::max(
            CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m2Max);
        OP_LOGD("CalcleTNDCausalDeterParam", "r1: %ld, r2: %ld!", r1, r2);
        deterPrefixData.prefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix2 = SliceVector(deterPrefixData.prefix2, fBaseParams.deterPrefixStep);
        deterPrefixData.prefix1.push_back(r1);
        deterPrefixData.prefix2.push_back(r2);
        OP_LOGD("CalcleTNDCausalDeterParam", "fBaseParams.deterMaxRound: %ld!", fBaseParams.deterMaxRound);
        OP_LOGD("CalcleTNDCausalDeterParam",
                "deterPrefixData.prefix1.back(): %ld, deterPrefixData.prefix2.back(): %ld!",
                deterPrefixData.prefix1.back(), deterPrefixData.prefix2.back());
        fBaseParams.deterMaxRound += deterPrefixData.prefix1.back();
        OP_LOGD("CalcleTNDCausalDeterParam", "fBaseParams.deterMaxRound: %ld!", fBaseParams.deterMaxRound);
        fBaseParams.deterMaxRound += deterPrefixData.prefix2.back();
        OP_LOGD("CalcleTNDCausalDeterParam", "fBaseParams.deterMaxRound: %ld!", fBaseParams.deterMaxRound);
        std::copy(deterPrefixData.prefix1.begin(), deterPrefixData.prefix1.end(), fBaseParams.deterPrefix1);
        std::copy(deterPrefixData.prefix2.begin(), deterPrefixData.prefix2.end(), fBaseParams.deterPrefix2);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDCausalDeterParamGQA(DeterPrefixData& deterPrefixData,
                                                                              const int64_t m0Max, const int64_t m1Max,
                                                                              const int64_t m2Max)
{
    int64_t N11 = fBaseParams.n2 / NUM_TWO;
    int64_t N12 = fBaseParams.n2 % NUM_TWO;

    int64_t prefix0Max{0}, prefix1Max{0}, prefix2Max{0};
    if (fBaseParams.n2 == 1) {
        prefix0Max = 0;
        prefix1Max = std::max(
            CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m1Max);
        prefix2Max = std::max(
            CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m2Max);
        fBaseParams.deterMaxRound = prefix1Max + prefix2Max;
    } else if (N12 == 0) {
        prefix0Max = std::max(CeilDivideBy(deterPrefixData.prefix0.back() * fBaseParams.g * N11,
                                           static_cast<int64_t>(fBaseParams.aicNum)),
                              m0Max);
        prefix1Max = 0;
        prefix2Max = 0;
        fBaseParams.deterMaxRound = prefix0Max;
    } else {
        prefix0Max = std::max(CeilDivideBy(deterPrefixData.prefix0.back() * fBaseParams.g * N11,
                                           static_cast<int64_t>(fBaseParams.aicNum)),
                              m0Max);
        prefix1Max = std::max(
            CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m1Max);
        prefix2Max = std::max(
            CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, static_cast<int64_t>(fBaseParams.aicNum)),
            m2Max);
        int64_t totalRound = prefix0Max + prefix1Max + prefix2Max;

        int64_t k2 = CeilDivideBy(static_cast<int64_t>(fBaseParams.aicNum), fBaseParams.n2);
        int64_t k1 = static_cast<int64_t>(fBaseParams.aicNum) - k2;

        int64_t prefix0MaxNew = std::max(CeilDivideBy(deterPrefixData.prefix0.back() * fBaseParams.g * N11, k1), m0Max);
        int64_t prefix1MaxNew = std::max(CeilDivideBy(deterPrefixData.prefix1.back() * fBaseParams.g, k2), m1Max);
        int64_t prefix2MaxNew = std::max(CeilDivideBy(deterPrefixData.prefix2.back() * fBaseParams.g, k2), m2Max);
        int64_t totalRoundNew = std::max(prefix0MaxNew, prefix1MaxNew + prefix2MaxNew);
        if (totalRoundNew < totalRound) {
            fBaseParams.coreDivide = true;
            prefix0Max = prefix0MaxNew;
            prefix1Max = prefix1MaxNew;
            prefix2Max = prefix2MaxNew;
            totalRound = totalRoundNew;
        }
        fBaseParams.deterMaxRound = totalRound;
    }

    deterPrefixData.prefix0 = SliceVector(deterPrefixData.prefix0, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix1 = SliceVector(deterPrefixData.prefix1, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix2 = SliceVector(deterPrefixData.prefix2, fBaseParams.deterPrefixStep);
    deterPrefixData.prefix0.push_back(prefix0Max);
    deterPrefixData.prefix1.push_back(prefix1Max);
    deterPrefixData.prefix2.push_back(prefix2Max);
    std::copy(deterPrefixData.prefix0.begin(), deterPrefixData.prefix0.end(), fBaseParams.deterPrefix0);
    std::copy(deterPrefixData.prefix1.begin(), deterPrefixData.prefix1.end(), fBaseParams.deterPrefix1);
    std::copy(deterPrefixData.prefix2.begin(), deterPrefixData.prefix2.end(), fBaseParams.deterPrefix2);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleActualToken(int64_t batchIdx, int64_t& actualCalcS1Token,
                                                                   int64_t& actualCalcS2Token)
{
    int64_t actualS1Len = fBaseParams.actualSeqQlen[batchIdx];
    int64_t actualS2Len = fBaseParams.actualSeqKvlen[batchIdx];
    // 对unpad场景的token值做二次校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    actualCalcS1Token = fBaseParams.s1Token;
    actualCalcS2Token = fBaseParams.s2Token;
    if ((fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) &&
         batchIdx != fBaseParams.bandIdx) ||
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
         batchIdx != fBaseParams.bandIdx)) {
        actualCalcS1Token = INT32_MAX;
        actualCalcS2Token = 0;
    }
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) ||
        (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL) &&
         batchIdx == fBaseParams.bandIdx)) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDBandDeterPrefix(DeterPrefixData& deterPrefixData, int64_t N11,
                                                                          int64_t& mnMax)
{
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t m, n, p, q, mNew, nNew;
        m = CeilDivideBy(fBaseParams.actualSeqQlen[i], fBaseParams.s1Inner * fBaseParams.s1CvRatio);
        n = CeilDivideBy(fBaseParams.actualSeqKvlen[i], fBaseParams.s2Inner * fBaseParams.s2CvRatio);
        deterPrefixData.deterPrefix.push_back(deterPrefixData.deterPrefix.back() +
                                              fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i]);
        deterPrefixData.deterPrefixAlign.push_back(
            deterPrefixData.deterPrefixAlign.back() +
            fBaseParams.actualSeqQlen[i] *
                AlignTo(fBaseParams.actualSeqKvlen[i], static_cast<int64_t>(ConstAxisTemplateNum::NUM16)));

        int64_t actualCalcS1Token, actualCalcS2Token;
        CalcleActualToken(i, actualCalcS1Token, actualCalcS2Token);
        p = CeilDivideBy(actualCalcS1Token, fBaseParams.s1Inner * fBaseParams.s1CvRatio) + 1;
        q = CeilDivideBy(actualCalcS2Token, fBaseParams.s2Inner * fBaseParams.s2CvRatio) + 1;
        p = p > m ? m : p;
        q = q > n ? n : q;

        // 负数场景变换
        if (p < 0) {
            n = n + p;
            q = p + q;
            p = 1;
        } else if (q < 0) {
            m = m + q;
            p = p + q;
            q = 1;
        }
        if (p + q <= m) {
            int64_t L1{q - 1}, L2{std::min(n - q + 1, m + NUM_TWO - p - q)},
                L3{std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - NUM_TWO))};
            mNew = L3 == 0 ? p + q + L2 - NUM_TWO : m;
            nNew = L1 + L2 + L3;
            mnMax = n <= m || fBaseParams.g == 1 ? std::max({mnMax, (p + q - 1) * fBaseParams.g}) :
                                                   std::max({mnMax, mNew * fBaseParams.g, p + q - 1});
            deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + std::min(mNew, nNew) * (p + q - 1));
        } else {
            mNew = m;
            nNew = std::min(m - 1 + q, n);
            if (p + q <= n) {
                mnMax = std::max({mnMax, mNew * fBaseParams.g, p + q - 1});
                deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + mNew * (p + q - 1));
            } else {
                mnMax = std::max({mnMax, mNew * fBaseParams.g, nNew});
                deterPrefixData.prefix1.push_back(deterPrefixData.prefix1.back() + mNew * nNew);
            }
        }

        // 注意，这里更新后的mNewList和nNewList是可以替代list_m和list_n的，它们的引入是避免空白行/列
        deterPrefixData.mNewList.push_back(mNew);
        deterPrefixData.nNewList.push_back(nNew);
        deterPrefixData.pNewList.push_back(p);
        deterPrefixData.qNewList.push_back(q);
        if (N11 > 0 && fBaseParams.g == 1) {
            int64_t R0 =
                deterPrefixData.prefix0.back() +
                (mNew * nNew - (mNew - p) * (mNew - p + 1) / NUM_TWO - (nNew - q) * (nNew - q + 1) / NUM_TWO) * N11;
            deterPrefixData.prefix0.push_back(R0);
        }
    }
}

template <const uint32_t deterSparseType>
std::tuple<int64_t, int64_t, int64_t> FlashAttentionScoreGradTilingUs1s2Bs2David::CalTNDDenseIndex(
    DeterPrefixData& deterPrefixData, int64_t coreId, int64_t roundId, int64_t N1)
{
    int64_t unPadRoundMax{deterPrefixData.prefix1[fBaseParams.b + 1]}, ID{(coreId - 1) * unPadRoundMax + roundId}, w{0};
    while (w < fBaseParams.b && ID > deterPrefixData.prefix1[w + 1] * N1) {
        w += 1;
    }
    int64_t delta = ID - deterPrefixData.prefix1[w] * N1;

    if (w >= fBaseParams.b) {
        return std::make_tuple(-1, -1, -1);
    }

    int64_t m{deterPrefixData.mNewList[w]}, n{deterPrefixData.nNewList[w]}, p, q;
    if constexpr (deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
        p = deterPrefixData.pNewList[w];
        q = deterPrefixData.qNewList[w];
        if (p + q <= m) {
            if (n >= m) {
                n = p + q - 1;
            } else {
                m = p + q - 1;
            }
        } else {
            if (p + q <= n) {
                n = p + q - 1;
            }
        }
    }

    int64_t currentBaseNum = m * n;
    int64_t batchId = w + 1;
    int64_t deltaN = (delta - 1) / currentBaseNum + 1;
    delta = delta % currentBaseNum;
    delta = delta != 0 ? delta : currentBaseNum;

    int64_t g = Gcd(m, unPadRoundMax);
    if (g == 0) {
        return std::make_tuple(-1, -1, -1);
    }
    int64_t t1{unPadRoundMax / g}, t2{m / g}, x{((delta - 1) % m) + 1}, y{(delta - 1) / m + 1};
    if (t1 < n) {
        int64_t n1 = n % t1;
        n1 = n1 == 0 ? t1 : n1;
        if (y <= n - n1) {
            int64_t deltaAdj = CeilDivideBy(y, t1);
            delta += deltaAdj;
            if (delta > deltaAdj * t2 * unPadRoundMax) {
                delta -= t2 * unPadRoundMax;
            }
            x = ((delta - 1) % m) + 1;
            y = (delta - 1) / m + 1;
        }
    }
    return std::make_tuple((batchId - 1) * N1 + deltaN, x, y);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::UpdateSeparateDkOffsetLargeM(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
    TndBandDeterRoundInfo& tndBandDeterRoundInfo)
{
    auto actualSeqKvlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
    const int64_t* kvValue = actualSeqKvlenTensor->GetData<int64_t>();
    int64_t m, n, p, q, x, y, w, coreId, round;
    std::tie(m, n, p, q, x, y, w, coreId, round) = coordinateInfo;
    if (n >= m) {
        if (y - q + 1 <= x && x <= p + y - 1) {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        } else {
            y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
            bool isValid = 1 <= y && y <= n;
            if (fBaseParams.separateDkOffset[coreId] == -1 && isValid && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            if (isValid) {
                SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
            }
        }
    } else {
        // 优先处理拼出来后有空白格的情况
        if (q + 1 <= y && y <= std::min(n, p + NUM_TWO * q - NUM_TWO)) {
            if (x != p + q - 1 && fBaseParams.separateDkOffset[coreId] == -1) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
        } else {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
        }
        SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
    }
}

std::vector<uint64_t> FlashAttentionScoreGradTilingUs1s2Bs2David::CalculateSyncRound(
    std::vector<std::pair<uint64_t, uint64_t>> syncRounds)
{
    if (syncRounds.size() == 0) {
        return {};
    }
    if (syncRounds.size() == 1) {
        return {syncRounds[0].first};
    }
    uint64_t minStartSyncRound = syncRounds[0].first;
    for (auto syncRound : syncRounds) {
        minStartSyncRound = syncRound.first < minStartSyncRound ? syncRound.first : minStartSyncRound;
    }

    std::vector<std::pair<uint64_t, uint64_t>> smallSyncRounds;
    for (auto syncRound : syncRounds) {
        if (syncRound.second > minStartSyncRound) {
            smallSyncRounds.push_back(syncRound);
        }
    }
    std::vector<uint64_t> needSyncRounds = CalculateSyncRound(smallSyncRounds);
    needSyncRounds.push_back(minStartSyncRound);
    return needSyncRounds;
}

int64_t FlashAttentionScoreGradTilingUs1s2Bs2David::GetKeyOffset(const int64_t* kvValue, int64_t w, int64_t y)
{
    int64_t batchId = CeilDivideBy(w, fBaseParams.n2) - 1;
    int64_t n2Idx = w - batchId * fBaseParams.n2 - 1;
    int64_t s2Idx = y - 1;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    int64_t seqKvLenPrefix = batchId == 0 ? 0 : kvValue[batchId - 1];
    bOffset = seqKvLenPrefix * fBaseParams.n2 * fBaseParams.d;
    n2Offset = n2Idx * fBaseParams.d;
    s2Offset = s2Idx * fBaseParams.s2Inner * fBaseParams.n2 * fBaseParams.d;
    return bOffset + n2Offset + s2Offset;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::IsSeparateS2(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo)
{
    int64_t m, n, p, q, x, y;
    std::tie(m, n, p, q, x, y, std::ignore, std::ignore, std::ignore) = coordinateInfo;
    // 补充的 is_max 运算
    bool isSeparate = true;

    if (p + q <= m && m > n) {
        // 这里要注意None的情况
        if (y <= q) {
            if (x - y >= p - 1) {
                isSeparate = false;
            }
            // 有空白块的情况交给上面处理了，因为有可能R_max坐标是None
        } else if (y > std::min(n, p + NUM_TWO * q - NUM_TWO)) {
            x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
            if (x == m || x < y - q + 1) {
                isSeparate = false;
            }
        }
    } else if (p + q <= m && n >= m) {
        if (y <= std::min(n, m + 1 - p)) {
            if (x - y == p - 1) {
                isSeparate = false;
            }
        } else {
            if (x == m) {
                isSeparate = false;
            }
        }
    } else {
        if (y <= m - p) {
            if (x - y == p - 1) {
                isSeparate = false;
            }
        } else {
            if (x == m) {
                isSeparate = false;
            }
        }
    }
    return isSeparate;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::UpdateSeparateDkOffsetSmallM(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
    TndBandDeterRoundInfo& tndBandDeterRoundInfo)
{
    auto actualSeqKvlenTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::ACTUAL_SEQ_KV_LEN));
    const int64_t* kvValue = actualSeqKvlenTensor->GetData<int64_t>();
    int64_t m, n, p, q, x, y, w, coreId, round;
    std::tie(m, n, p, q, x, y, w, coreId, round) = coordinateInfo;
    if (p + q <= n) {
        if (x - p + 1 <= y && y <= x + q - 1) {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        } else if (y < x - p + 1 && 1 <= y + p + q - 1 && y + p + q - 1 <= n) {
            y = y + p + q - 1;
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
    } else {
        if (x - p + 1 <= y && y <= x + q - 1) {
            if (fBaseParams.separateDkOffset[coreId] == -1 && IsSeparateS2(coordinateInfo)) {
                fBaseParams.separateDkOffset[coreId] = GetKeyOffset(kvValue, w, y);
            }
            SetCoreRoundInfo(tndBandDeterRoundInfo, round, w);
        }
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::UpdateSeparateDkOffset(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
    TndBandDeterRoundInfo& tndBandDeterRoundInfo)
{
    int64_t m, p, q;
    std::tie(m, std::ignore, p, q, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = coordinateInfo;
    if (p + q <= m) {
        UpdateSeparateDkOffsetLargeM(coordinateInfo, tndBandDeterRoundInfo);
    } else {
        UpdateSeparateDkOffsetSmallM(coordinateInfo, tndBandDeterRoundInfo);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDBandDeterSplitDkOffset(
    DeterPrefixData& deterPrefixData, std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
    std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges)
{
    int64_t precoreLastBatchStartRound = 0;
    int64_t N11 = fBaseParams.n1 / fBaseParams.aicNum;
    int64_t N12 = fBaseParams.n1 % fBaseParams.aicNum;
    for (uint32_t coreId = 0; coreId < CORE_LIST_NUM; coreId++) {
        if (deterPrefixData.prefix1.back() == 0 || coreId >= fBaseParams.aicNum - 1) {
            continue;
        }
        TndBandDeterRoundInfo tndBandDeterRoundInfo;
        for (uint64_t round = deterPrefixData.prefix1.back(); round > 0; round--) {
            auto oriCoordinateInfo = CalTNDDenseIndex<static_cast<uint32_t>(DeterSparseType::DETER_BAND)>(
                deterPrefixData, coreId + 1, round, fBaseParams.n1 % static_cast<int64_t>(fBaseParams.aicNum));
            int64_t w, x, y;
            std::tie(w, x, y) = oriCoordinateInfo;
            int64_t batchId = CeilDivideBy(w, N12);
            if (w == -1 || batchId > fBaseParams.b) {
                continue;
            }

            int64_t m = deterPrefixData.mNewList[batchId - 1];
            int64_t n = deterPrefixData.nNewList[batchId - 1];
            int64_t p = deterPrefixData.pNewList[batchId - 1];
            int64_t q = deterPrefixData.qNewList[batchId - 1];

            int64_t wTail = w % N12;
            wTail = wTail != 0 ? wTail : N12;
            w = (CeilDivideBy(w, N12) - 1) * fBaseParams.n1 + N11 * fBaseParams.aicNum + wTail;

            std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> coordinateInfo =
                std::make_tuple(m, n, p, q, x, y, w, coreId, round);
            UpdateSeparateDkOffset(coordinateInfo, tndBandDeterRoundInfo);
        }

        if (coreId != 0) {
            uint64_t startSyncRound = precoreLastBatchStartRound + deterPrefixData.prefix0.back();
            uint64_t endSyncRound = tndBandDeterRoundInfo.coreFirstBatchLastRound + deterPrefixData.prefix0.back();
            if (startSyncRound > endSyncRound) {
                syncRounds.push_back(std::make_pair(startSyncRound, endSyncRound));
            } else {
                syncRoundRanges.push_back(std::make_pair(startSyncRound, endSyncRound));
            }
        }
        precoreLastBatchStartRound = tndBandDeterRoundInfo.coreLastBatchStartRound;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalcleTNDDeterSyncRounds(
    std::vector<std::pair<uint64_t, uint64_t>>& syncRounds, std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges)
{
    if (syncRounds.size() + syncRoundRanges.size() > CORE_LIST_NUM) {
        fBaseParams.startNeedSyncRound[0] = 1;
        fBaseParams.endNeedSyncRound[0] = std::numeric_limits<uint64_t>::max();
        OP_LOGD("CalcleTNDBandDeterParam", "All rounds need sync!");
        return;
    }
    std::vector<uint64_t> needSyncRoundsTmp = CalculateSyncRound(syncRounds);
    std::vector<uint64_t> needSyncRounds;
    for (uint64_t needSyncRound : needSyncRoundsTmp) {
        bool isValid = true;
        for (auto needSyncRoundRange : syncRoundRanges) {
            if (needSyncRound >= needSyncRoundRange.first && needSyncRound <= needSyncRoundRange.second) {
                isValid = false;
                break;
            }
        }
        if (isValid) {
            needSyncRounds.push_back(needSyncRound);
        }
    }

    for (uint32_t i = 0; i < needSyncRounds.size(); i++) {
        fBaseParams.startNeedSyncRound[i] = needSyncRounds[i];
        fBaseParams.endNeedSyncRound[i] = needSyncRounds[i];
    }
    for (uint32_t i = 0; i < syncRoundRanges.size(); i++) {
        auto syncRoundPair = syncRoundRanges[i];
        fBaseParams.startNeedSyncRound[i + needSyncRounds.size()] = syncRoundPair.first;
        fBaseParams.endNeedSyncRound[i + needSyncRounds.size()] = syncRoundPair.second;
    }

    uint64_t allNeedSyncLoopNums = 0;
    for (uint64_t loopIdx = 0; static_cast<int64_t>(loopIdx) < fBaseParams.deterMaxRound; loopIdx++) {
        for (uint32_t i = 0; i < CORE_LIST_NUM; i++) {
            if (fBaseParams.endNeedSyncRound[i] == 0) {
                break;
            }
            if (loopIdx >= fBaseParams.startNeedSyncRound[i] && loopIdx <= fBaseParams.endNeedSyncRound[i]) {
                allNeedSyncLoopNums++;
                break;
            }
        }
    }
    if (fBaseParams.deterMaxRound < static_cast<int64_t>(allNeedSyncLoopNums) * NUM_TWO) {
        fBaseParams.splitAxis = SplitAxisEnum::BN2GS1S2;
    }
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::IsNewDeter()
{
    return fBaseParams.deterSparseType >= static_cast<uint32_t>(DeterSparseType::DETER_DENSE) &&
           fBaseParams.deterSparseType <= static_cast<uint32_t>(DeterSparseType::DETER_BAND) &&
           (fBaseParams.layoutType == INPUT_FORMAT_TND);
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::SetPrefixSparseParams()
{
    auto prefixNTensor = context_->GetOptionalInputTensor(static_cast<size_t>(InputIndex::PREFIX_N));
    if (prefixNTensor == nullptr) {
        OP_LOGW(context_, "FAG Us1s2Bbn2gs1s2 sparseMode is prefix, but prefixN tensor is null!");
        return false;
    }

    auto& prefixShape = prefixNTensor->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1 || prefixShape.GetDim(0) != fBaseParams.b) {
        OP_LOGW(context_, "FAG Us1s2Bbn2gs1s2 sparseMode is prefix, but prefixshape size[%zu] or value is invalid!",
                prefixShape.GetDimNum());
        return false;
    }

    std::vector<int64_t> prefixN;
    const int64_t* value = prefixNTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "FAG Us1s2Bbn2gs1s2 sparseMode is prefix, but prefixN data is null pointer!");
        return false;
    }
    const size_t shapeSize = prefixNTensor->GetShapeSize();
    for (size_t i = 0; i < shapeSize; i++) {
        prefixN.push_back(value[i]);
    }

    if (static_cast<int64_t>(prefixN.size()) == fBaseParams.b && prefixN.size() <= BATCH_MAX_SIZE) {
        std::copy(prefixN.begin(), prefixN.end(), fBaseParams.prefixN);
        return true;
    } else {
        OP_LOGW(context_, "FAG Us1s2Bbn2gs1s2 sparseMode is prefix, but prefixN size[%zu] or value is invalid!",
                prefixN.size());
        return false;
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessPseSparseMode8()
{
    for (int64_t boIdx = 0; boIdx < fBaseParams.b; boIdx++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[boIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[boIdx];
        if (boIdx == 0) {
            if (actualS1Len - actualS2Len + fBaseParams.qStartIdx - fBaseParams.kvStartIdx == 0) {
                continue;
            } else {
                OP_LOGE(context_,
                        "INNER Pse sparse mode 8 is only supported actualSeqQlen %ld + qStartIdx %ld - actualSeqKvlen "
                        "%ld - kvStartIdx %ld ==0.",
                        actualS1Len, fBaseParams.qStartIdx, actualS2Len, fBaseParams.kvStartIdx);
                return ge::GRAPH_FAILED;
            }
        }
        if (actualS1Len != actualS2Len) {
            OP_LOGE(context_,
                    "INNER Pse sparse mode 8 is only supported when actualSeqQlen %ld equal actualSeqKvlen %ld.",
                    actualS1Len, actualS2Len);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::ProcessInnerPseInfo(size_t pseShapeDim)
{
    auto pseShape = context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::PSE_SHIFT));
    // sparse mode 7 不支持 pse inner
    OP_CHECK_IF(fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "INNER pse does not support sparse mode 7."),
                return ge::GRAPH_FAILED);
    // sparse mode 8 支持pse inner的条件
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        auto ret = ProcessPseSparseMode8();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    // 输入为[N1]或者[B, N1]
    if (pseShapeDim == PSE_DIM_NUM_1) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        OP_CHECK_IF(
            dim0 != fBaseParams.n1,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                        "FAG invalid pse shape %ld, should be same with n1 %ld", dim0, fBaseParams.n1),
            return ge::GRAPH_FAILED);
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_1_N2_G_SLOPE);
    } else if (pseShapeDim == PSE_DIM_NUM_2) {
        auto dim0 = pseShape->GetStorageShape().GetDim(DIM_0);
        auto dim1 = pseShape->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(dim0 != fBaseParams.b || dim1 != fBaseParams.n1,
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                                "FAG invalid pse shape {%ld, %ld}, should be same with b n1 {%ld, %ld}",
                                                dim0, dim1, fBaseParams.b, fBaseParams.n1),
                    return ge::GRAPH_FAILED);
        fBaseParams.pseShapeType = static_cast<uint32_t>(PseShapeType::PSE_B_N2_G_SLOPE);
    } else {
        OP_LOGE(context_, "pse inner mode, unsupported pse shape");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::SetPseLayout()
{
    if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NSS) ||
        fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNHS) ||
        fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_1NHS) ||
        fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BNSS)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pseS1S2);
    } else if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_B_N2_G_SLOPE)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pseSlopeBn);
    } else if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_1_N2_G_SLOPE)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pseSlopeN);
    } else if (fBaseParams.pseShapeType == static_cast<uint32_t>(PseShapeType::PSE_SHAPE_TYPE_BN1S)) {
        fBaseParams.pseLayoutType = static_cast<uint32_t>(PseLayoutType::pse1S2);
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::SetAttenMaskShapeType(
    const gert::StorageShape* attenMaskShape, size_t dimNum)
{
    if (dimNum == ATTEN_MASK_DIM_LENGTH_2) {
        fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2);
    } else if (dimNum == ATTEN_MASK_DIM_LENGTH_4) {
        auto dim0 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_0);
        auto dim1 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_1);
        if ((dim0 == fBaseParams.b) && (dim1 == fBaseParams.n2 * fBaseParams.g)) {
            fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBN2GS1S2);
        } else if ((dim0 == fBaseParams.b) && (dim1 == 1)) {
            fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKBS1S2);
        } else if ((dim0 == 1) && (dim1 == 1)) {
            fBaseParams.attenMaskShapeType = static_cast<uint32_t>(AttenMaskShapeType::ATTENMASKS1S2);
        } else {
            OP_LOGE("FAG attenMask", "dim value error, dim0 = %ld, dim1 = %ld", dim0, dim1);
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE("FAG attenMask", "dim num error, dimNum = %lu", dimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2David::SupportTrans2BS2N2GD()
{
    return (fBaseParams.sparseMode <= static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS)) && fBaseParams.isAllSame &&
           (fBaseParams.layoutType == INPUT_FORMAT_TND);
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::GetCommonS1S2OuterIndex(int64_t (*parseInfo)[ARRAY_LENGTH],
                                                                         int64_t gTail, int64_t& s1oIdx,
                                                                         int64_t& s2oIdx)
{
    int64_t preSize = 0;
    int64_t nextSize = 0;
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        if (gTail >= preSize) {
            nextSize = parseInfo[i][LENGTH_IDX];
            if (gTail < nextSize) {
                s2oIdx = i;
                s1oIdx = parseInfo[i][BEGIN_IDX] + gTail - preSize - 1;
                OP_LOGD("Sparse", " s1oIdx = %ld, s2oIdx = %ld, preSize = %ld, nextSize = %ld", s1oIdx, s2oIdx, preSize,
                        nextSize);
                break;
            }
            preSize = parseInfo[i][LENGTH_IDX];
        }
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::SetSparsePrefixBlockInterval(
    int64_t bIdx, int64_t nIdx, std::vector<std::vector<std::pair<int64_t, int64_t>>>& s1ValidIdx,
    int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM], uint32_t& coreNum, int64_t& tmepBlock)
{
    for (int64_t gIdx = 0; gIdx < fBaseParams.g; ++gIdx) {
        for (int64_t s2Idx = 0; s2Idx < fBaseParams.s2Outer; ++s2Idx) {
            tmepBlock += s1ValidIdx[bIdx][s2Idx].first;
            while (tmepBlock >= fBaseParams.blockFactor && coreNum < CORE_LIST_NUM - 1) {
                blockEnds[coreNum++] =
                    (((bIdx * fBaseParams.n2 + nIdx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2Idx) *
                        fBaseParams.s1Outer +
                    fBaseParams.s1Outer - (tmepBlock - fBaseParams.blockFactor);
                blockStarts[coreNum] = blockEnds[coreNum - 1];
                tmepBlock = tmepBlock - fBaseParams.blockFactor;
            }
        }
    }
    return;
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::CalValidUnpadBlockInfo(
    int64_t batchIdx, std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo)
{
    int64_t actualS1Len = fBaseParams.actualSeqQlen[batchIdx];
    int64_t actualS2Len = fBaseParams.actualSeqKvlen[batchIdx];
    // 对unpad场景的token值做二次校正
    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    int64_t actualCalcS1Token, actualCalcS2Token;
    CalcleActualToken(batchIdx, actualCalcS1Token, actualCalcS2Token);

    bool isPrefixCompress = fBaseParams.attenMaskOptional != EMPTY_TENSOR &&
                            fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::PREFIX_COMPRESS);

    // unpad 场景下s2Outer是按照最大的s2计算得到的
    for (int64_t j = 0; j < fBaseParams.s2Outer; j++) {
        if (fBaseParams.cvS2Inner * j >= actualS2Len) {
            calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = 0;
            calculatedBlockInfo[batchIdx][j][END_IDX] = 0;
        } else if (isPrefixCompress) {
            int64_t s1OuterTmp = (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
            int64_t s1oBegin = 0;
            if (fBaseParams.cvS2Inner * j >= fBaseParams.prefixN[batchIdx]) {
                int64_t s1CausalStart = fBaseParams.cvS2Inner * j - actualS2Len + actualS1Len;
                s1oBegin = s1CausalStart > 0 ? s1CausalStart / fBaseParams.s1CvInner : 0;
            }
            calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = std::min(s1oBegin, s1OuterTmp);
            calculatedBlockInfo[batchIdx][j][END_IDX] = s1OuterTmp;
        } else {
            int64_t leftIntersectionPoint = std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, 0L);
            if (leftIntersectionPoint > int64_t(actualS1Len)) {
                calculatedBlockInfo[batchIdx][j][BEGIN_IDX] =
                    (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
            } else {
                calculatedBlockInfo[batchIdx][j][BEGIN_IDX] = leftIntersectionPoint / fBaseParams.s1CvInner;
            }
            int64_t cvBlockTail = fBaseParams.cvS2Inner * (j + 1) > actualS2Len ?
                                      actualS2Len - fBaseParams.cvS2Inner * j :
                                      fBaseParams.cvS2Inner;
            calculatedBlockInfo[batchIdx][j][END_IDX] =
                int64_t(std::min(int64_t(actualS1Len),
                                 std::max(fBaseParams.cvS2Inner * j + cvBlockTail + actualCalcS1Token, 0L)) +
                        fBaseParams.s1CvInner - 1) /
                fBaseParams.s1CvInner;
        }

        int64_t tmpLength =
            calculatedBlockInfo[batchIdx][j][END_IDX] > calculatedBlockInfo[batchIdx][j][BEGIN_IDX] ?
                calculatedBlockInfo[batchIdx][j][END_IDX] - calculatedBlockInfo[batchIdx][j][BEGIN_IDX] :
                0;
        if (j == 0) {
            calculatedBlockInfo[batchIdx][j][SUM_S1S2] = tmpLength;
        } else {
            calculatedBlockInfo[batchIdx][j][SUM_S1S2] = calculatedBlockInfo[batchIdx][j - 1][SUM_S1S2] + tmpLength;
        }

        calculatedBlockInfo[batchIdx][j][SUM_ALL] = 0; // 初始化清零

        OP_LOGD("FillBlockInfo", " s2Outer idx = %ld: Begin = %ld, End = %ld, Sum_S1S2 = %ld", j,
                calculatedBlockInfo[batchIdx][j][BEGIN_IDX], calculatedBlockInfo[batchIdx][j][END_IDX],
                calculatedBlockInfo[batchIdx][j][SUM_S1S2]);
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2David::GetUnpadS1S2OuterIndex(
    int64_t& s1oIdx, int64_t& s2oIdx, int64_t gTail, int64_t bIdx,
    std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo)
{
    int64_t s1oTail = 0;
    for (int64_t i = 0; i < fBaseParams.s2Outer; i++) {
        if (calculatedBlockInfo[bIdx][i][SUM_S1S2] > gTail) {
            s2oIdx = i;
            s1oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2];
            s1oIdx = calculatedBlockInfo[bIdx][i][BEGIN_IDX] + s1oTail;
            break;
        }
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2David::CheckUnpadTokensInfo()
{
    // 7  8
    if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CASUAL_BAND) ||
        fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND_LEFT_UP_CASUAL)) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[fBaseParams.bandIdx];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[fBaseParams.bandIdx];
        if (-fBaseParams.s1Token > actualS1Len || -fBaseParams.s2Token > actualS2Len ||
            (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
            OP_LOGE("ProcessTokensInfo",
                    "pre_token and next_token is invalid in the unpad scene, got b %u, s1 %ld, s2 %ld,  pre_token %ld, "
                    "next_token %ld, sparse_mode %u",
                    fBaseParams.bandIdx, actualS1Len, actualS2Len, fBaseParams.s1Token, fBaseParams.s2Token,
                    fBaseParams.sparseMode);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 0  4
    for (int64_t i = 0; i < fBaseParams.b; i++) {
        int64_t actualS1Len = fBaseParams.actualSeqQlen[i];
        int64_t actualS2Len = fBaseParams.actualSeqKvlen[i];
        if (actualS1Len == 0 || actualS2Len == 0) {
            continue;
        }
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::NO_MASK)) {
            if (-fBaseParams.s1Token > actualS2Len || -fBaseParams.s2Token > actualS1Len ||
                (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                OP_LOGE("ProcessTokensInfo",
                        "pre_token and next_token is invalid in the unpad scene, got b %ld, s1 %ld, s2 %ld,  "
                        "pre_token %ld, "
                        "next_token %ld, sparse_mode %u",
                        i, actualS1Len, actualS2Len, fBaseParams.s1Token, fBaseParams.s2Token, fBaseParams.sparseMode);
                return ge::GRAPH_FAILED;
            }
        }
        if (fBaseParams.sparseMode == static_cast<uint32_t>(SparseMode::BAND)) {
            if (-fBaseParams.s1Token > actualS1Len || -fBaseParams.s2Token > actualS2Len ||
                (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
                OP_LOGE("ProcessTokensInfo",
                        "pre_token and next_token is invalid in the unpad scene, got b %ld, s1 %ld, s2 %ld,  "
                        "pre_token %ld, "
                        "next_token %ld, sparse_mode %u",
                        i, actualS1Len, actualS2Len, fBaseParams.s1Token, fBaseParams.s2Token, fBaseParams.sparseMode);
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScoreGrad, FlashAttentionScoreGradTilingUs1s2Bs2David,
                                   std::vector<int32_t>({(int32_t)NpuArch::DAV_9201, (int32_t)NpuArch::DAV_9202}), 800);
} // namespace fag
} // namespace optiling
