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
 * \file mla_prolog_tiling_check.cpp
 * \brief
 */

#include "mla_prolog_tiling_check.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

template <typename E>
std::string ElemToString(const E &elem)
{
    return std::to_string(elem);
}

std::string FormatToString(const ge::Format format)
{
    return std::string(ge::GetFormatName(format));
}

template <typename C, typename Func = std::string (*)(const typename C::value_type &)>
std::string ConvertContainerToString(const C &container, Func func = ElemToString<typename C::value_type>)
{
    if (container.empty() || func == nullptr) {
        return "[]";
    }
    std::stringstream ss;
    ss << "[";
    bool isFirst = true;
    for (const auto &elem : container) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << func(elem);
        isFirst = false;
    }
    ss << "]";
    return ss.str();
}

// =================================全量参数校验=================================
ge::graphStatus MlaPrologTilingCheck::CheckDims() const
{
    OP_CHECK_IF(baseShapeInfo_.bSize > MAX_B_SIZE,
        OP_LOGE(context_.opName, "B should not be greater than %u, got %u.",
            MAX_B_SIZE, baseShapeInfo_.bSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.s1Size > MAX_S1_SIZE,
        OP_LOGE(context_.opName, "S should not be greater than %u, got %u.",
            MAX_S1_SIZE, baseShapeInfo_.s1Size),
        return ge::GRAPH_FAILED);
    const std::set<uint32_t> supportedHeSize {7168U, 7680U, 6144U};
    OP_CHECK_IF(supportedHeSize.find(baseShapeInfo_.heSize) == supportedHeSize.end(),
        OP_LOGE(context_.opName, "He allows only %s, got %u.",
            ConvertContainerToString(supportedHeSize).c_str(), baseShapeInfo_.heSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.hcqSize != HCQ_SIZE,
        OP_LOGE(context_.opName, "Hcq allows only %u, got %u.",
            HCQ_SIZE, baseShapeInfo_.hcqSize),
        return ge::GRAPH_FAILED);
    const std::set<uint32_t> supportedNSize {1, 2, 4, 8, 16, 32, 64, 128};
    OP_CHECK_IF((supportedNSize.find(baseShapeInfo_.nSize) == supportedNSize.end()),
        OP_LOGE(context_.opName, "N allows only %s, but got %u.",
            ConvertContainerToString(supportedNSize).c_str(), baseShapeInfo_.nSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.hckvSize != HCKV_SIZE,
        OP_LOGE(context_.opName, "Hckv allows only %u, got %u.",
            HCKV_SIZE, baseShapeInfo_.hckvSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.dSize != D_SIZE,
        OP_LOGE(context_.opName, "D allows only %u, got %u.",
            D_SIZE, baseShapeInfo_.dSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.drSize != DR_SIZE,
        OP_LOGE(context_.opName, "Dr allows only %u, got %u.",
            DR_SIZE, baseShapeInfo_.drSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.nkvSize != NKV_SIZE,
        OP_LOGE(context_.opName, "Nkv allows only %u, got %u.",
            NKV_SIZE, baseShapeInfo_.nkvSize),
        return ge::GRAPH_FAILED);
    const std::set<uint32_t> supportedBlockSize {16, 128};
    OP_CHECK_IF((supportedBlockSize.find(baseShapeInfo_.blockSize) == supportedBlockSize.end()),
        OP_LOGE(context_.opName, "BlockSize allows only %s, got %u.",
            ConvertContainerToString(supportedBlockSize).c_str(), baseShapeInfo_.blockSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(baseShapeInfo_.tSize > MAX_T_SIZE,
        OP_LOGE(context_.opName, "T should not be greater than %u, got %u.",
            MAX_T_SIZE, baseShapeInfo_.tSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MlaPrologTilingCheck::GenExpectedParamInfo()
{
    FillCommonParamInfo();
    FillScenarioParamInfo();
}

void MlaPrologTilingCheck::FillCommonParamInfo()
{
    FillRequiredParamShapeWithDims();

    if (context_.weightDq.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_DQ_NAME].dimNum = MLA_PROLOG_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightDq.desc->GetDataType());
        expectedParamInfo_[WEIGHT_DQ_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.hcqSize) / weightAxisSize,
                static_cast<int64_t>(baseShapeInfo_.heSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }
    if (context_.weightUqQr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_UQ_QR_NAME].dimNum = MLA_PROLOG_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightUqQr.desc->GetDataType());
        expectedParamInfo_[WEIGHT_UQ_QR_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.headSizeUqQr) / weightAxisSize,
                static_cast<int64_t>(baseShapeInfo_.hcqSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }
    if (context_.weightDkvKr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
        expectedParamInfo_[WEIGHT_DKV_KR_NAME].dimNum = MLA_PROLOG_DIM_NUM_4;
        int64_t weightAxisSize = 32L / ge::GetSizeByDataType(context_.weightDkvKr.desc->GetDataType());
        expectedParamInfo_[WEIGHT_DKV_KR_NAME].shape =
            std::vector<int64_t>{static_cast<int64_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) / weightAxisSize,
                static_cast<int64_t>(baseShapeInfo_.heSize) / NZ_H0_SIZE, NZ_H0_SIZE, weightAxisSize};
    }

    expectedParamInfo_[WEIGHT_DQ_NAME].format = ge::FORMAT_FRACTAL_NZ;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].format = ge::FORMAT_FRACTAL_NZ;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].format = ge::FORMAT_FRACTAL_NZ;
}

void MlaPrologTilingCheck::FillRequiredParamShapeWithDims()
{
    if (scenarioInfo_.batchSeqFusedFlag_) {
        expectedParamInfo_.emplace(TOKEN_X_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.heSize});
        expectedParamInfo_.emplace(ROPE_SIN_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.drSize});
        expectedParamInfo_.emplace(ROPE_COS_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.drSize});
        expectedParamInfo_.emplace(CACHE_INDEX_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize});

        expectedParamInfo_.emplace(QUERY_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, baseShapeInfo_.hckvSize});
        expectedParamInfo_.emplace(QUERY_ROPE_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, baseShapeInfo_.drSize});
    } else {
        expectedParamInfo_.emplace(TOKEN_X_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.heSize});
        expectedParamInfo_.emplace(ROPE_SIN_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.drSize});
        expectedParamInfo_.emplace(ROPE_COS_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.drSize});
        expectedParamInfo_.emplace(CACHE_INDEX_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size});

        expectedParamInfo_.emplace(QUERY_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.nSize, baseShapeInfo_.hckvSize});
        expectedParamInfo_.emplace(QUERY_ROPE_NAME, std::vector<uint32_t>{baseShapeInfo_.bSize, baseShapeInfo_.s1Size, baseShapeInfo_.nSize, baseShapeInfo_.drSize});
    }
    expectedParamInfo_.emplace(WEIGHT_DQ_NAME, std::vector<uint32_t>{baseShapeInfo_.heSize, baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(WEIGHT_UQ_QR_NAME, std::vector<uint32_t>{baseShapeInfo_.hcqSize, baseShapeInfo_.headSizeUqQr});
    expectedParamInfo_.emplace(WEIGHT_UK_NAME, std::vector<uint32_t>{baseShapeInfo_.nSize, baseShapeInfo_.dSize,
        baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(WEIGHT_DKV_KR_NAME, std::vector<uint32_t>{baseShapeInfo_.heSize,
        baseShapeInfo_.hckvSize + baseShapeInfo_.drSize});
    expectedParamInfo_.emplace(RMSNORM_GAMMA_CQ_NAME, std::vector<uint32_t>{baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(RMSNORM_GAMMA_CKV_NAME, std::vector<uint32_t>{baseShapeInfo_.hckvSize});

    expectedParamInfo_.emplace(KV_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.blockNum, baseShapeInfo_.blockSize,
        baseShapeInfo_.nkvSize, baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(KR_CACHE_NAME, std::vector<uint32_t>{baseShapeInfo_.blockNum, baseShapeInfo_.blockSize,
        baseShapeInfo_.nkvSize, baseShapeInfo_.drSize});
    expectedParamInfo_.emplace(KV_CACHE_OUT_NAME, expectedParamInfo_[KV_CACHE_NAME]);
    expectedParamInfo_.emplace(KR_CACHE_OUT_NAME, expectedParamInfo_[KR_CACHE_NAME]);
}

void MlaPrologTilingCheck::FillScenarioParamInfo()
{
    switch (scenarioInfo_.quantMode_) {
        case QUANT_MODE::NO_QUANT:
            FillNonQuantParamInfo();
            break;
        case QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT:
            FillPartialQuantParamInfo();
            break;
        case QUANT_MODE::PARTIAL_QUANT_KV_QUANT:
            FillPartialKVQuantParamInfo();
            break;
        case QUANT_MODE::FULL_QUANT_KV_NO_QUANT:
            FillFullQuantParamInfo();
            break;
        case QUANT_MODE::FULL_QUANT_KV_QUANT:
            FillFullKVQuantParamInfo();
            break;
        default:
            break;
    }
}

void MlaPrologTilingCheck::FillNonQuantParamInfo()
{
    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_UK_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[RMSNORM_GAMMA_CQ_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[RMSNORM_GAMMA_CKV_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[ROPE_SIN_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[ROPE_COS_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[CACHE_INDEX_NAME].dtype = ge::DT_INT64;
    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KR_CACHE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[QUERY_ROPE_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_BF16;
    expectedParamInfo_[KR_CACHE_OUT_NAME].dtype = ge::DT_BF16;

    if (!scenarioInfo_.isV1Flag_) {
        // 仅校验dequantScaleQNope有传入
        expectedParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, context_.dequantScaleQNope);
        expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].isValid = true;
    }
}

void MlaPrologTilingCheck::FillPartialQuantParamInfo()
{
    FillNonQuantParamInfo();

    expectedParamInfo_.emplace(DEQUANT_SCALE_W_UQ_QR_NAME, std::vector<uint32_t>{1, baseShapeInfo_.headSizeUqQr});
    expectedParamInfo_.emplace(SMOOTH_SCALES_CQ_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hcqSize});

    expectedParamInfo_[WEIGHT_UQ_QR_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[CACHE_INDEX_NAME].dtype = ge::DT_INT64;

    expectedParamInfo_[DEQUANT_SCALE_W_UQ_QR_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].dtype = ge::DT_FLOAT;

    expectedParamInfo_[SMOOTH_SCALES_CQ_NAME].isValid = (context_.smoothScalesCq.desc != nullptr);
}

void MlaPrologTilingCheck::FillPartialKVQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hckvSize});
    expectedParamInfo_.emplace(QUANT_SCALE_CKR_NAME, std::vector<uint32_t>{1, baseShapeInfo_.drSize});

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KR_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KR_CACHE_OUT_NAME].dtype = ge::DT_INT8;

    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[QUANT_SCALE_CKR_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillFullQuantParamInfo()
{
    FillPartialQuantParamInfo();

    expectedParamInfo_.emplace(DEQUANT_SCALE_X_NAME, std::vector<uint32_t>{baseShapeInfo_.tSize, 1});
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hcqSize});
    expectedParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME,
        std::vector<uint32_t>{1, baseShapeInfo_.hckvSize + baseShapeInfo_.drSize});

    expectedParamInfo_[TOKEN_X_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[WEIGHT_DQ_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[WEIGHT_DKV_KR_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[DEQUANT_SCALE_X_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[DEQUANT_SCALE_W_DQ_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[DEQUANT_SCALE_W_DKV_KR_NAME].dtype = ge::DT_FLOAT;
}

void MlaPrologTilingCheck::FillFullKVQuantParamInfo()
{
    FillFullQuantParamInfo();

    expectedParamInfo_.emplace(QUANT_SCALE_CKV_NAME, std::vector<uint32_t>{1, baseShapeInfo_.hckvSize});
    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME] =
        ParamInfo(std::vector<uint32_t>{baseShapeInfo_.tSize, baseShapeInfo_.nSize, 1});

    expectedParamInfo_[KV_CACHE_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[KV_CACHE_OUT_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[QUANT_SCALE_CKV_NAME].dtype = ge::DT_FLOAT;
    expectedParamInfo_[QUERY_NAME].dtype = ge::DT_INT8;
    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].dtype = ge::DT_FLOAT;

    expectedParamInfo_[DEQUANT_SCALE_Q_NOPE_NAME].isValid = true;
}

void MlaPrologTilingCheck::GenActualParamInfo()
{
    actualParamInfo_.emplace(TOKEN_X_NAME, context_.tokenX);
    actualParamInfo_.emplace(WEIGHT_DQ_NAME, context_.weightDq);
    actualParamInfo_.emplace(WEIGHT_UQ_QR_NAME, context_.weightUqQr);
    actualParamInfo_.emplace(WEIGHT_UK_NAME, context_.weightUk);
    actualParamInfo_.emplace(WEIGHT_DKV_KR_NAME, context_.weightDkvKr);
    actualParamInfo_.emplace(RMSNORM_GAMMA_CQ_NAME, context_.rmsnormGammaCq);
    actualParamInfo_.emplace(RMSNORM_GAMMA_CKV_NAME, context_.rmsnormGammaCkv);
    actualParamInfo_.emplace(ROPE_SIN_NAME, context_.ropeSin);
    actualParamInfo_.emplace(ROPE_COS_NAME, context_.ropeCos);
    actualParamInfo_.emplace(CACHE_INDEX_NAME, context_.cacheIndex);
    actualParamInfo_.emplace(KV_CACHE_NAME, context_.kvCache);
    actualParamInfo_.emplace(KR_CACHE_NAME, context_.krCache);
    actualParamInfo_.emplace(DEQUANT_SCALE_X_NAME, context_.dequantScaleX);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_DQ_NAME, context_.dequantScaleWDq);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_UQ_QR_NAME, context_.dequantScaleWUqQr);
    actualParamInfo_.emplace(DEQUANT_SCALE_W_DKV_KR_NAME, context_.dequantScaleWDkvKr);
    actualParamInfo_.emplace(QUANT_SCALE_CKV_NAME, context_.quantScaleCkv);
    actualParamInfo_.emplace(QUANT_SCALE_CKR_NAME, context_.quantScaleCkr);
    actualParamInfo_.emplace(SMOOTH_SCALES_CQ_NAME, context_.smoothScalesCq);
    actualParamInfo_.emplace(QUERY_NAME, context_.query);
    actualParamInfo_.emplace(QUERY_ROPE_NAME, context_.queryRope);
    actualParamInfo_.emplace(KV_CACHE_OUT_NAME, context_.kvCacheOut);
    actualParamInfo_.emplace(KR_CACHE_OUT_NAME, context_.krCacheOut);
    actualParamInfo_.emplace(DEQUANT_SCALE_Q_NOPE_NAME, context_.dequantScaleQNope);
}

ge::graphStatus MlaPrologTilingCheck::CheckParamByScenario()
{
    GenExpectedParamInfo();
    GenActualParamInfo();
    ge::graphStatus isCorrect {ge::GRAPH_SUCCESS};
    for (const auto &it : actualParamInfo_) {
        const auto &expectedParam {expectedParamInfo_[it.first]};
        if (__builtin_expect((expectedParam != it.second), 0)) {
            isCorrect = ge::GRAPH_FAILED;
            if (expectedParam.isValid != it.second.isValid) {
                OP_LOGE(context_.opName, "%s expected %s, got %s.",
                    it.first.c_str(),
                    expectedParam.isValid ? "not null" : "null",
                    it.second.isValid ? "not null" : "null");
                continue;
            }
            if (expectedParam.dtype != it.second.dtype) {
                OP_LOGE(context_.opName, "%s expected dtype %s, but got %s.",
                    it.first.c_str(),
                    TypeUtils::DataTypeToSerialString(expectedParam.dtype).c_str(),
                    TypeUtils::DataTypeToSerialString(it.second.dtype).c_str());
            }
            if (expectedParam.format != it.second.format) {
                OP_LOGE(context_.opName, "%s expected format %s, but got %s.",
                    it.first.c_str(),
                    ge::GetFormatName(expectedParam.format),
                    ge::GetFormatName(it.second.format));
            }
            if (expectedParam.shape != it.second.shape) {
                OP_LOGE(context_.opName, "%s expected shape %s, but got %s.",
                    it.first.c_str(),
                    ConvertContainerToString(expectedParam.shape).c_str(),
                    ConvertContainerToString(it.second.shape).c_str());
            }
        }
    }
    return isCorrect;
}
// =================================全量参数校验=================================

// ==================================单参数校验==================================
bool MlaPrologTilingCheck::IsSingleParamValid(const RequiredParaInfo &param, const std::string &paramName,
                                              const std::set<ge::DataType> &expectedDtype,
                                              const std::set<ge::Format> &expectedFormat,
                                              const std::set<size_t> &expectedDimNum) const
{
    OP_CHECK_IF((param.shape == nullptr) || (param.desc == nullptr),
        OP_LOGE(context_.opName, "%s should not be null.", paramName.c_str()), return false);

    ge::DataType dtype = param.desc->GetDataType();
    OP_CHECK_IF((expectedDtype.find(dtype) == expectedDtype.end()),
        OP_LOGE(context_.opName, "%s datatype only supports %s, but got %s.",
            paramName.c_str(),
            ConvertContainerToString(expectedDtype, TypeUtils::DataTypeToSerialString).c_str(),
            TypeUtils::DataTypeToSerialString(dtype).c_str()),
        return false);

    ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(param.desc->GetStorageFormat()));
    OP_CHECK_IF((expectedFormat.find(format) == expectedFormat.end()),
        OP_LOGE(context_.opName, "%s format only supports %s, but got %s.",
            paramName.c_str(),
            ConvertContainerToString(expectedFormat, FormatToString).c_str(),
            ge::GetFormatName(format)),
        return false);

    size_t dimNum = param.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((expectedDimNum.find(dimNum) == expectedDimNum.end()),
        OP_LOGE(context_.opName, "%s dim num supports only %s, but got %zu.",
            paramName.c_str(), ConvertContainerToString(expectedDimNum).c_str(), dimNum),
        return false);
    return true;
}

ge::graphStatus MlaPrologTilingCheck::CheckSingleRequiredParam() const
{
    if (!CheckTokenX() || !CheckWDq() || !CheckWUqQr() || !CheckWUk() || !CheckWDkvKr() || !CheckRmsnormGammaCq() ||
        !CheckRmsnormGammaCkv() || !CheckRopeSin() || !CheckRopeCos() || !CheckCacheIndex() || !CheckKvCache() ||
        !CheckKrCache()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool MlaPrologTilingCheck::CheckTokenX() const
{
    return IsSingleParamValid(context_.tokenX, TOKEN_X_NAME, {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologTilingCheck::CheckWDq() const
{
    return IsSingleParamValid(context_.weightDq, WEIGHT_DQ_NAME, {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_FRACTAL_NZ},
                              {2, 4});
}

bool MlaPrologTilingCheck::CheckWUqQr() const
{
    return IsSingleParamValid(context_.weightUqQr, WEIGHT_UQ_QR_NAME, {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_FRACTAL_NZ},
                              {2, 4});
}

bool MlaPrologTilingCheck::CheckWUk() const
{
    return IsSingleParamValid(context_.weightUk, WEIGHT_UK_NAME, {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {3});
}

bool MlaPrologTilingCheck::CheckWDkvKr() const
{
    return IsSingleParamValid(context_.weightDkvKr, WEIGHT_DKV_KR_NAME, {ge::DT_BF16, ge::DT_INT8},
                              {ge::FORMAT_FRACTAL_NZ}, {2, 4});
}

bool MlaPrologTilingCheck::CheckRmsnormGammaCq() const
{
    return IsSingleParamValid(context_.rmsnormGammaCq, RMSNORM_GAMMA_CQ_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW},
                              {1});
}

bool MlaPrologTilingCheck::CheckRmsnormGammaCkv() const
{
    return IsSingleParamValid(context_.rmsnormGammaCkv, RMSNORM_GAMMA_CKV_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1});
}

bool MlaPrologTilingCheck::CheckRopeSin() const
{
    return IsSingleParamValid(context_.ropeSin, ROPE_SIN_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologTilingCheck::CheckRopeCos() const
{
    return IsSingleParamValid(context_.ropeCos, ROPE_COS_NAME, {ge::DT_BF16}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {2, 3});
}

bool MlaPrologTilingCheck::CheckCacheIndex() const
{
    return IsSingleParamValid(context_.cacheIndex, CACHE_INDEX_NAME, {ge::DT_INT64}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {1, 2});
}

bool MlaPrologTilingCheck::CheckKvCache() const
{
    return IsSingleParamValid(context_.kvCache, KV_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {4});
}

bool MlaPrologTilingCheck::CheckKrCache() const
{
    return IsSingleParamValid(context_.krCache, KR_CACHE_NAME, {ge::DT_BF16, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_NCHW}, {4});
}

ge::graphStatus MlaPrologTilingCheck::CheckCacheMode() const
{
    if ((std::strcmp(context_.cacheMode, CACHE_MODE_PA_BSND) == 0) ||
        (std::strcmp(context_.cacheMode, CACHE_MODE_PA_NZ) == 0)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context_.opName, "Only support cacheMode (PA_BSND, PA_NZ), actually is %s.", context_.cacheMode);
    return ge::GRAPH_FAILED;
}
// ==================================单参数校验==================================
}  // namespace optiling