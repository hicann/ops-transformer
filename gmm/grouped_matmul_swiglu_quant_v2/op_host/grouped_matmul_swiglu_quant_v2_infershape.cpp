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
 * \file grouped_matmul_swiglu_quant_proto.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "graph/utils/type_utils.h"
#include "common/include/op_host/op_const_def_transformer.h"

#include <sstream>

using namespace ge;
namespace ops {
const int64_t X_INDEX = 0;
const int64_t WEIGHT_INDEX = 3;
const int64_t WEIGHTSCALE_DIM_PERTOKEN = 2;
const int64_t WEIGHTSCALE_INDEX = 4;
const int64_t M_DIM_INDEX = 0;
const int64_t DIM_LEN = 2;
const int64_t SPLIT_RATIO = 2;
const int64_t OUT_DIM_LEN = 3;
const int64_t N_SPLIT_RATIO = 128;
const int64_t MX_MULTI_WEIGHT_SCALE_DIM = 3;
const int64_t MX_SINGLE_WEIGHT_SCALE_DIM = 4;
constexpr size_t GMMSQ_INDEX_ATTR_QUANT_DTYPE = 3UL;
constexpr size_t GMMSQ_INDEX_ATTR_QUANT_MODE = 2UL;
constexpr size_t GMMSQ_INDEX_ATTR_TRANSPOSE_WEIGHT = 4UL;
constexpr size_t QUANT_MODE_MX_TYPE = 2;
constexpr size_t QUANT_MODE_PERTOKEN_TYPE = 0;
constexpr size_t QUANT_MODE_HIF4_TYPE = 3;
constexpr const char* GROUPED_MATMUL_SWIGLU_QUANT_V2_OP_TYPE = "GroupedMatmulSwigluQuantV2";

template <typename... Args>
std::string ListToString(const Args&... args)
{
    std::ostringstream oss;
    bool isFirst = true;
    using Expander = int[];
    (void)Expander{0, ((void)(oss << (isFirst ? "" : ", ") << args, isFirst = false), 0)...};
    return oss.str();
}
constexpr int64_t DYNAMIC_GRAPH_FIRST_INFERSHAPE_DIM_VALUE = -1;

static std::set<std::string> GmmDavidSupportSoc = {"Ascend950", "Ascend960DT", "Ascend960PR"};
static const std::unordered_set<ge::DataType> DavidSupportedInputDtypes = {
    ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT4_E2M1,
    ge::DataType::DT_FLOAT4_E1M2, ge::DataType::DT_INT8,          ge::DataType::DT_HIFLOAT8};
bool isSupportedInputDtypeForDavid(ge::DataType dtype)
{
    return DavidSupportedInputDtypes.find(dtype) != DavidSupportedInputDtypes.end();
}

static ge::graphStatus InferShape4GroupedMatmulSwigluQuantV2(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Shape* xShape = context->GetDynamicInputShape(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* weightScaleShape = context->GetDynamicInputShape(WEIGHTSCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* quantModePtr = attrs->GetInt(GMMSQ_INDEX_ATTR_QUANT_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    const bool* transposeWeightPtr = attrs->GetBool(GMMSQ_INDEX_ATTR_TRANSPOSE_WEIGHT);
    const bool transposeWeight = (transposeWeightPtr != nullptr ? *transposeWeightPtr : false);
    const auto* xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
    const auto* weightDesc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    const bool isHif4Shape = *quantModePtr == QUANT_MODE_HIF4_TYPE && xDesc != nullptr && weightDesc != nullptr &&
                             xDesc->GetDataType() == Ops::Transformer::DT_HIFLOAT4 &&
                             weightDesc->GetDataType() == Ops::Transformer::DT_HIFLOAT4;

    int64_t m = xShape->GetDim(M_DIM_INDEX);
    int64_t weightScaleDimNum = weightScaleShape->GetDimNum();
    int64_t nDimIndex = weightScaleDimNum - 1;
    // A8W4/A4W4 per-group weightScale is also 3D, so per-token mode must not be identified as MX by rank.
    // Preserve the original shape inference for invalid non-per-token modes; parameter validation rejects them later.
    bool isMxShape = *quantModePtr != QUANT_MODE_PERTOKEN_TYPE && (*quantModePtr == QUANT_MODE_MX_TYPE || isHif4Shape ||
                                                                   weightScaleDimNum == MX_SINGLE_WEIGHT_SCALE_DIM ||
                                                                   weightScaleDimNum == MX_MULTI_WEIGHT_SCALE_DIM);
    if (isMxShape) {
        if (weightScaleDimNum == MX_SINGLE_WEIGHT_SCALE_DIM) {
            nDimIndex = transposeWeight ? weightScaleDimNum - OUT_DIM_LEN : weightScaleDimNum - DIM_LEN;
        } else if (weightScaleDimNum == MX_MULTI_WEIGHT_SCALE_DIM) {
            nDimIndex = transposeWeight ? 0 : weightScaleDimNum - DIM_LEN;
        }
    }

    auto outScaleShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outScaleShape);
    if (isMxShape) {
        int64_t dimValue = static_cast<int64_t>(weightScaleShape->GetDim(nDimIndex));
        int64_t n = 0;
        if (dimValue == DYNAMIC_GRAPH_FIRST_INFERSHAPE_DIM_VALUE) {
            n = dimValue;
        } else {
            n = static_cast<int64_t>(Ops::Base::CeilDiv(weightScaleShape->GetDim(nDimIndex), N_SPLIT_RATIO));
        }
        outScaleShape->SetDimNum(OUT_DIM_LEN);
        outScaleShape->SetDim(0, m);
        outScaleShape->SetDim(1, n);
        // 2：scale shape 的末维索引，表示每组 scale 数量。
        outScaleShape->SetDim(2, isHif4Shape ? 1 : SPLIT_RATIO);
    } else {
        outScaleShape->SetDimNum(1);
        outScaleShape->SetDim(0, m);
    }

    int64_t dimValue = static_cast<int64_t>(weightScaleShape->GetDim(nDimIndex));
    int64_t n = 0;
    if (dimValue == DYNAMIC_GRAPH_FIRST_INFERSHAPE_DIM_VALUE) {
        n = dimValue;
    } else {
        n = static_cast<int64_t>(weightScaleShape->GetDim(nDimIndex) / SPLIT_RATIO);
        if (weightScaleShape->GetDimNum() == WEIGHTSCALE_DIM_PERTOKEN) {
            n = static_cast<int64_t>(weightScaleShape->GetDim(1) / SPLIT_RATIO);
        }
    }
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(DIM_LEN);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmulSwigluQuantV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outDtype = attrs->GetInt(GMMSQ_INDEX_ATTR_QUANT_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDtype);
    const int64_t* quantMode = attrs->GetInt(GMMSQ_INDEX_ATTR_QUANT_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantMode);

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        auto xDtype = context->GetInputDataType(X_INDEX);
        auto weightDtype = context->GetDynamicInputDataType(WEIGHT_INDEX, 0);
        const bool is9201 = platformInfo.str_info.short_soc_version == "Ascend960DT" ||
                            platformInfo.str_info.short_soc_version == "Ascend960PR";
        const bool xSupported =
            isSupportedInputDtypeForDavid(xDtype) || (is9201 && xDtype == Ops::Transformer::DT_HIFLOAT4);
        const bool weightSupported =
            isSupportedInputDtypeForDavid(weightDtype) || (is9201 && weightDtype == Ops::Transformer::DT_HIFLOAT4);
        OP_CHECK_IF(!xSupported || !weightSupported,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        GROUPED_MATMUL_SWIGLU_QUANT_V2_OP_TYPE, "x, weight",
                        ListToString(ge::TypeUtils::DataTypeToSerialString(xDtype),
                                     ge::TypeUtils::DataTypeToSerialString(weightDtype)),
                        "On this platform, the dtypes of x and weight must be within the range FLOAT8_E4M3, "
                        "FLOAT8_E5M2, FLOAT4_E2M1, INT8 or HIFLOAT8"),
                    return GRAPH_FAILED);

        OP_CHECK_IF(
            *quantMode != QUANT_MODE_MX_TYPE && *quantMode != QUANT_MODE_PERTOKEN_TYPE &&
                !(is9201 && *quantMode == QUANT_MODE_HIF4_TYPE),
            OP_LOGE_FOR_INVALID_VALUE(GROUPED_MATMUL_SWIGLU_QUANT_V2_OP_TYPE, "quant_mode", std::to_string(*quantMode),
                                      is9201 ? "0(Pertoken), 2(MX) or 3(HIF4)" : "0(Pertoken) or 2(MX)"),
            return GRAPH_FAILED);
        OP_CHECK_IF(*quantMode == QUANT_MODE_HIF4_TYPE &&
                        (xDtype != Ops::Transformer::DT_HIFLOAT4 || weightDtype != Ops::Transformer::DT_HIFLOAT4),
                    OP_LOGE(context->GetNodeName(), "HIF4 mode requires HIFLOAT4 x and weight."), return GRAPH_FAILED);
    }
    auto weightScaleDtype = context->GetDynamicInputDataType(WEIGHTSCALE_INDEX, 0);
    if (*quantMode == QUANT_MODE_MX_TYPE) {
        const bool isMxA8Hif4 = context->GetInputDataType(X_INDEX) == ge::DT_FLOAT8_E4M3FN &&
                                context->GetDynamicInputDataType(WEIGHT_INDEX, 0) == Ops::Transformer::DT_HIFLOAT4;
        if ((isMxA8Hif4 && weightScaleDtype == Ops::Transformer::DT_HIFLOAT4_SCALE) ||
            (!isMxA8Hif4 && weightScaleDtype == ge::DT_FLOAT8_E8M0)) {
            context->SetOutputDataType(1, DataType::DT_FLOAT8_E8M0);
        } else {
            OP_LOGE(context->GetNodeName(),
                    "In mx quant mode, weightScale must be FLOAT8_E8M0 (or HIFLOAT4_SCALE for MXA8HIF4), but got %s.",
                    ge::TypeUtils::DataTypeToSerialString(weightScaleDtype).c_str());
            return GRAPH_FAILED;
        }
    } else if (*quantMode == QUANT_MODE_HIF4_TYPE) {
        if (weightScaleDtype != Ops::Transformer::DT_HIFLOAT4_SCALE) {
            OP_LOGE(context->GetNodeName(), "HIF4 mode requires HIFLOAT4_SCALE weightScale.");
            return GRAPH_FAILED;
        }
        context->SetOutputDataType(1, Ops::Transformer::DT_HIFLOAT4_SCALE);
    } else if (*quantMode == QUANT_MODE_PERTOKEN_TYPE) {
        context->SetOutputDataType(1, DataType::DT_FLOAT);
    }
    context->SetOutputDataType(0, static_cast<ge::DataType>(*outDtype));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSwigluQuantV2)
    .InferShape(InferShape4GroupedMatmulSwigluQuantV2)
    .InferDataType(InferDataType4GroupedMatmulSwigluQuantV2);
} // namespace ops
