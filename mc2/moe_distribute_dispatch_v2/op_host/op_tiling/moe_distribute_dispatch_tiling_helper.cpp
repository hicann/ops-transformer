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
 * \file moe_distribute_dispatch_tiling_helper.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_helper.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_log_compat.h"

using namespace ge;

namespace optiling {
inline bool MoeDistributeDispatchTilingHelper::CheckInputTensorDim(const gert::TilingContext *context,
                                                                   const char *nodeName, const bool isScales,
                                                                   const uint32_t quantMode)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShape"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "xShape", std::to_string(xStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of xShape must be 2D."),
                    return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdShape"),
                    return false);
    OP_TILING_CHECK(
        expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expertIdShape", std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expertIdShape must be 2D."),
        return false);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));
    // 如果scales不为空进行shape维度检查
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesShape"),
                        return false);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) {
            // the cond is compatible with A2/A3 because static quant is only supported on A5
            if (scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
                std::string dimStr = std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", dimStr.c_str(), "2D");
                return false;
            }
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
        } else {
            size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
            if ((scalesDimNum != ONE_DIM) && (scalesDimNum != TWO_DIMS)) {
                std::string dimStr = std::to_string(scalesDimNum) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", dimStr.c_str(), "1D or 2D");
                return false;
            }
            // additional check for hif8 quant
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
            if ((expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDimNum != ONE_DIM)) {
                std::string dimStr = std::to_string(scalesDimNum) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", dimStr.c_str(), "1D");
                return false;
            }
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            if (scalesStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
                OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
            }
        }
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckDynamicScalesDim(const gert::TilingContext *context,
                                                                     const char *nodeName, const uint32_t quantMode)
{
    const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesShape"),
                    return false);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT))) {
        // quantMode 2: 1dim, the same in A2/A3/A5
        if (dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_ONE_DIM_NUM) {
            std::string dimStr = std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "dynamicScales", dimStr.c_str(),
                                         (std::to_string(DYNAMIC_SCALE_ONE_DIM_NUM) + "D").c_str());
            return false;
        }
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    } else {
        // MX/PERTILE
        if (dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_TWO_DIM_NUM) {
            std::string dimStr = std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "dynamicScales", dimStr.c_str(),
                                         (std::to_string(DYNAMIC_SCALE_TWO_DIM_NUM) + "D").c_str());
            return false;
        }
        OP_LOGD(nodeName, "dynamicScales dim0=%ld, dim1=%ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0),
                dynamicScalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckOutputTensorDim(gert::TilingContext *context, const char *nodeName,
                                                                    const uint32_t quantMode)
{
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXShape"), return false);
    OP_TILING_CHECK(
        expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expandXShape", std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expandXShape must be 2D."),
        return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    // Skip checking dynamicScales when quantMode is 0 or 1, the same in A2/A3/A5
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        if (!CheckDynamicScalesDim(context, nodeName, quantMode)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "dynamicScales", "", "dynamicScales shape check failed");
            return false;
        }
    }

    const gert::StorageShape *expandIdxStorageShape = context->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxShape"),
                    return false);
    OP_TILING_CHECK(
        expandIdxStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expandIdxShape", std::to_string(expandIdxStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expandIdxShape must be 1D."),
        return false);
    OP_LOGD(nodeName, "expandIdx dim0 = %ld", expandIdxStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsShape"), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "expertTokenNumsShape",
                        std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of expertTokenNumsShape must be 1D."),
                    return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckEpTpRecvTensorDim(const gert::TilingContext *context,
                                                                      const char *nodeName)
{
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountShape"),
                    return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "epRecvCountShape",
                        std::to_string(epRecvCountStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of epRecvCountShape must be 1D."),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorDim(gert::TilingContext *context, const char *nodeName,
                                                       const bool isScales, const uint32_t quantMode,
                                                       const uint32_t opVersion)
{
    if (!CheckInputTensorDim(context, nodeName, isScales, quantMode)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "input tensors", "", "input param shape is invalid");
        return false;
    }

    // A3/A5 v1接口的x_active_mask不支持传入
    if (opVersion == OP_VERSION_1) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        if (xActiveMaskStorageShape != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "x_active_mask", "provided",
                                      "only None is supported on this interface version");
            return false;
        }
    }

    if ((!CheckOutputTensorDim(context, nodeName, quantMode)) || (!CheckEpTpRecvTensorDim(context, nodeName))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "output tensors", "", "output param shape is invalid");
        return false;
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckCommonOutputTensorDataType(const gert::TilingContext *context,
                                                                               const char *nodeName)
{
    auto expandIdxDesc = context->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    if (expandIdxDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc");
        return false;
    }
    if (expandIdxDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(expandIdxDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandIdx", dtypeStr.c_str(), "INT32");
        return false;
    }

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    if (expertTokenNumsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsDesc");
        return false;
    }
    if (expertTokenNumsDesc->GetDataType() != ge::DT_INT64) {
        std::string dtypeStr = Ops::Base::ToString(expertTokenNumsDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertTokenNums", dtypeStr.c_str(), "INT64");
        return false;
    }

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    if (epRecvCountsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountsDesc");
        return false;
    }
    if (epRecvCountsDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(epRecvCountsDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "epRecvCounts", dtypeStr.c_str(), "INT32");
        return false;
    }

    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckInputTensorDataType(const gert::TilingContext *context,
                                                                        const char *nodeName, const bool isScales)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    if (xDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc");
        return false;
    }
    if ((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16)) {
        std::string dtypeStr = Ops::Base::ToString(xDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", dtypeStr.c_str(), "BF16 or FLOAT16");
        return false;
    }

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    if (expertIdDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc");
        return false;
    }
    if (expertIdDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(expertIdDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expert_ids", dtypeStr.c_str(), "INT32");
        return false;
    }

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        if (scalesDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc");
            return false;
        }
        if (scalesDesc->GetDataType() != ge::DT_FLOAT) {
            std::string dtypeStr = Ops::Base::ToString(scalesDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", dtypeStr.c_str(), "FLOAT");
            return false;
        }
    }
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorDataType(gert::TilingContext *context, const char *nodeName,
                                                            const bool isScales, const uint32_t quantMode)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(!CheckInputTensorDataType(context, nodeName, isScales),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Input param data type is invalid."), return false);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
    if (quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expand_x",
                                                  Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), "INT8"),
                        return false);
    } else {
        if (expandXDesc->GetDataType() != xDesc->GetDataType()) {
            std::string dtypeStr = Ops::Base::ToString(expandXDesc->GetDataType());
            std::string xDtypeStr = Ops::Base::ToString(xDesc->GetDataType());
            std::string reason = "The dtype of expandX must be the same as that of x";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expandX", dtypeStr.c_str(), reason.c_str());
            return false;
        }
    }

    if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                        return false);
        OP_TILING_CHECK(
            dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamic_scales",
                                      Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "FLOAT"),
            return false);
    }

    OP_TILING_CHECK(!CheckCommonOutputTensorDataType(context, nodeName),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CheckCommonOutputTensorDataType failed."), return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeNoScales(const gert::TilingContext *context,
                                                                           const char *nodeName, const bool isScales)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
    OP_TILING_CHECK((NON_QUANT_DTYPE.find(static_cast<ge::DataType>(xDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "BF16, FLOAT16, FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8"),
                    return false);
    // ExpandX: the same as X
    if (expandXDesc->GetDataType() != xDesc->GetDataType()) {
        std::string dtypeStr = Ops::Base::ToString(expandXDesc->GetDataType());
        std::string xDtypeStr = Ops::Base::ToString(xDesc->GetDataType());
        std::string reason = "The dtype of expandX must be the same as that of x";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expandX", dtypeStr.c_str(), reason.c_str());
        return false;
    }
    // Scales: bf16/fp16: nullptr; hif8: fp32; e5m2/e4m3fn: float/e8m0
    // Dynamic scales: the same as scales, and no validations for bf16/fp16
    // If X is bf16/fp16, the scales must be nullptr, which is validated in CheckQuantModeAndScales
    // Hence the datatype of X must be e5m2/e4m3fn/hif8 when isScales is true
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                        return false);
        OP_TILING_CHECK((xDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "FLOAT"),
                        return false);
        OP_TILING_CHECK(
            (scalesDesc->GetDataType() != ge::DT_FLOAT) && (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                                      "FLOAT or FLOAT8_E8M0"),
            return false);
        OP_TILING_CHECK(
            dynamicScalesDesc->GetDataType() != scalesDesc->GetDataType(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                nodeName, "dynamic_scales", Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(),
                (std::string("must match scales dtype ") + Ops::Base::ToString(scalesDesc->GetDataType())).c_str()),
            return false);
    }
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeStaticOrDynamic(const gert::TilingContext *context,
                                                                                  const char *nodeName, bool isScales)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                    return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "BF16 or FLOAT16"),
        return false);
    // Scales: fp32, optional for dynamic/pertoken/pertile, required for static/hif8
    // isScales has been checked in CheckQuantModeAndScales
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "FLOAT"),
                        return false);
    }
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamic_scales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "FLOAT"),
                    return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeMxfp8(const gert::TilingContext *context,
                                                                        const char *nodeName)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                    return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "BF16 or FLOAT16"),
        return false);
    // No Scales input
    OP_TILING_CHECK(
        dynamicScalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamic_scales",
                                  Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "FLOAT8_E8M0"),
        return false);
    return true;
}

inline bool MoeDistributeDispatchTilingHelper::CheckDistinctTensorDataType(gert::TilingContext *context,
                                                                           const char *nodeName, const bool isScales,
                                                                           const uint32_t quantMode)
{
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeNoScales(context, nodeName, isScales),
                        OP_LOGE_WITHOUT_REPORT(nodeName, "CheckTensorDataType for nonquant mode failed."),
                        return false);
    } else if ((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) ||
               (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP))) {
        OP_TILING_CHECK(!CheckTensorDataTypeMxfp8(context, nodeName),
                        OP_LOGE_WITHOUT_REPORT(nodeName, "CheckTensorDataType for mx quant mode failed."),
                        return false);
    } else {
        // static/dynamic/pertolen/pertile/hif8
        OP_TILING_CHECK(!CheckTensorDataTypeStaticOrDynamic(context, nodeName, isScales),
                        OP_LOGE_WITHOUT_REPORT(nodeName, "CheckTensorDataType for quantMode %u failed.", quantMode),
                        return false);
    }
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorDataTypeA5(gert::TilingContext *context, const char *nodeName,
                                                              const bool isScales, const uint32_t quantMode)
{
    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    if (expertIdDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc");
        return false;
    }
    if (expertIdDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(expertIdDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expert_ids", dtypeStr.c_str(), "INT32");
        return false;
    }

    OP_TILING_CHECK(!CheckDistinctTensorDataType(context, nodeName, isScales, quantMode),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CheckDistinctTensorDataType failed."), return false);

    OP_TILING_CHECK(!CheckCommonOutputTensorDataType(context, nodeName),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CheckCommonOutputTensorDataType failed."), return false);
    return true;
}

bool MoeDistributeDispatchTilingHelper::CheckTensorFormat(const gert::TilingContext *context, const char *nodeName,
                                                          const bool isScales, const uint32_t quantMode)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    auto xDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    if (xDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(xDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x", fmtStr.c_str(), "ND");
        return false;
    }

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    if (expertIdDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc");
        return false;
    }
    auto expertIdDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat()));
    if (expertIdDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expertIdDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expert_ids", fmtStr.c_str(), "ND");
        return false;
    }

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        if (scalesDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc");
            return false;
        }
        auto scalesDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat()));
        if (scalesDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(scalesDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "scales", fmtStr.c_str(), "ND");
            return false;
        }
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if (expandXDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc");
        return false;
    }
    auto expandXDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat()));
    if (expandXDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expandXDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandX", fmtStr.c_str(), "ND");
        return false;
    }

    // quantMode 2, compatible with A2/A3
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        if (dynamicScalesDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc");
            return false;
        }
        auto dynamicScalesDescFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat()));
        if (dynamicScalesDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(dynamicScalesDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "dynamicScales", fmtStr.c_str(), "ND");
            return false;
        }
    }

    auto expandIdxDesc = context->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    if (expandIdxDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandIdxDesc");
        return false;
    }
    auto expandIdxDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expandIdxDesc->GetStorageFormat()));
    if (expandIdxDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expandIdxDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandIdx", fmtStr.c_str(), "ND");
        return false;
    }

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    if (expertTokenNumsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsDesc");
        return false;
    }
    auto expertTokenNumsDescFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat()));
    if (expertTokenNumsDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expertTokenNumsDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertTokenNums", fmtStr.c_str(), "ND");
        return false;
    }

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    if (epRecvCountsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountsDesc");
        return false;
    }
    auto epRecvCountsDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat()));
    if (epRecvCountsDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(epRecvCountsDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "epRecvCounts", fmtStr.c_str(), "ND");
        return false;
    }
    return true;
}

ge::graphStatus MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatch(gert::TilingContext *context,
                                                                                    const char *nodeName,
                                                                                    const bool isScales,
                                                                                    const uint32_t quantMode)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, OP_VERSION_1),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params shape"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params dataType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params format"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatchA5(gert::TilingContext *context,
                                                                                      const bool isScales,
                                                                                      const uint32_t quantMode,
                                                                                      const bool isTokenMask)
{
    // nodeName已在调用处判空
    const char *nodeName = context->GetNodeName();
    auto opVersion = OpVersionManager::GetInstance().GetVersion();
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, opVersion),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params shape"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataTypeA5(context, nodeName, isScales, quantMode),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params dataType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params format"), return ge::GRAPH_FAILED);
    if ((opVersion != OP_VERSION_1) && isTokenMask) {
        OP_TILING_CHECK(!CheckTokenMask(context, nodeName), OP_LOGE_WITHOUT_REPORT(nodeName, "xActiveMask is invalid."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchTilingHelper::CheckTokenMask(const gert::TilingContext *context, const char *nodeName)
{
    // Check Dim/DType/Format
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    if (xActiveMaskStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskStorageShape");
        return false;
    }
    if (xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIM) {
        std::string dimStr = std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "xActiveMask", dimStr.c_str(), "1D");
        return false;
    }
    auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    if (xActiveMaskDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc");
        return false;
    }
    if (xActiveMaskDesc->GetDataType() != ge::DT_BOOL) {
        std::string dtypeStr = Ops::Base::ToString(xActiveMaskDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "xActiveMask", dtypeStr.c_str(), "BOOL");
        return false;
    }
    auto xActiveMaskDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat()));
    if (xActiveMaskDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(xActiveMaskDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "xActiveMask", fmtStr.c_str(), "ND");
        return false;
    }

    return true;
}
} // namespace optiling
