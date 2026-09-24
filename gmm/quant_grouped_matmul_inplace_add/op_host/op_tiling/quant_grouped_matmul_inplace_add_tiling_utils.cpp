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
 * \file quant_grouped_matmul_inplace_add_tiling_utils.cpp
 * \brief
 */
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"

#include <graph/utils/type_utils.h>

#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
#include "../quant_grouped_matmul_inplace_add_host_utils.h"
#include "log/log.h"

using namespace optiling;
using namespace optiling::GmmConstant;

namespace QuantGroupedMatmulInplaceAdd {

bool AnalyzeAttrsForInplaceAdd(const gert::TilingContext *context, GQmmInputInfo &inputParams)
{
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
        if (groupListTypePtr != nullptr) {
            OP_CHECK_IF(*groupListTypePtr != 0 && *groupListTypePtr != 1,
                        OP_LOGE_FOR_INVALID_VALUE(inputParams.opType, "groupListType",
                                                  std::to_string(*groupListTypePtr), "0 or 1"),
                        return false);
            inputParams.groupListType = static_cast<int8_t>(*groupListTypePtr);
        }
        const int64_t *groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_SIZE);
        if (groupSizePtr != nullptr) {
            OP_CHECK_IF(*groupSizePtr != 0,
                        OP_LOGE_FOR_INVALID_VALUE(inputParams.opType, "groupSize", std::to_string(*groupSizePtr), "0"),
                        return false);
        }
    }
    inputParams.transA = true;
    return true;
}

bool AnalyzeDtypeForInplaceAdd(const gert::TilingContext *context, GQmmInputInfo &inputParams)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "x1", "nullptr", "xDesc cannot be nullptr"),
                return false);
    inputParams.aDtype = xDesc->GetDataType();
    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "x2", "nullptr", "wDesc cannot be nullptr"),
                return false);
    inputParams.bDtype = wDesc->GetDataType();
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(
        scaleDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "scale2", "nullptr", "scaleDesc cannot be nullptr"),
        return false);
    inputParams.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams.perTokenScaleDtype;
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "y", "nullptr", "yDesc cannot be nullptr"),
                return false);
    inputParams.cDtype = yDesc->GetDataType();
    return true;
}

bool CheckDtypeForInplaceAdd(const GQmmInputInfo &inputParams)
{
    OP_CHECK_IF(inputParams.cDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "yRef",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams.cDtype), "DT_FLOAT"),
                return false);

    bool isHif8 = inputParams.aDtype == ge::DT_HIFLOAT8 && inputParams.bDtype == ge::DT_HIFLOAT8;
    bool isFp8 = (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
                 (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2);

    if (isHif8) {
        OP_CHECK_IF(
            inputParams.scaleDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale2",
                                      ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype), "DT_FLOAT"),
            return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype),
                                              "DT_FLOAT"),
                    return false);
        return true;
    }

    if (isFp8) {
        OP_CHECK_IF(
            inputParams.scaleDtype != ge::DT_FLOAT8_E8M0,
            OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale2",
                                      ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype), "DT_FLOAT8_E8M0"),
            return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype),
                                              "DT_FLOAT8_E8M0"),
                    return false);
        return true;
    }

    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        inputParams.opType, "x1, x2",
        ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams.aDtype),
                     ge::TypeUtils::DataTypeToSerialString(inputParams.bDtype)),
        "in quant case, the dtypes of x1 and x2 must be both DT_HIFLOAT8 or both DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2");
    return false;
}

bool CheckCoreNumForInplaceAdd(const gert::TilingContext *context, const GQmmInputInfo &inputParams)
{
    auto compileInfo = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(inputParams.opName, "compileInfo is nullptr."), return false);
    auto aicNum = compileInfo->aicNum;
    auto aivNum = compileInfo->aivNum;
    OP_CHECK_IF(aicNum == 0, OP_LOGE(inputParams.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    OP_CHECK_IF(
        aivNum != GmmConstant::CORE_RATIO * aicNum,
        OP_LOGE(inputParams.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
        return false);
    return true;
}

bool CheckShapeForHif8Quant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                            const GQmmInputInfo &inputParams)
{
    // T-T / T-C 合并校验，与 aclnn CheckHif8QuantParamsShape 对齐：
    //   scale1 (x1Scale/perToken): 1D 或 2D；firstDim == groupNum；若 2D 则 lastDim == 1
    //   scale2 (x2Scale):          1D 或 2D；firstDim == groupNum；若 2D 则 lastDim ∈ {1, nSize}
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale1", std::to_string(x1ScaleDimNum), "1 or 2"),
                return false);
    auto x1FirstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    OP_CHECK_IF(x1FirstDim != inputParams.groupNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    inputParams.opType, "scale1", "[" + ShapeToString(x1ScaleShape) + "]",
                    BuildErrorMsgStr("in T-T/T-C mode, first dim of scale1 must be equal to groupNum[",
                                     inputParams.groupNum, "]")),
                return false);
    if (x1ScaleDimNum == QuantGroupedMatmulInplaceAdd::DIM_NUM_2D) {
        auto x1LastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
        OP_CHECK_IF(
            x1LastDim != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams.opType, "scale1", "[" + ShapeToString(x1ScaleShape) + "]",
                                                  "in T-T/T-C mode, last dim of scale1 must be equal to 1"),
            return false);
    }

    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != 1 && x2ScaleDimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale2", std::to_string(x2ScaleDimNum), "1 or 2"),
                return false);
    auto x2FirstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    OP_CHECK_IF(
        x2FirstDim != inputParams.groupNum,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams.opType, "scale2", "[" + ShapeToString(x2ScaleShape) + "]",
                                              BuildErrorMsgStr("in T-T/T-C mode, first dim of scale2 must "
                                                               "be equal to groupNum[",
                                                               inputParams.groupNum, "]")),
        return false);
    if (x2ScaleDimNum == QuantGroupedMatmulInplaceAdd::DIM_NUM_2D) {
        auto x2LastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
        OP_CHECK_IF(
            x2LastDim != 1 && x2LastDim != inputParams.nSize,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams.opType, "scale2", "[" + ShapeToString(x2ScaleShape) + "]",
                                                  BuildErrorMsgStr("in T-T/T-C mode, last dim of scale2 must be "
                                                                   "equal to 1 or n[",
                                                                   inputParams.nSize, "]")),
            return false);
    }
    return true;
}

bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const GQmmInputInfo &inputParams)
{
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale2", std::to_string(x2ScaleDimNum), "3"),
                return false);
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale1", std::to_string(x1ScaleDimNum), "3"),
                return false);
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams.kSize / MXFP_BASEK_FACTOR + inputParams.groupNum;
    OP_CHECK_IF(
        xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue || xScaleMDim != inputParams.mSize,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale1", ShapeToString(x1ScaleShape),
                                  ShapeDimsToString(expectedKDimValue, inputParams.mSize, MXFP_MULTI_BASE_SIZE)),
        return false);
    OP_CHECK_IF(
        wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue || wScaleNDim != inputParams.nSize,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale2", ShapeToString(x2ScaleShape),
                                  ShapeDimsToString(expectedKDimValue, inputParams.nSize, MXFP_MULTI_BASE_SIZE)),
        return false);
    return true;
}

uint64_t GetTilingKeyForInplaceAdd(const GQmmInputInfo &inputParams)
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams.transB), static_cast<uint64_t>(inputParams.transA),
                              static_cast<uint64_t>(inputParams.kernelType));
}

} // namespace QuantGroupedMatmulInplaceAdd
