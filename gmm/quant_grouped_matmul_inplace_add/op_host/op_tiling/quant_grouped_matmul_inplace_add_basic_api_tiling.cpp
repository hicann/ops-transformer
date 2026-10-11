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
 * \file quant_grouped_matmul_inplace_add_basic_api_tiling.cpp
 * \brief
 */
#include "quant_grouped_matmul_inplace_add_basic_api_tiling.h"

#include <alog_pub.h>

#include <climits>

#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
#include "log/log.h"
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"

using namespace Ops::Transformer::OpTiling;
using namespace QuantGroupedMatmulInplaceAdd;
using namespace optiling::GmmConstant;
namespace optiling {

QGMMInplaceAddBasicApiTiling::QGMMInplaceAddBasicApiTiling(gert::TilingContext* context)
    : GroupedQmmBasicApiTiling(context)
{}

bool QGMMInplaceAddBasicApiTiling::IsCapable()
{
    // T-T
    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    if (scaleStorageShape == nullptr) {
        return false;
    }
    const gert::Shape& wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();

    return inputParams_.aDtype == ge::DT_HIFLOAT8 &&
           (scaleDimNum == 1 || (scaleDimNum == QuantGroupedMatmulInplaceAdd::DIM_NUM_2D &&
                                 static_cast<uint64_t>(wScaleShape.GetDim(1)) == 1));
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeAttrs()
{
    return AnalyzeAttrsForInplaceAdd(context_, inputParams_);
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeDtype()
{
    return AnalyzeDtypeForInplaceAdd(context_, inputParams_) && CheckDtypeForInplaceAdd(inputParams_);
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x1", "nullptr", "xStorageShape cannot be nullptr"),
        return false);
    const gert::Shape& xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_IF(
        wStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x2", "nullptr", "wStorageShape cannot be nullptr"),
        return false);
    const gert::Shape& wShape = wStorageShape->GetOriginShape();

    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale2", "nullptr",
                                                      "scaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape& wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        scaleDimNum < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale1", std::to_string(scaleDimNum), "positive integer"),
        return false);

    auto x1ScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale1", "nullptr",
                                                      "x1ScaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape& x1ScaleShape = x1ScaleStorageShape->GetOriginShape();

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);

    // HIFLOAT8 T-T 分支：IsCapable 已保证 wScale 为 (g,) 或 (g,1)，复用合并后的 T-T/T-C 校验
    OP_CHECK_IF(!IsMicroScaling() && !CheckShapeForHif8Quant(x1ScaleShape, wScaleShape, inputParams_),
                OP_LOGE(inputParams_.opName, "CheckShapeForHif8Quant failed."), return false);

    OP_CHECK_IF(!SetQuantModeForQGmmInplaceAdd(), OP_LOGE(inputParams_.opName, "SetQuantModeForQGmmInplaceAdd failed."),
                return false);
    SetKernelType();
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool QGMMInplaceAddBasicApiTiling::SetQuantModeForQGmmInplaceAdd()
{
    inputParams_.bQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    return true;
}

bool QGMMInplaceAddBasicApiTiling::CheckCoreNum() const
{
    return CheckCoreNumForInplaceAdd(context_, inputParams_);
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams());
    return ge::GRAPH_SUCCESS;
}

int64_t QGMMInplaceAddBasicApiTiling::LogQuantParams() const
{
    std::ostringstream oss;
    oss << "QGmmInplaceAddBasicApiParams: groupNum = " << tilingData_.gmmQuantParams.groupNum
        << ", aQuantMode = " << tilingData_.gmmQuantParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmQuantParams.bQuantMode
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmQuantParams.groupListType);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
    return 0;
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    return ge::GRAPH_SUCCESS;
}

uint64_t QGMMInplaceAddBasicApiTiling::GetTilingKey() const
{
    return GetTilingKeyForInplaceAdd(inputParams_);
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    OP_CHECK_IF(sizeof(tilingData_) % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", sizeof(tilingData_)),
                return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
