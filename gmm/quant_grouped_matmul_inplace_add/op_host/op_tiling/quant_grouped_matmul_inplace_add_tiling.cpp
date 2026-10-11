/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "register/op_impl_registry.h"
#include "quant_grouped_matmul_inplace_add_tiling.h"
#include "quant_grouped_matmul_inplace_add_basic_api_tiling.h"
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"
#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace QuantGroupedMatmulInplaceAdd;
using namespace optiling::GmmConstant;
namespace optiling {

void QuantGroupedInplaceAddTiling::Reset()
{
    tilingData_ = QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams();
    return;
}

ge::graphStatus QuantGroupedInplaceAddTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool QuantGroupedInplaceAddTiling::AnalyzeAttrs()
{
    return AnalyzeAttrsForInplaceAdd(context_, inputParams_);
}

bool QuantGroupedInplaceAddTiling::AnalyzeDtype()
{
    return AnalyzeDtypeForInplaceAdd(context_, inputParams_) && CheckDtypeForInplaceAdd(inputParams_);
}

bool QuantGroupedInplaceAddTiling::SetQuantModeForQGmmInplaceAdd()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    return true;
}

bool QuantGroupedInplaceAddTiling::AnalyzeInputs()
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
    if (IsMicroScaling()) {
        OP_CHECK_IF(!CheckShapeForMxQuant(x1ScaleShape, wScaleShape, inputParams_),
                    OP_LOGE(inputParams_.opName, "CheckShapeForMxQuant failed."), return false);
    } else {
        OP_CHECK_IF(!CheckShapeForHif8Quant(x1ScaleShape, wScaleShape, inputParams_),
                    OP_LOGE(inputParams_.opName, "CheckShapeForHif8Quant failed."), return false);
    }
    OP_CHECK_IF(!SetQuantModeForQGmmInplaceAdd(), OP_LOGE(inputParams_.opName, "SetQuantModeForQGmmInplaceAdd failed."),
                return false);
    SetKernelType();
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool QuantGroupedInplaceAddTiling::CheckCoreNum() const
{
    return CheckCoreNumForInplaceAdd(context_, inputParams_);
}

ge::graphStatus QuantGroupedInplaceAddTiling::DoOpTiling()
{
    tilingData_.quantGmmInplaceAddParams.groupNum = inputParams_.groupNum;
    tilingData_.quantGmmInplaceAddParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams());
    return ge::GRAPH_SUCCESS;
}

int64_t QuantGroupedInplaceAddTiling::LogQuantParams() const
{
    const auto& params = tilingData_.quantGmmInplaceAddParams;
    std::ostringstream oss;
    oss << "QGMMIAParams: groupNum = " << params.groupNum
        << ", groupListType = " << static_cast<uint32_t>(params.groupListType);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
    return 0;
}

ge::graphStatus QuantGroupedInplaceAddTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.M = inputParams_.mSize;
    tilingData_.mmTilingData.N = inputParams_.nSize;
    tilingData_.mmTilingData.Ka = inputParams_.kSize;
    tilingData_.mmTilingData.Kb = inputParams_.kSize;
    tilingData_.mmTilingData.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.mmTilingData.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.mmTilingData.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.mmTilingData.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.depthA1 = basicTiling_.depthA1;
    tilingData_.mmTilingData.depthB1 = basicTiling_.depthB1;
    tilingData_.mmTilingData.stepM = basicTiling_.stepM;
    tilingData_.mmTilingData.stepN = basicTiling_.stepN;
    tilingData_.mmTilingData.stepKa = basicTiling_.stepKa;
    tilingData_.mmTilingData.stepKb = basicTiling_.stepKb;
    tilingData_.mmTilingData.isBias = 0;
    tilingData_.mmTilingData.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.mmTilingData.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        tilingData_.mmTilingData.mxTypePara =
            (SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.mxTypePara +=
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.mmTilingData.mxTypePara +=
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedInplaceAddTiling::PostTiling()
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

ASCENDC_EXTERN_C ge::graphStatus TilingGMMInplaceAdd(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGMMInplaceAdd(gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_CHECK_IF(
        (compileInfoPtr->aicNum == 0 || compileInfoPtr->aivNum == 0 || compileInfoPtr->ubSize == 0 ||
         compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 ||
         compileInfoPtr->l0BSize == 0 || compileInfoPtr->l2Size == 0),
        OP_LOGE(context->GetNodeName(),
                "Platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, "
                "l0CSize=%lu, l0ASize=%lu, l0BSize=%lu, l2Size=%lu",
                compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
                compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize, compileInfoPtr->l2Size),
        return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
    return ge::GRAPH_SUCCESS;
}
uint64_t QuantGroupedInplaceAddTiling::GetTilingKey() const
{
    return GetTilingKeyForInplaceAdd(inputParams_);
}

REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatmulInplaceAdd, QGMMInplaceAddBasicApiTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatmulInplaceAdd, QuantGroupedInplaceAddTiling, 1);
IMPL_OP_OPTILING(QuantGroupedMatmulInplaceAdd)
    .Tiling(TilingGMMInplaceAdd)
    .TilingParse<GMMCompileInfo>(TilingPrepareForGMMInplaceAdd); // register into the framework
} // namespace optiling
