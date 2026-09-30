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
 * \file grouped_matmul_swiglu_quant_v2_tiling.cpp
 * \brief
 */

#include "grouped_matmul_swiglu_quant_v2_tiling.h"
#include <climits>
#include <string>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "op_host/tiling_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "grouped_matmul_swiglu_quant_v2_fusion_tiling.h"
#include "grouped_matmul_swiglu_quant_v2_base_tiling.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_basic_api_tiling.h"
#include "arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::GroupedMatmulSwigluQuantV2Tiling;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace {
constexpr size_t WORKSPACE_SLOT_COUNT = 1;
} // namespace

constexpr int64_t GMMSQ_FUSING_TILING_TEMPLATE = 0;
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2FusionTiling,
                             GMMSQ_FUSING_TILING_TEMPLATE);

constexpr int64_t GMMSQ_BASE_TILING_TEMPLATE = 1;
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2BaseTiling,
                             GMMSQ_BASE_TILING_TEMPLATE);

constexpr int64_t GMMSQ_950_TILING_TEMPLATE = 2;
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2Tiling950,
                             GMMSQ_950_TILING_TEMPLATE);

constexpr int64_t GMMSQ_WEIGHT_QUANT_TILING_TEMPLATE = 3;
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2WeightQuantTiling,
                             GMMSQ_WEIGHT_QUANT_TILING_TEMPLATE);

constexpr int64_t GMMSQ_TENSOR_API_TILING_TEMPLATE = 4;
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2BasicApiTiling950,
                             GMMSQ_TENSOR_API_TILING_TEMPLATE);

constexpr int64_t SWIGLU_MODE_V3 = 2;

// Check every list instance before routing: a present tensor list does not imply
// that every descriptor and shape exists. Reject missing metadata before use.
static ge::graphStatus CheckDynamicInputDescriptorsAndShapes(const gert::TilingContext* context, size_t irIndex,
                                                             const char* inputName)
{
    constexpr char OP_NAME[] = "GroupedMatmulSwigluQuantV2";
    const auto* instanceInfo = context->GetIrInputInstanceInfo(irIndex);
    OP_CHECK_IF(instanceInfo == nullptr || instanceInfo->GetInstanceNum() == 0U,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, inputName, "requires at least one instance"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < instanceInfo->GetInstanceNum(); ++i) {
        const std::string instanceName = std::string(inputName) + "[" + std::to_string(i) + "]";
        OP_CHECK_IF(
            context->GetDynamicInputDesc(irIndex, i) == nullptr || context->GetDynamicInputShape(irIndex, i) == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, instanceName.c_str(), "does not support nullptr"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRequiredTilingContext(gert::TilingContext* context)
{
    constexpr char OP_NAME[] = "GroupedMatmulSwigluQuantV2";
    OP_CHECK_IF(context == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "context", "does not support nullptr"),
                return ge::GRAPH_FAILED);

    const struct {
        const gert::CompileTimeTensorDesc* desc;
        const gert::StorageShape* shape;
        const char* name;
    } inputs[] = {
        {context->GetRequiredInputDesc(X_INDEX), context->GetRequiredInputShape(X_INDEX), "x"},
        {context->GetRequiredInputDesc(X_SCALE_INDEX), context->GetRequiredInputShape(X_SCALE_INDEX), "xScale"},
        {context->GetRequiredInputDesc(GROUPLIST_INDEX), context->GetRequiredInputShape(GROUPLIST_INDEX), "groupList"},
    };
    for (const auto& input : inputs) {
        OP_CHECK_IF(input.desc == nullptr || input.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, input.name, "does not support nullptr"),
                    return ge::GRAPH_FAILED);
    }
    if (CheckDynamicInputDescriptorsAndShapes(context, WEIGHT_INDEX, "weight") != ge::GRAPH_SUCCESS ||
        CheckDynamicInputDescriptorsAndShapes(context, WEIGHT_SCALE_INDEX, "weightScale") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "attrs", "does not support nullptr"),
                return ge::GRAPH_FAILED);

    // Legacy V2 tiling does not consume output metadata. Both outputs are mandatory for the V3 interface.
    const auto* swigluMode = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SWIGLU_MODE);
    if (swigluMode != nullptr && *swigluMode == SWIGLU_MODE_V3) {
        OP_CHECK_IF(context->GetOutputDesc(Y_INDEX) == nullptr || context->GetOutputShape(Y_INDEX) == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "y", "does not support nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context->GetOutputDesc(Y_SCALE_INDEX) == nullptr || context->GetOutputShape(Y_SCALE_INDEX) == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "yScale", "does not support nullptr"),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(context->GetWorkspaceSizes(WORKSPACE_SLOT_COUNT) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "workspace", "does not support nullptr"),
                return ge::GRAPH_FAILED);
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "tilingData", "does not support nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetData() == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "tilingData.data", "does not support nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupedMatmulSwigluQuantV2TilingFunc(gert::TilingContext* context)
{
    if (CheckRequiredTilingContext(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto compileInfoPtr = context->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("GroupedMatmulSwigluQuantV2TilingFunc", "compileInfo is null"),
                return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    const auto* swigluMode = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SWIGLU_MODE);
    if (swigluMode != nullptr && *swigluMode != 0) {
        OP_CHECK_IF(*swigluMode != SWIGLU_MODE_V3,
                    OP_LOGE(context->GetNodeName(), "swiglu_mode only supports 0 or 2, but got %ld", *swigluMode),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(compileInfoPtr->npuArch_ != static_cast<int32_t>(NpuArch::DAV_3510),
                    OP_LOGE(context->GetNodeName(), "swiglu_mode=2 only supports Ascend 950"), return ge::GRAPH_FAILED);
        OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc", "Using the shared MXFP8 weight-NZ tiling with V3 attributes");
        return TilingRegistry::GetInstance().DoTilingImpl(context, {GMMSQ_950_TILING_TEMPLATE});
    }
    if (compileInfoPtr->supportL12BtBf16) {
        std::vector<int32_t> registerList = {GMMSQ_950_TILING_TEMPLATE};
        auto xDesc = context->GetInputDesc(GroupedMatmulSwigluQuantV2Tiling::X_INDEX);
        auto wDesc = context->GetInputDesc(GroupedMatmulSwigluQuantV2Tiling::WEIGHT_INDEX);
        auto xScaleDesc = context->GetInputDesc(GroupedMatmulSwigluQuantV2Tiling::X_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
        OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
        OP_CHECK_NULL_WITH_CONTEXT(context, xScaleDesc);
        ge::DataType xDtype = xDesc->GetDataType();
        ge::DataType wDtype = wDesc->GetDataType();
        ge::DataType xScaleDtype = xScaleDesc->GetDataType();
        ge::Format weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetFormat().GetStorageFormat()));
        if (xDtype == ge::DT_FLOAT8_E4M3FN && wDtype == ge::DT_FLOAT4_E2M1) {
            // 伪量化场景 MxA8W4: x=FP8_E4M3FN, w=FP4_E2M1
            OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc", "Using the weight quant tiling for MxA8W4");
            registerList[0] = GMMSQ_WEIGHT_QUANT_TILING_TEMPLATE;
        } else if (xScaleDtype == ge::DT_FLOAT8_E8M0 &&
                   (weightFormat == ge::FORMAT_ND || weightFormat == ge::FORMAT_NCL)) {
            // Tensor API is preferred for MX quant with ND weight. Unsupported cases fall back to template 2.
            registerList = {GMMSQ_TENSOR_API_TILING_TEMPLATE, GMMSQ_950_TILING_TEMPLATE};
            OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc",
                    "Using Tensor API tiling first and falling back to the original MX tiling");
        } else {
            // 全量化场景
            OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc", "Using the tiling strategy in the mxfp8");
        }
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    } else {
        std::vector<int32_t> registerList = {GMMSQ_FUSING_TILING_TEMPLATE, GMMSQ_BASE_TILING_TEMPLATE};
        OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc", "Using the tiling strategy in the int8");
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}

ASCENDC_EXTERN_C graphStatus TilingPrepareForGMMSwigluQuantV2(gert::TilingParseContext* context)
{
    // get info
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulSwigluQuantV2", "context", "does not support nullptr"),
        return GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum_ = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->npuArch_ = static_cast<int32_t>(ascendcPlatform.GetCurNpuArch());
    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize_);
    OP_LOGD(context->GetNodeName(), "ubSize is %lu, aicNum is %u.", compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluQuantV2)
    .Tiling(GroupedMatmulSwigluQuantV2TilingFunc)
    .TilingParse<GMMSwigluV2CompileInfo>(TilingPrepareForGMMSwigluQuantV2);
} // namespace optiling
