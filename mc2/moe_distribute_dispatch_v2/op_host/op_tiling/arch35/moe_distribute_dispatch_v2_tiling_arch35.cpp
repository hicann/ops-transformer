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
 * \file moe_distribute_dispatch_v2_tiling_arch35.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v2_tiling_arch35.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "mc2_log_compat.h"
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t ATTR_COMM_ALG_INDEX = 13;

const std::string OP_NAME = "MoeDistributeDispatchA5";
} // namespace

namespace optiling {
static bool CheckTensorDataTypeStaticOrDynamic(const gert::TilingContext* context, const char* nodeName, bool isScales,
                                               DispatchV2Config& config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                    return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16 or float16"),
        return false);
    // Scales: fp32, optional for dynamic/pertoken/pertile, required for static/hif8
    // isScales has been checked in CheckQuantModeAndScales
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
    }
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "float"),
                    return false);
    return true;
}

static bool CheckTensorDataTypeHiF4(const gert::TilingContext* context, const char* nodeName, DispatchV2Config& config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16 or float16"),
        return false);
    // No Scales input
    OP_TILING_CHECK(
        dynamicScalesDesc->GetDataType() != Ops::Transformer::DT_HIFLOAT4_SCALE,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                  Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "hifloat4_scale"),
        return false);
    return true;
}

static bool CheckTensorDataTypeMxfp8(const gert::TilingContext* context, const char* nodeName, DispatchV2Config& config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16 or float16"),
        return false);
    // No Scales input
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "e8m0"),
                    return false);
    return true;
}

static bool CheckTensorDataTypeNonQuant(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                                        DispatchV2Config& config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK((NON_QUANT_DTYPE.find(static_cast<ge::DataType>(xDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "bf16, fp16, e5m2, e4m3fn, e1m2, e2m1, hif8, or hif4"),
                    return false);
    // ExpandX: the same as X
    OP_TILING_CHECK(
        (expandXDesc->GetDataType() != xDesc->GetDataType() && (expandXDesc->GetDataType() != ge::DT_HIFLOAT8)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "x and expandX",
                                               (std::string(Ops::Base::ToString(xDesc->GetDataType())) + " and " +
                                                Ops::Base::ToString(expandXDesc->GetDataType()))
                                                   .c_str(),
                                               "The dtype of expandX must be the same as the dtype of x"),
        return false);
    // Scales: bf16/fp16: nullptr; hif8: fp32; e5m2/e4m3fn: float/e8m0
    // Dynamic scales: the same as scales, and no validations for bf16/fp16
    // If X is bf16/fp16, the scales must be nullptr, which is validated in CheckQuantModeAndScales
    // Hence the datatype of X must be e5m2/e4m3fn/hif8 when isScales is true
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                        return false);
        OP_TILING_CHECK((xDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
        OP_TILING_CHECK(
            ((xDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (xDesc->GetDataType() == ge::DT_FLOAT4_E1M2)) &&
                (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                                      "e8m0"),
            return false);
        OP_TILING_CHECK(
            ((xDesc->GetDataType() == Ops::Transformer::DT_HIFLOAT4) &&
             (scalesDesc->GetDataType() != Ops::Transformer::DT_HIFLOAT4_SCALE)),
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                                      "hifloat4_scale"),
            return false);
        OP_TILING_CHECK(
            (scalesDesc->GetDataType() != ge::DT_FLOAT) && (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0) &&
                (scalesDesc->GetDataType() != Ops::Transformer::DT_HIFLOAT4_SCALE),
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                                      "float, e8m0, or hifloat4_scale"),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != scalesDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                            nodeName, "scales and dynamicScales",
                            (std::string(Ops::Base::ToString(scalesDesc->GetDataType())) + " and " +
                             Ops::Base::ToString(dynamicScalesDesc->GetDataType()))
                                .c_str(),
                            "The dtype of dynamicScales must be the same as the dtype of scales"),
                        return false);
    }
    return true;
}

static bool CheckDistinctTensorDataType(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                                        const uint32_t quantMode, DispatchV2Config& config)
{
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(
            !CheckTensorDataTypeNonQuant(context, nodeName, isScales, config),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                     "One or more input tensors or output tensors have invalid dtypes"),
            return false);
    } else if ((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) ||
               (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP))) {
        OP_TILING_CHECK(
            !CheckTensorDataTypeMxfp8(context, nodeName, config),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                     "One or more input tensors or output tensors have invalid dtypes"),
            return false);
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::HIF4_QUANT)) {
        OP_TILING_CHECK(
            !CheckTensorDataTypeHiF4(context, nodeName, config),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                     "One or more input tensors or output tensors have invalid dtypes"),
            return false);
    } else {
        // static/perToken/perGroup
        OP_TILING_CHECK(
            !CheckTensorDataTypeStaticOrDynamic(context, nodeName, isScales, config),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                     "One or more input tensors or output tensors have invalid dtypes"),
            return false);
    }
    return true;
}

static bool CheckQuantModeAndExpandXType(const gert::TilingContext* context, const char* nodeName,
                                         DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    QuantModeA5 quantMode = static_cast<QuantModeA5>(*quantModePtr);
    auto modeToFind = QUANT_MODE_MAP.find({quantMode, static_cast<ge::DataType>(expandXDesc->GetDataType())});
    OP_TILING_CHECK(modeToFind == QUANT_MODE_MAP.end(),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                        (std::string("The dtype of expandX is not supported by quantMode=") +
                         std::to_string(static_cast<uint32_t>(quantMode)))
                            .c_str()),
                    return false);
    OP_LOGD(nodeName, "quantMode=%u, expandx datatype=%s, get realMode=%u\n", static_cast<uint32_t>(quantMode),
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), static_cast<uint32_t>(modeToFind->second));
    return true;
}

bool MoeDistributeDispatchV2TilingFuncA5::CheckTensorDataType(const gert::TilingContext* context, const char* nodeName,
                                                              const bool isScales, const uint32_t quantMode,
                                                              const bool isActiveMask, const bool hasElasticInfo,
                                                              const bool isPerformance, DispatchV2Config& config)
{
    OP_TILING_CHECK(!CheckQuantModeAndExpandXType(context, nodeName, config),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "quantMode and expandX",
                                                             "The quantMode is not supported by the dtype of expandX"),
                    return false);
    OP_TILING_CHECK(
        !CheckDistinctTensorDataType(context, nodeName, isScales, quantMode, config),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors and output tensors",
                                                 "One or more input tensors or output tensors have invalid dtypes"),
        return false);
    OP_TILING_CHECK(
        !CheckCommomOtherInputTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config),
        OP_LOGE(nodeName, "CheckCommomOtherInputTensorDataType failed."), return false);
    OP_TILING_CHECK(!CheckCommomOutputTensorDataType(context, nodeName),
                    OP_LOGE(nodeName, "CheckCommomOutputTensorDataType failed."), return false);

    return true;
}

uint64_t MoeDistributeDispatchV2TilingFuncA5::CalTilingKey(const bool isScales, const uint32_t quantMode,
                                                           const bool isSetFullMeshV2, bool isLayered)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_MTE;
    if (isScales) {
        scaleMode = true;
    }
    if (isSetFullMeshV2) {
        templateDispatch = TILINGKEY_ENABLE_FULLMESH;
    }
    if (isLayered) {
        templateDispatch = TILINGKEY_ENABLE_HIERARCHY;
    }
    tilingKey = GET_TPL_TILING_KEY(tilingKeyQuantMode, scaleMode, templateDispatch, commMode, TILINGKEY_TPL_A5);
    return tilingKey;
}

static inline ge::graphStatus CheckHif4QuantOnCurrentPlatform(const gert::TilingContext* context, const char* nodeName,
                                                              const uint32_t quantMode)
{
    if (quantMode != static_cast<uint32_t>(QuantModeA5::HIF4_QUANT)) {
        return ge::GRAPH_SUCCESS;
    }
    const NpuArch npuArch = mc2tiling::GetNpuArch(context);
    if ((npuArch == Ops::Base::DAV_2201) || (npuArch == Ops::Base::DAV_3510)) {
        std::string npuArchStr = std::to_string(static_cast<int32_t>(npuArch));
        std::string reason = "HiF4 quant is only supported on Ascend960(A6) and later, but npuArch=" + npuArchStr;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quantMode", std::to_string(quantMode).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext* context, const char* nodeName, bool isScales,
                                               const uint32_t quantMode, DispatchV2Config& config)
{
    OP_TILING_CHECK(CheckHif4QuantOnCurrentPlatform(context, nodeName, quantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "HiF4 quant is not supported on the current platform."), return ge::GRAPH_FAILED);
    if (isScales && ((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) ||
                     (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP)) ||
                     (quantMode == static_cast<uint32_t>(QuantModeA5::HIF4_QUANT)))) {
        std::string reason = "scales must be nullptr when quantMode is MX_QUANT, MX_QUANT_CLIP or HIF4_QUANT";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context->GetInputDesc(config.xIndex);
    if (xDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc");
        return ge::GRAPH_FAILED;
    }
    std::string xDtypeStr = Ops::Base::ToString(xDesc->GetDataType());
    if (isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16))) {
        std::string reason = "scales must be nullptr when quantMode is NON_QUANT and x dtype is " + xDtypeStr;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) ||
         (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN) || (xDesc->GetDataType() == Ops::Transformer::DT_HIFLOAT4) ||
         (xDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (xDesc->GetDataType() == ge::DT_FLOAT4_E1M2))) {
        std::string reason = "scales must be provided when quantMode is NON_QUANT and x dtype is " + xDtypeStr;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "not provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        std::string reason = "scales must be provided when quantMode is STATIC_QUANT";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "not provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::CheckQuantModeMatchScales(gert::TilingContext* context,
                                                                               const char* nodeName, bool isScales,
                                                                               uint32_t quantMode,
                                                                               DispatchV2Config& config)
{
    OP_TILING_CHECK(CheckQuantModeAndScales(context, nodeName, isScales, quantMode, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d,quantMode is %u.",
                            static_cast<int32_t>(isScales), quantMode),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::CheckCommAlgPtr(const char* commAlgPtr, const char* nodeName)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::CheckQuantModePtr(const int64_t* quantModePtr,
                                                                       const char* nodeName)
{
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
                        (*quantModePtr >= static_cast<int64_t>(QuantModeA5::BUTT)),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "quantMode", std::to_string(*quantModePtr).c_str(),
                        std::string("[") + std::to_string(static_cast<int64_t>(QuantModeA5::NON_QUANT)) + "," +
                            std::to_string(static_cast<int64_t>(QuantModeA5::BUTT) - 1) + std::string("]").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::MoeDistributeDispatchTilingFuncImpl(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE(OP_NAME, "Fail to get tiling context."), return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE(OP_NAME, "Fail to get nodeName."), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    if (commAlgPtr != nullptr) {
        OP_LOGD(nodeName, "Set 'commAlg' as '%s'", commAlgPtr);
    }
    // 检查 commAlg 参数合法性校验
    bool isNullOrEmpty = (commAlgPtr == nullptr) || (std::strlen(commAlgPtr) == 0);

    bool isFullmeshV1 = (!isNullOrEmpty) && (std::strcmp(commAlgPtr, "fullmesh_v1") == 0);
    bool isFullmeshV2 = (!isNullOrEmpty) && (std::strcmp(commAlgPtr, "fullmesh_v2") == 0);
    bool isMte = isFullmeshV1 || isFullmeshV2;
    OP_TILING_CHECK(!(isNullOrEmpty || isMte),
                    OP_LOGE(nodeName,
                            "Invalid parameter: 'commAlg'='%s'. "
                            "Only 'fullmesh_v1' and 'fullmesh_v2' are supported. "
                            "Nullptr and empty char* are also allowed but will be interpreted as 'fullmesh_v1'.",
                            commAlgPtr),
                    return ge::GRAPH_FAILED);

    if (isNullOrEmpty) {
        OP_LOGI(nodeName, "Parameter 'commAlg' is nullptr/empty, defaulting to 'fullmesh_v1'.");
    }
    DispatchV2Config config;
    config.xIndex = 0U;                      // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    config.expertIdsIndex = 1U;              // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
    config.scalesIndex = 2U;                 // 2: 根据dispatchV2算子原型标志位初始化scales索引
    config.xActiveMaskIndex = 3U;            // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
    config.expertScalesIndex = 4U;           // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
    config.elasticInfoIndex = 5U;            // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
    config.performanceInfoIndex = 6U;        // 6: 根据dispatchV2算子原型标志位初始化performanceInfo索引
    config.attrGroupEpIndex = 0;             // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    config.attrEpWorldSizeIndex = 1;         // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
    config.attrEpRankIdIndex = 2;            // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
    config.attrMoeExpertNumIndex = 3;        // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
    config.attrGroupTpIndex = 4;             // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
    config.attrTpWorldSizeIndex = 5;         // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
    config.attrTpRankIdIndex = 6;            // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
    config.attrExpertSharedTypeIndex = 7;    // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
    config.attrSharedExpertNumIndex = 8;     // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
    config.attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
    config.attrQuantModeIndex = 10;          // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
    config.attrGlobalBsIndex = 11;           // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
    config.attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
    config.attrCommAlgIndex = 13;             // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
    config.attrZeroExpertNumIndex = 14;  // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
    config.attrCopyExpertNumIndex = 15;  // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
    config.attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
    config.isMc2Context = false;
    // MTE 调用 A3 tiling 实现
    return MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context)
{
    ge::graphStatus ret = MoeDistributeDispatchTilingFuncImpl(context);
    return ret;
}

static ge::graphStatus MoeDistributeDispatchV2TilingFuncImpl(gert::TilingContext* context)
{
    MoeDistributeDispatchV2TilingFuncA5 impl;
    return impl.MoeDistributeDispatchV2TilingFunc(context);
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFuncImpl)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
} // namespace optiling
