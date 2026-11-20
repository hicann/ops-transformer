/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_floyd_attention_grad_tiling.h"
#include <register/op_impl_registry.h>
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"
using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace FFAG {
constexpr uint32_t OUTPUT_IDX_DQ = 0;
constexpr uint32_t OUTPUT_IDX_DK1 = 1;
constexpr uint32_t OUTPUT_IDX_DV1 = 2;
constexpr uint32_t OUTPUT_IDX_DK2 = 3;
constexpr uint32_t OUTPUT_IDX_DV2 = 4;

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY1_INPUT_INDEX = 1;
constexpr uint32_t VALUE1_INPUT_INDEX = 2;
constexpr uint32_t KEY2_INPUT_INDEX = 3;
constexpr uint32_t VALUE2_INPUT_INDEX = 4;
constexpr uint32_t DY_INPUT_INDEX = 5;
constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 6;
constexpr uint32_t SOFTMAX_MAX = 7;
constexpr uint32_t SOFTMAX_SUM = 8;
constexpr uint32_t ATTENTION_IN = 9;

constexpr uint32_t LAYOUT_ATTR_IDX = 5;

constexpr uint32_t FAG_EMPTY_TILING_KEY = 90;

static uint32_t CalculateTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

// tiling func + tiling prepare
class FusedFloydAttentionGradTiling {
public:
    FusedFloydAttentionGradTilingData tilingData;
    FusedFloydAttentionGradTiling() {};

    ge::graphStatus RunEmptyTiling(gert::TilingContext* context)
    {
        uint64_t aicNum = 40; // 40: B3 default aicNum
        uint64_t aivNum = 20; // 20: B3 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context->GetCompileInfo());
            if (compilePtr == nullptr) {
                // GetCompileInfo may return nullptr per exe_graph tiling_context.h contract;
                // both platform and compile metadata unavailable is a config/platform loading failure.
                std::string reason = "platform info and compile info are both unavailable";
                const std::vector<const char*> msgKey = {"opName", "reason"};
                const std::vector<const char*> msgValue = {context->GetNodeName(), reason.c_str()};
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "platformInfo and compileInfo are both null.");
                REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
                return ge::GRAPH_FAILED;
            }
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        if (aivNum == 0) {
            std::string reason = "invalid platform core count";
            const std::vector<const char*> msgKey = {"opName", "reason"};
            const std::vector<const char*> msgValue = {"fused_floyd_attention_grad", reason.c_str()};
            OP_LOGE_WITHOUT_REPORT("fused_floyd_attention_grad", "num of aiv is 0.");
            REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
            return GRAPH_FAILED;
        }
        uint64_t dqNum =
            static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DQ)->GetStorageShape().GetShapeSize());
        if (dqNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDqNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDqNum(dqNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(dqNum / aivNum);
        }
        uint64_t dkNum =
            static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DK1)->GetStorageShape().GetShapeSize());
        if (dkNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDkNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDkNum(dkNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(dkNum / aivNum);
        }

        context->SetTilingKey(FAG_EMPTY_TILING_KEY);
        auto sliceNum = (dqNum < aivNum && dkNum < aivNum) ? std::max(dqNum, dkNum) : aivNum;
        context->SetBlockDim(CalculateTschBlockDim(sliceNum, aicNum, aivNum));
        context->SetScheduleMode(1);
        size_t* workspaces = context->GetWorkspaceSizes(1);
        // workspace上预留100M
        workspaces[0] = 100 * 1024 * 1024;
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
};

static bool IsEmptyOutput(gert::TilingContext* context)
{
    const gert::StorageShape* dqShape = context->GetOutputShape(OUTPUT_IDX_DQ);
    const gert::StorageShape* dk1Shape = context->GetOutputShape(OUTPUT_IDX_DK1);
    const gert::StorageShape* dv1Shape = context->GetOutputShape(OUTPUT_IDX_DV1);
    const gert::StorageShape* dk2Shape = context->GetOutputShape(OUTPUT_IDX_DK2);
    const gert::StorageShape* dv2Shape = context->GetOutputShape(OUTPUT_IDX_DV2);
    if (dqShape != nullptr && dk1Shape != nullptr && dv1Shape != nullptr && dk2Shape != nullptr &&
        dv2Shape != nullptr) {
        if (dqShape->GetStorageShape().GetShapeSize() == 0 || dk1Shape->GetStorageShape().GetShapeSize() == 0 ||
            dv1Shape->GetStorageShape().GetShapeSize() == 0 || dk2Shape->GetStorageShape().GetShapeSize() == 0 ||
            dv2Shape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckAttrs(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "attribute container is unavailable"})),
                return ge::GRAPH_FAILED);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    size_t* workspaces = context->GetWorkspaceSizes(1);

    OP_CHECK_IF(scaleValuePtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale_value", "attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(workspaces == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "workspace size storage is unavailable"})),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckBaseInput(gert::TilingContext* context)
{
    // query: [B, N2, G, S1, D], key1: [B, N2, G, S2, D], key2: [B, N2, S2, S1, D]
    auto& queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto& keyShape = context->GetInputShape(KEY1_INPUT_INDEX)->GetStorageShape();
    OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0) || queryShape.GetDim(1) != keyShape.GetDim(1) ||
                 queryShape.GetDim(4) != keyShape.GetDim(4)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "query and key1",
                    (std::to_string(queryShape.GetDim(0)) + "," + std::to_string(queryShape.GetDim(1)) + "," +
                     std::to_string(queryShape.GetDim(4)) + " and " + std::to_string(keyShape.GetDim(0)) + "," +
                     std::to_string(keyShape.GetDim(1)) + "," + std::to_string(keyShape.GetDim(4)))
                        .c_str(),
                    "query and key1 must have the same dims 0, 1 and 4"),
                return ge::GRAPH_FAILED);

    // issue #7529: dim 2 (G) mismatch used to be silently ignored, leaving part of dqOut unwritten.
    // Only warn here to keep compatibility with existing callers.
    if (queryShape.GetDim(2) != keyShape.GetDim(2)) {
        OP_LOGW(context->GetNodeName(),
                "query dim 2 (G=%ld) differs from key1 dim 2 (G=%ld), part of dqOut will not be written.",
                queryShape.GetDim(2), keyShape.GetDim(2));
    }

    // issue #7530: key2 must be [B, N2, S2, S1, D] derived from query/key1; mismatch used to leave part of
    // dk2Out/dv2Out unwritten. Only warn here to keep compatibility with existing callers.
    auto& key2Shape = context->GetInputShape(KEY2_INPUT_INDEX)->GetStorageShape();
    if (key2Shape.GetDim(0) != queryShape.GetDim(0) || key2Shape.GetDim(1) != queryShape.GetDim(1) ||
        key2Shape.GetDim(4) != queryShape.GetDim(4) || key2Shape.GetDim(2) != keyShape.GetDim(3) ||
        key2Shape.GetDim(3) != queryShape.GetDim(3)) {
        OP_LOGW(context->GetNodeName(),
                "key2 shape [%ld,%ld,%ld,%ld,%ld] does not match [B, N2, S2, S1, D] derived from query/key1, "
                "part of dk2Out/dv2Out will not be written.",
                key2Shape.GetDim(0), key2Shape.GetDim(1), key2Shape.GetDim(2), key2Shape.GetDim(3),
                key2Shape.GetDim(4));
    }

    return ge::SUCCESS;
}

static ge::graphStatus CheckOutputShapes(gert::TilingContext* context)
{
    constexpr uint32_t inputIndices[] = {QUERY_INPUT_INDEX, KEY1_INPUT_INDEX, VALUE1_INPUT_INDEX, KEY2_INPUT_INDEX,
                                         VALUE2_INPUT_INDEX};
    const char* inputNames[] = {"query", "key1", "value1", "key2", "value2"};
    const char* outputNames[] = {"dq", "dk1", "dv1", "dk2", "dv2"};
    for (uint32_t i = 0; i < sizeof(inputIndices) / sizeof(inputIndices[0]); ++i) {
        const auto* input = context->GetInputShape(inputIndices[i]);
        const auto* output = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, input);
        OP_CHECK_NULL_WITH_CONTEXT(context, output);
        const auto& expected = input->GetStorageShape();
        const auto& actual = output->GetStorageShape();
        OP_CHECK_IF(actual.GetDimNum() != expected.GetDimNum(),
                    OP_LOGE(context, "%s rank is invalid: got %zu, expected %zu to match %s.", outputNames[i],
                            actual.GetDimNum(), expected.GetDimNum(), inputNames[i]),
                    return ge::GRAPH_FAILED);
        for (uint64_t dim = 0; dim < expected.GetDimNum(); ++dim) {
            OP_CHECK_IF(actual.GetDim(dim) != expected.GetDim(dim),
                        OP_LOGE(context, "%s shape is invalid at dimension %zu: got %ld, expected %ld to match %s.",
                                outputNames[i], dim, actual.GetDim(dim), expected.GetDim(dim), inputNames[i]),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParams(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FusedFloydAttentionGrad", "context is unavailable"})),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrs(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "invalid attrs"), return ge::GRAPH_FAILED);
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY1_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE1_INPUT_INDEX) != nullptr && context->GetInputShape(KEY2_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE2_INPUT_INDEX) != nullptr && context->GetInputShape(DY_INPUT_INDEX) != nullptr &&
        context->GetOptionalInputShape(SOFTMAX_MAX) != nullptr &&
        context->GetOptionalInputShape(SOFTMAX_SUM) != nullptr &&
        context->GetOptionalInputShape(ATTENTION_IN) != nullptr) {
        if (CheckOutputShapes(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CheckBaseInput(context) == ge::GRAPH_SUCCESS) {
            return ge::SUCCESS;
        }
        return ge::GRAPH_FAILED;
    }
    const char* missingInput = context->GetInputShape(QUERY_INPUT_INDEX) == nullptr   ? "query" :
                               context->GetInputShape(KEY1_INPUT_INDEX) == nullptr    ? "key1" :
                               context->GetInputShape(VALUE1_INPUT_INDEX) == nullptr  ? "value1" :
                               context->GetInputShape(KEY2_INPUT_INDEX) == nullptr    ? "key2" :
                               context->GetInputShape(VALUE2_INPUT_INDEX) == nullptr  ? "value2" :
                               context->GetInputShape(DY_INPUT_INDEX) == nullptr      ? "dy" :
                               context->GetOptionalInputShape(SOFTMAX_MAX) == nullptr ? "softmax_max" :
                               context->GetOptionalInputShape(SOFTMAX_SUM) == nullptr ? "softmax_sum" :
                                                                                        "attention_in";
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), missingInput,
                                             "input shape is missing; this input is required by the current template");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFusedFloydAttentionGrad(gert::TilingContext* context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyOutput(context)) {
        FusedFloydAttentionGradTiling FusedFloydAttentionGradTiling;
        return FusedFloydAttentionGradTiling.RunEmptyTiling(context);
    } else {
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFusedFloydAttentionGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"FusedFloydAttentionGrad", "context is unavailable"})),
        return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();

    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreGradCompileInfo>();

    if (platformInfoPtr == nullptr || compileInfoPtr == nullptr) {
        // platform/compile metadata loading failure at tiling-prepare stage
        std::string reason = "platform info or compile info is unavailable at tiling prepare";
        const std::vector<const char*> msgKey = {"opName", "reason"};
        const std::vector<const char*> msgValue = {context->GetNodeName(), reason.c_str()};
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "platformInfoPtr or compileInfoPtr is null.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    OP_LOGI(context,
            "parse TilingParseContext succ. aivNum:%u, aicNum:%u, "
            "ubSize:%lu, l1Size:%lu, l0aSize:%lu, l0bSize:%lu, l0cSize:%lu, l2CacheSize:%lu",
            compileInfoPtr->aivNum, compileInfoPtr->aicNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
            compileInfoPtr->l0aSize, compileInfoPtr->l0bSize, compileInfoPtr->l0cSize, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedFloydAttentionGrad)
    .Tiling(TilingFusedFloydAttentionGrad)
    .TilingParse<FlashAttentionScoreGradCompileInfo>(TilingPrepareForFusedFloydAttentionGrad); // 向框架注册入口函数

} // namespace FFAG
} // namespace optiling
