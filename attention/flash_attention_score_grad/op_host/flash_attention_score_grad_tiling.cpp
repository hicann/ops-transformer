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
 * \file flash_attention_score_grad_tiling.cpp
 * \brief
 */

#include "../op_kernel/flash_attention_score_grad_tiling.h"
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
constexpr uint32_t OUTPUT_IDX_DQ = 0;
constexpr uint32_t OUTPUT_IDX_DK = 1;
constexpr uint32_t OUTPUT_IDX_DV = 2;
constexpr uint32_t OUTPUT_IDX_DPSE = 3;
constexpr uint32_t OUTPUT_IDX_DQ_ROPE = 4;
constexpr uint32_t OUTPUT_IDX_DK_ROPE = 5;

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t DY_INPUT_INDEX = 3;
constexpr uint32_t SOFTMAX_MAX = 8;
constexpr uint32_t SOFTMAX_SUM = 9;
constexpr uint32_t ATTENTION_IN = 11;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 22;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 23;

constexpr uint32_t HEAD_NUM_IDX = 4;
constexpr uint32_t LAYOUT_ATTR_IDX = 5;

constexpr uint32_t FAG_EMPTY_TILING_KEY = 90;
constexpr uint32_t TILING_KEY_1 = 1U;
constexpr size_t WORKSPACE_SIZE = 100 * 1024 * 1024;

static uint32_t CalculateTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0U || aivCoreNum == 0U || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

// tiling func + tiling prepare
class FlashAttentionScoreGradTiling {
public:
    FlashAttentionScoreGradTilingData tilingData;
    FlashAttentionScoreGradTiling(){};

    ge::graphStatus RunEmptyTiling(gert::TilingContext *context)
    {
        uint64_t aicNum = 40; // 40: B3 default aicNum
        uint64_t aivNum = 20; // 20: B3 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
            OP_CHECK_IF(compilePtr == nullptr, OP_LOGE(context, "compile_info is null"),
                       return ge::GRAPH_FAILED);
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        OP_CHECK_IF(aivNum == 0, OP_LOGE("flash_attention_score_grad", "num of aiv is 0."),
                   return GRAPH_FAILED);
        uint64_t dqNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DQ)->GetStorageShape().GetShapeSize());
        if (dqNum % aivNum == 0ULL) {
            tilingData.emptyTensorTilingData.set_formerDqNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDqNum(dqNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(dqNum / aivNum);
        }
        uint64_t dkNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DK)->GetStorageShape().GetShapeSize());
        if (dkNum % aivNum == 0ULL) {
            tilingData.emptyTensorTilingData.set_formerDkNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDkNum(dkNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(dkNum / aivNum);
        }
        uint64_t dvNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DV)->GetStorageShape().GetShapeSize());
        if (dvNum % aivNum == 0ULL) {
            tilingData.emptyTensorTilingData.set_formerDvNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDvNum(dvNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDvNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDvNum(dvNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDvNum(dvNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDvNum(dvNum / aivNum);
        }
        const gert::StorageShape *dpseShape = context->GetOutputShape(OUTPUT_IDX_DPSE);
        uint64_t dpseNum = (dpseShape == nullptr) ? 0 : static_cast<uint64_t>(dpseShape->GetStorageShape().GetShapeSize());
        if (dpseNum % aivNum == 0ULL) {
            tilingData.emptyTensorTilingData.set_formerDpseNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDpseNum(dpseNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDpseNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDpseNum(dpseNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDpseNum(dpseNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDpseNum(dpseNum / aivNum);
        }

        context->SetTilingKey(FAG_EMPTY_TILING_KEY);
        auto sliceNum =
            (dqNum < aivNum && dkNum < aivNum && dpseNum < aivNum) ? std::max(std::max(dqNum, dkNum), dpseNum) : aivNum;
        context->SetBlockDim(CalculateTschBlockDim(sliceNum, aicNum, aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = WORKSPACE_SIZE;
        return ge::GRAPH_SUCCESS;
    }
};

static bool IsEmptyOutput(gert::TilingContext *context)
{
    const gert::StorageShape *dqShape = context->GetOutputShape(OUTPUT_IDX_DQ);
    const gert::StorageShape *dqRopeShape = context->GetOutputShape(OUTPUT_IDX_DQ_ROPE);
    const gert::StorageShape *dkShape = context->GetOutputShape(OUTPUT_IDX_DK);
    const gert::StorageShape *dkRopeShape = context->GetOutputShape(OUTPUT_IDX_DK_ROPE);
    const gert::StorageShape *dvShape = context->GetOutputShape(OUTPUT_IDX_DV);
    if (dqShape != nullptr && dkShape != nullptr && dvShape != nullptr) {
        if (dqShape->GetStorageShape().GetShapeSize() == 0 || dkShape->GetStorageShape().GetShapeSize() == 0 ||
            dvShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
    }
    if (dqRopeShape != nullptr && dkRopeShape != nullptr) {
        if (dqRopeShape->GetStorageShape().GetShapeSize() == 0 || dkRopeShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    size_t *workspaces = context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepProbPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, preTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, nextTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, n1SizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckBaseInput(gert::TilingContext *context){
    auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto &keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto &valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
    int64_t headNum = *context->GetAttrs()->GetAttrPointer<int>(HEAD_NUM_IDX);
    OP_CHECK_IF(headNum == 0,
               OP_LOGE(context, "headNum is 0."),
               return ge::GRAPH_FAILED);
    const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (strlen(inputLayout) == 3) { // 3: BSH or SBH or TND
        if (inputLayout[0] == 'B') {
            // layout is BSH
            OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                OP_LOGE(context, "query or key shape is invalid"),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(queryShape.GetDim(2) % headNum != 0,
               OP_LOGE(context, "h1 [%ld] should be a multiple of headNum [%ld].",
               queryShape.GetDim(2), headNum),
               return ge::GRAPH_FAILED);
        } else if (inputLayout[0] == 'T') { // TND  N1 != N2
            OP_CHECK_IF(headNum != queryShape.GetDim(1),
               OP_LOGE(context, "headNum is [%ld], but got n1 [%ld].",
               headNum, queryShape.GetDim(1)),
               return ge::GRAPH_FAILED);
            return ge::SUCCESS;
        } else {
            // layout is SBH
            OP_CHECK_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                OP_LOGE(context, "query or key shape is invalid"),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(queryShape.GetDim(2) % headNum != 0,
               OP_LOGE(context, "h1 [%ld] should be a multiple of headNum [%ld].",
               queryShape.GetDim(2), headNum),
               return ge::GRAPH_FAILED);
        }
        // kD < vD
        OP_CHECK_IF((keyShape.GetDim(2) < valueShape.GetDim(2)),
            OP_LOGE(context, "key or value shape is invalid"),
            return ge::GRAPH_FAILED);
    } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
        OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
            OP_LOGE(context, "query or key shape is invalid"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
            OP_LOGE(context, "query or key shape is invalid"),
            return ge::GRAPH_FAILED);
        if (inputLayout[1] == 'N') {
            OP_CHECK_IF(headNum != queryShape.GetDim(1),
                   OP_LOGE(context, "headNum is [%ld], but got n1 [%ld].",
                   headNum, queryShape.GetDim(1)),
                   return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(headNum != queryShape.GetDim(2),
                   OP_LOGE(context, "headNum is [%ld], but got n1 [%ld].",
                   headNum, queryShape.GetDim(2)),
                   return ge::GRAPH_FAILED);  
        }
        OP_CHECK_IF((keyShape.GetDim(3) < valueShape.GetDim(3)),
            OP_LOGE(context, "key or value shape is invalid"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "invalid input_layout[%s].", inputLayout);
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

static ge::graphStatus CheckParams(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OP_LOGE("CheckParams", "context is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrs(context) != ge::GRAPH_SUCCESS,
               OP_LOGE(context->GetNodeName(), "invalid attrs"), return ge::GRAPH_FAILED);
    if ((context->GetInputShape(QUERY_ROPE_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_ROPE_INPUT_INDEX) == nullptr) ||
        (context->GetInputShape(QUERY_ROPE_INPUT_INDEX) == nullptr && context->GetInputShape(KEY_ROPE_INPUT_INDEX) != nullptr)) {
        OP_LOGE(context, "input shape Query Rope and Key Rope must be either both defined or both undefined.");
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetInputShape(DY_INPUT_INDEX) != nullptr &&
        context->GetOptionalInputShape(SOFTMAX_MAX) != nullptr &&
        context->GetOptionalInputShape(SOFTMAX_SUM) != nullptr &&
        context->GetOptionalInputShape(ATTENTION_IN) != nullptr) {
        if (CheckBaseInput(context) == ge::GRAPH_SUCCESS) {
            return ge::SUCCESS;
        }
    }
    OP_LOGE(context, "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_IF(compilePtr == nullptr, OP_LOGE(context, "compile_info is null"),
               return ge::GRAPH_FAILED);
    auto socVersion = compilePtr->socVersion;
    if (IsEmptyOutput(context)) {
        FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
        return flashAttentionScoreGradTiling.RunEmptyTiling(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScoreGrad(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OP_LOGE("TilingPrepare", "context is null."), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null."),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context, "compileInfoPtr is null."),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
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

IMPL_OP(FlashAttentionScoreGrad)
    .Tiling(TilingFlashAttentionGradScore)
    .TilingInputsDataDependency({12, 13, 14, 15, 16})
    .TilingParse<FlashAttentionScoreGradCompileInfo>(TilingPrepareForFlashAttentionScoreGrad); // 向框架注册入口函数

} // namespace optiling
