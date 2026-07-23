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
 * \file flash_attention_score_grad_tiling.cpp
 * \brief
 */

#include "../op_kernel/arch22/flash_attention_score_grad_tiling.h"
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/arch35/flash_attention_score_grad_template_tiling_key.h"
#include "../op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h"

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
constexpr uint32_t PSE_SHIFT_INPUT_INDEX = 4;
constexpr uint32_t PREFIX_INPUT_INDEX = 12;
constexpr uint32_t ACTUAL_SEQ_Q_LEN_INPUT_INDEX = 13;
constexpr uint32_t ACTUAL_SEQ_KV_LEN_INPUT_INDEX = 14;
constexpr uint32_t QUERY_ROPE_INPUT_INDEX = 22;
constexpr uint32_t KEY_ROPE_INPUT_INDEX = 23;

constexpr uint32_t HEAD_NUM_IDX = 4;
constexpr uint32_t LAYOUT_ATTR_IDX = 5;

constexpr int64_t T_MAX_LIMIT = 1000000;
constexpr int64_t N_MAX_LIMIT = 256;
constexpr int64_t D_MAX_LIMIT = 768;
constexpr int64_t ACTUAL_SEQ_Q_LEN_MAX_LIMIT = 2048;
constexpr int64_t ACTUAL_SEQ_Q_LEN_PREFIX_MAX_LIMIT = 1000;

constexpr uint32_t FAG_EMPTY_TILING_KEY = 0;
constexpr uint32_t TILING_KEY_1 = 1U;
constexpr size_t WORKSPACE_SIZE = 100 * 1024 * 1024;
constexpr uint64_t SMLEST_ROPE_SIZE = 64;

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

    ge::graphStatus RunEmptyTiling(gert::TilingContext *context)
    {
        FlashAttentionScoreGradTilingData *tilingData = context->GetTilingData<FlashAttentionScoreGradTilingData>();
        uint64_t aicNum = 40; // 40: B3 default aicNum
        uint64_t aivNum = 20; // 20: B3 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
            OP_CHECK_IF(compilePtr == nullptr, OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [compile_info is null]"),
                       return ge::GRAPH_FAILED);
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        OP_CHECK_IF(aivNum == 0, OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [num of aiv is 0]"),
                   return GRAPH_FAILED);
        uint64_t dqNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DQ)->GetStorageShape().GetShapeSize());
        if (dqNum % aivNum == 0ULL) {
            tilingData->emptyTensorTilingData.set_formerDqNum(aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum);
            tilingData->emptyTensorTilingData.set_tailCoreDqNum(0);
        } else {
            tilingData->emptyTensorTilingData.set_formerDqNum(dqNum % aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum + 1);
            tilingData->emptyTensorTilingData.set_tailCoreDqNum(dqNum / aivNum);
        }
        uint64_t dkNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DK)->GetStorageShape().GetShapeSize());
        if (dkNum % aivNum == 0ULL) {
            tilingData->emptyTensorTilingData.set_formerDkNum(aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum);
            tilingData->emptyTensorTilingData.set_tailCoreDkNum(0);
        } else {
            tilingData->emptyTensorTilingData.set_formerDkNum(dkNum % aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum + 1);
            tilingData->emptyTensorTilingData.set_tailCoreDkNum(dkNum / aivNum);
        }
        uint64_t dvNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DV)->GetStorageShape().GetShapeSize());
        if (dvNum % aivNum == 0ULL) {
            tilingData->emptyTensorTilingData.set_formerDvNum(aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDvNum(dvNum / aivNum);
            tilingData->emptyTensorTilingData.set_tailCoreDvNum(0);
        } else {
            tilingData->emptyTensorTilingData.set_formerDvNum(dvNum % aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDvNum(dvNum / aivNum + 1);
            tilingData->emptyTensorTilingData.set_tailCoreDvNum(dvNum / aivNum);
        }
        const gert::StorageShape *dpseShape = context->GetOutputShape(OUTPUT_IDX_DPSE);
        uint64_t dpseNum = (dpseShape == nullptr) ? 0 : static_cast<uint64_t>(dpseShape->GetStorageShape().GetShapeSize());
        if (dpseNum % aivNum == 0ULL) {
            tilingData->emptyTensorTilingData.set_formerDpseNum(aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDpseNum(dpseNum / aivNum);
            tilingData->emptyTensorTilingData.set_tailCoreDpseNum(0);
        } else {
            tilingData->emptyTensorTilingData.set_formerDpseNum(dpseNum % aivNum);
            tilingData->emptyTensorTilingData.set_singleCoreDpseNum(dpseNum / aivNum + 1);
            tilingData->emptyTensorTilingData.set_tailCoreDpseNum(dpseNum / aivNum);
        }

        context->SetTilingKey(FAG_EMPTY_TILING_KEY);
        auto sliceNum =
            (dqNum < aivNum && dkNum < aivNum && dpseNum < aivNum) ? std::max(std::max(dqNum, dkNum), dpseNum) : aivNum;
        context->SetBlockDim(CalculateTschBlockDim(sliceNum, aicNum, aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = WORKSPACE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RunEmptyTilingRegbase(gert::TilingContext *context)
    {
        fag::FlashAttentionScoreGradEmptyTensorTilingDataRegbase* emptyTensorTilingDataRegbase = context->GetTilingData<fag::FlashAttentionScoreGradEmptyTensorTilingDataRegbase>();
        uint64_t aicNum = 32; // 32: A5 default aicNum
        uint64_t aivNum = 64; // 64: A5 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
            OP_CHECK_IF(compilePtr == nullptr, OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [compile_info is null]"),
                       return ge::GRAPH_FAILED);
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        OP_CHECK_IF(aivNum == 0, OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [num of aiv is 0]"),
                   return GRAPH_FAILED);
        uint64_t dqNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DQ)->GetStorageShape().GetShapeSize());
        if (dqNum % aivNum == 0ULL) {
            emptyTensorTilingDataRegbase->set_formerDqNum(aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDqNum(dqNum / aivNum);
            emptyTensorTilingDataRegbase->set_tailCoreDqNum(0);
        } else {
            emptyTensorTilingDataRegbase->set_formerDqNum(dqNum % aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDqNum(dqNum / aivNum + 1);
            emptyTensorTilingDataRegbase->set_tailCoreDqNum(dqNum / aivNum);
        }
        uint64_t dkNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DK)->GetStorageShape().GetShapeSize());
        if (dkNum % aivNum == 0ULL) {
            emptyTensorTilingDataRegbase->set_formerDkNum(aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDkNum(dkNum / aivNum);
            emptyTensorTilingDataRegbase->set_tailCoreDkNum(0);
        } else {
            emptyTensorTilingDataRegbase->set_formerDkNum(dkNum % aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDkNum(dkNum / aivNum + 1);
            emptyTensorTilingDataRegbase->set_tailCoreDkNum(dkNum / aivNum);
        }
        uint64_t dvNum = static_cast<uint64_t>(context->GetOutputShape(OUTPUT_IDX_DV)->GetStorageShape().GetShapeSize());
        if (dvNum % aivNum == 0ULL) {
            emptyTensorTilingDataRegbase->set_formerDvNum(aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDvNum(dvNum / aivNum);
            emptyTensorTilingDataRegbase->set_tailCoreDvNum(0);
        } else {
            emptyTensorTilingDataRegbase->set_formerDvNum(dvNum % aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDvNum(dvNum / aivNum + 1);
            emptyTensorTilingDataRegbase->set_tailCoreDvNum(dvNum / aivNum);
        }
        const gert::StorageShape *dpseShape = context->GetOutputShape(OUTPUT_IDX_DPSE);
        uint64_t dpseNum = (dpseShape == nullptr) ? 0 : static_cast<uint64_t>(dpseShape->GetStorageShape().GetShapeSize());
        if (dpseNum % aivNum == 0ULL) {
            emptyTensorTilingDataRegbase->set_formerDpseNum(aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDpseNum(dpseNum / aivNum);
            emptyTensorTilingDataRegbase->set_tailCoreDpseNum(0);
        } else {
            emptyTensorTilingDataRegbase->set_formerDpseNum(dpseNum % aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDpseNum(dpseNum / aivNum + 1);
            emptyTensorTilingDataRegbase->set_tailCoreDpseNum(dpseNum / aivNum);
        }

        // rope场景清零的处理
        const gert::StorageShape *dqRopeShape = context->GetOutputShape(OUTPUT_IDX_DQ_ROPE);
        const gert::StorageShape *dkRopeShape = context->GetOutputShape(OUTPUT_IDX_DK_ROPE);
        uint64_t dqRopeNum = (dqRopeShape == nullptr) ? 0 : static_cast<uint64_t>(dqRopeShape->GetStorageShape().GetShapeSize());
        dqRopeNum = (dqRopeNum < SMLEST_ROPE_SIZE) ? 0 : dqRopeNum;
        if (dqRopeNum % aivNum == 0ULL) {
            emptyTensorTilingDataRegbase->set_formerDqRopeNum(aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDqRopeNum(dqRopeNum / aivNum);
            emptyTensorTilingDataRegbase->set_tailCoreDqRopeNum(0);
        } else {
            emptyTensorTilingDataRegbase->set_formerDqRopeNum(dqRopeNum % aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDqRopeNum(dqRopeNum / aivNum + 1);
            emptyTensorTilingDataRegbase->set_tailCoreDqRopeNum(dqRopeNum / aivNum);
        }

        uint64_t dkRopeNum = (dkRopeShape == nullptr) ? 0 : static_cast<uint64_t>(dkRopeShape->GetStorageShape().GetShapeSize());
        dkRopeNum = (dkRopeNum < SMLEST_ROPE_SIZE) ? 0 : dkRopeNum;
        if (dkRopeNum % aivNum == 0ULL) {
            emptyTensorTilingDataRegbase->set_formerDkRopeNum(aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDkRopeNum(dkRopeNum / aivNum);
            emptyTensorTilingDataRegbase->set_tailCoreDkRopeNum(0);
        } else {
            emptyTensorTilingDataRegbase->set_formerDkRopeNum(dkRopeNum % aivNum);
            emptyTensorTilingDataRegbase->set_singleCoreDkRopeNum(dkRopeNum / aivNum + 1);
            emptyTensorTilingDataRegbase->set_tailCoreDkRopeNum(dkRopeNum / aivNum);
        }
        emptyTensorTilingDataRegbase->isRope = (dqRopeNum > 0 || dkRopeNum > 0);

        context->SetTilingKey(GET_TPL_TILING_KEY(TILING_KEY_1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, TILING_KEY_1));
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

// EOD场景: TND输入下actualSeqQlen/actualSeqKvlen累加和全是0，无有效计算时，直接走空tensor模板
static bool IsTndAllSeqLenZero(gert::TilingContext *context)
{
    const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (inputLayout == nullptr || strlen(inputLayout) != 3 || inputLayout[0] != 'T') { // 3: TND
        return false;
    }
    // InputIndex::ACTUAL_SEQ_Q_LEN=13, ACTUAL_SEQ_KV_LEN=14
    auto actualSeqQlenTensor = context->GetOptionalInputTensor(13);
    auto actualSeqKvlenTensor = context->GetOptionalInputTensor(14);
    if (actualSeqQlenTensor == nullptr || actualSeqKvlenTensor == nullptr) {
        return false;
    }
    auto qShapeSize = actualSeqQlenTensor->GetShapeSize();
    auto kvShapeSize = actualSeqKvlenTensor->GetShapeSize();
    if (qShapeSize == 0 || kvShapeSize == 0 || qShapeSize != kvShapeSize) {
        return false;
    }
    const int64_t *qValue = actualSeqQlenTensor->GetData<int64_t>();
    const int64_t *kvValue = actualSeqKvlenTensor->GetData<int64_t>();
    if (qValue == nullptr || kvValue == nullptr) {
        return false;
    }
    // EOD场景: 累加和尾部连续为0的batch为无效数据，全部为0时走空tensor模板
    size_t realBSize = qShapeSize;
    while (realBSize > 0 && qValue[realBSize - 1] == 0 && kvValue[realBSize - 1] == 0) {
        --realBSize;
    }
    return realBSize == 0;
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
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum",
                    std::to_string(headNum).c_str(), "The value of headNum must be greater than 0"),
                return ge::GRAPH_FAILED);
    const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (strlen(inputLayout) == 3) { // 3: BSH or SBH or TND
        if (inputLayout[0] == 'B') {
            // layout is BSH
            std::string shapeMsg = "{" + Ops::Base::ToString(queryShape) + ", " + Ops::Base::ToString(keyShape) + "}";
            OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapeMsg.c_str(),
                    "When inputLayout is BSH, b axis of query must be equal to b axis of keyIn"),
                return ge::GRAPH_FAILED);
            std::string reasonMsg = "When inputLayout is BSH, h axis of query must be exactly divisible " +
                                    std::to_string(headNum);
            OP_CHECK_IF(queryShape.GetDim(2) % headNum != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query",
                    Ops::Base::ToString(queryShape).c_str(), reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
        } else if (inputLayout[0] == 'T') { // TND  N1 != N2
            std::string reasonMsg = "When inputLayout is TND, n axis of query must be equal to " +
                                    std::to_string(headNum);
            OP_CHECK_IF(headNum != queryShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query",
                    Ops::Base::ToString(queryShape).c_str(), reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
            return ge::SUCCESS;
        } else {
            // layout is SBH
            std::string shapeMsg = "{" + Ops::Base::ToString(queryShape) + ", " + Ops::Base::ToString(keyShape) + "}";
            OP_CHECK_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapeMsg.c_str(),
                    "When inputLayout is SBH, b axis of query must be equal to b axis of keyIn"),
                return ge::GRAPH_FAILED);
            std::string reasonMsg = "When inputLayout is SBH, h axis of query must be exactly divisible " +
                                    std::to_string(headNum);
            OP_CHECK_IF(queryShape.GetDim(2) % headNum != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query",
                    Ops::Base::ToString(queryShape).c_str(), reasonMsg.c_str()),
                return ge::GRAPH_FAILED);
        }
        // kD < vD
        std::string kvShapeMsg = "{" + Ops::Base::ToString(keyShape) + ", " + Ops::Base::ToString(valueShape) + "}";
        OP_CHECK_IF((keyShape.GetDim(2) < valueShape.GetDim(2)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "keyIn, value", kvShapeMsg.c_str(),
                "D axis of keyIn cannnot be less than d axis of value"),
            return ge::GRAPH_FAILED);
    } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
        std::string shapeMsg = "{" + Ops::Base::ToString(queryShape) + ", " + Ops::Base::ToString(keyShape) + "}";
        OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapeMsg.c_str(),
                "When inputLayout is BNSD or BSND, b axis of query must be equal to b axis of keyIn"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapeMsg.c_str(),
                "When inputLayout is BNSD or BSND, d axis of query must be equal to d axis of keyIn"),
            return ge::GRAPH_FAILED);
        if (inputLayout[1] == 'N') {
            std::string qReasonMsg = "When inputLayout is BNSD, n axis of query must be equal to " +
                                     std::to_string(headNum);
            OP_CHECK_IF(headNum != queryShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query",
                    Ops::Base::ToString(queryShape).c_str(), qReasonMsg.c_str()),
                return ge::GRAPH_FAILED);
        } else {
            std::string qReasonMsg = "When inputLayout is BSND, n axis of query must be equal to " +
                                     std::to_string(headNum);
            OP_CHECK_IF(headNum != queryShape.GetDim(2),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query",
                    Ops::Base::ToString(queryShape).c_str(), qReasonMsg.c_str()),
                return ge::GRAPH_FAILED);
        }
        std::string kvShapeMsg = "{" + Ops::Base::ToString(keyShape) + ", " + Ops::Base::ToString(valueShape) + "}";
        OP_CHECK_IF((keyShape.GetDim(3) < valueShape.GetDim(3)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "keyIn, value", kvShapeMsg.c_str(),
                "When inputLayout is BNSD, d axis of keyIn cannot be less than d axis of value"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout,
            "BSH or SBH or TND or BSND or BNSD");
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

static ge::graphStatus CheckParams(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [context is null]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrs(context) != ge::GRAPH_SUCCESS,
               OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [invalid attrs]"), return ge::GRAPH_FAILED);
    if ((context->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX) != nullptr && context->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX) == nullptr) ||
        (context->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX) == nullptr && context->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX) != nullptr)) {
        auto &qRShapePtr = context->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX)->GetStorageShape();
        auto &kRShapePtr = context->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX)->GetStorageShape();
        std::string qRShape = context->GetOptionalInputShape(QUERY_ROPE_INPUT_INDEX) != nullptr ?
                              Ops::Base::ToString(qRShapePtr) : "null";
        std::string kRShape = context->GetOptionalInputShape(KEY_ROPE_INPUT_INDEX) != nullptr ?
                              Ops::Base::ToString(kRShapePtr) : "null";
        std::string qkRShape = "{" + qRShape + ", " + kRShape + "}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "queryRopeOptional, keyRopeOptional",
            qkRShape.c_str(), "If queryRopeOptional is an empty tensor, keyRopeOptional must also be and empty tensor");
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
    } else {
        OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad",
            "query or keyIn or value or dy or softmaxMaxOptional or softmaxSumOptional or attentionInOptional");
    }
    OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, "
            "the reason is: [fail to get shape or attr from context]");
    return ge::GRAPH_FAILED;
}

static void LogConstraintWarnings(gert::TilingContext *context)
{
    const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(LAYOUT_ATTR_IDX);
    if (inputLayout == nullptr || strlen(inputLayout) != 3 || inputLayout[0] != 'T') {
        return;
    }

    auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    int64_t tVal = queryShape.GetDim(0);
    int64_t nVal = queryShape.GetDim(1);
    int64_t dVal = queryShape.GetDim(2);

    if (tVal > T_MAX_LIMIT) {
        OP_LOGW(context, "In op [FlashAttentionScoreGrad], T value %ld exceeds the recommended range [1, %ld], "
                "which may cause aicore timeout or trap error. Consider axis splitting for large computation.",
                tVal, T_MAX_LIMIT);
    }
    if (nVal > N_MAX_LIMIT) {
        OP_LOGW(context, "In op [FlashAttentionScoreGrad], N value %ld exceeds the recommended range [1, %ld].",
                nVal, N_MAX_LIMIT);
    }
    if (dVal > D_MAX_LIMIT) {
        OP_LOGW(context, "In op [FlashAttentionScoreGrad], D value %ld exceeds the recommended range [1, %ld].",
                dVal, D_MAX_LIMIT);
    }

    auto actualSeqQLenTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN_INPUT_INDEX);
    auto pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INPUT_INDEX);

    if (actualSeqQLenTensor != nullptr && actualSeqQLenTensor->GetData<int64_t>() != nullptr &&
        actualSeqQLenTensor->GetShapeSize() > 0) {
        int64_t seqQLenSize = static_cast<int64_t>(actualSeqQLenTensor->GetShapeSize());

        auto prefixTensor = context->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
        bool hasPrefix = (prefixTensor != nullptr && prefixTensor->GetShapeSize() > 0);

        if (seqQLenSize > ACTUAL_SEQ_Q_LEN_MAX_LIMIT) {
            OP_LOGW(context, "In op [FlashAttentionScoreGrad], actualSeqQLenOptional length %ld exceeds "
                    "the supported range [1, %ld].",
                    seqQLenSize, ACTUAL_SEQ_Q_LEN_MAX_LIMIT);
        }

        if (hasPrefix && seqQLenSize > ACTUAL_SEQ_Q_LEN_PREFIX_MAX_LIMIT) {
            OP_LOGW(context, "In op [FlashAttentionScoreGrad], when prefixOptional exists, "
                    "actualSeqQLenOptional length %ld exceeds the max supported value %ld.",
                    seqQLenSize, ACTUAL_SEQ_Q_LEN_PREFIX_MAX_LIMIT);
        }
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LogConstraintWarnings(context);
    auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_IF(compilePtr == nullptr, OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [compile_info is null]"),
               return ge::GRAPH_FAILED);
    auto npuArch = compilePtr->npuArch;
    if (npuArch == NpuArch::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
        if (IsEmptyOutput(context)) {
            FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
            return flashAttentionScoreGradTiling.RunEmptyTilingRegbase(context);
        }
    } else {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
        if (IsEmptyOutput(context) || IsTndAllSeqLenZero(context)) {
            FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
            return flashAttentionScoreGradTiling.RunEmptyTiling(context);
        }
    }
    return TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScoreGrad(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [context is null]"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [platformInfoPtr is null]"),
        return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<FlashAttentionScoreGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
        OP_LOGE(context, "The op [FlashAttentionScoreGrad] received bad params, the reason is: [compileInfoPtr is null]"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
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

IMPL_OP_OPTILING(FlashAttentionScoreGrad)
    .Tiling(TilingFlashAttentionGradScore)
    .TilingInputsDataDependency({12, 13, 14, 15, 16})
    .TilingParse<FlashAttentionScoreGradCompileInfo>(TilingPrepareForFlashAttentionScoreGrad); // 向框架注册入口函数

} // namespace optiling
