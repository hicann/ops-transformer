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
 * \file moe_compute_expert_tokens_tiling.cpp
 * \brief
 */
#include <cmath>
#include "moe_compute_expert_tokens_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {

static const size_t SHAPE_SIZE = 1;
static const int64_t ONCE_HANDLE_COUNT = 320;
static const int64_t ONCE_RPG_MAX_COUNT = 255;
static const int64_t MAX_CORE_NUM = 64;
static const int64_t SYS_WORKSPACE = static_cast<int64_t>(16 * 1024 * 1024);

static const int64_t E_BOUND_NUM = 256;
static const int64_t BSK_BOUND_NUM = 12000;
static const int64_t NET_E_BOUND_NUM = 8;
static const int64_t NET_BSK_BOUND_NUM = 65536;
static const int64_t EXTRA_WORKSAPCE = 32;
static const int64_t E_BOUND_NUM_UP_LIMIT = 2048;
static const int64_t BSK_BOUND_NUM_UP_LIMIT = std::pow(2, 24);

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return value;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

inline static int64_t CalcWorkLocal(int64_t handleNum)
{
    int64_t elementsPerBlock = static_cast<int64_t>(32 / sizeof(float));
    int64_t elementsPerRepeat = static_cast<int64_t>(256 / sizeof(float));

    int64_t firstMaxRepeat = RoundUp(handleNum, elementsPerRepeat);
    int64_t iter1OutputCount = firstMaxRepeat * 2;
    int64_t iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;
    int64_t iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;
    int64_t iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock;
    int64_t iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;

    int64_t iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;
    int64_t finalWorkLocalNeedSize = iter2AlignStart + iter3AlignStart + iter3AlignEnd;
    return finalWorkLocalNeedSize;
}

inline static ge::graphStatus MoeComputeExpertTokensSetTilingData(gert::TilingContext *context,
                                                                  MoeComputeExpertTokensTilingData &tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static bool CheckParamsShape(const gert::TilingContext *context)
{
    auto sortedExpertShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedExpertShapePtr);
    auto sortedExpertShape = sortedExpertShapePtr->GetStorageShape();

    auto outShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    auto outShape = outShapePtr->GetStorageShape();

    auto sortedExpertTypePtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedExpertTypePtr);
    auto sortedExpertType = sortedExpertTypePtr->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(sortedExpertType);

    auto attrs = context->GetAttrs();
    auto numOfExpert = *(attrs->GetAttrPointer<int64_t>(0));

    std::string sortedExpertDimNum = std::to_string(sortedExpertShape.GetDimNum());
    OP_CHECK_IF(sortedExpertShape.GetDimNum() != SHAPE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "sorted_experts",
                                             sortedExpertDimNum.c_str(), "1D"),
                return false);

    std::string outDimNum = std::to_string(outShape.GetDimNum());
    OP_CHECK_IF(outShape.GetDimNum() != SHAPE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "total_rows_before_expert",
                                             outDimNum.c_str(), "1D"),
                return false);

    std::string sortedExpertDtype = Ops::Base::ToString(sortedExpertType);
    OP_CHECK_IF(sortedExpertType != ge::DT_INT32 || dtypeSize != 4,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "sorted_experts",
                                          sortedExpertDtype.c_str(), "INT32"),
                return false);

    std::string numOfExpertStr = std::to_string(numOfExpert);
    OP_CHECK_IF(numOfExpert <= 0 || numOfExpert > 2048,
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "num_experts",
                                          numOfExpertStr.c_str(), "greater than 0 and not greater than 2048"),
                return false);

    return true;
}

static inline void PrintTilingData(MoeComputeExpertTokensTilingData &tilingData)
{
    const std::string opName = "MoeComputeExpertTokens";
    OP_LOGI(opName, "[totalCoreNum]: %ld", tilingData.get_totalCoreNum());
    OP_LOGI(opName, "[usedCoreNumBefore]: %ld", tilingData.get_usedCoreNumBefore());
    OP_LOGI(opName, "[usedCoreNumBefore3]: %ld", tilingData.get_usedCoreNumBefore3());
    OP_LOGI(opName, "[usedCoreNumAfter]: %ld", tilingData.get_usedCoreNumAfter());
    OP_LOGI(opName, "[ubSize]: %ld", tilingData.get_ubSize());
    OP_LOGI(opName, "[workLocalNeedSize]: %ld", tilingData.get_workLocalNeedSize());
    OP_LOGI(opName, "[sortedExpertNum]: %ld", tilingData.get_sortedExpertNum());
    OP_LOGI(opName, "[normalCoreHandleNumBefore]: %ld", tilingData.get_normalCoreHandleNumBefore());
    OP_LOGI(opName, "[normalCoreLoopNumBefore]: %ld", tilingData.get_normalCoreLoopNumBefore());
    OP_LOGI(opName, "[normalCoreHandleNumPerLoopBefore]: %ld", tilingData.get_normalCoreHandleNumPerLoopBefore());
    OP_LOGI(opName, "[normalCoreHandleNumTailLoopBefore]: %ld", tilingData.get_normalCoreHandleNumTailLoopBefore());
    OP_LOGI(opName, "[tailCoreHandleNumBefore]: %ld", tilingData.get_tailCoreHandleNumBefore());
    OP_LOGI(opName, "[tailCoreLoopNumBefore]: %ld", tilingData.get_tailCoreLoopNumBefore());
    OP_LOGI(opName, "[tailCoreHandleNumPerLoopBefore]: %ld", tilingData.get_tailCoreHandleNumPerLoopBefore());
    OP_LOGI(opName, "[tailCoreHandleNumTailLoopBefore]: %ld", tilingData.get_tailCoreHandleNumTailLoopBefore());
    OP_LOGI(opName, "[numOfExpert]: %ld", tilingData.get_numOfExpert());
    OP_LOGI(opName, "[normalCoreHandleNumAfter]: %ld", tilingData.get_normalCoreHandleNumAfter());
    OP_LOGI(opName, "[normalCoreLoopNumAfter]: %ld", tilingData.get_normalCoreLoopNumAfter());
    OP_LOGI(opName, "[normalCoreHandleNumPerLoopAfter]: %ld", tilingData.get_normalCoreHandleNumPerLoopAfter());
    OP_LOGI(opName, "[normalCoreHandleNumTailLoopAfter]: %ld", tilingData.get_normalCoreHandleNumTailLoopAfter());
    OP_LOGI(opName, "[tailCoreHandleNumAfter]: %ld", tilingData.get_tailCoreHandleNumAfter());
    OP_LOGI(opName, "[tailCoreLoopNumAfter]: %ld", tilingData.get_tailCoreLoopNumAfter());
    OP_LOGI(opName, "[tailCoreHandleNumPerLoopAfter]: %ld", tilingData.get_tailCoreHandleNumPerLoopAfter());
    OP_LOGI(opName, "[tailCoreHandleNumTailLoopAfter]: %ld", tilingData.get_tailCoreHandleNumTailLoopAfter());
    OP_LOGI(opName, "[handleNumPerCoreBefore]: %ld", tilingData.get_handleNumPerCoreBefore());
    OP_LOGI(opName, "[handleNumTailCoreBefore]: %ld", tilingData.get_handleNumTailCoreBefore());
    OP_LOGI(opName, "[loopCountBefore]: %ld", tilingData.get_loopCountBefore());
    OP_LOGI(opName, "[loopCountTailBefore]: %ld", tilingData.get_loopCountTailBefore());
    OP_LOGI(opName, "[handleNumPerLoopBefore]: %ld", tilingData.get_handleNumPerLoopBefore());
    OP_LOGI(opName, "[handleNumTailCorePerLoopBefore]: %ld", tilingData.get_handleNumTailCorePerLoopBefore());
    OP_LOGI(opName, "[handleExpertNumLoopCount]: %ld", tilingData.get_handleExpertNumLoopCount());
    OP_LOGI(opName, "[handleExpertNumMainCorePerLoop]: %ld", tilingData.get_handleExpertNumMainCorePerLoop());
    OP_LOGI(opName, "[handleExpertNumTailCorePerLoop]: %ld", tilingData.get_handleExpertNumTailCorePerLoop());
    OP_LOGI(opName, "[handleNumTailCoreMainLoop]: %ld", tilingData.get_handleNumTailCoreMainLoop());
    OP_LOGI(opName, "[loopCountTailCoreMainLoop]: %ld", tilingData.get_loopCountTailCoreMainLoop());
    OP_LOGI(opName, "[handleNumTailCoreTailLoop]: %ld", tilingData.get_handleNumTailCoreTailLoop());
    OP_LOGI(opName, "[loopCountTailCoreTailLoop]: %ld", tilingData.get_loopCountTailCoreTailLoop());
    OP_LOGI(opName, "[userWorkspaceSize]: %ld", tilingData.get_userWorkspaceSize());
    OP_LOGI(opName, "[tilingKey]: %ld", tilingData.get_tilingKey());
}

static inline ge::graphStatus CalcNumOfExpertTiling(const gert::TilingContext *context,
                                                    MoeComputeExpertTokensTilingData &tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingComputeExpertTokens Enter CalcNumOfExpertTiling.");

    int64_t normalCoreHandleNumAfter =
        (tilingData.get_numOfExpert() + tilingData.get_totalCoreNum() - 1) / tilingData.get_totalCoreNum();
    int64_t usedCoreNumAfter =
        min(RoundUp(tilingData.get_numOfExpert(), normalCoreHandleNumAfter), tilingData.get_totalCoreNum());
    int64_t tailCoreHandleNumAfter = tilingData.get_numOfExpert() - (usedCoreNumAfter - 1) * normalCoreHandleNumAfter;

    int64_t normalCoreHandleNumPerLoopAfter =
        (normalCoreHandleNumAfter > ONCE_RPG_MAX_COUNT) ? ONCE_RPG_MAX_COUNT : normalCoreHandleNumAfter;
    int64_t normalCoreLoopNumAfter = CeilDiv(normalCoreHandleNumAfter, normalCoreHandleNumPerLoopAfter);
    int64_t normalCoreHandleNumTailLoopAfter =
        normalCoreHandleNumAfter - (normalCoreLoopNumAfter - 1) * normalCoreHandleNumPerLoopAfter;

    int64_t tailCoreHandleNumPerLoopAfter =
        (tailCoreHandleNumAfter > ONCE_RPG_MAX_COUNT) ? ONCE_RPG_MAX_COUNT : tailCoreHandleNumAfter;
    int64_t tailCoreLoopNumAfter = CeilDiv(tailCoreHandleNumAfter, tailCoreHandleNumPerLoopAfter);
    int64_t tailCoreHandleNumTailLoopAfter =
        tailCoreHandleNumAfter - (tailCoreLoopNumAfter - 1) * tailCoreHandleNumPerLoopAfter;

    tilingData.set_normalCoreHandleNumAfter(normalCoreHandleNumAfter);
    tilingData.set_normalCoreLoopNumAfter(normalCoreLoopNumAfter);
    tilingData.set_normalCoreHandleNumPerLoopAfter(normalCoreHandleNumPerLoopAfter);
    tilingData.set_normalCoreHandleNumTailLoopAfter(normalCoreHandleNumTailLoopAfter);

    tilingData.set_tailCoreHandleNumAfter(tailCoreHandleNumAfter);
    tilingData.set_tailCoreLoopNumAfter(tailCoreLoopNumAfter);
    tilingData.set_tailCoreHandleNumPerLoopAfter(tailCoreHandleNumPerLoopAfter);
    tilingData.set_tailCoreHandleNumTailLoopAfter(tailCoreHandleNumTailLoopAfter);
    tilingData.set_usedCoreNumAfter(usedCoreNumAfter);

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcSortedExpertTiling(const gert::TilingContext *context,
                                                     MoeComputeExpertTokensTilingData &tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingComputeExpertTokens Enter CalcSortedExpertTiling.");

    int64_t normalCoreHandleNumBefore =
        (tilingData.get_sortedExpertNum() + tilingData.get_totalCoreNum() - 1) / tilingData.get_totalCoreNum();
    int64_t usedCoreNumBefore =
        min(RoundUp(tilingData.get_sortedExpertNum(), normalCoreHandleNumBefore), tilingData.get_totalCoreNum());
    int64_t tailCoreHandleNumBefore =
        tilingData.get_sortedExpertNum() - (usedCoreNumBefore - 1) * normalCoreHandleNumBefore;

    int64_t normalCoreHandleNumPerLoopBefore =
        (normalCoreHandleNumBefore > ONCE_HANDLE_COUNT) ? ONCE_HANDLE_COUNT : normalCoreHandleNumBefore;
    int64_t normalCoreLoopNumBefore = CeilDiv(normalCoreHandleNumBefore, normalCoreHandleNumPerLoopBefore);
    int64_t normalCoreHandleNumTailLoopBefore =
        normalCoreHandleNumBefore - (normalCoreLoopNumBefore - 1) * normalCoreHandleNumPerLoopBefore;

    int64_t tailCoreHandleNumPerLoopBefore =
        (tailCoreHandleNumBefore > ONCE_HANDLE_COUNT) ? ONCE_HANDLE_COUNT : tailCoreHandleNumBefore;
    int64_t tailCoreLoopNumBefore = CeilDiv(tailCoreHandleNumBefore, tailCoreHandleNumPerLoopBefore);
    int64_t tailCoreHandleNumTailLoopBefore =
        tailCoreHandleNumBefore - (tailCoreLoopNumBefore - 1) * tailCoreHandleNumPerLoopBefore;

    tilingData.set_normalCoreHandleNumBefore(normalCoreHandleNumBefore);
    tilingData.set_normalCoreLoopNumBefore(normalCoreLoopNumBefore);
    tilingData.set_normalCoreHandleNumPerLoopBefore(normalCoreHandleNumPerLoopBefore);
    tilingData.set_normalCoreHandleNumTailLoopBefore(normalCoreHandleNumTailLoopBefore);

    tilingData.set_tailCoreHandleNumBefore(tailCoreHandleNumBefore);
    tilingData.set_tailCoreLoopNumBefore(tailCoreLoopNumBefore);
    tilingData.set_tailCoreHandleNumPerLoopBefore(tailCoreHandleNumPerLoopBefore);
    tilingData.set_tailCoreHandleNumTailLoopBefore(tailCoreHandleNumTailLoopBefore);
    tilingData.set_usedCoreNumBefore(usedCoreNumBefore);

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcTemplate3ParamTiling(const gert::TilingContext *context,
                                                       MoeComputeExpertTokensTilingData &tilingData)
{
    OP_LOGD(context->GetNodeName(), "TilingComputeExpertTokens Enter CalcTemplate3ParamTiling.");

    int64_t handleNumPerCoreBefore =
        CeilDiv(CeilDiv(tilingData.get_sortedExpertNum(), tilingData.get_totalCoreNum()), ONCE_HANDLE_COUNT) *
        ONCE_HANDLE_COUNT;
    int64_t loopCountBefore = CeilDiv(handleNumPerCoreBefore, ONCE_HANDLE_COUNT);
    int64_t handleNumPerLoopBefore = ONCE_HANDLE_COUNT;
    int64_t usedCoreNumBefore3 =
        min(RoundUp(tilingData.get_sortedExpertNum(), handleNumPerCoreBefore), tilingData.get_totalCoreNum());
    int64_t handleNumTailCoreBefore =
        tilingData.get_sortedExpertNum() - (usedCoreNumBefore3 - 1) * handleNumPerCoreBefore;
    int64_t loopCountTailBefore = 1;
    int64_t handleNumTailCorePerLoopBefore = handleNumTailCoreBefore;

    int64_t handleNumTailCoreMainLoop =
        (handleNumTailCoreBefore > ONCE_HANDLE_COUNT) ? ONCE_HANDLE_COUNT : handleNumTailCoreBefore;
    int64_t loopCountTailCoreMainLoop =
        (handleNumTailCoreBefore > ONCE_HANDLE_COUNT) ? handleNumTailCoreBefore / ONCE_HANDLE_COUNT : 1;
    int64_t handleNumTailCoreTailLoop = handleNumTailCoreBefore - loopCountTailCoreMainLoop * handleNumTailCoreMainLoop;
    int64_t loopCountTailCoreTailLoop = handleNumTailCoreTailLoop == 0 ? 0 : 1;
    int64_t handleExpertNumMainCorePerLoop =
        (tilingData.get_numOfExpert() >= E_BOUND_NUM) ? E_BOUND_NUM : tilingData.get_numOfExpert();
    int64_t handleExpertNumLoopCount = CeilDiv(tilingData.get_numOfExpert(), E_BOUND_NUM);
    int64_t handleExpertNumTailCorePerLoop =
        tilingData.get_numOfExpert() - (handleExpertNumLoopCount - 1) * handleExpertNumMainCorePerLoop;

    tilingData.set_handleNumPerCoreBefore(handleNumPerCoreBefore);
    tilingData.set_handleNumTailCoreBefore(handleNumTailCoreBefore);
    tilingData.set_loopCountBefore(loopCountBefore);
    tilingData.set_loopCountTailBefore(loopCountTailBefore);
    tilingData.set_handleNumPerLoopBefore(handleNumPerLoopBefore);
    tilingData.set_handleNumTailCorePerLoopBefore(handleNumTailCorePerLoopBefore);

    tilingData.set_handleNumTailCoreMainLoop(handleNumTailCoreMainLoop);
    tilingData.set_loopCountTailCoreMainLoop(loopCountTailCoreMainLoop);
    tilingData.set_handleNumTailCoreTailLoop(handleNumTailCoreTailLoop);
    tilingData.set_loopCountTailCoreTailLoop(loopCountTailCoreTailLoop);
    tilingData.set_handleExpertNumMainCorePerLoop(handleExpertNumMainCorePerLoop);
    tilingData.set_handleExpertNumLoopCount(handleExpertNumLoopCount);
    tilingData.set_handleExpertNumTailCorePerLoop(handleExpertNumTailCorePerLoop);
    tilingData.set_usedCoreNumBefore3(usedCoreNumBefore3);

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus InitPlatformTilingData(gert::TilingContext *context,
                                                     MoeComputeExpertTokensTilingData &tilingData)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    const int64_t effectiveCoreNum = min(totalCoreNum, MAX_CORE_NUM);
    std::string totalCoreNumStr = std::to_string(totalCoreNum);
    OP_CHECK_IF((totalCoreNum <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "total_core_num",
                                                      totalCoreNumStr.c_str(),
                                                      "The platform core num should be greater than 0."),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    std::string ubSizePlatFormStr = std::to_string(ubSizePlatForm);
    OP_CHECK_IF((ubSizePlatForm <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size",
                                                      ubSizePlatFormStr.c_str(),
                                                      "The platform UB size should be greater than 0."),
                return ge::GRAPH_FAILED);

    tilingData.set_totalCoreNum(effectiveCoreNum);
    tilingData.set_ubSize(static_cast<int64_t>(ubSizePlatForm));
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus InitInputTilingData(gert::TilingContext *context,
                                                  MoeComputeExpertTokensTilingData &tilingData)
{
    auto sortedExpertInput = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedExpertInput);
    auto sortedExpertInputShape = sortedExpertInput->GetStorageShape();
    std::string sortedExpertInputDimNum = std::to_string(sortedExpertInputShape.GetDimNum());
    OP_CHECK_IF(sortedExpertInputShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "sorted_experts",
                                             sortedExpertInputDimNum.c_str(), "1D"),
                return ge::GRAPH_FAILED);

    tilingData.set_sortedExpertNum(sortedExpertInputShape.GetDim(0));
    std::string sortedExpertNumStr = std::to_string(tilingData.get_sortedExpertNum());
    OP_CHECK_IF(tilingData.get_sortedExpertNum() > BSK_BOUND_NUM_UP_LIMIT,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "sorted_experts",
                                          sortedExpertNumStr.c_str(), "not greater than 2**24"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *numOfExpertPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, numOfExpertPtr);
    const int64_t numOfExpert = *numOfExpertPtr;
    tilingData.set_numOfExpert(numOfExpert);
    std::string numOfExpertStr = std::to_string(tilingData.get_numOfExpert());
    OP_CHECK_IF(tilingData.get_numOfExpert() <= 0 || tilingData.get_numOfExpert() > E_BOUND_NUM_UP_LIMIT,
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "num_experts",
                                          numOfExpertStr.c_str(), "greater than 0 and not greater than 2048"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcMoeComputeExpertTokensTiling(gert::TilingContext *context,
                                                               MoeComputeExpertTokensTilingData &tilingData)
{
    OP_CHECK_IF(CalcTemplate3ParamTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CalcTemplate3ParamTiling",
                                                      "GRAPH_FAILED", "CalcTemplate3ParamTiling failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CalcSortedExpertTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CalcSortedExpertTiling",
                                                      "GRAPH_FAILED", "CalcSortedExpertTiling failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CalcNumOfExpertTiling(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CalcNumOfExpertTiling",
                                                      "GRAPH_FAILED", "CalcNumOfExpertTiling failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline void SetMoeComputeExpertTokensTilingKey(gert::TilingContext *context,
                                                      MoeComputeExpertTokensTilingData &tilingData)
{
    static const int64_t SCHEDULE_MODE_BATCH = 1;
    context->SetScheduleMode(SCHEDULE_MODE_BATCH);
    bool isSortedExpertOverBound = (tilingData.get_sortedExpertNum() > BSK_BOUND_NUM);
    bool isNumOfExpertOverBound = (tilingData.get_numOfExpert() > E_BOUND_NUM);
    if (!isSortedExpertOverBound && !isNumOfExpertOverBound) {
        tilingData.set_tilingKey(COM_SCENE_FLAG_1);
    } else if (isSortedExpertOverBound && !isNumOfExpertOverBound) {
        tilingData.set_tilingKey(COM_SCENE_FLAG_2);
    } else {
        tilingData.set_tilingKey(COM_SCENE_FLAG_3);
    }

    // 特殊网络case场景
    bool isNetScene =
        (tilingData.get_sortedExpertNum() == NET_BSK_BOUND_NUM) && (tilingData.get_numOfExpert() == NET_E_BOUND_NUM);
    if (isNetScene) {
        tilingData.set_tilingKey(COM_SCENE_FLAG_1);
    }
}

static inline ge::graphStatus SetWorkspaceAndWorkLocal(gert::TilingContext *context,
                                                       MoeComputeExpertTokensTilingData &tilingData)
{
    int64_t handleNum = tilingData.get_normalCoreHandleNumBefore();
    int64_t workLocalNeedSize = CalcWorkLocal(handleNum);
    tilingData.set_workLocalNeedSize(workLocalNeedSize);

    size_t userSize = tilingData.get_numOfExpert() * tilingData.get_totalCoreNum() * sizeof(int32_t);
    size_t sysWorkspaceSize = SYS_WORKSPACE;
    size_t *userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    userWorkspaceSize[0] = userSize + sysWorkspaceSize;
    tilingData.set_userWorkspaceSize(userWorkspaceSize[0]);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus SaveTilingDataToContext(gert::TilingContext *context,
                                                      MoeComputeExpertTokensTilingData &tilingData)
{
    OP_CHECK_IF(MoeComputeExpertTokensSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tiling_data",
                                                      "save failed",
                                                      "MoeComputeExpertTokensSetTilingData failed to set tiling data."),
                return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_totalCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4MoeComputeExpertTokens(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "[MoeComputeExpertTokens] Tiling4MoeComputeExpertTokens running begin");
    OP_CHECK_IF(!CheckParamsShape(context),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckParamsShape",
                                                      "GRAPH_FAILED",
                                                      "Tiling4MoeComputeExpertTokens check shape failed."),
                return ge::GRAPH_FAILED);

    MoeComputeExpertTokensTilingData tilingData;
    if (InitPlatformTilingData(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (InitInputTilingData(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CalcMoeComputeExpertTokensTiling(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetMoeComputeExpertTokensTilingKey(context, tilingData);
    if (SetWorkspaceAndWorkLocal(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SaveTilingDataToContext(context, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMoeComputeExpertTokens(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeComputeExpertTokens)
    .Tiling(Tiling4MoeComputeExpertTokens)
    .TilingParse<MoeComputeExpertTokensCompileInfo>(TilingPrepareForMoeComputeExpertTokens);

} // namespace optiling
