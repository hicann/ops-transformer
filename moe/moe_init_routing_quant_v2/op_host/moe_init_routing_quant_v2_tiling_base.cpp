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
 * \file moe_init_routing_quant_v2_tiling_base.cpp
 * \brief
 */
#include "moe_init_routing_quant_v2_tiling_base.h"
#include "op_host/tiling_util.h"

namespace optiling {
const static int64_t TILING_KEY_DROPLESS_SORT_ONE_CORE = 10001;
const static int64_t TILING_KEY_DROPLESS_SORT_MULTI_CORE = 10002;
const static int64_t TILING_KEY_DROPLESS_HIST_REGBASE_SORT_ONE_CORE = 11001;
const static int64_t TILING_KEY_DROPLESS_HIST_REGBASE_SORT_MULTI_CORE = 11002;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE = 10011;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE = 10012;
const static int64_t TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_ONE_CORE = 11011;
const static int64_t TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_MULTI_CORE = 11012;
const static int64_t TILING_KEY_HIGH_PERFORMANCE = 20000;
const static int64_t HIST_REGBASE_MAX_EXPERT_NUM = 256;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t INDEX_INPUT_X = 0;
const static int64_t INDEX_INPUT_EXPERT_IDX = 1;
const static int64_t ATTR_ACTIVE_ROWS = 0;
const static int64_t ATTR_EXPERT_CAPACITY = 1;
const static int64_t ATTR_EXPERT_NUM = 2;
const static int64_t ATTR_DROP_PAD_MODE = 3;
const static int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
const static int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
const static int64_t OUTOUT_EXPANDED_X = 0;
const static int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
const static int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
const static int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_TOKENS_COUNT = 2;
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

#define CHECK_FAIL(context, cond, log_func) \
    do { \
        if (cond) { \
            log_func; \
            return ge::GRAPH_FAILED; \
        } \
    } while (0)

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return x <= y ? x : x % y;
}

void InnerMoeInitRoutingV2TilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "platformInfo", "nullptr", "fail to get platform info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    regBase = false;
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
        mrgSortListMaxElement = 2048; // 单次排序大小，在david上设置成2048
        regBase = true;
    }
    moeInitRoutingTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::CheckTokenCount(int64_t num, const char* tag)
{
    auto expertTokensShapePtr = context_->GetOutputShape(num);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensShapePtr);

    auto expertTokensDesc = context_->GetOutputDesc(num);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensDesc);
    auto dt = expertTokensDesc->GetDataType();
    CHECK_FAIL(context_, dt != ge::DT_INT32,
               OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), tag, Ops::Base::ToString(dt).c_str(), "INT32"));

    const gert::Shape expertTokensShape = expertTokensShapePtr->GetStorageShape();
    size_t expertTokensDimNum = expertTokensShape.GetDimNum();
    std::string dimStr = std::to_string(expertTokensDimNum) + "D";
    CHECK_FAIL(context_, expertTokensDimNum != DIM_ONE,
               OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), tag, dimStr.c_str(), "1D"));
    CHECK_FAIL(context_, expertTokensShape.GetDim(0) != expertNum,
               OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), tag,
                                                     Ops::Base::ToString(expertTokensShape).c_str(),
                                                     "The first dim should be expertNum"));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::CheckOutShape()
{
    // 获取输出shape
    auto expandedXShapePtr = context_->GetOutputShape(OUTOUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShapePtr);
    const gert::Shape expandedXShape = expandedXShapePtr->GetStorageShape();

    auto expandedRowIdxShapePtr = context_->GetOutputShape(OUTOUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr);
    const gert::Shape expandedRowIdxShape = expandedRowIdxShapePtr->GetStorageShape();

    size_t expandedXDimNum = expandedXShape.GetDimNum();
    if (dropPadMode > 0) {
        std::string dimStr = std::to_string(expandedXDimNum) + "D";
        CHECK_FAIL(context_, expandedXDimNum != DIM_THREE,
                   OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expandedX", dimStr.c_str(), "3D"));
        CHECK_FAIL(context_, expandedXShape.GetDim(0) != expertNum,
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expandedX",
                                                         Ops::Base::ToString(expandedXShape).c_str(),
                                                         "The first dim of expandedX should be expertNum"));
        CHECK_FAIL(context_, expandedXShape.GetDim(1) != expertCapacity,
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expandedX",
                                                         Ops::Base::ToString(expandedXShape).c_str(),
                                                         "The second dim of expandedX should be expertCapacity"));
        CHECK_FAIL(context_, expandedXShape.GetDim(NUM_TWO) != moeInitRoutingTilingData.get_cols(),
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expandedX",
                                                         Ops::Base::ToString(expandedXShape).c_str(),
                                                         "The third dim of expandedX should be cols"));
    } else {
        std::string dimStr = std::to_string(expandedXDimNum) + "D";
        CHECK_FAIL(context_, expandedXDimNum != DIM_TWO,
                   OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expandedX", dimStr.c_str(), "2D"));
        int64_t firstDim = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();
        firstDim = activateNum == 0 ? firstDim : std::min(firstDim, activateNum);
        CHECK_FAIL(context_, expandedXShape.GetDim(0) != firstDim,
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expandedX",
                                                         Ops::Base::ToString(expandedXShape).c_str(),
                                                         "The first dim of expandedX should be n * k or activeNum"));
        CHECK_FAIL(context_, expandedXShape.GetDim(1) != moeInitRoutingTilingData.get_cols(),
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expandedX",
                                                         Ops::Base::ToString(expandedXShape).c_str(),
                                                         "The second dim of expandedX should be cols"));
    }

    size_t expandedRowIdxDimNum = expandedRowIdxShape.GetDimNum();
    std::string expandedRowIdxDimStr = std::to_string(expandedRowIdxDimNum) + "D";
    CHECK_FAIL(
        context_, expandedRowIdxDimNum != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expandedRowIdx", expandedRowIdxDimStr.c_str(), "1D"));
    CHECK_FAIL(context_, expandedRowIdxShape.GetDim(0) != totalLength,
               OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expandedRowIdx",
                                                     Ops::Base::ToString(expandedRowIdxShape).c_str(),
                                                     "The first dim of expandedRowIdx should be totalLength"));

    if (dropPadMode == 0 && expertTokensCountOrCumsumFlag != 0) {
        if (CheckTokenCount(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, "expertTokensCountOrCumsum") == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }

    if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
        if (CheckTokenCount(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY, "expertTokensBeforeCapacity") == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();

    // 获取输入shape
    auto xShapePtr = context_->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    auto expertIdxShapePtr = context_->GetInputShape(INDEX_INPUT_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr);
    const gert::Shape expertIdxShape = expertIdxShapePtr->GetStorageShape();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* activateNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_ROWS);
    if (activateNumPtr != nullptr) {
        activateNum = *activateNumPtr;
    }
    const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
    if (expertCapacityPtr != nullptr) {
        expertCapacity = *expertCapacityPtr;
    }
    const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
    if (expertNumPtr != nullptr) {
        expertNum = *expertNumPtr;
    }
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    if (dropPadModePtr != nullptr) {
        dropPadMode = *dropPadModePtr;
    }
    const int64_t* expertTokensCountOrCumsumFlagPtr =
        attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
    if (expertTokensCountOrCumsumFlagPtr != nullptr) {
        expertTokensCountOrCumsumFlag = *expertTokensCountOrCumsumFlagPtr;
    }
    const bool* expertTokensBeforeCapacityFlagPtr =
        attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
    if (expertTokensBeforeCapacityFlagPtr != nullptr) {
        expertTokensBeforeCapacityFlag = *expertTokensBeforeCapacityFlagPtr;
    }

    // 参数校验
    size_t xDimNnum = xShape.GetDimNum();
    size_t expertIdxDimNum = expertIdxShape.GetDimNum();
    CHECK_FAIL(context_, xDimNnum != DIM_TWO || expertIdxDimNum != DIM_TWO,
               OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                   context_->GetNodeName(), "x and expertIdx",
                   (std::to_string(xDimNnum) + " and " + std::to_string(expertIdxDimNum)).c_str(),
                   "The dim number of x and expertIdx should be 2"));
    CHECK_FAIL(context_, xShape.GetDim(0) != expertIdxShape.GetDim(0),
               OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                   context_->GetNodeName(), "x and expertIdx",
                   (std::to_string(xShape.GetDim(0)) + " and " + std::to_string(expertIdxShape.GetDim(0))).c_str(),
                   "The first dim of x and expertIdx should be equal"));
    CHECK_FAIL(context_, xShape.GetDim(0) < 0,
               OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                     "The first dim of x cannot be less than 0"));
    CHECK_FAIL(context_, xShape.GetDim(1) < 0,
               OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                     "The second dim of x cannot be less than 0"));
    CHECK_FAIL(context_, expertIdxShape.GetDim(0) < 0,
               OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expertIdx",
                                                     Ops::Base::ToString(expertIdxShape).c_str(),
                                                     "The first dim of expertIdx cannot be less than 0"));
    CHECK_FAIL(context_, expertIdxShape.GetDim(1) < 0,
               OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expertIdx",
                                                     Ops::Base::ToString(expertIdxShape).c_str(),
                                                     "The second dim of expertIdx cannot be less than 0"));
    CHECK_FAIL(context_, activateNum < 0,
               OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "activeNum", std::to_string(activateNum).c_str(),
                                         "greater than or equal to 0"));
    CHECK_FAIL(context_, expertCapacity < 0,
               OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expertCapacity",
                                         std::to_string(expertCapacity).c_str(), "greater than or equal to 0"));
    CHECK_FAIL(context_, expertNum < 0,
               OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expertNum", std::to_string(expertNum).c_str(),
                                         "greater than or equal to 0"));
    CHECK_FAIL(context_, dropPadMode < 0 || dropPadMode > 1,
               OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "dropPadMode", std::to_string(dropPadMode).c_str(),
                                         "0 or 1"));
    CHECK_FAIL(context_, dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1),
               OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                   context_->GetNodeName(), "expertCapacity and expertNum",
                   (std::to_string(expertCapacity) + " and " + std::to_string(expertNum)).c_str(),
                   "The expertCapacity and expertNum should be greater than 0 when dropPadMode is 1"));
    CHECK_FAIL(context_, expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT,
               OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expertTokensCountOrCumsumFlag",
                                         std::to_string(expertTokensCountOrCumsumFlag).c_str(), "0, 1 or 2"));
    CHECK_FAIL(context_, expertTokensCountOrCumsumFlag > 0 && expertNum <= 0,
               OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                   context_->GetNodeName(), "expertNum", std::to_string(expertNum).c_str(),
                   "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0"));
    CHECK_FAIL(context_, dropPadMode > 0 && expertCapacity > xShape.GetDim(0),
               OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                   context_->GetNodeName(), "x and expertCapacity",
                   (std::to_string(xShape.GetDim(0)) + " and " + std::to_string(expertCapacity)).c_str(),
                   "The first dim of x cannot be less than expertCapacity"));
    if (dropPadMode == 1) {
        // droppad场景下不输出expertTokensCountOrCumsum
        expertTokensCountOrCumsumFlag = 0;
    } else {
        // dropless场景下不输出expertTokensBeforeCapacity
        expertTokensBeforeCapacityFlag = false;
    }
    moeInitRoutingTilingData.set_cols(xShape.GetDim(1));
    moeInitRoutingTilingData.set_n(expertIdxShape.GetDim(0));
    moeInitRoutingTilingData.set_k(expertIdxShape.GetDim(1));
    moeInitRoutingTilingData.set_expertCapacity(expertCapacity);
    moeInitRoutingTilingData.set_expertNum(expertNum);
    moeInitRoutingTilingData.set_dropPadMode(dropPadMode);
    moeInitRoutingTilingData.set_expertTokensCountOrCumsumFlag(expertTokensCountOrCumsumFlag);
    moeInitRoutingTilingData.set_expertTokensBeforeCapacityFlag(expertTokensBeforeCapacityFlag);
    totalLength = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();

    auto ret = CheckOutShape();
    inuptXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ret;
}

void InnerMoeInitRoutingV2TilingBase::ShowTilingData()
{
    OP_LOGI(opName,
            "moeInitRoutingTilingData is coreNum:%ld, n:%ld, cols:%ld, k:%ld, expertCapacity:%ld, expertNum:%ld, "
            "dropPadMode:%ld, expertTokensCountOrCumsumFlag:%ld, expertTokensBeforeCapacityFlag:%ld",
            moeInitRoutingTilingData.get_coreNum(), moeInitRoutingTilingData.get_n(),
            moeInitRoutingTilingData.get_cols(), moeInitRoutingTilingData.get_k(),
            moeInitRoutingTilingData.get_expertCapacity(), moeInitRoutingTilingData.get_expertNum(),
            moeInitRoutingTilingData.get_dropPadMode(), moeInitRoutingTilingData.get_expertTokensCountOrCumsumFlag(),
            moeInitRoutingTilingData.get_expertTokensBeforeCapacityFlag());
    OP_LOGI(opName,
            "InnerMoeV2VBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
            "perCorePerLoopElements:%ld, "
            "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
            "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld",
            moeInitRoutingTilingData.vbsComputeParamsOp.get_needCoreNum(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreElements(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLoops(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreElements(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
            moeInitRoutingTilingData.vbsComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(opName, "VMSMiddleComputeTilingData is needCoreNum:%ld",
            moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGI(opName, "SortOutComputeTilingData is oneLoopMaxElements:%ld",
            moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(opName,
            "SrcToDstComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
            "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_needCoreNum(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_activateRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCorePerLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreLastLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCorePerLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreLastLoopRows());
    OP_LOGI(opName,
            "SrcToDstComputeCapacityTilingData is needCoreNum:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
            "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_needCoreNum(),
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCoreRows(),
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCorePerLoopRows(),
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_perCoreLastLoopRows(),
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCoreRows(),
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCorePerLoopRows(),
            moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_lastCoreLastLoopRows());
    OP_LOGI(opName,
            "GatherOutComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
            "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld,",
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_needCoreNum(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_activateRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopRows());
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::DoOpTiling()
{
    // NUM_TWO sort value and indices
    // NUM_FOUR sort need space
    // SORT32_ALIGN_ELEMENT 32Bytes aligned
    sortLoopMaxElement =
        (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;

    if (!regBase) {
        isFullLoad = IsFullLoad();
    }
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4SrcToDstCapacityCompute();
    Tiling4GatherOutCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t InnerMoeInitRoutingV2TilingBase::GetTilingKey() const
{
    if (isFullLoad) {
        return TILING_KEY_HIGH_PERFORMANCE;
    }

    bool histWithRegBase = regBase && expertNum <= HIST_REGBASE_MAX_EXPERT_NUM;
    if (dropPadMode == 0) {
        if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
            return histWithRegBase ? TILING_KEY_DROPLESS_HIST_REGBASE_SORT_ONE_CORE : TILING_KEY_DROPLESS_SORT_ONE_CORE;
        } else {
            return histWithRegBase ? TILING_KEY_DROPLESS_HIST_REGBASE_SORT_MULTI_CORE :
                                     TILING_KEY_DROPLESS_SORT_MULTI_CORE;
        }
    } else {
        if (totalLength <= sortLoopMaxElement) {
            return histWithRegBase ? TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_ONE_CORE :
                                     TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;
        } else {
            return histWithRegBase ? TILING_KEY_DROP_PAD_MODE_HIST_REGBASE_SORT_MULTI_CORE :
                                     TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE;
        }
    }
    return tilingKey_;
}

ge::graphStatus InnerMoeInitRoutingV2TilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = sizeof(float) * static_cast<size_t>(totalLength * NUM_TWO * NUM_THREE); // 排序需要的空间
    size_t scatterWorkspaceSize = sizeof(int32_t) * static_cast<size_t>(totalLength * NUM_TWO);
    size_t expertTokenFlagSize = static_cast<size_t>(aivNum * 2) * sizeof(int32_t);
    size_t expertNumSize = static_cast<size_t>(expertNum) * sizeof(int32_t);
    workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + expertNumSize +
                     SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

void InnerMoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void InnerMoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, sortLoopMaxElement); // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));    // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                               // 不能超过物理核数
    if (needCoreNum == 0) {
        return;
    }
    int64_t perCoreElements = totalLength / needCoreNum; // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            Ops::Base::CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                                (tilingData->get_perCoreLoops() - 1) *
                                                    tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(totalLength -
                                         (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                               SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                                 (tilingData->get_lastCoreLoops() - 1) *
                                                     tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    OP_CHECK_IF(tilingData->get_lastCoreLastLoopElements() <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "vbs tiling", "failed", "vbs tiling failed"),
                ;);
}

void InnerMoeInitRoutingV2TilingBase::Tiling4VBSCompute()
{
    auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (totalLength <= sortLoopMaxElement) { // 只用到一个核
        Tiling4VBSOneCoreCompute(tilingData);
        return;
    }
    Tiling4VBSMultiCoreCompute(tilingData);
}

void InnerMoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void InnerMoeInitRoutingV2TilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

    int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                             (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);

    tilingData->set_perCoreRows(perCoreRows);

    if (perLoopMaxRows >= tilingData->get_perCoreRows()) { // 一个loop结束
        tilingData->set_perCorePerLoopRows(tilingData->get_perCoreRows());
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows());
    } else {
        tilingData->set_perCorePerLoopRows(perLoopMaxRows);
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows() -
                                            (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), perLoopMaxRows) - 1) *
                                                perLoopMaxRows);
    }

    tilingData->set_lastCoreRows(lastCoreNum);
    if (perLoopMaxRows >= tilingData->get_lastCoreRows()) {
        tilingData->set_lastCorePerLoopRows(tilingData->get_lastCoreRows());
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows());
    } else {
        tilingData->set_lastCorePerLoopRows(perLoopMaxRows);
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows() -
                                             (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), perLoopMaxRows) - 1) *
                                                 perLoopMaxRows);
    }
}

void InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;

    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t cols = moeInitRoutingTilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize =
        (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        tilingData->set_perCorePerLoopRows(perCoreRows);
        tilingData->set_perCoreLastLoopRows(perCoreRows);
        tilingData->set_lastCorePerLoopRows(lastCoreRows);
        tilingData->set_lastCoreLastLoopRows(lastCoreRows);
        tilingData->set_perCoreLoops(1);
        tilingData->set_lastCoreLoops(1);
        tilingData->set_perLoopCols(cols);
        tilingData->set_lastLoopCols(cols);
        tilingData->set_colLoops(1);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                     static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE *
                          ONE_BLOCK_BYTE;
        }
        tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
        tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
        tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}
} // namespace optiling
