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
 * \file ffn_worker_batching_tiling.cpp
 * \brief
 */
#include "ffn_worker_batching_tiling.h"
#include "op_host/tiling_util.h"
#include "context_util.h"

namespace optiling {
constexpr uint32_t EXPERT_NUM_ATTR = 0;
constexpr uint32_t MAX_OUT_SHAPE_ATTR = 1;
constexpr uint32_t TOKEN_DTYPE_ATTR = 2;
constexpr uint32_t NEED_SCHEDULE_ATTR = 3;
constexpr uint32_t LAY_NUM_ATTR = 4;
constexpr int64_t X_SHAPE_VALUE = 1024;

constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
constexpr uint32_t BATCH_MODE = 1;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t MAX_RESERVE_WK_NUM = 128;

constexpr int64_t TILING_KEY_NORM = 100;
constexpr int64_t TILING_KEY_RECV = 101;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t EXPERT_IDX_MAX = 8192;
constexpr int64_t MAX_SESSION_NUM = 1024;
constexpr int64_t MAX_K_NUM = 64;
constexpr int64_t TH_RECV_CORE_NUM = 32;
constexpr int64_t TH_RECV_Y_NUM = 200000;

ge::graphStatus FfnWorkerBatchingTiling::SetBatchingWorkspaceSize(int64_t expertNum)
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);

    // 前128个int32: 总的有效专家数以及recv功能遍历的micro_batch_id
    // Y: 记录每个核、每次loop排序后的有效专家个数
    // 剩下的是排序需要占用的空间，并且需要记录排序后的 expert_ids以及专家序号expert_idx
    currentWorkspace[0] = MAX_RESERVE_WK_NUM * sizeof(int32_t) + Y_ * sizeof(int32_t) +
                          Y_ * sizeof(int32_t) * NUM_TWO * NUM_FOUR + expertNum * sizeof(int32_t) + sysWorkspaceSize;
    OP_LOGI(context_->GetNodeName(), "total workspace size:%lu", currentWorkspace[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0, OP_LOGE(context_->GetNodeName(), "Get aivNum:%u failed.", aivNum),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context_->GetNodeName(), "Get ubSize:%lu failed.", ubSize),
                return ge::GRAPH_FAILED);

    context_->SetBlockDim(aivNum);
    tilingData_.set_coreNum(aivNum);
    tilingData_.set_ubSize(ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTiling::CheckInputParam()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType xdtype = inputDesc->GetDataType();
    OP_CHECK_IF(xdtype != ge::DT_INT8,
                OP_LOGE(context_->GetNodeName(), "Input dtype:%s not int8", Ops::Base::ToString(xdtype).c_str()),
                return ge::GRAPH_FAILED);

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Transformer::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(), "x shape %s dim num not 1", Ops::Base::ToString(xShape).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FfnWorkerBatchingTiling::GetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr);
    OP_CHECK_IF(
        *expertNumPtr > EXPERT_IDX_MAX || *expertNumPtr <= 0,
        OP_LOGE(context_->GetNodeName(), "expert_num:%ld should be in range (0, %ld]", *expertNumPtr, EXPERT_IDX_MAX),
        return ge::GRAPH_FAILED);
    tilingData_.set_expertNum(*expertNumPtr);

    const gert::ContinuousVector *maxOutShapePtr = attrs->GetAttrPointer<gert::ContinuousVector>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxOutShapePtr);
    OP_CHECK_IF(maxOutShapePtr->GetSize() != NUM_FOUR,
                OP_LOGE(context_->GetNodeName(), "The max_out_shape size:%lu not equal 4.", maxOutShapePtr->GetSize()),
                return ge::GRAPH_FAILED);
    const int64_t *maxOutShapeArray = reinterpret_cast<const int64_t *>(maxOutShapePtr->GetData());
    A_ = maxOutShapeArray[INDEX_ZERO];
    BS_ = maxOutShapeArray[INDEX_ONE];
    K_ = maxOutShapeArray[INDEX_TWO];
    OP_CHECK_IF(
        (A_ > MAX_SESSION_NUM || A_ <= 0),
        OP_LOGE(context_->GetNodeName(), "max_out_shape[0]:%ld should be in range of (0, %ld]", A_, MAX_SESSION_NUM),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(BS_ <= 0, OP_LOGE(context_->GetNodeName(), "max_out_shape[1]:%ld should be greater than 0", BS_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((K_ > MAX_K_NUM || K_ <= 0),
                OP_LOGE(context_->GetNodeName(), "max_out_shape[2]:%ld should be in range of (0, %ld]", K_, MAX_K_NUM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxOutShapeArray[INDEX_THREE] <= 0,
                OP_LOGE(context_->GetNodeName(), "max_out_shape[3]:%ld should be greater than 0",
                        maxOutShapeArray[INDEX_THREE]),
                return ge::GRAPH_FAILED);

    Y_ = A_ * BS_ * K_;
    tilingData_.set_Y(Y_);
    tilingData_.set_H(maxOutShapeArray[INDEX_THREE]);

    const int64_t *tokenDtype = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    if (tokenDtype != nullptr) {
        OP_CHECK_IF((*tokenDtype < 0 || *tokenDtype > NUM_TWO),
                    OP_LOGE(context_->GetNodeName(), "token_dtype:%ld must be one of [0, 1, 2]", *tokenDtype),
                    return ge::GRAPH_FAILED);
        tilingData_.set_tokenDtype(*tokenDtype);
    } else {
        tilingData_.set_tokenDtype(0);
    }

    const int64_t *needSchedulePtr = attrs->GetAttrPointer<int64_t>(NEED_SCHEDULE_ATTR);
    if (needSchedulePtr != nullptr) {
        OP_CHECK_IF((*needSchedulePtr < 0 || *needSchedulePtr > 1),
                    OP_LOGE(context_->GetNodeName(), "need_schedule:%ld must be one of [0, 1]", *needSchedulePtr),
                    return ge::GRAPH_FAILED);
        needSchedule_ = *needSchedulePtr;
    }

    const int64_t *layNumPtr = attrs->GetAttrPointer<int64_t>(LAY_NUM_ATTR);
    if (layNumPtr != nullptr) {
        OP_CHECK_IF(
            (*layNumPtr < 0 || *layNumPtr > *expertNumPtr),
            OP_LOGE(context_->GetNodeName(), "layer_num:%ld must be in range of [0, %ld]", *layNumPtr, *expertNumPtr),
            return ge::GRAPH_FAILED);
        layerNum_ = *layNumPtr;
    }

    return ge::GRAPH_SUCCESS;
}

void FfnWorkerBatchingTiling::SetBatchingTilingKey(int64_t sortLoopMaxElement)
{
    if (needSchedule_ == 0) {
        context_->SetTilingKey(TILING_KEY_NORM);
        return;
    }

    // needSchedule_ 为 1
    context_->SetTilingKey(TILING_KEY_RECV);
}

ge::graphStatus FfnWorkerBatchingTiling::RunFfnWorkerBatchingTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputParam() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t sortLoopMaxElement = static_cast<int64_t>(tilingData_.get_ubSize()) /
                                 (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM;
    tilingData_.set_sortLoopMaxElement(sortLoopMaxElement);
    tilingData_.set_sortNumWorkSpace(Y_);

    SetBatchingTilingKey(sortLoopMaxElement);
    context_->SetScheduleMode(BATCH_MODE);
    if (needSchedule_ == 1 && Y_ <= TH_RECV_Y_NUM) {
        int64_t useCoreNum = std::min(tilingData_.get_coreNum(), TH_RECV_CORE_NUM);
        context_->SetBlockDim(useCoreNum);
        tilingData_.set_coreNum(useCoreNum);
    }

    OP_LOGI(context_->GetNodeName(),
            "coreNum:%ld ubSize:%ld Y:%ld H:%ld tokenDtype:%ld expertNum:%ld "
            "sortLoopMaxElement:%ld sortNumWorkSpace:%ld tilingKey:%ld",
            tilingData_.get_coreNum(), tilingData_.get_ubSize(), tilingData_.get_Y(), tilingData_.get_H(),
            tilingData_.get_tokenDtype(), tilingData_.get_expertNum(), tilingData_.get_sortLoopMaxElement(),
            tilingData_.get_sortNumWorkSpace(), context_->GetTilingKey());

    gert::TilingData *batchingTilingD = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, batchingTilingD);
    tilingData_.SaveToBuffer(batchingTilingD->GetData(), batchingTilingD->GetCapacity());
    batchingTilingD->SetDataSize(tilingData_.GetDataSize());

    int64_t expertNum = tilingData_.get_expertNum();
    return SetBatchingWorkspaceSize(expertNum);
}

static ge::graphStatus Tiling4FfnWorkerBatching(gert::TilingContext *context)
{
    FfnWorkerBatchingTiling tilingObject(context);
    return tilingObject.RunFfnWorkerBatchingTiling();
}

static ge::graphStatus TilingPrepare4FfnWorkerBatching(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<FfnWorkerBatchingCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FfnWorkerBatching)
    .Tiling(Tiling4FfnWorkerBatching)
    .TilingParse<FfnWorkerBatchingCompileInfo>(TilingPrepare4FfnWorkerBatching);
} // namespace optiling
