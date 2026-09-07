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
 * \file moe_distribute_dispatch_teardown_tiling_base.cpp
 * \brief
 */

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_dispatch_teardown_tiling_base.h"

namespace {
constexpr uint32_t INPUT_X_INDEX = 0U;
constexpr uint32_t INPUT_Y_INDEX = 1U;
constexpr uint32_t INPUT_EXPERT_IDS_INDEX = 2U;
constexpr uint32_t INPUT_COMM_CMD_INFO_INDEX = 3U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0U;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2U;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3U;

constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4U;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5U;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6U;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 7U;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 8U;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 9U;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 10U;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 11U;

constexpr uint32_t ONE_DIMS = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint64_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512U;

constexpr uint32_t INIT_TILINGKEY = 10000U;
constexpr uint64_t NUM_1000 = 1000U;

constexpr uint32_t NO_SCALES = 0U;
constexpr uint32_t STATIC_SCALES = 1U;
constexpr uint32_t DYNAMIC_SCALES = 2U;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr size_t MAX_COMM_ALG_LENGTH = 128UL;
constexpr int64_t MIN_SHARED_EXPERT_NUM = 0;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MIN_SHARED_EXPERT_RANK_NUM = 0;
constexpr int64_t MIN_GROUP_EP_SIZE = 2;
constexpr int64_t MAX_GROUP_EP_SIZE = 384;
constexpr int64_t UNQUANT = 0;
constexpr int64_t STATIC_QUANT = 1;
constexpr int64_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr int64_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr int64_t MX_QUANT = 4;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t SDMA_COMM = 0;
constexpr int64_t URMA_COMM = 2;
constexpr int64_t QUANT_HS_OFFSET = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 4;
constexpr int64_t BS_UPPER_BOUND = 4;
constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;
constexpr int64_t COMM_CMD_INFO_BASE = 16;
constexpr int64_t QUANT_ALIGN_OFFSET = 4;

constexpr int64_t MIN_H = 1024;
constexpr int64_t MAX_H = 8192;
constexpr int64_t MAX_BS = 512;
constexpr int64_t MAX_K = 16;

constexpr uint32_t LOCAL_STREAM_MAX_NUM = 40U;
constexpr uint32_t TILINGKEY_SCALES = 10U;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512U;
constexpr uint32_t SYSTEM_NEED_WORKSAPCE = 16 * 1024 * 1024U;
constexpr int64_t MB_SIZE = 1024 * 1024UL;
constexpr int64_t WIN_ADDR_ALIGN = 512UL;
constexpr int64_t SCALE_EXPAND_IDX_BUFFER = 44UL;
constexpr int64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr int64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr int64_t EVEN_ALIGN = 2;
constexpr int64_t UB_ALIGN = 32UL;
constexpr int64_t ALIGN_32 = 32UL;
constexpr int64_t ALIGN_128 = 128UL;
constexpr int64_t ALIGN_256 = 256UL;
constexpr int64_t ALIGN_512 = 512UL;

constexpr int64_t MIN_AVAILABLE_BUFF_SIZE = 2;
constexpr int64_t HCCL_BUFFER_SIZE = 44;

constexpr uint64_t DIM_ZERO = 0;
constexpr uint64_t DIM_ONE = 1;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t DIM_THREE = 3;

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocal;
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};
} // namespace

namespace optiling {

uint64_t MoeDistributeDispatchTeardownTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(nodeName_, "%s get tiling key %lu", this->socTilingName_, tilingKey);
    return tilingKey;
}

const ge::graphStatus MoeDistributeDispatchTeardownTilingBase::CheckRequiredAttrValue()
{
    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    // 判空
    OP_TILING_CHECK(
        ((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
         (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH)),
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName_, "groupEp",
            (groupEpPtr != nullptr ?
                 (std::string("length=") + std::to_string(strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH))).c_str() :
                 "null"),
            "valid group name length"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*epWorldSizePtr < MIN_GROUP_EP_SIZE) || (*epWorldSizePtr > MAX_GROUP_EP_SIZE)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(),
                                              (std::string("[") + std::to_string(MIN_GROUP_EP_SIZE) + ", " +
                                               std::to_string(MAX_GROUP_EP_SIZE) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MAX_MOE_EXPERT_NUM)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_MOE_EXPERT_NUM) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTeardownTilingBase::GetRequiredAttrAndSetTilingData()
{
    OP_LOGD("GetRequiredAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "attrs"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    // 判空
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "groupEp"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "epRankId"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "moeExpertNum"),
                    return ge::GRAPH_FAILED);

    if (CheckRequiredAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置 tilingdata
    groupEp_ = string(groupEpPtr);
    tilingData_->moeDistributeDispatchTeardownInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData_->moeDistributeDispatchTeardownInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData_->moeDistributeDispatchTeardownInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchTeardownTilingBase::CheckOptionalAttrValue()
{
    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    OP_TILING_CHECK((xShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, "x",
                                                 std::to_string(xShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                    return ge::GRAPH_FAILED);
    auto bs = xShape->GetStorageShape().GetDim(0);
    auto attrs = context_->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);

    OP_TILING_CHECK(
        (*expertShardTypePtr != 0),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "expertShardType", std::to_string(*expertShardTypePtr).c_str(), "0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*sharedExpertNumPtr < MIN_SHARED_EXPERT_NUM) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(),
                                              (std::string("[") + std::to_string(MIN_SHARED_EXPERT_NUM) + ", " +
                                               std::to_string(MAX_SHARED_EXPERT_NUM) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*sharedExpertRankNumPtr < MIN_SHARED_EXPERT_RANK_NUM) || (*sharedExpertRankNumPtr > *epWorldSizePtr / 2)),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertRankNum", std::to_string(*sharedExpertRankNumPtr).c_str(),
                                  (std::string("[") + std::to_string(MIN_SHARED_EXPERT_RANK_NUM) + ", " +
                                   std::to_string(*epWorldSizePtr / 2) + "]")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*quantModePtr < UNQUANT) || (*quantModePtr > MX_QUANT)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "quantMode", std::to_string(*quantModePtr).c_str(), "[0, 4]"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*globalBsPtr != 0) && ((*globalBsPtr < bs * *epWorldSizePtr) || (*globalBsPtr > MAX_BS * *epWorldSizePtr))),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "globalBs", std::to_string(*globalBsPtr).c_str(),
                                  (std::string("0 or [") + std::to_string(bs * *epWorldSizePtr) + ", " +
                                   std::to_string(MAX_BS * *epWorldSizePtr) + "]")
                                      .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 1),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "expertTokenNumsType",
                                              std::to_string(*expertTokenNumsTypePtr).c_str(), "1"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*commTypePtr != URMA_COMM),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "commType", std::to_string(*commTypePtr).c_str(),
                                              std::to_string(URMA_COMM).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((strnlen(commAlgPtr, MAX_COMM_ALG_LENGTH) != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "commAlg", commAlgPtr, "empty"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTeardownTilingBase::GetOptionalAttrAndSetTilingData()
{
    OP_LOGD("GetOptionalAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "attrs"), return ge::GRAPH_FAILED);

    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsType = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);

    // 判空
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertShardType"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "quantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "globalBs"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsType == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertTokenNumsType"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commAlg"), return ge::GRAPH_FAILED);

    if (CheckOptionalAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingData_->moeDistributeDispatchTeardownInfo.bs =
        context_->GetInputShape(INPUT_X_INDEX)->GetStorageShape().GetDim(0);
    tilingData_->moeDistributeDispatchTeardownInfo.h =
        context_->GetInputShape(INPUT_X_INDEX)->GetStorageShape().GetDim(1);
    tilingData_->moeDistributeDispatchTeardownInfo.k =
        context_->GetInputShape(INPUT_EXPERT_IDS_INDEX)->GetStorageShape().GetDim(1);

    // 设置 tilingdata
    tilingData_->moeDistributeDispatchTeardownInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData_->moeDistributeDispatchTeardownInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData_->moeDistributeDispatchTeardownInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    tilingData_->moeDistributeDispatchTeardownInfo.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsType);
    tilingData_->moeDistributeDispatchTeardownInfo.isQuant = (*quantModePtr != UNQUANT);

    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchTeardownTilingBase::CheckTensorShape()
{
    const gert::StorageShape *xShape = context_->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *yShape = context_->GetInputShape(INPUT_Y_INDEX);
    const gert::StorageShape *expertIdsShape = context_->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const gert::StorageShape *commCmdInfoShape = context_->GetInputShape(INPUT_COMM_CMD_INFO_INDEX);
    const gert::StorageShape *expandXOutShape = context_->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const gert::StorageShape *dynamicScalesOutShape = context_->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    const gert::StorageShape *assitInfoForCombineOutShape =
        context_->GetOutputShape(OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX);
    const gert::StorageShape *expertTokenNumsOutShape = context_->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);

    OP_TILING_CHECK((xShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((yShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((commCmdInfoShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commCmdInfo"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expandXOutShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expandXOut"),
                    return ge::GRAPH_FAILED);
    auto quantMode = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.quantMode);

    OP_TILING_CHECK(((quantMode != UNQUANT) && (quantMode != STATIC_QUANT) && (dynamicScalesOutShape == nullptr)),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "dynamicScalesOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((assitInfoForCombineOutShape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "assistInfoForCombineOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertTokenNumsOutShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertTokenNumsOut"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckInputTensorShapeDim(), OP_LOGE(nodeName_, "Check input dims failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckOutputTensorShapeDim(), OP_LOGE(nodeName_, "Check output dims failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorShapeRelation(), OP_LOGE(nodeName_, "Check shape relation failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorShapeSize(), OP_LOGE(nodeName_, "Check shape size failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckInputTensorShapeDim()
{
    const gert::StorageShape *xShape = context_->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *yShape = context_->GetInputShape(INPUT_Y_INDEX);
    const gert::StorageShape *expertIdsShape = context_->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const gert::StorageShape *commCmdInfoShape = context_->GetInputShape(INPUT_COMM_CMD_INFO_INDEX);

    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, "x",
                                                 std::to_string(xShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                    return false);
    OP_TILING_CHECK(yShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, "y",
                                                 std::to_string(yShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                    return false);
    OP_TILING_CHECK(
        expertIdsShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, "expertIds",
                                     std::to_string(expertIdsShape->GetStorageShape().GetDimNum()).c_str(), "2"),
        return false);
    OP_TILING_CHECK(
        commCmdInfoShape->GetStorageShape().GetDimNum() != ONE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, "commCmdInfo",
                                     std::to_string(commCmdInfoShape->GetStorageShape().GetDimNum()).c_str(), "1"),
        return false);

    return true;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckOutputTensorShapeDim()
{
    const gert::StorageShape *expandXOutShape = context_->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const gert::StorageShape *dynamicScalesOutShape = context_->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    const gert::StorageShape *assitInfoForCombineOutShape =
        context_->GetOutputShape(OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX);
    const gert::StorageShape *expertTokenNumsOutShape = context_->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(
        expandXOutShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, "expandXOut",
                                     std::to_string(expandXOutShape->GetStorageShape().GetDimNum()).c_str(), "2"),
        return false);
    auto quantMode = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.quantMode);
    if (quantMode == PERTOKEN_DYNAMIC_QUANT) {
        OP_TILING_CHECK(dynamicScalesOutShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName_, "dynamicScalesOut",
                            std::to_string(dynamicScalesOutShape->GetStorageShape().GetDimNum()).c_str(), "1"),
                        return false);
    } else if ((quantMode == PERGROUP_DYNAMIC_QUANT) || (quantMode == MX_QUANT)) {
        OP_TILING_CHECK(dynamicScalesOutShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName_, "dynamicScalesOut",
                            std::to_string(dynamicScalesOutShape->GetStorageShape().GetDimNum()).c_str(), "2"),
                        return false);
    }
    OP_TILING_CHECK(assitInfoForCombineOutShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName_, "assistInfoForCombineOut",
                        std::to_string(assitInfoForCombineOutShape->GetStorageShape().GetDimNum()).c_str(), "1"),
                    return false);
    OP_TILING_CHECK(expertTokenNumsOutShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName_, "expertTokenNumsOut",
                        std::to_string(expertTokenNumsOutShape->GetStorageShape().GetDimNum()).c_str(), "1"),
                    return false);

    return true;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckTensorShapeRelation()
{
    const gert::StorageShape *xShape = context_->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *expertIdsShape = context_->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const gert::StorageShape *expandXOutShape = context_->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const gert::StorageShape *dynamicScalesOutShape = context_->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);

    auto bs1 = xShape->GetStorageShape().GetDim(DIM_ZERO);
    auto bs2 = expertIdsShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(
        (bs1 != bs2),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName_, "x/expertIds",
            (std::string("x dim0=") + std::to_string(bs1) + ", expertIds dim0=" + std::to_string(bs2)).c_str(),
            "Dim0 of x must be equal to dim0 of expertIds"),
        return false);
    auto h1 = xShape->GetStorageShape().GetDim(DIM_ONE);
    auto h2 = expandXOutShape->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(
        (h1 != h2),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName_, "x/expandXOut",
            (std::string("x dim1=") + std::to_string(h1) + ", expandXOut dim1=" + std::to_string(h2)).c_str(),
            "Dim1 of x must be equal to dim1 of expandXOut"),
        return false);
    auto quantMode = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.quantMode);
    if ((quantMode != UNQUANT) && (quantMode != STATIC_QUANT)) {
        auto a1 = expandXOutShape->GetStorageShape().GetDim(DIM_ZERO);
        auto a2 = dynamicScalesOutShape->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(
            (a1 != a2),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName_, "expandXOut/dynamicScalesOut",
                (std::string("expandXOut dim0=") + std::to_string(a1) + ", dynamicScalesOut dim0=" + std::to_string(a2))
                    .c_str(),
                "Dim0 of expandXOut must be equal to dim0 of dynamicScalesOut"),
            return false);
    }
    return true;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckTensorShapeSize()
{
    const gert::StorageShape *xShape = context_->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *yShape = context_->GetInputShape(INPUT_Y_INDEX);
    const gert::StorageShape *expertIdsShape = context_->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const gert::StorageShape *commCmdInfoShape = context_->GetInputShape(INPUT_COMM_CMD_INFO_INDEX);
    const gert::StorageShape *expandXOutShape = context_->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const gert::StorageShape *dynamicScalesOutShape = context_->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    const gert::StorageShape *assitInfoForCombineOutShape =
        context_->GetOutputShape(OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX);
    const gert::StorageShape *expertTokenNumsOutShape = context_->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);

    auto bs = xShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(((bs <= 0) || (bs > MAX_BS)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "x", (std::string("dim0=") + std::to_string(bs)).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_BS) + "]").c_str()),
                    return false);

    auto h = xShape->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(((h < MIN_H) || (h > MAX_H)),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName_, "x", (std::string("dim1=") + std::to_string(h)).c_str(),
                        (std::string("[") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
                    return false);

    auto k = expertIdsShape->GetStorageShape().GetDim(DIM_ONE);
    auto moeExpertNum = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.moeExpertNum);
    OP_TILING_CHECK(
        ((k <= 0) || (k > std::min(MAX_K, moeExpertNum))),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "expertIds", (std::string("dim1=") + std::to_string(k)).c_str(),
                                  (std::string("(0, ") + std::to_string(std::min(MAX_K, moeExpertNum)) + "]").c_str()),
        return false);

    auto sharedExpertNum = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertNum);
    auto yDim0Golden = bs * (k + sharedExpertNum);
    auto yDim0 = yShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(
        (yDim0Golden != yDim0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "y", (std::string("dim0=") + std::to_string(yDim0)).c_str(),
                                              "Dim0 of y must be equal to bs * (k + sharedExpertNum)"),
        return false);
    int64_t tokenMsgSize1;
    auto quantMode = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.quantMode);
    if (quantMode == UNQUANT) {
        tokenMsgSize1 = ops::CeilAlign(h, ALIGN_256);
    } else if (quantMode == STATIC_QUANT) {
        tokenMsgSize1 = ops::CeilAlign(h, ALIGN_512);
    } else if (quantMode == PERTOKEN_DYNAMIC_QUANT) {
        tokenMsgSize1 = ops::CeilAlign(ops::CeilAlign(h, ALIGN_32) + QUANT_ALIGN_OFFSET, ALIGN_512);
    } else if (quantMode == PERGROUP_DYNAMIC_QUANT) {
        tokenMsgSize1 = ops::CeilAlign(ops::CeilAlign(h, ALIGN_128) + ops::CeilDiv(h, ALIGN_128), ALIGN_512);
    } else if (quantMode == MX_QUANT) {
        tokenMsgSize1 = ops::CeilAlign(
            ops::CeilAlign(h, ALIGN_256) + ops::CeilAlign(ops::CeilDiv(h, ALIGN_32), EVEN_ALIGN), ALIGN_512);
    }
    auto tokenMsgSize2 = yShape->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK((tokenMsgSize1 != tokenMsgSize2),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_, "y", (std::string("dim1=") + std::to_string(tokenMsgSize2)).c_str(),
                        "Dim1 of y must be equal to tokenMsgSize"),
                    return false);

    int64_t a1;
    int64_t localExpertNum1;
    auto epWorldSize = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.epWorldSize);
    auto epRankId = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.epRankId);
    auto sharedExpertRankNum = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertRankNum);
    auto globalBs = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.globalBs);
    auto globalBsReal = (globalBs == 0) ? (bs * epWorldSize) : globalBs;
    if (epRankId < sharedExpertRankNum) {
        localExpertNum1 = 1;
        a1 = bs * epWorldSize * sharedExpertNum / sharedExpertRankNum;
    } else {
        localExpertNum1 = moeExpertNum / (epWorldSize - sharedExpertRankNum);
        a1 = globalBsReal * std::min(localExpertNum1, k);
    }
    auto commCmdInfoSize1 = (bs * (k + sharedExpertNum) + epWorldSize * localExpertNum1) * COMM_CMD_INFO_BASE;
    auto commCmdINfoSize2 = commCmdInfoShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK((commCmdInfoSize1 != commCmdINfoSize2),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_, "commCmdInfo", (std::string("dim0=") + std::to_string(commCmdINfoSize2)).c_str(),
                        "Dim0 of commCmdInfo must be equal to the expected value"),
                    return false);

    auto a2 = expandXOutShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_LOGD(nodeName_, "expect dim 0 is %ld", a1);
    OP_LOGD(nodeName_, "expandXOut dim 0 is %ld", a2);
    OP_TILING_CHECK((a1 != a2),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "expandXOut",
                                                          (std::string("dim0=") + std::to_string(a2)).c_str(),
                                                          "Dim0 of expandXOut must be equal to the expected value"),
                    return false);

    if ((quantMode != UNQUANT) && (quantMode != STATIC_QUANT)) {
        auto a3 = dynamicScalesOutShape->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK((a1 != a3),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName_, "dynamicScalesOut", (std::string("dim0=") + std::to_string(a3)).c_str(),
                            "Dim0 of dynamicScalesOut must be equal to the expected value"),
                        return false);
    }
    auto dynamicScalesDim1 = dynamicScalesOutShape->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK((quantMode == PERGROUP_DYNAMIC_QUANT) && (dynamicScalesDim1 != ops::CeilDiv(h, ALIGN_128)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "dynamicScalesOut",
                                                          (std::string("dim1=") + std::to_string(dynamicScalesDim1) +
                                                           ", quantMode=" + std::to_string(quantMode))
                                                              .c_str(),
                                                          "Dim1 of dynamicScalesOut must be equal to CeilDiv(h, 128)"),
                    return false);
    OP_TILING_CHECK(
        (quantMode == MX_QUANT) && (dynamicScalesDim1 != ops::CeilAlign(ops::CeilDiv(h, ALIGN_32), EVEN_ALIGN)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName_, "dynamicScalesOut",
            (std::string("dim1=") + std::to_string(dynamicScalesDim1) + ", quantMode=" + std::to_string(quantMode))
                .c_str(),
            "Dim1 of dynamicScalesOut must be equal to CeilAlign(CeilDiv(h, 32), 2)"),
        return false);
    auto localExpertNum2 = expertTokenNumsOutShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(
        (localExpertNum1 != localExpertNum2),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "expertTokenNumsOut",
                                              (std::string("dim0=") + std::to_string(localExpertNum2)).c_str(),
                                              "Dim0 of expertTokenNumsOut must be equal to the expected value"),
        return false);

    auto assistInfoForCombineOutSize1 = a1 * ASSIST_INFO_NUM_PER_A;
    auto assistInfoForCombineOutSize2 = assitInfoForCombineOutShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK((assistInfoForCombineOutSize1 != assistInfoForCombineOutSize2),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_, "assistInfoForCombineOut",
                        (std::string("dim0=") + std::to_string(assistInfoForCombineOutSize2)).c_str(),
                        "Dim0 of assistInfoForCombineOut must be equal to a1 * 128"),
                    return false);

    // 设置 tilingdata
    tilingData_->moeDistributeDispatchTeardownInfo.bs = static_cast<uint32_t>(bs);
    tilingData_->moeDistributeDispatchTeardownInfo.k = static_cast<uint32_t>(k);
    tilingData_->moeDistributeDispatchTeardownInfo.h = static_cast<uint32_t>(h);

    return true;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckInputTensorDataType()
{
    OP_TILING_CHECK((context_->GetInputDesc(INPUT_X_INDEX) == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "x"),
                    return false);
    OP_TILING_CHECK((context_->GetInputDesc(INPUT_Y_INDEX) == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName_, "y"),
                    return false);
    OP_TILING_CHECK((context_->GetInputDesc(INPUT_EXPERT_IDS_INDEX) == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertIds"), return false);
    OP_TILING_CHECK((context_->GetInputDesc(INPUT_COMM_CMD_INFO_INDEX) == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commCmdInfo"), return false);

    OP_TILING_CHECK(
        ((context_->GetInputDesc(INPUT_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
         (context_->GetInputDesc(INPUT_X_INDEX)->GetDataType() != ge::DT_BF16)),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "x",
                                  Ops::Base::ToString(context_->GetInputDesc(INPUT_X_INDEX)->GetDataType()).c_str(),
                                  "float16 or bfloat16"),
        return false);
    OP_TILING_CHECK(
        ((context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
         (context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType() != ge::DT_BF16) &&
         (context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType() != ge::DT_INT8) &&
         (context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType() != ge::DT_HIFLOAT8)),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "y",
                                  Ops::Base::ToString(context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType()).c_str(),
                                  "float16/bfloat16/int8/fp8/hif8"),
        return false);
    OP_TILING_CHECK(
        (context_->GetInputDesc(INPUT_EXPERT_IDS_INDEX)->GetDataType() != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(
            nodeName_, "expertIds",
            Ops::Base::ToString(context_->GetInputDesc(INPUT_EXPERT_IDS_INDEX)->GetDataType()).c_str(), "int32"),
        return false);
    OP_TILING_CHECK(
        (context_->GetInputDesc(INPUT_COMM_CMD_INFO_INDEX)->GetDataType() != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(
            nodeName_, "commCmdInfo",
            Ops::Base::ToString(context_->GetInputDesc(INPUT_COMM_CMD_INFO_INDEX)->GetDataType()).c_str(), "int32"),
        return false);
    return true;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckOutputTensorDataType()
{
    OP_TILING_CHECK((context_->GetOutputDesc(OUTPUT_EXPAND_X_INDEX)->GetDataType() !=
                     context_->GetInputDesc(INPUT_Y_INDEX)->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName_, "expandXOut",
                        Ops::Base::ToString(context_->GetOutputDesc(OUTPUT_EXPAND_X_INDEX)->GetDataType()).c_str(),
                        "The dtype of expandXOut must be the same as that of y"),
                    return false);
    return true;
}

const bool MoeDistributeDispatchTeardownTilingBase::CheckRelationTensorDataType()
{
    OP_TILING_CHECK((context_->GetOutputDesc(OUTPUT_EXPAND_X_INDEX) == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expandXOut"), return false);
    OP_TILING_CHECK((context_->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX) == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "dynamicScalesOut"), return false);
    auto quantMode = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.quantMode);
    if ((quantMode != UNQUANT) && (quantMode != STATIC_QUANT)) {
        OP_TILING_CHECK((context_->GetOutputDesc(OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX) == nullptr),
                        OP_LOGE_WITH_INVALID_INPUT(nodeName_, "assistInfoForCombineOut"), return false);
    }
    OP_TILING_CHECK((context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX) == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertTokenNumsOut"), return false);

    if ((quantMode == PERTOKEN_DYNAMIC_QUANT) && (quantMode = PERGROUP_DYNAMIC_QUANT)) {
        OP_TILING_CHECK(
            (context_->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX)->GetDataType() != ge::DT_FLOAT),
            OP_LOGE_FOR_INVALID_DTYPE(
                nodeName_, "dynamicScalesOut",
                Ops::Base::ToString(context_->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX)->GetDataType()).c_str(),
                "float32"),
            return false);
    } else if (quantMode == MX_QUANT) {
        OP_TILING_CHECK(
            (context_->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX)->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE(
                nodeName_, "dynamicScalesOut",
                Ops::Base::ToString(context_->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX)->GetDataType()).c_str(),
                "fp8_e8m0"),
            return false);
    }

    OP_TILING_CHECK(
        (context_->GetOutputDesc(OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX)->GetDataType() != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(
            nodeName_, "assistInfoForCombineOut",
            Ops::Base::ToString(context_->GetOutputDesc(OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX)->GetDataType()).c_str(),
            "int32"),
        return false);
    OP_TILING_CHECK(
        (context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX)->GetDataType() != ge::DT_INT64),
        OP_LOGE_FOR_INVALID_DTYPE(
            nodeName_, "expertTokenNumsOut",
            Ops::Base::ToString(context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX)->GetDataType()).c_str(),
            "int64"),
        return false);

    return true;
}

const ge::graphStatus MoeDistributeDispatchTeardownTilingBase::CheckTensorDataType()
{
    OP_TILING_CHECK(!CheckInputTensorDataType(), OP_LOGE(nodeName_, "Check input dataType failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckOutputTensorDataType(), OP_LOGE(nodeName_, "Check output dataType failed!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckRelationTensorDataType(), OP_LOGE(nodeName_, "Check relation tensor dataType failed!"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchTeardownTilingBase::CheckHcclBuffSize()
{
    const int64_t hcclBuffSize = static_cast<int64_t>(mc2tiling::Mc2TilingUtils::GetMaxWindowSize()) / 2;
    OP_TILING_CHECK(hcclBuffSize < MIN_AVAILABLE_BUFF_SIZE,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%ld] < [%ld].", hcclBuffSize, MIN_AVAILABLE_BUFF_SIZE),
                    return ge::GRAPH_FAILED);

    auto bs = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.bs);
    auto k = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.k);
    auto h = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.h);
    auto epWorldSize = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.epWorldSize);
    auto epRankId = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.epRankId);
    auto sharedExpertNum = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertNum);
    auto sharedExpertRankNum = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertRankNum);
    auto moeExpertNum = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.moeExpertNum);
    auto globalBs = static_cast<int64_t>(tilingData_->moeDistributeDispatchTeardownInfo.globalBs);
    int64_t localExpertNum;
    if (epRankId < sharedExpertRankNum) {
        localExpertNum = 1;
    } else {
        localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    }
    auto maxBs = bs;
    if (globalBs != 0) {
        maxBs = globalBs / epWorldSize;
    }
    auto align = ops::CeilAlign((ops::CeilAlign(2 * h, ALIGN_32) + HCCL_BUFFER_SIZE), ALIGN_512);
    const auto hcclBuffSizeGolden =
        MIN_AVAILABLE_BUFF_SIZE * ((localExpertNum * maxBs * epWorldSize * align) +
                                   (k + sharedExpertNum) * maxBs * ops::CeilAlign(2 * h, ALIGN_512));
    OP_TILING_CHECK(hcclBuffSize < hcclBuffSizeGolden,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%ld] < [%ld].", hcclBuffSize, hcclBuffSizeGolden),
                    return ge::GRAPH_FAILED);
    tilingData_->moeDistributeDispatchTeardownInfo.totalWinSize = hcclBuffSize;
    return ge::GRAPH_SUCCESS;
}

void MoeDistributeDispatchTeardownTilingBase::SetTilingKey()
{
    bool isSharedExpert(tilingData_->moeDistributeDispatchTeardownInfo.epRankId <
                        tilingData_->moeDistributeDispatchTeardownInfo.sharedExpertRankNum);

    uint64_t tilingKey = INIT_TILINGKEY;
    tilingKey += static_cast<uint64_t>(tilingData_->moeDistributeDispatchTeardownInfo.quantMode);
    tilingKey += (isSharedExpert ? NUM_1000 : 0UL);

    OP_LOGD(nodeName_, "tilingKey is %lu", tilingKey);
    context_->SetTilingKey(tilingKey);
}

void MoeDistributeDispatchTeardownTilingBase::SetHcommCfg()
{
    OP_LOGD(nodeName_, "MoeDistributeDispatchTeardown groupEp = %s.", groupEp_.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(3); // AIV_UB-MEM or AIV_URMA
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
    reinterpret_cast<Mc2CcTilingInner *>(&tilingData_->mc2CcTiling)->protocal = 1; // 0: UB-MEM, 1: URMA
}

void MoeDistributeDispatchTeardownTilingBase::SetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    ;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(1); // 设置为batch mode模式, 所有核同时启动
    tilingData_->moeDistributeDispatchTeardownInfo.totalUbSize = ubSize;
    tilingData_->moeDistributeDispatchTeardownInfo.aivNum = aivNum;
    OP_LOGD(nodeName_, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
}

ge::graphStatus MoeDistributeDispatchTeardownTilingBase::SetWorkSpace()
{
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName_, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = static_cast<uint64_t>(SYSTEM_NEED_WORKSPACE +
                                          tilingData_->moeDistributeDispatchTeardownInfo.aivNum * ALIGN_512);
    return ge::GRAPH_SUCCESS;
}

const void MoeDistributeDispatchTeardownTilingBase::PrintTilingDataInfo()
{
    const MoeDistributeDispatchTeardownInfo &info = tilingData_->moeDistributeDispatchTeardownInfo;
    OP_LOGD(nodeName_, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(nodeName_, "epRankId is %u.", info.epRankId);
    OP_LOGD(nodeName_, "expertShardType is %u.", info.expertShardType);
    OP_LOGD(nodeName_, "sharedExpertNum is %u.", info.sharedExpertNum);
    OP_LOGD(nodeName_, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(nodeName_, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(nodeName_, "quantMode is %u.", info.quantMode);
    OP_LOGD(nodeName_, "globalBs is %u.", info.globalBs);
    OP_LOGD(nodeName_, "bs is %u.", info.bs);
    OP_LOGD(nodeName_, "k is %u.", info.k);
    OP_LOGD(nodeName_, "h is %u.", info.h);
    OP_LOGD(nodeName_, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName_, "isQuant is %u.", static_cast<uint32_t>(info.isQuant));
    OP_LOGD(nodeName_, "totalUbSize is %lu.", info.totalUbSize);
    OP_LOGD(nodeName_, "totalWinSize is %lu.", info.totalWinSize);
    OP_LOGD(nodeName_, "expertTokenNumsType is %u.", info.expertTokenNumsType);
    OP_LOGD(nodeName_, "sdmaUsedStreamPerCore is %u.", info.sdmaUsedStreamPerCore);
}

ge::graphStatus MoeDistributeDispatchTeardownTilingBase::MoeDistributeDispatchTeardownTilingFuncImpl()
{
    OP_LOGD(nodeName_, "MoeDistributeDispatchTeardownTilingFunc start");
    tilingData_ = context_->GetTilingData<MoeDistributeDispatchTeardownTilingData>();

    // 实现 A3 Tiling 拦截
    if (!((GetRequiredAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetOptionalAttrAndSetTilingData() == ge::GRAPH_SUCCESS) && (CheckTensorShape() == ge::GRAPH_SUCCESS) &&
          (CheckTensorDataType() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }
    if (CheckHcclBuffSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetHcommCfg();
    SetTilingKey();
    SetPlatformInfo();
    PrintTilingDataInfo();
    if (SetWorkSpace() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(nodeName_, "MoeDistributeDispatchTeardownTilingFunc success");
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
