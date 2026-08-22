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
 * \file moe_distribute_dispatch_setup_tiling_base.cpp
 * \brief
 */

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_dispatch_setup_tiling_base.h"

namespace {
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t OUTPUT_EXPAND_IDX_INDEX = 1U;
constexpr uint32_t OUTPUT_COMM_CMD_INFO_INDEX = 2U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0U;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2U;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3U;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4U;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5U;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6U;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 7U;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 8U;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 9U;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 10U;

constexpr uint32_t ONE_DIMS = 1U;
constexpr uint32_t TWO_DIMS = 2U;

constexpr uint32_t INIT_TILINGKEY = 1000U;
constexpr uint32_t NO_SCALES = 0U;
constexpr uint32_t STATIC_SCALES = 1U;
constexpr uint32_t DYNAMIC_SCALES = 2U;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr size_t MAX_COMM_ALG_LENGTH = 1UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4UL;
constexpr int64_t MIN_GROUP_EP_SIZE = 2UL;
constexpr int64_t MAX_GROUP_EP_SIZE = 384UL;
constexpr int64_t UNQUANT = 0;
constexpr int64_t STATIC_QUANT = 1;
constexpr int64_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr int64_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr int64_t MX_QUANT = 4;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512UL;
constexpr int64_t SDMA_COMM = 0UL;
constexpr int64_t URMA_COMM = 2UL;
constexpr int64_t QUANT_HS_OFFSET = 4UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 4UL;
constexpr int64_t BS_UPPER_BOUND = 4UL;
constexpr int64_t COMM_CMD_INFO_MULTIPLY = 16UL;
constexpr int64_t MIN_AVAILABLE_BUFF_SIZE = 2UL;
constexpr int64_t HCCL_BUFFER_SIZE = 44UL;
constexpr int64_t STATIC_QUANT_DIM_0 = 1;

constexpr int64_t MIN_H = 1024UL;
constexpr int64_t MAX_H = 8192UL;
constexpr int64_t MAX_BS = 512UL;
constexpr int64_t MAX_K = 16UL;

constexpr uint32_t LOCAL_STREAM_MAX_NUM = 40U;
constexpr uint32_t TILINGKEY_SCALES = 10U;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512U;
constexpr int64_t MB_SIZE = 1024UL * 1024UL;
constexpr int64_t WIN_ADDR_ALIGN = 512UL;
constexpr int64_t EVEN_ALIGN = 2;
constexpr int64_t UB_ALIGN = 32UL;
constexpr int64_t ALIGN_32 = 32UL;
constexpr int64_t ALIGN_128 = 128UL;
constexpr int64_t ALIGN_256 = 256UL;
constexpr int64_t ALIGN_512 = 512UL;
constexpr int64_t SCALE_EXPAND_IDX_BUFFER = 44UL;
constexpr int64_t DOUBLE_DATA_BUFFER = 2UL;
constexpr int64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr int64_t AICPUNUM = 4UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t SDMA_NEED_WORKSPACE = 16U * 1024 * 1024;
} // namespace

namespace optiling {

uint64_t MoeDistributeDispatchSetupTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(nodeName_, "%s get tiling key %lu", this->socTilingName_, tilingKey);
    return tilingKey;
}

const void MoeDistributeDispatchSetupTilingBase::PrintTilingDataInfo()
{
    const MoeDistributeDispatchSetupInfo &info = tilingData_->moeDistributeDispatchSetupInfo;
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
    OP_LOGD(nodeName_, "isActiveMask is %u.", static_cast<uint32_t>(info.isActiveMask));
    OP_LOGD(nodeName_, "totalUbSize is %lu.", info.totalUbSize);
    OP_LOGD(nodeName_, "totalWinSize is %lu.", info.totalWinSize);
    OP_LOGD(nodeName_, "sdmaUsedStreamPerCore is %u.", info.sdmaUsedStreamPerCore);
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckRequiredAttrValue()
{
    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    OP_TILING_CHECK(((strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                     (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH)),
                    OP_LOGE_WITH_INVALID_ATTR(
                        nodeName_, "groupEp",
                        (std::string("length=") + std::to_string(strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH))).c_str(),
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

ge::graphStatus MoeDistributeDispatchSetupTilingBase::GetRequiredAttrAndSetTilingData()
{
    OP_LOGD("GetRequiredAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "attrs"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);

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
    tilingData_->moeDistributeDispatchSetupInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData_->moeDistributeDispatchSetupInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckSharedExpertAttrValue(
    const uint32_t sharedExpertNum, const uint32_t sharedExpertRankNum)
{
    // 共享专家卡数>=共享专家数且可以整除
    if (sharedExpertRankNum == 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((sharedExpertNum == 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertNum",
                                              (std::string("sharedExpertNum=") + std::to_string(sharedExpertNum) +
                                               ", sharedExpertRankNum=" + std::to_string(sharedExpertRankNum))
                                                  .c_str(),
                                              "sharedExpertNum != 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertNum > sharedExpertRankNum),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertNum",
                                              (std::string("sharedExpertNum=") + std::to_string(sharedExpertNum) +
                                               ", sharedExpertRankNum=" + std::to_string(sharedExpertRankNum))
                                                  .c_str(),
                                              "sharedExpertNum <= sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertRankNum % sharedExpertNum != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertRankNum",
                                              (std::string("sharedExpertNum=") + std::to_string(sharedExpertNum) +
                                               ", sharedExpertRankNum=" + std::to_string(sharedExpertRankNum))
                                                  .c_str(),
                                              "sharedExpertRankNum % sharedExpertNum == 0"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOptionalAttrValue()
{
    auto attrs = context_->GetAttrs();
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);

    OP_TILING_CHECK(
        (*expertShardTypePtr != 0),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "expertShardType", std::to_string(*expertShardTypePtr).c_str(), "0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM)),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(),
                                  (std::string("[0, ") + std::to_string(MAX_SHARED_EXPERT_NUM) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((*sharedExpertRankNumPtr < 0) ||
         (*sharedExpertRankNumPtr > (tilingData_->moeDistributeDispatchSetupInfo.epWorldSize / 2))),
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName_, "sharedExpertRankNum", std::to_string(*sharedExpertRankNumPtr).c_str(),
            (std::string("[0, ") + std::to_string(tilingData_->moeDistributeDispatchSetupInfo.epWorldSize / 2) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*quantModePtr < UNQUANT) || (*quantModePtr > MX_QUANT)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "quantMode", std::to_string(*quantModePtr).c_str(), "0 to 4"),
                    return ge::GRAPH_FAILED);
    // globalBs 会在后面获取 BS 后再次校验
    OP_TILING_CHECK((*globalBsPtr < 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "globalBs", std::to_string(*globalBsPtr).c_str(),
                                              "0 or maxBs * epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*commTypePtr != URMA_COMM),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "commType", std::to_string(*commTypePtr).c_str(),
                                              std::to_string(URMA_COMM).c_str()),
                    return ge::GRAPH_FAILED);
    if (commAlgPtr != nullptr) {
        const std::string commAlg = std::string(commAlgPtr);
        OP_TILING_CHECK((commAlg != ""),
                        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "commAlg", commAlg.c_str(), "null or empty string"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::GetOptionalAttrAndSetTilingData()
{
    OP_LOGD("GetOptionalAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "attrs"), return ge::GRAPH_FAILED);

    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);

    // 判空
    OP_TILING_CHECK(expertShardTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertShardType"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "quantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "globalBs"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commType"), return ge::GRAPH_FAILED);

    if (CheckOptionalAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 校验共享专家限制
    const uint32_t sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    const uint32_t sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    if (CheckSharedExpertAttrValue(sharedExpertNum, sharedExpertRankNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置 tilingdata
    tilingData_->moeDistributeDispatchSetupInfo.expertShardType = static_cast<uint32_t>(*expertShardTypePtr);
    tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData_->moeDistributeDispatchSetupInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData_->moeDistributeDispatchSetupInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData_->moeDistributeDispatchSetupInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);

    if (tilingData_->moeDistributeDispatchSetupInfo.quantMode != UNQUANT) {
        tilingData_->moeDistributeDispatchSetupInfo.isQuant = true;
    } else {
        tilingData_->moeDistributeDispatchSetupInfo.isQuant = false;
    }
    if (context_->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX) != nullptr) {
        tilingData_->moeDistributeDispatchSetupInfo.isActiveMask = true;
    } else {
        tilingData_->moeDistributeDispatchSetupInfo.isActiveMask = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::GetComplexAttrAndSetTilingData()
{
    const uint32_t &epRankId = tilingData_->moeDistributeDispatchSetupInfo.epRankId;
    const uint32_t &epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    const uint32_t &sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    const uint32_t &sharedExpertRankNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertRankNum;
    const uint32_t &moeExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum;

    // localMoeExpertNum
    if (epRankId >= sharedExpertRankNum) {
        // MoE 专家卡
        tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum =
            moeExpertNum / (epWorldSize - sharedExpertNum);
    } else {
        // 共享专家卡
        tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum = 1U;
    }

    OP_TILING_CHECK((moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "moeExpertNum",
                                              (std::string("moeExpertNum=") + std::to_string(moeExpertNum) +
                                               ", epWorldSize=" + std::to_string(epWorldSize) +
                                               ", sharedExpertRankNum=" + std::to_string(sharedExpertRankNum))
                                                  .c_str(),
                                              "moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOneTensorDim(std::string name, TensorType tensortype,
                                                                              uint32_t index, uint32_t dims)
{
    const gert::StorageShape *shape;
    if (tensortype == TensorType::INPUT) {
        shape = context_->GetInputShape(index);
    } else if (tensortype == TensorType::OUTPUT) {
        shape = context_->GetOutputShape(index);
    } else if (tensortype == TensorType::OPTIONINPUT) {
        shape = context_->GetOptionalInputShape(index);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "tensorType", std::to_string(static_cast<int>(tensortype)).c_str(),
                                  "input or output");
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, name.c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(shape->GetStorageShape().GetDimNum() != dims,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName_, name.c_str(),
                                                 std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                                                 std::to_string(dims).c_str()),
                    return ge::GRAPH_FAILED);

    for (uint32_t d = 0; d < dims; d++) {
        OP_LOGD(nodeName_, "%s %u dim = %ld", name.c_str(), d, shape->GetStorageShape().GetDim(d));
    }
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckInputTensorDim()
{
    OP_TILING_CHECK(CheckOneTensorDim("x", TensorType::INPUT, X_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "x", "checkdim failed", "x must be 2D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("expertIds", TensorType::INPUT, EXPERT_IDS_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "expertIds", "checkdim failed", "expertIds must be 2D"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOptionalInputTensorDim()
{
    const gert::StorageShape *scalesStorageShape = context_->GetOptionalInputShape(SCALES_INDEX);
    const gert::StorageShape *xActiveMaskStorageShape = context_->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    uint64_t scalesCount = 0;
    if (scalesStorageShape != nullptr) {
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        if (scalesDimNum == ONE_DIMS) {
            // A3不会进此分支
            auto YDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    if (xActiveMaskStorageShape != nullptr) {
        OP_TILING_CHECK(xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName_, "xActiveMask",
                            std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDimNum()).c_str(), "1"),
                        return ge::GRAPH_FAILED);
    }
    tilingData_->moeDistributeDispatchSetupInfo.scalesCount = scalesCount;
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOutputTensorDim()
{
    OP_TILING_CHECK(CheckOneTensorDim("yOut", TensorType::OUTPUT, OUTPUT_Y_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "yOut", "checkdim failed", "yOut must be 2D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("expandIdxOut", TensorType::OUTPUT, OUTPUT_EXPAND_IDX_INDEX, ONE_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "expandIdxOut", "checkdim failed",
                                                 "expandIdxOut must be 1D"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOneTensorDim("commCmdInfoOut", TensorType::OUTPUT, OUTPUT_COMM_CMD_INFO_INDEX, ONE_DIMS) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "commCmdInfoOut", "checkdim failed",
                                                             "commCmdInfoOut must be 1D"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorDim()
{
    OP_TILING_CHECK(CheckInputTensorDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "input", "shape invalid",
                                                          "Input param shape validation failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOptionalInputTensorDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "optional input", "shape invalid",
                                                          "Optional input param shape validation failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputTensorDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "output", "shape invalid",
                                                          "Output param shape validation failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorShapeRelation()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto expertIdsShape = context_->GetInputShape(EXPERT_IDS_INDEX);
    auto xActiveMaskShape = context_->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    auto scalesShape = context_->GetOptionalInputShape(SCALES_INDEX);

    // BS校验
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDim(0) != expertIdsShape->GetStorageShape().GetDim(0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_, "x/expertIds",
                        (std::string("x dim0=") + std::to_string(xStorageShape->GetStorageShape().GetDim(0)) +
                         ", expertIds dim0=" + std::to_string(expertIdsShape->GetStorageShape().GetDim(0)))
                            .c_str(),
                        "Dim0 of x must be equal to dim0 of expertIds"),
                    return ge::GRAPH_FAILED);
    if (xActiveMaskShape != nullptr) {
        OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDim(0) != xActiveMaskShape->GetStorageShape().GetDim(0),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName_, "x/xActiveMask",
                            (std::string("x dim0=") + std::to_string(xStorageShape->GetStorageShape().GetDim(0)) +
                             ", xActiveMask dim0=" + std::to_string(xActiveMaskShape->GetStorageShape().GetDim(0)))
                                .c_str(),
                            "Dim0 of x must be equal to dim0 of xActiveMask"),
                        return ge::GRAPH_FAILED);
    }

    // H校验
    int64_t quantMode = tilingData_->moeDistributeDispatchSetupInfo.quantMode;
    if (scalesShape != nullptr) {
        size_t scalesDimNum = scalesShape->GetStorageShape().GetDimNum();
        // 1D: Static quant only
        if (scalesDimNum == ONE_DIMS) {
            auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
            const int64_t scalesDim0 = scalesShape->GetStorageShape().GetDim(0);
            OP_TILING_CHECK(
                (quantMode == STATIC_QUANT) && (yDesc->GetDataType() == ge::DT_INT8) &&
                    (scalesDim0 != xStorageShape->GetStorageShape().GetDim(1)) && (scalesDim0 != STATIC_QUANT_DIM_0),
                OP_LOGE_FOR_INVALID_VALUE(
                    nodeName_, "scales", (std::string("dim0=") + std::to_string(scalesDim0)).c_str(),
                    (std::string("H(") + std::to_string(xStorageShape->GetStorageShape().GetDim(1)) + ") or 1")
                        .c_str()),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK((quantMode == STATIC_QUANT) && (yDesc->GetDataType() == ge::DT_HIFLOAT8) &&
                                (scalesDim0 != STATIC_QUANT_DIM_0),
                            OP_LOGE_FOR_INVALID_VALUE(nodeName_, "scales",
                                                      (std::string("dim0=") + std::to_string(scalesDim0)).c_str(), "1"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDim(1) != scalesShape->GetStorageShape().GetDim(1),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                nodeName_, "x/scales",
                                (std::string("x dim1=") + std::to_string(xStorageShape->GetStorageShape().GetDim(1)) +
                                 ", scales dim1=" + std::to_string(scalesShape->GetStorageShape().GetDim(1)))
                                    .c_str(),
                                "Dim1 of x must be equal to dim1 of scales"),
                            return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckInputTensorDataType()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto expertIdsDesc = context_->GetInputDesc(EXPERT_IDS_INDEX);

    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertIds"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_FLOAT16) && (xDesc->GetDataType() != ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "float16 or bfloat16"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "expertIds",
                                              Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str(), "int32"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOptionalInputTensorDataType()
{
    auto scalesDesc = context_->GetOptionalInputDesc(SCALES_INDEX);
    auto xActiveMaskDesc = context_->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    if (scalesDesc != nullptr) {
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return ge::GRAPH_FAILED);
    }

    if (xActiveMaskDesc != nullptr) {
        OP_TILING_CHECK((xActiveMaskDesc->GetDataType() != ge::DT_BOOL),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "xActiveMask",
                                                  Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str(), "bool"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckOutputTensorDataType()
{
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    auto expandIdxDesc = context_->GetOutputDesc(OUTPUT_EXPAND_IDX_INDEX);
    auto commCmdInfoDesc = context_->GetOutputDesc(OUTPUT_COMM_CMD_INFO_INDEX);

    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "yOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expandIdxOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commCmdInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commCmdInfoOut"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((expandIdxDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "expandIdxOut",
                                              Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str(), "int32"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((commCmdInfoDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "commCmdInfoOut",
                                              Ops::Base::ToString(commCmdInfoDesc->GetDataType()).c_str(), "int32"),
                    return ge::GRAPH_FAILED);

    int64_t quantMode = tilingData_->moeDistributeDispatchSetupInfo.quantMode;
    OP_TILING_CHECK(
        (quantMode == UNQUANT) && (yDesc->GetDataType() != context_->GetInputDesc(X_INDEX)->GetDataType()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName_, "yOut", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "If quantMode is 0, the dtype of yOut must be the same as that of x"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == STATIC_QUANT) && (yDesc->GetDataType() != ge::DT_HIFLOAT8) &&
            (yDesc->GetDataType() != ge::DT_INT8),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName_, "yOut", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of yOut must be int8 or hif8"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == PERTOKEN_DYNAMIC_QUANT) && (yDesc->GetDataType() != ge::DT_INT8) &&
            (yDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) && (yDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName_, "yOut", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of yOut must be int8, fp8_e4m3fn or fp8_e5m2"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((quantMode == PERGROUP_DYNAMIC_QUANT) || (quantMode == MX_QUANT)) &&
            (yDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) && (yDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName_, "yOut", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of yOut must be fp8_e4m3fn or fp8_e5m2"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorDataType()
{
    if (!((CheckInputTensorDataType() == ge::GRAPH_SUCCESS) &&
          (CheckOptionalInputTensorDataType() == ge::GRAPH_SUCCESS) &&
          (CheckOutputTensorDataType() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckTensorShapeSizeAndSetTilingData()
{
    auto *xStorageShape = context_->GetInputShape(X_INDEX);
    auto *expertIdsShape = context_->GetInputShape(EXPERT_IDS_INDEX);

    int64_t H = xStorageShape->GetStorageShape().GetDim(1);
    int64_t BS = xStorageShape->GetStorageShape().GetDim(0);
    int64_t K = expertIdsShape->GetStorageShape().GetDim(1);

    uint32_t &epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    uint32_t &moeExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum;
    uint32_t &globalBs = tilingData_->moeDistributeDispatchSetupInfo.globalBs;

    OP_TILING_CHECK((H < MIN_H) || (H > MAX_H),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName_, "H", std::to_string(H).c_str(),
                        (std::string("[") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((BS <= 0) || (BS > MAX_BS),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "BS", std::to_string(BS).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_BS) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (K < 0) || (K > MAX_K) || (K > static_cast<int64_t>(moeExpertNum)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "K", std::to_string(K).c_str(),
                                  (std::string("[0, min(16, ") + std::to_string(moeExpertNum) + ")]").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (globalBs != 0) && ((globalBs < BS * epWorldSize) || (globalBs > MAX_BS * epWorldSize)),
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName_, "globalBs", std::to_string(globalBs).c_str(),
            (std::string(">= ") + std::to_string(BS * epWorldSize) + " and <= 512 * epWorldSize, or 0").c_str()),
        return ge::GRAPH_FAILED);

    tilingData_->moeDistributeDispatchSetupInfo.h = static_cast<uint32_t>(H);
    tilingData_->moeDistributeDispatchSetupInfo.bs = static_cast<uint32_t>(BS);
    tilingData_->moeDistributeDispatchSetupInfo.k = static_cast<uint32_t>(K);

    if (CheckComplexTensorShapeSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckComplexTensorShapeSize()
{
    int64_t sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    int64_t moeExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertNum;

    int64_t BS = tilingData_->moeDistributeDispatchSetupInfo.bs;
    int64_t K = tilingData_->moeDistributeDispatchSetupInfo.k;

    auto scalesShape = context_->GetOptionalInputShape(SCALES_INDEX);
    if ((scalesShape != nullptr)) {
        size_t scalesDimNum = scalesShape->GetStorageShape().GetDimNum();
        if (scalesDimNum != ONE_DIMS) {
            int64_t scalesDim0GoldenSize = moeExpertNum + sharedExpertNum;
            int64_t scalesDim0Size = context_->GetOptionalInputShape(SCALES_INDEX)->GetStorageShape().GetDim(0);
            OP_TILING_CHECK(scalesDim0Size != scalesDim0GoldenSize,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                nodeName_, "scales", (std::string("dim0=") + std::to_string(scalesDim0Size)).c_str(),
                                "Dim0 of scales must be equal to moeExpertNum + sharedExpertNum"),
                            return ge::GRAPH_FAILED);
        }
    }

    int64_t yOutDim0GoldenSize = (BS * (K + sharedExpertNum));
    int64_t yOutDim0Size = context_->GetOutputShape(OUTPUT_Y_INDEX)->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(yOutDim0Size != yOutDim0GoldenSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "yOut",
                                                          (std::string("dim0=") + std::to_string(yOutDim0Size)).c_str(),
                                                          "Dim0 of yOut must be equal to BS * (K + sharedExpertNum)"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckCalcTensorShapeSizeAndSetTilingData()
{
    auto yOutShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    auto expandIdxOutShape = context_->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    auto commCmdInfoOutShape = context_->GetOutputShape(OUTPUT_COMM_CMD_INFO_INDEX);
    int64_t K = tilingData_->moeDistributeDispatchSetupInfo.k;
    int64_t BS = tilingData_->moeDistributeDispatchSetupInfo.bs;
    int64_t H = tilingData_->moeDistributeDispatchSetupInfo.h;

    int64_t epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    int64_t sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    int64_t localExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum;

    int64_t tokenMsgSize = yOutShape->GetStorageShape().GetDim(1);
    int64_t expandIdxOutSize = expandIdxOutShape->GetStorageShape().GetDim(0);
    int64_t commCmdInfoOutSize = commCmdInfoOutShape->GetStorageShape().GetDim(0);
    int64_t quantMode = tilingData_->moeDistributeDispatchSetupInfo.quantMode;
    int64_t tokenMsgSizeGolden = 0;
    if (quantMode == UNQUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(H, ALIGN_256);
    } else if (quantMode == STATIC_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(H, ALIGN_512);
    } else if (quantMode == PERTOKEN_DYNAMIC_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(ops::CeilAlign(H, ALIGN_32) + QUANT_HS_OFFSET, ALIGN_512);
    } else if (quantMode == PERGROUP_DYNAMIC_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(ops::CeilAlign(H, ALIGN_128) + ops::CeilDiv(H, ALIGN_128), ALIGN_512);
    } else if (quantMode == MX_QUANT) {
        tokenMsgSizeGolden = ops::CeilAlign(
            ops::CeilAlign(H, ALIGN_256) + ops::CeilAlign(ops::CeilDiv(H, ALIGN_32), EVEN_ALIGN), ALIGN_512);
    }

    OP_TILING_CHECK(tokenMsgSize != tokenMsgSizeGolden,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "yOut",
                                                          (std::string("dim1=") + std::to_string(tokenMsgSize)).c_str(),
                                                          "Dim1 of yOut must be equal to tokenMsgSize"),
                    return ge::GRAPH_FAILED);

    const int64_t expandIdxOutSizeGolden = BS * K;
    OP_TILING_CHECK(expandIdxOutSize != expandIdxOutSizeGolden,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_, "expandIdxOut", (std::string("dim0=") + std::to_string(expandIdxOutSize)).c_str(),
                        "Dim0 of expandIdxOut must be equal to BS * K"),
                    return ge::GRAPH_FAILED);

    const int64_t commCmdInfoOutSizeGolden =
        (BS * (K + sharedExpertNum) + epWorldSize * localExpertNum) * COMM_CMD_INFO_MULTIPLY;
    OP_TILING_CHECK(
        commCmdInfoOutSize != commCmdInfoOutSizeGolden,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "commCmdInfoOut",
                                              (std::string("dim1=") + std::to_string(commCmdInfoOutSize)).c_str(),
                                              "Dim1 of commCmdInfoOut must be equal to the expected value"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MoeDistributeDispatchSetupTilingBase::SetTilingKey()
{
    /*
     * tilingkey说明
     * 4位的十进制数
     * 第1位（个位）：quantMode:
     *     0: 不量化, 1: 静态量化, 2: 动态量化
     * 第2位（十位）：是否有smoothScale:
     *     0: 无, 1: 有
     * 第3位（百位）：0
     * 第4位（千位）：1
     */
    uint64_t tilingKey = INIT_TILINGKEY;
    tilingKey += static_cast<uint64_t>(tilingData_->moeDistributeDispatchSetupInfo.quantMode);
    // 这里的是判断scales的
    if (context_->GetOptionalInputShape(SCALES_INDEX) != nullptr) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_SCALES);
    }
    context_->SetTilingKey(tilingKey);
    return;
}

const ge::graphStatus MoeDistributeDispatchSetupTilingBase::CheckHcclBuffSize()
{
    const uint64_t hcclBuffSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() / 2;
    OP_TILING_CHECK(hcclBuffSize < MIN_AVAILABLE_BUFF_SIZE,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%ld] < [%ld].", hcclBuffSize, MIN_AVAILABLE_BUFF_SIZE),
                    return ge::GRAPH_FAILED);

    uint32_t &epWorldSize = tilingData_->moeDistributeDispatchSetupInfo.epWorldSize;
    uint32_t &sharedExpertNum = tilingData_->moeDistributeDispatchSetupInfo.sharedExpertNum;
    uint32_t &globalBs = tilingData_->moeDistributeDispatchSetupInfo.globalBs;
    uint32_t &BS = tilingData_->moeDistributeDispatchSetupInfo.bs;
    uint32_t &H = tilingData_->moeDistributeDispatchSetupInfo.h;
    uint32_t &K = tilingData_->moeDistributeDispatchSetupInfo.k;
    uint32_t &localExpertNum = tilingData_->moeDistributeDispatchSetupInfo.moeExpertPerRankNum;
    uint32_t maxBs = BS;
    if (globalBs != 0) {
        maxBs = globalBs / epWorldSize;
    }
    uint64_t align = ops::CeilAlign(
        (ops::CeilAlign(2U * H, static_cast<uint32_t>(ALIGN_32)) + static_cast<uint64_t>(HCCL_BUFFER_SIZE)),
        static_cast<uint64_t>(ALIGN_512));
    const uint64_t hcclBuffSizeGolden =
        (MIN_AVAILABLE_BUFF_SIZE * localExpertNum * maxBs * epWorldSize * align) +
        (K + sharedExpertNum) * maxBs * ops::CeilAlign(2U * H, static_cast<uint32_t>(ALIGN_512));

    OP_TILING_CHECK(hcclBuffSize < hcclBuffSizeGolden,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%lu] < [%lu].", hcclBuffSize, hcclBuffSizeGolden),
                    return ge::GRAPH_FAILED);

    tilingData_->moeDistributeDispatchSetupInfo.totalWinSize = hcclBuffSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchSetupTilingBase::SetWorkspace()
{
    size_t *workspace = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspace[0] = static_cast<size_t>(SYSTEM_NEED_WORKSPACE) + SDMA_NEED_WORKSPACE;
    OP_LOGD(nodeName_, "workspace[0] size is %lu", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

void MoeDistributeDispatchSetupTilingBase::SetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetAicpuBlockDim(AICPUNUM);
    tilingData_->moeDistributeDispatchSetupInfo.totalUbSize = ubSize;
    tilingData_->moeDistributeDispatchSetupInfo.aivNum = aivNum;
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName_, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    uint32_t sdmaUsedStreamPerCore = LOCAL_STREAM_MAX_NUM / aivNum;
    tilingData_->moeDistributeDispatchSetupInfo.sdmaUsedStreamPerCore = sdmaUsedStreamPerCore;
}

void MoeDistributeDispatchSetupTilingBase::SetHcommCfg() {};

ge::graphStatus MoeDistributeDispatchSetupTilingBase::MoeDistributeDispatchSetupTilingFuncImpl()
{
    OP_LOGD(nodeName_, "MoeDistributeDispatchSetupTilingFunc start");
    tilingData_ = context_->GetTilingData<MoeDistributeDispatchSetupTilingData>();
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(nodeName_, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // 获取入参属性
    if (!((GetRequiredAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetOptionalAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetComplexAttrAndSetTilingData() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }

    if (!((CheckTensorDataType() == ge::GRAPH_SUCCESS) && (CheckTensorDim() == ge::GRAPH_SUCCESS) &&
          (CheckTensorShapeRelation() == ge::GRAPH_SUCCESS) &&
          (CheckTensorShapeSizeAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (CheckCalcTensorShapeSizeAndSetTilingData() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }

    if (CheckHcclBuffSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetPlatformInfo();

    if (SetWorkspace() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetTilingKey();
    SetHcommCfg();
    PrintTilingDataInfo();
    OP_LOGD(nodeName_, "MoeDistributeDispatchSetupTilingFunc success");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
