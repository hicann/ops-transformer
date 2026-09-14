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
 * \file moe_distribute_combine_setup_tiling_base.cpp
 * \brief
 */

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_combine_setup_tiling_base.h"

namespace {
constexpr uint32_t EXPAND_X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t ASSIST_INFO_INDEX = 2U;
constexpr uint32_t QUANT_EXPAND_X_OUT_INDEX = 0U;
constexpr uint32_t COMM_CMD_INFO_OUT_INDEX = 1U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 7;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 8;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 9;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 10;

constexpr uint32_t LOCAL_STREAM_MAX_NUM = 40U;
constexpr uint32_t USED_AIV_NUMS = 40U;
constexpr int64_t MIN_H = 1024;
constexpr int64_t MAX_H = 8192;
constexpr int64_t MAX_BS = 512;
constexpr int64_t MAX_K = 16;

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;

constexpr uint32_t COMM_QUANT_NONE = 0U;
constexpr uint32_t COMM_QUANT_INT12 = 1U;
constexpr uint32_t COMM_QUANT_INT8 = 2U;

constexpr uint32_t NO_SCALES = 0U;
constexpr uint32_t STATIC_SCALES = 1U;
constexpr uint32_t DYNAMIC_SCALES = 2U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MIN_GROUP_EP_SIZE = 2;
constexpr int64_t MAX_GROUP_EP_SIZE = 384;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t SDMA_COMM = 0;
constexpr int64_t URMA_COMM = 2;
constexpr int64_t QUANT_HS_OFFSET = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 4;
constexpr int64_t BS_UPPER_BOUND = 4;
constexpr int64_t ALIGN_8 = 8UL;
constexpr int64_t ALIGN_32 = 32UL;
constexpr int64_t ALIGN_512 = 512UL;
constexpr uint32_t AICPUNUM = 4U;

constexpr size_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr int64_t COMM_CMD_INFO_SIZE = 16;
constexpr uint64_t MIN_AVAILABLE_BUFF_SIZE = 2;
constexpr int64_t HCCL_BUFFER_SIZE = 44;
} // namespace

namespace MC2Tiling {
void MoeDistributeCombineSetupTilingBase::PrintTilingDataInfo()
{
    const MoeDistributeCombineSetupInfo &info = tilingData_->moeDistributeCombineSetupInfo;
    OP_LOGD(nodeName_, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(nodeName_, "epRankId is %u.", info.epRankId);
    OP_LOGD(nodeName_, "expertShardType is %u.", info.expertShardType);
    OP_LOGD(nodeName_, "sharedExpertNum is %u.", info.sharedExpertNum);
    OP_LOGD(nodeName_, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(nodeName_, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(nodeName_, "moeExpertPerRankNum is %u.", info.moeExpertPerRankNum);
    OP_LOGD(nodeName_, "commQuantMode is %u.", info.commQuantMode);
    OP_LOGD(nodeName_, "globalBs is %u.", info.globalBs);
    OP_LOGD(nodeName_, "bs is %u.", info.bs);
    OP_LOGD(nodeName_, "k is %u.", info.k);
    OP_LOGD(nodeName_, "h is %u.", info.h);
    OP_LOGD(nodeName_, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName_, "totalUbSize is %lu.", info.totalUbSize);
    OP_LOGD(nodeName_, "totalWinSize is %lu.", info.totalWinSize);
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckEpWorldSize()
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckMoeExpertNum()
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckRequiredAttrValue()
{
    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    OP_TILING_CHECK(((strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                     (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH)),
                    OP_LOGE_WITH_INVALID_ATTR(
                        nodeName_, "groupEp",
                        (std::string("length=") + std::to_string(strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH))).c_str(),
                        "valid group name length"),
                    return ge::GRAPH_FAILED);

    if (CheckEpWorldSize() != ge::GRAPH_SUCCESS || CheckMoeExpertNum() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::GetRequiredAttrAndSetTilingData()
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
    tilingData_->moeDistributeCombineSetupInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData_->moeDistributeCombineSetupInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData_->moeDistributeCombineSetupInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckSharedExpertAttrValue()
{
    const uint32_t &sharedExpertNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertNum;
    const uint32_t &sharedExpertRankNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertRankNum;
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

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckSharedExpertAttr()
{
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckOptionalAttrValue()
{
    if (CheckSharedExpertAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);

    // globalBs 会在后面获取 BS 后再次校验
    OP_TILING_CHECK((*globalBsPtr < 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "globalBs", std::to_string(*globalBsPtr).c_str(),
                                              "0 or maxBs * epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*commQuantModePtr != COMM_QUANT_NONE),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "commQuantMode", std::to_string(*commQuantModePtr).c_str(), "0"),
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

ge::graphStatus MoeDistributeCombineSetupTilingBase::GetOptionalAttrAndSetTilingData()
{
    OP_LOGD("GetOptionalAttrAndSetTilingData");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "attrs"), return ge::GRAPH_FAILED);

    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);

    // 判空
    OP_TILING_CHECK(expertShardTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertShardType"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "globalBs"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commType"), return ge::GRAPH_FAILED);

    if (CheckOptionalAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 设置 tilingdata
    tilingData_->moeDistributeCombineSetupInfo.expertShardType = static_cast<uint32_t>(*expertShardTypePtr);
    tilingData_->moeDistributeCombineSetupInfo.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData_->moeDistributeCombineSetupInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData_->moeDistributeCombineSetupInfo.commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    tilingData_->moeDistributeCombineSetupInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);

    // 校验共享专家限制
    if (CheckSharedExpertAttrValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckMoeExpertNumPerRank()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::GetComplexAttrAndSetTilingData()
{
    const uint32_t &epRankId = tilingData_->moeDistributeCombineSetupInfo.epRankId;
    const uint32_t &epWorldSize = tilingData_->moeDistributeCombineSetupInfo.epWorldSize;
    const uint32_t &sharedExpertNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertNum;
    const uint32_t &sharedExpertRankNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertRankNum;
    const uint32_t &moeExpertNum = tilingData_->moeDistributeCombineSetupInfo.moeExpertNum;

    OP_TILING_CHECK((moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "moeExpertNum",
                                              (std::string("moeExpertNum=") + std::to_string(moeExpertNum) +
                                               ", epWorldSize=" + std::to_string(epWorldSize) +
                                               ", sharedExpertRankNum=" + std::to_string(sharedExpertRankNum))
                                                  .c_str(),
                                              "moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0"),
                    return ge::GRAPH_FAILED);

    // localMoeExpertNum
    if (epRankId >= sharedExpertRankNum) {
        // MoE 专家卡
        tilingData_->moeDistributeCombineSetupInfo.moeExpertPerRankNum =
            moeExpertNum / (epWorldSize - sharedExpertRankNum);
        if (CheckMoeExpertNumPerRank() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 共同专家卡
        tilingData_->moeDistributeCombineSetupInfo.moeExpertPerRankNum = 1U;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckOneTensorDim(std::string name, TensorType tensortype,
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

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckInputTensorDim()
{
    OP_TILING_CHECK(
        CheckOneTensorDim("expandX", TensorType::INPUT, EXPAND_X_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "expandX", "checkdim failed", "expandX must be 2D"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("expertIds", TensorType::INPUT, EXPERT_IDS_INDEX, TWO_DIMS) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "expertIds", "checkdim failed", "expertIds must be 2D"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("assistInfoForCombine", TensorType::INPUT, ASSIST_INFO_INDEX, ONE_DIM) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "assistInfoForCombine", "checkdim failed",
                                                 "assistInfoForCombine must be 1D"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckOutputTensorDim()
{
    OP_TILING_CHECK(CheckOneTensorDim("quantExpandXOut", TensorType::OUTPUT, QUANT_EXPAND_X_OUT_INDEX, TWO_DIMS) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "quantExpandXOut", "checkdim failed",
                                                             "quantExpandXOut must be 2D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckOneTensorDim("commCmdInfoOut", TensorType::OUTPUT, COMM_CMD_INFO_OUT_INDEX, ONE_DIM) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_, "commCmdInfoOut", "checkdim failed",
                                                 "commCmdInfoOut must be 1D"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorDim()
{
    OP_TILING_CHECK(CheckInputTensorDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "input", "shape invalid",
                                                          "Input param shape validation failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputTensorDim() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "output", "shape invalid",
                                                          "Output param shape validation failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorShapeRelation()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);
    auto quantExpandXOutStorageShape = context_->GetOutputShape(QUANT_EXPAND_X_OUT_INDEX);
    OP_TILING_CHECK(
        (expandXStorageShape->GetStorageShape().GetDim(0) != quantExpandXOutStorageShape->GetStorageShape().GetDim(0)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName_, "expandX/quantExpandXOut",
            (std::string("expandX dim0=") + std::to_string(expandXStorageShape->GetStorageShape().GetDim(0)) +
             ", quantExpandXOut dim0=" + std::to_string(quantExpandXOutStorageShape->GetStorageShape().GetDim(0)))
                .c_str(),
            "Dim0 of expandX must be equal to dim0 of quantExpandXOut"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorShapeSizeAInMoeRank()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);
    int64_t A = expandXStorageShape->GetStorageShape().GetDim(0U);

    uint32_t &BS = tilingData_->moeDistributeCombineSetupInfo.bs;
    uint32_t &K = tilingData_->moeDistributeCombineSetupInfo.k;
    uint32_t &epWorldSize = tilingData_->moeDistributeCombineSetupInfo.epWorldSize;
    uint32_t &globalBs = tilingData_->moeDistributeCombineSetupInfo.globalBs;
    uint32_t &localMoeExpertNum = tilingData_->moeDistributeCombineSetupInfo.moeExpertPerRankNum;

    if (globalBs == 0) {
        // MoE 专家卡 均分
        if (!(A >= (BS * epWorldSize * std::min(localMoeExpertNum, K)))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName_, "expandX", (std::string("dim0(A)=") + std::to_string(A)).c_str(),
                "A should >= BS * epWorldSize * min(localMoeExpertNum, K) when globalBs is 0");
            return ge::GRAPH_FAILED;
        }
    } else {
        // MoE 专家卡 非均分
        if (!(A >= (globalBs * std::min(localMoeExpertNum, K)))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "expandX",
                                                  (std::string("dim0(A)=") + std::to_string(A)).c_str(),
                                                  "A should >= globalBs * min(localMoeExpertNum, K)");
            return ge::GRAPH_FAILED;
        }
    }

    tilingData_->moeDistributeCombineSetupInfo.a = static_cast<uint32_t>(A);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorShapeSizeAInSharedRank()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);
    int64_t A = expandXStorageShape->GetStorageShape().GetDim(0U);

    uint32_t &BS = tilingData_->moeDistributeCombineSetupInfo.bs;
    uint32_t &epWorldSize = tilingData_->moeDistributeCombineSetupInfo.epWorldSize;
    uint32_t &sharedExpertNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertNum;
    uint32_t &sharedExpertRankNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertRankNum;
    uint32_t &globalBs = tilingData_->moeDistributeCombineSetupInfo.globalBs;

    if (globalBs == 0) {
        // 共享专家卡 均分
        if (!(A == BS * epWorldSize * sharedExpertNum / sharedExpertRankNum)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName_, "expandX", (std::string("dim0(A)=") + std::to_string(A)).c_str(),
                "A should == BS * epWorldSize * sharedExpertNum / sharedExpertRankNum when globalBs is 0");
            return ge::GRAPH_FAILED;
        }
    } else {
        // 共享专家卡 非均分
        if (!(A == globalBs * sharedExpertNum / sharedExpertRankNum)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "expandX",
                                                  (std::string("dim0(A)=") + std::to_string(A)).c_str(),
                                                  "A should == globalBs * sharedExpertNum / sharedExpertRankNum");
            return ge::GRAPH_FAILED;
        }
    }
    tilingData_->moeDistributeCombineSetupInfo.a = static_cast<uint32_t>(A);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorShapeSize(int64_t h, int64_t bs, int64_t k)
{
    (void)h;
    (void)bs;
    (void)k;
    return ge::GRAPH_FAILED;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorShapeSizeAndSetTilingData()
{
    auto expandXStorageShape = context_->GetInputShape(EXPAND_X_INDEX);
    auto expertIdsShape = context_->GetInputShape(EXPERT_IDS_INDEX);
    int64_t H = expandXStorageShape->GetStorageShape().GetDim(1U);
    int64_t BS = expertIdsShape->GetStorageShape().GetDim(0U);
    int64_t K = expertIdsShape->GetStorageShape().GetDim(1U);
    if (CheckTensorShapeSize(H, BS, K) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t &globalBs = tilingData_->moeDistributeCombineSetupInfo.globalBs;
    uint32_t &epWorldSize = tilingData_->moeDistributeCombineSetupInfo.epWorldSize;
    OP_TILING_CHECK(
        (globalBs != 0) && ((globalBs < BS * epWorldSize) || (globalBs > MAX_BS * epWorldSize)),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "globalBs", std::to_string(globalBs).c_str(),
                                  (std::string("0 or [") + std::to_string(BS * epWorldSize) + std::string(", ") +
                                   std::to_string(MAX_BS * epWorldSize) + std::string("]"))
                                      .c_str()),
        return ge::GRAPH_FAILED);

    tilingData_->moeDistributeCombineSetupInfo.h = static_cast<uint32_t>(H);
    tilingData_->moeDistributeCombineSetupInfo.bs = static_cast<uint32_t>(BS);
    tilingData_->moeDistributeCombineSetupInfo.k = static_cast<uint32_t>(K);

    uint32_t &epRankId = tilingData_->moeDistributeCombineSetupInfo.epRankId;
    uint32_t &sharedExpertRankNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertRankNum;
    if (epRankId >= sharedExpertRankNum) {
        // MoE 专家卡
        if (CheckTensorShapeSizeAInMoeRank() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 共享专家卡
        if (CheckTensorShapeSizeAInSharedRank() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckCalcTensorShapeSizeAndSetTilingData()
{
    auto assistInfoForCombineShape = context_->GetInputShape(ASSIST_INFO_INDEX);
    auto quantExpandXOutShape = context_->GetOutputShape(QUANT_EXPAND_X_OUT_INDEX);
    auto commCmdInfoShape = context_->GetOutputShape(COMM_CMD_INFO_OUT_INDEX);
    int64_t A = tilingData_->moeDistributeCombineSetupInfo.a;
    int64_t H = tilingData_->moeDistributeCombineSetupInfo.h;
    int64_t epWorldSize = tilingData_->moeDistributeCombineSetupInfo.epWorldSize;
    int64_t assistInfoForCombineOutSize = assistInfoForCombineShape->GetStorageShape().GetDim(0);
    int64_t tokenMsgSize = quantExpandXOutShape->GetStorageShape().GetDim(1);
    int64_t commCmdInfoOutSize = commCmdInfoShape->GetStorageShape().GetDim(0);

    int64_t assistInfoForCombineOutSizeGolden = A * 128;
    OP_TILING_CHECK(assistInfoForCombineOutSize != assistInfoForCombineOutSizeGolden,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_, "assistInfoForCombine",
                        (std::string("dim0=") + std::to_string(assistInfoForCombineOutSize)).c_str(),
                        "Dim0 of assistInfoForCombine must be equal to A * 128"),
                    return ge::GRAPH_FAILED);

    int64_t tokenMsgSizeGolden = ops::CeilAlign(
        static_cast<int64_t>(ops::CeilAlign(H, ALIGN_32) + ops::CeilAlign(H, ALIGN_8) / ALIGN_8 * sizeof(float)),
        ALIGN_512);
    OP_TILING_CHECK(tokenMsgSize != tokenMsgSizeGolden,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "quantExpandXOut",
                                                          (std::string("dim1=") + std::to_string(tokenMsgSize)).c_str(),
                                                          "Dim1 of quantExpandXOut must be equal to tokenMsgSize"),
                    return ge::GRAPH_FAILED);

    int64_t commCmdInfoOutSizeGolden = (A + epWorldSize) * COMM_CMD_INFO_SIZE;
    OP_TILING_CHECK(
        commCmdInfoOutSize != commCmdInfoOutSizeGolden,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_, "commCmdInfoOut",
                                              (std::string("dim0=") + std::to_string(commCmdInfoOutSize)).c_str(),
                                              "Dim0 of commCmdInfoOut must be equal to (A + epWorldSize) * 16"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckTensorDataTypeAndFormat()
{
    auto expandXDesc = context_->GetInputDesc(EXPAND_X_INDEX);
    auto expertIdsDesc = context_->GetInputDesc(EXPERT_IDS_INDEX);
    auto assistInfoForCombineDesc = context_->GetInputDesc(ASSIST_INFO_INDEX);
    auto quantExpandXOutDesc = context_->GetOutputDesc(QUANT_EXPAND_X_OUT_INDEX);
    auto commCmdInfoOutDesc = context_->GetOutputDesc(COMM_CMD_INFO_OUT_INDEX);

    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expandX"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "expertIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(assistInfoForCombineDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "assistInfoForCombine"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantExpandXOutDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "quantExpandXOut"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commCmdInfoOutDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName_, "commCmdInfoOut"),
                    return ge::GRAPH_FAILED);

    if (CheckTensorDataType() && CheckTensorDataFormat()) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

bool MoeDistributeCombineSetupTilingBase::CheckTensorDataType()
{
    auto expandXDesc = context_->GetInputDesc(EXPAND_X_INDEX);
    auto expertIdsDesc = context_->GetInputDesc(EXPERT_IDS_INDEX);
    auto assistInfoForCombineDesc = context_->GetInputDesc(ASSIST_INFO_INDEX);
    auto quantExpandXOutDesc = context_->GetOutputDesc(QUANT_EXPAND_X_OUT_INDEX);
    auto commCmdInfoOutDesc = context_->GetOutputDesc(COMM_CMD_INFO_OUT_INDEX);

    OP_TILING_CHECK(
        (expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "expandX", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                  "bfloat16 or float16"),
        return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "expertIds",
                                              Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str(), "int32"),
                    return false);
    OP_TILING_CHECK(
        (assistInfoForCombineDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "assistInfoForCombine",
                                  Ops::Base::ToString(assistInfoForCombineDesc->GetDataType()).c_str(), "int32"),
        return false);
    OP_TILING_CHECK((quantExpandXOutDesc->GetDataType() != ge::DT_INT8),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "quantExpandXOut",
                                              Ops::Base::ToString(quantExpandXOutDesc->GetDataType()).c_str(), "int8"),
                    return false);
    OP_TILING_CHECK((commCmdInfoOutDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_, "commCmdInfoOut",
                                              Ops::Base::ToString(commCmdInfoOutDesc->GetDataType()).c_str(), "int32"),
                    return false);
    return true;
}

bool MoeDistributeCombineSetupTilingBase::CheckTensorDataFormat()
{
    auto expandXDesc = context_->GetInputDesc(EXPAND_X_INDEX);
    auto expertIdsDesc = context_->GetInputDesc(EXPERT_IDS_INDEX);
    auto assistInfoForCombineDesc = context_->GetInputDesc(ASSIST_INFO_INDEX);
    auto quantExpandXOutDesc = context_->GetOutputDesc(QUANT_EXPAND_X_OUT_INDEX);
    auto commCmdInfoOutDesc = context_->GetOutputDesc(COMM_CMD_INFO_OUT_INDEX);

    ge::Format expandXFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        (expandXFormat == ge::FORMAT_FRACTAL_NZ),
        OP_LOGE_FOR_INVALID_FORMAT(nodeName_, "expandX", Ops::Base::ToString(expandXFormat).c_str(), "FRACTAL_NZ"),
        return false);

    ge::Format expertIdsFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat()));
    OP_TILING_CHECK(
        (expertIdsFormat == ge::FORMAT_FRACTAL_NZ),
        OP_LOGE_FOR_INVALID_FORMAT(nodeName_, "expertIds", Ops::Base::ToString(expertIdsFormat).c_str(), "FRACTAL_NZ"),
        return false);

    ge::Format assistInfoFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoForCombineDesc->GetStorageFormat()));
    OP_TILING_CHECK((assistInfoFormat == ge::FORMAT_FRACTAL_NZ),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName_, "assistInfoForCombine",
                                               Ops::Base::ToString(assistInfoFormat).c_str(), "FRACTAL_NZ"),
                    return false);

    ge::Format quantExpandXFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(quantExpandXOutDesc->GetStorageFormat()));
    OP_TILING_CHECK((quantExpandXFormat == ge::FORMAT_FRACTAL_NZ),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName_, "quantExpandXOut",
                                               Ops::Base::ToString(quantExpandXFormat).c_str(), "FRACTAL_NZ"),
                    return false);

    ge::Format commCmdInfoFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(commCmdInfoOutDesc->GetStorageFormat()));
    OP_TILING_CHECK((commCmdInfoFormat == ge::FORMAT_FRACTAL_NZ),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName_, "commCmdInfoOut",
                                               Ops::Base::ToString(commCmdInfoFormat).c_str(), "FRACTAL_NZ"),
                    return false);

    return true;
}

void MoeDistributeCombineSetupTilingBase::SetTilingKey()
{
    bool tp = false;

    // 设置tilingKey模板参数
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tp);
    context_->SetTilingKey(tilingKey);
    OP_LOGD(nodeName_, "tilingKey is [%lu].", tilingKey);
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::CheckHcclBuffSize()
{
    const uint64_t hcclBuffSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() / 2;
    OP_TILING_CHECK(
        hcclBuffSize < MIN_AVAILABLE_BUFF_SIZE,
        OP_LOGE(nodeName_, "HCCL_BUFFSIZE too short, [%ld] < [%ld].", hcclBuffSize, MIN_AVAILABLE_BUFF_SIZE),
        return ge::GRAPH_FAILED);

    uint32_t &epWorldSize = tilingData_->moeDistributeCombineSetupInfo.epWorldSize;
    uint32_t &moeExpertNum = tilingData_->moeDistributeCombineSetupInfo.moeExpertNum;
    uint32_t &sharedExpertNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertNum;
    uint32_t &sharedExpertRankNum = tilingData_->moeDistributeCombineSetupInfo.sharedExpertRankNum;
    uint32_t &globalBs = tilingData_->moeDistributeCombineSetupInfo.globalBs;
    uint32_t &BS = tilingData_->moeDistributeCombineSetupInfo.bs;
    uint32_t &H = tilingData_->moeDistributeCombineSetupInfo.h;
    uint32_t &K = tilingData_->moeDistributeCombineSetupInfo.k;
    uint32_t localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    uint32_t maxBs = BS;
    if (globalBs != 0) {
        maxBs = globalBs / epWorldSize;
    }
    uint32_t align = static_cast<uint32_t>(
        ops::CeilAlign((ops::CeilAlign(static_cast<int64_t>(2 * H), ALIGN_32) + HCCL_BUFFER_SIZE), ALIGN_512));
    const uint64_t hcclBuffSizeGolden =
        (MIN_AVAILABLE_BUFF_SIZE * localExpertNum * maxBs * epWorldSize * align) +
        (K + sharedExpertNum) * maxBs * ops::CeilAlign(2U * H, static_cast<uint32_t>(ALIGN_512));

    OP_TILING_CHECK(hcclBuffSize < hcclBuffSizeGolden,
                    OP_LOGE(nodeName_, "HCCL_BUFFSIZE [%lu] < [%lu].", hcclBuffSize, hcclBuffSizeGolden),
                    return ge::GRAPH_FAILED);

    tilingData_->moeDistributeCombineSetupInfo.totalWinSize = hcclBuffSize;

    return ge::GRAPH_SUCCESS;
}

void MoeDistributeCombineSetupTilingBase::SetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetAicpuBlockDim(AICPUNUM);
    tilingData_->moeDistributeCombineSetupInfo.totalUbSize = ubSize;
    tilingData_->moeDistributeCombineSetupInfo.aivNum = aivNum;
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName_, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::SetWorkspace()
{
    size_t *workspace = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName_, "get workspace failed."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspace[0] = libApiWorkSpaceSize;
    OP_LOGD(nodeName_, "workspace[0] size is %lu", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

void MoeDistributeCombineSetupTilingBase::SetHcommCfg()
{
    return;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::MoeDistributeCombineSetupTilingFuncImpl()
{
    OP_LOGD(nodeName_, "MoeDistributeCombineSetupTilingFunc start");
    tilingData_ = context_->GetTilingData<MoeDistributeCombineSetupTilingData>();
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(nodeName_, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // 获取入参属性
    if (!((GetRequiredAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetOptionalAttrAndSetTilingData() == ge::GRAPH_SUCCESS) &&
          (GetComplexAttrAndSetTilingData() == ge::GRAPH_SUCCESS))) {
        return ge::GRAPH_FAILED;
    }

    if (!((CheckTensorDataTypeAndFormat() == ge::GRAPH_SUCCESS) && (CheckTensorDim() == ge::GRAPH_SUCCESS) &&
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
    OP_LOGD(nodeName_, "MoeDistributeCombineSetupTilingFunc success");
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeCombineSetupTilingBase::IsCapable()
{
    return false;
}

ge::graphStatus MoeDistributeCombineSetupTilingBase::DoOpTiling()
{
    return MoeDistributeCombineSetupTilingFuncImpl();
}

uint64_t MoeDistributeCombineSetupTilingBase::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(nodeName_, "%s get tiling key %lu", this->socTilingName_, tilingKey);
    return tilingKey;
}
} // namespace MC2Tiling
