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
 * \file moe_distribute_combine_teardown_tiling_arch22.cpp
 * \brief
 */

#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "moe_distribute_combine_teardown_tiling_arch22.h"

namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 8;
constexpr uint32_t ATTR_COMM_TYPE_INDEX = 9;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MIN_GROUP_EP_SIZE = 2;
constexpr int64_t MAX_GROUP_EP_SIZE = 384;
constexpr int64_t NON_QUANT = 0;
constexpr int64_t DYNAMIC_QUANT = 2;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t SDMA_COMM = 0;
constexpr int64_t URMA_COMM = 2;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t SDMA_NEED_WORKSPACE = 16U * 1024 * 1024;
} // namespace

namespace MC2Tiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombineTeardown, MoeDistributeCombineTeardownTilingA3, 1);
ge::graphStatus MoeDistributeCombineTeardownTilingA3::CheckAttrsWithoutRelation()
{
    auto attrs = context_->GetAttrs();

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    auto commTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_TYPE_INDEX);

    OP_TILING_CHECK((strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName_, "groupEp", std::to_string(strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH)).c_str(),
                        (std::string("[1, ") + std::to_string(MAX_GROUP_NAME_LENGTH) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epWorldSizePtr < MIN_GROUP_EP_SIZE) || (*epWorldSizePtr > MAX_GROUP_EP_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(),
                                              (std::string("[") + std::to_string(MIN_GROUP_EP_SIZE) + ", " +
                                               std::to_string(MAX_GROUP_EP_SIZE) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *expertShardTypePtr != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "expertShardType", std::to_string(*expertShardTypePtr).c_str(), "0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(),
                                  (std::string("[0, ") + std::to_string(MAX_SHARED_EXPERT_NUM) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*quantModePtr != NON_QUANT) && (*quantModePtr != DYNAMIC_QUANT),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "quantMode", std::to_string(*quantModePtr).c_str(),
                                  (std::to_string(NON_QUANT) + " or " + std::to_string(DYNAMIC_QUANT)).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*commTypePtr < SDMA_COMM) || (*commTypePtr > URMA_COMM),
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName_, "commType", std::to_string(*commTypePtr).c_str(),
            (std::string("[") + std::to_string(SDMA_COMM) + ", " + std::to_string(URMA_COMM) + "]").c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA3::CheckAttrsComplex()
{
    auto attrs = context_->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MAX_MOE_EXPERT_NUM),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_MOE_EXPERT_NUM) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OP_TILING_CHECK(
        (*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr > *epWorldSizePtr / 2),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "sharedExpertRankNum", std::to_string(*sharedExpertRankNumPtr).c_str(),
                                  (std::string("[0, ") + std::to_string(*epWorldSizePtr / 2) + "]").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*moeExpertNumPtr % (*epWorldSizePtr - *sharedExpertRankNumPtr) != 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              "moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA3::SetHcommCfg()
{
    const char* nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombine groupEp = %s", groupEp_.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
