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
 * \file moe_distribute_combine_setup_tiling_arch22.cpp
 * \brief
 */

#include "moe_distribute_combine_setup_tiling_arch22.h"

#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace {
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
constexpr int64_t MIN_GROUP_EP_SIZE = 2;
constexpr int64_t MAX_GROUP_EP_SIZE = 384;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 40U;
constexpr int64_t MIN_H = 1024;
constexpr int64_t MAX_H = 8192;
constexpr int64_t MAX_BS = 512;
constexpr int64_t MAX_K = 16;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
} // namespace

namespace MC2Tiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombineSetup, MoeDistributeCombineSetupTilingA3, 1);
ge::graphStatus MoeDistributeCombineSetupTilingA3::CheckEpWorldSize()
{
    auto attrs = context_->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);

    OP_TILING_CHECK(((*epWorldSizePtr < MIN_GROUP_EP_SIZE) || (*epWorldSizePtr > MAX_GROUP_EP_SIZE)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(),
                                              (std::string("[") + std::to_string(MIN_GROUP_EP_SIZE) + ", " +
                                               std::to_string(MAX_GROUP_EP_SIZE) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingA3::CheckMoeExpertNum()
{
    auto attrs = context_->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    OP_TILING_CHECK(((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MAX_MOE_EXPERT_NUM)),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_MOE_EXPERT_NUM) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingA3::CheckSharedExpertAttr()
{
    auto attrs = context_->GetAttrs();
    auto expertShardTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);

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
         (*sharedExpertRankNumPtr > (tilingData_->moeDistributeCombineSetupInfo.epWorldSize / 2))),
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName_, "sharedExpertRankNum", std::to_string(*sharedExpertRankNumPtr).c_str(),
            (std::string("[0, ") + std::to_string(tilingData_->moeDistributeCombineSetupInfo.epWorldSize / 2) + "]")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingA3::CheckTensorShapeSize(int64_t h, int64_t bs, int64_t k)
{
    OP_TILING_CHECK((h < MIN_H) || (h > MAX_H),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName_, "H", std::to_string(h).c_str(),
                        (std::string("[") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((bs <= 0) || (bs > MAX_BS),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "BS", std::to_string(bs).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_BS) + "]").c_str()),
                    return ge::GRAPH_FAILED);

    uint32_t& moeExpertNum = tilingData_->moeDistributeCombineSetupInfo.moeExpertNum;
    OP_TILING_CHECK(
        (k <= 0) || (k > MAX_K) || (k > static_cast<int64_t>(moeExpertNum)),
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName_, "K", std::to_string(k).c_str(),
            (std::string("(0, min(") + std::to_string(MAX_K) + ", " + std::to_string(moeExpertNum) + ")]").c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeDistributeCombineSetupTilingA3::SetHcommCfg()
{
    OP_LOGD(nodeName_, "MoeDistributeCombineSetup groupEp = %s", groupEp_.c_str());
    uint32_t opType = OP_TYPE_BATCH_WRITE;
    std::string algConfigStr = "BatchWrite=level0:fullmesh";
    uint32_t reduceType = 0U;

    uint32_t& aivNum = tilingData_->moeDistributeCombineSetupInfo.aivNum;
    OP_LOGD(nodeName_, "QueueNum = %u, aivNum = %u", (LOCAL_STREAM_MAX_NUM / aivNum), aivNum);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType, algConfigStr, reduceType);
    mc2CcTilingConfig.SetQueueNum(LOCAL_STREAM_MAX_NUM / aivNum);
    mc2CcTilingConfig.SetCommBlockNum(aivNum);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
}
} // namespace MC2Tiling
