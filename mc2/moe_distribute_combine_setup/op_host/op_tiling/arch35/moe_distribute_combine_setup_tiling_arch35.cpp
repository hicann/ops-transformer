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
 * \file moe_distribute_combine_setup_tiling_arch35.cpp
 * \brief
 */

#include "moe_distribute_combine_setup_tiling_arch35.h"

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
constexpr int64_t GROUP_EP_SIZE_2 = 2;
constexpr int64_t GROUP_EP_SIZE_4 = 4;
constexpr int64_t GROUP_EP_SIZE_8 = 8;
constexpr int64_t MOE_EXPERT_NUM_32 = 32;
constexpr int64_t MOE_EXPERT_NUM_64 = 64;
constexpr int64_t MOE_EXPERT_NUM_128 = 128;
constexpr int64_t BS_SIZE_8 = 8;
constexpr int64_t BS_SIZE_16 = 16;
constexpr int64_t BS_SIZE_256 = 256;
constexpr int64_t H_SIZE_4096 = 4096;
constexpr int64_t H_SIZE_7168 = 7168;
constexpr int64_t K_SIZE_6 = 6;
constexpr int64_t K_SIZE_8 = 8;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocol;
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};
} // namespace

namespace MC2Tiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombineSetup, MoeDistributeCombineSetupTilingA5, 0);
bool MoeDistributeCombineSetupTilingA5::IsCapable()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    if (ascendcPlatform.GetCurNpuArch() == Ops::Base::DAV_3510) {
        return true;
    }
    return false;
}

ge::graphStatus MoeDistributeCombineSetupTilingA5::CheckEpWorldSize()
{
    auto attrs = context_->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);

    OP_TILING_CHECK(
        (!((*epWorldSizePtr == GROUP_EP_SIZE_2) || (*epWorldSizePtr == GROUP_EP_SIZE_4) ||
           (*epWorldSizePtr == GROUP_EP_SIZE_8))),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(), "{2, 4, 8}"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingA5::CheckMoeExpertNum()
{
    auto attrs = context_->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);

    OP_TILING_CHECK(
        !(*moeExpertNumPtr == MOE_EXPERT_NUM_32),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(), "32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingA5::CheckSharedExpertAttr()
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
        (*sharedExpertNumPtr != 0),
        OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(), "0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertRankNumPtr != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName_, "sharedExpertRankNum",
                                              std::to_string(*sharedExpertRankNumPtr).c_str(), "0"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineSetupTilingA5::CheckTensorShapeSize(int64_t h, int64_t bs, int64_t k)
{
    OP_TILING_CHECK(((bs != BS_SIZE_8) && (bs != BS_SIZE_16) && (bs != BS_SIZE_256)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "BS", std::to_string(bs).c_str(), "{8, 16, 256}"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(((h != H_SIZE_4096) && (h != H_SIZE_7168)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "H", std::to_string(h).c_str(), "{4096, 7168}"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(((k != K_SIZE_6) && (k != K_SIZE_8)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "K", std::to_string(k).c_str(), "{6, 8}"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeDistributeCombineSetupTilingA5::SetHcommCfg()
{
    OP_LOGD(nodeName_, "MoeDistributeCombineSetup groupEp = %s", groupEp_.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint8_t aivEngineValue = mc2tiling::AIV_ENGINE;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(aivEngineValue); // AIV_UB-MEM or AIV_URMA
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
    reinterpret_cast<Mc2CcTilingInner *>(&tilingData_->mc2CcTiling)->protocol = 1; // 0: UB-MEM, 1: URMA
}
} // namespace MC2Tiling
