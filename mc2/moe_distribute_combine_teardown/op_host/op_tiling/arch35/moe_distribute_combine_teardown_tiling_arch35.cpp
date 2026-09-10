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
 * \file moe_distribute_combine_teardown_tiling_arch35.cpp
 * \brief
 */

#include "mc2_log.h"
#include "moe_distribute_combine_teardown_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 5;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 8;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t NON_QUANT = 0;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
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

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombineTeardown, MoeDistributeCombineTeardownTilingA5, 0);
bool MoeDistributeCombineTeardownTilingA5::IsCapable()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        return true;
    }
    return false;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA5::CheckAttrsWithoutRelation()
{
    auto attrs = context_->GetAttrs();

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);

    OP_TILING_CHECK((strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName_, "groupEp", std::to_string(strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH)).c_str(),
                        (std::string("[1, ") + std::to_string(MAX_GROUP_NAME_LENGTH) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !((*epWorldSizePtr == GROUP_EP_SIZE_2) || (*epWorldSizePtr == GROUP_EP_SIZE_4) ||
          (*epWorldSizePtr == GROUP_EP_SIZE_8)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(), "2, 4, or 8"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (*sharedExpertNumPtr != 0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(), "0"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*quantModePtr != NON_QUANT),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "quantMode", std::to_string(*quantModePtr).c_str(),
                                              std::to_string(NON_QUANT).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA5::CheckAttrsComplex()
{
    auto attrs = context_->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    OP_TILING_CHECK(
        !(*moeExpertNumPtr == MOE_EXPERT_NUM_32),
        OP_LOGE_FOR_INVALID_VALUE(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(), "32"),
        return ge::GRAPH_FAILED);

    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OP_TILING_CHECK((*sharedExpertRankNumPtr != 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "sharedExpertRankNum",
                                              std::to_string(*sharedExpertRankNumPtr).c_str(), "0"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*moeExpertNumPtr % (*epWorldSizePtr - *sharedExpertRankNumPtr) != 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              "moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA5::CheckBsHKSize(int64_t bs, int64_t h, int64_t k)
{
    OP_TILING_CHECK(((bs != BS_SIZE_8) && (bs != BS_SIZE_16) && (bs != BS_SIZE_256)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "Bs", std::to_string(bs).c_str(), "8, 16, or 256"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(((h != H_SIZE_4096) && (h != H_SIZE_7168)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "H", std::to_string(h).c_str(), "4096 or 7168"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(((k != K_SIZE_6) && (k != K_SIZE_8)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName_, "K", std::to_string(k).c_str(), "6 or 8"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTeardownTilingA5::SetHcommCfg()
{
    OP_LOGD(nodeName_, "MoeDistributeCombineTeardown groupEp = %s", groupEp_.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint8_t aivEngineValue = mc2tiling::AIV_ENGINE;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(aivEngineValue); // AIV_UB-MEM or AIV_URMA
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling) != 0,
                    OP_LOGE(nodeName_, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling) != 0,
                    OP_LOGE(nodeName_, "mc2CcTilingConfig mc2CcTiling1 GetTiling failed"), return ge::GRAPH_FAILED);
    reinterpret_cast<Mc2CcTilingInner *>(&tilingData_->mc2CcTiling)->protocol = 1; // 0: UB-MEM, 1: URMA

    return ge::GRAPH_SUCCESS;
}
} // namespace MC2Tiling
