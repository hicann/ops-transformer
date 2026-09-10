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
 * \file aclnn_moe_distribute_combine_setup.cpp
 * \brief
 */
#include "aclnn_moe_distribute_combine_setup.h"
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "mc2_log_compat.h"

namespace {

using namespace op;

static inline bool CheckEmptyTensor(const aclTensor *tensor, const char *name)
{
    const auto &shape = tensor->GetViewShape();
    if (shape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnMoeDistributeCombineSetup", name, "[]",
                                              (std::string("The shape of ") + name + " cannot be empty.").c_str());
        return false;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMoeDistributeCombineSetup", name,
                                                     std::to_string(shape.GetDim(i)).c_str(),
                                                     "dimension must be positive");
            return false;
        }
    }
    return true;
}

#define OP_CHECK_EMPTY_TENSOR(tensor, retExpr) \
    if (!CheckEmptyTensor(tensor, #tensor)) { \
        retExpr; \
    }

static constexpr size_t TWO_DIM = 2;
static constexpr int64_t ALIGN_8 = 8;
static constexpr int64_t ALIGN_32 = 32;
static constexpr int64_t ALIGN_512 = 512;
static constexpr int64_t COMM_CMD_INFO_BASE = 16;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static inline int64_t AlignUp(int64_t x, int64_t base)
{
    if (base == 0) {
        OP_LOGD("AlignUp: base cannot be zero");
        return 0;
    }
    return ((x + base - 1) / base) * base;
}

// check nullptr
static bool CheckNotNull(const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine,
                         const char *groupEp, aclTensor *quantExpandXOut, aclTensor *commCmdInfoOut)
{
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckNotNull start");
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(quantExpandXOut, return false);
    OP_CHECK_NULL(commCmdInfoOut, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMoeDistributeCombineSetup", "groupEp");
        return false;
    }
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckNotNull success");
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *expandX, const aclTensor *expertIds,
                               const aclTensor *assistInfoForCombine, const char *groupEp, int64_t epWorldSize,
                               int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
                               int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType,
                               const char *commAlg, aclTensor *quantExpandXOut, aclTensor *commCmdInfoOut)
{
    (void)epWorldSize;
    (void)epRankId;
    (void)moeExpertNum;
    (void)expertShardType;
    (void)sharedExpertNum;
    (void)sharedExpertRankNum;
    (void)globalBs;
    (void)commQuantMode;
    (void)commType;
    (void)commAlg;
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckParams start");
    CHECK_RET(CheckNotNull(expandX, expertIds, assistInfoForCombine, groupEp, quantExpandXOut, commCmdInfoOut),
              ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_EMPTY_TENSOR(expandX, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_EMPTY_TENSOR(expertIds, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_EMPTY_TENSOR(assistInfoForCombine, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_EMPTY_TENSOR(quantExpandXOut, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_EMPTY_TENSOR(commCmdInfoOut, return ACLNN_ERR_PARAM_INVALID);
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMoeDistributeCombineSetup", "groupEp",
                                              "length exceeds " + std::to_string(HCCL_GROUP_NAME_MAX),
                                              "groupEp name too long");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("aclnn_moe_distribute_combine_setup CheckParams success");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnInnerMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
    aclTensor *quantExpandXOut, aclTensor *commCmdInfoOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerMoeDistributeCombineSetup(void *workspace, uint64_t workspaceSize,
                                                           aclOpExecutor *executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnMoeDistributeCombineSetupGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
    aclTensor *quantExpandXOut, aclTensor *commCmdInfoOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMoeDistributeCombineSetupGetWorkspaceSize start.");
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnMoeDistributeCombineSetup", "npuArch",
            std::to_string(static_cast<int64_t>(GetCurrentPlatformInfo().GetCurNpuArch())).c_str(),
            "only support NpuArch::DAV_3510");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto retParam = CheckParams(expandX, expertIds, assistInfoForCombine, groupEp, epWorldSize, epRankId, moeExpertNum,
                                expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode,
                                commType, commAlg, quantExpandXOut, commCmdInfoOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclnnStatus retStatus = aclnnInnerMoeDistributeCombineSetupGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, const_cast<char *>(groupEp), epWorldSize, epRankId, moeExpertNum,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, commQuantMode, commType,
        const_cast<char *>(commAlg), quantExpandXOut, commCmdInfoOut, workspaceSize, executor);
    return retStatus;
}

extern "C" aclnnStatus aclnnMoeDistributeCombineSetupTeardownCalcOutputSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const char *groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, int64_t commType, const char *commAlg,
    uint64_t &tokenMsgSize, uint64_t &commCmdInfoOutSize)
{
    (void)expertIds;
    (void)assistInfoForCombine;
    (void)groupEp;
    (void)epRankId;
    (void)moeExpertNum;
    (void)expertShardType;
    (void)sharedExpertNum;
    (void)sharedExpertRankNum;
    (void)globalBs;
    (void)commQuantMode;
    (void)commType;
    (void)commAlg;
    OP_CHECK_NULL(expandX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_EMPTY_TENSOR(expandX, return ACLNN_ERR_PARAM_INVALID);
    if (expandX->GetViewShape().GetDimNum() != TWO_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMoeDistributeCombineSetup", "expandX",
                                                 "dim=" + std::to_string(expandX->GetViewShape().GetDimNum()),
                                                 "The shape dim of expandX must be 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t a = expandX->GetViewShape().GetDim(0);
    int64_t h = expandX->GetViewShape().GetDim(1);
    if (a <= 0 || h <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnMoeDistributeCombineSetup", "expandX",
                                              "[" + std::to_string(a) + "," + std::to_string(h) + "]",
                                              "dimensions must be greater than zero");
        return ACLNN_ERR_PARAM_INVALID;
    }

    tokenMsgSize =
        static_cast<uint64_t>(AlignUp(AlignUp(h, ALIGN_32) + AlignUp(h, ALIGN_8) / ALIGN_8 * sizeof(float), ALIGN_512));
    commCmdInfoOutSize = static_cast<uint64_t>((a + epWorldSize) * COMM_CMD_INFO_BASE);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnMoeDistributeCombineSetup(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream)
{
    OP_LOGD("aclnnMoeDistributeCombineSetup start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerMoeDistributeCombineSetup(workspace, workspaceSize, executor, stream);
    OP_LOGD("aclnnMoeDistributeCombineSetup success");
    return ret;
}
