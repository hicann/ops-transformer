/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_dispatch_setup.h"
#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_moe_distribute_dispatch_setup.h"
#include "mc2_log_compat.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t TWO_DIM = 2;
static constexpr int64_t ALIGN_32 = 32;
static constexpr int64_t ALIGN_256 = 256;
static constexpr int64_t ALIGN_512 = 512;
static constexpr int64_t QUANT_ALIGN_OFFSET = 4;
static constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;
static constexpr int64_t COMM_CMD_INFO_BASE = 16;

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

inline int64_t Align(int64_t x, int64_t base)
{
    if (base <= 0) {
        return x;
    }
    return ((x + base - 1) / base) * base;
}

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor *x, const aclTensor *expertIds, const char *groupEp, aclTensor *yOut,
                         aclTensor *expandIdxOut, aclTensor *commCmdInfoOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(expandIdxOut, return false);
    OP_CHECK_NULL(commCmdInfoOut, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMoeDistributeDispatchSetup", "groupEp");
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *expertIds, const char *groupEp, aclTensor *yOut,
                               aclTensor *expandIdxOut, aclTensor *commCmdInfoOut)
{
    CHECK_RET(CheckNotNull(x, expertIds, groupEp, yOut, expandIdxOut, commCmdInfoOut), ACLNN_ERR_PARAM_NULLPTR);
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMoeDistributeDispatchSetup", "groupEp",
                                              "length exceeds " + std::to_string(HCCL_GROUP_NAME_MAX),
                                              "groupEp name too long");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t shareExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t commType, const char *commAlg, aclTensor *yOut,
    aclTensor *expandIdxOut, aclTensor *commCmdInfoOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnMoeDistributeDispatchSetupGetWorkspaceSize start");
    auto ret_param = CheckParams(x, expertIds, groupEp, yOut, expandIdxOut, commCmdInfoOut);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, const_cast<char *>(groupEp), epWorldSize, epRankId,
        moeExpertNum, expertShardType, sharedExpertNum, shareExpertRankNum, quantMode, globalBs, commType,
        const_cast<char *>(commAlg), yOut, expandIdxOut, commCmdInfoOut, workspaceSize, executor);
    return ret;
}

static aclnnStatus CheckCalcOutputShape(const aclTensor *x, const aclTensor *expertIds)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expertIds, return ACLNN_ERR_PARAM_NULLPTR);
    if (x->GetViewShape().GetDimNum() != TWO_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMoeDistributeDispatchSetup", "x",
                                                 "dim=" + std::to_string(x->GetViewShape().GetDimNum()),
                                                 "The shape dim of x must be 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (expertIds->GetViewShape().GetDimNum() != TWO_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnMoeDistributeDispatchSetup", "expertIds",
                                                 "dim=" + std::to_string(expertIds->GetViewShape().GetDimNum()),
                                                 "The shape dim of expertIds must be 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeDispatchSetupTeardownCalcOutputSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t commType, const char *commAlg,
    uint64_t &tokenMsgSize, uint64_t &expandIdxOutSize, uint64_t &assistInfoForCombineOutSize,
    uint64_t &commCmdInfoOutSize)
{
    // 对外接口，参数暂使用 void 修改
    (void)scalesOptional;
    (void)xActiveMaskOptional;
    (void)groupEp;
    (void)expertShardType;
    (void)expertTokenNumsType;
    (void)commType;
    (void)commAlg;
    aclnnStatus retCheckShape = CheckCalcOutputShape(x, expertIds);
    CHECK_RET(retCheckShape == ACLNN_SUCCESS, retCheckShape);
    int64_t bs = x->GetViewShape().GetDim(0);
    int64_t h = x->GetViewShape().GetDim(1);
    int64_t k = expertIds->GetViewShape().GetDim(1);
    if (quantMode == 0) {
        tokenMsgSize = static_cast<uint64_t>(Align(h, ALIGN_256));
    } else {
        tokenMsgSize = static_cast<uint64_t>(Align(Align(h, ALIGN_32) + QUANT_ALIGN_OFFSET, ALIGN_512));
    }
    expandIdxOutSize = static_cast<uint64_t>(bs * k);
    int64_t a;
    int64_t localExpertNum;
    int64_t globalBsReal = (globalBs == 0) ? (bs * epWorldSize) : globalBs;
    if (sharedExpertRankNum >= epWorldSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMoeDistributeDispatchSetup", "sharedExpertRankNum",
                                              std::to_string(sharedExpertRankNum).c_str(),
                                              "should be less than epWorldSize=" + std::to_string(epWorldSize));
        return ACLNN_ERR_PARAM_INVALID;
    } else if (epRankId < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMoeDistributeDispatchSetup", "epRankId",
                                              std::to_string(epRankId).c_str(), "should not be less than 0");
        return ACLNN_ERR_PARAM_INVALID;
    } else {
        if (epRankId < sharedExpertRankNum) {
            localExpertNum = 1;
            a = bs * epWorldSize * sharedExpertNum / sharedExpertRankNum;
        } else {
            localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
            a = globalBsReal * std::min(localExpertNum, k);
        }
    }
    assistInfoForCombineOutSize = static_cast<uint64_t>(a * ASSIST_INFO_NUM_PER_A);
    commCmdInfoOutSize =
        static_cast<uint64_t>((bs * (k + sharedExpertNum) + epWorldSize * localExpertNum) * COMM_CMD_INFO_BASE);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeDispatchSetup(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream)
{
    OP_LOGD("aclnnMoeDistributeDispatchSetup start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerMoeDistributeDispatchSetup(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
