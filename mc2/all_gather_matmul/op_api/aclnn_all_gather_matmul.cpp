/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_all_gather_matmul.h"
#include "all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.h"
#include "securec.h"
#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mc2_log_compat.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/utils/hccl_util.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "aclnnInner_all_gather_matmul.h"
#include "op_host/util/op_const_def.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t NUM_ACL_STOP_ON_FAILURE = 1;
static constexpr size_t TWO_DIMS = 2;
static constexpr int64_t KVALUE_MIN = 256;
static constexpr int64_t KVALUE_MAX = 65535;
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;

extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

// check nullptr
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}
static inline bool IsAscend910A5(void)
{
    return op::GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510;
}
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static bool CheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(bias, x1, return false);
    }
    return true;
}
static bool CheckAttr(int64_t streamMode)
{
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAllGatherMatmulGetWorkspaceSize", "streamMode",
                                  std::to_string(streamMode).c_str(), "1");
        return false;
    }
    return true;
}

static bool IsGatherOut(const aclTensor *gatherOut)
{
    OP_CHECK_NULL(gatherOut, return false);
    if (gatherOut->IsEmpty()) {
        OP_LOGD("AllGatherMatmul, get gather out is false.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *output, const aclTensor *gatherOut,
                       bool isTransA)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAllGatherMatmulGetWorkspaceSize", "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAllGatherMatmulGetWorkspaceSize", "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return false;
    }
    OP_API_CHECK(isTransA, {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAllGatherMatmulGetWorkspaceSize", "x1", "transposed",
                                              "The value of x1 must not be transposed.");
        return false;
    });
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = x2->GetViewShape().GetDim(0);
    OP_API_CHECK((kVal1 != kVal2), {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnAllGatherMatmulGetWorkspaceSize", "x1/x2",
                                  (std::string("k=") + std::to_string(kVal1) + "/" + std::to_string(kVal2)).c_str(),
                                  "equal k values");
        return false;
    });

    if (IsGatherOut(gatherOut)) {
        auto kVal3 = gatherOut->GetViewShape().GetDim(1);
        OP_API_CHECK((kVal1 != kVal3), {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnAllGatherMatmulGetWorkspaceSize", "x1/gatherOut",
                                      (std::string("k=") + std::to_string(kVal1) + "/" + std::to_string(kVal3)).c_str(),
                                      "equal k values");
            return false;
        });
    }

    OP_API_CHECK((kVal1 < KVALUE_MIN || kVal1 >= KVALUE_MAX), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAllGatherMatmulGetWorkspaceSize", "k-axis", std::to_string(kVal1).c_str(),
                                  "[256, 65535)");
        return false;
    });

    auto nVal1 = x2->GetViewShape().GetDim(1);
    auto nVal2 = output->GetViewShape().GetDim(1);
    OP_API_CHECK((nVal1 != nVal2), {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnAllGatherMatmulGetWorkspaceSize", "x2/output",
                                  (std::string("n=") + std::to_string(nVal1) + "/" + std::to_string(nVal2)).c_str(),
                                  "equal n values");
        return false;
    });

    return true;
}

static bool CheckNotEmptyTensor(const aclTensor *x1)
{
    auto kValue = x1->GetViewShape().GetDim(1);
    OP_API_CHECK((kValue == 0), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAllGatherMatmulGetWorkspaceSize", "x1 k-dim", "0", "non-zero");
        return false;
    });
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, int64_t streamMode,
                               const aclTensor *output)
{
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);

    if (bias != nullptr) {
        OP_LOGW("AllGatherMatmul, The current version does not support the scenario where bias is not 0. "
                "Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
    }

    CHECK_RET(CheckDtypeValid(x1, x2, bias, output), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAttr(streamMode), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckNotEmptyTensor(x1), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus DealWithX1Empty(uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("AllGatherMatmul, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                                 const char *group, int64_t gatherIndex, int64_t commTurn,
                                                 int64_t streamMode, const aclTensor *output,
                                                 const aclTensor *gatherOut, uint64_t *workspaceSize,
                                                 aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    auto retParam = CheckParams(x1, x2, bias, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 处理空tensor 如果x1不为空 x2为空 需要进行gatherOut
    if (x1->IsEmpty()) {
        return DealWithX1Empty(workspaceSize, executor);
    }

    OP_LOGD("X1 is %s, X2 is %s.", x1->ToString().GetString(), x2->ToString().GetString());
    OP_LOGD("Output is %s, gatherOut is %s.", output->ToString().GetString(), gatherOut->ToString().GetString());

    uint32_t rankSize = 0;
    bool transposeX1 = IsTransposeLastTwoDims(x1);
    bool transposeX2 = IsTransposeLastTwoDims(x2);
    CHECK_RET(CheckShape(x1, x2, output, gatherOut, transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 【A2】校验非连续入参合法性
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        OP_API_CHECK(!transposeX2 && !MC2Aclnn::IsTensorContiguous(x2), {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAllGatherMatmulGetWorkspaceSize", "x2", "non-contiguous",
                                                  "The value of x2 must be contiguous when not transposed.");
            return ACLNN_ERR_PARAM_INVALID;
        });
    }
    bool isGatherOut = IsGatherOut(gatherOut);
    if (IsAscend910A5()) {
        const char *commMode = "ai_cpu";
        return aclnnAllGatherMatmulV2GetWorkspaceSize(
            x1, x2, bias, nullptr, nullptr, nullptr, 0, group, gatherIndex, commTurn, streamMode, 0, commMode,
            const_cast<aclTensor *>(output), const_cast<aclTensor *>(gatherOut), nullptr, workspaceSize, executor);
    }
    aclnnStatus ret = aclnnInnerAllGatherMatmulGetWorkspaceSize(
        x1, x2, bias, const_cast<char *>(group), transposeX1, transposeX2, gatherIndex, commTurn, rankSize, isGatherOut,
        output, gatherOut, workspaceSize, executor);
    OP_LOGD("AllGatherMatmul, aclnnInnerGetWorkspaceSize ret = %d.", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnAllGatherMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (IsAscend910A5()) {
        return aclnnAllGatherMatmulV2(workspace, workspaceSize, executor, stream);
    }
    if (workspace == nullptr || workspaceSize == 0UL) {
        OP_LOGD("Skip the api for empty tensor, workspace size %lu.", workspaceSize);
        return ACLNN_SUCCESS;
    }
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    auto ret = aclnnInnerAllGatherMatmul(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAllGatherMatmul", "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }

    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
