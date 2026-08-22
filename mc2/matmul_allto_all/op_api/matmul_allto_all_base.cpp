/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_allto_all_base.cpp
 * \brief
 */
#include "matmul_allto_all_base.h"
#include "matmul_allto_all_util.h"
#include "securec.h"
#include "common/utils/op_mc2.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "common/op_api/mc2_aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/op_executor.h"
#include "common/utils/hccl_util.h"
#include "opdev/platform.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "mc2_comm_utils.h"

namespace {

using namespace op;
using namespace matmul_allto_all_check;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *output)
{
    if (x1 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x1");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x2");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "output");
        return false;
    }
    return true;
}

// 检查是否有空tensor
// 950 非量化场景支持x1的m轴为0，即token提示词为空
static bool CheckNotEmptyTensor(const aclTensor *x1, const aclTensor *x2, bool transposeX2)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        auto mVal = x1->GetViewShape().GetDim(0);
        OP_API_CHECK((mVal == ZERO), {
            OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x1.dimM",
                                      std::to_string(mVal).c_str(), "non-zero");
            return false;
        });
    }
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    OP_API_CHECK((kVal1 == ZERO), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x1.dimK", std::to_string(kVal1).c_str(),
                                  "non-zero");
        return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x2.dimK", std::to_string(kVal2).c_str(),
                                  "non-zero");
        return false;
    });
    OP_API_CHECK((nVal == ZERO), {
        OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x2.dimN", std::to_string(nVal).c_str(),
                                  "non-zero");
        return false;
    });
    return true;
}

// 非量化模式下X支持的FP16数据类型
static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};
// 非量化模式下Output支持的FP16数据类型
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_BF16};
// 非量化模式下910B对Bias数据类型有特殊规范
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                            op::DataType::DT_FLOAT};

// 非量化场景下，校验所有输入的参数类型是否正确（A5）
static bool CheckAllDtypesValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    // biasDtype可以为输入xDtype，也可以为fp32
    if (biasOptional != nullptr) {
        if (biasOptional->GetDataType() != op::DataType::DT_FLOAT && biasOptional->GetDataType() != x1->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "biasOptional",
                                      op::ToString(biasOptional->GetDataType()).GetString(), "x1 dtype or float32");
            return false;
        }
    }
    return true;
}

// 910B在输入为bfloat16时，bias不支持bfloat16，只能为float32
static bool CheckAllDtypesValid910B(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                    const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, output, return false);
    if (biasOptional != nullptr) {
        auto biasDtype = biasOptional->GetDataType();
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, BIAS_DTYPE_SUPPORT_LIST, return false);
        if (x1->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(x1, biasOptional, return false);
        } else if (biasDtype != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "biasOptional",
                                      op::ToString(biasDtype).GetString(), "DT_FLOAT when x is DT_BFLOAT16");
            return false;
        }
    }
    return true;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                        const aclTensor *output)
{
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x1",
                                   op::ToString(x1->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x2",
                                   op::ToString(x2->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "biasOptional",
                                       op::ToString(biasOptional->GetStorageFormat()).GetString(), "ND");
            return false;
        }
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "output",
                                   op::ToString(output->GetStorageFormat()).GetString(), "ND");
        return false;
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                          const aclTensor *output)
{
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                                        const aclIntArray *alltoAllAxesOptional, const char *group,
                                        const char *commMode, bool transposeX1, bool transposeX2,
                                        const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape
    CHECK_RET(CheckShapeMMAA("matmul_allto_all", x1, x2, biasOptional, transposeX2, output), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    // bias的数据类型限制在950和910B上有所区别，这里根据芯片版本做区分
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        CHECK_RET(CheckAllDtypesValid(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckAllDtypesValid910B(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    }
    // 5. 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 6. 兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 7. 检查alltoallAxes是否为空或者[-1,-2]
    CHECK_RET(CheckAlltoAllAxes("matmul_allto_all", alltoAllAxesOptional, true), ACLNN_ERR_PARAM_INVALID);
    // 8. 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1("matmul_allto_all", transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 9. 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength("matmul_allto_all", group), ACLNN_ERR_PARAM_INVALID);
    // 10. 检查commMode非空
    CHECK_RET(commMode != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}

static aclnnStatus DealWithEmptyTensor(uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("MatmulAlltoAll, dealing with empty tensor.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

// L0层两段式接口Inner，根据算子原型op_graph/matmul_allto_all_proto.h，由模板自动生成。非量化L2层接口和量化L2层接口共用一套L0层接口。
// worldSize为硬件方参数，在aclnn侧不感知。yDtype在aclnn侧不感知。这两个参数需要在Inner接口处声明，在aclnn侧通过默认值传参。
extern "C" aclnnStatus aclnnInnerMatmulAlltoAllGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, char *group, int64_t worldSize, const aclIntArray *all2allAxesOptional,
    int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype,
    bool transposeX1, bool transposeX2, int64_t groupSize, char *commModeOptional, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

// 非量化L2接口调用L0时需要设置较多默认值，通过InnerMatmulAlltoAllGetWorkspaceSize完成默认值传参和调用L0层接口
extern "C" aclnnStatus InnerMatmulAlltoAllGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                           const aclTensor *biasOptional,
                                                           const aclIntArray *alltoAllAxesOptional, const char *group,
                                                           const char *commMode, bool transposeX1, bool transposeX2,
                                                           const aclTensor *output, uint64_t *workspaceSize,
                                                           aclOpExecutor **executor)
{
    // ACL和GE的datatype枚举值对undefined定义不同，inner接口进入到算子内部，需要使用GE枚举值，因此此处使用的枚举值为28
    const int64_t GE_UNDEFINED = 28;
    // 根据算子原型定义默认值
    aclTensor *x1ScaleOptional = nullptr;
    aclTensor *x2ScaleOptional = nullptr;
    aclTensor *commScaleOptional = nullptr;
    aclTensor *x1OffsetOptional = nullptr;
    aclTensor *x2OffsetOptional = nullptr;
    const aclTensor *out = output;
    char *str_group = const_cast<char *>(group);
    char *str_commMode = const_cast<char *>(commMode);
    int64_t worldSize = -1;
    int64_t yDtype = GE_UNDEFINED;
    int64_t x1QuantMode = 0;
    int64_t x2QuantMode = 0;
    int64_t commQuantMode = 0;
    int64_t commQuantDtype = GE_UNDEFINED;
    int64_t groupSize = 0;
    aclnnStatus ret = aclnnInnerMatmulAlltoAllGetWorkspaceSize(
        x1, x2, biasOptional, x1ScaleOptional, x2ScaleOptional, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, yDtype, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype,
        transposeX1, transposeX2, groupSize, str_commMode, out, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, aclnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

// 两段式接口
aclnnStatus aclnnMatmulAlltoAllBaseGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                    const aclTensor *biasOptional,
                                                    const aclIntArray *alltoAllAxesOptional, const char *group,
                                                    const char *commMode, bool transposeX1, bool transposeX2,
                                                    const aclTensor *output, uint64_t *workspaceSize,
                                                    aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前只有支持转置的x2涉及该处理
    aclnnStatus checkX2Ret = CheckX2Valid("matmul_allto_all", x2);
    CHECK_RET(checkX2Ret == ACLNN_SUCCESS, checkX2Ret);                  // 先检查x2是否合法，避免非法操作
    auto transX2 = x2;                                                   // 复制一个x2
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) { // 只有当非连续时，才会涉及到转连续等情况
        bool notContiguous =
            IsTransposeLastTwoDims(x2); // notContiguous标识x2是否是非连续的，通常在pytorch经过.t()会导致x2非连续
        if (notContiguous && transposeX2) { // 当非连续和转置同时生效时，判断为错误用法，直接报错
            OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x2", "non-contiguous and transposed",
                                      "contiguous or not transposed");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous) {
            transposeX2 = !transposeX2;
            // 把非连续x2转成连续
            transX2 = TransX2Tensor(x2);
            CHECK_RET(transX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            OP_LOGD("X2 is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. "
                    "After processing, transX2 dim0 is %ld, and dim1 is %ld.",
                    x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1), transX2->GetViewShape().GetDim(0),
                    transX2->GetViewShape().GetDim(1));
        }
    } else {
        // 对于 A2 和 A3，非连续则报错
        OP_API_CHECK(!MC2Aclnn::IsTensorContiguous(x2), {
            OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAlltoAllBaseGetWorkspaceSize", "x2", "non-contiguous", "contiguous");
            return ACLNN_ERR_PARAM_INVALID;
        });
    }
    aclnnStatus retParam = CheckAndHandleParams(x1, transX2, biasOptional, alltoAllAxesOptional, group, commMode,
                                                transposeX1, transposeX2, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // 处理空tensor，目前非量化matmulalltoall只支持x1第一维度bs为0，空tensor作异常处理
    if (x1->GetViewShape().GetDim(0) == 0 && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return DealWithEmptyTensor(workspaceSize, executor);
    }

    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode("matmul_allto_all", group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);

    aclnnStatus ret =
        InnerMatmulAlltoAllGetWorkspaceSize(x1, transX2, biasOptional, alltoAllAxesOptional, group, commMode,
                                            transposeX1, transposeX2, output, workspaceSize, executor);
    OP_LOGD("MatmulAlltoAll, end ret %d", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT(
            "aclnnMatmulAlltoAllBaseGetWorkspaceSize",
            "This is an error in launch aicore, aclnnMatmulAlltoAllBaseGetWorkspaceSize interface call failed.");
    }

    if (ret == ACLNN_SUCCESS && *executor != nullptr) {
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, arg);
    }

    return ret;
}

aclnnStatus aclnnMatmulAlltoAllBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnMatmulAlltoAllBase", "Param executor is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);

            if (commMode == Mc2Comm::COMM_MODE_CCU) {
                OP_LOGD("MatmulAlltoAll: use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            } else {
                OP_LOGD("MatmulAlltoAll: use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            }
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAlltoAll(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnMatmulAlltoAllBase",
                                "This is an error in launch aicore, aclnnInnerMatmulAlltoAll interface call failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
