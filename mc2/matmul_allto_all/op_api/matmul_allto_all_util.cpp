/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_allto_all_util.h"
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
#include "mc2_log_compat.h"
#include "opdev/platform.h"
#include "common/utils/hccl_util.h"
#include "mc2_comm_utils.h"

// 量化与非量化共用的方法和常量、枚举值
namespace matmul_allto_all_check {
using namespace op;

// 校验AlltoAll和Permute数据交换的方向参数, 可以为空和{-1,-2}, 不允许为其他值
bool CheckAlltoAllAxes(const char *opName, const aclIntArray *alltoAllAxesOptional, bool isMatmulAlltoAll)
{
    // alltoAllAxesOptional为空时会兼容性处理，不报错
    if (alltoAllAxesOptional == nullptr) {
        OP_LOGW("The alltoAllAxesOptional is nullptr.");
        return true;
    }
    uint64_t alltoallAxesSize = 0U; // alltoallAxes的大小
    aclGetIntArraySize(alltoAllAxesOptional, &alltoallAxesSize);
    if (alltoallAxesSize != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "alltoAllAxesOptional.size()", std::to_string(alltoallAxesSize).c_str(), "2");
        return false;
    }
    int64_t data1 = (*alltoAllAxesOptional)[0];
    int64_t data2 = (*alltoAllAxesOptional)[1];
    if (isMatmulAlltoAll) {
        OP_API_CHECK((data1 != NEG_ONE), {
            OP_LOGE_FOR_INVALID_VALUE(opName, "alltoAllAxesOptional[0]", std::to_string(data1).c_str(), "-1");
            return false;
        });
        OP_API_CHECK((data2 != NEG_TWO), {
            OP_LOGE_FOR_INVALID_VALUE(opName, "alltoAllAxesOptional[1]", std::to_string(data2).c_str(), "-2");
            return false;
        });
    } else {
        OP_API_CHECK((data1 != NEG_TWO), {
            OP_LOGE_FOR_INVALID_VALUE(opName, "alltoAllAxesOptional[0]", std::to_string(data1).c_str(), "-2");
            return false;
        });
        OP_API_CHECK((data2 != NEG_ONE), {
            OP_LOGE_FOR_INVALID_VALUE(opName, "alltoAllAxesOptional[1]", std::to_string(data2).c_str(), "-1");
            return false;
        });
    }
    return true;
}

// 校验输入的转置配置，x1不允许转置
bool CheckTransposeX1(const char *opName, bool transposeX1)
{
    OP_API_CHECK(transposeX1, {
        OP_LOGE_FOR_INVALID_VALUE(opName, "transposeX1", "true", "false");
        return false;
    });
    return true;
}

// 校验通信域名的字符串长度是否符合要求
bool CheckGroupLength(const char *opName, const char *group)
{
    if (group == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "group");
        return false;
    }
    auto len = strnlen(group, MAX_GROUP_LEN);
    if ((len >= MAX_GROUP_LEN) || (len == ZERO)) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "group.length()", std::to_string(len).c_str(), "in range (0, 128)");
        return false;
    }
    return true;
}

// 校验输入tensor的维度
bool CheckInputDimensions(const char *opName, const aclTensor *x1, const aclTensor *x2, const aclTensor *output,
                          const aclTensor *alltoAllOutOptional = nullptr)
{
    if (x1->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x1",
                                                 (std::to_string(x1->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x1 must be 2D.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return false;
    }
    if (output->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "output",
                                                 (std::to_string(output->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of output must be 2D.");
        return false;
    }
    if (alltoAllOutOptional != nullptr) {
        if (alltoAllOutOptional->GetViewShape().GetDimNum() != TWO_DIMS) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName, "alltoAllOutOptional",
                (std::to_string(alltoAllOutOptional->GetViewShape().GetDimNum()) + "D").c_str(),
                "The shape of alltoAllOutOptional must be 2D.");
            return false;
        }
    }
    return true;
}

// 校验bias的维度和shape
bool CheckBiasShape(const char *opName, const aclTensor *biasOptional, int64_t nVal)
{
    if (biasOptional != nullptr) {
        if (biasOptional->GetViewShape().GetDimNum() != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName, "biasOptional", (std::to_string(biasOptional->GetViewShape().GetDimNum()) + "D").c_str(),
                "The shape of biasOptional must be 1D.");
            return false;
        }
        auto biasDim = biasOptional->GetViewShape().GetDim(0);
        if (biasDim != nVal) {
            OP_LOGE_FOR_INVALID_VALUE(opName, "biasDim", std::to_string(biasDim).c_str(),
                                      ("should equal x2.n-axis " + std::to_string(nVal)).c_str());
            return false;
        }
    }
    return true;
}

// 校验MatmulAlltoAll和QuantMatmulAlltoAll输入属性shape
bool CheckShapeMMAA(const char *opName, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                    bool transposeX2, const aclTensor *output)
{
    if (!CheckInputDimensions(opName, x1, x2, output)) {
        return false;
    }
    auto kdimX1 = x1->GetViewShape().GetDim(1);
    auto kdimX2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);

    OP_LOGI("The input transposeX2 is: %d, x1 dim0 is: %ld, x1 dim1 is: %ld, x2 dim0 is: %ld, x2 dim1 is: %ld",
            transposeX2, x1->GetViewShape().GetDim(0), x1->GetViewShape().GetDim(1), x2->GetViewShape().GetDim(0),
            x2->GetViewShape().GetDim(1));

    if (kdimX1 != kdimX2) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "kdimX1", std::to_string(kdimX1).c_str(),
                                  ("should equal x2.dimK " + std::to_string(kdimX2)).c_str());
        return false;
    }
    if (kdimX1 < KVALUE_MIN || kdimX1 > KVALUE_MAX) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "kdimX1", std::to_string(kdimX1).c_str(), "in range [1, 65535]");
        return false;
    }
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    return CheckBiasShape(opName, biasOptional, nVal);
}

// 校验AlltoAllMatmul和AlltoAllQuantMatmul输入属性shape
bool CheckShapeAAMM(const char *opName, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
                    bool transposeX2, const aclTensor *output, const aclTensor *alltoAllOutOptional)
{
    if (!CheckInputDimensions(opName, x1, x2, output, alltoAllOutOptional)) {
        return false;
    }
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    return CheckBiasShape(opName, biasOptional, nVal);
}

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        OP_LOGD("The view_shape dim is: %ld", tensor->GetViewShape().GetDimNum());
        return false;
    }

    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // BMM 场景下，Batch维度的stride需要等于 N, D 的乘积
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
            OP_LOGI("The input tensor is contiguous.");
            return false;
        }
        OP_LOGI("The input tensor is not contiguous.");
        return true;
    }
    return false;
}

// 检查x2是否合法
aclnnStatus CheckX2Valid(const char *opName, const aclTensor *x2)
{
    if (x2 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "x2");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (x2->IsEmpty()) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "x2", "empty tensor", "non-empty tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x2",
                                                 (std::to_string(x2->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of x2 must be 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 检查commMode参数是否合法，并获取commMode对应枚举值
aclnnStatus CheckAndHandleCommMode(const char *opName, const char *group, const char *commModeStr,
                                   uint8_t &commModeEnum)
{
    const size_t maxLength = 7UL;
    // 获取通信引擎参数
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_CCU;
        } else {
            OP_LOGE_FOR_INVALID_VALUE(opName, "commMode", commModeStr, "\"ccu\" or \"ai_cpu\"");
            return ACLNN_ERR_PARAM_INVALID;
        }
        OP_LOGD("The commMode is [%s].", commModeStr);
    } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        if (strncmp(commModeStr, "ai_cpu", maxLength) != 0) {
            OP_LOGE_FOR_INVALID_VALUE(opName, "commMode", commModeStr, "\"ai_cpu\"");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        if (strncmp(commModeStr, "aiv", maxLength) != 0) {
            OP_LOGE_FOR_INVALID_VALUE(opName, "commMode", commModeStr, "\"aiv\"");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// 检查是否有alltoallout输出
bool IsAll2AllOut(const char *opName, const aclTensor *alltoAllOut)
{
    if (alltoAllOut == nullptr) {
        return false;
    }
    if (alltoAllOut->IsEmpty() && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "alltoAllOutOptional", "empty tensor", "non-empty tensor");
        return false;
    }
    return true;
}

// 处理支持转置的tensor物理排布不连续问题
aclTensor *TransX2Tensor(const aclTensor *x2)
{
    uint64_t storageShapeDimNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = x2->GetStorageShape().GetDim(i);
    }
    uint64_t viewShapeDimNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = x2->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = x2->GetViewShape().GetDim(1);
    viewDim[1] = x2->GetViewShape().GetDim(0);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = x2->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;
    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, x2->GetTensor()->GetAddr());
}
} // namespace matmul_allto_all_check
