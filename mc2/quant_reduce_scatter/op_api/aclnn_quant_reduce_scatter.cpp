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
 * \file aclnn_quant_reduce_scatter.cpp
 * \brief
 */
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
#include "common/utils/hccl_util.h"
#include "mc2_log_compat.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "aclnnInner_quant_reduce_scatter.h"

using namespace op;

namespace {
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static constexpr size_t HCCL_GROUP_NAME_LENGTH_MAX = 128U; // group长度小于128字符

// 根据API定义，列出K-G量化所能支持的所有dtype
const std::initializer_list<op::DataType> X_DTYPE_KG_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2};
const std::initializer_list<op::DataType> SCALES_DTYPE_KG_SUPPORT_LIST = {op::DataType::DT_FLOAT};

// 根据API定义，列出MX量化所能支持的所有dtype
const std::initializer_list<op::DataType> X_DTYPE_MX_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                     op::DataType::DT_FLOAT8_E5M2};
const std::initializer_list<op::DataType> SCALES_DTYPE_MX_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};

const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
                                                                       op::DataType::DT_FLOAT};

// 检查入参是否为nullptr
static bool CheckNotNull(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scales, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

// 检查x、scales、output的数据类型是否在算子的支持列表之内
static bool CheckKGAllDtypesValid(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    if (CheckType(x->GetDataType(), X_DTYPE_KG_SUPPORT_LIST) &&
        CheckType(scales->GetDataType(), SCALES_DTYPE_KG_SUPPORT_LIST) &&
        CheckType(output->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        return true;
    } else {
        return false;
    }
}

static bool CheckMXAllDtypesValid(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    if (CheckType(x->GetDataType(), X_DTYPE_MX_SUPPORT_LIST) &&
        CheckType(scales->GetDataType(), SCALES_DTYPE_MX_SUPPORT_LIST) &&
        CheckType(output->GetDataType(), OUTPUT_DTYPE_SUPPORT_LIST)) {
        return true;
    } else {
        return false;
    }
}

static bool CheckAllDtypesValid(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    bool isAllDtypesValid = false;
    isAllDtypesValid = CheckKGAllDtypesValid(x, scales, output) || CheckMXAllDtypesValid(x, scales, output);
    if (!isAllDtypesValid) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantReduceScatter", "x/scales/output",
            (std::string(op::ToString(x->GetDataType()).GetString()) + "/" +
             op::ToString(scales->GetDataType()).GetString() + "/" + op::ToString(output->GetDataType()).GetString())
                .c_str(),
            "The dtypes of x, scales and output must be valid");
    }
    return isAllDtypesValid;
}

static bool QuantReduceScatterCheckAllFormatValid(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    if (IsPrivateFormat(x->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnQuantReduceScatterGetWorkspaceSize", "x",
                                   op::ToString(x->GetStorageFormat()).GetString(), "non-Private Format");
        return false;
    }
    if (IsPrivateFormat(scales->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnQuantReduceScatterGetWorkspaceSize", "scales",
                                   op::ToString(scales->GetStorageFormat()).GetString(), "non-Private Format");
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT("aclnnQuantReduceScatterGetWorkspaceSize", "output",
                                   op::ToString(output->GetStorageFormat()).GetString(), "non-Private Format");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *scales, const aclTensor *output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, scales, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckAllDtypesValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查参数数据格式是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(QuantReduceScatterCheckAllFormatValid(x, scales, output), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus aclnnQuantReduceScatterGetWorkspaceSize(const aclTensor *context, const aclTensor *x,
                                                               const aclTensor *scales, int64_t hcclBufferSize,
                                                               int64_t worldSize, const char *reduceOp,
                                                               aclTensor *output, uint64_t *workspaceSize,
                                                               aclOpExecutor **executor)
{
    aclnnStatus retParam = CheckParams(x, scales, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    // 内部只处理ND格式：非ND(非私有)格式统一转为ND后重绑定形参，再传给inner
    if (x->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x origin format is: %s.", op::ToString(x->GetStorageFormat()).GetString());
        x = l0op::ReFormat(x, op::Format::FORMAT_ND);
        CHECK_RET(x != nullptr, ACLNN_ERR_PARAM_INVALID);
    }
    if (scales->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("scales origin format is: %s.", op::ToString(scales->GetStorageFormat()).GetString());
        scales = l0op::ReFormat(scales, op::Format::FORMAT_ND);
        CHECK_RET(scales != nullptr, ACLNN_ERR_PARAM_INVALID);
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is: %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = const_cast<aclTensor *>(l0op::ReFormat(output, op::Format::FORMAT_ND));
        CHECK_RET(output != nullptr, ACLNN_ERR_PARAM_INVALID);
    }

    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret =
        aclnnInnerQuantReduceScatterGetWorkspaceSize(context, x, scales, hcclBufferSize, const_cast<char *>(reduceOp),
                                                     yDtype, worldSize, output, workspaceSize, executor);
    OP_LOGD("QuantReduceScatter, aclnnQuantReduceScatterGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnQuantReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               const aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerQuantReduceScatter(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnQuantReduceScatter", "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
