/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdlib>

#include "securec.h"
#include "common/utils/op_mc2.h"
#include "acl/acl.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "mc2_log_compat.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "common/utils/hccl_util.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "aclnn_allto_allv_quant_grouped_mat_mul.h"
#include "op_host/util/op_const_def.h"

namespace {
using namespace op;

#include "mc2_comm_utils.h"

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
};

// 需要使用的常量定义
static constexpr size_t TWO_DIMS = 2U;
static constexpr size_t THREE_DIMS = 3U;

static bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                            const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                            bool permuteOutFlag, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // // 检查必选入参出参为非空
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "sendCountsTensorOptional/recvCountsTensorOptional",
            "non-null", "The value of sendCountsTensorOptional/recvCountsTensorOptional must be nullptr.");
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize",
                                              "permuteOutFlag/permuteOutOptional", "mismatched",
                                              "The values of permuteOutFlag and permuteOutOptional must be the same.");
        return false;
    }
    return true;
}

// 检查必要输入是否为空/quantMode为1或6，必须非空/1或6
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                         const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, int64_t gmmXQuantMode,
                         int64_t gmmWeightQuantMode)
{
    if (gmmX == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmX");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmWeight");
        return false;
    }
    if (gmmY == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmY");
        return false;
    }
    if (gmmXScale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmXScale");
        return false;
    }
    if (gmmWeightScale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmWeightScale");
        return false;
    }
    if ((gmmXQuantMode != static_cast<int64_t>(QuantModeType::PERTENSOR_QUANT)) &&
        (gmmXQuantMode != static_cast<int64_t>(QuantModeType::MX_QUANT))) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmXQuantMode",
                                  std::to_string(gmmXQuantMode).c_str(), "1 or 6");
        return false;
    }
    return true;
}

static bool CheckGmmWeightValid(const aclTensor *gmmWeight)
{
    if (gmmWeight == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmWeight");
        return false;
    }
    if (gmmWeight->GetViewShape().GetDimNum() != THREE_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmWeight",
                                                 (std::to_string(gmmWeight->GetViewShape().GetDimNum()) + "D").c_str(),
                                                 "The shape of gmmWeight must be 3D.");
        return false;
    }
    return true;
}

static bool CheckMmWeightValid(const aclTensor *mmWeightOptional)
{
    if (mmWeightOptional == nullptr) {
        return false;
    }
    if (mmWeightOptional->GetViewShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "mmWeightOptional",
            (std::to_string(mmWeightOptional->GetViewShape().GetDimNum()) + "D").c_str(),
            "The shape of mmWeightOptional must be 2D.");
        return false;
    }
    return true;
}

/**
 * 转置gmmWeight tensor的viewShape中间两维（H和N维度）
 * 用于处理tensor物理排布不连续问题，通过stride交换实现转置
 * @param gmmWeight 输入tensor，要求viewShape为[e, H, N, K]（4维）
 * @return 返回新创建的tensor指针，viewShape变为[e, N, H, K]
 */
static const aclTensor *TransGmmWeightTensor(const aclTensor *gmmWeight)
{
    uint64_t storageShapeDimNum = gmmWeight->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeight->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeight->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeight->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[1] = gmmWeight->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeight->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(gmmWeight, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize",
                                "aclGetDataType failed for gmmWeight, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = gmmWeight->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeight->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeight->GetTensor()->GetAddr());
}

/**
 * 转置gmmWeightScale tensor的viewShape中间两维（H和N维度）
 * pertensor量化场景（dimNum < 4）直接返回原tensor，无需转置
 * @param gmmWeightScale 输入tensor，MX量化时viewShape为[e, H, N, 2]
 * @return 返回tensor指针，MX量化时viewShape变为[e, N, H, 2]
 */
static const aclTensor *TransGmmWeightScaleTensor(const aclTensor *gmmWeightScale)
{
    // pertensor量化场景：gmmWeightScale shape [1]，dimNum == 1，不需要转置处理
    if (gmmWeightScale->GetViewShape().GetDimNum() < 4) {
        return gmmWeightScale;
    }

    uint64_t storageShapeDimNum = gmmWeightScale->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeightScale->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeightScale->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeightScale->GetViewShape().GetDim(i);
    }
    viewDim[1] = gmmWeightScale->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeightScale->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(gmmWeightScale, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize",
                                "aclGetDataType failed for gmmWeightScale, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = gmmWeightScale->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeightScale->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeightScale->GetTensor()->GetAddr());
}

/**
 * 转置mmWeightOptional tensor的viewShape两维
 * 用于处理tensor物理排布不连续问题，通过stride交换实现转置
 * @param mmWeightOptional 输入tensor，要求viewShape为[K, N]（2维）
 * @return 返回新创建的tensor指针，viewShape变为[N, K]
 */
static const aclTensor *TransMmWeightOptionalTensor(const aclTensor *mmWeightOptional)
{
    uint64_t storageShapeDimNum = mmWeightOptional->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightOptional->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightOptional->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightOptional->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = mmWeightOptional->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightOptional->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(mmWeightOptional, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize",
                                "aclGetDataType failed for mmWeightOptional, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = mmWeightOptional->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightOptional->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightOptional->GetTensor()->GetAddr());
}

/**
 * 转置mmWeightScale tensor的viewShape前两维
 * pertensor量化场景（dimNum < 3）直接返回原tensor，无需转置
 * @param mmWeightScale 输入tensor，MX量化时viewShape为[CeilDiv(H2, 64), N2, 2]
 * @return 返回tensor指针，MX量化时viewShape变为[N2, CeilDiv(H2, 64), 2]
 */
static const aclTensor *TransMmWeightScaleTensor(const aclTensor *mmWeightScale)
{
    // pertensor量化场景：mmWeightScale shape [1]，dimNum == 1，不需要转置处理
    if (mmWeightScale->GetViewShape().GetDimNum() < 3) {
        return mmWeightScale;
    }

    uint64_t storageShapeDimNum = mmWeightScale->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightScale->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightScale->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightScale->GetViewShape().GetDim(i);
    }
    viewDim[0] = mmWeightScale->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightScale->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclnnStatus dtypeRet = aclGetDataType(mmWeightScale, &dataType);
    if (dtypeRet != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize",
                                "aclGetDataType failed for mmWeightScale, ret=%d", dtypeRet);
        return nullptr;
    }
    auto transStride = mmWeightScale->GetViewStrides();
    std::vector<int64_t> stride(transStride.begin(), transStride.end());
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightScale->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightScale->GetTensor()->GetAddr());
}

bool IsTransposeTwoDims(const aclTensor *tensor, int64_t firstDimIndex, int64_t secondDimIndex)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() < 2) {
        return false;
    }
    int64_t dimNum = tensor->GetViewShape().GetDimNum();
    int64_t dim1 = firstDimIndex < 0 ? dimNum + firstDimIndex : firstDimIndex;
    int64_t dim2 = secondDimIndex < 0 ? dimNum + secondDimIndex : secondDimIndex;
    if (dim1 < 0 || dim1 >= dimNum || dim2 < 0 || dim2 >= dimNum || dim1 == dim2) {
        return false;
    }
    int64_t shape1 = tensor->GetViewShape().GetDim(dim1);
    int64_t shape2 = tensor->GetViewShape().GetDim(dim2);
    int64_t stride1 = tensor->GetViewStrides()[dim1];
    int64_t stride2 = tensor->GetViewStrides()[dim2];
    if (shape1 == 1 && shape2 == 1) {
        return false;
    }
    if (stride1 == 1) {
        return (stride2 != shape1);
    }
    if (stride2 == 1) {
        return true;
    }
    return (stride2 != shape1 * stride1);
}

static aclnnStatus CheckWeightScaleTransposeConsistency(const aclTensor *gmmWeight, const aclTensor *gmmWeightScale,
                                                        const aclTensor *mmWeightOptional,
                                                        const aclTensor *mmWeightScaleOptional)
{
    // pertensor量化场景：weightScale dimNum < 4，不需要转置，跳过校验
    if (gmmWeight != nullptr && gmmWeightScale != nullptr && gmmWeightScale->GetViewShape().GetDimNum() >= 4) {
        bool gmmWeightNotContiguous = IsTransposeTwoDims(gmmWeight, -1, -2);
        bool gmmWeightScaleNotContiguous = IsTransposeTwoDims(gmmWeightScale, -2, -3);
        if (gmmWeightNotContiguous != gmmWeightScaleNotContiguous) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmWeight/gmmWeightScale",
                (std::string(gmmWeightNotContiguous ? "transposed" : "not transposed") + "/" +
                 std::string(gmmWeightScaleNotContiguous ? "transposed" : "not transposed"))
                    .c_str(),
                "gmmWeight and gmmWeightScale transpose state must be consistent");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // pertensor量化场景：mmWeightScale dimNum < 3，不需要转置，跳过校验
    if (mmWeightOptional != nullptr && mmWeightScaleOptional != nullptr &&
        mmWeightScaleOptional->GetViewShape().GetDimNum() >= 3) {
        bool mmWeightNotContiguous = IsTransposeTwoDims(mmWeightOptional, -1, -2);
        bool mmWeightScaleNotContiguous = IsTransposeTwoDims(mmWeightScaleOptional, -2, -3);
        if (mmWeightNotContiguous != mmWeightScaleNotContiguous) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "mmWeight/mmWeightScale",
                (std::string(mmWeightNotContiguous ? "transposed" : "not transposed") + "/" +
                 std::string(mmWeightScaleNotContiguous ? "transposed" : "not transposed"))
                    .c_str(),
                "mmWeight and mmWeightScale transpose state must be consistent");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                               const aclTensor *gmmWeightScale, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                               const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // 检查空状态
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                              mmXScaleOptional, mmWeightScaleOptional, permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查参数是否为空
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, gmmXQuantMode, gmmWeightQuantMode),
              ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("AlltoAllvQuantGroupedMatmul checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype,
    const char *commMode, aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                             aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, aclTensor *gmmY, aclTensor *mmYOptional,
    aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const char *commMode = "ai_cpu";
    char *str_commMode = const_cast<char *>(commMode);
    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, gmmXScale, gmmWeightScale, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
        mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional, group, epWorldSize, sendCounts, recvCounts,
        gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight, permuteOutFlag, mmXQuantMode,
        mmWeightQuantMode, groupSize, yDtype, mmDtype, str_commMode, gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);
    if (*executor != nullptr) {
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(Mc2Comm::COMM_MODE_AICPU));
        NnopbaseSetUserHandle(*executor, args);
    }
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnInnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, int64_t groupSize, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前支持转置的gmmWeight涉及该处理
    CHECK_RET(CheckGmmWeightValid(gmmWeight), ACLNN_ERR_PARAM_NULLPTR); // 先检查gmmWeight是否合法，避免非法操作
    bool notContiguous = IsTransposeTwoDims(gmmWeight, -1, -2);
    auto transposeGmmWeight = gmmWeight;   // 复制一个gmmWeight
    if (notContiguous && transGmmWeight) { // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize", "gmmWeight",
                                              "not contiguous and transpose set",
                                              "contiguous tensor or transpose disabled");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous &&
        GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) { // 只有当非连续时，才会涉及到转连续等情况
        transGmmWeight = !transGmmWeight;
        // 把非连续gmmWeight转成连续
        transposeGmmWeight = TransGmmWeightTensor(gmmWeight);
        CHECK_RET(transposeGmmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("gmmWeight is a non-contiguous tensor. The original dim1 is %ld, and dim2 is %ld. After processing, "
                "transposeGmmWeight dim1 is %ld, and dim2 is %ld.",
                gmmWeight->GetViewShape().GetDim(1), gmmWeight->GetViewShape().GetDim(2),
                transposeGmmWeight->GetViewShape().GetDim(1), transposeGmmWeight->GetViewShape().GetDim(2));
    }

    auto transposeGmmWeightScale = gmmWeightScale;
    if (notContiguous && gmmWeightScale != nullptr && GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
        transposeGmmWeightScale = TransGmmWeightScaleTensor(gmmWeightScale);
        CHECK_RET(transposeGmmWeightScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 保存原始指针用于一致性校验
    const aclTensor *originalMmWeightOptional = mmWeightOptional;
    const aclTensor *originalMmWeightScaleOptional = mmWeightScaleOptional;
    if (CheckMmWeightValid(mmWeightOptional)) {
        bool notContiguous = IsTransposeTwoDims(mmWeightOptional, -1, -2);
        auto transMmWeightOptional = mmWeightOptional;
        int64_t mmDim0 = mmWeightOptional->GetViewShape().GetDim(0);
        int64_t mmDim1 = mmWeightOptional->GetViewShape().GetDim(1);
        int64_t mmStride0 = mmWeightOptional->GetViewStrides()[0];
        int64_t mmStride1 = mmWeightOptional->GetViewStrides()[1];
        if (notContiguous && transMmWeight) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize",
                                                  "mmWeightOptional", "not contiguous and transpose set",
                                                  "contiguous tensor or transpose disabled");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
            transMmWeight = !transMmWeight;
            transMmWeightOptional = TransMmWeightOptionalTensor(mmWeightOptional);
            CHECK_RET(transMmWeightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            mmWeightOptional = transMmWeightOptional;
        }
    }

    auto transMmWeightScaleOptional = mmWeightScaleOptional;
    bool mmWeightScaleNotContiguous = false;
    if (mmWeightScaleOptional != nullptr && mmWeightScaleOptional->GetViewShape().GetDimNum() >= 3 &&
        GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
        mmWeightScaleNotContiguous = IsTransposeTwoDims(mmWeightScaleOptional, -2, -3);
        if (mmWeightScaleNotContiguous) {
            transMmWeightScaleOptional = TransMmWeightScaleTensor(mmWeightScaleOptional);
            CHECK_RET(transMmWeightScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    aclnnStatus ret_consistency = CheckWeightScaleTransposeConsistency(
        gmmWeight, gmmWeightScale, originalMmWeightOptional, originalMmWeightScaleOptional);
    CHECK_RET(ret_consistency == ACLNN_SUCCESS, ret_consistency);

    aclnnStatus ret_param =
        CheckParams(gmmX, transposeGmmWeight, gmmXScale, transposeGmmWeightScale, sendCountsTensorOptional,
                    recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional, mmWeightScaleOptional,
                    gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, group, epWorldSize,
                    permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, transposeGmmWeight, gmmXScale, transposeGmmWeightScale, sendCountsTensorOptional,
        recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional, transMmWeightScaleOptional, group,
        epWorldSize, sendCounts, recvCounts, gmmXQuantMode, gmmWeightQuantMode, transGmmWeight, transMmWeight,
        permuteOutFlag, mmXQuantMode, mmWeightQuantMode, groupSize, gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                OP_LOGD("arch35, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                OP_LOGD("arch35, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvQuantGroupedMatMul(workspace, workspaceSize, executor, stream);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnInnerAlltoAllvQuantGroupedMatMul ret %d.", ret);
    return ret;
}
} // namespace
