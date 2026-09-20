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
 * \file aclnn_rope_with_sin_cos_cache.cpp
 * \brief
 */

#include "rope_with_sin_cos_cache.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "level0/add.h"
#include "rsqrt.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_rope_with_sin_cos_cache.h"
#include "aclnn_rope_with_sin_cos_cache_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "iostream"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t QUERY_OUT_INDEX = 0;
static const int64_t KEY_OUT_INDEX = 1;

static const int64_t DIM_ZERO = 0;
static const int64_t DIM_ONE = 1;
static const int64_t DIM_NUM = 2;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_BF16,
    op::DataType::DT_FLOAT,
};

static const std::initializer_list<op::DataType> positions_dtype_list = {op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> emptyDtypes = {};

static const std::initializer_list<std::vector<int64_t>> mropeSupportList = {
    {16, 24, 24}, {8, 12, 12}, {24, 20, 20}, {11, 11, 10}, {16, 16, 16, 16}}; // 支持的mrope输入

static const std::initializer_list<int64_t> cacheModeSupportList = {0, 1};

static const std::initializer_list<DataType> &GetSupportDtypeList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        Ops::Transformer::AclnnUtil::IsRegbase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckNotNull(const aclTensor *positions, const aclTensor *queryIn, const aclTensor *keyIn,
                         const aclTensor *cosSinCache, const aclTensor *queryOut, const aclTensor *keyOut)
{
    OP_CHECK_NULL(positions, return false);
    OP_CHECK_NULL(queryIn, return false);
    OP_CHECK_NULL(keyIn, return false);
    OP_CHECK_NULL(cosSinCache, return false);
    OP_CHECK_NULL(queryOut, return false);
    OP_CHECK_NULL(keyOut, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *positions, const aclTensor *queryIn, const aclTensor *keyIn,
                            const aclTensor *cosSinCache, const aclTensor *queryOut, const aclTensor *keyOut)
{
    // 检查positions queryIn的数据类型是否在支持列表内
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    const auto &supportList = GetSupportDtypeList();
    if (supportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", ToString(socVersion).GetString());
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(positions, positions_dtype_list, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(queryIn, supportList, return false);

    // keyIn, cosSinCache, queryOut, keyOut的数据类型是否与out一致
    OP_CHECK_DTYPE_NOT_MATCH(keyIn, queryIn->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(cosSinCache, queryIn->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(queryOut, queryIn->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(keyOut, queryIn->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *positions, const aclTensor *queryIn, const aclTensor *keyIn,
                       const aclTensor *cosSinCache, const aclTensor *queryOut, const aclTensor *keyOut)
{
    // 检查输入的所有shape是不是2维；
    if (queryIn->GetViewShape().GetDimNum() != DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected queryIn to be a vector of size 2, "
                "but got queryIn of shape %s.",
                op::ToString(queryIn->GetViewShape()).GetString());
        return false;
    }

    if (keyIn->GetViewShape().GetDimNum() != DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected keyIn to be a vector of size 2, "
                "but got keyIn of shape %s.",
                op::ToString(keyIn->GetViewShape()).GetString());
        return false;
    }

    if (cosSinCache->GetViewShape().GetDimNum() != DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected cosSinCache to be a vector of size 2, "
                "but got cosSinCache of shape %s.",
                op::ToString(cosSinCache->GetViewShape()).GetString());
        return false;
    }

    // 检查是否有维度为0
    if (queryIn->IsEmpty() || keyIn->IsEmpty() || cosSinCache->IsEmpty() || positions->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "RopeWithSinCosCache not support to process empty tensor currently",
                op::ToString(positions->GetViewShape()).GetString());
        return false;
    }

    // 检查keyIn的1维度是不是等于queryIn的一维
    OP_CHECK(keyIn->GetViewShape()[0] == queryIn->GetViewShape()[0],
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Expected keyIn.shape[0] == queryIn->GetViewShape()[0] to be true, but got false."),
             return false);
    // 检查输入和输出的shape 是否相同
    OP_CHECK_SHAPE_NOT_EQUAL(queryOut, queryIn, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(keyOut, keyIn, return false);

    return true;
}

static bool CheckPositionsShape(const aclTensor *positions, const aclTensor *queryIn, bool isMrope,
                                uint64_t mropeSectionSize)
{
    const auto &positionsShape = positions->GetViewShape();
    int64_t numTokens = queryIn->GetViewShape()[DIM_ZERO];
    if (!isMrope) {
        OP_CHECK(positionsShape.GetDimNum() == DIM_ONE && positionsShape[DIM_ZERO] == numTokens,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected positions shape to be [%ld] in rope mode, but got %s.",
                         numTokens, op::ToString(positionsShape).GetString()),
                 return false);
        return true;
    }

    OP_CHECK(positionsShape.GetDimNum() == DIM_NUM &&
                 positionsShape[DIM_ZERO] == static_cast<int64_t>(mropeSectionSize) &&
                 positionsShape[DIM_ONE] == numTokens,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected positions shape to be [%lu, %ld] in mrope mode, but got %s.",
                     mropeSectionSize, numTokens, op::ToString(positionsShape).GetString()),
             return false);
    return true;
}

static bool CheckMropeSection(const std::vector<int64_t> &target, const std::vector<std::vector<int64_t>> &support)
{
    for (const auto &vec : support) {
        if (vec == target)
            return true;
    }
    return false;
}

static bool CheckCacheMode(const int64_t target, const std::initializer_list<int64_t> support)
{
    for (const auto &item : support) {
        if (item == target) {
            return true;
        }
    }
    return false;
}

static aclnnStatus CheckParams(const aclTensor *positions, const aclTensor *queryIn, const aclTensor *keyIn,
                               const aclTensor *cosSinCache, const aclIntArray *mropeSection, aclTensor *queryOut,
                               aclTensor *keyOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(positions, queryIn, keyIn, cosSinCache, queryOut, keyOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(positions, queryIn, keyIn, cosSinCache, queryOut, keyOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(positions, queryIn, keyIn, cosSinCache, queryOut, keyOut), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查positions shape是否与当前rope模式匹配
    if (mropeSection == nullptr) {
        CHECK_RET(CheckPositionsShape(positions, queryIn, false, 0), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    // 5. 检查mrope模式下是否满足mropeSection[0] + mropeSection[1] + mropeSection[2] == rotaryDim/2
    uint64_t mropeSectionSize = 0U;
    aclGetIntArraySize(mropeSection, &mropeSectionSize);
    OP_CHECK((mropeSectionSize == 3 || mropeSectionSize == 4),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "[aclnnRopeWithSinCosCache] Expected mropeSectionSize is 3 or 4, "
                     "but got %ld.",
                     mropeSectionSize),
             return ACLNN_ERR_PARAM_INVALID);
    std::vector<int64_t> mropeSectionIn;
    int64_t mropeSectionSum = 0;
    mropeSectionIn.reserve(mropeSectionSize);
    for (size_t i = 0; i < mropeSectionSize; ++i) {
        OP_CHECK(static_cast<int64_t>((*mropeSection)[i]) >= 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "[aclnnRopeWithSinCosCache] The value of mropeSection must be non-negative, "
                         "but got %ld.",
                         static_cast<int64_t>((*mropeSection)[i])),
                 return ACLNN_ERR_PARAM_INVALID);
        int64_t val = static_cast<int64_t>((*mropeSection)[i]);
        mropeSectionIn.push_back(val);
        mropeSectionSum += val;
    }
    int64_t rotary_dim = cosSinCache->GetViewShape()[1];
    // kernel中mropesection[0]>0为mrope模式，否则rope模式
    if (mropeSectionIn[0] > 0) {
        CHECK_RET(CheckPositionsShape(positions, queryIn, true, mropeSectionSize), ACLNN_ERR_PARAM_INVALID);
        OP_CHECK(mropeSectionSum * 2 == rotary_dim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "[aclnnRopeWithSinCosCache] The accumulated value of mropeSection"
                         "should be equal to rotaryDim/2, but got %ld.",
                         mropeSectionSum),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK(CheckMropeSection(mropeSectionIn, mropeSupportList),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[aclnnRopeWithSinCosCache] The input mropeSection "
                                                  "must be in the supported list."),
                 return ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckPositionsShape(positions, queryIn, false, 0), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRopeWithSinCosCacheGetWorkspaceSizeCommon(const aclTensor *positions, const aclTensor *queryIn,
                                                           const aclTensor *keyIn, const aclTensor *cosSinCache,
                                                           const aclIntArray *mropeSection, int64_t headSize,
                                                           bool isNeoxStyle, int64_t cacheMode, aclTensor *queryOut,
                                                           aclTensor *keyOut, uint64_t *workspaceSize,
                                                           aclOpExecutor **executor)
{
    // 固定写法，参数检查
    auto ret = CheckParams(positions, queryIn, keyIn, cosSinCache, mropeSection, queryOut, keyOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(headSize > 0, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入positions转换成连续的tensor
    auto positionsContiguous = l0op::Contiguous(positions, uniqueExecutor.get());
    CHECK_RET(positionsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入cosSinCosCache转换成连续的tensor
    auto cosSinCacheContiguous = l0op::Contiguous(cosSinCache, uniqueExecutor.get());
    CHECK_RET(cosSinCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t numQheads = queryIn->GetViewShape()[1] / headSize;
    int64_t numKheads = keyIn->GetViewShape()[1] / headSize;

    int64_t queryStride = queryIn->GetViewStrides()[0];
    int64_t keyStride = keyIn->GetViewStrides()[0];

    std::tuple<aclTensor *, aclTensor *> result = l0op::RopeWithSinCosCache(
        positionsContiguous, queryIn, keyIn, cosSinCacheContiguous, mropeSection, headSize, isNeoxStyle, queryStride,
        keyStride, numQheads, numKheads, cacheMode, uniqueExecutor.get());
    auto query = std::get<0>(result);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto key = std::get<1>(result);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto queryViewCopyResult = l0op::ViewCopy(query, queryOut, uniqueExecutor.get());
    CHECK_RET(queryViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyViewCopyResult = l0op::ViewCopy(key, keyOut, uniqueExecutor.get());
    CHECK_RET(keyViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRopeWithSinCosCacheGetWorkspaceSize(const aclTensor *positions, const aclTensor *queryIn,
                                                     const aclTensor *keyIn, const aclTensor *cosSinCache,
                                                     const aclIntArray *mropeSection, int64_t headSize,
                                                     bool isNeoxStyle, aclTensor *queryOut, aclTensor *keyOut,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // V1版本只支持cacheMode为0
    int64_t cacheMode = 0;
    L2_DFX_PHASE_1(aclnnRopeWithSinCosCache,
                   DFX_IN(positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle, cacheMode),
                   DFX_OUT(queryOut, keyOut));

    return aclnnRopeWithSinCosCacheGetWorkspaceSizeCommon(positions, queryIn, keyIn, cosSinCache, mropeSection,
                                                          headSize, isNeoxStyle, cacheMode, queryOut, keyOut,
                                                          workspaceSize, executor);
}

// 固定写法，获取计算过程中需要使用的workspace大小
aclnnStatus aclnnRopeWithSinCosCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRopeWithSinCosCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnRopeWithSinCosCacheV2GetWorkspaceSize(const aclTensor *positions, const aclTensor *queryIn,
                                                       const aclTensor *keyIn, const aclTensor *cosSinCache,
                                                       const aclIntArray *mropeSection, int64_t headSize,
                                                       bool isNeoxStyle, int64_t cacheMode, aclTensor *queryOut,
                                                       aclTensor *keyOut, uint64_t *workspaceSize,
                                                       aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRopeWithSinCosCacheV2,
                   DFX_IN(positions, queryIn, keyIn, cosSinCache, mropeSection, headSize, isNeoxStyle, cacheMode),
                   DFX_OUT(queryOut, keyOut));

    OP_CHECK(CheckCacheMode(cacheMode, cacheModeSupportList),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[aclnnRopeWithSinCosCache] cacheMode only support 0 or 1, but got %ld",
                     cacheMode),
             return ACLNN_ERR_PARAM_INVALID);

    return aclnnRopeWithSinCosCacheGetWorkspaceSizeCommon(positions, queryIn, keyIn, cosSinCache, mropeSection,
                                                          headSize, isNeoxStyle, cacheMode, queryOut, keyOut,
                                                          workspaceSize, executor);
}

// 固定写法，获取计算过程中需要使用的workspace大小
aclnnStatus aclnnRopeWithSinCosCacheV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRopeWithSinCosCacheV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
