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
 * \file aclnn_attention_update.cpp
 * \brief
 */

#include "aclnn_attention_update.h"
#include "attention_update.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace attention_update {

static const int64_t LSE_TOTAL_LENGTH_DIM = 0;
static const int64_t GO_TOTAL_LENGTH_DIM = 0;
static const int64_t GO_HD_DIM = 1;

static const int64_t OUT_TOTAL_LENGTH_DIM = 0;
static const int64_t OUT_HD_DIM = 1;
static const int64_t ATTENTION_UPDATE_OUT_DIM_NUM = 2;
static const int64_t LSE_DIM_NUM = 1;
static const int64_t GO_DIM_NUM = 2;
static const int64_t HD_MULTIPLE = 8;
static const int64_t HD_MAX = 512;
static const int64_t SP_MAX = 128;
static const int64_t SP_MAX_95 = 16;

static const std::initializer_list<op::DataType> ATTENTION_UPDATE_DTYPE_SUPPORT_LIST = {
                                DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ATTENTION_UPDATE_DTYPE_SUPPORT_LIST_LOCALOUT = {
                                DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> ATTENTION_UPDATE_DTYPE_SUPPORT_LIST_95 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensorList* lse, const aclTensorList* localOut, const aclTensor* out) {
    OP_CHECK_NULL(lse, return false);
    for (uint64_t i = 0; i < lse->Size(); i++) {
        OP_CHECK_NULL((*lse)[i], return false);
    }
    OP_CHECK_NULL(localOut, return false);
    for (uint64_t i = 0; i < localOut->Size(); i++) {
        OP_CHECK_NULL((*localOut)[i], return false);
    }
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensorList* lse, const aclTensorList* localOut, const aclTensor* out) {
    for (size_t i = 0; i < lse->Size(); i++) {
        if ((*lse)[i]->GetViewShape().GetShapeSize() != 0) {
            OP_CHECK_DTYPE_NOT_SUPPORT((*lse)[i], ATTENTION_UPDATE_DTYPE_SUPPORT_LIST, return false);
        }
    }
    for (size_t i = 0; i < localOut->Size(); i++) {
        if ((*localOut)[i]->GetViewShape().GetShapeSize() != 0) {
            OP_CHECK_DTYPE_NOT_SUPPORT((*localOut)[i], ATTENTION_UPDATE_DTYPE_SUPPORT_LIST_LOCALOUT, return false);
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, ATTENTION_UPDATE_DTYPE_SUPPORT_LIST_LOCALOUT, return false);

    return true;
}

static inline bool CheckDtypeValid_95(const aclTensorList *lse, const aclTensorList *localOut, const aclTensor *out,
                                      const aclTensor *lseOut, uint64_t updateType)
{
    for (size_t i = 0; i < lse->Size(); i++) {
        if ((*lse)[i]->GetViewShape().GetShapeSize() != 0) {
            OP_CHECK_DTYPE_NOT_SUPPORT((*lse)[i], {DataType::DT_FLOAT}, return false);
        }
    }

    // localOut[0]检查dtype是否支持，其他检查dtype是否等于localOut[0]
    for (size_t i = 0; i < localOut->Size(); i++) {
        if ((*localOut)[i]->GetViewShape().GetShapeSize() != 0) {
            if (i == 0) {
                OP_CHECK_DTYPE_NOT_SUPPORT((*localOut)[i], ATTENTION_UPDATE_DTYPE_SUPPORT_LIST_95, return false);
            } else {
                OP_CHECK_DTYPE_NOT_SAME((*localOut)[i], (*localOut)[0], return false);
            }
        }
    }

    // out检查dtype是否等于localOut[0]
    OP_CHECK_DTYPE_NOT_SAME(out, (*localOut)[0], return false);

    if (updateType == static_cast<uint64_t>(1)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(lseOut, {DataType::DT_FLOAT}, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* lse, const aclTensorList* localOut, const aclTensor* out,
                                const aclTensor* lseOut, const int64_t updateType) {
    auto lseSp = lse->Size();
    auto localOutSp = localOut->Size();

    OP_CHECK(lseSp > 0 && lseSp <= SP_MAX,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse must be greater than zero and less than %ld, "
                                         "but which is %ld.", SP_MAX, lseSp), return false);
    OP_CHECK(localOutSp == lseSp,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse: %ld must equal to size of localOut: %ld.", lseSp, localOutSp),
        return false);
    
    auto lseShape = (*lse)[0]->GetViewShape();
    auto goShape = (*localOut)[0]->GetViewShape();

    OP_CHECK(lseShape.GetDimNum() == LSE_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of lse should be %ld, but which is %ld.",
                                                    LSE_DIM_NUM, lseShape.GetDimNum()),
        return false);
    OP_CHECK(goShape.GetDimNum() == GO_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of localOut should be %ld, but which is %ld.",
                                                    GO_DIM_NUM, goShape.GetDimNum()),
        return false);
    for (size_t i = 1; i < lseSp; i++) {
        OP_CHECK(lseShape == (*lse)[i]->GetViewShape(),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of every tensor of lse must be same."),
            return false);
        OP_CHECK(goShape == (*localOut)[i]->GetViewShape(),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of every tensor of localOut must be same."),
            return false);
    }

    auto outShape = out->GetViewShape();

    auto lseBshc =  lseShape.GetDim(LSE_TOTAL_LENGTH_DIM);
    auto localOutBshc =  goShape.GetDim(GO_TOTAL_LENGTH_DIM);

    auto goHd =  goShape.GetDim(GO_HD_DIM);

    OP_CHECK(goHd > 0 && goHd % HD_MULTIPLE == 0 && goHd <= HD_MAX ,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of localOut[1] must be a multiple of 8 and less than %ld, but which is %ld.",
                HD_MAX, goHd), return false);

    OP_CHECK(lseBshc == localOutBshc,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[0]: %ld must equal to size of localOut[0]: %ld.",
                lseBshc, localOutBshc), return false);

    if (updateType == 0 || updateType == 1) { // ATTENTION_UPDATE
        auto outDimNum = outShape.GetDimNum();
        OP_CHECK(outDimNum == ATTENTION_UPDATE_OUT_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of out should be %ld, but which is %ld.",
                                                    ATTENTION_UPDATE_OUT_DIM_NUM, outDimNum),
        return false);

        auto outBshc = outShape.GetDim(OUT_TOTAL_LENGTH_DIM);
        OP_CHECK(lseBshc == outBshc,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of out[%ld]: %ld.",
                                LSE_TOTAL_LENGTH_DIM, lseBshc, OUT_TOTAL_LENGTH_DIM, outBshc),
        return false);

        auto outHd = outShape.GetDim(OUT_HD_DIM);
        OP_CHECK(goHd == outHd,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of localOut[%ld]: %ld must equal to size of out[%ld]: %ld.",
                                GO_HD_DIM, goHd, OUT_HD_DIM, outHd),
        return false);
        if (updateType == 1) {
            auto lseOutShape = lseOut->GetViewShape();
            auto lseOutDimNum = lseOutShape.GetDimNum();
            OP_CHECK(lseOutDimNum  == LSE_DIM_NUM,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of lseOut should be %ld, but which is %ld.",
                                                        LSE_DIM_NUM, lseOutDimNum),
                return false);

            auto lseOutBshc = lseOutShape.GetDim(OUT_TOTAL_LENGTH_DIM);
            OP_CHECK(lseBshc == lseOutBshc,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of lseOut[%ld]: %ld.",
                                    LSE_TOTAL_LENGTH_DIM, lseBshc, OUT_TOTAL_LENGTH_DIM, lseOutBshc),
                return false);
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported updateType: %ld.", updateType);
        return false;
    }
    return true;
}

static inline bool CheckShape_95(const aclTensorList *lse, const aclTensorList *localOut, const aclTensor *out,
                                 const aclTensor *lseOut, const int64_t updateType)
{
    auto lseShape = (*lse)[0]->GetViewShape();
    auto goShape = (*localOut)[0]->GetViewShape();

    // 检查go和lse维度
    OP_CHECK(lseShape.GetDimNum() == LSE_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of lse should be %ld, but which is %ld.", LSE_DIM_NUM,
                     lseShape.GetDimNum()),
             return false);
    OP_CHECK(goShape.GetDimNum() == GO_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of localOut should be %ld, but which is %ld.", GO_DIM_NUM,
                     goShape.GetDimNum()),
             return false);

    // 检查go和lse的第1,2维
    auto lseBshc = lseShape.GetDim(LSE_TOTAL_LENGTH_DIM);
    auto localOutBshc = goShape.GetDim(GO_TOTAL_LENGTH_DIM);

    auto goHd = goShape.GetDim(GO_HD_DIM);

    OP_CHECK(goHd > 0 && goHd % HD_MULTIPLE == 0 && goHd <= HD_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Size of localOut[1] must be a multiple of 8 and less than %ld, but which is %ld.", HD_MAX, goHd),
             return false);

    OP_CHECK(lseBshc == localOutBshc,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[0]: %ld must equal to size of localOut[0]: %ld.", lseBshc,
                     localOutBshc),
             return false);

    // 检查lse和go每个shape是否相同
    for (size_t i = 1; i < lse->Size(); i++) {
        OP_CHECK(lseShape == (*lse)[i]->GetViewShape(),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of every tensor of lse must be same."), return false);
        OP_CHECK(goShape == (*localOut)[i]->GetViewShape(),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of every tensor of localOut must be same."), return false);
    }

    // 检查update，out，lseOut
    if (updateType == 0 || updateType == 1) {
        // 校验out的维度，第1,2维
        auto outShape = out->GetViewShape();
        auto outDimNum = outShape.GetDimNum();
        OP_CHECK(outDimNum == ATTENTION_UPDATE_OUT_DIM_NUM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of out should be %ld, but which is %ld.",
                         ATTENTION_UPDATE_OUT_DIM_NUM, outDimNum),
                 return false);

        auto outBshc = outShape.GetDim(OUT_TOTAL_LENGTH_DIM);
        OP_CHECK(lseBshc == outBshc,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of out[%ld]: %ld.",
                         LSE_TOTAL_LENGTH_DIM, lseBshc, OUT_TOTAL_LENGTH_DIM, outBshc),
                 return false);

        auto outHd = outShape.GetDim(OUT_HD_DIM);
        OP_CHECK(goHd == outHd,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of localOut[%ld]: %ld must equal to size of out[%ld]: %ld.",
                         GO_HD_DIM, goHd, OUT_HD_DIM, outHd),
                 return false);

        // 校验lseOut的维度，第1维
        if (updateType == 1) {
            auto lseOutShape = lseOut->GetViewShape();
            auto lseOutDimNum = lseOutShape.GetDimNum();
            OP_CHECK(lseOutDimNum == LSE_DIM_NUM,
                     OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of lseOut should be %ld, but which is %ld.", LSE_DIM_NUM,
                             lseOutDimNum),
                     return false);

            auto lseOutBshc = lseOutShape.GetDim(OUT_TOTAL_LENGTH_DIM);
            OP_CHECK(lseBshc == lseOutBshc,
                     OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of lseOut[%ld]: %ld.",
                             LSE_TOTAL_LENGTH_DIM, lseBshc, OUT_TOTAL_LENGTH_DIM, lseOutBshc),
                     return false);
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported updateType: %ld.", updateType);
        return false;
    }

    return true;
}

static inline bool CheckSp(const aclTensorList *lse, const aclTensorList *localOut)
{
    auto lseSp = lse->Size();
    auto localOutSp = localOut->Size();

    OP_CHECK(lseSp > 0 && lseSp <= SP_MAX,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Size of lse must be greater than zero and less than %ld, "
                     "but which is %ld.",
                     SP_MAX, lseSp),
             return false);
    OP_CHECK(
        localOutSp == lseSp,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse: %ld must equal to size of localOut: %ld.", lseSp, localOutSp),
        return false);
    return true;
}

static inline bool CheckSp_95(const aclTensorList *lse, const aclTensorList *localOut)
{
    auto lseSp = lse->Size();
    auto localOutSp = localOut->Size();

    OP_CHECK(lseSp > 0 && lseSp <= SP_MAX_95,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Size of lse must be greater than zero and less than %ld, "
                     "but which is %ld.",
                     SP_MAX_95, lseSp),
             return false);
    OP_CHECK(
        localOutSp == lseSp,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse: %ld must equal to size of localOut: %ld.", lseSp, localOutSp),
        return false);
    return true;
}

static inline bool CheckUpdateTypeAndLseOut(int64_t updateType, const aclTensor *lseOut)
{
    if (updateType == 0 && lseOut != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lseOut should be nullptr when updateType is %ld.", updateType);
        return false;
    }
    if (updateType == 1 && lseOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lseOut should not be nullptr when updateType is %ld.", updateType);
        return false;
    }

    if (updateType != 1 && updateType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updateType should be 0 or 1 but got %ld.", updateType);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams_95(const aclTensorList *lse, const aclTensorList *localOut, const aclTensor *out,
                                  const aclTensor *lseOut, const int64_t updateType)
{
    // lseOut作输出
    CHECK_RET(CheckUpdateTypeAndLseOut(updateType, lseOut), ACLNN_ERR_PARAM_INVALID);

    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(lse, localOut, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入sp是否支持
    CHECK_RET(CheckSp_95(lse, localOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid_95(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckShape_95(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensorList* lse, const aclTensorList* localOut,
            const aclTensor* out, const aclTensor* lseOut, const int64_t updateType) {
    // 启用lseout
    CHECK_RET(CheckUpdateTypeAndLseOut(updateType, lseOut), ACLNN_ERR_PARAM_INVALID);

    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(lse, localOut, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(lse, localOut, out), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入sp是否支持
    CHECK_RET(CheckSp(lse, localOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

} // namespace attention_update

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(
    const aclTensorList* lse, const aclTensorList* localOut, int64_t updateType,
    aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAttentionUpdate,
                   DFX_IN(lse, localOut, updateType),
                   DFX_OUT(out, lseOut));
    
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    // 固定写法，参数检查
    aclnnStatus ret;
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        ret = attention_update::CheckParams_95(lse, localOut, out, lseOut, updateType);
    } else {
        ret = attention_update::CheckParams(lse, localOut, out, lseOut, updateType);
    }
    
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if ((*lse)[0]->IsEmpty() || (*localOut)[0]->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto sp = lse->Size();

    FVector<const aclTensor*, attention_update::SP_MAX> lseContiguousVec;
    FVector<const aclTensor*, attention_update::SP_MAX> localOutContiguousVec;
    for (size_t i = 0; i < sp; i++) {
        auto ilseContiguous = l0op::Contiguous((*lse)[i], uniqueExecutor.get());
        lseContiguousVec.emplace_back(ilseContiguous);

        auto ilocalOutContiguous = l0op::Contiguous((*localOut)[i], uniqueExecutor.get());
        localOutContiguousVec.emplace_back(ilocalOutContiguous);
    }

    auto lseContiguous = uniqueExecutor.get()->AllocTensorList(lseContiguousVec.data(), sp);
    auto localOutContiguous = uniqueExecutor.get()->AllocTensorList(localOutContiguousVec.data(), sp);

    auto [AttentionUpdateRes, lseM] = l0op::AttentionUpdate(lseContiguous, localOutContiguous, updateType, sp, uniqueExecutor.get());
    CHECK_RET(AttentionUpdateRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(AttentionUpdateRes, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (updateType == 1) {
        CHECK_RET(lseM != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResultLseM = l0op::ViewCopy(lseM, lseOut, uniqueExecutor.get());
        CHECK_RET(viewCopyResultLseM != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAttentionUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAttentionUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif