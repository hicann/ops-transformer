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
 * \file aclnn_mhc_post_backward.cpp
 * \brief MhcPostBackward ACLNN API implementation
 */

#include "aclnn_mhc_post_backward.h"
#include "mhc_post_backward.h"

#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DIM_NUM_2 = 2;
static const int64_t DIM_NUM_3 = 3;
static const int64_t DIM_NUM_4 = 4;
static const int64_t DIM_IDX_0 = 0;
static const int64_t DIM_IDX_1 = 1;
static const int64_t DIM_IDX_2 = 2;
static const int64_t DIM_IDX_3 = 3;
static const int64_t D_ALIGN = 128;
static const int64_t D_MAX = 100000;
static const int64_t N_VALUE = 4;

struct MhcPostBackwardParams {
    const aclTensor *gradOutput = nullptr;
    const aclTensor *x = nullptr;
    const aclTensor *hRes = nullptr;
    const aclTensor *hOut = nullptr;
    const aclTensor *hPost = nullptr;
    aclTensor *gradX = nullptr;
    aclTensor *gradHres = nullptr;
    aclTensor *gradHout = nullptr;
    aclTensor *gradHpost = nullptr;
};

static aclnnStatus CheckNotNull(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *hRes,
                                const aclTensor *hOut, const aclTensor *hPost, aclTensor *gradX, aclTensor *gradHres,
                                aclTensor *gradHout, aclTensor *gradHpost)
{
    CHECK_COND(gradOutput != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradOutput must not be nullptr.");
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    // hRes is optional on A5 (DAV_3510), required on A2/A3
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510) {
        CHECK_COND(hRes != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hRes must not be nullptr.");
    }
    CHECK_COND(hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(hPost != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hPost must not be nullptr.");
    CHECK_COND(gradX != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradX must not be nullptr.");
    CHECK_COND(gradHres != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradHres must not be nullptr.");
    CHECK_COND(gradHout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradHout must not be nullptr.");
    CHECK_COND(gradHpost != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradHpost must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const MhcPostBackwardParams &params)
{
    // x , gradOutput: FP16 or BF16
    const std::initializer_list<op::DataType> SupportDtypeList = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(params.gradOutput, SupportDtypeList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x, SupportDtypeList, return ACLNN_ERR_PARAM_INVALID);

    // hRes: FP32 only (optional, skip check when nullptr)
    if (params.hRes != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(params.hRes, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    }

    // hOut: FP16 or BF16
    OP_CHECK_DTYPE_NOT_SUPPORT(params.hOut, SupportDtypeList, return ACLNN_ERR_PARAM_INVALID);

    // hPost: FP32 only
    OP_CHECK_DTYPE_NOT_MATCH(params.hPost, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    // gradX: must be same as x
    OP_CHECK_DTYPE_NOT_SAME(params.x, params.gradX, return ACLNN_ERR_PARAM_INVALID);

    // gradHres must be same as hRes (or FP32 when hRes is nullptr)
    if (params.hRes != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(params.hRes, params.gradHres, return ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_CHECK_DTYPE_NOT_MATCH(params.gradHres, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    }

    // gradHout: must be same as hOut
    OP_CHECK_DTYPE_NOT_SAME(params.hOut, params.gradHout, return ACLNN_ERR_PARAM_INVALID);

    // gradHpost: must be same as hPost
    OP_CHECK_DTYPE_NOT_SAME(params.hPost, params.gradHpost, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsTNDFormat(size_t dimNum)
{
    return dimNum == DIM_NUM_3;
}

static bool IsBSNDFormat(size_t dimNum)
{
    return dimNum == DIM_NUM_4;
}

static aclnnStatus CheckDimD(int64_t dimD)
{
    CHECK_COND(dimD > 0 && dimD <= D_MAX, ACLNN_ERR_PARAM_INVALID, "the dim D should be > 0, <= %ld, but got %ld",
               D_MAX, dimD);
    CHECK_COND(dimD % D_ALIGN == 0, ACLNN_ERR_PARAM_INVALID, "the dim D should be aligned to %ld, but got %ld", D_ALIGN,
               dimD);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape3D(const MhcPostBackwardParams &params)
{
    // x shape: [T, n, D]
    auto xDim0 = params.x->GetViewShape().GetDim(DIM_IDX_0);
    auto xDim1 = params.x->GetViewShape().GetDim(DIM_IDX_1);
    auto xDim2 = params.x->GetViewShape().GetDim(DIM_IDX_2);

    // gradOutput: [T, n, D]
    auto gradOutputDim0 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_0);
    auto gradOutputDim1 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_1);
    auto gradOutputDim2 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(gradOutputDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[0] %ld is not equal to x dim[0] %ld",
               gradOutputDim0, xDim0);
    CHECK_COND(gradOutputDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[1] %ld is not equal to x dim[1] %ld",
               gradOutputDim1, xDim1);
    CHECK_COND(gradOutputDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[2] %ld is not equal to x dim[2] %ld",
               gradOutputDim2, xDim2);

    // hRes shape: [T, n, n] (optional, skip check when nullptr)
    if (params.hRes != nullptr) {
        auto hResDim0 = params.hRes->GetViewShape().GetDim(DIM_IDX_0);
        auto hResDim1 = params.hRes->GetViewShape().GetDim(DIM_IDX_1);
        auto hResDim2 = params.hRes->GetViewShape().GetDim(DIM_IDX_2);
        CHECK_COND(hResDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "hRes dim[0] %ld is not equal to x dim[0] %ld", hResDim0,
                   xDim0);
        CHECK_COND(hResDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "hRes dim[1] %ld is not equal to x dim[1] %ld", hResDim1,
                   xDim1);
        CHECK_COND(hResDim2 == xDim1, ACLNN_ERR_PARAM_INVALID,
                   "hRes dim[2] %ld must equal hRes dim[1] %ld (n x n matrix)", hResDim2, hResDim1);
    }

    // hOut shape: [T, D]
    auto hOutDim0 = params.hOut->GetViewShape().GetDim(DIM_IDX_0);
    auto hOutDim1 = params.hOut->GetViewShape().GetDim(DIM_IDX_1);
    CHECK_COND(hOutDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "hOut dim[0] %ld is not equal to x dim[0] %ld", hOutDim0,
               xDim0);
    CHECK_COND(hOutDim1 == xDim2, ACLNN_ERR_PARAM_INVALID, "hOut dim[1] %ld is not equal to x dim[2] %ld", hOutDim1,
               xDim2);

    // hPost shape: [T, n]
    auto hPostDim0 = params.hPost->GetViewShape().GetDim(DIM_IDX_0);
    auto hPostDim1 = params.hPost->GetViewShape().GetDim(DIM_IDX_1);
    CHECK_COND(hPostDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "hPost dim[0] %ld is not equal to x dim[0] %ld", hPostDim0,
               xDim0);
    CHECK_COND(hPostDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "hPost dim[1] %ld is not equal to x dim[1] %ld", hPostDim1,
               xDim1);

    // gradX shape: [T, n, D]
    auto gradXDim0 = params.gradX->GetViewShape().GetDim(DIM_IDX_0);
    auto gradXDim1 = params.gradX->GetViewShape().GetDim(DIM_IDX_1);
    auto gradXDim2 = params.gradX->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(gradXDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "gradX dim[0] %ld is not equal to x dim[0] %ld", gradXDim0,
               xDim0);
    CHECK_COND(gradXDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "gradX dim[1] %ld is not equal to x dim[1] %ld", gradXDim1,
               xDim1);
    CHECK_COND(gradXDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "gradX dim[2] %ld is not equal to x dim[2] %ld", gradXDim2,
               xDim2);

    // gradHres shape: [T, n, n]
    auto gradHresDim0 = params.gradHres->GetViewShape().GetDim(DIM_IDX_0);
    auto gradHresDim1 = params.gradHres->GetViewShape().GetDim(DIM_IDX_1);
    auto gradHresDim2 = params.gradHres->GetViewShape().GetDim(DIM_IDX_2);
    if (params.hRes != nullptr) {
        auto hResDim0 = params.hRes->GetViewShape().GetDim(DIM_IDX_0);
        auto hResDim1 = params.hRes->GetViewShape().GetDim(DIM_IDX_1);
        auto hResDim2 = params.hRes->GetViewShape().GetDim(DIM_IDX_2);
        CHECK_COND(gradHresDim0 == hResDim0, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim[0] %ld is not equal to hRes dim[0] %ld", gradHresDim0, hResDim0);
        CHECK_COND(gradHresDim1 == hResDim1, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim[1] %ld is not equal to hRes dim[1] %ld", gradHresDim1, hResDim1);
        CHECK_COND(gradHresDim2 == hResDim2, ACLNN_ERR_PARAM_INVALID, "gradHres dim[2] %ld must equal hRes dim[2] %ld",
                   gradHresDim2, hResDim2);
    } else {
        CHECK_COND(gradHresDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "gradHres dim[0] %ld is not equal to x dim[0] %ld",
                   gradHresDim0, xDim0);
        CHECK_COND(gradHresDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "gradHres dim[1] %ld is not equal to x dim[1] %ld",
                   gradHresDim1, xDim1);
        CHECK_COND(gradHresDim2 == xDim1, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim[2] %ld must equal x dim[1] %ld (n x n matrix)", gradHresDim2, xDim1);
    }

    // gradHout shape: [T, D]
    auto gradHoutDim0 = params.gradHout->GetViewShape().GetDim(DIM_IDX_0);
    auto gradHoutDim1 = params.gradHout->GetViewShape().GetDim(DIM_IDX_1);
    CHECK_COND(gradHoutDim0 == hOutDim0, ACLNN_ERR_PARAM_INVALID, "gradHout dim[0] %ld is not equal to hOut dim[0] %ld",
               gradHoutDim0, hOutDim0);
    CHECK_COND(gradHoutDim1 == hOutDim1, ACLNN_ERR_PARAM_INVALID, "gradHout dim[1] %ld is not equal to hOut dim[1] %ld",
               gradHoutDim1, hOutDim1);

    // gradHpost shape: [T, n]
    auto gradHpostDim0 = params.gradHpost->GetViewShape().GetDim(DIM_IDX_0);
    auto gradHpostDim1 = params.gradHpost->GetViewShape().GetDim(DIM_IDX_1);
    CHECK_COND(gradHpostDim0 == hPostDim0, ACLNN_ERR_PARAM_INVALID,
               "gradHpost dim[0] %ld is not equal to hPost dim[0] %ld", gradHpostDim0, hPostDim0);
    CHECK_COND(gradHpostDim1 == hPostDim1, ACLNN_ERR_PARAM_INVALID,
               "gradHpost dim[1] %ld is not equal to hPost dim[1] %ld", gradHpostDim1, hPostDim1);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape4D(const MhcPostBackwardParams &params)
{
    // x shape: [B, S, n, D]
    auto xDim0 = params.x->GetViewShape().GetDim(DIM_IDX_0);
    auto xDim1 = params.x->GetViewShape().GetDim(DIM_IDX_1);
    auto xDim2 = params.x->GetViewShape().GetDim(DIM_IDX_2);
    auto xDim3 = params.x->GetViewShape().GetDim(DIM_IDX_3);

    // gradOutput: [B, S, n, D]
    auto gradOutputDim0 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_0);
    auto gradOutputDim1 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_1);
    auto gradOutputDim2 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_2);
    auto gradOutputDim3 = params.gradOutput->GetViewShape().GetDim(DIM_IDX_3);
    CHECK_COND(gradOutputDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[0] %ld is not equal to x dim[0] %ld",
               gradOutputDim0, xDim0);
    CHECK_COND(gradOutputDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[1] %ld is not equal to x dim[1] %ld",
               gradOutputDim1, xDim1);
    CHECK_COND(gradOutputDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[2] %ld is not equal to x dim[2] %ld",
               gradOutputDim2, xDim2);
    CHECK_COND(gradOutputDim3 == xDim3, ACLNN_ERR_PARAM_INVALID, "gradOutput dim[3] %ld is not equal to x dim[3] %ld",
               gradOutputDim3, xDim3);

    // hRes shape: [B, S, n, n] (optional, skip check when nullptr)
    if (params.hRes != nullptr) {
        auto hResDim0 = params.hRes->GetViewShape().GetDim(DIM_IDX_0);
        auto hResDim1 = params.hRes->GetViewShape().GetDim(DIM_IDX_1);
        auto hResDim2 = params.hRes->GetViewShape().GetDim(DIM_IDX_2);
        auto hResDim3 = params.hRes->GetViewShape().GetDim(DIM_IDX_3);
        CHECK_COND(hResDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "hRes dim[0] %ld is not equal to x dim[0] %ld", hResDim0,
                   xDim0);
        CHECK_COND(hResDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "hRes dim[1] %ld is not equal to x dim[1] %ld", hResDim1,
                   xDim1);
        CHECK_COND(hResDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "hRes dim[2] %ld is not equal to x dim[2] %ld", hResDim2,
                   xDim2);
        CHECK_COND(hResDim3 == xDim2, ACLNN_ERR_PARAM_INVALID,
                   "hRes dim[3] %ld must equal hRes dim[2] %ld (n x n matrix)", hResDim3, hResDim2);
    }

    // hOut shape: [B, S, D]
    auto hOutDim0 = params.hOut->GetViewShape().GetDim(DIM_IDX_0);
    auto hOutDim1 = params.hOut->GetViewShape().GetDim(DIM_IDX_1);
    auto hOutDim2 = params.hOut->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(hOutDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "hOut dim[0] %ld is not equal to x dim[0] %ld", hOutDim0,
               xDim0);
    CHECK_COND(hOutDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "hOut dim[1] %ld is not equal to x dim[1] %ld", hOutDim1,
               xDim1);
    CHECK_COND(hOutDim2 == xDim3, ACLNN_ERR_PARAM_INVALID, "hOut dim[2] %ld is not equal to x dim[3] %ld", hOutDim2,
               xDim3);

    // hPost shape: [B, S, n]
    auto hPostDim0 = params.hPost->GetViewShape().GetDim(DIM_IDX_0);
    auto hPostDim1 = params.hPost->GetViewShape().GetDim(DIM_IDX_1);
    auto hPostDim2 = params.hPost->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(hPostDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "hPost dim[0] %ld is not equal to x dim[0] %ld", hPostDim0,
               xDim0);
    CHECK_COND(hPostDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "hPost dim[1] %ld is not equal to x dim[1] %ld", hPostDim1,
               xDim1);
    CHECK_COND(hPostDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "hPost dim[2] %ld is not equal to x dim[2] %ld", hPostDim2,
               xDim2);

    // gradX shape: [B, S, n, D]
    auto gradXDim0 = params.gradX->GetViewShape().GetDim(DIM_IDX_0);
    auto gradXDim1 = params.gradX->GetViewShape().GetDim(DIM_IDX_1);
    auto gradXDim2 = params.gradX->GetViewShape().GetDim(DIM_IDX_2);
    auto gradXDim3 = params.gradX->GetViewShape().GetDim(DIM_IDX_3);
    CHECK_COND(gradXDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "gradX dim[0] %ld is not equal to x dim[0] %ld", gradXDim0,
               xDim0);
    CHECK_COND(gradXDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "gradX dim[1] %ld is not equal to x dim[1] %ld", gradXDim1,
               xDim1);
    CHECK_COND(gradXDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "gradX dim[2] %ld is not equal to x dim[2] %ld", gradXDim2,
               xDim2);
    CHECK_COND(gradXDim3 == xDim3, ACLNN_ERR_PARAM_INVALID, "gradX dim[3] %ld is not equal to x dim[3] %ld", gradXDim3,
               xDim3);

    // gradHres shape: [B, S, n, n]
    auto gradHresDim0 = params.gradHres->GetViewShape().GetDim(DIM_IDX_0);
    auto gradHresDim1 = params.gradHres->GetViewShape().GetDim(DIM_IDX_1);
    auto gradHresDim2 = params.gradHres->GetViewShape().GetDim(DIM_IDX_2);
    auto gradHresDim3 = params.gradHres->GetViewShape().GetDim(DIM_IDX_3);
    if (params.hRes != nullptr) {
        auto hResDim0 = params.hRes->GetViewShape().GetDim(DIM_IDX_0);
        auto hResDim1 = params.hRes->GetViewShape().GetDim(DIM_IDX_1);
        auto hResDim2 = params.hRes->GetViewShape().GetDim(DIM_IDX_2);
        auto hResDim3 = params.hRes->GetViewShape().GetDim(DIM_IDX_3);
        CHECK_COND(gradHresDim0 == hResDim0, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim[0] %ld is not equal to hRes dim[0] %ld", gradHresDim0, hResDim0);
        CHECK_COND(gradHresDim1 == hResDim1, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim[1] %ld is not equal to hRes dim[1] %ld", gradHresDim1, hResDim1);
        CHECK_COND(gradHresDim2 == hResDim2, ACLNN_ERR_PARAM_INVALID, "gradHres dim[2] %ld must equal hRes dim[2] %ld",
                   gradHresDim2, hResDim2);
        CHECK_COND(gradHresDim3 == hResDim3, ACLNN_ERR_PARAM_INVALID, "gradHres dim[3] %ld must equal hRes dim[3] %ld",
                   gradHresDim3, hResDim3);
    } else {
        CHECK_COND(gradHresDim0 == xDim0, ACLNN_ERR_PARAM_INVALID, "gradHres dim[0] %ld is not equal to x dim[0] %ld",
                   gradHresDim0, xDim0);
        CHECK_COND(gradHresDim1 == xDim1, ACLNN_ERR_PARAM_INVALID, "gradHres dim[1] %ld is not equal to x dim[1] %ld",
                   gradHresDim1, xDim1);
        CHECK_COND(gradHresDim2 == xDim2, ACLNN_ERR_PARAM_INVALID, "gradHres dim[2] %ld must equal x dim[2] %ld",
                   gradHresDim2, xDim2);
        CHECK_COND(gradHresDim3 == xDim2, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim[3] %ld must equal x dim[2] %ld (n x n matrix)", gradHresDim3, xDim2);
    }

    // gradHout shape: [B, S, D]
    auto gradHoutDim0 = params.gradHout->GetViewShape().GetDim(DIM_IDX_0);
    auto gradHoutDim1 = params.gradHout->GetViewShape().GetDim(DIM_IDX_1);
    auto gradHoutDim2 = params.gradHout->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(gradHoutDim0 == hOutDim0, ACLNN_ERR_PARAM_INVALID, "gradHout dim[0] %ld is not equal to hOut dim[0] %ld",
               gradHoutDim0, hOutDim0);
    CHECK_COND(gradHoutDim1 == hOutDim1, ACLNN_ERR_PARAM_INVALID, "gradHout dim[1] %ld is not equal to hOut dim[1] %ld",
               gradHoutDim1, hOutDim1);
    CHECK_COND(gradHoutDim2 == hOutDim2, ACLNN_ERR_PARAM_INVALID, "gradHout dim[2] %ld is not equal to hOut dim[2] %ld",
               gradHoutDim2, hOutDim2);

    // gradHpost shape: [B, S, n]
    auto gradHpostDim0 = params.gradHpost->GetViewShape().GetDim(DIM_IDX_0);
    auto gradHpostDim1 = params.gradHpost->GetViewShape().GetDim(DIM_IDX_1);
    auto gradHpostDim2 = params.gradHpost->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(gradHpostDim0 == hPostDim0, ACLNN_ERR_PARAM_INVALID,
               "gradHpost dim[0] %ld is not equal to hPost dim[0] %ld", gradHpostDim0, hPostDim0);
    CHECK_COND(gradHpostDim1 == hPostDim1, ACLNN_ERR_PARAM_INVALID,
               "gradHpost dim[1] %ld is not equal to hPost dim[1] %ld", gradHpostDim1, hPostDim1);
    CHECK_COND(gradHpostDim2 == hPostDim2, ACLNN_ERR_PARAM_INVALID,
               "gradHpost dim[2] %ld is not equal to hPost dim[2] %ld", gradHpostDim2, hPostDim2);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const MhcPostBackwardParams &params)
{
    auto gradOutputDimNum = params.gradOutput->GetViewShape().GetDimNum();
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    auto hOutDimNum = params.hOut->GetViewShape().GetDimNum();
    auto hPostDimNum = params.hPost->GetViewShape().GetDimNum();
    auto gradXDimNum = params.gradX->GetViewShape().GetDimNum();
    auto gradHresDimNum = params.gradHres->GetViewShape().GetDimNum();
    auto gradHoutDimNum = params.gradHout->GetViewShape().GetDimNum();
    auto gradHpostDimNum = params.gradHpost->GetViewShape().GetDimNum();
    CHECK_COND(IsTNDFormat(xDimNum) || IsBSNDFormat(xDimNum), ACLNN_ERR_PARAM_INVALID,
               "x dim should be 3 (TND format) or 4 (BSND format), but got %zu", xDimNum);

    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();

    if (IsTNDFormat(xDimNum)) {
        auto xDim1 = params.x->GetViewShape().GetDim(DIM_IDX_1);
        if (npuArch == NpuArch::DAV_3510) {
            CHECK_COND((xDim1 == 4) || (xDim1 == 6) || (xDim1 == 8), ACLNN_ERR_PARAM_INVALID,
                       "the dim n should be 4 / 6 / 8 for 3D format");
        } else {
            CHECK_COND(xDim1 == N_VALUE, ACLNN_ERR_PARAM_INVALID, "the dim n should be %ld for 3D format", N_VALUE);
            auto xDimD = params.x->GetViewShape().GetDim(DIM_IDX_2);
            CHECK_COND(CheckDimD(xDimD) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dim D for 3D format");
        }
        CHECK_COND(gradOutputDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "GradOutput dim num should be 3 for 3D format, but got %zu", gradOutputDimNum);
        if (params.hRes != nullptr) {
            auto hResDimNum = params.hRes->GetViewShape().GetDimNum();
            CHECK_COND(hResDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                       "hRes dim num should be 3 for 3D format, but got %zu", hResDimNum);
        }
        CHECK_COND(hOutDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "hOut dim num should be 2 for 3D format, but got %zu", hOutDimNum);
        CHECK_COND(hPostDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "hPost dim num should be 2 for 3D format, but got %zu", hPostDimNum);
        CHECK_COND(gradXDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "gradX dim num should be 3 for 3D format, but got %zu", gradXDimNum);
        CHECK_COND(gradHresDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim num should be 3 for 3D format, but got %zu", gradHresDimNum);
        CHECK_COND(gradHoutDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "gradHout dim num should be 2 for 3D format, but got %zu", gradHoutDimNum);
        CHECK_COND(gradHpostDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "gradHpost dim num should be 2 for 3D format, but got %zu", gradHpostDimNum);
        CHECK_COND(CheckShape3D(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape for 3D format");
    } else {
        auto xDim2 = params.x->GetViewShape().GetDim(DIM_IDX_2);
        if (npuArch == NpuArch::DAV_3510) {
            CHECK_COND((xDim2 == 4) || (xDim2 == 6) || (xDim2 == 8), ACLNN_ERR_PARAM_INVALID,
                       "the dim n should be 4 / 6 / 8 for 4D format");
        } else {
            CHECK_COND(xDim2 == N_VALUE, ACLNN_ERR_PARAM_INVALID, "the dim n should be %ld for 4D format", N_VALUE);
            auto xDimD = params.x->GetViewShape().GetDim(DIM_IDX_3);
            CHECK_COND(CheckDimD(xDimD) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dim D for 4D format");
        }
        CHECK_COND(gradOutputDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
                   "GradOutput dim num should be 4 for 4D format, but got %zu", gradOutputDimNum);
        if (params.hRes != nullptr) {
            auto hResDimNum = params.hRes->GetViewShape().GetDimNum();
            CHECK_COND(hResDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
                       "hRes dim num should be 4 for 4D format, but got %zu", hResDimNum);
        }
        CHECK_COND(hOutDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "hOut dim num should be 3 for 4D format, but got %zu", hOutDimNum);
        CHECK_COND(hPostDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "hPost dim num should be 3 for 4D format, but got %zu", hPostDimNum);
        CHECK_COND(gradXDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
                   "gradX dim num should be 4 for 4D format, but got %zu", gradXDimNum);
        CHECK_COND(gradHresDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
                   "gradHres dim num should be 4 for 4D format, but got %zu", gradHresDimNum);
        CHECK_COND(gradHoutDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "gradHout dim num should be 3 for 4D format, but got %zu", gradHoutDimNum);
        CHECK_COND(gradHpostDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "gradHpost dim num should be 3 for 4D format, but got %zu", gradHpostDimNum);
        CHECK_COND(CheckShape4D(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape for 4D format");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const MhcPostBackwardParams &params)
{
    op::Format gradOutputFormat = params.gradOutput->GetStorageFormat();
    op::Format xFormat = params.x->GetStorageFormat();
    op::Format hOutFormat = params.hOut->GetStorageFormat();
    op::Format hPostFormat = params.hPost->GetStorageFormat();
    CHECK_COND(!op::IsPrivateFormat(gradOutputFormat), ACLNN_ERR_PARAM_INVALID, "format of gradOutput %s is invalid.",
               op::ToString(gradOutputFormat).GetString());
    CHECK_COND(!op::IsPrivateFormat(xFormat), ACLNN_ERR_PARAM_INVALID, "format of x %s is invalid.",
               op::ToString(xFormat).GetString());
    if (params.hRes != nullptr) {
        op::Format hResFormat = params.hRes->GetStorageFormat();
        CHECK_COND(!op::IsPrivateFormat(hResFormat), ACLNN_ERR_PARAM_INVALID, "format of hRes %s is invalid.",
                   op::ToString(hResFormat).GetString());
    }
    CHECK_COND(!op::IsPrivateFormat(hOutFormat), ACLNN_ERR_PARAM_INVALID, "format of hOut %s is invalid.",
               op::ToString(hOutFormat).GetString());
    CHECK_COND(!op::IsPrivateFormat(hPostFormat), ACLNN_ERR_PARAM_INVALID, "format of hPost %s is invalid.",
               op::ToString(hPostFormat).GetString());
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParam(const MhcPostBackwardParams &params)
{
    CHECK_COND(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid format.");
    CHECK_COND(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
    CHECK_COND(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape.");

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPostBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *hRes,
                                                 const aclTensor *hOut, const aclTensor *hPost, aclTensor *gradX,
                                                 aclTensor *gradHres, aclTensor *gradHout, aclTensor *gradHpost,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(CheckNotNull(gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost) == ACLNN_SUCCESS,
               ACLNN_ERR_PARAM_NULLPTR, "one of required inputs for aclnnMhcPostBackwardGetWorkspaceSize is nullptr.");
    // Check if input tensors are empty
    if (gradOutput->IsEmpty() || x->IsEmpty() || hOut->IsEmpty() || hPost->IsEmpty() ||
        (hRes != nullptr && hRes->IsEmpty())) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnn_mhc_post_backward do not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    MhcPostBackwardParams params{gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost};
    aclnnStatus ret = CheckParam(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclTensor *outGradX = nullptr;
    aclTensor *outGradHres = nullptr;
    aclTensor *outGradHout = nullptr;
    aclTensor *outGradHpost = nullptr;

    L2_DFX_PHASE_1(aclnnMhcPostBackward, DFX_IN(params.gradOutput, params.x, params.hRes, params.hOut, params.hPost),
                   DFX_OUT(params.gradX, params.gradHres, params.gradHout, params.gradHpost));

    // Convert input tensors to contiguous
    auto reformatedGradOutput = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_COND(reformatedGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR, "reformatedGradOutput Contiguous failed.");
    auto reformatedX = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(reformatedX != nullptr, ACLNN_ERR_INNER_NULLPTR, "x Contiguous failed.");
    const aclTensor *reformatedHRes = nullptr;
    if (hRes != nullptr) {
        reformatedHRes = l0op::Contiguous(hRes, uniqueExecutor.get());
        CHECK_COND(reformatedHRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "hRes Contiguous failed.");
    }
    auto reformatedHOut = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_COND(reformatedHOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "hOut Contiguous failed.");
    auto reformatedHPost = l0op::Contiguous(hPost, uniqueExecutor.get());
    CHECK_COND(reformatedHPost != nullptr, ACLNN_ERR_INNER_NULLPTR, "hPost Contiguous failed.");

    // Call l0 interface: MhcPostBackward kernel
    const auto Result = l0op::MhcPostBackward(reformatedGradOutput, reformatedX, reformatedHRes, reformatedHOut,
                                              reformatedHPost, uniqueExecutor.get());

    CHECK_RET(std::get<0>(Result) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outGradX = std::get<0>(Result);
    outGradHres = std::get<1>(Result);
    outGradHout = std::get<2>(Result);
    outGradHpost = std::get<3>(Result);
    CHECK_RET(outGradHres != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outGradHout != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outGradHpost != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Convert output tensor to contiguous tensor and copy to output
    auto viewCopygradXResult = l0op::ViewCopy(outGradX, gradX, uniqueExecutor.get());
    CHECK_RET(viewCopygradXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopygradHresResult = l0op::ViewCopy(outGradHres, gradHres, uniqueExecutor.get());
    CHECK_RET(viewCopygradHresResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopygradHoutResult = l0op::ViewCopy(outGradHout, gradHout, uniqueExecutor.get());
    CHECK_RET(viewCopygradHoutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopygradHpostResult = l0op::ViewCopy(outGradHpost, gradHpost, uniqueExecutor.get());
    CHECK_RET(viewCopygradHpostResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPostBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPostBackward);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in MhcPostBackward launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
