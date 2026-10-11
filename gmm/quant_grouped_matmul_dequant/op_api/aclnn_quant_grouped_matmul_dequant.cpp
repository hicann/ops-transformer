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
 * \file aclnn_quant_grouped_matmul_dequant.cpp
 * \brief
 */

#include "aclnn_quant_grouped_matmul_dequant.h"
#include "aclnn_quant_grouped_matmul_dequant_weight_nz.h"
#include "quant_grouped_matmul_dequant.h"
#include "level0/padv3.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "util/math_util.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr uint64_t ND_DIMNUM = 3;
static constexpr uint64_t NZ_DIMNUM = 5;
static constexpr uint64_t FRACTAL_K_INT8 = 32;
static constexpr uint64_t FRACTAL_N_INT8 = 16;
static constexpr uint64_t N_IDX = 1;
static constexpr uint64_t K_IDX = 2;
static constexpr uint64_t X_IDX = 0;
static constexpr uint64_t WEIGTH_IDX = 1;
static constexpr uint64_t WEIGHT_SCALE_IDX = 2;
static constexpr uint64_t GROUPLIST_IDX = 5;
static constexpr uint64_t X_SCALE_IDX = 3;
static constexpr uint64_t SMOOTH_SCALE_IDX = 4;
static constexpr int64_t PAD_VEC_QGMMDQ[6] = {0, 0, 0, 0, 0, 16};
static constexpr uint64_t PAD_VEC_SIZE_QGMMDQ = 6;
static constexpr int64_t PERM_VEC_QGMMDQ[5] = {0, 3, 1, 2, 4};
static constexpr uint64_t PERM_VEC_SIZE_QGMMDQ = 5;
static const std::string PAD_MODE_QGMMDQ = "constant";

static const inline std::vector<std::vector<DataType>>& GetSupportDtypeList(SocVersion socVersion)
{
    static const std::vector<std::vector<DataType>> emptyDtypes = {};
    static const std::map<SocVersion, std::vector<std::vector<DataType>>> dataTypeSupportedMap = {
        {SocVersion::ASCEND310P,
         {{DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_FLOAT, DataType::DT_FLOAT, DataType::DT_FLOAT16,
           DataType::DT_INT64},
          {DataType::DT_FLOAT16, DataType::DT_INT8, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_FLOAT16,
           DataType::DT_INT64}}}};
    auto found = dataTypeSupportedMap.find(socVersion);
    if (found == dataTypeSupportedMap.end()) {
        return emptyDtypes;
    }
    return found->second;
}

static aclnnStatus CheckParamsNullOrNot(const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale,
                                        const aclTensor* groupList, const aclTensor* biasOptional,
                                        const aclTensor* xOffsetOptional, const aclTensor* out)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weight, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weightScale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(groupList, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    if (biasOptional != nullptr || xOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "not support biasOptional and xOffsetOptional for now, should be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsDateType(const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale,
                                       const aclTensor* groupList, const aclTensor* xScaleOptional,
                                       const aclTensor* smoothScaleOptional, const aclTensor* out)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();

    OP_CHECK_DTYPE_NOT_SAME(out, x, return ACLNN_ERR_PARAM_INVALID);

    const auto& DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(socVersion);
    if (DTYPE_SUPPORT_LIST_CURRENT.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (int i = 0; i < DTYPE_SUPPORT_LIST_CURRENT.size(); i++) {
        // 检查输入的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_MATCH(x, DTYPE_SUPPORT_LIST_CURRENT[i][X_IDX], continue);
        OP_CHECK_DTYPE_NOT_MATCH(weight, DTYPE_SUPPORT_LIST_CURRENT[i][WEIGTH_IDX], continue);
        OP_CHECK_DTYPE_NOT_MATCH(weightScale, DTYPE_SUPPORT_LIST_CURRENT[i][WEIGHT_SCALE_IDX], continue);
        OP_CHECK_DTYPE_NOT_MATCH(groupList, DTYPE_SUPPORT_LIST_CURRENT[i][GROUPLIST_IDX], continue);
        if (xScaleOptional != nullptr) {
            OP_CHECK_DTYPE_NOT_MATCH(xScaleOptional, DTYPE_SUPPORT_LIST_CURRENT[i][X_SCALE_IDX], continue);
        }
        if (smoothScaleOptional != nullptr) {
            OP_CHECK_DTYPE_NOT_MATCH(smoothScaleOptional, DTYPE_SUPPORT_LIST_CURRENT[i][SMOOTH_SCALE_IDX], continue);
        }

        return ACLNN_SUCCESS;
    }

    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckParamsDateType failed, please check log");
    return ACLNN_ERR_PARAM_INVALID;
}

static op::Shape GetWeightNzShape(const aclTensor* weight)
{
    uint64_t n = weight->GetViewShape().GetDim(N_IDX);
    uint64_t k = weight->GetViewShape().GetDim(K_IDX);

    uint64_t k1 = Ops::Base::CeilDiv(k, FRACTAL_K_INT8);
    uint64_t n1 = Ops::Base::CeilDiv(n, FRACTAL_N_INT8);

    op::Shape weightNzShape;
    weightNzShape.AppendDim(weight->GetViewShape().GetDim(0));
    weightNzShape.AppendDim(n1);
    weightNzShape.AppendDim(FRACTAL_N_INT8);
    weightNzShape.AppendDim(k1);
    weightNzShape.AppendDim(FRACTAL_K_INT8);
    return weightNzShape;
}
} // namespace

static aclnnStatus ConvertWeightNdToNz(const aclTensor*& weight, aclOpExecutor* exec)
{
    if (weight->GetViewShape().GetDim(N_IDX) % FRACTAL_N_INT8 != 0 ||
        weight->GetViewShape().GetDim(K_IDX) % FRACTAL_N_INT8 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight shape (N, K) must both align to 16, please check");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (weight->GetViewShape().GetDim(K_IDX) % FRACTAL_K_INT8 != 0) {
        auto padArray = exec->AllocIntArray(PAD_VEC_QGMMDQ, PAD_VEC_SIZE_QGMMDQ);
        auto padTensor = exec->ConvertToTensor(padArray, DataType::DT_INT64);
        auto valueTensor = exec->ConvertToTensor(exec->AllocScalar(0), DataType::DT_INT8);
        auto weight_ = exec->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
        weight_->SetStorageShape(weight->GetViewShape());
        weight = l0op::PadV3(weight_, padTensor, valueTensor, PAD_MODE_QGMMDQ, true, exec);
    }
    op::Shape weightNzShape = GetWeightNzShape(weight);
    auto weight_ = exec->CreateView(weight, weightNzShape, weight->GetViewOffset());
    auto perm = exec->AllocIntArray(PERM_VEC_QGMMDQ, PERM_VEC_SIZE_QGMMDQ);
    weight = l0op::Transpose(weight_, perm, exec);
    weight_ = exec->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
    weight_->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    weight = weight_;
    return ACLNN_SUCCESS;
}

static aclnnStatus PrepareWeightFormat(const aclTensor*& weight, aclOpExecutor* exec)
{
    uint64_t weightDimNum = weight->GetViewShape().GetDimNum();
    uint64_t weightStorageDimNum = weight->GetStorageShape().GetDimNum();
    if (weightDimNum == ND_DIMNUM &&
        (weight->GetStorageFormat() == op::Format::FORMAT_ND || weight->GetStorageFormat() == op::Format::FORMAT_NCL)) {
        return ConvertWeightNdToNz(weight, exec);
    } else if (weightStorageDimNum != NZ_DIMNUM || weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight is not in 5-dim (G, K//32, N//16, 16, 32), please check");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSizeCommon(
    const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* biasOptional, const aclTensor* xScaleOptional, const aclTensor* xOffsetOptional,
    const aclTensor* smoothScaleOptional, char* xQuantMode, bool transposeWeight, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (!transposeWeight) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto ret = CheckParamsNullOrNot(x, weight, weightScale, groupList, biasOptional, xOffsetOptional, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckParamsDateType(x, weight, weightScale, groupList, xScaleOptional, smoothScaleOptional, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || weight->IsEmpty() || weightScale->IsEmpty() || groupList->IsEmpty()) {
        *workspaceSize = static_cast<uint64_t>(0);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    auto weightScaleContiguous = l0op::Contiguous(weightScale, uniqueExecutor.get());
    auto groupListContiguous = l0op::Contiguous(groupList, uniqueExecutor.get());
    auto xScaleContiguous = xScaleOptional;
    if (xScaleOptional != nullptr) {
        xScaleContiguous = l0op::Contiguous(xScaleOptional, uniqueExecutor.get());
    }
    auto smoothScaleContiguous = smoothScaleOptional;
    if (smoothScaleContiguous != nullptr) {
        smoothScaleContiguous = l0op::Contiguous(smoothScaleOptional, uniqueExecutor.get());
    }
    if (!IsContiguous(weight) || !IsContiguous(out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight and out must be contiguous, please check");
        return ACLNN_ERR_PARAM_INVALID;
    }

    ret = PrepareWeightFormat(weight, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto out_ = uniqueExecutor.get()->CreateView(out, out->GetViewShape(), out->GetViewOffset());
    out_->SetStorageShape(out->GetViewShape());
    out = out_;

    auto matmulRet = l0op::QuantGroupedMatmulDequant(
        xContiguous, weight, weightScaleContiguous, groupListContiguous, biasOptional, xScaleContiguous,
        xOffsetOptional, smoothScaleContiguous, xQuantMode, transposeWeight, out, uniqueExecutor.get());

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* biasOptional, const aclTensor* xScaleOptional, const aclTensor* xOffsetOptional,
    const aclTensor* smoothScaleOptional, char* xQuantMode, bool transposeWeight, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnQuantGroupedMatmulDequant,
                   DFX_IN(x, weight, weightScale, groupList, biasOptional, xScaleOptional, xOffsetOptional,
                          smoothScaleOptional, xQuantMode, transposeWeight),
                   DFX_OUT(out));
    return aclnnQuantGroupedMatmulDequantGetWorkspaceSizeCommon(
        x, weight, weightScale, groupList, biasOptional, xScaleOptional, xOffsetOptional, smoothScaleOptional,
        xQuantMode, transposeWeight, out, workspaceSize, executor);
}

aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* biasOptional, const aclTensor* xScaleOptional, const aclTensor* xOffsetOptional,
    const aclTensor* smoothScaleOptional, char* xQuantMode, bool transposeWeight, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnQuantGroupedMatmulDequantWeightNZ,
                   DFX_IN(x, weight, weightScale, groupList, biasOptional, xScaleOptional, xOffsetOptional,
                          smoothScaleOptional, xQuantMode, transposeWeight),
                   DFX_OUT(out));
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto storageShape = weight->GetStorageShape();
    auto viewShape = weight->GetViewShape();
    aclTensor* weightNZ = const_cast<aclTensor*>(weight);
    CHECK_COND((storageShape.GetDimNum() == NZ_DIMNUM), ACLNN_ERR_PARAM_INVALID,
               "aclnnQuantGroupedMatmulDequantWeightNZ, The dimnum of storageShape for second input (weight)"
               "must be 5. \n But StorageShape got %s , and dimNum is %lu.",
               op::ToString(storageShape).GetString(), storageShape.GetDimNum());
    weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    if (viewShape.GetDimNum() == NZ_DIMNUM) {
        weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
    } else if (viewShape.GetDimNum() == ND_DIMNUM) {
        weightNZ->SetViewFormat(op::Format::FORMAT_ND);
    }
    return aclnnQuantGroupedMatmulDequantGetWorkspaceSizeCommon(
        x, weight, weightScale, groupList, biasOptional, xScaleOptional, xOffsetOptional, smoothScaleOptional,
        xQuantMode, transposeWeight, out, workspaceSize, executor);
}

aclnnStatus aclnnQuantGroupedMatmulDequant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantGroupedMatmulDequant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZ(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantGroupedMatmulDequantWeightNZ);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
