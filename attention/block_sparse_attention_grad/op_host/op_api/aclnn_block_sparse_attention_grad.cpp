/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_sparse_attention_grad_v2.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "block_sparse_attention_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <utility>
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct BSAGParams {
    const aclTensor* query = nullptr;
    const aclTensor* key = nullptr;
    const aclTensor* value = nullptr;
    const aclTensor* attentionOut = nullptr;
    const aclTensor* attentionOutGrad = nullptr;
    const aclTensor* softmaxLse = nullptr;
    const aclTensor* blockSparseMaskOptional = nullptr;
    const aclTensor* attenMaskOptional = nullptr;
    const aclTensor* qDequantScaleOptional = nullptr;
    const aclTensor* kDequantScaleOptional = nullptr;
    const aclTensor* vDequantScaleOptional = nullptr;
    const aclTensor* doDequantScaleOptional = nullptr;
    const aclTensor* pQuantScaleOptional = nullptr;
    const aclTensor* dsQuantScaleOptional = nullptr;
    const aclIntArray* actualSeqLengthsOptional = nullptr;
    const aclIntArray* actualSeqLengthsKvOptional = nullptr;
    char* qInputLayout;
    char* kvInputLayout;
    const aclIntArray* blockShapeOptional;
    int64_t numKeyValueHeads;
    int64_t maskType;
    int64_t preTokens;
    int64_t nextTokens;
    const aclTensor* dqOut = nullptr;
    const aclTensor* dkOut = nullptr;
    const aclTensor* dvOut = nullptr;
};

static aclnnStatus CheckMandatoryTensors(const aclTensor* dout, const aclTensor* query, const aclTensor* key,
                                         const aclTensor* value, const aclTensor* attentionOut,
                                         const aclTensor* softmaxLse, const aclTensor* blockSparseMaskOptional,
                                         const aclTensor* dqOut, const aclTensor* dkOut, const aclTensor* dvOut)
{
    CHECK_RET(dout != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseBlockShape(const aclIntArray* blockShapeOptional)
{
    if (blockShapeOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "blockShapeOptional is null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    uint64_t size = blockShapeOptional->Size();
    if (size < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional must contain at least two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t* data = blockShapeOptional->GetData();
    if (data == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional data is null.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (data[0] <= 0 || data[1] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional values must be positive, got [%ld, %ld].", data[0],
                data[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    constexpr int64_t blockAlignment = 16;
    if ((data[0] % blockAlignment != 0 || data[1] % blockAlignment != 0) &&
        op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Ascend 950 requires blockShapeOptional values to be multiples of 16, got [%ld, %ld].", data[0],
                data[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const BSAGParams& params)
{
    CHECK_COND(params.qInputLayout != nullptr && params.kvInputLayout != nullptr, ACLNN_ERR_PARAM_INVALID,
               "qInputLayout or kvInputLayout is nullptr.");
    std::string qLayout(params.qInputLayout);
    std::string kvLayout(params.kvInputLayout);
    CHECK_RET(CheckMandatoryTensors(params.attentionOutGrad, params.query, params.key, params.value,
                                    params.attentionOut, params.softmaxLse, params.blockSparseMaskOptional,
                                    params.dqOut, params.dkOut, params.dvOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_COND(qLayout == kvLayout, ACLNN_ERR_PARAM_INVALID, "qInputLayout and kvInputLayout must be the same.");
    CHECK_COND(qLayout == "BNSD" || qLayout == "TND" || qLayout == "BSND", ACLNN_ERR_PARAM_INVALID,
               "qInputLayout and kvInputLayout must be BNSD, TND, or BSND.");
    CHECK_COND(qLayout != "BSND" || op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510,
               ACLNN_ERR_PARAM_INVALID, "BSND is supported only on Ascend 950. Use BNSD or TND on Atlas A2/A3.");
    CHECK_COND(params.dqOut != nullptr && params.dkOut != nullptr && params.dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "dq/dk/dv cannot be nullptr.");
    // dtype检查
    DataType qDtype = params.query->GetDataType();
    DataType kDtype = params.key->GetDataType();
    DataType vDtype = params.value->GetDataType();
    DataType attnOutDtype = params.attentionOut->GetDataType();
    DataType gradDtype = params.attentionOutGrad->GetDataType(); // dout
    DataType lseDtype = params.softmaxLse->GetDataType();
    DataType maskDtype = params.blockSparseMaskOptional->GetDataType();
    DataType dqDtype = params.dqOut->GetDataType();
    DataType dkDtype = params.dkOut->GetDataType();
    DataType dvDtype = params.dvOut->GetDataType();
    const bool isHif8 = qDtype == DataType::DT_HIFLOAT8;
    CHECK_COND(!isHif8 || op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510, ACLNN_ERR_PARAM_INVALID,
               "HIFLOAT8 is supported only on Ascend 950.");
    CHECK_COND(qDtype == ACL_FLOAT16 || qDtype == ACL_BF16 || isHif8, ACLNN_ERR_PARAM_INVALID,
               "The dtype of query is not support.");
    CHECK_COND(qDtype == kDtype, ACLNN_ERR_PARAM_INVALID, "key dtype error.");
    CHECK_COND(qDtype == vDtype, ACLNN_ERR_PARAM_INVALID, "value dtype error.");
    const DataType expectedOutputType = isHif8 ? DataType::DT_BF16 : qDtype;
    CHECK_COND(attnOutDtype == expectedOutputType, ACLNN_ERR_PARAM_INVALID,
               "attentionOut must be BF16 for HiF8 input, otherwise match query dtype.");
    CHECK_COND(gradDtype == qDtype, ACLNN_ERR_PARAM_INVALID, "attentionOutGrad dtype must match query dtype.");
    CHECK_COND(lseDtype == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID, "softmaxLse dtype must be FLOAT32.");
    CHECK_COND(maskDtype == ACL_BOOL || maskDtype == ACL_UINT8, ACLNN_ERR_PARAM_INVALID,
               "blockSparseMask dtype must be BOOL or UINT8.");
    CHECK_COND(dqDtype == expectedOutputType, ACLNN_ERR_PARAM_INVALID, "dqOut dtype is invalid.");
    CHECK_COND(dkDtype == expectedOutputType, ACLNN_ERR_PARAM_INVALID, "dkOut dtype is invalid.");
    CHECK_COND(dvDtype == expectedOutputType, ACLNN_ERR_PARAM_INVALID, "dvOut dtype is invalid.");
    const std::array<std::pair<const aclTensor*, const char*>, 6> scales = {
        {{params.qDequantScaleOptional, "qDequantScaleOptional"},
         {params.kDequantScaleOptional, "kDequantScaleOptional"},
         {params.vDequantScaleOptional, "vDequantScaleOptional"},
         {params.doDequantScaleOptional, "doDequantScaleOptional"},
         {params.pQuantScaleOptional, "pQuantScaleOptional"},
         {params.dsQuantScaleOptional, "dsQuantScaleOptional"}}};
    for (const auto& scale : scales) {
        if (!isHif8) {
            CHECK_COND(scale.first == nullptr, ACLNN_ERR_PARAM_INVALID,
                       "%s must be nullptr for FLOAT16 or BFLOAT16 inputs.", scale.second);
            continue;
        }
        CHECK_COND(scale.first != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s cannot be nullptr for HIFLOAT8 inputs.",
                   scale.second);
        CHECK_COND(scale.first->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "%s must be FLOAT32.",
                   scale.second);
        const auto& shape = scale.first->GetViewShape();
        CHECK_COND(shape.GetDimNum() == 1 && shape.GetDim(0) == 1, ACLNN_ERR_PARAM_INVALID, "%s must have shape [1].",
                   scale.second);
    }
    // format检查
    if (params.query->GetStorageFormat() != ge::FORMAT_ND || params.key->GetStorageFormat() != ge::FORMAT_ND ||
        params.value->GetStorageFormat() != ge::FORMAT_ND || params.attentionOut->GetStorageFormat() != ge::FORMAT_ND ||
        params.attentionOutGrad->GetStorageFormat() != ge::FORMAT_ND ||
        params.softmaxLse->GetStorageFormat() != ge::FORMAT_ND ||
        params.blockSparseMaskOptional->GetStorageFormat() != ge::FORMAT_ND ||
        params.dqOut->GetStorageFormat() != ge::FORMAT_ND || params.dkOut->GetStorageFormat() != ge::FORMAT_ND ||
        params.dvOut->GetStorageFormat() != ge::FORMAT_ND) {
        OP_LOGW("Format of input is not ND, this format may lead to precision failure.");
    }
    // shape检查
    auto queryShape = params.query->GetViewShape();
    auto keyShape = params.key->GetViewShape();
    auto valueShape = params.value->GetViewShape();
    auto attnOutShape = params.attentionOut->GetViewShape();
    auto gradShape = params.attentionOutGrad->GetViewShape();
    auto lseShape = params.softmaxLse->GetViewShape();
    auto dqShape = params.dqOut->GetViewShape();
    auto dkShape = params.dkOut->GetViewShape();
    auto dvShape = params.dvOut->GetViewShape();
    if (isHif8) {
        CHECK_COND(qLayout == kvLayout && (qLayout == "BNSD" || qLayout == "BSND" || qLayout == "TND"),
                   ACLNN_ERR_PARAM_INVALID, "HiF8 requires matching BNSD/BSND/TND layouts.");
        const size_t rank = qLayout == "TND" ? 3 : 4;
        CHECK_COND(queryShape.GetDimNum() == rank && keyShape.GetDimNum() == rank && valueShape.GetDimNum() == rank &&
                       attnOutShape.GetDimNum() == rank && gradShape.GetDimNum() == rank &&
                       dqShape.GetDimNum() == rank && dkShape.GetDimNum() == rank && dvShape.GetDimNum() == rank,
                   ACLNN_ERR_PARAM_INVALID, "HiF8 tensor rank must match the input layout.");
        CHECK_COND(lseShape.GetDimNum() == rank - 1 || (lseShape.GetDimNum() == rank && lseShape.GetDim(rank - 1) == 1),
                   ACLNN_ERR_PARAM_INVALID, "LSE must match the forward layout, with an optional trailing 1.");
        for (size_t axis = 0; axis < rank - 1; ++axis) {
            CHECK_COND(lseShape.GetDim(axis) == queryShape.GetDim(axis), ACLNN_ERR_PARAM_INVALID,
                       "LSE axes must match the forward query axes.");
        }
    }
    if (qLayout == "TND") {
        // check head_dim
        CHECK_COND(queryShape.GetDim(2) == keyShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and key must be same.");
        CHECK_COND(queryShape.GetDim(2) == valueShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and value must be same.");
        CHECK_COND(queryShape.GetDim(2) == attnOutShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and attentionOut must be same.");
        CHECK_COND(queryShape.GetDim(2) == gradShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and attentionOutGrad must be same.");
        CHECK_COND(queryShape.GetDim(2) == dqShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and dqOut must be same.");
        CHECK_COND(queryShape.GetDim(2) == dkShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and dkOut must be same.");
        CHECK_COND(queryShape.GetDim(2) == dvShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 2 of query and dvOut must be same.");
    } else {
        // check batch_num, head_dim
        CHECK_COND(queryShape.GetDim(0) == keyShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and key must be same.");
        CHECK_COND(queryShape.GetDim(0) == valueShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and value must be same.");
        CHECK_COND(queryShape.GetDim(0) == attnOutShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and attentionOut must be same.");
        CHECK_COND(queryShape.GetDim(0) == gradShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and attentionOutGrad must be same.");
        CHECK_COND(queryShape.GetDim(0) == lseShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and softmaxLse must be same.");
        CHECK_COND(queryShape.GetDim(0) == dqShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and dqOut must be same.");
        CHECK_COND(queryShape.GetDim(0) == dkShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and dkOut must be same.");
        CHECK_COND(queryShape.GetDim(0) == dvShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 0 of query and dvOut must be same.");
        CHECK_COND(queryShape.GetDim(3) == keyShape.GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 3 of query and key must be same.");
        CHECK_COND(valueShape.GetDim(3) == attnOutShape.GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 3 of value and attentionOut must be same.");
        CHECK_COND(valueShape.GetDim(3) == gradShape.GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 3 of value and attentionOutGrad must be same.");
        CHECK_COND(queryShape.GetDim(3) == dqShape.GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 3 of query and dqOut must be same.");
        CHECK_COND(keyShape.GetDim(3) == dkShape.GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 3 of key and dkOut must be same.");
        CHECK_COND(valueShape.GetDim(3) == dvShape.GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "The dim's 3 of value and dvOut must be same.");
    }

    if (params.maskType == 0) {
        CHECK_COND(params.attenMaskOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "attenMaskOptional must be nullptr when maskType=0.");
    } else if (params.maskType == 1) {
        CHECK_COND(op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510, ACLNN_ERR_PARAM_INVALID,
                   "maskType=1 are currently supported only on A2 or A3.");
        CHECK_COND(params.attenMaskOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "maskType=1 requires attenMaskOptional.");
        CHECK_COND(params.attenMaskOptional->GetDataType() == ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "attenMask must have INT32 dtype.");
        const auto maskShape = params.attenMaskOptional->GetViewShape();
        CHECK_COND(maskShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID, "attenMask must have shape [B,Nq,M,2].");
        CHECK_COND(maskShape.GetDim(3) == 2, ACLNN_ERR_PARAM_INVALID, "attenMask last dimension must be 2.");
        CHECK_COND(maskShape.GetDim(1) == queryShape.GetDim(1), ACLNN_ERR_PARAM_INVALID,
                   "attenMask head dimension must equal query heads.");
        if (qLayout == "BNSD") {
            CHECK_COND(maskShape.GetDim(0) == queryShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                       "attenMask batch dimension must equal query batch.");
        } else {
            CHECK_COND(params.actualSeqLengthsOptional != nullptr &&
                           maskShape.GetDim(0) == static_cast<int64_t>(params.actualSeqLengthsOptional->Size()),
                       ACLNN_ERR_PARAM_INVALID, "attenMask batch must match TND sequence lengths.");
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType only supports 0 or 1, got %ld.", params.maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.preTokens != 2147483647 || params.nextTokens != 2147483647) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "preTokens and nextTokens must be 2147483647, got [%ld,%ld].",
                params.preTokens, params.nextTokens);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qLayout == "TND" && params.actualSeqLengthsOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, " actualSeqLengthsOptional  is mandatory when qInputLayout is TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kvLayout == "TND" && params.actualSeqLengthsKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, " actualSeqLengthsKvOptional  is mandatory when kvInputLayout is TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ParseBlockShape(params.blockShapeOptional);
}

static aclnnStatus MakeContiguous(const aclTensor*& dout, const aclTensor*& query, const aclTensor*& key,
                                  const aclTensor*& value, const aclTensor*& attentionOut, const aclTensor*& softmaxLse,
                                  const aclTensor*& blockSparseMaskOptional, const aclTensor*& attenMaskOptional,
                                  aclOpExecutor* executor)
{
    dout = l0op::Contiguous(dout, executor);
    CHECK_RET(dout != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    attentionOut = l0op::Contiguous(attentionOut, executor);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    softmaxLse = l0op::Contiguous(softmaxLse, executor);
    CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executor);
    CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static string ConvertLayoutString(char* layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

} // namespace

static aclnnStatus BlockSparseAttentionGradGetWorkspaceSizeImpl(
    const aclTensor* dout, const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* attentionOut, const aclTensor* softmaxLse, const aclTensor* blockSparseMaskOptional,
    const aclTensor* attenMaskOptional, const aclIntArray* blockShapeOptional,
    const aclIntArray* actualSeqLengthsOptional, const aclIntArray* actualSeqLengthsKvOptional,
    const aclTensor* qDequantScaleOptional, const aclTensor* kDequantScaleOptional,
    const aclTensor* vDequantScaleOptional, const aclTensor* doDequantScaleOptional,
    const aclTensor* pQuantScaleOptional, const aclTensor* dsQuantScaleOptional, char* qInputLayout,
    char* kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t preTokens,
    int64_t nextTokens, int64_t quantMode, aclTensor* dq, aclTensor* dk, aclTensor* dv, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    CHECK_COND(quantMode == 0 || quantMode == 5, ACLNN_ERR_PARAM_INVALID,
               "quantMode only supports 0 (no quantization) or 5 (HiF8 per-tensor), got %ld.", quantMode);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    const bool modeIsHif8 = query->GetDataType() == DataType::DT_HIFLOAT8;
    CHECK_COND((quantMode == 5 && modeIsHif8) || (quantMode == 0 && (query->GetDataType() == DataType::DT_FLOAT16 ||
                                                                     query->GetDataType() == DataType::DT_BF16)),
               ACLNN_ERR_PARAM_INVALID, "quantMode must match query dtype: 0=FP16/BF16, 5=HIFLOAT8.");
    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    BSAGParams bsag_params;
    bsag_params.query = query;
    bsag_params.key = key;
    bsag_params.value = value;
    bsag_params.attentionOut = attentionOut;
    bsag_params.attentionOutGrad = dout;
    bsag_params.softmaxLse = softmaxLse;
    bsag_params.blockSparseMaskOptional = blockSparseMaskOptional;
    bsag_params.attenMaskOptional = attenMaskOptional;
    bsag_params.qDequantScaleOptional = qDequantScaleOptional;
    bsag_params.kDequantScaleOptional = kDequantScaleOptional;
    bsag_params.vDequantScaleOptional = vDequantScaleOptional;
    bsag_params.doDequantScaleOptional = doDequantScaleOptional;
    bsag_params.pQuantScaleOptional = pQuantScaleOptional;
    bsag_params.dsQuantScaleOptional = dsQuantScaleOptional;
    bsag_params.actualSeqLengthsOptional = actualSeqLengthsOptional;
    bsag_params.actualSeqLengthsKvOptional = actualSeqLengthsKvOptional;
    bsag_params.qInputLayout = qInputLayout;
    bsag_params.kvInputLayout = kvInputLayout;
    bsag_params.blockShapeOptional = blockShapeOptional;
    bsag_params.numKeyValueHeads = numKeyValueHeads;
    bsag_params.maskType = maskType;
    bsag_params.preTokens = preTokens;
    bsag_params.nextTokens = nextTokens;
    bsag_params.dqOut = dq;
    bsag_params.dkOut = dk;
    bsag_params.dvOut = dv;
    aclnnStatus ret = ValidateParams(bsag_params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto* executorImpl = uniqueExecutor.get();

    if (blockSparseMaskOptional->GetDataType() != DataType::DT_UINT8) {
        blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executorImpl);
        CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        blockSparseMaskOptional = l0op::Cast(blockSparseMaskOptional, DataType::DT_UINT8, executorImpl);
        CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const bool isHif8 = query->GetDataType() == DataType::DT_HIFLOAT8;
    // HiF8 kernels consume BSND LSE in native BSN[1] order.
    // The existing contiguous conversion below handles arbitrary input strides.
    if (isHif8) {
        qDequantScaleOptional = l0op::Contiguous(qDequantScaleOptional, executorImpl);
        CHECK_RET(qDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        kDequantScaleOptional = l0op::Contiguous(kDequantScaleOptional, executorImpl);
        CHECK_RET(kDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        vDequantScaleOptional = l0op::Contiguous(vDequantScaleOptional, executorImpl);
        CHECK_RET(vDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        doDequantScaleOptional = l0op::Contiguous(doDequantScaleOptional, executorImpl);
        CHECK_RET(doDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        pQuantScaleOptional = l0op::Contiguous(pQuantScaleOptional, executorImpl);
        CHECK_RET(pQuantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dsQuantScaleOptional = l0op::Contiguous(dsQuantScaleOptional, executorImpl);
        CHECK_RET(dsQuantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (blockSparseMaskOptional != nullptr) {
            int64_t valuePerm[4] = {0, 1, 3, 2};
            auto perm = executorImpl->AllocIntArray(valuePerm, 4);
            blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executorImpl);
            const bool nativeSmallMask = isHif8 && (*blockShapeOptional)[0] == 128 && (*blockShapeOptional)[1] == 128 &&
                                         blockSparseMaskOptional->GetViewShape().GetShapeSize() <= 32768;
            if (!nativeSmallMask)
                blockSparseMaskOptional = l0op::Transpose(blockSparseMaskOptional, perm, executorImpl);
            CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    ret = MakeContiguous(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional,
                         executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    string qInputLayoutStr = ConvertLayoutString(qInputLayout);
    string kvInputLayoutStr = ConvertLayoutString(kvInputLayout);

    // Materialize long single-sequence HiF8 inputs in head-major order.
    // Linear reorders are included in the executor and in measured latency.
    const bool singleTnd = qInputLayoutStr == "TND" && actualSeqLengthsOptional != nullptr &&
                           actualSeqLengthsKvOptional != nullptr && actualSeqLengthsOptional->Size() == 1 &&
                           actualSeqLengthsKvOptional->Size() == 1 &&
                           (*actualSeqLengthsOptional)[0] == query->GetViewShape().GetDim(0) &&
                           (*actualSeqLengthsKvOptional)[0] == key->GetViewShape().GetDim(0);
    const bool singleBsnd = qInputLayoutStr == "BSND" && query->GetViewShape().GetDim(0) == 1;
    const int64_t seqLen = query->GetViewShape().GetDim(singleTnd ? 0 : 1);
    // Below this range, measured transpose costs offset the strided-access savings.
    const bool reorderHif8 = isHif8 && (singleTnd || singleBsnd) && seqLen >= 100000;
    auto transposeMaterialized = [&](const aclTensor* x, const int64_t* perm, size_t rank) -> const aclTensor* {
        auto axes = executorImpl->AllocIntArray(perm, rank);
        if (axes == nullptr)
            return nullptr;
        auto y = l0op::Transpose(x, axes, executorImpl);
        if (y == nullptr)
            return nullptr;
        return l0op::Contiguous(y, executorImpl);
    };
    if (reorderHif8) {
        auto toHeadMajor = [&](const aclTensor* x) -> const aclTensor* {
            if (singleTnd) {
                op::Shape shape;
                shape.AppendDim(1);
                for (size_t i = 0; i < 3; ++i)
                    shape.AppendDim(x->GetViewShape().GetDim(i));
                x = l0op::Reshape(x, shape, executorImpl);
                if (x == nullptr)
                    return nullptr;
            }
            const int64_t perm[] = {0, 2, 1, 3};
            return transposeMaterialized(x, perm, 4);
        };
        dout = toHeadMajor(dout);
        query = toHeadMajor(query);
        key = toHeadMajor(key);
        value = toHeadMajor(value);
        attentionOut = toHeadMajor(attentionOut);
        CHECK_RET(dout && query && key && value && attentionOut, ACLNN_ERR_INNER_NULLPTR);
        // Normalize optional trailing unit dimension before the BSN -> BNS reorder.
        op::Shape lseShape;
        lseShape.AppendDim(1);
        lseShape.AppendDim(seqLen);
        lseShape.AppendDim(query->GetViewShape().GetDim(1));
        softmaxLse = l0op::Reshape(softmaxLse, lseShape, executorImpl);
        CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t lsePerm[] = {0, 2, 1};
        softmaxLse = transposeMaterialized(softmaxLse, lsePerm, 3);
        CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_INNER_NULLPTR);
        qInputLayoutStr = "BNSD";
        kvInputLayoutStr = "BNSD";
    }

    auto outputs = l0op::BlockSparseAttentionGrad(
        dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional,
        blockShapeOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional, qDequantScaleOptional,
        kDequantScaleOptional, vDequantScaleOptional, doDequantScaleOptional, pQuantScaleOptional, dsQuantScaleOptional,
        const_cast<char*>(qInputLayoutStr.c_str()), const_cast<char*>(kvInputLayoutStr.c_str()), numKeyValueHeads,
        maskType, scaleValue, preTokens, nextTokens, quantMode, executorImpl);
    if (outputs[0] == nullptr || outputs[1] == nullptr || outputs[2] == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "BlockSparseAttentionGrad returned nullptr outputs.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    if (reorderHif8) {
        const aclTensor* destinations[] = {dq, dk, dv};
        const int64_t perm[] = {0, 2, 1, 3};
        for (size_t i = 0; i < 3; ++i) {
            outputs[i] = transposeMaterialized(outputs[i], perm, 4);
            CHECK_RET(outputs[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
            outputs[i] = l0op::Reshape(outputs[i], destinations[i]->GetViewShape(), executorImpl);
            CHECK_RET(outputs[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    auto viewCopyResult0 = l0op::ViewCopy(outputs[0], dq, executorImpl);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(outputs[1], dk, executorImpl);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult2 = l0op::ViewCopy(outputs[2], dv, executorImpl);
    CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor* dout, const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* attentionOut, const aclTensor* softmaxLse, const aclTensor* blockSparseMaskOptional,
    const aclTensor* attenMaskOptional, const aclIntArray* blockShapeOptional,
    const aclIntArray* actualSeqLengthsOptional, const aclIntArray* actualSeqLengthsKvOptional, char* qInputLayout,
    char* kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t preTokens,
    int64_t nextTokens, aclTensor* dq, aclTensor* dk, aclTensor* dv, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBlockSparseAttentionGrad,
                   DFX_IN(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional,
                          blockShapeOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional, qInputLayout,
                          kvInputLayout, numKeyValueHeads, maskType, scaleValue, preTokens, nextTokens),
                   DFX_OUT(dq, dk, dv));
    // Keep the pre-quantization ABI. HiF8 and all scale parameters belong to V2.
    return BlockSparseAttentionGradGetWorkspaceSizeImpl(
        dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional,
        blockShapeOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue, preTokens, nextTokens, 0,
        dq, dk, dv, workspaceSize, executor);
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
    const aclTensor* dout, const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* attentionOut, const aclTensor* softmaxLse, const aclTensor* blockSparseMaskOptional,
    const aclTensor* attenMaskOptional, const aclIntArray* blockShapeOptional,
    const aclIntArray* actualSeqLengthsOptional, const aclIntArray* actualSeqLengthsKvOptional,
    const aclTensor* qDequantScaleOptional, const aclTensor* kDequantScaleOptional,
    const aclTensor* vDequantScaleOptional, const aclTensor* doDequantScaleOptional,
    const aclTensor* pQuantScaleOptional, const aclTensor* dsQuantScaleOptional, char* qInputLayout,
    char* kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t preTokens,
    int64_t nextTokens, int64_t quantMode, aclTensor* dq, aclTensor* dk, aclTensor* dv, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBlockSparseAttentionGradV2,
                   DFX_IN(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional,
                          blockShapeOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                          qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, doDequantScaleOptional,
                          pQuantScaleOptional, dsQuantScaleOptional, qInputLayout, kvInputLayout, numKeyValueHeads,
                          maskType, scaleValue, preTokens, nextTokens, quantMode),
                   DFX_OUT(dq, dk, dv));
    return BlockSparseAttentionGradGetWorkspaceSizeImpl(
        dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, attenMaskOptional,
        blockShapeOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional, qDequantScaleOptional,
        kDequantScaleOptional, vDequantScaleOptional, doDequantScaleOptional, pQuantScaleOptional, dsQuantScaleOptional,
        qInputLayout, kvInputLayout, numKeyValueHeads, maskType, scaleValue, preTokens, nextTokens, quantMode, dq, dk,
        dv, workspaceSize, executor);
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGrad(void* workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor* executor,
                                                                                 const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockSparseAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradV2(void* workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor* executor,
                                                                                   const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockSparseAttentionGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
