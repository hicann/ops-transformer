/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <string>
#include <unordered_map>
#include <acl/acl.h>
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"
#include "block_sparse_attention.h"
#include "aclnn_block_sparse_attention_v3.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr uint64_t LSE_OUT = 1;

// V3 新增:量化模式枚举
// 0=不量化, 1=FP8 量化, 2=mxfp4 量化
enum QuantMode : int64_t {
    NO_QUANT = 0,
    FP8_QUANT = 1,
    MXFP4_OCP_QUANT = 2,
    MXFP4_CX_QUANT = 3,
};

static bool CheckDataType(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    const DataType vDtype = value->GetDataType();

    // V3 新增支持 DT_FLOAT4_E2M1 (mxfp4) 数据类型
    static const std::unordered_map<DataType, std::vector<DataType>> validKvType = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
        {DataType::DT_FLOAT8_E4M3FN, {DataType::DT_FLOAT8_E4M3FN}},
        {DataType::DT_FLOAT4_E2M1, {DataType::DT_FLOAT4_E2M1}}};

    auto iter = validKvType.find(qDtype);
    if (iter == validKvType.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported query datatype %d.", static_cast<int>(qDtype));
        return false;
    }

    if (std::find(iter->second.begin(), iter->second.end(), kDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), vDtype) == iter->second.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key/Value datatype mismatch with query.");
        return false;
    }

    return true;
}

// V3 新增:校验单个 scale:必须非空且 dtype 匹配 expectedScaleDtype
static bool CheckScaleMatchDtype(const aclTensor *scale, DataType expectedScaleDtype)
{
    return scale != nullptr && scale->GetDataType() == expectedScaleDtype;
}

// quantMode=0 校验: QKV=FP16/BF16(相同) + scales=null
static bool CheckNoQuantParams(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                               const aclTensor *qScale, const aclTensor *kScale, const aclTensor *vScale)
{
    DataType qDtype = query->GetDataType();
    auto sameQkv = [&]() { return key->GetDataType() == qDtype && value->GetDataType() == qDtype; };
    auto scalesNull = [&]() { return qScale == nullptr && kScale == nullptr && vScale == nullptr; };
    if ((qDtype != DataType::DT_FLOAT16 && qDtype != DataType::DT_BF16) || !sameQkv() || !scalesNull()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode 0 requires QKV=FP16/BF16 (same) and null scales, got Q=%d.",
                static_cast<int>(qDtype));
        return false;
    }
    return true;
}

// quantMode=1/2 校验: QKV 同 expectedDtype + scales 非空且 dtype 同 expectedScaleDtype
static bool CheckQuantParams(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                             const aclTensor *qScale, const aclTensor *kScale, const aclTensor *vScale,
                             DataType expectedDtype, DataType expectedScaleDtype, int64_t quantMode,
                             const char *expectedDesc, const char *scaleDesc)
{
    DataType qDtype = query->GetDataType();
    if (qDtype != expectedDtype || key->GetDataType() != qDtype || value->GetDataType() != qDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode %ld requires QKV dtype to be %s (same), got Q=%d.", quantMode,
                expectedDesc, static_cast<int>(qDtype));
        return false;
    }
    if (!CheckScaleMatchDtype(qScale, expectedScaleDtype) || !CheckScaleMatchDtype(kScale, expectedScaleDtype) ||
        !CheckScaleMatchDtype(vScale, expectedScaleDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode %ld requires q/k/vDequantScale non-null with dtype %s.", quantMode,
                scaleDesc);
        return false;
    }
    return true;
}

// V3 新增:按 quantMode 分发校验 QKV dtype 及 scales
static bool CheckQuantModeAndDtype(int64_t quantMode, const aclTensor *query, const aclTensor *key,
                                   const aclTensor *value, const aclTensor *qScale, const aclTensor *kScale,
                                   const aclTensor *vScale)
{
    switch (quantMode) {
        case NO_QUANT:
            return CheckNoQuantParams(query, key, value, qScale, kScale, vScale);
        case FP8_QUANT: // quantMode=1: QKV=FP8_E4M3FN, scales=FP32
            return CheckQuantParams(query, key, value, qScale, kScale, vScale, DataType::DT_FLOAT8_E4M3FN,
                                    DataType::DT_FLOAT, quantMode, "FP8_E4M3FN", "FP32");
        case MXFP4_OCP_QUANT:
        case MXFP4_CX_QUANT: // quantMode=2/3: QKV=FP4_E2M1, scales=FP8_E8M0
            return CheckQuantParams(query, key, value, qScale, kScale, vScale, DataType::DT_FLOAT4_E2M1,
                                    DataType::DT_FLOAT8_E8M0, quantMode, "FP4_E2M1", "FP8_E8M0");
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid quantMode %ld, must be 0/1/2/3.", quantMode);
            return false;
    }
}

static aclnnStatus CheckMandatoryTensors(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseblockShapeOptional(const aclIntArray *blockShapeOptional)
{
    if (blockShapeOptional != nullptr) {
        uint64_t size = blockShapeOptional->Size();
        if (size != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional must contain two elements [x, y].");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const int64_t *data = blockShapeOptional->GetData();
        if (data == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional data is null.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (data[0] <= 0 || data[1] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional values must be positive, got [%ld, %ld].", data[0],
                    data[1]);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                  const aclTensor *attentionOut, const aclTensor *qDequantScaleOptional,
                                  const aclTensor *kDequantScaleOptional, const aclTensor *vDequantScaleOptional,
                                  char *qInputLayout, char *kvInputLayout, const aclIntArray *blockShapeOptional,
                                  int64_t quantMode)
{
    CHECK_RET(CheckMandatoryTensors(query, key, value) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckDataType(query, key, value)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // V3 新增:校验 quantMode 与 QKV dtype 及 scales 一致性
    if (!CheckQuantModeAndDtype(quantMode, query, key, value, qDequantScaleOptional, kDequantScaleOptional,
                                vDequantScaleOptional)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qInputLayout == nullptr || kvInputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input layout strings are null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string qLayout(qInputLayout);
    std::string kvLayout(kvInputLayout);

    // 验证Q layout
    if (qLayout != "TND" && qLayout != "BNSD" && qLayout != "BSND") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qInputLayout only supports TND, BNSD or BSND, got %s.", qLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证KV layout
    if (kvLayout != "TND" && kvLayout != "BNSD" && kvLayout != "BSND") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvInputLayout only supports TND, BNSD or BSND, got %s.", kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证Q和KV格式一致性：两者必须相同
    if (qLayout != kvLayout) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The parameters qInputLayout and kvInputLayout must be consistent, but currently qInputLayout is %s "
                "and kvInputLayout is %s.",
                qLayout.c_str(), kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ParseblockShapeOptional(blockShapeOptional);
}

// V3 新增:校验 mxfp4 scale 维度,TND=4维, BNSD/BSND=5维
static aclnnStatus CheckMxfp4ScaleDim(const aclTensor *qScale, const aclTensor *kScale, const aclTensor *vScale,
                                      const std::string &layout)
{
    size_t expectedDim = (layout == "TND") ? 4 : 5;
    auto checkOne = [&](const aclTensor *scale, const char *name) -> bool {
        if (scale->GetStorageShape().GetDimNum() != expectedDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxfp4 %s must be %zuD tensor for %s layout, got %zuD.", name, expectedDim,
                    layout.c_str(), scale->GetStorageShape().GetDimNum());
            return false;
        }
        return true;
    };
    if (!checkOne(qScale, "qDequantScaleOptional") || !checkOne(kScale, "kDequantScaleOptional") ||
        !checkOne(vScale, "vDequantScaleOptional")) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeContiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                  const aclTensor *&blockSparseMaskOptional, const aclTensor *&attenMaskOptional,
                                  const aclTensor *&blockTableOptional, const aclTensor *&qDequantScaleOptional,
                                  const aclTensor *&kDequantScaleOptional, const aclTensor *&vDequantScaleOptional,
                                  const aclTensor *&pQuantScaleOptional, int64_t quantMode, const std::string &qLayout,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // 新增blockSparseMaskOptional非空校验且必须为四维
    if (blockSparseMaskOptional != nullptr) {
        blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executor);
        CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (blockSparseMaskOptional->GetStorageShape().GetDimNum() != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockSparseMask must be 4D tensor.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (blockTableOptional != nullptr) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (quantMode == FP8_QUANT) {
        qDequantScaleOptional = l0op::Contiguous(qDequantScaleOptional, executor);
        CHECK_RET(qDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (qDequantScaleOptional->GetStorageShape().GetDimNum() != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qDequantScaleOptional must be 4D tensor.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        kDequantScaleOptional = l0op::Contiguous(kDequantScaleOptional, executor);
        CHECK_RET(kDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (kDequantScaleOptional->GetStorageShape().GetDimNum() != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kDequantScaleOptional must be 4D tensor.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        vDequantScaleOptional = l0op::Contiguous(vDequantScaleOptional, executor);
        CHECK_RET(vDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (vDequantScaleOptional->GetStorageShape().GetDimNum() != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "vDequantScaleOptional must be 4D tensor.");
            return ACLNN_ERR_PARAM_INVALID;
        }

    } else if (quantMode == MXFP4_OCP_QUANT || quantMode == MXFP4_CX_QUANT) {
        // mxfp4(OCP/CX): scale 维度按 layout 校验, TND=4维, BNSD/BSND=5维
        qDequantScaleOptional = l0op::Contiguous(qDequantScaleOptional, executor);
        CHECK_RET(qDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        kDequantScaleOptional = l0op::Contiguous(kDequantScaleOptional, executor);
        CHECK_RET(kDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        vDequantScaleOptional = l0op::Contiguous(vDequantScaleOptional, executor);
        CHECK_RET(vDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclnnStatus ret =
            CheckMxfp4ScaleDim(qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, qLayout);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }

    // pQuantScaleOptional: 当前所有 quantMode 下都必须为 nullptr
    if (pQuantScaleOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pQuantScaleOptional must be nullptr in current quantMode(%lld).", quantMode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateAdditionalParams(int64_t innerPrecise, int64_t quantMode, double dstTypeMax,
                                            const aclTensor *attentionOut, uint64_t *workspaceSize,
                                            aclOpExecutor **executor)
{
    if (innerPrecise != 0 && innerPrecise != 1 && innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "innerPrecise must be 0 or 1 or 4, got %ld.", innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 只有 quantMode=2/3 (mxfp4 OCP/CX) 需要 dstTypeMax,合法取值:0 或 [6.0, 12.0]
    if (quantMode == MXFP4_CX_QUANT) {
        bool valid = (dstTypeMax == 0.0) || (dstTypeMax >= 6.0 && dstTypeMax <= 12.0);
        if (!valid) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode 3 (mxfp4 CX) requires dstTypeMax = 0 or in [6, 12], got %f.",
                    dstTypeMax);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateMxfp4Constraints(int64_t quantMode, int64_t softmaxLseFlag, int64_t innerPrecise,
                                            const aclTensor *attenMaskOptional, const aclTensor *blockTableOptional)
{
    if (quantMode != MXFP4_OCP_QUANT && quantMode != MXFP4_CX_QUANT) {
        return ACLNN_SUCCESS;
    }
    // mxfp4 量化下 innerPrecise 必须为 4
    if (innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "innerPrecise must be 4 in mxfp4 quantMode(%lld), got %lld.", quantMode,
                innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mxfp4 量化场景下暂不支持 LSE、pageAttention; attenMask(blockEffRows) 在 mxfp4 下支持
    if (softmaxLseFlag != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxLseFlag must be 0 in mxfp4 quantMode(%lld), got %lld.", quantMode,
                softmaxLseFlag);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockTableOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockTableOptional must be nullptr in mxfp4 quantMode(%lld).", quantMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static string ConvertLayoutString(char *layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMask,
    const aclTensor *attenMaskOptional, const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional,
    const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, char *qInputLayout,
    char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise,
    int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode,
    double dstTypeMax, aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret = ValidateParams(query, key, value, attentionOut, qDequantScaleOptional, kDequantScaleOptional,
                                     vDequantScaleOptional, qInputLayout, kvInputLayout, blockShape, quantMode);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = ValidateAdditionalParams(innerPrecise, quantMode, dstTypeMax, attentionOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // mxfp4 量化限制检查
    ret = ValidateMxfp4Constraints(quantMode, softmaxLseFlag, innerPrecise, attenMaskOptional, blockTableOptional);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // 去掉了idx ,idxnums
    L2_DFX_PHASE_1(
        aclnnBlockSparseAttentionV3,
        DFX_IN(query, key, value, blockSparseMask, attenMaskOptional, blockShape, actualSeqLengthsOptional,
               actualSeqLengthsKvOptional, blockTableOptional, qDequantScaleOptional, kDequantScaleOptional,
               vDequantScaleOptional, pQuantScaleOptional, qInputLayout, kvInputLayout, numKeyValueHeads, maskType,
               scaleValue, innerPrecise, blockSize, preTokens, nextTokens, softmaxLseFlag, quantMode, dstTypeMax),
        DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();
    string qInputLayoutStr = ConvertLayoutString(qInputLayout);
    string kvInputLayoutStr = ConvertLayoutString(kvInputLayout);
    // 新增blockSparseMaskOptional参数
    ret = MakeContiguous(query, key, value, blockSparseMask, attenMaskOptional, blockTableOptional,
                         qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional,
                         quantMode, qInputLayoutStr, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto outputs = l0op::BlockSparseAttention(
        query, key, value, blockSparseMask, attenMaskOptional, blockShape, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, blockTableOptional, qDequantScaleOptional, kDequantScaleOptional,
        vDequantScaleOptional, pQuantScaleOptional, qInputLayoutStr.c_str(), kvInputLayoutStr.c_str(), numKeyValueHeads,
        maskType, scaleValue, innerPrecise, blockSize, preTokens, nextTokens, softmaxLseFlag, quantMode, dstTypeMax,
        attentionOut, executorImpl);
    if (outputs[0] == nullptr || outputs[1] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockSparseAttention returned nullptr outputs.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (softmaxLseFlag == LSE_OUT) {
        auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
        CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionV3(void *workspace, uint64_t workspaceSize,
                                                                               aclOpExecutor *executor,
                                                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockSparseAttentionV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
