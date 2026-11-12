/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <limits>

#include "aclnn_generic_block_sparse_attention_metadata.h"
#include "generic_block_sparse_attention_metadata.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../../generic_block_sparse_attention_metadata_tiling.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// 参数校验实现自 generic_block_sparse_attention_metadata_check.h 迁入:
// 校验仅被本编译单元使用,且头文件层无 ErrorManager 报错入口,
// 与仓库 aclnn 惯例(校验内联于 cpp 匿名命名空间)保持一致。

constexpr int64_t GBSA_METADATA_DIM_NUM = 1;
constexpr int64_t GBSA_SEQ_LENGTH_DIM_NUM = 1;
constexpr int64_t GBSA_TND_SPARSE_BLOCK_IDX_DIM_NUM = 3;
constexpr int64_t GBSA_TND_SPARSE_BLOCK_COUNT_DIM_NUM = 2;
constexpr int64_t GBSA_TND_BLOCK_CAPACITY_DIM_INDEX = 2;

constexpr int64_t GBSA_CURRENT_HEAD_DIM = 128;
constexpr int64_t GBSA_ARCH35_NONQUANT_MAX_HEAD_DIM = 512;
constexpr int64_t GBSA_HEAD_DIM_192 = 192;
constexpr int64_t GBSA_HEAD_DIM_256 = 256;
constexpr int64_t GBSA_CURRENT_MAX_GROUP_SIZE = 128;
constexpr int64_t GBSA_CURRENT_BLOCK_SHAPE_X = 1;
constexpr int64_t GBSA_CURRENT_BLOCK_SHAPE_Y = 128;
constexpr int64_t GBSA_MAX_BLOCK_SHAPE_Y = 128;
constexpr int64_t GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB = 4;
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_NONE = 0;
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_FORCE = 1;
constexpr int64_t GBSA_ARCH35_QUANT_MAX_SPARSE_BLOCK_COUNT = 256;
constexpr int64_t GBSA_QUANT_TYPE_NONE = 0;
constexpr int64_t GBSA_QUANT_TYPE_FLOAT8 = 5;
constexpr int64_t GBSA_QUANT_TYPE_MXFP8 = 6;
constexpr int64_t GBSA_SOFTMAX_PRECISION_HIGH = 0;
constexpr int64_t GBSA_SOFTMAX_PRECISION_MIXED = 1;
constexpr int64_t GBSA_CURRENT_MASK_TYPE = 1;
constexpr int64_t GBSA_CURRENT_WINDOW_SIZE = -1;
constexpr int64_t GBSA_DEFAULT_MAX_SEQ_LEN = -1;

bool GbsaTensorValid(const aclTensor* tensor)
{
    return tensor != nullptr && tensor->GetViewShape().GetDimNum() > 0;
}

bool GbsaIsLayout(const char* layout, const char* expected)
{
    return layout != nullptr && std::strcmp(layout, expected) == 0;
}

aclnnStatus CheckGbsaTensorDataType(const aclTensor* tensor, aclDataType expectedDataType, const char* expectedTypeName,
                                    const char* tensorName)
{
    aclDataType dataType = ACL_DT_UNDEFINED;
    aclGetDataType(tensor, &dataType);
    if (dataType != expectedDataType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be %s.", tensorName, expectedTypeName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaTensorDim(const aclTensor* tensor, int64_t expectedDim, const char* tensorName)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dimension must be %ld.", tensorName, expectedDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaRequiredInt32Tensor(const aclTensor* tensor, const char* tensorName)
{
    if (!GbsaTensorValid(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be provided.", tensorName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckGbsaTensorDataType(tensor, ACL_INT32, "INT32", tensorName);
}

aclnnStatus CheckGbsaRequiredInputs(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                    const aclTensor* metadata)
{
    aclnnStatus status = CheckGbsaRequiredInt32Tensor(sparseBlockIdx, "sparseBlockIdx");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaRequiredInt32Tensor(sparseBlockCount, "SparseBlockCount");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaRequiredInt32Tensor(metadata, "metadata");
}

aclnnStatus CheckGbsaMetadataShape(const aclTensor* metadata)
{
    aclnnStatus status = CheckGbsaTensorDim(metadata, GBSA_METADATA_DIM_NUM, "metadata");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (metadata->GetViewShape().GetDim(0) !=
        static_cast<int64_t>(optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata shape must be [%u].",
                optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaPositiveScalarAttr(int64_t attrValue, const char* attrName)
{
    if (attrValue <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be greater than 0, but got %lld.", attrName, attrValue);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaHeadAttrs(int64_t numQHeads, int64_t numKvHeads)
{
    if (numQHeads < numKvHeads || numQHeads % numKvHeads != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "numQHeads must be greater than or equal to and divisible by numKvHeads, but got numQHeads=%lld "
                "and numKvHeads=%lld.",
                numQHeads, numKvHeads);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t groupSize = numQHeads / numKvHeads;
    if (groupSize > GBSA_CURRENT_MAX_GROUP_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupSize must be less than or equal to %lld, but got %lld.",
                GBSA_CURRENT_MAX_GROUP_SIZE, groupSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaBlockAttrs(int64_t blockShapeX, int64_t blockShapeY, int64_t layoutSparsePattern,
                                bool arch35NonQuant = false, bool arch22GeneralizedY = false,
                                bool arch22PagedGeneralizedY = false, bool mxfp8 = false)
{
    if (blockShapeX != GBSA_CURRENT_BLOCK_SHAPE_X) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeX currently only supports %lld, but got %lld.",
                GBSA_CURRENT_BLOCK_SHAPE_X, blockShapeX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (mxfp8 && blockShapeY != 4 && blockShapeY != 8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MXFP8 blockShapeY only supports 4 or 8, but got %lld.", blockShapeY);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isPowerOfTwoY = blockShapeY > 0 && (blockShapeY & (blockShapeY - 1)) == 0;
    const bool supportedBlockY = mxfp8              ? true :
                                 arch35NonQuant     ? (blockShapeY >= 1 && blockShapeY <= GBSA_CURRENT_BLOCK_SHAPE_Y) :
                                 arch22GeneralizedY ? (blockShapeY <= GBSA_MAX_BLOCK_SHAPE_Y && isPowerOfTwoY) :
                                 arch22PagedGeneralizedY ? (blockShapeY <= GBSA_MAX_BLOCK_SHAPE_Y && isPowerOfTwoY) :
                                                           (blockShapeY == GBSA_CURRENT_BLOCK_SHAPE_Y);
    if (!supportedBlockY) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "blockShapeY=%lld is unsupported (arch35NonQuant=%d, arch22TndGeneralizedY=%d, "
                "arch22PaGeneralizedY=%d); supported: 128, any in [1, 128] for arch35 non-quant, "
                "powers of two in [1, %lld] for arch22 dense or PA.",
                blockShapeY, arch35NonQuant, arch22GeneralizedY, arch22PagedGeneralizedY, GBSA_MAX_BLOCK_SHAPE_Y);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutSparsePattern != GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout_sparse_pattern currently only supports %lld (KVN_TotalQB_KB), but got %lld.",
                GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB, layoutSparsePattern);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaScalarAttrs(int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t blockShapeX,
                                 int64_t blockShapeY, int64_t layoutSparsePattern, bool isArch35, bool arch35NonQuant,
                                 bool arch22GeneralizedY = false, bool arch22PagedGeneralizedY = false,
                                 bool mxfp8 = false)
{
    const struct {
        int64_t value;
        const char* name;
    } positiveAttrs[] = {
        {numQHeads, "numQHeads"}, {numKvHeads, "numKvHeads"}, {headDim, "headDim"}, {blockShapeY, "blockShapeY"}};
    for (const auto& attr : positiveAttrs) {
        const aclnnStatus status = CheckGbsaPositiveScalarAttr(attr.value, attr.name);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    aclnnStatus status = CheckGbsaHeadAttrs(numQHeads, numKvHeads);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (mxfp8) {
        if (headDim != GBSA_HEAD_DIM_192 && headDim != GBSA_HEAD_DIM_256) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MXFP8 headDim only supports 192 or 256, but got %lld.", headDim);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (isArch35) {
        if (arch35NonQuant ? (headDim > GBSA_ARCH35_NONQUANT_MAX_HEAD_DIM) : headDim != GBSA_CURRENT_HEAD_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "headDim=%lld is unsupported (arch35NonQuant=%d).", headDim,
                    arch35NonQuant);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (headDim > GBSA_HEAD_DIM_256) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A2/A3 headDim must be in [1, 256], got %lld.", headDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckGbsaBlockAttrs(blockShapeX, blockShapeY, layoutSparsePattern, arch35NonQuant, arch22GeneralizedY,
                               arch22PagedGeneralizedY, mxfp8);
}

aclnnStatus CheckGbsaMaxSeqLenAttr(int64_t maxSeqLen, const char* attrName)
{
    if (maxSeqLen < GBSA_DEFAULT_MAX_SEQ_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be -1 or greater than or equal to 0, but got %lld.", attrName,
                maxSeqLen);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMaxSeqLenAttrs(int64_t maxQSeqLen, int64_t maxKvSeqLen)
{
    aclnnStatus status = CheckGbsaMaxSeqLenAttr(maxQSeqLen, "maxQSeqLen");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaMaxSeqLenAttr(maxKvSeqLen, "maxKvSeqLen");
}

aclnnStatus CheckGbsaLayouts(const char* qInputLayout, const char* kvInputLayout)
{
    if (!GbsaIsLayout(qInputLayout, "TND") && !GbsaIsLayout(qInputLayout, "BSND") &&
        !GbsaIsLayout(qInputLayout, "BNSD")) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qInputLayout must be TND, BSND or BNSD.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!GbsaIsLayout(kvInputLayout, qInputLayout) && !GbsaIsLayout(kvInputLayout, "PA_BBND") &&
        !GbsaIsLayout(kvInputLayout, "PA_BNBD")) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dense Q/K/V must use the same layout; paged KV uses PA_BBND/PA_BNBD.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaOptionalSeqLength(const aclTensor* tensor, int64_t batch, aclDataType dataType,
                                       int64_t expectedElements, const char* typeName, const char* tensorName)
{
    if (!GbsaTensorValid(tensor)) {
        return ACLNN_SUCCESS;
    }
    aclnnStatus status = CheckGbsaTensorDataType(tensor, dataType, typeName, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaTensorDim(tensor, GBSA_SEQ_LENGTH_DIM_NUM, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (tensor->GetViewShape().GetDim(0) != expectedElements) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape does not match batch %lld.", tensorName, batch);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GetGbsaBatch(const aclTensor* sparseBlockIdx, const aclTensor* cuSeqLengthsOptional,
                         const char* qInputLayout, int64_t maxQSeqLen, int64_t& batch)
{
    if (!GbsaIsLayout(qInputLayout, "TND")) {
        const auto& shape = sparseBlockIdx->GetViewShape();
        if (shape.GetDimNum() != GBSA_TND_SPARSE_BLOCK_IDX_DIM_NUM || maxQSeqLen <= 0 || shape.GetDim(1) < 0 ||
            shape.GetDim(1) % maxQSeqLen != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Batched sparse Q capacity must be B * maxQSeqLen.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        batch = shape.GetDim(1) / maxQSeqLen;
        return ACLNN_SUCCESS;
    }
    if (!GbsaTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqLengthsOptional->GetViewShape().GetDimNum() != GBSA_SEQ_LENGTH_DIM_NUM ||
        cuSeqLengthsOptional->GetViewShape().GetDim(0) < GBSA_SEQ_LENGTH_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths shape must be [batch + 1].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    batch = cuSeqLengthsOptional->GetViewShape().GetDim(0) - 1;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSparseTensorShapes(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                        int64_t numQHeads, int64_t numKvHeads, int64_t layoutSparsePattern)
{
    const int64_t sparseHeads = layoutSparsePattern == GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB ? numKvHeads : numQHeads;
    if (CheckGbsaTensorDim(sparseBlockIdx, GBSA_TND_SPARSE_BLOCK_IDX_DIM_NUM, "sparseBlockIdx") != ACLNN_SUCCESS ||
        CheckGbsaTensorDim(sparseBlockCount, GBSA_TND_SPARSE_BLOCK_COUNT_DIM_NUM, "sparseBlockCount") !=
            ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseBlockIdx->GetViewShape().GetDim(0) != sparseHeads ||
        sparseBlockCount->GetViewShape().GetDim(0) != sparseHeads ||
        sparseBlockIdx->GetViewShape().GetDim(1) != sparseBlockCount->GetViewShape().GetDim(1) ||
        sparseBlockIdx->GetViewShape().GetDim(GBSA_TND_BLOCK_CAPACITY_DIM_INDEX) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Sparse tensors must be [N2,totalQBlocks,topK] and [N2,totalQBlocks].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSeqLengthInputs(const aclTensor* cuSeqLengthsOptional, const aclTensor* cuSeqLengthsKvOptional,
                                     const aclTensor* seqUsedQOptional, const aclTensor* seqUsedKvOptional,
                                     int64_t batch, const char* qInputLayout, const char* kvInputLayout)
{
    // Batched layouts use seqused exclusively; shape capacity only defines storage strides.
    if (!GbsaIsLayout(qInputLayout, "TND") &&
        (!GbsaTensorValid(seqUsedQOptional) || !GbsaTensorValid(seqUsedKvOptional))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND/BNSD query requires seqUsedQ and seqUsedKv.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (GbsaIsLayout(qInputLayout, "TND") && !GbsaTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isPaKv = GbsaIsLayout(kvInputLayout, "PA_BBND") || GbsaIsLayout(kvInputLayout, "PA_BNBD");
    if (GbsaIsLayout(kvInputLayout, "TND")) {
        if (!GbsaTensorValid(cuSeqLengthsKvOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengthsKv is required for TND KV.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (isPaKv) {
        // FA-style PA: sequsedKv required; cuSeqLengthsKv only for TND.
        if (!GbsaTensorValid(seqUsedKvOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "seqUsedKv is required for paged KV.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (GbsaTensorValid(cuSeqLengthsKvOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "cuSeqLengthsKv must be empty when layoutKv is PA, only supported in TND layout.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    aclnnStatus status =
        CheckGbsaOptionalSeqLength(cuSeqLengthsOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengths");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaOptionalSeqLength(cuSeqLengthsKvOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengthsKv");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaOptionalSeqLength(seqUsedQOptional, batch, ACL_INT32, batch, "INT32", "seqUsedQ");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaOptionalSeqLength(seqUsedKvOptional, batch, ACL_INT32, batch, "INT32", "seqUsedKv");
}

aclnnStatus CheckGbsaQuantType(int64_t quantType, bool isArch35)
{
    if (quantType != GBSA_QUANT_TYPE_NONE && quantType != GBSA_QUANT_TYPE_FLOAT8 &&
        quantType != GBSA_QUANT_TYPE_MXFP8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantType only supports 0, 5 or 6, but got %lld.", quantType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (quantType != GBSA_QUANT_TYPE_NONE && !isArch35) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantType %lld is only supported on Atlas A5.", quantType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSoftmaxPrecision(int64_t softmaxPrecision, int64_t quantType)
{
    if (softmaxPrecision != GBSA_SOFTMAX_PRECISION_HIGH && softmaxPrecision != GBSA_SOFTMAX_PRECISION_MIXED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxPrecision only supports %lld or %lld, but got %lld.",
                GBSA_SOFTMAX_PRECISION_HIGH, GBSA_SOFTMAX_PRECISION_MIXED, softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (quantType == GBSA_QUANT_TYPE_FLOAT8 && softmaxPrecision != GBSA_SOFTMAX_PRECISION_MIXED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Per-tensor FP8 quantization only supports softmaxPrecision=%lld, but got %lld.",
                GBSA_SOFTMAX_PRECISION_MIXED, softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaPlatformAttrs(int64_t quantType, int64_t softmaxPrecision, bool isArch35)
{
    aclnnStatus status = CheckGbsaQuantType(quantType, isArch35);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaSoftmaxPrecision(softmaxPrecision, quantType);
}

aclnnStatus CheckGbsaResidualBlockMode(int64_t residualBlockMode, bool arch35NonQuant, bool arch22NonQuant = false)
{
    if (residualBlockMode != GBSA_RESIDUAL_BLOCK_MODE_NONE &&
        !((arch35NonQuant || arch22NonQuant) && residualBlockMode == GBSA_RESIDUAL_BLOCK_MODE_FORCE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "residual_block_mode=%lld is unsupported; mode 1 only supports non-quant currently.",
                residualBlockMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMaskAttrs(int64_t maskType, int64_t windowSizeLeft, int64_t windowSizeRight, bool isArch35,
                               bool arch35NonQuant)
{
    if (maskType != GBSA_CURRENT_MASK_TYPE && !((!isArch35 || arch35NonQuant) && maskType == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType supports %lld, and 0 on A2/A3 or A5 non-quant/MXFP8, but got %lld.",
                GBSA_CURRENT_MASK_TYPE, maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (windowSizeLeft != GBSA_CURRENT_WINDOW_SIZE || windowSizeRight != GBSA_CURRENT_WINDOW_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "windowSizeLeft and windowSizeRight currently only support %lld, but got %lld and %lld.",
                GBSA_CURRENT_WINDOW_SIZE, windowSizeLeft, windowSizeRight);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSparseBlockCapacity(const aclTensor* sparseBlockIdx, bool arch35NonQuant, bool isArch35)
{
    const int64_t dimNum = sparseBlockIdx->GetViewShape().GetDimNum();
    const int64_t capacity = sparseBlockIdx->GetViewShape().GetDim(dimNum - 1);
    const int64_t maxCapacity =
        (!isArch35 || arch35NonQuant) ? std::numeric_limits<int32_t>::max() : GBSA_ARCH35_QUANT_MAX_SPARSE_BLOCK_COUNT;
    if (capacity > maxCapacity) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseBlockIdx last dimension cannot exceed %lld, but got %lld.", maxCapacity,
                capacity);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMetadataParams(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                    const aclTensor* cuSeqLengthsOptional, const aclTensor* cuSeqLengthsKvOptional,
                                    const aclTensor* seqUsedQOptional, const aclTensor* seqUsedKvOptional,
                                    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads,
                                    int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, const char* qInputLayout,
                                    const char* kvInputLayout, int64_t layoutSparsePattern, int64_t maskType,
                                    int64_t quantType, int64_t softmaxPrecision, int64_t windowSizeLeft,
                                    int64_t windowSizeRight, int64_t residualBlockMode, bool isArch35,
                                    const aclTensor* metadata)
{
    aclnnStatus status = CheckGbsaRequiredInputs(sparseBlockIdx, sparseBlockCount, metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaLayouts(qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if ((!GbsaIsLayout(kvInputLayout, "PA_BBND") || !GbsaIsLayout(qInputLayout, "TND")) &&
        quantType != GBSA_QUANT_TYPE_NONE && !(isArch35 && quantType == GBSA_QUANT_TYPE_MXFP8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Batched Q and dense/PA_BNBD KV require quantType=0 or A5 MXFP8.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    status = CheckGbsaMaxSeqLenAttrs(maxQSeqLen, maxKvSeqLen);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (!GbsaIsLayout(qInputLayout, "TND") && (maxQSeqLen <= 0 || maxQSeqLen > std::numeric_limits<int32_t>::max())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND/BNSD require maxQSeqLen=Sq > 0 within INT32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    status = CheckGbsaScalarAttrs(
        numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, layoutSparsePattern, isArch35,
        isArch35 && quantType == GBSA_QUANT_TYPE_NONE,
        !isArch35 && quantType == GBSA_QUANT_TYPE_NONE && GbsaIsLayout(kvInputLayout, qInputLayout),
        !isArch35 && quantType == GBSA_QUANT_TYPE_NONE &&
            (GbsaIsLayout(kvInputLayout, "PA_BBND") || GbsaIsLayout(kvInputLayout, "PA_BNBD")),
        isArch35 && quantType == GBSA_QUANT_TYPE_MXFP8);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaMetadataShape(metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    int64_t batch = 0;
    status = GetGbsaBatch(sparseBlockIdx, cuSeqLengthsOptional, qInputLayout, maxQSeqLen, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSparseTensorShapes(sparseBlockIdx, sparseBlockCount, numQHeads, numKvHeads, layoutSparsePattern);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSparseBlockCapacity(
        sparseBlockIdx, isArch35 && (quantType == GBSA_QUANT_TYPE_NONE || quantType == GBSA_QUANT_TYPE_MXFP8),
        isArch35);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSeqLengthInputs(cuSeqLengthsOptional, cuSeqLengthsKvOptional, seqUsedQOptional, seqUsedKvOptional,
                                      batch, qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaPlatformAttrs(quantType, softmaxPrecision, isArch35);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaResidualBlockMode(residualBlockMode, isArch35 && quantType == GBSA_QUANT_TYPE_NONE,
                                        !isArch35 && quantType == GBSA_QUANT_TYPE_NONE);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaMaskAttrs(maskType, windowSizeLeft, windowSizeRight, isArch35,
                              isArch35 && (quantType == GBSA_QUANT_TYPE_NONE || quantType == GBSA_QUANT_TYPE_MXFP8));
}

aclnnStatus ParseBlockShape(const aclIntArray* blockShape, int64_t& blockShapeX, int64_t& blockShapeY)
{
    if (blockShape == nullptr || blockShape->Size() != 2U || blockShape->GetData() == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape must contain exactly two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    blockShapeX = blockShape->GetData()[0];
    blockShapeY = blockShape->GetData()[1];
    return ACLNN_SUCCESS;
}

const aclTensor* MakeContiguousOptional(const aclTensor* tensor, aclOpExecutor* executor)
{
    return tensor == nullptr ? nullptr : l0op::Contiguous(tensor, executor);
}

} // namespace

aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
    const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount, const aclTensor* cuSeqLengthsQOptional,
    const aclTensor* cuSeqLengthsKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray* blockShape, const char* layoutQ, const char* layoutKv, int64_t layoutSparsePattern,
    int64_t maskType, int64_t quantType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight,
    int64_t residualBlockMode, bool isConsistentTopk, const aclTensor* metadataOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    int64_t blockShapeX = 0;
    int64_t blockShapeY = 0;
    aclnnStatus status = ParseBlockShape(blockShape, blockShapeX, blockShapeY);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const op::PlatformInfo& platformInfo = op::GetCurrentPlatformInfo();
    const uint32_t aicCoreNum = platformInfo.GetCubeCoreNum();
    const bool isArch35 = platformInfo.GetCurNpuArch() == NpuArch::DAV_3510;
    status = CheckGbsaMetadataParams(sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                                     sequsedQOptional, sequsedKvOptional, maxQSeqLen, maxKvSeqLen, numQHeads,
                                     numKvHeads, headDim, blockShapeX, blockShapeY, layoutQ, layoutKv,
                                     layoutSparsePattern, maskType, quantType, softmaxPrecision, winLeft, winRight,
                                     residualBlockMode, isArch35, metadataOptional);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    // Map the layout pattern to the AICPU-side packed-GQA flag (1: packed by KV head, 0: unpacked).
    const int64_t isPackedGqaFlag = layoutSparsePattern == GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB ? 1 : 0;
    // is_consistent_topk crosses the AICPU attr boundary as an int64 flag (0/1).
    const int64_t isConsistentTopkFlag = isConsistentTopk ? 1 : 0;

    const aclTensor* idxContiguous = l0op::Contiguous(sparseBlockIdx, uniqueExecutor.get());
    CHECK_RET(idxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* countContiguous = l0op::Contiguous(sparseBlockCount, uniqueExecutor.get());
    CHECK_RET(countContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* cuQContiguous = MakeContiguousOptional(cuSeqLengthsQOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLengthsQOptional == nullptr || cuQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* seqUsedQContiguous = MakeContiguousOptional(sequsedQOptional, uniqueExecutor.get());
    CHECK_RET(sequsedQOptional == nullptr || seqUsedQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* cuKvContiguous = MakeContiguousOptional(cuSeqLengthsKvOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLengthsKvOptional == nullptr || cuKvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* seqUsedKvContiguous = MakeContiguousOptional(sequsedKvOptional, uniqueExecutor.get());
    CHECK_RET(sequsedKvOptional == nullptr || seqUsedKvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* output = l0op::GenericBlockSparseAttentionMetadata(
        idxContiguous, countContiguous, cuQContiguous, cuKvContiguous, seqUsedQContiguous, seqUsedKvContiguous,
        maxQSeqLen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGqaFlag, layoutQ, aicCoreNum,
        residualBlockMode, isConsistentTopkFlag, maskType, quantType, metadataOptional, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGenericBlockSparseAttentionMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
