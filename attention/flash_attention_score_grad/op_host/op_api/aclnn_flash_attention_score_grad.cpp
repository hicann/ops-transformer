/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention_score_grad.h"
#include "flash_attention_score_grad.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/fast_vector.h"
#include "runtime/context.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
#define CHECK_SCALAR_TENSOR(condition)                                                                                             \
    do {                                                                                                                           \
        if (condition) {                                                                                                           \
            OP_LOGW("There is a scalar tensor in the input optional parameters, and we will treat this input parameter as null."); \
        }                                                                                                                          \
    } while (0)

typedef struct FagInShapeInfoS {
    int64_t n1Dim;
    int64_t n2Dim;
    int64_t h1Dim;
    int64_t h2Dim;
    int64_t s1Dim;
    int64_t s2Dim;
    int64_t dDim;
    int64_t dkDim;
    int64_t dvDim;
    int64_t alignDim;

    int64_t querySDimStrideSize;
    int64_t kvSDimStrideSize;

    std::string inputLayoutStr;

    bool needPadDimD;
    bool needTranspose;
    bool passThrowInnerFag;
    bool needBackwordReshape;
    bool needPadValueD;
} FagInShapeInfo;

typedef struct FagShapeArrayS {
    aclIntArray *queryShapeArray = nullptr;
    aclIntArray *keyShapeArray = nullptr;
    aclIntArray *dqShapeArray = nullptr;
    aclIntArray *dkShapeArray = nullptr;
    aclIntArray *queryBwShapeArray = nullptr;
    aclIntArray *keyBwShapeArray = nullptr;
    aclIntArray *dqBwShapeArray = nullptr;
    aclIntArray *dkBwShapeArray = nullptr;
    aclIntArray *valueReshapeBefore = nullptr;
    aclIntArray *valueReshapeAfter = nullptr;
    aclIntArray *attenInReshapeBefore = nullptr;
    aclIntArray *attenInReshapeAfter = nullptr;
    aclIntArray *dvReshapeBefore = nullptr;
    aclIntArray *dvReshapeAfter = nullptr;
} FagShapeArray;

static constexpr int64_t ALIGN_D_DIM_SIZE = 128;
static constexpr int64_t SPARE_ALIGN_D_DIM_SIZE = 16;
static constexpr int64_t MAX_BSN_DIMS_SIZE = 65535;
static constexpr int64_t MAX_LAYOUT_SIZE = 5;
static constexpr int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t HEAD_DIM_8 = 8;
static const int64_t HEAD_DIM_72 = 72;
static const int64_t HEAD_DIM_88 = 88;
static const int64_t HEAD_DIM_128 = 128;
static const int64_t HEAD_DIM_192 = 192;

static const int64_t SEQ_LEN_4096 = 4096;
static constexpr size_t MIN_DIM = 3;
static const int64_t TND_MAX_S2 = 1024;
static const int64_t TND_MAX_S1_SUM = 160 * 1024;
static const int64_t TND_MAX_DDIM = 96;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;

char defaultSoftmaxInLayout[] = "";

static bool CheckIsNeedPad(const FagInShapeInfo &fagShape)
{
    if (fagShape.dDim == HEAD_DIM_192 &&  fagShape.inputLayoutStr != "TND" && fagShape.needTranspose == false) {
        OP_LOGD("D=192, Scenarios that do not require pad processing");
        return false;
    }
    if ((fagShape.dDim == HEAD_DIM_72 || fagShape.dDim == HEAD_DIM_88) && fagShape.s1Dim <= SEQ_LEN_4096 &&
         fagShape.s2Dim <= SEQ_LEN_4096 &&  fagShape.inputLayoutStr != "BNSD" && fagShape.inputLayoutStr != "TND" &&
         fagShape.n1Dim == fagShape.n2Dim && fagShape.needTranspose == false) {
        OP_LOGD("Scenarios that do not require pad processing");
        return false;
    }
    return true;
}

static int64_t GetSumIntArrayMaxValue(const aclIntArray *intArrayValue) {
    // 获取targetLengthsList中的最大值
    int64_t maxLength = 0;
    int64_t tmpMaxLength = 0;
    if (intArrayValue->Size() == 1) {
      maxLength = static_cast<int64_t>((*intArrayValue)[0]);
      return maxLength;
    }
    maxLength = static_cast<int64_t>((*intArrayValue)[0]);
    for (size_t i = 1; i < intArrayValue->Size(); i++) {
        tmpMaxLength = static_cast<int64_t>((*intArrayValue)[i]) - static_cast<int64_t>((*intArrayValue)[i - 1]);
        if (tmpMaxLength > maxLength) {
            maxLength = tmpMaxLength;
        }
    }
    return maxLength;
}

static int64_t getSeqLenQSum(const aclIntArray *actualSeqQLenOptional) {
    if (actualSeqQLenOptional->Size() < 1) {
        return 0;
    }
    int64_t sQLenSum = 0;
    for (int64_t i = actualSeqQLenOptional->Size() - 1; i >= 0; --i) {
        sQLenSum = static_cast<int64_t>((*actualSeqQLenOptional)[i]);
        if (sQLenSum > 0) {
            break;
        }
    }
    return sQLenSum;
}

static bool CheckTndIsNeedPad(const FagInShapeInfo &fagShape, const aclIntArray *actualSeqQLenOptional,
                       const aclIntArray *actualSeqKvLenOptional, int64_t dDim, double keepProb)
{
    int64_t sKvLenMax = 0;
    int64_t sQLenSum = 0;
    int64_t deterministicValue = 0;
    rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
    if (retRts != RT_ERROR_NONE) {
        OP_LOGW("Fag aclnn unable to get system param determinstic.");
        // 如果determinstic参数获取失败，则不主动去除pad
        return true;
    }
    OP_LOGD("Fag aclnn deterministic is = %ld.", deterministicValue);
    if (fagShape.inputLayoutStr == "TND" && fagShape.dDim % HEAD_DIM_8 == 0 && deterministicValue == 0) {
        return false;
    }
    // TND并且是非确定性计算
    if (fagShape.inputLayoutStr == "TND" && deterministicValue == 0 &&
        actualSeqQLenOptional != nullptr && actualSeqKvLenOptional != nullptr) {
        if (actualSeqQLenOptional->Size() == actualSeqKvLenOptional->Size()) {
            sKvLenMax = GetSumIntArrayMaxValue(actualSeqKvLenOptional);
            sQLenSum = getSeqLenQSum(actualSeqQLenOptional);
        }
    }
    if (sKvLenMax == 0 || sQLenSum == 0) {
        // 走原来逻辑是否pad
        OP_LOGD("Fag aclnn TND case sKvLenMax(%ld) or sQLenSum(%ld) is 0.", sKvLenMax, sQLenSum);
        return true;
    }

    OP_LOGD("Fag aclnn TND case deterministic: %ld, s2 max: %ld, dDim: %ld, s1 sum: %ld.", deterministicValue,
            sKvLenMax, dDim, sQLenSum);
    bool notHasDropMask = ((!(keepProb < 1.0)) && (!(keepProb > 1.0)));
    if ((sKvLenMax <= TND_MAX_S2) && (dDim < TND_MAX_DDIM) && (sQLenSum < TND_MAX_S1_SUM) && notHasDropMask) {
        // 去除pad
        OP_LOGD("Fag aclnn TND case do not do pad dimD operation.");
        return false;
    }
    return true;
}

static aclnnStatus InvalidTensorDimCheck(const aclTensor *query, const aclTensor *queryRope, const aclTensor *key, const aclTensor *keyRope, const aclTensor *value,
                                         const aclTensor *dy, const aclTensor *attentionIn, const aclTensor *dq, const aclTensor *dqRope,
                                         const aclTensor *dk, const aclTensor *dkRope, const aclTensor *dv)
{
    if (queryRope != nullptr && keyRope != nullptr && dqRope != nullptr && dkRope != nullptr) {
        auto queryRopeDimNum = queryRope->GetViewShape().GetDimNum();
        auto keyRopeDimNum = keyRope->GetViewShape().GetDimNum();
        auto dqRopeDimNum = dqRope->GetViewShape().GetDimNum();
        auto dkRopeDimNum = dkRope->GetViewShape().GetDimNum();
        if (queryRopeDimNum < MIN_DIM || keyRopeDimNum < MIN_DIM || dqRopeDimNum < MIN_DIM || dkRopeDimNum < MIN_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input or output of FAG does not support tensors with dim less than 3.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto queryDimNum = query->GetViewShape().GetDimNum();
    auto keyDimNum = key->GetViewShape().GetDimNum();
    auto valueDimNum = value->GetViewShape().GetDimNum();
    auto dyDimNum = dy->GetViewShape().GetDimNum();
    auto attentionInDimNum = attentionIn->GetViewShape().GetDimNum();
    auto dqDimNum = dq->GetViewShape().GetDimNum();
    auto dkDimNum = dk->GetViewShape().GetDimNum();
    auto dvDimNum = dv->GetViewShape().GetDimNum();
    if (queryDimNum < MIN_DIM || keyDimNum < MIN_DIM || valueDimNum < MIN_DIM || dyDimNum < MIN_DIM ||
        attentionInDimNum < MIN_DIM || dqDimNum < MIN_DIM || dkDimNum < MIN_DIM || dvDimNum < MIN_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input or output of FAG does not support tensors with dim less than 3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus isSupportMultiInput(const aclTensor *query, const aclTensor *queryRope,
                                       const aclTensor *key, const aclTensor *keyRope, const aclTensor *value, 
                                       const aclTensor *attenMaskOptional, const aclTensor *pseShiftOptional,
                                       const aclTensor *dropMaskOptional, double keepProb, FagInShapeInfo &fagShape,
                                       int64_t sparseMode)
{
    CHECK_RET((queryRope == nullptr && keyRope == nullptr) || (queryRope != nullptr && keyRope != nullptr),
            ACLNN_ERR_PARAM_NULLPTR);
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    auto kRopeDtype = keyRope->GetDataType();
    auto qRopeDtype = queryRope->GetDataType();
    auto qRopeShape = queryRope->GetViewShape();
    auto kRopeShape = keyRope->GetViewShape();
    if (qRopeShape.GetDim(DIM_NUM_2) > fagShape.dDim || kRopeShape.GetDim(DIM_NUM_2) > fagShape.dDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, do not support query_rope and key_rope when"
                " the head-dim of query_rope or key_rope is larger than the head-dim of query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype !=  ge::DataType::DT_BF16 || kDtype != ge::DataType::DT_BF16 || vDtype != ge::DataType::DT_BF16
        || qRopeDtype != ge::DataType::DT_BF16 || kRopeDtype != ge::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], queryRope[%s], key[%s], keyRope[%s], value[%s]"
                " should be BFloat16.", op::ToString(DataType(qDtype)).GetString(),
                op::ToString(DataType(qRopeDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(kRopeDtype)).GetString(), op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseMode == 6) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, do not support query_rope and key_rope when sparseMode is 6.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (queryRope != nullptr) {
        if (attenMaskOptional == nullptr ||
            attenMaskOptional->GetViewShape().GetDimNum() == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Invalid input, only support query_rope and key_rope when attentionMask is given.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (pseShiftOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope when pseShift is nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (dropMaskOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope when dropMask is nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (keepProb < 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope when keepProb = 1.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (fagShape.inputLayoutStr != "TND") {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope as input for layout TND.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (fagShape.needPadDimD || fagShape.needTranspose || fagShape.needBackwordReshape || fagShape.needPadValueD) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, do not support query_rope and key_rope as input when shape is not aligned with 128 or other corner cases.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GetInputShapeInfo(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                     int64_t headNum, const char *inputLayout, FagInShapeInfo &fagShape,
                                     const aclIntArray *actualSeqQLenOptional,
                                     const aclIntArray *actualSeqKvLenOptional,
                                     double keepProb)
{
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    auto valueShape = value->GetViewShape();
    auto queryDimSize = query->Size();
    auto kvDimSize = key->Size();
    fagShape.inputLayoutStr = op::ToString(inputLayout).GetString();
    fagShape.n1Dim = (fagShape.inputLayoutStr == "BNSD") ? queryShape.GetDim(1) : queryShape.GetDim(2); // 1 or 2:n1
    fagShape.n2Dim = (fagShape.inputLayoutStr == "BNSD") ? keyShape.GetDim(1) : keyShape.GetDim(2);       // 1 or 2:n2
    fagShape.s1Dim = (fagShape.inputLayoutStr == "BNSD") ? queryShape.GetDim(2) : queryShape.GetDim(1); // 1 or 2:s1
    fagShape.s2Dim = (fagShape.inputLayoutStr == "BNSD") ? keyShape.GetDim(2) : keyShape.GetDim(1);       // 1 or 2:s2
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "SBH") {
        fagShape.h1Dim = queryShape.GetDim(2); // 2:h1
        fagShape.h2Dim = keyShape.GetDim(2);    // 2:h2
        fagShape.dDim = fagShape.h1Dim / headNum; // q Head-dim
        if (fagShape.dDim == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of D is zero.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        fagShape.n1Dim = headNum;
        fagShape.n2Dim = fagShape.h2Dim / fagShape.dDim;
        fagShape.s1Dim = (fagShape.inputLayoutStr == "BSH") ? queryShape.GetDim(1) : queryShape.GetDim(0);
        fagShape.s2Dim = (fagShape.inputLayoutStr == "BSH") ? keyShape.GetDim(1) : keyShape.GetDim(0);
        fagShape.dkDim = keyShape.GetDim(DIM_NUM_2) / fagShape.n2Dim;
        fagShape.dvDim = valueShape.GetDim(DIM_NUM_2) / fagShape.n2Dim;
    } else if (fagShape.inputLayoutStr == "TND") {
        fagShape.dDim = queryShape.GetDim(2);  // 2:d
        fagShape.n1Dim = queryShape.GetDim(1); // 1:n1
        fagShape.n2Dim = keyShape.GetDim(1);    // 1:n2
        fagShape.dkDim = keyShape.GetDim(DIM_NUM_2);
        fagShape.dvDim = valueShape.GetDim(DIM_NUM_2);
    } else if (queryShape.GetDimNum() > MIN_DIM) {
        fagShape.dDim = queryShape.GetDim(3); // 3:d
        fagShape.dkDim = keyShape.GetDim(3); // key Head-dim
        fagShape.dvDim = valueShape.GetDim(3); // value Head-dim
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of the tensor whose input is BNSD/BSND is less than 4.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim != fagShape.dkDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", fagShape.dDim, fagShape.dkDim);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim < fagShape.dvDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The headDim of value should be smaller than headDim of key.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t deterministicValue = 0;
    rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
    if (retRts != RT_ERROR_NONE) {
        OP_LOGW("Fag aclnn unable to get system param determinstic.");
        // 如果determinstic参数获取失败，则不主动去除pad
        deterministicValue = DIM_NUM_3;
    }
    fagShape.needPadValueD = (fagShape.dDim != fagShape.dvDim) && (!(fagShape.dDim == HEAD_DIM_192 && fagShape.dvDim == HEAD_DIM_128 && deterministicValue == 0));
    fagShape.querySDimStrideSize = 0;
    fagShape.kvSDimStrideSize = 0;
    if (fagShape.inputLayoutStr == "BSND") { // stride is N * D
        fagShape.querySDimStrideSize = fagShape.n1Dim * fagShape.dDim;
        fagShape.kvSDimStrideSize = fagShape.n2Dim * fagShape.dDim;
    } else if (fagShape.inputLayoutStr == "BSH") {           // stride is H
        fagShape.querySDimStrideSize = queryShape.GetDim(2); // 2:dv
        fagShape.kvSDimStrideSize = keyShape.GetDim(2);       // 2:dv
    } else if (fagShape.inputLayoutStr == "SBH") {           // stride is B * H
        fagShape.querySDimStrideSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
        fagShape.kvSDimStrideSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
    }

    fagShape.alignDim = (fagShape.dDim < ALIGN_D_DIM_SIZE) ? SPARE_ALIGN_D_DIM_SIZE : ALIGN_D_DIM_SIZE;
    auto dDimAlignSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim;

    // 判断是否需要PAD和transpose, 同时判断是否为如下特殊场景 (SBH下，只需要PAD不需要transpose)
    fagShape.needPadDimD =
        (fagShape.dDim % fagShape.alignDim != 0 && queryShape.GetShapeSize() != 0 && keyShape.GetShapeSize() != 0) ?
            true :
            false;

    // 计算是否超过65535时，应该使用对齐以后的D值
    if (fagShape.needPadDimD) {
        if (fagShape.inputLayoutStr == "BSND") { // stride is N * D
            fagShape.querySDimStrideSize = fagShape.n1Dim * dDimAlignSize;
            fagShape.kvSDimStrideSize = fagShape.n2Dim * dDimAlignSize;
        } else if (fagShape.inputLayoutStr == "BSH") {           // stride is H
            fagShape.querySDimStrideSize = fagShape.dDim == 0 ? 0 :
                (queryShape.GetDim(2) / fagShape.dDim * dDimAlignSize); // 2:dv
            fagShape.kvSDimStrideSize = fagShape.dDim == 0 ? 0 :
                (keyShape.GetDim(2) / fagShape.dDim * dDimAlignSize);       // 2:dv
        } else if (fagShape.inputLayoutStr == "SBH") {           // stride is B * H
            int64_t queryBHSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
            int64_t kvBHSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
            fagShape.querySDimStrideSize = fagShape.dDim == 0 ? 0 : (queryBHSize / fagShape.dDim * dDimAlignSize);
            fagShape.kvSDimStrideSize = fagShape.dDim == 0 ? 0 : (kvBHSize / fagShape.dDim * dDimAlignSize);
        }
    }

    bool needTranspose =
        queryShape.GetShapeSize() != 0 && keyShape.GetShapeSize() != 0 &&
        (fagShape.inputLayoutStr != "BNSD" && fagShape.inputLayoutStr != "TND" &&
         (fagShape.querySDimStrideSize > MAX_BSN_DIMS_SIZE || fagShape.kvSDimStrideSize > MAX_BSN_DIMS_SIZE));
    fagShape.needTranspose = needTranspose;

    if (!CheckIsNeedPad(fagShape) ||
        !CheckTndIsNeedPad(fagShape, actualSeqQLenOptional, actualSeqKvLenOptional, fagShape.dDim, keepProb)) {
        fagShape.needPadDimD = false;
    }
    // 特殊情况的TND的D不等长无需处理
    if (fagShape.inputLayoutStr == "TND" && fagShape.dDim == HEAD_DIM_192 && fagShape.dvDim == HEAD_DIM_128 && deterministicValue == 0) {
        fagShape.needPadDimD = false;
    }

    fagShape.passThrowInnerFag = (!(fagShape.needPadDimD) && !(fagShape.needTranspose));
    fagShape.needBackwordReshape =
        (fagShape.inputLayoutStr == "SBH" && fagShape.needPadDimD && !(fagShape.needTranspose));
    return ACLNN_SUCCESS;
}

static inline aclnnStatus ContiguousTensorWithCheck(const aclTensor *inputTensor, const aclTensor **outTensor,
                                                    aclOpExecutor *executor)
{
    if (inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() != 0) {
        *outTensor = l0op::Contiguous(inputTensor, executor);
        CHECK_RET(*outTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    // 输入入参如果是标量tensor，将会按照此可选输入为null处理 ;
    CHECK_SCALAR_TENSOR(inputTensor != nullptr && inputTensor->GetViewShape().GetDimNum() == 0);

    return ACLNN_SUCCESS;
}

static inline void ConvertInputLayout(FagInShapeInfo fagShape, const char *inputLayout, char *inputLayoutUnderTrans,
                                      size_t layoutUnderTransSize)
{
    if (fagShape.needTranspose) {                  // 1. 只要是需要transpose，输入FAG layout必然是BNSD
        inputLayoutUnderTrans[0] = 'B';            // 0: 'B'
        inputLayoutUnderTrans[1] = 'N';            // 1: 'N'
        inputLayoutUnderTrans[2] = 'S';            // 2: 'S'
        inputLayoutUnderTrans[3] = 'D';            // 3: 'D'
    } else if (fagShape.needBackwordReshape) {     // 2. 如果是SBH仅PAD场景，输入FAG layout必然还是SBH
        inputLayoutUnderTrans[0] = inputLayout[0]; // 0: 'S'
        inputLayoutUnderTrans[1] = inputLayout[1]; // 1: 'B'
        inputLayoutUnderTrans[2] = 'H';            // 2: 'H'
    } else if (fagShape.needPadDimD) { // 3. 如果是仅PAD场景，根据BSH/SBH/BNSD/BSND自适应reshape后的layout
        /* BSH  -> BSND
           SBH  -> SBND
           TND  -> TND
           BNSD -> BNSD
           BSND -> BSND */
        for (size_t i = 0; i < strlen(inputLayout) && i < layoutUnderTransSize - 1; i++) {
            if (inputLayout[i] == 'H') {
                inputLayoutUnderTrans[i] = 'N';
                inputLayoutUnderTrans[i + 1] = 'D';
                break;
            }
            inputLayoutUnderTrans[i] = inputLayout[i];
        }
    } else { // 4. 其他情况，保持原始layout
        for (size_t i = 0; i < strlen(inputLayout) && i < layoutUnderTransSize - 1; i++) {
            inputLayoutUnderTrans[i] = inputLayout[i];
        }
    }
}

static aclnnStatus ContiguousInputTensor(const aclTensor *query, const aclTensor *queryRope, const aclTensor *key,
                                         const aclTensor *keyRope, const aclTensor *value,
                                         const aclTensor *dy, const aclTensor *attentionInOptional,
                                         const aclTensor **queryCngs, const aclTensor **queryRopeCngs,
                                         const aclTensor **keyCngs,const aclTensor **keyRopeCngs,
                                         const aclTensor **valueCngs, const aclTensor **dyCngs,
                                         const aclTensor **attentionInOptionalCngs, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // query如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(query, queryCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    if (queryRope != nullptr) {
        // query如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(queryRope, queryRopeCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    }

    // key如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(key, keyCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    if (keyRope != nullptr) {
        // key如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(keyRope, keyRopeCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    }

    // value如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(value, valueCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // dy如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dy, dyCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // attentionInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(attentionInOptional, attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    return ret;
}

static aclnnStatus ContiguousOptionalInputTensor(
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor **pseShiftOptionalCngs, const aclTensor **dropMaskOptionalCngs,
    const aclTensor **paddingMaskOptionalCngs, const aclTensor **attenMaskOptionalCngs,
    const aclTensor **softmaxMaxOptionalCngs, const aclTensor **softmaxSumOptionalCngs,
    const aclTensor **softmaxInOptionalCngs, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // pseShiftOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(pseShiftOptional, pseShiftOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // dropMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dropMaskOptional, dropMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // paddingMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(paddingMaskOptional, paddingMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // attenMaskOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(attenMaskOptional, attenMaskOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // softmaxMaxOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxMaxOptional, softmaxMaxOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // softmaxSumOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxSumOptional, softmaxSumOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    // softmaxInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(softmaxInOptional, softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    return ret;
}

static void GetInputAndOutputReshapeArray(const aclTensor *query, const aclTensor *key, FagInShapeInfo fagShape,
                                          FagShapeArray &fagShapeArray, aclOpExecutor *executor)
{
    if (fagShape.passThrowInnerFag) {
        return;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return;
    }

    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    FVector<int64_t, 0> queryReshapeList;
    FVector<int64_t, 0> keyReshapeList;
    FVector<int64_t, 0> dqReshapeList;
    FVector<int64_t, 0> dkReshapeList;
    for (size_t i = 0; i < 3; i++) { // 3: sizeof("BSH")
        dqReshapeList.emplace_back(queryShape.GetDim(i));
        dkReshapeList.emplace_back(keyShape.GetDim(i));
        if (i < 2) { // 2: split last Dim
            queryReshapeList.emplace_back(queryShape.GetDim(i));
            keyReshapeList.emplace_back(keyShape.GetDim(i));
        }
    }

    queryReshapeList.emplace_back(fagShape.n1Dim);
    queryReshapeList.emplace_back(fagShape.dDim);
    keyReshapeList.emplace_back(fagShape.n2Dim);
    keyReshapeList.emplace_back(fagShape.dDim);

    // get shape array
    fagShapeArray.queryShapeArray = executor->AllocIntArray(queryReshapeList.data(), queryReshapeList.size());
    fagShapeArray.dqShapeArray = executor->AllocIntArray(dqReshapeList.data(), dqReshapeList.size());
    fagShapeArray.keyShapeArray = executor->AllocIntArray(keyReshapeList.data(), keyReshapeList.size());
    fagShapeArray.dkShapeArray = executor->AllocIntArray(dkReshapeList.data(), dkReshapeList.size());

    return;
}

static void GetInputAndOutputBackwordReshapeArrayForSBH(const aclTensor *query, const aclTensor *key,
                                                        FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                                        aclOpExecutor *executor)
{
    if (!(fagShape.needBackwordReshape)) {
        return;
    }

    if (query == nullptr || key == nullptr) {
        return;
    }
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    FVector<int64_t, 0> queryReshapeList;
    FVector<int64_t, 0> keyReshapeList;
    FVector<int64_t, 0> dqReshapeList;
    FVector<int64_t, 0> dkReshapeList;
    for (size_t i = 0; i < 2; i++) { // 2: get SBH pre shape size 'SB'
        queryReshapeList.emplace_back(queryShape.GetDim(i));
        dqReshapeList.emplace_back(queryShape.GetDim(i));
        keyReshapeList.emplace_back(keyShape.GetDim(i));
        dkReshapeList.emplace_back(keyShape.GetDim(i));
    }

    auto dDimAlignSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim;
    auto queryHDimAlignSize = fagShape.n1Dim * dDimAlignSize;
    auto keyHDimAlignSize = fagShape.n2Dim * dDimAlignSize;

    queryReshapeList.emplace_back(queryHDimAlignSize);
    keyReshapeList.emplace_back(keyHDimAlignSize);

    dqReshapeList.emplace_back(fagShape.n1Dim);
    dqReshapeList.emplace_back(dDimAlignSize);
    dkReshapeList.emplace_back(fagShape.n2Dim);
    dkReshapeList.emplace_back(dDimAlignSize);

    // get shape array
    fagShapeArray.queryBwShapeArray = executor->AllocIntArray(queryReshapeList.data(), queryReshapeList.size());
    fagShapeArray.dqBwShapeArray = executor->AllocIntArray(dqReshapeList.data(), dqReshapeList.size());
    fagShapeArray.keyBwShapeArray = executor->AllocIntArray(keyReshapeList.data(), keyReshapeList.size());
    fagShapeArray.dkBwShapeArray = executor->AllocIntArray(dkReshapeList.data(), dkReshapeList.size());

    return;
}

static void GetKvUnequalReshapeArray(const aclTensor *value, FagInShapeInfo fagShape, FagShapeArray &fagShapeArray, aclOpExecutor *executor)
{
    if (!(fagShape.needPadValueD)) {
        return;
    }
 
    if (!(fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH")) {
        return;
    }
 
    FVector<int64_t, DIM_NUM_4> valueReshapeBeforeList;
    FVector<int64_t, DIM_NUM_4> attenInReshapeBeforeList;
    FVector<int64_t, DIM_NUM_4> dvReshapeBeforeList;
    FVector<int64_t, DIM_NUM_3> valueReshapeAfterList;
    FVector<int64_t, DIM_NUM_3> attenInReshapeAfterList;
    FVector<int64_t, DIM_NUM_3> dvReshapeAfterList;
 
    if (fagShape.inputLayoutStr == "SBH") {
        auto bDim = value->GetViewShape().GetDim(1);
        valueReshapeBeforeList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim, fagShape.dvDim});
        valueReshapeAfterList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim * fagShape.dDim});
        attenInReshapeBeforeList.assign({fagShape.s1Dim, bDim, fagShape.n1Dim, fagShape.dvDim});
        attenInReshapeAfterList.assign({fagShape.s1Dim, bDim, fagShape.n1Dim * fagShape.dDim});
        dvReshapeBeforeList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim, fagShape.dDim});
        dvReshapeAfterList.assign({fagShape.s2Dim, bDim, fagShape.n2Dim * fagShape.dvDim});
    } else { // BSH
        auto bDim = value->GetViewShape().GetDim(0);
        valueReshapeBeforeList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim, fagShape.dvDim});
        valueReshapeAfterList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim * fagShape.dDim});
        attenInReshapeBeforeList.assign({bDim, fagShape.s1Dim, fagShape.n1Dim, fagShape.dvDim});
        attenInReshapeAfterList.assign({bDim, fagShape.s1Dim, fagShape.n1Dim * fagShape.dDim});
        dvReshapeBeforeList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim, fagShape.dDim});
        dvReshapeAfterList.assign({bDim, fagShape.s2Dim, fagShape.n2Dim * fagShape.dvDim});
    }
 
    fagShapeArray.valueReshapeBefore = executor->AllocIntArray(valueReshapeBeforeList.data(), valueReshapeBeforeList.size());
    fagShapeArray.valueReshapeAfter = executor->AllocIntArray(valueReshapeAfterList.data(), valueReshapeAfterList.size());
 
    fagShapeArray.attenInReshapeBefore = executor->AllocIntArray(attenInReshapeBeforeList.data(), attenInReshapeBeforeList.size());
    fagShapeArray.attenInReshapeAfter = executor->AllocIntArray(attenInReshapeAfterList.data(), attenInReshapeAfterList.size());
 
    fagShapeArray.dvReshapeBefore = executor->AllocIntArray(dvReshapeBeforeList.data(), dvReshapeBeforeList.size());
    fagShapeArray.dvReshapeAfter = executor->AllocIntArray(dvReshapeAfterList.data(), dvReshapeAfterList.size());
}

static aclnnStatus ReshapeInputTensor(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                      const aclTensor **dy, const aclTensor **attentionInOptional,
                                      FagInShapeInfo fagShape, FagShapeArray fagShapeArray, bool isBackWord,
                                      aclOpExecutor *executor)
{
    bool needReshape = isBackWord ? fagShape.needBackwordReshape : !(fagShape.passThrowInnerFag);
    if (!needReshape) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return ACLNN_SUCCESS;
    }

    auto queryShapeArray = isBackWord ? fagShapeArray.queryBwShapeArray : fagShapeArray.queryShapeArray;
    auto keyShapeArray = isBackWord ? fagShapeArray.keyBwShapeArray : fagShapeArray.keyShapeArray;

    // reshape input
    *query = l0op::Reshape(*query, queryShapeArray, executor);
    CHECK_RET(*query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *key = l0op::Reshape(*key, keyShapeArray, executor);
    CHECK_RET(*key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *value = l0op::Reshape(*value, keyShapeArray, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *dy = l0op::Reshape(*dy, queryShapeArray, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Reshape(*attentionInOptional, queryShapeArray, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ReshapeOutputTensor(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                       FagInShapeInfo fagShape, FagShapeArray fagShapeArray, bool isBackWord,
                                       aclOpExecutor *executor)
{
    bool needReshape = isBackWord ? fagShape.needBackwordReshape : !(fagShape.passThrowInnerFag);
    if (!needReshape) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr != "BSH" && fagShape.inputLayoutStr != "SBH") {
        return ACLNN_SUCCESS;
    }

    aclIntArray *dqShapeArray = isBackWord ? fagShapeArray.dqBwShapeArray : fagShapeArray.dqShapeArray;
    aclIntArray *dkShapeArray = isBackWord ? fagShapeArray.dkBwShapeArray : fagShapeArray.dkShapeArray;

    // reshape
    fagOut[0] = l0op::Reshape(fagOut[0], dqShapeArray, executor);
    CHECK_RET(fagOut[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    fagOut[1] = l0op::Reshape(fagOut[1], dkShapeArray, executor);
    CHECK_RET(fagOut[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    fagOut[2] = l0op::Reshape(fagOut[2], dkShapeArray, executor); // 2:dv
    CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);     // 2:dv

    return ACLNN_SUCCESS;
}

static aclnnStatus PaddingInputTensorDdim(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                          const aclTensor **dy, const aclTensor **attentionInOptional,
                                          FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needPadDimD)) {
        OP_LOGD("Fag aclnn case do not do pad dimD operation.");
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Fag aclnn case do pad dimD operation.");

    // padding
    // query
    auto padSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim - fagShape.dDim;
    aclIntArray *paddingArray = nullptr;
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> padding = {0, 0, 0, 0, 0, padSize};
        paddingArray = executor->AllocIntArray(padding.data(), 6); // 6: TND 3dims, padding D dim
    } else {
        FVector<int64_t> padding = {0, 0, 0, 0, 0, 0, 0, padSize};
        paddingArray = executor->AllocIntArray(padding.data(), 8); // 8: BNSD 4dims, padding D dim
    }
    auto padTensor = executor->ConvertToTensor(paddingArray, DataType::DT_INT64);
    CHECK_RET(padTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *query = l0op::Pad(*query, padTensor, executor);
    CHECK_RET(*query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // key
    *key = l0op::Pad(*key, padTensor, executor);
    CHECK_RET(*key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // value
    *value = l0op::Pad(*value, padTensor, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // dy
    *dy = l0op::Pad(*dy, padTensor, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // attenmask_in
    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Pad(*attentionInOptional, padTensor, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus SliceOutputTensorDdim(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                         FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needPadDimD)) {
        return ACLNN_SUCCESS;
    }

    auto dqOutShape = (fagOut[0])->GetViewShape(); // 0: dq
    auto dkOutShape = (fagOut[1])->GetViewShape(); // 1: dk

    // slice
    FVector<int64_t> dqOutSizeVector;
    FVector<int64_t> dkOutSizeVector;
    for (size_t i = 0; i < dqOutShape.GetDimNum() - 1; i++) {
        dqOutSizeVector.emplace_back(dqOutShape.GetDim(i));
    }

    for (size_t i = 0; i < dkOutShape.GetDimNum() - 1; i++) {
        dkOutSizeVector.emplace_back(dkOutShape.GetDim(i));
    }

    aclIntArray *offsets = nullptr;
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> offsetsVector = {0, 0, 0};
        offsets = executor->AllocIntArray(offsetsVector.data(), offsetsVector.size());
    } else {
        FVector<int64_t> offsetsVector = {0, 0, 0, 0};
        offsets = executor->AllocIntArray(offsetsVector.data(), offsetsVector.size());
    }

    dqOutSizeVector.emplace_back(fagShape.dDim);
    auto dqOutSize = executor->AllocIntArray(dqOutSizeVector.data(), dqOutSizeVector.size());
    fagOut[0] = l0op::Slice(fagOut[0], offsets, dqOutSize, executor); // 0: dq
    CHECK_RET(fagOut[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    dkOutSizeVector.emplace_back(fagShape.dDim);
    auto dkOutSize = executor->AllocIntArray(dkOutSizeVector.data(), dkOutSizeVector.size());
    fagOut[1] = l0op::Slice(fagOut[1], offsets, dkOutSize, executor); // 1: dk
    CHECK_RET(fagOut[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    fagOut[2] = l0op::Slice(fagOut[2], offsets, dkOutSize, executor); // 2: dv
    CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static aclnnStatus PaddingValueDim(const aclTensor **value, const aclTensor **dy, const aclTensor **attentionInOptional,
                                   FagInShapeInfo fagShape, FagShapeArray fagShapeArray, aclOpExecutor *executor)
{
    if (!(fagShape.needPadValueD)) {
        return ACLNN_SUCCESS;
    }
 
    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        *value = l0op::Reshape(*value, fagShapeArray.valueReshapeBefore, executor);
        CHECK_RET(*value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        *dy = l0op::Reshape(*dy, fagShapeArray.attenInReshapeBefore, executor);
        CHECK_RET(*dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        *attentionInOptional = l0op::Reshape(*attentionInOptional, fagShapeArray.attenInReshapeBefore, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
 
    int32_t dimNum = (fagShape.inputLayoutStr == "TND") ? DIM_NUM_3 : DIM_NUM_4;
    auto paddings = GeneratePaddings(dimNum, fagShape.dDim - fagShape.dvDim, executor);
    *value = l0op::Pad(*value, paddings, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
 
    *dy = l0op::Pad(*dy, paddings, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
 
    *attentionInOptional = l0op::Pad(*attentionInOptional, paddings, executor);
    CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
 
    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        *value = l0op::Reshape(*value, fagShapeArray.valueReshapeAfter, executor);
        CHECK_RET(*value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        *dy = l0op::Reshape(*dy, fagShapeArray.attenInReshapeAfter, executor);
        CHECK_RET(*dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        *attentionInOptional = l0op::Reshape(*attentionInOptional, fagShapeArray.attenInReshapeAfter, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
 
    return ACLNN_SUCCESS;
}
 
static aclnnStatus SliceDvOut(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                              FagInShapeInfo fagShape, FagShapeArray fagShapeArray, aclOpExecutor *executor)
{
    if (!(fagShape.needPadValueD)) {
        return ACLNN_SUCCESS;
    }
 
    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        fagOut[2] = l0op::Reshape(fagOut[2], fagShapeArray.dvReshapeBefore, executor);
        CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR); // 2: dv
    }
 
    FVector<int64_t, MAX_DIM_NUM> dvOutSizeVector = ToShapeVector(fagOut[2]->GetViewShape());
    dvOutSizeVector.back() -= fagShape.dDim - fagShape.dvDim;
    auto dvOutSize = executor->AllocIntArray(dvOutSizeVector.data(), dvOutSizeVector.size());
    if (fagShape.inputLayoutStr != "TND") {
        FVector<int64_t, DIM_NUM_4> offsets(DIM_NUM_4, 0);
        fagOut[2] = l0op::Slice(fagOut[2], executor->AllocIntArray(offsets.data(), offsets.size()),
                                dvOutSize, executor); // 2: dv
        CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    } else {
        FVector<int64_t, DIM_NUM_3> offsets(DIM_NUM_3, 0);
        fagOut[2] = l0op::Slice(fagOut[2], executor->AllocIntArray(offsets.data(), offsets.size()),
                                dvOutSize, executor); // 2: dv
        CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    
    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        fagOut[2] = l0op::Reshape(fagOut[2], fagShapeArray.dvReshapeAfter, executor);
        CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR); // 2: dv
    }
 
    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeInputTensor(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                        const aclTensor **dy, const aclTensor **attentionInOptional,
                                        FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needTranspose)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "BNSD" || fagShape.inputLayoutStr == "TND") {
        return ACLNN_SUCCESS;
    }

    FVector<int64_t> transposeDim;
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "BSND") {
        transposeDim = {0, 2, 1, 3};
    } else {
        transposeDim = {1, 2, 0, 3};
    }

    auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());

    // query
    *query = l0op::Transpose(*query, perm, executor);
    CHECK_RET(*query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // key
    *key = l0op::Transpose(*key, perm, executor);
    CHECK_RET(*key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // value
    *value = l0op::Transpose(*value, perm, executor);
    CHECK_RET(*value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // dy
    *dy = l0op::Transpose(*dy, perm, executor);
    CHECK_RET(*dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // attentionInOptional
    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Transpose(*attentionInOptional, perm, executor);
        CHECK_RET(*attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeSoftMaxTensor(const aclTensor **softmax, const aclTensor **softmaxSum,FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (fagShape.inputLayoutStr == "TND") {
        FVector<int64_t> transposeDim = {1, 0, 2};
        auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());
        *softmax = l0op::Transpose(*softmax, perm, executor);
        CHECK_RET(*softmax != nullptr, ACLNN_ERR_PARAM_NULLPTR);

        *softmaxSum = l0op::Transpose(*softmaxSum, perm, executor);
        CHECK_RET(*softmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeOutputTensor(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                         FagInShapeInfo fagShape, aclOpExecutor *executor)
{
    if (!(fagShape.needTranspose)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "BNSD" || fagShape.inputLayoutStr == "TND") {
        return ACLNN_SUCCESS;
    }

    FVector<int64_t> transposeDim;
    if (fagShape.inputLayoutStr == "BSH" || fagShape.inputLayoutStr == "BSND") {
        transposeDim = {0, 2, 1, 3};
    } else {
        transposeDim = {2, 0, 1, 3};
    }

    auto perm = executor->AllocIntArray(transposeDim.data(), transposeDim.size());

    // dqOut
    fagOut[0] = l0op::Transpose(fagOut[0], perm, executor);
    CHECK_RET(fagOut[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // dkOut
    fagOut[1] = l0op::Transpose(fagOut[1], perm, executor);
    CHECK_RET(fagOut[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // dvOut
    fagOut[2] = l0op::Transpose(fagOut[2], perm, executor);   // 2:dvOut
    CHECK_RET(fagOut[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR); // 2:dvOut

    // dpseOut
    return ACLNN_SUCCESS;
}

static aclnnStatus PreFlashAttentionScoreGrad(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                              const aclTensor **dy, const aclTensor **attentionInOptional,
                                              FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                              aclOpExecutor *executor)
{
    // 获取reshape array, SBH特殊场景下，需要提前获取调用FAG前反向reshape成SBH时所需的reshape array
    GetInputAndOutputReshapeArray(*query, *key, fagShape, fagShapeArray, executor);
    GetInputAndOutputBackwordReshapeArrayForSBH(*query, *key, fagShape, fagShapeArray, executor);
    GetKvUnequalReshapeArray(*value, fagShape, fagShapeArray, executor);

    // 特定情况下，KV Dim不等长时，将HeadDim Padding到与K相同
    auto ret = PaddingValueDim(value, dy, attentionInOptional, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输入tensor从三维扩展成四维
    ret = ReshapeInputTensor(query, key, value, dy, attentionInOptional, fagShape, fagShapeArray, false, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 执行D轴Padding到对齐值
    ret = PaddingInputTensorDdim(query, key, value, dy, attentionInOptional, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 执行输入transpose到BNSD
    ret = TransposeInputTensor(query, key, value, dy, attentionInOptional, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是SBH特殊场景，在调用FAG前，需要将SBND重新改成SBH，否则FAG将报错不支持layout
    ret = ReshapeInputTensor(query, key, value, dy, attentionInOptional, fagShape, fagShapeArray, true, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static aclnnStatus PostFlashAttentionScoreGrad(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut,
                                               const aclTensor **dqOut, const aclTensor **dqRopeOut,
                                               const aclTensor **dkOut, const aclTensor **dkRopeOut,
                                               const aclTensor **dvOut, const aclTensor **dpseOut,
                                               FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                               aclOpExecutor *executor)
{
    // 如果是SBH特殊场景，在调用FAG后，需要将SBH重新改成SBND，以完成后续的slice等操作
    auto ret = ReshapeOutputTensor(fagOut, fagShape, fagShapeArray, true, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输出由BNSD转为原始shape
    ret = TransposeOutputTensor(fagOut, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将D轴padding脏数据切掉
    ret = SliceOutputTensorDdim(fagOut, fagShape, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 将输出tensor由四维还原成三维
    ret = ReshapeOutputTensor(fagOut, fagShape, fagShapeArray, false, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // KV不等长时，将dVOut HeadDim还原
    ret = SliceDvOut(fagOut, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto dqOutViewCopyRes = l0op::ViewCopy(fagOut[0], *dqOut, executor);
    CHECK_RET(dqOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto dkOutViewCopyRes = l0op::ViewCopy(fagOut[1], *dkOut, executor);
    CHECK_RET(dkOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto dvOutViewCopyRes = l0op::ViewCopy(fagOut[2], *dvOut, executor);
    CHECK_RET(dvOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (!(*dpseOut == nullptr || (*dpseOut)->GetDataType() == ge::DataType::DT_FLOAT)) {
        auto dpseOutViewCopyRes = l0op::ViewCopy(fagOut[3], *dpseOut, executor);
        CHECK_RET(dpseOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dqRopeOut != nullptr && *dqRopeOut != nullptr) {
        auto dqRopeOutViewCopyRes = l0op::ViewCopy(fagOut[4], *dqRopeOut, executor);
        CHECK_RET(dqRopeOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dkRopeOut != nullptr && *dkRopeOut != nullptr) {
        auto dkRopeOutViewCopyRes = l0op::ViewCopy(fagOut[5], *dkRopeOut, executor);
        CHECK_RET(dkRopeOutViewCopyRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus FlashAttentionScoreGradGetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor* dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, char *softmaxInLayout,
    const uint64_t *workspaceSize, aclOpExecutor *executor) {
    (void) workspaceSize;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, nullptr, key, nullptr, value, dy, attentionInOptional, dqOut, nullptr, dkOut, nullptr, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional, 
        actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, pls check input shape.");
        return ret;
    }

    // 输入连续性转换
    const aclTensor *queryCngs = nullptr;
    const aclTensor *keyCngs = nullptr;
    const aclTensor *valueCngs = nullptr;
    const aclTensor *dyCngs = nullptr;
    const aclTensor *attentionInOptionalCngs = nullptr;
    const aclTensor *pseShiftOptionalCngs = nullptr;
    const aclTensor *dropMaskOptionalCngs = nullptr;
    const aclTensor *paddingMaskOptionalCngs = nullptr;
    const aclTensor *attenMaskOptionalCngs = nullptr;
    const aclTensor *softmaxMaxOptionalCngs = nullptr;
    const aclTensor *softmaxSumOptionalCngs = nullptr;
    const aclTensor *softmaxInOptionalCngs = nullptr;
    ret = ContiguousInputTensor(query, nullptr, key, nullptr, value, dy, attentionInOptional, &queryCngs, nullptr, &keyCngs, nullptr, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs,
        &attenMaskOptionalCngs, &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (strcmp(softmaxInLayout, "same_as_input") == 0) {
        ret = TransposeSoftMaxTensor(&softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, fagShape, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 调整input layout
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);

    // 调用FAG ascendc接口
    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayoutUnderTrans,
        innerPrecise, sparseMode, PSE_TYPE_V1, 0, 0, 0, softmaxInLayout, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_PARAM_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, nullptr, &dkOut, nullptr, &dvOut, &dpseOut, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreGrad,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, scaleValue, keepProb, preTokens,
                          nextTokens, headNum, inputLayout, innerPrecise, sparseMode),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // 空Tensor处理
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional, nullptr,
        nullptr, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout,
        innerPrecise, sparseMode, dqOut, dkOut, dvOut, dpseOut, defaultSoftmaxInLayout, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGrad,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          scaleValue, keepProb, preTokens, nextTokens, headNum,
                          inputLayout, innerPrecise, sparseMode),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout %s is not TND, invalid shape, pls check", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, keepProb, preTokens,
        nextTokens, headNum, inputLayout, innerPrecise, sparseMode, dqOut, dkOut, dvOut,
        dpseOut, defaultSoftmaxInLayout, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGrad);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


static aclnnStatus FlashAttentionScoreGradV2GetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    const uint64_t *workspaceSize, aclOpExecutor *executor) {
    (void) workspaceSize;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, nullptr, key, nullptr, value, dy, attentionInOptional, dqOut, nullptr, dkOut, nullptr, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional, 
        actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, pls check input shape.");
        return ret;
    }

    // 输入连续性转换
    const aclTensor *queryCngs = nullptr;
    const aclTensor *keyCngs = nullptr;
    const aclTensor *valueCngs = nullptr;
    const aclTensor *dyCngs = nullptr;
    const aclTensor *attentionInOptionalCngs = nullptr;
    const aclTensor *pseShiftOptionalCngs = nullptr;
    const aclTensor *dropMaskOptionalCngs = nullptr;
    const aclTensor *paddingMaskOptionalCngs = nullptr;
    const aclTensor *attenMaskOptionalCngs = nullptr;
    const aclTensor *softmaxMaxOptionalCngs = nullptr;
    const aclTensor *softmaxSumOptionalCngs = nullptr;
    const aclTensor *softmaxInOptionalCngs = nullptr;
    ret = ContiguousInputTensor(query, nullptr, key, nullptr, value, dy, attentionInOptional, &queryCngs, nullptr, &keyCngs, nullptr, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs,
        &attenMaskOptionalCngs, &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调整input layout
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);

    // 调用FAG ascendc接口
    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb, preTokens, nextTokens,
        headNum, inputLayoutUnderTrans, innerPrecise, sparseMode, pseType, 0, 0, 0, defaultSoftmaxInLayout, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_PARAM_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, nullptr, &dkOut, nullptr, &dvOut, &dpseOut, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreGradV2,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, qStartIdxOptional, kvStartIdxOptional,
                          scaleValue, keepProb, preTokens, nextTokens, headNum,
                          inputLayout, innerPrecise, sparseMode, pseType),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradV2GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional, nullptr,
        nullptr, qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb, preTokens,
        nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType, dqOut,
        dkOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGradV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGradV2,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
               keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise,
               sparseMode, pseType),
        DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout %s is not TND, invalid shape, pls check", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradV2GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
        keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise,
        sparseMode, pseType, dqOut, dkOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static aclnnStatus FlashAttentionScoreGradV3GetWorkspace(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *key, const aclTensor *keyRope, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut, const aclTensor *dkRopeOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    const uint64_t *workspaceSize, aclOpExecutor *executor) {
    (void) workspaceSize;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, queryRope, key, keyRope, value, dy, attentionInOptional, dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional, 
        actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, pls check input shape.");
        return ret;
    }
    ret = isSupportMultiInput(query, queryRope, key, keyRope, value, attenMaskOptional, pseShiftOptional, dropMaskOptional, keepProb, fagShape, sparseMode);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, pls check input shape.");
        return ret;
    }

    // 输入连续性转换
    const aclTensor *queryCngs = nullptr;
    const aclTensor *queryRopeCngs = nullptr;
    const aclTensor *keyCngs = nullptr;
    const aclTensor *keyRopeCngs = nullptr;
    const aclTensor *valueCngs = nullptr;
    const aclTensor *dyCngs = nullptr;
    const aclTensor *attentionInOptionalCngs = nullptr;
    const aclTensor *pseShiftOptionalCngs = nullptr;
    const aclTensor *dropMaskOptionalCngs = nullptr;
    const aclTensor *paddingMaskOptionalCngs = nullptr;
    const aclTensor *attenMaskOptionalCngs = nullptr;
    const aclTensor *softmaxMaxOptionalCngs = nullptr;
    const aclTensor *softmaxSumOptionalCngs = nullptr;
    const aclTensor *softmaxInOptionalCngs = nullptr;
    ret = ContiguousInputTensor(query, queryRope, key, keyRope, value, dy, attentionInOptional, &queryCngs, &queryRopeCngs, &keyCngs, &keyRopeCngs, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs,
        &attenMaskOptionalCngs, &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);

    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调整input layout
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);
    // 调用FAG ascendc接口

    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr, queryRopeCngs, keyRopeCngs, scaleValue, keepProb, preTokens, nextTokens, headNum,
        inputLayoutUnderTrans, innerPrecise, sparseMode, pseType, 0, 0, 0, defaultSoftmaxInLayout, executor);

    if (queryRope != nullptr && keyRope != nullptr) {
        CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr && fagRes[4] != nullptr && fagRes[5] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_PARAM_NULLPTR);
    } else {
        CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr,  // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_PARAM_NULLPTR);
    }

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dqRopeOut, &dkOut, &dkRopeOut, &dvOut, &dpseOut, fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn, const aclTensor *keyInRope, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut, const aclTensor *dkRopeOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGradV3,
        DFX_IN(query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
               keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise,
               sparseMode, pseType),
        DFX_OUT(dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut));

    // layout检查1
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout %s is not TND, invalid shape, pls check", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyIn != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionInOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradV3GetWorkspace(
        query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
        keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise,
        sparseMode, pseType, dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut, char *softmaxInLayout ,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGradV4,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          scaleValue, keepProb, preTokens, nextTokens, headNum,
                          inputLayout, innerPrecise, sparseMode, softmaxInLayout),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout %s is not TND, invalid shape, pls check", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (strcmp(softmaxInLayout, "same_as_input") != 0 && strcmp(softmaxInLayout, "") != 0 ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxInLayout %s is not same_as_input or Empty string, invalid softmaxInLayout, please check", softmaxInLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr || keyIn != nullptr || value != nullptr || dy != nullptr || attentionInOptional != nullptr || dqOut != nullptr || dkOut != nullptr || dvOut != nullptr || workspaceSize != nullptr || executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (dqOut->IsEmpty() && dkOut->IsEmpty() && dvOut->IsEmpty()) {
        if (dpseOut == nullptr || dpseOut->IsEmpty()) {
            OP_LOGD("All out tensor is empty");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 异常防护
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid HeadNum, pls check input attr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, keepProb, preTokens,
        nextTokens, headNum, inputLayout, innerPrecise, sparseMode, dqOut, dkOut, dvOut,
        dpseOut, softmaxInLayout, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV4);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
