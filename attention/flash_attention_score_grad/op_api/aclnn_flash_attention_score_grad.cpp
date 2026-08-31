/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "acl/acl_rt.h"
#include "op_common/log/log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
#define CHECK_SCALAR_TENSOR(condition) \
    do { \
        if (condition) { \
            OP_LOGW("There is a scalar tensor in the input optional parameters, and we will treat this input " \
                    "parameter as null."); \
        } \
    } while (0)

#define OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(tensor1, tensor2) \
    if (tensor1 != nullptr && tensor2 != nullptr) { \
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(tensor1, tensor2->GetViewShape(), return ACLNN_ERR_PARAM_INVALID); \
    }

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
    int64_t t1;
    int64_t t2;

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
static const uint64_t DIM_NUM_1 = 1;

char defaultSoftmaxInLayout[] = "";

static bool StrideLimited()
{
    NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return true;
    }
    return false;
}

static bool IsAscend950()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
}

static bool NeedTransposeSoftmaxTensor(const char *softmaxInLayout)
{
    return softmaxInLayout != nullptr && strcmp(softmaxInLayout, "same_as_input") == 0 && !IsAscend950();
}

static bool CheckIsNeedPad(const FagInShapeInfo &fagShape)
{
    if (fagShape.dDim == HEAD_DIM_192 && fagShape.inputLayoutStr != "TND" && fagShape.needTranspose == false) {
        OP_LOGD("D=192, Scenarios that do not require pad processing");
        return false;
    }
    if ((fagShape.dDim == HEAD_DIM_72 || fagShape.dDim == HEAD_DIM_88) && fagShape.s1Dim <= SEQ_LEN_4096 &&
        fagShape.s2Dim <= SEQ_LEN_4096 && fagShape.inputLayoutStr != "BNSD" && fagShape.inputLayoutStr != "TND" &&
        fagShape.n1Dim == fagShape.n2Dim && fagShape.needTranspose == false) {
        OP_LOGD("Scenarios that do not require pad processing");
        return false;
    }
    return true;
}

static int64_t GetSumIntArrayMaxValue(const aclIntArray *intArrayValue)
{
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

static int64_t getSeqLenQSum(const aclIntArray *actualSeqQLenOptional)
{
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

static void GetActSeqLen(const aclIntArray *cuSeqLen, int64_t &sum)
{
    if (cuSeqLen->Size() < 1) {
        return;
    }
    sum += static_cast<int64_t>((*cuSeqLen)[0]);
    for (int64_t i = 1; i < cuSeqLen->Size(); ++i) {
        int64_t cur = static_cast<int64_t>((*cuSeqLen)[i]) - static_cast<int64_t>((*cuSeqLen)[i - 1]);
        if (cur > 0) {
            sum += cur;
        }
    }
}

static bool CheckDimT(const FagInShapeInfo &fagShape, const aclIntArray *actualSeqQLenOptional,
                      const aclIntArray *actualSeqKvLenOptional, int64_t &sumSeqQLen, int64_t &sumSeqKvLen)
{
    if (actualSeqQLenOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "actualSeqQLenOptional");
        return false;
    }
    if (actualSeqKvLenOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "actualSeqKvLenOptional");
        return false;
    }
    GetActSeqLen(actualSeqQLenOptional, sumSeqQLen);
    GetActSeqLen(actualSeqKvLenOptional, sumSeqKvLen);
    if (fagShape.t1 < sumSeqQLen || fagShape.t2 < sumSeqKvLen) {
        return false;
    }
    return true;
}

static bool CheckTndIsNeedPad(const FagInShapeInfo &fagShape, const aclIntArray *actualSeqQLenOptional,
                              const aclIntArray *actualSeqKvLenOptional, int64_t dDim, double keepProb)
{
    int64_t sKvLenMax = 0;
    int64_t sQLenSum = 0;
    int64_t deterministicValue = 0;
    aclError aclRet = aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicValue);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGW("Fag aclnn unable to get system param determinstic.");
        // 如果determinstic参数获取失败，则不主动去除pad
        return true;
    }
    OP_LOGD("Fag aclnn deterministic is = %ld.", deterministicValue);
    if (fagShape.inputLayoutStr == "TND" && fagShape.dDim % HEAD_DIM_8 == 0 && deterministicValue == 0) {
        return false;
    }
    // TND并且是非确定性计算
    if (fagShape.inputLayoutStr == "TND" && deterministicValue == 0 && actualSeqQLenOptional != nullptr &&
        actualSeqKvLenOptional != nullptr) {
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

static aclnnStatus InvalidTensorDimCheck(const aclTensor *query, const aclTensor *queryRope, const aclTensor *key,
                                         const aclTensor *keyRope, const aclTensor *value, const aclTensor *dy,
                                         const aclTensor *attentionIn, const aclTensor *dq, const aclTensor *dqRope,
                                         const aclTensor *dk, const aclTensor *dkRope, const aclTensor *dv,
                                         const aclTensor *sinkIn, const aclTensor *dsinkOut)
{
    if (queryRope != nullptr && keyRope != nullptr && dqRope != nullptr && dkRope != nullptr) {
        auto queryRopeDimNum = queryRope->GetViewShape().GetDimNum();
        auto keyRopeDimNum = keyRope->GetViewShape().GetDimNum();
        auto dqRopeDimNum = dqRope->GetViewShape().GetDimNum();
        auto dkRopeDimNum = dkRope->GetViewShape().GetDimNum();
        if (queryRopeDimNum < MIN_DIM || keyRopeDimNum < MIN_DIM || dqRopeDimNum < MIN_DIM || dkRopeDimNum < MIN_DIM) {
            std::string dimsMsg = "{" + std::to_string(queryRopeDimNum) + ", " + std::to_string(keyRopeDimNum) + ", " +
                                  std::to_string(dqRopeDimNum) + ", " + std::to_string(dkRopeDimNum) + "}";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                "FlashAttentionScoreGrad", "queryRopeOptional, keyRopeOptional, dqRopeOut, dkRopeOut", dimsMsg.c_str(),
                "The shape dims of queryRopeOptional, keyRopeOptional, dqRopeOut, dkRopeOut cannot be less than 3");
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
        std::string dimsMsg = "{" + std::to_string(queryDimNum) + ", " + std::to_string(keyDimNum) + ", " +
                              std::to_string(valueDimNum) + ", " + std::to_string(dyDimNum) + ", " +
                              std::to_string(attentionInDimNum) + ", " + std::to_string(dqDimNum) + ", " +
                              std::to_string(dkDimNum) + ", " + std::to_string(dvDimNum) + "}";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            "FlashAttentionScoreGrad", "query, keyIn, value, attentionIn, dqOut, dkOut, dvOut", dimsMsg.c_str(),
            "The shape dims of query, keyIn, value, attentionIn, dqOut, dkOut, dvOut cannot be less than 3");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (sinkIn != nullptr && dsinkOut != nullptr && !sinkIn->IsEmpty() && !dsinkOut->IsEmpty()) {
        auto sinkInDim = sinkIn->GetViewShape().GetDimNum();
        auto dsinkDim = dsinkOut->GetViewShape().GetDimNum();
        if (sinkInDim != DIM_NUM_1 || dsinkDim != DIM_NUM_1) {
            std::string dimsMsg = "{" + std::to_string(sinkInDim) + ", " + std::to_string(dsinkDim) + "}";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("FlashAttentionScoreGrad", "sinkInOptional, dsinkOut",
                                                      dimsMsg.c_str(),
                                                      "The shape dims of sinkInOptional, dsinkOut must be 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus isSupportMultiInput(const aclTensor *query, const aclTensor *queryRope, const aclTensor *key,
                                       const aclTensor *keyRope, const aclTensor *value,
                                       const aclTensor *attenMaskOptional, const aclTensor *pseShiftOptional,
                                       const aclTensor *dropMaskOptional, double keepProb, FagInShapeInfo &fagShape,
                                       int64_t sparseMode)
{
    CHECK_RET((queryRope == nullptr && keyRope == nullptr) || (queryRope != nullptr && keyRope != nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    if (queryRope != nullptr && keyRope != nullptr) {
        auto kRopeDtype = keyRope->GetDataType();
        auto kRopeShape = keyRope->GetViewShape();
        auto qRopeDtype = queryRope->GetDataType();
        auto qRopeShape = queryRope->GetViewShape();
        if (qRopeShape.GetDim(DIM_NUM_2) > fagShape.dDim || kRopeShape.GetDim(DIM_NUM_2) > fagShape.dDim) {
            std::string shapesMsg = std::string("{") + op::ToString(qRopeShape).GetString() + ", " +
                                    op::ToString(kRopeShape).GetString() + "}";
            std::string reasonMsg =
                "The head dimensions of queryRope and keyRope cannot be greater than " + std::to_string(fagShape.dDim);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "queryRope, keyRope", shapesMsg.c_str(),
                                                   reasonMsg.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (qRopeDtype != ge::DataType::DT_BF16 || kRopeDtype != ge::DataType::DT_BF16) {
            std::string dtypesMsg = std::string("{") + op::ToString(DataType(qRopeDtype)).GetString() + ", " +
                                    op::ToString(DataType(kRopeDtype)).GetString() + "}";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("FlashAttentionScoreGrad", "queryRope, keyRope", dtypesMsg.c_str(),
                                                   "The dtypes of queryRope and keyRope must be BFLOAT16");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (qDtype != ge::DataType::DT_BF16 || kDtype != ge::DataType::DT_BF16 || vDtype != ge::DataType::DT_BF16) {
            std::string dtypesMsg = std::string("{") + op::ToString(DataType(qDtype)).GetString() + ", " +
                                    op::ToString(DataType(kDtype)).GetString() + ", " +
                                    op::ToString(DataType(vDtype)).GetString() + "}";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                "FlashAttentionScoreGrad", "query, keyIn, value", dtypesMsg.c_str(),
                "The dtypes of query, keyIn and value must be BFLOAT16 when rope inputs exist");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sparseMode == 6) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "sparseMode", std::to_string(sparseMode),
                                                  "sparseMode cannot be 6 when queryRope and keyRope are provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (queryRope != nullptr) {
        if (attenMaskOptional == nullptr || attenMaskOptional->GetViewShape().GetDimNum() == 0) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                "FlashAttentionScoreGrad", "attenMaskOptional",
                "attenMaskOptional must be provided and non-empty when queryRope and keyRope are provided");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (pseShiftOptional != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                "FlashAttentionScoreGrad", "pseShiftOptional",
                "pseShiftOptional must be empty when queryRope and keyRope are provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (dropMaskOptional != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                "FlashAttentionScoreGrad", "dropMaskOptional",
                "dropMaskOptional must be empty when queryRope and keyRope are provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (keepProb < 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "keepProb", std::to_string(keepProb),
                                                  "keepProb must be 1 when queryRope and keyRope are provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (fagShape.inputLayoutStr != "TND") {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "inputLayout", fagShape.inputLayoutStr,
                                                  "inputLayout must be TND when queryRope and keyRope are provided");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (fagShape.needPadDimD || fagShape.needTranspose || fagShape.needBackwordReshape || fagShape.needPadValueD) {
            std::string shapesMsg = std::string("{") + op::ToString(query->GetViewShape()).GetString() + ", " +
                                    op::ToString(queryRope->GetViewShape()).GetString() + ", " +
                                    op::ToString(key->GetViewShape()).GetString() + ", " +
                                    op::ToString(keyRope->GetViewShape()).GetString() + "}";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "query, queryRope, keyIn, keyRope", shapesMsg.c_str(),
                "The input shapes cannot require padding, transposition or backward reshape when rope inputs exist");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus GetInputShapeInfo(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                     int64_t headNum, const char *inputLayout, FagInShapeInfo &fagShape,
                                     const aclIntArray *actualSeqQLenOptional,
                                     const aclIntArray *actualSeqKvLenOptional, double keepProb)
{
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    auto valueShape = value->GetViewShape();
    auto queryDimSize = queryShape.GetShapeSize();
    auto kvDimSize = keyShape.GetShapeSize();
    fagShape.inputLayoutStr = op::ToString(inputLayout).GetString();
    bool isLayoutBNSD = (fagShape.inputLayoutStr == "BNSD");
    bool isLayoutBSND = (fagShape.inputLayoutStr == "BSND");
    bool isLayoutBSH = (fagShape.inputLayoutStr == "BSH");
    bool isLayoutSBH = (fagShape.inputLayoutStr == "SBH");
    bool isLayoutTND = (fagShape.inputLayoutStr == "TND");
    fagShape.n1Dim = isLayoutBNSD ? queryShape.GetDim(1) : queryShape.GetDim(2); // 1 or 2:n1
    fagShape.n2Dim = isLayoutBNSD ? keyShape.GetDim(1) : keyShape.GetDim(2);     // 1 or 2:n2
    fagShape.s1Dim = isLayoutBNSD ? queryShape.GetDim(2) : queryShape.GetDim(1); // 1 or 2:s1
    fagShape.s2Dim = isLayoutBNSD ? keyShape.GetDim(2) : keyShape.GetDim(1);     // 1 or 2:s2
    if (isLayoutBSH || isLayoutSBH) {
        fagShape.h1Dim = queryShape.GetDim(2); // 2:h1
        fagShape.h2Dim = keyShape.GetDim(2);   // 2:h2
        auto h3Dim = valueShape.GetDim(2);     // 3:h3，h of v
        if (headNum == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                                  "The value of headNum must be positive number");
            return ACLNN_ERR_PARAM_INVALID;
        }
        fagShape.dDim = fagShape.h1Dim / headNum; // q Head-dim
        if (fagShape.dDim == 0) {
            fagShape.n1Dim = headNum;
            fagShape.n2Dim = headNum;
            fagShape.s1Dim = isLayoutBSH ? queryShape.GetDim(1) : queryShape.GetDim(0);
            fagShape.s2Dim = isLayoutBSH ? keyShape.GetDim(1) : keyShape.GetDim(0);
            fagShape.dkDim = fagShape.h2Dim;
            fagShape.dvDim = h3Dim;
        } else {
            fagShape.n1Dim = headNum;
            fagShape.n2Dim = fagShape.h2Dim / fagShape.dDim;
            fagShape.s1Dim = isLayoutBSH ? queryShape.GetDim(1) : queryShape.GetDim(0);
            fagShape.s2Dim = isLayoutBSH ? keyShape.GetDim(1) : keyShape.GetDim(0);
            fagShape.dkDim = fagShape.n2Dim == 0 ? fagShape.dDim : keyShape.GetDim(DIM_NUM_2) / fagShape.n2Dim;
            fagShape.dvDim = fagShape.n2Dim == 0 ? 0 : valueShape.GetDim(DIM_NUM_2) / fagShape.n2Dim;
        }
    } else if (isLayoutTND) {
        fagShape.dDim = queryShape.GetDim(2);  // 2:d
        fagShape.n1Dim = queryShape.GetDim(1); // 1:n1
        fagShape.n2Dim = keyShape.GetDim(1);   // 1:n2
        fagShape.dkDim = keyShape.GetDim(DIM_NUM_2);
        fagShape.dvDim = valueShape.GetDim(DIM_NUM_2);
        fagShape.t1 = queryShape.GetDim(0);
        fagShape.t2 = keyShape.GetDim(0);
    } else if (queryShape.GetDimNum() > MIN_DIM) {
        if (!(isLayoutBNSD || isLayoutBSND)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "inputLayout",
                                                  fagShape.inputLayoutStr.c_str(),
                                                  "The value of inputLayout must be BNSD or BSND or BSH or SBH or TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // query dim > 3
        fagShape.dDim = queryShape.GetDim(3);  // 3:d
        fagShape.dkDim = keyShape.GetDim(3);   // key Head-dim
        fagShape.dvDim = valueShape.GetDim(3); // value Head-dim
    } else {
        // query dimNum <= 3 and layout is not BSH/SBH/TND
        if (!(isLayoutBNSD || isLayoutBSND)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "inputLayout",
                                                  fagShape.inputLayoutStr.c_str(),
                                                  "The value of inputLayout must be BNSD or BSND or BSH or SBH or TND");
        } else {
            // query dimNum <= 3 and layout is not BSH/SBH/TND
            std::string dimsMsg = "{" + std::to_string(queryShape.GetDimNum()) + ", " +
                                  std::to_string(keyShape.GetDimNum()) + ", " + std::to_string(valueShape.GetDimNum()) +
                                  "}";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                "FlashAttentionScoreGrad", "query, keyIn, value", dimsMsg.c_str(),
                "When inputLayout is BNSD or BSND, the shape dims of query, keyIn and value must be greater than 3");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim != fagShape.dkDim && queryShape.GetDimNum() == DIM_NUM_3) {
        std::string shapesMsg = "{[" + std::to_string(queryShape.GetDim(0)) + ", " +
                                std::to_string(queryShape.GetDim(1)) + ", " + std::to_string(queryShape.GetDim(2)) +
                                "], [" + std::to_string(keyShape.GetDim(0)) + ", " +
                                std::to_string(keyShape.GetDim(1)) + ", " + std::to_string(keyShape.GetDim(2)) + "]}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapesMsg.c_str(),
                                               "D axis of query must be equal to d axis of keyIn");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim != fagShape.dkDim && queryShape.GetDimNum() == DIM_NUM_4) {
        std::string shapesMsg = "{[" + std::to_string(queryShape.GetDim(0)) + ", " +
                                std::to_string(queryShape.GetDim(1)) + ", " + std::to_string(queryShape.GetDim(2)) +
                                ", " + std::to_string(queryShape.GetDim(3)) + "], [" +
                                std::to_string(keyShape.GetDim(0)) + ", " + std::to_string(keyShape.GetDim(1)) + ", " +
                                std::to_string(keyShape.GetDim(2)) + ", " + std::to_string(keyShape.GetDim(3)) + "]}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapesMsg.c_str(),
                                               "D axis of query must be equal to d axis of keyIn");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (fagShape.dDim < fagShape.dvDim) {
        std::string qvShapesMsg = "{[";
        for (int i = 0; i < queryShape.GetDimNum(); i++) {
            if (i < queryShape.GetDimNum() - 1) {
                qvShapesMsg += std::to_string(queryShape.GetDim(i)) + ", ";
            } else {
                qvShapesMsg += std::to_string(queryShape.GetDim(i)) + "], [";
            }
        }
        for (int i = 0; i < valueShape.GetDimNum(); i++) {
            if (i < valueShape.GetDimNum() - 1) {
                qvShapesMsg += std::to_string(valueShape.GetDim(i)) + ", ";
            } else {
                qvShapesMsg += std::to_string(valueShape.GetDim(i)) + "]}";
            }
        }
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, value", qvShapesMsg.c_str(),
                                               "D axis for query cannot be less than d axis of value");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t deterministicValue = 0;

    if (StrideLimited()) {
        aclError aclRet = aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &deterministicValue);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGW("Fag aclnn unable to get system param determinstic.");
            // 如果determinstic参数获取失败，则不主动去除pad
            deterministicValue = DIM_NUM_3;
        }
        fagShape.needPadValueD =
            (fagShape.dDim != fagShape.dvDim) &&
            (!(fagShape.dDim == HEAD_DIM_192 && fagShape.dvDim == HEAD_DIM_128 && deterministicValue == 0));
    } else {
        fagShape.needPadValueD = fagShape.dDim != fagShape.dvDim;
    }

    fagShape.querySDimStrideSize = 0;
    fagShape.kvSDimStrideSize = 0;

    if (isLayoutBSND) { // stride is N * D
        fagShape.querySDimStrideSize = fagShape.n1Dim * fagShape.dDim;
        fagShape.kvSDimStrideSize = fagShape.n2Dim * fagShape.dDim;
    } else if (isLayoutBSH) {                                // stride is H
        fagShape.querySDimStrideSize = queryShape.GetDim(2); // 2:dv
        fagShape.kvSDimStrideSize = keyShape.GetDim(2);      // 2:dv
    } else if (isLayoutSBH) {                                // stride is B * H
        fagShape.querySDimStrideSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
        fagShape.kvSDimStrideSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
    }

    fagShape.alignDim = (fagShape.dDim < ALIGN_D_DIM_SIZE) ? SPARE_ALIGN_D_DIM_SIZE : ALIGN_D_DIM_SIZE;

    if (StrideLimited()) {
        auto dDimAlignSize = (fagShape.dDim + fagShape.alignDim - 1) / fagShape.alignDim * fagShape.alignDim;

        // 判断是否需要PAD和transpose, 同时判断是否为如下特殊场景 (SBH下，只需要PAD不需要transpose)
        fagShape.needPadDimD =
            (fagShape.dDim % fagShape.alignDim != 0 && queryDimSize != 0 && kvDimSize != 0) ? true : false;

        // 计算是否超过65535时，应该使用对齐以后的D值
        if (fagShape.needPadDimD) {
            if (isLayoutBSND) { // stride is N * D
                fagShape.querySDimStrideSize = fagShape.n1Dim * dDimAlignSize;
                fagShape.kvSDimStrideSize = fagShape.n2Dim * dDimAlignSize;
            } else if (isLayoutBSH) { // stride is H
                fagShape.querySDimStrideSize =
                    fagShape.dDim == 0 ? 0 : (queryShape.GetDim(2) / fagShape.dDim * dDimAlignSize); // 2:dv
                fagShape.kvSDimStrideSize =
                    fagShape.dDim == 0 ? 0 : (keyShape.GetDim(2) / fagShape.dDim * dDimAlignSize); // 2:dv
            } else if (isLayoutSBH) {                                                              // stride is B * H
                int64_t queryBHSize = fagShape.s1Dim == 0 ? 0 : (queryDimSize / fagShape.s1Dim);
                int64_t kvBHSize = fagShape.s2Dim == 0 ? 0 : (kvDimSize / fagShape.s2Dim);
                fagShape.querySDimStrideSize = fagShape.dDim == 0 ? 0 : (queryBHSize / fagShape.dDim * dDimAlignSize);
                fagShape.kvSDimStrideSize = fagShape.dDim == 0 ? 0 : (kvBHSize / fagShape.dDim * dDimAlignSize);
            }
        }

        bool needTranspose =
            queryDimSize != 0 && kvDimSize != 0 &&
            (fagShape.inputLayoutStr != "BNSD" && fagShape.inputLayoutStr != "TND" &&
             (fagShape.querySDimStrideSize > MAX_BSN_DIMS_SIZE || fagShape.kvSDimStrideSize > MAX_BSN_DIMS_SIZE));
        fagShape.needTranspose = needTranspose;

        if (!CheckIsNeedPad(fagShape) ||
            !CheckTndIsNeedPad(fagShape, actualSeqQLenOptional, actualSeqKvLenOptional, fagShape.dDim, keepProb)) {
            fagShape.needPadDimD = false;
        }
        // 特殊情况的TND的D不等长无需处理
        if (isLayoutTND && fagShape.dDim == HEAD_DIM_192 && fagShape.dvDim == HEAD_DIM_128 && deterministicValue == 0) {
            fagShape.needPadDimD = false;
        }
        if (isLayoutTND) {
            int64_t sumSeqQLen = 0;
            int64_t sumSeqKvLen = 0;
            if (!CheckDimT(fagShape, actualSeqQLenOptional, actualSeqKvLenOptional, sumSeqQLen, sumSeqKvLen)) {
                std::string shapesMsg = "{[" + std::to_string(fagShape.t1) + ", " + std::to_string(fagShape.n1Dim) +
                                        ", " + std::to_string(fagShape.dDim) + "], [" + std::to_string(fagShape.t2) +
                                        ", " + std::to_string(fagShape.n2Dim) + ", " + std::to_string(fagShape.dDim) +
                                        "]}";
                std::string reasonMsg = "T axis of query and keyIn cannot be less than " + std::to_string(sumSeqQLen) +
                                        " and " + std::to_string(sumSeqKvLen);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn", shapesMsg.c_str(),
                                                       reasonMsg.c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验EOD场景尾部是否补0
            if (fagShape.t1 > sumSeqQLen && fagShape.t2 > sumSeqKvLen) {
                int64_t actSeqQLenEnd =
                    static_cast<int64_t>((*actualSeqQLenOptional)[actualSeqQLenOptional->Size() - 1]);
                int64_t actSeqKvLenEnd =
                    static_cast<int64_t>((*actualSeqKvLenOptional)[actualSeqKvLenOptional->Size() - 1]);
                if (actSeqQLenEnd != 0 || actSeqKvLenEnd != 0) {
                    std::string valuesMsg = std::to_string(actSeqQLenEnd) + ", " + std::to_string(actSeqKvLenEnd);
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "FlashAttentionScoreGrad", "actualSeqQLenOptional, actualSeqKvLenOptional", valuesMsg.c_str(),
                        "The values of the actualSeqQLenOptional end and actualSeqKvLenOptional end must be 0");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
    } else {
        fagShape.needPadDimD = false;
        fagShape.needTranspose = false;
    }

    fagShape.passThrowInnerFag = (!(fagShape.needPadDimD) && !(fagShape.needTranspose));
    fagShape.needBackwordReshape = (isLayoutSBH && fagShape.needPadDimD && !(fagShape.needTranspose));
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
                                         const aclTensor *keyRope, const aclTensor *value, const aclTensor *dy,
                                         const aclTensor *attentionInOptional, const aclTensor **queryCngs,
                                         const aclTensor **queryRopeCngs, const aclTensor **keyCngs,
                                         const aclTensor **keyRopeCngs, const aclTensor **valueCngs,
                                         const aclTensor **dyCngs, const aclTensor **attentionInOptionalCngs,
                                         aclOpExecutor *executor)
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
    const aclTensor *softmaxInOptional, const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional,
    const aclTensor *dScaleDyOptional, const aclTensor *dScaleOOptional, const aclTensor *sinkInOptional,
    const aclTensor **pseShiftOptionalCngs, const aclTensor **dropMaskOptionalCngs,
    const aclTensor **paddingMaskOptionalCngs, const aclTensor **attenMaskOptionalCngs,
    const aclTensor **softmaxMaxOptionalCngs, const aclTensor **softmaxSumOptionalCngs,
    const aclTensor **softmaxInOptionalCngs, const aclTensor **queryRopeOptionalCngs,
    const aclTensor **keyRopeOptionalCngs, const aclTensor **dScaleQOptionalCngs, const aclTensor **dScaleKOptionalCngs,
    const aclTensor **dScaleVOptionalCngs, const aclTensor **dScaleDyOptionalCngs,
    const aclTensor **dScaleOOptionalCngs, const aclTensor **sinkInOptionalCngs, aclOpExecutor *executor)
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

    // sinkInOptional如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(sinkInOptional, sinkInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    if (!StrideLimited()) {
        // queryRopeOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(queryRopeOptional, queryRopeOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        // keyRopeOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(keyRopeOptional, keyRopeOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        // dScaleQOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(dScaleQOptional, dScaleQOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        // dScaleKOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(dScaleKOptional, dScaleKOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        // dScaleVOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(dScaleVOptional, dScaleVOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        // dScaleDyOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(dScaleDyOptional, dScaleDyOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        // dScaleOOptional如果非连续，需要转连续
        ret = ContiguousTensorWithCheck(dScaleOOptional, dScaleOOptionalCngs, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ret;
}

static aclnnStatus ContiguousQuantOptionalInputTensor(const aclTensor *dsScaleOptional, const aclTensor *pScaleOptional,
                                                      const aclTensor **dsScaleOptionalCngs,
                                                      const aclTensor **pScaleOptionalCngs, aclOpExecutor *executor)
{
    auto ret = ACLNN_SUCCESS;

    // 如果非连续，需要转连续
    ret = ContiguousTensorWithCheck(dsScaleOptional, dsScaleOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    ret = ContiguousTensorWithCheck(pScaleOptional, pScaleOptionalCngs, executor);
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

static void GetKvUnequalReshapeArray(const aclTensor *value, FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                     aclOpExecutor *executor)
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

    fagShapeArray.valueReshapeBefore =
        executor->AllocIntArray(valueReshapeBeforeList.data(), valueReshapeBeforeList.size());
    fagShapeArray.valueReshapeAfter =
        executor->AllocIntArray(valueReshapeAfterList.data(), valueReshapeAfterList.size());

    fagShapeArray.attenInReshapeBefore =
        executor->AllocIntArray(attenInReshapeBeforeList.data(), attenInReshapeBeforeList.size());
    fagShapeArray.attenInReshapeAfter =
        executor->AllocIntArray(attenInReshapeAfterList.data(), attenInReshapeAfterList.size());

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
    OP_CHECK(*query != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape query failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);
    *key = l0op::Reshape(*key, keyShapeArray, executor);
    OP_CHECK(*key != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape keyIn failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);
    *value = l0op::Reshape(*value, keyShapeArray, executor);
    OP_CHECK(*value != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape value failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);
    *dy = l0op::Reshape(*dy, queryShapeArray, executor);
    OP_CHECK(*dy != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dy failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Reshape(*attentionInOptional, queryShapeArray, executor);
        OP_CHECK(*attentionInOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The op [FlashAttentionScoreGrad] received bad params, "
                                                  "the reason is: [Reshape attentionInOptional failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
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
    OP_CHECK(fagOut[0] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dq_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);
    fagOut[1] = l0op::Reshape(fagOut[1], dkShapeArray, executor);
    OP_CHECK(fagOut[1] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dk_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);
    fagOut[2] = l0op::Reshape(fagOut[2], dkShapeArray, executor); // 2:dv
    OP_CHECK(fagOut[2] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dv_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR); // 2:dv

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
    OP_CHECK(*query != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Pad query failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // key
    *key = l0op::Pad(*key, padTensor, executor);
    OP_CHECK(*key != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Pad keyIn failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // value
    *value = l0op::Pad(*value, padTensor, executor);
    OP_CHECK(*value != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Pad value failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // dy
    *dy = l0op::Pad(*dy, padTensor, executor);
    OP_CHECK(*dy != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Pad dy failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // attenmask_in
    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Pad(*attentionInOptional, padTensor, executor);
        OP_CHECK(*attentionInOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The op [FlashAttentionScoreGrad] received bad params, "
                                                  "the reason is: [Pad attentionInOptional failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
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
    OP_CHECK(fagOut[0] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Slice dq_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    dkOutSizeVector.emplace_back(fagShape.dDim);
    auto dkOutSize = executor->AllocIntArray(dkOutSizeVector.data(), dkOutSizeVector.size());
    fagOut[1] = l0op::Slice(fagOut[1], offsets, dkOutSize, executor); // 1: dk
    OP_CHECK(fagOut[1] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Slice dk_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);
    fagOut[2] = l0op::Slice(fagOut[2], offsets, dkOutSize, executor); // 2: dv
    OP_CHECK(fagOut[2] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Slice dv_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "The op [FlashAttentionScoreGrad] received bad params, the reason is: [AllocIntArray padVec failed]");
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
        OP_CHECK(*value != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape value failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
        *dy = l0op::Reshape(*dy, fagShapeArray.attenInReshapeBefore, executor);

        OP_CHECK(*dy != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dy failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
        *attentionInOptional = l0op::Reshape(*attentionInOptional, fagShapeArray.attenInReshapeBefore, executor);
        OP_CHECK(*attentionInOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The op [FlashAttentionScoreGrad] received bad params, "
                                                  "the reason is: [Reshape attentionInOptional failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }

    int32_t dimNum = (fagShape.inputLayoutStr == "TND") ? DIM_NUM_3 : DIM_NUM_4;
    auto paddings = GeneratePaddings(dimNum, fagShape.dDim - fagShape.dvDim, executor);
    *value = l0op::Pad(*value, paddings, executor);
    OP_CHECK(*value != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Pad value failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    *dy = l0op::Pad(*dy, paddings, executor);
    OP_CHECK(*dy != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Pad dy failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    *attentionInOptional = l0op::Pad(*attentionInOptional, paddings, executor);
    OP_CHECK(*attentionInOptional != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The op [FlashAttentionScoreGrad] received bad params, "
                                              "the reason is: [Pad attentionInOptional failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        *value = l0op::Reshape(*value, fagShapeArray.valueReshapeAfter, executor);
        OP_CHECK(*value != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape value failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
        *dy = l0op::Reshape(*dy, fagShapeArray.attenInReshapeAfter, executor);
        OP_CHECK(*dy != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dy failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
        *attentionInOptional = l0op::Reshape(*attentionInOptional, fagShapeArray.attenInReshapeAfter, executor);
        OP_CHECK(*attentionInOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The op [FlashAttentionScoreGrad] received bad params, "
                                                  "the reason is: [Reshape attentionInOptional failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus SliceDvOut(std::array<const aclTensor *, l0op::MAX_FAG_OUTPUT_CNT> &fagOut, FagInShapeInfo fagShape,
                              FagShapeArray fagShapeArray, aclOpExecutor *executor)
{
    if (!(fagShape.needPadValueD)) {
        return ACLNN_SUCCESS;
    }

    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        fagOut[2] = l0op::Reshape(fagOut[2], fagShapeArray.dvReshapeBefore, executor);
        OP_CHECK(
            fagOut[2] != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                    "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dv_out failed]"),
            return ACLNN_ERR_PARAM_NULLPTR);
    }

    FVector<int64_t, MAX_DIM_NUM> dvOutSizeVector = ToShapeVector(fagOut[2]->GetViewShape());
    dvOutSizeVector.back() -= fagShape.dDim - fagShape.dvDim;
    auto dvOutSize = executor->AllocIntArray(dvOutSizeVector.data(), dvOutSizeVector.size());
    if (fagShape.inputLayoutStr != "TND") {
        FVector<int64_t, DIM_NUM_4> offsets(DIM_NUM_4, 0);
        fagOut[2] = l0op::Slice(fagOut[2], executor->AllocIntArray(offsets.data(), offsets.size()), dvOutSize,
                                executor); // 2: dv
        OP_CHECK(fagOut[2] != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Slice dv_out failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    } else {
        FVector<int64_t, DIM_NUM_3> offsets(DIM_NUM_3, 0);
        fagOut[2] = l0op::Slice(fagOut[2], executor->AllocIntArray(offsets.data(), offsets.size()), dvOutSize,
                                executor); // 2: dv
        OP_CHECK(fagOut[2] != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Slice dv_out failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (fagShape.inputLayoutStr == "SBH" || fagShape.inputLayoutStr == "BSH") {
        fagOut[2] = l0op::Reshape(fagOut[2], fagShapeArray.dvReshapeAfter, executor);
        OP_CHECK(
            fagOut[2] != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                    "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Reshape dv_out failed]"),
            return ACLNN_ERR_PARAM_NULLPTR);
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
    OP_CHECK(*query != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose query failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // key
    *key = l0op::Transpose(*key, perm, executor);
    OP_CHECK(*key != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose keyIn failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // value
    *value = l0op::Transpose(*value, perm, executor);
    OP_CHECK(*value != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose value failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // dy
    *dy = l0op::Transpose(*dy, perm, executor);
    OP_CHECK(*dy != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose dy failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // attentionInOptional
    if (*attentionInOptional != nullptr && (*attentionInOptional)->GetViewShape().GetDimNum() != 0) {
        *attentionInOptional = l0op::Transpose(*attentionInOptional, perm, executor);
        OP_CHECK(*attentionInOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The op [FlashAttentionScoreGrad] received bad params, "
                                                  "the reason is: [Transpose attentionInOptional failed]"),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus TransposeSoftMaxTensor(const aclTensor **softmax, const aclTensor **softmaxSum,
                                          FagInShapeInfo fagShape, aclOpExecutor *executor)
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
    OP_CHECK(fagOut[0] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose dq_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // dkOut
    fagOut[1] = l0op::Transpose(fagOut[1], perm, executor);
    OP_CHECK(fagOut[1] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose dk_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // dvOut
    fagOut[2] = l0op::Transpose(fagOut[2], perm, executor); // 2:dvOut
    OP_CHECK(fagOut[2] != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                     "The op [FlashAttentionScoreGrad] received bad params, the reason is: [Transpose dv_out failed]"),
             return ACLNN_ERR_PARAM_NULLPTR); // 2:dvOut

    // dpseOut
    return ACLNN_SUCCESS;
}

static aclnnStatus InputDtypeCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                   const aclTensor *dy)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    auto dyDtype = dy->GetDataType();
    if (!(qDtype == op::DataType::DT_FLOAT8_E4M3FN || qDtype == op::DataType::DT_FLOAT8_E5M2 ||
          qDtype == op::DataType::DT_HIFLOAT8) &&
        (qDtype != kDtype || kDtype != vDtype || vDtype != dyDtype)) {
        std::string dTypesMsg = std::string("{") + op::ToString(DataType(qDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(kDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(vDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(dyDtype)).GetString() + std::string("}");
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "FlashAttentionScoreGrad", "query, keyIn, value, dy", dTypesMsg.c_str(),
            "when the dtype of query is  not FLOAT8_E5M2 or FLOAT8_E4M3FN or HIFLOAT8, "
            "the dtypes of query, keyIn, value, dy must be the same");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (StrideLimited() &&
        !(qDtype == op::DataType::DT_FLOAT || qDtype == op::DataType::DT_FLOAT16 || qDtype == op::DataType::DT_BF16)) {
        std::string dTypesMsg = std::string("{") + op::ToString(DataType(qDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(kDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(vDtype)).GetString() + std::string("}");
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "FlashAttentionScoreGrad", "query, keyIn, value", dTypesMsg.c_str(),
            "the dtypes of query, keyIn, value, dy must be FLOAT32 or FLOAT16 or BFLOAT16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!StrideLimited() && !(qDtype == op::DataType::DT_FLOAT || qDtype == op::DataType::DT_FLOAT16 ||
                              qDtype == op::DataType::DT_BF16 || qDtype == op::DataType::DT_FLOAT8_E4M3FN ||
                              qDtype == op::DataType::DT_FLOAT8_E5M2 || qDtype == op::DataType::DT_HIFLOAT8)) {
        std::string dTypesMsg = std::string("{") + op::ToString(DataType(qDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(kDtype)).GetString() + std::string(", ") +
                                op::ToString(DataType(vDtype)).GetString() + std::string("}");
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "FlashAttentionScoreGrad", "query, keyIn, value", dTypesMsg.c_str(),
            "the dtypes of query, keyIn, value, dy must be FLOAT32 or FLOAT16 or BFLOAT16 or FLOAT8_E5M2 or "
            "FLOAT8_E4M3FN or HIFLOAT8");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool IsSameShapeExceptLastDim(const op::Shape &leftShape, const op::Shape &rightShape)
{
    if (leftShape.GetDimNum() != rightShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i + 1 < leftShape.GetDimNum(); ++i) {
        if (leftShape.GetDim(i) != rightShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static aclnnStatus InputOutputShapeCheck(const aclTensor *query, const aclTensor *queryRope, const aclTensor *key,
                                         const aclTensor *keyRope, const aclTensor *value, const aclTensor *dy,
                                         const aclTensor *attentionIn, const aclTensor *dq, const aclTensor *dqRope,
                                         const aclTensor *dk, const aclTensor *dkRope, const aclTensor *dv,
                                         const aclTensor *sinkIn, const aclTensor *dsinkOut, const char *inputLayout)
{
    auto queryShape = query->GetViewShape();
    auto keyShape = key->GetViewShape();
    auto valueShape = value->GetViewShape();
    auto dyShape = dy->GetViewShape();
    auto attentionInShape = attentionIn->GetViewShape();
    auto dqShape = dq->GetViewShape();
    auto dkShape = dk->GetViewShape();
    auto dvShape = dv->GetViewShape();
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    auto expectedDimNum = (inputLayoutStr == "BNSD" || inputLayoutStr == "BSND") ? DIM_NUM_4 : DIM_NUM_3;
    if (!(inputLayoutStr == "BNSD" || inputLayoutStr == "BSND" || inputLayoutStr == "BSH" || inputLayoutStr == "SBH" ||
          inputLayoutStr == "TND")) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayoutStr,
                                   "BNSD, BSND, BSH, SBH or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (queryShape.GetDimNum() != expectedDimNum || keyShape.GetDimNum() != expectedDimNum ||
        valueShape.GetDimNum() != expectedDimNum || dyShape.GetDimNum() != expectedDimNum ||
        attentionInShape.GetDimNum() != expectedDimNum || dqShape.GetDimNum() != expectedDimNum ||
        dkShape.GetDimNum() != expectedDimNum || dvShape.GetDimNum() != expectedDimNum) {
        std::string dimsMsg =
            "{" + std::to_string(queryShape.GetDimNum()) + ", " + std::to_string(keyShape.GetDimNum()) + ", " +
            std::to_string(valueShape.GetDimNum()) + ", " + std::to_string(dyShape.GetDimNum()) + ", " +
            std::to_string(attentionInShape.GetDimNum()) + ", " + std::to_string(dqShape.GetDimNum()) + ", " +
            std::to_string(dkShape.GetDimNum()) + ", " + std::to_string(dvShape.GetDimNum()) + "}";
        std::string reasonMsg = "The shape dimensions must all be " + std::to_string(expectedDimNum) +
                                " when inputLayout is " + inputLayoutStr;
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("FlashAttentionScoreGrad",
                                                  "query, keyIn, value, dy, attentionInOptional, dqOut, dkOut, dvOut",
                                                  dimsMsg.c_str(), reasonMsg.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!IsSameShapeExceptLastDim(keyShape, valueShape)) {
        std::string shapesMsg =
            std::string("{") + op::ToString(keyShape).GetString() + ", " + op::ToString(valueShape).GetString() + "}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "FlashAttentionScoreGrad", "keyIn, value", shapesMsg.c_str(),
            "The shapes of keyIn and value must be the same except for the last dimension");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsSameShapeExceptLastDim(queryShape, dyShape)) {
        std::string shapesMsg =
            std::string("{") + op::ToString(queryShape).GetString() + ", " + op::ToString(dyShape).GetString() + "}";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "FlashAttentionScoreGrad", "query, dy", shapesMsg.c_str(),
            "The shapes of query and dy must be the same except for the last dimension");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (inputLayoutStr == "BNSD" || inputLayoutStr == "BSND" || inputLayoutStr == "BSH") {
        if (!(queryShape.GetDim(0) == keyShape.GetDim(0) && keyShape.GetDim(0) == valueShape.GetDim(0))) {
            std::string shapesMsg = std::string("{") + op::ToString(queryShape).GetString() + ", " +
                                    op::ToString(keyShape).GetString() + ", " + op::ToString(valueShape).GetString() +
                                    "}";
            std::string reasonMsg = "The batch dimensions of query, keyIn and value must be the same when "
                                    "inputLayout is " +
                                    inputLayoutStr;
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("FlashAttentionScoreGrad", "query, keyIn, value", shapesMsg.c_str(),
                                                   reasonMsg.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (inputLayoutStr == "SBH") {
        if (!(queryShape.GetDim(1) == keyShape.GetDim(1) && keyShape.GetDim(1) == valueShape.GetDim(1))) {
            std::string shapesMsg = std::string("{") + op::ToString(queryShape).GetString() + ", " +
                                    op::ToString(keyShape).GetString() + ", " + op::ToString(valueShape).GetString() +
                                    "}";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "query, keyIn, value", shapesMsg.c_str(),
                "The batch dimensions of query, keyIn and value must be the same when inputLayout is SBH");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(attentionIn, dyShape, return ACLNN_ERR_PARAM_INVALID);

    if (queryRope != nullptr && keyRope != nullptr && dqRope != nullptr && dkRope != nullptr) {
        auto queryRopeShape = queryRope->GetViewShape();
        auto keyRopeShape = keyRope->GetViewShape();
        auto dqRopeShape = dqRope->GetViewShape();
        auto dkRopeShape = dkRope->GetViewShape();
        if (queryRopeShape.GetDimNum() != DIM_NUM_3 || keyRopeShape.GetDimNum() != DIM_NUM_3 ||
            dqRopeShape.GetDimNum() != DIM_NUM_3 || dkRopeShape.GetDimNum() != DIM_NUM_3) {
            std::string dimsMsg =
                "{" + std::to_string(queryRopeShape.GetDimNum()) + ", " + std::to_string(keyRopeShape.GetDimNum()) +
                ", " + std::to_string(dqRopeShape.GetDimNum()) + ", " + std::to_string(dkRopeShape.GetDimNum()) + "}";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("FlashAttentionScoreGrad",
                                                      "queryRope, keyRope, dqRopeOut, dkRopeOut", dimsMsg.c_str(),
                                                      "The shape dimensions of all rope tensors must be 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(queryRopeShape.GetDim(0) == queryShape.GetDim(0) && queryRopeShape.GetDim(1) == queryShape.GetDim(1) &&
              queryRopeShape.GetDim(DIM_NUM_2) <= queryShape.GetDim(DIM_NUM_2))) {
            std::string shapesMsg = std::string("{") + op::ToString(queryRopeShape).GetString() + ", " +
                                    op::ToString(queryShape).GetString() + "}";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "queryRope, query", shapesMsg.c_str(),
                "The T and N dimensions must be the same and the D dimension of queryRope cannot exceed query");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(keyRopeShape.GetDim(0) == keyShape.GetDim(0) && keyRopeShape.GetDim(1) == keyShape.GetDim(1) &&
              keyRopeShape.GetDim(DIM_NUM_2) <= keyShape.GetDim(DIM_NUM_2))) {
            std::string shapesMsg = std::string("{") + op::ToString(keyRopeShape).GetString() + ", " +
                                    op::ToString(keyShape).GetString() + "}";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "keyRope, keyIn", shapesMsg.c_str(),
                "The T and N dimensions must be the same and the D dimension of keyRope cannot exceed keyIn");
            return ACLNN_ERR_PARAM_INVALID;
        }
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(dqRope, queryRope->GetViewShape(), return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(dkRope, keyRope->GetViewShape(), return ACLNN_ERR_PARAM_INVALID);
    }

    if (sinkIn != nullptr && dsinkOut != nullptr && !sinkIn->IsEmpty() && !dsinkOut->IsEmpty()) {
        if (sinkIn->GetViewShape().GetDimNum() != DIM_NUM_1 || dsinkOut->GetViewShape().GetDimNum() != DIM_NUM_1) {
            std::string dimsMsg = "{" + std::to_string(sinkIn->GetViewShape().GetDimNum()) + ", " +
                                  std::to_string(dsinkOut->GetViewShape().GetDimNum()) + "}";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("FlashAttentionScoreGrad", "sinkInOptional, dsinkOut",
                                                      dimsMsg.c_str(),
                                                      "The shape dimensions of sinkInOptional and dsinkOut must be 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(dsinkOut, sinkIn->GetViewShape(), return ACLNN_ERR_PARAM_INVALID);
    }

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(dq, queryShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(dk, keyShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(dv, valueShape, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus InputOutputDtypeAndShapeCheck(const aclTensor *query, const aclTensor *queryRope,
                                                 const aclTensor *key, const aclTensor *keyRope, const aclTensor *value,
                                                 const aclTensor *dy, const aclTensor *attentionIn, const aclTensor *dq,
                                                 const aclTensor *dqRope, const aclTensor *dk, const aclTensor *dkRope,
                                                 const aclTensor *dv, const aclTensor *sinkIn,
                                                 const aclTensor *dsinkOut, const char *inputLayout)
{
    auto ret = InputDtypeCheck(query, key, value, dy);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputOutputShapeCheck(query, queryRope, key, keyRope, value, dy, attentionIn, dq, dqRope, dk, dkRope, dv,
                                sinkIn, dsinkOut, inputLayout);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto dqDtype = dq->GetDataType();
    auto dkDtype = dk->GetDataType();
    auto dvDtype = dv->GetDataType();
    if (!(dqDtype == query->GetDataType() && dkDtype == key->GetDataType() && dvDtype == value->GetDataType())) {
        std::string dtypesMsg = std::string("{") + op::ToString(DataType(query->GetDataType())).GetString() + ", " +
                                op::ToString(DataType(key->GetDataType())).GetString() + ", " +
                                op::ToString(DataType(value->GetDataType())).GetString() + ", " +
                                op::ToString(DataType(dqDtype)).GetString() + ", " +
                                op::ToString(DataType(dkDtype)).GetString() + ", " +
                                op::ToString(DataType(dvDtype)).GetString() + "}";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "FlashAttentionScoreGrad", "query, keyIn, value, dqOut, dkOut, dvOut", dtypesMsg.c_str(),
            "The dtypes of dqOut, dkOut and dvOut must match query, keyIn and value respectively");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (queryRope != nullptr && keyRope != nullptr && dqRope != nullptr && dkRope != nullptr) {
        if (!(dqRope->GetDataType() == queryRope->GetDataType() && dkRope->GetDataType() == keyRope->GetDataType())) {
            std::string dtypesMsg = std::string("{") + op::ToString(DataType(queryRope->GetDataType())).GetString() +
                                    ", " + op::ToString(DataType(keyRope->GetDataType())).GetString() + ", " +
                                    op::ToString(DataType(dqRope->GetDataType())).GetString() + ", " +
                                    op::ToString(DataType(dkRope->GetDataType())).GetString() + "}";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                "FlashAttentionScoreGrad", "queryRope, keyRope, dqRopeOut, dkRopeOut", dtypesMsg.c_str(),
                "The dtypes of dqRopeOut and dkRopeOut must match queryRope and keyRope respectively");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (sinkIn != nullptr && dsinkOut != nullptr && !sinkIn->IsEmpty() && !dsinkOut->IsEmpty()) {
        if (!(sinkIn->GetDataType() == op::DataType::DT_FLOAT && dsinkOut->GetDataType() == op::DataType::DT_FLOAT)) {
            std::string dtypesMsg = std::string("{") + op::ToString(DataType(sinkIn->GetDataType())).GetString() +
                                    ", " + op::ToString(DataType(dsinkOut->GetDataType())).GetString() + "}";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("FlashAttentionScoreGrad", "sinkInOptional, dsinkOut",
                                                   dtypesMsg.c_str(),
                                                   "The dtypes of sinkInOptional and dsinkOut must be FLOAT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus PreFlashAttentionScoreGrad(const aclTensor **query, const aclTensor **key, const aclTensor **value,
                                              const aclTensor **dy, const aclTensor **attentionInOptional,
                                              FagInShapeInfo fagShape, FagShapeArray &fagShapeArray,
                                              aclOpExecutor *executor)
{
    // 输入dtype异常拦截校验
    CHECK_RET(InputDtypeCheck(*query, *key, *value, *dy) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // 获取reshape array, SBH特殊场景下，需要提前获取调用FAG前反向reshape成SBH时所需的reshape array
    GetInputAndOutputReshapeArray(*query, *key, fagShape, fagShapeArray, executor);
    GetInputAndOutputBackwordReshapeArrayForSBH(*query, *key, fagShape, fagShapeArray, executor);
    GetKvUnequalReshapeArray(*value, fagShape, fagShapeArray, executor);

    auto ret = ACLNN_SUCCESS;
    // 特定情况下，KV Dim不等长时，将HeadDim Padding到与K相同
    if (StrideLimited()) {
        ret = PaddingValueDim(value, dy, attentionInOptional, fagShape, fagShapeArray, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

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
                                               const aclTensor **dsinkOut, FagInShapeInfo fagShape,
                                               FagShapeArray &fagShapeArray, aclOpExecutor *executor)
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
    if (StrideLimited()) {
        ret = SliceDvOut(fagOut, fagShape, fagShapeArray, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto dqOutViewCopyRes = l0op::ViewCopy(fagOut[0], *dqOut, executor);
    OP_CHECK(
        dqOutViewCopyRes != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "the dqOutViewCopyRes cannot be nullptr, dqOut is invalid, "
                "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                op::ToString((*dqOut)->GetDataType()).GetString(), op::ToString((fagOut[0])->GetDataType()).GetString(),
                op::ToString((*dqOut)->GetStorageFormat()).GetString(),
                op::ToString((fagOut[0])->GetStorageFormat()).GetString(),
                op::ToString((*dqOut)->GetViewShape()).GetString(),
                op::ToString((fagOut[0])->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_NULLPTR);
    auto dkOutViewCopyRes = l0op::ViewCopy(fagOut[1], *dkOut, executor);
    OP_CHECK(
        dkOutViewCopyRes != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "the dkOutViewCopyRes cannot be nullptr, dkOut is invalid, "
                "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                op::ToString((*dkOut)->GetDataType()).GetString(), op::ToString((fagOut[1])->GetDataType()).GetString(),
                op::ToString((*dkOut)->GetStorageFormat()).GetString(),
                op::ToString((fagOut[1])->GetStorageFormat()).GetString(),
                op::ToString((*dkOut)->GetViewShape()).GetString(),
                op::ToString((fagOut[1])->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_NULLPTR);
    auto dvOutViewCopyRes = l0op::ViewCopy(fagOut[2], *dvOut, executor);
    OP_CHECK(
        dvOutViewCopyRes != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "the dvOutViewCopyRes cannot be nullptr, dvOut is invalid, "
                "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                op::ToString((*dvOut)->GetDataType()).GetString(), op::ToString((fagOut[2])->GetDataType()).GetString(),
                op::ToString((*dvOut)->GetStorageFormat()).GetString(),
                op::ToString((fagOut[2])->GetStorageFormat()).GetString(),
                op::ToString((*dvOut)->GetViewShape()).GetString(),
                op::ToString((fagOut[2])->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_NULLPTR);
    if (!(dpseOut == nullptr || *dpseOut == nullptr || (*dpseOut)->GetDataType() == ge::DataType::DT_FLOAT)) {
        auto dpseOutViewCopyRes = l0op::ViewCopy(fagOut[3], *dpseOut, executor);
        OP_CHECK(dpseOutViewCopyRes != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "the dpseOutViewCopyRes cannot be nullptr, dpseOut is invalid, "
                         "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                         op::ToString((*dpseOut)->GetDataType()).GetString(),
                         op::ToString((fagOut[3])->GetDataType()).GetString(),
                         op::ToString((*dpseOut)->GetStorageFormat()).GetString(),
                         op::ToString((fagOut[3])->GetStorageFormat()).GetString(),
                         op::ToString((*dpseOut)->GetViewShape()).GetString(),
                         op::ToString((fagOut[3])->GetViewShape()).GetString()),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }

    if (dqRopeOut != nullptr && *dqRopeOut != nullptr &&
        !((*dqRopeOut)->GetViewShape().GetDimNum() == 1 && (*dqRopeOut)->GetViewShape()[0] == 0)) {
        auto dqRopeOutViewCopyRes = l0op::ViewCopy(fagOut[4], *dqRopeOut, executor);
        OP_CHECK(dqRopeOutViewCopyRes != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "the dqRopeOutViewCopyRes cannot be nullptr, dqRopeOut is invalid, "
                         "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                         op::ToString((*dqRopeOut)->GetDataType()).GetString(),
                         op::ToString((fagOut[4])->GetDataType()).GetString(),
                         op::ToString((*dqRopeOut)->GetStorageFormat()).GetString(),
                         op::ToString((fagOut[4])->GetStorageFormat()).GetString(),
                         op::ToString((*dqRopeOut)->GetViewShape()).GetString(),
                         op::ToString((fagOut[4])->GetViewShape()).GetString()),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dkRopeOut != nullptr && *dkRopeOut != nullptr &&
        !((*dkRopeOut)->GetViewShape().GetDimNum() == 1 && (*dkRopeOut)->GetViewShape()[0] == 0)) {
        auto dkRopeOutViewCopyRes = l0op::ViewCopy(fagOut[5], *dkRopeOut, executor);
        OP_CHECK(dkRopeOutViewCopyRes != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "the dkRopeOutViewCopyRes cannot be nullptr, dkRopeOut is invalid, "
                         "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                         op::ToString((*dkRopeOut)->GetDataType()).GetString(),
                         op::ToString((fagOut[5])->GetDataType()).GetString(),
                         op::ToString((*dkRopeOut)->GetStorageFormat()).GetString(),
                         op::ToString((fagOut[5])->GetStorageFormat()).GetString(),
                         op::ToString((*dkRopeOut)->GetViewShape()).GetString(),
                         op::ToString((fagOut[5])->GetViewShape()).GetString()),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dsinkOut != nullptr && *dsinkOut != nullptr &&
        !((*dsinkOut)->GetViewShape().GetDimNum() == 1 && (*dsinkOut)->GetViewShape()[0] == 0)) {
        auto dsinkOutViewCopyRes = l0op::ViewCopy(fagOut[6], *dsinkOut, executor);
        OP_CHECK(dsinkOutViewCopyRes != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                         "the dsinkOutViewCopyRes cannot be nullptr, dsinkOut is invalid, "
                         "dtype[%s] should be %s, format[%s] should be %s, shape[%s] should be %s",
                         op::ToString((*dsinkOut)->GetDataType()).GetString(),
                         op::ToString((fagOut[6])->GetDataType()).GetString(),
                         op::ToString((*dsinkOut)->GetStorageFormat()).GetString(),
                         op::ToString((fagOut[6])->GetStorageFormat()).GetString(),
                         op::ToString((*dsinkOut)->GetViewShape()).GetString(),
                         op::ToString((fagOut[6])->GetViewShape()).GetString()),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus FlashAttentionScoreGradGetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, char *softmaxInLayout, const uint64_t *workspaceSize, aclOpExecutor *executor)
{
    (void)workspaceSize;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, nullptr, key, nullptr, value, dy, attentionInOptional, dqOut, nullptr,
                                     dkOut, nullptr, dvOut, nullptr, nullptr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional,
                            actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
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
    ret = ContiguousInputTensor(query, nullptr, key, nullptr, value, dy, attentionInOptional, &queryCngs, nullptr,
                                &keyCngs, nullptr, &valueCngs, &dyCngs, &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs, &attenMaskOptionalCngs,
        &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // reshape + PAD + Transpose
    FagShapeArray fagShapeArray;
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (NeedTransposeSoftmaxTensor(softmaxInLayout)) {
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
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb,
        preTokens, nextTokens, headNum, inputLayoutUnderTrans, innerPrecise, sparseMode, PSE_TYPE_V1, 0, 0, 0,
        softmaxInLayout, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr, // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_PARAM_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, nullptr, &dkOut, nullptr, &dvOut, &dpseOut, nullptr, fagShape,
                                      fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionScoreGrad,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode),
        DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet = InputOutputDtypeAndShapeCheck(query, nullptr, keyIn, nullptr, value, dy, attentionInOptional, dqOut,
                                                  nullptr, dkOut, nullptr, dvOut, nullptr, nullptr, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional, nullptr,
        nullptr, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, dqOut,
        dkOut, dvOut, dpseOut, defaultSoftmaxInLayout, workspaceSize, uniqueExecutor.get());
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
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionUnpaddingScoreGrad,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout, "TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet = InputOutputDtypeAndShapeCheck(query, nullptr, keyIn, nullptr, value, dy, attentionInOptional, dqOut,
                                                  nullptr, dkOut, nullptr, dvOut, nullptr, nullptr, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, keepProb, preTokens, nextTokens, headNum,
        inputLayout, innerPrecise, sparseMode, dqOut, dkOut, dvOut, dpseOut, defaultSoftmaxInLayout, workspaceSize,
        uniqueExecutor.get());
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
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    const uint64_t *workspaceSize, aclOpExecutor *executor)
{
    (void)workspaceSize;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, nullptr, key, nullptr, value, dy, attentionInOptional, dqOut, nullptr,
                                     dkOut, nullptr, dvOut, nullptr, nullptr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional,
                            actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
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
    ret = ContiguousInputTensor(query, nullptr, key, nullptr, value, dy, attentionInOptional, &queryCngs, nullptr,
                                &keyCngs, nullptr, &valueCngs, &dyCngs, &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs, &attenMaskOptionalCngs,
        &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, executor);
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
        kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayoutUnderTrans, innerPrecise, sparseMode, pseType,
        0, 0, 0, defaultSoftmaxInLayout, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, nullptr, &dkOut, nullptr, &dvOut, &dpseOut, nullptr, fagShape,
                                      fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreGradV2,
                   DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional,
                          attentionInOptional, prefixOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
                          keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType),
                   DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet = InputOutputDtypeAndShapeCheck(query, nullptr, keyIn, nullptr, value, dy, attentionInOptional, dqOut,
                                                  nullptr, dkOut, nullptr, dvOut, nullptr, nullptr, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradV2GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional, nullptr,
        nullptr, qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum,
        inputLayout, innerPrecise, sparseMode, pseType, dqOut, dkOut, dvOut, dpseOut, workspaceSize,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionUnpaddingScoreGradV2,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
               keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType),
        DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout, "TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet = InputOutputDtypeAndShapeCheck(query, nullptr, keyIn, nullptr, value, dy, attentionInOptional, dqOut,
                                                  nullptr, dkOut, nullptr, dvOut, nullptr, nullptr, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradV2GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb,
        preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType, dqOut, dkOut, dvOut, dpseOut,
        workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static aclnnStatus FlashAttentionScoreGradV3GetWorkspace(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *key, const aclTensor *keyRope,
    const aclTensor *value, const aclTensor *dy, const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut, const aclTensor *dkRopeOut,
    const aclTensor *dvOut, const aclTensor *dpseOut, const uint64_t *workspaceSize, aclOpExecutor *executor)
{
    (void)workspaceSize;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, queryRope, key, keyRope, value, dy, attentionInOptional, dqOut, dqRopeOut,
                                     dkOut, dkRopeOut, dvOut, nullptr, nullptr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 获取基本参数
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional,
                            actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = isSupportMultiInput(query, queryRope, key, keyRope, value, attenMaskOptional, pseShiftOptional,
                              dropMaskOptional, keepProb, fagShape, sparseMode);
    if (ret != ACLNN_SUCCESS) {
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
    ret = ContiguousInputTensor(query, queryRope, key, keyRope, value, dy, attentionInOptional, &queryCngs,
                                &queryRopeCngs, &keyCngs, &keyRopeCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs,
                                executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs, &attenMaskOptionalCngs,
        &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, executor);
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
        kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, queryRopeCngs, keyRopeCngs,
        nullptr, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayoutUnderTrans, innerPrecise, sparseMode,
        pseType, 0, 0, 0, defaultSoftmaxInLayout, executor);

    if (queryRope != nullptr && keyRope != nullptr) {
        CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr && fagRes[4] != nullptr &&
                      fagRes[5] != nullptr, // 0: dqOut 1: dkOut 2:dvOut
                  ACLNN_ERR_PARAM_NULLPTR);
    } else {
        CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr, // 0: dqOut 1: dkOut 2:dvOut
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dqRopeOut, &dkOut, &dkRopeOut, &dvOut, &dpseOut, nullptr,
                                      fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn, const aclTensor *keyInRope,
    const aclTensor *value, const aclTensor *dy, const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut, const aclTensor *dkRopeOut,
    const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionUnpaddingScoreGradV3,
        DFX_IN(query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
               attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
               prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional,
               scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType),
        DFX_OUT(dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut));

    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout, "TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet =
        InputOutputDtypeAndShapeCheck(query, queryRope, keyIn, keyInRope, value, dy, attentionInOptional, dqOut,
                                      dqRopeOut, dkOut, dkRopeOut, dvOut, nullptr, nullptr, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradV3GetWorkspace(
        query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
        attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
        prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional,
        scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType, dqOut,
        dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, char *softmaxInLayout, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionUnpaddingScoreGradV4,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, keepProb, preTokens, nextTokens, headNum,
               inputLayout, innerPrecise, sparseMode, softmaxInLayout),
        DFX_OUT(dqOut, dkOut, dvOut, dpseOut));
    // layout检查
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(softmaxInLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    // layout检查
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout, "TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (strcmp(softmaxInLayout, "same_as_input") != 0 && strcmp(softmaxInLayout, "") != 0) {
        OP_LOGE_FOR_INVALID_VALUE("FlashAttentionScoreGrad", "softmaxInLayout", softmaxInLayout,
                                  "same_as_input or empty string");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    CHECK_RET(query != nullptr && keyIn != nullptr && value != nullptr && dy != nullptr &&
                  attentionInOptional != nullptr && dqOut != nullptr && dkOut != nullptr && dvOut != nullptr &&
                  workspaceSize != nullptr && executor != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet = InputOutputDtypeAndShapeCheck(query, nullptr, keyIn, nullptr, value, dy, attentionInOptional, dqOut,
                                                  nullptr, dkOut, nullptr, dvOut, nullptr, nullptr, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradGetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, keepProb, preTokens, nextTokens, headNum,
        inputLayout, innerPrecise, sparseMode, dqOut, dkOut, dvOut, dpseOut, softmaxInLayout, workspaceSize,
        uniqueExecutor.get());
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

static aclnnStatus FlashAttentionScoreGradV5GetWorkspace(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *key, const aclTensor *keyRope,
    const aclTensor *value, const aclTensor *dy, const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclTensor *sinkInOptional, const aclIntArray *prefixOptional, // already 补充了sink
    const aclIntArray *actualSeqQLenOptional, const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut,
    const aclTensor *dkRopeOut, const aclTensor *dvOut, const aclTensor *dpseOut, const aclTensor *dsinkOut,
    char *softmaxInLayout, const uint64_t *workspaceSize, aclOpExecutor *executor, bool isMaxWorkspace = false)
{
    (void)workspaceSize;

    // 检查除sink外tensor维度是否大于2且sink维度为1
    auto ret = InvalidTensorDimCheck(query, queryRope, key, keyRope, value, dy, attentionInOptional, dqOut, dqRopeOut,
                                     dkOut, dkRopeOut, dvOut, sinkInOptional, dsinkOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (sinkInOptional != nullptr) {
        auto queryDtype = query->GetDataType();
        if (queryDtype == ge::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                "FlashAttentionScoreGrad", "query", op::ToString(DataType(queryDtype)).GetString(),
                "The dtype of query cannot be FLOAT32 when sinkInOptional is provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // fagShape needed by Pre/Post even in isMaxWorkspace mode
    FagInShapeInfo fagShape;
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional,
                            actualSeqKvLenOptional, keepProb);

    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (!isMaxWorkspace) {
        ret = isSupportMultiInput(query, queryRope, key, keyRope, value, attenMaskOptional, pseShiftOptional,
                                  dropMaskOptional, keepProb, fagShape, sparseMode);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
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
    const aclTensor *sinkInOptionalCngs = nullptr;
    ret = ContiguousInputTensor(query, queryRope, key, keyRope, value, dy, attentionInOptional, &queryCngs,
                                &queryRopeCngs, &keyCngs, &keyRopeCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs,
                                executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // already 增加sink的连续性检测
    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        sinkInOptional, &pseShiftOptionalCngs, &dropMaskOptionalCngs, &paddingMaskOptionalCngs, &attenMaskOptionalCngs,
        &softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, &softmaxInOptionalCngs, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, &sinkInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    FagShapeArray fagShapeArray;
    char inputLayoutUnderTrans[MAX_LAYOUT_SIZE] = {0};
    ret = PreFlashAttentionScoreGrad(&queryCngs, &keyCngs, &valueCngs, &dyCngs, &attentionInOptionalCngs, fagShape,
                                     fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (NeedTransposeSoftmaxTensor(softmaxInLayout)) {
        ret = TransposeSoftMaxTensor(&softmaxMaxOptionalCngs, &softmaxSumOptionalCngs, fagShape, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }
    ConvertInputLayout(fagShape, inputLayout, inputLayoutUnderTrans, MAX_LAYOUT_SIZE);

    // 调用FAG ascendc接口
    auto fagRes = l0op::FlashAttentionScoreGrad(
        queryCngs, keyCngs, valueCngs, dyCngs, pseShiftOptionalCngs, dropMaskOptionalCngs, paddingMaskOptionalCngs,
        attenMaskOptionalCngs, softmaxMaxOptionalCngs, softmaxSumOptionalCngs, softmaxInOptionalCngs,
        attentionInOptionalCngs, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, queryRopeCngs, keyRopeCngs,
        sinkInOptionalCngs, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayoutUnderTrans, innerPrecise,
        sparseMode, pseType, 0, 0, 0, softmaxInLayout, executor, isMaxWorkspace);

    if (queryRope != nullptr && keyRope != nullptr) {
        CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr && fagRes[4] != nullptr &&
                      fagRes[5] != nullptr, // 0: dqOut 1: dkOut 2:dvOut
                  ACLNN_ERR_PARAM_NULLPTR);
    } else {
        CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr, // 0: dqOut 1: dkOut 2:dvOut
                  ACLNN_ERR_PARAM_NULLPTR);
    }
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dqRopeOut, &dkOut, &dkRopeOut, &dvOut, &dpseOut, &dsinkOut,
                                      fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclTensor *sinkInOptional,
    const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, int64_t pseType, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dpseOut, const aclTensor *dsinkOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionScoreGradV3,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, prefixOptional,
               qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout,
               innerPrecise, sparseMode, pseType, sinkInOptional),
        DFX_OUT(dqOut, dkOut, dvOut, dpseOut, dsinkOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet =
        InputOutputDtypeAndShapeCheck(query, nullptr, keyIn, nullptr, value, dy, attentionInOptional, dqOut, nullptr,
                                      dkOut, nullptr, dvOut, sinkInOptional, dsinkOut, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    auto ret = FlashAttentionScoreGradV5GetWorkspace(
        query, nullptr, keyIn, nullptr, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
        attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
        sinkInOptional, prefixOptional, nullptr, nullptr, qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb,
        preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType, dqOut, nullptr, dkOut, nullptr,
        dvOut, dpseOut, dsinkOut, nullptr, workspaceSize, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGradV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn, const aclTensor *keyInRope,
    const aclTensor *value, const aclTensor *dy, const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclTensor *sinkInOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    char *softmaxInLayout, const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut,
    const aclTensor *dkRopeOut, const aclTensor *dvOut, const aclTensor *dpseOut, const aclTensor *dsinkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionUnpaddingScoreGradV5,
        DFX_IN(query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
               attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
               sinkInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
               kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise,
               sparseMode, pseType, softmaxInLayout),
        DFX_OUT(dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut, dsinkOut));

    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout, "TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet =
        InputOutputDtypeAndShapeCheck(query, queryRope, keyIn, keyInRope, value, dy, attentionInOptional, dqOut,
                                      dqRopeOut, dkOut, dkRopeOut, dvOut, sinkInOptional, dsinkOut, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradV5GetWorkspace(
        query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
        attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
        sinkInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode,
        pseType, dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut, dsinkOut, softmaxInLayout, workspaceSize,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionUnpaddingScoreGradV5);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetMaxWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn, const aclTensor *keyInRope,
    const aclTensor *value, const aclTensor *dy, const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclTensor *sinkInOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    char *softmaxInLayout, const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut,
    const aclTensor *dkRopeOut, const aclTensor *dvOut, const aclTensor *dpseOut, const aclTensor *dsinkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (dropMaskOptional != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "FlashAttentionScoreGrad", "dropMaskOptional",
            "dropMaskOptional must be empty when querying the maximum workspace size");
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE_FOR_INVALID_FORMAT("FlashAttentionScoreGrad", "inputLayout", inputLayout, "TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(workspaceSize != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "workspaceSize"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "executor"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto checkRet =
        InputOutputDtypeAndShapeCheck(query, queryRope, keyIn, keyInRope, value, dy, attentionInOptional, dqOut,
                                      dqRopeOut, dkOut, dkRopeOut, dvOut, sinkInOptional, dsinkOut, inputLayout);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // calculate fag
    auto ret = FlashAttentionScoreGradV5GetWorkspace(
        query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional,
        attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional,
        sinkInOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode,
        pseType, dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, dpseOut, dsinkOut, softmaxInLayout, workspaceSize,
        uniqueExecutor.get(), true);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSinkOptionalInput(const aclTensor *sinkInOptional, const aclTensor *dsinkOut,
                                          const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                          const aclTensor *dy, const aclTensor *attentionInOptional,
                                          const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
                                          const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut)
{
    if (sinkInOptional != nullptr) {
        OP_CHECK(dsinkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dsinkOut"),
                 return ACLNN_ERR_PARAM_NULLPTR);
        auto qDtype = query->GetDataType();
        auto kDtype = key->GetDataType();
        auto vDtype = value->GetDataType();
        auto dyDtype = dy->GetDataType();
        auto yDtype = attentionInOptional->GetDataType();
        auto dqOutDtype = dqOut->GetDataType();
        auto dkOutDtype = dkOut->GetDataType();
        auto dvOutDtype = dvOut->GetDataType();
        auto sinkDtype = sinkInOptional->GetDataType();
        auto dsinkDtype = dsinkOut->GetDataType();
        OP_CHECK((sinkDtype == ge::DataType::DT_FLOAT),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "sinkInOptional", op::ToString(DataType(sinkDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of sinkInOptional must be FLOAT32"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((dsinkDtype == ge::DataType::DT_FLOAT),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "dsinkOut", op::ToString(DataType(dsinkDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of dsinkOut must be FLOAT32"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((qDtype == ge::DataType::DT_BF16 || qDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "query", op::ToString(DataType(qDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of query must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((kDtype == ge::DataType::DT_BF16 || kDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "keyIn", op::ToString(DataType(kDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of keyIn must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((vDtype == ge::DataType::DT_BF16 || vDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "value", op::ToString(DataType(vDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of value must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((dyDtype == ge::DataType::DT_BF16 || dyDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "dy", op::ToString(DataType(dyDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of dy must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((yDtype == ge::DataType::DT_BF16 || yDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("FlashAttentionScoreGrad", "attentionInOptional",
                                                       op::ToString(DataType(yDtype)).GetString(),
                                                       "When optional parameter sinkInOptional exists, "
                                                       "the dtype of attentionInOptional must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((dqOutDtype == ge::DataType::DT_BF16 || dqOutDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "dqOut", op::ToString(DataType(dqOutDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of dqOut must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((dkOutDtype == ge::DataType::DT_BF16 || dkOutDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "dkOut", op::ToString(DataType(dkOutDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of dkOut must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK((dvOutDtype == ge::DataType::DT_BF16 || dvOutDtype == ge::DataType::DT_FLOAT16),
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                     "FlashAttentionScoreGrad", "dvOut", op::ToString(DataType(dvOutDtype)).GetString(),
                     "When optional parameter sinkInOptional exists, the dtype of dvOut must be FLOAT16 or BFLOAT16"),
                 return ACLNN_ERR_PARAM_INVALID);
        if (queryRopeOptional != nullptr && keyRopeOptional != nullptr) {
            auto qRopeDtype = queryRopeOptional->GetDataType();
            auto kRopeDtype = keyRopeOptional->GetDataType();
            OP_CHECK((qRopeDtype == ge::DataType::DT_BF16 || qRopeDtype == ge::DataType::DT_FLOAT16),
                     OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                         "FlashAttentionScoreGrad", "queryRopeOptional", op::ToString(DataType(qRopeDtype)).GetString(),
                         "When optional parameter sinkInOptional exists, "
                         "the dtype of queryRopeOptional must be FLOAT16 or BFLOAT16"),
                     return ACLNN_ERR_PARAM_INVALID);
            OP_CHECK((kRopeDtype == ge::DataType::DT_BF16 || kRopeDtype == ge::DataType::DT_FLOAT16),
                     OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("FlashAttentionScoreGrad", "keyRopeOptional",
                                                           op::ToString(DataType(kRopeDtype)).GetString(),
                                                           "When optional parameter sinkInOptional exists, "
                                                           "the dtype of keyRopeOptional must be FLOAT16 or BFLOAT16"),
                     return ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus FlashAttentionScoreGradV4GetWorkspace(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclTensor *sinkInOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, char *softmaxInLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, int64_t seed, int64_t offset, int64_t outDtype, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dqRopeOut, const aclTensor *dkRopeOut, const aclTensor *dpseOut,
    const aclTensor *dsinkOut, aclOpExecutor *executor)
{
    // 获取基本参数
    FagInShapeInfo fagShape;
    // 检查tensor维度是否大于2
    auto ret = InvalidTensorDimCheck(query, queryRopeOptional, key, keyRopeOptional, value, dy, attentionInOptional,
                                     dqOut, dqRopeOut, dkOut, dkRopeOut, dvOut, sinkInOptional, dsinkOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = GetInputShapeInfo(query, key, value, headNum, inputLayout, fagShape, actualSeqQLenOptional,
                            actualSeqKvLenOptional, keepProb);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = CheckSinkOptionalInput(sinkInOptional, dsinkOut, query, key, value, dy, attentionInOptional,
                                 queryRopeOptional, keyRopeOptional, dqOut, dkOut, dvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (softmaxInLayout != nullptr) {
        if (strcmp(inputLayout, "TND") == 0 && strcmp(softmaxInLayout, "same_as_input") != 0 &&
            strcmp(softmaxInLayout, "") != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "FlashAttentionScoreGrad", "softmaxInLayout", softmaxInLayout,
                "When inputLayout is TND, the value of softmaxInLayout must be same_as_input or empty string");
            return ACLNN_ERR_PARAM_INVALID;
        } else if (strcmp(inputLayout, "TND") != 0 && strcmp(softmaxInLayout, "") != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "FlashAttentionScoreGrad", "softmaxInLayout", softmaxInLayout,
                "When inputLayout is not TND, the value of softmaxInLayout must be empty string");
            return ACLNN_ERR_PARAM_INVALID;
        }
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
    const aclTensor *queryRopeOptionalCngs = nullptr;
    const aclTensor *keyRopeOptionalCngs = nullptr;
    const aclTensor *dScaleQOptionalCngs = nullptr;
    const aclTensor *dScaleKOptionalCngs = nullptr;
    const aclTensor *dScaleVOptionalCngs = nullptr;
    const aclTensor *dScaleDyOptionalCngs = nullptr;
    const aclTensor *dScaleOOptionalCngs = nullptr;
    const aclTensor *sinkInOptionalCngs = nullptr;
    const aclTensor *dsink = nullptr;
    ret = ContiguousInputTensor(query, queryRopeOptional, key, keyRopeOptional, value, dy, attentionInOptional,
                                &queryCngs, &queryRopeOptionalCngs, &keyCngs, &keyRopeOptionalCngs, &valueCngs, &dyCngs,
                                &attentionInOptionalCngs, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ContiguousOptionalInputTensor(
        pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional,
        softmaxSumOptional, softmaxInOptional, queryRopeOptional, keyRopeOptional, dScaleQOptional, dScaleKOptional,
        dScaleVOptional, dScaleDyOptional, dScaleOOptional, sinkInOptional, &pseShiftOptionalCngs,
        &dropMaskOptionalCngs, &paddingMaskOptionalCngs, &attenMaskOptionalCngs, &softmaxMaxOptionalCngs,
        &softmaxSumOptionalCngs, &softmaxInOptionalCngs, &queryRopeOptionalCngs, &keyRopeOptionalCngs,
        &dScaleQOptionalCngs, &dScaleKOptionalCngs, &dScaleVOptionalCngs, &dScaleDyOptionalCngs, &dScaleOOptionalCngs,
        &sinkInOptionalCngs, executor);
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
        kvStartIdxOptional, dScaleQOptionalCngs, dScaleKOptionalCngs, dScaleVOptionalCngs, dScaleDyOptionalCngs,
        dScaleOOptionalCngs, nullptr, nullptr, queryRopeOptionalCngs, keyRopeOptionalCngs, sinkInOptionalCngs,
        scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayoutUnderTrans, innerPrecise, sparseMode, pseType,
        seed, offset, outDtype, softmaxInLayout, executor);
    CHECK_RET(fagRes[0] != nullptr && fagRes[1] != nullptr && fagRes[2] != nullptr, // 0: dqOut 1: dkOut 2:dvOut
              ACLNN_ERR_PARAM_NULLPTR);

    // transpose + slice + reshape + viewCopy
    ret = PostFlashAttentionScoreGrad(fagRes, &dqOut, &dqRopeOut, &dkOut, &dkRopeOut, &dvOut, &dpseOut, &dsinkOut,
                                      fagShape, fagShapeArray, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional, const aclTensor *sinkInOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional, const aclTensor *dScaleDyOptional,
    const aclTensor *dScaleOOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, char *softmaxInLayout, int64_t innerPrecise, int64_t sparseMode,
    int64_t pseType, int64_t seed, int64_t offset, int64_t outDtype, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dqRopeOut, const aclTensor *dkRopeOut, const aclTensor *dpseOut,
    const aclTensor *dsinkOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnFlashAttentionScoreGradV4,
        DFX_IN(query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
               softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, sinkInOptional,
               queryRopeOptional, keyRopeOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional,
               dScaleOOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
               kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, softmaxInLayout,
               innerPrecise, sparseMode, pseType, seed, offset, outDtype),
        DFX_OUT(dqOut, dkOut, dvOut, dqRopeOut, dkRopeOut, dpseOut, dsinkOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    OP_CHECK(query != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(keyIn != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "keyIn"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(value != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "value"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dy != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dy"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(attentionInOptional != nullptr,
             OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dqOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dqOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dkOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dkOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(dvOut != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "dvOut"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(inputLayout != nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "inputLayout"),
             return ACLNN_ERR_PARAM_NULLPTR);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FlashAttentionScoreGrad", "headNum", std::to_string(headNum).c_str(),
                                              "The value of headNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(dqOut, query);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(dkOut, keyIn);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(dvOut, value);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(dqRopeOut, queryRopeOptional);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(dkRopeOut, keyRopeOptional);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE_FAGV4(dsinkOut, sinkInOptional);
    // calculate fag
    auto ret = FlashAttentionScoreGradV4GetWorkspace(
        query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
        softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, sinkInOptional,
        queryRopeOptional, keyRopeOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional, dScaleDyOptional,
        dScaleOOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional,
        kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout, softmaxInLayout,
        innerPrecise, sparseMode, pseType, seed, offset, outDtype, dqOut, dkOut, dvOut, dqRopeOut, dkRopeOut, dpseOut,
        dsinkOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGradV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreGradV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
