/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mixed_quant_flash_attn_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include <limits>

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char* MQFA_ACLNN_OP_NAME = "MixedQuantFlashAttnMetadata";
// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
// batch size 上限（不含）
constexpr uint32_t MQFA_BATCH_SIZE_LIMIT = 65536;
// max_seqlen 取值范围
constexpr int64_t MQFA_MAX_SEQLEN_LOWER_BOUND = -1;
constexpr int64_t MQFA_MAX_SEQLEN_UPPER_BOUND = std::numeric_limits<int32_t>::max();
// win_left / win_right 上限
constexpr int64_t MQFA_WIN_LOWER_BOUND = -1;
constexpr int64_t MQFA_WIN_UPPER_BOUND = std::numeric_limits<int32_t>::max();

enum class MaskModeMqfa : uint8_t {
    NO_MASK = 0,
    CAUSAL = 3,
    SLIDING_WINDOW = 4,
};

inline bool IsTensorExistMqfa(const aclTensor* tensor)
{
    // 零元素张量视为已传入（如 batch_size=0 时的 seqused_kv/seqused_q），元素个数约束由各张量校验单独把关
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0);
}

int64_t GetDimNumMqfa(const aclTensor* tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeMqfa(const aclTensor* tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

std::string DataTypeToSerialStringMqfa(aclDataType dataType)
{
    switch (dataType) {
        case aclDataType::ACL_DT_UNDEFINED:
            return "UNDEFINED";
        case aclDataType::ACL_FLOAT:
            return "FLOAT";
        case aclDataType::ACL_FLOAT16:
            return "FLOAT16";
        case aclDataType::ACL_INT8:
            return "INT8";
        case aclDataType::ACL_INT32:
            return "INT32";
        case aclDataType::ACL_UINT8:
            return "UINT8";
        case aclDataType::ACL_INT16:
            return "INT16";
        case aclDataType::ACL_UINT16:
            return "UINT16";
        case aclDataType::ACL_UINT32:
            return "UINT32";
        case aclDataType::ACL_INT64:
            return "INT64";
        case aclDataType::ACL_UINT64:
            return "UINT64";
        case aclDataType::ACL_DOUBLE:
            return "DOUBLE";
        case aclDataType::ACL_BOOL:
            return "BOOL";
        case aclDataType::ACL_STRING:
            return "STRING";
        case aclDataType::ACL_COMPLEX64:
            return "COMPLEX64";
        case aclDataType::ACL_COMPLEX128:
            return "COMPLEX128";
        case aclDataType::ACL_BF16:
            return "BFLOAT16";
        case aclDataType::ACL_INT4:
            return "INT4";
        case aclDataType::ACL_UINT1:
            return "UINT1";
        case aclDataType::ACL_COMPLEX32:
            return "COMPLEX32";
        case aclDataType::ACL_HIFLOAT8:
            return "HIFLOAT8";
        case aclDataType::ACL_FLOAT8_E5M2:
            return "FLOAT8_E5M2";
        case aclDataType::ACL_FLOAT8_E4M3FN:
            return "FLOAT8_E4M3FN";
        case aclDataType::ACL_FLOAT8_E8M0:
            return "FLOAT8_E8M0";
        case aclDataType::ACL_FLOAT6_E3M2:
            return "FLOAT6_E3M2";
        case aclDataType::ACL_FLOAT6_E2M3:
            return "FLOAT6_E2M3";
        case aclDataType::ACL_FLOAT4_E2M1:
            return "FLOAT4_E2M1";
        case aclDataType::ACL_FLOAT4_E1M2:
            return "FLOAT4_E1M2";
        default:
            return std::to_string(static_cast<int32_t>(dataType));
    }
}

inline bool IsTensorSourceMqfa(const std::string& source)
{
    return source != "batch_size";
}

inline int64_t GetRawShapeSizeMqfa(const std::string& source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

inline std::string GetSourceDescMqfa(const std::string& source, int64_t value)
{
    std::string valueStr = " (" + std::to_string(value) + ")";
    if (source == "batch_size") {
        return "batch_size" + valueStr;
    }
    if (source.find("cu_seqlens") != std::string::npos) {
        return "the shape size" + valueStr + " of " + source + " minus 1";
    }
    return "the shape size" + valueStr + " of " + source;
}

// 与分配侧（torch extension 的 _calculate_metadata_size）及主算子 metadata_checker 的计算方式保持一致：
// head/FA/FD 每个调度单元均为 16 个 int32，结果按 4096 个 INT32 向上对齐
int64_t CalcMetaDataSizeMqfa(int64_t batchSize, int64_t numHeadsKv, int64_t aicCoreNum, int64_t aivCoreNum)
{
    constexpr int64_t METADATA_UNIT = 16;
    constexpr int64_t METADATA_ALIGN_SIZE = 4096;
    int64_t rawSize = ((aicCoreNum + aivCoreNum) * batchSize * numHeadsKv + 1) * METADATA_UNIT;
    return (rawSize + METADATA_ALIGN_SIZE - 1) / METADATA_ALIGN_SIZE * METADATA_ALIGN_SIZE;
}

// batch size 推导优先级与 aicpu 侧一致：seqused 优先于 cu_seqlens（仅 TND），batch_size 属性仅作兜底
int64_t GetQueryBatchSizeMqfa(const aclTensor* sequsedQOptional, const aclTensor* cuSeqlensQOptional,
                              const char* layoutQOptional, int64_t batchSize, std::string* source)
{
    if (IsTensorExistMqfa(sequsedQOptional)) {
        *source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistMqfa(cuSeqlensQOptional)) {
            *source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

int64_t GetKvBatchSizeMqfa(const aclTensor* sequsedKvOptional, int64_t batchSize, std::string* source)
{
    if (IsTensorExistMqfa(sequsedKvOptional)) {
        *source = "seqused_kv";
        return sequsedKvOptional->GetViewShape().GetDim(0);
    }
    *source = "batch_size";
    return batchSize;
}

aclnnStatus CheckSingleParamMqfa(const aclTensor* sequsedQOptional, int64_t batchSize, int64_t maxSeqlenQ,
                                 int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                                 int64_t quantComputeMode, int64_t maskMode, int64_t winLeft, int64_t winRight,
                                 const char* layoutQOptional, const char* layoutKvOptional,
                                 const char* layoutAttnOutOptional, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    // batch_size >= 0
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q [-1, INT32_MAX]
    if (maxSeqlenQ < MQFA_MAX_SEQLEN_LOWER_BOUND || maxSeqlenQ > MQFA_MAX_SEQLEN_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The current value is not within the valid range. The valid range is [" +
                                                  std::to_string(MQFA_MAX_SEQLEN_LOWER_BOUND) + ", " +
                                                  std::to_string(MQFA_MAX_SEQLEN_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_kv [-1, INT32_MAX]
    if (maxSeqlenKv < MQFA_MAX_SEQLEN_LOWER_BOUND || maxSeqlenKv > MQFA_MAX_SEQLEN_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "max_seqlen_kv", std::to_string(maxSeqlenKv),
                                              "The current value is not within the valid range. The valid range is [" +
                                                  std::to_string(MQFA_MAX_SEQLEN_LOWER_BOUND) + ", " +
                                                  std::to_string(MQFA_MAX_SEQLEN_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q > 0
    if (numHeadsQ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                              "The value of num_heads_q must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv > 0
    if (numHeadsKv <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "num_heads_kv", std::to_string(numHeadsKv),
                                              "The value of num_heads_kv must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim = 128
    if (headDim != 128) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "head_dim", std::to_string(headDim),
                                              "The value of head_dim must be 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // quant_mode: 1 or 2
    if (quantComputeMode != 1 && quantComputeMode != 2) {
        OP_LOGE_FOR_INVALID_VALUE(MQFA_ACLNN_OP_NAME, "quant_mode", std::to_string(quantComputeMode), "1 or 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode: 0, 3 or 4
    if (maskMode != static_cast<int64_t>(MaskModeMqfa::NO_MASK) &&
        maskMode != static_cast<int64_t>(MaskModeMqfa::CAUSAL) &&
        maskMode != static_cast<int64_t>(MaskModeMqfa::SLIDING_WINDOW)) {
        OP_LOGE_FOR_INVALID_VALUE(MQFA_ACLNN_OP_NAME, "mask_mode", std::to_string(maskMode), "0, 3 or 4");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // win_left [-1, INT32_MAX]
    if (winLeft < MQFA_WIN_LOWER_BOUND || winLeft > MQFA_WIN_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "win_left", std::to_string(winLeft),
                                              "The current value is not within the valid range. The valid range is [" +
                                                  std::to_string(MQFA_WIN_LOWER_BOUND) + ", " +
                                                  std::to_string(MQFA_WIN_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // win_right [-1, INT32_MAX]
    if (winRight < MQFA_WIN_LOWER_BOUND || winRight > MQFA_WIN_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "win_right", std::to_string(winRight),
                                              "The current value is not within the valid range. The valid range is [" +
                                                  std::to_string(MQFA_WIN_LOWER_BOUND) + ", " +
                                                  std::to_string(MQFA_WIN_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maskMode == static_cast<int64_t>(MaskModeMqfa::NO_MASK) ||
        maskMode == static_cast<int64_t>(MaskModeMqfa::CAUSAL)) {
        if (winLeft != -1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "win_left", std::to_string(winLeft),
                                                  "When mask_mode is 0 or 3, the value of win_left must be -1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (winRight != -1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "win_right", std::to_string(winRight),
                                                  "When mask_mode is 0 or 3, the value of win_right must be -1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "layout_q", "layout_q cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKvOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "layout_kv", "layout_kv cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutAttnOutOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "layout_attn_out",
                                                 "layout_attn_out cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q: BSND, TND or BNSD
    if (strcmp(layoutQOptional, "BSND") != 0 && strcmp(layoutQOptional, "TND") != 0 &&
        strcmp(layoutQOptional, "BNSD") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(MQFA_ACLNN_OP_NAME, "layout_q", layoutQOptional, "BSND, TND or BNSD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_attn_out: BSND, TND or BNSD
    if (strcmp(layoutAttnOutOptional, "BSND") != 0 && strcmp(layoutAttnOutOptional, "TND") != 0 &&
        strcmp(layoutAttnOutOptional, "BNSD") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(MQFA_ACLNN_OP_NAME, "layout_attn_out", layoutAttnOutOptional, "BSND, TND or BNSD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv: PA_BNBD, PA_BBND or PA_NZ
    if (strcmp(layoutKvOptional, "PA_BNBD") != 0 && strcmp(layoutKvOptional, "PA_BBND") != 0 &&
        strcmp(layoutKvOptional, "PA_NZ") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(MQFA_ACLNN_OP_NAME, "layout_kv", layoutKvOptional, "PA_BNBD, PA_BBND or PA_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_q 为 BSND 或 BNSD 且 seqused_q 未传入时，max_seqlen_q 必须大于 -1
    if (strcmp(layoutQOptional, "TND") != 0 && !IsTensorExistMqfa(sequsedQOptional) && maxSeqlenQ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "When layout_q is BSND or BNSD and seqused_q is not passed, "
                                              "the value of max_seqlen_q must be greater than -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q 必须能被 num_heads_kv 整除
    if (numHeadsQ % numHeadsKv != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(MQFA_ACLNN_OP_NAME, "num_heads_q and num_heads_kv",
                                               std::to_string(numHeadsQ) + " and " + std::to_string(numHeadsKv),
                                               "The value of num_heads_q must be exactly divisible by "
                                               "that of num_heads_kv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核数校验
    if (aicCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "aic_core_num", std::to_string(aicCoreNum),
                                              "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(MQFA_ACLNN_OP_NAME, "aiv_core_num", std::to_string(aivCoreNum),
                                              "The value of aiv_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceMqfa(const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedKvOptional,
                               const char* layoutQOptional, const char* layoutKvOptional, const aclTensor* metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistMqfa(cuSeqlensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "cu_seqlens_q",
                                                     "When layout_q is TND, cu_seqlens_q cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cu_seqlens_q 至少有1个元素
        if (cuSeqlensQOptional->GetViewShape().GetDim(0) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(MQFA_ACLNN_OP_NAME, "cu_seqlens_q",
                                                      std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)),
                                                      "The shape size of cu_seqlens_q must be greater than 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        // layout_q 非 TND 时，不应该传入 cu_seqlens_q
        if (cuSeqlensQOptional != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "cu_seqlens_q",
                                                     "When layout_q is not TND, cu_seqlens_q cannot be passed");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // seqused_kv 存在性校验
    if (strcmp(layoutKvOptional, "PA_BNBD") == 0 || strcmp(layoutKvOptional, "PA_BBND") == 0 ||
        strcmp(layoutKvOptional, "PA_NZ") == 0) {
        if (!IsTensorExistMqfa(sequsedKvOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "seqused_kv",
                                                     "When layout_kv is PA_BBND, PA_BNBD or PA_NZ, "
                                                     "seqused_kv cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistMqfa(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(MQFA_ACLNN_OP_NAME, "metadata", "metadata cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyMqfa(const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
                                 const aclTensor* sequsedKvOptional, const char* layoutQ, int64_t batchSize,
                                 int64_t numHeadsKv, int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor* metadata)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    int64_t dimNum = -1;
    // 校验 cu_seqlens_q
    if (IsTensorExistMqfa(cuSeqlensQOptional)) {
        // 校验 cu_seqlens_q 维度
        dimNum = GetDimNumMqfa(cuSeqlensQOptional);
        if (dimNum != DIM_NUM_ONE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(MQFA_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum) + "D", "1D");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        dataType = GetDataTypeMqfa(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(MQFA_ACLNN_OP_NAME, "cu_seqlens_q",
                                                  DataTypeToSerialStringMqfa(dataType),
                                                  "The dtype of cu_seqlens_q must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistMqfa(sequsedQOptional)) {
        // 校验 seqused_q 维度
        dimNum = GetDimNumMqfa(sequsedQOptional);
        if (dimNum != DIM_NUM_ONE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(MQFA_ACLNN_OP_NAME, "seqused_q", std::to_string(dimNum) + "D", "1D");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        dataType = GetDataTypeMqfa(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(MQFA_ACLNN_OP_NAME, "seqused_q", DataTypeToSerialStringMqfa(dataType),
                                                  "The dtype of seqused_q must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_kv
    if (IsTensorExistMqfa(sequsedKvOptional)) {
        // 校验 seqused_kv 维度
        dimNum = GetDimNumMqfa(sequsedKvOptional);
        if (dimNum != DIM_NUM_ONE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(MQFA_ACLNN_OP_NAME, "seqused_kv", std::to_string(dimNum) + "D", "1D");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_kv 数据类型
        dataType = GetDataTypeMqfa(sequsedKvOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(MQFA_ACLNN_OP_NAME, "seqused_kv",
                                                  DataTypeToSerialStringMqfa(dataType),
                                                  "The dtype of seqused_kv must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 推导 q/kv 侧 batch size，优先级与 aicpu 侧一致：seqused_kv > seqused_q > cu_seqlens_q（仅 TND） > batch_size
    std::string querySource;
    int64_t queryBatchSize =
        GetQueryBatchSizeMqfa(sequsedQOptional, cuSeqlensQOptional, layoutQ, batchSize, &querySource);
    std::string kvSource;
    int64_t kvBatchSize = GetKvBatchSizeMqfa(sequsedKvOptional, batchSize, &kvSource);
    // 校验 TND 场景 q 维度一致性（cu_seqlens_q 与 seqused_q 同时传入时）
    if (strcmp(layoutQ, "TND") == 0 && IsTensorExistMqfa(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                MQFA_ACLNN_OP_NAME, "cu_seqlens_q and seqused_q",
                std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)) + " and " +
                    std::to_string(sequsedQOptional->GetViewShape().GetDim(0)),
                "When layout_q is TND and seqused_q is passed, the shape size (" +
                    std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)) +
                    ") of cu_seqlens_q minus 1 must be equal to the shape size (" +
                    std::to_string(sequsedQOptional->GetViewShape().GetDim(0)) + ") of seqused_q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    std::string queryDesc = GetSourceDescMqfa(querySource, GetRawShapeSizeMqfa(querySource, queryBatchSize));
    // 校验 q/kv 维度一致性
    if (queryBatchSize != kvBatchSize) {
        std::string kvDesc = GetSourceDescMqfa(kvSource, GetRawShapeSizeMqfa(kvSource, kvBatchSize));
        if (IsTensorSourceMqfa(querySource) && IsTensorSourceMqfa(kvSource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                MQFA_ACLNN_OP_NAME, querySource + " and " + kvSource,
                std::to_string(GetRawShapeSizeMqfa(querySource, queryBatchSize)) + " and " +
                    std::to_string(GetRawShapeSizeMqfa(kvSource, kvBatchSize)),
                queryDesc + " must be equal to " + kvDesc);
        } else if (IsTensorSourceMqfa(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(MQFA_ACLNN_OP_NAME, querySource,
                                                      std::to_string(GetRawShapeSizeMqfa(querySource, queryBatchSize)),
                                                      queryDesc + " must be equal to " + kvDesc);
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(MQFA_ACLNN_OP_NAME, kvSource,
                                                      std::to_string(GetRawShapeSizeMqfa(kvSource, kvBatchSize)),
                                                      kvDesc + " must be equal to " + queryDesc);
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 batch_size 与 q 侧 cu_seqlens_q、seqused_q 推导 batch 一致
    if (IsTensorSourceMqfa(querySource) && batchSize != queryBatchSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            MQFA_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
            "batch_size (" + std::to_string(batchSize) + ") must be equal to " + queryDesc);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 batch size 上限
    if (queryBatchSize >= MQFA_BATCH_SIZE_LIMIT) {
        if (IsTensorSourceMqfa(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                MQFA_ACLNN_OP_NAME, querySource, std::to_string(GetRawShapeSizeMqfa(querySource, queryBatchSize)),
                queryDesc + " must be less than " + std::to_string(MQFA_BATCH_SIZE_LIMIT));
        } else {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                MQFA_ACLNN_OP_NAME, querySource, std::to_string(queryBatchSize),
                queryDesc + " must be less than " + std::to_string(MQFA_BATCH_SIZE_LIMIT));
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 metadata（置于一致性/上限校验之后：batch 不一致时优先报出根因，避免尺寸症状掩盖根因）
    if (IsTensorExistMqfa(metadata)) {
        // 校验 metadata 维度
        dimNum = GetDimNumMqfa(metadata);
        if (dimNum != DIM_NUM_ONE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(MQFA_ACLNN_OP_NAME, "metadata", std::to_string(dimNum) + "D", "1D");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素个数（按 kv 侧推导的 batch size 计算，与 aicpu 分核实际使用的 baseInfo.batchSize 口径一致）
        int64_t expectedSize = CalcMetaDataSizeMqfa(kvBatchSize, numHeadsKv, aicCoreNum, aivCoreNum);
        if (metadata->GetViewShape().GetDim(0) != expectedSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(MQFA_ACLNN_OP_NAME, "metadata",
                                          std::to_string(metadata->GetViewShape().GetDim(0)),
                                          std::to_string(expectedSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        dataType = GetDataTypeMqfa(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(MQFA_ACLNN_OP_NAME, "metadata", DataTypeToSerialStringMqfa(dataType),
                                                  "The dtype of metadata must be INT32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheck(const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
                        const aclTensor* sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                        int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t quantComputeMode,
                        int64_t maskMode, int64_t winLeft, int64_t winRight, const char* layoutQ, const char* layoutKv,
                        const char* layoutAttnOut, uint32_t aicCoreNum, uint32_t aivCoreNum, const aclTensor* metaData)
{
    if (CheckSingleParamMqfa(sequsedQOptional, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim,
                             quantComputeMode, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutAttnOut,
                             aicCoreNum, aivCoreNum) == ACLNN_SUCCESS &&
        CheckExistenceMqfa(cuSeqlensQOptional, sequsedKvOptional, layoutQ, layoutKv, metaData) == ACLNN_SUCCESS &&
        CheckConsistencyMqfa(cuSeqlensQOptional, sequsedQOptional, sequsedKvOptional, layoutQ, batchSize, numHeadsKv,
                             aicCoreNum, aivCoreNum, metaData) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}
} // namespace

#ifdef __cplusplus
}
#endif
