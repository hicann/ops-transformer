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
 * \file fused_causal_conv1d_cut_bsh_tiling_arch35.cpp
 * \brief
 */

#include "fused_causal_conv1d_cut_bsh_tiling_arch35.h"
#include <limits>

namespace optiling {
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;

constexpr uint64_t INPUT_X_INDEX = 0;
constexpr uint64_t INPUT_WEIGHT_INDEX = 1;
constexpr uint64_t INPUT_CACHE_STATES_INDEX = 2;
constexpr uint64_t INPUT_QUERY_START_LOC_INDEX = 3;
constexpr uint64_t INPUT_CACHE_INDICES_INDEX = 4;
constexpr uint64_t INPUT_INITIAL_STATE_MODE_INDEX = 5;
constexpr uint64_t INPUT_BIAS_INDEX = 6;
constexpr uint64_t INPUT_NUM_ACCEPTED_TOKEN_INDEX = 7;
constexpr uint64_t INPUT_NUM_COMPUTED_TOKENS_INDEX = 8;
constexpr uint64_t INPUT_BLOCK_IDX_FIRST_INDEX = 9;
constexpr uint64_t INPUT_BLOCK_IDX_LAST_INDEX = 10;
constexpr uint64_t INPUT_INITIAL_STATE_IDX_INDEX = 11;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;
constexpr int32_t ATTR_MAX_QUERY_LEN_INDEX = 3;
constexpr int32_t ATTR_RESIDUAL_CONNECTION_INDEX = 4;
constexpr int32_t ATTR_BLOCK_SIZE_INDEX = 5;
constexpr int32_t ATTR_CONV_MODE_INDEX = 6;
constexpr int32_t ATTR_MAX_DRAFT_TOKENS_INDEX = 7;

constexpr uint64_t OUTPUT_CONV_STATES_INDEX = 0;
constexpr uint64_t OUTPUT_Y_INDEX = 1;
constexpr int64_t BH_MAX_QUERY_LEN_THRESHOLD = 8;
constexpr uint64_t X_DIM_NUM = 2;
constexpr uint64_t WEIGHT_DIM_NUM = 2;
constexpr uint64_t CACHE_STATES_DIM_NUM = 3;
constexpr uint64_t SEQ_START_INDEX_DIM_NUM = 1;

constexpr uint64_t DIM_MIN = 16;
constexpr uint64_t DIM_MAX = 16384;
constexpr uint64_t CU_SEQ_LEN_MIN = 1;
constexpr uint64_t CU_SEQ_LEN_MAX = 1048576;
constexpr uint64_t BATCH_MIN = 1;
constexpr uint64_t BATCH_MAX = 256;
constexpr uint64_t KERNEL_WIDTH_MAX = 6;

constexpr uint64_t DIM_ALIGN_ELEMENTS = 128;           // 256 bytes / 2 bytes per element (fp16/bf16)
constexpr uint64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024; // 8 KB system reserved UB space
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t TILING_KEY_BSH_BF16 = 10000UL;
constexpr uint64_t TILING_KEY_BSH_FP16 = 10001UL;
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool FusedCausalConv1dCutBSHTiling::IsCapable()
{
    if (context_ == nullptr) {
        return false;
    }
    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    if (xShape == nullptr) {
        return false;
    }
    size_t xDimNum = xShape->GetOriginShape().GetDimNum();
    if (xDimNum != X_DIM_NUM) {
        // BSH 模板仅支持 2D 输入
        return false;
    }
    int64_t maxQueryLen = -1;
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *p = attrs->GetAttrPointer<int64_t>(ATTR_MAX_QUERY_LEN_INDEX);
        if (p != nullptr) {
            maxQueryLen = *p;
        }
    }
    // 短序列 (1..8) → BH；其他 (>8 或 -1 未知) → BSH
    if (maxQueryLen >= 1 && maxQueryLen <= BH_MAX_QUERY_LEN_THRESHOLD) {
        return false;
    }
    return true;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    if (totalCoreNum_ == 0UL) {
        OP_LOGE(context_->GetNodeName(), "coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE(context_->GetNodeName(), "ubSize is 0");
        return ge::GRAPH_FAILED;
    }

    // 核内 UB 有固定 8KB 的系统保留空间，需要扣除
    if (ubSize <= SYSTEM_RESERVED_UB_SIZE) {
        OP_LOGE(context_->GetNodeName(), "ubSize %lu is too small, must be > %lu", ubSize, SYSTEM_RESERVED_UB_SIZE);
        return ge::GRAPH_FAILED;
    }
    ubSize_ = ubSize - SYSTEM_RESERVED_UB_SIZE;

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据类型
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckInputDtype()
{
    if (xType_ != ge::DataType::DT_FLOAT16 && xType_ != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xType_).c_str(),
                                              "The dtype of x must be FLOAT16 or BF16");
        return ge::GRAPH_FAILED;
    }

    if (weightType_ != xType_) {
        std::string dtypeMsg = Ops::Base::ToString(xType_) + " and " + Ops::Base::ToString(weightType_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and weight", dtypeMsg.c_str(),
                                               "The dtypes of x and weight must be the same");
        return ge::GRAPH_FAILED;
    }

    auto cacheStatesType = context_->GetInputDesc(INPUT_CACHE_STATES_INDEX)->GetDataType();
    if (cacheStatesType != xType_) {
        std::string dtypeMsg = Ops::Base::ToString(xType_) + " and " + Ops::Base::ToString(cacheStatesType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and cache_states", dtypeMsg.c_str(),
                                               "The dtypes of x and cache_states must be the same");
        return ge::GRAPH_FAILED;
    }

    // 检查 cacheIndices (OPTIONAL)
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(INPUT_CACHE_INDICES_INDEX);
    if (cacheIndicesDesc != nullptr) {
        auto cacheIndicesType = cacheIndicesDesc->GetDataType();
        if (cacheIndicesType != ge::DataType::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                  Ops::Base::ToString(cacheIndicesType).c_str(),
                                                  "The dtype of cache_indices must be INT32");
            return ge::GRAPH_FAILED;
        }
    }

    // 检查 queryStartLoc (OPTIONAL)
    auto seqStartIndexDesc = context_->GetOptionalInputDesc(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexDesc != nullptr) {
        auto seqStartIndexType = seqStartIndexDesc->GetDataType();
        if (seqStartIndexType != ge::DataType::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                                  Ops::Base::ToString(seqStartIndexType).c_str(),
                                                  "The dtype of query_start_loc must be INT32");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据维度
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckXDim()
{
    uint64_t xDimNum = xShape_.GetDimNum();
    if (xDimNum != X_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                 "The shape dim of x must be 2");
        return ge::GRAPH_FAILED;
    }
    if (!(cuSeqLen_ >= CU_SEQ_LEN_MIN && cuSeqLen_ <= CU_SEQ_LEN_MAX)) {
        std::string reasonMsg = "The value of cu_seq_len must be within the range [" + std::to_string(CU_SEQ_LEN_MIN) +
                                ", " + std::to_string(CU_SEQ_LEN_MAX) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(cuSeqLen_).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!(dim_ >= DIM_MIN && dim_ <= DIM_MAX && dim_ % DIM_MIN == 0)) {
        std::string reasonMsg = "The value of dim must be within the range [" + std::to_string(DIM_MIN) + ", " +
                                std::to_string(DIM_MAX) + "] and must be a multiple of " + std::to_string(DIM_MIN);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim_).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckWeightDim()
{
    uint64_t weightDimNum = weightShape_.GetDimNum();
    if (weightDimNum != WEIGHT_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "weight",
                                                 std::to_string(weightDimNum).c_str(),
                                                 "The shape dim of weight must be 2");
        return ge::GRAPH_FAILED;
    }
    uint64_t weightDim = weightShape_.GetDim(DIM_1);
    if (weightDim != dim_) {
        std::string reasonMsg = "Shape [1] of weight must be equal to shape [1] of x (" + std::to_string(dim_) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight", std::to_string(weightDim).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kernelWidth_ != 3) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "weight", std::to_string(kernelWidth_).c_str(),
                                              "The value of kernel_width must be equal to 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckCacheStatesDim()
{
    uint64_t cacheStatesDimNum = cacheStatesShape_.GetDimNum();
    if (cacheStatesDimNum != CACHE_STATES_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "cache_states",
                                                 std::to_string(cacheStatesDimNum).c_str(),
                                                 "The shape dim of cache_states must be 3");
        return ge::GRAPH_FAILED;
    }
    uint64_t cacheStatesDim1 = cacheStatesShape_.GetDim(DIM_1);
    if (cacheStatesDim1 < (kernelWidth_ - 1)) {
        std::string reasonMsg =
            "Shape [1] of cache_states must be greater than or equal to " + std::to_string(kernelWidth_ - 1);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cache_states",
                                              std::to_string(cacheStatesDim1).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    uint64_t cacheStatesDim2 = cacheStatesShape_.GetDim(DIM_2);
    if (cacheStatesDim2 != dim_) {
        std::string reasonMsg = "Shape [2] of cache_states must be equal to " + std::to_string(dim_);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cache_states",
                                              std::to_string(cacheStatesDim2).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckIndexDims()
{
    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexStorageShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query_start_loc", "nullptr",
                                              "query_start_loc cannot be nullptr");
        return ge::GRAPH_FAILED;
    }
    // cache_indices 可选：为空时 kernel 使用 batch_idx 作为 cache line
    auto seqStartIndexShape = seqStartIndexStorageShape->GetOriginShape();
    uint64_t seqStartIndexDimNum = seqStartIndexShape.GetDimNum();
    if (seqStartIndexDimNum != SEQ_START_INDEX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                                 std::to_string(seqStartIndexDimNum).c_str(),
                                                 "The shape dim of query_start_loc must be 1");
        return ge::GRAPH_FAILED;
    }
    uint64_t seqStartIndexDim0 = seqStartIndexShape.GetDim(DIM_0);
    if (seqStartIndexDim0 != (batch_ + 1)) {
        std::string reasonMsg = "Shape [0] of query_start_loc must be equal to " + std::to_string(batch_ + 1);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                              std::to_string(seqStartIndexDim0).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// cache_indices 校验：非 APC 为 1D [batch]；APC 为 2D [batch, maxNumBlocks]；shape[0] 必须等于 batch
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckCacheIndicesDim()
{
    auto cacheIndicesStorage = context_->GetOptionalInputShape(INPUT_CACHE_INDICES_INDEX);
    if (cacheIndicesStorage == nullptr) {
        return ge::GRAPH_SUCCESS; // 可选输入，缺省即跳过
    }
    auto cacheIndicesShape = cacheIndicesStorage->GetOriginShape();
    size_t cacheIndicesDimNum = cacheIndicesShape.GetDimNum();
    // APC 开启时检测逻辑已强制 cache_indices 为 2D；非 APC 时必须为 1D
    if (apcEnabled_ != 1 && cacheIndicesDimNum != SEQ_START_INDEX_DIM_NUM) {
        std::string reasonMsg;
        if (cacheIndicesDimNum == DIM_2) {
            // 2D cache_indices 是 APC 模式特征；当前 APC 未启用，无法匹配该维度
            reasonMsg = "cache_indices must be 2D when APC is enabled; "
                        "currently APC is not enabled, provide a 1D cache_indices instead";
        } else {
            reasonMsg = "The shape dim of cache_indices must be 1";
        }

        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                 std::to_string(cacheIndicesDimNum).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    uint64_t cacheIndicesDim0 = static_cast<uint64_t>(cacheIndicesShape.GetDim(DIM_0));
    if (cacheIndicesDim0 != batch_) {
        std::string reasonMsg = "Shape [0] of cache_indices must be equal to batch (" + std::to_string(batch_) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                              std::to_string(cacheIndicesDim0).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 属性合法性校验：activation_mode / pad_slot_id / run_mode / residual_connection /
//                 block_size / max_query_len / conv_mode / max_draft_tokens
// BSH 模板仅处理 2D 输入，max_query_len 不允许为 -1
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // activation_mode ∈ {0, 1, 2}
    int64_t activationMode = 0;
    const int64_t *p = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    if (p != nullptr) {
        activationMode = *p;
    }
    if (activationMode != 0 && activationMode != 1 && activationMode != 2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "activation_mode",
                                              std::to_string(activationMode).c_str(),
                                              "The value of activation_mode must be 0, 1 or 2");
        return ge::GRAPH_FAILED;
    }

    // pad_slot_id 不做范围校验（允许任意 int64 值）

    // run_mode ∈ {0, 1}
    if (runMode_ != 0 && runMode_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "run_mode", std::to_string(runMode_).c_str(),
                                              "The value of run_mode must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    // residual_connection ∈ {0, 1}
    if (residualConnection_ != 0UL && residualConnection_ != 1UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "residual_connection",
                                              std::to_string(residualConnection_).c_str(),
                                              "The value of residual_connection must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    // block_size ∈ {0} ∪ [2, INT64_MAX]（属性类型 int64，读入 uint64_t；负值会被 reinterpret 为大于 INT64_MAX 的 uint64
    // 而被拦截）
    constexpr uint64_t blockSizeUpper = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (blockSize_ != 0UL && (blockSize_ < 2UL || blockSize_ > blockSizeUpper)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", std::to_string(blockSize_).c_str(),
                                              "The value of block_size must be 0 or greater than or equal to 2");
        return ge::GRAPH_FAILED;
    }

    // BSH 模板仅处理 2D 输入 x，max_query_len 必须 > 0
    if (maxQueryLen_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "max_query_len",
                                              std::to_string(maxQueryLen_).c_str(),
                                              "The value of max_query_len must be greater than 0 when x is 2D");
        return ge::GRAPH_FAILED;
    }

    // 2D 输入 x 的第一维 cu_seq_len 不能超过 batch * max_query_len
    if (cuSeqLen_ > batch_ * static_cast<uint64_t>(maxQueryLen_)) {
        std::string reasonMsg = "Shape [0] of x must be less than or equal to batch * max_query_len (" +
                                std::to_string(batch_ * static_cast<uint64_t>(maxQueryLen_)) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(cuSeqLen_).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // conv_mode ∈ {0, 1}
    if (convMode_ != 0UL && convMode_ != 1UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "conv_mode", std::to_string(convMode_).c_str(),
                                              "The value of conv_mode must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    // max_draft_tokens ∈ [0, 16]
    if (maxDraftTokens_ < 0 || maxDraftTokens_ > 16) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "max_draft_tokens",
                                              std::to_string(maxDraftTokens_).c_str(),
                                              "The value of max_draft_tokens must be between 0 and 16");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckInputDim()
{
    if (CheckXDim() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckWeightDim() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckCacheStatesDim() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckIndexDims() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckCacheIndicesDim() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckAttrs() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (!(batch_ >= BATCH_MIN && batch_ <= BATCH_MAX)) {
        std::string reasonMsg = "The value of batch must be within the range [" + std::to_string(BATCH_MIN) + ", " +
                                std::to_string(BATCH_MAX) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "batch", std::to_string(batch_).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 检查输入参数
ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckInputParams()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::CheckOutputParams()
{
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    auto outputYType = outputYDesc->GetDataType();
    if (xType_ != outputYType) {
        std::string dtypeMsg = Ops::Base::ToString(xType_) + " and " + Ops::Base::ToString(outputYType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and y", dtypeMsg.c_str(),
                                               "The dtypes of x and y must be the same");
        return ge::GRAPH_FAILED;
    }

    auto outputCacheStatesDesc = context_->GetOutputDesc(OUTPUT_CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputCacheStatesDesc);
    auto outputCacheStatesType = outputCacheStatesDesc->GetDataType();
    if (xType_ != outputCacheStatesType) {
        std::string dtypeMsg = Ops::Base::ToString(xType_) + " and " + Ops::Base::ToString(outputCacheStatesType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and output_cache_states", dtypeMsg.c_str(),
                                               "The dtypes of x and output_cache_states must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetInputShapes()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X_INDEX));
    xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetOriginShape();
    cuSeqLen_ = xShape_.GetDim(DIM_0);
    dim_ = xShape_.GetDim(DIM_1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_WEIGHT_INDEX));
    weightShape_ = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetOriginShape();
    kernelWidth_ = weightShape_.GetDim(DIM_0);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_CACHE_STATES_INDEX));
    cacheStatesShape_ = context_->GetInputShape(INPUT_CACHE_STATES_INDEX)->GetOriginShape();

    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexStorageShape != nullptr) {
        seqStartIndexShape_ = seqStartIndexStorageShape->GetOriginShape();
        // 完整 dim 校验由 CheckIndexDims 完成（要求 dim==1），此处仅保证读 dim0 不越界
        if (seqStartIndexShape_.GetDimNum() < SEQ_START_INDEX_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                                     std::to_string(seqStartIndexShape_.GetDimNum()).c_str(),
                                                     "The shape dim of query_start_loc must be 1");
            return ge::GRAPH_FAILED;
        }
        // query_start_loc 形状为 (batch + 1,)，dim0 必须 >= 2 才能反推非空 batch（README 约束 batch >= 1）
        int64_t seqStartDim0 = seqStartIndexShape_.GetDim(DIM_0);
        if (seqStartDim0 < 2) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "query_start_loc", std::to_string(seqStartDim0).c_str(),
                "The dim 0 of query_start_loc must be >= 2 (shape is [batch+1], batch range [1, 256])");
            return ge::GRAPH_FAILED;
        }
        batch_ = seqStartDim0 - 1;
    } else {
        batch_ = 1;
        seqStartIndexShape_ = gert::Shape({0, static_cast<int64_t>(cuSeqLen_)});
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetInputDtypes()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X_INDEX));
    xType_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_WEIGHT_INDEX));
    weightType_ = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();

    xDtypeSize_ = GetSizeByDataType(xType_);
    if (xDtypeSize_ == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xType_).c_str(),
                                              "The dtype size of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetInputStrides()
{
    if (context_->InputIsView(INPUT_X_INDEX)) {
        auto *xStride = context_->GetInputStride(INPUT_X_INDEX);
        OP_CHECK_IF(xStride == nullptr || xStride->GetDimNum() == 0,
                    OP_LOGE(context_->GetNodeName(), "x stride is invalid."), return ge::GRAPH_FAILED);
        if (xStride->GetDimNum() != xShape_.GetDimNum()) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x stride",
                                                     std::to_string(xStride->GetDimNum()).c_str(),
                                                     "The shape dim of x stride must be equal to that of x");
            return ge::GRAPH_FAILED;
        }
        xStride_ = xStride->GetStride(DIM_0);
    } else {
        xStride_ = dim_;
    }

    if (context_->InputIsView(INPUT_CACHE_STATES_INDEX)) {
        auto *cacheStride = context_->GetInputStride(INPUT_CACHE_STATES_INDEX);
        OP_CHECK_IF(cacheStride == nullptr || cacheStride->GetDimNum() == 0,
                    OP_LOGE(context_->GetNodeName(), "cache_states stride is invalid."), return ge::GRAPH_FAILED);
        if (cacheStride->GetDimNum() != cacheStatesShape_.GetDimNum()) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "cache_states stride", std::to_string(cacheStride->GetDimNum()).c_str(),
                "The shape dim of cache_states stride must be equal to that of cache_states");
            return ge::GRAPH_FAILED;
        }
        cacheStride0_ = cacheStride->GetStride(DIM_0);
        cacheStride1_ = cacheStride->GetStride(DIM_1);
    } else {
        cacheStride0_ = cacheStatesShape_.GetDim(DIM_1) * dim_;
        cacheStride1_ = dim_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("FusedCausalConv1dCutBSH", "context is null"), return ge::GRAPH_FAILED);

    if (GetInputShapes() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (GetInputDtypes() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    padSlotId_ = -1;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_PAD_SLOT_ID_INDEX) != nullptr) {
        padSlotId_ = *(context_->GetAttrs()->GetInt(ATTR_PAD_SLOT_ID_INDEX));
    }
    residualConnection_ = 0;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_RESIDUAL_CONNECTION_INDEX) != nullptr) {
        residualConnection_ = *(context_->GetAttrs()->GetInt(ATTR_RESIDUAL_CONNECTION_INDEX));
    }
    runMode_ = 0;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_RUN_MODE_INDEX) != nullptr) {
        runMode_ = *(context_->GetAttrs()->GetInt(ATTR_RUN_MODE_INDEX));
    }

    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *p;
        if ((p = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX)) != nullptr) {
            blockSize_ = static_cast<uint64_t>(*p);
        }
        if ((p = attrs->GetAttrPointer<int64_t>(ATTR_CONV_MODE_INDEX)) != nullptr) {
            convMode_ = static_cast<uint64_t>(*p);
        }
        // 原地更新开关：由构造时传入的 isInplace_ 决定（拆分后不再从 attr 读取）
        inplace_ = isInplace_ ? 1UL : 0UL;
        if ((p = attrs->GetAttrPointer<int64_t>(ATTR_MAX_QUERY_LEN_INDEX)) != nullptr) {
            maxQueryLen_ = *p;
        }
    }

    // APC 初判：block_idx_first_scheduled_token 存在即视为用户意图开启 APC，
    // 随后立即校验全部必要条件，任一不符直接报错
    apcEnabled_ = 0;
    maxNumBlocks_ = 0;
    hasCacheIndices_ = (context_->GetOptionalInputShape(INPUT_CACHE_INDICES_INDEX) != nullptr) ? 1UL : 0UL;

    if (context_->GetOptionalInputShape(INPUT_BLOCK_IDX_FIRST_INDEX) != nullptr) {
        apcEnabled_ = 1;
        if (context_->GetOptionalInputShape(INPUT_BLOCK_IDX_LAST_INDEX) == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_idx_last_scheduled_token", "nullptr",
                                                  "block_idx_last_scheduled_token cannot be nullptr when APC");
            return ge::GRAPH_FAILED;
        }
        if (context_->GetOptionalInputShape(INPUT_INITIAL_STATE_IDX_INDEX) == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "initial_state_idx", "nullptr",
                                                  "initial_state_idx cannot be nullptr when APC");
            return ge::GRAPH_FAILED;
        }
        auto cacheIndicesStorage = context_->GetOptionalInputShape(INPUT_CACHE_INDICES_INDEX);
        if (cacheIndicesStorage == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cache_indices", "nullptr",
                                                  "cache_indices cannot be nullptr when APC is enabled");
            return ge::GRAPH_FAILED;
        }
        auto cacheIndicesShape = cacheIndicesStorage->GetOriginShape();
        if (cacheIndicesShape.GetDimNum() != DIM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                     std::to_string(cacheIndicesShape.GetDimNum()).c_str(),
                                                     "The shape dim of cache_indices must be 2 when APC is enabled");
            return ge::GRAPH_FAILED;
        }
        if (blockSize_ == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", "0",
                                                  "The value of block_size must not be 0 when APC is enabled");
            return ge::GRAPH_FAILED;
        }
        maxNumBlocks_ = static_cast<uint64_t>(cacheIndicesShape.GetDim(DIM_1));
    }
    hasAcceptTokenNum_ = (context_->GetOptionalInputShape(INPUT_NUM_ACCEPTED_TOKEN_INDEX) != nullptr) ? 1UL : 0UL;
    hasNumComputedTokens_ = (context_->GetOptionalInputShape(INPUT_NUM_COMPUTED_TOKENS_INDEX) != nullptr) ? 1UL : 0UL;

    // convMode!=0 或 apcEnabled 时，必须提供 num_computed_tokens
    if (convMode_ != 0 || apcEnabled_ == 1) {
        if (hasNumComputedTokens_ == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "num_computed_tokens", "nullptr",
                "num_computed_tokens cannot be nullptr when conv_mode != 0 or APC is enabled");
            return ge::GRAPH_FAILED;
        }
    }

    if (GetInputStrides() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    if (CheckInputParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputParams() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 辅助函数：计算切cu_seq_len时的核间切分信息（均分多尾核策略）
FusedCausalConv1dCutBSHTiling::CuSeqLenSplitInfo FusedCausalConv1dCutBSHTiling::CalculateCuSeqLenSplitInfo(
    uint64_t cuSeqLen, uint64_t bsOverlap, uint64_t coreNum) const
{
    CuSeqLenSplitInfo info;

    if (coreNum == 0) {
        return info;
    }

    // 均分策略：将带重叠的总长度均分到所有核
    // effectiveTotal = cuSeqLen + (coreNum - 1) * overlap
    info.effectiveTotal = cuSeqLen + (coreNum - 1) * bsOverlap;

    // 向下取整的基础长度
    info.baseLen = info.effectiveTotal / coreNum;

    // 余数（需要多分配的块数）
    info.remainder = info.effectiveTotal % coreNum;

    // 前 remainder 个核是大核，后面是小核
    if (info.remainder > 0) {
        info.blockFactor = info.baseLen + 1; // 大核载入长度
        info.blockTailFactor = info.baseLen; // 小核载入长度
    } else {
        // 所有核均匀分配
        info.blockFactor = info.baseLen;
        info.blockTailFactor = info.baseLen;
    }
    info.realCoreNum = coreNum; // 所有核都使用

    return info;
}

// 计算二维切分时的tiling（支持不均匀切分 + dim循环）
ge::graphStatus FusedCausalConv1dCutBSHTiling::CalcCoreUbTiling(uint64_t coreDim, uint64_t coreBS,
                                                                uint64_t bsBlockFactor, int64_t availableUbSize,
                                                                uint64_t weightCacheCoeffPerDim, uint64_t bsOverlap,
                                                                uint64_t &ubFactorBS, uint64_t &ubFactorDim,
                                                                uint64_t &loopNumBS, uint64_t &ubTailFactorBS,
                                                                uint64_t &loopNumDim, uint64_t &ubTailFactorDim)
{
    constexpr uint64_t XY_BUFFER_FACTOR = 2 * DOUBLE_BUFFER_NUM; // 2(x双) + 2(y双) = 4
    int64_t maxUbDim = availableUbSize / (weightCacheCoeffPerDim + coreBS * xDtypeSize_ * XY_BUFFER_FACTOR);
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (coreDim < DIM_ALIGN_ELEMENTS) {
        ubFactorDim = coreDim;
        ubFactorBS = coreBS;
        loopNumDim = 1;
        ubTailFactorDim = coreDim;
    } else if (maxUbDim >= static_cast<int64_t>(DIM_ALIGN_ELEMENTS)) {
        ubFactorBS = coreBS;
        ubFactorDim = (std::min(static_cast<uint64_t>(maxUbDim), coreDim) / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (ubFactorDim == 0)
            ubFactorDim = DIM_ALIGN_ELEMENTS;
    } else {
        ubFactorDim = DIM_ALIGN_ELEMENTS;
        int64_t availableForXY = availableUbSize - static_cast<int64_t>(weightCacheCoeffPerDim * ubFactorDim);
        int64_t maxBS = availableForXY / static_cast<int64_t>(ubFactorDim * xDtypeSize_ * XY_BUFFER_FACTOR);
        ubFactorBS =
            static_cast<uint64_t>(std::min(std::max(maxBS, static_cast<int64_t>(1)), static_cast<int64_t>(coreBS)));
        if (ubFactorBS == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling");
            return ge::GRAPH_FAILED;
        }
    }
    if (ubFactorDim > coreDim) {
        ubFactorDim = coreDim;
    }
    loopNumDim = (coreDim <= ubFactorDim) ? 1 : (coreDim + ubFactorDim - 1) / ubFactorDim;
    ubTailFactorDim = (coreDim <= ubFactorDim) ? coreDim : coreDim - (loopNumDim - 1) * ubFactorDim;

    if (bsBlockFactor <= ubFactorBS) {
        loopNumBS = 1;
    } else {
        uint64_t remaining = bsBlockFactor - ubFactorBS;
        loopNumBS = 1 + Ops::Base::CeilDiv(remaining, ubFactorBS - bsOverlap);
    }
    uint64_t lastLoopInput = bsBlockFactor - (loopNumBS - 1) * (ubFactorBS - bsOverlap);
    ubTailFactorBS = std::min(lastLoopInput, ubFactorBS);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::Calculate2DTiling()
{
    uint64_t bsOverlap = kernelWidth_ - 1;
    auto alignUp = [](uint64_t x, uint64_t a) -> uint64_t { return ((x + a - 1) / a) * a; };
    constexpr uint64_t META_ALIGN = 32UL;
    uint64_t metaInt32Bytes = batch_ * sizeof(int32_t);
    uint64_t metaTBufBytes = alignUp((batch_ + 1) * sizeof(int32_t), META_ALIGN); // startLoc
    if (hasCacheIndices_ == 1 && apcEnabled_ == 0) {
        metaTBufBytes += alignUp(metaInt32Bytes, META_ALIGN); // 1D cacheIdx
    }
    if (hasNumComputedTokens_ == 1) {
        metaTBufBytes += alignUp(metaInt32Bytes, META_ALIGN); // num_computed_tokens
    }
    if (hasAcceptTokenNum_ == 1) {
        metaTBufBytes += alignUp(metaInt32Bytes, META_ALIGN); // num_accepted_tokens
    }
    if (apcEnabled_ == 1) {
        metaTBufBytes += alignUp(metaInt32Bytes, META_ALIGN) * 3; // blockIdxFirst/Last + initialStateIdx
        // metaTBufBytes += alignUp(maxNumBlocks_ * sizeof(int32_t), META_ALIGN);  // 2D cacheIndices (单 batch)
    }
    uint64_t fixedUbSize = metaTBufBytes;

    // weight 双 buffer + cache 双 buffer：(K×2 + stateLen×2) × sizeof(T)
    // 注：cache UB 按 stateLen_ × maxUbDim × sizeof(T) 分配（正向加载，cachedStateLen 最大为 stateLen_）
    uint64_t cacheStatesDim1 = cacheStatesShape_.GetDim(DIM_1); // stateLen
    uint64_t weightCacheCoeffPerDim =
        (kernelWidth_ * DOUBLE_BUFFER_NUM + cacheStatesDim1 * DOUBLE_BUFFER_NUM) * xDtypeSize_;
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - static_cast<int64_t>(fixedUbSize);
    OP_CHECK_IF(
        availableUbSize <= 0,
        OP_LOGE(context_->GetNodeName(), "FusedCausalConv1dCutBSH availableUbSize <= 0 (ubSize_=%lu fixedUbSize=%lu)",
                ubSize_, fixedUbSize),
        return ge::GRAPH_FAILED);

    uint64_t mainCoreDim = (dimBlockFactor_ > 0) ? dimBlockFactor_ : DIM_ALIGN_ELEMENTS;
    if (CalcCoreUbTiling(mainCoreDim, bsBlockFactor_, bsBlockFactor_, availableUbSize, weightCacheCoeffPerDim,
                         bsOverlap, ubFactorBS_, ubFactorDim_, loopNumBS_, ubTailFactorBS_, loopNumDim_,
                         ubTailFactorDim_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tailCoreDim = (dimRemainderCores_ > 0) ? dimBlockTailFactor_ : dimBlockFactor_;

    if (tailCoreDim == 0) {
        tailCoreDim = DIM_ALIGN_ELEMENTS;
    }
    if (CalcCoreUbTiling(tailCoreDim, bsBlockTailFactor_, bsBlockTailFactor_, availableUbSize, weightCacheCoeffPerDim,
                         bsOverlap, tailBlockubFactorBS_, tailBlockubFactorDim_, tailBlockloopNumBS_,
                         tailBlockubTailFactorBS_, tailBlockloopNumDim_,
                         tailBlockubTailFactorDim_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (dimRemainderElems_ > 0) {
        uint64_t lastCoreBaseDim = (dimRemainderCores_ < dimCoreNum_) ? dimBlockTailFactor_ : dimBlockFactor_;
        uint64_t lastCoreDim = lastCoreBaseDim + dimRemainderElems_;
        // last core 的 BS 长度与普通尾核相同（BS 切分与 dim 切分独立）
        uint64_t lastCoreBS = bsBlockTailFactor_;
        uint64_t lastCoreubFactorBS_unused = 0;
        uint64_t lastCoreloopNumBS_unused = 0;
        uint64_t lastCoreubTailFactorBS_unused = 0;
        if (CalcCoreUbTiling(lastCoreDim, lastCoreBS, lastCoreBS, availableUbSize, weightCacheCoeffPerDim, bsOverlap,
                             lastCoreubFactorBS_unused, lastCoreubFactorDim_, lastCoreloopNumBS_unused,
                             lastCoreubTailFactorBS_unused, lastCoreloopNumDim_,
                             lastCoreubTailFactorDim_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        // 兜底：last core 参数 = 基础核参数
        // dimRemainderCores_ < dimCoreNum_ 表示有尾核，last core 基于尾核；否则基于主核
        bool hasTailCore = (dimRemainderCores_ < dimCoreNum_);
        if (hasTailCore) {
            lastCoreloopNumDim_ = tailBlockloopNumDim_;
            lastCoreubFactorDim_ = tailBlockubFactorDim_;
            lastCoreubTailFactorDim_ = tailBlockubTailFactorDim_;
        } else {
            lastCoreloopNumDim_ = loopNumDim_;
            lastCoreubFactorDim_ = ubFactorDim_;
            lastCoreubTailFactorDim_ = ubTailFactorDim_;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::SearchBestCoreSplit(uint64_t N, uint64_t bsOverlap,
                                                                   uint64_t &bestDimCores,
                                                                   CuSeqLenSplitInfo &bestBSSplitInfo)
{
    uint64_t bestUsed = 0;
    bool bestIsBalanced = false; // 当前最佳是否 BS 均分
    for (uint64_t dc = N; dc >= 1; --dc) {
        uint64_t base = N / dc;
        if (base == 0)
            continue;

        uint64_t maxAllowedBSByCore = totalCoreNum_ / dc;
        if (maxAllowedBSByCore == 0)
            continue;

        uint64_t maxAllowedBSBySeqLen = (cuSeqLen_ > bsOverlap) ? (cuSeqLen_ - bsOverlap) : 1;
        uint64_t maxAllowedBS = std::min(maxAllowedBSByCore, maxAllowedBSBySeqLen);

        // 尝试一组候选 coreNum：从 maxAllowedBS 起递减，至少检查均分点
        // 简化：只检查 maxAllowedBS 与最近的均分候选（找一个能整除 effectiveTotal 的）
        for (uint64_t bsCore = maxAllowedBS; bsCore >= 1; --bsCore) {
            auto splitInfo = CalculateCuSeqLenSplitInfo(cuSeqLen_, bsOverlap, bsCore);
            uint64_t usedCores = dc * splitInfo.realCoreNum;
            bool isBalanced = (splitInfo.remainder == 0);

            // 选优条件
            bool better = false;
            if (!bestIsBalanced && isBalanced) {
                better = true; // 不均分→均分 升级
            } else if (bestIsBalanced == isBalanced) {
                if (usedCores > bestUsed)
                    better = true;
                else if (usedCores == bestUsed && dc > bestDimCores)
                    better = true;
            }
            if (better) {
                bestDimCores = dc;
                bestUsed = usedCores;
                bestIsBalanced = isBalanced;
                bestBSSplitInfo = splitInfo;
            }

            // 内层：只要找到当前 dc 下的均分方案即可跳出（更小 bsCore 不会更优）
            if (isBalanced)
                break;
            // 非均分：只评估最大 bsCore 一种（避免 O(N²) 全枚举）
            if (bsCore == maxAllowedBS)
                break;
        }
        if (bestUsed == totalCoreNum_ && bestIsBalanced)
            break;
    }

    if (bestUsed == 0) {
        OP_LOGE(context_->GetNodeName(), "Failed to find valid tiling strategy");
        return ge::GRAPH_FAILED;
    }
    realCoreNum_ = bestUsed;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::ApplyDimSplit(uint64_t N, uint64_t bestDimCores)
{
    if (bestDimCores == 0) {
        OP_LOGE(context_->GetNodeName(), "ApplyDimSplit: bestDimCores is 0");
        return ge::GRAPH_FAILED;
    }
    constexpr uint64_t DIM_GRANULARITY = DIM_ALIGN_ELEMENTS;
    uint64_t base = N / bestDimCores;
    uint64_t remainder = N % bestDimCores;

    dimCoreNum_ = bestDimCores;
    dimRemainderCores_ = remainder;
    dimBlockFactor_ = (remainder > 0 ? base + 1 : base) * DIM_GRANULARITY;
    dimBlockTailFactor_ = base * DIM_GRANULARITY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::DoOpTiling()
{
    uint64_t bsOverlap = kernelWidth_ - 1;
    constexpr uint64_t DIM_GRANULARITY = DIM_ALIGN_ELEMENTS;

    uint64_t N = dim_ / DIM_GRANULARITY;
    // 允许 dim % 128 != 0，余数元素由最后一个 dim 核承担
    dimRemainderElems_ = dim_ % DIM_GRANULARITY;

    // dim < 128 场景（如 dim=16/32/64）
    //   N=0 时没有可用的"整 128 块"供切分，整个 dim 全部走 dimRemainderElems_。
    //   贪心搜索用 searchN=1（最多 1 个 dim 核，即 last core），避免 base==0 的非法切分。
    //   此时 dimCoreNum_=1，dimBlockFactor_=0，dimBlockTailFactor_=0，
    //   lastCoreDim = 0 + dim_ = dim_，该唯一核承担全部 dim 元素。
    uint64_t searchN = (N > 0) ? N : 1;

    uint64_t bestDimCores = 0;
    CuSeqLenSplitInfo bestBSSplitInfo = {};
    if (SearchBestCoreSplit(searchN, bsOverlap, bestDimCores, bestBSSplitInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ApplyDimSplit(N, bestDimCores) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    bsCoreNum_ = bestBSSplitInfo.realCoreNum;
    bsRemainderCores_ = bestBSSplitInfo.remainder;
    bsBlockFactor_ = bestBSSplitInfo.blockFactor;
    bsBlockTailFactor_ = bestBSSplitInfo.blockTailFactor;

    return Calculate2DTiling();
}

uint64_t FusedCausalConv1dCutBSHTiling::GetTilingKey() const
{
    // 根据数据类型返回不同的 tiling key
    if (xType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_BSH_BF16;
    } else if (xType_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_BSH_FP16;
    }
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;

    // inplace=true 场景：y 需要一块独立 workspace 暂存，避免主循环写 y 污染
    // xGM_（否则 SyncAll 后 WriteDeferredCacheSimple / FillApcBlockCaches 从
    // xGM_ 读到的是 y，不是原始 x）。主循环结束后再从 y workspace 搬回 xGM_。
    // 大小：cuSeqLen × xStride × sizeof(T)，按 32B 对齐，位于 SYS_WORKSPACE_SIZE 之后。
    if (inplace_ == 1UL) {
        constexpr uint64_t WS_ALIGN = 32UL;
        uint64_t yBackupBytes = cuSeqLen_ * xStride_ * xDtypeSize_;
        yBackupBytes = ((yBackupBytes + WS_ALIGN - 1) / WS_ALIGN) * WS_ALIGN;
        workspaceSize_ += yBackupBytes;
    }

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBSHTiling::PostTiling()
{
    // Set block dimension (number of cores to use)
    context_->SetBlockDim(realCoreNum_);

    // Populate tiling data
    tilingData_.loopNumBS = loopNumBS_;
    tilingData_.loopNumDim = loopNumDim_;
    tilingData_.ubFactorBS = ubFactorBS_;
    tilingData_.ubTailFactorBS = ubTailFactorBS_;
    tilingData_.ubFactorDim = ubFactorDim_;
    tilingData_.ubTailFactorDim = ubTailFactorDim_;
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;

    // dim方向核间切分信息
    tilingData_.dimCoreNum = dimCoreNum_;
    tilingData_.dimRemainderCores = dimRemainderCores_;
    tilingData_.dimBlockFactor = dimBlockFactor_;
    tilingData_.dimBlockTailFactor = dimBlockTailFactor_;

    // BS方向核间切分信息
    tilingData_.bsCoreNum = bsCoreNum_;
    tilingData_.bsRemainderCores = bsRemainderCores_;
    tilingData_.bsBlockFactor = bsBlockFactor_;
    tilingData_.bsBlockTailFactor = bsBlockTailFactor_;

    // 核数信息
    tilingData_.realCoreNum = realCoreNum_;

    // 其他参数
    tilingData_.kernelWidth = kernelWidth_;
    tilingData_.cuSeqLen = cuSeqLen_;
    tilingData_.dim = dim_;
    tilingData_.batch = batch_;
    tilingData_.padSlotId = padSlotId_;
    tilingData_.xStride = xStride_;
    tilingData_.cacheStride0 = cacheStride0_;
    tilingData_.cacheStride1 = cacheStride1_;
    tilingData_.stateLen = cacheStatesShape_.GetDim(DIM_1);
    tilingData_.residualConnection = residualConnection_;
    tilingData_.apcEnabled = apcEnabled_;
    tilingData_.blockSize = blockSize_;
    tilingData_.maxNumBlocks = maxNumBlocks_;
    tilingData_.convMode = convMode_;
    tilingData_.inplace = inplace_;
    tilingData_.hasAcceptTokenNum = hasAcceptTokenNum_;
    tilingData_.hasNumComputedTokens = hasNumComputedTokens_;
    tilingData_.hasCacheIndices = hasCacheIndices_;
    tilingData_.dimRemainderElems = dimRemainderElems_;
    tilingData_.lastCoreloopNumDim = lastCoreloopNumDim_;
    tilingData_.lastCoreubFactorDim = lastCoreubFactorDim_;
    tilingData_.lastCoreubTailFactorDim = lastCoreubTailFactorDim_;

    // Save tiling data to buffer
    auto tilingDataSize = sizeof(FusedCausalConv1dCutBSHTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void FusedCausalConv1dCutBSHTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "cuSeqLen: " << cuSeqLen_ << std::endl;
    info << "dim: " << dim_ << std::endl;
    info << "kernelWidth: " << kernelWidth_ << std::endl;
    info << "batch: " << batch_ << std::endl;
    info << "padSlotId: " << padSlotId_ << std::endl;

    // dim方向核间切分信息
    info << "dimCoreNum: " << dimCoreNum_ << std::endl;
    info << "dimRemainderCores: " << dimRemainderCores_ << std::endl;
    info << "dimBlockFactor: " << dimBlockFactor_ << std::endl;
    info << "dimBlockTailFactor: " << dimBlockTailFactor_ << std::endl;

    // BS方向核间切分信息
    info << "bsCoreNum: " << bsCoreNum_ << std::endl;
    info << "bsRemainderCores: " << bsRemainderCores_ << std::endl;
    info << "bsBlockFactor: " << bsBlockFactor_ << std::endl;
    info << "bsBlockTailFactor: " << bsBlockTailFactor_ << std::endl;

    // 核数信息
    info << "realCoreNum: " << realCoreNum_ << std::endl;

    // 核内切分参数
    info << "loopNumBS: " << loopNumBS_ << std::endl;
    info << "loopNumDim: " << loopNumDim_ << std::endl;
    info << "ubFactorBS: " << ubFactorBS_ << std::endl;
    info << "ubTailFactorBS: " << ubTailFactorBS_ << std::endl;
    info << "ubFactorDim: " << ubFactorDim_ << std::endl;
    info << "ubTailFactorDim: " << ubTailFactorDim_ << std::endl;
    info << "tailBlockloopNumBS: " << tailBlockloopNumBS_ << std::endl;
    info << "tailBlockloopNumDim: " << tailBlockloopNumDim_ << std::endl;
    info << "tailBlockubFactorBS: " << tailBlockubFactorBS_ << std::endl;
    info << "tailBlockubTailFactorBS: " << tailBlockubTailFactorBS_ << std::endl;
    info << "tailBlockubFactorDim: " << tailBlockubFactorDim_ << std::endl;
    info << "tailBlockubTailFactorDim: " << tailBlockubTailFactorDim_ << std::endl;
    info << "residualConnection: " << residualConnection_ << std::endl;
    info << "xStride: " << xStride_ << std::endl;
    info << "cacheStride0: " << cacheStride0_ << std::endl;
    info << "cacheStride1: " << cacheStride1_ << std::endl;
    info << "apcEnabled: " << apcEnabled_ << std::endl;
    info << "blockSize: " << blockSize_ << std::endl;
    info << "maxNumBlocks: " << maxNumBlocks_ << std::endl;
    info << "convMode: " << convMode_ << std::endl;
    info << "inplace: " << inplace_ << std::endl;
    info << "hasAcceptTokenNum: " << hasAcceptTokenNum_ << std::endl;
    info << "hasNumComputedTokens: " << hasNumComputedTokens_ << std::endl;
    info << "hasCacheIndices: " << hasCacheIndices_ << std::endl;
    info << "maxQueryLen: " << maxQueryLen_ << std::endl;
    info << "dimRemainderElems: " << dimRemainderElems_ << std::endl;
    info << "lastCoreloopNumDim: " << lastCoreloopNumDim_ << std::endl;
    info << "lastCoreubFactorDim: " << lastCoreubFactorDim_ << std::endl;
    info << "lastCoreubTailFactorDim: " << lastCoreubTailFactorDim_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

} // namespace optiling
