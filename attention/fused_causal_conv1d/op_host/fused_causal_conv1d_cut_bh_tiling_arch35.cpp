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
 * \file fused_causal_conv1d_cut_bh_tiling_arch35.cpp
 * \brief FusedCausalConv1dCutBH tiling implementation
 */

#include "fused_causal_conv1d_cut_bh_tiling_arch35.h"
#include <algorithm>
#include "securec.h"

namespace optiling {

bool FusedCausalConv1dCutBHTiling::IsCapable()
{
    return true;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        // 无运行时平台信息，降级使用编译期 CompileInfo
        auto compileInfoPtr = reinterpret_cast<const FusedCausalConv1dCutBHCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        // 与运行时路径对齐：coreNum/ubSize 必须 > 0，否则后续核内 UB 切分会下溢
        if (compileInfoPtr->coreNum == 0UL) {
            OP_LOGE(context_->GetNodeName(), "compile-time coreNum is 0");
            return ge::GRAPH_FAILED;
        }
        if (compileInfoPtr->ubSize == 0UL) {
            OP_LOGE(context_->GetNodeName(), "compile-time ubSize is 0");
            return ge::GRAPH_FAILED;
        }
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        // 正常路径：从运行时平台信息读取
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
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("FusedCausalConv1dCutBH", "context is null"), return ge::GRAPH_FAILED);

    if (GetShapeInfo() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    if (GetTypeInfo() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    if (GetAttrInfo() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    // 形状、类型、属性读取完成后才能做合法性校验
    if (CheckInputParams() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    if (GetStrideInfo() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetShapeInfo()
{
    // 获取 x 的 shape
    auto xShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xOriginShape = xShape->GetOriginShape();
    // 支持 3D [batch, seq_len, dim] 和 2D [cu_seq_len, dim] 两种输入格式
    if (xOriginShape.GetDimNum() == DIM_3) {
        xInputMode_ = X_INPUT_3D;
        batchSize_ = xOriginShape.GetDim(DIM_0);
        seqLen_ = xOriginShape.GetDim(DIM_1);
        dim_ = xOriginShape.GetDim(DIM_2);
    } else if (xOriginShape.GetDimNum() == DIM_2) {
        // 2D 输入：cu_seq_len = 所有 batch token 总数
        xInputMode_ = X_INPUT_2D;
        cuSeqLen_ = xOriginShape.GetDim(DIM_0);
        dim_ = xOriginShape.GetDim(DIM_1);

        // 2D 输入必须提供 query_start_loc 才能反推 batch 数
        // query_start_loc 形状为 (batch + 1,)，因此 batch = dim0 - 1
        auto queryStartLocShape = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, queryStartLocShape);
        auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();
        // 完整形状校验由 ValidateQueryStartLocShape 完成（要求 dim==1），此处仅保证读 dim0 不越界
        if (queryStartLocOriginShape.GetDimNum() < DIM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                                     std::to_string(queryStartLocOriginShape.GetDimNum()).c_str(),
                                                     "The shape dim of query_start_loc must be 1");
            return ge::GRAPH_FAILED;
        }
        // query_start_loc 形状为 (batch + 1,)，dim0 必须 >= 2 才能反推非空 batch（README 约束 batch >= 1）
        int64_t queryStartLocDim0 = queryStartLocOriginShape.GetDim(DIM_0);
        if (queryStartLocDim0 < 2) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "query_start_loc", std::to_string(queryStartLocDim0).c_str(),
                "The dim 0 of query_start_loc must be >= 2 (shape is [batch+1], batch range [1, 256])");
            return ge::GRAPH_FAILED;
        }
        batchSize_ = queryStartLocDim0 - 1;
        seqLen_ = 0; // 2D 模式下 seqLen_ 在 UB 计算时由 maxQueryLen_ 替代
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                 std::to_string(xOriginShape.GetDimNum()).c_str(),
                                                 "The shape dim of x must be 2 or 3");
        return ge::GRAPH_FAILED;
    }

    // 获取 weight 形状，提取卷积核大小 kernelSize_
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();
    if (weightOriginShape.GetDimNum() != DIM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "weight",
                                                 std::to_string(weightOriginShape.GetDimNum()).c_str(),
                                                 "The shape dim of weight must be 2");
        return ge::GRAPH_FAILED;
    }

    kernelSize_ = weightOriginShape.GetDim(DIM_0);
    if (kernelSize_ != 3) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "weight", std::to_string(kernelSize_).c_str(),
                                              "The dim 0 of weight must be equal to 3");
        return ge::GRAPH_FAILED;
    }

    // 获取 conv_states 形状，提取 stateLen_（= kernel_size - 1 + max_seq_len）
    auto convStatesShape = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShape);
    auto convStatesOriginShape = convStatesShape->GetOriginShape();
    if (convStatesOriginShape.GetDimNum() != DIM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "conv_states",
                                                 std::to_string(convStatesOriginShape.GetDimNum()).c_str(),
                                                 "The shape dim of conv_states must be 3");
        return ge::GRAPH_FAILED;
    }
    stateLen_ = convStatesOriginShape.GetDim(DIM_1);

    // APC 初判：blockIdxFirstScheduledToken 存在即视为用户意图开启 APC，
    // 随后立即校验全部必要条件，任一不符直接报错（block_size 在 GetAttrInfo 中再校验）
    auto blockIdxFirstDesc = context_->GetOptionalInputDesc(BLOCK_IDX_FIRST_SCHEDULED_TOKEN_INDEX);
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    hasCacheIndices_ = (cacheIndicesDesc != nullptr) ? 1 : 0;

    apcEnable_ = (blockIdxFirstDesc != nullptr) ? 1 : 0;
    if (apcEnable_) {
        auto blockIdxLastDesc = context_->GetOptionalInputDesc(BLOCK_IDX_LAST_SCHEDULED_TOKEN_INDEX);
        if (blockIdxLastDesc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_idx_last_scheduled_token", "nullptr",
                                                  "block_idx_last_scheduled_token cannot be nullptr when APC");
            return ge::GRAPH_FAILED;
        }
        auto initialStateIdxDesc = context_->GetOptionalInputDesc(INITIAL_STATE_IDX_INDEX);
        if (initialStateIdxDesc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "initial_state_idx", "nullptr",
                                                  "initial_state_idx cannot be nullptr when APC");
            return ge::GRAPH_FAILED;
        }
        auto cacheIndicesShape = context_->GetOptionalInputShape(CACHE_INDICES_INDEX);
        if (cacheIndicesShape == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cache_indices", "nullptr",
                                                  "cache_indices cannot be nullptr when APC is enabled");
            return ge::GRAPH_FAILED;
        }
        auto cacheIndicesOriginShape = cacheIndicesShape->GetOriginShape();
        if (cacheIndicesOriginShape.GetDimNum() != DIM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                     std::to_string(cacheIndicesOriginShape.GetDimNum()).c_str(),
                                                     "The shape dim of cache_indices must be 2 when APC is enabled");
            return ge::GRAPH_FAILED;
        }
        maxNumBlocks_ = cacheIndicesOriginShape.GetDim(DIM_1);
    }

    // 判断 numComputedTokens 是否提供
    auto numComputedTokensDesc = context_->GetOptionalInputDesc(NUM_COMPUTED_TOKENS_INDEX);
    hasNumComputedTokens_ = (numComputedTokensDesc != nullptr) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetTypeInfo()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    auto convStatesDesc = context_->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesDesc);

    xDtype_ = xDesc->GetDataType();
    weightDtype_ = weightDesc->GetDataType();
    convStatesDtype_ = convStatesDesc->GetDataType();

    // 可选输入：query_start_loc
    auto queryStartLocDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    if (queryStartLocDesc != nullptr) {
        queryStartLocDtype_ = queryStartLocDesc->GetDataType();
    }

    // 可选输入：cache_indices
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    if (cacheIndicesDesc != nullptr) {
        cacheIndicesDtype_ = cacheIndicesDesc->GetDataType();
    }

    // 可选输入：num_accepted_token（存在时设置 hasAcceptTokenNum_ = 1）
    auto numAcceptedTokenDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKEN_INDEX);
    if (numAcceptedTokenDesc != nullptr) {
        numAcceptedTokenDtype_ = numAcceptedTokenDesc->GetDataType();
        hasAcceptTokenNum_ = 1;
    } else {
        hasAcceptTokenNum_ = 0;
    }

    // 计算 x 数据类型的字节大小（用于后续内存计算）
    xDtypeSize_ = GetSizeByDataType(xDtype_);
    if (xDtypeSize_ == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype_).c_str(),
                                              "The dtype size of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // 激活函数模式（可选，默认 0=无激活）
    const int64_t *activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    if (activationModePtr != nullptr) {
        activationMode_ = *activationModePtr;
    }

    // padding slot id（可选，默认 -1 表示不使用 padding）
    const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    if (padSlotIdPtr != nullptr) {
        padSlotId_ = *padSlotIdPtr;
    }

    // 运行模式（可选，预留扩展）
    const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
    if (runModePtr != nullptr) {
        runMode_ = *runModePtr;
    }

    // 2D 输入时的最大序列长度（决定 UB 分配时 seqLen 取值）
    const int64_t *maxQueryLenPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_QUERY_LEN_INDEX);
    if (maxQueryLenPtr != nullptr) {
        maxQueryLen_ = *maxQueryLenPtr;
    }

    // 是否使用残差连接（0=否，1=是）
    const int64_t *residualConnectionPtr = attrs->GetAttrPointer<int64_t>(ATTR_RESIDUAL_CONNECTION_INDEX);
    if (residualConnectionPtr != nullptr) {
        residualConnection_ = *residualConnectionPtr;
    }

    // APC block 大小（APC 开启时必须非零）
    const int64_t *blockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX);
    if (blockSizePtr != nullptr) {
        blockSize_ = *blockSizePtr;
    }

    // APC 已初判开启时，block_size 必须非零，否则直接报错
    if (apcEnable_ && blockSize_ == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", "0",
                                              "The value of block_size must not be 0 when APC is enabled");
        return ge::GRAPH_FAILED;
    }

    // 卷积模式（0=Qwen3 默认，1=Pangu v2 前 width-1 token 置零）
    const int64_t *convModePtr = attrs->GetAttrPointer<int64_t>(ATTR_CONV_MODE_INDEX);
    if (convModePtr != nullptr) {
        convMode_ = *convModePtr;
    }

    const int64_t *maxDraftTokens = attrs->GetAttrPointer<int64_t>(ATTR_MAX_DRAFT_TOKENS_INDEX);
    if (maxDraftTokens != nullptr) {
        maxDraftTokens_ = *maxDraftTokens;
    }

    // 原地更新开关：由构造时传入的 isInplace_ 决定（拆分后不再从 attr 读取）
    inplace_ = isInplace_ ? 1 : 0;

    // convMode!=0 或 apcEnabled 时，必须提供 num_computed_tokens
    if (convMode_ != 0 || apcEnable_) {
        if (hasNumComputedTokens_ == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "num_computed_tokens", "nullptr",
                "num_computed_tokens cannot be nullptr when conv_mode != 0 or APC is enabled");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetStrideInfo()
{
    // 获取 x 的 stride（3D 输入取 DIM_1 stride；2D 输入取 DIM_0 stride）
    bool xIsView = context_->InputIsView(X_INDEX);
    if (xIsView) {
        auto *xStride = context_->GetInputStride(X_INDEX);
        OP_CHECK_IF(xStride == nullptr, OP_LOGE(context_->GetNodeName(), "x stride is invalid."),
                    return ge::GRAPH_FAILED);

        if (xInputMode_ == X_INPUT_3D) {
            if (xStride->GetDimNum() != DIM_3) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x stride",
                                                         std::to_string(xStride->GetDimNum()).c_str(),
                                                         "The shape dim of x stride must be 3");
                return ge::GRAPH_FAILED;
            }
            xStride_ = xStride->GetStride(DIM_1); // [batch, seq_len, dim] 中 seq_len 维的 stride
        } else {
            if (xStride->GetDimNum() != DIM_2) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x stride",
                                                         std::to_string(xStride->GetDimNum()).c_str(),
                                                         "The shape dim of x stride must be 2");
                return ge::GRAPH_FAILED;
            }
            xStride_ = xStride->GetStride(DIM_0); // [cu_seq_len, dim] 中 cu_seq_len 维的 stride
        }
    } else {
        // 连续张量：stride = dim_（每行 dim_ 个元素）
        xStride_ = dim_;
    }

    // 获取 conv_states 的 stride（形状为 [batch, state_len, dim]）
    bool cacheIsView = context_->InputIsView(CONV_STATES_INDEX);
    if (cacheIsView) {
        auto *cacheStride = context_->GetInputStride(CONV_STATES_INDEX);
        OP_CHECK_IF(cacheStride == nullptr, OP_LOGE(context_->GetNodeName(), "conv_states stride is invalid."),
                    return ge::GRAPH_FAILED);
        if (cacheStride->GetDimNum() != DIM_3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "conv_states stride",
                                                     std::to_string(cacheStride->GetDimNum()).c_str(),
                                                     "The shape dim of conv_states stride must be 3");
            return ge::GRAPH_FAILED;
        }
        cacheStride0_ = cacheStride->GetStride(DIM_0); // batch 维 stride
        cacheStride1_ = cacheStride->GetStride(DIM_1); // state_len 维 stride
    } else {
        // 连续张量默认 stride
        cacheStride0_ = dim_ * stateLen_; // batch 步长
        cacheStride1_ = dim_;             // state_len 步长
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateXShape()
{
    if (batchSize_ < MIN_BATCH || batchSize_ > MAX_BATCH) {
        std::string reasonMsg = "The value of batch_size must be within the range [" + std::to_string(MIN_BATCH) +
                                ", " + std::to_string(MAX_BATCH) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(batchSize_).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (xInputMode_ == X_INPUT_3D) {
        // seqLen = m+1，m 表示当前 decode 步数（BH 模板只支持短序列）
        int64_t m = seqLen_ - 1;
        if (m < MIN_M || m > MAX_M) {
            std::string reasonMsg = "The value of seq_len must satisfy m in the range [" + std::to_string(MIN_M) +
                                    ", " + std::to_string(MAX_M) + "], where m = seq_len - 1";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(seqLen_).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (xInputMode_ == X_INPUT_2D) {
        // 2D 模式下通过属性传入最大序列长度
        if (maxQueryLen_ < 1 || maxQueryLen_ > MAX_M + 1) {
            std::string reasonMsg = "The value of max_query_len must be greater than 0";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "max_query_len",
                                                  std::to_string(maxQueryLen_).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        // 2D 模式下 x 的第一维 cu_seq_len 不能超过 batch * max_query_len
        if (cuSeqLen_ > batchSize_ * maxQueryLen_) {
            std::string reasonMsg = "Shape [0] of x must be less than or equal to batch_size * max_query_len (" +
                                    std::to_string(batchSize_ * maxQueryLen_) + ")";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(cuSeqLen_).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // dim 必须在合法范围内，且为 16 的整数倍（不足 128 的余数元素由 dim 尾核承担）
    if (dim_ < MIN_DIM || dim_ > MAX_DIM) {
        std::string reasonMsg = "The value of dim must be within the range [" + std::to_string(MIN_DIM) + ", " +
                                std::to_string(MAX_DIM) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim_).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (dim_ % 16 != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim_).c_str(),
                                              "The value of dim must be a multiple of 16");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateWeightShape()
{
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();

    int64_t weightDim = weightOriginShape.GetDim(1);
    if (weightDim != dim_) {
        std::string reasonMsg = "Shape [1] of weight must be equal to shape [1] of x (" + std::to_string(dim_) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "weight", std::to_string(weightDim).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateConvStatesShape()
{
    auto convStatesShape = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShape);
    auto convStatesOriginShape = convStatesShape->GetOriginShape();
    // conv_states shape: [-1, K-1+m, dim]，state_len 需能容纳所有历史状态
    if (xInputMode_ == X_INPUT_3D) {
        int64_t expectedCacheLen = kernelSize_ + seqLen_ - 2; // = (K-1) + (seqLen-1)
        int64_t state_len = convStatesOriginShape.GetDim(DIM_1);
        if (state_len < expectedCacheLen) {
            std::string reasonMsg =
                "Shape [1] of conv_states must be greater than or equal to " + std::to_string(expectedCacheLen);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "conv_states",
                                                  std::to_string(state_len).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // dim 维度必须与 x 一致
    int64_t convStatesDim = convStatesOriginShape.GetDim(DIM_2);
    if (convStatesDim != dim_) {
        std::string reasonMsg =
            "Shape [2] of conv_states must be equal to shape [1] of x (" + std::to_string(dim_) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "conv_states",
                                              std::to_string(convStatesDim).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateCacheIndicesShape()
{
    auto indicesShape = context_->GetOptionalInputShape(CACHE_INDICES_INDEX);
    if (indicesShape == nullptr) {
        return ge::GRAPH_SUCCESS; // 可选输入，不存在时跳过
    }
    auto indicesOriginShape = indicesShape->GetOriginShape();

    if (apcEnable_) {
        // APC 模式：2D [batch, maxNumBlocks]
        if (indicesOriginShape.GetDimNum() != DIM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                     std::to_string(indicesOriginShape.GetDimNum()).c_str(),
                                                     "The shape dim of cache_indices must be 2 when APC is enabled");
            return ge::GRAPH_FAILED;
        }
        int64_t indicesBatch = indicesOriginShape.GetDim(DIM_0);
        if (indicesBatch != batchSize_) {
            std::string reasonMsg =
                "Shape [0] of cache_indices must be equal to batch_size (" + std::to_string(batchSize_) + ")";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                  std::to_string(indicesBatch).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        // 非 APC 模式：1D [batch]
        if (indicesOriginShape.GetDimNum() != DIM_1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                     std::to_string(indicesOriginShape.GetDimNum()).c_str(),
                                                     "The shape dim of cache_indices must be 1");
            return ge::GRAPH_FAILED;
        }
        int64_t indicesLen = indicesOriginShape.GetDim(DIM_0);
        if (indicesLen != batchSize_) {
            std::string reasonMsg =
                "Shape [0] of cache_indices must be equal to batch_size (" + std::to_string(batchSize_) + ")";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                                  std::to_string(indicesLen).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateNumAcceptedTokenShape()
{
    if (context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKEN_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS; // 可选输入，不存在时跳过
    }

    auto acceptShape = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, acceptShape);
    auto acceptOriginShape = acceptShape->GetOriginShape();
    if (acceptOriginShape.GetDimNum() != DIM_1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "num_accepted_token",
                                                 std::to_string(acceptOriginShape.GetDimNum()).c_str(),
                                                 "The shape dim of num_accepted_token must be 1");
        return ge::GRAPH_FAILED;
    }

    int64_t acceptLen = acceptOriginShape.GetDim(DIM_0);
    if (acceptLen != batchSize_) {
        std::string reasonMsg =
            "Shape [0] of num_accepted_token must be equal to batch_size (" + std::to_string(batchSize_) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "num_accepted_token",
                                              std::to_string(acceptLen).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateQueryStartLocShape()
{
    if (xInputMode_ == X_INPUT_3D) {
        return ge::GRAPH_SUCCESS; // 3D 输入不需要 query_start_loc
    }
    auto queryStartLocShape = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    if (queryStartLocShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query_start_loc", "nullptr",
                                              "query_start_loc cannot be nullptr when input x is 2D");
        return ge::GRAPH_FAILED;
    }

    auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();
    if (queryStartLocOriginShape.GetDimNum() != DIM_1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                                 std::to_string(queryStartLocOriginShape.GetDimNum()).c_str(),
                                                 "The shape dim of query_start_loc must be 1");
        return ge::GRAPH_FAILED;
    }

    // 形状应为 (batch + 1,)，多出的 1 用于表示结束偏移
    int64_t queryStartLocLen = queryStartLocOriginShape.GetDim(DIM_0);
    if (queryStartLocLen != batchSize_ + 1) {
        std::string reasonMsg = "Shape [0] of query_start_loc must be equal to " + std::to_string(batchSize_ + 1);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                              std::to_string(queryStartLocLen).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateXType()
{
    if (xDtype_ != ge::DataType::DT_FLOAT16 && xDtype_ != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype_).c_str(),
                                              "The dtype of x must be FLOAT16 or BFLOAT16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateWeightType()
{
    if (weightDtype_ != xDtype_) {
        std::string dtypeMsg = Ops::Base::ToString(xDtype_) + " and " + Ops::Base::ToString(weightDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and weight", dtypeMsg.c_str(),
                                               "The dtypes of x and weight must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateConvStatesType()
{
    if (convStatesDtype_ != xDtype_) {
        std::string dtypeMsg = Ops::Base::ToString(xDtype_) + " and " + Ops::Base::ToString(convStatesDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and conv_states", dtypeMsg.c_str(),
                                               "The dtypes of x and conv_states must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateCacheIndicesType()
{
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    if (cacheIndicesDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (cacheIndicesDtype_ != ge::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "cache_indices",
                                              Ops::Base::ToString(cacheIndicesDtype_).c_str(),
                                              "The dtype of cache_indices must be INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateQueryStartLocType()
{
    if (xInputMode_ == X_INPUT_3D) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryStartLocDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    if (queryStartLocDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query_start_loc", "nullptr",
                                              "query_start_loc cannot be nullptr when input x is 2D");
        return ge::GRAPH_FAILED;
    }
    if (queryStartLocDtype_ != ge::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "query_start_loc",
                                              Ops::Base::ToString(queryStartLocDtype_).c_str(),
                                              "The dtype of query_start_loc must be INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateNumAcceptedTokenType()
{
    auto numAcceptedTokenDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKEN_INDEX);
    if (numAcceptedTokenDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (numAcceptedTokenDtype_ != ge::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "num_accepted_token",
                                              Ops::Base::ToString(numAcceptedTokenDtype_).c_str(),
                                              "The dtype of num_accepted_token must be INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 属性合法性校验：activation_mode / pad_slot_id / run_mode / residual_connection /
//                 block_size / max_query_len / conv_mode / max_draft_tokens
ge::graphStatus FusedCausalConv1dCutBHTiling::ValidateAttrs()
{
    // activation_mode ∈ {0, 1, 2}
    if (activationMode_ != 0 && activationMode_ != 1 && activationMode_ != 2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "activation_mode",
                                              std::to_string(activationMode_).c_str(),
                                              "The value of activation_mode must be 0, 1 or 2");
        return ge::GRAPH_FAILED;
    }

    // pad_slot_id 不做范围校验（允许任意 int64 值，包括负数表示 padding sample）

    // run_mode ∈ {0, 1}
    if (runMode_ != 0 && runMode_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "run_mode", std::to_string(runMode_).c_str(),
                                              "The value of run_mode must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    // residual_connection ∈ {0, 1}
    if (residualConnection_ != 0 && residualConnection_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "residual_connection",
                                              std::to_string(residualConnection_).c_str(),
                                              "The value of residual_connection must be 0 or 1");
        return ge::GRAPH_FAILED;
    }

    // block_size ∈ {0} ∪ [2, INT64_MAX]（属性类型为 int64）
    if (blockSize_ != 0 && blockSize_ < 2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", std::to_string(blockSize_).c_str(),
                                              "The value of block_size must be 0 or greater than or equal to 2");
        return ge::GRAPH_FAILED;
    }

    // max_query_len：3D 输入允许 -1（表示未指定）；其它情况必须 > 0
    if (xInputMode_ == X_INPUT_3D) {
        if (maxQueryLen_ != -1 && maxQueryLen_ <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "max_query_len", std::to_string(maxQueryLen_).c_str(),
                "The value of max_query_len must be greater than 0 (only -1 is allowed for 3D x)");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (maxQueryLen_ <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "max_query_len",
                                                  std::to_string(maxQueryLen_).c_str(),
                                                  "The value of max_query_len must be greater than 0 when x is 2D");
            return ge::GRAPH_FAILED;
        }
    }

    // conv_mode ∈ {0, 1}
    if (convMode_ != 0 && convMode_ != 1) {
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

ge::graphStatus FusedCausalConv1dCutBHTiling::CheckInputParams()
{
    // 形状校验
    if (ValidateXShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateWeightShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateConvStatesShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateQueryStartLocShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateCacheIndicesShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateNumAcceptedTokenShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    // 类型校验
    if (ValidateXType() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateWeightType() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateConvStatesType() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateQueryStartLocType() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateCacheIndicesType() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (ValidateNumAcceptedTokenType() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    // 属性合法性校验
    if (ValidateAttrs() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;

    return ge::GRAPH_SUCCESS;
}

// DoOpTiling
// Tiling 主入口，根据 dim 大小选择细粒度（< 1024）或粗粒度（>= 1024）策略：
//   细粒度：dim 按 128 元素分块（DIM_ALIGN_ELEMENT）
//   粗粒度：dim 按 1024 元素分块（COARSE_GRAIN_ELEMENT）
ge::graphStatus FusedCausalConv1dCutBHTiling::DoOpTiling()
{
    if (dim_ >= COARSE_GRAIN_THRESHOLD) {
        // 粗粒度模式（dim >= 1024）：更大的切块粒度
        OP_CHECK_IF(ComputeInterCoreSplitCoarse() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplitCoarse failed"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ComputeIntraCoreUbTilingCoarse() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTilingCoarse failed"), return ge::GRAPH_FAILED);
    } else {
        // 细粒度模式（dim < 1024）：128 元素为切块单元，适合小 dim 场景充分利用所有核
        OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ComputeInterCoreSplit（细粒度核间切分）
ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeInterCoreSplit()
{
    // N = dim / 128，即 dim 方向可用的对齐块数；dimRemainder 为不足 128 的余数元素
    int64_t N = dim_ / DIM_ALIGN_ELEMENT;
    dimRemainderElems_ = dim_ % DIM_ALIGN_ELEMENT;
    if (dim_ <= 0) {
        OP_LOGE(context_->GetNodeName(), "dim %ld is invalid", dim_);
        return ge::GRAPH_FAILED;
    }
    // dim < 128 时（如 dim=64），N=0，dimRemainderElems_=dim_
    // 贪心搜索时用 searchN=1 代替 N=0，让 batch 方向能正常分核
    // 切分结果里 mainCoredimLen_ = base*128 = 0，最后一个核完全靠 dimRemainderElems_ 处理
    int64_t searchN = (N > 0) ? N : 1;

    // 根据数据量计算可用核数上限
    limitedCoreNum_ = CalculateLimitedCoreNum();
    int64_t maxCoresAvailable = std::min<int64_t>(totalCoreNum_, limitedCoreNum_);

    // 贪心搜索：从 searchN 到 1 遍历 dim 方向核数 dc，
    // 找使 dc × actualBS 最大的组合（优先总利用率高，相同时 dc 更大）
    int64_t bestDimCores = 1;
    int64_t bestBSCores = 1;
    int64_t bestUsed = 1;
    for (int64_t dc = searchN; dc >= 1; --dc) {
        int64_t maxAllowedBSByCore = maxCoresAvailable / dc;
        if (maxAllowedBSByCore == 0)
            continue;
        int64_t actualBS = std::min<int64_t>(batchSize_, maxAllowedBSByCore); // 确保每个bs核上有1哥bacth
        if (actualBS <= 0)
            continue;
        int64_t usedCores = dc * actualBS;
        if (usedCores > bestUsed || (usedCores == bestUsed && dc > bestDimCores)) {
            bestDimCores = dc;
            bestBSCores = actualBS;
            bestUsed = usedCores;
        }
        if (bestUsed == maxCoresAvailable) {
            break; // 已满核，提前退出
        }
    }

    // dim 方向多主核/多尾核切分：
    //   blockRemainder 个大核（每核 (base+1)*128 元素），其余为小核（每核 base*128 元素）
    //   最后一个 dim 核额外承担 dimRemainderElems_ 个余数元素（通过 lastCore* 参数传递）
    int64_t base = N / bestDimCores;
    int64_t remainder = N % bestDimCores;

    dimCoreCnt_ = bestDimCores;
    if (remainder == 0) {
        // 完全整除：所有核均等，无尾核
        dimMainCoreCnt_ = bestDimCores;
        dimTailCoreCnt_ = 0;
        mainCoredimLen_ = base * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = base * DIM_ALIGN_ELEMENT;
    } else {
        // 有块余数：remainder 个大核，其余为小核
        dimMainCoreCnt_ = remainder;
        dimTailCoreCnt_ = bestDimCores - remainder;
        mainCoredimLen_ = (base + 1) * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = base * DIM_ALIGN_ELEMENT;
    }

    // batch 方向多主核/多尾核切分：bsRemainder 个大核，其余为小核
    batchCoreCnt_ = bestBSCores;
    int64_t bsBase = batchSize_ / batchCoreCnt_;
    int64_t bsRemainder = batchSize_ % batchCoreCnt_;
    if (bsRemainder == 0) {
        batchMainCoreCnt_ = batchCoreCnt_;
        batchTailCoreCnt_ = 0;
        mainCoreBatchNum_ = bsBase;
        tailCoreBatchNum_ = bsBase;
    } else {
        batchMainCoreCnt_ = bsRemainder;
        batchTailCoreCnt_ = batchCoreCnt_ - bsRemainder;
        mainCoreBatchNum_ = bsBase + 1;
        tailCoreBatchNum_ = bsBase;
    }

    usedCoreNum_ = dimCoreCnt_ * batchCoreCnt_;

    return ge::GRAPH_SUCCESS;
}

// ComputeUbFor（细粒度核内 UB 切分核心计算）
// UB 占用估算（per dim element）：
//   weight + conv_states 部分：BUFFER_NUM * (kernelSize + (kernelSize-1)*2) * DTYPE_SIZE
//   x 部分（包含 x_in 和 y_out 双 buffer）：BUFFER_NUM * coreBS * seqLen * DTYPE_SIZE * 2
// 优先保持 batch 完整（outUbBS = coreBS），若 UB 不足则逐步减小 batch
void FusedCausalConv1dCutBHTiling::ComputeUbFor(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize,
                                                int64_t &outUbDim, int64_t &outUbBS, int64_t &outLoopDim,
                                                int64_t &outLoopBS, int64_t &outUbTailDim, int64_t &outUbTailBS)
{
    // 2D 输入模式：seqLen 不固定，用属性中的最大值保守估算 UB 需求
    if (xInputMode_ == X_INPUT_2D) {
        seqLen_ = maxQueryLen_;
    }

    // 每个 dim 元素对应的 weight/conv_states UB 开销（字节）
    // weight 单 buffer（1份），cacheQueueIn + cacheQueueOut 各 DB（2×2份）
    int64_t weightConvStatesCoeffPerDim = (kernelSize_ + BUFFER_NUM * stateLen_ * 2) * DTYPE_SIZE;
    // 每个 dim 元素对应的 x（含 y）UB 开销（单 buffer × batch × seqLen × 2 方向）
    int64_t xCoeffPerDimFullBS = coreBS * seqLen_ * DTYPE_SIZE * 2;
    int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBS;

    // 计算一次循环最多能放多少 dim 元素（对齐到 128 的倍数）
    int64_t maxUbDim = (totalCoeffPerDim > 0) ? (availableUbSize / totalCoeffPerDim) : 0;
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENT) * DIM_ALIGN_ELEMENT;

    if (maxUbDim >= DIM_ALIGN_ELEMENT) {
        // UB 足够放 >= 1 个 128 元素块且保持 batch 完整
        outUbBS = coreBS;
        outUbDim = std::min(maxUbDim, coreDimElems);
        outUbDim = (outUbDim / DIM_ALIGN_ELEMENT) * DIM_ALIGN_ELEMENT;
        if (outUbDim == 0)
            outUbDim = DIM_ALIGN_ELEMENT;
        // 尾核 coreDimElems 可能不足 128（dim % 128 != 0），outUbDim 不能超过实际 dim 大小
        outUbDim = std::min(outUbDim, coreDimElems);
    } else {
        // UB 不足：固定 dim=128（或 coreDimElems 若更小），尽量多放 batch
        outUbDim = std::min(DIM_ALIGN_ELEMENT, coreDimElems);
        int64_t weightConvStatesSize = weightConvStatesCoeffPerDim * outUbDim;
        int64_t availableForX = availableUbSize - weightConvStatesSize;
        int64_t xSizePerBatch = seqLen_ * outUbDim * DTYPE_SIZE * 2;
        outUbBS = (xSizePerBatch > 0) ? (availableForX / xSizePerBatch) : 0;
        outUbBS = std::max<int64_t>(outUbBS, 1);
        outUbBS = std::min(outUbBS, coreBS);
    }

    // 计算循环次数和尾块大小
    outLoopDim = (coreDimElems + outUbDim - 1) / outUbDim;
    // 若只有 1 次循环，尾块 = 主块大小；否则尾块 = 剩余元素
    outUbTailDim = (outLoopDim == 1) ? outUbDim : (coreDimElems - (outLoopDim - 1) * outUbDim);
    outLoopBS = (coreBS + outUbBS - 1) / outUbBS;
    outUbTailBS = (outLoopBS == 1) ? outUbBS : (coreBS - (outLoopBS - 1) * outUbBS);
}

// ComputeIntraCoreUbTiling（细粒度核内 UB 切分）
// 1. 计算固定 UB 占用（acceptTokenBuf / indicesBuf / queryStartLocBuf 等辅助 buffer）
// 2. 分别对大核（main core）和小核（tail core）调用 ComputeUbFor 计算 UB 循环参数
ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeIntraCoreUbTiling()
{
    // 固定 UB 占用：acceptTokenBuf（每核始终分配，存放 batch 级别的 accepted token 数）
    int64_t numAcceptedTokensUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    fixedUBSize = numAcceptedTokensUBSize;

    // indicesBuf：非 APC 时存 [batch] 个 cache 索引；APC 时存单个 batch 的 [maxNumBlocks] 个 block 索引
    int64_t cacheIndicesUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    if (apcEnable_) {
        cacheIndicesUBSize = (maxNumBlocks_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    }
    fixedUBSize += cacheIndicesUBSize;

    // queryStartLocBuf：2D 输入时需要存储 [batch+1] 个位置信息
    if (xInputMode_ == X_INPUT_2D) {
        int64_t queryStartLocUBSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += queryStartLocUBSize;
    }

    // numComputedTokensBuf：按 numComputedTokens 是否提供来分配
    if (hasNumComputedTokens_) {
        int64_t numComputedTokensUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += numComputedTokensUBSize;
    }

    // APC 开启时额外分配 3 个辅助 buffer：initialStateIdx / blockIdxFirst / blockIdxLast
    if (apcEnable_) {
        int64_t apcAuxBufSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += apcAuxBufSize * 3;
    }

    // 扣除系统预留和固定占用后的可用 UB
    int64_t availableUbSize = static_cast<int64_t>(ubSize_ - SYSTEM_RESERVED_UB_SIZE) - fixedUBSize;

    // 最后一个 dim 核的实际 dim 大小（基础块 + 余数）
    int64_t lastBaseDimLen = (dimTailCoreCnt_ > 0) ? tailCoredimLen_ : mainCoredimLen_;
    int64_t lastCoreDimLen = lastBaseDimLen + dimRemainderElems_;

    // dim < 128 时 mainCoredimLen_=0，唯一的核就是 last core，但 batch 方向仍需区分主核/尾核
    if (mainCoredimLen_ == 0) {
        // 1) batch 大核（main batch core）
        ComputeUbFor(lastCoreDimLen, mainCoreBatchNum_, availableUbSize, lastCoreubMainFactorDim_, ubMainFactorBS_,
                     lastCoreloopNumDim_, loopNumBS_, lastCoreubTailFactorDim_, ubTailFactorBS_);
        ubMainFactorDim_ = lastCoreubMainFactorDim_;
        ubTailFactorDim_ = lastCoreubTailFactorDim_;
        loopNumDim_ = lastCoreloopNumDim_;

        // 2) batch 小核（tail batch core）—— 如果有的话，单独计算
        if (batchTailCoreCnt_ > 0) {
            ComputeUbFor(lastCoreDimLen, tailCoreBatchNum_, availableUbSize, tailBlockubFactorDim_,
                         tailBlockubFactorBS_, tailBlockloopNumDim_, tailBlockloopNumBS_, tailBlockubTailFactorDim_,
                         tailBlockubTailFactorBS_);
        } else {
            // 无 batch 尾核，复制主核参数
            tailBlockubFactorDim_ = ubMainFactorDim_;
            tailBlockubTailFactorDim_ = ubTailFactorDim_;
            tailBlockloopNumDim_ = loopNumDim_;
            tailBlockubFactorBS_ = ubMainFactorBS_;
            tailBlockubTailFactorBS_ = ubTailFactorBS_;
            tailBlockloopNumBS_ = loopNumBS_;
        }
        return ge::GRAPH_SUCCESS;
    }

    // 主核 UB 参数
    ComputeUbFor(mainCoredimLen_, mainCoreBatchNum_, availableUbSize, ubMainFactorDim_, ubMainFactorBS_, loopNumDim_,
                 loopNumBS_, ubTailFactorDim_, ubTailFactorBS_);

    // 尾核 UB 参数（存在尾核时才单独计算；否则复制主核参数保持一致）
    if (dimTailCoreCnt_ > 0 || batchTailCoreCnt_ > 0) {
        // 多主核/多尾核策略：
        //   dim 尾核的 dim 大小取 tailCoredimLen_（若无 dim 尾核则与主核相同）
        //   batch 尾核的 batch 数取 tailCoreBatchNum_（若无 batch 尾核则与主核相同）
        int64_t tailCoreDim = (dimTailCoreCnt_ > 0) ? tailCoredimLen_ : mainCoredimLen_;
        int64_t tailCoreBS = (batchTailCoreCnt_ > 0) ? tailCoreBatchNum_ : mainCoreBatchNum_;
        ComputeUbFor(tailCoreDim, tailCoreBS, availableUbSize, tailBlockubFactorDim_, tailBlockubFactorBS_,
                     tailBlockloopNumDim_, tailBlockloopNumBS_, tailBlockubTailFactorDim_, tailBlockubTailFactorBS_);
    } else {
        // 无尾核：直接复制主核参数，确保 kernel 逻辑无需区分主/尾核分支
        tailBlockubFactorDim_ = ubMainFactorDim_;
        tailBlockubFactorBS_ = ubMainFactorBS_;
        tailBlockloopNumDim_ = loopNumDim_;
        tailBlockloopNumBS_ = loopNumBS_;
        tailBlockubTailFactorDim_ = ubTailFactorDim_;
        tailBlockubTailFactorBS_ = ubTailFactorBS_;
    }

    // 最后一个 dim 核 UB 参数
    // dimRemainderElems_>0：lastCoreDimLen = 基础块 + 余数，需单独计算
    // dimRemainderElems_=0：最后一个 dim 核与普通尾核参数相同
    if (dimRemainderElems_ > 0) {
        // 用临时变量接收 batch 输出，避免覆盖 tailBlock* 参数
        int64_t tmpUbBS = 0;
        int64_t tmpLoopBS = 0;
        int64_t tmpUbTailBS = 0;
        ComputeUbFor(lastCoreDimLen, mainCoreBatchNum_, availableUbSize, lastCoreubMainFactorDim_, tmpUbBS,
                     lastCoreloopNumDim_, tmpLoopBS, lastCoreubTailFactorDim_, tmpUbTailBS);
    } else {
        lastCoreloopNumDim_ = tailBlockloopNumDim_;
        lastCoreubMainFactorDim_ = tailBlockubFactorDim_;
        lastCoreubTailFactorDim_ = tailBlockubTailFactorDim_;
    }

    return ge::GRAPH_SUCCESS;
}

// CalculateLimitedCoreNum（细粒度）
// 根据输入数据量估算合理的最大核数：
//   每 256 字节（= 128 元素 × 2 字节）分配 1 核，最多不超过 totalCoreNum_
int64_t FusedCausalConv1dCutBHTiling::CalculateLimitedCoreNum()
{
    // 只考虑 batch × dim 的数据量（不含 seqLen，因为 seqLen 较小且固定）
    int64_t xSizeBytes = batchSize_ * dim_ * DTYPE_SIZE;
    if (dim_ < 128) {
        xSizeBytes = batchSize_ * DIM_ALIGN_ELEMENT * DTYPE_SIZE;
    }

    // ceil(xSizeBytes / 256)，每 256 字节（128 元素）分配 1 核
    int64_t effectiveCoreNum = (xSizeBytes + DIM_ALIGN_SIZE - 1) / DIM_ALIGN_SIZE;
    effectiveCoreNum = std::max(effectiveCoreNum, static_cast<int64_t>(1));

    return std::min(effectiveCoreNum, static_cast<int64_t>(totalCoreNum_));
}

// CalculateLimitedCoreNumCoarse（粗粒度）
// 粗粒度模式下按 2048 字节（= 1024 元素）粒度估算合理核数
int64_t FusedCausalConv1dCutBHTiling::CalculateLimitedCoreNumCoarse()
{
    int64_t xSizeBytes = batchSize_ * dim_ * DTYPE_SIZE;

    // ceil(xSizeBytes / 2048)，每 2048 字节分配 1 核
    int64_t effectiveCoreNum = (xSizeBytes + COARSE_GRAIN_SIZE - 1) / COARSE_GRAIN_SIZE;
    effectiveCoreNum = std::max(effectiveCoreNum, static_cast<int64_t>(1));

    return std::min(effectiveCoreNum, static_cast<int64_t>(totalCoreNum_));
}

// GetTilingKey
// 根据 x 数据类型返回对应的 TilingKey，框架据此选择 kernel 分支
uint64_t FusedCausalConv1dCutBHTiling::GetTilingKey() const
{
    if (xDtype_ == ge::DataType::DT_BF16) {
        return TILING_KEY_BH_BF16;
    } else if (xDtype_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_BH_FP16;
    }
}

// PostTiling
// 将所有 tiling 计算结果写入 tilingData_，并通过 memcpy 复制到 RawTilingData
// 同时调用 SetBlockDim 告知框架实际使用的核数
ge::graphStatus FusedCausalConv1dCutBHTiling::PostTiling()
{
    // 设置实际使用的核数（由框架传递给 runtime 调度器）
    context_->SetBlockDim(usedCoreNum_);

    // ---- 核分布参数 ----
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.dimCoreCnt = dimCoreCnt_;
    tilingData_.batchCoreCnt = batchCoreCnt_;

    // ---- dim 方向切分参数 ----
    tilingData_.dimMainCoreCnt = dimMainCoreCnt_;
    tilingData_.dimTailCoreCnt = dimTailCoreCnt_;
    tilingData_.mainCoredimLen = mainCoredimLen_;
    tilingData_.tailCoredimLen = tailCoredimLen_;

    // ---- batch 方向切分参数 ----
    tilingData_.batchMainCoreCnt = batchMainCoreCnt_;
    tilingData_.batchTailCoreCnt = batchTailCoreCnt_;
    tilingData_.mainCoreBatchNum = mainCoreBatchNum_;
    tilingData_.tailCoreBatchNum = tailCoreBatchNum_;

    // ---- 大核（main core）UB 循环参数 ----
    tilingData_.loopNumBS = loopNumBS_;
    tilingData_.loopNumDim = loopNumDim_;
    tilingData_.ubMainFactorBS = ubMainFactorBS_;
    tilingData_.ubTailFactorBS = ubTailFactorBS_;
    tilingData_.ubMainFactorDim = ubMainFactorDim_;
    tilingData_.ubTailFactorDim = ubTailFactorDim_;

    // ---- 小核（tail core）UB 循环参数 ----
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;

    // ---- 形状信息（kernel 运行时使用）----
    tilingData_.batchSize = batchSize_;
    tilingData_.seqLen = seqLen_;
    tilingData_.cuSeqLen = cuSeqLen_;
    tilingData_.dim = dim_;
    tilingData_.kernelSize = kernelSize_;
    tilingData_.stateLen = stateLen_;
    tilingData_.xStride = xStride_;
    tilingData_.cacheStride0 = cacheStride0_;
    tilingData_.cacheStride1 = cacheStride1_;
    tilingData_.padSlotId = padSlotId_;
    tilingData_.xInputMode = xInputMode_;
    tilingData_.hasAcceptTokenNum = hasAcceptTokenNum_;
    tilingData_.residualConnection = residualConnection_;

    tilingData_.apcEnable = apcEnable_;
    tilingData_.blockSize = blockSize_;
    tilingData_.maxNumBlocks = maxNumBlocks_;
    tilingData_.hasCacheIndices = hasCacheIndices_;
    tilingData_.inplace = inplace_;
    tilingData_.convMode = convMode_;
    tilingData_.hasNumComputedTokens = hasNumComputedTokens_;
    tilingData_.maxQueryLen = maxQueryLen_;

    // ---- dimRemainder（last dim core）相关字段 ----
    tilingData_.dimRemainderElems = dimRemainderElems_;
    tilingData_.lastCoreloopNumDim = lastCoreloopNumDim_;
    tilingData_.lastCoreubMainFactorDim = lastCoreubMainFactorDim_;
    tilingData_.lastCoreubTailFactorDim = lastCoreubTailFactorDim_;

    // 将 tilingData_ 序列化到 RawTilingData（供 kernel 侧反序列化）
    auto tilingDataSize = sizeof(FusedCausalConv1dCutBHTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

// DumpTilingInfo
// 打印所有 tiling 参数（INFO 级别日志），用于调试和问题定位
void FusedCausalConv1dCutBHTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== FusedCausalConv1dCutBH DumpTilingInfo ===");

    // 核分布参数
    OP_LOGI(context_->GetNodeName(), "usedCoreNum: %ld", tilingData_.usedCoreNum);
    OP_LOGI(context_->GetNodeName(), "dimCoreCnt: %ld", tilingData_.dimCoreCnt);
    OP_LOGI(context_->GetNodeName(), "batchCoreCnt: %ld", tilingData_.batchCoreCnt);

    // dim 方向切分参数
    OP_LOGI(context_->GetNodeName(), "dimMainCoreCnt: %ld", tilingData_.dimMainCoreCnt);
    OP_LOGI(context_->GetNodeName(), "dimTailCoreCnt: %ld", tilingData_.dimTailCoreCnt);
    OP_LOGI(context_->GetNodeName(), "mainCoredimLen: %ld", tilingData_.mainCoredimLen);
    OP_LOGI(context_->GetNodeName(), "tailCoredimLen: %ld", tilingData_.tailCoredimLen);

    // batch 方向切分参数
    OP_LOGI(context_->GetNodeName(), "batchMainCoreCnt: %ld", tilingData_.batchMainCoreCnt);
    OP_LOGI(context_->GetNodeName(), "batchTailCoreCnt: %ld", tilingData_.batchTailCoreCnt);
    OP_LOGI(context_->GetNodeName(), "mainCoreBatchNum: %ld", tilingData_.mainCoreBatchNum);
    OP_LOGI(context_->GetNodeName(), "tailCoreBatchNum: %ld", tilingData_.tailCoreBatchNum);

    // 大核 UB 循环参数
    OP_LOGI(context_->GetNodeName(), "loopNumBS: %ld", tilingData_.loopNumBS);
    OP_LOGI(context_->GetNodeName(), "loopNumDim: %ld", tilingData_.loopNumDim);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorBS: %ld", tilingData_.ubMainFactorBS);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorBS: %ld", tilingData_.ubTailFactorBS);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorDim: %ld", tilingData_.ubMainFactorDim);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorDim: %ld", tilingData_.ubTailFactorDim);

    // 小核 UB 循环参数
    OP_LOGI(context_->GetNodeName(), "tailBlockloopNumBS: %ld", tilingData_.tailBlockloopNumBS);
    OP_LOGI(context_->GetNodeName(), "tailBlockloopNumDim: %ld", tilingData_.tailBlockloopNumDim);
    OP_LOGI(context_->GetNodeName(), "tailBlockubFactorBS: %ld", tilingData_.tailBlockubFactorBS);
    OP_LOGI(context_->GetNodeName(), "tailBlockubTailFactorBS: %ld", tilingData_.tailBlockubTailFactorBS);
    OP_LOGI(context_->GetNodeName(), "tailBlockubFactorDim: %ld", tilingData_.tailBlockubFactorDim);
    OP_LOGI(context_->GetNodeName(), "tailBlockubTailFactorDim: %ld", tilingData_.tailBlockubTailFactorDim);

    // 形状信息
    OP_LOGI(context_->GetNodeName(), "batchSize: %ld", tilingData_.batchSize);
    OP_LOGI(context_->GetNodeName(), "seqLen: %ld", tilingData_.seqLen);
    OP_LOGI(context_->GetNodeName(), "cuSeqLen: %ld", tilingData_.cuSeqLen);
    OP_LOGI(context_->GetNodeName(), "dim: %ld", tilingData_.dim);
    OP_LOGI(context_->GetNodeName(), "kernelSize: %ld", tilingData_.kernelSize);
    OP_LOGI(context_->GetNodeName(), "stateLen: %ld", tilingData_.stateLen);
    OP_LOGI(context_->GetNodeName(), "xStride: %ld", tilingData_.xStride);
    OP_LOGI(context_->GetNodeName(), "cacheStride0: %ld", tilingData_.cacheStride0);
    OP_LOGI(context_->GetNodeName(), "cacheStride1: %ld", tilingData_.cacheStride1);
    OP_LOGI(context_->GetNodeName(), "padSlotId: %ld", tilingData_.padSlotId);
    OP_LOGI(context_->GetNodeName(), "xInputMode: %ld", tilingData_.xInputMode);
    OP_LOGI(context_->GetNodeName(), "hasAcceptTokenNum: %ld", tilingData_.hasAcceptTokenNum);
    OP_LOGI(context_->GetNodeName(), "residualConnection: %ld", tilingData_.residualConnection);
    OP_LOGI(context_->GetNodeName(), "apcEnable: %ld", tilingData_.apcEnable);
    OP_LOGI(context_->GetNodeName(), "blockSize: %ld", tilingData_.blockSize);
    OP_LOGI(context_->GetNodeName(), "maxNumBlocks: %ld", tilingData_.maxNumBlocks);
    OP_LOGI(context_->GetNodeName(), "hasCacheIndices: %ld", tilingData_.hasCacheIndices);
    OP_LOGI(context_->GetNodeName(), "inplace: %ld", tilingData_.inplace);
    OP_LOGI(context_->GetNodeName(), "convMode: %ld", tilingData_.convMode);
    OP_LOGI(context_->GetNodeName(), "hasNumComputedTokens: %ld", tilingData_.hasNumComputedTokens);
    OP_LOGI(context_->GetNodeName(), "maxQueryLen: %ld", maxQueryLen_);
    // dimRemainder（last dim core）相关的字段
    OP_LOGI(context_->GetNodeName(), "dimRemainderElems: %ld", tilingData_.dimRemainderElems);
    OP_LOGI(context_->GetNodeName(), "lastCoreloopNumDim: %ld", tilingData_.lastCoreloopNumDim);
    OP_LOGI(context_->GetNodeName(), "lastCoreubMainFactorDim: %ld", tilingData_.lastCoreubMainFactorDim);
    OP_LOGI(context_->GetNodeName(), "lastCoreubTailFactorDim: %ld", tilingData_.lastCoreubTailFactorDim);
}

ge::graphStatus FusedCausalConv1dCutBHTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCausalConv1dCutBHTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

// ComputeInterCoreSplitCoarse（粗粒度核间切分，dim >= 1024）
ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeInterCoreSplitCoarse()
{
    // 计算粗粒度块数和余数细块数
    int64_t coarseBlocks = dim_ / COARSE_GRAIN_ELEMENT; // 1024 元素块数
    int64_t remainder = dim_ % COARSE_GRAIN_ELEMENT;    // 不足 1024 的余数元素
    int64_t fineBlocks = remainder / DIM_ALIGN_ELEMENT; // 将余数划分为 128 元素细块
    dimRemainderElems_ = remainder % DIM_ALIGN_ELEMENT; // 不足 128 的余数元素

    limitedCoreNum_ = CalculateLimitedCoreNumCoarse();
    int64_t maxCoresAvailable = std::min<int64_t>(totalCoreNum_, limitedCoreNum_);

    // dim 方向：最多使用 coarseBlocks 个核（一核一粗块），不超过可用核上限
    dimCoreCnt_ = std::min(coarseBlocks, maxCoresAvailable);
    if (dimCoreCnt_ <= 0) {
        OP_LOGE(context_->GetNodeName(), "dim %ld is smaller than COARSE_GRAIN_ELEMENT %ld", dim_,
                COARSE_GRAIN_ELEMENT);
        return ge::GRAPH_FAILED;
    }

    // 将余数细块平均分给各 dim 核（多主核/多尾核）
    int64_t fineBase = fineBlocks / dimCoreCnt_;
    int64_t fineRemainder = fineBlocks % dimCoreCnt_;

    if (fineRemainder == 0) {
        // 完全整除：所有 dim 核负责相同大小，无尾核
        dimMainCoreCnt_ = dimCoreCnt_;
        dimTailCoreCnt_ = 0;
        mainCoredimLen_ = COARSE_GRAIN_ELEMENT + fineBase * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = mainCoredimLen_;
    } else {
        // 有细块余数：fineRemainder 个大核，其余为小核
        dimMainCoreCnt_ = fineRemainder;
        dimTailCoreCnt_ = dimCoreCnt_ - fineRemainder;
        mainCoredimLen_ = COARSE_GRAIN_ELEMENT + (fineBase + 1) * DIM_ALIGN_ELEMENT;
        tailCoredimLen_ = COARSE_GRAIN_ELEMENT + fineBase * DIM_ALIGN_ELEMENT;
    }

    // batch 方向：用剩余核均分 batch（至少 1 个核）
    int64_t remainingCores = maxCoresAvailable / dimCoreCnt_;
    batchCoreCnt_ = std::min(batchSize_, remainingCores);
    batchCoreCnt_ = std::max(batchCoreCnt_, static_cast<int64_t>(1));

    // batch 多主核/多尾核切分（与细粒度相同逻辑）
    int64_t bsBase = batchSize_ / batchCoreCnt_;
    int64_t bsRemainder = batchSize_ % batchCoreCnt_;
    if (bsRemainder == 0) {
        batchMainCoreCnt_ = batchCoreCnt_;
        batchTailCoreCnt_ = 0;
        mainCoreBatchNum_ = bsBase;
        tailCoreBatchNum_ = bsBase;
    } else {
        batchMainCoreCnt_ = bsRemainder;
        batchTailCoreCnt_ = batchCoreCnt_ - bsRemainder;
        mainCoreBatchNum_ = bsBase + 1;
        tailCoreBatchNum_ = bsBase;
    }

    usedCoreNum_ = dimCoreCnt_ * batchCoreCnt_;

    return ge::GRAPH_SUCCESS;
}

// ComputeUbForCoarse（粗粒度 UB 切分核心计算）
void FusedCausalConv1dCutBHTiling::ComputeUbForCoarse(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize,
                                                      int64_t &outUbDim, int64_t &outUbBS, int64_t &outLoopDim,
                                                      int64_t &outLoopBS, int64_t &outUbTailDim, int64_t &outUbTailBS)
{
    // 2D 输入模式：用最大序列长度估算 UB
    if (xInputMode_ == X_INPUT_2D) {
        seqLen_ = maxQueryLen_;
    }

    // UB 占用系数（weight 单 buffer，cacheQueue DB，x/y 单 buffer）
    int64_t weightConvStatesCoeffPerDim = (kernelSize_ + BUFFER_NUM * stateLen_ * 2) * DTYPE_SIZE;
    int64_t xCoeffPerDimFullBS = coreBS * seqLen_ * DTYPE_SIZE * 2;
    int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBS;

    // 粗粒度模式：dim 对齐到 1024 元素
    int64_t maxUbDim = (totalCoeffPerDim > 0) ? (availableUbSize / totalCoeffPerDim) : 0;
    maxUbDim = (maxUbDim / COARSE_GRAIN_ELEMENT) * COARSE_GRAIN_ELEMENT;

    if (maxUbDim >= COARSE_GRAIN_ELEMENT) {
        // UB 足够：保持 batch 完整，尽量多放 dim
        outUbBS = coreBS;
        outUbDim = std::min(maxUbDim, coreDimElems);
        outUbDim = (outUbDim / COARSE_GRAIN_ELEMENT) * COARSE_GRAIN_ELEMENT;
        if (outUbDim == 0)
            outUbDim = COARSE_GRAIN_ELEMENT;
        // 尾核 coreDimElems 可能含 dimRemainder（不足 1024），outUbDim 不能超过实际 dim 大小
        outUbDim = std::min(outUbDim, coreDimElems);
    } else {
        // UB 不足：固定 dim=1024（或 coreDimElems 若更小），减小 batch
        outUbDim = std::min(COARSE_GRAIN_ELEMENT, coreDimElems);
        int64_t weightConvStatesSize = weightConvStatesCoeffPerDim * outUbDim;
        int64_t availableForX = availableUbSize - weightConvStatesSize;
        int64_t xSizePerBatch = seqLen_ * outUbDim * DTYPE_SIZE * 2;
        outUbBS = (xSizePerBatch > 0) ? (availableForX / xSizePerBatch) : 0;
        outUbBS = std::max<int64_t>(outUbBS, 1);
        outUbBS = std::min(outUbBS, coreBS);
    }

    // 循环次数和尾块大小
    outLoopDim = (coreDimElems + outUbDim - 1) / outUbDim;
    outUbTailDim = (outLoopDim == 1) ? outUbDim : (coreDimElems - (outLoopDim - 1) * outUbDim);
    outLoopBS = (coreBS + outUbBS - 1) / outUbBS;
    outUbTailBS = (outLoopBS == 1) ? outUbBS : (coreBS - (outLoopBS - 1) * outUbBS);
}

ge::graphStatus FusedCausalConv1dCutBHTiling::ComputeIntraCoreUbTilingCoarse()
{
    // 固定 UB 占用（与细粒度版本相同）
    int64_t numAcceptedTokensUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    fixedUBSize = numAcceptedTokensUBSize;

    int64_t cacheIndicesUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    if (apcEnable_) {
        cacheIndicesUBSize = (maxNumBlocks_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    }
    fixedUBSize += cacheIndicesUBSize;

    if (xInputMode_ == X_INPUT_2D) {
        int64_t queryStartLocUBSize =
            ((batchSize_ + 1) * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += queryStartLocUBSize;
    }

    if (hasNumComputedTokens_) {
        int64_t numComputedTokensUBSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += numComputedTokensUBSize;
    }

    if (apcEnable_) {
        int64_t apcAuxBufSize = (batchSize_ * sizeof(int32_t) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        fixedUBSize += apcAuxBufSize * 3;
    }

    int64_t availableUbSize = static_cast<int64_t>(ubSize_ - SYSTEM_RESERVED_UB_SIZE) - fixedUBSize;

    // 大核 UB 参数（粗粒度版本）
    ComputeUbForCoarse(mainCoredimLen_, mainCoreBatchNum_, availableUbSize, ubMainFactorDim_, ubMainFactorBS_,
                       loopNumDim_, loopNumBS_, ubTailFactorDim_, ubTailFactorBS_);

    // 尾核 UB 参数（粗粒度版本）
    if (dimTailCoreCnt_ > 0 || batchTailCoreCnt_ > 0) {
        // 多主核/多尾核策略：
        //   dim 尾核的 dim 大小取 tailCoredimLen_（若无 dim 尾核则与主核相同）
        //   batch 尾核的 batch 数取 tailCoreBatchNum_（若无 batch 尾核则与主核相同）
        int64_t tailCoreDim = (dimTailCoreCnt_ > 0) ? tailCoredimLen_ : mainCoredimLen_;
        int64_t tailCoreBS = (batchTailCoreCnt_ > 0) ? tailCoreBatchNum_ : mainCoreBatchNum_;
        ComputeUbForCoarse(tailCoreDim, tailCoreBS, availableUbSize, tailBlockubFactorDim_, tailBlockubFactorBS_,
                           tailBlockloopNumDim_, tailBlockloopNumBS_, tailBlockubTailFactorDim_,
                           tailBlockubTailFactorBS_);
    } else {
        // 无尾核：复制主核参数
        tailBlockubFactorDim_ = ubMainFactorDim_;
        tailBlockubFactorBS_ = ubMainFactorBS_;
        tailBlockloopNumDim_ = loopNumDim_;
        tailBlockloopNumBS_ = loopNumBS_;
        tailBlockubTailFactorDim_ = ubTailFactorDim_;
        tailBlockubTailFactorBS_ = ubTailFactorBS_;
    }

    // 最后一个 dim 核 UB 参数（含 dimRemainderElems_ 余数元素，粗粒度版本）
    if (dimRemainderElems_ > 0) {
        int64_t lastBaseDimLen = (dimTailCoreCnt_ > 0) ? tailCoredimLen_ : mainCoredimLen_;
        int64_t lastCoreDimLen = lastBaseDimLen + dimRemainderElems_;
        // 用临时变量接收 batch 输出，避免覆盖 tailBlock* 参数
        int64_t tmpUbBS = 0;
        int64_t tmpLoopBS = 0;
        int64_t tmpUbTailBS = 0;
        ComputeUbForCoarse(lastCoreDimLen, mainCoreBatchNum_, availableUbSize, lastCoreubMainFactorDim_, tmpUbBS,
                           lastCoreloopNumDim_, tmpLoopBS, lastCoreubTailFactorDim_, tmpUbTailBS);
    } else {
        lastCoreloopNumDim_ = tailBlockloopNumDim_;
        lastCoreubMainFactorDim_ = tailBlockubFactorDim_;
        lastCoreubTailFactorDim_ = tailBlockubTailFactorDim_;
    }

    return ge::GRAPH_SUCCESS;
}

// 向框架注册 BH 模板，优先级为 1
REGISTER_OPS_TILING_TEMPLATE(FusedCausalConv1dCutBH, FusedCausalConv1dCutBHTiling, 1);

} // namespace optiling
