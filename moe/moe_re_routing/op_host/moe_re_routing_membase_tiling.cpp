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
 * \file moe_re_routing_membase_tiling.cpp
 * \brief
 */
#include "moe_re_routing_membase_tiling.h"
#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "log/log.h"

namespace optiling {

constexpr int64_t TOKENS_INDEX = 0;
constexpr int64_t EXPERT_TOKEN_NUM_PER_RANK_INDEX = 1;
constexpr int64_t PER_TOKEN_SCALES_INDEX = 2;
constexpr int64_t PERMUTE_TOKENS_INDEX = 0;
constexpr int64_t PERMUTE_PER_TOKEN_SCALES_INDEX = 1;
constexpr int64_t PERMUTE_TOKEN_IDX_INDEX = 2;
constexpr int64_t EXPERT_TOKEN_NUM_INDEX = 3;
constexpr int64_t EXPERT_TOKEN_NUM_TYPE_INDEX = 0;
constexpr int64_t EXPERT_TOKEN_NUM_TYPE = 1;
constexpr int64_t IDX_TYPE_INDEX = 1;
constexpr int64_t IDX_TYPE = 0;
constexpr int64_t SHAPE_IDX_A = 0;
constexpr int64_t SHAPE_IDX_H = 1;
constexpr int64_t SHAPE_IDX_N = 0;
constexpr int64_t SHAPE_IDX_E = 1;
constexpr int64_t DIM_SIZE = 2;
constexpr int64_t HAS_SCALE = 1;
constexpr int64_t HAS_NO_SCALE = 0;
constexpr int64_t RANK_NUM_UPPER_BOUND = 288;
constexpr int64_t TOKENS_LENGTH_UPPER_BOUND = 16384;
constexpr int64_t RESERVE_UB_SIZE = 12288;
constexpr int64_t TILING_KEY_TOKENS_INT8_EXPERTS_INT32_SCALES_FLOAT32 = 100000;
constexpr int64_t TILING_KEY_TOKENS_INT8_EXPERTS_INT64_SCALES_FLOAT32 = 100010;
constexpr int64_t TILING_KEY_TOKENS_FP16_EXPERTS_INT32_SCALES_FLOAT32 = 100100;
constexpr int64_t TILING_KEY_TOKENS_FP16_EXPERTS_INT64_SCALES_FLOAT32 = 100110;
constexpr int64_t TILING_KEY_TOKENS_BF16_EXPERTS_INT32_SCALES_FLOAT32 = 100200;
constexpr int64_t TILING_KEY_TOKENS_BF16_EXPERTS_INT64_SCALES_FLOAT32 = 100210;

inline int64_t CeilDiv(int64_t N, int64_t n) { return ((((N) + (n)-1) / (n))); }

static const std::map<std::pair<ge::DataType, ge::DataType>, int64_t> TILING_KEY_MAP = {
    {{ge::DT_INT8, ge::DT_INT32}, TILING_KEY_TOKENS_INT8_EXPERTS_INT32_SCALES_FLOAT32},
    {{ge::DT_INT8, ge::DT_INT64}, TILING_KEY_TOKENS_INT8_EXPERTS_INT64_SCALES_FLOAT32},
    {{ge::DT_FLOAT16, ge::DT_INT32}, TILING_KEY_TOKENS_FP16_EXPERTS_INT32_SCALES_FLOAT32},
    {{ge::DT_FLOAT16, ge::DT_INT64}, TILING_KEY_TOKENS_FP16_EXPERTS_INT64_SCALES_FLOAT32},
    {{ge::DT_BF16, ge::DT_INT32}, TILING_KEY_TOKENS_BF16_EXPERTS_INT32_SCALES_FLOAT32},
    {{ge::DT_BF16, ge::DT_INT64}, TILING_KEY_TOKENS_BF16_EXPERTS_INT64_SCALES_FLOAT32},
};

static std::tuple<int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context, const int64_t index = 0)
{
    const gert::StorageShape *shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."),
                return std::make_tuple(0, 0));
    // check shape length is DIM_SIZE
    OP_CHECK_IF(shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE,
                OP_LOGE(context->GetNodeName(), "Shape must be (A, H)."), return std::make_tuple(0, 0));
    return std::make_tuple(shapePtr->GetStorageShape().GetDim(SHAPE_IDX_A),
                           shapePtr->GetStorageShape().GetDim(SHAPE_IDX_H));
}

ge::graphStatus MoeReRoutingTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MoeReRoutingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        int64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ((uint64_t &)ubSize));
        ubSize_ = ubSize;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."),
                return ge::GRAPH_FAILED);
    auto tokenShapeTuple = GetShapeTuple(context_, TOKENS_INDEX);
    auto expertShapeTuple = GetShapeTuple(context_, EXPERT_TOKEN_NUM_PER_RANK_INDEX);
    int64_t tokenNum = std::get<SHAPE_IDX_A>(tokenShapeTuple);
    int64_t tokenSize = std::get<SHAPE_IDX_H>(tokenShapeTuple);
    int64_t rankNums = std::get<SHAPE_IDX_N>(expertShapeTuple);
    auto tokenDtype = context_->GetInputDesc(TOKENS_INDEX)->GetDataType();
    commonParams_.tokenDtype = tokenDtype;
    auto expertDataType = context_->GetInputDesc(EXPERT_TOKEN_NUM_PER_RANK_INDEX)->GetDataType();
    commonParams_.expertDtype = expertDataType;
    auto scalePtr = context_->GetOptionalInputDesc(PER_TOKEN_SCALES_INDEX);
    if (scalePtr != nullptr) {
        auto scaleDtype = scalePtr->GetDataType();
        tilingData_.set_hasScale(HAS_SCALE);
        commonParams_.scaleDtype = scaleDtype;
    } else {
        tilingData_.set_hasScale(HAS_NO_SCALE);
        commonParams_.scaleDtype = ge::DT_FLOAT;
    }

    OP_CHECK_IF(tokenSize > TOKENS_LENGTH_UPPER_BOUND,
                OP_LOGE(context_->GetNodeName(), "tokenSize[%ld] should <= 16384,", tokenSize),
                return ge::GRAPH_FAILED);
    tilingData_.set_tokensNum(tokenNum);
    tilingData_.set_tokensSize(tokenSize);
    tilingData_.set_rankNum(rankNums);
    tilingData_.set_expertNumPerRank(std::get<SHAPE_IDX_E>(expertShapeTuple));

    // attr info
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto expertTokenNumType = attrs->GetInt(EXPERT_TOKEN_NUM_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokenNumType);
    OP_CHECK_IF(*expertTokenNumType != EXPERT_TOKEN_NUM_TYPE,
                OP_LOGE(context_->GetNodeName(), "expertTokenNumType[%ld] should == 1.", *expertTokenNumType),
                return ge::GRAPH_FAILED);

    auto idxType = attrs->GetInt(IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxType);
    OP_CHECK_IF(*idxType != IDX_TYPE, OP_LOGE(context_->GetNodeName(), "idxType[%ld] should == 0.", *idxType),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool MoeReRoutingTiling::IsCapable() { return true; }

ge::graphStatus MoeReRoutingTiling::DoOpTiling()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."),
                return ge::GRAPH_FAILED);
    int64_t tokenNum = tilingData_.get_tokensNum();
    int64_t tokenSize = tilingData_.get_tokensSize();
    int64_t rankNums = tilingData_.get_rankNum();
    int64_t ubFactor;
    int64_t usedUbBuffer = (ubSize_ - RESERVE_UB_SIZE) / 2;
    if (commonParams_.tokenDtype == ge::DT_INT8) {
        if (commonParams_.scaleDtype != ge::DT_UNDEFINED) {
            ubFactor = static_cast<int64_t>(usedUbBuffer / (tokenSize + sizeof(float) + sizeof(int32_t)));
        } else {
            ubFactor = static_cast<int64_t>(usedUbBuffer / (tokenSize + sizeof(int32_t)));
        }
    } else {
        if (commonParams_.scaleDtype != ge::DT_UNDEFINED) {
            ubFactor =
                static_cast<int64_t>(usedUbBuffer / (tokenSize * sizeof(short) + sizeof(float) + sizeof(int32_t)));
        } else {
            ubFactor = static_cast<int64_t>(usedUbBuffer / (tokenSize * sizeof(short) + sizeof(int32_t)));
        }
    }

    std::pair<ge::DataType, ge::DataType> keyToFind = {commonParams_.tokenDtype, commonParams_.expertDtype};
    auto valueToFind = TILING_KEY_MAP.find(keyToFind);
    OP_CHECK_IF(valueToFind == TILING_KEY_MAP.end(), OP_LOGE(context_->GetNodeName(), "Failed to find tilingkey."),
                return ge::GRAPH_FAILED);
    tilingKey_ = valueToFind->second;

    int64_t blockDimFactor = CeilDiv(rankNums, coreNum_);
    int64_t numBlocks = CeilDiv(rankNums, blockDimFactor);
    if (tokenNum == 0) {
        numBlocks = 1;
    }
    context_->SetBlockDim(numBlocks);
    tilingData_.set_coreNum(numBlocks);
    tilingData_.set_ubFactor(ubFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MoeReRoutingTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

uint64_t MoeReRoutingTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus MoeReRoutingTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(32); // 设置工作区大小为32字节
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MoeReRouting, MoeReRoutingTiling, 20000);

} // namespace optiling
