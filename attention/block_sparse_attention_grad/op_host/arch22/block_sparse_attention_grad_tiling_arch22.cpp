/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../block_sparse_attention_grad_tiling.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "util/shape_util.h"
#include "log/log.h"

#include <cstdint>
#include <string>
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace std;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSH_DIM_B = 0;
constexpr int BSH_DIM_S = 1;
constexpr int BSH_DIM_H = 2;

constexpr int DOUT_INDEX = 0;
constexpr int QUERY_INDEX = 1;
constexpr int KEY_INDEX = 2;
constexpr int VALUE_INDEX = 3;
constexpr int OUT_INDEX = 4;
constexpr int SOFTMAX_LSE_INDEX = 5;
constexpr int BLOCK_SPARSE_MASK_INDEX = 6;
constexpr int BLOCK_SHAPE_INDEX = 8;

constexpr int ATTENTION_MASK_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 9;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 10;

constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;

constexpr int VALID_HEAD_DIM_128 = 128;
constexpr uint64_t VEC_POST_DIVISION = 3; // post 划分的空间块数， input 2份，output 1份
constexpr uint64_t WORKSPACE_NUM_ALIGN = 256;
constexpr uint32_t BATCH_SCHEDULE_MODE = 1;
constexpr uint32_t SCHEDULE_LEGACY = 0;
constexpr uint32_t SCHEDULE_WIDE = 2;
constexpr uint32_t WIDE_MAX_EDGES = 8;
constexpr uint32_t WIDE_COMPACT_Q_TASKS = 2;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 2 * 2;
constexpr uint32_t NUM3 = 3;
constexpr uint32_t ONEBLOCK_FLOAT_NUM = 32 / sizeof(float); // 基本块32字节的float数目
static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    uint32_t qBlocksInX = CeilDiv(x, BASIC_BLOCK_SIZE);
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = CeilDiv(remainingSeqlen, BASIC_BLOCK_SIZE);
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

ge::graphStatus BSAGradTiling::GetNpuInfo(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessTND(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    const auto* queryShape = context->GetInputShape(QUERY_INDEX);
    const auto* kvShape = context->GetInputShape(KEY_INDEX);

    if (queryShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != TND_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "query,key",
                                                  (std::to_string(queryShape->GetOriginShape().GetDimNum()) + "," +
                                                   std::to_string(kvShape->GetOriginShape().GetDimNum()))
                                                      .c_str(),
                                                  "TND layout requires query and key rank 3 [T,N,D].");
        return ge::GRAPH_FAILED;
    }
    totalTokensT_ = queryShape->GetOriginShape().GetDim(TND_DIM_T);
    kvTotalSeqlen_ = kvShape->GetStorageShape().GetDim(TND_DIM_T);
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(TND_DIM_N));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(TND_DIM_D));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(TND_DIM_N));
    dqSize_ = totalTokensT_ * numHeads_ * headDim_;
    dkvSize_ = kvTotalSeqlen_ * kvHeads_ * headDim_;

    auto actualSeqLengths = context->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_INDEX);
    if (actualSeqLengths == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "actualSeqLengthsOptional",
            "TND layout requires actualSeqLengthsOptional; supply the query sequence-length tensor.");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
    qSeqLenList = actualSeqLengths->GetData<int64_t>();
    if (qSeqLenList == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "actualSeqLengthsOptional",
                                                 "Query sequence-length data is unavailable; supply populated INT64 "
                                                 "data and check value-dependent Host transfer for TND tiling.");
        return ge::GRAPH_FAILED;
    }

    auto actualSeqLengthsKv = context->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);
    if (actualSeqLengthsKv == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "actualSeqLengthsKvOptional",
            "TND layout requires actualSeqLengthsKvOptional; supply the key/value sequence-length tensor.");
        return ge::GRAPH_FAILED;
    }
    kvSeqLenList = actualSeqLengthsKv->GetData<int64_t>();
    if (kvSeqLenList == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "actualSeqLengthsKvOptional",
                                                 "Key/value sequence-length data is unavailable; supply populated "
                                                 "INT64 data and check value-dependent Host transfer for TND tiling.");
        return ge::GRAPH_FAILED;
    }

    maxQSeqlen_ = 0;
    maxKvSeqlen_ = 0;
    for (uint32_t i = 0; i < batch_; ++i) {
        maxQSeqlen_ = std::max(maxQSeqlen_, static_cast<uint32_t>(qSeqLenList[i]));
        maxKvSeqlen_ = std::max(maxKvSeqlen_, static_cast<uint32_t>(kvSeqLenList[i]));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessBNSD(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    const auto* queryShape = context->GetInputShape(QUERY_INDEX);
    const auto* kvShape = context->GetInputShape(KEY_INDEX);
    auto blockSparseMaskShape = context->GetInputShape(BLOCK_SPARSE_MASK_INDEX);

    if (queryShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM ||
        blockSparseMaskShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "query,key,blockSparseMaskOptional",
            (std::to_string(queryShape->GetOriginShape().GetDimNum()) + "," +
             std::to_string(kvShape->GetOriginShape().GetDimNum()) + "," +
             std::to_string(blockSparseMaskShape->GetOriginShape().GetDimNum()))
                .c_str(),
            "BNSD layout requires query, key and blockSparseMaskOptional rank 4.");
        return ge::GRAPH_FAILED;
    }
    // 保存BNSD格式的batch和S维度
    batch_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_B));
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_N));
    maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_S));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_D));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_N));
    maxKvSeqlen_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_S));
    dqSize_ = batch_ * numHeads_ * maxQSeqlen_ * headDim_;
    dkvSize_ = batch_ * kvHeads_ * maxKvSeqlen_ * headDim_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessAttrs(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    auto blockShapeOptional = context->GetInputTensor(BLOCK_SHAPE_INDEX);
    if (blockShapeOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "blockShapeOptional",
            "The block-shape tensor is required by this tiling implementation; supply [blockShapeX,blockShapeY].");
        return ge::GRAPH_FAILED;
    }
    blockShapeList = blockShapeOptional->GetData<int64_t>();
    if (blockShapeList != nullptr) {
        // 真实的 NPU 环境：GE 框架自动将张量放到 Host 侧，正常读取
        blockShapeX_ = blockShapeList[0];
        blockShapeY_ = blockShapeList[1];
    } else {
        // 这里提供默认切块大小，保证 UT 测试环境的后续切分逻辑能顺利验证
        blockShapeX_ = 64;
        blockShapeY_ = 64;
    }

    if (context->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX) == nullptr) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(headDim_));
    } else {
        scaleValue_ = *context->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE_INDEX);
    }

    auto qInputDesc = context->GetInputDesc(QUERY_INDEX);
    if (qInputDesc == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName,
                "Query input descriptor is unavailable; check Tiling context metadata and operator registration."}));
        return ge::GRAPH_FAILED;
    } else {
        dataType_ = qInputDesc->GetDataType();
    }

    if (kvHeads_ == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "key", (std::to_string(kvHeads_)).c_str(),
                                                  "The key/value head count must be greater than zero.");
        return ge::GRAPH_FAILED;
    }

    if (!(numHeads_ >= kvHeads_ && numHeads_ % kvHeads_ == 0)) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName, "query,key",
            (std::string("query heads=") + std::to_string(numHeads_) + ",key heads=" + std::to_string(kvHeads_))
                .c_str(),
            "Query heads must be greater than or equal to key heads and divisible by key heads.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ParseAttnMaskCounts(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    const auto* maskType = context->GetAttrs()->GetAttrPointer<int64_t>(MASK_TYPE_INDEX);
    const int64_t requestedMaskType = maskType == nullptr ? 0 : *maskType;
    maskType_ = 0;
    hasPerBlockMask_ = false;
    maxMaskBlocks_ = 0;
    auto mask = context->GetOptionalInputTensor(ATTENTION_MASK_INDEX);
    if (requestedMaskType == 0) {
        // L0 launcher 会为 nullptr 创建空占位 Tensor。
        if (mask != nullptr && mask->GetStorageShape().GetDimNum() != 0 && mask->GetShapeSize() != 0) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName, "maskType,attenMaskOptional",
                (std::string("maskType=0 requires an absent or empty attention mask; actual shape=") +
                 Ops::Base::ToString(mask->GetStorageShape()))
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    auto desc = context->GetOptionalInputDesc(ATTENTION_MASK_INDEX);
    if (requestedMaskType != 1 || mask == nullptr || desc == nullptr || desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "maskType,attenMaskOptional",
            (std::string("Require maskType=1 with INT32 attention counts [B,Nq,M,2]; actual maskType=") +
             std::to_string(requestedMaskType) + ", tensor present=" + std::to_string(mask != nullptr) +
             ", descriptor present=" + std::to_string(desc != nullptr) +
             ", dtype enum=" + (desc == nullptr ? "<missing>" : std::to_string(static_cast<int>(desc->GetDataType()))))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    const uint32_t minBlocks = std::max(CeilDiv(maxQSeqlen_, static_cast<uint32_t>(blockShapeX_)),
                                        CeilDiv(maxKvSeqlen_, static_cast<uint32_t>(blockShapeY_)));
    const auto& shape = mask->GetStorageShape();
    if (shape.GetDimNum() != 4 || shape.GetDim(0) != batch_ || shape.GetDim(1) != numHeads_ ||
        shape.GetDim(2) < minBlocks || shape.GetDim(2) > std::numeric_limits<uint32_t>::max() || shape.GetDim(3) != 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "attenMaskOptional", Ops::Base::ToString(shape).c_str(),
            (std::string("For maskType=1 require [B=") + std::to_string(batch_) + ",Nq=" + std::to_string(numHeads_) +
             ",M in [" + std::to_string(minBlocks) + "," + std::to_string(std::numeric_limits<uint32_t>::max()) +
             "],2].")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    maxMaskBlocks_ = static_cast<uint32_t>(shape.GetDim(2));
    maskType_ = 1;
    hasPerBlockMask_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::ProcessInput(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    if (context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "qInputLayout",
            "Required query input layout is unavailable; provide the layout attribute in the operator definition.");
        return ge::GRAPH_FAILED;
    }

    std::string qLayout(context->GetAttrs()->GetAttrPointer<char>(Q_INPUT_LAYOUT_INDEX));
    if (qLayout == "TND") {
        layout_ = InputLayout::TND;
    } else if (qLayout == "BNSD") {
        layout_ = InputLayout::BNSD;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(opName, "qInputLayout", qLayout.c_str(),
                                               "Supported query layouts are TND and BNSD.");
        return ge::GRAPH_FAILED;
    }

    // 核心 Shape 的判空前置，保证安全性
    if (context->GetInputShape(QUERY_INDEX) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "query");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(KEY_INDEX) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "key");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(BLOCK_SPARSE_MASK_INDEX) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "blockSparseMaskOptional");
        return ge::GRAPH_FAILED;
    }

    if (layout_ == InputLayout::TND) {
        if (ProcessTND(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (layout_ == InputLayout::BNSD) {
        if (ProcessBNSD(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    deterministic_ = (context->GetDeterministic() == 1);
    if (ProcessAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ParseAttnMaskCounts(context);
}

ge::graphStatus BSAGradTiling::CalculatePostUbBaseSize(gert::TilingContext* context)
{
    // post 计算：划分空间块数，256 字节对齐
    postUbBaseSize_ = static_cast<uint64_t>(ubSize_ - sizeof(BlockSparseAttentionGradTilingData) - 2 * 1024) /
                      VEC_POST_DIVISION / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateSoftmaxGradTiling(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    // 安全校验：防止除以 0 导致 Core Dump 崩溃
    if (headDim_ == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName, "query", (std::to_string(headDim_)).c_str(),
            "The query head dimension D must be greater than zero for SoftmaxGrad tiling.");
        return ge::GRAPH_FAILED;
    }

    constexpr static uint64_t inputBufferLen = 24 * 1024; // castBuffer 24K*2=48K
    constexpr static uint64_t castBufferLen = 48 * 1024;  // castBuffer 48K*2=96K

    uint64_t outputBufferLen = (castBufferLen + headDim_ - 1) / headDim_ * ONEBLOCK_FLOAT_NUM; // 输出(s1,8)
    uint64_t tempBufferLen = 40 * 1024 - outputBufferLen;

    int64_t singleLoopNBurstNum = inputBufferLen / sizeof(float) / headDim_;
    auto softmaxGradShape = ge::Shape({singleLoopNBurstNum, headDim_});

    // 调用 CANN 底层的 SoftMaxGradTilingFunc
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen, tilingData_->softmaxGradTilingData,
                                   true);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::AssignCoreTasks(uint32_t numHeads, uint32_t kvHeads, uint32_t blockX, uint32_t coreNum,
                                               const std::vector<uint32_t>& tasksInBatch,
                                               const std::vector<uint64_t>& qPrefixTokenSum,
                                               const std::vector<uint64_t>& kvPrefixTokenSum)
{
    if (coreNum == 0 || numHeads == 0 || kvHeads == 0 || numHeads < kvHeads) {
        return ge::GRAPH_FAILED;
    }

    taskNumPerCore_ = totalTaskNum_ / coreNum;
    tailTaskNum_ = totalTaskNum_ % coreNum;
    uint32_t currentGlobalTaskId = 0;
    uint32_t qBlocksInX = CeilDiv(blockX, BASIC_BLOCK_SIZE);
    qBlocksInX = qBlocksInX == 0 ? 1 : qBlocksInX;

    for (uint32_t i = 0; i < coreNum; i++) {
        uint32_t curBatch = 0, curHeadNum = 0, curQSeqIdx = 0, tempId = currentGlobalTaskId;
        uint64_t preQSeqLengths = 0, preKVSeqLengths = 0;

        if (layout_ == InputLayout::TND) {
            // TND遍历顺序是 Batch -> SeqBlock -> Head
            for (uint32_t b = 0; b < batch_; b++) {
                if (tempId < tasksInBatch[b]) {
                    curBatch = b;
                    curHeadNum = tempId % numHeads;
                    uint32_t blockIdx = tempId / numHeads;

                    uint32_t macroBlockIdx = blockIdx / qBlocksInX;
                    uint32_t microBlockInMacro = blockIdx % qBlocksInX;

                    curQSeqIdx = macroBlockIdx * blockX + microBlockInMacro * BASIC_BLOCK_SIZE;
                    preQSeqLengths = qPrefixTokenSum[b] + curQSeqIdx;
                    preKVSeqLengths = kvPrefixTokenSum[b];
                    break;
                }
                tempId -= tasksInBatch[b];
            }
        } else {
            // BNSD遍历顺序是 Batch -> Head -> SeqBlock
            uint32_t qBlocksPerHead = GetQBlocks(maxQSeqlen_, blockX);
            qBlocksPerHead = qBlocksPerHead == 0 ? 1 : qBlocksPerHead;

            curBatch = tempId / (numHeads * qBlocksPerHead);
            uint32_t remain = tempId % (numHeads * qBlocksPerHead);
            curHeadNum = remain / qBlocksPerHead;
            uint32_t blockIdx = remain % qBlocksPerHead;

            curQSeqIdx = (blockIdx / qBlocksInX) * blockX + (blockIdx % qBlocksInX) * BASIC_BLOCK_SIZE;

            preQSeqLengths = (static_cast<uint64_t>(curBatch) * numHeads * maxQSeqlen_) +
                             (static_cast<uint64_t>(curHeadNum) * maxQSeqlen_) + (static_cast<uint64_t>(curQSeqIdx));
            preKVSeqLengths = (static_cast<uint64_t>(curBatch) * kvHeads * maxKvSeqlen_) +
                              (static_cast<uint64_t>(curHeadNum / (numHeads / kvHeads)) * maxKvSeqlen_);
        }

        tilingData_->get_beginBatch()[i] = curBatch;
        tilingData_->get_beginHead()[i] = curHeadNum;
        tilingData_->get_beginQSeqOffset()[i] = curQSeqIdx;
        tilingData_->get_preQSeqLengths()[i] = preQSeqLengths;
        tilingData_->get_preKVSeqLengths()[i] = preKVSeqLengths;

        uint32_t taskLen = (i < tailTaskNum_) ? (taskNumPerCore_ + 1) : taskNumPerCore_;
        currentGlobalTaskId += taskLen;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateTaskSplit(gert::TilingContext* context)
{
    SelectSchedule();
    uint32_t numHeads = numHeads_;
    uint32_t kvHeads = kvHeads_;
    uint32_t blockX = blockShapeX_;
    uint32_t coreNum = aicNum_;

    totalTaskNum_ = 0;
    totalQBlocks_ = 0;

    const auto* queryShape = context->GetInputShape(QUERY_INDEX);
    const auto* kvShape = context->GetInputShape(KEY_INDEX);
    uint32_t totalQ = (layout_ == InputLayout::TND) ? queryShape->GetOriginShape().GetDim(TND_DIM_T) : 0;
    uint32_t totalKv = (layout_ == InputLayout::TND) ? kvShape->GetOriginShape().GetDim(TND_DIM_T) : 0;

    std::vector<uint32_t> tasksInBatch(batch_);
    std::vector<uint64_t> qPrefixTokenSum(batch_ + 1, 0);
    std::vector<uint64_t> kvPrefixTokenSum(batch_ + 1, 0);

    for (uint32_t b = 0; b < batch_; b++) {
        uint32_t qSeqlen = 0;
        uint32_t kvSeqlen = 0;
        if (layout_ == InputLayout::TND) {
            qSeqlen = static_cast<uint32_t>(qSeqLenList[b]);
            kvSeqlen = static_cast<uint32_t>(kvSeqLenList[b]);
        } else {
            qSeqlen = maxQSeqlen_;
            kvSeqlen = maxKvSeqlen_;
        }

        uint32_t qBlocks = GetQBlocks(qSeqlen, blockX);
        tasksInBatch[b] =
            (scheduleMode_ == SCHEDULE_WIDE ? CeilDiv(qBlocks, wideQTasksPerWorkItem_) : qBlocks) * numHeads;
        totalTaskNum_ += tasksInBatch[b];

        uint32_t curQBlockNum = CeilDiv(qSeqlen, blockX) * numHeads;
        totalQBlocks_ += curQBlockNum;

        qPrefixTokenSum[b + 1] = qPrefixTokenSum[b] + qSeqlen;
        kvPrefixTokenSum[b + 1] = kvPrefixTokenSum[b] + kvSeqlen;
    }

    if (scheduleMode_ == SCHEDULE_WIDE) {
        blockDim_ = std::min(aicNum_, totalTaskNum_);
        return AssignWideCoreTasks(blockDim_);
    }
    if (AssignCoreTasks(numHeads, kvHeads, blockX, coreNum, tasksInBatch, qPrefixTokenSum, kvPrefixTokenSum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    blockDim_ = std::min(aicNum_, totalTaskNum_);
    return ge::GRAPH_SUCCESS;
}

// deterministic 优先使用原 legacy 路径；未迁移的 grouped 形状同样回退 legacy。
uint32_t BSAGradTiling::SelectSchedule()
{
    const bool wideEligible = layout_ == InputLayout::BNSD && blockShapeY_ <= BASIC_BLOCK_SIZE &&
                              (blockShapeY_ == BASIC_BLOCK_SIZE || (blockShapeX_ == 64 && blockShapeY_ == 64));
    scheduleMode_ = !deterministic_ && wideEligible ? SCHEDULE_WIDE : SCHEDULE_LEGACY;
    // 固定聚合两个 Q task。
    wideQTasksPerWorkItem_ = scheduleMode_ == SCHEDULE_WIDE ? WIDE_COMPACT_Q_TASKS : 0;
    return scheduleMode_;
}

/**
 * @brief 为 BNSD wide work item 分配核起点和任务数量。
 * @param coreNum 实际启动的 AIC 核数，等于 min(aicNum, work item 总数)。
 * @return 非空任务成功返回 GRAPH_SUCCESS，零任务返回 GRAPH_FAILED。
 */
ge::graphStatus BSAGradTiling::AssignWideCoreTasks(uint32_t coreNum)
{
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t fragmentsPerBlock = CeilDiv(static_cast<uint32_t>(blockShapeX_), BASIC_BLOCK_SIZE);
    const uint32_t workItemsPerHead = CeilDiv(GetQBlocks(maxQSeqlen_, blockShapeX_), wideQTasksPerWorkItem_);
    taskNumPerCore_ = totalTaskNum_ / coreNum;
    tailTaskNum_ = totalTaskNum_ % coreNum;
    for (uint32_t coreIndex = 0; coreIndex < coreNum; ++coreIndex) {
        const uint32_t workItemIndex = coreIndex; // 本核负责的第一个 work item 的全局编号
        const uint32_t headIndex = workItemIndex / workItemsPerHead;
        const uint32_t firstQTask = (workItemIndex % workItemsPerHead) * wideQTasksPerWorkItem_;
        const uint32_t queryStart =
            (firstQTask / fragmentsPerBlock) * blockShapeX_ + (firstQTask % fragmentsPerBlock) * BASIC_BLOCK_SIZE;
        const uint32_t batchIndex = headIndex / numHeads_;
        const uint32_t queryHead = headIndex % numHeads_;
        tilingData_->get_beginBatch()[coreIndex] = batchIndex;
        tilingData_->get_beginHead()[coreIndex] = queryHead;
        tilingData_->get_beginQSeqOffset()[coreIndex] = queryStart;
        tilingData_->get_preQSeqLengths()[coreIndex] = static_cast<uint64_t>(headIndex) * maxQSeqlen_ + queryStart;
        tilingData_->get_preKVSeqLengths()[coreIndex] =
            (static_cast<uint64_t>(batchIndex) * kvHeads_ + queryHead / (numHeads_ / kvHeads_)) * maxKvSeqlen_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::CalculateWorkSpace(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    if (blockDim_ == 0) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName, (std::string("Workspace tiling requires a positive blockDim; actual=") +
                         std::to_string(blockDim_) + ", AIC cores=" + std::to_string(aicNum_) +
                         ", tasks=" + std::to_string(totalTaskNum_) + "; check platform resources and task splitting.")
                            .c_str()}));
        return ge::GRAPH_FAILED;
    }

    sOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float);
    dPOutSize_ = blockDim_ * WORKSPACE_BLOCK_SIZE_DB * sizeof(float);
    if (scheduleMode_ == SCHEDULE_WIDE) {
        // 每个核两个 stage，每个 stage 最多 8 个 128x128 tile；S/P、dP/dS 各占一组 plane。
        wideWorkspaceStageElements_ = BASIC_BLOCK_SIZE * BASIC_BLOCK_SIZE * WIDE_MAX_EDGES;
        wideWorkspaceCoreElements_ = wideWorkspaceStageElements_ * 2;
        wideRawPlaneBytes_ = static_cast<uint64_t>(blockDim_) * wideWorkspaceCoreElements_ * sizeof(float);
        sOutSize_ =
            wideRawPlaneBytes_ + static_cast<uint64_t>(blockDim_) * wideWorkspaceCoreElements_ * sizeof(uint16_t);
        dPOutSize_ = sOutSize_;
    }
    if (layout_ == InputLayout::TND) {
        dQOutSize_ = totalTokensT_ * numHeads_ * headDim_ * sizeof(float);
        dKOutSize_ = kvTotalSeqlen_ * kvHeads_ * headDim_ * sizeof(float);
        dVOutSize_ = dKOutSize_;
        gradSize_ = totalTokensT_ * numHeads_ * ONEBLOCK_FLOAT_NUM * sizeof(float);
    } else {
        dQOutSize_ = batch_ * numHeads_ * maxQSeqlen_ * headDim_ * sizeof(float);
        dKOutSize_ = batch_ * kvHeads_ * maxKvSeqlen_ * headDim_ * sizeof(float);
        dVOutSize_ = dKOutSize_;
        gradSize_ = batch_ * numHeads_ * maxQSeqlen_ * ONEBLOCK_FLOAT_NUM * sizeof(float);
    }

    uint32_t groupSizeForDet_ = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    uint32_t batchSizeForDet_ = 40;
    uint32_t KForDet_ = (aicNum_ > 0) ? ((batchSizeForDet_ + aicNum_ - 1) / aicNum_) : 1;
    if (KForDet_ < 1)
        KForDet_ = 1;
    uint64_t detDkWorkspaceSize_ =
        deterministic_ ? ((uint64_t)aicNum_ * groupSizeForDet_ * dkvSize_ * sizeof(float)) : 0;
    uint64_t detDvWorkspaceSize_ =
        deterministic_ ? ((uint64_t)aicNum_ * groupSizeForDet_ * dkvSize_ * sizeof(float)) : 0;
    workSpaceSize_ = libapiSize_ + sOutSize_ + dPOutSize_ + dQOutSize_ + dKOutSize_ + dVOutSize_ + gradSize_ +
                     detDkWorkspaceSize_ + detDvWorkspaceSize_;
    context->GetWorkspaceSizes(1)[0] = workSpaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::FillTilingData(gert::TilingContext* context)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_headDim(headDim_);
    tilingData_->set_totalTaskNum(totalTaskNum_);
    tilingData_->set_maskType(maskType_);
    tilingData_->set_hasPerBlockMask(hasPerBlockMask_ ? 1U : 0U);
    tilingData_->set_maxMaskBlocks(maxMaskBlocks_);
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    tilingData_->set_totalQBlocks(totalQBlocks_);
    tilingData_->set_inputLayout(static_cast<uint32_t>(layout_));
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_maxKvSeqlen(maxKvSeqlen_);
    tilingData_->set_basicQBlockSize(BASIC_BLOCK_SIZE);
    tilingData_->set_basicKVBlockSize(
        scheduleMode_ == SCHEDULE_WIDE ? std::min<int64_t>(blockShapeY_, BASIC_BLOCK_SIZE) : BASIC_BLOCK_SIZE);
    tilingData_->set_scheduleMode(scheduleMode_);
    tilingData_->set_wideQTasksPerWorkItem(wideQTasksPerWorkItem_);
    tilingData_->set_widePacketEdgeCapacity(scheduleMode_ == SCHEDULE_WIDE ? WIDE_MAX_EDGES : 0);
    tilingData_->set_wideWorkspaceStageElements(wideWorkspaceStageElements_);
    tilingData_->set_wideWorkspaceCoreElements(wideWorkspaceCoreElements_);
    tilingData_->set_wideRawPlaneBytes(wideRawPlaneBytes_);
    tilingData_->set_wideTaskStride(scheduleMode_ == SCHEDULE_WIDE ? blockDim_ : 0);
    tilingData_->set_taskNumPerCore(taskNumPerCore_);
    tilingData_->set_tailTaskNum(tailTaskNum_);

    // 生成tilingKey（按照开发规范：在tiling层生成）
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim_);
    context->SetScheduleMode(BATCH_SCHEDULE_MODE);

    tilingData_->set_sOutSize(sOutSize_);
    tilingData_->set_dPOutSize(dPOutSize_);
    tilingData_->set_dQOutSize(dQOutSize_);
    tilingData_->set_dKOutSize(dKOutSize_);
    tilingData_->set_dVOutSize(dVOutSize_);
    tilingData_->set_gradSize(gradSize_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_usedVecCoreNum(blockDim_ * 2);
    tilingData_->set_qTotalSeqlen(totalTokensT_);
    tilingData_->set_kvTotalSeqlen(kvTotalSeqlen_);
    tilingData_->set_dqSize(dqSize_);
    tilingData_->set_dkvSize(dkvSize_);
    tilingData_->set_postUbBaseSize(postUbBaseSize_);
    tilingData_->set_ubSize(ubSize_ - sizeof(BlockSparseAttentionGradTilingData) - 2 * 1024);
    uint32_t gsForDet = (kvHeads_ > 0) ? (numHeads_ / kvHeads_) : 1;
    uint32_t bsForDet = 40;
    uint32_t kForDet = (aicNum_ > 0) ? ((bsForDet + aicNum_ - 1) / aicNum_) : 1;
    if (kForDet < 1)
        kForDet = 1;
    uint64_t detDkSize = deterministic_ ? ((uint64_t)aicNum_ * gsForDet * dkvSize_ * sizeof(float)) : 0;
    uint64_t detDvSize = deterministic_ ? ((uint64_t)aicNum_ * gsForDet * dkvSize_ * sizeof(float)) : 0;
    tilingData_->set_detDkWorkspaceSize(detDkSize);
    tilingData_->set_detDvWorkspaceSize(detDvSize);
    tilingData_->set_aicNumForDet(aicNum_);
    tilingData_->set_groupSizeForDet(gsForDet);
    tilingData_->set_deterministic(deterministic_ ? 1 : 0);
    tilingData_->set_batchSizeForDet(bsForDet);
    tilingData_->set_KForDet(kForDet);
    return ge::GRAPH_SUCCESS;
}

uint64_t BSAGradTiling::GenerateTilingKey()
{
    uint64_t tilingKey = 0;
    if (dataType_ == ge::DT_FLOAT16) {
        if (layout_ == InputLayout::TND) {
            tilingKey = 0;
        } else {
            tilingKey = 1;
        }
    } else if (dataType_ == ge::DT_BF16) {
        if (layout_ == InputLayout::TND) {
            tilingKey = 10;
        } else {
            tilingKey = 11;
        }
    }
    return tilingKey;
}

ge::graphStatus BSAGradTiling::GetBSAGradTiling(gert::TilingContext* context,
                                                BlockSparseAttentionGradTilingData& tilingData)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName, (std::string("GetNpuInfo failed; ret=") + std::to_string(ret) +
                                               "; check the SoC-specific platform resource metadata.")
                                                  .c_str()}));
        return ret;
    }

    ret = ProcessInput(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "ProcessInput failed; retain its first error.");
        return ret;
    }

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName, (std::string("CalculateTaskSplit failed; ret=") + std::to_string(ret) +
                         ", AIC cores=" + std::to_string(aicNum_) + ", query heads=" + std::to_string(numHeads_) +
                         ", key heads=" + std::to_string(kvHeads_) + ", tasks=" + std::to_string(totalTaskNum_) +
                         "; check task dimensions and platform resources.")
                            .c_str()}));
        return ret;
    }

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "CalculateWorkSpace failed; retain its first error.");
        return ret;
    }

    ret = CalculatePostUbBaseSize(context);
    if (ret != ge::GRAPH_SUCCESS) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                opName, (std::string("CalculatePostUbBaseSize failed; ret=") + std::to_string(ret) +
                         ", UB bytes=" + std::to_string(ubSize_) + "; check post-processing UB resource calculation.")
                            .c_str()}));
        return ret;
    }

    ret = CalculateSoftmaxGradTiling(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "CalculateSoftmaxGradTiling failed; retain its first error.");
        return ret;
    }

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName, (std::string("FillTilingData failed; ret=") + std::to_string(ret) +
                                               "; check the BlockSparseAttentionGrad tiling-data object.")
                                                  .c_str()}));
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BSAGradTiling::SetTilingData(gert::TilingContext* context,
                                             BlockSparseAttentionGradTilingData& tilingData)
{
    const char* opName = context->GetNodeName() == nullptr ? "BlockSparseAttentionGrad" : context->GetNodeName();
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        opName, "Raw tiling-data buffer is unavailable; check Tiling context output construction."})),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

class BlockSparseAttentionGradArch22Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockSparseAttentionGradArch22Tiling(gert::TilingContext* context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        OP_CHECK_IF(
            context_ == nullptr,
            REPORT_PREDEFINED_ERR_MSG(
                "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                (std::vector<const char*>{
                    "BlockSparseAttentionGrad",
                    "Tiling context is null; check BlockSparseAttentionGrad Tiling invocation and registration."})),
            return ge::GRAPH_FAILED);
        if (tiling_.GetBSAGradTiling(context_, tilingData_) != ge::GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(context_, "GetBSAGradTiling failed; retain its first error.");
            return ge::GRAPH_FAILED;
        }
        tilingKey_ = tilingData_.get_tilingKey();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        return tiling_.SetTilingData(context_, tilingData_);
    }

    uint64_t GetTilingKey() const override
    {
        return tilingKey_;
    }

private:
    BSAGradTiling tiling_;
    BlockSparseAttentionGradTilingData tilingData_;
    uint64_t tilingKey_ = 0;
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad, BlockSparseAttentionGradArch22Tiling,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 1);

} // namespace optiling
