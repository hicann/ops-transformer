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
 * \file generic_block_sparse_attention_grad_tiling_arch22.cpp
 * \brief Arch22 tiling for GenericBlockSparseAttentionGrad.
 */

#include "../generic_block_sparse_attention_grad_tiling.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "util/shape_util.h"

#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t TND_DIM_T = 0;
constexpr uint32_t TND_DIM_N = 1;
constexpr uint32_t TND_DIM_D = 2;
constexpr uint32_t TND_DIM_NUM = 3;

constexpr uint32_t BNSD_DIM_B = 0;
constexpr uint32_t BNSD_DIM_N = 1;
constexpr uint32_t BNSD_DIM_S = 2;
constexpr uint32_t BNSD_DIM_D = 3;
constexpr uint32_t BNSD_DIM_NUM = 4;

constexpr uint64_t TILING_KEY_100 = 100;
constexpr uint64_t TILING_KEY_101 = 101;
constexpr uint64_t TILING_KEY_102 = 102;
constexpr uint64_t TILING_KEY_110 = 110;
constexpr uint64_t TILING_KEY_111 = 111;
constexpr uint64_t TILING_KEY_112 = 112;

enum class InputId : uint32_t {
    Query = 0,
    Key = 1,
    Value = 2,
    DOut = 3,
    Out = 4,
    SoftmaxLse = 5,
    RsvdBlockIdx = 6,
    RsvdBlockCount = 7,
    Metadata = 8,
    AttentionMask = 9,
    CuSeqLengthsQ = 10,
    CuSeqLengthsKv = 11,
    ActualSeqLengths = 12,
    ActualSeqLengthsKv = 13,
};

// attr 索引与 def.cpp 中 Attr 声明顺序一致（block_shape 为首个 attr）。
enum class AttrId : uint32_t {
    BlockShape = 0,
    QInputLayout = 1,
    KvInputLayout = 2,
    LayoutSparsePattern = 3,
    ScaleValue = 4,
    MaskType = 5,
    SoftmaxPrecision = 6,
    WinLeft = 7,
    WinRight = 8,
    ResidualBlockMode = 9,
    IsConsistentTopk = 10,
};

constexpr uint32_t ToIndex(InputId id)
{
    return static_cast<uint32_t>(id);
}

constexpr uint32_t ToIndex(AttrId id)
{
    return static_cast<uint32_t>(id);
}

constexpr uint64_t VEC_POST_DIVISION = 3; // post 划分的空间块数， input 2份，output 1份
constexpr uint64_t WORKSPACE_NUM_ALIGN = 256;
constexpr uint32_t BATCH_SCHEDULE_MODE = 1;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;

constexpr int64_t ARC22_METADATA_MIN_ELEMS = 198;

constexpr uint32_t WORKSPACE_BLOCK_SIZE = 128 * 128 * 2;
constexpr uint32_t WORKSPACE_BLOCK_NUM = 3;
constexpr uint32_t Q_AGGREGATE_M = 128;                     // Cube 单次计算允许的最大聚合 Q 行数
constexpr uint32_t Q_AGGREGATE_MAX_SEGMENTS = 128;          // blockX=1 时的最坏分段数
constexpr uint32_t ONEBLOCK_FLOAT_NUM = 32 / sizeof(float); // 基本块32字节的float数目
constexpr uint64_t Q_PACKET_INPUT_BYTES = static_cast<uint64_t>(Q_AGGREGATE_M) * 128 * 2;            // 单份 Q/dOut
constexpr uint64_t Q_PACKET_GRAD_BYTES = static_cast<uint64_t>(Q_AGGREGATE_M) * 128 * sizeof(float); // 单份 dQ
constexpr uint64_t Q_PACKET_META_BYTES = Q_AGGREGATE_MAX_SEGMENTS * 4 * sizeof(uint32_t);            // 分段元数据
static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline const gert::Tensor* NormalizeOptionalTensor(const gert::Tensor* tensor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    const auto& shape = tensor->GetStorageShape();
    const size_t dimNum = shape.GetDimNum();
    const bool absent = (dimNum == 0) || (dimNum == 1 && shape.GetDim(0) <= 0);
    return absent ? nullptr : tensor;
}

// Tiling类
class GBSAGTiling {
public:
    GBSAGTiling() = default;
    ~GBSAGTiling() = default;

    ge::graphStatus GetGBSAGTiling(gert::TilingContext* context, GenericBlockSparseAttentionGradTilingData& tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext* context, GenericBlockSparseAttentionGradTilingData& tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext* context);
    ge::graphStatus ProcessInput(gert::TilingContext* context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext* context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext* context);
    ge::graphStatus FillTilingData(gert::TilingContext* context);

    ge::graphStatus CalculatePostUbBaseSize(gert::TilingContext* context);
    ge::graphStatus CalculateSoftmaxGradTiling(gert::TilingContext* context);

    ge::graphStatus ProcessTND(gert::TilingContext* context);
    ge::graphStatus ProcessBNSD(gert::TilingContext* context);
    ge::graphStatus ProcessBSND(gert::TilingContext* context);

    ge::graphStatus ProcessAttrs(gert::TilingContext* context);

    uint64_t GenerateTilingKey() const;

    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t headDim_ = 0;
    int64_t blockShapeX_ = 0; // block的x维度
    int64_t blockShapeY_ = 0; // block的y维度
    float scaleValue_ = 0.0f;
    uint32_t maskType_ = 0;
    uint32_t layoutSparsePattern_ = 1;
    uint32_t softmaxPrecision_ = 0;
    int64_t winLeft_ = -1;
    int64_t winRight_ = -1;

    bool useUniformQSeqlen_ = false;  // 是否使用统一的qseqlen值（使用maxQSeqlen_）
    bool useUniformKvSeqlen_ = false; // 是否使用统一的kvseqlen值（使用maxKvSeqlen_）

    uint64_t sOutSize_ = 0;
    uint64_t dPOutSize_ = 0;
    uint64_t dQOutSize_ = 0;
    uint64_t dKOutSize_ = 0;
    uint64_t dVOutSize_ = 0;
    uint64_t gradSize_ = 0;
    uint32_t maxQSegmentsPerPacket_ = 1; // 单 packet 最大分段数
    uint64_t packetWorkspaceSize_ = 0;   // 所有 AIC core 的 packet workspace 总字节数

    GsagInputLayout layout_ = GsagInputLayout::TND;
    uint32_t kvTotalSeqlen_ = 0;

    uint32_t blockDim_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aiVCRatio = 2;
    uint64_t ubSize_ = 0;
    uint64_t postUbBaseSize_ = 0;
    uint64_t workSpaceSize_ = 0;
    uint64_t libapiSize_ = 0;

    uint64_t dqSize_ = 0;      // dq 元素总量
    uint64_t dkvSize_ = 0;     // dkv dv元素总量
    uint32_t maxQSeqlen_ = 0;  // BNSD格式Q的第三维（S维度）
    uint32_t maxKvSeqlen_ = 0; // BNSD格式KV的第三维（S维度）
    int64_t totalTokensT_ = 0; // TND格式Q的第一维（T维度，总token数）

    ge::DataType dataType_ = ge::DT_FLOAT16;

    GenericBlockSparseAttentionGradTilingData* tilingData_ = nullptr;
};

ge::graphStatus GBSAGTiling::GetNpuInfo(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aiVCRatio = aivNum_ / aicNum_;
    OP_LOGI(context->GetNodeName(), "[GBSAG-meta] GetNpuInfo: aicNum_=%u aivNum_=%u ubSize_=%lu libapiSize_=%lu",
            aicNum_, aivNum_, ubSize_, libapiSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessTND(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    const auto* queryShape = context->GetInputShape(ToIndex(InputId::Query));
    const auto* kvShape = context->GetInputShape(ToIndex(InputId::Key));
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
    kvTotalSeqlen_ = kvShape->GetOriginShape().GetDim(TND_DIM_T);
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(TND_DIM_N));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(TND_DIM_D));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(TND_DIM_N));
    dqSize_ = totalTokensT_ * numHeads_ * headDim_;
    dkvSize_ = kvTotalSeqlen_ * kvHeads_ * headDim_;

    auto cuQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::CuSeqLengthsQ)));
    auto cuKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::CuSeqLengthsKv)));

    // 契约：TND 场景 cuSeq 必传（int64 前缀和数组，长度 b+1）；seqused 不作为 TND 长度来源。
    if (cuQ == nullptr || cuKv == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "cu_seq_lengths_q,cu_seq_lengths_kv",
            "TND layout requires both cumulative INT64 sequence-length arrays [B+1].");
        return ge::GRAPH_FAILED;
    }

    const uint32_t qBatch = static_cast<uint32_t>(cuQ->GetShapeSize() - 1);
    const uint32_t kvBatch = static_cast<uint32_t>(cuKv->GetShapeSize() - 1);
    if (qBatch == 0 || qBatch != kvBatch) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName, "cu_seq_lengths_q,cu_seq_lengths_kv",
            (std::string("elements=") + std::to_string(cuQ->GetShapeSize()) + "," +
             std::to_string(cuKv->GetShapeSize()) + "; batches=" + std::to_string(qBatch) + "," +
             std::to_string(kvBatch))
                .c_str(),
            "Cumulative sequence-length arrays must have equal lengths and describe a positive number of batches.");
        return ge::GRAPH_FAILED;
    }
    batch_ = qBatch;

    useUniformQSeqlen_ = false;
    useUniformKvSeqlen_ = false;

    // 用于静态 workspace/stride 规划，动态调度和实际长度由 metadata/K_OUT 消费。
    const auto* idxShape = context->GetInputShape(ToIndex(InputId::RsvdBlockIdx));
    maxQSeqlen_ = static_cast<uint32_t>(idxShape->GetOriginShape().GetDim(3));
    maxKvSeqlen_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessBNSD(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    const auto* queryShape = context->GetInputShape(ToIndex(InputId::Query));
    const auto* kvShape = context->GetInputShape(ToIndex(InputId::Key));
    auto idxShape = context->GetInputShape(ToIndex(InputId::RsvdBlockIdx));
    auto countShape = context->GetInputShape(ToIndex(InputId::RsvdBlockCount));
    if (queryShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM || idxShape->GetOriginShape().GetDimNum() != 4 ||
        countShape->GetOriginShape().GetDimNum() != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "query,key,sparse_block_idx,sparse_block_count",
            (std::to_string(queryShape->GetOriginShape().GetDimNum()) + "," +
             std::to_string(kvShape->GetOriginShape().GetDimNum()) + "," +
             std::to_string(idxShape->GetOriginShape().GetDimNum()) + "," +
             std::to_string(countShape->GetOriginShape().GetDimNum()))
                .c_str(),
            "BNSD requires query/key/sparse_block_idx rank 4 and sparse_block_count rank 3.");
        return ge::GRAPH_FAILED;
    }
    batch_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_B));
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_N));
    maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_S));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_D));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_N));
    maxKvSeqlen_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_S));
    dqSize_ = static_cast<uint64_t>(batch_) * numHeads_ * maxQSeqlen_ * headDim_;
    dkvSize_ = static_cast<uint64_t>(batch_) * kvHeads_ * maxKvSeqlen_ * headDim_;
    auto cuQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::CuSeqLengthsQ)));
    auto cuKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::CuSeqLengthsKv)));
    if (cuQ != nullptr || cuKv != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "cu_seq_lengths_q,cu_seq_lengths_kv",
            "Cumulative sequence-length arrays are only valid for TND; omit them for BNSD layout.");
        return ge::GRAPH_FAILED;
    }
    // seqused 仅 TND 生效：BNSD 为定长布局，长度一律取物理 S 维，不再读取 seqused。
    const auto* usedQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengths)));
    const auto* usedKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengthsKv)));
    if (usedQ != nullptr || usedKv != nullptr) {
        OP_LOGW(context->GetNodeName(),
                "sequsedQ/sequsedKv only take effect for TND and are ignored for BNSD; must be nullptr.");
    }
    useUniformQSeqlen_ = true;
    useUniformKvSeqlen_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessBSND(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    const auto* queryShape = context->GetInputShape(ToIndex(InputId::Query));
    const auto* kvShape = context->GetInputShape(ToIndex(InputId::Key));
    batch_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_B));
    maxQSeqlen_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_N));
    numHeads_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_S));
    headDim_ = static_cast<uint32_t>(queryShape->GetOriginShape().GetDim(BNSD_DIM_D));
    maxKvSeqlen_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_N));
    kvHeads_ = static_cast<uint32_t>(kvShape->GetOriginShape().GetDim(BNSD_DIM_S));
    dqSize_ = static_cast<uint64_t>(batch_) * maxQSeqlen_ * numHeads_ * headDim_;
    dkvSize_ = static_cast<uint64_t>(batch_) * maxKvSeqlen_ * kvHeads_ * headDim_;
    auto cuQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::CuSeqLengthsQ)));
    auto cuKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::CuSeqLengthsKv)));
    if (cuQ != nullptr || cuKv != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "cu_seq_lengths_q,cu_seq_lengths_kv",
            "Cumulative sequence-length arrays are only valid for TND; omit them for BSND layout.");
        return ge::GRAPH_FAILED;
    }
    // seqused 仅 TND 生效：BSND 为定长布局，长度一律取物理 S 维，不再读取 seqused。
    const auto* usedQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengths)));
    const auto* usedKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengthsKv)));
    if (usedQ != nullptr || usedKv != nullptr) {
        OP_LOGW(context->GetNodeName(),
                "sequsedQ/sequsedKv only take effect for TND and are ignored for BSND; must be nullptr.");
    }
    useUniformQSeqlen_ = true;
    useUniformKvSeqlen_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessAttrs(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    // block_shape 为 def 级 OPTIONAL ListInt 属性（默认 {1, 128}），与前向算子一致；
    // aclnn 路径总是显式传入，图模式缺省时框架会落地 def 默认值。
    const auto* blockShapeArr = context->GetAttrs()->GetListInt(ToIndex(AttrId::BlockShape));
    if (blockShapeArr == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "block_shape",
            "Required block_shape is unavailable; provide [1,Y] or check the operator-definition defaults.");
        return ge::GRAPH_FAILED;
    }
    if (blockShapeArr->GetSize() != 2) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(opName, "block_shape", std::to_string(blockShapeArr->GetSize()).c_str(), "2");
        return ge::GRAPH_FAILED;
    }
    blockShapeX_ = blockShapeArr->GetData()[0];
    blockShapeY_ = blockShapeArr->GetData()[1];

    if (blockShapeX_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "block_shape[0]", std::to_string(blockShapeX_).c_str(), "1");
        return ge::GRAPH_FAILED;
    }
    if (blockShapeY_ < 128 || (blockShapeY_ % 64) != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "block_shape[1]", std::to_string(blockShapeY_).c_str(),
                                              "The block width must be at least 128 and divisible by 64.");
        return ge::GRAPH_FAILED;
    }
    if (layout_ == GsagInputLayout::TND) {
        const auto* countShape = context->GetInputShape(ToIndex(InputId::RsvdBlockCount));
        maxKvSeqlen_ =
            static_cast<uint32_t>(countShape->GetOriginShape().GetDim(2)) * static_cast<uint32_t>(blockShapeY_);
    }
    // 按 blockX 估算单 packet 最大分段数；设备侧使用定长元数据，避免动态内存。
    // packet 总行数始终不超过 Cube 的 M 上限 128。
    const uint32_t segmentRows = std::min<uint32_t>(Q_AGGREGATE_M, static_cast<uint32_t>(blockShapeX_));
    maxQSegmentsPerPacket_ = segmentRows == 0 ? Q_AGGREGATE_MAX_SEGMENTS : CeilDiv(Q_AGGREGATE_M, segmentRows);
    maxQSegmentsPerPacket_ = std::min(maxQSegmentsPerPacket_, Q_AGGREGATE_MAX_SEGMENTS);
    // [方案 A] packet workspace 单核独占 + 三槽软流水缓冲；allocation 在 CalculateWorkSpace
    // 里按 blockDim=aicNum_ 拉满。此处即每核字节数（blockDim 倍之前）。
    packetWorkspaceSize_ = WORKSPACE_BLOCK_NUM * (3 * Q_PACKET_INPUT_BYTES + Q_PACKET_GRAD_BYTES + Q_PACKET_META_BYTES);

    if (context->GetAttrs()->GetAttrPointer<float>(ToIndex(AttrId::ScaleValue)) == nullptr) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(headDim_));
    } else {
        scaleValue_ = *context->GetAttrs()->GetAttrPointer<float>(ToIndex(AttrId::ScaleValue));
    }

    // maskType 仅支持 1（CAUSAL）：缺失或非 1 直接拒绝。
    auto maskTypeAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::MaskType));
    if (maskTypeAttr == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "mask_type", "Required mask_type is unavailable; provide mask_type=1 for this implementation.");
        return ge::GRAPH_FAILED;
    }
    if (*maskTypeAttr != 1) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "mask_type", std::to_string(*maskTypeAttr).c_str(), "1");
        return ge::GRAPH_FAILED;
    }
    maskType_ = static_cast<uint32_t>(*maskTypeAttr);

    const auto* sparseLayout = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::LayoutSparsePattern));
    const auto* residualMode = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::ResidualBlockMode));
    const int64_t layoutSparsePattern = sparseLayout != nullptr ? *sparseLayout : 1;
    const int64_t residualBlockMode = residualMode != nullptr ? *residualMode : 0;
    if (layoutSparsePattern != 1 || residualBlockMode != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "layout_sparse_pattern,residual_block_mode",
            (std::to_string(layoutSparsePattern) + "," + std::to_string(residualBlockMode)).c_str(),
            "This tiling implementation requires layout_sparse_pattern=1 and residual_block_mode=0.");
        return ge::GRAPH_FAILED;
    }
    layoutSparsePattern_ = static_cast<uint32_t>(layoutSparsePattern);
    // Both is_consistent_topk values use the existing per-block count path.

    auto softmaxPrecisionAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::SoftmaxPrecision));
    softmaxPrecision_ = (softmaxPrecisionAttr == nullptr || *softmaxPrecisionAttr == 0) ? 0 : 0;
    if (softmaxPrecisionAttr != nullptr && *softmaxPrecisionAttr != 0) {
        OP_LOGW(context->GetNodeName(), "softmaxPrecision=%ld is unsupported; using 0", *softmaxPrecisionAttr);
    }

    auto winLeftAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::WinLeft));
    auto winRightAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::WinRight));
    winLeft_ = (winLeftAttr == nullptr || *winLeftAttr == -1) ? -1 : -1;
    winRight_ = (winRightAttr == nullptr || *winRightAttr == -1) ? -1 : -1;
    if (winLeftAttr != nullptr && *winLeftAttr != -1) {
        OP_LOGW(context->GetNodeName(), "winLeft=%ld is unsupported; using -1", *winLeftAttr);
    }
    if (winRightAttr != nullptr && *winRightAttr != -1) {
        OP_LOGW(context->GetNodeName(), "winRight=%ld is unsupported; using -1", *winRightAttr);
    }

    auto qInputDesc = context->GetInputDesc(ToIndex(InputId::Query));
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

ge::graphStatus GBSAGTiling::ProcessInput(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    if (context->GetAttrs()->GetAttrPointer<char>(ToIndex(AttrId::QInputLayout)) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "layout_q",
            "Required query input layout is unavailable; provide the layout attribute in the operator definition.");
        return ge::GRAPH_FAILED;
    }

    std::string qLayout(context->GetAttrs()->GetAttrPointer<char>(ToIndex(AttrId::QInputLayout)));
    if (qLayout == "TND") {
        layout_ = GsagInputLayout::TND;
    } else if (qLayout == "BNSD") {
        layout_ = GsagInputLayout::BNSD;
    } else if (qLayout == "BSND") {
        layout_ = GsagInputLayout::BSND;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(opName, "layout_q", qLayout.c_str(),
                                               "Supported query layouts are TND, BNSD and BSND.");
        return ge::GRAPH_FAILED;
    }

    // 核心 Shape 的判空前置，保证安全性
    if (context->GetInputShape(ToIndex(InputId::Query)) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "query");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(ToIndex(InputId::Key)) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "key");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(ToIndex(InputId::RsvdBlockIdx)) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "sparse_block_idx");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(ToIndex(InputId::RsvdBlockCount)) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "sparse_block_count");
        return ge::GRAPH_FAILED;
    }
    const auto* rsvdBlockIdxShape = context->GetInputShape(ToIndex(InputId::RsvdBlockIdx));
    const auto* rsvdBlockCountShape = context->GetInputShape(ToIndex(InputId::RsvdBlockCount));
    if (rsvdBlockIdxShape->GetOriginShape().GetDimNum() != 4 ||
        rsvdBlockCountShape->GetOriginShape().GetDimNum() != 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "sparse_block_idx,sparse_block_count",
            (std::to_string(rsvdBlockIdxShape->GetOriginShape().GetDimNum()) + "," +
             std::to_string(rsvdBlockCountShape->GetOriginShape().GetDimNum()))
                .c_str(),
            "sparse_block_idx must have rank 4 and sparse_block_count must have rank 3.");
        return ge::GRAPH_FAILED;
    }

    // metadata 为必传输入；arch22 ABI 按最大核数固定预留，容量须 >= 198 个 int32。
    const auto* metadataTensor = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::Metadata)));
    if (metadataTensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "metadata",
            "The metadata input is required for the arch22 tiling ABI; provide at least 198 INT32 elements.");
        return ge::GRAPH_FAILED;
    }
    const int64_t metadataElems = metadataTensor->GetShapeSize();
    if (metadataElems < ARC22_METADATA_MIN_ELEMS) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName, "metadata", (std::to_string(metadataElems)).c_str(),
            "Arch22 metadata must contain at least 198 INT32 elements (6 + 64 * 3).");
        return ge::GRAPH_FAILED;
    }

    if (layout_ == GsagInputLayout::TND) {
        if (ProcessTND(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (layout_ == GsagInputLayout::BNSD) {
        if (ProcessBNSD(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (layout_ == GsagInputLayout::BSND) {
        if (ProcessBSND(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // 稀疏张量的 batch/kv-head 维必须与 Q/K 对齐
    const int64_t idxBatchNum = rsvdBlockIdxShape->GetOriginShape().GetDim(0);
    const int64_t idxKvHeadNum = rsvdBlockIdxShape->GetOriginShape().GetDim(1);
    if (idxBatchNum != static_cast<int64_t>(batch_) || idxKvHeadNum != static_cast<int64_t>(kvHeads_)) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            opName, "sparse_block_idx,query,key",
            (std::string("sparse B,N2=[") + std::to_string(idxBatchNum) + "," + std::to_string(idxKvHeadNum) +
             "],query/key B,N2=[" + std::to_string(batch_) + "," + std::to_string(kvHeads_) + "]")
                .c_str(),
            "Sparse block-index batch and key-head dimensions must match query/key batch and key heads.");
        return ge::GRAPH_FAILED;
    }

    return ProcessAttrs(context);
}

ge::graphStatus GBSAGTiling::CalculatePostUbBaseSize(gert::TilingContext* context)
{
    // post 计算：划分空间块数，256 字节对齐
    postUbBaseSize_ = static_cast<uint64_t>(ubSize_ - sizeof(GenericBlockSparseAttentionGradTilingData) - 2 * 1024) /
                      VEC_POST_DIVISION / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::CalculateSoftmaxGradTiling(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    // 安全校验：防止除以 0 导致 Core Dump 崩溃
    if (headDim_ == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName, "query", (std::to_string(headDim_)).c_str(),
            "The query head dimension D must be greater than zero for SoftmaxGrad tiling.");
        return ge::GRAPH_FAILED;
    }

    constexpr static int64_t packetRowsPerAiv = Q_AGGREGATE_M / 2;
    constexpr static uint64_t outputBufferLen = packetRowsPerAiv * ONEBLOCK_FLOAT_NUM * sizeof(float);
    uint64_t tempBufferLen = 40 * 1024 - outputBufferLen;

    auto softmaxGradShape = ge::Shape({packetRowsPerAiv, headDim_});

    // 调用 CANN 底层的 SoftMaxGradTilingFunc
    AscendC::SoftMaxGradTilingFunc(softmaxGradShape, sizeof(float), tempBufferLen, tilingData_->softmaxGradTilingData,
                                   true);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::CalculateTaskSplit(gert::TilingContext* context)
{
    (void)context;
    if (aicNum_ == 0) {
        return ge::GRAPH_FAILED;
    }
    blockDim_ = aicNum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::CalculateWorkSpace(gert::TilingContext* context)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
    if (blockDim_ == 0) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName, (std::string("Workspace tiling requires a positive blockDim; actual=") +
                                               std::to_string(blockDim_) + ", AIC cores=" + std::to_string(aicNum_) +
                                               "; check task splitting and platform resources.")
                                                  .c_str()}));
        return ge::GRAPH_FAILED;
    }

    sOutSize_ = blockDim_ * WORKSPACE_BLOCK_NUM * WORKSPACE_BLOCK_SIZE * sizeof(float);
    dPOutSize_ = blockDim_ * WORKSPACE_BLOCK_NUM * WORKSPACE_BLOCK_SIZE * sizeof(float);
    if (layout_ == GsagInputLayout::TND) {
        dQOutSize_ = totalTokensT_ * numHeads_ * headDim_ * sizeof(float);
        dKOutSize_ = kvTotalSeqlen_ * kvHeads_ * headDim_ * sizeof(float);
        dVOutSize_ = dKOutSize_;
    } else {
        dQOutSize_ = batch_ * numHeads_ * maxQSeqlen_ * headDim_ * sizeof(float);
        dKOutSize_ = batch_ * kvHeads_ * maxKvSeqlen_ * headDim_ * sizeof(float);
        dVOutSize_ = dKOutSize_;
    }
    gradSize_ = 0;

    // 阶段三 packet 缓冲按实际 AIC core 独占并采用双缓冲。
    // 新区域追加在原梯度和 softmax workspace 之后，保持阶段二地址偏移不变。
    packetWorkspaceSize_ *= blockDim_;
    workSpaceSize_ =
        libapiSize_ + sOutSize_ + dPOutSize_ + dQOutSize_ + dKOutSize_ + dVOutSize_ + gradSize_ + packetWorkspaceSize_;
    context->GetWorkspaceSizes(1)[0] = workSpaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::FillTilingData(gert::TilingContext* context)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_headDim(headDim_);
    tilingData_->set_maskType(maskType_);
    tilingData_->set_layoutSparsePattern(layoutSparsePattern_);
    tilingData_->set_softmaxPrecision(softmaxPrecision_);
    tilingData_->set_winLeft(winLeft_);
    tilingData_->set_winRight(winRight_);
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    tilingData_->set_inputLayout(static_cast<uint32_t>(layout_));
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_maxKvSeqlen(maxKvSeqlen_);
    tilingData_->set_useUniformQSeqlen(useUniformQSeqlen_ ? 1 : 0);
    tilingData_->set_useUniformKvSeqlen(useUniformKvSeqlen_ ? 1 : 0);
    tilingData_->set_basicKVBlockSize(BASIC_BLOCK_SIZE);

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
    tilingData_->set_usedVecCoreNum(blockDim_ * aiVCRatio);
    tilingData_->set_dqSize(dqSize_);
    tilingData_->set_dkvSize(dkvSize_);
    tilingData_->set_postUbBaseSize(postUbBaseSize_);
    tilingData_->set_ubSize(ubSize_ - sizeof(GenericBlockSparseAttentionGradTilingData) - 2 * 1024);
    tilingData_->set_packetWorkspaceSize(packetWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

uint64_t GBSAGTiling::GenerateTilingKey() const
{
    uint64_t tilingKey = 0;
    if (dataType_ == ge::DT_FLOAT16) {
        tilingKey = layout_ == GsagInputLayout::TND ?
                        TILING_KEY_100 :
                        (layout_ == GsagInputLayout::BNSD ? TILING_KEY_101 : TILING_KEY_102);
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey = layout_ == GsagInputLayout::TND ?
                        TILING_KEY_110 :
                        (layout_ == GsagInputLayout::BNSD ? TILING_KEY_111 : TILING_KEY_112);
    }
    return tilingKey;
}

ge::graphStatus GBSAGTiling::GetGBSAGTiling(gert::TilingContext* context,
                                            GenericBlockSparseAttentionGradTilingData& tilingData)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
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
                         ", AIC cores=" + std::to_string(aicNum_) + "; a positive AIC core count is required.")
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
                                               "; check the GenericBlockSparseAttentionGrad tiling-data object.")
                                                  .c_str()}));
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::SetTilingData(gert::TilingContext* context,
                                           GenericBlockSparseAttentionGradTilingData& tilingData)
{
    const char* opName = context->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context->GetNodeName();
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

class GenericBlockSparseAttentionGradArch22Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GenericBlockSparseAttentionGradArch22Tiling(gert::TilingContext* context)
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
        OP_CHECK_IF(context_ == nullptr,
                    REPORT_PREDEFINED_ERR_MSG(
                        "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                        (std::vector<const char*>{"GenericBlockSparseAttentionGrad",
                                                  "Tiling context is null; check GenericBlockSparseAttentionGrad "
                                                  "Tiling invocation and registration."})),
                    return ge::GRAPH_FAILED);
        if (tiling_.GetGBSAGTiling(context_, tilingData_) != ge::GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(context_, "GetGBSAGTiling failed; retain its first error.");
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
    GBSAGTiling tiling_;
    GenericBlockSparseAttentionGradTilingData tilingData_;
    uint64_t tilingKey_ = 0;
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(GenericBlockSparseAttentionGrad, GenericBlockSparseAttentionGradArch22Tiling,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 1);

} // namespace optiling
