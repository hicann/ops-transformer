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
    IsPackedGqa = 1,
    QInputLayout = 2,
    KvInputLayout = 3,
    ScaleValue = 4,
    MaskType = 5,
    SoftmaxPrecision = 6,
    WinLeft = 7,
    WinRight = 8,
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
// 三槽软流水：单槽 = 「FP32 S/dP tile + 低精度 P/dS tile」成对预留（128*128*2 元素）；
// 每核槽数 = 3（Scatter 延后 2 拍 + 预取 1 拍），须与 kernel 侧 WORKSPACE_SOFTPIPE_SLOT_NUM 一致。
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

// 可选 aclTensor 的 shape-only presence 归一化：op_api 对「未传入」的可选输入会分配空的
// 占位 tensor（AllocTensor 无 shape，dim_num==0 的标量），GetOptionalInputTensor 仍返回
// 非 null，且标量 GetShapeSize()==1（空乘积）。因此不能用 GetShapeSize() 判定 presence，
// 这里按仓库通行口径（causal_conv1d / grouped_matmul / moe_finalize_routing）用 rank 判定：
// dim_num==0 或 [0] 视为缺省。绝不读 device 数据值。
static inline const gert::Tensor *NormalizeOptionalTensor(const gert::Tensor *tensor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    const auto &shape = tensor->GetStorageShape();
    const size_t dimNum = shape.GetDimNum();
    const bool absent = (dimNum == 0) || (dimNum == 1 && shape.GetDim(0) <= 0);
    return absent ? nullptr : tensor;
}

// Tiling类
class GBSAGTiling {
public:
    GBSAGTiling() = default;
    ~GBSAGTiling() = default;

    ge::graphStatus GetGBSAGTiling(gert::TilingContext *context, GenericBlockSparseAttentionGradTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context, GenericBlockSparseAttentionGradTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ProcessInput(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);

    ge::graphStatus CalculatePostUbBaseSize(gert::TilingContext *context);
    ge::graphStatus CalculateSoftmaxGradTiling(gert::TilingContext *context);

    ge::graphStatus ProcessTND(gert::TilingContext *context);
    ge::graphStatus ProcessBNSD(gert::TilingContext *context);
    ge::graphStatus ProcessBSND(gert::TilingContext *context);

    ge::graphStatus ProcessAttrs(gert::TilingContext *context);

    uint64_t GenerateTilingKey() const;

    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t headDim_ = 0;
    int64_t blockShapeX_ = 0; // block的x维度
    int64_t blockShapeY_ = 0; // block的y维度
    float scaleValue_ = 0.0f;
    uint32_t maskType_ = 0;
    uint32_t isPackedGQA_ = 1;
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

    GenericBlockSparseAttentionGradTilingData *tilingData_ = nullptr;
};

ge::graphStatus GBSAGTiling::GetNpuInfo(gert::TilingContext *context)
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

ge::graphStatus GBSAGTiling::ProcessTND(gert::TilingContext *context)
{
    const auto *queryShape = context->GetInputShape(ToIndex(InputId::Query));
    const auto *kvShape = context->GetInputShape(ToIndex(InputId::Key));
    if (queryShape->GetOriginShape().GetDimNum() != TND_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "TND format must have 3 dimensions");
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
        OP_LOGE(context->GetNodeName(), "TND requires both cuSeqLengthsQ and cuSeqLengthsKv to be provided");
        return ge::GRAPH_FAILED;
    }

    const uint32_t qBatch = static_cast<uint32_t>(cuQ->GetShapeSize() - 1);
    const uint32_t kvBatch = static_cast<uint32_t>(cuKv->GetShapeSize() - 1);
    if (qBatch == 0 || qBatch != kvBatch) {
        OP_LOGE(context->GetNodeName(), "Q/KV sequence metadata batch mismatch: q=%u kv=%u", qBatch, kvBatch);
        return ge::GRAPH_FAILED;
    }
    batch_ = qBatch;

    useUniformQSeqlen_ = false;
    useUniformKvSeqlen_ = false;

    // 迭代 2 不在主算子 Host 侧读取动态长度 tensor 的值。最大物理维度只
    // 用于静态 workspace/stride 规划，动态调度和实际长度由 metadata/K_OUT 消费。
    const auto *idxShape = context->GetInputShape(ToIndex(InputId::RsvdBlockIdx));
    maxQSeqlen_ = static_cast<uint32_t>(idxShape->GetOriginShape().GetDim(3));
    maxKvSeqlen_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessBNSD(gert::TilingContext *context)
{
    const auto *queryShape = context->GetInputShape(ToIndex(InputId::Query));
    const auto *kvShape = context->GetInputShape(ToIndex(InputId::Key));
    auto idxShape = context->GetInputShape(ToIndex(InputId::RsvdBlockIdx));
    auto countShape = context->GetInputShape(ToIndex(InputId::RsvdBlockCount));
    if (queryShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM ||
        kvShape->GetOriginShape().GetDimNum() != BNSD_DIM_NUM || idxShape->GetOriginShape().GetDimNum() != 4 ||
        countShape->GetOriginShape().GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "BNSD requires rank-4 inputs and rank-4/rank-3 sparse tensors");
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
        OP_LOGE(context->GetNodeName(), "cuSeqLengths are only valid for TND");
        return ge::GRAPH_FAILED;
    }
    auto usedQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengths)));
    auto usedKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengthsKv)));
    useUniformQSeqlen_ = usedQ == nullptr;
    useUniformKvSeqlen_ = usedKv == nullptr;
    if (!useUniformQSeqlen_) {
        if (usedQ->GetShapeSize() != batch_) {
            OP_LOGE(context->GetNodeName(), "sequsedQ must contain one value per batch for BNSD");
            return ge::GRAPH_FAILED;
        }
    }
    if (!useUniformKvSeqlen_) {
        if (usedKv->GetShapeSize() != batch_) {
            OP_LOGE(context->GetNodeName(), "sequsedKv must contain one value per batch for BNSD");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessBSND(gert::TilingContext *context)
{
    const auto *queryShape = context->GetInputShape(ToIndex(InputId::Query));
    const auto *kvShape = context->GetInputShape(ToIndex(InputId::Key));
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
        OP_LOGE(context->GetNodeName(), "cuSeqLengths are only valid for TND");
        return ge::GRAPH_FAILED;
    }
    auto usedQ = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengths)));
    auto usedKv = NormalizeOptionalTensor(context->GetOptionalInputTensor(ToIndex(InputId::ActualSeqLengthsKv)));
    useUniformQSeqlen_ = usedQ == nullptr;
    useUniformKvSeqlen_ = usedKv == nullptr;
    if (!useUniformQSeqlen_) {
        if (usedQ->GetShapeSize() != batch_) {
            OP_LOGE(context->GetNodeName(), "sequsedQ must contain one value per batch for BSND");
            return ge::GRAPH_FAILED;
        }
    }
    if (!useUniformKvSeqlen_) {
        if (usedKv->GetShapeSize() != batch_) {
            OP_LOGE(context->GetNodeName(), "sequsedKv must contain one value per batch for BSND");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessAttrs(gert::TilingContext *context)
{
    // block_shape 为 def 级 OPTIONAL ListInt 属性（默认 {1, 128}），与前向算子一致；
    // aclnn 路径总是显式传入，图模式缺省时框架会落地 def 默认值。
    const auto *blockShapeArr = context->GetAttrs()->GetListInt(ToIndex(AttrId::BlockShape));
    if (blockShapeArr == nullptr) {
        OP_LOGE(context->GetNodeName(), "block_shape attr is missing");
        return ge::GRAPH_FAILED;
    }
    if (blockShapeArr->GetSize() != 2) {
        OP_LOGE(context->GetNodeName(), "block_shape must contain two elements [x, y], got size %zu.",
                blockShapeArr->GetSize());
        return ge::GRAPH_FAILED;
    }
    blockShapeX_ = blockShapeArr->GetData()[0];
    blockShapeY_ = blockShapeArr->GetData()[1];
    if (blockShapeX_ != 1) {
        OP_LOGW(context->GetNodeName(), "blockShapeX=%ld is unsupported; using 1", blockShapeX_);
        blockShapeX_ = 1;
    }
    if (blockShapeY_ < 128 || (blockShapeY_ % 64) != 0) {
        int64_t normalizedY = std::max<int64_t>(128, blockShapeY_);
        normalizedY = ((normalizedY + 63) / 64) * 64;
        OP_LOGW(context->GetNodeName(), "blockShapeY=%ld is unsupported; using %ld", blockShapeY_, normalizedY);
        blockShapeY_ = normalizedY;
    }
    if (layout_ == GsagInputLayout::TND) {
        const auto *countShape = context->GetInputShape(ToIndex(InputId::RsvdBlockCount));
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

    auto maskTypeAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::MaskType));
    if (maskTypeAttr == nullptr) {
        maskType_ = 0;
    } else if (*maskTypeAttr == 0 || *maskTypeAttr == 1) {
        maskType_ = static_cast<uint32_t>(*maskTypeAttr);
    } else {
        OP_LOGW(context->GetNodeName(), "maskType=%ld is unsupported; using 0", *maskTypeAttr);
        maskType_ = 0;
    }

    auto packedGqaAttr = context->GetAttrs()->GetAttrPointer<int64_t>(ToIndex(AttrId::IsPackedGqa));
    isPackedGQA_ = (packedGqaAttr == nullptr || *packedGqaAttr == 1) ? 1 : 1;
    if (packedGqaAttr != nullptr && *packedGqaAttr != 1) {
        OP_LOGW(context->GetNodeName(), "isPackedGQA=%ld is unsupported; using 1", *packedGqaAttr);
    }

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
        OP_LOGE(context->GetNodeName(), "Query inputDesc is null");
        return ge::GRAPH_FAILED;
    } else {
        dataType_ = qInputDesc->GetDataType();
    }

    if (kvHeads_ == 0) {
        OP_LOGE(context->GetNodeName(), "kvHeads can not be zero.");
        return ge::GRAPH_FAILED;
    }

    if (!(numHeads_ >= kvHeads_ && numHeads_ % kvHeads_ == 0)) {
        OP_LOGE(context->GetNodeName(),
                "Invalid head config: query heads(%u) must be >=kv heads(%u) and divisible by it.", numHeads_,
                kvHeads_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::ProcessInput(gert::TilingContext *context)
{
    if (context->GetAttrs()->GetAttrPointer<char>(ToIndex(AttrId::QInputLayout)) == nullptr) {
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
        OP_LOGE(context->GetNodeName(), "Unsupported layout: %s. Supported formats: TND, BNSD, BSND",
                context->GetAttrs()->GetAttrPointer<char>(ToIndex(AttrId::QInputLayout)));
        return ge::GRAPH_FAILED;
    }

    // 核心 Shape 的判空前置，保证安全性
    if (context->GetInputShape(ToIndex(InputId::Query)) == nullptr) {
        OP_LOGE(context->GetNodeName(), "Query shape is null");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(ToIndex(InputId::Key)) == nullptr) {
        OP_LOGE(context->GetNodeName(), "KV shape is null");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(ToIndex(InputId::RsvdBlockIdx)) == nullptr) {
        OP_LOGE(context->GetNodeName(), "rsvdBlockIdx shape is null");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(ToIndex(InputId::RsvdBlockCount)) == nullptr) {
        OP_LOGE(context->GetNodeName(), "rsvdBlockCount shape is null");
        return ge::GRAPH_FAILED;
    }
    const auto *rsvdBlockIdxShape = context->GetInputShape(ToIndex(InputId::RsvdBlockIdx));
    const auto *rsvdBlockCountShape = context->GetInputShape(ToIndex(InputId::RsvdBlockCount));
    if (rsvdBlockIdxShape->GetOriginShape().GetDimNum() != 4 ||
        rsvdBlockCountShape->GetOriginShape().GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "rsvdBlockIdx/count must be rank 4/3");
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

    return ProcessAttrs(context);
}

ge::graphStatus GBSAGTiling::CalculatePostUbBaseSize(gert::TilingContext *context)
{
    // post 计算：划分空间块数，256 字节对齐
    postUbBaseSize_ = static_cast<uint64_t>(ubSize_ - sizeof(GenericBlockSparseAttentionGradTilingData) - 2 * 1024) /
                      VEC_POST_DIVISION / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::CalculateSoftmaxGradTiling(gert::TilingContext *context)
{
    // 安全校验：防止除以 0 导致 Core Dump 崩溃
    if (headDim_ == 0) {
        OP_LOGE(context->GetNodeName(), "CalculateSoftmaxGradTiling failed: headDim_ is 0");
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

ge::graphStatus GBSAGTiling::CalculateTaskSplit(gert::TilingContext *context)
{
    (void)context;
    if (aicNum_ == 0) {
        return ge::GRAPH_FAILED;
    }
    blockDim_ = aicNum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::CalculateWorkSpace(gert::TilingContext *context)
{
    if (blockDim_ == 0) {
        OP_LOGE(context->GetNodeName(), "blockDim is 0");
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

ge::graphStatus GBSAGTiling::FillTilingData(gert::TilingContext *context)
{
    if (tilingData_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_headDim(headDim_);
    tilingData_->set_maskType(maskType_);
    tilingData_->set_isPackedGQA(isPackedGQA_);
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

ge::graphStatus GBSAGTiling::GetGBSAGTiling(gert::TilingContext *context,
                                            GenericBlockSparseAttentionGradTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }

    ret = ProcessInput(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "ProcessInput failed");
        return ret;
    }

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateTaskSplit failed");
        return ret;
    }

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateWorkSpace failed");
        return ret;
    }

    ret = CalculatePostUbBaseSize(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculatePostUbBaseSize failed");
        return ret;
    }

    ret = CalculateSoftmaxGradTiling(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateSoftmaxGradTiling failed");
        return ret;
    }

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "FillTilingData failed");
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSAGTiling::SetTilingData(gert::TilingContext *context,
                                           GenericBlockSparseAttentionGradTilingData &tilingData)
{
    OP_CHECK_IF(
        context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttentionGrad", "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

class GenericBlockSparseAttentionGradArch22Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GenericBlockSparseAttentionGradArch22Tiling(gert::TilingContext *context)
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
                    OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttentionGrad", "Context is nullptr."),
                    return ge::GRAPH_FAILED);
        if (tiling_.GetGBSAGTiling(context_, tilingData_) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "GetGBSAGTiling failed");
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
