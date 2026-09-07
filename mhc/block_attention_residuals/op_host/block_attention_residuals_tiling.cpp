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
 * \file block_attention_residuals_tiling.cpp
 * \brief BlockAttentionResiduals host tiling — CanResidentAllBlocks → RESIDENT / RELOAD；无 FP32-v WS
 */
#include "block_attention_residuals_tiling.h"

#include <algorithm>

#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(BlockAttentionResiduals, BlockAttentionResidualsTiling, 0);

constexpr size_t PARTIAL_BLOCK_INDEX = 0;
constexpr size_t BLOCK_RES_INDEX = 1;
constexpr size_t PROJ_WEIGHT_INDEX = 2;
constexpr size_t NORM_WEIGHT_INDEX = 3;
constexpr size_t HIDDEN_STATES_INDEX = 0;
constexpr size_t INV_NORM_INDEX = 1;
constexpr size_t PROBS_INDEX = 2;

constexpr size_t PARTIAL_BLOCK_DIM = 2;
constexpr size_t BLOCK_RES_DIM = 3;

constexpr size_t ATTR_VALID_BLOCK_NUM_INDEX = 0;
constexpr size_t ATTR_NORM_EPS_INDEX = 1;
constexpr size_t ATTR_NEED_BACKWARD_INDEX = 2;

constexpr uint32_t UB_AVAIL_BYTES = 192U * 1024U; // DAV_2201 按 192KB 估算
// TPipe/对齐/Que 头开销；过小会导致大 H RESIDENT 运行时 507035
constexpr uint32_t UB_OVERHEAD_BYTES = 32U * 1024U;
constexpr uint32_t ELEM_PER_BLK_BF16 = 16U;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8U;
constexpr uint32_t SCALAR_LOCAL_ELEMS = 8U;
constexpr uint32_t MAX_NUM_BLOCKS = 100U;
constexpr uint32_t STAGING_ALIGN_BYTES = 512U;

static inline uint32_t AlignUpU32(uint32_t val, uint32_t align)
{
    if (align == 0) {
        return val;
    }
    return (val + align - 1U) / align * align;
}

static inline uint32_t AlignDownU32(uint32_t val, uint32_t align)
{
    if (align == 0) {
        return val;
    }
    return (val / align) * align;
}

static inline uint64_t AlignUpU64(uint64_t val, uint64_t align)
{
    if (align == 0) {
        return val;
    }
    return (val + align - 1UL) / align * align;
}

void BlockAttentionResidualsTiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
    tilingData_.usedCoreNum = static_cast<uint32_t>(compileInfo_.aivNum);
}

ge::graphStatus BlockAttentionResidualsTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetValidBlockNum() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid valid_block_num attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetNormEps() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid norm_eps attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetNeedBackward() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid need_backward attr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint32_t BlockAttentionResidualsTiling::GetMinStagingBytes() const
{
    // inv/probs 均经 Que 逐 token 直写 GM，不再占用 staging
    return 0;
}

uint32_t BlockAttentionResidualsTiling::CalcMaxResidentRows(int64_t hiddenSize) const
{
    // UB ≈ B*H*2 + 4*H*4(scoreWeight/vRow/out/brc) + B*4 + 2*H*2(Que) + overhead (+ minStaging)
    // N_ub_max = floor( (UB - 4*H*4 - 2*H*2 - overhead - minStaging) / (H*2 + 4) )
    // 保持保守估计，避免误选 RESIDENT（大 H 下 B 稍大即可能 UB 打满/精度异常）
    // fp16/fp32 输入按元素字节数等比放大（fp32 为 2 倍）
    if (hiddenSize <= 0) {
        return 0;
    }
    const uint32_t elemBytes = DtypeElemBytes();
    const uint64_t h = static_cast<uint64_t>(hiddenSize);
    const uint64_t workBytes = static_cast<uint64_t>(4U) * h * sizeof(float) +
                               static_cast<uint64_t>(2U) * h * elemBytes + UB_OVERHEAD_BYTES +
                               static_cast<uint64_t>(GetMinStagingBytes());
    if (workBytes >= UB_AVAIL_BYTES) {
        return 0;
    }
    const uint64_t remain = UB_AVAIL_BYTES - workBytes;
    const uint64_t perRow = h * elemBytes + sizeof(float);
    return static_cast<uint32_t>(remain / perRow);
}

bool BlockAttentionResidualsTiling::CanResidentAllBlocks(int64_t blockCount, int64_t hiddenSize) const
{
    return blockCount > 0 && blockCount <= static_cast<int64_t>(CalcMaxResidentRows(hiddenSize));
}

int64_t BlockAttentionResidualsTiling::CalcHSliceChunk() const
{
    // TK-HSLICE 的 h 切块大小：取满足切块 UB 预算的最大 chunk。
    // 切块缓冲：inQue 2×chunk×elemBytes + outQue 1×chunk×elemBytes + 3 个 fp32 buf ×chunk×4B；
    // 元数据：vecMeta+metaSoftmax+sumSq+sumSqComp+score+scoreComp(各 metaAlign×4B)
    //         + metaBrc(metaAlign×8×4B) + scalar + 全局开销。
    const int64_t B = tilingData_.blockCount;
    const int64_t H = tilingData_.hiddenSize;
    if (B <= 0 || H <= 0) {
        return 0;
    }
    const uint32_t elemBytes = DtypeElemBytes();
    const uint32_t elemsPerBlk = DtypeElemsPerBlk();
    const uint32_t metaAlign = AlignUpU32(static_cast<uint32_t>(B), ELEM_PER_BLK_FP32);
    constexpr uint32_t META_FLOAT_PER_ALIGN =
        1U + 1U + 8U + 2U + 2U; // vecMeta + metaSoftmax + metaBrc + Kahan(sumSq, score)
    uint64_t metaBytes = static_cast<uint64_t>(metaAlign) * META_FLOAT_PER_ALIGN * sizeof(float) +
                         static_cast<uint64_t>(SCALAR_LOCAL_ELEMS) * sizeof(float) + UB_OVERHEAD_BYTES;
    constexpr uint32_t BACKWARD_QUE_NUM = 2U; // invQue + probsQue
    if (tilingData_.needBackward != 0) {
        metaBytes += static_cast<uint64_t>(BACKWARD_QUE_NUM) * metaAlign * sizeof(float);
    }
    if (metaBytes >= UB_AVAIL_BYTES) {
        return H; // 保护：退化为整行单块（此时 kernel 按 hiddenSizeChunk=H 处理）
    }
    const uint64_t remain = UB_AVAIL_BYTES - metaBytes;
    constexpr uint64_t FP32_COMPUTE_PER_ELEM = 3U * sizeof(float);        // scoreWeight + vRow + outFp32
    const uint64_t perElemBytes = 3U * elemBytes + FP32_COMPUTE_PER_ELEM; // inQue 2× + outQue 1×
    uint32_t chunk = static_cast<uint32_t>(remain / perElemBytes);
    chunk = AlignDownU32(chunk, elemsPerBlk);
    if (chunk < elemsPerBlk) {
        chunk = elemsPerBlk;
    }
    if (static_cast<int64_t>(chunk) > H) {
        return H;
    }
    return static_cast<int64_t>(chunk);
}

uint64_t BlockAttentionResidualsTiling::EstimateUbComputeBytes(bool resident) const
{
    const uint64_t H = static_cast<uint64_t>(tilingData_.hiddenSize);
    const uint64_t B = static_cast<uint64_t>(tilingData_.blockCount);
    const uint64_t HAlignBf16 = AlignUpU64(H, ELEM_PER_BLK_BF16);
    const uint64_t HAlignFp32 = AlignUpU64(H, ELEM_PER_BLK_FP32);
    const uint32_t elemBytes = DtypeElemBytes();
    // 与 arch22 / kernel 一致：meta 按 32B block（8 fp32）对齐
    const uint32_t metaAlign = AlignUpU32(static_cast<uint32_t>(B), ELEM_PER_BLK_FP32);

    uint64_t ub = 0;
    if (resident) {
        ub += static_cast<uint64_t>(1) * HAlignBf16 * elemBytes; // inQue
        ub += static_cast<uint64_t>(1) * HAlignBf16 * elemBytes; // outQue
        ub += static_cast<uint64_t>(B) * HAlignBf16 * elemBytes; // vBf16 resident
    } else {
        ub += static_cast<uint64_t>(2) * HAlignBf16 * elemBytes; // inQue BUFFER_NUM=2
        ub += static_cast<uint64_t>(1) * HAlignBf16 * elemBytes; // outQue
    }
    ub += static_cast<uint64_t>(3) * HAlignFp32 * sizeof(float);                // scoreWeight + vRow + outFp32
    ub += static_cast<uint64_t>(metaAlign) * sizeof(float);                     // vecMeta
    ub += static_cast<uint64_t>(metaAlign) * sizeof(float);                     // metaSoftmax
    ub += static_cast<uint64_t>(metaAlign) * ELEM_PER_BLK_FP32 * sizeof(float); // metaBrc[n*8] after Softmax Brcb
    ub += static_cast<uint64_t>(SCALAR_LOCAL_ELEMS) * sizeof(float);
    if (tilingData_.needBackward != 0) {
        ub += static_cast<uint64_t>(ELEM_PER_BLK_FP32) * sizeof(float); // invQue_ 1 block
        ub += static_cast<uint64_t>(metaAlign) * sizeof(float);         // probsQue_ AlignUp(B, 8)
    }
    ub += UB_OVERHEAD_BYTES;
    return ub;
}

ge::graphStatus BlockAttentionResidualsTiling::FillStagingFields()
{
    // 无 staging：inv/probs 均 Que 直写
    tilingData_.stagingBytes = 0;
    tilingData_.tokensPerFlush = 0;
    tilingData_.elemsPerToken = (tilingData_.needBackward != 0) ? static_cast<uint32_t>(tilingData_.blockCount) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::DoOpTiling()
{
    if (tilingData_.numTokens == 0 || tilingData_.hiddenSize == 0) {
        tilingData_.tokensPerCore = 0;
        tilingData_.blockCount = tilingData_.numBlocks + 1;
        tilingData_.stagingBytes = 0;
        tilingData_.tokensPerFlush = 0;
        tilingData_.elemsPerToken = 0;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t coreNum = std::min(tilingData_.numTokens, static_cast<int64_t>(tilingData_.usedCoreNum));
    tilingData_.usedCoreNum = static_cast<uint32_t>(coreNum);
    tilingData_.tokensPerCore = (coreNum == 0) ? 0 : (tilingData_.numTokens + coreNum - 1) / coreNum;
    tilingData_.blockCount = tilingData_.numBlocks + 1;
    tilingData_.invHiddenSize =
        (tilingData_.hiddenSize > 0) ? (1.0f / static_cast<float>(tilingData_.hiddenSize)) : 0.0f;
    tilingData_.wsSizePerToken = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::DoLibApiTiling()
{
    const int64_t B = tilingData_.blockCount;
    const int64_t H = tilingData_.hiddenSize;
    // dtype 不编码进 key：框架按 dtype 编译独立 kernel 二进制，key 只区分算法分支。
    // 此处仍按运行时 dtype 估算 UB，保证 RELOAD/RESIDENT/HSLICE 选择对 fp16/fp32 成立。
    uint64_t variantKey = TILING_KEY_RELOAD;
    if (CanResidentAllBlocks(B, H)) {
        variantKey = TILING_KEY_RESIDENT;
        tilingData_.hiddenSizeChunk = 0;
        OP_LOGI(context_->GetNodeName(), "Select TK-RESIDENT (%lu): B=%ld H=%ld N_ub_max=%u needBackward=%u",
                static_cast<unsigned long>(variantKey), B, H, CalcMaxResidentRows(H), tilingData_.needBackward);
    } else if (EstimateUbComputeBytes(false) <= UB_AVAIL_BYTES) {
        variantKey = TILING_KEY_RELOAD;
        tilingData_.hiddenSizeChunk = 0;
        OP_LOGI(context_->GetNodeName(), "Select TK-RELOAD (%lu): B=%ld H=%ld N_ub_max=%u needBackward=%u",
                static_cast<unsigned long>(variantKey), B, H, CalcMaxResidentRows(H), tilingData_.needBackward);
    } else {
        variantKey = TILING_KEY_HSLICE;
        tilingData_.hiddenSizeChunk = CalcHSliceChunk();
        OP_LOGI(context_->GetNodeName(), "Select TK-HSLICE (%lu): B=%ld H=%ld chunk=%ld N_ub_max=%u needBackward=%u",
                static_cast<unsigned long>(variantKey), B, H, tilingData_.hiddenSizeChunk, CalcMaxResidentRows(H),
                tilingData_.needBackward);
    }
    tilingKey_ = variantKey;
    OP_CHECK_IF(FillStagingFields() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "FillStagingFields failed"),
                return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttentionResidualsTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus BlockAttentionResidualsTiling::GetWorkspaceSize()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    workspaceSize_ = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.usedCoreNum);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());

    const auto tilingDataSize = sizeof(BlockAttentionResiduals::BlockAttentionResidualsTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity %zu < size %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::CheckContext()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(PARTIAL_BLOCK_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(PARTIAL_BLOCK_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(BLOCK_RES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(BLOCK_RES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(PROJ_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(PROJ_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(NORM_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(NORM_WEIGHT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(HIDDEN_STATES_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(HIDDEN_STATES_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::AnalyzeDtype()
{
    auto partialBlockDtype = context_->GetInputDesc(PARTIAL_BLOCK_INDEX)->GetDataType();
    auto blockDtype = context_->GetInputDesc(BLOCK_RES_INDEX)->GetDataType();
    auto projDtype = context_->GetInputDesc(PROJ_WEIGHT_INDEX)->GetDataType();
    auto normDtype = context_->GetInputDesc(NORM_WEIGHT_INDEX)->GetDataType();
    auto outDtype = context_->GetOutputDesc(HIDDEN_STATES_INDEX)->GetDataType();

    const auto dtypeOk = [](ge::DataType dt) {
        return dt == ge::DT_BF16 || dt == ge::DT_FLOAT16 || dt == ge::DT_FLOAT;
    };
    OP_CHECK_IF(!dtypeOk(partialBlockDtype),
                OP_LOGE(context_->GetNodeName(), "partial_block dtype must be BF16/FP16/FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!dtypeOk(blockDtype), OP_LOGE(context_->GetNodeName(), "block_res dtype must be BF16/FP16/FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!dtypeOk(projDtype), OP_LOGE(context_->GetNodeName(), "proj_weight dtype must be BF16/FP16/FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!dtypeOk(normDtype), OP_LOGE(context_->GetNodeName(), "norm_weight dtype must be BF16/FP16/FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!dtypeOk(outDtype), OP_LOGE(context_->GetNodeName(), "hidden_states dtype must be BF16/FP16/FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(partialBlockDtype != blockDtype || partialBlockDtype != projDtype || partialBlockDtype != normDtype ||
                    partialBlockDtype != outDtype,
                OP_LOGE(context_->GetNodeName(), "all inputs and hidden_states must share the same dtype"),
                return ge::GRAPH_FAILED);
    inputDtype_ = partialBlockDtype;
    return ge::GRAPH_SUCCESS;
}

uint32_t BlockAttentionResidualsTiling::DtypeElemBytes() const
{
    return (inputDtype_ == ge::DT_FLOAT) ? sizeof(float) : sizeof(uint16_t);
}

uint32_t BlockAttentionResidualsTiling::DtypeElemsPerBlk() const
{
    // 32B block 可容纳的元素数：16bit → 16，fp32 → 8
    return (inputDtype_ == ge::DT_FLOAT) ? ELEM_PER_BLK_FP32 : ELEM_PER_BLK_BF16;
}

ge::graphStatus BlockAttentionResidualsTiling::AnalyzeShapes()
{
    const auto &partialBlockShape = context_->GetInputShape(PARTIAL_BLOCK_INDEX)->GetOriginShape();
    const auto &blockShape = context_->GetInputShape(BLOCK_RES_INDEX)->GetOriginShape();
    const auto &projShape = context_->GetInputShape(PROJ_WEIGHT_INDEX)->GetOriginShape();
    const auto &normShape = context_->GetInputShape(NORM_WEIGHT_INDEX)->GetOriginShape();

    OP_CHECK_IF(partialBlockShape.GetDimNum() != PARTIAL_BLOCK_DIM,
                OP_LOGE(context_->GetNodeName(), "partial_block dim num must be %zu", PARTIAL_BLOCK_DIM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape.GetDimNum() != BLOCK_RES_DIM,
                OP_LOGE(context_->GetNodeName(), "block_res dim num must be %zu", BLOCK_RES_DIM),
                return ge::GRAPH_FAILED);

    const int64_t numTokens = partialBlockShape.GetDim(0);
    const int64_t hiddenSize = partialBlockShape.GetDim(1);
    const int64_t numBlocks = blockShape.GetDim(1);
    OP_CHECK_IF(numTokens < 0 || hiddenSize <= 0 || numBlocks < 1 || numBlocks > static_cast<int64_t>(MAX_NUM_BLOCKS),
                OP_LOGE(context_->GetNodeName(), "shape range requires T>=0, H>=1, and 1<=N<=%u", MAX_NUM_BLOCKS),
                return ge::GRAPH_FAILED);

    tilingData_.numTokens = numTokens;
    tilingData_.hiddenSize = hiddenSize;
    tilingData_.numBlocks = numBlocks;

    OP_CHECK_IF(blockShape.GetDim(0) != tilingData_.numTokens,
                OP_LOGE(context_->GetNodeName(), "T mismatch between partial_block and block_res"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape.GetDim(2) != tilingData_.hiddenSize,
                OP_LOGE(context_->GetNodeName(), "H mismatch between partial_block and block_res"),
                return ge::GRAPH_FAILED);
    const bool projShapeOk =
        (projShape.GetDimNum() == 1 && projShape.GetDim(0) == hiddenSize) ||
        (projShape.GetDimNum() == 2 && projShape.GetDim(0) == 1 && projShape.GetDim(1) == hiddenSize);
    OP_CHECK_IF(!projShapeOk, OP_LOGE(context_->GetNodeName(), "proj_weight shape must be [H] or [1,H]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape.GetDimNum() != 1 || normShape.GetDim(0) != tilingData_.hiddenSize,
                OP_LOGE(context_->GetNodeName(), "norm_weight shape must be [H]"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::GetValidBlockNum()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const int64_t *validBlockNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_VALID_BLOCK_NUM_INDEX);
    int64_t validBlockNum = (validBlockNumPtr != nullptr) ? *validBlockNumPtr : -1;
    const int64_t inputValidBlockNum = validBlockNum;
    if (validBlockNum == -1) {
        validBlockNum = tilingData_.numBlocks;
    }
    OP_CHECK_IF(
        validBlockNum != tilingData_.numBlocks,
        OP_LOGE(context_->GetNodeName(), "valid_block_num must be -1 (default, use N) or equal to N(%ld), got %ld",
                tilingData_.numBlocks, inputValidBlockNum),
        return ge::GRAPH_FAILED);
    tilingData_.validBlockNum = validBlockNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::GetNormEps()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float *normEpsPtr = attrs->GetAttrPointer<float>(ATTR_NORM_EPS_INDEX);
    tilingData_.normEps = (normEpsPtr != nullptr) ? *normEpsPtr : 1e-6F;
    OP_CHECK_IF(tilingData_.normEps <= 0.0f, OP_LOGE(context_->GetNodeName(), "norm_eps must be > 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttentionResidualsTiling::GetNeedBackward()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const bool *needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD_INDEX);
    const bool needBackward = (needBackwardPtr != nullptr) ? *needBackwardPtr : false;
    tilingData_.needBackward = needBackward ? 1U : 0U;

    if (!needBackward) {
        return ge::GRAPH_SUCCESS;
    }

    auto invNormDesc = context_->GetOutputDesc(INV_NORM_INDEX);
    auto probsDesc = context_->GetOutputDesc(PROBS_INDEX);
    auto invNormShape = context_->GetOutputShape(INV_NORM_INDEX);
    auto probsShape = context_->GetOutputShape(PROBS_INDEX);
    OP_CHECK_IF(invNormDesc == nullptr || probsDesc == nullptr || invNormShape == nullptr || probsShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "need_backward=true requires inv_norm and probs outputs"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(invNormDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "inv_norm dtype must be FLOAT"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(probsDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(context_->GetNodeName(), "probs dtype must be FLOAT"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void BlockAttentionResidualsTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "numTokens=%ld numBlocks=%ld hiddenSize=%ld tokensPerCore=%ld usedCoreNum=%u",
            tilingData_.numTokens, tilingData_.numBlocks, tilingData_.hiddenSize, tilingData_.tokensPerCore,
            tilingData_.usedCoreNum);
    OP_LOGD(context_->GetNodeName(),
            "blockCount=%ld wsSizePerToken=%lu normEps=%f tilingKey=%lu N_ub_max=%u "
            "needBackward=%u stagingBytes=%u tokensPerFlush=%u elemsPerToken=%u",
            tilingData_.blockCount, static_cast<unsigned long>(tilingData_.wsSizePerToken), tilingData_.normEps,
            static_cast<unsigned long>(tilingKey_), CalcMaxResidentRows(tilingData_.hiddenSize),
            tilingData_.needBackward, tilingData_.stagingBytes, tilingData_.tokensPerFlush, tilingData_.elemsPerToken);
}

static ge::graphStatus BlockAttentionResidualsTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("BlockAttentionResiduals", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForBlockAttentionResiduals(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("BlockAttentionResiduals", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttentionResiduals)
    .Tiling(BlockAttentionResidualsTilingFunc)
    .TilingParse<BlockAttentionResidualsCompileInfo>(TilingPrepareForBlockAttentionResiduals);
} // namespace optiling
