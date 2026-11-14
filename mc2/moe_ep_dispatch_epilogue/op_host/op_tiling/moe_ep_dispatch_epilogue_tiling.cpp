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
 * \file moe_ep_dispatch_epilogue_tiling.cpp
 * \brief Expert-sorted epilogue tiling — perSlotBytes = direct: tokenSize + ALIGN_UP(meta, 512);
 *        hybrid: ALIGN_UP(hAlign + meta, 512),
 *        eAlloc from output shape, workspace for rank+expert prefix sums.
 */

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "moe_ep_window_layout.h"
#include "../../../common/utils/moe_ep_exception_dump.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling.h"
#include "../../../common/op_kernel/moe_shared_jetty_context.h"
#include "../../../common/op_api/moe_ep_comm_alg.h"
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling_key.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t X_INDEX = 1U;
constexpr uint32_t TOPK_IDX_INDEX = 2U;
constexpr uint32_t NUM_RECV_PER_RANK_INDEX = 3U;
constexpr uint32_t NUM_RECV_PER_EXPERT_INDEX = 4U;
constexpr uint32_t CACHED_RECV_SRC_METADATA_INDEX = 5U;

constexpr uint32_t OUT_RECV_X_INDEX = 0U;
constexpr uint32_t OUT_RECV_SRC_METADATA_INDEX = 1U;
constexpr uint32_t OUT_RECV_TOPK_WEIGHTS_INDEX = 2U;
constexpr uint32_t OUT_RECV_SCALES_INDEX = 3U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;
constexpr uint32_t ATTR_HAS_TOPK_WEIGHTS_INDEX = 5;
constexpr uint32_t ATTR_TOPO_TYPE_INDEX = 6;
constexpr uint32_t ATTR_RANK_NUM_PER_SERVER_INDEX = 7;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 8;

constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 1024;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_NUM_EXPERTS = 2048;
constexpr int64_t MIN_NUM_EXPERTS = 2;
constexpr int64_t K_MAX = 32;
constexpr int64_t H_MAX = 8192;
constexpr uint32_t HIDDEN_ALIGN = 32U;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t FP8_DTYPE_SIZE = 1UL;
constexpr uint64_t METADATA_DTYPE_SIZE = 4UL; // sizeof(int32)=sizeof(float)=4
constexpr uint64_t METADATA_FIELDS = 5UL;
constexpr int64_t SCALES_GROUP_SIZE_MXFP = 32;
constexpr int64_t SCALES_GROUP_SIZE_PERGROUP = 128;
constexpr int64_t SCALES_ALIGN_EVEN = 2; // fp8 align 2
constexpr uint32_t NETWORK_DIRECT = 0U;
constexpr uint32_t NETWORK_HYBRID = 1U;
constexpr uint32_t TOPK_AND_TOPK_WEIGHT_NUMBER = 2U;

static void PrintTilingDataInfo(const char* nodeName, const MoeEpDispatchEpilogueInfo& info)
{
    OP_LOGD(nodeName, "epWorldSize=%u, epRankId=%u, numExperts=%u, numLocalExperts=%u", info.cfg.epWorldSize,
            info.cfg.epRankId, info.cfg.numExperts, info.cfg.numLocalExperts);
    OP_LOGD(nodeName, "hidden=%u, topK=%u, numTokens=%u, numMaxTokensPerRank=%u, perSlotBytes=%u", info.cfg.hidden,
            info.cfg.topK, info.cfg.numTokens, info.cfg.numMaxTokensPerRank, info.cfg.perSlotBytes);
    OP_LOGD(nodeName, "aivNum=%u, cached=%u, scalesBytes=%u", info.aivNum, info.cached, info.cfg.scalesBytes);
}

// ---------------------------------------------------------------------------
// 属性合法性
// ---------------------------------------------------------------------------
static ge::graphStatus CheckAttrParams(const gert::TilingContext* context, const char* nodeName,
                                       MoeEpDispatchEpilogueInfo& info, uint32_t& networkMode, uint32_t& serverNum,
                                       uint32_t& rankNumPerServerOut, uint32_t& commAlg)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto requestedNetworkModePtr = attrs->GetAttrPointer<int64_t>(ATTR_TOPO_TYPE_INDEX);
    auto rankNumPerServerPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_NUM_PER_SERVER_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);

    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nmtPtr == nullptr, OP_LOGE(nodeName, "nmtPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(requestedNetworkModePtr == nullptr, OP_LOGE(nodeName, "requestedNetworkModePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankNumPerServerPtr == nullptr, OP_LOGE(nodeName, "rankNumPerServerPtr is null."),
                    return ge::GRAPH_FAILED);
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t requestedNetworkMode = *requestedNetworkModePtr;
    int64_t rankNumPerServer = *rankNumPerServerPtr;
    OP_TILING_CHECK(
        (epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
                                              "The value of ep_world_size must be in the supported range"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_rank_id", std::to_string(*epRankIdPtr).c_str(),
                                              "The value of ep_rank_id must be in the range [0, ep_world_size)"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*numExpertsPtr < MIN_NUM_EXPERTS) || (*numExpertsPtr > MAX_NUM_EXPERTS) || (*numExpertsPtr % epWorldSize != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "num_experts", std::to_string(*numExpertsPtr).c_str(),
            "The value of num_experts must be in the supported range and divisible by ep_world_size"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *nmtPtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_max_tokens_per_rank", std::to_string(*nmtPtr).c_str(), "positive"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *cclBufferSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "ccl_buffer_size", std::to_string(*cclBufferSizePtr).c_str(), "positive"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!Mc2Aclnn::IsValidMoeEpCommAlg(commAlgPtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "comm_alg", commAlgPtr == nullptr ? "null" : commAlgPtr,
                                              "a supported communication algorithm"),
                    return ge::GRAPH_FAILED);
    if (Mc2Aclnn::IsMoeEpClosOnly(commAlgPtr)) {
        OP_TILING_CHECK(
            epWorldSize < Mc2Aclnn::MOE_EP_CLOS_ONLY_MIN_RANK_SIZE,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
                                                  "clos_only requires at least 2 ranks"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            requestedNetworkMode != NETWORK_DIRECT,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "topo_type", "non-direct topology",
                                                  "clos_only dispatch epilogue supports direct topology only"),
            return ge::GRAPH_FAILED);
    }
    MoeEpTopology topology{};
    OP_TILING_CHECK(ResolveMoeEpTopology(static_cast<uint32_t>(epWorldSize), requestedNetworkMode, rankNumPerServer,
                                         topology) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName,
                            "Invalid Moe EP topology: epWorldSize=%ld, topoType=%ld, "
                            "rankNumPerServer=%ld.",
                            epWorldSize, requestedNetworkMode, rankNumPerServer),
                    return ge::GRAPH_FAILED);
    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);
    serverNum = topology.serverNum;
    networkMode = topology.networkMode;
    rankNumPerServerOut = topology.rankNumPerServer;
    commAlg = Mc2Aclnn::ParseMoeEpCommAlg(commAlgPtr, epWorldSize);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 输入 dtype（cached 路径才校验 OPTIONAL cachedRecvSrcMetadata）
// ---------------------------------------------------------------------------
static ge::graphStatus CheckInputDataType(const gert::TilingContext* context, const char* nodeName, bool cached)
{
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_TILING_CHECK(
        contextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(
            nodeName, "context", ge::TypeUtils::DataTypeToSerialString(contextDesc->GetDataType()).c_str(), "DT_INT32"),
        return ge::GRAPH_FAILED);

    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16) &&
            (xDesc->GetDataType() != ge::DT_FLOAT8_E5M2) && (xDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()).c_str(),
                                  "DT_BF16/DT_FLOAT16/DT_FLOAT8_E5M2/DT_FLOAT8_E4M3FN"),
        return ge::GRAPH_FAILED);

    auto topkIdxDesc = context->GetInputDesc(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxDesc);
    OP_TILING_CHECK(topkIdxDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "topk_idx",
                                              ge::TypeUtils::DataTypeToSerialString(topkIdxDesc->GetDataType()).c_str(),
                                              "DT_INT32"),
                    return ge::GRAPH_FAILED);

    if (cached) {
        auto recvSrcMetaDesc = context->GetInputDesc(CACHED_RECV_SRC_METADATA_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaDesc);
        OP_TILING_CHECK(recvSrcMetaDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE(
                            nodeName, "cached_recv_src_metadata",
                            ge::TypeUtils::DataTypeToSerialString(recvSrcMetaDesc->GetDataType()).c_str(), "DT_INT32"),
                        return ge::GRAPH_FAILED);
    }

    auto numRecvRankDesc = context->GetInputDesc(NUM_RECV_PER_RANK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvRankDesc);
    OP_TILING_CHECK(numRecvRankDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        nodeName, "num_recv_tokens_per_rank",
                        ge::TypeUtils::DataTypeToSerialString(numRecvRankDesc->GetDataType()).c_str(), "DT_INT32"),
                    return ge::GRAPH_FAILED);

    auto numRecvExpertDesc = context->GetInputDesc(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvExpertDesc);
    OP_TILING_CHECK(numRecvExpertDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(
                        nodeName, "num_recv_tokens_per_expert",
                        ge::TypeUtils::DataTypeToSerialString(numRecvExpertDesc->GetDataType()).c_str(), "DT_INT64"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// Input shapes; cached packed metadata is checked with the output capacity below.
// 必须先于本函数完成 attr 校验，依赖 epWorldSize / numLocalExperts / nmt
// ---------------------------------------------------------------------------
static ge::graphStatus CheckInputTensorShape(const gert::TilingContext* context, const char* nodeName,
                                             MoeEpDispatchEpilogueInfo& info, const uint32_t commAlg)
{
    // ---- context: dim 必须 = 1 ----
    const gert::StorageShape* contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextStorageShape);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "context",
                        (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
    // Preserve the legacy auto-context shape contract; clos_only has a new, fixed shared-Jetty context ABI.
    if (commAlg == Mc2Tiling::MOE_EP_COMM_ALG_SHARED_JETTY) {
        constexpr int64_t expectedContextElements = sizeof(Mc2Kernel::MoeSharedJettyContext) / sizeof(int32_t);
        OP_TILING_CHECK(
            contextStorageShape->GetStorageShape().GetDim(0) != expectedContextElements,
            OP_LOGE_FOR_INVALID_SHAPESIZE(nodeName, "context",
                                          std::to_string(contextStorageShape->GetStorageShape().GetDim(0)).c_str(),
                                          std::to_string(expectedContextElements).c_str()),
            return ge::GRAPH_FAILED);
    }
    // ---- x [num_tokens, hidden]，hidden 为 cfg 唯一真值来源 ----
    const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "x", (std::to_string(xShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);
    const int64_t numTokens = xShape->GetStorageShape().GetDim(0);
    const int64_t hidden = xShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(numTokens <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x", std::to_string(numTokens).c_str(), "positive"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((hidden <= 0) || (hidden > H_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "x", std::to_string(hidden).c_str(),
                                                          "The second dimension of x must be in the range (0, H_MAX]"),
                    return ge::GRAPH_FAILED);

    // ---- topkIdx [num_tokens, top_k] int32 ----
    const gert::StorageShape* topkIdxShape = context->GetInputShape(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);
    OP_TILING_CHECK(
        topkIdxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "topk_idx",
                                     (std::to_string(topkIdxShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t topK = topkIdxShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        (topK <= 0) || (topK > K_MAX),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "topk_idx", std::to_string(topK).c_str(),
                                              "The second dimension of topk_idx must be in the range (0, K_MAX]"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        topkIdxShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "topk_idx and x",
            (std::to_string(topkIdxShape->GetStorageShape().GetDim(0)) + " and " + std::to_string(numTokens)).c_str(),
            "The first dimension of topk_idx must be equal to the first dimension of x"),
        return ge::GRAPH_FAILED);
    info.cfg.numTokens = static_cast<uint32_t>(numTokens);
    info.cfg.hidden = static_cast<uint32_t>(hidden);
    info.cfg.topK = static_cast<uint32_t>(topK);

    // Cached packed shape is checked against recv_x capacity in CheckOutputTensors.

    // ---- numRecvPerRank [ep_world_size] int32 ----
    const gert::StorageShape* numRecvRankShape = context->GetInputShape(NUM_RECV_PER_RANK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvRankShape);
    OP_TILING_CHECK(numRecvRankShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "num_recv_tokens_per_rank",
                        (std::to_string(numRecvRankShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
    const int64_t numRecvRankDim0 = numRecvRankShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numRecvRankDim0 != static_cast<int64_t>(info.cfg.epWorldSize),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "num_recv_tokens_per_rank", std::to_string(numRecvRankDim0).c_str(),
                        "The first dimension of num_recv_tokens_per_rank must be equal to ep_world_size"),
                    return ge::GRAPH_FAILED);

    // ---- numRecvPerExpert [num_local_experts] int64 ----
    const gert::StorageShape* numRecvExpertShape = context->GetInputShape(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvExpertShape);
    OP_TILING_CHECK(numRecvExpertShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "num_recv_tokens_per_expert",
                        (std::to_string(numRecvExpertShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
    const int64_t numRecvExpertDim0 = numRecvExpertShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numRecvExpertDim0 != static_cast<int64_t>(info.cfg.numLocalExperts),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "num_recv_tokens_per_expert", std::to_string(numRecvExpertDim0).c_str(),
                        "The first dimension of num_recv_tokens_per_expert must be equal to num_local_experts"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus CheckRecvScalesTensor(const gert::TilingContext* context, const char* nodeName,
                                             ge::DataType recvXDtype, int64_t aAlloc, int64_t hidden,
                                             MoeEpDispatchEpilogueInfo& info)
{
    auto recvScalesDesc = context->GetOutputDesc(OUT_RECV_SCALES_INDEX);
    auto recvScalesShape = context->GetOutputShape(OUT_RECV_SCALES_INDEX);
    const bool isFp8 = (recvXDtype == ge::DT_FLOAT8_E5M2 || recvXDtype == ge::DT_FLOAT8_E4M3FN);
    if (!isFp8) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_NULL_WITH_CONTEXT(context, recvScalesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvScalesShape);
    OP_TILING_CHECK(recvScalesShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "recv_scales",
                        (std::to_string(recvScalesShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvScalesShape->GetStorageShape().GetDim(0) != aAlloc,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "recv_scales and recv_x",
            (std::to_string(recvScalesShape->GetStorageShape().GetDim(0)) + " and " + std::to_string(aAlloc)).c_str(),
            "The first dimension of recv_scales must be equal to the first dimension of recv_x"),
        return ge::GRAPH_FAILED);

    const ge::DataType scalesDtype = recvScalesDesc->GetDataType();
    OP_TILING_CHECK(
        scalesDtype != ge::DT_FLOAT && scalesDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "recv_scales", ge::TypeUtils::DataTypeToSerialString(scalesDtype).c_str(),
                                  "DT_FLOAT/DT_FLOAT8_E8M0"),
        return ge::GRAPH_FAILED);
    const int64_t groupSize = (scalesDtype == ge::DT_FLOAT) ? SCALES_GROUP_SIZE_PERGROUP : SCALES_GROUP_SIZE_MXFP;
    int64_t expectedDim1 = (hidden + groupSize - 1) / groupSize;
    if (scalesDtype == ge::DT_FLOAT8_E8M0) {
        expectedDim1 = (expectedDim1 + SCALES_ALIGN_EVEN - 1) / SCALES_ALIGN_EVEN * SCALES_ALIGN_EVEN;
    }
    OP_TILING_CHECK(
        recvScalesShape->GetStorageShape().GetDim(1) != expectedDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "recv_scales", std::to_string(recvScalesShape->GetStorageShape().GetDim(1)).c_str(),
            "The second dimension of recv_scales must be aligned according to the scale dtype and hidden size"),
        return ge::GRAPH_FAILED);

    uint32_t scalesSize = (scalesDtype == ge::DT_FLOAT) ? sizeof(float) : FP8_DTYPE_SIZE;
    info.cfg.scalesBytes = static_cast<uint32_t>(expectedDim1 * scalesSize);
    info.isMxQuant = (scalesDtype == ge::DT_FLOAT8_E8M0) ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensors(const gert::TilingContext* context, const char* nodeName,
                                          MoeEpDispatchEpilogueInfo& info, int64_t topK, bool hasTopkWeights,
                                          bool cached)
{
    // ---- recvX [A_alloc, hidden] bf16/fp16 ----
    auto recvXShape = context->GetOutputShape(OUT_RECV_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvXShape);
    OP_TILING_CHECK(
        recvXShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "recv_x",
                                     (std::to_string(recvXShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t aAlloc = recvXShape->GetStorageShape().GetDim(0);
    const int64_t hidden = static_cast<int64_t>(info.cfg.hidden);
    const int64_t minTopKLocalExperts =
        (topK < static_cast<int64_t>(info.cfg.numLocalExperts)) ? topK : static_cast<int64_t>(info.cfg.numLocalExperts);
    const int64_t aUpper = static_cast<int64_t>(info.cfg.epWorldSize) *
                           static_cast<int64_t>(info.cfg.numMaxTokensPerRank) * minTopKLocalExperts;
    OP_TILING_CHECK(
        aAlloc < 0 || aAlloc > aUpper,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "recv_x", std::to_string(aAlloc).c_str(),
                                              "The first dimension of recv_x must be in the range [0, ep_world_size * "
                                              "num_max_tokens_per_rank * min(top_k, num_local_experts)]"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        aAlloc > INT32_MAX,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "recv_x", std::to_string(aAlloc).c_str(),
            "The first dimension of recv_x must not exceed INT32_MAX because metadata stores recv_x_idx as int32"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvXShape->GetStorageShape().GetDim(1) != hidden,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "recv_x and x",
            (std::to_string(recvXShape->GetStorageShape().GetDim(1)) + " and " + std::to_string(hidden)).c_str(),
            "The second dimension of recv_x must be equal to the second dimension of x"),
        return ge::GRAPH_FAILED);

    auto recvXDesc = context->GetOutputDesc(OUT_RECV_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvXDesc);
    const ge::DataType recvXDtype = recvXDesc->GetDataType();
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(
        recvXDtype != xDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "recv_x and x",
                                               (std::string(ge::TypeUtils::DataTypeToSerialString(recvXDtype)) +
                                                " and " + ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()))
                                                   .c_str(),
                                               "The dtype of recv_x must be equal to the dtype of x"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRecvScalesTensor(context, nodeName, recvXDtype, aAlloc, hidden, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "recv_scales",
                                                             "The shape or dtype of recv_scales is invalid"),
                    return ge::GRAPH_FAILED);

    // ---- recvTopkWeights [A_alloc] float when topk weights are enabled ----
    auto recvTopkWeightsShape = context->GetOutputShape(OUT_RECV_TOPK_WEIGHTS_INDEX);
    if (hasTopkWeights) {
        OP_CHECK_NULL_WITH_CONTEXT(context, recvTopkWeightsShape);
        OP_TILING_CHECK(recvTopkWeightsShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "recv_topk_weights",
                            (std::to_string(recvTopkWeightsShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            recvTopkWeightsShape->GetStorageShape().GetDim(0) != aAlloc,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                nodeName, "recv_topk_weights and recv_x",
                (std::to_string(recvTopkWeightsShape->GetStorageShape().GetDim(0)) + " and " + std::to_string(aAlloc))
                    .c_str(),
                "The first dimension of recv_topk_weights must be equal to the first dimension of recv_x"),
            return ge::GRAPH_FAILED);
        auto recvTopkWeightsDesc = context->GetOutputDesc(OUT_RECV_TOPK_WEIGHTS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, recvTopkWeightsDesc);
        OP_TILING_CHECK(
            recvTopkWeightsDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "recv_topk_weights",
                                      ge::TypeUtils::DataTypeToSerialString(recvTopkWeightsDesc->GetDataType()).c_str(),
                                      "DT_FLOAT"),
            return ge::GRAPH_FAILED);
    }

    // Full storage descriptor includes five-column rows, rank offsets, and the local reverse index.
    info.metadataRankOffsetsOffset =
        AlignMoeEpWin(static_cast<uint64_t>(aAlloc) * METADATA_FIELDS * METADATA_DTYPE_SIZE);
    info.localRecvIndexOffset = info.metadataRankOffsetsOffset +
                                AlignMoeEpWin((static_cast<uint64_t>(info.cfg.epWorldSize) + 1U) * METADATA_DTYPE_SIZE);
    // 旧字段的偏移保持不变；尾部仅追加slot终点和metadata行号。
    info.slotEndsOffset = info.localRecvIndexOffset + AlignMoeEpWin(static_cast<uint64_t>(info.cfg.numTokens) *
                                                                    info.cfg.topK * METADATA_DTYPE_SIZE);
    info.slotRowIdsOffset = info.slotEndsOffset + AlignMoeEpWin(static_cast<uint64_t>(aAlloc) * METADATA_DTYPE_SIZE);
    const uint64_t packedElements =
        (info.slotRowIdsOffset + AlignMoeEpWin(static_cast<uint64_t>(aAlloc) * METADATA_DTYPE_SIZE)) /
        METADATA_DTYPE_SIZE;
    auto recvSrcMetaShape = context->GetOutputShape(OUT_RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaShape);
    OP_TILING_CHECK(recvSrcMetaShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "recv_src_metadata", "unknown",
                                                             "The shape dim of recv_src_metadata must be 1D"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvSrcMetaShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(packedElements),
        OP_LOGE_FOR_INVALID_SHAPESIZE(nodeName, "recv_src_metadata", "unknown", std::to_string(packedElements).c_str()),
        return ge::GRAPH_FAILED);
    auto recvSrcMetaDesc = context->GetOutputDesc(OUT_RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaDesc);
    OP_TILING_CHECK(recvSrcMetaDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "recv_src_metadata", "unknown",
                                                          "The dtype of recv_src_metadata must be DT_INT32"),
                    return ge::GRAPH_FAILED);
    if (cached) {
        auto cachedShape = context->GetInputShape(CACHED_RECV_SRC_METADATA_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, cachedShape);
        OP_TILING_CHECK(
            cachedShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "cached_recv_src_metadata", "unknown",
                                                     "The shape dim of cached_recv_src_metadata must be 1D"),
            return ge::GRAPH_FAILED);
        // Cached input and output use the same A_alloc, hence the same tail offset.
        OP_TILING_CHECK(cachedShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(packedElements),
                        OP_LOGE_FOR_INVALID_SHAPESIZE(nodeName, "cached_recv_src_metadata", "unknown",
                                                      std::to_string(packedElements).c_str()),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildAndCheckWindowLayout(const gert::TilingContext* context, const char* nodeName,
                                                 MoeEpDispatchEpilogueInfo& info, uint32_t networkMode,
                                                 uint32_t serverNum, uint32_t rankNumPerServer, const uint32_t commAlg)
{
    auto attrs = context->GetAttrs();
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
                    return ge::GRAPH_FAILED);
    const uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    uint32_t aivNum = info.aivNum;
    const MoeEpWindowLayoutParams params = {info.cfg.epWorldSize,
                                            info.cfg.numLocalExperts,
                                            info.cfg.numMaxTokensPerRank,
                                            info.cfg.topK,
                                            info.cfg.hidden,
                                            networkMode,
                                            rankNumPerServer,
                                            serverNum,
                                            commAlg};
    MoeEpWindowLayout layout{};
    OP_TILING_CHECK(CalcMoeEpWindowLayout(params, layout) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Calculate Moe EP window layout failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMoeEpWindowCapacity(layout.requiredBytes, maxWindowSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check Moe EP window capacity failed."), return ge::GRAPH_FAILED);

    info.dumpMetadata = BuildMoeEpDumpMetadata(params, layout, aivNum);
    info.winDataOffset = layout.winDataOffset;
    info.slotWinStateOffset = layout.slotWinStateOffset;
    info.dispatchNotifyCount = layout.dispatchNotifyCount;
    info.totalWinSizeEp = maxWindowSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpDispatchEpilogueTilingFunc(gert::TilingContext* context)
{
    context->SetScheduleMode(1U);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueTilingData* tilingData = context->GetTilingData<MoeEpDispatchEpilogueTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueInfo& info = tilingData->moeEpDispatchEpilogueInfo;
    uint32_t commAlg = Mc2Tiling::MOE_EP_COMM_ALG_AUTO;
    uint32_t networkMode = NETWORK_DIRECT;
    uint32_t serverNum = 1U;
    uint32_t rankNumPerServer = 1U;

    OP_TILING_CHECK(CheckAttrParams(context, nodeName, info, networkMode, serverNum, rankNumPerServer, commAlg) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);

    bool cached = (context->GetInputShape(CACHED_RECV_SRC_METADATA_INDEX) != nullptr);
    info.cached = cached ? 1U : 0U;
    info.networkMode = networkMode;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto hasTopkWeightsPtr = attrs->GetAttrPointer<bool>(ATTR_HAS_TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(hasTopkWeightsPtr == nullptr, OP_LOGE(nodeName, "hasTopkWeightsPtr is null."),
                    return ge::GRAPH_FAILED);
    bool hasTopkWeights = *hasTopkWeightsPtr;

    OP_TILING_CHECK(
        CheckInputDataType(context, nodeName, cached) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors", "The dtype of an input tensor is invalid"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckInputTensorShape(context, nodeName, info, commAlg) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "input tensors", "The shape of an input tensor is invalid"),
        return ge::GRAPH_FAILED);

    const int64_t topK = static_cast<int64_t>(info.cfg.topK);

    OP_TILING_CHECK(CheckOutputTensors(context, nodeName, info, topK, hasTopkWeights, cached) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "output tensors",
                                                             "The shape or dtype of an output tensor is invalid"),
                    return ge::GRAPH_FAILED);

    auto recvXDesc = context->GetOutputDesc(OUT_RECV_X_INDEX);
    OP_TILING_CHECK(recvXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "recv_x"), return ge::GRAPH_FAILED);
    ge::DataType recvXDtype = recvXDesc->GetDataType();
    bool isFp8 = (recvXDtype == ge::DT_FLOAT8_E5M2 || recvXDtype == ge::DT_FLOAT8_E4M3FN);
    uint32_t recvXDtypeSize = isFp8 ? FP8_DTYPE_SIZE : MAX_OUT_DTYPE_SIZE;
    uint32_t tokenSize = info.cfg.hidden * recvXDtypeSize;
    uint32_t kAlign32 = ((info.cfg.topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t scalesSizeAlign32 = isFp8 ? ((info.cfg.scalesBytes + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN : 0;
    uint32_t metaSlotBytes = scalesSizeAlign32 + kAlign32 * TOPK_AND_TOPK_WEIGHT_NUMBER + UB_ALIGN;
    if (networkMode == NETWORK_HYBRID) {
        // hybrid 槽布局(与 dispatch hybrid 一致): [x: ALIGN32(tokenSize)][scales][meta]，整槽 512 对齐
        uint32_t hAlign32 = ((tokenSize + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
        info.cfg.perSlotBytes = ((hAlign32 + metaSlotBytes + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    } else {
        // direct 紧凑槽布局: [x: tokenSize][meta: ALIGN512(metaSlotBytes)]
        info.cfg.perSlotBytes = tokenSize + (metaSlotBytes + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN * WIN_ADDR_ALIGN;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_TILING_CHECK(aivNum == 0U, OP_LOGE(nodeName, "Platform reports aiv_num=0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        aivNum > MOE_EP_PER_CORE_DIAG_SLOT_COUNT,
        OP_LOGE(nodeName, "aiv_num=%u exceeds diagnostic slot capacity %lu.", aivNum, MOE_EP_PER_CORE_DIAG_SLOT_COUNT),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ubSize == 0UL, OP_LOGE(nodeName, "Platform reports ub_size=0."), return ge::GRAPH_FAILED);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_TILING_CHECK(
        BuildAndCheckWindowLayout(context, nodeName, info, networkMode, serverNum, rankNumPerServer, commAlg) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "window layout", "The computed HCCL window size is invalid"),
        return ge::GRAPH_FAILED);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    // Joint counts replace the old per-core rank matrix; each core derives private metadata cursors ordered by rank,
    // expert and contributing core. They sit at the very start of this op's user workspace.
    // 必须与 kernel 的 rankExpertCountStride_ 逐字节一致：kernel 按 [rank][expert] 存表，每个 rank 的行先补到
    // ELEM_ALIGN 再乘 epWorldSize_。写成 ceil(numExperts / 8) * 8 只在 numLocalExperts 本身对齐时才相等。
    uint32_t jointCountStride = ((info.cfg.numLocalExperts + 7U) / 8U) * 8U * info.cfg.epWorldSize;
    uint64_t jointCountBytes = static_cast<uint64_t>(aivNum) * jointCountStride * sizeof(int32_t);
    uint64_t jointCountBytesAlign512 = ((jointCountBytes + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + jointCountBytesAlign512;

    uint64_t tilingKey =
        GET_TPL_TILING_KEY(TILINGKEY_TPL_A5, cached ? 1U : 0U, hasTopkWeights ? 1U : 0U, info.isMxQuant ? 1U : 0U,
                           commAlg == Mc2Tiling::MOE_EP_COMM_ALG_SHARED_JETTY);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u, cached=%u, hasTopkWeights=%u, isMxQuant=%u", tilingKey,
            blockDim, aivNum, info.cached, hasTopkWeights ? 1U : 0U, info.isMxQuant);

    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

struct MoeEpDispatchEpilogueCompileInfo {};
ge::graphStatus TilingParseForMoeEpDispatchEpilogue(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatchEpilogue)
    .Tiling(MoeEpDispatchEpilogueTilingFunc)
    .TilingParse<MoeEpDispatchEpilogueCompileInfo>(TilingParseForMoeEpDispatchEpilogue);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
inline void MoeEpDispatchEpilogueExceptionImplWrapper(aclrtExceptionInfo* args, void* userdata)
{
    Mc2Exception::MoeEpExceptionImpl(args, userdata, "MoeEpDispatchEpilogue");
}

__attribute__((constructor)) void RegisterMoeEpDispatchEpilogueExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;
    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS ||
        aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS ||
        runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeEpDispatchEpilogue", "Runtime or metadef does not support exception dump registration.");
        return;
    }
    IMPL_OP(MoeEpDispatchEpilogue).ExceptionDumpParseFunc(MoeEpDispatchEpilogueExceptionImplWrapper);
}
#endif
} // namespace optiling
