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
 * \brief Expert-sorted epilogue tiling — perSlotBytes = ALIGN_UP(hidden*2, 512) + 512,
 *        eAlloc from output shape, workspace for rank+expert prefix sums.
 */

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
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling_key.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t DST_BUFFER_SLOT_IDX_INDEX = 1U;
constexpr uint32_t NUM_RECV_PER_RANK_INDEX = 2U;
constexpr uint32_t NUM_RECV_PER_EXPERT_INDEX = 3U;
constexpr uint32_t CACHED_RECV_SRC_METADATA_INDEX = 4U;

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

constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr int64_t META_INNER_DIM = 4; // recvSrcMetadata dim(1) = 4
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
constexpr int64_t SCALES_GROUP_SIZE_MXFP = 32;
constexpr int64_t SCALES_GROUP_SIZE_PERGROUP = 128;
constexpr int64_t SCALES_ALIGN_EVEN = 2; // fp8 align 2
constexpr uint32_t NETWORK_DIRECT = 0U;
constexpr uint32_t NETWORK_HYBRID = 1U;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpDispatchEpilogueInfo &info)
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
static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, const char *nodeName,
                                       MoeEpDispatchEpilogueInfo &info, uint32_t &networkMode, uint32_t &serverNum,
                                       uint32_t &rankNumPerServerOut)
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
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
                    OP_LOGE(nodeName, "ep_world_size is invalid, should be in [%ld, %ld], but got %ld.",
                            MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "ep_rank_id is invalid, should be in [0, %ld), but got %ld.", epWorldSize, *epRankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*numExpertsPtr < MIN_NUM_EXPERTS) || (*numExpertsPtr > MAX_NUM_EXPERTS) || (*numExpertsPtr % epWorldSize != 0),
        OP_LOGE(nodeName,
                "num_experts is invalid, should be in [%ld, %ld] and divisible by ep_world_size, but got "
                "num_experts=%ld, ep_world_size=%ld.",
                MIN_NUM_EXPERTS, MAX_NUM_EXPERTS, *numExpertsPtr, epWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*nmtPtr <= 0, OP_LOGE(nodeName, "num_max_tokens_per_rank must be positive, but got %ld.", *nmtPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*cclBufferSizePtr <= 0,
                    OP_LOGE(nodeName, "ccl_buffer_size must be positive, but got %ld.", *cclBufferSizePtr),
                    return ge::GRAPH_FAILED);
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

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 输入 dtype（cached 路径才校验 OPTIONAL cachedRecvSrcMetadata）
// ---------------------------------------------------------------------------
static ge::graphStatus CheckInputDataType(const gert::TilingContext *context, const char *nodeName, bool cached)
{
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "context dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(contextDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto dstSlotDesc = context->GetInputDesc(DST_BUFFER_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSlotDesc);
    OP_TILING_CHECK(dstSlotDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "dst_buffer_slot_idx dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(dstSlotDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    if (cached) {
        auto recvSrcMetaDesc = context->GetInputDesc(CACHED_RECV_SRC_METADATA_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaDesc);
        OP_TILING_CHECK(recvSrcMetaDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(nodeName, "cached_recv_src_metadata dtype must be DT_INT32, but got %s.",
                                ge::TypeUtils::DataTypeToSerialString(recvSrcMetaDesc->GetDataType()).c_str()),
                        return ge::GRAPH_FAILED);
    }

    auto numRecvRankDesc = context->GetInputDesc(NUM_RECV_PER_RANK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvRankDesc);
    OP_TILING_CHECK(numRecvRankDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(numRecvRankDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto numRecvExpertDesc = context->GetInputDesc(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvExpertDesc);
    OP_TILING_CHECK(numRecvExpertDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dtype must be DT_INT64, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(numRecvExpertDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 输入 shape（cached 路径才校验 OPTIONAL cachedRecvSrcMetadata）
// 必须先于本函数完成 attr 校验，依赖 epWorldSize / numLocalExperts / nmt
// ---------------------------------------------------------------------------
static ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context, const char *nodeName,
                                             const MoeEpDispatchEpilogueInfo &info, bool cached)
{
    // ---- context: dim 必须 = 1 ----
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextStorageShape);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "context dims must be 1, but got %lu.", contextStorageShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    // ---- dstBufferSlotIdx [num_tokens, top_k] int32 ----
    const gert::StorageShape *dstSlotShape = context->GetInputShape(DST_BUFFER_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSlotShape);
    OP_TILING_CHECK(dstSlotShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "dst_buffer_slot_idx dims must be 2, but got %lu.",
                            dstSlotShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t numTokens = dstSlotShape->GetStorageShape().GetDim(0);
    const int64_t topK = dstSlotShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(numTokens <= 0,
                    OP_LOGE(nodeName, "dst_buffer_slot_idx dim0(num_tokens) must be positive, but got %ld.", numTokens),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((topK <= 0) || (topK > K_MAX),
                    OP_LOGE(nodeName, "dst_buffer_slot_idx dim1(top_k) must be in (0, %ld], but got %ld.", K_MAX, topK),
                    return ge::GRAPH_FAILED);

    // ---- cachedRecvSrcMetadata [*, 4] int32 (OPTIONAL，cached 路径才校验) ----
    if (cached) {
        const gert::StorageShape *recvSrcMetaShape = context->GetInputShape(CACHED_RECV_SRC_METADATA_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaShape);
        OP_TILING_CHECK(recvSrcMetaShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(nodeName, "cached_recv_src_metadata dims must be 2, but got %lu.",
                                recvSrcMetaShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        const int64_t minTopKLocalExperts = (topK < static_cast<int64_t>(info.cfg.numLocalExperts)) ?
                                                topK :
                                                static_cast<int64_t>(info.cfg.numLocalExperts);
        const int64_t metaUpper = static_cast<int64_t>(info.cfg.epWorldSize) *
                                  static_cast<int64_t>(info.cfg.numMaxTokensPerRank) * minTopKLocalExperts;
        const int64_t metaDim0 = recvSrcMetaShape->GetStorageShape().GetDim(0);
        const int64_t metaDim1 = recvSrcMetaShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            metaDim0 < 0 || metaDim0 > metaUpper,
            OP_LOGE(nodeName, "cached_recv_src_metadata dim0 must be in [0, %ld], but got %ld.", metaUpper, metaDim0),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            metaDim1 != META_INNER_DIM,
            OP_LOGE(nodeName, "cached_recv_src_metadata dim1 must be %ld, but got %ld.", META_INNER_DIM, metaDim1),
            return ge::GRAPH_FAILED);
    }

    // ---- numRecvPerRank [ep_world_size] int32 ----
    const gert::StorageShape *numRecvRankShape = context->GetInputShape(NUM_RECV_PER_RANK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvRankShape);
    OP_TILING_CHECK(numRecvRankShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank dims must be 1, but got %lu.",
                            numRecvRankShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t numRecvRankDim0 = numRecvRankShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numRecvRankDim0 != static_cast<int64_t>(info.cfg.epWorldSize),
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank dim0 must equal ep_world_size=%u, but got %ld.",
                            info.cfg.epWorldSize, numRecvRankDim0),
                    return ge::GRAPH_FAILED);

    // ---- numRecvPerExpert [num_local_experts] int64 ----
    const gert::StorageShape *numRecvExpertShape = context->GetInputShape(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvExpertShape);
    OP_TILING_CHECK(numRecvExpertShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dims must be 1, but got %lu.",
                            numRecvExpertShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t numRecvExpertDim0 = numRecvExpertShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numRecvExpertDim0 != static_cast<int64_t>(info.cfg.numLocalExperts),
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dim0 must equal num_local_experts=%u, but got %ld.",
                            info.cfg.numLocalExperts, numRecvExpertDim0),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus CheckRecvScalesTensor(const gert::TilingContext *context, const char *nodeName,
                                             ge::DataType recvXDtype, int64_t aAlloc, int64_t hidden,
                                             MoeEpDispatchEpilogueInfo &info)
{
    auto recvScalesDesc = context->GetOutputDesc(OUT_RECV_SCALES_INDEX);
    auto recvScalesShape = context->GetOutputShape(OUT_RECV_SCALES_INDEX);
    const bool isFp8 = (recvXDtype == ge::DT_FLOAT8_E5M2 || recvXDtype == ge::DT_FLOAT8_E4M3FN);
    if (!isFp8) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_NULL_WITH_CONTEXT(context, recvScalesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvScalesShape);
    OP_TILING_CHECK(
        recvScalesShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "recv_scales dims must be 2, but got %lu.", recvScalesShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvScalesShape->GetStorageShape().GetDim(0) != aAlloc,
                    OP_LOGE(nodeName, "recv_scales dim0 must equal recv_x dim0=%ld, but got %ld.", aAlloc,
                            recvScalesShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);

    const ge::DataType scalesDtype = recvScalesDesc->GetDataType();
    OP_TILING_CHECK(scalesDtype != ge::DT_FLOAT && scalesDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(nodeName, "recv_scales dtype must be DT_FLOAT or DT_FLOAT8_E8M0, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(scalesDtype).c_str()),
                    return ge::GRAPH_FAILED);
    const int64_t groupSize = (scalesDtype == ge::DT_FLOAT) ? SCALES_GROUP_SIZE_PERGROUP : SCALES_GROUP_SIZE_MXFP;
    int64_t expectedDim1 = (hidden + groupSize - 1) / groupSize;
    if (scalesDtype == ge::DT_FLOAT8_E8M0) {
        expectedDim1 = (expectedDim1 + SCALES_ALIGN_EVEN - 1) / SCALES_ALIGN_EVEN * SCALES_ALIGN_EVEN;
    }
    OP_TILING_CHECK(recvScalesShape->GetStorageShape().GetDim(1) != expectedDim1,
                    OP_LOGE(nodeName, "recv_scales dim1 must be %ld, but got %ld.", expectedDim1,
                            recvScalesShape->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    uint32_t scalesSize = (scalesDtype == ge::DT_FLOAT) ? sizeof(float) : FP8_DTYPE_SIZE;
    info.cfg.scalesBytes = static_cast<uint32_t>(expectedDim1 * scalesSize);
    info.isMxQuant = (scalesDtype == ge::DT_FLOAT8_E8M0) ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensors(const gert::TilingContext *context, const char *nodeName,
                                          MoeEpDispatchEpilogueInfo &info, int64_t topK, bool hasTopkWeights)
{
    // ---- recvX [A_alloc, hidden] bf16/fp16 ----
    auto recvXShape = context->GetOutputShape(OUT_RECV_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvXShape);
    OP_TILING_CHECK(recvXShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "recv_x dims must be 2, but got %lu.", recvXShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t aAlloc = recvXShape->GetStorageShape().GetDim(0);
    const int64_t hidden = recvXShape->GetStorageShape().GetDim(1);
    const int64_t minTopKLocalExperts =
        (topK < static_cast<int64_t>(info.cfg.numLocalExperts)) ? topK : static_cast<int64_t>(info.cfg.numLocalExperts);
    const int64_t aUpper = static_cast<int64_t>(info.cfg.epWorldSize) *
                           static_cast<int64_t>(info.cfg.numMaxTokensPerRank) * minTopKLocalExperts;
    OP_TILING_CHECK(
        aAlloc < 0 || aAlloc > aUpper,
        OP_LOGE(nodeName, "recv_x dim0(A_alloc) must be in [0, ep*nmt*min(top_k,num_local_experts)=%ld], but got %ld.",
                aUpper, aAlloc),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((hidden <= 0) || (hidden > H_MAX),
                    OP_LOGE(nodeName, "recv_x dim1(hidden) must be in (0, %ld], but got %ld.", H_MAX, hidden),
                    return ge::GRAPH_FAILED);

    auto recvXDesc = context->GetOutputDesc(OUT_RECV_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvXDesc);
    const ge::DataType recvXDtype = recvXDesc->GetDataType();
    OP_TILING_CHECK(
        recvXDtype != ge::DT_BF16 && recvXDtype != ge::DT_FLOAT16 && recvXDtype != ge::DT_FLOAT8_E5M2 &&
            recvXDtype != ge::DT_FLOAT8_E4M3FN,
        OP_LOGE(
            nodeName,
            "recv_x dtype must be in support list [DT_BF16, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN], but got %s.",
            ge::TypeUtils::DataTypeToSerialString(recvXDtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRecvScalesTensor(context, nodeName, recvXDtype, aAlloc, hidden, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check recvScales tensor failed."), return ge::GRAPH_FAILED);

    // ---- recvTopkWeights [A_alloc] float when topk weights are enabled ----
    auto recvTopkWeightsShape = context->GetOutputShape(OUT_RECV_TOPK_WEIGHTS_INDEX);
    if (hasTopkWeights) {
        OP_CHECK_NULL_WITH_CONTEXT(context, recvTopkWeightsShape);
        OP_TILING_CHECK(recvTopkWeightsShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE(nodeName, "recv_topk_weights dims must be 1, but got %lu.",
                                recvTopkWeightsShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(recvTopkWeightsShape->GetStorageShape().GetDim(0) != aAlloc,
                        OP_LOGE(nodeName, "recv_topk_weights dim0 must equal recv_x dim0=%ld, but got %ld.", aAlloc,
                                recvTopkWeightsShape->GetStorageShape().GetDim(0)),
                        return ge::GRAPH_FAILED);
        auto recvTopkWeightsDesc = context->GetOutputDesc(OUT_RECV_TOPK_WEIGHTS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, recvTopkWeightsDesc);
        OP_TILING_CHECK(recvTopkWeightsDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "recv_topk_weights dtype must be DT_FLOAT, but got %s.",
                                ge::TypeUtils::DataTypeToSerialString(recvTopkWeightsDesc->GetDataType()).c_str()),
                        return ge::GRAPH_FAILED);
    }

    // ---- recvSrcMetadata [A_alloc, 4] int32 ----
    auto recvSrcMetaShape = context->GetOutputShape(OUT_RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaShape);
    OP_TILING_CHECK(recvSrcMetaShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "recv_src_metadata dims must be 2, but got %lu.",
                            recvSrcMetaShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvSrcMetaShape->GetStorageShape().GetDim(0) != aAlloc,
                    OP_LOGE(nodeName, "recv_src_metadata dim0 must equal recv_x dim0=%ld, but got %ld.", aAlloc,
                            recvSrcMetaShape->GetStorageShape().GetDim(0)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvSrcMetaShape->GetStorageShape().GetDim(1) != META_INNER_DIM,
                    OP_LOGE(nodeName, "recv_src_metadata dim1 must be %ld, but got %ld.", META_INNER_DIM,
                            recvSrcMetaShape->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);
    auto recvSrcMetaDesc = context->GetOutputDesc(OUT_RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaDesc);
    OP_TILING_CHECK(recvSrcMetaDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "recv_src_metadata dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(recvSrcMetaDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BuildAndCheckWindowLayout(const gert::TilingContext *context, const char *nodeName,
                                                 MoeEpDispatchEpilogueInfo &info, uint32_t networkMode,
                                                 uint32_t serverNum, uint32_t rankNumPerServer)
{
    auto attrs = context->GetAttrs();
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
                    return ge::GRAPH_FAILED);
    const uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    const MoeEpWindowLayoutParams params = {info.cfg.epWorldSize,
                                            info.cfg.numLocalExperts,
                                            info.cfg.numMaxTokensPerRank,
                                            info.cfg.topK,
                                            info.cfg.hidden,
                                            networkMode,
                                            rankNumPerServer,
                                            serverNum};
    MoeEpWindowLayout layout{};
    OP_TILING_CHECK(CalcMoeEpWindowLayout(params, layout) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Calculate Moe EP window layout failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckMoeEpWindowCapacity(layout.requiredBytes, maxWindowSize, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check Moe EP window capacity failed."), return ge::GRAPH_FAILED);

    info.dumpMetadata = BuildMoeEpDumpMetadata(params, layout, info.aivNum);
    info.winDataOffset = layout.winDataOffset;
    info.slotWinStateOffset = layout.slotWinStateOffset;
    info.dispatchNotifyCount = layout.dispatchNotifyCount;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpDispatchEpilogueTilingFunc(gert::TilingContext *context)
{
    context->SetScheduleMode(1U);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueTilingData *tilingData = context->GetTilingData<MoeEpDispatchEpilogueTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueInfo &info = tilingData->moeEpDispatchEpilogueInfo;
    uint32_t networkMode = NETWORK_DIRECT;
    uint32_t serverNum = 1U;
    uint32_t rankNumPerServer = 1U;

    OP_TILING_CHECK(
        CheckAttrParams(context, nodeName, info, networkMode, serverNum, rankNumPerServer) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);

    bool cached = (context->GetInputShape(CACHED_RECV_SRC_METADATA_INDEX) != nullptr);
    info.cached = cached ? 1U : 0U;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto hasTopkWeightsPtr = attrs->GetAttrPointer<bool>(ATTR_HAS_TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(hasTopkWeightsPtr == nullptr, OP_LOGE(nodeName, "hasTopkWeightsPtr is null."),
                    return ge::GRAPH_FAILED);
    bool hasTopkWeights = *hasTopkWeightsPtr;

    OP_TILING_CHECK(CheckInputDataType(context, nodeName, cached) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input dtype failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, info, cached) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    const gert::StorageShape *dstBufferSlotIdxShape = context->GetInputShape(DST_BUFFER_SLOT_IDX_INDEX);
    OP_TILING_CHECK(dstBufferSlotIdxShape == nullptr, OP_LOGE(nodeName, "dst_buffer_slot_idx shape is nullptr."),
                    return ge::GRAPH_FAILED);
    info.cfg.numTokens = static_cast<uint32_t>(dstBufferSlotIdxShape->GetStorageShape().GetDim(0));
    info.cfg.topK = static_cast<uint32_t>(dstBufferSlotIdxShape->GetStorageShape().GetDim(1));
    const int64_t topK = static_cast<int64_t>(info.cfg.topK);

    OP_TILING_CHECK(CheckOutputTensors(context, nodeName, info, topK, hasTopkWeights) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check output tensors failed."), return ge::GRAPH_FAILED);

    auto recvXShape = context->GetOutputShape(OUT_RECV_X_INDEX);
    info.cfg.hidden = static_cast<uint32_t>(recvXShape->GetStorageShape().GetDim(1));

    auto recvXDesc = context->GetOutputDesc(OUT_RECV_X_INDEX);
    OP_TILING_CHECK(recvXDesc == nullptr, OP_LOGE(nodeName, "recv_x desc is nullptr."), return ge::GRAPH_FAILED);
    ge::DataType recvXDtype = recvXDesc->GetDataType();
    bool isFp8 = (recvXDtype == ge::DT_FLOAT8_E5M2 || recvXDtype == ge::DT_FLOAT8_E4M3FN);
    uint32_t recvXDtypeSize = isFp8 ? FP8_DTYPE_SIZE : MAX_OUT_DTYPE_SIZE;
    uint32_t hAlign32 = ((info.cfg.hidden * recvXDtypeSize + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t kAlign32 = ((info.cfg.topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t scalesSizeAlign32 = isFp8 ? ((info.cfg.scalesBytes + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN : 0;
    info.cfg.perSlotBytes =
        ((hAlign32 + scalesSizeAlign32 + kAlign32 * 2 + UB_ALIGN + WIN_ADDR_ALIGN - 1) / WIN_ADDR_ALIGN) *
        WIN_ADDR_ALIGN;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_TILING_CHECK(aivNum == 0U, OP_LOGE(nodeName, "Platform reports aiv_num=0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ubSize == 0UL, OP_LOGE(nodeName, "Platform reports ub_size=0."), return ge::GRAPH_FAILED);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_TILING_CHECK(BuildAndCheckWindowLayout(context, nodeName, info, networkMode, serverNum, rankNumPerServer) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check HCCL Window size failed."), return ge::GRAPH_FAILED);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    uint32_t numLocalExpertsAlign8 = (info.cfg.numLocalExperts + 7) / 8 * 8;
    uint64_t hitCountBytes = static_cast<uint64_t>(aivNum) * numLocalExpertsAlign8 * sizeof(int32_t);
    uint64_t hitCountBytesAlign512 = ((hitCountBytes + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + hitCountBytesAlign512;

    uint64_t tilingKey =
        GET_TPL_TILING_KEY(TILINGKEY_TPL_A5, cached ? 1U : 0U, hasTopkWeights ? 1U : 0U, info.isMxQuant ? 1U : 0U);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u, cached=%u, hasTopkWeights=%u, isMxQuant=%u", tilingKey,
            blockDim, aivNum, info.cached, hasTopkWeights ? 1U : 0U, info.isMxQuant);

    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

struct MoeEpDispatchEpilogueCompileInfo {};
ge::graphStatus TilingParseForMoeEpDispatchEpilogue(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatchEpilogue)
    .Tiling(MoeEpDispatchEpilogueTilingFunc)
    .TilingParse<MoeEpDispatchEpilogueCompileInfo>(TilingParseForMoeEpDispatchEpilogue);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
inline void MoeEpDispatchEpilogueExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
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
