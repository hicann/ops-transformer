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
 * \file moe_ep_combine_tiling.cpp
 * \brief
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
#include "../../op_kernel/moe_ep_combine_tiling.h"
#include "../../op_kernel/moe_ep_combine_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t RECVX_INDEX = 1U;
constexpr uint32_t TOPK_IDX_INDEX = 2U;
constexpr uint32_t RECV_SRC_METADATA_INDEX = 3U;
constexpr uint32_t NUM_RECV_PER_EXPERT_INDEX = 4U;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 5U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;
constexpr uint32_t ATTR_TOPO_TYPE_INDEX = 5;
constexpr uint32_t ATTR_RANK_NUM_PER_SERVER_INDEX = 6;

constexpr uint32_t ONE_DIMS = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr int64_t MAX_EP_WORLD_SIZE = 1024;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_NUM_EXPERTS = 2048;
constexpr int64_t MIN_NUM_EXPERTS = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t COMM_ALIGN = 512UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr int64_t H_MIN = 1;
constexpr int64_t H_MAX = 8192;
constexpr int64_t K_MAX = 32;
constexpr int64_t META_INNER_DIM = 4;
constexpr uint32_t NETWORK_DIRECT = 0U;
constexpr uint32_t NETWORK_HYBRID = 1U;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpCombineInfo &info)
{
    OP_LOGD(nodeName, "epWorldSize=%u, epRankId=%u, numExperts=%u, numLocalExperts=%u", info.cfg.epWorldSize,
            info.cfg.epRankId, info.cfg.numExperts, info.cfg.numLocalExperts);
    OP_LOGD(nodeName, "numTokens=%u, hidden=%u, topK=%u, numMaxTokensPerRank=%u", info.cfg.numTokens, info.cfg.hidden,
            info.cfg.topK, info.cfg.numMaxTokensPerRank);
    OP_LOGD(nodeName, "perSlotBytes=%u, hasTopkWeights=%u, aivNum=%u", info.cfg.perSlotBytes, info.hasTopkWeights,
            info.aivNum);
    OP_LOGD(nodeName,
            "totalWinSizeEp=%lu, combineStateWinOffset=%lu, combineDataWinOffset=%lu, "
            "sendDataWorkspaceSizePerRank=%lu, totalUbSize=%lu",
            info.totalWinSizeEp, info.combineStateWinOffset, info.combineDataWinOffset,
            info.sendDataWorkspaceSizePerRank, info.totalUbSize);
}

static ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context, const char *nodeName,
                                             MoeEpCombineInfo &info)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextStorageShape);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(nodeName, "context dims must be 1, but got %lu.", contextStorageShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *recvxShape = context->GetInputShape(RECVX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvxShape);
    OP_TILING_CHECK(recvxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "x dims must be 2, but got %lu.", recvxShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t recvxDim0 = recvxShape->GetStorageShape().GetDim(0);
    const int64_t recvxDim1 = recvxShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(recvxDim0 < 0, OP_LOGE(nodeName, "x dim0(A) must be positive, but got %ld.", recvxDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (recvxDim1 < H_MIN) || (recvxDim1 > H_MAX),
        OP_LOGE(nodeName, "x dim1(hidden) is invalid, should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, recvxDim1),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *topkIdxShape = context->GetInputShape(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);
    OP_TILING_CHECK(
        topkIdxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topk_idx dims must be 2, but got %lu.", topkIdxShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    const int64_t topkDim0 = topkIdxShape->GetStorageShape().GetDim(0);
    const int64_t topkDim1 = topkIdxShape->GetStorageShape().GetDim(1);
    int64_t numExperts = static_cast<int64_t>(info.cfg.numExperts);
    OP_TILING_CHECK(topkDim0 <= 0,
                    OP_LOGE(nodeName, "topk_idx dim0(num_tokens) must be positive, but got %ld.", topkDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (topkDim1 <= 0) || (topkDim1 > K_MAX) || (topkDim1 > numExperts),
        OP_LOGE(nodeName, "topk_idx dim1(top_k) is invalid, should be in (0, min(%ld, num_experts=%ld)], but got %ld.",
                K_MAX, numExperts, topkDim1),
        return ge::GRAPH_FAILED);

    int64_t numLocalExperts = static_cast<int64_t>(info.cfg.numLocalExperts);
    int64_t epWorldSize = static_cast<int64_t>(info.cfg.epWorldSize);
    int64_t nmt = static_cast<int64_t>(info.cfg.numMaxTokensPerRank);
    int64_t minTopKLocalExperts = (topkDim1 < numLocalExperts) ? topkDim1 : numLocalExperts;
    int64_t aAllocUpper = epWorldSize * nmt * minTopKLocalExperts;
    OP_TILING_CHECK(recvxDim0 > aAllocUpper,
                    OP_LOGE(nodeName,
                            "x dim0(A) must not exceed A_Upper=%ld (ep_world_size=%ld * num_max_tokens_per_rank=%ld * "
                            "min(top_k=%ld, num_local_experts=%ld)), but got %ld.",
                            aAllocUpper, epWorldSize, nmt, topkDim1, numLocalExperts, recvxDim0),
                    return ge::GRAPH_FAILED);

    info.cfg.hidden = static_cast<uint32_t>(recvxDim1);
    info.cfg.numTokens = static_cast<uint32_t>(topkDim0);
    info.cfg.topK = static_cast<uint32_t>(topkDim1);

    const gert::StorageShape *recvSrcMetadataShape = context->GetInputShape(RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetadataShape);
    OP_TILING_CHECK(recvSrcMetadataShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "recv_src_metadata dims must be 2, but got %lu.",
                            recvSrcMetadataShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t recvSrcMetadataDim0 = recvSrcMetadataShape->GetStorageShape().GetDim(0);
    const int64_t recvSrcMetadataDim1 = recvSrcMetadataShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(recvSrcMetadataDim0 != recvxDim0,
                    OP_LOGE(nodeName, "recv_src_metadata dim0 must equal x dim0(%ld), but got %ld.", recvxDim0,
                            recvSrcMetadataDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        recvSrcMetadataDim1 != META_INNER_DIM,
        OP_LOGE(nodeName, "recv_src_metadata dim1 must be %ld, but got %ld.", META_INNER_DIM, recvSrcMetadataDim1),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *numRecvPerExpertShape = context->GetInputShape(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpertShape);
    OP_TILING_CHECK(numRecvPerExpertShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dims must be 1, but got %lu.",
                            numRecvPerExpertShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        numRecvPerExpertShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.cfg.numLocalExperts),
        OP_LOGE(nodeName, "num_recv_tokens_per_expert dim0 must equal num_local_experts=%u, but got %ld.",
                info.cfg.numLocalExperts, numRecvPerExpertShape->GetStorageShape().GetDim(0)),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *topkWeightsShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    bool hasTopkWeights = (topkWeightsShape != nullptr);
    if (hasTopkWeights) {
        OP_TILING_CHECK(topkWeightsShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                        OP_LOGE(nodeName, "topk_weights dims must be 1, but got %lu.",
                                topkWeightsShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(topkWeightsShape->GetStorageShape().GetDim(0) != recvxDim0,
                        OP_LOGE(nodeName, "topk_weights dim0 must equal x dim0(%ld), but got %ld.", recvxDim0,
                                topkWeightsShape->GetStorageShape().GetDim(0)),
                        return ge::GRAPH_FAILED);
    }
    info.hasTopkWeights = hasTopkWeights ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDataType(const gert::TilingContext *context, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "context dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(contextDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto recvxDesc = context->GetInputDesc(RECVX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvxDesc);
    OP_TILING_CHECK(recvxDesc->GetDataType() != ge::DT_BF16 && recvxDesc->GetDataType() != ge::DT_FLOAT16,
                    OP_LOGE(nodeName, "x dtype must be DT_BF16 or DT_FLOAT16, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(recvxDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto topkIdxDesc = context->GetInputDesc(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxDesc);
    OP_TILING_CHECK(topkIdxDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "topk_idx dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(topkIdxDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto recvSrcMetadataDesc = context->GetInputDesc(RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetadataDesc);
    OP_TILING_CHECK(recvSrcMetadataDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "recv_src_metadata dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(recvSrcMetadataDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto numRecvPerExpertDesc = context->GetInputDesc(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpertDesc);
    OP_TILING_CHECK(numRecvPerExpertDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dtype must be DT_INT64, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(numRecvPerExpertDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    auto topkWeightsDesc = context->GetOptionalInputDesc(TOPK_WEIGHTS_INDEX);
    if (topkWeightsDesc != nullptr) {
        OP_TILING_CHECK(topkWeightsDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "topk_weights dtype must be DT_FLOAT, but got %s.",
                                ge::TypeUtils::DataTypeToSerialString(topkWeightsDesc->GetDataType()).c_str()),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, const char *nodeName, MoeEpCombineInfo &info,
                                       uint32_t &networkMode, uint32_t &serverNum, uint32_t &rankNumPerServerOut)
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

static ge::graphStatus BuildAndCheckWindowLayout(const gert::TilingContext *context, MoeEpCombineInfo &info,
                                                 uint32_t networkMode, uint32_t serverNum, uint32_t rankNumPerServer,
                                                 const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);

    const uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    const MoeEpWindowLayoutParams params = {info.cfg.epWorldSize,
                                            info.cfg.numLocalExperts,
                                            static_cast<uint32_t>(*nmtPtr),
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
    info.totalWinSizeEp = maxWindowSize;
    info.combineStateWinOffset = layout.combineStateWinOffset;
    info.combineDataWinOffset = layout.combineDataWinOffset;
    info.sendDataWorkspaceSizePerRank = layout.combineDataSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpCombineTilingFunc(gert::TilingContext *context)
{
    context->SetScheduleMode(1U);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);

    MoeEpCombineTilingData *tilingData = context->GetTilingData<MoeEpCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter MoeEpCombine tiling func.");

    MoeEpCombineInfo &info = tilingData->moeEpCombineInfo;
    uint32_t networkMode = NETWORK_DIRECT;
    uint32_t serverNum = 1U;
    uint32_t rankNumPerServer = 1U;

    OP_TILING_CHECK(
        CheckAttrParams(context, nodeName, info, networkMode, serverNum, rankNumPerServer) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputDataType(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor dtype failed."), return ge::GRAPH_FAILED);

    uint32_t hAlign32 = ((info.cfg.hidden * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    info.cfg.perSlotBytes = static_cast<uint32_t>(((hAlign32 + UB_ALIGN + COMM_ALIGN - 1UL) / COMM_ALIGN) * COMM_ALIGN);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_TILING_CHECK(BuildAndCheckWindowLayout(context, info, networkMode, serverNum, rankNumPerServer, nodeName) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check window size failed."), return ge::GRAPH_FAILED);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    uint64_t perCoreRankCountStride =
        ops::CeilDiv(static_cast<uint64_t>(info.cfg.epWorldSize) * UB_ALIGN, COMM_ALIGN) * COMM_ALIGN;
    // Address tables, per-AIV flags, rank totals, and one aligned per-rank count row for every AIV.
    workSpaces[0] =
        SYSTEM_NEED_WORKSPACE + static_cast<uint64_t>(info.cfg.epWorldSize) * info.sendDataWorkspaceSizePerRank +
        static_cast<uint64_t>(aivNum) * COMM_ALIGN + static_cast<uint64_t>(info.cfg.epWorldSize) * COMM_ALIGN +
        static_cast<uint64_t>(aivNum) * perCoreRankCountStride;

    uint32_t tplHasTopkWeights = info.hasTopkWeights ? 1 : 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(tplHasTopkWeights, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);

    OP_LOGD(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u", tilingKey, blockDim, aivNum);
    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

struct MoeEpCombineCompileInfo {};
ge::graphStatus TilingParseForMoeEpCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpCombine)
    .Tiling(MoeEpCombineTilingFunc)
    .TilingParse<MoeEpCombineCompileInfo>(TilingParseForMoeEpCombine);

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
inline void MoeEpCombineExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::MoeEpExceptionImpl(args, userdata, "MoeEpCombine");
}

__attribute__((constructor)) void RegisterMoeEpCombineExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;
    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS ||
        aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS ||
        runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeEpCombine", "Runtime or metadef does not support exception dump registration.");
        return;
    }
    IMPL_OP(MoeEpCombine).ExceptionDumpParseFunc(MoeEpCombineExceptionImplWrapper);
}
#endif
} // namespace
