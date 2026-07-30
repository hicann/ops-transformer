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
 * \file moe_ep_dispatch_tiling.cpp
 * \brief
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_dispatch_tiling.h"
#include "../../op_kernel/moe_ep_dispatch_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace Mc2Tiling {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t X_INDEX = 1U;
constexpr uint32_t TOPK_IDX_INDEX = 2U;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 3U;
constexpr uint32_t SCALES_INDEX = 4U;
constexpr uint32_t CACHED_SLOT_IDX_INDEX = 5U;

constexpr uint32_t NUM_RECV_PER_RANK_INDEX = 0U;
constexpr uint32_t NUM_RECV_PER_EXPERT_INDEX = 1U;
constexpr uint32_t DST_BUFFER_SLOT_IDX_INDEX = 2U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;
constexpr uint32_t ATTR_EXPERT_ALIGNMENT_INDEX = 5;
constexpr uint32_t ATTR_DO_CPU_SYNC_INDEX = 6;
constexpr uint32_t ATTR_HOST_PINNED_COUNTER_ADDR_INDEX = 7;

constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr int64_t H_MAX = 8192;
constexpr int64_t K_MAX = 32;
constexpr int64_t MAX_EP_WORLD_SIZE = 128;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_NUM_EXPERTS = 2048;
constexpr int64_t MIN_NUM_EXPERTS = 2;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t FP8_DTYPE_SIZE = 1UL;
constexpr uint64_t METADATA_DTYPE_SIZE = 4UL; // sizeof(int32)=sizeof(float)=4
constexpr int64_t SCALES_GROUP_SIZE_MXFP = 32;
constexpr int64_t SCALES_GROUP_SIZE_PERGROUP = 128;
constexpr int64_t SCALES_ALIGN_EVEN = 2; // fp8 align 2

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t UB_ALIGN = 32UL;

constexpr uint32_t DIRECT_MODE = 0;
constexpr uint32_t HYBRID_MODE = 1;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpDispatchInfo &info)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", info.cfg.epWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", info.cfg.epRankId);
    OP_LOGD(nodeName, "numExperts is %u.", info.cfg.numExperts);
    OP_LOGD(nodeName, "numLocalExperts is %u.", info.cfg.numLocalExperts);
    OP_LOGD(nodeName, "numTokens is %u.", info.cfg.numTokens);
    OP_LOGD(nodeName, "hidden is %u.", info.cfg.hidden);
    OP_LOGD(nodeName, "topK is %u.", info.cfg.topK);
    OP_LOGD(nodeName, "numMaxTokensPerRank is %u.", info.cfg.numMaxTokensPerRank);
    OP_LOGD(nodeName, "perSlotBytes is %u.", info.cfg.perSlotBytes);
    OP_LOGD(nodeName, "expertAlignment is %u.", info.cfg.expertAlignment);
    OP_LOGD(nodeName, "doCpuSync is %u.", info.doCpuSync);
    OP_LOGD(nodeName, "isCached is %u.", info.isCached);
    OP_LOGD(nodeName, "isTopkWeights is %u.", info.isTopkWeights);
    OP_LOGD(nodeName, "networkMode is %u.", info.networkMode);
    OP_LOGD(nodeName, "numScaleupRanks is %u.", info.numScaleupRanks);
    OP_LOGD(nodeName, "numScaleoutRanks is %u.", info.numScaleoutRanks);
    OP_LOGD(nodeName, "numAivStage1 is %u.", info.numAivStage1);
    OP_LOGD(nodeName, "numAivStage2 is %u.", info.numAivStage2);
    OP_LOGD(nodeName, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName, "hostPinnedCounterAddr is %lu.", info.hostPinnedCounterAddr);
    OP_LOGD(nodeName, "dispatchSendWinOffset is %lu.", info.dispatchSendWinOffset);
    OP_LOGD(nodeName, "totalWinSizeEp is %lu.", info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalUbSize is %lu.", info.totalUbSize);
}

static bool CheckInputTensorShape(const gert::TilingContext *context, const char *nodeName, MoeEpDispatchInfo &info)
{
    const gert::StorageShape *contextShape = context->GetInputShape(CONTEXT_INDEX);
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *topkIdxShape = context->GetInputShape(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);
    OP_TILING_CHECK(
        contextShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "context dims must be 1, but got %lu.", contextShape->GetStorageShape().GetDimNum()),
        return false);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "x dims must be 2, but got %lu.", xShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_TILING_CHECK(
        topkIdxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topk_idx dims must be 2, but got %lu.", topkIdxShape->GetStorageShape().GetDimNum()),
        return false);

    int64_t xDim0 = xShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xShape->GetStorageShape().GetDim(1);
    int64_t topkDim0 = topkIdxShape->GetStorageShape().GetDim(0);
    int64_t topkDim1 = topkIdxShape->GetStorageShape().GetDim(1);
    int64_t numMaxTokensPerRank = static_cast<int64_t>(info.cfg.numMaxTokensPerRank);
    int64_t numExperts = static_cast<int64_t>(info.cfg.numExperts);
    OP_TILING_CHECK(
        (xDim0 <= 0) || (xDim0 > numMaxTokensPerRank),
        OP_LOGE(nodeName, "x dim0 is invalid, should be in [1, %ld], but got %ld.", numMaxTokensPerRank, xDim0),
        return false);
    OP_TILING_CHECK((xDim1 <= 0) || (xDim1 > H_MAX),
                    OP_LOGE(nodeName, "x dim1(hidden) is invalid, should be in (0, %ld], but got %ld.", H_MAX, xDim1),
                    return false);
    OP_TILING_CHECK(
        xDim0 != topkDim0,
        OP_LOGE(nodeName, "topk_idx dim0 must equal x dim0, but got x dim0=%ld, topk_idx dim0=%ld.", xDim0, topkDim0),
        return false);
    OP_TILING_CHECK(
        (topkDim1 <= 0) || (topkDim1 > K_MAX) || (topkDim1 > numExperts),
        OP_LOGE(nodeName, "topk_idx dim1(topK) is invalid, should be in (0, min(%ld, num_experts=%ld)], but got %ld.",
                K_MAX, numExperts, topkDim1),
        return false);

    info.cfg.numTokens = static_cast<uint32_t>(xDim0);
    info.cfg.hidden = static_cast<uint32_t>(xDim1);
    info.cfg.topK = static_cast<uint32_t>(topkDim1);
    return true;
}

static bool CheckOptionalTensorShape(const gert::TilingContext *context, const char *nodeName, int64_t topkDim0,
                                     int64_t topkDim1)
{
    const gert::StorageShape *weightsShape = context->GetOptionalInputShape(TOPK_WEIGHTS_INDEX);
    const gert::StorageShape *cachedShape = context->GetOptionalInputShape(CACHED_SLOT_IDX_INDEX);

    if (weightsShape != nullptr) {
        OP_TILING_CHECK(
            weightsShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "topk_weights dims must be 2, but got %lu.", weightsShape->GetStorageShape().GetDimNum()),
            return false);
        OP_TILING_CHECK(
            weightsShape->GetStorageShape().GetDim(0) != topkDim0,
            OP_LOGE(nodeName,
                    "topk_weights dim0 must equal topk_idx dim0, but got topk_weights dim0=%ld, topk_idx dim0=%ld.",
                    weightsShape->GetStorageShape().GetDim(0), topkDim0),
            return false);
        OP_TILING_CHECK(
            weightsShape->GetStorageShape().GetDim(1) != topkDim1,
            OP_LOGE(nodeName,
                    "topk_weights dim1 must equal topk_idx dim1, but got topk_weights dim1=%ld, topk_idx dim1=%ld.",
                    weightsShape->GetStorageShape().GetDim(1), topkDim1),
            return false);
    }

    if (cachedShape != nullptr) {
        OP_TILING_CHECK(cachedShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE(nodeName, "cached_slot_idx dims must be 2, but got %lu.",
                                cachedShape->GetStorageShape().GetDimNum()),
                        return false);
        OP_TILING_CHECK(
            cachedShape->GetStorageShape().GetDim(0) != topkDim0,
            OP_LOGE(
                nodeName,
                "cached_slot_idx dim0 must equal topk_idx dim0, but got cached_slot_idx dim0=%ld, topk_idx dim0=%ld.",
                cachedShape->GetStorageShape().GetDim(0), topkDim0),
            return false);
        OP_TILING_CHECK(
            cachedShape->GetStorageShape().GetDim(1) != topkDim1,
            OP_LOGE(
                nodeName,
                "cached_slot_idx dim1 must equal topk_idx dim1, but got cached_slot_idx dim1=%ld, topk_idx dim1=%ld.",
                cachedShape->GetStorageShape().GetDim(1), topkDim1),
            return false);
    }

    return true;
}

static bool CheckInputTensorScales(const gert::TilingContext *context, const char *nodeName, MoeEpDispatchInfo &info,
                                   const bool isXFp8)
{
    const gert::StorageShape *scalesShape = context->GetOptionalInputShape(SCALES_INDEX);
    OP_TILING_CHECK(isXFp8 && (scalesShape == nullptr),
                    OP_LOGE(nodeName, "scales is required when x is fp8, but not provided."), return false);
    OP_TILING_CHECK(!isXFp8 && (scalesShape != nullptr), OP_LOGE(nodeName, "scales is only valid when x is fp8."),
                    return false);

    if (scalesShape != nullptr) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        // check dtype
        ge::DataType scalesDtype = scalesDesc->GetDataType();
        OP_TILING_CHECK((scalesDtype != ge::DT_FLOAT) && (scalesDtype != ge::DT_FLOAT8_E8M0),
                        OP_LOGE(nodeName, "scales dtype must be DT_FLOAT or DT_FLOAT8_E8M0, but got %s.",
                                ge::TypeUtils::DataTypeToSerialString(scalesDtype).c_str()),
                        return false);
        // check format
        auto scalesFormat = ge::GetPrimaryFormat(scalesDesc->GetStorageFormat());
        OP_TILING_CHECK(static_cast<ge::Format>(scalesFormat) == ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "scales format is invalid."), return false);

        // check shape
        OP_TILING_CHECK(
            scalesShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "scales dims must be 2, but got %lu.", scalesShape->GetStorageShape().GetDimNum()),
            return false);
        int64_t scalesDim0 = scalesShape->GetStorageShape().GetDim(0);
        int64_t scalesDim1 = scalesShape->GetStorageShape().GetDim(1);
        int64_t groupSize = (scalesDtype == ge::DT_FLOAT) ? SCALES_GROUP_SIZE_PERGROUP : SCALES_GROUP_SIZE_MXFP;
        int64_t expectedDim1 = (static_cast<int64_t>(info.cfg.hidden) + groupSize - 1) / groupSize;
        if (scalesDtype == ge::DT_FLOAT8_E8M0) {
            expectedDim1 = (expectedDim1 + SCALES_ALIGN_EVEN - 1) / SCALES_ALIGN_EVEN * SCALES_ALIGN_EVEN;
        }
        OP_TILING_CHECK(scalesDim0 != static_cast<int64_t>(info.cfg.numTokens),
                        OP_LOGE(nodeName, "scales dim0 must equal x dim0, but got scales dim0=%ld, x dim0=%u.",
                                scalesDim0, info.cfg.numTokens),
                        return false);
        OP_TILING_CHECK(
            scalesDim1 != expectedDim1,
            OP_LOGE(nodeName, "scales dim1 is invalid, expected %ld but got %ld.", expectedDim1, scalesDim1),
            return false);

        uint32_t scalesSize = scalesDtype == ge::DT_FLOAT ? sizeof(float) : FP8_DTYPE_SIZE;
        info.cfg.scalesBytes = static_cast<uint32_t>(expectedDim1 * scalesSize);
        info.isMxQuant = (scalesDtype == ge::DT_FLOAT8_E8M0) ? 1U : 0U;
    }
    return true;
}

static bool CheckOutputTensorShape(const gert::TilingContext *context, const char *nodeName,
                                   const MoeEpDispatchInfo &info, int64_t topkDim0, int64_t topkDim1)
{
    const gert::StorageShape *recvPerRankShape = context->GetOutputShape(NUM_RECV_PER_RANK_INDEX);
    const gert::StorageShape *recvPerExpertShape = context->GetOutputShape(NUM_RECV_PER_EXPERT_INDEX);
    const gert::StorageShape *dstSlotIdxShape = context->GetOutputShape(DST_BUFFER_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvPerRankShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvPerExpertShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSlotIdxShape);
    OP_TILING_CHECK(recvPerRankShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank dims must be 1, but got %lu.",
                            recvPerRankShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_TILING_CHECK(recvPerRankShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.cfg.epWorldSize),
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank dim0 must equal ep_world_size=%u, but got %ld.",
                            info.cfg.epWorldSize, recvPerRankShape->GetStorageShape().GetDim(0)),
                    return false);
    OP_TILING_CHECK(recvPerExpertShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dims must be 1, but got %lu.",
                            recvPerExpertShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_TILING_CHECK(recvPerExpertShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.cfg.numLocalExperts),
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dim0 must equal num_local_experts=%u, but got %ld.",
                            info.cfg.numLocalExperts, recvPerExpertShape->GetStorageShape().GetDim(0)),
                    return false);
    OP_TILING_CHECK(dstSlotIdxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "dst_buffer_slot_idx dims must be 2, but got %lu.",
                            dstSlotIdxShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_TILING_CHECK(dstSlotIdxShape->GetStorageShape().GetDim(0) != topkDim0,
                    OP_LOGE(nodeName,
                            "dst_buffer_slot_idx dim0 must equal topk_idx dim0, but got dst_buffer_slot_idx dim0=%ld, "
                            "topk_idx dim0=%ld.",
                            dstSlotIdxShape->GetStorageShape().GetDim(0), topkDim0),
                    return false);
    OP_TILING_CHECK(dstSlotIdxShape->GetStorageShape().GetDim(1) != topkDim1,
                    OP_LOGE(nodeName,
                            "dst_buffer_slot_idx dim1 must equal topk_idx dim1, but got dst_buffer_slot_idx dim1=%ld, "
                            "topk_idx dim1=%ld.",
                            dstSlotIdxShape->GetStorageShape().GetDim(1), topkDim1),
                    return false);
    return true;
}

static bool CheckInputTensorDtype(const gert::TilingContext *context, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    auto xDesc = context->GetInputDesc(X_INDEX);
    auto topkIdxDesc = context->GetInputDesc(TOPK_IDX_INDEX);
    auto topkWeightsDesc = context->GetOptionalInputDesc(TOPK_WEIGHTS_INDEX);
    auto cachedSlotIdxDesc = context->GetOptionalInputDesc(CACHED_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxDesc);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "context dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(contextDesc->GetDataType()).c_str()),
                    return false);
    ge::DataType xDtype = xDesc->GetDataType();
    OP_TILING_CHECK(
        (xDtype != ge::DT_BF16) && (xDtype != ge::DT_FLOAT16) && (xDtype != ge::DT_FLOAT8_E5M2) &&
            (xDtype != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE(nodeName,
                "x dtype must be in support list [DT_BF16, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN], but got %s.",
                ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return false);

    OP_TILING_CHECK(topkIdxDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "topk_idx dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(topkIdxDesc->GetDataType()).c_str()),
                    return false);

    if (topkWeightsDesc != nullptr) {
        OP_TILING_CHECK(topkWeightsDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(nodeName, "topk_weights dtype must be DT_FLOAT, but got %s.",
                                ge::TypeUtils::DataTypeToSerialString(topkWeightsDesc->GetDataType()).c_str()),
                        return false);
    }
    if (cachedSlotIdxDesc != nullptr) {
        OP_TILING_CHECK(cachedSlotIdxDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE(nodeName, "cached_slot_idx dtype must be DT_INT32, but got %s.",
                                ge::TypeUtils::DataTypeToSerialString(cachedSlotIdxDesc->GetDataType()).c_str()),
                        return false);
    }

    return true;
}

static bool CheckOutputTensorDtype(const gert::TilingContext *context, const char *nodeName)
{
    auto numRecvPerRankDesc = context->GetOutputDesc(NUM_RECV_PER_RANK_INDEX);
    auto numRecvPerExpertDesc = context->GetOutputDesc(NUM_RECV_PER_EXPERT_INDEX);
    auto dstSlotIdxDesc = context->GetOutputDesc(DST_BUFFER_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerRankDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpertDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSlotIdxDesc);
    OP_TILING_CHECK(numRecvPerRankDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(numRecvPerRankDesc->GetDataType()).c_str()),
                    return false);
    OP_TILING_CHECK(numRecvPerExpertDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert dtype must be DT_INT64, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(numRecvPerExpertDesc->GetDataType()).c_str()),
                    return false);
    OP_TILING_CHECK(dstSlotIdxDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "dst_buffer_slot_idx dtype must be DT_INT32, but got %s.",
                            ge::TypeUtils::DataTypeToSerialString(dstSlotIdxDesc->GetDataType()).c_str()),
                    return false);
    return true;
}

static ge::graphStatus CheckInputTensorFormat(const gert::TilingContext *context, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    auto xDesc = context->GetInputDesc(X_INDEX);
    auto topkIdxDesc = context->GetInputDesc(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxDesc);
    auto contextFormat = ge::GetPrimaryFormat(contextDesc->GetStorageFormat());
    auto xFormat = ge::GetPrimaryFormat(xDesc->GetStorageFormat());
    auto topkIdxFormat = ge::GetPrimaryFormat(topkIdxDesc->GetStorageFormat());
    OP_TILING_CHECK(static_cast<ge::Format>(contextFormat) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "context format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(xFormat) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(topkIdxFormat) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "topk_idx format is invalid."), return ge::GRAPH_FAILED);
    auto topkWeightsDesc = context->GetOptionalInputDesc(TOPK_WEIGHTS_INDEX);
    if (topkWeightsDesc != nullptr) {
        auto topkWeightsFormat = ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat());
        OP_TILING_CHECK(static_cast<ge::Format>(topkWeightsFormat) == ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "topk_weights format is invalid."), return ge::GRAPH_FAILED);
    }
    auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
    if (scalesDesc != nullptr) {
        auto scalesFormat = ge::GetPrimaryFormat(scalesDesc->GetStorageFormat());
        OP_TILING_CHECK(static_cast<ge::Format>(scalesFormat) == ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "scales format is invalid."), return ge::GRAPH_FAILED);
    }
    auto cachedSlotIdxDesc = context->GetOptionalInputDesc(CACHED_SLOT_IDX_INDEX);
    if (cachedSlotIdxDesc != nullptr) {
        auto cachedSlotIdxFormat = ge::GetPrimaryFormat(cachedSlotIdxDesc->GetStorageFormat());
        OP_TILING_CHECK(static_cast<ge::Format>(cachedSlotIdxFormat) == ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE(nodeName, "cached_slot_idx format is invalid."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorFormat(const gert::TilingContext *context, const char *nodeName)
{
    auto numRecvPerRankDesc = context->GetOutputDesc(NUM_RECV_PER_RANK_INDEX);
    auto numRecvPerExpertDesc = context->GetOutputDesc(NUM_RECV_PER_EXPERT_INDEX);
    auto dstSlotIdxDesc = context->GetOutputDesc(DST_BUFFER_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerRankDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpertDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstSlotIdxDesc);
    auto numRecvPerRankFormat = ge::GetPrimaryFormat(numRecvPerRankDesc->GetStorageFormat());
    auto numRecvPerExpertFormat = ge::GetPrimaryFormat(numRecvPerExpertDesc->GetStorageFormat());
    auto dstSlotIdxFormat = ge::GetPrimaryFormat(dstSlotIdxDesc->GetStorageFormat());
    OP_TILING_CHECK(static_cast<ge::Format>(numRecvPerRankFormat) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "num_recv_tokens_per_rank format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(numRecvPerExpertFormat) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "num_recv_tokens_per_expert format is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(dstSlotIdxFormat) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "dst_buffer_slot_idx format is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputTensor(const gert::TilingContext *context, const char *nodeName,
                                        MoeEpDispatchInfo &info)
{
    OP_TILING_CHECK(!CheckInputTensorShape(context, nodeName, info),
                    OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckOptionalTensorShape(context, nodeName, info.cfg.numTokens, info.cfg.topK),
                    OP_LOGE(nodeName, "Check optional input tensor shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputTensorDtype(context, nodeName), OP_LOGE(nodeName, "Check input tensor dtype failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckInputTensorFormat(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor format failed."), return ge::GRAPH_FAILED);

    ge::DataType xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    bool isXFp8 = ((xDtype == ge::DT_FLOAT8_E5M2) || (xDtype == ge::DT_FLOAT8_E4M3FN));
    OP_TILING_CHECK(!CheckInputTensorScales(context, nodeName, info, isXFp8),
                    OP_LOGE(nodeName, "Check scales input failed."), return ge::GRAPH_FAILED);

    uint32_t xDtypeSize = isXFp8 ? FP8_DTYPE_SIZE : MAX_OUT_DTYPE_SIZE;
    uint32_t hAlign32 = ((info.cfg.hidden * xDtypeSize + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t kAlign32 = ((info.cfg.topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t scalesSizeAlign32 = isXFp8 ? ((info.cfg.scalesBytes + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN : 0;
    info.cfg.perSlotBytes =
        ((hAlign32 + scalesSizeAlign32 + kAlign32 * 2 + UB_ALIGN + WIN_ADDR_ALIGN - 1) / WIN_ADDR_ALIGN) *
        WIN_ADDR_ALIGN;
    info.isTopkWeights = (context->GetOptionalInputShape(TOPK_WEIGHTS_INDEX) != nullptr) ? 1 : 0;
    OP_LOGD(nodeName, "perSlotBytes = %u (hidden=%u)", info.cfg.perSlotBytes, info.cfg.hidden);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensor(const gert::TilingContext *context, const char *nodeName,
                                         const MoeEpDispatchInfo &info)
{
    OP_TILING_CHECK(!CheckOutputTensorShape(context, nodeName, info, info.cfg.numTokens, info.cfg.topK),
                    OP_LOGE(nodeName, "Check output tensor shape failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckOutputTensorDtype(context, nodeName), OP_LOGE(nodeName, "Check output tensor dtype failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputTensorFormat(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check output tensor format failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommAttr(const gert::TilingContext *context, const char *nodeName, MoeEpDispatchInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
                    return ge::GRAPH_FAILED);

    int64_t epWorldSize = *epWorldSizePtr;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
                    OP_LOGE(nodeName, "ep_world_size is invalid, should be in [%ld, %ld], but got %ld.",
                            MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "ep_rank_id is invalid, should be in [0, %ld), but got %ld.", epWorldSize, *epRankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*cclBufferSizePtr <= 0,
                    OP_LOGE(nodeName, "ccl_buffer_size must be positive, but got %ld.", *cclBufferSizePtr),
                    return ge::GRAPH_FAILED);

    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckComputeAttr(const gert::TilingContext *context, const char *nodeName,
                                        MoeEpDispatchInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto expertAlignmentPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_ALIGNMENT_INDEX);
    auto hpCounterAddrPtr = attrs->GetAttrPointer<int64_t>(ATTR_HOST_PINNED_COUNTER_ADDR_INDEX);
    auto doCpuSyncPtr = attrs->GetAttrPointer<bool>(ATTR_DO_CPU_SYNC_INDEX);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nmtPtr == nullptr, OP_LOGE(nodeName, "numMaxTokensPerRankPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertAlignmentPtr == nullptr, OP_LOGE(nodeName, "expertAlignmentPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hpCounterAddrPtr == nullptr, OP_LOGE(nodeName, "hpCounterAddrPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(doCpuSyncPtr == nullptr, OP_LOGE(nodeName, "doCpuSyncPtr is null."), return ge::GRAPH_FAILED);

    int64_t epWorldSize = static_cast<int64_t>(info.cfg.epWorldSize);
    OP_TILING_CHECK(
        (*numExpertsPtr < MIN_NUM_EXPERTS) || (*numExpertsPtr > MAX_NUM_EXPERTS) || (*numExpertsPtr % epWorldSize != 0),
        OP_LOGE(nodeName,
                "num_experts is invalid, should be in [%ld, %ld] and divisible by ep_world_size, but got "
                "num_experts=%ld, ep_world_size=%ld.",
                MIN_NUM_EXPERTS, MAX_NUM_EXPERTS, *numExpertsPtr, epWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*nmtPtr <= 0),
                    OP_LOGE(nodeName, "num_max_tokens_per_rank must be positive, but got %ld.", *nmtPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertAlignmentPtr != 1),
                    OP_LOGE(nodeName, "expert_alignment must be 1, but got %ld.", *expertAlignmentPtr),
                    return ge::GRAPH_FAILED);

    bool cached = (context->GetOptionalInputShape(CACHED_SLOT_IDX_INDEX) != nullptr);
    OP_TILING_CHECK(cached && *doCpuSyncPtr,
                    OP_LOGE(nodeName, "do_cpu_sync and cached can't be true at the same time."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cached && (context->GetOptionalInputShape(TOPK_WEIGHTS_INDEX) != nullptr),
                    OP_LOGE(nodeName, "topk_weights is not supported when cached."), return ge::GRAPH_FAILED);

    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);
    info.cfg.expertAlignment = static_cast<uint32_t>(*expertAlignmentPtr);
    info.hostPinnedCounterAddr = static_cast<uint64_t>(*hpCounterAddrPtr);
    info.doCpuSync = (*doCpuSyncPtr && !cached) ? 1 : 0;
    info.isCached = cached ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, const char *nodeName,
                                       MoeEpDispatchInfo &info)
{
    OP_TILING_CHECK(CheckCommAttr(context, nodeName, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check comm attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckComputeAttr(context, nodeName, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check compute attr failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t AlignUpWin(const uint64_t data)
{
    return (data + WIN_ADDR_ALIGN - 1) / WIN_ADDR_ALIGN * WIN_ADDR_ALIGN;
}

static uint64_t CalcDispatchWorkspace(const MoeEpDispatchInfo &info)
{
    uint64_t epWorldSize = static_cast<uint64_t>(info.cfg.epWorldSize);
    uint64_t moeExpertNumPerRank = static_cast<uint64_t>(info.cfg.numLocalExperts);
    uint64_t aivNum = static_cast<uint64_t>(info.aivNum);
    uint64_t counterBytes = aivNum * AlignUpWin(epWorldSize * sizeof(int32_t));
    uint64_t sendCntPerRankBytes = epWorldSize * WIN_ADDR_ALIGN;
    uint64_t sendCntPerExpertBytes = AlignUpWin(moeExpertNumPerRank * epWorldSize * sizeof(int32_t));

    uint64_t sendCntBytes = counterBytes + sendCntPerRankBytes + sendCntPerExpertBytes;
    uint64_t globalABytes = UB_ALIGN;
    return SYSTEM_NEED_WORKSPACE + sendCntBytes + globalABytes;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeEpDispatchInfo &info, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
                    return ge::GRAPH_FAILED);
    uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    uint64_t epWorldSize = static_cast<uint64_t>(info.cfg.epWorldSize);
    uint64_t nmt = static_cast<uint64_t>(info.cfg.numMaxTokensPerRank);
    uint64_t moeExpertNumPerRank = static_cast<uint64_t>(info.cfg.numLocalExperts);
    uint64_t topK = static_cast<uint64_t>(info.cfg.topK);

    uint64_t cntWinStateSize =
        epWorldSize * AlignUpWin(moeExpertNumPerRank * sizeof(int32_t)) + epWorldSize * WIN_ADDR_ALIGN;
    uint64_t dispatchWinStateSize = cntWinStateSize + epWorldSize * WIN_ADDR_ALIGN;
    uint64_t combineWinStateSize = nmt * topK * WIN_ADDR_ALIGN + epWorldSize * WIN_ADDR_ALIGN;
    uint64_t hiddenAlign = (info.cfg.hidden * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint64_t topKAlign = (topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint64_t dispatchReservedPerSlotBytes = AlignUpWin(hiddenAlign + topKAlign * 2 + UB_ALIGN);
    uint64_t dispatchRecvWinDataReservedSize = epWorldSize * nmt * dispatchReservedPerSlotBytes;
    uint64_t combineWinDataSize = nmt * topK * AlignUpWin(static_cast<uint64_t>(hiddenAlign + UB_ALIGN));

    uint64_t totalStateWinSizeEp = dispatchWinStateSize + combineWinStateSize;
    uint64_t stateAndRecvDataWinSize = dispatchRecvWinDataReservedSize + combineWinDataSize + totalStateWinSizeEp;
    uint64_t dispatchSendWinDataReservedSize = dispatchRecvWinDataReservedSize;
    uint64_t winNeed = stateAndRecvDataWinSize + dispatchSendWinDataReservedSize;
    OP_TILING_CHECK(winNeed > maxWindowSize,
                    OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, need %luMB, available %luMB.",
                            (winNeed / MB_SIZE) + 1UL, maxWindowSize / MB_SIZE),
                    return ge::GRAPH_FAILED);
    info.totalWinSizeEp = maxWindowSize;
    info.cntWinStateOffset = 0UL;
    info.slotWinStateOffset = cntWinStateSize;
    info.winDataOffset = totalStateWinSizeEp;
    info.dispatchSendWinOffset = stateAndRecvDataWinSize;
    OP_LOGD(nodeName,
            "windowSize=%lu, perSlotBytes=%u, dispatchReservedPerSlotBytes=%lu, stateAndRecvDataWinSize=%lu, "
            "dispatchSendWinOffset=%lu, dispatchSendWinDataReservedSize=%lu",
            maxWindowSize, info.cfg.perSlotBytes, dispatchReservedPerSlotBytes, stateAndRecvDataWinSize,
            info.dispatchSendWinOffset, dispatchSendWinDataReservedSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const MoeEpDispatchInfo &info, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = CalcDispatchWorkspace(info);
    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const uint32_t doCpuSync, const uint32_t isCached, const uint32_t isTopkWeights,
                             const uint32_t isMxQuant, const uint32_t networkMode)
{
    bool cpuSyncMode = false;
    bool CachedMode = false;
    bool topkWeightsMode = false;
    bool mxQuantMode = false;
    if (doCpuSync) {
        cpuSyncMode = true;
    }
    if (isCached) {
        CachedMode = true;
    }
    if (isTopkWeights) {
        topkWeightsMode = true;
    }
    if (isMxQuant) {
        mxQuantMode = true;
    }

    return GET_TPL_TILING_KEY(cpuSyncMode, CachedMode, topkWeightsMode, mxQuantMode, static_cast<uint8_t>(networkMode));
}

static void SetPlatformAndNetworkInfo(gert::TilingContext *context, MoeEpDispatchInfo &info, const char *nodeName)
{
    uint32_t epWorldSize = info.cfg.epWorldSize;
    info.networkMode = (epWorldSize > 128) ? 1 : 0;
    info.numScaleupRanks = epWorldSize;
    info.numScaleoutRanks = 1;
    info.numAivStage1 = 0;
    info.numAivStage2 = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1U);
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
}

static ge::graphStatus MoeEpDispatchTilingFunc(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);
    MoeEpDispatchTilingData *tilingData = context->GetTilingData<MoeEpDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter MoeEpDispatch tiling func.");
    MoeEpDispatchInfo &info = tilingData->moeEpDispatchInfo;

    OP_TILING_CHECK(CheckAttrParams(context, nodeName, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckInputTensor(context, nodeName, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputTensor(context, nodeName, info) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check output tensor failed."), return ge::GRAPH_FAILED);

    SetPlatformAndNetworkInfo(context, info, nodeName);
    OP_TILING_CHECK(CheckWinSize(context, info, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, info, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Set workspace failed."), return ge::GRAPH_FAILED);

    uint64_t tilingKey =
        CalTilingKey(info.doCpuSync, info.isCached, info.isTopkWeights, info.isMxQuant, info.networkMode);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    PrintTilingDataInfo(nodeName, info);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatch).Tiling(MoeEpDispatchTilingFunc);
} // namespace Mc2Tiling
