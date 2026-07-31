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
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "../mega_moe.h"
#include "../../../op_kernel/arch35/mega_moe_tiling.h"
#include "../../../op_kernel/arch35/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch35/mega_moe_workspace_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
namespace {
const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t UB_BLOCK_SIZE = 32LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_TOPK = 1LL;
const static int64_t MAX_TOPK = 32LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = 1024LL;
const static int64_t MIN_H = 1024LL;
const static int64_t MAX_H = 8LL * 1024LL; // 8K
const static int64_t H_ALIGN = 32LL;
const static int64_t W4_K_ALIGN = 64LL;
const static int64_t HIDDEN_DIM_BASE = 1024LL;
const static int64_t MAX_HIDDEN_DIM = 8LL * 1024LL; // 8K
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_MOE_EXPERT_NUM = 2048LL;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
} // namespace

void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    OP_LOGD(nodeName, "shape: bs=%u, h=%u, hiddenDim=%u, topK=%u, maxOutputSize=%u", tilingData->bs, tilingData->h,
            tilingData->hiddenDim, tilingData->topK, tilingData->maxOutputSize);
    OP_LOGD(nodeName,
            "topology: expertPerRank=%u, moeExpertPerRank=%u, sharedExpertNum=%u, epWorldSize=%u, aicNum=%u, "
            "blockAivNum=%u, blockNumPerEP=%u, topoType=%ld",
            tilingData->expertPerRank, tilingData->moeExpertPerRank, tilingData->sharedExpertNum,
            tilingData->epWorldSize, tilingData->aicNum, tilingData->blockAivNum, tilingData->blockNumPerEP,
            tilingData->topoType);
    OP_LOGD(nodeName, "mode: groupedMatmulMode=%u, combineQuantMode=%ld, clampLimit=%f",
            static_cast<uint32_t>(tilingData->groupedMatmulMode), tilingData->combineQuantMode, tilingData->clampLimit);
    OP_LOGD(nodeName, "combineSync: slotCountPerExpert=%lu", tilingData->combineSyncSlotCountPerExpert);

    const auto &dispatchConfig = tilingData->dispatchBufferConfig;
    OP_LOGD(nodeName, "dispatch: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, copyBufferBytes=%u",
            dispatchConfig.routeItemsPerBatch, dispatchConfig.routeBatchCount, dispatchConfig.bufferCount,
            dispatchConfig.copyBufferBytes);

    const auto &sendMaskConfigWithExtraExpert = tilingData->sendMaskConfigForCoreWithExtraExpert;
    const auto &sendMaskConfigWithoutExtraExpert = tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    OP_LOGD(nodeName, "sendMask: coreCountWithExtraExpert=%u", tilingData->sendMaskCoreCountWithExtraExpert);
    OP_LOGD(nodeName,
            "sendMaskWithExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithExtraExpert.routeItemsPerBatch, sendMaskConfigWithExtraExpert.routeBatchCount,
            sendMaskConfigWithExtraExpert.bufferCount, sendMaskConfigWithExtraExpert.bufferBytes);
    OP_LOGD(nodeName,
            "sendMaskWithoutExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithoutExtraExpert.routeItemsPerBatch, sendMaskConfigWithoutExtraExpert.routeBatchCount,
            sendMaskConfigWithoutExtraExpert.bufferCount, sendMaskConfigWithoutExtraExpert.bufferBytes);

    const auto &unpermuteFullChunkConfig = tilingData->unpermuteConfigForFullTokenChunk;
    const auto &unpermuteTailChunkConfig = tilingData->unpermuteConfigForTailTokenChunk;
    OP_LOGD(nodeName, "unpermute: fullTokenChunkCoreCount=%u", tilingData->unpermuteFullTokenChunkCoreCount);
    OP_LOGD(nodeName,
            "unpermuteFullChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteFullChunkConfig.tokensPerBatch, unpermuteFullChunkConfig.inputBufferCount,
            unpermuteFullChunkConfig.bf16SlotElementCount, unpermuteFullChunkConfig.fp32SlotElementCount,
            unpermuteFullChunkConfig.topKWeightsBufferBytes, unpermuteFullChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName,
            "unpermuteTailChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteTailChunkConfig.tokensPerBatch, unpermuteTailChunkConfig.inputBufferCount,
            unpermuteTailChunkConfig.bf16SlotElementCount, unpermuteTailChunkConfig.fp32SlotElementCount,
            unpermuteTailChunkConfig.topKWeightsBufferBytes, unpermuteTailChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
}

void PrintWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataPtr:         %ld\n", info->dispatchRevDataPtr);
    OP_LOGD(nodeName, "dispatchRevScalePtr:        %ld\n", info->dispatchRevScalePtr);
    OP_LOGD(nodeName, "swigluQuantDataPtr:         %ld\n", info->swigluQuantDataPtr);
    OP_LOGD(nodeName, "swigluQuantScalePtr:        %ld\n", info->swigluQuantScalePtr);
    OP_LOGD(nodeName, "expertRevTokenNumsPtr:      %ld\n", info->expertRevTokenNumsPtr);
    OP_LOGD(nodeName, "metaInfoPtr:                %ld\n", info->metaInfoPtr);
    OP_LOGD(nodeName, "flagSwiGluToGmm2Ptr:        %ld\n", info->flagSwiGluToGmm2Ptr);
    OP_LOGD(nodeName, "flagDispatchToGmm1Ptr:      %ld\n", info->flagDispatchToGmm1Ptr);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsPtr:      %ld\n", info->flagSendCntCalToUpdParamsPtr);
    OP_LOGD(nodeName, "flagGmmToEpiloguePtr: %ld\n", info->flagGmmToEpiloguePtr);
}

void PrintPeermemInfo(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t rankSyncInWorldSize = PEERMEM_DATA_OFFSET;
    OP_LOGD(nodeName, "rankSyncInWorldSize: {%ld}\n", rankSyncInWorldSize);
    int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
    int64_t compareCount =
        ops::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) / (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = ops::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
    int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32; // mask + 32B count
    int64_t maskRecvSize =
        ops::CeilAlign((int64_t)tilingData->expertPerRank * tilingData->epWorldSize * maskSlotSize, (int64_t)ALIGN_512);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", maskRecvSize);
    uint32_t mxScaleNum = ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
    uint32_t dataBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
    uint32_t scaleAlignBytes =
        ops::CeilAlign(mxScaleNum * static_cast<uint32_t>(sizeof(int8_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tokenBytes = dataBytes + scaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(tilingData->topK * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        tokenBytes += weightBytes;
    }
    int64_t quantTokenScaleSize =
        ops::CeilAlign((int64_t)((int64_t)(tilingData->bs) * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
    OP_LOGD(nodeName, "quantTokenScaleSize: {%ld}\n", quantTokenScaleSize);
    // Non-quantized Combine stores one BF16 value (2 bytes) for each hidden-dimension element.
    int64_t combineTokenBytes = static_cast<int64_t>(tilingData->h) * 2;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        int64_t tokenStorageBytes =
            ops::CeilAlign(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(ALIGN_256));
        int64_t scaleCount =
            ops::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
        int64_t storedScaleBytes = ops::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
        combineTokenBytes = ops::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<int64_t>(ALIGN_32));
    }
    int64_t combineSendSize = ops::CeilAlign(sendTotalNum * combineTokenBytes, static_cast<int64_t>(ALIGN_512));
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n",
            rankSyncInWorldSize + maskRecvSize + quantTokenScaleSize + combineSendSize);
}

static ge::DataType GetDataTypeByOpQuantMode(const int64_t opQuantMode)
{
    // unsupport UNQUANT, STATIC, DYNAMIC currently
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_DTYPE_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case DISPATCH_QUANT_OUT_DTYPE_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        case DISPATCH_QUANT_OUT_DTYPE_E2M1:
            return ge::DT_FLOAT4_E2M1;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

static int64_t GetOpQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    int64_t dispatchQuantOutDtype = static_cast<int64_t>(*dispatchQuantOutDtypePtr);

    int64_t opQuantMode;
    if (dispatchQuantOutDtype == static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E5M2;
    } else if (dispatchQuantOutDtype == static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E4M3FN;
    } else {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E2M1;
    }

    return opQuantMode;
}

static int64_t GetCombineQuantModeByAttr(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    if (combineQuantModePtr == nullptr) {
        return COMBINE_QUANT_OUT_TYPE_NO_QUANT;
    }
    return static_cast<int64_t>(*combineQuantModePtr);
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config, MegaMoeTilingData *tilingData,
                             const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    int64_t combineQuantMode = GetCombineQuantModeByAttr(context, config);
    int64_t topoType = TILINGKEY_TPL_MTE;
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));

    return GET_TPL_TILING_KEY(static_cast<int64_t>(*dispatchQuantModePtr), opQuantMode, combineQuantMode, topoType,
                              topkWeightsType);
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    auto activationClampPtr = attrs->GetAttrPointer<float>((config.attrActivationClampIndex));

    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "cclBufferSize"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "maxRecvTokenNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutDtypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantOutDtype"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "combineQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commAlg"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationClampPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationClamp"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*activationClampPtr < 0 || std::isnan(*activationClampPtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp", std::to_string(*activationClampPtr).c_str(),
                                              "should be >= 0 and not NAN"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    auto yDesc = context->GetOutputDesc(config.yIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t n = weightOneStorageShape->GetStorageShape().GetDim(1);

    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                                               std::to_string(MAX_EP_WORLD_SIZE) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              (std::string("should in [") + std::to_string(epWorldSize) + ", " +
                                               std::to_string(MAX_MOE_EXPERT_NUM) + "] and mod(..., epWorldSize(" +
                                               std::to_string(epWorldSize) + ")) == 0")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t moeExpertPerRank = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(
        expertPerRank < moeExpertPerRank,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "expertPerRank", std::to_string(expertPerRank).c_str(),
            (std::string("should >= moeExpertNum/epWorldSize = ") + std::to_string(moeExpertPerRank)).c_str()),
        return ge::GRAPH_FAILED);

    // maskRecv Size
    int64_t compareCount =
        ops::CeilAlign((int64_t)(bs * topK * sizeof(int32_t)), (int64_t)(ALIGN_256)) / (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = ops::CeilAlign(compareCount / 8, (int64_t)ALIGN_32); // 8 = block_32 / sizeof(int32_t)
    int64_t maskSlotSize = maskAlignSize + ALIGN_32;                             // mask + 32B count
    int64_t maskRecvSize = ops::CeilAlign(expertPerRank * epWorldSize * maskSlotSize, ALIGN_512);
    // quantTokenScale Size
    uint32_t mxScaleNum = ops::CeilDiv(h, static_cast<int64_t>(ALIGN_32));
    uint32_t dataBytes = ops::CeilAlign(h, static_cast<int64_t>(ALIGN_256)) * sizeof(int8_t);
    uint32_t scaleAlignBytes =
        ops::CeilAlign(mxScaleNum * static_cast<uint32_t>(sizeof(int8_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tokenBytes = dataBytes + scaleAlignBytes;
    if (*attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex)) == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        tokenBytes = ops::CeilAlign(tokenBytes + weightBytes, static_cast<uint32_t>(ALIGN_32));
    }
    int64_t quantTokenScaleSize = ops::CeilAlign((int64_t)(bs * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
    // 必须与 kernel InitCombineBuffers 的 combine record 布局一致。
    int64_t combineTokenBytes = h * yDtypeSize;
    if (GetCombineQuantModeByAttr(context, config) != COMBINE_NO_QUANT) {
        int64_t tokenStorageBytes = ops::CeilAlign(h, static_cast<int64_t>(ALIGN_256));
        int64_t scaleCount = ops::CeilDiv(h, static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
        int64_t storedScaleBytes = ops::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
        combineTokenBytes = ops::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<int64_t>(ALIGN_32));
    }
    int64_t combineSendSize = ops::CeilAlign(bs * topK * combineTokenBytes, static_cast<int64_t>(ALIGN_512));
    int64_t leastCclBufferSize = PEERMEM_DATA_OFFSET + maskRecvSize + quantTokenScaleSize + combineSendSize;
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    int64_t cclBufferSize = static_cast<int64_t>(*cclBufferSizePtr);
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                              (std::string("should >= ") + std::to_string(leastCclBufferSize)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    int64_t maxRecvTokenNum = static_cast<int64_t>(*maxRecvTokenNumPtr);
    OP_TILING_CHECK(maxRecvTokenNum < 0 || maxRecvTokenNum > bs * epWorldSize * std::min(topK, expertPerRank),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                              (std::string("should in [0, ") +
                                               std::to_string(bs * epWorldSize * std::min(topK, expertPerRank)) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(dispatchQuantMode != DISPATCH_QUANT_MODE_MXFP,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "dispatchQuantMode", std::to_string(dispatchQuantMode).c_str(),
                        (std::string("only support mxfp(") + std::to_string(DISPATCH_QUANT_MODE_MXFP) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    int64_t dispatchQuantOutDtype = static_cast<int64_t>(*dispatchQuantOutDtypePtr);
    OP_TILING_CHECK(
        dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) &&
            dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
            dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "dispatchQuantOutDtype", std::to_string(dispatchQuantOutDtype).c_str(),
                                  "only support fp8_e5m2, fp8_e4m3fn and fp4_e2m1"),
        return ge::GRAPH_FAILED);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType refWeightDataType = GetDataTypeByOpQuantMode(opQuantMode);
    OP_TILING_CHECK(
        refWeightDataType == ge::DT_UNDEFINED,
        OP_LOGE(nodeName, "unsupported dispatchQuantMode(%ld), leading out data type to being DT_UNDEFINED.",
                dispatchQuantMode),
        return ge::GRAPH_FAILED);
    // weight1 must match dispatch quant dtype; the only allowed mismatch is A8W4 (fp4_e2m1 + fp8_e4m3fn).
    if (refWeightDataType != weightOneDesc->GetDataType()) {
        std::string weightDtypeErrMsg = std::string("The dtype of weightOne (") +
                                        Ops::Base::ToString(weightOneDesc->GetDataType()) +
                                        ") must match dispatch quant dtype (" + Ops::Base::ToString(refWeightDataType) +
                                        "), or be fp4_e2m1 with fp8_e4m3fn dispatch quant.";
        OP_TILING_CHECK(
            weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1 || opQuantMode != DISPATCH_QUANT_OUT_DTYPE_E4M3FN,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "dispatchQuantOutDtype/weight1", weightDtypeErrMsg.c_str(),
                                                  "weight1 dtype mismatch."),
            return ge::GRAPH_FAILED);
    }

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));

    OP_TILING_CHECK(
        *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combineQuantMode", std::to_string(*combineQuantModePtr).c_str(),
                                  "only support no_quant(0), fp8_e5m2(3) and fp8_e4m3fn(4)"),
        return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr, "not support, need empty string"),
                    return ge::GRAPH_FAILED);

    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    int64_t numMaxTokensPerRank = static_cast<int64_t>(*numMaxTokensPerRankPtr);
    if (numMaxTokensPerRank != 0) {
        OP_TILING_CHECK(
            bs != numMaxTokensPerRank,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank", std::to_string(numMaxTokensPerRank).c_str(),
                                      (std::string("should be 0 or Bs, Bs is ") + std::to_string(bs).c_str())),
            return ge::GRAPH_FAILED);
    }

    auto sharedWeight1StorageShape = context->GetDynamicInputShape(config.sharedWeight1Index, 0);
    int64_t sharedExpertNum = 0;
    if (sharedWeight1StorageShape != nullptr) {
        sharedExpertNum = sharedWeight1StorageShape->GetStorageShape().GetDim(0);
    }
    OP_TILING_CHECK(sharedExpertNum > NUM_FOUR,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(sharedExpertNum).c_str(),
                                              "only support 0-4"),
                    return ge::GRAPH_FAILED);

    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                     MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto activationClampPtr = attrs->GetAttrPointer<float>((config.attrActivationClampIndex));

    tilingData->epWorldSize = *epWorldSizePtr;
    tilingData->maxOutputSize = *maxRecvTokenNumPtr != 0 ? *maxRecvTokenNumPtr :
                                                           tilingData->bs * tilingData->epWorldSize *
                                                               std::min(tilingData->topK, tilingData->expertPerRank);
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->combineQuantMode = GetCombineQuantModeByAttr(context, config);
    tilingData->clampLimit = *activationClampPtr;

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t moeExpertPerRank = moeExpertNum / static_cast<int64_t>(tilingData->epWorldSize);
    tilingData->moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    auto sharedWeight1StorageShape = context->GetDynamicInputShape(config.sharedWeight1Index, 0);
    if (sharedWeight1StorageShape != nullptr) {
        tilingData->sharedExpertNum = static_cast<uint32_t>(sharedWeight1StorageShape->GetStorageShape().GetDim(0));
    } else {
        tilingData->sharedExpertNum = 0;
    }

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN) {
        // A8W4: fp4_e2m1 weight in NZ_C0_32, dispatched via separate template instantiation
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    } else if (opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN &&
               weightOneDesc->GetDataType() == GetDataTypeByOpQuantMode(opQuantMode) &&
               static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) ==
                   ge::FORMAT_FRACTAL_NZ) {
        // A8W8_NZ: fp8_e4m3fn activation × fp8_e4m3fn weight in NZ format (E5M2 not supported)
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W8_NZ;
    } else if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E2M1) {
        // A4W4: weight1 和 activation 都是 fp4，GMM1 走 generic，GMM2 走 A8W4。
        // NZ format: weight1 为 FRACTAL_NZ → A4W4_NZ；否则为 A4W4（ND 格式）。
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ) {
            tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A4W4_NZ;
        } else {
            tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A4W4;
        }
    } else {
        // Generic: fp8 activation × fp8 weight in ND format
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_GENERAL;
    }

    // GMM1 tile 状态位区每 expert 的 tile 上限（仅 prefetch 软同步路径使用）
    // outputN = hiddenDim / SWIGLU_N_HALF(2)，tileM=L1_TILE_M_256(256)，tileN=L1_TILE_N(256)
    // INT_CACHELINE=16：每 expert 的状态位按 16 对齐，与 kernel 侧 dispatchFlagSlotsPerExpert 一致
    if (tilingData->topkWeightsPrefetch == 1) {
        int64_t outputN = static_cast<int64_t>(tilingData->hiddenDim) / 2;
        int64_t maxTilesM = ops::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(256));
        int64_t maxTilesN = ops::CeilDiv(outputN, static_cast<int64_t>(256));
        tilingData->maxTilesPerExpert =
            static_cast<uint32_t>(ops::CeilAlign(maxTilesM * maxTilesN, static_cast<int64_t>(16)));
    }

    return ge::GRAPH_SUCCESS;
}

static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->bs) * tilingData->topK;
    uint64_t alignedTotalRouteItems =
        (sendTotalNum + static_cast<uint64_t>(ALIGN_256) - 1U) / static_cast<uint64_t>(ALIGN_256) * ALIGN_256;

    // Stage 1：使用基准 batch 确定 dispatch ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);

    // copyBufferBytes 是每个 dispatch slot 中量化 token 和 scale 的拷贝区大小。
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t copyBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    bufferConfig.copyBufferBytes = copyBufferBytes;

    // fixedBufferBytes 包含 expertTokenCntTensor_、cumsumInfoTensor_、sendCntTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ALIGN_32) +
        static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->expertPerRank * sizeof(int32_t),
                           static_cast<uint64_t>(ALIGN_32))) +
        tilingData->epWorldSize * static_cast<uint32_t>(ALIGN_32) +
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->expertPerRank) * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32)));
    uint32_t maskBufferBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    // routeIndexBufferBytes 是单个 int32 route index tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutDispatchSlots 包含固定 tensor、maskBatchTensor_、topkIndexTensor_ 和
    // validTopkIndexTensor_（此二者等大）。
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + maskBufferBytes + 2U * routeIndexBufferBytes;
    // dispatchSlotBytes 表示一个 ring slot：一份 token/scale copy buffer 加一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 ring slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是 maskBatchTensor_ 和两个 route index tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        // maskBatchTensor_ 占 1bit，topkIndexTensor_ 和 validTopkIndexTensor_ 各占一个 int32：
        // bytesPerItem = 1/8 + 4 + 4 = 65/8B，因此 maxItems = budgetBytes * 8 / 65。
        uint32_t expandedRouteItems = routeItemBudgetBytes * 8U / 65U;
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) /
                                                                bufferConfig.routeItemsPerBatch);
        }
    }
    return bufferConfig;
}

static MegaMoeSendMaskBufferConfig CalcSendMaskBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                            uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->bs) * tilingData->topK;
    uint64_t alignedTotalRouteItems =
        (sendTotalNum + static_cast<uint64_t>(ALIGN_256) - 1U) / static_cast<uint64_t>(ALIGN_256) * ALIGN_256;

    // Stage 1：使用基准 batch 确定 mask push ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);
    bufferConfig.bufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);

    // routeIndexBufferBytes 是单个 int32 route tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutMaskBuffers 包含固定 tensor、topkIdsTensor_ 和 sendGatherOutTensor_。
    uint32_t bytesWithoutMaskBuffers = fixedBufferBytes + 2U * routeIndexBufferBytes;
    // maskBufferBudgetBytes 是扣除非 ring tensor 后可用于分配 mask slot 的 UB。
    uint32_t maskBufferBudgetBytes =
        availableUbBytes > bytesWithoutMaskBuffers ? availableUbBytes - bytesWithoutMaskBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(maskBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    // maskPushCount 是当前 core 实际执行的 mask push 次数，更多 slot 不会增加流水重叠。
    uint64_t maskPushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (maskPushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > maskPushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(maskPushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithMaskCount 包含固定 tensor 和所有 slot 的 32B count 尾部。
        uint32_t fixedBytesWithMaskCount =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(ALIGN_32);
        // routeItemBudgetBytes 是两个 route tensor 和所有 slot 的 mask 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithMaskCount ? availableUbBytes - fixedBytesWithMaskCount : 0U;
        // topkIdsTensor_ 和 sendGatherOutTensor_ 各占一个 int32，N 个 sendMask slot 各占 1bit：
        // bytesPerItem = 4 + 4 + N/8 = (64 + N)/8B。
        uint32_t expandedRouteItems =
            routeItemBudgetBytes * 8U / (64U + static_cast<uint32_t>(bufferConfig.bufferCount));
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) /
                                                                bufferConfig.routeItemsPerBatch);
            bufferConfig.bufferBytes =
                static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);
        }
    }
    return bufferConfig;
}

static MegaMoeUnpermuteBufferConfig CalcUnpermuteBufferConfig(const MegaMoeTilingData *tilingData,
                                                              uint32_t coreTokenCount,
                                                              uint32_t topKWeightsConversionElementBytes,
                                                              uint32_t availableUbBytes)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t bf16SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    uint32_t dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    uint32_t scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }

    // Stage 1：构造基准 weight batch，再选择输入 ring 深度。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->topK;
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(ALIGN_32)));
    }

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        availableUbBytes > bytesBeforeInputBuffers ? availableUbBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / dataSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch * static_cast<int32_t>(tilingData->topK + tilingData->sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 weight batch。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes = scaleBytes + (static_cast<uint32_t>(bufferConfig.inputBufferCount) + 1U) * dataSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = availableUbBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->topK;
            bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(ops::CeilAlign(
                static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
            if (topKWeightsConversionElementBytes > 0U) {
                bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
                    ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                                   static_cast<uint64_t>(ALIGN_32)));
            }
        }
    }
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData *tilingData)
{
    if (tilingData->combineQuantMode == COMBINE_NO_QUANT && tilingData->topoType != TOPO_TYPE_URMA ||
        tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    uint64_t logicalCoreCount = tilingData->blockAivNum;
    if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        // A8W4/A4W4 每两个 AIV 配对，只有 subBlockIdx=1 参与 Combine，因此逻辑核数是物理 AIV 数的一半。
        logicalCoreCount /= 2U;
    }
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert = static_cast<uint64_t>(tilingData->bs) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, logicalCoreCount);
}

static ge::graphStatus SetAdaptiveBufferConfigs(const gert::TilingContext *context, MegaMoeConfig &config,
                                                MegaMoeTilingData *tilingData, uint32_t availableUbBytes)
{
    const char *nodeName = context->GetNodeName();
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    OP_TILING_CHECK(topKWeightsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);

    uint32_t activationElementsPerByte =
        GetOpQuantModeByAttrDispatchOutType(context, config) == DISPATCH_QUANT_OUT_DTYPE_E2M1 ? 2U : 1U;
    // DispatchBuffInit 的 UB 布局不依赖 AIV 分核结果，所有 AIV core 使用同一套 dispatch 配置。
    // 若 kernel 后续按 core 拆分 DispatchBuffInit 的 tensor 或改变 copyTmp slot 布局，需同步调整此处。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    tilingData->combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);

    int64_t maxWavesPerExpert =
        (static_cast<int64_t>(tilingData->maxOutputSize) + static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256) - 1) /
        static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256);
    int64_t dispatchFlagSlotsPerExpert = (maxWavesPerExpert + INT_CACHELINE - 1) / INT_CACHELINE * INT_CACHELINE;
    uint64_t totalFlagElementCount =
        static_cast<uint64_t>(tilingData->moeExpertPerRank) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert) +
         static_cast<uint64_t>(INT_CACHELINE) * tilingData->aicNum);
    if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        totalFlagElementCount += static_cast<uint64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint64_t combineFlagElementCount = tilingData->combineSyncSlotCountPerExpert * tilingData->moeExpertPerRank *
                                           static_cast<uint64_t>(INT_CACHELINE);
        totalFlagElementCount = std::max(totalFlagElementCount, combineFlagElementCount);
    }
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    // 与 kernel xOutTensorSize 一致：token 和 scale 分别对齐，prefetch 模式再追加对齐后的 weight。
    uint32_t quantOutputBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        quantOutputBufferBytes +=
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    // sendCntAccTensor_ 按本卡 weight 容量分配，与 kernel 地址布局一致；实际 mask push 次数按 routed MoE
    // expert 数计算，不包含共享专家和未参与路由的预留 weight。
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->expertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    uint32_t sendMaskFixedBufferBytes = resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes +
                                        2U * quantInputBufferBytes + sendCountAccumulatorBytes;

    /*
     * 与 kernel SendMaskCal 的 expert 遍历一一对应：
     *   globalExpertId = aivCoreIdx + ownedIdx * blockAivNum。
     * totalExpertCount 除以 blockAivNum 后，前 remainder 个 AIV core 各多处理一个 expert，其余 core
     * 处理 quotient 个 expert，因此这里只需预计算两套配置。
     *
     * 若修改 SendMaskCal 的 expert 分核方式、ownedExpertCount 或 maskPushCount 计算，必须同步更新
     * 这里的两类 core 划分和 kernel 配置选择条件。
     */
    // SendMaskCal 只遍历 routed MoE 专家；expertPerRank 是本卡 weight 容量，可能包含未参与路由的预留项。
    uint32_t totalExpertCount = tilingData->epWorldSize * tilingData->moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / tilingData->blockAivNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % tilingData->blockAivNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcSendMaskBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcSendMaskBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);

    /*
     * 与 kernel Unpermute 开头的 TilingByCore(m_, ..., align=1) 一一对应。TilingByCore 使用：
     *   fullTokenChunkSize = ceil(bs / blockAivNum)
     * 为连续 core 分配等长完整 chunk，最后一个活跃 core 可能只处理 tail，后续 core 的 coreLen 为 0
     * 并在读取配置前返回。因此 host 只需预计算“完整 chunk”和“tail chunk”两套配置，并记录完整
     * chunk 对应的 core 数作为 kernel 选择边界。
     *
     * 若修改 TilingByCore、Unpermute 的 align 参数或分核方式，必须同步更新下面的 chunk 推导以及
     * kernel 中 UnpermuteBuffInit 的配置选择条件。
     */
    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    // FP32 可直接搬入计算 buffer；其他已支持类型按实际元素大小预留转换中转区。
    uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    uint32_t fullTokenChunkSize = ops::CeilDiv(tilingData->bs, tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(tilingData->bs, fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk =
        CalcUnpermuteBufferConfig(tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ? MegaMoeUnpermuteBufferConfig{} :
                               CalcUnpermuteBufferConfig(tilingData, tailTokenChunkSize,
                                                         topKWeightsConversionElementBytes, availableUbBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
                                                 MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, WorkspaceInfo &workspaceInfo, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(workspaceInfo.workspaceSize == 0LL,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "workspaceSize",
                                              std::to_string(workspaceInfo.workspaceSize).c_str(), "non-zero"),
                    return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                             const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_TILING_CHECK(weightOneStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "weight1",
                        (std::to_string(weightOneStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
                    return ge::GRAPH_FAILED);
    const int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightOneDim2 = weightOneStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightOne dim0 = %ld", weightOneDim0);
    OP_LOGD(nodeName, "weightOne dim1 = %ld", weightOneDim1);
    OP_LOGD(nodeName, "weightOne dim2 = %ld", weightOneDim2);

    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weight2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    OP_TILING_CHECK(weightTwoStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "weight2",
                        (std::to_string(weightTwoStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
                    return ge::GRAPH_FAILED);
    const int64_t weightTwoDim0 = weightTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightTwoDim1 = weightTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightTwoDim2 = weightTwoStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightTwo dim0 = %ld", weightTwoDim0);
    OP_LOGD(nodeName, "weightTwo dim1 = %ld", weightTwoDim1);
    OP_LOGD(nodeName, "weightTwo dim2 = %ld", weightTwoDim2);

    OP_TILING_CHECK(
        weightOneDim0 != weightTwoDim0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weight1 and weight2",
            (std::string("[") + std::to_string(weightOneDim0) + ", " + std::to_string(weightTwoDim0) + "]").c_str(),
            "Dim0 of weight1 must be equal to dim0 of weight2."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        weightOneDim2 != weightTwoDim1 || h != weightOneDim2,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1, weight2 and x",
                                               (std::string("[") + std::to_string(weightOneDim2) + ", " +
                                                std::to_string(weightTwoDim1) + ", " + std::to_string(h) + "]")
                                                   .c_str(),
                                               "Dim2 of weight1 must be equal to dim1 of weight2 and h of x."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightOneDim1 != weightTwoDim2 * NUM_TWO,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weight1 and weight2",
            (std::string("[") + std::to_string(weightOneDim1) + ", " + std::to_string(weightTwoDim2) + "]").c_str(),
            "Dim1 of weight1 must be equal to dim2 of weight2 multiplied by 2."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedWeightShape(const gert::StorageShape *sharedShape, const gert::StorageShape *refShape,
                                              const char *sharedName, const char *refName, const char *nodeName)
{
    OP_TILING_CHECK(
        sharedShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, sharedName,
                                     (std::to_string(sharedShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedShape->GetStorageShape().GetDim(1) != refShape->GetStorageShape().GetDim(1),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedName) + " dim1").c_str(),
                                              std::to_string(sharedShape->GetStorageShape().GetDim(1)).c_str(),
                                              (std::string("must be equal to ") + refName + " dim1").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedShape->GetStorageShape().GetDim(2) != refShape->GetStorageShape().GetDim(2),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedName) + " dim2").c_str(),
                                              std::to_string(sharedShape->GetStorageShape().GetDim(2)).c_str(),
                                              (std::string("must be equal to ") + refName + " dim2").c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedWeightScalesInput(const gert::TilingContext *context, uint32_t sharedIdx,
                                                    uint32_t refIdx, const char *sharedName, const char *refName,
                                                    int64_t sharedExpertNum, const char *nodeName)
{
    auto sharedDesc = context->GetDynamicInputDesc(sharedIdx, 0);
    auto refDesc = context->GetDynamicInputDesc(refIdx, 0);
    OP_TILING_CHECK(sharedDesc->GetDataType() != refDesc->GetDataType(),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedName) + " dtype").c_str(),
                                              std::to_string(sharedDesc->GetDataType()).c_str(),
                                              (std::string("must be same as ") + refName).c_str()),
                    return ge::GRAPH_FAILED);
    auto sharedShape = context->GetDynamicInputShape(sharedIdx, 0);
    auto refShape = context->GetDynamicInputShape(refIdx, 0);
    OP_TILING_CHECK(sharedShape->GetStorageShape().GetDim(0) != sharedExpertNum,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, (std::string(sharedName) + " dim0").c_str(),
                        std::to_string(sharedShape->GetStorageShape().GetDim(0)).c_str(),
                        (std::string("must be equal to sharedExpertNum = ") + std::to_string(sharedExpertNum)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        sharedShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, sharedName,
                                     (std::to_string(sharedShape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedShape->GetStorageShape().GetDim(1) != refShape->GetStorageShape().GetDim(1),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedName) + " dim1").c_str(),
                                              std::to_string(sharedShape->GetStorageShape().GetDim(1)).c_str(),
                                              (std::string("must be equal to ") + refName + " dim1").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedShape->GetStorageShape().GetDim(2) != refShape->GetStorageShape().GetDim(2),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedName) + " dim2").c_str(),
                                              std::to_string(sharedShape->GetStorageShape().GetDim(2)).c_str(),
                                              (std::string("must be equal to ") + refName + " dim2").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedShape->GetStorageShape().GetDim(3) != refShape->GetStorageShape().GetDim(3),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, (std::string(sharedName) + " dim3").c_str(),
                                              std::to_string(sharedShape->GetStorageShape().GetDim(3)).c_str(),
                                              (std::string("must be equal to ") + refName + " dim3").c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedExpertInputs(const gert::TilingContext *context, MegaMoeConfig &config,
                                               const char *nodeName)
{
    auto sharedWeight1Desc = context->GetDynamicInputDesc(config.sharedWeight1Index, 0);
    if (sharedWeight1Desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto sharedWeight1Shape = context->GetDynamicInputShape(config.sharedWeight1Index, 0);
    int64_t sharedExpertNum = sharedWeight1Shape->GetStorageShape().GetDim(0);
    if (sharedExpertNum <= 0) {
        return ge::GRAPH_SUCCESS;
    }

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_TILING_CHECK(
        sharedWeight1Desc->GetDataType() != weightOneDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "shared_weight1 dtype",
                                  std::to_string(sharedWeight1Desc->GetDataType()).c_str(), "must be same as weight1"),
        return ge::GRAPH_FAILED);
    if (CheckSharedWeightShape(sharedWeight1Shape, weightOneStorageShape, "shared_weight1", "weight1", nodeName) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto sharedWeight2Desc = context->GetDynamicInputDesc(config.sharedWeight2Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    OP_TILING_CHECK(
        sharedWeight2Desc->GetDataType() != weightTwoDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "shared_weight2 dtype",
                                  std::to_string(sharedWeight2Desc->GetDataType()).c_str(), "must be same as weight2"),
        return ge::GRAPH_FAILED);
    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weight2Index, 0);
    auto sharedWeight2Shape = context->GetDynamicInputShape(config.sharedWeight2Index, 0);
    OP_TILING_CHECK(
        sharedWeight2Shape->GetStorageShape().GetDim(0) != sharedExpertNum,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "shared_weight2 dim0", std::to_string(sharedWeight2Shape->GetStorageShape().GetDim(0)).c_str(),
            (std::string("must be equal to sharedExpertNum = ") + std::to_string(sharedExpertNum)).c_str()),
        return ge::GRAPH_FAILED);
    if (CheckSharedWeightShape(sharedWeight2Shape, weightTwoStorageShape, "shared_weight2", "weight2", nodeName) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(CheckSharedWeightScalesInput(context, config.sharedWeightScales1Index, config.weightScales1Index,
                                                 "shared_weight_scales1", "weight_scales1", sharedExpertNum,
                                                 nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check shared_weight_scales1 failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedWeightScalesInput(context, config.sharedWeightScales2Index, config.weightScales2Index,
                                                 "shared_weight_scales2", "weight_scales2", sharedExpertNum,
                                                 nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check shared_weight_scales2 failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);

    auto yStorageShape = context->GetOutputShape(config.yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "y", (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "y dim0 = %ld", yDim0);
    OP_LOGD(nodeName, "y dim1 = %ld", yDim1);

    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsStorageShape);
    OP_TILING_CHECK(
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "expert_token_nums",
            (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsDim0);

    // expertTokenNums 仅报告路由专家的 token 数，不包含共享专家.
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    int64_t moeExpertPerRank = static_cast<int64_t>(*moeExpertNumPtr) / static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(
        expertTokenNumsDim0 != moeExpertPerRank,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "expertTokenNums", (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
            (std::string("The shape [dim0] of expertTokenNums must be equal to moeExpertPerRank(") +
             std::to_string(moeExpertPerRank) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightScalesTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                  const char *nodeName)
{
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(config.weightScales1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneStorageShape);
    OP_TILING_CHECK(
        weightScalesOneStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "weight_scales1",
            (std::to_string(weightScalesOneStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
        return ge::GRAPH_FAILED);
    const int64_t weightScalesOneDim0 = weightScalesOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesOneDim1 = weightScalesOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesOneDim2 = weightScalesOneStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesOneDim3 = weightScalesOneStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesOne dim0 = %ld", weightScalesOneDim0);
    OP_LOGD(nodeName, "weightScalesOne dim1 = %ld", weightScalesOneDim1);
    OP_LOGD(nodeName, "weightScalesOne dim2 = %ld", weightScalesOneDim2);
    OP_LOGD(nodeName, "weightScalesOne dim3 = %ld", weightScalesOneDim3);

    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(config.weightScales2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoStorageShape);
    OP_TILING_CHECK(weightScalesTwoStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "weightScalesTwo",
                        (std::to_string(weightScalesTwoStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of weightScalesTwo must be 4D."),
                    return ge::GRAPH_FAILED);
    const int64_t weightScalesTwoDim0 = weightScalesTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesTwoDim1 = weightScalesTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesTwoDim2 = weightScalesTwoStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesTwoDim3 = weightScalesTwoStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesTwo dim0 = %ld", weightScalesTwoDim0);
    OP_LOGD(nodeName, "weightScalesTwo dim1 = %ld", weightScalesTwoDim1);
    OP_LOGD(nodeName, "weightScalesTwo dim2 = %ld", weightScalesTwoDim2);
    OP_LOGD(nodeName, "weightScalesTwo dim3 = %ld", weightScalesTwoDim3);

    OP_TILING_CHECK(
        weightScalesOneDim0 != weightScalesTwoDim0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weightScalesOne, weightScalesTwo",
            (std::string("[") + std::to_string(weightScalesOneDim0) + ", " + std::to_string(weightScalesTwoDim0) + "]")
                .c_str(),
            "The shape [dim0] of weightScalesOne and weightScalesTwo must be equal."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightScalesOneDim3 != NUM_TWO || weightScalesOneDim3 != weightScalesTwoDim3,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weightScalesOne, weightScalesTwo",
            (std::string("[") + std::to_string(weightScalesOneDim3) + ", " + std::to_string(weightScalesTwoDim3) + "]")
                .c_str(),
            "The shape [dim3] of weightScalesOne and weightScalesTwo must be 2."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        weightScalesTwoDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScalesTwo", (std::string("dim1=") + std::to_string(weightScalesTwoDim1)).c_str(),
            (std::string("The shape [dim1] of weightScalesTwo must be equal to h(") + std::to_string(h) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        weightScalesOneDim2 != ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScalesOne", (std::string("dim2=") + std::to_string(weightScalesOneDim2)).c_str(),
            (std::string(
                 "The shape [dim2] of weightScalesOne must be equal to CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN) = ") +
             std::to_string(ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN)) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);

    const int64_t n = weightScalesOneDim1;
    OP_TILING_CHECK(
        weightScalesTwoDim2 != ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScalesTwo", (std::string("dim2=") + std::to_string(weightScalesTwoDim2)).c_str(),
            (std::string("The shape [dim2] of weightScalesTwo must be equal to CeilDiv(n / 2, "
                         "INPUT_WEIGHT_SCALES_CEIL_ALIGN) = ") +
             std::to_string(ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN)) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "context", (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of context must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x must be 2D."),
                    return ge::GRAPH_FAILED);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "topkIds", (std::to_string(topkIdsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of topkIds must be 2D."),
        return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "topkWeights",
                        (std::to_string(topkWeightsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of topkWeights must be 2D."),
                    return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 && xDim0 != topkWeightsDim0 && topkIdsDim0 != topkWeightsDim0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x, topkIds, topkWeights",
                        (std::string("[") + std::to_string(xDim0) + ", " + std::to_string(topkIdsDim0) + ", " +
                         std::to_string(topkWeightsDim0) + "]")
                            .c_str(),
                        "The shape [dim0] of x, topkIds, and topkWeights must be equal."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        topkIdsDim1 != topkWeightsDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "topkIds, topkWeights",
            (std::string("[") + std::to_string(topkIdsDim1) + ", " + std::to_string(topkWeightsDim1) + "]").c_str(),
            "The shape [dim1] of topkIds and topkWeights must be equal."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "optional params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check weight scales tensor dim failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckSharedExpertInputs(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check shared expert inputs failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "output params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "context",
                                                          Ops::Base::ToString(contextDesc->GetDataType()).c_str(),
                                                          "The dtype of context must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkIds",
                                                          Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of topkIds must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkWeights",
                                              Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str(),
                                              "The dtype of topkWeights must be DT_FLOAT or DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "weightOne", Ops::Base::ToString(weightOneDesc->GetDataType()).c_str(),
            "The dtype of weightOne must be within the range DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "weightTwo", Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str(),
            "The dtype of weightTwo must be within the range DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightOneDesc->GetDataType() != weightTwoDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "weightOne, weightTwo",
                                               (std::string("[") + Ops::Base::ToString(weightOneDesc->GetDataType()) +
                                                ", " + Ops::Base::ToString(weightTwoDesc->GetDataType()) + "]")
                                                   .c_str(),
                                               "The dtypes of weightOne and weightTwo must be the same."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "weightScalesOne", Ops::Base::ToString(weightScalesOneDesc->GetDataType()).c_str(),
                        "The dtype of weightScalesOne must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "weightScalesTwo", Ops::Base::ToString(weightScalesTwoDesc->GetDataType()).c_str(),
                        "The dtype of weightScalesTwo must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of y must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "expertTokenNums", Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(),
                        "The dtype of expertTokenNums must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context, MegaMoeConfig &config,
                                         const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkIds format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkWeights format is invalid."), return ge::GRAPH_FAILED);

    // A8W4 path: weight1 must use NZ_C0_32 format now.
    if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 &&
        GetOpQuantModeByAttrDispatchOutType(context, config) == DISPATCH_QUANT_OUT_DTYPE_E4M3FN) {
        OP_TILING_CHECK(weightOneDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ_C0_32,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "weight1",
                                                   Ops::Base::ToString(weightOneDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_FRACTAL_NZ_C0_32"),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesOne format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesTwo format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "y format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertTokenNums format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const char *nodeName)
{
    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // topk范围
    OP_TILING_CHECK(
        topkIdsDim1 < MIN_TOPK || topkIdsDim1 > MAX_TOPK,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topK", std::to_string(topkIdsDim1).c_str(), "only support [1, 32]"),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    // 检查 H 范围 [1K, 8K]
    OP_TILING_CHECK(
        xDim1 < MIN_H || xDim1 > MAX_H,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "H", std::to_string(xDim1).c_str(),
            (std::string("should in [") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
        return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    bool isW4NzC0_32 = weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 &&
                       weightOneDesc->GetStorageFormat() == ge::FORMAT_FRACTAL_NZ_C0_32;
    // URMA/Layered 保持 1K 对齐；W4 的 NZ_C0_32 分形要求 GMM K 按 64 对齐；其余 MTE 路径按 32 对齐。
    int64_t requiredHAlignment =
        *topoTypePtr == TOPO_TYPE_URMA ? HIDDEN_DIM_BASE : (isW4NzC0_32 ? W4_K_ALIGN : H_ALIGN);
    OP_TILING_CHECK(
        xDim1 % requiredHAlignment != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(),
                                  (std::string("multiple of ") + std::to_string(requiredHAlignment)).c_str()),
        return ge::GRAPH_FAILED);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightOneDim0 < MIN_EXPERT_PER_RANK || weightOneDim0 > MAX_EXPERT_PER_RANK,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertPerRank", std::to_string(weightOneDim0).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
                                               std::to_string(MAX_EXPERT_PER_RANK) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightOneDim1 < HIDDEN_DIM_BASE || weightOneDim1 > MAX_HIDDEN_DIM,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "hiddenDim", std::to_string(weightOneDim1).c_str(),
                        (std::string("should in [") + std::to_string(HIDDEN_DIM_BASE) + ", " +
                         std::to_string(MAX_HIDDEN_DIM) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightOneDim1 % HIDDEN_DIM_BASE != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(weightOneDim1).c_str(),
                                              (std::string("multiple of ") + std::to_string(HIDDEN_DIM_BASE)).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                     MegaMoeConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t hiddenDim = weightOneStorageShape->GetStorageShape().GetDim(1);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(h);
    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->expertPerRank = static_cast<uint32_t>(expertPerRank);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                        MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    tilingData->aicNum = aicNum;
    tilingData->blockAivNum = aivNum;
    auto attrs = context->GetAttrs();
    tilingData->topoType = *attrs->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));
    tilingData->topkWeightsPrefetch =
        static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex)));
    OP_TILING_CHECK(aivNum <= 0 || aicNum <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum/aicNum",
                                              (std::to_string(aivNum) + ", " + std::to_string(aicNum)).c_str(),
                                              "should both be > 0"),
                    return ge::GRAPH_FAILED);

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        SetAdaptiveBufferConfigs(context, config, tilingData, static_cast<uint32_t>(ubSize)) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Setting adaptive buffer configs failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceInfo workspaceInfo((uint8_t *)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceInfo, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceInfo(&workspaceInfo, nodeName);
    PrintPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext *context)
{
    MegaMoeConfig config;
    ge::graphStatus ret;

    ret = MegaMoeTilingFuncImplPublic(context, config);

    return ret;
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe).Tiling(MegaMoeTilingFunc).TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);
} // namespace optiling
