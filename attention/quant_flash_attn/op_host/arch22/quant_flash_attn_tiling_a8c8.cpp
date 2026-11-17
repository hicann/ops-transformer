/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_flash_attn_tiling_a8c8.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "quant_flash_attn_tiling_data_a8c8.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include "../../op_kernel/arch22/quant_flash_attn_tiling_data_int8.h"

#include "../../../common/op_host/split_core.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>
#include "tiling/softmax/softmax_tiling.h"

// The decode host also includes the Host tiling data class. Declare only the
// shared seven-field key ABI here, without including the Kernel selector header.
ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      ASCENDC_TPL_UINT_DECL(Config, 10, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
                      ASCENDC_TPL_UINT_DECL(QuantMode, 5, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, 0, 1),
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      ASCENDC_TPL_BOOL_DECL(IsFd, 0, 1), ASCENDC_TPL_BOOL_DECL(HasVTail, 0, 1));

namespace optiling {
namespace quant_flash_attn {

namespace {

ge::graphStatus ValidateA8C8(const gert::TilingContext* context, const QfaTilingInfo& qfaInfo)
{
    OP_CHECK_IF(qfaInfo.quantMode != QfaQuantMode::A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR,
                OP_LOGE(context, "A2/A3 QFA only supports A8C8 INT8 quantization"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.qLayout != QfaLayout::TND, OP_LOGE(context, "A3 QFA requires layout_q=TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.layoutQDescale != QfaLayout::NT, OP_LOGE(context, "A3 QFA requires layout_q_descale=NT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.outLayout != QfaLayout::TND, OP_LOGE(context, "A3 QFA requires layout_out=TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.kvLayout != QfaLayout::PA_BBND && qfaInfo.kvLayout != QfaLayout::PA_BNBD &&
                    qfaInfo.kvLayout != QfaLayout::PA_NZ,
                OP_LOGE(context, "A3 QFA requires PA_BBND, PA_BNBD or PA_NZ cache layout"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.inputQType != ge::DT_INT8 || qfaInfo.inputKvType != ge::DT_INT8 ||
                    qfaInfo.opParamInfo.value.desc->GetDataType() != ge::DT_INT8,
                OP_LOGE(context, "A2/A3 QFA requires INT8 Q/K/V"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.outputType != ge::DT_BF16, OP_LOGE(context, "A2/A3 QFA requires BF16 output"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.qDescaleType != ge::DT_FLOAT || qfaInfo.kDescaleType != ge::DT_FLOAT ||
                    qfaInfo.vDescaleType != ge::DT_FLOAT,
                OP_LOGE(context, "A3 QFA requires FP32 descale tensors"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.qkHeadDim <= 0 || qfaInfo.qkHeadDim > 256,
                OP_LOGE(context, "A3 QFA head_dim must be in [1,256]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.pageAttentionFlag == false || qfaInfo.blockSize <= 0 || qfaInfo.maxBlockNumPerBatch <= 0,
                OP_LOGE(context, "A3 QFA requires paged KV cache and block_table"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.metadata.tensor != nullptr,
                OP_LOGE(context, "A2/A3 paged INT8 QFA does not support metadata"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.pScale.tensor == nullptr || qfaInfo.opParamInfo.pScale.tensor->GetShapeSize() != 1,
                OP_LOGE(context, "A3 QFA requires scalar p_scale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.maskMode != 0 && qfaInfo.maskMode != 3, OP_LOGE(context, "A3 QFA supports mask_mode 0 and 3"),
                return ge::GRAPH_FAILED);
    if (qfaInfo.softmaxLseFlag) {
        const auto& lse = qfaInfo.opParamInfo.lseOut;
        OP_CHECK_IF(lse.desc == nullptr || lse.shape == nullptr || lse.desc->GetDataType() != ge::DT_FLOAT ||
                        lse.shape->GetStorageShape().GetDimNum() != 2U ||
                        lse.shape->GetStorageShape().GetDim(0) != qfaInfo.n1Size ||
                        lse.shape->GetStorageShape().GetDim(1) != qfaInfo.qTSize,
                    OP_LOGE(context, "A2/A3 QFA requires FP32 LSE in [Hq, T]"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(qfaInfo.sinksFlag || qfaInfo.hasVTail,
                OP_LOGE(context, "A2/A3 INT8 QFA does not support sinks or high-precision V tails"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qfaInfo.opParamInfo.attnMask.tensor != nullptr || qfaInfo.maskMode == 1 || qfaInfo.maskMode == 2,
                OP_LOGE(context, "A2/A3 INT8 QFA does not support explicit attention masks or sliding windows"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint32_t KvLayoutCode(QfaLayout layout)
{
    if (layout == QfaLayout::PA_BBND) {
        return QFA_A8C8_KV_LAYOUT_PA_BBND;
    }
    if (layout == QfaLayout::PA_BNBD) {
        return QFA_A8C8_KV_LAYOUT_PA_BNBD;
    }
    return QFA_A8C8_KV_LAYOUT_PA_NZ;
}

ge::graphStatus MakeHostPlan(const gert::TilingContext* context, const QfaTilingInfo& info, uint32_t aicNum,
                             QfaHostPlanData& plan, bool& useOptimizedVectorService, bool allowPairedHeads)
{
    BaseInfo base;
    base.bSize = info.bSize;
    base.n2Size = info.n2Size;
    base.gSize = info.gSize;
    base.s1Size = info.maxSeqQ;
    base.s2Size = info.maxSeqKv;
    base.preToken = std::numeric_limits<int32_t>::max();
    base.isS1G = true;
    base.actualLenQDims = base.bSize;
    base.actualLenKvDims = base.bSize;
    const auto hostPtr = [](const QfaOptionalParaInfo& p) -> const int32_t* {
        if (p.tensor == nullptr ||
            (p.tensor->GetPlacement() != gert::kOnHost && p.tensor->GetPlacement() != gert::kFollowing)) {
            return nullptr;
        }
        return p.tensor->GetData<int32_t>();
    };
    const auto* cu = hostPtr(info.opParamInfo.cuSeqlensQ);
    const auto* uq = hostPtr(info.opParamInfo.sequsedQ);
    const auto* uk = hostPtr(info.opParamInfo.sequsedKv);
    for (uint32_t b = 0; b < base.bSize; ++b) {
        int64_t q = cu == nullptr ? base.s1Size : cu[b + 1U] - cu[b];
        if (uq != nullptr)
            q = std::min(q, static_cast<int64_t>(uq[b]));
        const int64_t k = uk == nullptr ? base.s2Size : uk[b];
        base.actualSeqS1Size.push_back(std::max(int64_t{0}, q));
        base.actualSeqS2Size.push_back(std::max(int64_t{0}, k));
    }
    const auto scratchBytes = [](uint32_t rows, uint32_t floor) {
        uint32_t bytes = floor;
        for (uint32_t r = 1U; r <= rows; ++r) {
            ge::Shape shape({r, 512U});
            bytes = std::max(bytes,
                             (AscendC::GetSoftMaxFlashV2MinTmpSize(shape, 4U, 4U, true, false, false) + 255U) & ~255U);
        }
        return bytes;
    };
    const bool valid = a8c8_plan::QfaBuildPlan(base, info.qkHeadDim, KvLayoutCode(info.kvLayout), info.blockSize,
                                               aicNum, plan, useOptimizedVectorService, allowPairedHeads, scratchBytes);
    OP_CHECK_IF(!valid, OP_LOGE(context, "No valid QFA N512 plan"), return ge::GRAPH_FAILED);
    plan.returnSoftmaxLse = info.softmaxLseFlag;
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus TilingQuantFlashAttnA8C8(gert::TilingContext* context, const QfaTilingInfo& qfaInfo)
{
    auto status = ValidateA8C8(context, qfaInfo);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t aivNum = platform.GetCoreNumAiv();
    const uint32_t aicNum = platform.GetCoreNumAic();
    OP_CHECK_IF(aivNum == 0 || aicNum == 0, OP_LOGE(context, "AIV/AIC core num is 0"), return ge::GRAPH_FAILED);

    QuantFlashAttnA8C8TilingData tilingData;
    tilingData.batchSize = (static_cast<uint32_t>(qfaInfo.bSize));
    tilingData.totalTokens = (static_cast<uint32_t>(qfaInfo.qTSize));
    tilingData.qHeadNum = (static_cast<uint32_t>(qfaInfo.n1Size));
    tilingData.kvHeadNum = (static_cast<uint32_t>(qfaInfo.n2Size));
    tilingData.groupSize = (static_cast<uint32_t>(qfaInfo.gSize));
    tilingData.headDim = (static_cast<uint32_t>(qfaInfo.qkHeadDim));
    tilingData.blockSize = (static_cast<uint32_t>(qfaInfo.blockSize));
    tilingData.maxBlockNum = (static_cast<uint32_t>(qfaInfo.maxBlockNumPerBatch));
    tilingData.maxSeqQ = (static_cast<uint32_t>(qfaInfo.maxSeqQ));
    tilingData.maxSeqKv = (static_cast<uint32_t>(qfaInfo.maxSeqKv));
    OP_CHECK_IF(qfaInfo.bSize > QFA_PLAN_MAX_BATCH || aicNum > QFA_PLAN_MAX_CORES,
                OP_LOGE(context, "A2/A3 QFA host-plan capacity exceeded"), return ge::GRAPH_FAILED);
    constexpr uint32_t kvTile = 512U;
    tilingData.kvTile = (kvTile);
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1U;
    const auto scheduleStatus = context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    OP_CHECK_IF(scheduleStatus != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetScheduleMode(BATCH_MODE) failed, ret=%u", static_cast<uint32_t>(scheduleStatus)),
                return ge::GRAPH_FAILED);
    tilingData.kvLayout = (KvLayoutCode(qfaInfo.kvLayout));
    tilingData.maskMode = (static_cast<uint32_t>(qfaInfo.maskMode));
    tilingData.softmaxScale = (qfaInfo.softmaxScale);

    const uint64_t dim = static_cast<uint64_t>(qfaInfo.qkHeadDim);
    const uint64_t blockSize = static_cast<uint64_t>(qfaInfo.blockSize);
    const uint64_t kvHeads = static_cast<uint64_t>(qfaInfo.n2Size);
    const uint64_t dBlocks = (dim + QFA_A8C8_D_BLOCK - 1U) / QFA_A8C8_D_BLOCK;
    uint64_t keyBlock = kvHeads * blockSize * dim;
    uint64_t keyHead = blockSize * dim;
    uint64_t keySlot = dim;
    uint64_t keyDim = 1U;
    uint64_t keyDBlock = 0U;
    uint64_t keyInner = 0U;
    uint64_t valueBlock = keyBlock;
    uint64_t valueHead = keyHead;
    uint64_t valueSlot = keySlot;
    uint64_t valueDim = 1U;
    uint64_t valueDBlock = 0U;
    uint64_t valueInner = 0U;
    if (qfaInfo.kvLayout == QfaLayout::PA_BBND) {
        keyBlock = blockSize * kvHeads * dim;
        keyHead = dim;
        keySlot = kvHeads * dim;
        valueBlock = keyBlock;
        valueHead = keyHead;
        valueSlot = keySlot;
    } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
        keyBlock = kvHeads * dBlocks * blockSize * QFA_A8C8_D_BLOCK;
        keyHead = dBlocks * blockSize * QFA_A8C8_D_BLOCK;
        keySlot = QFA_A8C8_D_BLOCK;
        keyDim = 1U;
        keyDBlock = blockSize * QFA_A8C8_D_BLOCK;
        keyInner = 1U;
        valueBlock = keyBlock;
        valueHead = keyHead;
        valueSlot = keySlot;
        valueDBlock = keyDBlock;
        valueInner = keyInner;
    }
    if (qfaInfo.keyStrides != nullptr) {
        keyBlock = qfaInfo.keyStrides->GetStride(0);
        keyHead = qfaInfo.keyStrides->GetStride(1);
        if (qfaInfo.kvLayout == QfaLayout::PA_BBND) {
            keySlot = qfaInfo.keyStrides->GetStride(1);
            keyHead = qfaInfo.keyStrides->GetStride(2);
            keyDim = qfaInfo.keyStrides->GetStride(3);
        } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
            keyDBlock = qfaInfo.keyStrides->GetStride(2);
            keySlot = qfaInfo.keyStrides->GetStride(3);
            keyInner = qfaInfo.keyStrides->GetStride(4);
        } else {
            keySlot = qfaInfo.keyStrides->GetStride(2);
            keyDim = qfaInfo.keyStrides->GetStride(3);
        }
    }
    if (qfaInfo.valueStrides != nullptr) {
        valueBlock = qfaInfo.valueStrides->GetStride(0);
        valueHead = qfaInfo.valueStrides->GetStride(1);
        if (qfaInfo.kvLayout == QfaLayout::PA_BBND) {
            valueSlot = qfaInfo.valueStrides->GetStride(1);
            valueHead = qfaInfo.valueStrides->GetStride(2);
            valueDim = qfaInfo.valueStrides->GetStride(3);
        } else if (qfaInfo.kvLayout == QfaLayout::PA_NZ) {
            valueDBlock = qfaInfo.valueStrides->GetStride(2);
            valueSlot = qfaInfo.valueStrides->GetStride(3);
            valueInner = qfaInfo.valueStrides->GetStride(4);
        } else {
            valueSlot = qfaInfo.valueStrides->GetStride(2);
            valueDim = qfaInfo.valueStrides->GetStride(3);
        }
    }
    const uint64_t kDescaleBlock =
        qfaInfo.kDescaleStrides != nullptr ? qfaInfo.kDescaleStrides->GetStride(0) : kvHeads * blockSize;
    const uint64_t kDescaleHead =
        qfaInfo.kDescaleStrides != nullptr ? qfaInfo.kDescaleStrides->GetStride(1) : blockSize;
    const uint64_t kDescaleSlot = qfaInfo.kDescaleStrides != nullptr ? qfaInfo.kDescaleStrides->GetStride(2) : 1U;
    const uint64_t vDescaleHead = qfaInfo.vDescaleStrides != nullptr ? qfaInfo.vDescaleStrides->GetStride(0) : 1U;
    tilingData.keyStrideBlock = (keyBlock);
    tilingData.keyStrideHead = (keyHead);
    tilingData.keyStrideSlot = (keySlot);
    tilingData.keyStrideDim = (keyDim);
    tilingData.keyStrideDBlock = (keyDBlock);
    tilingData.keyStrideInner = (keyInner);
    tilingData.valueStrideBlock = (valueBlock);
    tilingData.valueStrideHead = (valueHead);
    tilingData.valueStrideSlot = (valueSlot);
    tilingData.valueStrideDim = (valueDim);
    tilingData.valueStrideDBlock = (valueDBlock);
    tilingData.valueStrideInner = (valueInner);
    tilingData.kDescaleStrideBlock = (kDescaleBlock);
    tilingData.kDescaleStrideHead = (kDescaleHead);
    tilingData.kDescaleStrideSlot = (kDescaleSlot);
    tilingData.vDescaleStrideHead = (vDescaleHead);

    constexpr size_t QFA_A8C8_CUBE_SYSTEM_WORKSPACE_SIZE = 16U * 1024U * 1024U;

    tilingData.kScaleDenseStride = (0U);
    tilingData.kScaleDenseOffset = (0ULL);
    tilingData.shortQRoute = (1U);

    QfaHostPlanData plan{};
    bool useOptimizedVectorService = false;
    size_t plannedWorkspace = 0U;
    {
        const bool allowPairedHeads = keyHead == dim && valueHead == dim && keySlot == kvHeads * dim &&
                                      valueSlot == kvHeads * dim && keyDim == 1U && valueDim == 1U;
        const auto result = MakeHostPlan(context, qfaInfo, aicNum, plan, useOptimizedVectorService, allowPairedHeads);
        if (result != ge::GRAPH_SUCCESS)
            return result;
        tilingData.reservedNativeContext = (plan.m);
        tilingData.coreNum = (plan.cores);
        tilingData.smallDQtile = (plan.m > 64U ? 1U : 0U);
        const uint64_t rows = std::max(64U, plan.m * plan.headGroupSize);
        const uint64_t stride = QFA_A8C8_CUBE_SEQ_Q_BYTES + rows * 512ULL * 12ULL + rows * 256ULL * 20ULL + rows * 8ULL;
        plannedWorkspace = stride * std::max(plan.cores, plan.partials);
    }

    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr || rawTilingData->GetData() == nullptr,
                OP_LOGE(context, "raw tiling data is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(QfaHostPlanTilingPayload),
                OP_LOGE(context, "raw tiling capacity=%zu, required=%zu", rawTilingData->GetCapacity(),
                        sizeof(QfaHostPlanTilingPayload)),
                return ge::GRAPH_FAILED);
    QfaHostPlanTilingPayload payload{};
    payload.config = tilingData;
    payload.plan = plan;
    std::memcpy(rawTilingData->GetData(), &payload, sizeof(payload));
    rawTilingData->SetDataSize(sizeof(payload));

    context->SetBlockDim(plan.cores);
    const uint64_t key = !useOptimizedVectorService || plan.hpScratchBytes != 32768U ? 10003U :
                         plan.hpRows == 8U                                           ? 10005U :
                         qfaInfo.qkHeadDim == 128U && plan.hpVec2Rows == 16U         ? 10004U :
                                                                                       10006U;
    const bool compactD256 = qfaInfo.qkHeadDim == 256U && plan.m == 128U && plan.hpRows == 16U &&
                             plan.hpVec2Rows == 8U && plan.hpScratchBytes == 16384U;
    const uint64_t vectorKey = compactD256 ? 10011U : key;
    const uint64_t decodeKey = plan.headGroupSize == 2U ? (plan.m > 32U ? 8U : 7U) : vectorKey - 10000U;
    context->SetTilingKey(GET_TPL_TILING_KEY(QFA_INT8_TPL_TND, decodeKey, QFA_INT8_QUANT_MODE, false,
                                             QFA_INT8_TPL_PAGED_KV_LAYOUT, false, false));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspace == nullptr, OP_LOGE(context, "workspace size pointer is nullptr"), return ge::GRAPH_FAILED);
    workspace[0] = QFA_A8C8_CUBE_SYSTEM_WORKSPACE_SIZE + plannedWorkspace;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
