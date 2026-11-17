/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "quant_flash_attn_schedule_int8.h"
#include "../quant_flash_attn_tiling_common.h"
#include "../qfa_tiling_info.h"
#include "../checkers/base_checker_quant_flash_attn.h"
#include "../../op_kernel/arch22/quant_flash_attn_template_tiling_key_arch22.h"
#include "../../op_kernel/arch22/quant_flash_attn_tiling_data_int8.h"
using namespace ge;
using namespace AscendC;
namespace optiling {
using namespace quant_flash_attn;
using namespace BaseApi::Int8GqaTnd;

// A2/A3 fused INT8 GQA. Each task owns one grouped Q tile for all of S2, so
// online softmax and its output accumulation stay on the same AIC/AIV pair.
ge::graphStatus TilingQuantFlashAttnInt8(gert::TilingContext* context,
                                         const platform_ascendc::PlatformAscendC& platform)
{
    OP_CHECK_IF(platform.GetCurNpuArch() != NpuArch::DAV_2201,
                OP_LOGE(context, "Full INT8 GQA mode requires A2/A3 (DAV_2201)."), return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    const auto *q = context->GetInputShape(QUERY_INDEX), *k = context->GetInputShape(KEY_INDEX),
               *v = context->GetInputShape(VALUE_INDEX);
    const auto *qScale = context->GetInputShape(Q_DESCALE_INDEX), *kScale = context->GetInputShape(K_DESCALE_INDEX),
               *vScale = context->GetInputShape(V_DESCALE_INDEX);
    const auto *qDesc = context->GetInputDesc(QUERY_INDEX), *kDesc = context->GetInputDesc(KEY_INDEX),
               *vDesc = context->GetInputDesc(VALUE_INDEX);
    const auto *qScaleDesc = context->GetInputDesc(Q_DESCALE_INDEX),
               *kScaleDesc = context->GetInputDesc(K_DESCALE_INDEX),
               *vScaleDesc = context->GetInputDesc(V_DESCALE_INDEX);
    const auto* out = context->GetOutputShape(ATTN_OUT_INDEX);
    const auto* outDesc = context->GetOutputDesc(ATTN_OUT_INDEX);
    OP_CHECK_IF(attrs == nullptr || q == nullptr || k == nullptr || v == nullptr || qScale == nullptr ||
                    kScale == nullptr || vScale == nullptr || qDesc == nullptr || kDesc == nullptr ||
                    vDesc == nullptr || qScaleDesc == nullptr || kScaleDesc == nullptr || vScaleDesc == nullptr ||
                    out == nullptr || outDesc == nullptr,
                OP_LOGE(context, "Full INT8 GQA mode has a missing tensor or attribute."), return ge::GRAPH_FAILED);
    const auto* quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_CHECK_IF(quantMode == nullptr || *quantMode != QFA_INT8_QUANT_MODE,
                OP_LOGE(context, "A2/A3 requires quant_mode 12."), return ge::GRAPH_FAILED);
    // L0 may pass a zero-element tensor for an omitted optional input.
    const auto hasOptionalInput = [context](uint32_t index) {
        const auto* shape = context->GetOptionalInputShape(index);
        return context->GetOptionalInputDesc(index) != nullptr && shape != nullptr &&
               shape->GetStorageShape().GetShapeSize() != 0;
    };
    OP_CHECK_IF(hasOptionalInput(METADATA_INDEX),
                OP_LOGE(context, "A2/A3 INT8 Prefill requires metadata to be absent."), return ge::GRAPH_FAILED);
    const auto* softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    const auto* maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);
    const auto* layoutQ = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
    const auto* layoutKv = attrs->GetAttrPointer<char>(ATTR_LAYOUT_KV_INDEX);
    const auto* layoutOut = attrs->GetAttrPointer<char>(ATTR_LAYOUT_OUT_INDEX);
    const auto* returnLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_LSE_INDEX);
    auto layoutId = [](const char* name) -> int32_t {
        if (name == nullptr)
            return QFA_INVALID_LAYOUT;
        if (std::strcmp(name, "BNSD") == 0)
            return QFA_LAYOUT_BNSD;
        if (std::strcmp(name, "BSND") == 0)
            return QFA_LAYOUT_BSND;
        if (std::strcmp(name, "TND") == 0)
            return QFA_LAYOUT_TND;
        return QFA_INVALID_LAYOUT;
    };
    const int32_t lq = layoutId(layoutQ), lk = layoutId(layoutKv), lo = layoutId(layoutOut);
    OP_CHECK_IF(softmaxScale == nullptr || !std::isfinite(*softmaxScale) || *softmaxScale <= 0.0f ||
                    maskMode == nullptr || (*maskMode != QFA_MASK_NONE && *maskMode != QFA_MASK_CAUSAL_ATTR) ||
                    returnLse == nullptr || lq < 0 || lk < 0 || lo < 0 ||
                    ((lq == QFA_LAYOUT_TND || lk == QFA_LAYOUT_TND || lo == QFA_LAYOUT_TND) &&
                     !(lq == QFA_LAYOUT_TND && lk == QFA_LAYOUT_TND && lo == QFA_LAYOUT_TND)),
                OP_LOGE(context, "INT8 GQA requires BNSD/BSND or packed TND, mask 0/3, BF16 output."),
                return ge::GRAPH_FAILED);
    for (uint32_t i : {BLOCK_TABLE_INDEX, SEQUSED_Q_INDEX, SEQUSED_KV_INDEX, SINKS_INDEX, ATTN_MASK_INDEX, V_TAIL_INDEX,
                       BLOCK_TABLE_TAIL_INDEX, SEQUSED_V_TAIL_INDEX}) {
        OP_CHECK_IF(hasOptionalInput(i),
                    OP_LOGE(context, "INT8 GQA does not accept cache, seqused, sinks or explicit mask tensors."),
                    return ge::GRAPH_FAILED);
    }
    const auto* wl = attrs->GetAttrPointer<int64_t>(ATTR_WIN_LEFT_INDEX);
    const auto* wr = attrs->GetAttrPointer<int64_t>(ATTR_WIN_RIGHT_INDEX);
    OP_CHECK_IF(wl == nullptr || wr == nullptr || *wl < QFA_UNBOUNDED_WINDOW || *wr < QFA_UNBOUNDED_WINDOW ||
                    *wl > std::numeric_limits<int32_t>::max() || *wr > std::numeric_limits<int32_t>::max(),
                OP_LOGE(context, "INT8 win_left/right must be -1 or a nonnegative INT32 distance."),
                return ge::GRAPH_FAILED);
    const auto* pScaleDesc = context->GetOptionalInputDesc(P_SCALE_INDEX);
    const auto* pScale = context->GetOptionalInputShape(P_SCALE_INDEX);
    OP_CHECK_IF(pScaleDesc == nullptr || pScale == nullptr || pScaleDesc->GetDataType() != ge::DT_FLOAT ||
                    pScale->GetStorageShape().GetDimNum() != 1 || pScale->GetStorageShape().GetDim(0) != 1,
                OP_LOGE(context, "Full INT8 GQA requires FP32 p_scale [1]."), return ge::GRAPH_FAILED);
    const auto& qs = q->GetStorageShape();
    const auto& ks = k->GetStorageShape();
    const auto& vs = v->GetStorageShape();
    const auto& os = out->GetStorageShape();
    constexpr uint32_t tndRank = 3U;
    constexpr uint32_t batchedRank = 4U;
    const uint32_t rank = lq == QFA_LAYOUT_TND ? tndRank : batchedRank;
    OP_CHECK_IF(qs.GetDimNum() != rank || ks.GetDimNum() != rank || vs.GetDimNum() != rank || os.GetDimNum() != rank ||
                    qDesc->GetDataType() != ge::DT_INT8 || kDesc->GetDataType() != ge::DT_INT8 ||
                    vDesc->GetDataType() != ge::DT_INT8 || qScaleDesc->GetDataType() != ge::DT_FLOAT ||
                    kScaleDesc->GetDataType() != ge::DT_FLOAT || vScaleDesc->GetDataType() != ge::DT_FLOAT ||
                    outDesc->GetDataType() != ge::DT_BF16,
                OP_LOGE(context, "INT8 GQA tensor rank or dtype is invalid."), return ge::GRAPH_FAILED);
    // Heads are on axis 2 for BSND and axis 1 for BNSD/TND. Sequence is on axis 2 for BNSD
    // and axis 1 for BSND. The TND branch below binds sequence lengths from cu_seqlens and the bounds.
    int64_t b = qs.GetDim(0);
    const int64_t hq = qs.GetDim(lq == QFA_LAYOUT_BSND ? 2 : 1), hkv = ks.GetDim(lk == QFA_LAYOUT_BSND ? 2 : 1);
    int64_t sq = qs.GetDim(lq == QFA_LAYOUT_BNSD ? 2 : 1), sk = ks.GetDim(lk == QFA_LAYOUT_BNSD ? 2 : 1);
    const int64_t d = qs.GetDim(rank - 1);
    const int64_t outputDim = os.GetDim(rank - 1);
    const bool compactOutput =
        d > 0 && d <= QFA_MAX_HEAD_DIM && outputDim >= QFA_COMPACT_OUTPUT_MIN_DIM && outputDim < d &&
        outputDim % QFA_COMPACT_OUTPUT_ALIGNMENT == 0 &&
        ((outputDim + QFA_HEAD_DIM_ALIGNMENT - 1) / QFA_HEAD_DIM_ALIGNMENT) * QFA_HEAD_DIM_ALIGNMENT == d;
    const bool validOutputDim = outputDim == d || compactOutput;
    if (lq == QFA_LAYOUT_TND) {
        const auto* cqs = context->GetOptionalInputShape(CU_SEQLENS_Q_INDEX);
        const auto* cks = context->GetOptionalInputShape(CU_SEQLENS_KV_INDEX);
        const auto* cqd = context->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
        const auto* ckd = context->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
        constexpr int64_t minCumulativeLengths = 2; // Leading zero and the first sequence end.
        OP_CHECK_IF(!cqs || !cks || !cqd || !ckd || cqd->GetDataType() != ge::DT_INT32 ||
                        ckd->GetDataType() != ge::DT_INT32 || cqs->GetStorageShape().GetDimNum() != 1 ||
                        cks->GetStorageShape().GetDimNum() != 1 ||
                        cqs->GetStorageShape().GetDim(0) < minCumulativeLengths ||
                        cqs->GetStorageShape().GetDim(0) != cks->GetStorageShape().GetDim(0),
                    OP_LOGE(context, "TND requires INT32 cu_seqlens_q/kv [B+1]."), return ge::GRAPH_FAILED);
        b = cqs->GetStorageShape().GetDim(0) - 1;
        // Device cumulative lengths are consumed by the kernel, not dereferenced
        // by host tiling. Attributes are bounds; device code checks contents.
        const auto* mq = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);
        const auto* mk = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);
        sq = mq && *mq > 0 ? *mq : qs.GetDim(0);
        sk = mk && *mk > 0 ? *mk : ks.GetDim(0);
        OP_CHECK_IF(sq > qs.GetDim(0) || sk > ks.GetDim(0),
                    OP_LOGE(context, "TND max lengths cannot exceed total token counts."), return ge::GRAPH_FAILED);
    } else if (hasOptionalInput(CU_SEQLENS_Q_INDEX) || hasOptionalInput(CU_SEQLENS_KV_INDEX)) {
        OP_LOGE(context, "cu_seqlens are only accepted for TND.");
        return ge::GRAPH_FAILED;
    }
    bool validShapes = b > 0 && hq > 0 && hkv > 0 && sq > 0 && sk > 0 && d > 0 && d <= QFA_MAX_HEAD_DIM &&
                       hq % hkv == 0 && qs.GetDim(0) > 0 && ks.GetDim(0) > 0;
    for (uint32_t i = 0; i < rank; ++i)
        validShapes &= ks.GetDim(i) == vs.GetDim(i);
    validShapes &= ks.GetDim(rank - 1) == d;
    const auto& qss = qScale->GetStorageShape();
    const auto& kss = kScale->GetStorageShape();
    validShapes &= qss.GetDimNum() == rank - 1 && kss.GetDimNum() == rank - 1;
    if (validShapes) {
        for (uint32_t i = 0; i < rank - 1; ++i)
            validShapes &= qss.GetDim(i) == qs.GetDim(i) && kss.GetDim(i) == ks.GetDim(i);
    }
    validShapes &= vScale->GetStorageShape().GetDimNum() == 1 && vScale->GetStorageShape().GetDim(0) == hkv;
    if (lq == QFA_LAYOUT_TND) {
        for (uint32_t i = 0; i < rank - 1; ++i)
            validShapes &= os.GetDim(i) == qs.GetDim(i);
        validShapes &= validOutputDim;
    } else {
        // For the output, BNSD stores heads before sequence and BSND stores sequence before heads.
        validShapes &= ks.GetDim(0) == b && os.GetDim(0) == b && validOutputDim &&
                       os.GetDim(lo == QFA_LAYOUT_BNSD ? 1 : 2) == hq && os.GetDim(lo == QFA_LAYOUT_BNSD ? 2 : 1) == sq;
    }
    for (int64_t dim : {b, hq, hkv, sq, sk})
        validShapes &= dim <= std::numeric_limits<uint32_t>::max();
    OP_CHECK_IF(!validShapes,
                OP_LOGE(context, "INT8 GQA requires matching Q/K/V/output and per-token-head Q/K, per-head V scales."),
                return ge::GRAPH_FAILED);
    const QfaBaseChecker checker;
    for (uint32_t i : {KEY_INDEX, VALUE_INDEX, K_DESCALE_INDEX, V_DESCALE_INDEX}) {
        const auto& shape = context->GetInputShape(i)->GetStorageShape();
        const auto* strides = context->GetInputStride(i);
        int32_t axis = 0;
        // The shared checker treats 1D tensors as contiguous; per-head V scales still need a stride check.
        const bool invalidVectorStride = shape.GetDimNum() == 1 && shape.GetDim(0) > 1 && strides != nullptr &&
                                         strides->GetDimNum() > 0 && strides->GetStride(0) != 1;
        OP_CHECK_IF(invalidVectorStride ||
                        checker.CheckTensorContiguous(shape.GetDimNum(), shape, strides, axis) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "INT8 dense prefill input %u must be contiguous.", i), return ge::GRAPH_FAILED);
    }
    if (*returnLse) {
        const auto* shape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
        const auto* desc = context->GetOutputDesc(SOFTMAX_LSE_INDEX);
        OP_CHECK_IF(shape == nullptr || desc == nullptr || desc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context, "INT8 LSE requires an FP32 output tensor."), return ge::GRAPH_FAILED);
        const auto& ls = shape->GetStorageShape();
        // LSE drops D and always places heads before the sequence axis: [Hq,Tq] or [B,Hq,Sq].
        const bool validLse = lq == QFA_LAYOUT_TND ?
                                  ls.GetDimNum() == 2 && ls.GetDim(0) == hq && ls.GetDim(1) == qs.GetDim(0) :
                                  ls.GetDimNum() == 3 && ls.GetDim(0) == b && ls.GetDim(1) == hq && ls.GetDim(2) == sq;
        OP_CHECK_IF(!validLse, OP_LOGE(context, "INT8 LSE shape must be [Hq,Tq] or [B,Hq,Sq]."),
                    return ge::GRAPH_FAILED);
    }
    auto* rawTiling = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr || workspace == nullptr)
        return ge::GRAPH_FAILED;
    uint64_t l2Bytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
    BaseApi::Int8GqaTnd::QfaInt8ScheduleInput input{b,
                                                    hq,
                                                    hkv,
                                                    d,
                                                    outputDim,
                                                    sq,
                                                    sk,
                                                    lq == QFA_LAYOUT_TND ? qs.GetDim(0) : 0,
                                                    lk == QFA_LAYOUT_TND ? ks.GetDim(0) : 0,
                                                    lq,
                                                    lk,
                                                    lo,
                                                    *maskMode,
                                                    platform.GetCoreNumAic(),
                                                    l2Bytes,
                                                    platform.GetLibApiWorkSpaceSize(),
                                                    rawTiling->GetCapacity(),
                                                    *softmaxScale,
                                                    static_cast<int32_t>(*wl),
                                                    static_cast<int32_t>(*wr)};
    BaseApi::Int8GqaTnd::QfaInt8Schedule schedule;
    OP_CHECK_IF(!BaseApi::Int8GqaTnd::BuildInt8Schedule(input, schedule),
                OP_LOGE(context, "INT8 scheduling parameters exceed supported limits."), return ge::GRAPH_FAILED);
    const auto& td = schedule.payload.base.config;
    const uint32_t config = schedule.config;
    const size_t size = schedule.hasTaskBuffer ? sizeof(schedule.payload) : sizeof(schedule.payload.base);
    if (rawTiling->GetCapacity() < size)
        return ge::GRAPH_FAILED;
    std::memcpy(rawTiling->GetData(), &schedule.payload, size);
    rawTiling->SetDataSize(size);
    workspace[0] = schedule.workspaceBytes;
    context->SetBlockDim(td.usedCoreNum);
    // Independent BNSD, BSND/mixed and TND implementations preserve the
    // validated dense-layout pipelines when optimizing packed attention.
    const uint32_t kernelLayout =
        td.layoutQ == QFA_LAYOUT_TND ? QFA_INT8_TPL_TND :
        (td.layoutQ == QFA_LAYOUT_BNSD && td.layoutKv == QFA_LAYOUT_BNSD && td.layoutOut == QFA_LAYOUT_BNSD) ?
                                       QFA_INT8_TPL_BNSD :
                                       QFA_INT8_TPL_BSND_OR_MIXED;
    OP_LOGD(context, "QFA_INT8_DISPATCH sq=%lld sk=%lld hq=%lld hk=%lld d=%lld m=%u n=%u cfg=%u cores=%u",
            static_cast<long long>(sq), static_cast<long long>(sk), static_cast<long long>(hq),
            static_cast<long long>(hkv), static_cast<long long>(d), td.blockM, td.blockN, config, td.usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(kernelLayout, config | (*returnLse ? QFA_INT8_CFG_LSE_FLAG : 0U),
                                             static_cast<uint32_t>(*quantMode), false, QFA_INT8_TPL_KV_LAYOUT, false,
                                             false));
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
