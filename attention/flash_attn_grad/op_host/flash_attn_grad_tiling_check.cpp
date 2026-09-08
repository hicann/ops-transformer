/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_check.h"

namespace optiling {

bool FlashAttnGradCheck::IsTensorExist(gert::TilingContext *context, size_t index)
{
    auto shape = context->GetOptionalInputShape(index);
    return (shape != nullptr) && (shape->GetStorageShape().GetDimNum() > 0);
}

ge::graphStatus FlashAttnGradCheck::CheckInputExistence(gert::TilingContext *context)
{
    const char *opName = context->GetNodeName();
    auto qShape = context->GetInputShape(Q_INDEX);
    auto kShape = context->GetInputShape(K_INDEX);
    auto vShape = context->GetInputShape(V_INDEX);
    auto doutShape = context->GetInputShape(DOUT_INDEX);
    auto attnOutShape = context->GetInputShape(ATTN_OUT_INDEX);
    auto softmaxLseShape = context->GetInputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr || doutShape == nullptr ||
                    attnOutShape == nullptr || softmaxLseShape == nullptr,
                OP_LOGE(opName, "required input (q/k/v/dout/attn_out/softmax_lse) shape is nullptr."),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradCheck::CheckDtypeConsistency(gert::TilingContext *context)
{
    const char *opName = context->GetNodeName();
    auto qDesc = context->GetInputDesc(Q_INDEX);
    auto kDesc = context->GetInputDesc(K_INDEX);
    auto vDesc = context->GetInputDesc(V_INDEX);
    auto doutDesc = context->GetInputDesc(DOUT_INDEX);
    auto attnOutDesc = context->GetInputDesc(ATTN_OUT_INDEX);
    auto dqDesc = context->GetOutputDesc(0);
    auto dkDesc = context->GetOutputDesc(1);
    auto dvDesc = context->GetOutputDesc(2);
    OP_CHECK_IF(qDesc == nullptr || kDesc == nullptr || vDesc == nullptr || doutDesc == nullptr ||
                    attnOutDesc == nullptr || dqDesc == nullptr || dkDesc == nullptr || dvDesc == nullptr,
                OP_LOGE(opName, "failed to get input/output desc for dtype check."), return ge::GRAPH_PARAM_INVALID);

    ge::DataType qDtype = qDesc->GetDataType();
    OP_CHECK_IF(kDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of k must be the same as q(%d), but got %d.", static_cast<int32_t>(qDtype),
                        static_cast<int32_t>(kDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(vDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of v must be the same as q(%d), but got %d.", static_cast<int32_t>(qDtype),
                        static_cast<int32_t>(vDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(doutDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of dout must be the same as q(%d), but got %d.", static_cast<int32_t>(qDtype),
                        static_cast<int32_t>(doutDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(attnOutDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of attn_out must be the same as q(%d), but got %d.",
                        static_cast<int32_t>(qDtype), static_cast<int32_t>(attnOutDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dqDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of dq must be the same as q(%d), but got %d.", static_cast<int32_t>(qDtype),
                        static_cast<int32_t>(dqDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dkDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of dk must be the same as q(%d), but got %d.", static_cast<int32_t>(qDtype),
                        static_cast<int32_t>(dkDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dvDesc->GetDataType() != qDtype,
                OP_LOGE(opName, "dtype of dv must be the same as q(%d), but got %d.", static_cast<int32_t>(qDtype),
                        static_cast<int32_t>(dvDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradCheck::CheckAttrs(gert::TilingContext *context, int64_t &maskMode, int64_t &winLeft,
                                               int64_t &winRight, std::string &layoutQStr, std::string &layoutKvStr,
                                               std::string &layoutOutStr)
{
    const char *opName = context->GetNodeName();
    maskMode = MASK_MODE_NO_MASK;
    winLeft = -1;
    winRight = -1;
    layoutQStr = "BSND";
    layoutKvStr = "BSND";
    layoutOutStr = "BSND";
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;

    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(1);
        if (maskModePtr != nullptr) {
            maskMode = *maskModePtr;
        }
        auto winLeftPtr = attrs->GetAttrPointer<int64_t>(2);
        if (winLeftPtr != nullptr) {
            winLeft = *winLeftPtr;
        }
        auto winRightPtr = attrs->GetAttrPointer<int64_t>(3);
        if (winRightPtr != nullptr) {
            winRight = *winRightPtr;
        }
        auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(4);
        if (maxSeqlenQPtr != nullptr) {
            maxSeqlenQ = *maxSeqlenQPtr;
        }
        auto maxSeqlenKvPtr = attrs->GetAttrPointer<int64_t>(5);
        if (maxSeqlenKvPtr != nullptr) {
            maxSeqlenKv = *maxSeqlenKvPtr;
        }
        auto layoutQPtr = attrs->GetAttrPointer<char>(6);
        if (layoutQPtr != nullptr) {
            layoutQStr = std::string(layoutQPtr);
        }
        auto layoutKvPtr = attrs->GetAttrPointer<char>(7);
        if (layoutKvPtr != nullptr) {
            layoutKvStr = std::string(layoutKvPtr);
        }
        auto layoutOutPtr = attrs->GetAttrPointer<char>(8);
        if (layoutOutPtr != nullptr) {
            layoutOutStr = std::string(layoutOutPtr);
        }
    }

    OP_CHECK_IF(maskMode != MASK_MODE_NO_MASK && maskMode != MASK_MODE_CAUSAL && maskMode != MASK_MODE_WINDOW,
                OP_LOGE(opName, "mask_mode only supports 0, 3, 4, but got %ld.", maskMode),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(winLeft < -1, OP_LOGE(opName, "win_left must be -1 or >= 0, but got %ld.", winLeft),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(winRight < -1, OP_LOGE(opName, "win_right must be -1 or >= 0, but got %ld.", winRight),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(maxSeqlenQ < -1, OP_LOGE(opName, "max_seqlen_q must be -1 or >= 0, but got %ld.", maxSeqlenQ),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(maxSeqlenKv < -1, OP_LOGE(opName, "max_seqlen_kv must be -1 or >= 0, but got %ld.", maxSeqlenKv),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradCheck::CheckMaskAndAttnMask(gert::TilingContext *context, int64_t maskMode,
                                                         int64_t winLeft, int64_t winRight)
{
    const char *opName = context->GetNodeName();

    if (maskMode == MASK_MODE_NO_MASK) {
        OP_CHECK_IF(IsTensorExist(context, ATTN_MASK_INDEX),
                    OP_LOGE(opName, "attn_mask must not be provided when mask_mode is 0."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(winLeft != -1, OP_LOGE(opName, "win_left must be -1 when mask_mode is 0, but got %ld.", winLeft),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(winRight != -1, OP_LOGE(opName, "win_right must be -1 when mask_mode is 0, but got %ld.", winRight),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (maskMode == MASK_MODE_CAUSAL) {
        OP_CHECK_IF(!IsTensorExist(context, ATTN_MASK_INDEX),
                    OP_LOGE(opName, "attn_mask must be provided when mask_mode is 3."), return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(winLeft != -1, OP_LOGE(opName, "win_left must be -1 when mask_mode is 3, but got %ld.", winLeft),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(winRight != -1, OP_LOGE(opName, "win_right must be -1 when mask_mode is 3, but got %ld.", winRight),
                    return ge::GRAPH_PARAM_INVALID);
    } else if (maskMode == MASK_MODE_WINDOW) {
        OP_CHECK_IF(!IsTensorExist(context, ATTN_MASK_INDEX),
                    OP_LOGE(opName, "attn_mask must be provided when mask_mode is 4."), return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(winLeft < 0, OP_LOGE(opName, "win_left must be >= 0 when mask_mode is 4, but got %ld.", winLeft),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(winRight < 0, OP_LOGE(opName, "win_right must be >= 0 when mask_mode is 4, but got %ld.", winRight),
                    return ge::GRAPH_PARAM_INVALID);
    }

    if (IsTensorExist(context, ATTN_MASK_INDEX)) {
        auto attnMaskShape = context->GetOptionalInputShape(ATTN_MASK_INDEX);
        auto &maskStorageShape = attnMaskShape->GetStorageShape();
        OP_CHECK_IF(maskStorageShape.GetDimNum() != 2,
                    OP_LOGE(opName, "attn_mask must be 2D, but got %ld dims.", maskStorageShape.GetDimNum()),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(maskStorageShape.GetDim(0) != ATTN_MASK_DIM || maskStorageShape.GetDim(1) != ATTN_MASK_DIM,
                    OP_LOGE(opName, "attn_mask shape must be [2048, 2048], but got [%ld, %ld].",
                            maskStorageShape.GetDim(0), maskStorageShape.GetDim(1)),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradCheck::CheckLayout(gert::TilingContext *context, const std::string &layoutQStr,
                                                const std::string &layoutKvStr, const std::string &layoutOutStr)
{
    const char *opName = context->GetNodeName();
    OP_CHECK_IF(layoutQStr != "BSND" && layoutQStr != "TND" && layoutQStr != "BNSD",
                OP_LOGE(opName, "layout_q only supports BSND, TND, BNSD, but got %s.", layoutQStr.c_str()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(layoutKvStr != "BSND" && layoutKvStr != "TND" && layoutKvStr != "BNSD",
                OP_LOGE(opName, "layout_kv only supports BSND, TND, BNSD, but got %s.", layoutKvStr.c_str()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(layoutOutStr != "BSND" && layoutOutStr != "TND" && layoutOutStr != "BNSD",
                OP_LOGE(opName, "layout_out only supports BSND, TND, BNSD, but got %s.", layoutOutStr.c_str()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(layoutQStr != layoutKvStr || layoutQStr != layoutOutStr,
                OP_LOGE(opName, "layout_q(%s), layout_kv(%s), layout_out(%s) must be the same.", layoutQStr.c_str(),
                        layoutKvStr.c_str(), layoutOutStr.c_str()),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradCheck::CheckShape(gert::TilingContext *context, const std::string &layoutQStr)
{
    const char *opName = context->GetNodeName();
    bool isTnd = (layoutQStr == "TND");
    bool isBnsd = (layoutQStr == "BNSD");

    auto qShape = context->GetInputShape(Q_INDEX);
    auto kShape = context->GetInputShape(K_INDEX);
    auto vShape = context->GetInputShape(V_INDEX);
    auto doutShape = context->GetInputShape(DOUT_INDEX);
    auto attnOutShape = context->GetInputShape(ATTN_OUT_INDEX);
    auto softmaxLseShape = context->GetInputShape(SOFTMAX_LSE_INDEX);

    auto &qSS = qShape->GetStorageShape();
    auto &kSS = kShape->GetStorageShape();
    auto &vSS = vShape->GetStorageShape();
    auto &doutSS = doutShape->GetStorageShape();
    auto &attnOutSS = attnOutShape->GetStorageShape();
    auto &softmaxLseSS = softmaxLseShape->GetStorageShape();
    auto &dqSS = context->GetOutputShape(0)->GetStorageShape();
    auto &dkSS = context->GetOutputShape(1)->GetStorageShape();
    auto &dvSS = context->GetOutputShape(2)->GetStorageShape();

    size_t expectDims = isTnd ? 3 : 4;
    OP_CHECK_IF(qSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "q must be %zuD when layout is %s, but got %ld dims.", expectDims, layoutQStr.c_str(),
                        qSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(kSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "k must be %zuD when layout is %s, but got %ld dims.", expectDims, layoutQStr.c_str(),
                        kSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(vSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "v must be %zuD when layout is %s, but got %ld dims.", expectDims, layoutQStr.c_str(),
                        vSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(doutSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "dout must be %zuD when layout is %s, but got %ld dims.", expectDims,
                        layoutQStr.c_str(), doutSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(attnOutSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "attn_out must be %zuD when layout is %s, but got %ld dims.", expectDims,
                        layoutQStr.c_str(), attnOutSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dqSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "dq must be %zuD when layout is %s, but got %ld dims.", expectDims, layoutQStr.c_str(),
                        dqSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dkSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "dk must be %zuD when layout is %s, but got %ld dims.", expectDims, layoutQStr.c_str(),
                        dkSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dvSS.GetDimNum() != expectDims,
                OP_LOGE(opName, "dv must be %zuD when layout is %s, but got %ld dims.", expectDims, layoutQStr.c_str(),
                        dvSS.GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);

    if (isTnd) {
        OP_CHECK_IF(softmaxLseSS.GetDimNum() != 2,
                    OP_LOGE(opName, "softmax_lse must be 2D (N, T) when layout is TND, but got %ld dims.",
                            softmaxLseSS.GetDimNum()),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // D: Dq==Dk==Ddq==Ddk; Dv==Dv_dout==Ddv==Dattn_out
    // Dq may differ from Dv
    size_t dIdx = isTnd ? 2 : 3;
    int64_t qD = qSS.GetDim(dIdx);
    OP_CHECK_IF(kSS.GetDim(dIdx) != qD,
                OP_LOGE(opName, "D dim of k must equal q D(%ld), but got %ld.", qD, kSS.GetDim(dIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dqSS.GetDim(dIdx) != qD,
                OP_LOGE(opName, "D dim of dq must equal q D(%ld), but got %ld.", qD, dqSS.GetDim(dIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dkSS.GetDim(dIdx) != qD,
                OP_LOGE(opName, "D dim of dk must equal q D(%ld), but got %ld.", qD, dkSS.GetDim(dIdx)),
                return ge::GRAPH_PARAM_INVALID);
    int64_t vD = vSS.GetDim(dIdx);
    OP_CHECK_IF(doutSS.GetDim(dIdx) != vD,
                OP_LOGE(opName, "D dim of dout must equal v D(%ld), but got %ld.", vD, doutSS.GetDim(dIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dvSS.GetDim(dIdx) != vD,
                OP_LOGE(opName, "D dim of dv must equal v D(%ld), but got %ld.", vD, dvSS.GetDim(dIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(attnOutSS.GetDim(dIdx) != vD,
                OP_LOGE(opName, "D dim of attn_out must equal v D(%ld), but got %ld.", vD, attnOutSS.GetDim(dIdx)),
                return ge::GRAPH_PARAM_INVALID);
    // Dv <= D is required, not merely conventional: the kernel allocates every
    // on-chip tile at the D-derived width and fills only Dv columns on the V/dO
    // side (see the d_align tilingkey field). A Dv > D would overrun those tiles.
    // The AscendC reference does not check this; we do rather than inherit it.
    OP_CHECK_IF(vD > qD, OP_LOGE(opName, "D dim of v (%ld) must not exceed D dim of q (%ld).", vD, qD),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(vD <= 0 || qD <= 0, OP_LOGE(opName, "D dims must be positive, but got q D=%ld, v D=%ld.", qD, vD),
                return ge::GRAPH_PARAM_INVALID);

    // N: q.N==dout.N==dq.N==attn_out.N; k.N==v.N==dk.N==dv.N; N1%N2==0
    size_t nIdx = isTnd ? 1 : (isBnsd ? 1 : 2);
    int64_t qN = qSS.GetDim(nIdx);
    int64_t kN = kSS.GetDim(nIdx);
    OP_CHECK_IF(doutSS.GetDim(nIdx) != qN,
                OP_LOGE(opName, "N dim of dout must equal q N(%ld), but got %ld.", qN, doutSS.GetDim(nIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dqSS.GetDim(nIdx) != qN,
                OP_LOGE(opName, "N dim of dq must equal q N(%ld), but got %ld.", qN, dqSS.GetDim(nIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(attnOutSS.GetDim(nIdx) != qN,
                OP_LOGE(opName, "N dim of attn_out must equal q N(%ld), but got %ld.", qN, attnOutSS.GetDim(nIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(vSS.GetDim(nIdx) != kN,
                OP_LOGE(opName, "N dim of v must equal k N(%ld), but got %ld.", kN, vSS.GetDim(nIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dkSS.GetDim(nIdx) != kN,
                OP_LOGE(opName, "N dim of dk must equal k N(%ld), but got %ld.", kN, dkSS.GetDim(nIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(dvSS.GetDim(nIdx) != kN,
                OP_LOGE(opName, "N dim of dv must equal k N(%ld), but got %ld.", kN, dvSS.GetDim(nIdx)),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(qN % kN != 0, OP_LOGE(opName, "N dim of q(%ld) must be divisible by N dim of k(%ld).", qN, kN),
                return ge::GRAPH_PARAM_INVALID);

    // S: q.S==dout.S==dq.S==attn_out.S; k.S==v.S==dk.S==dv.S
    if (!isTnd) {
        size_t sIdx = isBnsd ? 2 : 1;
        int64_t qS = qSS.GetDim(sIdx);
        int64_t kS = kSS.GetDim(sIdx);
        OP_CHECK_IF(doutSS.GetDim(sIdx) != qS,
                    OP_LOGE(opName, "S dim of dout must equal q S(%ld), but got %ld.", qS, doutSS.GetDim(sIdx)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dqSS.GetDim(sIdx) != qS,
                    OP_LOGE(opName, "S dim of dq must equal q S(%ld), but got %ld.", qS, dqSS.GetDim(sIdx)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(attnOutSS.GetDim(sIdx) != qS,
                    OP_LOGE(opName, "S dim of attn_out must equal q S(%ld), but got %ld.", qS, attnOutSS.GetDim(sIdx)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(vSS.GetDim(sIdx) != kS,
                    OP_LOGE(opName, "S dim of v must equal k S(%ld), but got %ld.", kS, vSS.GetDim(sIdx)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dkSS.GetDim(sIdx) != kS,
                    OP_LOGE(opName, "S dim of dk must equal k S(%ld), but got %ld.", kS, dkSS.GetDim(sIdx)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dvSS.GetDim(sIdx) != kS,
                    OP_LOGE(opName, "S dim of dv must equal k S(%ld), but got %ld.", kS, dvSS.GetDim(sIdx)),
                    return ge::GRAPH_PARAM_INVALID);
    }

    if (!isTnd) {
        // B consistency
        int64_t bDim = qSS.GetDim(0);
        OP_CHECK_IF(kSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of k must equal q B(%ld), but got %ld.", bDim, kSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(vSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of v must equal q B(%ld), but got %ld.", bDim, vSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(doutSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of dout must equal q B(%ld), but got %ld.", bDim, doutSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dqSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of dq must equal q B(%ld), but got %ld.", bDim, dqSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dkSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of dk must equal q B(%ld), but got %ld.", bDim, dkSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(dvSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of dv must equal q B(%ld), but got %ld.", bDim, dvSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(attnOutSS.GetDim(0) != bDim,
                    OP_LOGE(opName, "B dim of attn_out must equal q B(%ld), but got %ld.", bDim, attnOutSS.GetDim(0)),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // cu_seqlens existence check
    if (isTnd) {
        OP_CHECK_IF(!IsTensorExist(context, CU_SEQLENS_Q_INDEX),
                    OP_LOGE(opName, "cu_seqlens_q must be provided when layout is TND."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(!IsTensorExist(context, CU_SEQLENS_KV_INDEX),
                    OP_LOGE(opName, "cu_seqlens_kv must be provided when layout is TND."),
                    return ge::GRAPH_PARAM_INVALID);
        auto cuSeqQShape = context->GetOptionalInputShape(CU_SEQLENS_Q_INDEX);
        OP_CHECK_IF(
            cuSeqQShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE(opName, "cu_seqlens_q must be 1D, but got %ld dims.", cuSeqQShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_PARAM_INVALID);
        auto cuSeqKvShape = context->GetOptionalInputShape(CU_SEQLENS_KV_INDEX);
        OP_CHECK_IF(
            cuSeqKvShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE(opName, "cu_seqlens_kv must be 1D, but got %ld dims.", cuSeqKvShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_PARAM_INVALID);
    } else {
        OP_CHECK_IF(IsTensorExist(context, CU_SEQLENS_Q_INDEX),
                    OP_LOGE(opName, "cu_seqlens_q must not be provided when layout is not TND."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(IsTensorExist(context, CU_SEQLENS_KV_INDEX),
                    OP_LOGE(opName, "cu_seqlens_kv must not be provided when layout is not TND."),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnGradCheck::CheckParams(gert::TilingContext *context)
{
    const char *opName = context->GetNodeName();

    auto ret = CheckInputExistence(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckInputExistence failed."), return ret);

    ret = CheckDtypeConsistency(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckDtypeConsistency failed."), return ret);

    int64_t maskMode = 0;
    int64_t winLeft = -1;
    int64_t winRight = -1;
    std::string layoutQStr = "BSND";
    std::string layoutKvStr = "BSND";
    std::string layoutOutStr = "BSND";
    ret = CheckAttrs(context, maskMode, winLeft, winRight, layoutQStr, layoutKvStr, layoutOutStr);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckAttrs failed."), return ret);

    ret = CheckMaskAndAttnMask(context, maskMode, winLeft, winRight);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckMaskAndAttnMask failed."), return ret);

    ret = CheckLayout(context, layoutQStr, layoutKvStr, layoutOutStr);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckLayout failed."), return ret);

    ret = CheckShape(context, layoutQStr);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName, "CheckShape failed."), return ret);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
