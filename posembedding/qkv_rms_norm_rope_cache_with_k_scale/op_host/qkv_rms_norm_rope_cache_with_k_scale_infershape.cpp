/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace QkvRmsNormRopeCacheWithKScale {
static constexpr uint64_t QKV_INDEX = 0;
static constexpr uint64_t K_CACHE_INDEX = 5;
static constexpr uint64_t V_CACHE_INDEX = 6;
static constexpr uint64_t K_SCALE_CACHE_INDEX = 7;
static constexpr uint64_t QUERY_START_LOC_INDEX = 8;
static constexpr uint64_t SEQ_LENS_INDEX = 9;
static constexpr uint64_t ROTATION_INDEX = 10;
static constexpr uint64_t MROPE_POSITION_INDEX = 12;
static constexpr uint64_t HEAD_NUMS_ATTR_INDEX = 0;
static constexpr uint64_t QKV_LAYOUT_ATTR_INDEX = 1;
static constexpr uint64_t Q_OUT_LAYOUT_ATTR_INDEX = 2;
static constexpr uint64_t MROPE_SECTION_ATTR_INDEX = 4;
static constexpr uint64_t Q_QUANT_MODE_ATTR_INDEX = 5;
static constexpr uint64_t Q_OUT_DTYPE_ATTR_INDEX = 6;
static constexpr uint64_t K_QUANT_MODE_ATTR_INDEX = 7;

static constexpr uint64_t Q_OUT_INDEX = 0;
static constexpr uint64_t Q_SCALE_INDEX = 1;
static constexpr uint64_t K_CACHE_OUT_INDEX = 2;
static constexpr uint64_t V_CACHE_OUT_INDEX = 3;
static constexpr uint64_t K_SCALE_CACHE_OUT_INDEX = 4;

static constexpr uint64_t QKV_DIM_NUM = 3;
static constexpr int64_t MAX_Q_HEAD_NUM = 64;
static constexpr int64_t SUPPORTED_HEAD_DIM = 128;
static constexpr const char *QKV_LAYOUT_NTD = "NTD";
static constexpr const char *QKV_LAYOUT_TND = "TND";
static constexpr const char *Q_QUANT_MODE_PER_TOKEN_PER_HEAD = "PerTokenPerHead";
static constexpr const char *Q_QUANT_MODE_NO_QUANT = "NoQuant";
static constexpr const char *Q_QUANT_MODE_MX = "Mx";
static constexpr const char *K_QUANT_MODE_PER_TOKEN_PER_HEAD = "PerTokenPerHead";
static constexpr const char *K_QUANT_MODE_MX = "Mx";
static constexpr const char *DEFAULT_OP_NAME = "QkvRmsNormRopeCacheWithKScale";
static constexpr int64_t MX_SCALE_GROUP_SIZE = 32;

enum class QScaleKind {
    FP32_RANK2,
    INTERNAL_PLACEHOLDER,
    E8M0_RANK3,
};

struct InferAttrs {
    int64_t qHeads = 0;
    int64_t kHeads = 0;
    int64_t vHeads = 0;
    bool isQkvNtd = true;
    bool isQOutNtd = true;
    QScaleKind qScaleKind = QScaleKind::FP32_RANK2;
};

struct InputShapes {
    const gert::Shape *qkv = nullptr;
    const gert::Shape *kCache = nullptr;
    const gert::Shape *vCache = nullptr;
    const gert::Shape *kScaleCache = nullptr;
};

struct OutputShapes {
    gert::Shape *qOut = nullptr;
    gert::Shape *qScale = nullptr;
    gert::Shape *kCache = nullptr;
    gert::Shape *vCache = nullptr;
    gert::Shape *kScaleCache = nullptr;
};

struct QOutputDims {
    int64_t totalTokens = 0;
    int64_t headDim = 0;
};

static const char *GetLogOpName(gert::InferShapeContext *context)
{
    return context == nullptr || context->GetNodeName() == nullptr ? DEFAULT_OP_NAME : context->GetNodeName();
}

static std::string RankString(uint64_t dimNum)
{
    return std::to_string(dimNum) + "D";
}

static std::string ShapeString(const gert::Shape *shape)
{
    return shape == nullptr ? "nullptr" : Ops::Base::ToString(*shape);
}

static ge::graphStatus CheckNotNull(gert::InferShapeContext *context, const void *ptr, const char *paramName,
                                    const char *reason)
{
    if (ptr != nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetLogOpName(context), paramName, "nullptr", reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus ParseLayoutQkv(gert::InferShapeContext *context, const gert::RuntimeAttrs *attrs, bool &isNtd)
{
    if (static_cast<uint64_t>(attrs->GetAttrNum()) <= QKV_LAYOUT_ATTR_INDEX) {
        isNtd = false;
        return ge::GRAPH_SUCCESS;
    }
    const char *layoutQkv = attrs->GetStr(QKV_LAYOUT_ATTR_INDEX);
    if (layoutQkv == nullptr || layoutQkv[0] == '\0') {
        isNtd = false;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(layoutQkv, QKV_LAYOUT_NTD) == 0 || std::strcmp(layoutQkv, QKV_LAYOUT_TND) == 0) {
        isNtd = std::strcmp(layoutQkv, QKV_LAYOUT_NTD) == 0;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE(GetLogOpName(context), "layout_qkv", layoutQkv, "NTD or TND");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus ParseLayoutQOut(gert::InferShapeContext *context, const gert::RuntimeAttrs *attrs, bool qkvIsNtd,
                                       bool &qOutIsNtd)
{
    if (static_cast<uint64_t>(attrs->GetAttrNum()) <= Q_OUT_LAYOUT_ATTR_INDEX) {
        qOutIsNtd = true;
        return ge::GRAPH_SUCCESS;
    }
    const char *layoutQOut = attrs->GetStr(Q_OUT_LAYOUT_ATTR_INDEX);
    if (layoutQOut == nullptr || layoutQOut[0] == '\0') {
        qOutIsNtd = true;
        return ge::GRAPH_SUCCESS;
    }
    const bool isNtd = std::strcmp(layoutQOut, QKV_LAYOUT_NTD) == 0;
    const bool isTnd = std::strcmp(layoutQOut, QKV_LAYOUT_TND) == 0;
    if (!isNtd && !isTnd) {
        OP_LOGE_FOR_INVALID_VALUE(GetLogOpName(context), "layout_q_out", layoutQOut, "NTD or TND");
        return ge::GRAPH_FAILED;
    }
    if (qkvIsNtd && isTnd) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetLogOpName(context), "layout_q_out", layoutQOut,
                                              "layout_qkv=NTD with layout_q_out=TND is not supported");
        return ge::GRAPH_FAILED;
    }
    qOutIsNtd = isNtd;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseHeadNums(gert::InferShapeContext *context, const gert::RuntimeAttrs *attrs,
                                     InferAttrs &inferAttrs)
{
    const auto headNums = attrs->GetListInt(HEAD_NUMS_ATTR_INDEX);
    if (CheckNotNull(context, headNums, "head_nums", "head_nums must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t headNumsSize = static_cast<uint64_t>(headNums->GetSize());
    if (headNumsSize != 3U) {
        const std::string incorrectSize = std::to_string(headNumsSize);
        const std::string correctSize = "3";
        OP_LOGE_FOR_INVALID_LISTSIZE(GetLogOpName(context), "head_nums", incorrectSize.c_str(), correctSize.c_str());
        return ge::GRAPH_FAILED;
    }
    const auto *headNumsData = headNums->GetData();
    if (CheckNotNull(context, headNumsData, "head_nums", "head_nums data must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    inferAttrs.qHeads = headNumsData[0];
    inferAttrs.kHeads = headNumsData[1];
    inferAttrs.vHeads = headNumsData[2];
    if (inferAttrs.qHeads <= 0 || inferAttrs.qHeads > MAX_Q_HEAD_NUM || inferAttrs.kHeads <= 0 ||
        inferAttrs.vHeads <= 0 || inferAttrs.qHeads % 8 != 0 || inferAttrs.kHeads != inferAttrs.qHeads / 8 ||
        inferAttrs.vHeads != inferAttrs.kHeads) {
        const std::string incorrectValue = std::to_string(inferAttrs.qHeads) + ", " +
                                           std::to_string(inferAttrs.kHeads) + ", " + std::to_string(inferAttrs.vHeads);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetLogOpName(context), "head_nums", incorrectValue.c_str(),
                                              "head_nums must satisfy 0<Nq<=64, Nq=8*Nk, and Nk=Nv");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static const char *GetQuantModeOrDefault(const gert::RuntimeAttrs *attrs, uint64_t attrIndex, const char *defaultMode)
{
    if (static_cast<uint64_t>(attrs->GetAttrNum()) <= attrIndex) {
        return defaultMode;
    }
    const char *mode = attrs->GetStr(attrIndex);
    return mode == nullptr || mode[0] == '\0' ? defaultMode : mode;
}

static bool HasMropeSection(const gert::RuntimeAttrs *attrs)
{
    if (static_cast<uint64_t>(attrs->GetAttrNum()) <= MROPE_SECTION_ATTR_INDEX) {
        return false;
    }
    const auto *section = attrs->GetListInt(MROPE_SECTION_ATTR_INDEX);
    return section != nullptr && section->GetSize() != 0;
}

static ge::graphStatus ParseSceneAndQScaleKind(gert::InferShapeContext *context, const gert::RuntimeAttrs *attrs,
                                               const InferAttrs &inferAttrs, QScaleKind &kind)
{
    const bool hasQueryStartLoc = context->GetOptionalInputShape(QUERY_START_LOC_INDEX) != nullptr;
    const bool hasSeqLens = context->GetOptionalInputShape(SEQ_LENS_INDEX) != nullptr;
    const bool hasRotation = context->GetOptionalInputShape(ROTATION_INDEX) != nullptr;
    const bool hasMropePosition = context->GetOptionalInputShape(MROPE_POSITION_INDEX) != nullptr;
    const bool hasMropeSection = HasMropeSection(attrs);
    const char *qMode = GetQuantModeOrDefault(attrs, Q_QUANT_MODE_ATTR_INDEX, Q_QUANT_MODE_PER_TOKEN_PER_HEAD);
    const char *kMode = GetQuantModeOrDefault(attrs, K_QUANT_MODE_ATTR_INDEX, K_QUANT_MODE_PER_TOKEN_PER_HEAD);

    if (hasMropePosition != hasMropeSection) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetLogOpName(context), "mrope_position/mrope_section",
                                              "presence mismatch", "both parameters must be provided together");
        return ge::GRAPH_FAILED;
    }

    const bool isRope = !hasMropePosition && hasQueryStartLoc && hasSeqLens && hasRotation &&
                        std::strcmp(qMode, Q_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0 &&
                        std::strcmp(kMode, K_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0;
    if (isRope) {
        kind = QScaleKind::FP32_RANK2;
        return ge::GRAPH_SUCCESS;
    }

    const bool isMropeLayout = !inferAttrs.isQkvNtd && !inferAttrs.isQOutNtd;
    const bool isMrope = hasMropePosition && !hasQueryStartLoc && !hasSeqLens && hasRotation && isMropeLayout &&
                         std::strcmp(qMode, Q_QUANT_MODE_NO_QUANT) == 0 &&
                         std::strcmp(kMode, K_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0;
    if (isMrope) {
        kind = QScaleKind::INTERNAL_PLACEHOLDER;
        return ge::GRAPH_SUCCESS;
    }

    const bool isMropeMx = hasMropePosition && !hasQueryStartLoc && !hasSeqLens && !hasRotation && isMropeLayout &&
                           std::strcmp(qMode, Q_QUANT_MODE_MX) == 0 && std::strcmp(kMode, K_QUANT_MODE_MX) == 0;
    if (isMropeMx) {
        kind = QScaleKind::E8M0_RANK3;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
        GetLogOpName(context), "scene/q_quant_mode/k_quant_mode", "unsupported combination",
        "only RoPE(PerTokenPerHead,PerTokenPerHead,rotation present), M-RoPE"
        "(NoQuant,PerTokenPerHead,rotation present), and M-RoPE MX(Mx,Mx,rotation absent) are supported");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus ValidateQkvShapeForOutput(gert::InferShapeContext *context, const gert::Shape *qkvShape,
                                                 const InferAttrs &attrs, QOutputDims &dims)
{
    const uint64_t actualDimNum = static_cast<uint64_t>(qkvShape->GetDimNum());
    if (actualDimNum != QKV_DIM_NUM) {
        const std::string actualRank = RankString(actualDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetLogOpName(context), "qkv", actualRank.c_str(),
                                                 "qkv rank must be exactly 3D");
        return ge::GRAPH_FAILED;
    }
    dims.totalTokens = qkvShape->GetDim(attrs.isQkvNtd ? 1 : 0);
    dims.headDim = qkvShape->GetDim(2);
    if (dims.totalTokens <= 0 || dims.headDim != SUPPORTED_HEAD_DIM) {
        const std::string incorrectShape = ShapeString(qkvShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetLogOpName(context), "qkv", incorrectShape.c_str(),
                                              "qkv token dimension must be positive and head dimension must be 128");
        return ge::GRAPH_FAILED;
    }

    const int64_t qkvHeads = qkvShape->GetDim(attrs.isQkvNtd ? 0 : 1);
    const int64_t expectedHeads = attrs.qHeads + attrs.kHeads + attrs.vHeads;
    if (qkvHeads != expectedHeads) {
        const std::string incorrectValues = std::to_string(qkvHeads) + ", " + std::to_string(expectedHeads);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(GetLogOpName(context), "qkv, head_nums", incorrectValues.c_str(),
                                               "qkv logical head dimension must equal Nq+Nk+Nv from head_nums");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseInferAttrs(gert::InferShapeContext *context, const gert::RuntimeAttrs *attrs,
                                       InferAttrs &inferAttrs)
{
    if (ParseHeadNums(context, attrs, inferAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseLayoutQkv(context, attrs, inferAttrs.isQkvNtd) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseLayoutQOut(context, attrs, inferAttrs.isQkvNtd, inferAttrs.isQOutNtd) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseSceneAndQScaleKind(context, attrs, inferAttrs, inferAttrs.qScaleKind) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputShapes(gert::InferShapeContext *context, InputShapes &input)
{
    input.qkv = context->GetInputShape(QKV_INDEX);
    if (CheckNotNull(context, input.qkv, "qkv", "qkv shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    input.kCache = context->GetInputShape(K_CACHE_INDEX);
    if (CheckNotNull(context, input.kCache, "k_cache", "k_cache shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    input.vCache = context->GetInputShape(V_CACHE_INDEX);
    if (CheckNotNull(context, input.vCache, "v_cache", "v_cache shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    input.kScaleCache = context->GetInputShape(K_SCALE_CACHE_INDEX);
    if (CheckNotNull(context, input.kScaleCache, "k_scale_cache", "k_scale_cache shape must exist") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetOutputShapes(gert::InferShapeContext *context, OutputShapes &output)
{
    output.qOut = context->GetOutputShape(Q_OUT_INDEX);
    if (CheckNotNull(context, output.qOut, "q_out", "q_out shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    output.qScale = context->GetOutputShape(Q_SCALE_INDEX);
    if (CheckNotNull(context, output.qScale, "q_scale", "q_scale shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    output.kCache = context->GetOutputShape(K_CACHE_OUT_INDEX);
    if (CheckNotNull(context, output.kCache, "k_cache_out", "k_cache_out shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    output.vCache = context->GetOutputShape(V_CACHE_OUT_INDEX);
    if (CheckNotNull(context, output.vCache, "v_cache_out", "v_cache_out shape must exist") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    output.kScaleCache = context->GetOutputShape(K_SCALE_CACHE_OUT_INDEX);
    if (CheckNotNull(context, output.kScaleCache, "k_scale_cache_out", "k_scale_cache_out shape must exist") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void SetOutputShapes(const InputShapes &input, const OutputShapes &output, const InferAttrs &attrs,
                            const QOutputDims &dims)
{
    output.qOut->SetDimNum(3);
    output.qOut->SetDim(0, attrs.isQOutNtd ? attrs.qHeads : dims.totalTokens);
    output.qOut->SetDim(1, attrs.isQOutNtd ? dims.totalTokens : attrs.qHeads);
    output.qOut->SetDim(2, dims.headDim);

    if (attrs.qScaleKind == QScaleKind::E8M0_RANK3) {
        output.qScale->SetDimNum(3);
        output.qScale->SetDim(0, dims.totalTokens);
        output.qScale->SetDim(1, attrs.qHeads);
        output.qScale->SetDim(2, (dims.headDim + MX_SCALE_GROUP_SIZE - 1) / MX_SCALE_GROUP_SIZE);
    } else {
        // NoQuant keeps a rank-2 FP32 internal placeholder for the fixed device ABI.
        output.qScale->SetDimNum(2);
        output.qScale->SetDim(0, attrs.isQOutNtd ? attrs.qHeads : dims.totalTokens);
        output.qScale->SetDim(1, attrs.isQOutNtd ? dims.totalTokens : attrs.qHeads);
    }

    *output.kCache = *input.kCache;
    *output.vCache = *input.vCache;
    *output.kScaleCache = *input.kScaleCache;
}

ge::graphStatus InferShape4QkvRmsNormRopeCacheWithKScale(gert::InferShapeContext *context)
{
    if (CheckNotNull(context, context, "context", "context can not be nullptr") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto *runtimeAttrs = context->GetAttrs();
    if (CheckNotNull(context, runtimeAttrs, "attrs", "attrs can not be nullptr") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    InputShapes input;
    if (GetInputShapes(context, input) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OutputShapes output;
    if (GetOutputShapes(context, output) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    InferAttrs attrs;
    if (ParseInferAttrs(context, runtimeAttrs, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QOutputDims dims;
    if (ValidateQkvShapeForOutput(context, input.qkv, attrs, dims) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetOutputShapes(input, output, attrs, dims);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4QkvRmsNormRopeCacheWithKScale(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DEFAULT_OP_NAME, "context", "nullptr", "context can not be nullptr");
        return ge::GRAPH_FAILED;
    }
    ge::DataType qOutDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType qScaleDtype = ge::DT_FLOAT;
    const auto *attrs = context->GetAttrs();
    const char *qQuantMode = attrs == nullptr ?
                                 Q_QUANT_MODE_PER_TOKEN_PER_HEAD :
                                 GetQuantModeOrDefault(attrs, Q_QUANT_MODE_ATTR_INDEX, Q_QUANT_MODE_PER_TOKEN_PER_HEAD);
    const char *kQuantMode = attrs == nullptr ?
                                 K_QUANT_MODE_PER_TOKEN_PER_HEAD :
                                 GetQuantModeOrDefault(attrs, K_QUANT_MODE_ATTR_INDEX, K_QUANT_MODE_PER_TOKEN_PER_HEAD);
    const bool isRopeModePair = std::strcmp(qQuantMode, Q_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0 &&
                                std::strcmp(kQuantMode, K_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0;
    const bool isMropeModePair = std::strcmp(qQuantMode, Q_QUANT_MODE_NO_QUANT) == 0 &&
                                 std::strcmp(kQuantMode, K_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0;
    const bool isMropeMxModePair =
        std::strcmp(qQuantMode, Q_QUANT_MODE_MX) == 0 && std::strcmp(kQuantMode, K_QUANT_MODE_MX) == 0;
    if (!isRopeModePair && !isMropeModePair && !isMropeMxModePair) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            DEFAULT_OP_NAME, "q_quant_mode/k_quant_mode", "unsupported combination",
            "only (PerTokenPerHead,PerTokenPerHead), (NoQuant,PerTokenPerHead), and (Mx,Mx) are supported");
        return ge::GRAPH_FAILED;
    }
    if (isMropeMxModePair) {
        qScaleDtype = ge::DT_FLOAT8_E8M0;
    }
    if (attrs != nullptr && static_cast<uint64_t>(attrs->GetAttrNum()) > Q_OUT_DTYPE_ATTR_INDEX) {
        const int64_t *qOutDtypeAttr = attrs->GetInt(Q_OUT_DTYPE_ATTR_INDEX);
        if (qOutDtypeAttr != nullptr) {
            const auto requestedDtype = static_cast<ge::DataType>(*qOutDtypeAttr);
            if (requestedDtype != ge::DT_FLOAT8_E4M3FN && requestedDtype != ge::DT_BF16) {
                const std::string incorrectValue = std::to_string(*qOutDtypeAttr);
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DEFAULT_OP_NAME, "q_out_dtype", incorrectValue.c_str(),
                                                      "q_out_dtype must be ge::DT_FLOAT8_E4M3FN or ge::DT_BF16");
                return ge::GRAPH_FAILED;
            }
            qOutDtype = requestedDtype;
        }
    }
    const ge::DataType expectedQOutDtype = isMropeModePair ? ge::DT_BF16 : ge::DT_FLOAT8_E4M3FN;
    if (qOutDtype != expectedQOutDtype) {
        const std::string incorrectValue = std::to_string(static_cast<int64_t>(qOutDtype));
        const std::string reason = isMropeModePair ? "M-RoPE q_out_dtype must be ge::DT_BF16" :
                                                     "RoPE and M-RoPE MX q_out_dtype must be ge::DT_FLOAT8_E4M3FN";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(DEFAULT_OP_NAME, "q_out_dtype", incorrectValue.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(Q_OUT_INDEX, qOutDtype);
    context->SetOutputDataType(Q_SCALE_INDEX, qScaleDtype);
    context->SetOutputDataType(K_CACHE_OUT_INDEX, context->GetInputDataType(K_CACHE_INDEX));
    context->SetOutputDataType(V_CACHE_OUT_INDEX, context->GetInputDataType(V_CACHE_INDEX));
    context->SetOutputDataType(K_SCALE_CACHE_OUT_INDEX, context->GetInputDataType(K_SCALE_CACHE_INDEX));
    return ge::GRAPH_SUCCESS;
}
} // namespace QkvRmsNormRopeCacheWithKScale

IMPL_OP_INFERSHAPE(QkvRmsNormRopeCacheWithKScale)
    .InferShape(QkvRmsNormRopeCacheWithKScale::InferShape4QkvRmsNormRopeCacheWithKScale)
    .InferDataType(QkvRmsNormRopeCacheWithKScale::InferDtype4QkvRmsNormRopeCacheWithKScale);
