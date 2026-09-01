/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../../../op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h"
#include "log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "securec.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace QkvRmsNormRopeCacheWithKScale {
namespace {
constexpr uint64_t QKV_INDEX = 0;
constexpr uint64_t Q_GAMMA_INDEX = 1;
constexpr uint64_t K_GAMMA_INDEX = 2;
constexpr uint64_t COS_SIN_INDEX = 3;
constexpr uint64_t SLOT_MAPPING_INDEX = 4;
constexpr uint64_t K_CACHE_INDEX = 5;
constexpr uint64_t V_CACHE_INDEX = 6;
constexpr uint64_t K_SCALE_CACHE_INDEX = 7;
constexpr uint64_t QUERY_START_LOC_INDEX = 8;
constexpr uint64_t SEQ_LENS_INDEX = 9;
constexpr uint64_t ROTATION_INDEX = 10;
constexpr uint64_t V_SCALE_INDEX = 11;
constexpr uint64_t MROPE_POSITION_INDEX = 12;

constexpr uint64_t Q_OUT_INDEX = 0;
constexpr uint64_t Q_SCALE_INDEX = 1;

constexpr uint64_t HEAD_NUMS_ATTR_INDEX = 0;
constexpr uint64_t QKV_LAYOUT_ATTR_INDEX = 1;
constexpr uint64_t Q_OUT_LAYOUT_ATTR_INDEX = 2;
constexpr uint64_t EPSILON_ATTR_INDEX = 3;
constexpr uint64_t MROPE_SECTION_ATTR_INDEX = 4;
constexpr uint64_t Q_QUANT_MODE_ATTR_INDEX = 5;
constexpr uint64_t Q_OUT_DTYPE_ATTR_INDEX = 6;
constexpr uint64_t K_QUANT_MODE_ATTR_INDEX = 7;

constexpr uint64_t MROPE_SECTION_SIZE = 3;

constexpr float DEFAULT_EPSILON = 1e-6f;
constexpr int32_t TILING_TEMPLATE_PRIORITY = 1000;
constexpr uint64_t WORKSPACE_COUNT = 1;
constexpr uint64_t RESERVED_WORKSPACE_SIZE = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t TOKEN_TILE_PER_AIV_CAP = 4;
constexpr uint64_t MX_SCALE_GROUP_SIZE = 32U;
constexpr uint64_t MROPE_MX_GM_STRIDE_MAX_BYTES = (1ULL << 40U) - 1U;
constexpr const char *QKV_LAYOUT_NTD = "NTD";
constexpr const char *QKV_LAYOUT_TND = "TND";
constexpr const char *Q_QUANT_MODE_PER_TOKEN_PER_HEAD = "PerTokenPerHead";
constexpr const char *Q_QUANT_MODE_NO_QUANT = "NoQuant";
constexpr const char *Q_QUANT_MODE_MX = "Mx";
constexpr const char *K_QUANT_MODE_PER_TOKEN_PER_HEAD = "PerTokenPerHead";
constexpr const char *K_QUANT_MODE_MX = "Mx";
constexpr const char *DEFAULT_OP_NAME = "QkvRmsNormRopeCacheWithKScale";

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool FitsUint32(uint64_t value)
{
    return value <= std::numeric_limits<uint32_t>::max();
}

bool FitsUint16(uint64_t value)
{
    return value <= std::numeric_limits<uint16_t>::max();
}

const char *CacheTensorName(uint64_t index)
{
    switch (index) {
        case K_CACHE_INDEX:
            return "kCache";
        case V_CACHE_INDEX:
            return "vCache";
        case K_SCALE_CACHE_INDEX:
            return "kScaleCache";
        default:
            return "cache";
    }
}

int64_t GetStrideDimOrZero(const gert::Stride &stride, uint64_t index)
{
    return index < static_cast<uint64_t>(stride.GetDimNum()) ? stride.GetStride(index) : 0;
}

template <typename DimAt, typename StrideAt>
bool IsNonOverlappingPositiveStrides(std::size_t rank, DimAt dimAt, StrideAt strideAt)
{
    uint64_t innerSpan = 1U;
    for (std::size_t offset = 0U; offset < rank; ++offset) {
        const std::size_t index = rank - 1U - offset;
        const int64_t dim = dimAt(index);
        const int64_t stride = strideAt(index);
        if (dim <= 0 || stride <= 0) {
            return false;
        }
        const uint64_t uDim = static_cast<uint64_t>(dim);
        const uint64_t uStride = static_cast<uint64_t>(stride);
        if (uDim > 1U && uStride < innerSpan) {
            return false;
        }
        const uint64_t extent = uDim - 1U;
        if (extent != 0U && uStride > (std::numeric_limits<uint64_t>::max() - innerSpan) / extent) {
            return false;
        }
        innerSpan += extent * uStride;
    }
    return true;
}
} // namespace

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateShapeRank(const TensorContractInfo &tensor,
                                                                           const char *tensorName,
                                                                           uint64_t expectedRank) const
{
    const uint64_t actualRank = static_cast<uint64_t>(tensor.shape.GetDimNum());
    OP_CHECK_IF(
        !tensor.shapePresent || actualRank != expectedRank,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, tensorName, (std::to_string(actualRank) + "D").c_str(),
                                                 ("shape rank must be " + std::to_string(expectedRank) + "D").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

namespace {
uint64_t CalcQkAlignedRows(uint64_t tokenTile, uint64_t numQHeads, uint64_t numKHeads)
{
    return Ops::Base::CeilAlign(tokenTile * numQHeads, 16UL) + Ops::Base::CeilAlign(tokenTile * numKHeads, 16UL);
}

uint64_t CalcQkPreprocessNzBytes(uint64_t rowCount)
{
    const uint64_t rowStride =
        Ops::Base::CeilAlign(rowCount - 1, QkvRmsNormRopeCacheWithKScaleBaseTiling::QK_PREPROCESS_UB_NZ_STRIDE_ALIGN) +
        1;
    const uint64_t blockCount =
        (QkvRmsNormRopeCacheWithKScaleBaseTiling::QK_PREPROCESS_NZ_D_BLOCKS - 1) * rowStride + rowCount;
    return blockCount * QkvRmsNormRopeCacheWithKScaleBaseTiling::QK_PREPROCESS_BLOCK_BYTES;
}

const char *DtypeName(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16:
            return "bf16";
        case ge::DT_FLOAT:
            return "float";
        case ge::DT_INT32:
            return "int32";
        case ge::DT_INT8:
            return "int8";
        case ge::DT_FLOAT8_E4M3FN:
            return "float8_e4m3fn";
        case ge::DT_FLOAT8_E8M0:
            return "float8_e8m0";
        case ge::DT_UNDEFINED:
            return "undefined";
        default:
            return "unknown";
    }
}

const char *QQuantModeName(QQuantMode mode)
{
    switch (mode) {
        case QQuantMode::PER_TOKEN_PER_HEAD:
            return Q_QUANT_MODE_PER_TOKEN_PER_HEAD;
        case QQuantMode::NO_QUANT:
            return Q_QUANT_MODE_NO_QUANT;
        case QQuantMode::MX_QUANT:
            return Q_QUANT_MODE_MX;
        default:
            return "INVALID";
    }
}

const char *KQuantModeName(KQuantMode mode)
{
    return mode == KQuantMode::MX_QUANT ? K_QUANT_MODE_MX : K_QUANT_MODE_PER_TOKEN_PER_HEAD;
}

const char *LayoutName(uint64_t layout)
{
    switch (layout) {
        case QKV_K_SCALE_LAYOUT_NTD:
            return QKV_LAYOUT_NTD;
        case QKV_K_SCALE_LAYOUT_TND:
            return QKV_LAYOUT_TND;
        default:
            return "INVALID";
    }
}
} // namespace

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateHeadNums() const
{
    OP_CHECK_IF(input_.numQHeads == 0 || input_.numKHeads == 0 || input_.numVHeads == 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "Nq, Nk, Nv",
                    (std::to_string(input_.numQHeads) + ", " + std::to_string(input_.numKHeads) + ", " +
                     std::to_string(input_.numVHeads))
                        .c_str(),
                    "head nums must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input_.numQHeads > MAX_Q_HEAD_NUM,
        OP_LOGE_FOR_INVALID_VALUE(opName_, "Nq", std::to_string(input_.numQHeads).c_str(), "less than or equal to 64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input_.numQHeads % Q_TO_K_HEAD_RATIO != 0 || input_.numKHeads != input_.numQHeads / Q_TO_K_HEAD_RATIO,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "Nq, Nk", (std::to_string(input_.numQHeads) + ", " + std::to_string(input_.numKHeads)).c_str(),
            "nq must be equal to 8 * nk"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input_.numVHeads != input_.numKHeads,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "Nv, Nk", (std::to_string(input_.numVHeads) + ", " + std::to_string(input_.numKHeads)).c_str(),
            "nv must be equal to nk"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateScalarInputs() const
{
    OP_CHECK_IF(input_.layoutQkv != QKV_K_SCALE_LAYOUT_NTD && input_.layoutQkv != QKV_K_SCALE_LAYOUT_TND,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_qkv", LayoutName(input_.layoutQkv), "NTD or TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input_.layoutQOut != QKV_K_SCALE_LAYOUT_NTD && input_.layoutQOut != QKV_K_SCALE_LAYOUT_TND,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_q_out", LayoutName(input_.layoutQOut), "NTD or TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input_.layoutQkv == QKV_K_SCALE_LAYOUT_NTD && input_.layoutQOut == QKV_K_SCALE_LAYOUT_TND,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "layout_qkv, layout_q_out", "NTD, TND",
                                                       "layout_qkv=NTD with layout_q_out=TND is not supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input_.totalTokens == 0,
        OP_LOGE_FOR_INVALID_VALUE(opName_, "totalTokens", std::to_string(input_.totalTokens).c_str(), "greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(input_.totalTokens > MAX_TOKENS,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "totalTokens", std::to_string(input_.totalTokens).c_str(),
                                          "less than or equal to 262144"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GetSceneTraits(scene_).needsMropePosition && input_.batch == 0,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "batch", std::to_string(input_.batch).c_str(), "greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input_.headDim != SUPPORTED_HEAD_DIM,
                OP_LOGE_FOR_INVALID_VALUE(opName_, "headDim", std::to_string(input_.headDim).c_str(),
                                          std::to_string(SUPPORTED_HEAD_DIM).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input_.maxSeqLen == 0,
        OP_LOGE_FOR_INVALID_VALUE(opName_, "maxSeqLen", std::to_string(input_.maxSeqLen).c_str(), "greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(input_.blockNum == 0 || input_.blockSize == 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "blockNum, blockSize",
                    (std::to_string(input_.blockNum) + ", " + std::to_string(input_.blockSize)).c_str(),
                    "blockNum and blockSize must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input_.vScale.shapePresent,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "vScale", "missing", "vScale must exist"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

const SceneTraits &QkvRmsNormRopeCacheWithKScaleBaseTiling::GetSceneTraits(Scene scene)
{
    static const SceneTraits sceneTraits[] = {
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, 1U, true, false, false, false,
         QKV_INPUT_ROWS_PER_AIV},
        {ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT, 2U, true, true, false, false, MROPE_COMPACT_INPUT_ROWS_PER_AIV},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, 2U, false, true, true, true,
         MROPE_COMPACT_INPUT_ROWS_PER_AIV},
    };
    return sceneTraits[static_cast<uint8_t>(scene)];
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateDtypes() const
{
    const auto &traits = GetSceneTraits(scene_);
    struct DtypeRule {
        const char *tensorName;
        const TensorContractInfo *tensor;
        ge::DataType expected;
    };
    std::vector<DtypeRule> rules = {
        {"qkv", &input_.qkv, ge::DT_BF16},
        {"qGamma", &input_.qGamma, ge::DT_FLOAT},
        {"kGamma", &input_.kGamma, ge::DT_FLOAT},
        {"cosSin", &input_.cosSin, ge::DT_FLOAT},
        {"slotMapping", &input_.slotMapping, ge::DT_INT32},
        {"vCache", &input_.vCache, ge::DT_FLOAT8_E4M3FN},
        {"kCache", &input_.kCache, traits.kCacheDtype},
        {"kScaleCache", &input_.kScaleCache, traits.kScaleCacheDtype},
        {"vScale", &input_.vScale, ge::DT_FLOAT},
    };

    if (traits.needsRotation) {
        rules.push_back({"rotation", &input_.rotation, ge::DT_BF16});
    }
    if (traits.needsMropePosition) {
        rules.push_back({"mropePosition", &input_.mropePosition, ge::DT_INT32});
    } else {
        rules.push_back({"queryStartLoc", &input_.queryStartLoc, ge::DT_INT32});
        rules.push_back({"seqLens", &input_.seqLens, ge::DT_INT32});
    }

    for (const auto &rule : rules) {
        OP_CHECK_IF(rule.tensor->descriptorPresent && rule.tensor->dtype != rule.expected,
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, rule.tensorName, DtypeName(rule.tensor->dtype),
                                              DtypeName(rule.expected)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateMropeSection()
{
    if (!GetSceneTraits(scene_).needsMropePosition) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        !hasMropeSection_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "mrope_section", "missing", "M-RoPE requires mrope_section"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(mropeSectionSize_ != MROPE_SECTION_SIZE,
                OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "mrope_section", std::to_string(mropeSectionSize_).c_str(), "3"),
                return ge::GRAPH_FAILED);
    const auto *sectionData = mropeSectionData_;
    OP_CHECK_IF(
        sectionData == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "mrope_section", "null", "mrope_section data must exist"),
        return ge::GRAPH_FAILED);

    const int64_t halfHeadDim = static_cast<int64_t>(input_.headDim / 2);
    int64_t sectionSum = 0;
    for (uint64_t axis = 0; axis < MROPE_SECTION_SIZE; ++axis) {
        const int64_t section = sectionData[axis];
        OP_CHECK_IF(section < 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "mrope_section", std::to_string(section).c_str(),
                        ("mrope_section[" + std::to_string(axis) + "] must be non-negative").c_str()),
                    return ge::GRAPH_FAILED);
        if (axis != 0U) {
            const int64_t axisOffset = static_cast<int64_t>(axis);
            const int64_t axisCapacity =
                halfHeadDim <= axisOffset ?
                    0 :
                    (halfHeadDim - 1 - axisOffset) / static_cast<int64_t>(MROPE_SECTION_SIZE) + 1;
            OP_CHECK_IF(section > axisCapacity,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            opName_, "mrope_section", std::to_string(section).c_str(),
                            ("mrope_section[" + std::to_string(axis) + "] must be less than or equal to " +
                             std::to_string(axisCapacity))
                                .c_str()),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(section > halfHeadDim - sectionSum,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "mrope_section", "sum exceeds D/2",
                                                          "the sum of mrope_section must not exceed D/2"),
                    return ge::GRAPH_FAILED);
        sectionSum += section;
    }
    mropeSectionH_ = static_cast<uint64_t>(sectionData[1]);
    mropeSectionW_ = static_cast<uint64_t>(sectionData[2]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateQOutDtypeAttr() const
{
    const ge::DataType expectedQOutDtype = GetSceneTraits(scene_).qOutDtype;
    OP_CHECK_IF(
        qOutDtypeAttr_ != expectedQOutDtype,
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "q_out_dtype", DtypeName(qOutDtypeAttr_), DtypeName(expectedQOutDtype)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateShapeInputsForDerivedFields() const
{
    const auto &traits = GetSceneTraits(scene_);
    OP_CHECK_IF(!input_.qkv.shapePresent || input_.qkv.shape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "qkv",
                                             (std::to_string(input_.qkv.shape.GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input_.cosSin.shapePresent || input_.cosSin.shape.GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "cosSin",
                                             (std::to_string(input_.cosSin.shape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    if (!traits.needsMropePosition) {
        OP_CHECK_IF(
            !input_.queryStartLoc.shapePresent || input_.queryStartLoc.shape.GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "queryStartLoc",
                                         (std::to_string(input_.queryStartLoc.shape.GetDimNum()) + "D").c_str(), "1D"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(input_.queryStartLoc.shape.GetDim(0) < 2,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "queryStartLoc",
                                                          Ops::Base::ToString(input_.queryStartLoc.shape).c_str(),
                                                          "dim 0 must be greater than or equal to 2"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!input_.seqLens.shapePresent || input_.seqLens.shape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        opName_, "seqLens", (std::to_string(input_.seqLens.shape.GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            input_.seqLens.shape.GetDim(0) != input_.queryStartLoc.shape.GetDim(0) - 1,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "seqLens, queryStartLoc",
                (Ops::Base::ToString(input_.seqLens.shape) + ", " + Ops::Base::ToString(input_.queryStartLoc.shape))
                    .c_str(),
                "dim 0 of seqLens must be equal to dim 0 of queryStartLoc minus 1"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            !input_.mropePosition.shapePresent || input_.mropePosition.shape.GetDimNum() != 2,
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "mropePosition",
                                         (std::to_string(input_.mropePosition.shape.GetDimNum()) + "D").c_str(), "2D"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!input_.kCache.shapePresent || input_.kCache.shape.GetDimNum() != 4,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "kCache",
                                             (std::to_string(input_.kCache.shape.GetDimNum()) + "D").c_str(), "4D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateNonCacheShapeRanks() const
{
    const auto &traits = GetSceneTraits(scene_);
    struct NonCacheRankRule {
        const char *tensorName;
        const TensorContractInfo *tensor;
        uint64_t expectedRank;
    };
    const NonCacheRankRule commonRankRules[] = {
        {"qGamma", &input_.qGamma, 1},
        {"kGamma", &input_.kGamma, 1},
        {"slotMapping", &input_.slotMapping, 1},
    };
    for (const auto &rule : commonRankRules) {
        if (ValidateShapeRank(*rule.tensor, rule.tensorName, rule.expectedRank) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (traits.needsRotation && ValidateShapeRank(input_.rotation, "rotation", 2) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ValidateShapeRank(input_.vScale, "vScale", traits.vScaleRank) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateQkvAndGammaShapes() const
{
    const uint64_t numQkHeads = input_.numQHeads + input_.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input_.numVHeads;
    const bool isNtd = input_.layoutQkv == QKV_K_SCALE_LAYOUT_NTD;
    const uint64_t qkvHeadDim = static_cast<uint64_t>(input_.qkv.shape.GetDim(isNtd ? 0 : 1));
    const uint64_t qkvTokenDim = static_cast<uint64_t>(input_.qkv.shape.GetDim(isNtd ? 1 : 0));
    const uint64_t expectedDim0 = isNtd ? numQkvHeads : input_.totalTokens;
    const uint64_t expectedDim1 = isNtd ? input_.totalTokens : numQkvHeads;
    OP_CHECK_IF(qkvHeadDim != numQkvHeads || qkvTokenDim != input_.totalTokens ||
                    static_cast<uint64_t>(input_.qkv.shape.GetDim(2)) != input_.headDim,
                OP_LOGE_FOR_INVALID_SHAPE(opName_, "qkv", Ops::Base::ToString(input_.qkv.shape).c_str(),
                                          (std::to_string(expectedDim0) + ", " + std::to_string(expectedDim1) + ", " +
                                           std::to_string(input_.headDim))
                                              .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<uint64_t>(input_.qGamma.shape.GetDim(0)) != input_.headDim ||
            static_cast<uint64_t>(input_.kGamma.shape.GetDim(0)) != input_.headDim,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "qGamma, kGamma",
            (Ops::Base::ToString(input_.qGamma.shape) + ", " + Ops::Base::ToString(input_.kGamma.shape)).c_str(),
            ("dim 0 must be equal to headDim " + std::to_string(input_.headDim)).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidatePositionInputShapes() const
{
    OP_CHECK_IF(static_cast<uint64_t>(input_.cosSin.shape.GetDim(0)) != input_.maxSeqLen ||
                    static_cast<uint64_t>(input_.cosSin.shape.GetDim(1)) != input_.headDim,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cosSin", Ops::Base::ToString(input_.cosSin.shape).c_str(),
                    ("shape must be maxSeqLen by headDim, maxSeqLen is " + std::to_string(input_.maxSeqLen) +
                     ", headDim is " + std::to_string(input_.headDim))
                        .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(input_.slotMapping.shape.GetDim(0)) != input_.totalTokens,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "slotMapping", Ops::Base::ToString(input_.slotMapping.shape).c_str(),
                    ("dim 0 must be equal to totalTokens " + std::to_string(input_.totalTokens)).c_str()),
                return ge::GRAPH_FAILED);
    if (GetSceneTraits(scene_).needsMropePosition) {
        OP_CHECK_IF(input_.mropePosition.shape.GetDim(0) != static_cast<int64_t>(input_.totalTokens) ||
                        input_.mropePosition.shape.GetDim(1) != static_cast<int64_t>(MROPE_SECTION_SIZE),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "mropePosition", Ops::Base::ToString(input_.mropePosition.shape).c_str(),
                        ("shape must be [T, 3], T is " + std::to_string(input_.totalTokens)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateRotationAndScaleShapes() const
{
    const auto &traits = GetSceneTraits(scene_);
    if (traits.needsRotation) {
        OP_CHECK_IF(static_cast<uint64_t>(input_.rotation.shape.GetDim(0)) != input_.headDim ||
                        static_cast<uint64_t>(input_.rotation.shape.GetDim(1)) != input_.headDim,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "rotation", Ops::Base::ToString(input_.rotation.shape).c_str(),
                        ("shape must be headDim by headDim, headDim is " + std::to_string(input_.headDim)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(static_cast<uint64_t>(input_.vScale.shape.GetDim(0)) != input_.numVHeads,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "vScale", Ops::Base::ToString(input_.vScale.shape).c_str(),
                    ("dim 0 must be equal to numVHeads " + std::to_string(input_.numVHeads)).c_str()),
                return ge::GRAPH_FAILED);
    if (traits.vScaleRank == 2U) {
        OP_CHECK_IF(static_cast<uint64_t>(input_.vScale.shape.GetDim(1)) != input_.headDim,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "vScale", Ops::Base::ToString(input_.vScale.shape).c_str(),
                        ("dim 1 must be equal to headDim " + std::to_string(input_.headDim)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateNonCacheShapes() const
{
    if (ValidateNonCacheShapeRanks() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateQkvAndGammaShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePositionInputShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateRotationAndScaleShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateCacheShapes() const
{
    const auto &traits = GetSceneTraits(scene_);
    if (ValidateShapeRank(input_.vCache, "vCache", 4) != ge::GRAPH_SUCCESS ||
        ValidateShapeRank(input_.kScaleCache, "kScaleCache", 4U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        static_cast<uint64_t>(input_.kCache.shape.GetDim(0)) != input_.blockNum ||
            static_cast<uint64_t>(input_.kCache.shape.GetDim(1)) != input_.numKHeads ||
            static_cast<uint64_t>(input_.kCache.shape.GetDim(2)) != input_.blockSize ||
            static_cast<uint64_t>(input_.kCache.shape.GetDim(3)) != input_.headDim,
        OP_LOGE_FOR_INVALID_SHAPE(opName_, "kCache", Ops::Base::ToString(input_.kCache.shape).c_str(),
                                  (std::to_string(input_.blockNum) + ", " + std::to_string(input_.numKHeads) + ", " +
                                   std::to_string(input_.blockSize) + ", " + std::to_string(input_.headDim))
                                      .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<uint64_t>(input_.vCache.shape.GetDim(0)) != input_.blockNum ||
            static_cast<uint64_t>(input_.vCache.shape.GetDim(1)) != input_.numVHeads ||
            static_cast<uint64_t>(input_.vCache.shape.GetDim(2)) != input_.blockSize ||
            static_cast<uint64_t>(input_.vCache.shape.GetDim(3)) != input_.headDim,
        OP_LOGE_FOR_INVALID_SHAPE(opName_, "vCache", Ops::Base::ToString(input_.vCache.shape).c_str(),
                                  (std::to_string(input_.blockNum) + ", " + std::to_string(input_.numVHeads) + ", " +
                                   std::to_string(input_.blockSize) + ", " + std::to_string(input_.headDim))
                                      .c_str()),
        return ge::GRAPH_FAILED);
    const uint64_t scaleCount = traits.usesMxScaleLayout ? Ops::Base::CeilDiv(input_.headDim, MX_SCALE_GROUP_SIZE) : 1U;
    const bool invalidScaleShape = static_cast<uint64_t>(input_.kScaleCache.shape.GetDim(0)) != input_.blockNum ||
                                   static_cast<uint64_t>(input_.kScaleCache.shape.GetDim(1)) != input_.numKHeads ||
                                   static_cast<uint64_t>(input_.kScaleCache.shape.GetDim(2)) != input_.blockSize ||
                                   static_cast<uint64_t>(input_.kScaleCache.shape.GetDim(3)) != scaleCount;
    const std::string expectedScaleShape = std::to_string(input_.blockNum) + ", " + std::to_string(input_.numKHeads) +
                                           ", " + std::to_string(input_.blockSize) + ", " + std::to_string(scaleCount);
    OP_CHECK_IF(invalidScaleShape,
                OP_LOGE_FOR_INVALID_SHAPE(opName_, "kScaleCache", Ops::Base::ToString(input_.kScaleCache.shape).c_str(),
                                          expectedScaleShape.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateShapes() const
{
    ge::graphStatus status = ValidateNonCacheShapes();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale non-cache shape validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateCacheShapes();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale cache shape validation failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateStrides() const
{
    const auto &traits = GetSceneTraits(scene_);
    const gert::Stride &kStride = input_.kCache.stride;
    const gert::Stride &vStride = input_.vCache.stride;
    const gert::Stride &kScaleStride = input_.kScaleCache.stride;
    const int64_t headDim = static_cast<int64_t>(input_.headDim);
    const int64_t minKScaleTokenStride =
        traits.usesMxScaleLayout ? static_cast<int64_t>(Ops::Base::CeilDiv(input_.headDim, MX_SCALE_GROUP_SIZE)) : 1;
    OP_CHECK_IF(!input_.kCache.stridePresent || kStride.GetDimNum() != 4 || kStride.GetStride(0) <= 0 ||
                    kStride.GetStride(1) < headDim || kStride.GetStride(2) < headDim || kStride.GetStride(3) != 1,
                OP_LOGE_FOR_INVALID_STRIDE(opName_, "kCache",
                                           (std::to_string(GetStrideDimOrZero(input_.kCache.stride, 0)) + ", " +
                                            std::to_string(GetStrideDimOrZero(input_.kCache.stride, 1)) + ", " +
                                            std::to_string(GetStrideDimOrZero(input_.kCache.stride, 2)) + ", " +
                                            std::to_string(GetStrideDimOrZero(input_.kCache.stride, 3)))
                                               .c_str(),
                                           "positive block stride, stride[1]/stride[2] >= headDim, and stride[3]=1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!input_.vCache.stridePresent || vStride.GetDimNum() != 4 || vStride.GetStride(0) <= 0 ||
                    vStride.GetStride(1) < headDim || vStride.GetStride(2) < headDim || vStride.GetStride(3) != 1,
                OP_LOGE_FOR_INVALID_STRIDE(opName_, "vCache",
                                           (std::to_string(GetStrideDimOrZero(input_.vCache.stride, 0)) + ", " +
                                            std::to_string(GetStrideDimOrZero(input_.vCache.stride, 1)) + ", " +
                                            std::to_string(GetStrideDimOrZero(input_.vCache.stride, 2)) + ", " +
                                            std::to_string(GetStrideDimOrZero(input_.vCache.stride, 3)))
                                               .c_str(),
                                           "positive block stride, stride[1]/stride[2] >= headDim, and stride[3]=1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStride.GetStride(0) != vStride.GetStride(0) || kStride.GetStride(1) != vStride.GetStride(1) ||
                    kStride.GetStride(2) != vStride.GetStride(2),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "kCache stride, vCache stride",
                    (std::to_string(input_.kCache.stride.GetStride(0)) + ", " +
                     std::to_string(input_.kCache.stride.GetStride(1)) + ", " +
                     std::to_string(input_.kCache.stride.GetStride(2)) + " and " +
                     std::to_string(input_.vCache.stride.GetStride(0)) + ", " +
                     std::to_string(input_.vCache.stride.GetStride(1)) + ", " +
                     std::to_string(input_.vCache.stride.GetStride(2)))
                        .c_str(),
                    "the first three dimensions of kCache stride and vCache stride must be equal"),
                return ge::GRAPH_FAILED);
    const bool kNonOverlapping = IsNonOverlappingPositiveStrides(
        4U, [this](std::size_t index) { return input_.kCache.shape.GetDim(index); },
        [&kStride](std::size_t index) { return kStride.GetStride(index); });
    const bool vNonOverlapping = IsNonOverlappingPositiveStrides(
        4U, [this](std::size_t index) { return input_.vCache.shape.GetDim(index); },
        [&vStride](std::size_t index) { return vStride.GetStride(index); });
    OP_CHECK_IF(!kNonOverlapping || !vNonOverlapping,
                OP_LOGE(context_, "K/V cache views overlap or their checked span overflows."), return ge::GRAPH_FAILED);
    const bool invalidKScaleStride = !input_.kScaleCache.stridePresent || kScaleStride.GetDimNum() != 4 ||
                                     kScaleStride.GetStride(0) <= 0 || kScaleStride.GetStride(1) <= 0 ||
                                     kScaleStride.GetStride(2) < minKScaleTokenStride || kScaleStride.GetStride(3) != 1;
    const std::string kScaleStrideValue = std::to_string(GetStrideDimOrZero(input_.kScaleCache.stride, 0)) + ", " +
                                          std::to_string(GetStrideDimOrZero(input_.kScaleCache.stride, 1)) + ", " +
                                          std::to_string(GetStrideDimOrZero(input_.kScaleCache.stride, 2)) + ", " +
                                          std::to_string(GetStrideDimOrZero(input_.kScaleCache.stride, 3));
    OP_CHECK_IF(invalidKScaleStride,
                OP_LOGE_FOR_INVALID_STRIDE(opName_, "kScaleCache", kScaleStrideValue.c_str(),
                                           traits.usesMxScaleLayout ?
                                               "positive 4D stride with stride[2] >= ceil(D/32) and stride[3]=1" :
                                               "positive 4D stride and stride[3]=1"),
                return ge::GRAPH_FAILED);
    const bool kScaleNonOverlapping = IsNonOverlappingPositiveStrides(
        4U, [this](std::size_t index) { return input_.kScaleCache.shape.GetDim(index); },
        [&kScaleStride](std::size_t index) { return kScaleStride.GetStride(index); });
    OP_CHECK_IF(!kScaleNonOverlapping, OP_LOGE(context_, "K scale cache view overlaps or its checked span overflows."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QkvRmsNormRopeCacheWithKScaleBaseTiling::TrySelectTokenTile(uint64_t tokenTile) const
{
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tokenTile, AIV_PER_AIC);
    const uint64_t numQkHeads = input_.numQHeads + input_.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input_.numVHeads;
    const uint64_t qRows = tokenTilePerAiv * input_.numQHeads;
    const uint64_t kRows = tokenTilePerAiv * input_.numKHeads;
    const uint64_t qkPreprocessBytes = CalcQkPreprocessNzBytes(qRows) + CalcQkPreprocessNzBytes(kRows);
    const uint64_t rowTile = tokenTile * numQkHeads;
    const uint64_t rowTileAligned = Ops::Base::CeilAlign(rowTile, 16UL);
    const uint64_t qkAlignedRows = CalcQkAlignedRows(tokenTile, input_.numQHeads, input_.numKHeads);
    const uint64_t qkvInputRowsPerAiv = GetSceneTraits(scene_).qkvInputRowsPerAiv;
    const bool fitsUb = qkPreprocessBytes <= QK_PREPROCESS_UB_BYTES &&
                        tokenTilePerAiv * numQkHeads <= QK_OUTPUT_ROWS_PER_AIV &&
                        tokenTilePerAiv * numQkvHeads <= qkvInputRowsPerAiv &&
                        tokenTilePerAiv * input_.numVHeads <= V_OUTPUT_ROWS_PER_AIV;
    const bool fitsL0c = rowTile <= L0C_MAX_ROWS && rowTileAligned <= L0C_MAX_ROWS && qkAlignedRows <= L0C_MAX_ROWS;
    const bool accepted = fitsUb && fitsL0c;
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale token tile candidate: tokenTilePerAiv=%llu tokenTile=%llu rowTile=%llu "
            "rowTileAligned=%llu qkAlignedRows=%llu qkPreprocessBytes=%llu qkvInputRowsPerAiv=%llu "
            "fitsUb=%u fitsSideBySideL0c=%u accepted=%u.",
            tokenTilePerAiv, tokenTile, rowTile, rowTileAligned, qkAlignedRows, qkPreprocessBytes, qkvInputRowsPerAiv,
            fitsUb ? 1U : 0U, fitsL0c ? 1U : 0U, accepted ? 1U : 0U);
    return accepted;
}

uint64_t QkvRmsNormRopeCacheWithKScaleBaseTiling::SelectTokenTile() const
{
    const uint64_t numQkHeads = input_.numQHeads + input_.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input_.numVHeads;
    const uint64_t qkvInputRowsPerAiv = GetSceneTraits(scene_).qkvInputRowsPerAiv;
    const uint64_t resourceTokenTilePerAiv =
        std::min({MAX_TOKEN_TILE / 2, QK_OUTPUT_ROWS_PER_AIV / numQkHeads, qkvInputRowsPerAiv / numQkvHeads,
                  V_OUTPUT_ROWS_PER_AIV / input_.numVHeads});
    // Limit each AIV's token tile so V-cache preprocess can overlap better with Q/K vector work.
    const uint64_t initialTokenTilePerAiv = std::min(resourceTokenTilePerAiv, TOKEN_TILE_PER_AIV_CAP);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale token tile search start: initialTokenTilePerAiv=%llu "
            "resourceTokenTilePerAiv=%llu tokenTilePerAivCap=%llu qkOutputLimit=%llu qkvInputLimit=%llu "
            "vOutputLimit=%llu.",
            initialTokenTilePerAiv, resourceTokenTilePerAiv, TOKEN_TILE_PER_AIV_CAP,
            QK_OUTPUT_ROWS_PER_AIV / numQkHeads, qkvInputRowsPerAiv / numQkvHeads,
            V_OUTPUT_ROWS_PER_AIV / input_.numVHeads);

    uint64_t searchTokenTilePerAiv = initialTokenTilePerAiv;
    while (searchTokenTilePerAiv > 0) {
        const uint64_t tokenTile = 2 * searchTokenTilePerAiv;
        if (TrySelectTokenTile(tokenTile)) {
            return tokenTile;
        }
        --searchTokenTilePerAiv;
    }
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale token tile search fallback: head contract guarantees tokenTile=1.");
    return 1;
}

bool QkvRmsNormRopeCacheWithKScaleBaseTiling::AreMropeMxCopyAndVfFieldsRepresentable(
    uint64_t tokenTile, const ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout &layout) const
{
    // The controller stores the tile size in uint32_t, while VF token/head
    // loop bounds are uint16_t. A zero totalBytes denotes an invalid layout.
    if (!FitsUint32(tokenTile) || !FitsUint16(tokenTile) || !FitsUint16(input_.numQHeads) ||
        !FitsUint16(input_.numKHeads) || !FitsUint16(input_.numVHeads) || layout.totalBytes == 0U) {
        return false;
    }

    // Cache scatter converts the head gap to an unsigned byte stride. Validate
    // the source stride before subtracting the logical D128 row width.
    const int64_t kvCacheHeadStrideValue = input_.kCache.stride.GetStride(1);
    if (kvCacheHeadStrideValue < 0 || static_cast<uint64_t>(kvCacheHeadStrideValue) < input_.headDim) {
        return false;
    }
    const uint64_t kvCacheHeadStride = static_cast<uint64_t>(kvCacheHeadStrideValue);

    uint64_t headCount = 0U;
    uint64_t inputTokenStride = 0U;
    uint64_t qkvElements = 0U;
    uint64_t qkvBytes = 0U;
    uint64_t rawCosSinElements = 0U;
    uint64_t rawCosSinBytes = 0U;
    uint64_t rawCosSinTokenStride = 0U;
    uint64_t slotBytes = 0U;
    uint64_t qDataElements = 0U;
    uint64_t qScaleElements = 0U;
    uint64_t vScaleElements = 0U;
    uint64_t kScaleTokenBytes = 0U;
    uint64_t kScaleTokenStrideBytes = 0U;
    uint64_t kScaleStagingBytes = 0U;
    uint64_t kvCacheDstStrideBytes = 0U;
    // Recompute every downstream Copy/VF field in uint64_t and in its actual
    // unit before narrowing: QKV/raw-cos/slot/V-scale lengths and K-scale
    // staging are bytes; VF strides and Q/Q-scale lengths are element counts.
    const bool arithmeticOk =
        ::QkvRmsNormRopeCacheWithKScale::MropeMxCheckedAdd(input_.numQHeads, input_.numKHeads, headCount) &&
        ::QkvRmsNormRopeCacheWithKScale::MropeMxCheckedAdd(headCount, input_.numVHeads, headCount) &&
        CheckedMul(headCount, input_.headDim, inputTokenStride) &&
        CheckedMul(tokenTile, inputTokenStride, qkvElements) && CheckedMul(qkvElements, 2U, qkvBytes) &&
        CheckedMul(3U, input_.headDim, rawCosSinTokenStride) &&
        CheckedMul(tokenTile, rawCosSinTokenStride, rawCosSinElements) &&
        CheckedMul(rawCosSinElements, sizeof(float), rawCosSinBytes) &&
        CheckedMul(tokenTile, sizeof(int32_t), slotBytes) && CheckedMul(tokenTile, input_.numQHeads, qDataElements) &&
        CheckedMul(qDataElements, input_.headDim, qDataElements) &&
        CheckedMul(tokenTile, input_.numQHeads, qScaleElements) &&
        CheckedMul(qScaleElements, ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_SCALE_COUNT_D128, qScaleElements) &&
        CheckedMul(input_.numKHeads, ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_SCALE_COUNT_D128, kScaleTokenBytes) &&
        ::QkvRmsNormRopeCacheWithKScale::MropeMxCheckedAlign(kScaleTokenBytes, kScaleTokenStrideBytes) &&
        CheckedMul(tokenTile, kScaleTokenStrideBytes, kScaleStagingBytes) &&
        CheckedMul(input_.numVHeads, input_.headDim, vScaleElements) &&
        CheckedMul(vScaleElements, sizeof(float), vScaleElements) &&
        CheckedMul(kvCacheHeadStride - input_.headDim, ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_FP8_BYTES,
                   kvCacheDstStrideBytes);
    // These limits mirror the uint32_t DataCopy lengths, LocalTensor sizes,
    // and VF address-register strides used by the device implementation.
    if (!arithmeticOk || !FitsUint32(inputTokenStride) || !FitsUint32(input_.headDim) || !FitsUint32(qkvBytes) ||
        !FitsUint32(rawCosSinBytes) || !FitsUint32(slotBytes) || !FitsUint32(qDataElements) ||
        !FitsUint32(qScaleElements) || !FitsUint32(kScaleStagingBytes) ||
        !FitsUint32(kScaleTokenStrideBytes / sizeof(uint32_t)) || !FitsUint32(vScaleElements) ||
        kvCacheDstStrideBytes > MROPE_MX_GM_STRIDE_MAX_BYTES) {
        return false;
    }

    // Compact K-scale scatter writes four E8M0 bytes per head and represents
    // the remaining destination head gap with a uint32_t DataCopy stride.
    const uint64_t kScaleElements = ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_SCALE_COUNT_D128;
    const uint64_t kScaleHeadStride = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(1));
    if (kScaleHeadStride < kScaleElements || !FitsUint32(kScaleHeadStride - kScaleElements)) {
        return false;
    }
    return true;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::SelectMropeMxTokenTile(
    uint64_t &tokenTile, ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout &selectedLayout) const
{
    OP_CHECK_IF(compileInfo_ == nullptr, OP_LOGE(context_, "compileInfo is nullptr during MX tile search."),
                return ge::GRAPH_FAILED);
    // A tile cannot exceed the input, the resident M-RoPE position window, or
    // the uint16_t token count accepted by the VF entry points.
    const uint64_t upper =
        std::min({input_.totalTokens, ::QkvRmsNormRopeCacheWithKScale::MROPE_MX_POSITION_WINDOW_TOKENS,
                  static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())});
    // Search downward so the first candidate passing all independent gates is
    // the largest feasible tile for this shape and UB size.
    for (uint64_t candidate = upper; candidate > 0U; --candidate) {
        ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout layout{};
        const bool layoutOk = ::QkvRmsNormRopeCacheWithKScale::TryMakeMropeMxQGlobalTileWaveUbLayout(
            candidate, input_.numQHeads, input_.numKHeads, input_.headDim, layout);
        // Layout construction validates aligned offsets; the remaining gates
        // validate physical UB capacity and downstream Copy/VF field widths.
        if (!layoutOk || static_cast<uint64_t>(layout.totalBytes) > compileInfo_->ubSize ||
            !AreMropeMxCopyAndVfFieldsRepresentable(candidate, layout)) {
            continue;
        }
        tokenTile = candidate;
        selectedLayout = layout;
        OP_LOGD(context_,
                "QkvRmsNormRopeCacheWithKScale M-RoPE MX tile selected: upper=%llu tokenTile=%llu ubSize=%llu "
                "layoutTotalBytes=%u.",
                upper, tokenTile, compileInfo_->ubSize, selectedLayout.totalBytes);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
        opName_, "totalTokens, ubSize",
        (std::to_string(input_.totalTokens) + ", " + std::to_string(compileInfo_->ubSize)).c_str(),
        "NO_UB_TILE: no M-RoPE MX token tile, including tokenTile=1, fits UB and copy/VF fields");
    return ge::GRAPH_FAILED;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillTilingData(uint64_t tokenTile)
{
    tilingData_ = {};
    tilingData_.totalTokens = input_.totalTokens;
    tilingData_.batch = input_.batch;
    tilingData_.qHeadNum = input_.numQHeads;
    tilingData_.kvHeadNum = input_.numKHeads;
    tilingData_.headDim = input_.headDim;
    tilingData_.blockSize = input_.blockSize;
    tilingData_.coreTokenTile = Ops::Base::CeilDiv(input_.totalTokens, aicNum_);
    tilingData_.coreGroupNum = Ops::Base::CeilDiv(input_.totalTokens, tilingData_.coreTokenTile);
    tilingData_.kvCacheStrideBlock = static_cast<uint64_t>(input_.kCache.stride.GetStride(0));
    tilingData_.kvCacheStrideHead = static_cast<uint64_t>(input_.kCache.stride.GetStride(1));
    tilingData_.kvCacheStrideToken = static_cast<uint64_t>(input_.kCache.stride.GetStride(2));
    tilingData_.kScaleCacheStrideBlock = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(0));
    tilingData_.kScaleCacheStrideHead = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(1));
    tilingData_.kScaleCacheStrideToken = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(2));
    tilingData_.tokenTile = tokenTile;
    tilingData_.epsilon = epsilon_;
    if (GetSceneTraits(scene_).needsMropePosition) {
        // ValidateMropeSection() stores the validated H/W projection before tiling.
        tilingData_.mropeSectionH = mropeSectionH_;
        tilingData_.mropeSectionW = mropeSectionW_;
    }
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillMropeMxTilingData(uint64_t tokenTile)
{
    tilingData_ = {};
    tilingData_.totalTokens = input_.totalTokens;
    tilingData_.qHeadNum = input_.numQHeads;
    tilingData_.kvHeadNum = input_.numKHeads;
    tilingData_.headDim = input_.headDim;
    tilingData_.blockSize = input_.blockSize;
    tilingData_.kvCacheStrideBlock = static_cast<uint64_t>(input_.kCache.stride.GetStride(0));
    tilingData_.kvCacheStrideHead = static_cast<uint64_t>(input_.kCache.stride.GetStride(1));
    tilingData_.kvCacheStrideToken = static_cast<uint64_t>(input_.kCache.stride.GetStride(2));
    tilingData_.kScaleCacheStrideBlock = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(0));
    tilingData_.kScaleCacheStrideHead = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(1));
    tilingData_.kScaleCacheStrideToken = static_cast<uint64_t>(input_.kScaleCache.stride.GetStride(2));
    tilingData_.tokenTile = tokenTile;
    tilingData_.epsilon = epsilon_;
    tilingData_.mropeSectionH = mropeSectionH_;
    tilingData_.mropeSectionW = mropeSectionW_;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateParsedInput() const
{
    OP_LOGD(context_, "QkvRmsNormRopeCacheWithKScale parsed input validation start.");
    ge::graphStatus status = ValidateHeadNums();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale head nums validation failed."),
                return ge::GRAPH_FAILED);

    status = ValidateScalarInputs();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale scalar validation failed."), return ge::GRAPH_FAILED);

    status = ValidateDtypes();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale dtype validation failed."), return ge::GRAPH_FAILED);
    status = ValidateQOutDtypeAttr();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale q_out_dtype validation failed."),
                return ge::GRAPH_FAILED);
    status = ValidateShapes();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale shape validation failed."), return ge::GRAPH_FAILED);
    status = ValidateStrides();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale stride validation failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ComputeTilingData()
{
    if (scene_ == Scene::MROPE_MX) {
        uint64_t tokenTile = 0U;
        ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout selectedLayout{};
        OP_CHECK_IF(SelectMropeMxTokenTile(tokenTile, selectedLayout) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "M-RoPE MX token tile selection failed."), return ge::GRAPH_FAILED);

        const uint64_t activeAiv = std::min(input_.totalTokens, aivNum_);
        OP_CHECK_IF(!FitsUint32(activeAiv),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "activeAiv", std::to_string(activeAiv).c_str(),
                                                          "activeAiv must fit uint32"),
                    return ge::GRAPH_FAILED);
        const uint32_t activeAiv32 = static_cast<uint32_t>(activeAiv);
        FillMropeMxTilingData(tokenTile);
        numBlocks_ = CalcTschBlockDim(activeAiv32, 0U, static_cast<uint32_t>(aivNum_));
        OP_CHECK_IF(numBlocks_ != activeAiv32,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "blockDim, activeAiv",
                        (std::to_string(numBlocks_) + ", " + std::to_string(activeAiv32)).c_str(),
                        "CalcTschBlockDim(activeAiv, 0, aivNum) must preserve activeAiv for AIV-only launch"),
                    return ge::GRAPH_FAILED);
        OP_LOGD(context_,
                "QkvRmsNormRopeCacheWithKScale M-RoPE MX tiling: T=%llu aivNum=%llu activeAiv=%u "
                "blockDim=%u tokenTile=%llu ubSize=%llu layoutTotalBytes=%u unusedFields={%llu,%llu,%llu}.",
                input_.totalTokens, aivNum_, activeAiv32, numBlocks_, tokenTile, compileInfo_->ubSize,
                selectedLayout.totalBytes, tilingData_.batch, tilingData_.coreTokenTile, tilingData_.coreGroupNum);
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t numQkHeads = input_.numQHeads + input_.numKHeads;
    const uint64_t numQkvHeads = numQkHeads + input_.numVHeads;
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale contract input: layout_qkv=%s layout_q_out=%s T=%llu batch=%llu "
            "Nq=%llu Nk=%llu Nv=%llu Nqk=%llu Nqkv=%llu headDim=%llu maxSeqLen=%llu blockNum=%llu "
            "blockSize=%llu aicNum=%llu.",
            LayoutName(input_.layoutQkv), LayoutName(input_.layoutQOut), input_.totalTokens, input_.batch,
            input_.numQHeads, input_.numKHeads, input_.numVHeads, numQkHeads, numQkvHeads, input_.headDim,
            input_.maxSeqLen, input_.blockNum, input_.blockSize, aicNum_);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale resource constants: maxTokenTile=%llu qkvInputRowsPerAiv=%llu "
            "qkOutputRowsPerAiv=%llu vOutputRowsPerAiv=%llu l0aFullDimSegmentRows=%llu l0cMaxRows=%llu "
            "aivPerAic=%llu.",
            MAX_TOKEN_TILE, QKV_INPUT_ROWS_PER_AIV, QK_OUTPUT_ROWS_PER_AIV, V_OUTPUT_ROWS_PER_AIV,
            L0A_FULL_DIM_SEGMENT_ROWS, L0C_MAX_ROWS, AIV_PER_AIC);
    const uint64_t tokenTile = SelectTokenTile();
    FillTilingData(tokenTile);
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tokenTile, AIV_PER_AIC);
    const uint64_t rowTile = tokenTile * numQkHeads;
    const uint64_t rowTileAligned = Ops::Base::CeilAlign(rowTile, 16UL);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale contract tiling result: coreTokenTile=%llu coreGroupNum=%llu "
            "tokenTile=%llu tokenTilePerAiv=%llu Nq=%llu Nk=%llu Nv=%llu Nqk=%llu Nqkv=%llu "
            "rowTile=%llu rowTileAligned=%llu dimTile=%llu.",
            tilingData_.coreTokenTile, tilingData_.coreGroupNum, tilingData_.tokenTile, tokenTilePerAiv,
            input_.numQHeads, input_.numKHeads, input_.numVHeads, numQkHeads, numQkvHeads, rowTile, rowTileAligned,
            DIM_TILE);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale contract stride result: kvCache={%llu,%llu,%llu} "
            "kScaleCache={%llu,%llu,%llu}.",
            tilingData_.kvCacheStrideBlock, tilingData_.kvCacheStrideHead, tilingData_.kvCacheStrideToken,
            tilingData_.kScaleCacheStrideBlock, tilingData_.kScaleCacheStrideHead, tilingData_.kScaleCacheStrideToken);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MakeContiguousStrideInfo(const char *opName, const gert::Shape &shape, gert::Stride &strideInfo)
{
    gert::Stride stride;
    stride.SetDimNum(shape.GetDimNum());
    int64_t currentStride = 1;
    for (int32_t dim = static_cast<int32_t>(shape.GetDimNum()) - 1; dim >= 0; --dim) {
        const int64_t dimValue = shape.GetDim(dim);
        OP_CHECK_IF(
            dimValue <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "cache", Ops::Base::ToString(shape).c_str(),
                                                  ("dim " + std::to_string(dim) + " must be greater than 0").c_str()),
            return ge::GRAPH_FAILED);
        stride.SetStride(dim, currentStride);
        OP_CHECK_IF(dimValue > 0 && currentStride > std::numeric_limits<int64_t>::max() / dimValue,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, "cache", Ops::Base::ToString(shape).c_str(),
                        "contiguous stride calculation must be less than or equal to int64 max"),
                    return ge::GRAPH_FAILED);
        currentStride *= dimValue;
    }
    strideInfo = stride;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ToTensorInfo(gert::TilingContext *context, const char *opName, const char *tensorName, uint64_t index,
                             TensorContractInfo &info)
{
    const auto desc = context->GetInputDesc(index);
    if (desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, tensorName, "nullptr", "input description must exist");
        return ge::GRAPH_FAILED;
    }
    const auto shape = context->GetInputShape(index);
    if (shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, tensorName, "nullptr", "input shape must exist");
        return ge::GRAPH_FAILED;
    }
    info.descriptorPresent = true;
    info.dtype = desc->GetDataType();
    info.shape = shape->GetStorageShape();
    info.shapePresent = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillViewCacheTensorInfo(gert::TilingContext *context, const char *opName, uint64_t index,
                                        ge::DataType dtype, const gert::Shape &logicalShapeInfo,
                                        TensorContractInfo &info)
{
    const auto inputStride = context->GetInputStride(index);
    const uint64_t inputStrideRank = inputStride == nullptr ? 0U : static_cast<uint64_t>(inputStride->GetDimNum());
    OP_LOGD(context,
            "QkvRmsNormRopeCacheWithKScale cache input index=%llu view stride query: stridePresent=%u "
            "strideRank=%llu.",
            index, inputStride == nullptr ? 0U : 1U, inputStrideRank);
    if (inputStride == nullptr || inputStride->GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_STRIDE(opName, CacheTensorName(index), "missing or empty", "4D stride");
        return ge::GRAPH_FAILED;
    }

    info.shape = logicalShapeInfo;
    info.shapePresent = true;
    info.stride = *inputStride;
    info.stridePresent = true;
    info.dtype = dtype;
    OP_LOGD(context,
            "QkvRmsNormRopeCacheWithKScale cache input index=%llu uses view shape/stride: "
            "shapeRank=%llu strideRank=%llu stride={%lld,%lld,%lld,%lld}.",
            index, static_cast<uint64_t>(info.shape.GetDimNum()), static_cast<uint64_t>(info.stride.GetDimNum()),
            info.stride.GetStride(0), info.stride.GetStride(1), info.stride.GetStride(2), info.stride.GetStride(3));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillContiguousCacheTensorInfo(gert::TilingContext *context, const char *opName, uint64_t index,
                                              ge::DataType dtype, const gert::Shape &storageShapeInfo,
                                              TensorContractInfo &info)
{
    info.shape = storageShapeInfo;
    info.shapePresent = true;
    OP_CHECK_IF(MakeContiguousStrideInfo(opName, storageShapeInfo, info.stride) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "make contiguous cache stride failed, input index=%llu.",
                        static_cast<unsigned long long>(index)),
                return ge::GRAPH_FAILED);
    info.stridePresent = true;
    info.dtype = dtype;
    OP_LOGD(context,
            "QkvRmsNormRopeCacheWithKScale cache input index=%llu uses contiguous storage shape/stride: "
            "shapeRank=%llu strideRank=%llu stride={%lld,%lld,%lld,%lld}.",
            index, static_cast<uint64_t>(info.shape.GetDimNum()), static_cast<uint64_t>(info.stride.GetDimNum()),
            info.stride.GetStride(0), info.stride.GetStride(1), info.stride.GetStride(2), info.stride.GetStride(3));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ToCacheTensorInfo(gert::TilingContext *context, const char *opName, uint64_t index,
                                  TensorContractInfo &info)
{
    const auto desc = context->GetInputDesc(index);
    if (desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CacheTensorName(index), "nullptr",
                                              "input description must exist");
        return ge::GRAPH_FAILED;
    }
    info.descriptorPresent = true;
    const auto storageShape = context->GetInputShape(index);
    if (storageShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, CacheTensorName(index), "nullptr",
                                              "input storage shape must exist");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &storageShapeInfo = storageShape->GetStorageShape();
    const gert::Shape &logicalShapeInfo = storageShape->GetShape();
    const bool inputIsView = context->InputIsView(index);
    OP_LOGD(context,
            "QkvRmsNormRopeCacheWithKScale cache input index=%llu InputIsView=%u storageRank=%llu "
            "logicalRank=%llu.",
            index, inputIsView ? 1U : 0U, static_cast<uint64_t>(storageShapeInfo.GetDimNum()),
            static_cast<uint64_t>(logicalShapeInfo.GetDimNum()));
    if (inputIsView) {
        return FillViewCacheTensorInfo(context, opName, index, desc->GetDataType(), logicalShapeInfo, info);
    }
    return FillContiguousCacheTensorInfo(context, opName, index, desc->GetDataType(), storageShapeInfo, info);
}

TensorContractInfo ToOptionalTensorInfo(gert::TilingContext *context, uint64_t index)
{
    TensorContractInfo info;
    const auto desc = context->GetOptionalInputDesc(index);
    if (desc == nullptr) {
        return info;
    }
    info.descriptorPresent = true;
    info.dtype = desc->GetDataType();
    const auto shape = context->GetOptionalInputShape(index);
    if (shape == nullptr) {
        return info;
    }
    info.shape = shape->GetStorageShape();
    info.shapePresent = true;
    return info;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::LogTensorInfo(const char *tensorName,
                                                            const TensorContractInfo &info) const
{
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale tensor %s: dtype=%s(%u) descriptorPresent=%u shapePresent=%u shapeRank=%llu "
            "stridePresent=%u strideRank=%llu stride={%lld,%lld,%lld,%lld}.",
            tensorName, DtypeName(info.dtype), static_cast<uint32_t>(info.dtype), info.descriptorPresent ? 1U : 0U,
            info.shapePresent ? 1U : 0U, static_cast<uint64_t>(info.shape.GetDimNum()), info.stridePresent ? 1U : 0U,
            static_cast<uint64_t>(info.stride.GetDimNum()), GetStrideDimOrZero(info.stride, 0),
            GetStrideDimOrZero(info.stride, 1), GetStrideDimOrZero(info.stride, 2), GetStrideDimOrZero(info.stride, 3));
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::LogContractInput() const
{
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale parsed attrs/input summary: head_nums={%llu,%llu,%llu} "
            "layout_qkv=%s layout_q_out=%s epsilon=%f T=%llu batch=%llu headDim=%llu maxSeqLen=%llu blockNum=%llu "
            "blockSize=%llu aicNum=%llu.",
            input_.numQHeads, input_.numKHeads, input_.numVHeads, LayoutName(input_.layoutQkv),
            LayoutName(input_.layoutQOut), epsilon_, input_.totalTokens, input_.batch, input_.headDim, input_.maxSeqLen,
            input_.blockNum, input_.blockSize, aicNum_);
    LogTensorInfo("qkv", input_.qkv);
    LogTensorInfo("qGamma", input_.qGamma);
    LogTensorInfo("kGamma", input_.kGamma);
    LogTensorInfo("cosSin", input_.cosSin);
    LogTensorInfo("slotMapping", input_.slotMapping);
    LogTensorInfo("kCache", input_.kCache);
    LogTensorInfo("vCache", input_.vCache);
    LogTensorInfo("kScaleCache", input_.kScaleCache);
    LogTensorInfo("queryStartLoc", input_.queryStartLoc);
    LogTensorInfo("seqLens", input_.seqLens);
    LogTensorInfo("rotation", input_.rotation);
    LogTensorInfo("vScale", input_.vScale);
    LogTensorInfo("mropePosition", input_.mropePosition);
}

ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *opName,
                             const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo, uint64_t &workspaceSize)
{
    auto *workspaces = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context, "get workspace failed."), return ge::GRAPH_FAILED);
    const uint64_t opWorkspaceSize = compileInfo.opWorkspaceSize;
    OP_CHECK_IF(
        opWorkspaceSize > std::numeric_limits<uint64_t>::max() - RESERVED_WORKSPACE_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "opWorkspaceSize", std::to_string(opWorkspaceSize).c_str(),
                                              "workspace plus 16 MiB reserve must fit uint64"),
        return ge::GRAPH_FAILED);
    workspaceSize = opWorkspaceSize + RESERVED_WORKSPACE_SIZE;
    workspaces[0] = workspaceSize;
    OP_LOGD(context,
            "QkvRmsNormRopeCacheWithKScale workspace: opWorkspaceSize=%llu reservedWorkspaceSize=%llu total=%llu.",
            opWorkspaceSize, RESERVED_WORKSPACE_SIZE, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::LogTilingData() const
{
    const uint64_t numQkHeads = input_.numQHeads + input_.numKHeads;
    const uint64_t rowTile = tilingData_.tokenTile * numQkHeads;
    const uint64_t rowTileAligned = Ops::Base::CeilAlign(rowTile, 16UL);
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tilingData_.tokenTile, AIV_PER_AIC);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale final tilingData: totalTokens=%llu batch=%llu qHeadNum=%llu "
            "kvHeadNum=%llu headDim=%llu blockSize=%llu coreTokenTile=%llu coreGroupNum=%llu tokenTile=%llu "
            "epsilon=%f.",
            tilingData_.totalTokens, tilingData_.batch, tilingData_.qHeadNum, tilingData_.kvHeadNum,
            tilingData_.headDim, tilingData_.blockSize, tilingData_.coreTokenTile, tilingData_.coreGroupNum,
            tilingData_.tokenTile, tilingData_.epsilon);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale final strides: kvCache={%llu,%llu,%llu} "
            "kScaleCache={%llu,%llu,%llu}.",
            tilingData_.kvCacheStrideBlock, tilingData_.kvCacheStrideHead, tilingData_.kvCacheStrideToken,
            tilingData_.kScaleCacheStrideBlock, tilingData_.kScaleCacheStrideHead, tilingData_.kScaleCacheStrideToken);
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale final launch: tilingKey=%llu blockDim=%u tilingDataSize=%llu "
            "workspaceSize=%llu layout_qkv=%s Nqk=%llu Nv=%llu rowTile=%llu rowTileAligned=%llu "
            "tokenTilePerAiv=%llu dimTile=%llu layout_q_out=%s.",
            context_->GetTilingKey(), numBlocks_, tilingDataSize_, workspaceSize_, LayoutName(input_.layoutQkv),
            numQkHeads, input_.numVHeads, rowTile, rowTileAligned, tokenTilePerAiv, DIM_TILE,
            LayoutName(input_.layoutQOut));
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseHeadNumsAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_, "attrs is nullptr."), return ge::GRAPH_FAILED);
    const auto headNums = attrs->GetListInt(HEAD_NUMS_ATTR_INDEX);
    OP_CHECK_IF(headNums == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "head_nums", "missing", "head_nums must exist"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headNums->GetSize() != 3,
                OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "head_nums", std::to_string(headNums->GetSize()).c_str(), "3"),
                return ge::GRAPH_FAILED);
    const auto *headNumsData = headNums->GetData();
    OP_CHECK_IF(headNumsData == nullptr, OP_LOGE(context_, "head_nums data is nullptr."), return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(headNumsData[i] <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "head_nums", std::to_string(headNumsData[i]).c_str(),
                        ("head_nums[" + std::to_string(i) + "] must be greater than 0").c_str()),
                    return ge::GRAPH_FAILED);
    }

    input_.numQHeads = static_cast<uint64_t>(headNumsData[0]);
    input_.numKHeads = static_cast<uint64_t>(headNumsData[1]);
    input_.numVHeads = static_cast<uint64_t>(headNumsData[2]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseLayoutQkvAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_, "attrs is nullptr."), return ge::GRAPH_FAILED);
    if (static_cast<uint64_t>(attrs->GetAttrNum()) <= QKV_LAYOUT_ATTR_INDEX) {
        input_.layoutQkv = QKV_K_SCALE_LAYOUT_TND;
        return ge::GRAPH_SUCCESS;
    }
    const char *layoutQkv = attrs->GetStr(QKV_LAYOUT_ATTR_INDEX);
    if (layoutQkv == nullptr || layoutQkv[0] == '\0') {
        input_.layoutQkv = QKV_K_SCALE_LAYOUT_TND;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(layoutQkv, QKV_LAYOUT_NTD) == 0) {
        input_.layoutQkv = QKV_K_SCALE_LAYOUT_NTD;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(layoutQkv, QKV_LAYOUT_TND) == 0) {
        input_.layoutQkv = QKV_K_SCALE_LAYOUT_TND;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_qkv", layoutQkv, "NTD or TND");
    return ge::GRAPH_FAILED;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseLayoutQOutAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_, "attrs is nullptr."), return ge::GRAPH_FAILED);
    if (static_cast<uint64_t>(attrs->GetAttrNum()) <= Q_OUT_LAYOUT_ATTR_INDEX) {
        input_.layoutQOut = QKV_K_SCALE_LAYOUT_NTD;
        return ge::GRAPH_SUCCESS;
    }
    const char *layoutQOut = attrs->GetStr(Q_OUT_LAYOUT_ATTR_INDEX);
    if (layoutQOut == nullptr || layoutQOut[0] == '\0') {
        input_.layoutQOut = QKV_K_SCALE_LAYOUT_NTD;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(layoutQOut, QKV_LAYOUT_NTD) == 0) {
        input_.layoutQOut = QKV_K_SCALE_LAYOUT_NTD;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(layoutQOut, QKV_LAYOUT_TND) == 0) {
        input_.layoutQOut = QKV_K_SCALE_LAYOUT_TND;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_q_out", layoutQOut, "NTD or TND");
    return ge::GRAPH_FAILED;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseQQuantModeAttr()
{
    const auto attrs = context_->GetAttrs();
    if (attrs == nullptr || static_cast<uint64_t>(attrs->GetAttrNum()) <= Q_QUANT_MODE_ATTR_INDEX) {
        qQuantMode_ = QQuantMode::PER_TOKEN_PER_HEAD;
        return ge::GRAPH_SUCCESS;
    }
    const char *qQuantMode = attrs->GetStr(Q_QUANT_MODE_ATTR_INDEX);
    if (qQuantMode == nullptr || qQuantMode[0] == '\0' ||
        std::strcmp(qQuantMode, Q_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0) {
        qQuantMode_ = QQuantMode::PER_TOKEN_PER_HEAD;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(qQuantMode, Q_QUANT_MODE_NO_QUANT) == 0) {
        qQuantMode_ = QQuantMode::NO_QUANT;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(qQuantMode, Q_QUANT_MODE_MX) == 0) {
        qQuantMode_ = QQuantMode::MX_QUANT;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE(opName_, "q_quant_mode", qQuantMode, "PerTokenPerHead, NoQuant, or Mx");
    return ge::GRAPH_FAILED;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseKQuantModeAttr()
{
    const auto attrs = context_->GetAttrs();
    if (attrs == nullptr || static_cast<uint64_t>(attrs->GetAttrNum()) <= K_QUANT_MODE_ATTR_INDEX) {
        kQuantMode_ = KQuantMode::PER_TOKEN_PER_HEAD;
        return ge::GRAPH_SUCCESS;
    }
    const char *kQuantMode = attrs->GetStr(K_QUANT_MODE_ATTR_INDEX);
    if (kQuantMode == nullptr || kQuantMode[0] == '\0' ||
        std::strcmp(kQuantMode, K_QUANT_MODE_PER_TOKEN_PER_HEAD) == 0) {
        kQuantMode_ = KQuantMode::PER_TOKEN_PER_HEAD;
        return ge::GRAPH_SUCCESS;
    }
    if (std::strcmp(kQuantMode, K_QUANT_MODE_MX) == 0) {
        kQuantMode_ = KQuantMode::MX_QUANT;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE(opName_, "k_quant_mode", kQuantMode, "PerTokenPerHead or Mx");
    return ge::GRAPH_FAILED;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseQOutDtypeAttr()
{
    const auto attrs = context_->GetAttrs();
    if (attrs == nullptr || static_cast<uint64_t>(attrs->GetAttrNum()) <= Q_OUT_DTYPE_ATTR_INDEX) {
        qOutDtypeAttr_ = ge::DT_FLOAT8_E4M3FN;
        return ge::GRAPH_SUCCESS;
    }
    const int64_t *qOutDtype = attrs->GetInt(Q_OUT_DTYPE_ATTR_INDEX);
    if (qOutDtype == nullptr) {
        qOutDtypeAttr_ = ge::DT_FLOAT8_E4M3FN;
        return ge::GRAPH_SUCCESS;
    }
    qOutDtypeAttr_ = static_cast<ge::DataType>(*qOutDtype);
    OP_CHECK_IF(qOutDtypeAttr_ != ge::DT_FLOAT8_E4M3FN && qOutDtypeAttr_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "q_out_dtype", std::to_string(*qOutDtype).c_str(),
                                                      "q_out_dtype must be ge::DT_FLOAT8_E4M3FN or ge::DT_BF16"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::CacheMropeSectionAttr()
{
    mropeSectionSize_ = 0;
    hasMropeSection_ = false;
    mropeSectionData_ = nullptr;
    mropeSectionH_ = 0;
    mropeSectionW_ = 0;

    const auto attrs = context_->GetAttrs();
    if (attrs == nullptr || static_cast<uint64_t>(attrs->GetAttrNum()) <= MROPE_SECTION_ATTR_INDEX) {
        return;
    }
    const auto mropeSection = attrs->GetListInt(MROPE_SECTION_ATTR_INDEX);
    if (mropeSection == nullptr || mropeSection->GetSize() == 0) {
        return;
    }

    hasMropeSection_ = true;
    mropeSectionSize_ = static_cast<uint64_t>(mropeSection->GetSize());
    mropeSectionData_ = mropeSection->GetData();
}

float QkvRmsNormRopeCacheWithKScaleBaseTiling::ParseEpsilonAttr() const
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr || static_cast<uint64_t>(attrs->GetAttrNum()) <= EPSILON_ATTR_INDEX) {
        return DEFAULT_EPSILON;
    }
    const float *epsilonAttr = attrs->GetFloat(EPSILON_ATTR_INDEX);
    return epsilonAttr == nullptr ? DEFAULT_EPSILON : *epsilonAttr;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::FillShapeDerivedFields()
{
    const auto &qkvShape = input_.qkv.shape;
    const auto &cosSinShape = input_.cosSin.shape;
    const auto &kCacheShape = input_.kCache.shape;
    const bool isNtd = input_.layoutQkv == QKV_K_SCALE_LAYOUT_NTD;
    const int64_t qkvHeadDim = qkvShape.GetDim(isNtd ? 0 : 1);
    const int64_t qkvTokenDim = qkvShape.GetDim(isNtd ? 1 : 0);
    OP_CHECK_IF(qkvHeadDim <= 0 || qkvTokenDim <= 0 || qkvShape.GetDim(2) <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "qkv", Ops::Base::ToString(qkvShape).c_str(),
                                                      "head, token and head_dim dimensions must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cosSinShape.GetDim(0) <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "cosSin", Ops::Base::ToString(cosSinShape).c_str(),
                                                      "dim 0 must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kCacheShape.GetDim(0) <= 0 || kCacheShape.GetDim(2) <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "kCache", Ops::Base::ToString(kCacheShape).c_str(),
                                                      "block_num and block_size dimensions must be greater than 0"),
                return ge::GRAPH_FAILED);
    input_.totalTokens = static_cast<uint64_t>(qkvTokenDim);
    input_.headDim = static_cast<uint64_t>(qkvShape.GetDim(2));
    input_.batch =
        GetSceneTraits(scene_).needsMropePosition ? 0 : static_cast<uint64_t>(input_.queryStartLoc.shape.GetDim(0)) - 1;
    input_.maxSeqLen = static_cast<uint64_t>(cosSinShape.GetDim(0));
    input_.blockNum = static_cast<uint64_t>(kCacheShape.GetDim(0));
    input_.blockSize = static_cast<uint64_t>(kCacheShape.GetDim(2));
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale derived shape fields: totalTokens=%llu batch=%llu headDim=%llu "
            "maxSeqLen=%llu blockNum=%llu blockSize=%llu.",
            input_.totalTokens, input_.batch, input_.headDim, input_.maxSeqLen, input_.blockNum, input_.blockSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::FillRequiredTensorInputs()
{
    OP_CHECK_IF(ToTensorInfo(context_, opName_, "qkv", QKV_INDEX, input_.qkv) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse qkv tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context_, opName_, "qGamma", Q_GAMMA_INDEX, input_.qGamma) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse qGamma tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context_, opName_, "kGamma", K_GAMMA_INDEX, input_.kGamma) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse kGamma tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToTensorInfo(context_, opName_, "cosSin", COS_SIN_INDEX, input_.cosSin) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse cosSin tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ToTensorInfo(context_, opName_, "slotMapping", SLOT_MAPPING_INDEX, input_.slotMapping) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "failed to parse slotMapping tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context_, opName_, K_CACHE_INDEX, input_.kCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse kCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context_, opName_, V_CACHE_INDEX, input_.vCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse vCache tensor info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ToCacheTensorInfo(context_, opName_, K_SCALE_CACHE_INDEX, input_.kScaleCache) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse kScaleCache tensor info."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::FillOptionalTensorInputs()
{
    input_.queryStartLoc = ToOptionalTensorInfo(context_, QUERY_START_LOC_INDEX);
    input_.seqLens = ToOptionalTensorInfo(context_, SEQ_LENS_INDEX);
    input_.rotation = ToOptionalTensorInfo(context_, ROTATION_INDEX);
    input_.vScale = ToOptionalTensorInfo(context_, V_SCALE_INDEX);
    input_.mropePosition = ToOptionalTensorInfo(context_, MROPE_POSITION_INDEX);
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::BuildContractInput()
{
    CacheMropeSectionAttr();
    FillOptionalTensorInputs();
    OP_CHECK_IF(FillRequiredTensorInputs() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to parse required tensor inputs."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseQQuantModeAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to parse q_quant_mode attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseKQuantModeAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to parse k_quant_mode attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseHeadNumsAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to parse head_nums attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseLayoutQkvAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to parse layout_qkv attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseLayoutQOutAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to parse layout_q_out attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseQOutDtypeAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to parse q_out_dtype attr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ResolveScene() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "failed to resolve operator scene."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateShapeInputsForDerivedFields() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "invalid shape inputs for derived fields."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillShapeDerivedFields() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to fill shape derived fields."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateMropeSection() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale M-RoPE section validation failed."),
                return ge::GRAPH_FAILED);
    epsilon_ = ParseEpsilonAttr();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ValidateCompileInfo(
    const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo) const
{
    OP_LOGD(context_,
            "QkvRmsNormRopeCacheWithKScale compile info: aicNum=%u aivNum=%u ubSize=%llu l1Size=%llu "
            "l0cSize=%llu opWorkspaceSize=%llu.",
            compileInfo.aicNum, compileInfo.aivNum, compileInfo.ubSize, compileInfo.l1Size, compileInfo.l0cSize,
            compileInfo.opWorkspaceSize);
    if (GetSceneTraits(scene_).usesAivOnlyKernel) {
        OP_CHECK_IF(compileInfo.aivNum == 0 || compileInfo.ubSize == 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "aivNum, ubSize",
                        (std::to_string(compileInfo.aivNum) + ", " + std::to_string(compileInfo.ubSize)).c_str(),
                        "M-RoPE MX requires aivNum and ubSize greater than 0"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        compileInfo.aicNum == 0 || compileInfo.ubSize == 0 || compileInfo.l1Size == 0 || compileInfo.l0cSize == 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "aicNum, ubSize, l1Size, l0cSize",
            (std::to_string(compileInfo.aicNum) + ", " + std::to_string(compileInfo.ubSize) + ", " +
             std::to_string(compileInfo.l1Size) + ", " + std::to_string(compileInfo.l0cSize))
                .c_str(),
            "RoPE and M-RoPE require AIC, UB, L1 and L0C resources greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(compileInfo.aivNum) != static_cast<uint64_t>(compileInfo.aicNum) * AIV_PER_AIC,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "aicNum, aivNum",
                    (std::to_string(compileInfo.aicNum) + ", " + std::to_string(compileInfo.aivNum)).c_str(),
                    ("aivNum in compile info must equal aicNum multiplied by " + std::to_string(AIV_PER_AIC)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyRawTilingData(gert::TilingContext *context, const char *opName,
                                  const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData, uint64_t &tilingDataSize)
{
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetData() == nullptr, OP_LOGE(context, "raw tiling buffer is nullptr."),
                return ge::GRAPH_FAILED);
    tilingDataSize = sizeof(QkvRmsNormRopeCacheWithKScaleTilingData);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "tilingDataSize", std::to_string(tilingDataSize).c_str(),
                                                      "tilingDataSize must be aligned to 8"),
                return ge::GRAPH_FAILED);
    const uint64_t tilingCapacity = static_cast<uint64_t>(rawTilingData->GetCapacity());
    OP_CHECK_IF(tilingDataSize > tilingCapacity,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName, "tilingDataSize, tilingCapacity",
                    (std::to_string(tilingDataSize) + ", " + std::to_string(tilingCapacity)).c_str(),
                    "tilingDataSize must be less than or equal to tilingCapacity"),
                return ge::GRAPH_FAILED);
    const auto ret = memcpy_s(rawTilingData->GetData(), tilingCapacity, &tilingData, tilingDataSize);
    OP_CHECK_IF(ret != EOK, OP_LOGE(context, "copy QkvRmsNormRopeCacheWithKScale tiling data failed, ret=%d.", ret),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::Reset(gert::TilingContext *context)
{
    TilingBaseClass::Reset(context);
    Reset();
    opName_ = context_ == nullptr || context_->GetNodeName() == nullptr ? DEFAULT_OP_NAME : context_->GetNodeName();
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::Reset()
{
    opName_ = DEFAULT_OP_NAME;
    compileInfo_ = nullptr;
    input_ = {};
    tilingData_ = {};
    aicNum_ = 0;
    aivNum_ = 0;
    ropeMode_ = RopeMode::ROPE;
    qQuantMode_ = QQuantMode::PER_TOKEN_PER_HEAD;
    kQuantMode_ = KQuantMode::PER_TOKEN_PER_HEAD;
    scene_ = Scene::ROPE;
    qOutDtypeAttr_ = ge::DT_FLOAT8_E4M3FN;
    epsilon_ = DEFAULT_EPSILON;
    tilingDataSize_ = 0;
    mropeSectionSize_ = 0;
    hasMropeSection_ = false;
    mropeSectionData_ = nullptr;
    mropeSectionH_ = 0;
    mropeSectionW_ = 0;
    numBlocks_ = 0;
    workspaceSize_ = 0;
    tilingKey_ = 0;
}

bool QkvRmsNormRopeCacheWithKScaleBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(BuildContractInput() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "failed to build QkvRmsNormRopeCacheWithKScale contract input."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateParsedInput() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "invalid QkvRmsNormRopeCacheWithKScale contract input."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::GetPlatformInfo()
{
    compileInfo_ = context_->GetCompileInfo<QkvRmsNormRopeCacheWithKScaleCompileInfo>();
    OP_CHECK_IF(compileInfo_ == nullptr, OP_LOGE(context_, "compileInfo is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateCompileInfo(*compileInfo_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "invalid QkvRmsNormRopeCacheWithKScale compile info."), return ge::GRAPH_FAILED);
    aicNum_ = compileInfo_->aicNum;
    aivNum_ = compileInfo_->aivNum;
    LogContractInput();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::DoOpTiling()
{
    OP_CHECK_IF(ComputeTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "QkvRmsNormRopeCacheWithKScale contract tiling build failed."),
                return ge::GRAPH_FAILED);
    if (!GetSceneTraits(scene_).usesAivOnlyKernel) {
        numBlocks_ = static_cast<uint32_t>(tilingData_.coreGroupNum);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t QkvRmsNormRopeCacheWithKScaleBaseTiling::GetTilingKey() const
{
    const bool kCacheIsInt8 = input_.kCache.dtype == ge::DT_INT8;
    // The template key uses its own 0/1 ABI values rather than ge::DataType values.
    const uint64_t kCacheDtypeKey = kCacheIsInt8 ? QKV_K_SCALE_CACHE_DTYPE_INT8 : QKV_K_SCALE_CACHE_DTYPE_FP8_E4M3FN;
    return GET_TPL_TILING_KEY(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128, input_.layoutQkv,
                              input_.layoutQOut, static_cast<uint64_t>(ropeMode_), kCacheDtypeKey,
                              static_cast<uint64_t>(qQuantMode_), static_cast<uint64_t>(kQuantMode_));
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::ResolveScene()
{
    const bool hasQueryStartLoc = input_.queryStartLoc.descriptorPresent;
    const bool hasSeqLens = input_.seqLens.descriptorPresent;
    const bool hasMropePosition = input_.mropePosition.descriptorPresent;
    const bool rotationPresent = input_.rotation.descriptorPresent;
    if (hasMropePosition != hasMropeSection_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "mrope_position/mrope_section", "presence mismatch",
                                               "both M-RoPE parameters must be provided together");
        return ge::GRAPH_FAILED;
    }
    if (hasMropePosition && (hasQueryStartLoc || hasSeqLens)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "query_start_loc/seq_lens", "not absent",
                                               "query_start_loc and seq_lens must be absent in the M-RoPE scene");
        return ge::GRAPH_FAILED;
    }
    ropeMode_ = hasMropePosition ? RopeMode::MROPE : RopeMode::ROPE;
    bool resolved = false;
    if (!hasMropePosition && qQuantMode_ == QQuantMode::PER_TOKEN_PER_HEAD &&
        kQuantMode_ == KQuantMode::PER_TOKEN_PER_HEAD) {
        scene_ = Scene::ROPE;
        resolved = true;
    } else if (hasMropePosition && qQuantMode_ == QQuantMode::NO_QUANT &&
               kQuantMode_ == KQuantMode::PER_TOKEN_PER_HEAD) {
        scene_ = Scene::MROPE;
        resolved = true;
    } else if (hasMropePosition && qQuantMode_ == QQuantMode::MX_QUANT && kQuantMode_ == KQuantMode::MX_QUANT) {
        scene_ = Scene::MROPE_MX;
        resolved = true;
    }
    if (resolved) {
        const auto &traits = GetSceneTraits(scene_);
        const bool layoutSupported = !traits.needsMropePosition || (input_.layoutQkv == QKV_K_SCALE_LAYOUT_TND &&
                                                                    input_.layoutQOut == QKV_K_SCALE_LAYOUT_TND);
        resolved = layoutSupported && rotationPresent == traits.needsRotation;
    }
    OP_CHECK_IF(!resolved,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "rope/q_quant_mode/k_quant_mode/layout/rotation",
                    (std::to_string(static_cast<uint64_t>(ropeMode_)) + ", " + QQuantModeName(qQuantMode_) + ", " +
                     KQuantModeName(kQuantMode_) + ", " + LayoutName(input_.layoutQkv) + "->" +
                     LayoutName(input_.layoutQOut) + ", " + (rotationPresent ? "present" : "absent"))
                        .c_str(),
                    "combination must match one of the three supported scenes"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::GetWorkspaceSize()
{
    OP_CHECK_IF(compileInfo_ == nullptr, OP_LOGE(context_, "compileInfo is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspace(context_, opName_, *compileInfo_, workspaceSize_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "set QkvRmsNormRopeCacheWithKScale workspace failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheWithKScaleBaseTiling::PostTiling()
{
    OP_CHECK_IF(CopyRawTilingData(context_, opName_, tilingData_, tilingDataSize_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "copy QkvRmsNormRopeCacheWithKScale raw tiling data failed."),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(numBlocks_);
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheWithKScaleBaseTiling::DumpTilingInfo()
{
    LogTilingData();
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QkvRmsNormRopeCacheWithKScale, QkvRmsNormRopeCacheWithKScaleBaseTiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), TILING_TEMPLATE_PRIORITY);

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling
