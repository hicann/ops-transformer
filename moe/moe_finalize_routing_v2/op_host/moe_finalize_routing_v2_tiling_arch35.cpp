/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_tiling_arch35.cpp
 * \brief
 */
#include "op_host/tiling_util.h"
#include "moe_finalize_routing_v2_tiling.h"
#include "../op_kernel/arch35/moe_finalize_routing_v2_tiling_struct.h"

namespace optiling {
static constexpr int64_t TWO = 2;
static constexpr int64_t EXPANDED_X_IDX = 0;
static constexpr int64_t EXPANDED_ROW_IDX_IDX = 1;
static constexpr int64_t X1_IDX = 2;
static constexpr int64_t X2_IDX = 3;
static constexpr int64_t BIAS_IDX = 4;
static constexpr int64_t SCALES_IDX = 5;
static constexpr int64_t EXPERTIDX_IDX = 6;
static constexpr int64_t X_IDX = 7;
static constexpr int64_t CONST_EXPERT_ALPHA1_IDX = 8;
static constexpr int64_t CONST_EXPERT_ALPHA2_IDX = 9;
static constexpr int64_t CONST_EXPERT_V_IDX = 10;
static constexpr int64_t BIAS_DIM_NUM = 2;
static constexpr int64_t SCALES_DIM_NUM = 2;
static constexpr int64_t DROPLESS_EXPANDED_X_DIM_NUM = 2;
static constexpr int64_t DROPPAD_EXPANDED_X_DIM_NUM = 3;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DROP_LESS_COL = 0;
static constexpr int64_t DROP_PAD_COL = 1;
static constexpr int64_t DROP_LESS_ROW = 2;
static constexpr int64_t DROP_PAD_ROW = 3;
static constexpr int64_t INPUT_BUFFER_NUM = 8;        // expaned_x bias x1 x2 x a1 a2 v
static constexpr int64_t CONST_EXPERT_BUFFER_NUM = 3; // a1 a2 v
static constexpr int64_t CONST_EXPERT_ATTR_LIST_SIZE = 2;
static constexpr int64_t OUTPUT_BUFFER_NUM = 1; // y
static constexpr uint64_t FULL_LOAD_H_BASE_TILING_KEY = 10000;
static constexpr uint64_t SPLIT_H_BASE_TILING_KEY = 20000;
static constexpr uint64_t FULL_LOAD_ROW_K_H_BASE_TILING_KEY = 30000;
static constexpr uint64_t FULL_LOAD_K_H_BASE_TILING_KEY = 40000;
static constexpr uint64_t DROPLESS_COL_TILING_LEY = 0;
static constexpr uint64_t DROPPAD_COL_TILING_KEY = 10;
static constexpr uint64_t DROPLESS_ROW_TILING_LEY = 20;
static constexpr uint64_t DROPPAD_ROW_TILING_KEY = 30;
static constexpr uint64_t FLOAT16_TILING_KEY = 1;
static constexpr uint64_t BFLOAT16_TILING_KEY = 2;
static constexpr size_t WORKSPACE_RESERVED = 16 * 1024 * 1024;

const static int64_t ATTR_DROP_PAD_MODE_IDX = 0LL;
const static int64_t ATTR_ZERO_EXPERT_RANGE_IDX = 1LL;
const static int64_t ATTR_COPY_EXPERT_RANGE_IDX = 2LL;
const static int64_t ATTR_CONSTANT_EXPERT_RANGE_IDX = 3LL;
const static int64_t ATTR_K_IDX = 4LL;

const static int64_t BATCH_COPY_EXPERT_NUM = 4;

class MoeFinalizeRoutingV2Regbase : public MoeFinalizeRoutingTilingV2 {
public:
    explicit MoeFinalizeRoutingV2Regbase(gert::TilingContext* context)
        : MoeFinalizeRoutingTilingV2(context)
    {}
    ~MoeFinalizeRoutingV2Regbase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        MoeFinalizeRoutingTilingV2::Reset(context);
    }

protected:
    ge::graphStatus DoGetPlatformInfo() override;
    ge::graphStatus DoGetShapeAttrsInfo() override;
    ge::graphStatus CalcOpTiling() override;
    ge::graphStatus CalcTilingKey() override;
    void DoPostTiling() override;
    void PrintTilingData() override;
    bool IsCapable() override;

    bool IsRowKHFullLoad();
    bool IsKHFullLoad();
    bool IsHFullLoad();
    ge::graphStatus CheckOptionalInputDtype(int64_t idx, const char* name, ge::DataType expectedDtype);
    ge::graphStatus CheckOptionalInputShape(int64_t idx, const char* name, const gert::Shape& expected,
                                            const char* reason);
    ge::graphStatus CheckShapeAndDtypeIsValid();
    ge::graphStatus CheckPartShapeAndDtypeIsValid();
    ge::graphStatus CheckConstExpertPartShapeAndDtypeIsValid(int64_t idx);
    ge::graphStatus DoGetZeroShapeAttrsInfo(int64_t idx, int64_t& start, int64_t& end);
    ge::graphStatus GetDropPadModeAndExpertRangeAttrs();
    ge::graphStatus GetRequiredInputInfo();
    ge::graphStatus GetOptionalInputFlags();
    ge::graphStatus FinalCheckShapeAndDtypeIsValid();
    ge::graphStatus GetRow(const gert::StorageShape* expandedRowIdxShape);
    ge::graphStatus CheckBiasShape(const gert::StorageShape* biasShape);
    ge::graphStatus GetECH(const gert::StorageShape* expandedXShape);
    ge::graphStatus GetK(const gert::StorageShape* scalesShape);
    int64_t RowsHSize(int64_t rowFactor, bool scalesInUb);
    int64_t RowsHSizeForKHFullLoad(int64_t rowFactor, bool scalesInUb);
    int64_t CalcRowFactor(int64_t ubSizeRemained, bool scalesInUb);
    int64_t CalcRowFactorForKHFullLoad(int64_t ubSizeRemained, bool scalesInUb);
    void SetFullLoadTilingData(int64_t rowOfFormerBlock, int64_t rowOfTailBlock, int64_t rowFactor);
    ge::graphStatus DoOpTilingRowKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus DoOpTilingKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus DoOpTilingHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus DoOpTilingSplitH(int64_t rowOfFormerBlock, int64_t rowOfTailBlock);
    ge::graphStatus CheckRange(int64_t start, int64_t end, const char* rangeName);
    ge::graphStatus CheckRangeOverlap(int64_t start1, int64_t end1, int64_t start2, int64_t end2, const char* name1,
                                      const char* name2);
    ge::graphStatus CheckExpertRangeValidity();

    bool hasX1_{false};
    bool hasX2_{false};
    bool hasScales_{false};
    bool hasBias_{false};
    bool hasX_{false}; // 有x输入，且constantEnd - constantStart >= 0
    bool hasConstantExpert_{false};
    bool rowKHFullLoad_{false};
    bool kHFullLoad_{false};
    bool hFullLoad_{false};
    uint32_t coreNum_{0};
    uint64_t blockSize_{0};
    uint64_t vlFp32_{0};
    uint64_t scaleDtypeKey{0};

    ge::DataType dtype{ge::DataType::DT_FLOAT};
    int64_t dtypeSize{0};
    int64_t scaleDtypeSize{0};
    int64_t dropPadMode{0};
    int64_t row{0};
    int64_t e{0};
    int64_t c{0};
    int64_t k{0};
    int64_t h{0};
    int64_t hAligned{0};
    int64_t dim0OfExpandedX{0};
    // 新增判断零专家、拷贝专家以及常量专家的范围
    int64_t zeroExpertStart{-1};
    int64_t zeroExpertEnd{-1};
    int64_t copyExpertStart{-1};
    int64_t copyExpertEnd{-1};
    int64_t constantExpertStart{-1};
    int64_t constantExpertEnd{-1};
    int64_t constExpertRangeNum{-1};
    MoeFinalizeRoutingV2RegbaseTilingData* tilingData{nullptr};
};

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoGetPlatformInfo()
{
    auto compileInfoPtr = reinterpret_cast<const MoeFinalizeRoutingCompileInfoV2*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "compile_info", "nullptr",
                                                      "compile info is null"),
                return ge::GRAPH_FAILED);

    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    OP_CHECK_IF(blockSize_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size",
                                                      std::to_string(blockSize_).c_str(), "Get blockSize failed."),
                return ge::GRAPH_FAILED);
    vlFp32_ = Ops::Base::GetVRegSize(context_) / sizeof(float);
    OP_CHECK_IF(vlFp32_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "VL32", std::to_string(vlFp32_).c_str(),
                                                      "Get VL32 failed."),
                return ge::GRAPH_FAILED);

    coreNum_ = static_cast<uint32_t>(compileInfoPtr->aivNum);
    OP_CHECK_IF(coreNum_ == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "core_num",
                                                      std::to_string(coreNum_).c_str(), "Get core num failed."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = compileInfoPtr->ubSize;
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ub_size",
                                                      std::to_string(ubSize).c_str(), "Get ubSize failed."),
                return ge::GRAPH_FAILED);
    ubSize_ = static_cast<int64_t>(ubSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetK(const gert::StorageShape* scalesShape)
{
    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);
    const int64_t* attrKPtr = attrsPtr->GetAttrPointer<int64_t>(ATTR_K_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrKPtr);
    k = *attrKPtr;
    if (!scalesShape) {
        hasScales_ = false;
        return ge::GRAPH_SUCCESS;
    }

    hasScales_ = true;
    OP_CHECK_IF(scalesShape->GetStorageShape().GetDimNum() != SCALES_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scales",
                                             std::to_string(scalesShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
                return ge::GRAPH_FAILED);
    int64_t scalesDim1 = scalesShape->GetStorageShape().GetDim(1);
    // k > 1 means user explicitly specified k, need to verify consistency with scales dim1
    if (*attrKPtr > 1) {
        OP_CHECK_IF(*attrKPtr != scalesDim1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "k", std::to_string(*attrKPtr).c_str(), "k must equal scales dim1"),
                    return ge::GRAPH_FAILED);
    }
    k = scalesDim1;
    OP_CHECK_IF(k <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "k", std::to_string(k).c_str(),
                                                      "k must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetECH(const gert::StorageShape* expandedXShape)
{
    if (dropPadMode == DROP_LESS_ROW || dropPadMode == DROP_LESS_COL) {
        OP_CHECK_IF(expandedXShape->GetStorageShape().GetDimNum() != DROPLESS_EXPANDED_X_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "expanded_x",
                        std::to_string(expandedXShape->GetStorageShape().GetDimNum()).c_str(),
                        "dim num of expanded_x should be 2 in dropless mode."),
                    return ge::GRAPH_FAILED);
        dim0OfExpandedX = expandedXShape->GetStorageShape().GetDim(0);
        h = expandedXShape->GetStorageShape().GetDim(1);
        OP_CHECK_IF(h <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "h", std::to_string(h).c_str(),
                                                          "h must be greater than 0."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        expandedXShape->GetStorageShape().GetDimNum() != DROPPAD_EXPANDED_X_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "expanded_x",
                                                 std::to_string(expandedXShape->GetStorageShape().GetDimNum()).c_str(),
                                                 "dim num of expanded_x should be 3 in drop pad mode."),
        return ge::GRAPH_FAILED);
    e = expandedXShape->GetStorageShape().GetDim(0);
    c = expandedXShape->GetStorageShape().GetDim(1);
    h = expandedXShape->GetStorageShape().GetDim(TWO);
    OP_CHECK_IF(e <= 0 || c <= 0 || h <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "e,c,h",
                    (std::to_string(e) + "," + std::to_string(c) + "," + std::to_string(h)).c_str(),
                    "e, c and h must be greater than 0."),
                return ge::GRAPH_FAILED);
    dim0OfExpandedX = e;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetRow(const gert::StorageShape* expandedRowIdxShape)
{
    OP_CHECK_IF(
        expandedRowIdxShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_row_idx",
                                     std::to_string(expandedRowIdxShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);
    row = expandedRowIdxShape->GetStorageShape().GetDim(0) / k;
    OP_CHECK_IF(row <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "row", std::to_string(row).c_str(),
                                                      "row must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckBiasShape(const gert::StorageShape* biasShape)
{
    OP_CHECK_IF(biasShape->GetStorageShape().GetDimNum() != BIAS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "bias",
                                             std::to_string(biasShape->GetStorageShape().GetDimNum()).c_str(), "2D"),
                return ge::GRAPH_FAILED);
    e = biasShape->GetStorageShape().GetDim(0);
    if (dropPadMode == DROP_PAD_ROW || dropPadMode == DROP_PAD_COL) {
        OP_CHECK_IF(dim0OfExpandedX != e,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "expanded_x and bias", "invalid",
                        "dim 0 of expanded_x should be equal to dim 0 of bias in droppad mode."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(e < k,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "e", std::to_string(e).c_str(),
                                                      "e must be greater than or equal to k."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckOptionalInputDtype(int64_t idx, const char* name,
                                                                     ge::DataType expectedDtype)
{
    auto desc = context_->GetOptionalInputDesc(idx);
    if (desc) {
        OP_CHECK_IF(
            desc->GetDataType() != expectedDtype,
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), name, Ops::Base::ToString(desc->GetDataType()).c_str(),
                                      Ops::Base::ToString(expectedDtype).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckOptionalInputShape(int64_t idx, const char* name,
                                                                     const gert::Shape& expected, const char* reason)
{
    auto shape = context_->GetOptionalInputShape(idx);
    if (shape) {
        OP_CHECK_IF(shape->GetStorageShape() != expected,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), name, "invalid", reason),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckShapeAndDtypeIsValid()
{
    gert::Shape rowIdxShape = {row * k};
    gert::Shape bsh = {row, h};

    auto expandedRowIdxDesc = context_->GetInputDesc(EXPANDED_ROW_IDX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxDesc);
    auto expandedRowIdxShape = context_->GetInputShape(EXPANDED_ROW_IDX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShape);
    OP_CHECK_IF(expandedRowIdxDesc->GetDataType() != ge::DataType::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expanded_row_idx",
                                          Ops::Base::ToString(expandedRowIdxDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expandedRowIdxShape->GetStorageShape() != rowIdxShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expanded_row_idx", "invalid",
                                                      "shape of expanded_row_idx must be (bs,k)."),
                return ge::GRAPH_FAILED);

    if (CheckOptionalInputDtype(X1_IDX, "x1", dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputShape(X1_IDX, "x1", bsh, "shape of x1 must be (bs,h).") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputDtype(X2_IDX, "x2", dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckPartShapeAndDtypeIsValid() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "shape_and_dtype", "GRAPH_FAILED",
                                                      "CheckPartShapeAndDtypeIsValid failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(FinalCheckShapeAndDtypeIsValid() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "shape_and_dtype", "GRAPH_FAILED",
                                                      "FinalCheckShapeAndDtypeIsValid failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckConstExpertPartShapeAndDtypeIsValid(int64_t idx)
{
    // const_expert_alpha等的shape信息要与const_expert_range_num一致
    constExpertRangeNum = constantExpertEnd - constantExpertStart;
    auto desc = context_->GetOptionalInputDesc(idx);
    if (desc) {
        OP_CHECK_IF(desc->GetDataType() != dtype,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "const_expert",
                                              Ops::Base::ToString(desc->GetDataType()).c_str(),
                                              Ops::Base::ToString(dtype).c_str()),
                    return ge::GRAPH_FAILED);
    }
    auto shape = context_->GetOptionalInputShape(idx);
    if (shape) {
        OP_CHECK_IF(shape->GetStorageShape().GetDim(1) != h,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "const_expert", "invalid",
                                                          "dim 1 of const expert should be h."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(shape->GetStorageShape().GetDim(0) != constExpertRangeNum,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "const_expert", "invalid",
                                                          "dim 0 of expert should be constExpertRangeNum."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckPartShapeAndDtypeIsValid()
{
    gert::Shape bsh = {row, h};

    if (CheckOptionalInputShape(X2_IDX, "x2", bsh, "shape of x2 must be (bs,h).") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputDtype(BIAS_IDX, "bias", dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto biasShape = context_->GetOptionalInputShape(BIAS_IDX);
    if (biasShape) {
        OP_CHECK_IF(
            CheckBiasShape(biasShape) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "bias", "GRAPH_FAILED", "failed to get e."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(biasShape->GetStorageShape().GetDim(1) != h,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "bias", "invalid",
                                                          "dim 1 of bias should be h."),
                    return ge::GRAPH_FAILED);
    }

    auto scalesDesc = context_->GetOptionalInputDesc(SCALES_IDX);
    scaleDtypeSize = dtypeSize;
    if (scalesDesc) {
        auto scaleDtype = scalesDesc->GetDataType();
        OP_CHECK_IF(scaleDtype != ge::DataType::DT_FLOAT && scaleDtype != ge::DataType::DT_FLOAT16 &&
                        scaleDtype != ge::DataType::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scales",
                                              Ops::Base::ToString(scaleDtype).c_str(), "FLOAT, FLOAT16 or BF16"),
                    return ge::GRAPH_FAILED);
        scaleDtypeSize = (scaleDtype == ge::DataType::DT_FLOAT) ? sizeof(float) : sizeof(short);
        if (scaleDtype == ge::DataType::DT_FLOAT16) {
            scaleDtypeKey = FLOAT16_TILING_KEY;
        } else if (scaleDtype == ge::DataType::DT_BF16) {
            scaleDtypeKey = BFLOAT16_TILING_KEY;
        }
    }

    if (CheckConstExpertPartShapeAndDtypeIsValid(CONST_EXPERT_ALPHA1_IDX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckConstExpertPartShapeAndDtypeIsValid(CONST_EXPERT_ALPHA2_IDX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckConstExpertPartShapeAndDtypeIsValid(CONST_EXPERT_V_IDX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::FinalCheckShapeAndDtypeIsValid()
{
    gert::Shape bsk = {row, k};
    gert::Shape bsh = {row, h};

    if (CheckOptionalInputShape(SCALES_IDX, "scales", bsk, "shape of scales must be (bs,k).") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputDtype(EXPERTIDX_IDX, "expert_idx", ge::DataType::DT_INT32) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputShape(EXPERTIDX_IDX, "expert_idx", bsk, "shape of expert_idx must be (bs,k).") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputDtype(X_IDX, "x", dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOptionalInputShape(X_IDX, "x", bsh, "shape of x must be (bs,h).") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    OP_CHECK_IF(
        yDesc->GetDataType() != dtype,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                  Ops::Base::ToString(dtype).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        yShape->GetStorageShape() != bsh,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y", "invalid", "shape of y must be (bs,h)."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoGetZeroShapeAttrsInfo(int64_t idx, int64_t& start, int64_t& end)
{
    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);
    const auto* attrPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(idx);
    if (attrPtr != nullptr && attrPtr->GetSize() == CONST_EXPERT_ATTR_LIST_SIZE) {
        const int64_t* attrList = static_cast<const int64_t*>(attrPtr->GetData());
        start = attrList[0];
        end = attrList[1];
        OP_LOGD(context_, "Extracted input attrs start = %ld, end = %ld.", start, end);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckRange(int64_t start, int64_t end, const char* rangeName)
{
    if (start == -1 && end == -1) {
        return ge::GRAPH_SUCCESS;
    }

    if (e == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (!(0 <= start && start < end && end <= e)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), (std::string(rangeName) + " range").c_str(),
            ("[" + std::to_string(start) + ", " + std::to_string(end) + "), E=" + std::to_string(e)).c_str(),
            "Range must satisfy 0 <= start < end <= E.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckRangeOverlap(int64_t start1, int64_t end1, int64_t start2,
                                                               int64_t end2, const char* name1, const char* name2)
{
    if (start1 == -1 || start2 == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (!(end1 <= start2 || end2 <= start1)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(),
                                               (std::string(name1) + " range and " + name2 + " range").c_str(),
                                               ("[" + std::to_string(start1) + ", " + std::to_string(end1) + ") and [" +
                                                std::to_string(start2) + ", " + std::to_string(end2) + ")")
                                                   .c_str(),
                                               "Ranges should not overlap.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CheckExpertRangeValidity()
{
    // 检查是否合法区间
    if (CheckRange(zeroExpertStart, zeroExpertEnd, "zero expert") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRange(copyExpertStart, copyExpertEnd, "copy expert") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRange(constantExpertStart, constantExpertEnd, "constant expert") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 检查区间是否重叠
    if (CheckRangeOverlap(zeroExpertStart, zeroExpertEnd, copyExpertStart, copyExpertEnd, "zero expert",
                          "copy expert") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRangeOverlap(zeroExpertStart, zeroExpertEnd, constantExpertStart, constantExpertEnd, "zero expert",
                          "constant expert") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckRangeOverlap(copyExpertStart, copyExpertEnd, constantExpertStart, constantExpertEnd, "copy expert",
                          "constant expert") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetDropPadModeAndExpertRangeAttrs()
{
    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);
    auto dropPadModePtr = attrsPtr->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dropPadModePtr);
    dropPadMode = *dropPadModePtr;
    OP_CHECK_IF(dropPadMode != DROP_LESS_ROW && dropPadMode != DROP_LESS_COL && dropPadMode != DROP_PAD_COL &&
                    dropPadMode != DROP_PAD_ROW,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "drop_pad_mode", std::to_string(dropPadMode).c_str(),
                                          "0, 1, 2 or 3"),
                return ge::GRAPH_FAILED);

    DoGetZeroShapeAttrsInfo(ATTR_ZERO_EXPERT_RANGE_IDX, zeroExpertStart, zeroExpertEnd);
    DoGetZeroShapeAttrsInfo(ATTR_COPY_EXPERT_RANGE_IDX, copyExpertStart, copyExpertEnd);
    DoGetZeroShapeAttrsInfo(ATTR_CONSTANT_EXPERT_RANGE_IDX, constantExpertStart, constantExpertEnd);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetRequiredInputInfo()
{
    auto expandedXDesc = context_->GetInputDesc(EXPANDED_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXDesc);
    dtype = expandedXDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DataType::DT_FLOAT && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expanded_x", Ops::Base::ToString(dtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);
    dtypeSize = (dtype == ge::DataType::DT_FLOAT) ? sizeof(float) : sizeof(short);
    auto expandedXShape = context_->GetInputShape(EXPANDED_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShape);
    OP_CHECK_IF(GetECH(expandedXShape) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "H", "GRAPH_FAILED", "failed to get H."),
                return ge::GRAPH_FAILED);

    auto scalesShape = context_->GetOptionalInputShape(SCALES_IDX);
    OP_CHECK_IF(GetK(scalesShape) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "K", "GRAPH_FAILED", "failed to get K."),
                return ge::GRAPH_FAILED);

    auto expandedRowIdxShape = context_->GetInputShape(EXPANDED_ROW_IDX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShape);
    OP_CHECK_IF(
        GetRow(expandedRowIdxShape) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "row", "GRAPH_FAILED", "failed to get row."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::GetOptionalInputFlags()
{
    auto x1Desc = context_->GetOptionalInputDesc(X1_IDX);
    hasX1_ = x1Desc != nullptr;
    auto x2Desc = context_->GetOptionalInputDesc(X2_IDX);
    hasX2_ = x2Desc != nullptr;
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_IDX);
    hasBias_ = biasDesc != nullptr;
    OP_CHECK_IF(hasX2_ && !hasX1_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", "nullptr",
                                                      "x1 must exist when x2 exists."),
                return ge::GRAPH_FAILED);

    auto xDesc = context_->GetOptionalInputDesc(X_IDX);
    hasX_ = xDesc != nullptr;
    auto constExpertAlpha1Desc = context_->GetOptionalInputDesc(CONST_EXPERT_ALPHA1_IDX);
    auto constExpertAlpha2Desc = context_->GetOptionalInputDesc(CONST_EXPERT_ALPHA2_IDX);
    auto constExpertVDesc = context_->GetOptionalInputDesc(CONST_EXPERT_V_IDX);
    hasConstantExpert_ =
        constExpertVDesc != nullptr && constExpertAlpha1Desc != nullptr && constExpertAlpha2Desc != nullptr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoGetShapeAttrsInfo()
{
    if (GetDropPadModeAndExpertRangeAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetRequiredInputInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetOptionalInputFlags() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeAndDtypeIsValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExpertRangeValidity() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t MoeFinalizeRoutingV2Regbase::RowsHSize(int64_t rowFactor, bool scalesInUb)
{
    return (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_)) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * dtypeSize), blockSize_) * blockSize_ +
           (hasConstantExpert_ ? 1 : 0) * CONST_EXPERT_BUFFER_NUM *
               Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_ +
           (scalesInUb && hasScales_ ? 1 : 0) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * scaleDtypeSize), blockSize_) * blockSize_ +
           /* y */ Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * sizeof(float)), blockSize_) * blockSize_;
}

int64_t MoeFinalizeRoutingV2Regbase::RowsHSizeForKHFullLoad(int64_t rowFactor, bool scalesInUb)
{
    return (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_)) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * dtypeSize), blockSize_) * blockSize_ +
           (scalesInUb && hasScales_ ? 1 : 0) *
               Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * scaleDtypeSize), blockSize_) * blockSize_ +
           (hasBias_ ? 1 : 0) * rowFactor * k * hAligned * dtypeSize +
           (hasConstantExpert_ ? 1 : 0) * constExpertRangeNum * hAligned * dtypeSize * CONST_EXPERT_BUFFER_NUM +
           rowFactor * k * hAligned * dtypeSize +
           Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * sizeof(int32_t)), blockSize_) * blockSize_ +
           Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * k * sizeof(int32_t)), blockSize_) * blockSize_ +
           Ops::Base::CeilDiv(static_cast<uint64_t>(rowFactor * h * sizeof(float)), blockSize_) * blockSize_;
}

int64_t MoeFinalizeRoutingV2Regbase::CalcRowFactorForKHFullLoad(int64_t ubSizeRemained, bool scalesInUb)
{
    int64_t rowFactor = 1;
    int64_t factor = 1;
    while (RowsHSizeForKHFullLoad(rowFactor, scalesInUb) <= ubSizeRemained) {
        factor *= TWO;
        rowFactor *= factor;
    }
    int64_t upper = rowFactor;
    int64_t lower = rowFactor / factor;
    while (upper - lower > 1) {
        int64_t mid = (lower + upper) / TWO;
        int64_t size = RowsHSizeForKHFullLoad(mid, scalesInUb);
        if (size <= ubSizeRemained) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    rowFactor = lower;
    return rowFactor;
}

int64_t MoeFinalizeRoutingV2Regbase::CalcRowFactor(int64_t ubSizeRemained, bool scalesInUb)
{
    int64_t rowFactor = 1;
    int64_t factor = 1;
    while (RowsHSize(rowFactor, scalesInUb) <= ubSizeRemained) {
        factor *= TWO;
        rowFactor *= factor;
    }
    int64_t upper = rowFactor;
    int64_t lower = rowFactor / factor;
    while (upper - lower > 1) {
        int64_t mid = (lower + upper) / TWO;
        int64_t size = RowsHSize(mid, scalesInUb);
        if (size <= ubSizeRemained) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    rowFactor = lower;
    return rowFactor;
}

void MoeFinalizeRoutingV2Regbase::SetFullLoadTilingData(int64_t rowOfFormerBlock, int64_t rowOfTailBlock,
                                                        int64_t rowFactor)
{
    int64_t rowLoopOfFormerBlock = Ops::Base::CeilDiv(rowOfFormerBlock, rowFactor);
    rowFactor = Ops::Base::CeilDiv(rowOfFormerBlock, rowLoopOfFormerBlock);
    int64_t rowLoopOfTailBlock = Ops::Base::CeilDiv(rowOfTailBlock, rowFactor);
    int64_t tailRowFactorOfFormerBlock = rowOfFormerBlock - (rowLoopOfFormerBlock - 1) * rowFactor;
    int64_t tailRowFactorOfTailBlock = rowOfTailBlock - (rowLoopOfTailBlock - 1) * rowFactor;
    tilingData->rowLoopOfFormerBlock = rowLoopOfFormerBlock;
    tilingData->rowLoopOfTailBlock = rowLoopOfTailBlock;
    tilingData->rowFactor = rowFactor;
    tilingData->constExpertRangeFactor = std::min(constExpertRangeNum, rowFactor);
    tilingData->tailRowFactorOfFormerBlock = tailRowFactorOfFormerBlock;
    tilingData->tailRowFactorOfTailBlock = tailRowFactorOfTailBlock;
    tilingData->hLoop = 1;
    tilingData->hFactor = h;
    tilingData->tailHFactor = h;
    tilingData->kLoop = k;
    tilingData->kFactor = 1;
    tilingData->tailKFactor = 1;
    tilingData->activeNum = dim0OfExpandedX;

    tilingData->zeroExpertStart = zeroExpertStart;
    tilingData->zeroExpertEnd = zeroExpertEnd;
    tilingData->copyExpertStart = copyExpertStart;
    tilingData->copyExpertEnd = copyExpertEnd;
    tilingData->constantExpertStart = constantExpertStart;
    tilingData->constantExpertEnd = constantExpertEnd;
    tilingData->constExpertRangeNum = constExpertRangeNum;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingRowKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    int64_t expandedXAlignedByte;
    if (dropPadMode == DROP_LESS_COL || dropPadMode == DROP_LESS_ROW) {
        expandedXAlignedByte =
            Ops::Base::CeilDiv(static_cast<uint64_t>(row * k * h * dtypeSize), blockSize_) * blockSize_;
    } else {
        expandedXAlignedByte =
            Ops::Base::CeilDiv(static_cast<uint64_t>(e * c * h * dtypeSize), blockSize_) * blockSize_;
    }
    int64_t eHAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(e * h * dtypeSize), blockSize_) * blockSize_;
    int64_t hasBiasvalue = hasBias_ ? eHAlignedByte : 0;
    int64_t ubSizeRemained = (ubSize_ - expandedXAlignedByte - hasBiasvalue) / DOUBLE_BUFFER;
    int64_t rowFactor = CalcRowFactor(ubSizeRemained, true);
    SetFullLoadTilingData(rowOfFormerBlock, rowOfTailBlock, rowFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingKHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    int64_t rowFactor = 1;
    if (k <= BATCH_COPY_EXPERT_NUM) {
        int64_t ubSizeRemained = ubSize_ / DOUBLE_BUFFER;
        rowFactor = CalcRowFactorForKHFullLoad(ubSizeRemained, true);
    } else {
        int64_t kHAlignedByte = k * hAligned * dtypeSize;
        int64_t hasBiasvalue = hasBias_ ? kHAlignedByte : 0;
        int64_t ubSizeRemained = ubSize_ / DOUBLE_BUFFER - kHAlignedByte - hasBiasvalue;
        rowFactor = CalcRowFactor(ubSizeRemained, true);
    }
    SetFullLoadTilingData(rowOfFormerBlock, rowOfTailBlock, rowFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingHFullLoad(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    int64_t expandedXAndBiasSize = (1 /* expanded_x */ + static_cast<int64_t>(hasBias_)) * hAligned * dtypeSize;
    int64_t kFactor = ubSize_ / DOUBLE_BUFFER / expandedXAndBiasSize;
    int64_t upper = kFactor;
    int64_t lower = 1;
    int64_t blockSize = static_cast<int64_t>(blockSize_);
    while (upper - lower > 1) {
        int64_t mid = (lower + upper) / TWO;
        int64_t useUbSize = mid * expandedXAndBiasSize + Ops::Base::CeilDiv(mid * dtypeSize, blockSize) * blockSize +
                            RowsHSize(1, false);
        if (useUbSize > ubSize_ / DOUBLE_BUFFER) {
            upper = mid;
        } else {
            lower = mid;
        }
    }
    kFactor = lower;
    int64_t kLoop = Ops::Base::CeilDiv(k, kFactor);
    kFactor = Ops::Base::CeilDiv(k, kLoop);
    int64_t tailKFactor = k - (kLoop - 1) * kFactor;

    int64_t ubSizeRemained = ubSize_ / DOUBLE_BUFFER - kFactor * expandedXAndBiasSize -
                             Ops::Base::CeilDiv(kFactor * dtypeSize, blockSize) * blockSize;
    int64_t rowFactor = CalcRowFactor(ubSizeRemained, false);
    SetFullLoadTilingData(rowOfFormerBlock, rowOfTailBlock, rowFactor);
    tilingData->kLoop = kLoop;
    tilingData->kFactor = kFactor;
    tilingData->tailKFactor = tailKFactor;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::DoOpTilingSplitH(int64_t rowOfFormerBlock, int64_t rowOfTailBlock)
{
    int64_t actualInputNum = INPUT_BUFFER_NUM - static_cast<int64_t>(!hasX1_) - static_cast<int64_t>(!hasX2_) -
                             static_cast<int64_t>(!hasBias_) - static_cast<int64_t>(!hasX_) -
                             static_cast<int64_t>(!hasConstantExpert_) * CONST_EXPERT_BUFFER_NUM;
    int64_t totalBufferNum = actualInputNum + OUTPUT_BUFFER_NUM + (dtype != ge::DataType::DT_FLOAT ? 1 : 0);
    int64_t hFactor = ubSize_ / DOUBLE_BUFFER / dtypeSize / totalBufferNum;
    int64_t hLoop = Ops::Base::CeilDiv(h, hFactor);
    hFactor = Ops::Base::CeilDiv(h, hLoop);
    int64_t tailHFactor = h - (hLoop - 1) * hFactor;
    tilingData->rowLoopOfFormerBlock = rowOfFormerBlock;
    tilingData->rowLoopOfTailBlock = rowOfTailBlock;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = rowOfFormerBlock;
    tilingData->tailRowFactorOfTailBlock = rowOfTailBlock;
    tilingData->hLoop = hLoop;
    tilingData->hFactor = hFactor;
    tilingData->tailHFactor = tailHFactor;
    tilingData->activeNum = dim0OfExpandedX;

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2Regbase::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "MoeFinalizeRoutingV2 tiling data: numBlocks[%ld] row[%ld] e[%ld] c[%ld] h[%ld] hAligned[%ld] k[%ld]"
            "rowOfFormerBlock[%ld] rowOfTailBlock[%ld] rowLoopOfFormerBlock[%ld] rowLoopOfTailBlock[%ld] "
            "rowFactor[%ld] tailRowFactorOfFormerBlock[%ld] tailRowFactorOfTailBlock[%ld] "
            "hLoop[%ld] hFactor[%ld] tailHFactor[%ld] zeroExpertStart[%ld] zeroExpertEnd[%ld] copyExpertStart[%ld] "
            "copyExpertEnd[%ld] constantExpertStart[%ld] constantExpertEnd[%ld] constExpertRangeNum[%ld]",
            usedCoreNum_, tilingData->row, tilingData->e, tilingData->c, tilingData->h, tilingData->hAligned,
            tilingData->k, tilingData->rowOfFormerBlock, tilingData->rowOfTailBlock, tilingData->rowLoopOfFormerBlock,
            tilingData->rowLoopOfTailBlock, tilingData->rowFactor, tilingData->tailRowFactorOfFormerBlock,
            tilingData->tailRowFactorOfTailBlock, tilingData->hLoop, tilingData->hFactor, tilingData->tailHFactor,
            tilingData->zeroExpertStart, tilingData->zeroExpertEnd, tilingData->copyExpertStart,
            tilingData->copyExpertEnd, tilingData->constantExpertStart, tilingData->constantExpertEnd,
            tilingData->constExpertRangeNum);
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CalcOpTiling()
{
    int64_t rowPerCore = Ops::Base::CeilDiv(row, static_cast<int64_t>(coreNum_));
    usedCoreNum_ = std::min(Ops::Base::CeilDiv(row, rowPerCore), static_cast<int64_t>(coreNum_));
    int64_t rowOfTailBlock = row - (usedCoreNum_ - 1) * rowPerCore;

    tilingData = context_->GetTilingData<MoeFinalizeRoutingV2RegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->row = row;
    tilingData->e = e;
    tilingData->c = c;
    tilingData->h = h;
    tilingData->hAligned = hAligned;
    tilingData->k = k;
    tilingData->rowOfFormerBlock = rowPerCore;
    tilingData->rowOfTailBlock = rowOfTailBlock;

    tilingData->zeroExpertStart = zeroExpertStart;
    tilingData->zeroExpertEnd = zeroExpertEnd;
    tilingData->copyExpertStart = copyExpertStart;
    tilingData->copyExpertEnd = copyExpertEnd;
    tilingData->constantExpertStart = constantExpertStart;
    tilingData->constantExpertEnd = constantExpertEnd;

    // 切分
    ge::graphStatus ret;
    if (rowKHFullLoad_) {
        ret = DoOpTilingRowKHFullLoad(rowPerCore, rowOfTailBlock);
    } else if (kHFullLoad_) {
        ret = DoOpTilingKHFullLoad(rowPerCore, rowOfTailBlock);
    } else if (hFullLoad_) {
        ret = DoOpTilingHFullLoad(rowPerCore, rowOfTailBlock);
    } else {
        ret = DoOpTilingSplitH(rowPerCore, rowOfTailBlock);
    }

    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tiling", "GRAPH_FAILED", "failed to do tiling"),
        return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

bool MoeFinalizeRoutingV2Regbase::IsRowKHFullLoad()
{
    int64_t expandedXAlignedByte;
    if (dropPadMode == DROP_LESS_COL || dropPadMode == DROP_LESS_ROW) {
        expandedXAlignedByte =
            Ops::Base::CeilDiv(static_cast<uint64_t>(row * k * h * dtypeSize), blockSize_) * blockSize_;
    } else {
        expandedXAlignedByte =
            Ops::Base::CeilDiv(static_cast<uint64_t>(e * c * h * dtypeSize), blockSize_) * blockSize_;
    }
    int64_t eHAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(e * h * dtypeSize), blockSize_) * blockSize_;
    int64_t scalesAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(k * scaleDtypeSize), blockSize_) * blockSize_;
    int64_t hAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_;
    hAligned = hAlignedByte / dtypeSize;
    int64_t hAligned32Byte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * sizeof(float)), blockSize_) * blockSize_;
    int64_t hasBiasvalue = hasBias_ ? eHAlignedByte : 0;
    int64_t hasScalevalue = hasScales_ ? scalesAlignedByte : 0;
    int64_t totalSize =
        expandedXAlignedByte + hasBiasvalue +
        DOUBLE_BUFFER * (hasScalevalue +
                         (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_) +
                          static_cast<int64_t>(hasConstantExpert_) * CONST_EXPERT_BUFFER_NUM) *
                             hAlignedByte +
                         hAligned32Byte * OUTPUT_BUFFER_NUM);

    return totalSize <= ubSize_;
}

bool MoeFinalizeRoutingV2Regbase::IsKHFullLoad()
{
    int64_t scalesAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(k * scaleDtypeSize), blockSize_) * blockSize_;
    int64_t hAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_;
    hAligned = hAlignedByte / dtypeSize;
    int64_t kHAlignedByte = k * hAligned * dtypeSize;
    int64_t hAligned32Byte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * sizeof(float)), blockSize_) * blockSize_;
    int64_t hasBiasvalue = hasBias_ ? kHAlignedByte : 0;
    int64_t hasScalevalue = hasScales_ ? scalesAlignedByte : 0;
    int64_t expandedRowIdxAlignedByte =
        Ops::Base::CeilDiv(static_cast<uint64_t>(k * sizeof(int32_t)), blockSize_) * blockSize_;
    int64_t hasExpandedRowIdxValue = k == 1 ? expandedRowIdxAlignedByte : 0;
    int64_t expertIdxAlignedByte =
        Ops::Base::CeilDiv(static_cast<uint64_t>(k * sizeof(int32_t)), blockSize_) * blockSize_;
    int64_t hasExpertIdxValue = (hasBias_ && k == 1) ? expertIdxAlignedByte : 0;
    int64_t totalSize =
        DOUBLE_BUFFER * (kHAlignedByte + hasBiasvalue + hasScalevalue + hasExpandedRowIdxValue + hasExpertIdxValue +
                         (static_cast<int64_t>(hasX1_) + static_cast<int64_t>(hasX2_) + static_cast<int64_t>(hasX_) +
                          static_cast<int64_t>(hasConstantExpert_) * CONST_EXPERT_BUFFER_NUM) *
                             hAlignedByte +
                         hAligned32Byte * OUTPUT_BUFFER_NUM);

    return totalSize <= ubSize_;
}

bool MoeFinalizeRoutingV2Regbase::IsHFullLoad()
{
    int64_t oneKAlignedByte = static_cast<int64_t>(blockSize_);
    int64_t hAlignedByte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * dtypeSize), blockSize_) * blockSize_;
    hAligned = hAlignedByte / dtypeSize;
    int64_t hAligned32Byte = Ops::Base::CeilDiv(static_cast<uint64_t>(h * sizeof(float)), blockSize_) * blockSize_;
    int64_t actualInputNum = INPUT_BUFFER_NUM - static_cast<int64_t>(!hasX1_) - static_cast<int64_t>(!hasX2_) -
                             static_cast<int64_t>(!hasBias_) - static_cast<int64_t>(!hasX_) -
                             static_cast<int64_t>(!hasConstantExpert_) * CONST_EXPERT_BUFFER_NUM;
    int64_t hasScalevalue = hasScales_ ? oneKAlignedByte : 0;
    int64_t totalSize =
        DOUBLE_BUFFER * (hAlignedByte * actualInputNum + hasScalevalue + hAligned32Byte * OUTPUT_BUFFER_NUM);

    return totalSize <= ubSize_;
}

bool MoeFinalizeRoutingV2Regbase::IsCapable()
{
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return false;
    }
    // IsRowKHFullLoad（全维度）→ IsKHFullLoad（K-H 维度）→ IsHFullLoad（仅 H 维度），粒度由粗到细，适配不同 MoE
    // 子任务；
    rowKHFullLoad_ = IsRowKHFullLoad();
    if (rowKHFullLoad_) {
        return true;
    }

    kHFullLoad_ = IsKHFullLoad();
    if (kHFullLoad_) {
        return true;
    }
    hFullLoad_ = IsHFullLoad();

    return true;
}

ge::graphStatus MoeFinalizeRoutingV2Regbase::CalcTilingKey()
{
    if (rowKHFullLoad_) {
        tilingKey_ = FULL_LOAD_ROW_K_H_BASE_TILING_KEY;
    } else if (kHFullLoad_) {
        tilingKey_ = FULL_LOAD_K_H_BASE_TILING_KEY;
    } else if (hFullLoad_) {
        tilingKey_ = FULL_LOAD_H_BASE_TILING_KEY;
    } else {
        tilingKey_ = SPLIT_H_BASE_TILING_KEY;
    }

    if (dropPadMode == DROP_LESS_COL) {
        tilingKey_ += DROPLESS_COL_TILING_LEY;
    } else if (dropPadMode == DROP_LESS_ROW) {
        tilingKey_ += DROPLESS_ROW_TILING_LEY;
    } else if (dropPadMode == DROP_PAD_COL) {
        tilingKey_ += DROPPAD_COL_TILING_KEY;
    } else {
        tilingKey_ += DROPPAD_ROW_TILING_KEY;
    }

    tilingKey_ += scaleDtypeKey;
    OP_LOGI(context_->GetNodeName(), "MoeFinalizeRoutingV2 get tiling key: %lx", tilingKey_);
    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2Regbase::DoPostTiling()
{
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeFinalizeRoutingV2, MoeFinalizeRoutingV2Regbase, 30000);

} // namespace optiling
