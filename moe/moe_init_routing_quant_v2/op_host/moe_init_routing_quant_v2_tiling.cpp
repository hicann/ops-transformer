/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_quant_v2_tiling.cpp
 * \brief
 */
#include "moe_init_routing_quant_v2_tiling.h"

namespace optiling {

const static int64_t ATTR_QUANT_MODE = 6;
const static int64_t TILING_KEY_BASE = 10000;
const static int64_t TILING_KEY_PERF_BASE = 20000;
const static int64_t TILING_KEY_QUANT_BASE = 1000;
const static int64_t TILING_KEY_DROP_MODE_BASE = 100;
const static int64_t TILING_KEY_SORT_BASE = 10;
const static int64_t NUM_TWO = 2;
const static int64_t ONE_BLOCK_BYTE = 32;
const static int64_t FOUR_BLOCK_BYTE = 128;
const static int64_t MAX_COLS_ONE_LOOP = 8192;
const static int64_t INDEX_SCALE = 2;
const static int64_t INDEX_OFFSET = 3;
const static int64_t SMOOTH_NONE = 0;
const static int64_t SMOOTH_1H = 1;
const static int64_t SMOOTH_EH = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MAX_COLS_DYNAMIC_QUANT = 6144;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21;
const static int64_t DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER = 13;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_64 = 64;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_128 = 128;
const static int64_t OUTOUT_DYNAMIC_QUANT_SCALE = 4;
const static int64_t OUTOUT_EXPANDED_X = 0;
const static int64_t FULLLOAD_H_LIMIT = 7168;
const static int64_t HIST_REGBASE_MAX_EXPERT_NUM = 256;
const static int64_t SIZE_INT32 = 4;
const static int64_t SIZE_INT16 = 2;
const static int64_t SIZE_INT8 = 1;
const static int64_t SIZE_FP32 = 4;
const static int64_t INT4_USED_BYTE = 32 * 2 + 32 * 2 + 256; // 2*32 for align 32*2+256 for compute

class MoeInitRoutingQuantV2TilingBase : public InnerMoeInitRoutingV2TilingBase {
public:
    explicit MoeInitRoutingQuantV2TilingBase(gert::TilingContext* context)
        : InnerMoeInitRoutingV2TilingBase(context)
    {}
    ~MoeInitRoutingQuantV2TilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        InnerMoeInitRoutingV2TilingBase::Reset(context);
    }

protected:
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckOutShape() override;
    ge::graphStatus CheckInt4Info();
    bool IsFullLoadQuant(int64_t space);
    bool IsFullLoadDynamicQuant(int64_t space);
    bool IsFullLoad() override;
    void SetGatherTilingData(InnerMoeV2GatherOutComputeTilingData* tilingData, int64_t perCoreRows,
                             int64_t lastCoreRows, int64_t cols);
    void SetGatherTilingDataCols(InnerMoeV2GatherOutComputeTilingData* tilingData, int64_t baseMaxCols, int64_t cols);
    void SetGatherTilingDataRows(InnerMoeV2GatherOutComputeTilingData* tilingData, int64_t perCoreRows,
                                 int64_t lastCoreRows, int64_t basePerLoopMaxRows);
    void Tiling4GatherQuant();
    void Tiling4GatherDynamicQuant();
    void Tiling4SrcToDstCapacityCompute() override;
    void Tiling4GatherOutCompute() override;
    void CopyGatherOutTiling(InnerMoeV2GatherOutComputeTilingData& dst, InnerMoeV2GatherOutComputeTilingData& src);
    void CopyTilingData();

    int64_t quantMode;
    bool isInt4;
    MoeInitRoutingQuantV2TilingData quantTilingData;
};

#define CHECK_FAIL(context, cond, log_func) \
    do { \
        if (cond) { \
            log_func; \
            return ge::GRAPH_FAILED; \
        } \
    } while (0)

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return x <= y ? x : x % y;
}

inline static int64_t AlignOneBlockByte(int64_t x)
{
    return (x + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

inline static int64_t AlignOneBlockByteCeil(int64_t x)
{
    return x / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::CheckOutShape()
{
    if (InnerMoeInitRoutingV2TilingBase::CheckOutShape() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (quantMode != 0) {
        auto dynamicShapePtr = context_->GetOutputShape(OUTOUT_DYNAMIC_QUANT_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dynamicShapePtr);

        auto dynamicQuantScaleDesc = context_->GetOutputDesc(OUTOUT_DYNAMIC_QUANT_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context_, dynamicQuantScaleDesc);
        auto dt = dynamicQuantScaleDesc->GetDataType();
        CHECK_FAIL(context_, dt != ge::DT_FLOAT,
                   OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "dynamicQuantScale",
                                             Ops::Base::ToString(dt).c_str(), "FLOAT"));

        const gert::Shape dynamicQuantScaleShape = dynamicShapePtr->GetStorageShape();
        size_t dynamicQuantScaleDimNum = dynamicQuantScaleShape.GetDimNum();
        std::string dynamicQuantScaleDimStr = std::to_string(dynamicQuantScaleDimNum) + "D";
        CHECK_FAIL(context_, dynamicQuantScaleDimNum != 1U,
                   OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "dynamicQuantScale",
                                                dynamicQuantScaleDimStr.c_str(), "1D"));
        if (dropPadMode == 0) {
            int64_t firstDim = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();
            firstDim = activateNum == 0 ? firstDim : std::min(firstDim, activateNum);
            CHECK_FAIL(
                context_, dynamicQuantScaleShape.GetDim(0) != firstDim,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "dynamicQuantScale", Ops::Base::ToString(dynamicQuantScaleShape).c_str(),
                    "The first dim of dynamicQuantScale should be n * k or activeNum"));
        } else {
            CHECK_FAIL(
                context_, dynamicQuantScaleShape.GetDim(0) != expertNum * expertCapacity,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "dynamicQuantScale", Ops::Base::ToString(dynamicQuantScaleShape).c_str(),
                    "The first dim of dynamicQuantScale should be expertNum * expertCapacity"));
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::CheckInt4Info()
{
    auto expandedXDesc = context_->GetOutputDesc(OUTOUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXDesc);
    auto expandedXType = expandedXDesc->GetDataType();
    isInt4 = expandedXType == ge::DT_INT4;
    if (isInt4) {
        int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
        CHECK_FAIL(context_, cols % NUM_TWO != 0,
                   OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                       context_->GetNodeName(), "x", std::to_string(cols).c_str(),
                       "The second dim of x should be a multiple of 2 when expandedX is int4"));
        CHECK_FAIL(context_, quantMode != 1,
                   OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "quantMode", std::to_string(quantMode).c_str(),
                                             "1 when expandedX is int4"));
        CHECK_FAIL(context_, dropPadMode != 0,
                   OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "dropPadMode",
                                             std::to_string(dropPadMode).c_str(), "0 when expandedX is int4"));
        auto scaleShapePtr = context_->GetOptionalInputShape(INDEX_SCALE);
        if (scaleShapePtr != nullptr) {
            auto scaleDesc = context_->GetOptionalInputDesc(INDEX_SCALE);
            OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
            auto smoothShape = scaleShapePtr->GetStorageShape();
            size_t smoothDimNum = smoothShape.GetDimNum();
            std::string smoothDimStr = std::to_string(smoothDimNum) + "D";
            CHECK_FAIL(context_, smoothDimNum != static_cast<size_t>(NUM_TWO),
                       OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", smoothDimStr.c_str(), "2D"));
            CHECK_FAIL(context_, smoothShape.GetDim(0) != 1,
                       OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                           context_->GetNodeName(), "scale", Ops::Base::ToString(smoothShape).c_str(),
                           "The first dim of scale should be 1 when expandedX is int4"));
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    const int64_t* quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
    if (quantModePtr != nullptr) {
        quantMode = *quantModePtr;
    }
    CHECK_FAIL(
        context_, quantMode < 0 || quantMode > 1,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "quantMode", std::to_string(quantMode).c_str(), "0 or 1"));
    if (InnerMoeInitRoutingV2TilingBase::GetShapeAttrsInfo() == ge::GRAPH_FAILED ||
        CheckInt4Info() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    auto scaleShapePtr = context_->GetOptionalInputShape(INDEX_SCALE);
    if (quantMode == 0) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
        auto scaleDesc = context_->GetOptionalInputDesc(INDEX_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
        auto dt_scale = scaleDesc->GetDataType();
        CHECK_FAIL(context_, dt_scale != ge::DT_FLOAT,
                   OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scale", Ops::Base::ToString(dt_scale).c_str(),
                                             "FLOAT"));

        auto offsetShapePtr = context_->GetOptionalInputShape(INDEX_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(context_, offsetShapePtr);
        auto offsetDesc = context_->GetOptionalInputDesc(INDEX_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
        auto dt_offset = offsetDesc->GetDataType();
        CHECK_FAIL(context_, dt_offset != ge::DT_FLOAT,
                   OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "offset", Ops::Base::ToString(dt_offset).c_str(),
                                             "FLOAT"));

        auto scaleShape = scaleShapePtr->GetStorageShape();
        std::string scaleDimStr = std::to_string(scaleShape.GetDimNum()) + "D";
        CHECK_FAIL(context_, scaleShape.GetDimNum() != 1,
                   OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", scaleDimStr.c_str(), "1D"));
        CHECK_FAIL(context_, scaleShape.GetDim(0) != 1,
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale",
                                                         Ops::Base::ToString(scaleShape).c_str(),
                                                         "The first dim of scale should be 1"));
        auto offsetShape = offsetShapePtr->GetStorageShape();
        std::string offsetDimStr = std::to_string(offsetShape.GetDimNum()) + "D";
        CHECK_FAIL(context_, offsetShape.GetDimNum() != 1,
                   OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "offset", offsetDimStr.c_str(), "1D"));
        CHECK_FAIL(context_, offsetShape.GetDim(0) != 1,
                   OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "offset",
                                                         Ops::Base::ToString(offsetShape).c_str(),
                                                         "The first dim of offset should be 1"));
    } else {
        if (scaleShapePtr != nullptr) {
            auto scaleDesc = context_->GetOptionalInputDesc(INDEX_SCALE);
            OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
            auto dt = scaleDesc->GetDataType();
            CHECK_FAIL(
                context_, dt != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scale", Ops::Base::ToString(dt).c_str(), "FLOAT"));

            int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
            auto smoothShape = scaleShapePtr->GetStorageShape();
            size_t smoothDimNum = smoothShape.GetDimNum();
            std::string smoothDimStr = std::to_string(smoothDimNum) + "D";
            CHECK_FAIL(context_, smoothDimNum != static_cast<size_t>(NUM_TWO),
                       OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", smoothDimStr.c_str(), "2D"));
            CHECK_FAIL(context_, smoothShape.GetDim(0) != 1 && smoothShape.GetDim(0) != expertNum,
                       OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale",
                                                             Ops::Base::ToString(smoothShape).c_str(),
                                                             "The first dim of scale should be 1 or expertNum"));
            CHECK_FAIL(context_, smoothShape.GetDim(1) != cols,
                       OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                           context_->GetNodeName(), "scale and x",
                           (std::to_string(smoothShape.GetDim(1)) + " and " + std::to_string(cols)).c_str(),
                           "The second dim of scale and x should be equal"));
            quantTilingData.set_smoothType((smoothShape.GetDim(0) == 1) ? SMOOTH_1H : SMOOTH_EH);
        } else {
            quantTilingData.set_smoothType(SMOOTH_NONE);
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool MoeInitRoutingQuantV2TilingBase::IsFullLoadQuant(int64_t space)
{
    int64_t perCoreXRows = moeInitRoutingTilingData.get_n() / aivNum;
    int64_t remainder = moeInitRoutingTilingData.get_n() % aivNum;
    // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
    perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
    int64_t quantBaseSpace = AlignOneBlockByte(moeInitRoutingTilingData.get_cols());
    int64_t quantSpace = quantBaseSpace * (inuptXDtypeSize_ + SIZE_INT8 + SIZE_FP32 + SIZE_INT16) * perCoreXRows;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - space - quantSpace;
    return remainUbAfterSort > 0;
}

bool MoeInitRoutingQuantV2TilingBase::IsFullLoadDynamicQuant(int64_t space)
{
    int64_t quantSpace = AlignOneBlockByte(moeInitRoutingTilingData.get_cols()) * DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER;
    int64_t scaleOutSpace = 64;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - space - scaleOutSpace - quantSpace;
    return remainUbAfterSort > 0;
}

bool MoeInitRoutingQuantV2TilingBase::IsFullLoad()
{
    if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.get_cols() > MAX_COLS_ONE_LOOP ||
        this->dropPadMode == 1) {
        return false;
    }
    int64_t sortSpace = AlignOneBlockByte(this->totalLength) * SIZE_INT32 * ONE_CORE_SORT_BUFFER;
    int64_t otherSpace = AlignOneBlockByte(this->totalLength) * SIZE_INT32 * NUM_THREE;
    int64_t expertSpace = AlignOneBlockByte(this->expertNum * SIZE_INT32);
    if (quantMode == 0) {
        return IsFullLoadQuant(sortSpace + otherSpace + expertSpace);
    } else {
        return IsFullLoadDynamicQuant(sortSpace + otherSpace + expertSpace);
    }
}

void MoeInitRoutingQuantV2TilingBase::SetGatherTilingData(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                                          int64_t perCoreRows, int64_t lastCoreRows, int64_t cols)
{
    tilingData->set_perCorePerLoopRows(perCoreRows);
    tilingData->set_perCoreLastLoopRows(perCoreRows);
    tilingData->set_lastCorePerLoopRows(lastCoreRows);
    tilingData->set_lastCoreLastLoopRows(lastCoreRows);
    tilingData->set_perCoreLoops(1);
    tilingData->set_lastCoreLoops(1);
    tilingData->set_perLoopCols(cols);
    tilingData->set_lastLoopCols(cols);
    tilingData->set_colLoops(1);
}
struct GatherTilingDataParam {
    int64_t perCorePerLoopRows = 1;
    int64_t lastCorePerLoopRows = 1;
    int64_t cols = 1;
    int64_t perCoreLastLoopRows = 1;
    int64_t lastCoreLastLoopRows = 1;
    int64_t perCoreLoops = 1;
    int64_t lastCoreLoops = 1;
};

void SetGatherTilingDatawithloop(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                 GatherTilingDataParam gatherTilingDataParam)
{
    tilingData->set_perCorePerLoopRows(gatherTilingDataParam.perCorePerLoopRows);
    tilingData->set_perCoreLastLoopRows(gatherTilingDataParam.perCoreLastLoopRows);
    tilingData->set_lastCorePerLoopRows(gatherTilingDataParam.lastCorePerLoopRows);
    tilingData->set_lastCoreLastLoopRows(gatherTilingDataParam.lastCoreLastLoopRows);
    tilingData->set_perCoreLoops(gatherTilingDataParam.perCoreLoops);
    tilingData->set_lastCoreLoops(gatherTilingDataParam.lastCoreLoops);
    tilingData->set_perLoopCols(gatherTilingDataParam.cols);
    tilingData->set_lastLoopCols(gatherTilingDataParam.cols);
    tilingData->set_colLoops(1);
}

void MoeInitRoutingQuantV2TilingBase::SetGatherTilingDataCols(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                                              int64_t baseMaxCols, int64_t cols)
{
    tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
    tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
    tilingData->set_colLoops(baseMaxCols == 0 ? 0 : (cols + baseMaxCols - 1) / baseMaxCols);
}
void MoeInitRoutingQuantV2TilingBase::SetGatherTilingDataRows(InnerMoeV2GatherOutComputeTilingData* tilingData,
                                                              int64_t perCoreRows, int64_t lastCoreRows,
                                                              int64_t basePerLoopMaxRows)
{
    tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->set_perCoreLoops(basePerLoopMaxRows == 0 ? 0 :
                                                           (perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

    tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->set_lastCoreLoops(
        basePerLoopMaxRows == 0 ? 0 : (lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
}

void MoeInitRoutingQuantV2TilingBase::Tiling4SrcToDstCapacityCompute()
{
    // 直方图是否使用regBase
    bool histWithRegBase = regBase && expertNum <= HIST_REGBASE_MAX_EXPERT_NUM;
    quantTilingData.set_histWithRegBase(histWithRegBase ? 1 : 0);

    if (quantMode == 0 || dropPadMode == 0) {
        InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute();
        return;
    }
    auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;

    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(totalLength, perCoreRows));
    int64_t cols = moeInitRoutingTilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize = AlignOneBlockByte(perCoreRows * SIZE_INT32) * NUM_FOUR;
    int64_t colSize = AlignOneBlockByte(cols * SIZE_INT8) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
    int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
    if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
        int64_t totalColSize = AlignOneBlockByte(baseMaxCols * SIZE_INT8) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - totalColSize - scaleSize) / SIZE_INT32) / NUM_FOUR;
        if (cols < MAX_COLS_DYNAMIC_QUANT) {
            basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize - scaleSize) / SIZE_INT32) / NUM_FOUR;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = AlignOneBlockByteCeil(ubSize - rowSize - scaleSize) / DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

void MoeInitRoutingQuantV2TilingBase::Tiling4GatherQuant()
{
    auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t sizeOfCol = SIZE_INT8 * NUM_TWO + SIZE_FP32 + SIZE_INT16 + inuptXDtypeSize_ * NUM_TWO;
    int64_t rowSize = AlignOneBlockByte((perCoreRows * SIZE_INT32 * NUM_TWO));
    int64_t colSize = AlignOneBlockByte(cols * sizeOfCol);
    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize = AlignOneBlockByte(baseMaxCols * sizeOfCol);
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - baseMaxColsSize) / NUM_TWO / SIZE_INT32);
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize) / NUM_TWO / SIZE_INT32);
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = AlignOneBlockByteCeil((ubSize - rowSize) / sizeOfCol);
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

void MoeInitRoutingQuantV2TilingBase::Tiling4GatherDynamicQuant()
{
    auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    aicoreParams_.ubSize = isInt4 ? aicoreParams_.ubSize - INT4_USED_BYTE : aicoreParams_.ubSize;
    int64_t cols = InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols();
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize = AlignOneBlockByte(perCoreRows * SIZE_INT32) * NUM_FOUR;
    int64_t colSize = AlignOneBlockByte(cols * SIZE_INT8) * DYNAMIC_QUANT_COLS_BUFFER;
    int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
    int64_t onceRowSize =
        (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - scaleSize - ONE_BLOCK_BYTE * NUM_FOUR * NUM_THREE) /
        (SIZE_INT32 * NUM_FOUR);
    int64_t oneBlockNumInt = static_cast<int64_t>(ONE_BLOCK_BYTE) / static_cast<int64_t>(SIZE_INT32);
    onceRowSize = onceRowSize / oneBlockNumInt * oneBlockNumInt;
    bool ifOneLoopInt4 =
        isInt4 &&
        ((static_cast<int64_t>(aicoreParams_.ubSize) > colSize + scaleSize + ONE_BLOCK_BYTE * NUM_FOUR * NUM_FOUR));
    bool ifOneLoop =
        ((static_cast<int64_t>(aicoreParams_.ubSize) > colSize + scaleSize + ONE_BLOCK_BYTE * NUM_FOUR * NUM_FOUR) &&
         quantTilingData.get_smoothType() == SMOOTH_NONE && cols == FULLLOAD_H_LIMIT) ||
        ifOneLoopInt4;
    int64_t perCoreOnceRowSize = ifOneLoop ? std::min(onceRowSize, perCoreRows) : perCoreRows;
    int64_t lastCoreOnceRowSize = ifOneLoop ? std::min(onceRowSize, lastCoreRows) : lastCoreRows;
    int64_t perCoreLoops = ifOneLoop ? Ops::Base::CeilDiv(perCoreRows, perCoreOnceRowSize) : 1;
    int64_t lastCoreLoops = ifOneLoop ? Ops::Base::CeilDiv(lastCoreRows, lastCoreOnceRowSize) : 1;
    int64_t perCoreLastLoopRows = ifOneLoop ? GetPerOrLastValue(perCoreRows, perCoreOnceRowSize) : perCoreRows;
    int64_t lastCoreLastLoopRows = ifOneLoop ? GetPerOrLastValue(lastCoreRows, lastCoreOnceRowSize) : lastCoreRows;
    if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize) || ifOneLoop) {
        GatherTilingDataParam gatherTilingDataParam;
        gatherTilingDataParam.perCorePerLoopRows = perCoreOnceRowSize;
        gatherTilingDataParam.lastCorePerLoopRows = lastCoreOnceRowSize;
        gatherTilingDataParam.cols = cols;
        gatherTilingDataParam.perCoreLastLoopRows = perCoreLastLoopRows;
        gatherTilingDataParam.lastCoreLastLoopRows = lastCoreLastLoopRows;
        gatherTilingDataParam.perCoreLoops = perCoreLoops;
        gatherTilingDataParam.lastCoreLoops = lastCoreLoops;
        SetGatherTilingDatawithloop(tilingData, gatherTilingDataParam);
    } else {
        int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
        int64_t totalColSize = AlignOneBlockByte(baseMaxCols * SIZE_INT8) * DYNAMIC_QUANT_COLS_BUFFER;
        int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
        int64_t basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - totalColSize - scaleSize) / SIZE_INT32) / NUM_FOUR;
        if (cols < MAX_COLS_DYNAMIC_QUANT) {
            basePerLoopMaxRows = AlignOneBlockByteCeil((ubSize - colSize - scaleSize) / SIZE_INT32) / NUM_FOUR;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols =
                AlignOneBlockByteCeil((ubSize - rowSize - scaleSize) / DYNAMIC_QUANT_COLS_BUFFER); // aligin ub
        }
        SetGatherTilingDataCols(tilingData, baseMaxCols, cols);
        SetGatherTilingDataRows(tilingData, perCoreRows, lastCoreRows, basePerLoopMaxRows);
    }
}

void MoeInitRoutingQuantV2TilingBase::Tiling4GatherOutCompute()
{
    auto tilingData = &quantTilingData.gatherOutComputeParamsOp;
    tilingData->set_activateRows(totalLength);
    if (dropPadMode == 0 && activateNum > 0) {
        tilingData->set_activateRows(std::min(activateNum, totalLength));
    }
    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(totalLength, perCoreRows));
    tilingData->set_perCoreRows(perCoreRows);

    if (quantMode == 0) {
        Tiling4GatherQuant();
    } else {
        Tiling4GatherDynamicQuant();
    }
}

void MoeInitRoutingQuantV2TilingBase::CopyGatherOutTiling(InnerMoeV2GatherOutComputeTilingData& dst,
                                                          InnerMoeV2GatherOutComputeTilingData& src)
{
    dst.set_needCoreNum(src.get_needCoreNum());
    dst.set_activateRows(src.get_activateRows());
    dst.set_perCoreRows(src.get_perCoreRows());
    dst.set_perCorePerLoopRows(src.get_perCorePerLoopRows());
    dst.set_perCoreLastLoopRows(src.get_perCoreLastLoopRows());
    dst.set_lastCoreRows(src.get_lastCoreRows());
    dst.set_lastCorePerLoopRows(src.get_lastCorePerLoopRows());
    dst.set_lastCoreLastLoopRows(src.get_lastCoreLastLoopRows());
    dst.set_perCoreLoops(src.get_perCoreLoops());
    dst.set_lastCoreLoops(src.get_lastCoreLoops());
    dst.set_perLoopCols(src.get_perLoopCols());
    dst.set_lastLoopCols(src.get_lastLoopCols());
    dst.set_colLoops(src.get_colLoops());
}

void MoeInitRoutingQuantV2TilingBase::CopyTilingData()
{
    quantTilingData.set_coreNum(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_coreNum());
    quantTilingData.set_n(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_n());
    quantTilingData.set_cols(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols());
    quantTilingData.set_k(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_k());
    quantTilingData.set_expertCapacity(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertCapacity());
    quantTilingData.set_expertNum(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertNum());
    quantTilingData.set_dropPadMode(InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_dropPadMode());
    quantTilingData.set_expertTokensCountOrCumsumFlag(
        InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertTokensCountOrCumsumFlag());
    quantTilingData.set_expertTokensBeforeCapacityFlag(
        InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_expertTokensBeforeCapacityFlag());

    auto vbsTilingData = &InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.vbsComputeParamsOp;
    quantTilingData.vbsComputeParamsOp.set_needCoreNum(vbsTilingData->get_needCoreNum());
    quantTilingData.vbsComputeParamsOp.set_perCoreElements(vbsTilingData->get_perCoreElements());
    quantTilingData.vbsComputeParamsOp.set_perCoreLoops(vbsTilingData->get_perCoreLoops());
    quantTilingData.vbsComputeParamsOp.set_perCorePerLoopElements(vbsTilingData->get_perCorePerLoopElements());
    quantTilingData.vbsComputeParamsOp.set_perCoreLastLoopElements(vbsTilingData->get_perCoreLastLoopElements());
    quantTilingData.vbsComputeParamsOp.set_lastCoreElements(vbsTilingData->get_lastCoreElements());
    quantTilingData.vbsComputeParamsOp.set_lastCoreLoops(vbsTilingData->get_lastCoreLoops());
    quantTilingData.vbsComputeParamsOp.set_lastCorePerLoopElements(vbsTilingData->get_lastCorePerLoopElements());
    quantTilingData.vbsComputeParamsOp.set_lastCoreLastLoopElements(vbsTilingData->get_lastCoreLastLoopElements());
    quantTilingData.vbsComputeParamsOp.set_oneLoopMaxElements(vbsTilingData->get_oneLoopMaxElements());

    quantTilingData.vmsMiddleComputeParamsOp.set_needCoreNum(
        InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    quantTilingData.sortOutComputeParamsOp.set_oneLoopMaxElements(
        InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());

    CopyGatherOutTiling(quantTilingData.srcToDstComputeParamsOp,
                        InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.srcToDstComputeParamsOp);
    CopyGatherOutTiling(quantTilingData.srcToDstCapacityComputeParamsOp,
                        InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp);
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::GetWorkspaceSize()
{
    InnerMoeInitRoutingV2TilingBase::GetWorkspaceSize();
    bool useCols =
        (quantTilingData.gatherOutComputeParamsOp.get_colLoops() > 1) ||
        (InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp.get_colLoops() > 1);
    if (quantMode == 1 && useCols) {
        workspaceSize_ += aivNum * InnerMoeInitRoutingV2TilingBase::moeInitRoutingTilingData.get_cols() * SIZE_FP32;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingQuantV2TilingBase::PostTiling()
{
    CopyTilingData();
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    context_->SetLocalMemorySize(aicoreParams_.ubSize);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    quantTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(quantTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingQuantV2TilingBase::GetTilingKey() const
{
    if (isFullLoad) {
        return static_cast<uint64_t>(TILING_KEY_PERF_BASE + static_cast<int64_t>(quantMode) * TILING_KEY_QUANT_BASE);
    }
    context_->SetScheduleMode(1);
    return static_cast<uint64_t>(static_cast<int64_t>(TILING_KEY_BASE) +
                                 static_cast<int64_t>(quantMode) * static_cast<int64_t>(TILING_KEY_QUANT_BASE) +
                                 static_cast<int64_t>(dropPadMode) * static_cast<int64_t>(TILING_KEY_DROP_MODE_BASE) +
                                 static_cast<int64_t>(totalLength > sortLoopMaxElement) *
                                     static_cast<int64_t>(TILING_KEY_SORT_BASE));
}

ge::graphStatus TilingForMoeInitRoutingQuantV2(gert::TilingContext* context)
{
    MoeInitRoutingQuantV2TilingBase tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForMoeInitRoutingQuantV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingQuantV2)
    .Tiling(TilingForMoeInitRoutingQuantV2)
    .TilingParse<MoeInitRoutingQuantV2CompileInfo>(TilingPrepareForMoeInitRoutingQuantV2);

} // namespace optiling
