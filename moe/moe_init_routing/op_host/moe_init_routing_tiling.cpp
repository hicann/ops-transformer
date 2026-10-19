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
 * \file moe_init_routing_tiling.cpp
 * \brief
 */
#include "moe_init_routing_tiling.h"

namespace optiling {
const static int64_t SPLIT_N = 0;
const static int64_t SPLIT_K = 1;
const static int64_t SPLIT_ACTIVATE_ROW = 2;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static int64_t MAX_SORT_REPEAT_TIMES = 255;
const static int64_t MAX_SORT_LOOP_ELEMENT = SORT32_ALIGN_ELEMENT * MAX_SORT_REPEAT_TIMES;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t SIZE_31 = 31;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t SPLIT_K_THRESHOLD = 512;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

class MoeInitRountingTilingBase : public TilingBaseClass {
public:
    explicit MoeInitRountingTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRountingTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckOutShape();
    void Tiling4GatherOutComputeSplitK();
    void Tiling4GatherOutComputeSplitN();
    void Tiling4GatherOutComputeRow();
    void Tiling4GatherOutCompute();
    void Tiling4SrcToDstCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(VBSComputeTilingData* tilingData);
    void Tinlig4VBSOneCoreCompute(VBSComputeTilingData* tilingData);
    bool IsFullLoad();

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1024;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t cols_ = 0;
    int64_t inuptXDtypeSize_;

    const char* opName = "";
    MoeInitRoutingTilingData moeInitRoutingTilingData;
};

void MoeInitRountingTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRountingTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platform_info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
        mrgSortListMaxElement = 2048; // 单次排序大小，在david上设置成2048
    }
    moeInitRoutingTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingTilingBase::CheckOutShape()
{
    // 获取输入shape
    const gert::Shape expandedXShape = context_->GetOutputShape(0)->GetStorageShape();
    const gert::Shape expandedRowIdxShape = context_->GetOutputShape(1)->GetStorageShape();
    const gert::Shape expandedExpertIdxShape = context_->GetOutputShape(2)->GetStorageShape();

    size_t expandedXDimNnum = expandedXShape.GetDimNum();
    if (expandedXDimNnum != DIM_TWO) {
        std::string expandedXDimNumStr = std::to_string(expandedXDimNnum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_x", expandedXDimNumStr.c_str(), "2D");
        return ge::GRAPH_FAILED;
    }

    size_t expandedRowIdxDimNnum = expandedRowIdxShape.GetDimNum();
    if (expandedRowIdxDimNnum != DIM_ONE) {
        std::string expandedRowIdxDimNumStr = std::to_string(expandedRowIdxDimNnum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_row_idx", expandedRowIdxDimNumStr.c_str(),
                                     "1D");
        return ge::GRAPH_FAILED;
    }

    if (expandedRowIdxShape != expandedExpertIdxShape) {
        std::string shapeMsg =
            Ops::Base::ToString(expandedRowIdxShape) + " and " + Ops::Base::ToString(expandedExpertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "expanded_row_idx and expanded_expert_idx",
                                               shapeMsg.c_str(),
                                               "The shape of expanded_row_idx and expanded_expert_idx should be same");
        return ge::GRAPH_FAILED;
    }

    if (expandedXShape.GetDim(0) !=
        std::min(moeInitRoutingTilingData.get_n(), activateNum) * moeInitRoutingTilingData.get_k()) {
        std::string expandedXShapeStr = Ops::Base::ToString(expandedXShape);
        std::string reasonMsg =
            "The first dim of expanded_x should be " +
            std::to_string(std::min(moeInitRoutingTilingData.get_n(), activateNum) * moeInitRoutingTilingData.get_k());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expanded_x", expandedXShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (expandedXShape.GetDim(1) != moeInitRoutingTilingData.get_cols()) {
        std::string expandedXShapeStr = Ops::Base::ToString(expandedXShape);
        std::string reasonMsg =
            "The second dim of expanded_x should be " + std::to_string(moeInitRoutingTilingData.get_cols());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "expanded_x", expandedXShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (expandedRowIdxShape.GetDim(0) != totalLength) {
        std::string shapeMsg =
            Ops::Base::ToString(expandedRowIdxShape) + " and " + Ops::Base::ToString(expandedExpertIdxShape);
        std::string reasonMsg =
            "The first dim of expanded_row_idx and expanded_expert_idx should be " + std::to_string(totalLength);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "expanded_row_idx and expanded_expert_idx",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingTilingBase::GetShapeAttrsInfo()
{
    // 获取输入shape
    const gert::Shape xShape = context_->GetInputShape(0)->GetStorageShape();
    const gert::Shape rowIdxShape = context_->GetInputShape(1)->GetStorageShape();
    const gert::Shape expertIdxShape = context_->GetInputShape(2)->GetStorageShape();

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* activateNumPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activateNumPtr);
    activateNum = *activateNumPtr;
    OP_LOGI(context_->GetNodeName(), "activateNum is: %ld", activateNum);

    // 参数校验
    size_t xDimNnum = xShape.GetDimNum();
    if (xDimNnum != DIM_TWO) {
        std::string xDimNumStr = std::to_string(xDimNnum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", xDimNumStr.c_str(), "2D");
        return ge::GRAPH_FAILED;
    }

    size_t rowIdxDimNum = rowIdxShape.GetDimNum();
    if (rowIdxDimNum != DIM_TWO) {
        std::string rowIdxDimNumStr = std::to_string(rowIdxDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "row_idx", rowIdxDimNumStr.c_str(), "2D");
        return ge::GRAPH_FAILED;
    }

    if (rowIdxShape != expertIdxShape) {
        std::string shapeMsg = Ops::Base::ToString(rowIdxShape) + " and " + Ops::Base::ToString(expertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "row_idx and expert_idx", shapeMsg.c_str(),
                                               "The shape of row_idx and expert_idx should be same");
        return ge::GRAPH_FAILED;
    }

    if (xShape.GetDim(0) != expertIdxShape.GetDim(0)) {
        std::string shapeMsg = Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(expertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and expert_idx", shapeMsg.c_str(),
                                               "Input rows should be same");
        return ge::GRAPH_FAILED;
    }

    if (activateNum < 0) {
        std::string activateNumStr = std::to_string(activateNum);
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "active_num", activateNumStr.c_str(),
                                  "a non-negative number");
        return ge::GRAPH_FAILED;
    }

    this->n_ = expertIdxShape.GetDim(0);
    this->k_ = expertIdxShape.GetDim(1);
    this->cols_ = xShape.GetDim(1);
    moeInitRoutingTilingData.set_n(expertIdxShape.GetDim(0));
    moeInitRoutingTilingData.set_k(expertIdxShape.GetDim(1));
    moeInitRoutingTilingData.set_cols(xShape.GetDim(1));

    totalLength = moeInitRoutingTilingData.get_n() * moeInitRoutingTilingData.get_k();

    auto ret = CheckOutShape();
    inuptXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ret;
}

void MoeInitRountingTilingBase::ShowTilingData()
{
    OP_LOGI(opName, "moeInitRoutingTilingData is coreNum:%ld, n:%ld, cols:%ld,k:%ld",
            moeInitRoutingTilingData.get_coreNum(), moeInitRoutingTilingData.get_n(),
            moeInitRoutingTilingData.get_cols(), moeInitRoutingTilingData.get_k());
    OP_LOGI(
        opName,
        "VBSComputeTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, perCorePerLoopElements:%ld, "
        "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
        "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld",
        moeInitRoutingTilingData.vbsComputeParamsOp.get_needCoreNum(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLoops(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
        moeInitRoutingTilingData.vbsComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(opName, "VMSMiddleComputeTilingData is needCoreNum:%ld",
            moeInitRoutingTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGI(opName, "SortOutComputeTilingData is oneLoopMaxElements:%ld",
            moeInitRoutingTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
    OP_LOGI(opName,
            "SrcToDstComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCorePerLoopRows:%ld, "
            "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld, "
            "maxColsOneLoop:%ld",
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_needCoreNum(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_activateRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCorePerLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_perCoreLastLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCorePerLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_lastCoreLastLoopRows(),
            moeInitRoutingTilingData.srcToDstComputeParamsOp.get_maxColsOneLoop());
    OP_LOGI(opName,
            "GatherOutComputeTilingData is needCoreNum:%ld, activateRows:%ld, perCoreRows:%ld, perCoreK:%ld, "
            "perCorePerLoopK:%ld, perCoreLastLoopK:%ld, perCorePerLoopRows:%ld, "
            "perCoreLastLoopRows:%ld, lastCoreRows:%ld, lastCoreK:%ld, lastCorePerLoopK:%ld, lastCoreLastLoopK:%ld, "
            "lastCorePerLoopRows:%ld, lastCoreLastLoopRows:%ld, "
            "maxColsOneLoop:%ld, splitFlag:%ld",
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_needCoreNum(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_activateRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreK(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopK(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopK(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCorePerLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_perCoreLastLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreK(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopK(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopK(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCorePerLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_lastCoreLastLoopRows(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_maxColsOneLoop(),
            moeInitRoutingTilingData.gatherOutComputeParamsOp.get_splitFlag());
}
ge::graphStatus MoeInitRountingTilingBase::DoOpTiling()
{
    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    sortLoopMaxElement = std::min(sortLoopMaxElement, MAX_SORT_LOOP_ELEMENT);
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4GatherOutCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRountingTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeInitRountingTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = sizeof(float) * static_cast<size_t>(totalLength * NUM_TWO * NUM_THREE); // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        moeInitRoutingTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    size_t scatterWorkspaceSize = sizeof(int32_t) * static_cast<size_t>(totalLength);
    workspaceSize_ =
        sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    context_->SetLocalMemorySize(aicoreParams_.ubSize);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    context_->SetScheduleMode(1);
    moeInitRoutingTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeInitRoutingTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void MoeInitRountingTilingBase::Tinlig4VBSOneCoreCompute(VBSComputeTilingData* tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeInitRountingTilingBase::Tinlig4VBSMultiCoreCompute(VBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, sortLoopMaxElement); // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));    // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                               // 不能超过物理核数

    int64_t perCoreElements = totalLength / needCoreNum; // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            Ops::Base::CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                                (tilingData->get_perCoreLoops() - 1) *
                                                    tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(totalLength -
                                         (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                               SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                                 (tilingData->get_lastCoreLoops() - 1) *
                                                     tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    OP_CHECK_IF(tilingData->get_lastCoreLastLoopElements() <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "lastCoreLastLoopElements",
                    std::to_string(tilingData->get_lastCoreLastLoopElements()).c_str(), "vbs tiling failed"),
                ;);
}

void MoeInitRountingTilingBase::Tiling4VBSCompute()
{
    if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
        tilingKey_ = 1UL;
    } else {
        tilingKey_ = 2UL;
    }

    auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (GetTilingKey() == 1UL) { // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void MoeInitRountingTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeInitRountingTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRountingTilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

    int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                             (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength, aivNum);
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->get_needCoreNum() - 1);

    tilingData->set_perCoreRows(perCoreRows);

    if (perLoopMaxRows >= tilingData->get_perCoreRows()) { // 一个loop结束
        tilingData->set_perCorePerLoopRows(tilingData->get_perCoreRows());
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows());
    } else {
        tilingData->set_perCorePerLoopRows(perLoopMaxRows);
        tilingData->set_perCoreLastLoopRows(tilingData->get_perCoreRows() -
                                            (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), perLoopMaxRows) - 1) *
                                                perLoopMaxRows);
    }

    tilingData->set_lastCoreRows(lastCoreNum);
    if (perLoopMaxRows >= tilingData->get_lastCoreRows()) {
        tilingData->set_lastCorePerLoopRows(tilingData->get_lastCoreRows());
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows());
    } else {
        tilingData->set_lastCorePerLoopRows(perLoopMaxRows);
        tilingData->set_lastCoreLastLoopRows(tilingData->get_lastCoreRows() -
                                             (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), perLoopMaxRows) - 1) *
                                                 perLoopMaxRows);
    }
}

void MoeInitRountingTilingBase::Tiling4GatherOutComputeSplitK()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_splitFlag(SPLIT_K);
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows);
    if (activateNum <= 0) {
        OP_LOGW(opName, "activateNum or numRows is %ld, less than or equal to0", activateNum);
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_perCoreRows(moeInitRoutingTilingData.get_n());
    tilingData->set_lastCoreRows(moeInitRoutingTilingData.get_n());
    tilingData->set_activateRows(activateNum * moeInitRoutingTilingData.get_k());
    int64_t perCoreK = Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), aivNum);
    tilingData->set_perCoreK(perCoreK);
    int needCoreNum = Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), perCoreK);
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_lastCoreK(moeInitRoutingTilingData.get_k() -
                              (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreK());

    int64_t maxColsOneLoop = std::min(MAX_COLS_ONE_LOOP, moeInitRoutingTilingData.get_cols());
    tilingData->set_maxColsOneLoop(maxColsOneLoop);

    int64_t kFactor = tilingData->get_perCoreK();
    int64_t perLoopMaxRows =
        (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) /
        (static_cast<int64_t>(sizeof(int32_t)) * kFactor + (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE));
    OP_LOGD(opName, "perLoopMaxRows is %ld", perLoopMaxRows);
    while (perLoopMaxRows <= 0) {
        OP_LOGW(opName, "perLoopMaxRows is %ld, less than 0", perLoopMaxRows);
        kFactor = kFactor / NUM_TWO;
        perLoopMaxRows =
            (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) /
            (static_cast<int64_t>(sizeof(int32_t)) * kFactor + (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE));
    }

    tilingData->set_perCorePerLoopK(kFactor);
    tilingData->set_perCoreLastLoopK(tilingData->get_perCoreK() -
                                     (Ops::Base::CeilDiv(tilingData->get_perCoreK(), kFactor) - 1) * kFactor);
    tilingData->set_lastCorePerLoopK(kFactor);
    tilingData->set_lastCoreLastLoopK(tilingData->get_perCoreK() -
                                      (Ops::Base::CeilDiv(tilingData->get_perCoreK(), kFactor) - 1) * kFactor);
    tilingData->set_perCorePerLoopRows(std::min(tilingData->get_perCoreRows(), perLoopMaxRows));
    tilingData->set_perCoreLastLoopRows(
        tilingData->get_perCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), tilingData->get_perCorePerLoopRows()) - 1) *
            tilingData->get_perCorePerLoopRows());
    tilingData->set_lastCorePerLoopRows(std::min(tilingData->get_lastCoreRows(), perLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(
        tilingData->get_lastCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), tilingData->get_lastCorePerLoopRows()) - 1) *
            tilingData->get_lastCorePerLoopRows());
}

void MoeInitRountingTilingBase::Tiling4GatherOutComputeSplitN()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_splitFlag(SPLIT_N);
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows);
    int perCoreRows = Ops::Base::CeilDiv(realRows, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_perCoreRows(0);
        return;
    }

    int64_t maxColsOneLoop = std::min(MAX_COLS_ONE_LOOP, moeInitRoutingTilingData.get_cols());
    tilingData->set_maxColsOneLoop(maxColsOneLoop);
    int64_t kFactor = moeInitRoutingTilingData.get_k();
    int64_t perLoopMaxRows =
        (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) /
        (static_cast<int64_t>(sizeof(int32_t)) * kFactor + (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE));
    OP_LOGD(opName, "perLoopMaxRows is %ld", perLoopMaxRows);
    while (perLoopMaxRows <= 0) {
        OP_LOGW(opName, "perLoopMaxRows is %ld, less than 0", perLoopMaxRows);
        kFactor = kFactor / NUM_TWO;
        perLoopMaxRows =
            (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - kFactor * ONE_BLOCK_BYTE) /
            (static_cast<int64_t>(sizeof(int32_t)) * kFactor + (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE));
    }

    tilingData->set_perCoreK(moeInitRoutingTilingData.get_k());
    tilingData->set_perCorePerLoopK(kFactor);
    tilingData->set_perCoreLastLoopK(moeInitRoutingTilingData.get_k() -
                                     (Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), kFactor) - 1) * kFactor);
    tilingData->set_lastCoreK(moeInitRoutingTilingData.get_k());
    tilingData->set_lastCorePerLoopK(kFactor);
    tilingData->set_lastCoreLastLoopK(moeInitRoutingTilingData.get_k() -
                                      (Ops::Base::CeilDiv(moeInitRoutingTilingData.get_k(), kFactor) - 1) * kFactor);

    tilingData->set_activateRows(activateNum * moeInitRoutingTilingData.get_k());
    tilingData->set_perCoreRows(perCoreRows);
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(realRows, tilingData->get_perCoreRows()));
    tilingData->set_perCorePerLoopRows(std::min(tilingData->get_perCoreRows(), perLoopMaxRows));

    tilingData->set_perCoreLastLoopRows(
        tilingData->get_perCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), tilingData->get_perCorePerLoopRows()) - 1) *
            tilingData->get_perCorePerLoopRows());
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(realRows, tilingData->get_perCoreRows()));
    tilingData->set_lastCoreRows(realRows - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreRows());
    tilingData->set_lastCorePerLoopRows(std::min(tilingData->get_lastCoreRows(), perLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(
        tilingData->get_lastCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), tilingData->get_lastCorePerLoopRows()) - 1) *
            tilingData->get_lastCorePerLoopRows());
}

void MoeInitRountingTilingBase::Tiling4GatherOutComputeRow()
{
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->set_splitFlag(SPLIT_ACTIVATE_ROW);
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows) * moeInitRoutingTilingData.get_k();
    int perCoreRows = Ops::Base::CeilDiv(activateNum, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_perCoreRows(0);
        return;
    }

    int64_t maxColsOneLoop = std::min(MAX_COLS_ONE_LOOP, moeInitRoutingTilingData.get_cols());
    int64_t xUbSize = (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t perLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - xUbSize) /
                             static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    OP_LOGD(opName, "perLoopMaxRows is %ld", perLoopMaxRows);
    while (perLoopMaxRows <= 0) {
        OP_LOGW(opName, "perLoopMaxRows is %ld, less than 0", perLoopMaxRows);
        maxColsOneLoop = maxColsOneLoop / NUM_TWO;
        xUbSize = (maxColsOneLoop * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        perLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - xUbSize) /
                         static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    }
    tilingData->set_maxColsOneLoop(maxColsOneLoop);
    tilingData->set_activateRows(activateNum);
    tilingData->set_perCoreRows(perCoreRows);
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(activateNum, tilingData->get_perCoreRows()));
    tilingData->set_perCorePerLoopRows(std::min(tilingData->get_perCoreRows(), perLoopMaxRows));

    tilingData->set_perCoreLastLoopRows(
        tilingData->get_perCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_perCoreRows(), tilingData->get_perCorePerLoopRows()) - 1) *
            tilingData->get_perCorePerLoopRows());
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(activateNum, tilingData->get_perCoreRows()));
    tilingData->set_lastCoreRows(activateNum - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreRows());
    tilingData->set_lastCorePerLoopRows(std::min(tilingData->get_lastCoreRows(), perLoopMaxRows));
    tilingData->set_lastCoreLastLoopRows(
        tilingData->get_lastCoreRows() -
        (Ops::Base::CeilDiv(tilingData->get_lastCoreRows(), tilingData->get_lastCorePerLoopRows()) - 1) *
            tilingData->get_lastCorePerLoopRows());
}

bool MoeInitRountingTilingBase::IsFullLoad()
{
    if (this->cols_ > MAX_COLS_ONE_LOOP) {
        return false;
    }
    int64_t realRows = moeInitRoutingTilingData.get_n();
    activateNum = std::min(activateNum, realRows);
    int64_t perCoreRows = Ops::Base::CeilDiv(realRows, aivNum);
    int64_t sortSpace = Ops::Base::CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT *
                        sizeof(int32_t) * KV_FACTOR * ONE_CORE_SORT_BUFFER;
    int64_t gatherSpace = 0;
    int64_t remainUbAfterSort = 0;
    if (this->n_ < aivNum && this->n_ < this->k_) {
        gatherSpace = Ops::Base::CeilDiv(moeInitRoutingTilingData.get_cols() * inuptXDtypeSize_, ONE_BLOCK_BYTE) *
                      ONE_BLOCK_BYTE * realRows;
        remainUbAfterSort = aicoreParams_.ubSize - sortSpace - gatherSpace;
        if (remainUbAfterSort > 0) {
            Tiling4GatherOutComputeSplitK();
            return true;
        }
    }

    gatherSpace = Ops::Base::CeilDiv(moeInitRoutingTilingData.get_cols() * inuptXDtypeSize_, ONE_BLOCK_BYTE) *
                  ONE_BLOCK_BYTE * perCoreRows;
    remainUbAfterSort = aicoreParams_.ubSize - sortSpace - gatherSpace;
    if (remainUbAfterSort > 0) {
        Tiling4GatherOutComputeSplitN();
        return true;
    }
    return false;
}

void MoeInitRountingTilingBase::Tiling4GatherOutCompute()
{
    if (GetTilingKey() == 1) {
        bool isFullLoad = IsFullLoad();
        if (isFullLoad) {
            tilingKey_ = 0;
            return;
        }
    }
    if (moeInitRoutingTilingData.get_n() / NUM_TWO > activateNum) {
        tilingKey_ = tilingKey_ + 2L;
        return Tiling4GatherOutComputeRow();
    }

    if (moeInitRoutingTilingData.get_n() < aivNum &&
        moeInitRoutingTilingData.get_n() < moeInitRoutingTilingData.get_k()) {
        return Tiling4GatherOutComputeSplitK();
    }
    return Tiling4GatherOutComputeSplitN();
}

static ge::graphStatus TilingForMoeInitRouting(gert::TilingContext* context)
{
    MoeInitRountingTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMoeInitRounting(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRouting)
    .Tiling(TilingForMoeInitRouting)
    .TilingParse<MoeInitRoutingCompileInfo>(TilingPrepareForMoeInitRounting);

} // namespace optiling
