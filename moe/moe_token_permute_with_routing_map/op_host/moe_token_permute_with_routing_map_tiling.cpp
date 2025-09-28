/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_permute_with_routing_map_tiling.cpp
 * \brief
 */
#include "moe_token_permute_with_routing_map_tiling.h"

namespace {
const static int64_t SPLIT_N = 0;
const static int64_t SPLIT_K = 1;
const static int64_t SPLIT_ACTIVATE_ROW = 2;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t SIZE_31 = 31;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t SPLIT_K_THRESHOLD = 512;
const static int64_t MAX_INDICES_NUM = 512;
const static int64_t INT32_DTYPE_SIZE = 4;
const static int64_t DATA_MOVE_ALIGN = 512;
const static int64_t BUFFER_NUM = 2;
const static int64_t MAX_BLOCK_COUNT = 4095;
const static uint64_t SORT_ONE_CORE_MODE = 1UL;
const static uint64_t SORT_MULTI_CORE_MODE = 2UL;
const static uint64_t ENABLE_NUMOUTTOKENS = 4L;
const static uint64_t SPLIT_D_MODE = 2L;
const static int64_t SORT_LIMIT_LENGTH = 16777215;
const static int64_t SORT_WORK_SPACE_NUM = 2;
constexpr static uint64_t BLOCK_SIZE = 256;
constexpr static uint64_t DOUBLE_BUFFER = 2;
constexpr static uint64_t IO_QUE = 2;
constexpr static uint64_t MASK_ONE_DATA_SIZE = 7;   // doublebufer 2* 1(int8)+4(int32)+1(int8)
constexpr static uint64_t INDEX_ONE_DATA_SIZE = 12; // doublebufer 2 *4(int32)
constexpr static uint64_t PROB_INDEX = 2;
constexpr static uint64_t PAD_KEY = 9;
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
template <typename T>
static auto GetCeilInt(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

template <typename T>
static auto GetDiv(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return (value1) / value2;
}

template <typename T>
static auto GetRem(const T& value1, const T& value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}

template <typename T1, typename T2>
inline auto FloorAlign(const T1& a, const T2& b) -> T1
{
    if (b != 0) {
        return (a) / b * b;
    }
    return a;
}

template <typename T1, typename T2>
inline auto UpAlign(const T1& a, const T2& b) -> T1
{
    if (b != 0) {
        return (a + b - 1) / b * b;
    }
    return a;
}

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            dsize = sizeof(int16_t);
            return true;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            dsize = sizeof(int32_t);
            return true;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            dsize = sizeof(int64_t);
            return true;
        default:
            return false;
    }
}

inline static int64_t CeilLog4(int64_t x)
{
    return (int64_t)std::ceil(std::log(x) / std::log(NUM_FOUR));
}

inline static int64_t VmsLoops(int64_t x)
{
    int64_t srcWsIndex = 0;
    for (int64_t i = 0; x >= 1; i++) {
        x = (x + NUM_FOUR - 1) / NUM_FOUR;
        srcWsIndex = (srcWsIndex + 1) % SORT_WORK_SPACE_NUM;
        if (x == 1) {
            break;
        }
    }
    return srcWsIndex;
}
} // namespace

namespace optiling {
class MoeTokenPermuteWithRoutingMapTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit MoeTokenPermuteWithRoutingMapTilingBase(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeTokenPermuteWithRoutingMapTilingBase() override = default;

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
    void Tiling4IndexCopyCompute();
    void Tiling4MaskedSelect();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4VBSComputeLastdim();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(PermuteVBSComputeRMTilingData* tilingData);
    void Tinlig4VBSMultiCoreComputeLastdim(PermuteVBSComputeRMTilingData* tilingData);
    void Tinlig4VBSOneCoreCompute(PermuteVBSComputeRMTilingData* tilingData);

    int64_t aivNum = 0;
    int64_t realCoreNumAiv = 0;
    int64_t inputDimNum = 0;
    int64_t numTokens = 0;
    int64_t numExperts = 0;
    int64_t numOutTokens = 0;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t tokenBtypeSize = 0;
    int64_t indicesBtypeSize = 0;
    int64_t sortLoopMaxElement = 0;
    int64_t capacity = 0;
    int64_t hasProb = 0;
    int64_t mrgSortListMaxElement = 1024;
    bool paddedMode = false;
    const char* opName = nullptr;
    MoeTokenPermuteWithRoutingMapTilingData moeTokenPermuteWithRoutingMapTilingData;
};

void MoeTokenPermuteWithRoutingMapTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::GetPlatformInfo()
{
    auto indicesPtr = context_->GetInputTensor(1);
    OP_CHECK_IF(
        indicesPtr == nullptr, OP_LOGE(opName, "fail to get input [indices]"),
        return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const MoeTokenPermuteWithRoutingMapCompileInfo*>(context_->GetCompileInfo());

    uint64_t aivNumLocal; // Vector核数量
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        aivNumLocal = compileInfo->aivNum; // Vector核数量
        OP_CHECK_IF(
            compileInfo == nullptr, OP_LOGE(context_, "compile info is null"),
            return ge::GRAPH_FAILED);
        aicoreParams_.ubSize = FloorAlign(compileInfo->ubSize, ONE_BLOCK_BYTE);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivNumLocal = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = FloorAlign(ubSizePlatForm, ONE_BLOCK_BYTE);
    }

    if (indicesPtr->GetShapeSize() <= SORT32_ALIGN_ELEMENT) {
        aivNumLocal = 1;
    } else {
        aivNumLocal = compileInfo->aivNum;
    }
    realCoreNumAiv = compileInfo->aivNum;
    aicoreParams_.blockDim = aivNumLocal;

    moeTokenPermuteWithRoutingMapTilingData.set_coreNum(aivNumLocal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::CheckOutShape()
{
    // 获取输入shape
    const auto tokenOutput = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenOutput);
    const gert::Shape tokensShape = tokenOutput->GetStorageShape();
    const auto indicesOutput = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesOutput);
    const gert::Shape IndicesShape = indicesOutput->GetStorageShape();

    size_t tokensDimNnum = tokensShape.GetDimNum();
    if (tokensDimNnum < DIM_TWO) {
        OP_LOGE(
            context_->GetNodeName(), "The dim number of Output permute_tokens should be greater than 1 but got [%lu].",
            tokensDimNnum);
        return ge::GRAPH_FAILED;
    }

    int64_t cols = 1;
    for (size_t i = 1; i < tokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    size_t IndicesDimNnum = IndicesShape.GetDimNum();
    if (IndicesDimNnum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of Output sort_indices should be 1.");
        return ge::GRAPH_FAILED;
    }

    if (cols != moeTokenPermuteWithRoutingMapTilingData.get_cols() && !paddedMode) {
        OP_LOGE(
            context_->GetNodeName(), "The hidden_size of output permuteTokens should be %ld but got %ld.",
            moeTokenPermuteWithRoutingMapTilingData.get_cols(), cols);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != numOutTokens && !paddedMode) {
        OP_LOGE(
            context_->GetNodeName(), "The dim 0 of output permuteTokens should be %ld but got %ld.", numOutTokens,
            tokensShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "MoeTokenPermuteWithRoutingMap Tiling initing.");

    // 获取输入shape
    const gert::Shape tokensShape = context_->GetInputShape(0)->GetStorageShape();
    const gert::Shape IndicesShape = context_->GetInputShape(1)->GetStorageShape();
    auto probInput = context_->GetOptionalInputTensor(PROB_INDEX);
    hasProb = probInput == nullptr ? 0 : 1;
    moeTokenPermuteWithRoutingMapTilingData.set_hasProb(hasProb);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int64_t id = 0;
    const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(id++);
    const bool* paddedModePtr = attrs->GetAttrPointer<bool>(id);
    numOutTokens = *numOutTokensPtr;
    paddedMode = *paddedModePtr;
    numTokens = tokensShape.GetDim(0);
    numExperts = IndicesShape.GetDim(0);
    if (numExperts < 0) {
        OP_LOGE(context_->GetNodeName(), "Input attr's num_out_tokens [%ld] should  large than 0.", numTokens);
        return ge::GRAPH_FAILED;
    }

    size_t TokensDimNnum = tokensShape.GetDimNum();

    int64_t cols = 1;
    for (size_t i = 1; i < TokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    size_t indicesDimNum = IndicesShape.GetDimNum();
    if (indicesDimNum != DIM_TWO && indicesDimNum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of indices should be 2 or 1 but got [%lu].", indicesDimNum);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != IndicesShape.GetDim(1)) {
        OP_LOGE(
            context_->GetNodeName(), "Input token's dim 0 [%ld] should be same with routingmap's tokennum [%ld].",
            tokensShape.GetDim(0), IndicesShape.GetDim(1));
        return ge::GRAPH_FAILED;
    }

    tokenBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType());
    indicesBtypeSize = ge::GetSizeByDataType(ge::DT_INT32);

    moeTokenPermuteWithRoutingMapTilingData.set_cols(cols);
    auto tokenOneBlockNum = GetDiv(ONE_BLOCK_BYTE, tokenBtypeSize);
    auto colsAlign = GetCeilInt(cols, tokenOneBlockNum) * tokenOneBlockNum;
    moeTokenPermuteWithRoutingMapTilingData.set_n(tokensShape.GetDim(0));
    moeTokenPermuteWithRoutingMapTilingData.set_colsAlign(colsAlign);
    if (numTokens <= 0) {
        OP_LOGE(context_->GetNodeName(), "Input attr's num_out_tokens [%ld] should  large than max 0.", numTokens);
        return ge::GRAPH_FAILED;
    }

    int64_t topK = numOutTokens / numTokens;

    if (topK > MAX_INDICES_NUM) {
        OP_LOGE(
            context_->GetNodeName(), "numOutTokens / numTokens [%ld] should not large than max topK[%ld].", topK,
            MAX_INDICES_NUM);
        return ge::GRAPH_FAILED;
    }

    moeTokenPermuteWithRoutingMapTilingData.set_topK(topK);
    totalLength = moeTokenPermuteWithRoutingMapTilingData.get_n() * moeTokenPermuteWithRoutingMapTilingData.get_topK();

    if (paddedMode == true) {
        capacity = numOutTokens / numExperts;
        totalLength = numTokens;
        numOutTokens = numExperts * capacity;
    } else {
        numOutTokens = totalLength;
    }

    moeTokenPermuteWithRoutingMapTilingData.set_capacity(capacity);

    if (totalLength >= SORT_LIMIT_LENGTH) {
        OP_LOGE(
            context_->GetNodeName(), "The elements num of indices [%ld] should be less than [%ld].", totalLength,
            SORT_LIMIT_LENGTH);
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckOutShape();
    return ret;
}

void MoeTokenPermuteWithRoutingMapTilingBase::ShowTilingData()
{
    OP_LOGD(
        opName,
        "indexCopyCTilingData is needCoreNum:%ld, frontCoreNum:%ld, "
        "tailCoreNum:%ld, coreCalcNum:%ld, coreCalcTail:%ld, oneTokenBtypeSize:%ld, "
        "onceIndicesTokenMoveTimes:%ld, onceUbTokenNums:%ld, onceIndicesTokenNums:%ld, "
        "onceIndices:%ld, oneTokenlastMove:%ld, oneTokenOnceMove:%ld, oneTokenMoveTimes:%ld, "
        "frontCoreLoop:%ld, frontCoreLastTokenNums:%ld, tailCoreLoop:%ld, tailCoreLastTokenNums:%ld, "
        "tailLastonceIndicesTokenMoveTimes:%ld, tailLastIndicesLastTokenNums:%ld, "
        "frontLastonceIndicesTokenMoveTimes:%ld, frontLastIndicesLastTokenNums:%ld, "
        "numOutTokens:%ld, tokenUB:%ld, indicesUB:%ld",
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_needCoreNum(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_frontCoreNum(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_tailCoreNum(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_coreCalcNum(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_coreCalcTail(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_oneTokenBtypeSize(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_onceIndicesTokenMoveTimes(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_onceUbTokenNums(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_onceIndicesTokenNums(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_onceIndices(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_oneTokenlastMove(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_oneTokenOnceMove(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_oneTokenMoveTimes(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_frontCoreLoop(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_frontCoreLastTokenNums(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_tailCoreLoop(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_tailCoreLastTokenNums(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_tailLastonceIndicesTokenMoveTimes(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_tailLastIndicesLastTokenNums(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_frontLastonceIndicesTokenMoveTimes(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_frontLastIndicesLastTokenNums(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_numOutTokens(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_tokenUB(),
        moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp.get_indicesUB());
    OP_LOGD(
        opName,
        "PermuteVBSComputeRMTilingData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
        "perCorePerLoopElements:%ld, "
        "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
        "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld, lastCoreWSindex:%ld",
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_needCoreNum(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_perCoreElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_perCoreLoops(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_lastCoreElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_oneLoopMaxElements(),
        moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp.get_lastCoreWSindex());
    OP_LOGD(
        opName, "PermuteVMSMiddleComputeRMTilingData is needCoreNum:%ld",
        moeTokenPermuteWithRoutingMapTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGD(
        opName, "moeTokenPermuteWithRoutingMapTilingData is coreNum:%ld, n:%ld, cols:%ld, colsAlign:%ld, k:%ld",
        moeTokenPermuteWithRoutingMapTilingData.get_coreNum(), moeTokenPermuteWithRoutingMapTilingData.get_n(),
        moeTokenPermuteWithRoutingMapTilingData.get_cols(), moeTokenPermuteWithRoutingMapTilingData.get_colsAlign(),
        moeTokenPermuteWithRoutingMapTilingData.get_topK());
    OP_LOGD(
        opName, "PermuteSortOutComputeRMTilingData is oneLoopMaxElements:%ld",
        moeTokenPermuteWithRoutingMapTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
}
ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::DoOpTiling()
{
    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    if (paddedMode == false) {
        Tiling4MaskedSelect();
        Tiling4VBSCompute();
        Tiling4VMSMiddleCompute();
        Tiling4SortOutCompute();
        Tiling4IndexCopyCompute();
    } else {
        Tiling4VBSComputeLastdim();
        Tiling4SortOutCompute();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeTokenPermuteWithRoutingMapTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = paddedMode == true ? moeTokenPermuteWithRoutingMapTilingData.get_coreNum() *
                                                        totalLength * sizeof(float) * NUM_TWO * NUM_TWO :
                                                    totalLength * sizeof(float) * NUM_TWO * NUM_TWO; // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        moeTokenPermuteWithRoutingMapTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    moeTokenPermuteWithRoutingMapTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(moeTokenPermuteWithRoutingMapTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tinlig4VBSOneCoreCompute(PermuteVBSComputeRMTilingData* tilingData)
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

void MoeTokenPermuteWithRoutingMapTilingBase::Tinlig4VBSMultiCoreComputeLastdim(
    PermuteVBSComputeRMTilingData* tilingData)
{
    int64_t perCoreElements = numTokens; // 每个核处理的元素数

    int64_t frontCoreNum =
        GetRem(numExperts, realCoreNumAiv) != 0 ? GetRem(numExperts, realCoreNumAiv) : realCoreNumAiv;
    int64_t tailCoreNum = numExperts <= realCoreNumAiv ? 0 : realCoreNumAiv - frontCoreNum;
    int64_t blockDim = frontCoreNum + tailCoreNum;
    aivNum = blockDim;
    int64_t coreCalcNum = GetCeilInt(numExperts, realCoreNumAiv);
    int64_t coreCalcTail = GetDiv(numExperts, realCoreNumAiv);
    tilingData->set_frontcoreTask(coreCalcNum);
    tilingData->set_tailcoreTask(coreCalcTail);
    tilingData->set_frontCoreNum(frontCoreNum);
    tilingData->set_tailCoreNum(tailCoreNum);
    tilingData->set_needCoreNum(blockDim);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_perCoreLoops(
        GetCeilInt(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
    tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

    tilingData->set_perCoreLastLoopElements(
        tilingData->get_perCoreElements() -
        (tilingData->get_perCoreLoops() - 1) * tilingData->get_perCorePerLoopElements());

    tilingData->set_lastCoreElements(
        totalLength - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(GetCeilInt(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCorePerLoopElements(std::min(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCoreLastLoopElements(
        tilingData->get_lastCoreElements() -
        (tilingData->get_lastCoreLoops() - 1) * tilingData->get_lastCorePerLoopElements());
    tilingData->set_lastCoreWSindex(0);
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tinlig4VBSMultiCoreCompute(PermuteVBSComputeRMTilingData* tilingData)
{
    int64_t needCoreNum = GetCeilInt(totalLength, sortLoopMaxElement);      // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum))); // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                            // 不能超过物理核数

    int64_t perCoreElements = GetDiv(totalLength, needCoreNum); // 每个核处理的元素数
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = GetCeilInt(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_perCoreLoops(
        GetCeilInt(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
    tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

    tilingData->set_perCoreLastLoopElements(
        tilingData->get_perCoreElements() -
        (tilingData->get_perCoreLoops() - 1) * tilingData->get_perCorePerLoopElements());

    tilingData->set_lastCoreElements(
        totalLength - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(GetCeilInt(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCorePerLoopElements(std::min(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCoreLastLoopElements(
        tilingData->get_lastCoreElements() -
        (tilingData->get_lastCoreLoops() - 1) * tilingData->get_lastCorePerLoopElements());
    tilingData->set_lastCoreWSindex(
        std::abs(VmsLoops(tilingData->get_lastCoreLoops()) - VmsLoops(tilingData->get_perCoreLoops())));
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4VBSCompute()
{
    if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
        tilingKey_ = SORT_ONE_CORE_MODE;
    } else {
        tilingKey_ = SORT_MULTI_CORE_MODE;
    }

    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (GetTilingKey() == 1UL) { // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}
void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4VBSComputeLastdim()
{
    tilingKey_ = PAD_KEY;
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    Tinlig4VBSMultiCoreComputeLastdim(tilingData);
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeTokenPermuteWithRoutingMapTilingData.vbsComputeParamsOp;
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = GetCeilInt(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}
void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4MaskedSelect()
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.maskedSelectParamsOp;

    uint64_t aivUseNum = realCoreNumAiv;    // Vector核数量
    uint64_t ubSize = aicoreParams_.ubSize; // ubSize大小
    uint64_t totalLengthLocal = numTokens * numExperts;

    uint64_t formerNum = 0;
    uint64_t formerLength = 0;
    uint64_t formerTileNum = 0;
    uint64_t formerTileLength = 0;
    uint64_t formerLastTileLength = 0;

    uint64_t tailNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailTileNum = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailLastTileLength = 0;

    uint64_t blockDim = 0;

    // 求单个元素大小
    uint64_t sizeOfDataType = 2;
    if (hasProb) {
        uint64_t dataType = context_->GetInputDesc(2)->GetDataType();
        switch (dataType) {
            case ge::DT_FLOAT:
            case ge::DT_INT32:
            case ge::DT_UINT32:
                sizeOfDataType = sizeof(int32_t);
                break;
            case ge::DT_DOUBLE:
            case ge::DT_INT64:
            case ge::DT_UINT64:
                sizeOfDataType = sizeof(int64_t);
                break;
            case ge::DT_FLOAT16:
            case ge::DT_BF16:
            case ge::DT_INT16:
            case ge::DT_UINT16:
                sizeOfDataType = sizeof(int16_t);
                break;
            case ge::DT_BOOL:
            case ge::DT_INT8:
            case ge::DT_UINT8:
                sizeOfDataType = sizeof(int8_t);
                break;
            default:
                break;
        }
    }

    // 一个block存放的元素
    uint32_t alignNum = BLOCK_SIZE / NUM_TWO; // 256/<8>=32

    // ub对齐后长度

    uint64_t oneDataSize = IO_QUE * DOUBLE_BUFFER * sizeOfDataType + MASK_ONE_DATA_SIZE + INDEX_ONE_DATA_SIZE;
    uint64_t ubLength = ((ubSize - ONE_BLOCK_BYTE * DOUBLE_BUFFER * DOUBLE_BUFFER) / oneDataSize) / alignNum * alignNum;
    ubLength = static_cast<int64_t>(ubLength) > numTokens ? numTokens : ubLength; // 一次ub能放多少数
    // 运行核数
    blockDim = (numExperts > static_cast<int64_t>(aivUseNum)) ? aivUseNum : numExperts;
    tilingData->set_needCoreNum(blockDim);

    // 切分流程
    formerNum = numExperts % blockDim;
    if (formerNum == 0) {
        formerNum = blockDim;
    }
    tailNum = blockDim - formerNum;

    formerLength = (numExperts + blockDim - 1) / blockDim * numTokens; // 算的多的核需要算多少数
    formerTileNum = (formerLength + ubLength - 1) / ubLength;          // 算的多的核要用多少次ub
    formerTileLength = ubLength;                                       // 算的多的核一次ub能放多少数
    formerLastTileLength = formerLength % ubLength; // 算的多的核最后一次ub需要算多少数
    if (formerLastTileLength == 0) {
        formerLastTileLength = ubLength;
    }

    if (tailNum > 0) {
        tailLength = (totalLengthLocal - formerLength * formerNum) / tailNum; // 一定可能整出
        tailTileNum = (tailLength + ubLength - 1) / ubLength;
        tailTileLength = ubLength;
        tailLastTileLength = tailLength % ubLength;
        if (tailLastTileLength == 0) {
            tailLastTileLength = ubLength;
        }
    }
    aivNum = std::max(aivNum, static_cast<int64_t>(blockDim));
    tilingData->set_formerNum(formerNum);
    tilingData->set_formerLength(formerLength);
    tilingData->set_formertileNum(formerTileNum);
    tilingData->set_formertileLength(formerTileLength);
    tilingData->set_formerlasttileLength(formerLastTileLength);
    tilingData->set_tokenNum(numTokens);

    tilingData->set_tailNum(tailNum);
    tilingData->set_tailLength(tailLength);
    tilingData->set_tailtileNum(tailTileNum);
    tilingData->set_tailtileLength(tailTileLength);
    tilingData->set_taillasttileLength(tailLastTileLength);
}
void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4IndexCopyCompute()
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp;
    int64_t tokenNums = moeTokenPermuteWithRoutingMapTilingData.get_n();
    int64_t topK = moeTokenPermuteWithRoutingMapTilingData.get_topK();
    int64_t cols = moeTokenPermuteWithRoutingMapTilingData.get_cols();

    tilingData->set_numOutTokens(numOutTokens);

    int64_t frontCoreNum = GetRem(tokenNums, realCoreNumAiv) != 0 ? GetRem(tokenNums, realCoreNumAiv) : realCoreNumAiv;
    int64_t tailCoreNum = tokenNums <= realCoreNumAiv ? 0 : realCoreNumAiv - frontCoreNum;
    int64_t blockDim = frontCoreNum + tailCoreNum;
    int64_t coreCalcNum = GetCeilInt(tokenNums, realCoreNumAiv);
    int64_t coreCalcTail = GetDiv(tokenNums, realCoreNumAiv);

    int64_t ubLeft = aicoreParams_.ubSize - MAX_INDICES_NUM * INT32_DTYPE_SIZE;
    int64_t oneTokenBtypeSize = cols * tokenBtypeSize;

    int64_t oneTokenBtypeSizeAlign32 = UpAlign(oneTokenBtypeSize, ONE_BLOCK_BYTE);

    int64_t oneTokenlastMove = 1;
    int64_t oneTokenOnceMove = 1;
    int64_t oneTokenMoveTimes = 1;
    int64_t onceIndicesTokenMoveTimes = 1;
    ;
    int64_t onceUbTokenNums = 1;
    ;
    int64_t onceIndicesTokenNums = 1;
    ;
    int64_t onceIndices = 1;
    int64_t tokenUB = 1;
    int64_t indicesUB = 1;
    if (ubLeft >= BUFFER_NUM * oneTokenBtypeSizeAlign32) {
        onceUbTokenNums = GetDiv(
            static_cast<int64_t>(aicoreParams_.ubSize),
            oneTokenBtypeSizeAlign32 * BUFFER_NUM + topK * BUFFER_NUM * INT32_DTYPE_SIZE);
        onceUbTokenNums = std::min(onceUbTokenNums, MAX_BLOCK_COUNT);
        int64_t TopKUbLeft = aicoreParams_.ubSize - onceUbTokenNums * oneTokenBtypeSizeAlign32 * BUFFER_NUM;
        onceIndicesTokenMoveTimes = GetDiv(TopKUbLeft, onceUbTokenNums * topK * INT32_DTYPE_SIZE);
        onceIndicesTokenNums = onceIndicesTokenMoveTimes * onceUbTokenNums;
        onceIndices = onceIndicesTokenNums * topK;
        tokenUB = onceUbTokenNums * oneTokenBtypeSizeAlign32;
        indicesUB = UpAlign(onceIndices, ONE_BLOCK_BYTE);
    } else {
        onceIndicesTokenNums = GetDiv(MAX_INDICES_NUM, topK);
        onceIndices = onceIndicesTokenNums * topK;
        oneTokenOnceMove = GetDiv(FloorAlign(GetDiv(ubLeft, BUFFER_NUM), DATA_MOVE_ALIGN), tokenBtypeSize);
        oneTokenMoveTimes = GetCeilInt(cols, oneTokenOnceMove);
        oneTokenlastMove = cols - (oneTokenMoveTimes - 1) * oneTokenOnceMove;
        tilingKey_ = tilingKey_ + SPLIT_D_MODE;
        tokenUB = oneTokenOnceMove * tokenBtypeSize;
        indicesUB = MAX_INDICES_NUM * INT32_DTYPE_SIZE;
    }

    int64_t frontCoreLoop = GetCeilInt(coreCalcNum, onceIndicesTokenNums);
    int64_t frontCoreLastTokenNums = coreCalcNum - (frontCoreLoop - 1) * onceIndicesTokenNums;
    int64_t tailCoreLoop = GetCeilInt(coreCalcTail, onceIndicesTokenNums);
    int64_t tailCoreLastTokenNums = coreCalcTail - (tailCoreLoop - 1) * onceIndicesTokenNums;
    int64_t tailLastonceIndicesTokenMoveTimes = GetCeilInt(tailCoreLastTokenNums, onceUbTokenNums);
    int64_t tailLastIndicesLastTokenNums =
        tailCoreLastTokenNums - (tailLastonceIndicesTokenMoveTimes - 1) * onceUbTokenNums;
    int64_t frontLastonceIndicesTokenMoveTimes = GetCeilInt(frontCoreLastTokenNums, onceUbTokenNums);

    int64_t frontLastIndicesLastTokenNums =
        frontCoreLastTokenNums - (frontLastonceIndicesTokenMoveTimes - 1) * onceUbTokenNums;
    tilingData->set_tokenUB(tokenUB);
    tilingData->set_indicesUB(indicesUB);
    tilingData->set_needCoreNum(blockDim);
    tilingData->set_frontCoreNum(frontCoreNum);
    tilingData->set_tailCoreNum(tailCoreNum);
    tilingData->set_coreCalcNum(coreCalcNum);
    tilingData->set_coreCalcTail(coreCalcTail);
    tilingData->set_oneTokenBtypeSize(oneTokenBtypeSize);
    tilingData->set_onceIndicesTokenMoveTimes(onceIndicesTokenMoveTimes);
    tilingData->set_onceUbTokenNums(onceUbTokenNums);
    tilingData->set_onceIndicesTokenNums(onceIndicesTokenNums);
    tilingData->set_onceIndices(onceIndices);
    tilingData->set_oneTokenlastMove(oneTokenlastMove);
    tilingData->set_oneTokenOnceMove(oneTokenOnceMove);
    tilingData->set_oneTokenMoveTimes(oneTokenMoveTimes);
    tilingData->set_frontCoreLoop(frontCoreLoop);
    tilingData->set_frontCoreLastTokenNums(frontCoreLastTokenNums);
    tilingData->set_tailCoreLoop(tailCoreLoop);
    tilingData->set_tailCoreLastTokenNums(tailCoreLastTokenNums);
    tilingData->set_tailLastonceIndicesTokenMoveTimes(tailLastonceIndicesTokenMoveTimes);
    tilingData->set_tailLastIndicesLastTokenNums(tailLastIndicesLastTokenNums);
    tilingData->set_frontLastonceIndicesTokenMoveTimes(frontLastonceIndicesTokenMoveTimes);
    tilingData->set_frontLastIndicesLastTokenNums(frontLastIndicesLastTokenNums);
    aivNum = std::max(aivNum, blockDim);
}

static ge::graphStatus TilingForMoeTokenPermuteWithRoutingMap(gert::TilingContext* context)
{
    MoeTokenPermuteWithRoutingMapTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMoeTokenPermuteWithRoutingMap(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForMoeTokenPermuteWithRoutingMap start.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto compileInfo = context->GetCompiledInfo<MoeTokenPermuteWithRoutingMapCompileInfo>();

    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context->GetNodeName(), "compileInfo->aivNum is %lu.", compileInfo->aivNum);

    compileInfo->workSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "compileInfo->workSpaceSize is %lu.", compileInfo->workSpaceSize);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_LOGD(context->GetNodeName(), "compileInfo->ubSize is %lu.", compileInfo->ubSize);

    OP_LOGD(context->GetNodeName(), "TilingPrepareForMoeTokenPermuteWithRoutingMap end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTokenPermuteWithRoutingMap)
    .Tiling(TilingForMoeTokenPermuteWithRoutingMap)
    .TilingParse<MoeTokenPermuteWithRoutingMapCompileInfo>(TilingPrepareForMoeTokenPermuteWithRoutingMap);
} // namespace optiling
