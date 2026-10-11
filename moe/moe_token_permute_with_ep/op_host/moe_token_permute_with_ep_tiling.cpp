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
 * \file moe_token_permute_with_ep_tiling.cpp
 * \brief
 */
#include "moe_token_permute_with_ep_tiling.h"
#include "op_host/tiling_base.h"

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
const static int64_t USED_SIZE = 96; // prob outsize + prob align + index align
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
const static uint64_t SPILT_D_MODE = 2L;
const static int64_t SORT_LIMIT_LENGTH = 16777215;
const static int64_t SORT_WORK_SPACE_NUM = 2;
const static int64_t PROBS_SPACE_SCALE = 2;
const static int64_t RANGE_SIZE = 2;
const static int64_t PERMUTE_WITH_EP_INPUT_TOKENS = 0;
const static int64_t PERMUTE_WITH_EP_INPUT_IDX = 1;
const static int64_t PERMUTE_WITH_EP_INPUT_PROBS = 2;
const static int64_t PERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
const static int64_t PERMUTE_WITH_EP_OUTPUT_IDX = 1;
const static int64_t PERMUTE_WITH_EP_OUTPUT_PROBS = 2;
const static int64_t PERMUTE_WITH_EP_ARRT_RANGE = 0;
const static int64_t PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS = 1;
const static int64_t PERMUTE_WITH_EP_ARRT_PADDED_MODE = 2;

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
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
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
class MoeTokenPermuteWithEpTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    struct CoreParams {
        int64_t frontCoreNum = 1;
        int64_t tailCoreNum = 1;
        int64_t numBlocks = 1;
        int64_t coreCalcNum = 1;
        int64_t coreCalcTail = 1;
    };
    struct UBParams {
        int64_t ubLeft = 1;
        int64_t oneTokenBtypeSize = 1;
        int64_t oneTokenBtypeSizeAlign32 = 1;
        int64_t oneTokenlastMove = 1;
        int64_t oneTokenOnceMove = 1;
        int64_t oneTokenMoveTimes = 1;
        int64_t onceIndicesTokenMoveTimes = 1;
        int64_t onceUbTokenNums = 1;
        int64_t onceIndicesTokenNums = 1;
        int64_t onceIndices = 1;
        int64_t tokenUB = 1;
        int64_t indicesUB = 1;
        int64_t probsUB = 1;
        int64_t oneProbBtypeSize = 1;
    };

    struct LoopParams {
        int64_t frontCoreLoop = 1;
        int64_t frontCoreLastTokenNums = 1;
        int64_t tailCoreLoop = 1;
        int64_t tailCoreLastTokenNums = 1;
        int64_t tailLastonceIndicesTokenMoveTimes = 1;
        int64_t tailLastIndicesLastTokenNums = 1;
        int64_t frontLastonceIndicesTokenMoveTimes = 1;
        int64_t frontLastIndicesLastTokenNums = 1;
    };

public:
    explicit MoeTokenPermuteWithEpTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeTokenPermuteWithEpTilingBase() override = default;

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
    ge::graphStatus CheckAndGetAttrsInfo();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttrsInfosAndInputShape();
    ge::graphStatus CheckOutputIndicesAndToken(const gert::Shape indicesShape, int64_t cols,
                                               const gert::Shape tokensShape);
    ge::graphStatus CheckOutputShape();
    void Tiling4IndexCopyCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void ShowIndexCopyCTilingData();
    void ShowVBSComputeTilingEPData();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(PermuteVBSComputeTilingEPData* tilingData);
    void Tinlig4VBSOneCoreCompute(PermuteVBSComputeTilingEPData* tilingData);
    void CalculateCoreParams(CoreParams& coreParams, int64_t tokenNums);
    void CalculateUBParams(UBParams& ubParams, int64_t topK, int64_t cols);
    void CalculateLoopParams(LoopParams& loopParams, const CoreParams& coreParams, int64_t onceIndicesTokenNums,
                             int64_t onceUbTokenNums) const;
    void SetTilingDataFinalParams(IndexMixCopyComputeTilingData* tilingData, const CoreParams& coreParams,
                                  const UBParams& ubParams, const LoopParams& loopParams) const;
    int64_t aivNum = 0;
    int64_t realCoreNumAiv = 0;
    int64_t inputDimNum = 0;
    int64_t numOutTokens = 0;
    int64_t start = 0;
    int64_t end = 0;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t tokenBtypeSize = 0;
    int64_t indicesBtypeSize = 0;
    int64_t probsBtypeSize = 0;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1024;
    bool paddedMode = false;
    const char* opName = nullptr;
    MoeTokenPermuteWithEpTilingData moeTokenPermuteWithEpTilingData;
};

void MoeTokenPermuteWithEpTilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(opName, "fail to get platform info");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto indicesPtr = context_->GetInputTensor(1);
    if (indicesPtr == nullptr) {
        OP_LOGE(opName, "fail to get input [indices]");
        return ge::GRAPH_FAILED;
    }
    if (indicesPtr->GetShapeSize() <= SORT32_ALIGN_ELEMENT) {
        aivNum = 1;
    } else {
        aivNum = ascendcPlatform.GetCoreNumAiv();
    }
    realCoreNumAiv = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.numBlocks = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = FloorAlign(ubSizePlatForm, ONE_BLOCK_BYTE);
    moeTokenPermuteWithEpTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::CheckOutputIndicesAndToken(const gert::Shape indicesShape,
                                                                            int64_t cols, const gert::Shape tokensShape)
{
    size_t IndicesDimNnum = indicesShape.GetDimNum();
    if (IndicesDimNnum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of Output sort_indices should be 1.");
        return ge::GRAPH_FAILED;
    }

    if (indicesShape.GetDim(0) != totalLength) {
        OP_LOGE(context_->GetNodeName(), "The size of Output sort_indices should be [%ld].", totalLength);
        return ge::GRAPH_FAILED;
    }

    if (cols != moeTokenPermuteWithEpTilingData.get_cols()) {
        OP_LOGE(context_->GetNodeName(), "The hidden_size of output permuteTokens should be %ld but got %ld.",
                moeTokenPermuteWithEpTilingData.get_cols(), cols);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != numOutTokens) {
        OP_LOGE(context_->GetNodeName(), "The dim 0 of output permuteTokens should be %ld but got %ld.", numOutTokens,
                tokensShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::CheckOutputShape()
{
    // 获取输入shape
    const auto tokenOutput = context_->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenOutput);
    const gert::Shape tokensShape = tokenOutput->GetStorageShape();
    const auto indicesOutput = context_->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesOutput);
    const gert::Shape IndicesShape = indicesOutput->GetStorageShape();

    size_t tokensDimNnum = tokensShape.GetDimNum();
    if (tokensDimNnum < DIM_TWO) {
        OP_LOGE(context_->GetNodeName(),
                "The dim number of Output permute_tokens should be greater than 1 but got [%lu].", tokensDimNnum);
        return ge::GRAPH_FAILED;
    }

    int64_t cols = 1;
    for (size_t i = 1; i < tokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    auto ret = CheckOutputIndicesAndToken(IndicesShape, cols, tokensShape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto probsTensor = context_->GetOptionalInputTensor(PERMUTE_WITH_EP_OUTPUT_PROBS);
    if (probsTensor != nullptr) {
        const auto probsOutput = context_->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_PROBS);
        OP_CHECK_NULL_WITH_CONTEXT(context_, probsOutput);
        const gert::Shape probsShape = probsOutput->GetStorageShape();
        size_t probsDimNum = probsShape.GetDimNum();
        if (probsDimNum != DIM_ONE) {
            OP_LOGE(context_->GetNodeName(), "The dim number of output permuteProbs should be 1.");
            return ge::GRAPH_FAILED;
        }

        if (probsShape.GetDim(0) != numOutTokens) {
            OP_LOGE(context_->GetNodeName(), "The size of output permuteProbs should be [%ld] but got [%ld].",
                    numOutTokens, probsShape.GetDim(0));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::CheckAndGetAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto rangePtr = attrs->GetAttrPointer<gert::ContinuousVector>(PERMUTE_WITH_EP_ARRT_RANGE);
    const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS);
    const bool* paddedModePtr = attrs->GetAttrPointer<bool>(PERMUTE_WITH_EP_ARRT_PADDED_MODE);

    if (rangePtr != nullptr) {
        if (rangePtr->GetSize() != RANGE_SIZE) {
            OP_LOGE(context_->GetNodeName(), "the size of range only support 2");
            return ge::GRAPH_FAILED;
        }
        const int64_t* rangeList = reinterpret_cast<const int64_t*>(rangePtr->GetData());
        start = rangeList[0];
        end = rangeList[1];
    } else {
        start = 0;
        end = *numOutTokensPtr;
    }
    paddedMode = *paddedModePtr;
    if (paddedMode == true) {
        OP_LOGE(context_->GetNodeName(), "Currently only support padded_mode is false");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::CheckInputShape()
{
    const gert::Shape tokensShape = context_->GetInputShape(PERMUTE_WITH_EP_INPUT_TOKENS)->GetStorageShape();
    const gert::Shape IndicesShape = context_->GetInputShape(PERMUTE_WITH_EP_INPUT_IDX)->GetStorageShape();
    auto probsTensor = context_->GetOptionalInputTensor(PERMUTE_WITH_EP_INPUT_PROBS);
    const gert::StorageShape* probsStorageShapePtr =
        (probsTensor == nullptr) ? nullptr : context_->GetOptionalInputShape(PERMUTE_WITH_EP_INPUT_PROBS);
    size_t indicesDimNum = IndicesShape.GetDimNum();
    if (indicesDimNum != DIM_TWO && indicesDimNum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of indices should be 2 or 1 but got [%lu].", indicesDimNum);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != IndicesShape.GetDim(0)) {
        OP_LOGE(context_->GetNodeName(), "Input token's dim 0 [%ld] should be same with indices' dim 0 [%ld].",
                tokensShape.GetDim(0), IndicesShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    if (probsTensor != nullptr) {
        const gert::Shape probsShape = probsStorageShapePtr->GetStorageShape();
        size_t probsDimNum = probsShape.GetDimNum();
        if (probsDimNum != DIM_TWO && probsDimNum != DIM_ONE) {
            OP_LOGE(context_->GetNodeName(), "The dim number of probs should be 2 or 1 but got [%lu].", probsDimNum);
            return ge::GRAPH_FAILED;
        }

        if (indicesDimNum != probsDimNum) {
            OP_LOGE(context_->GetNodeName(), "The dim number of probs [%zu] should be equal to indices [%lu].",
                    probsDimNum, indicesDimNum);
            return ge::GRAPH_FAILED;
        }

        if (tokensShape.GetDim(0) != probsShape.GetDim(0)) {
            OP_LOGE(context_->GetNodeName(), "Input token's dim 0 [%ld] should be same with probs' dim 0 [%ld].",
                    tokensShape.GetDim(0), probsShape.GetDim(0));
            return ge::GRAPH_FAILED;
        }

        if ((probsDimNum == DIM_TWO) && (IndicesShape.GetDim(1) != probsShape.GetDim(1))) {
            OP_LOGE(context_->GetNodeName(), "Input indices' dim 1 [%ld] should be same with probs' dim 1 [%ld].",
                    IndicesShape.GetDim(1), probsShape.GetDim(1));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::CheckAttrsInfosAndInputShape()
{
    auto ret = CheckAndGetAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "MoeTokenPermuteWithEp Tiling initing.");

    auto ret = CheckAttrsInfosAndInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape tokensShape = context_->GetInputShape(PERMUTE_WITH_EP_INPUT_TOKENS)->GetStorageShape();
    const gert::Shape IndicesShape = context_->GetInputShape(PERMUTE_WITH_EP_INPUT_IDX)->GetStorageShape();
    auto probsTensor = context_->GetOptionalInputTensor(PERMUTE_WITH_EP_INPUT_PROBS);

    tokenBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(PERMUTE_WITH_EP_INPUT_TOKENS)->GetDataType());
    indicesBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(PERMUTE_WITH_EP_INPUT_IDX)->GetDataType());
    probsBtypeSize = (probsTensor == nullptr) ?
                         0 :
                         ge::GetSizeByDataType(context_->GetInputDesc(PERMUTE_WITH_EP_INPUT_PROBS)->GetDataType());

    size_t TokensDimNnum = tokensShape.GetDimNum();
    int64_t cols = 1;
    for (size_t i = 1; i < TokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    moeTokenPermuteWithEpTilingData.set_cols(cols);
    auto tokenOneBlockNum = GetDiv(ONE_BLOCK_BYTE, tokenBtypeSize);
    auto colsAlign = GetCeilInt(cols, tokenOneBlockNum) * tokenOneBlockNum;
    moeTokenPermuteWithEpTilingData.set_n(IndicesShape.GetDim(0));
    moeTokenPermuteWithEpTilingData.set_colsAlign(colsAlign);

    int64_t topK = (IndicesShape.GetDimNum() == 1) ? 1 : IndicesShape.GetDim(1);
    if (topK > MAX_INDICES_NUM || topK < 1) {
        OP_LOGE(context_->GetNodeName(), "Input indices's dim 1 [%ld] should be in (0, [%ld]].", topK, MAX_INDICES_NUM);
        return ge::GRAPH_FAILED;
    }

    moeTokenPermuteWithEpTilingData.set_topK(topK);
    totalLength = moeTokenPermuteWithEpTilingData.get_n() * moeTokenPermuteWithEpTilingData.get_topK();
    if (totalLength >= SORT_LIMIT_LENGTH) {
        OP_LOGE(context_->GetNodeName(), "The elements num of indices [%ld] should be less than [%ld].", totalLength,
                SORT_LIMIT_LENGTH);
        return ge::GRAPH_FAILED;
    }

    if (start < 0 || start > totalLength || end < 0 || end > totalLength) {
        OP_LOGE(context_->GetNodeName(), "The start [%ld] and end [%ld] should be in [0, %ld].", start, end,
                totalLength);
        return ge::GRAPH_FAILED;
    }

    numOutTokens = end - start;
    numOutTokens = (numOutTokens <= 0) ? numOutTokens + totalLength : numOutTokens;
    numOutTokens = std::min(numOutTokens, totalLength);
    numOutTokens = std::max(numOutTokens, static_cast<int64_t>(0));

    return CheckOutputShape();
}

void MoeTokenPermuteWithEpTilingBase::ShowIndexCopyCTilingData()
{
    OP_LOGD(opName,
            "indexCopyCTilingData is needCoreNum:%ld, frontCoreNum:%ld, "
            "tailCoreNum:%ld, coreCalcNum:%ld, coreCalcTail:%ld, oneTokenBtypeSize:%ld, "
            "onceIndicesTokenMoveTimes:%ld, onceUbTokenNums:%ld, onceIndicesTokenNums:%ld, "
            "onceIndices:%ld, oneTokenlastMove:%ld, oneTokenOnceMove:%ld, oneTokenMoveTimes:%ld, "
            "frontCoreLoop:%ld, frontCoreLastTokenNums:%ld, tailCoreLoop:%ld, tailCoreLastTokenNums:%ld, "
            "tailLastonceIndicesTokenMoveTimes:%ld, tailLastIndicesLastTokenNums:%ld, "
            "frontLastonceIndicesTokenMoveTimes:%ld, frontLastIndicesLastTokenNums:%ld, "
            "numOutTokens:%ld, start:%ld, end:%ld, tokenUB:%ld, indicesUB:%ld, probsUB:%ld",
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_needCoreNum(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_frontCoreNum(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_tailCoreNum(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_coreCalcNum(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_coreCalcTail(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_oneTokenBtypeSize(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_onceIndicesTokenMoveTimes(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_onceUbTokenNums(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_onceIndicesTokenNums(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_onceIndices(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_oneTokenlastMove(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_oneTokenOnceMove(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_oneTokenMoveTimes(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_frontCoreLoop(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_frontCoreLastTokenNums(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_tailCoreLoop(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_tailCoreLastTokenNums(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_tailLastonceIndicesTokenMoveTimes(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_tailLastIndicesLastTokenNums(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_frontLastonceIndicesTokenMoveTimes(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_frontLastIndicesLastTokenNums(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_numOutTokens(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_start(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_end(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_tokenUB(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_indicesUB(),
            moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp.get_probsUB());
}

void MoeTokenPermuteWithEpTilingBase::ShowVBSComputeTilingEPData()
{
    OP_LOGD(opName,
            "PermuteVBSComputeTilingEPData is needCoreNum:%ld, perCoreElements:%ld, perCoreLoops:%ld, "
            "perCorePerLoopElements:%ld, "
            "perCoreLastLoopElements:%ld, lastCoreElements:%ld, lastCoreLoops:%ld, lastCorePerLoopElements:%ld, "
            "lastCoreLastLoopElements:%ld, oneLoopMaxElements:%ld, lastCoreWSindex:%ld",
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_needCoreNum(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_perCoreElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_perCoreLoops(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_perCorePerLoopElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_perCoreLastLoopElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_lastCoreElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_lastCoreLoops(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_lastCorePerLoopElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_lastCoreLastLoopElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_oneLoopMaxElements(),
            moeTokenPermuteWithEpTilingData.vbsComputeParamsOp.get_lastCoreWSindex());
}
void MoeTokenPermuteWithEpTilingBase::ShowTilingData()
{
    ShowIndexCopyCTilingData();
    ShowVBSComputeTilingEPData();
    OP_LOGD(opName, "PermuteVMSMiddleComputeTilingEPData is needCoreNum:%ld",
            moeTokenPermuteWithEpTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGD(opName, "moeTokenPermuteWithEpTilingData is coreNum:%ld, n:%ld, cols:%ld, colsAlign:%ld, k:%ld",
            moeTokenPermuteWithEpTilingData.get_coreNum(), moeTokenPermuteWithEpTilingData.get_n(),
            moeTokenPermuteWithEpTilingData.get_cols(), moeTokenPermuteWithEpTilingData.get_colsAlign(),
            moeTokenPermuteWithEpTilingData.get_topK());
    OP_LOGD(opName, "PermuteSortOutComputeTilingEPData is oneLoopMaxElements:%ld",
            moeTokenPermuteWithEpTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
}
ge::graphStatus MoeTokenPermuteWithEpTilingBase::DoOpTiling()
{
    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4IndexCopyCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeTokenPermuteWithEpTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = sizeof(float) * static_cast<size_t>(totalLength * NUM_TWO * NUM_TWO); // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        moeTokenPermuteWithEpTilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithEpTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    context_->SetScheduleMode(1);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    moeTokenPermuteWithEpTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                                 context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeTokenPermuteWithEpTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeTokenPermuteWithEpTilingBase::Tinlig4VBSOneCoreCompute(PermuteVBSComputeTilingEPData* tilingData)
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

void MoeTokenPermuteWithEpTilingBase::Tinlig4VBSMultiCoreCompute(PermuteVBSComputeTilingEPData* tilingData)
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

    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                            (tilingData->get_perCoreLoops() - 1) *
                                                tilingData->get_perCorePerLoopElements());

    tilingData->set_lastCoreElements(totalLength -
                                     (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(GetCeilInt(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCorePerLoopElements(std::min(tilingData->get_lastCoreElements(), sortLoopMaxElement));
    tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                             (tilingData->get_lastCoreLoops() - 1) *
                                                 tilingData->get_lastCorePerLoopElements());
    tilingData->set_lastCoreWSindex(
        std::abs(VmsLoops(tilingData->get_lastCoreLoops()) - VmsLoops(tilingData->get_perCoreLoops())));
}

void MoeTokenPermuteWithEpTilingBase::Tiling4VBSCompute()
{
    if (totalLength <= sortLoopMaxElement) { // 排序只用到一个核排序
        tilingKey_ = SORT_ONE_CORE_MODE;
    } else {
        tilingKey_ = SORT_MULTI_CORE_MODE;
    }

    auto tilingData = &moeTokenPermuteWithEpTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (GetTilingKey() == 1UL) { // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void MoeTokenPermuteWithEpTilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeTokenPermuteWithEpTilingData.vbsComputeParamsOp;
    auto tilingData = &moeTokenPermuteWithEpTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = GetCeilInt(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeTokenPermuteWithEpTilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeTokenPermuteWithEpTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeTokenPermuteWithEpTilingBase::CalculateCoreParams(MoeTokenPermuteWithEpTilingBase::CoreParams& coreParams,
                                                          int64_t tokenNums)
{
    coreParams.frontCoreNum =
        GetRem(tokenNums, realCoreNumAiv) != 0 ? GetRem(tokenNums, realCoreNumAiv) : realCoreNumAiv;
    coreParams.tailCoreNum = tokenNums <= realCoreNumAiv ? 0 : realCoreNumAiv - coreParams.frontCoreNum;
    coreParams.numBlocks = coreParams.frontCoreNum + coreParams.tailCoreNum;
    coreParams.coreCalcNum = GetCeilInt(tokenNums, realCoreNumAiv);
    coreParams.coreCalcTail = GetDiv(tokenNums, realCoreNumAiv);
}

void MoeTokenPermuteWithEpTilingBase::CalculateUBParams(MoeTokenPermuteWithEpTilingBase::UBParams& ubParams,
                                                        int64_t topK, int64_t cols)
{
    ubParams.ubLeft = aicoreParams_.ubSize - PROBS_SPACE_SCALE * MAX_INDICES_NUM * INT32_DTYPE_SIZE - ONE_BLOCK_BYTE;
    ubParams.oneTokenBtypeSize = cols * tokenBtypeSize;
    ubParams.oneProbBtypeSize = probsBtypeSize;
    ubParams.oneTokenBtypeSizeAlign32 = UpAlign(ubParams.oneTokenBtypeSize, ONE_BLOCK_BYTE);

    if (ubParams.ubLeft >= BUFFER_NUM * ubParams.oneTokenBtypeSizeAlign32) {
        ubParams.onceUbTokenNums = GetDiv(
            static_cast<int64_t>(aicoreParams_.ubSize - USED_SIZE),
            ubParams.oneTokenBtypeSizeAlign32 * BUFFER_NUM + PROBS_SPACE_SCALE * topK * BUFFER_NUM * INT32_DTYPE_SIZE);
        ubParams.onceUbTokenNums = std::min(ubParams.onceUbTokenNums, MAX_BLOCK_COUNT);

        int64_t TopKUbLeft = aicoreParams_.ubSize - USED_SIZE -
                             ubParams.onceUbTokenNums * ubParams.oneTokenBtypeSizeAlign32 * BUFFER_NUM;
        /* Probs的UB按照indices的UB计算, 这里乘以2*/
        ubParams.onceIndicesTokenMoveTimes =
            GetDiv(TopKUbLeft, PROBS_SPACE_SCALE * ubParams.onceUbTokenNums * topK * INT32_DTYPE_SIZE);

        ubParams.onceIndicesTokenNums = ubParams.onceIndicesTokenMoveTimes * ubParams.onceUbTokenNums;
        ubParams.onceIndices = ubParams.onceIndicesTokenNums * topK;
        ubParams.tokenUB = ubParams.onceUbTokenNums * ubParams.oneTokenBtypeSizeAlign32;
        ubParams.indicesUB = UpAlign(ubParams.onceIndices * INT32_DTYPE_SIZE, ONE_BLOCK_BYTE);
        ubParams.probsUB = ubParams.indicesUB;
    } else {
        ubParams.onceIndicesTokenNums = GetDiv(MAX_INDICES_NUM, topK);
        ubParams.onceIndices = ubParams.onceIndicesTokenNums * topK;
        ubParams.oneTokenOnceMove =
            GetDiv(FloorAlign(GetDiv(ubParams.ubLeft, BUFFER_NUM), DATA_MOVE_ALIGN), tokenBtypeSize);
        ubParams.oneTokenMoveTimes = GetCeilInt(cols, ubParams.oneTokenOnceMove);
        ubParams.oneTokenlastMove = cols - (ubParams.oneTokenMoveTimes - 1) * ubParams.oneTokenOnceMove;

        tilingKey_ = tilingKey_ + SPILT_D_MODE;
        ubParams.tokenUB = ubParams.oneTokenOnceMove * tokenBtypeSize;
        ubParams.indicesUB = MAX_INDICES_NUM * INT32_DTYPE_SIZE;
        ubParams.probsUB = ubParams.indicesUB;
    }
}

void MoeTokenPermuteWithEpTilingBase::CalculateLoopParams(MoeTokenPermuteWithEpTilingBase::LoopParams& loopParams,
                                                          const MoeTokenPermuteWithEpTilingBase::CoreParams& coreParams,
                                                          int64_t onceIndicesTokenNums, int64_t onceUbTokenNums) const
{
    loopParams.frontCoreLoop = GetCeilInt(coreParams.coreCalcNum, onceIndicesTokenNums);
    loopParams.frontCoreLastTokenNums = coreParams.coreCalcNum - (loopParams.frontCoreLoop - 1) * onceIndicesTokenNums;
    loopParams.tailCoreLoop = GetCeilInt(coreParams.coreCalcTail, onceIndicesTokenNums);
    loopParams.tailCoreLastTokenNums = coreParams.coreCalcTail - (loopParams.tailCoreLoop - 1) * onceIndicesTokenNums;
    loopParams.tailLastonceIndicesTokenMoveTimes = GetCeilInt(loopParams.tailCoreLastTokenNums, onceUbTokenNums);
    loopParams.tailLastIndicesLastTokenNums =
        loopParams.tailCoreLastTokenNums - (loopParams.tailLastonceIndicesTokenMoveTimes - 1) * onceUbTokenNums;
    loopParams.frontLastonceIndicesTokenMoveTimes = GetCeilInt(loopParams.frontCoreLastTokenNums, onceUbTokenNums);

    loopParams.frontLastIndicesLastTokenNums =
        loopParams.frontCoreLastTokenNums - (loopParams.frontLastonceIndicesTokenMoveTimes - 1) * onceUbTokenNums;
}

void MoeTokenPermuteWithEpTilingBase::SetTilingDataFinalParams(
    IndexMixCopyComputeTilingData* tilingData, const MoeTokenPermuteWithEpTilingBase::CoreParams& coreParams,
    const MoeTokenPermuteWithEpTilingBase::UBParams& ubParams,
    const MoeTokenPermuteWithEpTilingBase::LoopParams& loopParams) const
{
    tilingData->set_tokenUB(ubParams.tokenUB);
    tilingData->set_indicesUB(ubParams.indicesUB);
    tilingData->set_probsUB(ubParams.probsUB);

    tilingData->set_needCoreNum(coreParams.numBlocks);
    tilingData->set_frontCoreNum(coreParams.frontCoreNum);
    tilingData->set_tailCoreNum(coreParams.tailCoreNum);
    tilingData->set_coreCalcNum(coreParams.coreCalcNum);
    tilingData->set_coreCalcTail(coreParams.coreCalcTail);

    tilingData->set_oneTokenBtypeSize(ubParams.oneTokenBtypeSize);
    tilingData->set_oneProbBtypeSize(ubParams.oneProbBtypeSize);
    tilingData->set_onceIndicesTokenMoveTimes(ubParams.onceIndicesTokenMoveTimes);
    tilingData->set_onceUbTokenNums(ubParams.onceUbTokenNums);
    tilingData->set_onceIndicesTokenNums(ubParams.onceIndicesTokenNums);
    tilingData->set_onceIndices(ubParams.onceIndices);
    tilingData->set_oneTokenlastMove(ubParams.oneTokenlastMove);
    tilingData->set_oneTokenOnceMove(ubParams.oneTokenOnceMove);
    tilingData->set_oneTokenMoveTimes(ubParams.oneTokenMoveTimes);

    tilingData->set_frontCoreLoop(loopParams.frontCoreLoop);
    tilingData->set_frontCoreLastTokenNums(loopParams.frontCoreLastTokenNums);
    tilingData->set_tailCoreLoop(loopParams.tailCoreLoop);
    tilingData->set_tailCoreLastTokenNums(loopParams.tailCoreLastTokenNums);
    tilingData->set_tailLastonceIndicesTokenMoveTimes(loopParams.tailLastonceIndicesTokenMoveTimes);
    tilingData->set_tailLastIndicesLastTokenNums(loopParams.tailLastIndicesLastTokenNums);
    tilingData->set_frontLastonceIndicesTokenMoveTimes(loopParams.frontLastonceIndicesTokenMoveTimes);
    tilingData->set_frontLastIndicesLastTokenNums(loopParams.frontLastIndicesLastTokenNums);
}

void MoeTokenPermuteWithEpTilingBase::Tiling4IndexCopyCompute()
{
    auto tilingData = &moeTokenPermuteWithEpTilingData.indexCopyComputeParamsOp;
    int64_t tokenNums = moeTokenPermuteWithEpTilingData.get_n();
    int64_t topK = moeTokenPermuteWithEpTilingData.get_topK();
    int64_t cols = moeTokenPermuteWithEpTilingData.get_cols();

    if (numOutTokens != tokenNums * topK) {
        tilingKey_ = tilingKey_ + ENABLE_NUMOUTTOKENS;
    }
    tilingData->set_numOutTokens(numOutTokens);
    tilingData->set_start(start);
    tilingData->set_end(end);

    CoreParams coreParams;
    CalculateCoreParams(coreParams, tokenNums);
    UBParams ubParams;
    CalculateUBParams(ubParams, topK, cols);

    LoopParams loopParams;
    CalculateLoopParams(loopParams, coreParams, ubParams.onceIndicesTokenNums, ubParams.onceUbTokenNums);
    SetTilingDataFinalParams(tilingData, coreParams, ubParams, loopParams);

    aivNum = std::max(aivNum, coreParams.numBlocks);
}

static ge::graphStatus TilingForMoeTokenPermuteWithEp(gert::TilingContext* context)
{
    MoeTokenPermuteWithEpTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMoeTokenPermuteWithEp(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTokenPermuteWithEp)
    .Tiling(TilingForMoeTokenPermuteWithEp)
    .TilingParse<MoeTokenPermuteWithEpCompileInfo>(TilingPrepareForMoeTokenPermuteWithEp);
} // namespace optiling
