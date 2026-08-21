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
 * \file moe_token_permute_with_routing_map_tiling.cpp
 * \brief
 */
#include "op_host/tiling_util.h"
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
constexpr static uint64_t SIMT_KEY = 10;
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
const static int64_t INPUT_DTYPE_B64 = 8;
const static int64_t INPUT_DTYPE_B32 = 4;
const static int64_t INPUT_DTYPE_B16 = 2;
const static int64_t INPUT_DTYPE_B8 = 1;
const static int64_t INDICES_SIZE = 8192;
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

constexpr int32_t DCACHE_SIZE = 128 * 1024;
#ifdef DAVID_FPGA
const static int64_t SMALL_CASE_THREAD_NUM = 64;
#else
const static int64_t SMALL_CASE_THREAD_NUM = 128;
#endif
const static int64_t MAX_THREAD_NUM = 2048;
constexpr int32_t DCACHE = 32 * 1024;
const static int64_t SIMD_B32_THRES = 620;

template <typename T>
static auto GetCeilInt(const T &value1, const T &value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

template <typename T>
static auto GetDiv(const T &value1, const T &value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return (value1) / value2;
}

template <typename T>
static auto GetRem(const T &value1, const T &value2) -> T
{
    if (value2 == 0) {
        return value2;
    }
    return value1 % value2;
}

template <typename T1, typename T2>
inline auto FloorAlign(const T1 &a, const T2 &b) -> T1
{
    if (b != 0) {
        return (a) / b * b;
    }
    return a;
}

template <typename T1, typename T2>
inline auto UpAlign(const T1 &a, const T2 &b) -> T1
{
    if (b != 0) {
        return (a + b - 1) / b * b;
    }
    return a;
}

// 与 kernel moe_common.h Align 一致：按 32B 对齐元素个数
inline int64_t KernelElementAlign(int64_t elementNum, int64_t dtypeBytes)
{
    if (dtypeBytes == 0) {
        return 0;
    }
    return UpAlign(elementNum * dtypeBytes, ONE_BLOCK_BYTE) / dtypeBytes;
}

// 非对齐 cumsum+scatter 路径 workspace（int32 元素布局，见 moe_token_permute_with_routing_map.cpp）
inline size_t CalcNonAlignedScatterWorkspaceBytes(int64_t msNeedCoreNum, int64_t numTokens, int64_t numExperts,
                                                  int64_t totalLength)
{
    int64_t partialOffset = KernelElementAlign(msNeedCoreNum, INT32_DTYPE_SIZE);
    int64_t sortPositionOffset = partialOffset + msNeedCoreNum * numTokens;
    int64_t valueStreamOffset = sortPositionOffset + numExperts * numTokens;
    int64_t requiredInt32Elems = valueStreamOffset + totalLength;
    return static_cast<size_t>(requiredInt32Elems) * static_cast<size_t>(INT32_DTYPE_SIZE);
}

inline bool GetLengthByType(int32_t dtype, uint32_t &dsize)
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

inline static int64_t CeilLog4(int64_t x) { return (int64_t)std::ceil(std::log(x) / std::log(NUM_FOUR)); }

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

struct MoeTokenPermuteWithRoutingMapIndexCopyParams {
    int64_t topK = 0;
    int64_t cols = 0;
    int64_t frontCoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t numBlocks = 0;
    int64_t coreCalcNum = 0;
    int64_t coreCalcTail = 0;
    int64_t ubLeft = 0;
    int64_t oneTokenBtypeSize = 0;
    int64_t oneTokenBtypeSizeAlign32 = 0;

    int64_t oneTokenlastMove = 1;
    int64_t oneTokenOnceMove = 1;
    int64_t oneTokenMoveTimes = 1;
    int64_t onceIndicesTokenMoveTimes = 1;
    int64_t onceUbTokenNums = 1;
    int64_t onceIndicesTokenNums = 1;
    int64_t onceIndices = 1;
    int64_t tokenUB = 1;
    int64_t indicesUB = 1;

    int64_t frontCoreLoop = 0;
    int64_t frontCoreLastTokenNums = 0;
    int64_t tailCoreLoop = 0;
    int64_t tailCoreLastTokenNums = 0;
    int64_t tailLastonceIndicesTokenMoveTimes = 0;
    int64_t tailLastIndicesLastTokenNums = 0;
    int64_t frontLastonceIndicesTokenMoveTimes = 0;
    int64_t frontLastIndicesLastTokenNums = 0;
};

class MoeTokenPermuteWithRoutingMapTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeTokenPermuteWithRoutingMapTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeTokenPermuteWithRoutingMapTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override { return true; }
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
    ge::graphStatus InitShapeAttrsInfo();
    ge::graphStatus CheckOutShape();
    void Tiling4IndexCopyCompute();
    uint64_t CalcMaskedSelectUb();
    void Tiling4MaskedSelect();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void InitMoeTokenPermuteWithRoutingMapIndexCopyParams(MoeTokenPermuteWithRoutingMapIndexCopyParams &params);
    void SetMoeTokenPermuteWithRoutingMapIndexCopyParams(MoeTokenPermuteWithRoutingMapIndexCopyParams &params);
    void Tiling4VBSCompute();
    void Tiling4VBSComputeLastdim();
    void ShowIndexCopyComputeTilingDataTilingData();
    void Tiling4GatherCompute();
    void Tiling4GatherComputeSimt(bool isProb);

    int64_t XDtypeImprove();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(PermuteVBSComputeRMTilingData *tilingData);
    void Tinlig4VBSMultiCoreComputeLastdim(PermuteVBSComputeRMTilingData *tilingData);
    void Tinlig4VBSOneCoreCompute(PermuteVBSComputeRMTilingData *tilingData);

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
    int32_t improveDtypeSize_ = 0;
    bool regBase = false;
    bool paddedMode = false;
    const char *opName = nullptr;
    int64_t innerSize_ = 0;
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
    OP_CHECK_IF(indicesPtr == nullptr, OP_LOGE(opName, "fail to get input [indices]"), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const MoeTokenPermuteWithRoutingMapCompileInfo *>(context_->GetCompileInfo());

    uint64_t aivNumLocal; // Vector核数量
    auto platformInfo = context_->GetPlatformInfo();
    regBase = Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    if (platformInfo == nullptr) {
        aivNumLocal = compileInfo->aivNum; // Vector核数量
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
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
    aicoreParams_.numBlocks = aivNumLocal;

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
        OP_LOGE(context_->GetNodeName(),
                "The dim number of Output permute_tokens should be greater than 1 but got [%lu].", tokensDimNnum);
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
        OP_LOGE(context_->GetNodeName(), "The hidden_size of output permuteTokens should be %ld but got %ld.",
                moeTokenPermuteWithRoutingMapTilingData.get_cols(), cols);
        return ge::GRAPH_FAILED;
    }

    if (tokensShape.GetDim(0) != numOutTokens && !paddedMode) {
        OP_LOGE(context_->GetNodeName(), "The dim 0 of output permuteTokens should be %ld but got %ld.", numOutTokens,
                tokensShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::InitShapeAttrsInfo()
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
    const int64_t *numOutTokensPtr = attrs->GetAttrPointer<int64_t>(id++);
    const bool *paddedModePtr = attrs->GetAttrPointer<bool>(id);
    numOutTokens = *numOutTokensPtr;
    paddedMode = *paddedModePtr;
    numTokens = tokensShape.GetDim(0);
    numExperts = IndicesShape.GetDim(0);
    if (numExperts < 0) {
        OP_LOGE(context_->GetNodeName(), "Input attr's num_out_tokens [%ld] should be larger than 0.", numTokens);
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
        OP_LOGE(context_->GetNodeName(), "Input token's dim 0 [%ld] should be same with routingmap's tokennum [%ld].",
                tokensShape.GetDim(0), IndicesShape.GetDim(1));
        return ge::GRAPH_FAILED;
    }

    tokenBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType());
    indicesBtypeSize = ge::GetSizeByDataType(ge::DT_INT32);

    moeTokenPermuteWithRoutingMapTilingData.set_cols(cols);
    auto tokenOneBlockNum = GetDiv(ONE_BLOCK_BYTE, tokenBtypeSize);
    auto colsAlign = GetCeilInt(cols, tokenOneBlockNum) * tokenOneBlockNum;
    moeTokenPermuteWithRoutingMapTilingData.set_n(tokensShape.GetDim(0));
    moeTokenPermuteWithRoutingMapTilingData.set_expertNum(numExperts);
    moeTokenPermuteWithRoutingMapTilingData.set_colsAlign(colsAlign);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::GetShapeAttrsInfo()
{
    ge::graphStatus status = InitShapeAttrsInfo();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Init Shape Attrs Info failed.");
        return ge::GRAPH_FAILED;
    }
    if (numTokens <= 0) {
        OP_LOGE(context_->GetNodeName(), "Input attr's num_out_tokens [%ld] should be larger than max 0.", numTokens);
        return ge::GRAPH_FAILED;
    }

    int64_t topK = numOutTokens / numTokens;

    if (paddedMode == false && topK > MAX_INDICES_NUM) {
        OP_LOGE(context_->GetNodeName(), "numOutTokens / numTokens [%ld] should not be larger than max topK[%ld].",
                topK, MAX_INDICES_NUM);
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
        OP_LOGE(context_->GetNodeName(), "The elements num of indices [%ld] should be less than [%ld].", totalLength,
                SORT_LIMIT_LENGTH);
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckOutShape();
    return ret;
}

void MoeTokenPermuteWithRoutingMapTilingBase::ShowIndexCopyComputeTilingDataTilingData()
{
    OP_LOGD(opName,
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
}

void MoeTokenPermuteWithRoutingMapTilingBase::ShowTilingData()
{
    ShowIndexCopyComputeTilingDataTilingData();
    OP_LOGD(opName,
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
    OP_LOGD(opName, "PermuteVMSMiddleComputeRMTilingData is needCoreNum:%ld",
            moeTokenPermuteWithRoutingMapTilingData.vmsMiddleComputeParamsOp.get_needCoreNum());
    OP_LOGD(opName, "moeTokenPermuteWithRoutingMapTilingData is coreNum:%ld, n:%ld, cols:%ld, colsAlign:%ld, k:%ld",
            moeTokenPermuteWithRoutingMapTilingData.get_coreNum(), moeTokenPermuteWithRoutingMapTilingData.get_n(),
            moeTokenPermuteWithRoutingMapTilingData.get_cols(), moeTokenPermuteWithRoutingMapTilingData.get_colsAlign(),
            moeTokenPermuteWithRoutingMapTilingData.get_topK());
    OP_LOGD(opName, "PermuteSortOutComputeRMTilingData is oneLoopMaxElements:%ld",
            moeTokenPermuteWithRoutingMapTilingData.sortOutComputeParamsOp.get_oneLoopMaxElements());
}

int64_t MoeTokenPermuteWithRoutingMapTilingBase::XDtypeImprove()
{
    improveDtypeSize_ = tokenBtypeSize;
    int64_t lastAxisBytes = moeTokenPermuteWithRoutingMapTilingData.get_cols() * tokenBtypeSize;
    if ((tokenBtypeSize < INPUT_DTYPE_B64) && (lastAxisBytes % INPUT_DTYPE_B64) == 0) {
        OP_LOGD(context_->GetNodeName(), "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B64", lastAxisBytes);
        improveDtypeSize_ = INPUT_DTYPE_B64;
        return INPUT_DTYPE_B64;
    }

    if ((tokenBtypeSize < INPUT_DTYPE_B32) && (lastAxisBytes % INPUT_DTYPE_B32) == 0) {
        OP_LOGD(context_->GetNodeName(), "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B32", lastAxisBytes);
        improveDtypeSize_ = INPUT_DTYPE_B32;
        return INPUT_DTYPE_B32;
    }

    if ((tokenBtypeSize < INPUT_DTYPE_B16) && (lastAxisBytes % INPUT_DTYPE_B16) == 0) {
        OP_LOGD(context_->GetNodeName(), "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B16", lastAxisBytes);
        improveDtypeSize_ = INPUT_DTYPE_B16;
        return INPUT_DTYPE_B16;
    }
    return tokenBtypeSize;
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4GatherCompute()
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp;
    OP_LOGD(opName, "Tiling4GatherCompute start");

    int64_t cols = moeTokenPermuteWithRoutingMapTilingData.get_cols();

    int64_t blockFactor = numOutTokens / realCoreNumAiv;
    int64_t tailBlockFactor = numOutTokens - blockFactor * realCoreNumAiv;

    int64_t ubBlockSize = static_cast<int64_t>(ONE_BLOCK_BYTE);
    int64_t innerSize_ = cols / (XDtypeImprove() / tokenBtypeSize);

    int64_t ubAviable =
        (aicoreParams_.ubSize - INDICES_SIZE) / ubBlockSize * ubBlockSize / improveDtypeSize_ / BUFFER_NUM;

    int32_t indiceFactor = INDICES_SIZE / INT32_DTYPE_SIZE;
    int64_t needCoreNum_ = blockFactor > 0 ? realCoreNumAiv : tailBlockFactor;
    aivNum = std::max(aivNum, static_cast<int64_t>(needCoreNum_));

    tilingData->set_needCoreNum(needCoreNum_);
    tilingData->set_onceIndices(indiceFactor);
    tilingData->set_oneTokenBtypeSize(improveDtypeSize_);
    tilingData->set_numOutTokens(numOutTokens);
    tilingData->set_onceUbTokenNums(innerSize_);
    tilingData->set_coreCalcNum(blockFactor);
    tilingData->set_tailCoreNum(tailBlockFactor);
    tilingData->set_tokenUB(ubAviable);
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4GatherComputeSimt(bool isProb)
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp;
    int32_t threadNum = MAX_THREAD_NUM;
    int64_t ySize_ = isProb ? numOutTokens : innerSize_ * numOutTokens;
    while ((threadNum >= NUM_TWO * SMALL_CASE_THREAD_NUM) &&
           (GetDiv(ySize_, static_cast<int64_t>(threadNum)) < (aivNum / NUM_TWO))) {
        threadNum = threadNum / NUM_TWO;
    }
    int64_t needCoreNum_ = 1;
    if (isProb) {
        tilingData->set_onceIndices(threadNum);
        int64_t perCoreElements = GetDiv(ySize_, realCoreNumAiv);
        if (ySize_ < threadNum) {
            tilingData->set_frontCoreNum(needCoreNum_); // 复用旧的tiling结构体
            tilingData->set_frontCoreLoop(ySize_);
            tilingData->set_tailCoreLoop(ySize_);
        } else {
            perCoreElements = (perCoreElements + threadNum - 1) / threadNum * threadNum; // 对齐到threadNum_的倍数
            needCoreNum_ = GetDiv(ySize_, perCoreElements);
            int64_t lastCoreElements = ySize_ - perCoreElements * (needCoreNum_ - 1);
            tilingData->set_frontCoreNum(needCoreNum_);
            tilingData->set_frontCoreLoop(perCoreElements);
            tilingData->set_tailCoreLoop(lastCoreElements);
        }
    } else {
        tilingData->set_onceIndicesTokenNums(threadNum);
        tilingData->set_onceUbTokenNums(innerSize_);
        int64_t perCoreElements = GetDiv(ySize_, realCoreNumAiv);
        if (ySize_ < threadNum) {
            tilingData->set_needCoreNum(needCoreNum_);
            tilingData->set_coreCalcNum(ySize_);
            tilingData->set_tailCoreNum(ySize_);
        } else {
            perCoreElements = (perCoreElements + threadNum - 1) / threadNum * threadNum; // 对齐到threadNum_的倍数
            needCoreNum_ = GetDiv(ySize_, perCoreElements);
            int64_t lastCoreElements = ySize_ - perCoreElements * (needCoreNum_ - 1);
            tilingData->set_needCoreNum(needCoreNum_);
            tilingData->set_coreCalcNum(perCoreElements);
            tilingData->set_tailCoreNum(lastCoreElements);
        }
    }
    aivNum = std::max(aivNum, static_cast<int64_t>(needCoreNum_));
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::DoOpTiling()
{
    bool isSimd;
    if (regBase) {
        improveDtypeSize_ = tokenBtypeSize;
        int64_t cols = moeTokenPermuteWithRoutingMapTilingData.get_cols();
        isSimd = (cols >= SIMD_B32_THRES) && numOutTokens >= aivNum / NUM_TWO;
        innerSize_ = cols;
        aicoreParams_.ubSize = aicoreParams_.ubSize - SIMT_UB_SIZE_BYTE;
    }

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

        if (regBase) {
            Tiling4GatherComputeSimt(true);
            context_->SetLocalMemorySize(static_cast<uint32_t>(aicoreParams_.ubSize));
            if (isSimd) {
                Tiling4GatherCompute();
            } else {
                tilingKey_ = tilingKey_ + SIMT_KEY;
                Tiling4GatherComputeSimt(false);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MoeTokenPermuteWithRoutingMapTilingBase::GetTilingKey() const { return tilingKey_; }

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    int64_t realCoreNum = paddedMode == true ? aivNum : moeTokenPermuteWithRoutingMapTilingData.get_coreNum();
    size_t sortWorkspaceSize = paddedMode == true ? realCoreNum * totalLength * sizeof(float) * NUM_TWO * NUM_TWO :
                                                    totalLength * sizeof(float) * NUM_TWO * NUM_TWO; // 排序需要的空间
    if (paddedMode == false) {
        int64_t msNeedCoreNum = moeTokenPermuteWithRoutingMapTilingData.maskedSelectParamsOp.get_needCoreNum();
        size_t nonAlignedWs = CalcNonAlignedScatterWorkspaceBytes(msNeedCoreNum, numTokens, numExperts, totalLength);
        sortWorkspaceSize = std::max(sortWorkspaceSize, nonAlignedWs);
    }
    size_t coreSyncWorkspaceSize = realCoreNum * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteWithRoutingMapTilingBase::PostTiling()
{
    context_->SetBlockDim(aivNum);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    context_->SetScheduleMode(1);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    moeTokenPermuteWithRoutingMapTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(moeTokenPermuteWithRoutingMapTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tinlig4VBSOneCoreCompute(PermuteVBSComputeRMTilingData *tilingData)
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
    PermuteVBSComputeRMTilingData *tilingData)
{
    int64_t perCoreElements = numTokens; // 每个核处理的元素数

    int64_t frontCoreNum =
        GetRem(numExperts, realCoreNumAiv) != 0 ? GetRem(numExperts, realCoreNumAiv) : realCoreNumAiv;
    int64_t tailCoreNum = numExperts <= realCoreNumAiv ? 0 : realCoreNumAiv - frontCoreNum;
    int64_t numBlocks = frontCoreNum + tailCoreNum;
    aivNum = numBlocks;
    int64_t coreCalcNum = GetCeilInt(numExperts, realCoreNumAiv);
    int64_t coreCalcTail = GetDiv(numExperts, realCoreNumAiv);
    tilingData->set_frontcoreTask(coreCalcNum);
    tilingData->set_tailcoreTask(coreCalcTail);
    tilingData->set_frontCoreNum(frontCoreNum);
    tilingData->set_tailCoreNum(tailCoreNum);
    tilingData->set_needCoreNum(numBlocks);
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
    tilingData->set_lastCoreWSindex(0);
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tinlig4VBSMultiCoreCompute(PermuteVBSComputeRMTilingData *tilingData)
{
    int64_t needCoreNum = GetCeilInt(totalLength, sortLoopMaxElement);      // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum))); // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, realCoreNumAiv);                    // 不能超过物理核数

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
    aivNum = std::max(aivNum, static_cast<int64_t>(tilingData->get_needCoreNum()));
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

uint64_t MoeTokenPermuteWithRoutingMapTilingBase::CalcMaskedSelectUb()
{
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
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t ubLength = ((ubSize - ONE_BLOCK_BYTE * DOUBLE_BUFFER * DOUBLE_BUFFER) / oneDataSize) / alignNum * alignNum;
    ubLength = static_cast<int64_t>(ubLength) > numTokens ? numTokens : ubLength; // 一次ub能放多少数
    return ubLength;
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4MaskedSelect()
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.maskedSelectParamsOp;

    uint64_t aivUseNum = realCoreNumAiv; // Vector核数量
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

    uint64_t numBlocks = 0;
    uint64_t ubLength = CalcMaskedSelectUb();
    OP_CHECK_IF(ubLength == 0, OP_LOGE(opName, "Ub length is zero."), return);

    // 运行核数
    numBlocks = (numExperts > static_cast<int64_t>(aivUseNum)) ? aivUseNum : numExperts;
    tilingData->set_needCoreNum(numBlocks);

    // 切分流程
    formerNum = numExperts % numBlocks;
    if (formerNum == 0) {
        formerNum = numBlocks;
    }
    tailNum = numBlocks - formerNum;

    formerLength = (numExperts + numBlocks - 1) / numBlocks * numTokens; // 算的多的核需要算多少数
    formerTileNum = (formerLength + ubLength - 1) / ubLength;            // 算的多的核要用多少次ub
    formerTileLength = ubLength;                                         // 算的多的核一次ub能放多少数
    formerLastTileLength = formerLength % ubLength; // 算的多的核最后一次ub需要算多少数
    if (formerLastTileLength == 0) {
        formerLastTileLength = ubLength;
    }

    if (tailNum > 0) {
        tailLength = (totalLengthLocal - formerLength * formerNum) / tailNum;
        tailTileNum = (tailLength + ubLength - 1) / ubLength;
        tailTileLength = ubLength;
        tailLastTileLength = tailLength % ubLength;
        if (tailLastTileLength == 0) {
            tailLastTileLength = ubLength;
        }
    }
    aivNum = std::max(aivNum, static_cast<int64_t>(numBlocks));
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

void MoeTokenPermuteWithRoutingMapTilingBase::InitMoeTokenPermuteWithRoutingMapIndexCopyParams(
    MoeTokenPermuteWithRoutingMapIndexCopyParams &params)
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp;
    int64_t tokenNums = moeTokenPermuteWithRoutingMapTilingData.get_n();
    params.topK = moeTokenPermuteWithRoutingMapTilingData.get_topK();
    params.cols = moeTokenPermuteWithRoutingMapTilingData.get_cols();

    tilingData->set_numOutTokens(numOutTokens);

    params.frontCoreNum = GetRem(tokenNums, realCoreNumAiv) != 0 ? GetRem(tokenNums, realCoreNumAiv) : realCoreNumAiv;
    params.tailCoreNum = tokenNums <= realCoreNumAiv ? 0 : realCoreNumAiv - params.frontCoreNum;
    params.numBlocks = params.frontCoreNum + params.tailCoreNum;
    params.coreCalcNum = GetCeilInt(tokenNums, realCoreNumAiv);
    params.coreCalcTail = GetDiv(tokenNums, realCoreNumAiv);

    params.ubLeft = aicoreParams_.ubSize - MAX_INDICES_NUM * INT32_DTYPE_SIZE;
    params.oneTokenBtypeSize = params.cols * tokenBtypeSize;

    params.oneTokenBtypeSizeAlign32 = UpAlign(params.oneTokenBtypeSize, ONE_BLOCK_BYTE);

    params.oneTokenlastMove = 1;
    params.oneTokenOnceMove = 1;
    params.oneTokenMoveTimes = 1;
    params.onceIndicesTokenMoveTimes = 1;
    params.onceUbTokenNums = 1;
    params.onceIndicesTokenNums = 1;
    params.onceIndices = 1;
    params.tokenUB = 1;
    params.indicesUB = 1;
}

void MoeTokenPermuteWithRoutingMapTilingBase::SetMoeTokenPermuteWithRoutingMapIndexCopyParams(
    MoeTokenPermuteWithRoutingMapIndexCopyParams &params)
{
    auto tilingData = &moeTokenPermuteWithRoutingMapTilingData.indexCopyComputeParamsOp;
    tilingData->set_tokenUB(params.tokenUB);
    tilingData->set_indicesUB(params.indicesUB);
    tilingData->set_needCoreNum(params.numBlocks);
    tilingData->set_frontCoreNum(params.frontCoreNum);
    tilingData->set_tailCoreNum(params.tailCoreNum);
    tilingData->set_coreCalcNum(params.coreCalcNum);
    tilingData->set_coreCalcTail(params.coreCalcTail);
    tilingData->set_oneTokenBtypeSize(params.oneTokenBtypeSize);
    tilingData->set_onceIndicesTokenMoveTimes(params.onceIndicesTokenMoveTimes);
    tilingData->set_onceUbTokenNums(params.onceUbTokenNums);
    tilingData->set_onceIndicesTokenNums(params.onceIndicesTokenNums);
    tilingData->set_onceIndices(params.onceIndices);
    tilingData->set_oneTokenlastMove(params.oneTokenlastMove);
    tilingData->set_oneTokenOnceMove(params.oneTokenOnceMove);
    tilingData->set_oneTokenMoveTimes(params.oneTokenMoveTimes);
    tilingData->set_frontCoreLoop(params.frontCoreLoop);
    tilingData->set_frontCoreLastTokenNums(params.frontCoreLastTokenNums);
    tilingData->set_tailCoreLoop(params.tailCoreLoop);
    tilingData->set_tailCoreLastTokenNums(params.tailCoreLastTokenNums);
    tilingData->set_tailLastonceIndicesTokenMoveTimes(params.tailLastonceIndicesTokenMoveTimes);
    tilingData->set_tailLastIndicesLastTokenNums(params.tailLastIndicesLastTokenNums);
    tilingData->set_frontLastonceIndicesTokenMoveTimes(params.frontLastonceIndicesTokenMoveTimes);
    tilingData->set_frontLastIndicesLastTokenNums(params.frontLastIndicesLastTokenNums);
}

void MoeTokenPermuteWithRoutingMapTilingBase::Tiling4IndexCopyCompute()
{
    MoeTokenPermuteWithRoutingMapIndexCopyParams params;
    InitMoeTokenPermuteWithRoutingMapIndexCopyParams(params);
    if (params.ubLeft >= BUFFER_NUM * params.oneTokenBtypeSizeAlign32) {
        params.onceUbTokenNums =
            GetDiv(static_cast<int64_t>(aicoreParams_.ubSize),
                   params.oneTokenBtypeSizeAlign32 * BUFFER_NUM + params.topK * BUFFER_NUM * INT32_DTYPE_SIZE);
        params.onceUbTokenNums = std::min(params.onceUbTokenNums, MAX_BLOCK_COUNT);
        int64_t TopKUbLeft =
            aicoreParams_.ubSize - params.onceUbTokenNums * params.oneTokenBtypeSizeAlign32 * BUFFER_NUM;
        params.onceIndicesTokenMoveTimes = GetDiv(TopKUbLeft, params.onceUbTokenNums * params.topK * INT32_DTYPE_SIZE);
        params.onceIndicesTokenNums = params.onceIndicesTokenMoveTimes * params.onceUbTokenNums;
        params.onceIndices = params.onceIndicesTokenNums * params.topK;
        params.tokenUB = params.onceUbTokenNums * params.oneTokenBtypeSizeAlign32;
        params.indicesUB = UpAlign(params.onceIndices * INT32_DTYPE_SIZE, ONE_BLOCK_BYTE);
    } else {
        params.onceIndicesTokenNums = GetDiv(MAX_INDICES_NUM, params.topK);
        params.onceIndices = params.onceIndicesTokenNums * params.topK;
        params.oneTokenOnceMove =
            GetDiv(FloorAlign(GetDiv(params.ubLeft, BUFFER_NUM), DATA_MOVE_ALIGN), tokenBtypeSize);
        params.oneTokenMoveTimes = GetCeilInt(params.cols, params.oneTokenOnceMove);
        params.oneTokenlastMove = params.cols - (params.oneTokenMoveTimes - 1) * params.oneTokenOnceMove;
        tilingKey_ = tilingKey_ + SPLIT_D_MODE;
        params.tokenUB = params.oneTokenOnceMove * tokenBtypeSize;
        params.indicesUB = MAX_INDICES_NUM * INT32_DTYPE_SIZE;
    }

    params.frontCoreLoop = GetCeilInt(params.coreCalcNum, params.onceIndicesTokenNums);
    params.frontCoreLastTokenNums = params.coreCalcNum - (params.frontCoreLoop - 1) * params.onceIndicesTokenNums;
    params.tailCoreLoop = GetCeilInt(params.coreCalcTail, params.onceIndicesTokenNums);
    params.tailCoreLastTokenNums = params.coreCalcTail - (params.tailCoreLoop - 1) * params.onceIndicesTokenNums;
    params.tailLastonceIndicesTokenMoveTimes = GetCeilInt(params.tailCoreLastTokenNums, params.onceUbTokenNums);
    params.tailLastIndicesLastTokenNums =
        params.tailCoreLastTokenNums - (params.tailLastonceIndicesTokenMoveTimes - 1) * params.onceUbTokenNums;
    params.frontLastonceIndicesTokenMoveTimes = GetCeilInt(params.frontCoreLastTokenNums, params.onceUbTokenNums);
    params.frontLastIndicesLastTokenNums =
        params.frontCoreLastTokenNums - (params.frontLastonceIndicesTokenMoveTimes - 1) * params.onceUbTokenNums;
    SetMoeTokenPermuteWithRoutingMapIndexCopyParams(params);
    aivNum = std::max(aivNum, params.numBlocks);
}

static ge::graphStatus TilingForMoeTokenPermuteWithRoutingMap(gert::TilingContext *context)
{
    MoeTokenPermuteWithRoutingMapTilingBase tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForMoeTokenPermuteWithRoutingMap(gert::TilingParseContext *context)
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
