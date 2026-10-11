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
 * \file moe_init_routing_v3_tiling.cpp
 * \brief
 */
#include "moe_init_routing_v3_tiling.h"
#include "register/op_def_registry.h"

using Ops::Transformer::OpTiling::TilingBaseClass;

namespace optiling {
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t NUM_FIVE = 5;
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
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_IDX_MAX = 10240;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;
const static int64_t ACTIVE_NUM_MIN_VALUE = static_cast<int64_t>(-1);
const static int64_t EXPERT_CAPACITY_MIN_VALUE = static_cast<int64_t>(0);

// Counting sort constants
const static int64_t COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM = 128;
const static int64_t COUNTING_SORT_FILTER_CHUNK_SIZE = 4096;
const static int64_t COUNTING_SORT_VECTOR_FILTER_THRESHOLD = 64;
const static int64_t COUNTING_SORT_ONE_REPEAT_COMPARE_NUM = 64;
const static int64_t COUNTING_SORT_MAX_PER_LOOP_COLS_BYTES = 64 * 1024; // 128KB
const static int64_t COUNTING_SORT_MIN_N = 256;                         // 128KB
const static int64_t COUNTING_SORT_MAX_N = 1024 * 1024;                 // 128KB

const static int64_t INPUT_X_INDEX = 0;
const static int64_t INPUT_EXPERT_IDX_INDEX = 1;
const static int64_t INPUT_SCALE_INDEX = 2;
const static int64_t INPUT_OFFSET_INDEX = 3;
const static int64_t OUTPUT_EXPANDED_X_INDEX = 0;
const static int64_t OUTPUT_EXPANDED_ROW_IDX_INDEX = 1;
const static int64_t OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2;
const static int64_t OUTPUT_EXPANDED_SCALE_INDEX = 3;
const static int64_t ATTR_ACTIVE_NUM_INDEX = 0;
const static int64_t ATTR_EXPERT_CAPACITY_INDEX = 1;
const static int64_t ATTR_EXPERT_NUM_INDEX = 2;
const static int64_t ATTR_DROP_PAD_MODE_INDEX = 3;
const static int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 4;
const static int64_t ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX = 5;
const static int64_t ATTR_QUANT_MODE_INDEX = 6;
const static int64_t ATTR_EXPERT_RANGE_INDEX = 7;
const static int64_t ATTR_ROW_IDX_TYPE_INDEX = 8;
const static int64_t ATTR_EXPERT_RANGE_DIM = 2;
const static int64_t GATHER = 0;
const static int64_t SCATTER = 1;
const static int64_t UN_QUANT = -1L;
const static int64_t STATIC_QUANT = 0;
const static int64_t DYNAMIC_QUANT = 1;
const static int64_t CUMSUM = 0;
const static int64_t COUNT = 1;
const static int64_t KEY_VALUE = 2;
const static int64_t DROP_LESS = 0;
const static int64_t DROP_PAD = 1;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21;
const static int64_t DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER = 13;
const static int64_t STATIC_QUANT_FULLLOAD_COLS_BUFFER = 11;

const static int64_t DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15;
const static int64_t DYNAMIC_QUANT_SCALE_SIZE_64 = 64;
const static int64_t MAX_COLS_DYNAMIC_QUANT = 6144;
const static int64_t SIZE_INT32 = 4;
const static int64_t SIZE_INT16 = 2;
const static int64_t SIZE_INT8 = 1;
const static int64_t SIZE_FP32 = 4;

const static uint64_t TILINGKEY_BASE = 1000000;
const static uint64_t SORT_CORE_TILINGKEY_BASE = 100000;
const static uint64_t QUANT_MODE_TILINGKEY_BASE = 10000;
const static uint64_t ROWIDX_TYPE_TILINGKEY_BASE = 1000;
const static uint64_t DROP_MODE_TILINGKEY_BASE = 100;

// Tiling Key for performance puncturing
const static uint64_t PERFORMANCE_TILINGKEY_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168 = 2000000;
const static uint64_t UNQUANTIZED_FULLLOAD_TILINGKEY = 2100000;
const static uint64_t STATIC_QUANT_FULLLOAD_TILINGKEY = 2200000;
const static uint64_t DYNAMIC_QUANT_FULLLOAD_TILINGKEY = 2300000;
const static uint64_t DYNAMIC_QUANT_EPFULLLOAD_TILINGKEY = 10000;
const static uint64_t DYNAMIC_QUANT_SMOOTHTYPE_FULLLOAD_TILINGKEY = 1000;
const static uint64_t EMPTY_TENSOR_TILINGKEY = 3000000;

const static int64_t PERFORMANCE_MODE_TOP_K = 8;
const static int64_t PERFORMANCE_MODE_BS_MIN = 384;
const static int64_t PERFORMANCE_MODE_BS_MAX = 8192;
const static int64_t PERFORMANCE_MODE_RANGE_MAX = 32;
const static int64_t PERFORMANCE_MODE_MAX_BATCH_SIZE_TOP_K = PERFORMANCE_MODE_BS_MAX * PERFORMANCE_MODE_TOP_K;
const static int64_t PERFORMANCE_MODE_MAX_ONE_CORE_GATHER = 21845;

const static int64_t ONE_BLOCK_ELEMENT = 8;
const static int64_t COUNTING_SORT_THRESHOLD = 1536;

const static int64_t gatherFirstN = 100;
const static int64_t gatherFirstScale = 8;
const static int64_t scale1H = 1;
const static int64_t scaleEH = 2;
const static int64_t ONE_REPEAT_SORT_NUM = 32;

enum class PerformanceMode : int32_t {
    COMMON = 0,
    ONE_CORE_GATHER_SORT = 1,
    MULTI_CORE_GATHER_SORT = 2,
};

static constexpr int64_t KEY_VALUE_MODE_DIM0_NUM = 2;

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    return x <= y ? x : x % y;
}

inline static int64_t AlignOneBlockByteCeil(int64_t x)
{
    return x / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

class MoeInitRountingV3TilingBase : public TilingBaseClass {
public:
    explicit MoeInitRountingV3TilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRountingV3TilingBase() override = default;

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
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckDtype();
    void Tiling4GatherOutCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4ExpertTokensCountCompute();
    void ShowTilingData();
    void Tinlig4VBSMultiCoreCompute(MoeV3VBSComputeTilingData* tilingData);
    void Tinlig4VBSOneCoreCompute(MoeV3VBSComputeTilingData* tilingData);
    bool IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() const;
    bool ShouldEnableCountingSortByShape() const;
    bool IsFullLoad();
    int64_t IsGatherFirstFullLoad();
    void SetGatherTilingData(MoeV3SrcToDstCapacityComputeTilingData* tilingData, int64_t perCoreRows,
                             int64_t lastCoreRows, int64_t cols);
    void SetGatherTilingDataCols(MoeV3SrcToDstCapacityComputeTilingData* tilingData, int64_t baseMaxCols, int64_t cols);
    void SetGatherTilingDataRows(MoeV3SrcToDstCapacityComputeTilingData* tilingData, int64_t perCoreRows,
                                 int64_t lastCoreRows, int64_t basePerLoopMaxRows);
    void ComputeCountingSortTiling();
    int64_t EstimateCountingSortFullLoadUB(int64_t coreTokenNum, int64_t coreEntries, int64_t expertCountStride,
                                           int64_t filterNeedCoreNum, int64_t actualExpertNum);
    void ComputeCountingSortCutOriginTiling(int64_t filterNeedCoreNum, int64_t filterPerCoreTokens,
                                            int64_t expertCountStride, int64_t actualExpertNum);
    void Tiling4SrcToDstDropPadCompute();
    void Tiling4SrcToDstDropPadDynamicCompute();
    void Tiling4SrcToDstCompute();
    PerformanceMode GetPerformanceMode() const;

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 1504;
    int64_t totalLength_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t cols_ = 0;
    int64_t inuptXDtypeSize_;

    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t isInputScale_ = 0;
    int64_t isInputOffset_ = 0;

    int64_t sortMode_ = 0;
    int64_t rowIdxTytpe_ = 0;
    int64_t activeNum_ = -1L;
    int64_t expertCapacity_ = -1L;
    int64_t expertNum_ = -1L;
    int64_t dropPadMode_ = -1L;
    int64_t expertTokensNumType_ = -1L;
    bool expertTokensNumFlag_ = false;
    int64_t quantMode_ = 0;
    int64_t rowIdxType_ = -1L;

    bool isFullload_ = false;
    bool isEmptyTensor_ = false;
    int64_t gatherFirstFullload_ = 0;
    int64_t ep_ = 0;
    int64_t smoothType_ = 0;
    int64_t countingSortFullLoad_ = -1;
    size_t countingSortWorkspaceSize_ = 0;

    const gert::StorageShape* xShapePtr_ = nullptr;
    const gert::StorageShape* expertIdxShapePtr_ = nullptr;
    const gert::StorageShape* scaleShapePtr_ = nullptr;
    const gert::StorageShape* offsetShapePtr_ = nullptr;

    const int64_t* activeNumPtr_ = nullptr;
    const int64_t* expertCapacityPtr_ = nullptr;
    const int64_t* expertNumPtr_ = nullptr;
    const int64_t* dropPadModePtr_ = nullptr;
    const int64_t* expertTokensNumTypePtr_ = nullptr;
    const bool* expertTokensNumFlagPtr_ = nullptr;
    const int64_t* quantModePtr_ = nullptr;
    const gert::ContinuousVector* activeExpertRangeListPtr_;
    const int64_t* rowIdxTypePtr_ = nullptr;

    const gert::StorageShape* expandedXShapePtr_ = nullptr;
    const gert::StorageShape* expandedRowIdxShapePtr_ = nullptr;
    const gert::StorageShape* expertTokensCountOrCumsumShapePtr_ = nullptr;
    const gert::StorageShape* expandedScaleShapePtr_ = nullptr;

    const gert::Shape performXShape = gert::Shape({1, 7168});
    const gert::Shape performExpertIdxShape = gert::Shape({1, 8});
    const gert::Shape performScaleShape = gert::Shape({256, 7168});

    const char* opName = "";
    MoeInitRoutingV3TilingData moeInitRoutingV3TilingData;
};

void MoeInitRountingV3TilingBase::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRountingV3TilingBase::GetPlatformInfo()
{
    auto compileInfoPtr = reinterpret_cast<const MoeInitRoutingV3CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "compileInfo"),
                return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    aicoreParams_.numBlocks = aivNum;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    moeInitRoutingV3TilingData.set_coreNum(aivNum);
    OP_LOGI(context_, "---PlatformInfo--- aivNum is: %ld, ubSizePlatForm is: %ld ", aivNum, aicoreParams_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::CheckAttr()
{
    quantMode_ = *quantModePtr_;
    moeInitRoutingV3TilingData.set_quantMode(quantMode_);
    OP_LOGI(context_, "quant_mode is: %ld ", quantMode_);

    dropPadMode_ = *dropPadModePtr_;
    moeInitRoutingV3TilingData.set_dropPadMode(dropPadMode_);
    OP_CHECK_IF((dropPadMode_ != DROP_LESS) && (dropPadMode_ != DROP_PAD),
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "drop_pad_mode", std::to_string(dropPadMode_),
                                          (std::to_string(DROP_LESS) + " or " + std::to_string(DROP_PAD))),
                return ge::GRAPH_FAILED);

    rowIdxTytpe_ = *rowIdxTypePtr_;
    moeInitRoutingV3TilingData.set_rowIdxType(rowIdxTytpe_);
    OP_LOGI(context_, "row_idx_type is: %ld ", rowIdxTytpe_);

    activeNum_ = *activeNumPtr_;
    if (dropPadMode_ == DROP_LESS) {
        OP_CHECK_IF(activeNum_ < ACTIVE_NUM_MIN_VALUE,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "active_num", std::to_string(activeNum_),
                                              ("greater than or equal to " + std::to_string(ACTIVE_NUM_MIN_VALUE))),
                    return ge::GRAPH_FAILED);
    }

    expertNum_ = *expertNumPtr_;
    moeInitRoutingV3TilingData.set_expertNum(expertNum_);
    OP_CHECK_IF(
        expertNum_ <= 0,
        OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_num", std::to_string(expertNum_), "greater than 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        activeExpertRangeListPtr_->GetSize() != ATTR_EXPERT_RANGE_DIM && activeExpertRangeListPtr_->GetSize() != 0,
        OP_LOGE_WITH_INVALID_ATTR_SIZE(context_->GetNodeName(), "expert_range",
                                       std::to_string(activeExpertRangeListPtr_->GetSize()),
                                       (std::to_string(ATTR_EXPERT_RANGE_DIM) + " or 0(no input)")),
        return ge::GRAPH_FAILED);
    if (activeExpertRangeListPtr_->GetSize() == 0) {
        expertStart_ = 0;
        expertEnd_ = expertNum_;
    } else {
        const int64_t* expertRangeList = reinterpret_cast<const int64_t*>(activeExpertRangeListPtr_->GetData());
        expertStart_ = expertRangeList[0];
        expertEnd_ = expertRangeList[1];
    }
    moeInitRoutingV3TilingData.set_expertStart(expertStart_);
    moeInitRoutingV3TilingData.set_expertEnd(expertEnd_);
    moeInitRoutingV3TilingData.set_actualExpertNum(expertEnd_ - expertStart_);
    OP_LOGI(context_, "expert_start is: %ld, expert_end is: %ld, actualExpertNum is: %ld ", expertStart_, expertEnd_,
            expertEnd_ - expertStart_);

    n_ = xShapePtr_->GetStorageShape().GetDim(0);
    expertCapacity_ = *expertCapacityPtr_;
    moeInitRoutingV3TilingData.set_expertCapacity(expertCapacity_);
    if (dropPadMode_ == DROP_PAD) {
        OP_CHECK_IF(
            expertCapacity_ <= EXPERT_CAPACITY_MIN_VALUE || expertCapacity_ > n_,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_Capacity", std::to_string(expertCapacity_),
                                      ("greater than 0 and less than " + std::to_string(n_))),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(rowIdxTytpe_ == SCATTER,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "row_idx_type", std::to_string(rowIdxTytpe_),
                                              "0 when drop_pad_mode is 1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            expertStart_ != 0 || expertEnd_ != expertNum_,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_range",
                                      ("[" + std::to_string(expertStart_) + ", " + std::to_string(expertEnd_) + "]"),
                                      ("[0, " + std::to_string(expertNum_) + "] when drop_pad_mode is 1")),
            return ge::GRAPH_FAILED);
    }

    expertTokensNumType_ = *expertTokensNumTypePtr_;
    moeInitRoutingV3TilingData.set_expertTokensNumType(expertTokensNumType_);
    OP_CHECK_IF(
        (expertTokensNumType_ != COUNT) && (expertTokensNumType_ != KEY_VALUE) && (expertTokensNumType_ != CUMSUM),
        OP_LOGE_WITH_INVALID_ATTR(
            context_->GetNodeName(), "expert_tokens_num_type", std::to_string(expertTokensNumType_),
            (std::to_string(COUNT) + ", " + std::to_string(KEY_VALUE) + " or " + std::to_string(CUMSUM))),
        return ge::GRAPH_FAILED);

    expertTokensNumFlag_ = *expertTokensNumFlagPtr_;
    if (dropPadMode_ == DROP_PAD && expertTokensNumFlag_) {
        OP_CHECK_IF(expertTokensNumType_ != COUNT,
                    OP_LOGE_WITH_INVALID_ATTR(
                        context_->GetNodeName(), "expert_tokens_num_type", std::to_string(expertTokensNumType_),
                        (std::to_string(COUNT) + "when DROP_MODE and expert_tokens_num_flag is true")),
                    return ge::GRAPH_FAILED);
    }
    if (expertTokensNumFlag_) {
        moeInitRoutingV3TilingData.set_expertTokensNumFlag(1);
    } else {
        moeInitRoutingV3TilingData.set_expertTokensNumFlag(0);
    }

    OP_CHECK_IF(expertStart_ < 0,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_start", std::to_string(expertStart_),
                                          "greater than or equal to 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertStart_ >= expertEnd_,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_start", std::to_string(expertStart_),
                                          ("less than " + std::to_string(expertEnd_))),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertEnd_ > expertNum_,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_end", std::to_string(expertEnd_),
                                          ("less than or equal to " + std::to_string(expertNum_))),
                return ge::GRAPH_FAILED);
    if (expertTokensNumType_ == KEY_VALUE) {
        OP_CHECK_IF(expertEnd_ > KV_MODE_EXPERT_IDX_MAX,
                    OP_LOGE_WITH_INVALID_ATTR(
                        context_->GetNodeName(), "expert_end", std::to_string(expertEnd_),
                        ("less than or equal to " + std::to_string(KV_MODE_EXPERT_IDX_MAX) + " in KEY_VALUE mode")),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(expertEnd_ > EXPERT_IDX_MAX,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_end", std::to_string(expertEnd_),
                                              ("less than or equal to " + std::to_string(EXPERT_IDX_MAX))),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(quantMode_ != UN_QUANT && quantMode_ != DYNAMIC_QUANT && quantMode_ != STATIC_QUANT,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "quant_mode", std::to_string(quantMode_),
                                          (std::to_string(UN_QUANT) + ", " + std::to_string(DYNAMIC_QUANT) + " or " +
                                           std::to_string(STATIC_QUANT))),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rowIdxTytpe_ != SCATTER && rowIdxTytpe_ != GATHER,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "row_idx_type", std::to_string(rowIdxTytpe_),
                                          (std::to_string(SCATTER) + " or " + std::to_string(GATHER))),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::CheckInputShape()
{
    const gert::Shape xShape = xShapePtr_->GetStorageShape();
    OP_LOGI(context_, "input x shape: %s ", Ops::Base::ToString(xShape).c_str());
    const gert::Shape expertIdxShape = expertIdxShapePtr_->GetStorageShape();
    OP_LOGI(context_, "input expert_idx shape: %s.", Ops::Base::ToString(expertIdxShape).c_str());

    // 参数校验
    OP_CHECK_IF(xShape.GetDimNum() != DIM_TWO,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xShape.GetDimNum()),
                                             std::to_string(DIM_TWO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertIdxShape.GetDimNum() != DIM_TWO,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expert_idx",
                                             std::to_string(expertIdxShape.GetDimNum()), std::to_string(DIM_TWO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDim(0) != expertIdxShape.GetDim(0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "x dim[0], expert_idx dim[0]",
                    (std::to_string(xShape.GetDim(0)) + ", " + std::to_string(expertIdxShape.GetDim(0))),
                    "input rows should be same"),
                return ge::GRAPH_FAILED);

    n_ = expertIdxShape.GetDim(0);
    k_ = expertIdxShape.GetDim(1);
    cols_ = xShape.GetDim(1);
    moeInitRoutingV3TilingData.set_n(n_);
    moeInitRoutingV3TilingData.set_k(k_);
    moeInitRoutingV3TilingData.set_cols(cols_);
    totalLength_ = n_ * k_;

    // 空tensor场景：n_==0（行数为0）
    if (n_ == 0) {
        isEmptyTensor_ = true;
        int64_t expertCountElements = 0;
        if (expertTokensNumType_ == KEY_VALUE) {
            expertCountElements = expertNum_ * KEY_VALUE_MODE_DIM0_NUM;
        } else {
            expertCountElements = expertEnd_ - expertStart_;
        }
        moeInitRoutingV3TilingData.set_expertCountElements(expertCountElements);
    }

    if (activeNum_ == 0 || activeNum_ == ACTIVE_NUM_MIN_VALUE) {
        activeNum_ = totalLength_;
    } else {
        activeNum_ = std::min(activeNum_, totalLength_);
    }
    moeInitRoutingV3TilingData.set_activeNum(activeNum_);

    inuptXDtypeSize_ =
        static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(INPUT_X_INDEX)->GetDataType()));
    OP_LOGI(context_, "Input x dtype size is: %ld. ", inuptXDtypeSize_);

    if (quantMode_ == UN_QUANT && scaleShapePtr_ != nullptr) {
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OP_LOGI(context_, "input scale shape: %s", Ops::Base::ToString(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        OP_CHECK_IF(scaleDimNum != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", std::to_string(scaleDimNum), "1"),
                    return ge::GRAPH_FAILED);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        OP_CHECK_IF(scaleDim0 != n_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[0]", std::to_string(scaleDim0),
                                              std::to_string(n_)),
                    return ge::GRAPH_FAILED);
    }

    if (quantMode_ == STATIC_QUANT) {
        OP_CHECK_IF(scaleShapePtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "scale"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(offsetShapePtr_ == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "offset"),
                    return ge::GRAPH_FAILED);
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OP_LOGI(context_, "input scale shape: %s", Ops::Base::ToString(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        OP_CHECK_IF(scaleDimNum != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", std::to_string(scaleDimNum), "1"),
                    return ge::GRAPH_FAILED);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        OP_CHECK_IF(scaleDim0 != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[0]", std::to_string(scaleDim0), "1"),
                    return ge::GRAPH_FAILED);
        auto offsetShape = offsetShapePtr_->GetStorageShape();
        OP_LOGI(context_, "input offset shape: %s", Ops::Base::ToString(offsetShape).c_str());
        auto offsetDimNum = static_cast<int64_t>(offsetShape.GetDimNum());
        OP_CHECK_IF(offsetDimNum != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "offset", std::to_string(offsetDimNum), "1"),
                    return ge::GRAPH_FAILED);
        auto offsetDim0 = static_cast<int64_t>(offsetShape.GetDim(0));
        OP_CHECK_IF(
            offsetDim0 != 1,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "offset dim[0]", std::to_string(offsetDim0), "1"),
            return ge::GRAPH_FAILED);
    }

    if (quantMode_ == DYNAMIC_QUANT && scaleShapePtr_ != nullptr) {
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OP_LOGI(context_, "input scale shape: %s", Ops::Base::ToString(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        OP_CHECK_IF(scaleDimNum != NUM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", std::to_string(scaleDimNum),
                                                 std::to_string(NUM_TWO)),
                    return ge::GRAPH_FAILED);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        OP_CHECK_IF(scaleDim0 != (expertEnd_ - expertStart_) && scaleDim0 != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[0]", std::to_string(scaleDim0),
                                              (std::to_string(expertEnd_ - expertStart_) + " or 1")),
                    return ge::GRAPH_FAILED);
        auto scaleDim1 = static_cast<int64_t>(scaleShape.GetDim(1));
        OP_CHECK_IF(scaleDim1 != cols_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[1]", std::to_string(scaleDim1),
                                              std::to_string(cols_)),
                    return ge::GRAPH_FAILED);
        if (scaleDim0 == 1) {
            smoothType_ = scale1H;
        } else {
            smoothType_ = scaleEH;
        }
        moeInitRoutingV3TilingData.set_smoothType(smoothType_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::CheckOutShape()
{
    // 获取输入shape
    const gert::Shape expandedXShape = context_->GetOutputShape(0)->GetStorageShape();
    OP_LOGI(context_, "expanded_x shape: %s.", Ops::Base::ToString(expandedXShape).c_str());
    const gert::Shape expandedRowIdxShape = context_->GetOutputShape(1)->GetStorageShape();
    OP_LOGI(context_, "expanded_row_idx shape: %s.", Ops::Base::ToString(expandedRowIdxShape).c_str());
    const gert::Shape expertTokensCountOrCumsumShape = context_->GetOutputShape(NUM_TWO)->GetStorageShape();
    OP_LOGI(context_, "expert_tokens_count_or_cumsum shape: %s.",
            Ops::Base::ToString(expertTokensCountOrCumsumShape).c_str());

    size_t expandedXDimNum = expandedXShape.GetDimNum();
    if (dropPadMode_ > 0) {
        OP_CHECK_IF(expandedXDimNum != NUM_THREE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expandedX", std::to_string(expandedXDimNum),
                                                 std::to_string(NUM_THREE)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(expandedXShape.GetDim(0) != expertNum_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expandedX dim[0]",
                                              std::to_string(expandedXShape.GetDim(0)), std::to_string(expertNum_)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            expandedXShape.GetDim(1) != expertCapacity_,
            OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expandedX dim[1]",
                                      std::to_string(expandedXShape.GetDim(1)), std::to_string(expertCapacity_)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(expandedXShape.GetDim(NUM_TWO) != cols_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expandedX dim[2]",
                                              std::to_string(expandedXShape.GetDim(NUM_TWO)), std::to_string(cols_)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(expandedXDimNum != DIM_TWO,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expandedX", std::to_string(expandedXDimNum),
                                                 std::to_string(DIM_TWO)),
                    return ge::GRAPH_FAILED);
        int64_t firstDim = totalLength_;
        firstDim = activeNum_ == 0 ? firstDim : std::min(firstDim, activeNum_);
        OP_CHECK_IF(expandedXShape.GetDim(0) != firstDim,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expandedX dim[0]",
                                              std::to_string(expandedXShape.GetDim(0)), std::to_string(firstDim)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(expandedXShape.GetDim(1) != cols_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expandedX dim[1]",
                                              std::to_string(expandedXShape.GetDim(1)), std::to_string(cols_)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(expandedRowIdxShape.GetDimNum() != DIM_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_row_idx",
                                             std::to_string(expandedRowIdxShape.GetDimNum()), std::to_string(DIM_ONE)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expandedRowIdxShape.GetDim(0) != totalLength_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_row_idx dim[0]",
                                          std::to_string(expandedRowIdxShape.GetDim(0)), std::to_string(totalLength_)),
                return ge::GRAPH_FAILED);

    if (expertTokensNumFlag_) {
        if (expertTokensNumType_ == KEY_VALUE) {
            OP_CHECK_IF(expertTokensCountOrCumsumShape.GetDimNum() != DIM_TWO,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expert_tokens_count_or_cumsum",
                                                     std::to_string(expertTokensCountOrCumsumShape.GetDimNum()),
                                                     std::to_string(DIM_TWO)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(expertTokensCountOrCumsumShape.GetDim(0) != expertNum_,
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_tokens_count_or_cumsum dim[0]",
                                                  std::to_string(expertTokensCountOrCumsumShape.GetDim(0)),
                                                  std::to_string(expertNum_)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(expertTokensCountOrCumsumShape.GetDim(1) != KEY_VALUE_MODE_DIM0_NUM,
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_tokens_count_or_cumsum dim[1]",
                                                  std::to_string(expertTokensCountOrCumsumShape.GetDim(1)),
                                                  std::to_string(KEY_VALUE_MODE_DIM0_NUM)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(expertTokensCountOrCumsumShape.GetDimNum() != DIM_ONE,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expert_tokens_count_or_cumsum",
                                                     std::to_string(expertTokensCountOrCumsumShape.GetDimNum()),
                                                     std::to_string(DIM_ONE)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(expertTokensCountOrCumsumShape.GetDim(0) != (expertEnd_ - expertStart_),
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_tokens_count_or_cumsum dim[0]",
                                                  std::to_string(expertTokensCountOrCumsumShape.GetDim(0)),
                                                  std::to_string(expertEnd_ - expertStart_)),
                        return ge::GRAPH_FAILED);
        }
    }

    if (quantMode_ != STATIC_QUANT && scaleShapePtr_ != nullptr) {
        const gert::Shape expandedScaleShape = context_->GetOutputShape(3)->GetStorageShape();
        OP_LOGI(context_, "expanded_scale shape: %s.", Ops::Base::ToString(expandedScaleShape).c_str());
        size_t expandedScaleDimNum = expandedScaleShape.GetDimNum();
        OP_CHECK_IF(expandedScaleDimNum != DIM_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_scale",
                                                 std::to_string(expandedScaleDimNum), std::to_string(DIM_ONE)),
                    return ge::GRAPH_FAILED);
        if (dropPadMode_ > 0) {
            OP_CHECK_IF(expandedScaleShape.GetDim(0) != expertNum_ * expertCapacity_,
                        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_scale dim[0]",
                                                  std::to_string(expandedScaleShape.GetDim(0)),
                                                  std::to_string(expertNum_ * expertCapacity_)),
                        return ge::GRAPH_FAILED);
        } else {
            int64_t firstDim = totalLength_;
            firstDim = activeNum_ == 0 ? firstDim : std::min(firstDim, activeNum_);
            OP_CHECK_IF(
                expandedScaleShape.GetDim(0) != firstDim,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_scale dim[0]",
                                          std::to_string(expandedScaleShape.GetDim(0)), std::to_string(firstDim)),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::GetShapeAttrsInfo()
{
    OP_LOGI(context_, "TilingContext: %s.", context_->GetNodeName());

    // 获取输入shape
    xShapePtr_ = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr_);

    expertIdxShapePtr_ = context_->GetInputShape(INPUT_EXPERT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr_);

    // 可选输入scale
    scaleShapePtr_ = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    if (scaleShapePtr_ == nullptr) {
        OP_LOGI(context_, "optional input scale is null");
    } else {
        isInputScale_ = 1;
    }
    moeInitRoutingV3TilingData.set_isInputScale(isInputScale_);

    // 可选输入offset
    offsetShapePtr_ = context_->GetOptionalInputShape(INPUT_OFFSET_INDEX);
    if (offsetShapePtr_ == nullptr) {
        OP_LOGI(context_, "optional input offset is null");
    } else {
        isInputOffset_ = 1;
    }
    moeInitRoutingV3TilingData.set_isInputOffset(isInputOffset_);

    // 获取输出shape
    expandedXShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShapePtr_);
    expandedRowIdxShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_ROW_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr_);
    expertTokensCountOrCumsumShapePtr_ = context_->GetOutputShape(OUTPUT_EXPERT_TOKENS_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensCountOrCumsumShapePtr_);
    expandedScaleShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedScaleShapePtr_);

    // 获取属性
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    activeNumPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activeNumPtr_);
    expertCapacityPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertCapacityPtr_);
    expertNumPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr_);
    dropPadModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dropPadModePtr_);
    expertTokensNumTypePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensNumTypePtr_);
    expertTokensNumFlagPtr_ = attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensNumFlagPtr_);
    quantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantModePtr_);
    activeExpertRangeListPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_EXPERT_RANGE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activeExpertRangeListPtr_);
    rowIdxTypePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_ROW_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowIdxTypePtr_);
    return ge::GRAPH_SUCCESS;
}

void MoeInitRountingV3TilingBase::ShowTilingData()
{
    int64_t isFullloadInt = isFullload_ ? 1 : 0;
    OP_LOGI(context_, "isFullload: %ld, gatherFirstFullload: %ld, ep: %ld, countingSortFullLoad: %ld", isFullloadInt,
            gatherFirstFullload_, ep_, countingSortFullLoad_);
}

int64_t MoeInitRountingV3TilingBase::IsGatherFirstFullLoad()
{
    // 判断当前输入条件下，要不要先gather(剔除无效专家)再排序.
    if (ep_ == 0) {
        return 0;
    } else if (n_ >= gatherFirstN && (expertEnd_ - expertStart_) * gatherFirstScale <= expertNum_) {
        return 1;
    }
    return 0;
}

bool MoeInitRountingV3TilingBase::IsFullLoad()
{
    int64_t perCoreTokens = 1;
    if (expertStart_ == 0 && expertEnd_ == expertNum_) {
        ep_ = 0;
        if (quantMode_ != 1) {
            perCoreTokens = n_ / aivNum;
            int64_t remainder = n_ % aivNum;
            // NUM_TWO is Max xRows need add 2 becauseof the left and right row may be another row.
            perCoreTokens = remainder <= 1 ? perCoreTokens + 1 : perCoreTokens + NUM_TWO;
        }
    } else {
        ep_ = 1;
        perCoreTokens = 1;
    }
    moeInitRoutingV3TilingData.set_ep(ep_);

    if (totalLength_ > sortLoopMaxElement || this->dropPadMode_ == 1) {
        return false;
    }

    gatherFirstFullload_ = IsGatherFirstFullLoad();
    moeInitRoutingV3TilingData.set_gatherFirstFullload(gatherFirstFullload_);
    int64_t tileLength = Align(this->totalLength_, int64_t(sizeof(int32_t)));
    int64_t sortNum = Ops::Base::CeilDiv(tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

    int64_t sortSpace = sortNum * sizeof(int32_t) * ONE_CORE_SORT_BUFFER;
    int64_t rowIdxSpace = sortNum * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace =
        Ops::Base::CeilDiv(this->expertNum_ * int64_t(sizeof(int64_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * NUM_TWO;
    int64_t gatherSpace = Ops::Base::CeilDiv(cols_ * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * perCoreTokens;
    int64_t remainUb = aicoreParams_.ubSize - sortSpace - rowIdxSpace - expertSpace - LENGTH_1024;

    if (quantMode_ == -1) {
        remainUb -= (gatherSpace + ONE_BLOCK_BYTE);
    } else if (quantMode_ == 0) {
        int64_t quantSpace = 0;
        int64_t xAlignedCount = Align(this->cols_, int64_t(sizeof(int8_t)));
        quantSpace = xAlignedCount * STATIC_QUANT_FULLLOAD_COLS_BUFFER * perCoreTokens;
        remainUb -= (gatherSpace + quantSpace);
    } else {
        int64_t quantSpace =
            Ops::Base::CeilDiv(cols_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER;
        int64_t scaleOutSpace = ONE_BLOCK_BYTE * NUM_TWO;
        remainUb -= (quantSpace + scaleOutSpace);
    }
    return remainUb > 0;
}

bool MoeInitRountingV3TilingBase::ShouldEnableCountingSortByShape() const
{
    int64_t actualExpertNum = expertEnd_ - expertStart_;
    bool expertNumValid = expertNum_ <= 1024;
    bool dropPadValid =
        (dropPadMode_ == DROP_PAD) || (dropPadMode_ == DROP_LESS && actualExpertNum * NUM_TWO <= expertNum_);
    bool quantValid = quantMode_ != DYNAMIC_QUANT;
    bool excludeNonDropPadSmallN = !(n_ <= 2560 && dropPadMode_ == DROP_LESS && quantMode_ == STATIC_QUANT);
    bool excludeDropPadTinyN = !(dropPadMode_ == DROP_PAD && n_ < 512);
    return expertNumValid && dropPadValid && quantValid && excludeNonDropPadSmallN && excludeDropPadTinyN;
}

void MoeInitRountingV3TilingBase::ComputeCountingSortTiling()
{
    int64_t actualExpertNum = expertEnd_ - expertStart_;
    if (n_ < COUNTING_SORT_MIN_N || n_ > COUNTING_SORT_MAX_N) {
        return;
    }

    if (n_ <= COUNTING_SORT_THRESHOLD && rowIdxTytpe_ == SCATTER && ep_ == 1 && expertNum_ == ASSIST_NUM &&
        (expertEnd_ - expertStart_) <= PERFORMANCE_MODE_RANGE_MAX && quantMode_ == UN_QUANT &&
        k_ == PERFORMANCE_MODE_TOP_K && dropPadMode_ == DROP_LESS && expertTokensNumType_ == COUNT &&
        actualExpertNum <= COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM && actualExpertNum > 0) {
        return;
    }

    if (!ShouldEnableCountingSortByShape()) {
        return;
    }

    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize);
    int64_t filterPerCoreTokens = Ops::Base::CeilDiv(n_, aivNum);
    int64_t filterNeedCoreNum = Ops::Base::CeilDiv(n_, filterPerCoreTokens);
    int64_t expertCountStride = Ops::Base::CeilAlign(actualExpertNum, ONE_BLOCK_ELEMENT);
    int64_t coreEntries = filterPerCoreTokens * k_;
    int64_t fullLoadUB = EstimateCountingSortFullLoadUB(filterPerCoreTokens, coreEntries, expertCountStride,
                                                        filterNeedCoreNum, actualExpertNum);

    countingSortFullLoad_ =
        (fullLoadUB <= ubSize - LENGTH_1024 && ep_ == 1 && quantMode_ == UN_QUANT && k_ == PERFORMANCE_MODE_TOP_K &&
         dropPadMode_ == DROP_LESS && expertTokensNumType_ == COUNT &&
         actualExpertNum <= COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM && actualExpertNum > 0) ?
            1 :
            0;
    int64_t filterChunkSize = (countingSortFullLoad_ == 1) ? 0 : COUNTING_SORT_FILTER_CHUNK_SIZE;

    if (countingSortFullLoad_ == 1) {
        countingSortWorkspaceSize_ =
            static_cast<size_t>(filterNeedCoreNum) * static_cast<size_t>(expertCountStride) * sizeof(int32_t);
        moeInitRoutingV3TilingData.gatherOutComputeParamsOp.set_needCoreNum(filterNeedCoreNum);
        moeInitRoutingV3TilingData.gatherOutComputeParamsOp.set_perCoreIndicesElements(filterPerCoreTokens);
    } else {
        ComputeCountingSortCutOriginTiling(filterNeedCoreNum, filterPerCoreTokens, expertCountStride, actualExpertNum);
    }

    OP_LOGI(context_,
            "CountingSort tiling: countingSortFullLoad=%ld, filterNeedCoreNum=%ld, "
            "filterPerCoreTokens=%ld, filterChunkSize=%ld, fullLoadUB=%ld, ubSize=%ld, "
            "countingSortWorkspaceSize=%zu",
            countingSortFullLoad_, filterNeedCoreNum, filterPerCoreTokens, filterChunkSize, fullLoadUB, ubSize,
            countingSortWorkspaceSize_);
}

int64_t MoeInitRountingV3TilingBase::EstimateCountingSortFullLoadUB(int64_t coreTokenNum, int64_t coreEntries,
                                                                    int64_t expertCountStride,
                                                                    int64_t filterNeedCoreNum, int64_t actualExpertNum)
{
    int64_t colsAligned =
        Ops::Base::CeilDiv(cols_ * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE / inuptXDtypeSize_;
    int64_t scaleSlotSize = ONE_BLOCK_BYTE / SIZE_FP32;
    int64_t ub = 0;

    ub += Ops::Base::CeilAlign(coreTokenNum * colsAligned * inuptXDtypeSize_, ONE_BLOCK_BYTE);
    ub += Ops::Base::CeilAlign(coreEntries * SIZE_INT32, ONE_BLOCK_BYTE);
    if (isInputScale_) {
        ub += Ops::Base::CeilAlign(coreTokenNum * scaleSlotSize * SIZE_FP32, ONE_BLOCK_BYTE);
    }
    ub += Ops::Base::CeilAlign(expertCountStride * SIZE_INT32, ONE_BLOCK_BYTE);
    ub += Ops::Base::CeilAlign(filterNeedCoreNum * expertCountStride * SIZE_INT32, ONE_BLOCK_BYTE);
    ub += Ops::Base::CeilAlign(actualExpertNum * static_cast<int64_t>(sizeof(int64_t)), ONE_BLOCK_BYTE);
    ub += Ops::Base::CeilAlign(expertCountStride * SIZE_INT32, ONE_BLOCK_BYTE);
    ub += Ops::Base::CeilAlign(coreEntries * NUM_TWO * SIZE_INT32, ONE_BLOCK_BYTE);

    int64_t entriesAligned =
        Ops::Base::CeilDiv(coreEntries, COUNTING_SORT_ONE_REPEAT_COMPARE_NUM) * COUNTING_SORT_ONE_REPEAT_COMPARE_NUM;
    int64_t maskBytes = Ops::Base::CeilAlign(Ops::Base::CeilDiv(entriesAligned, ONE_BLOCK_ELEMENT), ONE_BLOCK_BYTE);
    ub += Ops::Base::CeilAlign(entriesAligned * SIZE_FP32, ONE_BLOCK_BYTE);
    ub += maskBytes * NUM_THREE;
    ub += Ops::Base::CeilAlign(entriesAligned * SIZE_INT32, ONE_BLOCK_BYTE) * NUM_THREE;

    return ub;
}

void MoeInitRountingV3TilingBase::ComputeCountingSortCutOriginTiling(int64_t filterNeedCoreNum,
                                                                     int64_t filterPerCoreTokens,
                                                                     int64_t expertCountStride, int64_t actualExpertNum)
{
    int64_t chunkAligned = Ops::Base::CeilAlign(COUNTING_SORT_FILTER_CHUNK_SIZE, ONE_BLOCK_ELEMENT);
    int64_t persistentSize = Ops::Base::CeilAlign(expertCountStride * SIZE_INT32, ONE_BLOCK_BYTE);
    // Phase A: chunked filtering
    int64_t phaseASize = Ops::Base::CeilAlign(chunkAligned * SIZE_INT32, ONE_BLOCK_BYTE);
    phaseASize += Ops::Base::CeilAlign(chunkAligned * SIZE_FP32, ONE_BLOCK_BYTE);
    int64_t maskBytesA =
        Ops::Base::CeilAlign(Ops::Base::CeilDiv(chunkAligned, ONE_BLOCK_ELEMENT), static_cast<int64_t>(sizeof(int8_t)));
    phaseASize += maskBytesA * NUM_THREE + Ops::Base::CeilAlign(chunkAligned * SIZE_INT32, ONE_BLOCK_BYTE) * NUM_THREE;
    phaseASize += Ops::Base::CeilAlign(chunkAligned * NUM_TWO * SIZE_INT32, ONE_BLOCK_BYTE);

    // Phase B
    int64_t phaseBSize = Ops::Base::CeilAlign(filterNeedCoreNum * expertCountStride * SIZE_INT32, ONE_BLOCK_BYTE);
    phaseBSize += Ops::Base::CeilAlign(expertCountStride * SIZE_INT32, ONE_BLOCK_BYTE);
    phaseBSize += Ops::Base::CeilAlign(actualExpertNum * static_cast<int64_t>(sizeof(int64_t)), ONE_BLOCK_BYTE);

    // Phase C: compute colsLoops/perLoopCols based on remaining UB space
    // When actualExpertNum is large (up to 10240), persistentSize grows significantly,
    // so compute maxPerLoopCols from actual Phase C budget rather than fixed constant
    int64_t phaseCBudget = static_cast<int64_t>(aicoreParams_.ubSize) - persistentSize - ONE_BLOCK_BYTE;

    // Fixed Phase C overhead (independent of perLoopCols)
    int64_t phaseCFixedOverhead = Ops::Base::CeilAlign(2048 * NUM_TWO * SIZE_INT32, ONE_BLOCK_BYTE); // pairsBuf
    phaseCFixedOverhead += ONE_BLOCK_BYTE;                                                           // idxBuf (32B)
    if (dropPadMode_ != DROP_PAD) {
        // Non-droppad: 3 stack arrays newPosArr/srcRowArr/origFlatIdxArr (int32_t[2048] each)
        phaseCFixedOverhead += NUM_THREE * static_cast<int64_t>(2048) * SIZE_INT32;
    }

    // Per-element byte cost for all perLoopCols-dependent buffers
    // 2 xBuf slots (double buffer) + quant buffers per mode
    int64_t colBytesPerElement = NUM_TWO * inuptXDtypeSize_;
    if (quantMode_ == STATIC_QUANT) {
        colBytesPerElement += SIZE_FP32 + SIZE_INT8 * NUM_TWO; // quantFloat + quantInt8
    } else if (isInputScale_) {
        // unquant with inputScale: 2 scale slots (fixed 32B each)
        phaseCFixedOverhead += Ops::Base::CeilAlign(ONE_BLOCK_ELEMENT * SIZE_FP32, ONE_BLOCK_BYTE) * NUM_TWO;
    }

    // Compute max perLoopCols from remaining budget (with alignment margin)
    int64_t remainingForCols = phaseCBudget - phaseCFixedOverhead - LENGTH_1024;
    int64_t dynamicMaxPerLoopCols = remainingForCols / colBytesPerElement;
    // Align down to block boundary for element alignment
    dynamicMaxPerLoopCols =
        (dynamicMaxPerLoopCols * inuptXDtypeSize_ / ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE / inuptXDtypeSize_;
    int64_t maxPerLoopColsBytes =
        std::min(dynamicMaxPerLoopCols * inuptXDtypeSize_, COUNTING_SORT_MAX_PER_LOOP_COLS_BYTES);

    int64_t csPerLoopCols =
        (cols_ * inuptXDtypeSize_ > maxPerLoopColsBytes) ? (maxPerLoopColsBytes / inuptXDtypeSize_) : cols_;
    int64_t csColsLoops = Ops::Base::CeilDiv(cols_, csPerLoopCols);
    int64_t csLastLoopCols = cols_ - (csColsLoops - 1) * csPerLoopCols;
    int64_t csPerLoopColsAligned =
        Ops::Base::CeilDiv(csPerLoopCols * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE / inuptXDtypeSize_;

    if (csColsLoops > 1 && quantMode_ != UN_QUANT) {
        countingSortFullLoad_ = -1;
        return;
    }

    // Phase C size (accurate estimation matching kernel layout)
    int64_t phaseCSize = phaseCFixedOverhead;
    phaseCSize += Ops::Base::CeilAlign(csPerLoopColsAligned * inuptXDtypeSize_, ONE_BLOCK_BYTE) * NUM_TWO;
    if (quantMode_ == STATIC_QUANT) {
        phaseCSize += Ops::Base::CeilAlign(csPerLoopColsAligned * SIZE_FP32, ONE_BLOCK_BYTE);
        phaseCSize += Ops::Base::CeilAlign(csPerLoopColsAligned * SIZE_INT8, ONE_BLOCK_BYTE) * NUM_TWO;
        phaseCSize += ONE_BLOCK_BYTE;
    }

    int64_t totalUB = persistentSize + std::max({phaseASize, phaseBSize, phaseCSize});
    OP_LOGI(context_, "totalUB is: %ld", totalUB);

    // Set tiling params
    auto gatherOp = &moeInitRoutingV3TilingData.gatherOutComputeParamsOp;
    gatherOp->set_needCoreNum(filterNeedCoreNum);
    gatherOp->set_perCoreIndicesElements(filterPerCoreTokens);
    gatherOp->set_colsLoops(csColsLoops);
    gatherOp->set_perLoopCols(csPerLoopCols);
    gatherOp->set_lastLoopCols(csLastLoopCols);

    // Workspace: pairs + expertCount
    int64_t lastCoreTokens = n_ - (filterNeedCoreNum - 1) * filterPerCoreTokens;
    int64_t maxCoreEntries = std::max(filterPerCoreTokens, lastCoreTokens) * k_;
    int64_t pairsPerCore = Ops::Base::CeilAlign(maxCoreEntries, ONE_BLOCK_ELEMENT) * NUM_TWO;
    size_t pairsWorkspaceSize =
        static_cast<size_t>(filterNeedCoreNum) * static_cast<size_t>(pairsPerCore) * sizeof(int32_t);
    size_t expertCountWorkspaceSize =
        static_cast<size_t>(filterNeedCoreNum) * static_cast<size_t>(expertCountStride) * sizeof(int32_t);
    countingSortWorkspaceSize_ = pairsWorkspaceSize + expertCountWorkspaceSize;
}

bool MoeInitRountingV3TilingBase::IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() const
{
    OP_LOGI(context_, "Begin IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() ...");
    bool result = false;

    // 性能模板： 当前支持 ((1, 7168), (1, 8),(256, 7168),None) ('bfloat16', 'int32','float32','float32')
    // expert_range [0,256), quant_mode=DYNAMIC_QUANT
    const gert::Shape performXShape_X_1_7168 = gert::Shape({1, 7168});
    const gert::Shape performExpertIdxShape_X_1_7168 = gert::Shape({1, 8});
    const gert::Shape performScaleShape_X_1_7168 = gert::Shape({256, 7168});

    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr_);
    if (nullptr == scaleShapePtr_) {
        result = false;
    } else if (xShapePtr_->GetStorageShape() == performXShape_X_1_7168 &&
               expertIdxShapePtr_->GetStorageShape() == performExpertIdxShape_X_1_7168 &&
               scaleShapePtr_->GetStorageShape() == performScaleShape_X_1_7168 && offsetShapePtr_ == nullptr &&
               context_->GetInputDesc(INPUT_X_INDEX)->GetDataType() == ge::DT_BF16 && expertStart_ == 0 &&
               expertEnd_ == ASSIST_NUM && quantMode_ == DYNAMIC_QUANT && expertTokensNumType_ == KEY_VALUE) {
        result = true;
    }
    OP_LOGI(context_, "End IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168() ...");
    return result;
}

PerformanceMode MoeInitRountingV3TilingBase::GetPerformanceMode() const
{
    PerformanceMode result = PerformanceMode::COMMON;
    if (expertNum_ != ASSIST_NUM || (expertEnd_ - expertStart_) > PERFORMANCE_MODE_RANGE_MAX ||
        n_ < PERFORMANCE_MODE_BS_MIN || n_ > PERFORMANCE_MODE_BS_MAX || k_ != PERFORMANCE_MODE_TOP_K) {
        return result;
    }

    // Judge performance mode according to totalLength_
    if (totalLength_ < PERFORMANCE_MODE_MAX_ONE_CORE_GATHER) {
        OP_LOGI(context_, "totalLength_: %ld, PerformanceMode::ONE_CORE_GATHER_SORT", totalLength_);
        result = PerformanceMode::ONE_CORE_GATHER_SORT;
    } else if (totalLength_ <= PERFORMANCE_MODE_MAX_BATCH_SIZE_TOP_K) {
        OP_LOGI(context_, "totalLength_: %ld, PerformanceMode::MULTI_CORE_GATHER_SORT", totalLength_);
        result = PerformanceMode::MULTI_CORE_GATHER_SORT;
    }
    return result;
}

ge::graphStatus MoeInitRountingV3TilingBase::CheckDtype()
{
    auto inputXDtype_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();
    OP_CHECK_IF(inputXDtype_ != ge::DT_INT8 && inputXDtype_ != ge::DT_FLOAT16 && inputXDtype_ != ge::DT_BF16 &&
                    inputXDtype_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "inputXDtype_",
                                          Ops::Base::ToString(inputXDtype_).c_str(), "INT8, FLOAT16, BF16 or FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputXDtype_ == ge::DT_INT8 && quantMode_ != UN_QUANT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "inputXDtype_",
                                                      Ops::Base::ToString(inputXDtype_).c_str(),
                                                      "quantization is not supported when input_X is INT8"),
                return ge::GRAPH_FAILED);

    auto expertIdxDtype_ = context_->GetInputDesc(INPUT_EXPERT_IDX_INDEX)->GetDataType();
    OP_CHECK_IF(expertIdxDtype_ != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expertIdxDtype_",
                                          Ops::Base::ToString(expertIdxDtype_).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (quantMode_ == STATIC_QUANT) {
        auto scaleDtype_ = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetDataType();
        OP_CHECK_IF(scaleDtype_ != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scaleDtype_",
                                              Ops::Base::ToString(scaleDtype_).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);

        auto offsetDtype_ = context_->GetOptionalInputDesc(INPUT_OFFSET_INDEX)->GetDataType();
        OP_CHECK_IF(offsetDtype_ != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "offsetDtype_",
                                              Ops::Base::ToString(offsetDtype_).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    } else {
        if (scaleShapePtr_ != nullptr) {
            auto scaleDtype_ = context_->GetOptionalInputDesc(INPUT_SCALE_INDEX)->GetDataType();
            OP_CHECK_IF(scaleDtype_ != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scaleDtype_",
                                                  Ops::Base::ToString(scaleDtype_).c_str(), "FLOAT"),
                        return ge::GRAPH_FAILED);
        }
    }

    auto expandedXDtype_ = context_->GetOutputDesc(OUTPUT_EXPANDED_X_INDEX)->GetDataType();
    OP_CHECK_IF(expandedXDtype_ != ge::DT_INT8 && expandedXDtype_ != ge::DT_FLOAT16 && expandedXDtype_ != ge::DT_BF16 &&
                    expandedXDtype_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expandedXDtype_",
                                          Ops::Base::ToString(expandedXDtype_).c_str(), "INT8, FLOAT16, BF16 or FLOAT"),
                return ge::GRAPH_FAILED);

    auto expandedRowIdxDtype_ = context_->GetOutputDesc(OUTPUT_EXPANDED_ROW_IDX_INDEX)->GetDataType();
    OP_CHECK_IF(expandedRowIdxDtype_ != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expandedRowIdxDtype_",
                                          Ops::Base::ToString(expandedRowIdxDtype_).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    auto expertTokensCountOrCusumDtype_ = context_->GetOutputDesc(OUTPUT_EXPERT_TOKENS_COUNT_INDEX)->GetDataType();
    OP_CHECK_IF(expertTokensCountOrCusumDtype_ != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expertTokensCountOrCusumDtype_",
                                          Ops::Base::ToString(expertTokensCountOrCusumDtype_).c_str(), "INT64"),
                return ge::GRAPH_FAILED);

    if (quantMode_ == DYNAMIC_QUANT || (quantMode_ == UN_QUANT && scaleShapePtr_ != nullptr)) {
        auto expandedScaleDtype_ = context_->GetOutputDesc(OUTPUT_EXPANDED_SCALE_INDEX)->GetDataType();
        OP_CHECK_IF(expandedScaleDtype_ != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expandedScaleDtype_",
                                              Ops::Base::ToString(expandedScaleDtype_).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::DoOpTiling()
{
    auto ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 空tensor快速返回，跳过后续tiling计算
    if (isEmptyTensor_) {
        return ge::GRAPH_SUCCESS;
    }

    if (IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168()) {
        aivNum = totalLength_;
    }

    sortLoopMaxElement = (aicoreParams_.ubSize - aivNum * ONE_BLOCK_BYTE) / (NUM_FOUR * NUM_TWO * NUM_FOUR) /
                         SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;

    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4ExpertTokensCountCompute();
    Tiling4SrcToDstCompute();
    Tiling4SrcToDstDropPadCompute();
    Tiling4GatherOutCompute();
    isFullload_ = IsFullLoad();
    if (!isFullload_ && !IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168()) {
        ComputeCountingSortTiling();
    }
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRountingV3TilingBase::GetTilingKey() const
{
    if (isEmptyTensor_) {
        return EMPTY_TENSOR_TILINGKEY;
    }

    if (isFullload_) {
        if (quantMode_ == UN_QUANT) {
            return UNQUANTIZED_FULLLOAD_TILINGKEY;
        } else if (quantMode_ == STATIC_QUANT) {
            return STATIC_QUANT_FULLLOAD_TILINGKEY;
        } else {
            return (DYNAMIC_QUANT_FULLLOAD_TILINGKEY + ep_ * DYNAMIC_QUANT_EPFULLLOAD_TILINGKEY +
                    smoothType_ * DYNAMIC_QUANT_SMOOTHTYPE_FULLLOAD_TILINGKEY);
        }
    } else if (IsPerformanceMode_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168()) {
        return PERFORMANCE_TILINGKEY_X_1_7168_EXPERT_IDX_1_8_SCALE_256_7168;
    } else if (countingSortFullLoad_ == 1 ||
               (PerformanceMode::ONE_CORE_GATHER_SORT == GetPerformanceMode() && quantMode_ == UN_QUANT &&
                rowIdxTytpe_ == SCATTER && expertTokensNumType_ == COUNT && countingSortFullLoad_ == -1)) {
        uint64_t sortMode = NUM_TWO;
        return static_cast<uint64_t>(TILINGKEY_BASE + sortMode * SORT_CORE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(quantMode_ + 1) * QUANT_MODE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(rowIdxTytpe_) * ROWIDX_TYPE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(dropPadMode_) * DROP_MODE_TILINGKEY_BASE);
    } else if (countingSortFullLoad_ == 0 ||
               (PerformanceMode::MULTI_CORE_GATHER_SORT == GetPerformanceMode() && quantMode_ == UN_QUANT &&
                rowIdxTytpe_ == SCATTER && expertTokensNumType_ == COUNT && countingSortFullLoad_ == -1)) {
        uint64_t sortMode = 3;
        return static_cast<uint64_t>(TILINGKEY_BASE + sortMode * SORT_CORE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(rowIdxTytpe_) * ROWIDX_TYPE_TILINGKEY_BASE);
    } else {
        return static_cast<uint64_t>(TILINGKEY_BASE + static_cast<uint64_t>(sortMode_) * SORT_CORE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(quantMode_ + 1) * QUANT_MODE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(rowIdxTytpe_) * ROWIDX_TYPE_TILINGKEY_BASE +
                                     static_cast<uint64_t>(dropPadMode_) * DROP_MODE_TILINGKEY_BASE);
    }
}

ge::graphStatus MoeInitRountingV3TilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    if (isEmptyTensor_) {
        workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
        return ge::GRAPH_SUCCESS;
    }
    size_t sortWorkspaceSize =
        sizeof(float) * static_cast<size_t>(totalLength_ * NUM_TWO * NUM_THREE); // 排序需要的空间
    size_t coreSyncWorkspaceSize =
        moeInitRoutingV3TilingData.get_coreNum() * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    size_t scatterWorkspaceSize = sizeof(int32_t) * static_cast<size_t>(totalLength_);
    size_t expertIdxValueWorkspaceSize = sizeof(int32_t) * static_cast<size_t>(aivNum) * 2U;
    size_t expertTokensCountWorkspaceSize = sizeof(int32_t) * static_cast<size_t>((expertEnd_ - expertStart_));
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, static_cast<int64_t>(sizeof(int32_t)));
    int64_t quantTempWorkspaceSize = aivNum * cols_ * static_cast<int64_t>(sizeof(float));
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + expertTokensCountWorkspaceSize +
                     expertTokenTotalCountWorkspace + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    if (quantMode_ == DYNAMIC_QUANT) {
        workspaceSize_ += quantTempWorkspaceSize;
    }
    if (dropPadMode_ == DROP_PAD) {
        workspaceSize_ += expertIdxValueWorkspaceSize;
    }
    // Counting sort may need additional workspace
    if (countingSortWorkspaceSize_ > workspaceSize_) {
        workspaceSize_ = countingSortWorkspaceSize_;
    }
    OP_LOGI(context_, "Allocate workspaceSize is: %ld.", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3TilingBase::PostTiling()
{
    if (isEmptyTensor_) {
        context_->SetBlockDim(1);
    } else {
        context_->SetBlockDim(aivNum);
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeInitRoutingV3TilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeInitRoutingV3TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void MoeInitRountingV3TilingBase::Tinlig4VBSOneCoreCompute(MoeV3VBSComputeTilingData* tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength_);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeInitRountingV3TilingBase::Tinlig4VBSMultiCoreCompute(MoeV3VBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, sortLoopMaxElement);    // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(NUM_FOUR, CeilLog4(needCoreNum))); // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                                   // 不能超过物理核数

    OP_CHECK_IF(needCoreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "needCoreNum",
                                                      std::to_string(needCoreNum), "needCoreNum cannot be 0"),
                return;);
    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (totalLength_ / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength_ - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreElements);
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

        tilingData->set_lastCoreElements(totalLength_ -
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
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "lastCoreLastLoopElements",
                                                      std::to_string(tilingData->get_lastCoreLastLoopElements()),
                                                      "vbs tiling failed"),
                ;);
}

void MoeInitRountingV3TilingBase::Tiling4VBSCompute()
{
    if (totalLength_ <= sortLoopMaxElement) { // 排序只用到一个核排序
        sortMode_ = 0;
    } else {
        sortMode_ = 1;
    }

    auto tilingData = &moeInitRoutingV3TilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (sortMode_ == 0UL) { // 只用到一个核
        Tinlig4VBSOneCoreCompute(tilingData);
        return;
    }
    Tinlig4VBSMultiCoreCompute(tilingData);
}

void MoeInitRountingV3TilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingV3TilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingV3TilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeInitRountingV3TilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRountingV3TilingBase::Tiling4ExpertTokensCountCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.expertTokensCountTilingDataOp;
    int64_t totalElements = moeInitRoutingV3TilingData.get_n() * moeInitRoutingV3TilingData.get_k();
    int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, aivNum);
    int64_t needCoreNum = Ops::Base::CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElements);

    int64_t expertNumElement = (moeInitRoutingV3TilingData.get_expertTokensNumType() != KEY_VALUE) ?
                                   moeInitRoutingV3TilingData.get_actualExpertNum() :
                                   (moeInitRoutingV3TilingData.get_actualExpertNum() + 1) * DIM_TWO;

    int64_t maxElementsPerLoop =
        (static_cast<int64_t>(aicoreParams_.ubSize) -
         Ops::Base::CeilAlign(expertNumElement, ONE_BLOCK_BYTE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         ONE_BLOCK_BYTE) /
        static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = Ops::Base::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    tilingData->set_perCoreLoops(perCoreLoops);
    tilingData->set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData->set_perCoreLastLoopElements(perCoreLastLoopElements);

    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = Ops::Base::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    tilingData->set_lastCoreLoops(lastCoreLoops);
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(lastCoreLastLoopElements);

    OP_LOGI(context_,
            "ExpertTokensCountCompute Tilingdata, needCoreNum is: %ld, perCoreElements is: %ld, lastCoreElements is: "
            "%ld, maxElementsPerLoop is: %ld, perCoreLoops is: %ld, perCorePerLoopElements is: %ld, "
            "perCoreLastLoopElements "
            "is: %ld, lastCoreLoops is: %ld, lastCorePerLoopElements is: %ld, lastCoreLastLoopElements is: %ld.",
            needCoreNum, perCoreElements, lastCoreElements, maxElementsPerLoop, perCoreLoops, perCorePerLoopElements,
            perCoreLastLoopElements, lastCoreLoops, lastCorePerLoopElements, lastCoreLastLoopElements);
}

// 增加SrcToDstCapacity切分
void MoeInitRountingV3TilingBase::Tiling4SrcToDstDropPadCompute()
{
    if (quantMode_ == DYNAMIC_QUANT && dropPadMode_ == DROP_PAD) {
        MoeInitRountingV3TilingBase::Tiling4SrcToDstDropPadDynamicCompute();
        return;
    }

    auto tilingData = &moeInitRoutingV3TilingData.srcToDstDropPadParamsOp;

    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength_, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t cols = moeInitRoutingV3TilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength_ - perCoreRows * (needCoreNum - 1);
    tilingData->set_lastCoreRows(lastCoreRows);
    int64_t inuptXDtypeSize = inuptXDtypeSize_ == SIZE_INT8 ? SIZE_INT16 : inuptXDtypeSize_;

    int64_t rowSize = (perCoreRows * sizeof(int32_t) * NUM_TWO + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE *
                      ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) { // 一行能够全载
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                     static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize / ONE_BLOCK_BYTE *
                          ONE_BLOCK_BYTE;
        }
        tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
        tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
        tilingData->set_colLoops((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLoops((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLoops((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}

void MoeInitRountingV3TilingBase::SetGatherTilingData(MoeV3SrcToDstCapacityComputeTilingData* tilingData,
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

void MoeInitRountingV3TilingBase::SetGatherTilingDataCols(MoeV3SrcToDstCapacityComputeTilingData* tilingData,
                                                          int64_t baseMaxCols, int64_t cols)
{
    tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
    tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
    tilingData->set_colLoops(baseMaxCols == 0 ? 0 : (cols + baseMaxCols - 1) / baseMaxCols);
}

void MoeInitRountingV3TilingBase::SetGatherTilingDataRows(MoeV3SrcToDstCapacityComputeTilingData* tilingData,
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

void MoeInitRountingV3TilingBase::Tiling4SrcToDstDropPadDynamicCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.srcToDstDropPadDynamicParamsOp;

    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength_, aivNum);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_needCoreNum(Ops::Base::CeilDiv(totalLength_, perCoreRows));
    int64_t cols = moeInitRoutingV3TilingData.get_cols();
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength_ - perCoreRows * (tilingData->get_needCoreNum() - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize = AlignBytes(perCoreRows, static_cast<int64_t>(sizeof(int32_t))) * NUM_FOUR;
    int64_t colSize = AlignBytes(cols, static_cast<int64_t>(sizeof(int8_t))) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
    int64_t scaleSize = DYNAMIC_QUANT_SCALE_SIZE_64;
    if (rowSize + colSize + scaleSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        SetGatherTilingData(tilingData, perCoreRows, lastCoreRows, cols);
    } else {
        int64_t baseMaxCols = MAX_COLS_DYNAMIC_QUANT;
        int64_t totalColSize =
            AlignBytes(baseMaxCols, static_cast<int64_t>(sizeof(int8_t))) * DYNAMIC_QUANT_SRC_TO_DST_BUFFER;
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

void MoeInitRountingV3TilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.srcToDstComputeParamsOp;

    int64_t useCore = aivNum;
    // ubsize减去32B对齐保留空间
    int64_t remainUbSize = aicoreParams_.ubSize - ASSIST_NUM * sizeof(int32_t) - ONE_BLOCK_BYTE * (ASSIST_NUM + 1);
    int64_t perLoopMaxElements = remainUbSize / (ONE_BLOCK_BYTE + SIZE_INT32);
    int64_t perCoreElements = Ops::Base::CeilDiv(totalLength_, useCore);
    if (perCoreElements <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreElements);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreElements = totalLength_ - perCoreElements * (needCoreNum - 1);

    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElements);
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, perLoopMaxElements);
    int64_t perCorePerLoopElements = Ops::Base::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, perLoopMaxElements);
    int64_t lastCorePerLoopElements = Ops::Base::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    tilingData->set_perCoreLoops(perCoreLoops);
    tilingData->set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData->set_perCoreLastLoopElements(perCoreLastLoopElements);
    tilingData->set_lastCoreLoops(lastCoreLoops);
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(lastCoreLastLoopElements);
}

void MoeInitRountingV3TilingBase::Tiling4GatherOutCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.gatherOutComputeParamsOp;
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivNum);
    if (perCoreIndicesElements <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = moeInitRoutingV3TilingData.get_cols();
    int64_t colMultiple = NUM_TWO * inuptXDtypeSize_;
    int64_t rowMultiple = NUM_TWO;
    if (quantMode_ == DYNAMIC_QUANT) {
        colMultiple = DYNAMIC_QUANT_COLS_BUFFER;
        rowMultiple = NUM_FOUR;
    }
    if (quantMode_ == STATIC_QUANT) {
        colMultiple = SIZE_INT8 * NUM_TWO + SIZE_FP32 + SIZE_INT16 + inuptXDtypeSize_ * NUM_TWO;
        rowMultiple = NUM_TWO;
    }
    int64_t perLoopMaxIndicesElements =
        (static_cast<int64_t>(aicoreParams_.ubSize) - Align(perLoopCols, inuptXDtypeSize_) * colMultiple -
         ONE_BLOCK_BYTE * NUM_TWO) /
        rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = Ops::Base::CeilDiv(perLoopCols, NUM_TWO);
        perLoopMaxIndicesElements = (static_cast<int64_t>(aicoreParams_.ubSize) -
                                     Align(perLoopCols, inuptXDtypeSize_) * colMultiple - ONE_BLOCK_BYTE * NUM_TWO) /
                                    rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        OP_LOGI(context_, "perLoopCols is: %ld, perLoopMaxIndicesElements is: %ld", perLoopCols,
                perLoopMaxIndicesElements);
    }
    int64_t colsLoops = Ops::Base::CeilDiv(moeInitRoutingV3TilingData.get_cols(), perLoopCols);
    int64_t lastLoopCols = moeInitRoutingV3TilingData.get_cols() - (colsLoops - 1) * perLoopCols;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreIndicesElements(perCoreIndicesElements);
    tilingData->set_lastCoreIndicesElements(lastCoreIndicesElements);
    tilingData->set_colsLoops(colsLoops);
    tilingData->set_perLoopCols(perLoopCols);
    tilingData->set_lastLoopCols(lastLoopCols);

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    tilingData->set_perCoreIndicesLoops(perCoreIndicesLoops);
    tilingData->set_perCorePerLoopIndicesElements(perCorePerLoopIndicesElements);
    tilingData->set_perCoreLastLoopIndicesElements(perCoreLastLoopIndicesElements);

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    tilingData->set_lastCoreIndicesLoops(lastCoreIndicesLoops);
    tilingData->set_lastCorePerLoopIndicesElements(lastCorePerLoopIndicesElements);
    tilingData->set_lastCoreLastLoopIndicesElements(lastCoreLastLoopIndicesElements);

    OP_LOGI(
        context_,
        "GatherOut Tilingdata, needCoreNum is: %ld, perCoreIndicesElements is: %ld, lastCoreIndicesElements is: %ld, "
        "colsLoops is: %ld, perLoopCols is: %ld, lastLoopCols is: %ld, perCoreIndicesLoops is: %ld, "
        "perCorePerLoopIndicesElements is: %ld, perCoreLastLoopIndicesElements is: %ld, lastCoreIndicesLoops is: "
        "%ld, lastCorePerLoopIndicesElements is: "
        "%ld, lastCoreLastLoopIndicesElements is: %ld.",
        needCoreNum, perCoreIndicesElements, lastCoreIndicesElements, colsLoops, perLoopCols, lastLoopCols,
        perCoreIndicesLoops, perCorePerLoopIndicesElements, perCoreLastLoopIndicesElements, lastCoreIndicesLoops,
        lastCorePerLoopIndicesElements, lastCoreLastLoopIndicesElements);
}

REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV3, MoeInitRountingV3TilingBase, 10000); // If not 950, fallback to this.
} // namespace optiling
