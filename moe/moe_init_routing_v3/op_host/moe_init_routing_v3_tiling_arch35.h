/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v3_tiling_arch35.h
 * \brief
 */

#ifndef MOE_INIT_ROUTING_V3_TILING_ARCH35_H
#define MOE_INIT_ROUTING_V3_TILING_ARCH35_H

#include <cmath>
#include <sstream>
#include <string>
#include <unordered_set>
#include "register/op_def_registry.h"
#include "moe_init_routing_v3_tiling.h"
#include "../op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h"
#include "op_host/tiling_util.h"

#define MIRV3_CHECK_GE_RET(expr) \
    if (ge::graphStatus ret = (expr); ret != ge::GRAPH_SUCCESS) { \
        return ret; \
    }

namespace optiling {
using Ops::Transformer::OpTiling::TilingBaseClass;
constexpr int64_t SIMT_DCACHE_SIZE = 64 * 1024LL; // UB要给SIMT预留64k的DCache空间，然后要用SetLocalMemSize()
constexpr int64_t SORT_API_MAX_ELEM = 32 * 255LL; // AscendC::Sort全排序模式最多支持一次排序(32*255rep)个元素
constexpr int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
constexpr int64_t MX_QUANT_BLOCK_SIZE = 32LL;
constexpr int64_t MXFPX_SCALE_BLOCK_SIZE = 64LL;
constexpr int64_t SCALE_FACTOR_WITH_X = 32LL;
constexpr int64_t NUM_THOUSAND = 1000LL;
constexpr int64_t FP8_PERBLOCK_BLOCK_SIZE = 128LL;
constexpr int64_t FP8_GROUP_SIZE = 128LL;
constexpr int64_t DOUBLE_BUFFER = 2LL;

constexpr int64_t MAX_QUEUE_BUFFER_NUM = 6LL;

constexpr int64_t NUM_TWO = 2LL;
constexpr int64_t NUM_THREE = 3LL;
constexpr int64_t NUM_FOUR = 4LL;
constexpr int64_t MRG_LIST_NUM = 4LL;
constexpr int64_t SORT32_ALIGN_ELEMENT = 32LL;
constexpr int64_t UB_BLOCK_SIZE = 32LL;
constexpr size_t DIM_ONE = 1ULL;
constexpr size_t DIM_TWO = 2ULL;
constexpr int32_t SIZE_16 = 16;
constexpr int32_t LENGTH_1024 = 1024;
constexpr int64_t KV_FACTOR = 2LL;
constexpr int64_t ONE_CORE_SORT_BUFFER = 6LL;
constexpr int64_t EXPERT_IDX_MAX = 10240LL;
constexpr int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;
constexpr int64_t RANK_ONE = 1LL;
constexpr int64_t RANK_TWO = 2LL;
constexpr int64_t RANK_THREE = 3LL;
constexpr int64_t BF16_TO_FP32_SIZE_FACTOR = 2LL;

// 输入输出的位置索引（V3 回归后 topk_weight 相关索引不再使用，V4 子类使用独立索引常量）
constexpr int64_t INPUT_X_INDEX = 0LL;
constexpr int64_t INPUT_EXPERT_IDX_INDEX = 1LL;
constexpr int64_t INPUT_SCALE_INDEX = 2LL;
constexpr int64_t INPUT_OFFSET_INDEX = 3LL;
constexpr int64_t OUTPUT_EXPANDED_X_INDEX = 0LL;
constexpr int64_t OUTPUT_EXPANDED_ROW_IDX_INDEX = 1LL;
constexpr int64_t OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2LL;
constexpr int64_t OUTPUT_EXPANDED_SCALE_INDEX = 3LL;
constexpr int64_t ATTR_ACTIVE_NUM_INDEX = 0LL;
constexpr int64_t ATTR_EXPERT_CAPACITY_INDEX = 1LL;
constexpr int64_t ATTR_EXPERT_NUM_INDEX = 2LL;
constexpr int64_t ATTR_DROP_PAD_MODE_INDEX = 3LL;
constexpr int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 4LL;
constexpr int64_t ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX = 5LL;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 6LL;
constexpr int64_t ATTR_EXPERT_RANGE_INDEX = 7LL;
constexpr int64_t ATTR_ROW_IDX_TYPE_INDEX = 8LL;

constexpr int64_t ACTIVE_NUM_MIN_VALUE = -1LL;
constexpr int64_t DYNAMIC_QUANT_COLS_BUFFER = 21LL;
constexpr int64_t HIF8_PERTENSOR_QUANT_COLS_BUFFER = 5LL;
constexpr int64_t HIF8_PERTOKEN_QUANT_COLS_BUFFER = 5LL;
constexpr int64_t STATIC_QUANT_ROW_MULTIPLE = 2LL; // 静态量化行方向Buffer乘数
constexpr int64_t STATIC_QUANT_FULLLOAD_COLS_BUFFER = 5LL;
constexpr int64_t DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER = 9LL;

// 输入attrs相关
constexpr int64_t ROW_IDX_GATHER = 0LL;
constexpr int64_t ROW_IDX_SCATTER = 1LL;
constexpr int64_t QUANT_MODE_UNQUANT = -1LL;
constexpr int64_t QUANT_MODE_STATIC = 0LL;
constexpr int64_t QUANT_MODE_DYNAMIC = 1LL;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2LL;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_E5M2 = 4LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_E4M3FN = 5LL;
constexpr int64_t QUANT_MODE_HIF8_CAST = 6LL;
constexpr int64_t QUANT_MODE_HIF8_PERTENSOR = 7LL;
constexpr int64_t QUANT_MODE_HIF8_PERTOKEN = 8LL;
constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9LL;
constexpr int64_t QUANT_MODE_FP8_PERBLOCK_E5M2 = 11LL;
constexpr int64_t QUANT_MODE_FP8_PERBLOCK_E4M3FN = 12LL;
constexpr int64_t QUANT_MODE_INT4_DYNAMIC = 13LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_AMAX_E5M2 = 14LL;
constexpr int64_t QUANT_MODE_FP8_GROUP_AMAX_E4M3FN = 15LL;
constexpr int64_t QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 = 16LL;
constexpr int64_t QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN = 17LL;
constexpr int64_t EXPERT_TOKENS_TYPE_CUMSUM = 0LL;

constexpr int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
constexpr int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
constexpr int64_t DROP_PAD_MODE_DROPLESS = 0LL;
constexpr int64_t DROP_PAD_MODE_DROPPAD = 1LL;
constexpr int64_t EXPERT_CAPACITY_MIN_VALUE = 0LL;
constexpr int64_t MAX_COLS_ONE_LOOP = 16376LL; // DropPad Tiling计算使用的最大列数
constexpr int64_t REG_SIZE = 256LL;

constexpr int64_t TILINGKEY_BASE = 10000000LL;
constexpr int64_t FULLLOAD_TILINGKEY_BASE = 200000LL; // 全载模版
constexpr int64_t SORT_CORE_TILINGKEY_BASE = 1000000LL;
constexpr int64_t QUANT_MODE_TILINGKEY_BASE = 10000LL;
constexpr int64_t ROWIDX_TYPE_TILINGKEY_BASE = 1000LL;
constexpr int64_t DROP_MODE_TILINGKEY_BASE = 100LL;
constexpr uint64_t EMPTY_TENSOR_TILINGKEY = 3000000ULL;
constexpr int64_t KEY_VALUE_MODE_DIM0_NUM = 2LL;

const static int64_t COUNT_SORT_BASE = 400000LL;
const static int64_t FULL_LOAD_BASE = 10000;

inline static int64_t CeilLog4(int64_t x)
{
    if (x <= 0) {
        return 0;
    }
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

struct MultipleParams {
    int64_t colMultiple = 0;
    int64_t rowMultiple = 0;
};

struct PerLoopParams {
    int64_t xCopyInQueueBufferNum = 2;
    int64_t perLoopCols = 0;
    int64_t perLoopMaxIndicesElements = 0;
};

class MoeInitRoutingV3TilingArch35 : public TilingBaseClass {
public:
    explicit MoeInitRoutingV3TilingArch35(gert::TilingContext *context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRoutingV3TilingArch35() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // 1、获取INPUT/OUTPUT/ATTR信息：DelayedGetShapeAttrsInfo()，延后到DoOpTiling内执行，以便先检查IsCapable()
    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetShapeAttrsInfo()");
        return ge::GRAPH_SUCCESS;
    }
    // 2、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 3、判断此Tiling模板是否适配当前SOC
    bool IsCapable() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::IsCapable()");
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }
    // 4、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::DoLibApiTiling()");
        return ge::GRAPH_SUCCESS;
    }
    // 6、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 8、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset()
    {
        opName = nullptr;
    }

    // DoGetShapeAttrsInfo使用的子函数（V4 需覆盖）
    virtual ge::graphStatus GetInputTensorsInfo();
    virtual ge::graphStatus GetOutputTensorsInfo();
    virtual ge::graphStatus GetInputAttrsInfo();

    // 辅助工具函数（V4 子类需要使用）
    template <bool IS_INPUT_TENSOR = true, bool IS_OPTIONAL_INPUT = false>
    ge::graphStatus GetTensorShapeDtype(gert::Shape &shape, ge::DataType &dtype, int64_t index)
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetTensorShapeDtype(...)");
        const gert::StorageShape *shapePtr{nullptr};
        const gert::CompileTimeTensorDesc *descPtr{nullptr};
        if constexpr (IS_INPUT_TENSOR) {
            if constexpr (IS_OPTIONAL_INPUT) {
                shapePtr = context_->GetOptionalInputShape(index);
                descPtr = context_->GetOptionalInputDesc(index);
            } else {
                shapePtr = context_->GetInputShape(index);
                descPtr = context_->GetInputDesc(index);
            }
        } else {
            shapePtr = context_->GetOutputShape(index);
            descPtr = context_->GetOutputDesc(index);
        }
        OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
        shape = shapePtr->GetStorageShape();
        OP_CHECK_NULL_WITH_CONTEXT(context_, descPtr);
        dtype = descPtr->GetDataType();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetOptionalInputShapeDtype(gert::Shape &shape, ge::DataType &dtype, int64_t &marker, int64_t index)
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetOptionalInputShapeDtype(...)");
        if (context_->GetOptionalInputShape(index) != nullptr) {
            marker = 1;
            return GetTensorShapeDtype<true, true>(shape, dtype, index);
        } else {
            // 该Tensor没有输入
            marker = 0;
            return ge::GRAPH_SUCCESS;
        }
    }

    template <typename ATTR_T>
    ge::graphStatus GetInputAttr(ATTR_T &attr, const gert::RuntimeAttrs *attrsPtr, int64_t index)
    {
        OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetInputAttr(...)");
        const auto *attrPtr = attrsPtr->GetAttrPointer<ATTR_T>(index);
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
        attr = *attrPtr;
        return ge::GRAPH_SUCCESS;
    }

    // op variables
    const char *opName = "";
    MoeInitRoutingV3Arch35TilingData *tilingDataPtr_{nullptr};

    // platform infos
    int64_t aivCoreNum_ = 0LL;
    int64_t totalUbSize_ = 0LL;
    int64_t availUbSize_ = 0LL;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;

    // important values
    int64_t sortLoopMaxElement_ = 0LL;
    int64_t totalLength_ = 0LL;
    int64_t n_ = 0LL;
    int64_t k_ = 0LL;
    int64_t cols_ = 0LL;
    int64_t inputXDtypeSize_ = 0LL;
    int64_t inputScaleDTypeSize_ = 0LL;
    int64_t isInputScale_ = 0LL;
    int64_t isInputOffset_ = 0LL;
    int64_t isInputTopkWeight_ = 0LL;
    int64_t isOutputExpandedTopkWeight_ = 0LL;
    int64_t sortMode_ = 0LL;

    // full load flag
    bool isFullload_ = false;
    bool isEmptyTensor_ = false;
    int64_t ep_ = 0LL;

    // CountingSort 性能模板标志：0=未启用, 1=FullLoad, 2=CutOrigin
    int64_t countingSortMode_ = 0LL;

    // input attrs
    int64_t activeNum_ = -1LL;
    int64_t expertCapacity_ = -1LL;
    int64_t expertNum_ = -1LL;
    int64_t dropPadMode_ = -1LL;
    int64_t expertTokensNumType_ = -1LL;
    bool expertTokensNumFlag_ = false;
    int64_t quantMode_ = -1LL;
    int64_t expertStart_ = -1LL;
    int64_t expertEnd_ = -1LL;
    int64_t rowIdxType_ = -1LL;

    // input tensors shape
    gert::Shape xShape_;
    gert::Shape expertIdxShape_;
    gert::Shape scaleShape_;
    gert::Shape offsetShape_;
    gert::Shape topkWeightShape_;
    // output tensors shape
    gert::Shape expandedXShape_;
    gert::Shape expandedRowIdxShape_;
    gert::Shape expertTokensCountOrCumsumShape_;
    gert::Shape expandedScaleShape_;
    gert::Shape expandedTopkWeightShape_;

    // input tensors dtype
    ge::DataType xDtype_;
    ge::DataType expertIdxDtype_;
    ge::DataType scaleDtype_;
    ge::DataType offsetDtype_;
    ge::DataType topkWeightDtype_;
    // output tensors dtype
    ge::DataType expandedXDtype_;
    ge::DataType expandedRowIdxDtype_;
    ge::DataType expertTokensCountOrCumsumDtype_;
    ge::DataType expandedScaleDtype_;
    ge::DataType expandedTopkWeightDtype_;

private:
    ge::graphStatus CheckSetPlatformInfo();
    ge::graphStatus DoGetShapeAttrsInfo();
    ge::graphStatus CheckSetAttrs();
    ge::graphStatus CheckSetListAttrs();
    ge::graphStatus ValidateExpertTokensNumType();
    ge::graphStatus ValidateExpertNum();
    ge::graphStatus ValidateDropPadMode();
    ge::graphStatus ValidateExpertCapacity();
    ge::graphStatus ValidateQuantMode();
    ge::graphStatus ValidateRowIdxType();
    ge::graphStatus CheckSetInputs();
    ge::graphStatus CheckSetEmptyTensor();
    ge::graphStatus CheckOutputs();
    ge::graphStatus GetEmptyTensorWorkspaceSize();
    ge::graphStatus GetCountingSortWorkspaceSize();
    ge::graphStatus GetNormalWorkspaceSize();

    // CheckInputShape使用的子函数
    ge::graphStatus CheckInputX();
    ge::graphStatus CheckInputExpertIdx();
    ge::graphStatus CheckInputScale();
    ge::graphStatus CheckStaticQuantScale();
    struct ScaleShapeCheckInfo {
        int64_t rank = -1;
        int64_t dim0 = -1;
        int64_t dim1 = -1;
        int64_t dim2 = -1;
    };
    ScaleShapeCheckInfo GetExpectedInputScaleShape() const;
    ge::graphStatus CheckInputScaleShape(const ScaleShapeCheckInfo &expected);
    ge::graphStatus CheckInputScaleDtype();
    ge::graphStatus CheckInputOffset();
    ge::graphStatus CheckInputTopkWeight();
    ge::graphStatus CheckTopkWeightConsistency();
    // CheckOutShape使用的子函数
    ge::graphStatus CheckOutputExpandedX();
    ge::graphStatus CheckOutputExpandedRowIdx();
    ge::graphStatus CheckOutputExpertTokensCountOrCumsum();
    ge::graphStatus CheckOutputExpandedScale();
    ge::graphStatus CheckOutputExpandedTopkWeight();
    ge::graphStatus ValidateExpandedXShapeDropPad();
    ge::graphStatus ValidateExpandedXShapeDropless();
    ge::graphStatus ValidateExpandedXDtype();
    void CalculateExpectedScaleShape(int64_t &expectedRank, int64_t &expectedDim0, int64_t &expectedDim1,
                                     int64_t &expectedDim2);
    ge::graphStatus ValidateScaleShape(int64_t expectedRank, int64_t expectedDim0, int64_t expectedDim1,
                                       int64_t expectedDim2);

    // 各阶段TilingData计算函数
    MultipleParams GetMultipleParams();
    PerLoopParams GetPerLoopParams(MultipleParams &multipleParams, int64_t perCoreIndicesElements);
    void AlignInt4DynamicQuantPerLoopCols(PerLoopParams &perLoopParams) const;
    void SetPerLoopParams4NoQuantDropPad(const MultipleParams &multipleParams, PerLoopParams &perLoopParams,
                                         const int64_t perCoreIndicesElements);
    int64_t GetXBufferNum(const int additionalBufferNum);
    void SetPerLoopParams4NoQuantDropLess(PerLoopParams &perLoopParams, const int64_t perCoreIndicesElements);
    void Tiling4GatherOutCompute();
    void Tiling4GatherOutMxFP8NoQuantCompute();
    void Tiling4GatherOutMxQuant();
    void Tiling4GatherOutFP8Quant();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4ExpertTokensCountCompute();
    void Tiling4VBSOneCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling);
    void Tiling4VBSMultiCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling);
    int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols);
    int64_t CalcMaxRowIdxPerLoopFP8Quant(int64_t perLoopCols);
    int64_t CalcMaxRowIdxPerLoopFP8GroupQuant(int64_t perLoopCols);
    bool IsFullLoad();
    bool IsSupportGatherCopyKernels() const;
    void ComputeUseGatherCopy();
    void SetIndicesLoopParams4GatherOut(int64_t perLoopMaxIndicesElements, int64_t perCoreIndicesElements,
                                        int64_t lastCoreIndicesElements);
    void SetLastCoreIndicesTiling(MoeV3Arch35GatherOutComputeTilingData *gatherOutTiling,
                                  int64_t lastCoreIndicesElements, int64_t perLoopMaxIndicesElements);

    // CountingSort 性能模板
    bool IsCountingSortApplicable();
    void ComputeCountingSortMode();
    int64_t EstimateArch35CountingSortFullLoadUB(int64_t perCoreTokens);
    void ComputeArch35CountingSortFullLoadTiling();
    void ComputeArch35CountingSortCutOriginTiling();

    // DropPad模式Tiling计算函数
    void SetCoreSplitParams4SrcToDstDropPad(int64_t &needCoreNum, int64_t &perCoreRows, int64_t &lastCoreRows);
    void SetLoopParams4SrcToDstDropPad(int64_t perCoreRows, int64_t lastCoreRows);
    void Tiling4SrcToDstDropPadCompute();
    void Tiling4GatherOutDropPadCompute();
    bool UseCompactGatherOutDropPad(int64_t outputRows) const;
    void SetGatherOutDropPadCoreSplitParams(int64_t &needCoreNum, int64_t &perCoreIndicesElements,
                                            int64_t &lastCoreIndicesElements);
    void SetGatherOutDropPadLoopParams(int64_t perCoreIndicesElements, int64_t lastCoreIndicesElements);

    // LogTilingData
    void LogBaseTilingData();
    void LogVbsTilingData();
    void LogVmsMiddleTilingData();
    void LogSortOutTilingData();
    void LogExpertTokensCountTilingData();
    void LogGatherOutTilingData();

    bool IsMXFPXNoQuantCase(int64_t quantMode, ge::DataType xDtype) const
    {
        return quantMode == QUANT_MODE_UNQUANT &&
               (xDtype == ge::DataType::DT_FLOAT8_E5M2 || xDtype == ge::DataType::DT_FLOAT8_E4M3FN ||
                xDtype == ge::DataType::DT_FLOAT4_E2M1);
    }

    bool IsSupportFullloadQuantMode() const
    {
        return (quantMode_ == QUANT_MODE_UNQUANT && !IsMXFPXNoQuantCase(quantMode_, xDtype_)) ||
               (quantMode_ == QUANT_MODE_STATIC) || IsAnyDynamicQuantCase();
    }

    bool IsMXFP8NoQuantCase(int64_t quantMode, ge::DataType xDtype) const
    {
        return quantMode == QUANT_MODE_UNQUANT &&
               (xDtype == ge::DataType::DT_FLOAT8_E5M2 || xDtype == ge::DataType::DT_FLOAT8_E4M3FN);
    }

    bool IsInt8DynamicQuantCase() const
    {
        return quantMode_ == QUANT_MODE_DYNAMIC;
    }

    bool IsInt4DynamicQuantCase() const
    {
        return quantMode_ == QUANT_MODE_INT4_DYNAMIC;
    }

    bool IsAnyDynamicQuantCase() const
    {
        return IsInt8DynamicQuantCase() || IsInt4DynamicQuantCase();
    }

    bool NeedQuantTempWorkspace() const
    {
        return (quantMode_ >= QUANT_MODE_DYNAMIC && quantMode_ != QUANT_MODE_HIF8_CAST &&
                quantMode_ != QUANT_MODE_HIF8_PERTENSOR && quantMode_ != QUANT_MODE_MXFP8_E5M2 &&
                quantMode_ != QUANT_MODE_MXFP8_E4M3FN && quantMode_ != QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 &&
                quantMode_ != QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN && quantMode_ != QUANT_MODE_FP8_GROUP_E5M2 &&
                quantMode_ != QUANT_MODE_FP8_GROUP_E4M3FN && quantMode_ != QUANT_MODE_FP8_GROUP_AMAX_E5M2 &&
                quantMode_ != QUANT_MODE_FP8_GROUP_AMAX_E4M3FN);
    }
};

} // namespace optiling

#endif // MOE_INIT_ROUTING_V3_TILING_ARCH35_H
