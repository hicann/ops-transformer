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
 * \file moe_init_routing_v3_tiling_arch35.cpp
 * \brief
 */
#include "moe_init_routing_v3_tiling_arch35.h"

namespace optiling {
constexpr int64_t DIM_VALUE_ONE = 1LL;
constexpr int64_t DIM_VALUE_TWO = 2LL;

const static int64_t CS_MIN_N = 256LL;
const static int64_t CS_MAX_N = 1024LL * 1024LL;
const static int64_t CS_FILTER_CHUNK_SIZE = 4096LL;
const static int64_t CS_MAX_PER_LOOP_COLS_BYTES = 64LL * 1024LL;
const static int64_t CS_MAX_ACTUAL_EXPERT_NUM = 256LL;
const static int64_t CS_ONE_BLOCK_ELEMENT = 8LL;
const static int64_t CS_HISTOGRAM_BIN_SIZE = 128LL; // Histogram BIN0[0-127]/BIN1[128-255]

const static int64_t CS_FULLLOAD_MAX_ACTUAL_EXPERT_NUM = 128LL;
const static int64_t CS_PERFORMANCE_MODE_TOP_K = 8LL;
const static int64_t CS_PERFORMANCE_MODE_RANGE_MAX = 32LL;
const static int64_t CS_COUNTING_SORT_THRESHOLD = 1536LL;
const static int64_t CS_ASSIST_NUM = 256LL;
const static int64_t NUM_32 = 32;
const static int64_t MAX_EXPERT_NUM = 1024LL;
const static int64_t AGGRBUFBYTES_A5 = 10 * 1024; // A5一次搬出数据量（较好带宽）

ge::graphStatus MoeInitRoutingV3TilingArch35::GetPlatformInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetPlatformInfo()");

    const auto *compileInfoPtr = reinterpret_cast<const MoeInitRoutingV3CompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "compileInfo"),
                return ge::GRAPH_FAILED);
    aivCoreNum_ = static_cast<int64_t>(compileInfoPtr->aivNum);
    totalUbSize_ = static_cast<int64_t>(compileInfoPtr->ubSize);
    socVersion_ = compileInfoPtr->socVersion;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckSetPlatformInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckSetPlatformInfo()");

    // check aivCoreNum
    OP_CHECK_IF(aivCoreNum_ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "aivCoreNum",
                                                      std::to_string(aivCoreNum_), "failed to get valid aivCoreNum"),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->coreNum = aivCoreNum_;
    // check availUbSize
    availUbSize_ = totalUbSize_ - SIMT_DCACHE_SIZE;
    OP_CHECK_IF(
        totalUbSize_ <= 0 || availUbSize_ <= 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "totalUbSize, availUbSize",
                                               (std::to_string(totalUbSize_) + ", " + std::to_string(availUbSize_)),
                                               ("Got invalid totalUbSize(<0) or availUbSize(<0)")),
        return ge::GRAPH_FAILED);
    // log info
    OP_LOGD(context_,
            "Got platform info aivCoreNum = %ld, totalUbSize = %ld bytes, availUbSize = %ld bytes (SIMT_DCACHE_SIZE = "
            "%ld bytes).",
            aivCoreNum_, totalUbSize_, availUbSize_, SIMT_DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::DoGetShapeAttrsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetInputAttrsInfo()");

    MIRV3_CHECK_GE_RET(GetInputTensorsInfo());
    MIRV3_CHECK_GE_RET(GetOutputTensorsInfo());
    MIRV3_CHECK_GE_RET(GetInputAttrsInfo());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::DoOpTiling()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::DoOpTiling()");

    // 获取tilingData指针
    tilingDataPtr_ = context_->GetTilingData<MoeInitRoutingV3Arch35TilingData>();

    MIRV3_CHECK_GE_RET(CheckSetPlatformInfo());
    MIRV3_CHECK_GE_RET(DoGetShapeAttrsInfo());
    MIRV3_CHECK_GE_RET(CheckSetAttrs());
    MIRV3_CHECK_GE_RET(CheckSetListAttrs());
    MIRV3_CHECK_GE_RET(CheckSetInputs());
    MIRV3_CHECK_GE_RET(CheckSetEmptyTensor());
    MIRV3_CHECK_GE_RET(CheckOutputs());
    MIRV3_CHECK_GE_RET(CheckTopkWeightConsistency());

    // 空tensor快速返回，跳过后续tiling计算
    if (isEmptyTensor_) {
        return ge::GRAPH_SUCCESS;
    }

    sortLoopMaxElement_ = availUbSize_ / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    sortLoopMaxElement_ =
        std::min(sortLoopMaxElement_, SORT_API_MAX_ELEM); // 限制单核排序的元素个数在AscendC::Sort全排序的能力范围内

    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4ExpertTokensCountCompute();

    isFullload_ = IsFullLoad();
    ComputeCountingSortMode();

    ComputeUseGatherCopy();

    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
        quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 || quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN ||
        quantMode_ == QUANT_MODE_MXFP4_E2M1) {
        Tiling4GatherOutMxQuant();
    } else if (quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 || quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN ||
               quantMode_ == QUANT_MODE_FP8_GROUP_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_E4M3FN ||
               quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN) {
        Tiling4GatherOutFP8Quant();
    } else if (dropPadMode_ == DROP_PAD_MODE_DROPPAD && !isFullload_) {
        Tiling4SrcToDstDropPadCompute();
        Tiling4GatherOutDropPadCompute();
    } else if (IsMXFP8NoQuantCase(quantMode_, xDtype_)) {
        Tiling4GatherOutMxFP8NoQuantCompute();
    } else {
        Tiling4GatherOutCompute();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRoutingV3TilingArch35::GetTilingKey() const
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetTilingKey()");

    if (isEmptyTensor_) {
        return EMPTY_TENSOR_TILINGKEY;
    }

    int64_t quantModeFactor = quantMode_ + 1;

    if (isFullload_ && IsSupportFullloadQuantMode()) {
        // INT4动态量化全载模板复用INT8动态量化tilingkey，kernel通过quantMode区分。
        quantModeFactor =
            quantMode_ == QUANT_MODE_INT4_DYNAMIC ? (QUANT_MODE_DYNAMIC - QUANT_MODE_UNQUANT) : quantModeFactor;
        return static_cast<uint64_t>(FULLLOAD_TILINGKEY_BASE + quantModeFactor * QUANT_MODE_TILINGKEY_BASE);
    }

    if (countingSortMode_ != 0) {
        return static_cast<uint64_t>(
            COUNT_SORT_BASE + countingSortMode_ * FULL_LOAD_BASE + quantModeFactor * QUANT_MODE_TILINGKEY_BASE +
            rowIdxType_ * ROWIDX_TYPE_TILINGKEY_BASE + dropPadMode_ * DROP_MODE_TILINGKEY_BASE);
    }

    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN) {
        // 对于MXFP8量化，两种模式在TilingKey体现的QuantMode都为3。
        // 其余非量化为0，静态量化为1，动态量化为2，即都是quantMode_+1
        // 可以用与最低的UNQUANT的数值的差值来作为quantModeFactor，这里值就为3
        quantModeFactor = QUANT_MODE_MXFP8_E5M2 - QUANT_MODE_UNQUANT;
    } else if (quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 ||
               quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN) {
        quantModeFactor = QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 - QUANT_MODE_UNQUANT;
    } else if (quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 || quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
        quantModeFactor = QUANT_MODE_FP8_PERBLOCK_E5M2 - QUANT_MODE_UNQUANT;
    } else if (quantMode_ == QUANT_MODE_FP8_GROUP_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_E4M3FN) {
        quantModeFactor = QUANT_MODE_FP8_GROUP_E5M2 - QUANT_MODE_UNQUANT;
    } else if (quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN) {
        quantModeFactor = QUANT_MODE_FP8_GROUP_AMAX_E5M2 - QUANT_MODE_UNQUANT;
    }
    return static_cast<uint64_t>(TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
                                 quantModeFactor * QUANT_MODE_TILINGKEY_BASE +
                                 rowIdxType_ * ROWIDX_TYPE_TILINGKEY_BASE + dropPadMode_ * DROP_MODE_TILINGKEY_BASE);
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetWorkspaceSize()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetWorkspaceSize()");
    // 空tensor场景：分配少量workspace即可
    if (isEmptyTensor_) {
        return GetEmptyTensorWorkspaceSize();
    }
    if (countingSortMode_ == 1) {
        return GetCountingSortWorkspaceSize();
    }
    return GetNormalWorkspaceSize();
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetCountingSortWorkspaceSize()
{
    // CountingSort 性能模板（仅模板1 FullLoad 非量化）的 workspace：核间专家计数归约表 + 过滤 pairs 表。
    // mode 1（FullLoad 单类）：offset0 = pairs，offset1 = expertCount（均从 ws 起始）
    int64_t filterNeedCoreNum = tilingDataPtr_->countingSortParamsOp.filterNeedCoreNum;
    int64_t expertCountStride = tilingDataPtr_->countingSortParamsOp.expertCountStride;
    // pairs 区：pairsPerCore = CeilAlign(max(filterPerCoreTokens, lastCoreTokens)*k_, CS_ONE_BLOCK_ELEMENT) * 2
    int64_t filterPerCoreTokens = tilingDataPtr_->countingSortParamsOp.filterPerCoreTokens;
    int64_t lastCoreTokens = tilingDataPtr_->countingSortParamsOp.lastCoreTokens;
    int64_t maxCoreEntries = std::max(filterPerCoreTokens, lastCoreTokens) * tilingDataPtr_->k;
    int64_t pairsPerCore = Ops::Base::CeilAlign(maxCoreEntries, CS_ONE_BLOCK_ELEMENT) * NUM_TWO;
    int64_t csWorkspace = filterNeedCoreNum * expertCountStride * static_cast<int64_t>(sizeof(int32_t));
    csWorkspace += filterNeedCoreNum * pairsPerCore * static_cast<int64_t>(sizeof(int32_t));

    csWorkspace += SIZE_16 * LENGTH_1024 * LENGTH_1024;
    workspaceSize_ = static_cast<uint32_t>(csWorkspace);
    OP_LOGD(context_, "CountingSort workspace size = %u bytes (pairsPerCore=%ld, stride=%ld)", workspaceSize_,
            pairsPerCore, expertCountStride);
    auto *csWsPtr = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, csWsPtr);
    csWsPtr[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetNormalWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = 0;
    int64_t sortWorkspaceSize =
        totalLength_ * static_cast<int64_t>(sizeof(float) * NUM_TWO * NUM_THREE);             // 排序需要的空间
    int64_t coreSyncWorkspaceSize = tilingDataPtr_->coreNum * SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    int64_t scatterWorkspaceSize = totalLength_ * static_cast<int64_t>(sizeof(int32_t));

    int64_t expertCountAlignedElements = Align(expertEnd_ - expertStart_, static_cast<int64_t>(sizeof(int32_t)));
    int64_t expertTokensCountWorkspaceSize = expertCountAlignedElements * static_cast<int64_t>(sizeof(int32_t));
    if (dropPadMode_ != DROP_PAD_MODE_DROPPAD) {
        expertTokensCountWorkspaceSize += expertCountAlignedElements * static_cast<int64_t>(sizeof(int32_t));
    }
    int64_t quantTempWorkspaceSize = aivCoreNum_ * cols_ * static_cast<int64_t>(sizeof(float));

    workspaceSize_ += sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + expertTokensCountWorkspaceSize;

    // DropPad模式需要额外的workspace空间
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        // expertIdxValueGm: 每个核需要存储2个int32 (lastExpertId和lastCoreExpertIdNum)
        int64_t expertIdxValueWorkspaceSize = tilingDataPtr_->coreNum * NUM_TWO * sizeof(int32_t);
        workspaceSize_ += expertIdxValueWorkspaceSize;

        int64_t outputToSrcRowWorkspaceSize = 0;
        if (tilingDataPtr_->gatherOutComputeParamsOp.useCompactGatherOutDropPad) {
            outputToSrcRowWorkspaceSize = expertNum_ * expertCapacity_ * static_cast<int64_t>(sizeof(int32_t));
            workspaceSize_ += outputToSrcRowWorkspaceSize;
        }
        OP_LOGD(context_, "DropPad workspace: expertIdxValueWorkspace=%ld, outputToSrcRowWorkspace=%ld",
                expertIdxValueWorkspaceSize, outputToSrcRowWorkspaceSize);
    }

    if (NeedQuantTempWorkspace()) {
        // Dynamic/INT4 dynamic quant and HIF8 per-token use quantTempGm_ when cols cannot be full-loaded.
        workspaceSize_ += quantTempWorkspaceSize;
    }
    // STATIC_QUANT: expandedRowIdxIndexGm_ 复用 expertTotalCountGm_ 之后的空间
    // 这里workspaceSize_除了计算必要的，还会加上16M的AscendC框架用大小
    int64_t frameworkOverhead = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    workspaceSize_ += frameworkOverhead;

    OP_LOGD(context_, "Computed workspace size to allocate is %u bytes", workspaceSize_);
    // 设置workspace
    auto *workspacePtr = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspacePtr);
    workspacePtr[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetEmptyTensorWorkspaceSize()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetEmptyTensorWorkspaceSize()");
    workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    auto *workspacePtr = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspacePtr);
    workspacePtr[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::PostTiling()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::PostTiling()");

    // 这个tilingKey_成员变量(TilingBaseClass)不能在GetTilingKey()方法里赋值的设计还挺抽象的
    tilingKey_ = GetTilingKey();
    LogBaseTilingData();

    // 设置启动核数：空tensor场景只用到1个核，否则全核启动
    if (isEmptyTensor_) {
        context_->SetBlockDim(1);
    } else {
        context_->SetBlockDim(aivCoreNum_);
    }
    // 设置UB可用大小（必须是减除SIMT用的DCACHE大小后的）
    auto ret = context_->SetLocalMemorySize(availUbSize_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "availUbSize",
                                                      std::to_string(availUbSize_), "failed to set local memory size"),
                return ge::GRAPH_FAILED);
    // 涉及核间同步的算子必须设置schedule_mode为1，独占全核
    ret = context_->SetScheduleMode(1);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "schedule_mode", "1",
                                                      "failed to set schedule mode for kernel that needs sync cores"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetInputTensorsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetInputTensorsInfo()");

    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<true>(xShape_, xDtype_, INPUT_X_INDEX));
    inputXDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(xDtype_));
    inputXDtypeSize_ = (inputXDtypeSize_ > NUM_THOUSAND) ? 1 : inputXDtypeSize_;
    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<true>(expertIdxShape_, expertIdxDtype_, INPUT_EXPERT_IDX_INDEX));
    // 可选输入scale
    MIRV3_CHECK_GE_RET(GetOptionalInputShapeDtype(scaleShape_, scaleDtype_, isInputScale_, INPUT_SCALE_INDEX));
    tilingDataPtr_->isInputScale = isInputScale_;
    if (isInputScale_) {
        inputScaleDTypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(scaleDtype_));
    }
    // 可选输入offset
    MIRV3_CHECK_GE_RET(GetOptionalInputShapeDtype(offsetShape_, offsetDtype_, isInputOffset_, INPUT_OFFSET_INDEX));
    tilingDataPtr_->isInputOffset = isInputOffset_;
    // V3 regression: topk_weight input not supported
    isInputTopkWeight_ = 0;
    tilingDataPtr_->isInputTopkWeight = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetOutputTensorsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetOutputTensorsInfo()");

    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expandedXShape_, expandedXDtype_, OUTPUT_EXPANDED_X_INDEX));
    MIRV3_CHECK_GE_RET(
        GetTensorShapeDtype<false>(expandedRowIdxShape_, expandedRowIdxDtype_, OUTPUT_EXPANDED_ROW_IDX_INDEX));
    MIRV3_CHECK_GE_RET(GetTensorShapeDtype<false>(expertTokensCountOrCumsumShape_, expertTokensCountOrCumsumDtype_,
                                                  OUTPUT_EXPERT_TOKENS_COUNT_INDEX));
    MIRV3_CHECK_GE_RET(
        GetTensorShapeDtype<false>(expandedScaleShape_, expandedScaleDtype_, OUTPUT_EXPANDED_SCALE_INDEX));
    // V3 regression: expanded_topk_weight output not supported
    isOutputExpandedTopkWeight_ = 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::GetInputAttrsInfo()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::GetInputAttrsInfo()");

    auto attrsPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrsPtr);

    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(activeNum_, attrsPtr, ATTR_ACTIVE_NUM_INDEX));
    OP_LOGD(context_, "Get input attr activeNum = %ld.", activeNum_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertCapacity_, attrsPtr, ATTR_EXPERT_CAPACITY_INDEX));
    OP_LOGD(context_, "Get input attr expertCapacity = %ld.", expertCapacity_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertNum_, attrsPtr, ATTR_EXPERT_NUM_INDEX));
    OP_LOGD(context_, "Get input attr expertNum = %ld.", expertNum_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(dropPadMode_, attrsPtr, ATTR_DROP_PAD_MODE_INDEX));
    OP_LOGD(context_, "Get input attr dropPadMode = %ld.", dropPadMode_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(expertTokensNumType_, attrsPtr, ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX));
    OP_LOGD(context_, "Get input attr expertTokensNumType = %ld.", expertTokensNumType_);
    MIRV3_CHECK_GE_RET(GetInputAttr<bool>(expertTokensNumFlag_, attrsPtr, ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX));
    OP_LOGD(context_, "Get input attr expertTokensNumFlag = %ld.", expertTokensNumFlag_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(quantMode_, attrsPtr, ATTR_QUANT_MODE_INDEX));
    OP_LOGD(context_, "Get input attr quantMode = %ld.", quantMode_);
    MIRV3_CHECK_GE_RET(GetInputAttr<int64_t>(rowIdxType_, attrsPtr, ATTR_ROW_IDX_TYPE_INDEX));
    OP_LOGD(context_, "Get input attr rowIdxType = %ld.", rowIdxType_);
    // expertStart, expertEnd
    const auto *aerPtr = attrsPtr->GetAttrPointer<gert::ContinuousVector>(ATTR_EXPERT_RANGE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, aerPtr);
    int64_t aerLen = aerPtr->GetSize();
    OP_CHECK_IF(
        aerLen != 2,
        OP_LOGE_WITH_INVALID_ATTR_SIZE(context_->GetNodeName(), "active_expert_range", std::to_string(aerLen), "2"),
        return ge::GRAPH_FAILED);
    const int64_t *aerList = reinterpret_cast<const int64_t *>(aerPtr->GetData());
    expertStart_ = aerList[0];
    expertEnd_ = aerList[1];
    OP_LOGD(context_, "Extracted input attrs expertStart = %ld, expertEnd = %ld.", expertStart_, expertEnd_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateExpertTokensNumType()
{
    OP_CHECK_IF((expertTokensNumType_ != EXPERT_TOKENS_TYPE_CUMSUM) &&
                    (expertTokensNumType_ != EXPERT_TOKENS_TYPE_COUNT) &&
                    (expertTokensNumType_ != EXPERT_TOKENS_TYPE_KEY_VALUE),
                OP_LOGE_WITH_INVALID_ATTR(
                    context_->GetNodeName(), "expert_tokens_num_type", std::to_string(expertTokensNumType_),
                    (std::to_string(EXPERT_TOKENS_TYPE_CUMSUM) + ", " + std::to_string(EXPERT_TOKENS_TYPE_COUNT) +
                     " or " + std::to_string(EXPERT_TOKENS_TYPE_KEY_VALUE))),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dropPadMode_ == DROP_PAD_MODE_DROPPAD && expertTokensNumType_ != EXPERT_TOKENS_TYPE_COUNT,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_tokens_num_type",
                                          std::to_string(expertTokensNumType_),
                                          std::to_string(EXPERT_TOKENS_TYPE_COUNT) + " when drop_pad_mode is " +
                                              std::to_string(DROP_PAD_MODE_DROPPAD)),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->expertTokensNumType = expertTokensNumType_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateExpertNum()
{
    int64_t maxExpertNum =
        (expertTokensNumType_ == EXPERT_TOKENS_TYPE_KEY_VALUE) ? KV_MODE_EXPERT_IDX_MAX : EXPERT_IDX_MAX;
    OP_CHECK_IF(expertNum_ <= 0 || expertNum_ > maxExpertNum,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_num", std::to_string(expertNum_),
                                          ("in range [1, " + std::to_string(maxExpertNum) + "]")),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->expertNum = expertNum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateDropPadMode()
{
    OP_CHECK_IF(dropPadMode_ != DROP_PAD_MODE_DROPLESS && dropPadMode_ != DROP_PAD_MODE_DROPPAD,
                OP_LOGE_WITH_INVALID_ATTR(
                    context_->GetNodeName(), "drop_pad_mode", std::to_string(dropPadMode_),
                    std::to_string(DROP_PAD_MODE_DROPLESS) + " or " + std::to_string(DROP_PAD_MODE_DROPPAD)),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->dropPadMode = dropPadMode_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateExpertCapacity()
{
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        OP_CHECK_IF(
            expertCapacity_ <= EXPERT_CAPACITY_MIN_VALUE,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expertCapacity_", std::to_string(expertCapacity_),
                                      " greater than " + std::to_string(EXPERT_CAPACITY_MIN_VALUE)),
            return ge::GRAPH_FAILED);
    }
    tilingDataPtr_->expertCapacity = expertCapacity_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateQuantMode()
{
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        OP_CHECK_IF(quantMode_ != QUANT_MODE_UNQUANT,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "quant_mode", std::to_string(quantMode_),
                                              "-1 in DropPad mode"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(quantMode_ != QUANT_MODE_UNQUANT && quantMode_ != QUANT_MODE_STATIC &&
                    quantMode_ != QUANT_MODE_DYNAMIC && quantMode_ != QUANT_MODE_MXFP8_E5M2 &&
                    quantMode_ != QUANT_MODE_MXFP8_E4M3FN && quantMode_ != QUANT_MODE_FP8_GROUP_E5M2 &&
                    quantMode_ != QUANT_MODE_FP8_GROUP_E4M3FN && quantMode_ != QUANT_MODE_HIF8_CAST &&
                    quantMode_ != QUANT_MODE_HIF8_PERTENSOR && quantMode_ != QUANT_MODE_HIF8_PERTOKEN &&
                    quantMode_ != QUANT_MODE_MXFP4_E2M1 && quantMode_ != QUANT_MODE_FP8_PERBLOCK_E5M2 &&
                    quantMode_ != QUANT_MODE_FP8_PERBLOCK_E4M3FN && quantMode_ != QUANT_MODE_INT4_DYNAMIC &&
                    quantMode_ != QUANT_MODE_FP8_GROUP_AMAX_E5M2 && quantMode_ != QUANT_MODE_FP8_GROUP_AMAX_E4M3FN &&
                    quantMode_ != QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 &&
                    quantMode_ != QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "quant_mode", std::to_string(quantMode_),
                                          "-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16 or 17"),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->quantMode = quantMode_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateRowIdxType()
{
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        OP_CHECK_IF(rowIdxType_ != ROW_IDX_GATHER,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "row_idx_type", std::to_string(rowIdxType_),
                                              std::to_string(ROW_IDX_GATHER)),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(rowIdxType_ != ROW_IDX_SCATTER && rowIdxType_ != ROW_IDX_GATHER,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "row_idx_type", std::to_string(rowIdxType_),
                                          (std::to_string(ROW_IDX_SCATTER) + " or " + std::to_string(ROW_IDX_GATHER))),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->rowIdxType = rowIdxType_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckSetAttrs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckSetAttrs()");

    MIRV3_CHECK_GE_RET(ValidateExpertTokensNumType());
    MIRV3_CHECK_GE_RET(ValidateExpertNum());
    MIRV3_CHECK_GE_RET(ValidateDropPadMode());
    MIRV3_CHECK_GE_RET(ValidateExpertCapacity());

    OP_CHECK_IF(expertTokensNumFlag_ != true,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_tokens_num_flag",
                                          (expertTokensNumFlag_ ? "True" : "False"), "True"),
                return ge::GRAPH_FAILED);
    tilingDataPtr_->expertTokensNumFlag = expertTokensNumFlag_;

    MIRV3_CHECK_GE_RET(ValidateQuantMode());
    MIRV3_CHECK_GE_RET(ValidateRowIdxType());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckSetListAttrs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckSetListAttrs()");

    // expertStart, expertEnd
    OP_CHECK_IF(expertStart_ < 0,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_start", std::to_string(expertStart_),
                                          "greater than or equal to 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expertStart_ >= expertEnd_,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "expert_start, expert_end",
                                               (std::to_string(expertStart_) + ", " + std::to_string(expertEnd_)),
                                               "expert_start should be less than expert_end"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertEnd_ > expertNum_,
                OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_end", std::to_string(expertEnd_),
                                          ("less than or equal to expert_num(" + std::to_string(expertNum_) + ")")),
                return ge::GRAPH_FAILED);

    // DropPad模式下activeExpertRange必须为[0, expertNum]
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        OP_CHECK_IF(
            expertStart_ != 0 || expertEnd_ != expertNum_,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "[expert_start, expert_end]",
                                      "[" + std::to_string(expertStart_) + ", " + std::to_string(expertEnd_) + "]",
                                      "[0, " + std::to_string(expertNum_) + "]"),
            return ge::GRAPH_FAILED);
    }

    tilingDataPtr_->expertStart = expertStart_;
    tilingDataPtr_->expertEnd = expertEnd_;
    tilingDataPtr_->actualExpertNum = expertEnd_ - expertStart_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputX()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckInputX()");

    // rank
    auto rank = static_cast<int64_t>(xShape_.GetDimNum());
    OP_CHECK_IF(rank != 2, OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(rank), "2"),
                return ge::GRAPH_FAILED);
    // dtype
    using ge::DataType;
    using std::unordered_set;

    static const unordered_set<DataType> STATIC_QUANT_SUPPORTED_DTYPES = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                          DataType::DT_BF16};
    static const unordered_set<DataType> UNQUANT_SUPPORTED_DTYPES = {
        DataType::DT_FLOAT,    DataType::DT_FLOAT16,     DataType::DT_BF16,          DataType::DT_INT8,
        DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT4_E2M1};
    static const unordered_set<DataType> MXFP4QUANT_SUPPORTED_DTYPES = {DataType::DT_FLOAT16, DataType::DT_BF16};
    static const unordered_set<DataType> INT4_DYNAMIC_SUPPORTED_DTYPES = {DataType::DT_FLOAT, DataType::DT_BF16};
    static const unordered_set<DataType> DYNAMIC_QUANT_SUPPORTED_DTYPES = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                           DataType::DT_BF16, DataType::DT_INT8};
    static const std::unordered_set<DataType> EIGHT_BIT_QUANT_SUPPORTED_DTYPES = {ge::DataType::DT_FLOAT16,
                                                                                  ge::DataType::DT_BF16};
    unordered_set<DataType> supportedDtypes;
    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
        quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 || quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN ||
        quantMode_ == QUANT_MODE_HIF8_CAST || quantMode_ == QUANT_MODE_HIF8_PERTENSOR ||
        quantMode_ == QUANT_MODE_HIF8_PERTOKEN || quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 ||
        quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN || quantMode_ == QUANT_MODE_FP8_GROUP_E5M2 ||
        quantMode_ == QUANT_MODE_FP8_GROUP_E4M3FN || quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E5M2 ||
        quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN) {
        supportedDtypes = EIGHT_BIT_QUANT_SUPPORTED_DTYPES;
    } else if (quantMode_ == QUANT_MODE_UNQUANT) {
        supportedDtypes = UNQUANT_SUPPORTED_DTYPES;
    } else if (quantMode_ == QUANT_MODE_STATIC) {
        supportedDtypes = STATIC_QUANT_SUPPORTED_DTYPES;
    } else if (quantMode_ == QUANT_MODE_MXFP4_E2M1) {
        supportedDtypes = MXFP4QUANT_SUPPORTED_DTYPES;
    } else if (quantMode_ == QUANT_MODE_INT4_DYNAMIC) {
        supportedDtypes = INT4_DYNAMIC_SUPPORTED_DTYPES;
    } else {
        //! 出于历史调用的兼容性，这里不拦截quant_mode=1（动态量化）下输入x为int8类型，仅资料说明此时算子输出expandedX、expandedScale无意义
        supportedDtypes = DYNAMIC_QUANT_SUPPORTED_DTYPES;
    }
    OP_CHECK_IF(
        supportedDtypes.count(xDtype_) == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dtype of x", Ops::Base::ToString(xDtype_),
                                              ("unsupported under quant_mode " + std::to_string(quantMode_))),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputExpertIdx()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckInputExpertIdx()");

    auto rank = static_cast<int64_t>(expertIdxShape_.GetDimNum());
    OP_CHECK_IF(rank != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expert_idx", std::to_string(rank), "2"),
                return ge::GRAPH_FAILED);
    int64_t expertIdxDim0 = expertIdxShape_.GetDim(0);
    int64_t xDim0 = xShape_.GetDim(0);
    OP_CHECK_IF(expertIdxDim0 != xDim0,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_idx dim[0]", std::to_string(expertIdxDim0),
                                          std::to_string(xDim0)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputScale()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckInputScale()");

    if (quantMode_ == QUANT_MODE_STATIC) {
        return CheckStaticQuantScale();
    }

    if (isInputScale_ == 0) {
        return ge::GRAPH_SUCCESS;
    }
    const auto expected = GetExpectedInputScaleShape();
    MIRV3_CHECK_GE_RET(CheckInputScaleShape(expected));
    MIRV3_CHECK_GE_RET(CheckInputScaleDtype());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckStaticQuantScale()
{
    // 静态量化模式：scale必须输入，shape为[1]，dtype为FLOAT
    if (quantMode_ == QUANT_MODE_STATIC) {
        OP_CHECK_IF(isInputScale_ == 0, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "scale"),
                    return ge::GRAPH_FAILED);
        auto rankScale = static_cast<int64_t>(scaleShape_.GetDimNum());
        OP_CHECK_IF(rankScale != RANK_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", std::to_string(rankScale), "1"),
                    return ge::GRAPH_FAILED);
        auto dim0 = scaleShape_.GetDim(0);
        OP_CHECK_IF(dim0 != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[0]", std::to_string(dim0), "1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(scaleDtype_ != ge::DataType::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scale",
                                              Ops::Base::ToString(scaleDtype_).c_str(), "DT_FLOAT"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

MoeInitRoutingV3TilingArch35::ScaleShapeCheckInfo MoeInitRoutingV3TilingArch35::GetExpectedInputScaleShape() const
{
    ScaleShapeCheckInfo expected;
    if (quantMode_ == QUANT_MODE_UNQUANT) {
        if (scaleDtype_ == ge::DataType::DT_FLOAT8_E8M0) {
            expected.rank = RANK_THREE;
            expected.dim0 = expertIdxShape_.GetDim(0);
            expected.dim1 = Ops::Base::CeilDiv(xShape_.GetDim(1), MXFPX_SCALE_BLOCK_SIZE);
            expected.dim2 = NUM_TWO;
        } else {
            expected.rank = RANK_ONE;
            expected.dim0 = xShape_.GetDim(0);
        }
    } else if (quantMode_ == QUANT_MODE_DYNAMIC) {
        expected.rank = RANK_TWO;
        expected.dim0 = expertEnd_ - expertStart_;
        expected.dim1 = xShape_.GetDim(1);
    } else if (quantMode_ == QUANT_MODE_INT4_DYNAMIC) {
        // INT4 dynamic quantization: scale can be None or shape (1, H), dtype FLOAT.
        expected.rank = RANK_TWO;
        expected.dim0 = 1;
        expected.dim1 = xShape_.GetDim(1);
    }
    return expected;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputScaleShape(const ScaleShapeCheckInfo &expected)
{
    if (expected.rank != -1) {
        auto rankScale = static_cast<int64_t>(scaleShape_.GetDimNum());
        OP_CHECK_IF(rankScale != expected.rank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale", std::to_string(rankScale),
                                                 std::to_string(expected.rank)),
                    return ge::GRAPH_FAILED);
    }
    if (expected.dim0 != -1) {
        auto dim0 = scaleShape_.GetDim(0);
        OP_CHECK_IF(dim0 != expected.dim0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[0]", std::to_string(dim0),
                                              std::to_string(expected.dim0)),
                    return ge::GRAPH_FAILED);
    }
    if (expected.dim1 != -1) {
        auto dim1 = scaleShape_.GetDim(1);
        OP_CHECK_IF(dim1 != expected.dim1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[1]", std::to_string(dim1),
                                              std::to_string(expected.dim1)),
                    return ge::GRAPH_FAILED);
    }
    if (expected.dim2 != -1) {
        auto dim2 = scaleShape_.GetDim(2);
        OP_CHECK_IF(dim2 != expected.dim2,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "scale dim[2]", std::to_string(dim2),
                                              std::to_string(expected.dim2)),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputScaleDtype()
{
    if (IsMXFPXNoQuantCase(quantMode_, xDtype_)) {
        OP_CHECK_IF(scaleDtype_ != ge::DataType::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "x, scale",
                        (Ops::Base::ToString(xDtype_) + ", " + Ops::Base::ToString(scaleDtype_)).c_str(),
                        "scale should be DT_FLOAT8_E8M0 in no quant case"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            scaleDtype_ != ge::DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "scale", std::to_string(scaleDtype_), "DT_FLOAT"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputOffset()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckInputOffset()");

    // 静态量化模式：offset必须输入，shape为[1]，dtype为FLOAT
    if (quantMode_ == QUANT_MODE_STATIC) {
        OP_CHECK_IF(isInputOffset_ == 0, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "offset"),
                    return ge::GRAPH_FAILED);
        auto rankOffset = static_cast<int64_t>(offsetShape_.GetDimNum());
        OP_CHECK_IF(rankOffset != RANK_ONE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "offset", std::to_string(rankOffset), "1"),
                    return ge::GRAPH_FAILED);
        auto dim0 = offsetShape_.GetDim(0);
        OP_CHECK_IF(dim0 != 1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "offset dim[0]", std::to_string(dim0), "1"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(offsetDtype_ != ge::DataType::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "offset",
                                              Ops::Base::ToString(offsetDtype_).c_str(), "DT_FLOAT"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckInputTopkWeight()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckInputTopkWeight()");

    if (isInputTopkWeight_ == 0) {
        return ge::GRAPH_SUCCESS;
    }

    // rank校验：topk_weight必须为2D
    auto rank = static_cast<int64_t>(topkWeightShape_.GetDimNum());
    OP_CHECK_IF(rank != RANK_TWO,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "topk_weight", std::to_string(rank), "2"),
                return ge::GRAPH_FAILED);
    // dim[0] 与 x/expert_idx 的 dim[0] 一致
    int64_t dim0 = topkWeightShape_.GetDim(0);
    OP_CHECK_IF(dim0 != n_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "topk_weight dim[0]", std::to_string(dim0),
                                          std::to_string(n_)),
                return ge::GRAPH_FAILED);
    // dim[1] 与 expert_idx 的 dim[1] 一致
    int64_t dim1 = topkWeightShape_.GetDim(1);
    OP_CHECK_IF(dim1 != k_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "topk_weight dim[1]", std::to_string(dim1),
                                          std::to_string(k_)),
                return ge::GRAPH_FAILED);
    // dtype校验：topk_weight必须为DT_FLOAT
    OP_CHECK_IF(topkWeightDtype_ != ge::DataType::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "topk_weight",
                                          Ops::Base::ToString(topkWeightDtype_).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckTopkWeightConsistency()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckTopkWeightConsistency()");

    // topk_weight和expanded_topk_weight必须同时传入或同时不传入
    OP_CHECK_IF(isInputTopkWeight_ != isOutputExpandedTopkWeight_,
                OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(),
                                           "topk_weight and expanded_topk_weight must be both present or both absent"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckSetInputs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckSetInputs()");

    MIRV3_CHECK_GE_RET(CheckInputX());
    MIRV3_CHECK_GE_RET(CheckInputExpertIdx());
    n_ = xShape_.GetDim(0);
    k_ = expertIdxShape_.GetDim(1);
    MIRV3_CHECK_GE_RET(CheckInputScale());
    MIRV3_CHECK_GE_RET(CheckInputOffset());
    MIRV3_CHECK_GE_RET(CheckInputTopkWeight());
    OP_CHECK_IF(
        k_ < 0,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_), "greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    cols_ = xShape_.GetDim(1);
    totalLength_ = n_ * k_;
    tilingDataPtr_->n = n_;
    tilingDataPtr_->k = k_;
    tilingDataPtr_->cols = cols_;

    // INT4 dynamic quantization packs two values per byte along the H/cols dimension.
    if (quantMode_ == QUANT_MODE_INT4_DYNAMIC && (cols_ % NUM_TWO != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cols", std::to_string(cols_).c_str(),
                                              "For INT4 dynamic quantization, cols must be even.");
        return ge::GRAPH_FAILED;
    }

    // DropPad模式下expertCapacity必须满足 ≤ numRows
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        OP_CHECK_IF(
            expertCapacity_ > n_,
            OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "expert_capacity", std::to_string(expertCapacity_),
                                      "less than or equal to " + std::to_string(n_)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckSetEmptyTensor()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckSetEmptyTensor()");

    // 空tensor场景：n_==0、k_==0、cols_==0
    if (n_ == 0 || k_ == 0 || cols_ == 0) {
        isEmptyTensor_ = true;
    }

    // 空tensor场景下activeNum无实际意义（如cols_==0但n_*k_>0时，调用方可能传入0），跳过校验并
    // 归一化为totalLength_，与非arch35版本保持一致，避免语义上合法的空输入误失败。
    if (isEmptyTensor_) {
        tilingDataPtr_->activeNum = totalLength_;
    } else if (activeNum_ != ACTIVE_NUM_MIN_VALUE) {
        //! 出于历史调用的兼容性，保留校验activeNum=n*k，但实际上不使用该属性
        OP_CHECK_IF(activeNum_ != totalLength_,
                    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "active_num", std::to_string(activeNum_),
                                              ("bs*k(" + std::to_string(totalLength_) + ")")),
                    return ge::GRAPH_FAILED);
        tilingDataPtr_->activeNum = totalLength_;
    } else {
        tilingDataPtr_->activeNum = totalLength_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateExpandedXShapeDropPad()
{
    auto rank = static_cast<int64_t>(expandedXShape_.GetDimNum());
    OP_CHECK_IF(rank != RANK_THREE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "rank", std::to_string(rank), "3"),
                return ge::GRAPH_FAILED);

    int64_t dim0 = expandedXShape_.GetDim(0);
    OP_CHECK_IF(dim0 != expertNum_,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "The dim0 of output expanded_x",
                                             std::to_string(dim0), std::to_string(expertNum_)),
                return ge::GRAPH_FAILED);

    int64_t dim1 = expandedXShape_.GetDim(1);
    OP_CHECK_IF(dim1 != expertCapacity_,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "The dim1 of output expanded_x",
                                             std::to_string(dim1), std::to_string(expertCapacity_)),
                return ge::GRAPH_FAILED);

    int64_t dim2 = expandedXShape_.GetDim(2);
    OP_CHECK_IF(dim2 != cols_,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "The dim2 of output expanded_x",
                                             std::to_string(dim2), std::to_string(cols_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateExpandedXShapeDropless()
{
    auto rank = static_cast<int64_t>(expandedXShape_.GetDimNum());
    OP_CHECK_IF(rank != RANK_TWO,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_x", std::to_string(rank), "2"),
                return ge::GRAPH_FAILED);

    int64_t dim0 = expandedXShape_.GetDim(0);
    OP_CHECK_IF(dim0 != totalLength_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_x dim[0]", std::to_string(dim0),
                                          std::to_string(totalLength_)),
                return ge::GRAPH_FAILED);

    int64_t dim1 = expandedXShape_.GetDim(1);
    OP_CHECK_IF(dim1 != cols_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_x dim[1]", std::to_string(dim1),
                                          std::to_string(cols_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateExpandedXDtype()
{
    using ge::DataType;
    DataType expectedDtype = DataType::DT_UNDEFINED;
    switch (quantMode_) {
        case QUANT_MODE_UNQUANT:
            expectedDtype = xDtype_;
            break;
        case QUANT_MODE_STATIC:
        case QUANT_MODE_DYNAMIC:
            expectedDtype = DataType::DT_INT8;
            break;
        case QUANT_MODE_MXFP8_E5M2:
        case QUANT_MODE_FP8_GROUP_E5M2:
        case QUANT_MODE_FP8_PERBLOCK_E5M2:
        case QUANT_MODE_FP8_GROUP_AMAX_E5M2:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2:
            expectedDtype = DataType::DT_FLOAT8_E5M2;
            break;
        case QUANT_MODE_MXFP8_E4M3FN:
        case QUANT_MODE_FP8_GROUP_E4M3FN:
        case QUANT_MODE_FP8_PERBLOCK_E4M3FN:
        case QUANT_MODE_FP8_GROUP_AMAX_E4M3FN:
        case QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN:
            expectedDtype = DataType::DT_FLOAT8_E4M3FN;
            break;
        case QUANT_MODE_HIF8_CAST:
        case QUANT_MODE_HIF8_PERTENSOR:
        case QUANT_MODE_HIF8_PERTOKEN:
            expectedDtype = DataType::DT_HIFLOAT8;
            break;
        case QUANT_MODE_MXFP4_E2M1:
            expectedDtype = DataType::DT_FLOAT4_E2M1;
            break;
        case QUANT_MODE_INT4_DYNAMIC:
            expectedDtype = DataType::DT_INT4;
            break;
        default:
            break;
    }

    if (expectedDtype != DataType::DT_UNDEFINED) {
        OP_CHECK_IF(
            expandedXDtype_ != expectedDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "expanded_x", Ops::Base::ToString(expandedXDtype_).c_str(),
                ("expected " + Ops::Base::ToString(expectedDtype) + " under quant_mode " + std::to_string(quantMode_))),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckOutputExpandedX()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckOutputExpandedX()");

    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        MIRV3_CHECK_GE_RET(ValidateExpandedXShapeDropPad());
    } else {
        MIRV3_CHECK_GE_RET(ValidateExpandedXShapeDropless());
    }

    MIRV3_CHECK_GE_RET(ValidateExpandedXDtype());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckOutputExpandedRowIdx()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckOutputExpandedRowIdx()");

    auto rank = static_cast<int64_t>(expandedRowIdxShape_.GetDimNum());
    OP_CHECK_IF(rank != RANK_ONE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_row_idx", std::to_string(rank), "1"),
                return ge::GRAPH_FAILED);
    int64_t dim0 = expandedRowIdxShape_.GetDim(0);
    OP_CHECK_IF(dim0 != totalLength_,
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_row_idx dim[0]", std::to_string(dim0),
                                          std::to_string(totalLength_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckOutputExpertTokensCountOrCumsum()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckOutputExpertTokensCountOrCumsum()");

    int64_t expectedRank{-1};
    int64_t expectedDim0{-1};
    int64_t expectedDim1{-1};
    if (expertTokensNumType_ == EXPERT_TOKENS_TYPE_CUMSUM || expertTokensNumType_ == EXPERT_TOKENS_TYPE_COUNT) {
        expectedRank = RANK_ONE;
        expectedDim0 = expertEnd_ - expertStart_;
    } else if (expertTokensNumType_ == EXPERT_TOKENS_TYPE_KEY_VALUE) {
        expectedRank = RANK_TWO;
        expectedDim0 = expertNum_;
        expectedDim1 = DIM_VALUE_TWO;
    }

    auto rank = static_cast<int64_t>(expertTokensCountOrCumsumShape_.GetDimNum());
    if (expectedRank != -1) {
        OP_CHECK_IF(rank != expectedRank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expert_tokens_count_or_cumsum",
                                                 std::to_string(rank), std::to_string(expectedRank)),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim0 != -1) {
        int64_t dim0 = expertTokensCountOrCumsumShape_.GetDim(0);
        OP_CHECK_IF(dim0 != expectedDim0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_tokens_count_or_cumsum dim[0]",
                                              std::to_string(dim0), std::to_string(expectedDim0)),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim1 != -1) {
        int64_t dim1 = expertTokensCountOrCumsumShape_.GetDim(1);
        OP_CHECK_IF(dim1 != expectedDim1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expert_tokens_count_or_cumsum dim[1]",
                                              std::to_string(dim1), std::to_string(expectedDim1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeInitRoutingV3TilingArch35::CalculateExpectedScaleShape(int64_t &expectedRank, int64_t &expectedDim0,
                                                               int64_t &expectedDim1, int64_t &expectedDim2)
{
    if ((quantMode_ == QUANT_MODE_UNQUANT && isInputScale_ == 1) || IsAnyDynamicQuantCase()) {
        if (IsMXFPXNoQuantCase(quantMode_, xDtype_)) {
            expectedRank = RANK_THREE;
            expectedDim0 = totalLength_;
            expectedDim1 = Ops::Base::CeilDiv(xShape_.GetDim(1), MXFPX_SCALE_BLOCK_SIZE);
            expectedDim2 = NUM_TWO;
        } else {
            expectedRank = RANK_ONE;
            expectedDim0 = (dropPadMode_ == DROP_PAD_MODE_DROPPAD) ? expertNum_ * expertCapacity_ : totalLength_;
        }
    } else if ((quantMode_ == QUANT_MODE_MXFP8_E5M2) || (quantMode_ == QUANT_MODE_MXFP8_E4M3FN) ||
               (quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2) ||
               (quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN)) {
        expectedRank = RANK_TWO;
        expectedDim0 = totalLength_;
        expectedDim1 = Ops::Base::CeilAlign<int64_t>(Ops::Base::CeilDiv<int64_t>(cols_, MX_QUANT_BLOCK_SIZE), 2LL);
    } else if ((quantMode_ == QUANT_MODE_HIF8_PERTOKEN)) {
        expectedRank = RANK_ONE;
        expectedDim0 = totalLength_;
    } else if (quantMode_ == QUANT_MODE_FP8_GROUP_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_E4M3FN ||
               quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN) {
        expectedRank = RANK_TWO;
        expectedDim0 = totalLength_;
        expectedDim1 = Ops::Base::CeilDiv<int64_t>(cols_, FP8_GROUP_SIZE);
    } else if (quantMode_ == QUANT_MODE_MXFP4_E2M1 || quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 ||
               quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
        expectedRank = RANK_THREE;
        expectedDim0 = totalLength_;
        if (quantMode_ == QUANT_MODE_MXFP4_E2M1) {
            expectedDim1 = Ops::Base::CeilDiv(cols_, UB_BLOCK_SIZE * NUM_TWO);
        } else {
            expectedDim1 = Ops::Base::CeilDiv(cols_, FP8_PERBLOCK_BLOCK_SIZE * NUM_TWO);
        }
        expectedDim2 = NUM_TWO;
    }
}

ge::graphStatus MoeInitRoutingV3TilingArch35::ValidateScaleShape(int64_t expectedRank, int64_t expectedDim0,
                                                                 int64_t expectedDim1, int64_t expectedDim2)
{
    auto rank = static_cast<int64_t>(expandedScaleShape_.GetDimNum());
    if (expectedRank != -1) {
        OP_CHECK_IF(rank != expectedRank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_scale", std::to_string(rank),
                                                 std::to_string(expectedRank)),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim0 != -1) {
        int64_t dim0 = expandedScaleShape_.GetDim(0);
        OP_CHECK_IF(dim0 != expectedDim0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_scale dim[0]", std::to_string(dim0),
                                              std::to_string(expectedDim0)),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim1 != -1) {
        int64_t dim1 = expandedScaleShape_.GetDim(1);
        OP_CHECK_IF(dim1 != expectedDim1,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_scale dim[1]", std::to_string(dim1),
                                              std::to_string(expectedDim1)),
                    return ge::GRAPH_FAILED);
    }
    if (expectedDim2 != -1) {
        int64_t dim2 = expandedScaleShape_.GetDim(2);
        OP_CHECK_IF(dim2 != expectedDim2,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_scale dim[2]", std::to_string(dim2),
                                              std::to_string(expectedDim2)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckOutputExpandedScale()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckOutputExpandedScale()");

    if (quantMode_ == QUANT_MODE_STATIC || quantMode_ == QUANT_MODE_HIF8_CAST) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t expectedRank{-1};
    int64_t expectedDim0{-1};
    int64_t expectedDim1{-1};
    int64_t expectedDim2{-1};

    CalculateExpectedScaleShape(expectedRank, expectedDim0, expectedDim1, expectedDim2);
    return ValidateScaleShape(expectedRank, expectedDim0, expectedDim1, expectedDim2);
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckOutputExpandedTopkWeight()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckOutputExpandedTopkWeight()");

    if (isInputTopkWeight_ == 0) {
        return ge::GRAPH_SUCCESS; // topk_weight不传入时，expanded_topk_weight也不传入，无需校验
    }

    // shape校验：expanded_topk_weight必须为2D
    auto rank = static_cast<int64_t>(expandedTopkWeightShape_.GetDimNum());
    OP_CHECK_IF(
        rank != RANK_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "expanded_topk_weight", std::to_string(rank), "2"),
        return ge::GRAPH_FAILED);
    // dim[0]校验：DropPad模式为expertNum*expertCapacity，DropLess模式为totalLength
    int64_t dim0 = expandedTopkWeightShape_.GetDim(0);
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        OP_CHECK_IF(dim0 != expertNum_ * expertCapacity_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_topk_weight dim[0]",
                                              std::to_string(dim0), std::to_string(expertNum_ * expertCapacity_)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(dim0 != totalLength_,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_topk_weight dim[0]",
                                              std::to_string(dim0), std::to_string(totalLength_)),
                    return ge::GRAPH_FAILED);
    }
    // dim[1]校验：必须为1
    int64_t dim1 = expandedTopkWeightShape_.GetDim(1);
    OP_CHECK_IF(
        dim1 != DIM_VALUE_ONE,
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "expanded_topk_weight dim[1]", std::to_string(dim1), "1"),
        return ge::GRAPH_FAILED);
    // dtype校验：expanded_topk_weight必须为DT_FLOAT
    OP_CHECK_IF(expandedTopkWeightDtype_ != ge::DataType::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "expanded_topk_weight",
                                          Ops::Base::ToString(expandedTopkWeightDtype_).c_str(), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRoutingV3TilingArch35::CheckOutputs()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CheckOutputs()");

    MIRV3_CHECK_GE_RET(CheckOutputExpandedX());
    MIRV3_CHECK_GE_RET(CheckOutputExpandedRowIdx());
    MIRV3_CHECK_GE_RET(CheckOutputExpertTokensCountOrCumsum());
    MIRV3_CHECK_GE_RET(CheckOutputExpandedScale());
    MIRV3_CHECK_GE_RET(CheckOutputExpandedTopkWeight());

    return ge::GRAPH_SUCCESS;
}

void MoeInitRoutingV3TilingArch35::LogBaseTilingData()
{
    std::stringstream ss;
    ss << "\n[TilingKey]\n" << tilingKey_ << "\n[WorkspaceSize]\n" << workspaceSize_ << "\n";
    ss << "[MoeInitRoutingV3Arch35TilingData]\n";
    ss << "coreNum = " << tilingDataPtr_->coreNum << "\n";
    ss << "n = " << tilingDataPtr_->n << "\n";
    ss << "cols = " << tilingDataPtr_->cols << "\n";
    ss << "k = " << tilingDataPtr_->k << "\n";
    ss << "expertStart = " << tilingDataPtr_->expertStart << "\n";
    ss << "expertEnd = " << tilingDataPtr_->expertEnd << "\n";
    ss << "actualExpertNum = " << tilingDataPtr_->actualExpertNum << "\n";
    ss << "quantMode = " << tilingDataPtr_->quantMode << "\n";
    ss << "rowIdxType = " << tilingDataPtr_->rowIdxType << "\n";
    ss << "useGatherCopy = " << tilingDataPtr_->useGatherCopy << "\n";
    ss << "isInputScale = " << tilingDataPtr_->isInputScale << "\n";
    ss << "isInputOffset = " << tilingDataPtr_->isInputOffset << "\n";
    ss << "expertNum = " << tilingDataPtr_->expertNum << "\n";
    ss << "expertTokensNumType = " << tilingDataPtr_->expertTokensNumType << "\n";
    ss << "expertTokensNumFlag = " << tilingDataPtr_->expertTokensNumFlag << "\n";
    ss << "gatherFirstFullload = " << tilingDataPtr_->gatherFirstFullload << "\n";
    ss << "epFullload = " << tilingDataPtr_->epFullload << "\n";
    ss << "activeNum = " << tilingDataPtr_->activeNum << "\n";
    ss << "dropPadMode = " << tilingDataPtr_->dropPadMode << "\n";
    ss << "smoothType = " << tilingDataPtr_->smoothType << "\n";
    ss << "isInputTopkWeight = " << tilingDataPtr_->isInputTopkWeight << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3TilingArch35::LogVbsTilingData()
{
    std::stringstream ss;
    auto vbsTiling = &(tilingDataPtr_->vbsComputeParamsOp);
    ss << "\n[MoeV3Arch35VBSComputeTilingData]\n";
    ss << "needCoreNum = " << vbsTiling->needCoreNum << "\n";
    ss << "perCoreElements = " << vbsTiling->perCoreElements << "\n";
    ss << "perCoreLoops = " << vbsTiling->perCoreLoops << "\n";
    ss << "perCorePerLoopElements = " << vbsTiling->perCorePerLoopElements << "\n";
    ss << "perCoreLastLoopElements = " << vbsTiling->perCoreLastLoopElements << "\n";
    ss << "lastCoreElements = " << vbsTiling->lastCoreElements << "\n";
    ss << "lastCoreLoops = " << vbsTiling->lastCoreLoops << "\n";
    ss << "lastCorePerLoopElements = " << vbsTiling->lastCorePerLoopElements << "\n";
    ss << "lastCoreLastLoopElements = " << vbsTiling->lastCoreLastLoopElements << "\n";
    ss << "oneLoopMaxElements = " << vbsTiling->oneLoopMaxElements << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3TilingArch35::LogVmsMiddleTilingData()
{
    std::stringstream ss;
    auto vmsMiddleTiling = &(tilingDataPtr_->vmsMiddleComputeParamsOp);
    ss << "\n[MoeV3Arch35VMSMiddleComputeTilingData]\n";
    ss << "needCoreNum = " << vmsMiddleTiling->needCoreNum << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3TilingArch35::LogSortOutTilingData()
{
    std::stringstream ss;
    auto sortOutTiling = &(tilingDataPtr_->sortOutComputeParamsOp);
    ss << "\n[MoeV3Arch35SortOutComputeTilingData]\n";
    ss << "oneLoopMaxElements = " << sortOutTiling->oneLoopMaxElements << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3TilingArch35::LogExpertTokensCountTilingData()
{
    std::stringstream ss;
    auto expertTokensCountTiling = &(tilingDataPtr_->expertTokensCountTilingDataOp);
    ss << "\n[MoeV3Arch35ExpertTokensCountTilingData]\n";
    ss << "needCoreNum = " << expertTokensCountTiling->needCoreNum << "\n";
    ss << "perCoreElements = " << expertTokensCountTiling->perCoreElements << "\n";
    ss << "lastCoreElements = " << expertTokensCountTiling->lastCoreElements << "\n";
    ss << "perCoreLoops = " << expertTokensCountTiling->perCoreLoops << "\n";
    ss << "perCorePerLoopElements = " << expertTokensCountTiling->perCorePerLoopElements << "\n";
    ss << "perCoreLastLoopElements = " << expertTokensCountTiling->perCoreLastLoopElements << "\n";
    ss << "lastCoreLoops = " << expertTokensCountTiling->lastCoreLoops << "\n";
    ss << "lastCorePerLoopElements = " << expertTokensCountTiling->lastCorePerLoopElements << "\n";
    ss << "lastCoreLastLoopElements = " << expertTokensCountTiling->lastCoreLastLoopElements << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3TilingArch35::LogGatherOutTilingData()
{
    std::stringstream ss;
    auto gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    ss << "\n[MoeV3Arch35GatherOutComputeTilingData]\n";
    ss << "needCoreNum = " << gatherOutTiling->needCoreNum << "\n";
    ss << "perCoreIndicesElements = " << gatherOutTiling->perCoreIndicesElements << "\n";
    ss << "lastCoreIndicesElements = " << gatherOutTiling->lastCoreIndicesElements << "\n";
    ss << "perCoreIndicesLoops = " << gatherOutTiling->perCoreIndicesLoops << "\n";
    ss << "perCorePerLoopIndicesElements = " << gatherOutTiling->perCorePerLoopIndicesElements << "\n";
    ss << "perCoreLastLoopIndicesElements = " << gatherOutTiling->perCoreLastLoopIndicesElements << "\n";
    ss << "lastCoreIndicesLoops = " << gatherOutTiling->lastCoreIndicesLoops << "\n";
    ss << "lastCorePerLoopIndicesElements = " << gatherOutTiling->lastCorePerLoopIndicesElements << "\n";
    ss << "colsLoops = " << gatherOutTiling->colsLoops << "\n";
    ss << "perLoopCols = " << gatherOutTiling->perLoopCols << "\n";
    ss << "lastLoopCols = " << gatherOutTiling->lastLoopCols << "\n";
    ss << "activeNum = " << gatherOutTiling->activeNum << "\n";
    ss << "useCompactGatherOutDropPad = " << gatherOutTiling->useCompactGatherOutDropPad << "\n";
    ss << "xCopyInQueueBufferNum = " << gatherOutTiling->xCopyInQueueBufferNum << "\n";
    OP_LOGI(context_, "%s", ss.str().c_str());
}

void MoeInitRoutingV3TilingArch35::Tiling4VBSOneCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4VBSOneCoreCompute(...)");

    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = totalLength_;
    vbsTiling->perCoreLoops = 1;
    vbsTiling->perCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

void MoeInitRoutingV3TilingArch35::Tiling4VBSMultiCoreCompute(MoeV3Arch35VBSComputeTilingData *vbsTiling)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4VBSMultiCoreCompute(...)");

    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, sortLoopMaxElement_); // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));      // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivCoreNum_);                            // 不能超过物理核数

    OP_CHECK_IF(needCoreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "needCoreNum", std::to_string(needCoreNum),
                                                      "needCoreNum cannot be 0"),
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

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops =
            Ops::Base::CeilDiv(vbsTiling->perCoreElements, sortLoopMaxElement_); // 每个核处理的loop数
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, sortLoopMaxElement_);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = totalLength_ - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops),
                               SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
    OP_CHECK_IF(
        vbsTiling->lastCoreLastLoopElements <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "vbsTiling->lastCoreLastLoopElements",
                                              std::to_string(vbsTiling->lastCoreLastLoopElements), "vbs tiling failed"),
        ;);
}

void MoeInitRoutingV3TilingArch35::Tiling4VBSCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4VBSCompute()");

    if (totalLength_ <= sortLoopMaxElement_) { // 排序只用到一个核排序
        sortMode_ = 0;
    } else {
        sortMode_ = 1;
    }

    auto *vbsTiling = &(tilingDataPtr_->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = sortLoopMaxElement_;
    if (sortMode_ == 0) { // 只用到一个核
        Tiling4VBSOneCoreCompute(vbsTiling);
    } else {
        Tiling4VBSMultiCoreCompute(vbsTiling);
    }

    LogVbsTilingData();
}

void MoeInitRoutingV3TilingArch35::Tiling4VMSMiddleCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4VMSMiddleCompute()");

    auto *vbsTiling = &(tilingDataPtr_->vbsComputeParamsOp);
    auto *vmsMiddleTiling = &(tilingDataPtr_->vmsMiddleComputeParamsOp);
    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        vmsMiddleTiling->needCoreNum = 0;         // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum; // 需要的核数

    LogVmsMiddleTilingData();
}

void MoeInitRoutingV3TilingArch35::Tiling4SortOutCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4SortOutCompute()");

    auto *sortOutTiling = &(tilingDataPtr_->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;

    LogSortOutTilingData();
}

void MoeInitRoutingV3TilingArch35::Tiling4ExpertTokensCountCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4ExpertTokensCountCompute()");

    auto *tokensCountTiling = &(tilingDataPtr_->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingDataPtr_->n * tilingDataPtr_->k;
    int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, aivCoreNum_);
    int64_t needCoreNum = Ops::Base::CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingDataPtr_->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE) ?
                                   tilingDataPtr_->actualExpertNum :
                                   (tilingDataPtr_->actualExpertNum + 1) * DIM_VALUE_TWO;
    int64_t maxElementsPerLoop =
        (availUbSize_ -
         Ops::Base::CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = Ops::Base::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = Ops::Base::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;

    LogExpertTokensCountTilingData();
}

MultipleParams MoeInitRoutingV3TilingArch35::GetMultipleParams()
{
    MultipleParams params;
    params.colMultiple = NUM_TWO * inputXDtypeSize_;
    params.rowMultiple = NUM_TWO;
    if (quantMode_ == QUANT_MODE_STATIC) {
        // 静态量化：输入(双缓冲) + 输出int8(双缓冲) + float中间buffer + half中间buffer
        // colMultiple = 2*inputXDtypeSize + 2*1 + 4 + 2 = 2*inputXDtypeSize + 8
        params.colMultiple = NUM_TWO * inputXDtypeSize_ + NUM_TWO + sizeof(float) + sizeof(uint16_t);
        params.rowMultiple = STATIC_QUANT_ROW_MULTIPLE;
    } else if (IsAnyDynamicQuantCase()) {
        params.colMultiple = DYNAMIC_QUANT_COLS_BUFFER;
        params.rowMultiple = NUM_FOUR;
    } else if (quantMode_ == QUANT_MODE_HIF8_CAST && xDtype_ == ge::DataType::DT_BF16) {
        // 当BF16->FP32->HIF8转换时，额外需要存储FP32的中间结果
        params.colMultiple = NUM_TWO * (inputXDtypeSize_ + inputXDtypeSize_ * BF16_TO_FP32_SIZE_FACTOR);
    } else if (quantMode_ == QUANT_MODE_HIF8_PERTOKEN) {
        params.colMultiple = HIF8_PERTOKEN_QUANT_COLS_BUFFER;
        params.rowMultiple = NUM_FOUR;
    } else if (quantMode_ == QUANT_MODE_HIF8_PERTENSOR) {
        params.colMultiple = HIF8_PERTENSOR_QUANT_COLS_BUFFER;
        params.rowMultiple = NUM_FOUR;
    }
    return params;
}

PerLoopParams MoeInitRoutingV3TilingArch35::GetPerLoopParams(MultipleParams &multipleParams,
                                                             int64_t perCoreIndicesElements)
{
    PerLoopParams perLoopParams;
    perLoopParams.perLoopCols = tilingDataPtr_->cols;
    if (quantMode_ == QUANT_MODE_HIF8_PERTENSOR) {
        perLoopParams.perLoopMaxIndicesElements =
            (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple) /
            multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        while (perLoopParams.perLoopMaxIndicesElements <= 0 && perLoopParams.perLoopCols > 1) {
            perLoopParams.perLoopCols = Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO);
            perLoopParams.perLoopMaxIndicesElements =
                (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple) /
                multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        }
        perLoopParams.perLoopMaxIndicesElements =
            std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);
    } else if (IsMXFPXNoQuantCase(quantMode_, xDtype_)) {
        perLoopParams.perLoopCols = Ops::Base::CeilAlign(perLoopParams.perLoopCols, MXFPX_SCALE_BLOCK_SIZE);
        perLoopParams.perLoopMaxIndicesElements =
            (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple -
             Align(perLoopParams.perLoopCols / SCALE_FACTOR_WITH_X, inputScaleDTypeSize_) * inputScaleDTypeSize_ *
                 NUM_TWO) /
            multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        while (perLoopParams.perLoopMaxIndicesElements <= 0) {
            perLoopParams.perLoopCols =
                Ops::Base::CeilAlign(Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO), MXFPX_SCALE_BLOCK_SIZE);
            perLoopParams.perLoopMaxIndicesElements =
                (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple -
                 Align(perLoopParams.perLoopCols / SCALE_FACTOR_WITH_X, inputScaleDTypeSize_) * inputScaleDTypeSize_ *
                     NUM_TWO) /
                multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        }
    } else {
        if (quantMode_ == QUANT_MODE_UNQUANT && dropPadMode_ == DROP_PAD_MODE_DROPLESS) {
            SetPerLoopParams4NoQuantDropLess(perLoopParams, perCoreIndicesElements);
        } else {
            SetPerLoopParams4NoQuantDropPad(multipleParams, perLoopParams, perCoreIndicesElements);
        }
    }
    return perLoopParams;
}

void MoeInitRoutingV3TilingArch35::AlignInt4DynamicQuantPerLoopCols(PerLoopParams &perLoopParams) const
{
    if (quantMode_ != QUANT_MODE_INT4_DYNAMIC || perLoopParams.perLoopCols >= tilingDataPtr_->cols ||
        perLoopParams.perLoopCols % NUM_TWO == 0) {
        return;
    }
    perLoopParams.perLoopCols = std::max(perLoopParams.perLoopCols - 1, NUM_TWO);
}

void MoeInitRoutingV3TilingArch35::SetPerLoopParams4NoQuantDropLess(PerLoopParams &perLoopParams,
                                                                    const int64_t perCoreIndicesElements)
{
    perLoopParams.perLoopMaxIndicesElements = availUbSize_ / NUM_TWO /
                                              (AlignBytes(perLoopParams.perLoopCols, inputXDtypeSize_) +
                                               AlignBytes(1, sizeof(float)) + static_cast<int64_t>(sizeof(int32_t)));
    while (perLoopParams.perLoopMaxIndicesElements <= 0) {
        perLoopParams.perLoopCols = Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO);
        perLoopParams.perLoopMaxIndicesElements =
            availUbSize_ / NUM_TWO /
            (AlignBytes(perLoopParams.perLoopCols, inputXDtypeSize_) + AlignBytes(1, sizeof(float)) +
             static_cast<int64_t>(sizeof(int32_t)));
    }
    perLoopParams.perLoopMaxIndicesElements = std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t rowIdxQueueSize = AlignBytes(perLoopParams.perLoopMaxIndicesElements, sizeof(int32_t));
    int64_t xQueueSize =
        perLoopParams.perLoopMaxIndicesElements * AlignBytes(perLoopParams.perLoopCols, inputXDtypeSize_);
    int64_t scaleQueueSize = perLoopParams.perLoopMaxIndicesElements * AlignBytes(1, sizeof(float));

    int64_t baseMemory = rowIdxQueueSize * NUM_TWO + xQueueSize * NUM_TWO + scaleQueueSize * NUM_TWO;

    int64_t remainingSpace = availUbSize_ - baseMemory;
    int64_t additionalBufferNum = remainingSpace / xQueueSize;
    perLoopParams.xCopyInQueueBufferNum = GetXBufferNum(additionalBufferNum);
}

void MoeInitRoutingV3TilingArch35::SetPerLoopParams4NoQuantDropPad(const MultipleParams &multipleParams,
                                                                   PerLoopParams &perLoopParams,
                                                                   const int64_t perCoreIndicesElements)
{
    perLoopParams.perLoopMaxIndicesElements =
        (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple -
         UB_BLOCK_SIZE * NUM_TWO) /
        multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    while (perLoopParams.perLoopMaxIndicesElements <= 0 && perLoopParams.perLoopCols > 1) {
        perLoopParams.perLoopCols = Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO);
        perLoopParams.perLoopMaxIndicesElements =
            (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * multipleParams.colMultiple -
             UB_BLOCK_SIZE * NUM_TWO) /
            multipleParams.rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    }
    perLoopParams.perLoopMaxIndicesElements = std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);

    int64_t rowIdxQueueSize = AlignBytes(perLoopParams.perLoopMaxIndicesElements, sizeof(int32_t));
    int64_t xQueueSize = AlignBytes(perLoopParams.perLoopCols, inputXDtypeSize_);
    int64_t scaleQueueSize = AlignBytes(1, sizeof(float));

    int64_t baseMemory = rowIdxQueueSize * NUM_TWO + xQueueSize * NUM_TWO + scaleQueueSize * NUM_TWO;

    int64_t remainingSpace = availUbSize_ - baseMemory;
    int64_t additionalBufferNum = remainingSpace / xQueueSize;
    perLoopParams.xCopyInQueueBufferNum = GetXBufferNum(additionalBufferNum);
}

int64_t MoeInitRoutingV3TilingArch35::GetXBufferNum(const int additionalBufferNum)
{
    if (additionalBufferNum > 0) {
        return std::min(additionalBufferNum + NUM_TWO, MAX_QUEUE_BUFFER_NUM);
    }
    return NUM_TWO;
}

void MoeInitRoutingV3TilingArch35::SetLastCoreIndicesTiling(MoeV3Arch35GatherOutComputeTilingData *gatherOutTiling,
                                                            int64_t lastCoreIndicesElements,
                                                            int64_t perLoopMaxIndicesElements)
{
    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
    gatherOutTiling->activeNum = tilingDataPtr_->activeNum;
}

void MoeInitRoutingV3TilingArch35::Tiling4GatherOutCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4GatherOutCompute()");

    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    MultipleParams multipleParams = GetMultipleParams();
    PerLoopParams perLoopParams = GetPerLoopParams(multipleParams, perCoreIndicesElements);
    int64_t originPerLoopCols = perLoopParams.perLoopCols;
    AlignInt4DynamicQuantPerLoopCols(perLoopParams);
    if (perLoopParams.perLoopCols != originPerLoopCols) {
        SetPerLoopParams4NoQuantDropPad(multipleParams, perLoopParams, perCoreIndicesElements);
        AlignInt4DynamicQuantPerLoopCols(perLoopParams);
    }

    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopParams.perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopParams.perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopParams.perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;
    gatherOutTiling->xCopyInQueueBufferNum = perLoopParams.xCopyInQueueBufferNum;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    SetLastCoreIndicesTiling(gatherOutTiling, lastCoreIndicesElements, perLoopParams.perLoopMaxIndicesElements);

    LogGatherOutTilingData();
    return;
}

void MoeInitRoutingV3TilingArch35::Tiling4GatherOutMxFP8NoQuantCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4GatherOutMxFP8NoQuantCompute()");

    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    PerLoopParams perLoopParams;
    perLoopParams.perLoopCols = tilingDataPtr_->cols;

    perLoopParams.perLoopCols = Ops::Base::CeilAlign(perLoopParams.perLoopCols, MXFPX_SCALE_BLOCK_SIZE);
    perLoopParams.perLoopMaxIndicesElements =
        (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * inputXDtypeSize_ * NUM_TWO -
         Align(perLoopParams.perLoopCols / SCALE_FACTOR_WITH_X, inputScaleDTypeSize_) * inputScaleDTypeSize_ *
             NUM_TWO) /
        NUM_TWO / static_cast<int64_t>(sizeof(int32_t));

    perLoopParams.xCopyInQueueBufferNum = DOUBLE_BUFFER;
    // 当能搬入一整行时，ub才可能充足，才可能适合匹配多buffers的场景
    if (perLoopParams.perLoopMaxIndicesElements > 0) {
        int64_t rowIdxSize = NUM_TWO * static_cast<int64_t>(sizeof(int32_t)) *
                             std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);
        int64_t oneRowSize = Align(perLoopParams.perLoopCols, inputXDtypeSize_) * inputXDtypeSize_ * NUM_TWO +
                             Align(perLoopParams.perLoopCols / SCALE_FACTOR_WITH_X, inputScaleDTypeSize_) *
                                 inputScaleDTypeSize_ * NUM_TWO;
        perLoopParams.xCopyInQueueBufferNum =
            std::min(MAX_QUEUE_BUFFER_NUM, (availUbSize_ - rowIdxSize) / oneRowSize * NUM_TWO);
    }

    while (perLoopParams.perLoopMaxIndicesElements <= 0) {
        perLoopParams.perLoopCols =
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(perLoopParams.perLoopCols, NUM_TWO), MXFPX_SCALE_BLOCK_SIZE);
        perLoopParams.perLoopMaxIndicesElements =
            (availUbSize_ - Align(perLoopParams.perLoopCols, inputXDtypeSize_) * inputXDtypeSize_ * NUM_TWO -
             Align(perLoopParams.perLoopCols / SCALE_FACTOR_WITH_X, inputScaleDTypeSize_) * inputScaleDTypeSize_ *
                 NUM_TWO) /
            NUM_TWO / static_cast<int64_t>(sizeof(int32_t));
    }

    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopParams.perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopParams.perLoopCols;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopParams.perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;
    gatherOutTiling->xCopyInQueueBufferNum = perLoopParams.xCopyInQueueBufferNum;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopParams.perLoopMaxIndicesElements, perCoreIndicesElements);
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    SetLastCoreIndicesTiling(gatherOutTiling, lastCoreIndicesElements, perLoopParams.perLoopMaxIndicesElements);

    LogGatherOutTilingData();
    return;
}

int64_t MoeInitRoutingV3TilingArch35::CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CalcMaxRowIdxPerLoopMxQuant(...)");

    // 输入x[cols]所占大小：cols*sizeof(dtypeX)
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize_);
    // 输出scale[cols]所占大小：scaleCols*sizeof(dtypeX)*2+scaleCols*sizeof(Byte)
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize_) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    // 输出xOut[cols]所占大小：
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    // 返回的是(availUbSize-每行输入x、输出scale、输出xOut所占的大小)/sizeof(int32)，应该是留给sortedRowIdx元素的数目
    return (availUbSize_ / 2 - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

void MoeInitRoutingV3TilingArch35::SetIndicesLoopParams4GatherOut(int64_t perLoopMaxIndicesElements,
                                                                  int64_t perCoreIndicesElements,
                                                                  int64_t lastCoreIndicesElements)
{
    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

void MoeInitRoutingV3TilingArch35::Tiling4GatherOutMxQuant()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4GatherOutMxQuant()");

    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = Ops::Base::CeilAlign(tilingDataPtr_->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols);
    while (perLoopMaxIndicesElements <= 0 && perLoopCols > MX_QUANT_BLOCK_SIZE) {
        perLoopCols = Ops::Base::CeilAlign(Ops::Base::CeilDiv(perLoopCols, NUM_TWO), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols);
    }
    if (perLoopMaxIndicesElements <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "availUbSize, cols",
            (std::to_string(availUbSize_) + ", " + std::to_string(tilingDataPtr_->cols)),
            "UB space insufficient for MX quantization");
        return;
    }
    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->perCoreIndicesElements =
        perCoreIndicesElements; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->lastCoreIndicesElements =
        lastCoreIndicesElements; // 没用这个，kernel根据读取到的expertTotalCount重新计算tiling相关值
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    SetIndicesLoopParams4GatherOut(perLoopMaxIndicesElements, perCoreIndicesElements, lastCoreIndicesElements);
    gatherOutTiling->activeNum = tilingDataPtr_->activeNum;

    LogGatherOutTilingData();
    return;
}

int64_t MoeInitRoutingV3TilingArch35::CalcMaxRowIdxPerLoopFP8Quant(int64_t perLoopCols)
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::CalcMaxRowIdxPerLoopFP8Quant(...)");

    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize_);
    int64_t xOutSize = AlignBytes(perLoopCols, sizeof(uint8_t));
    int64_t scaleSize =
        std::max(AlignBytes(Ops::Base::CeilDiv(perLoopCols, FP8_PERBLOCK_BLOCK_SIZE), sizeof(float)), REG_SIZE);
    return (availUbSize_ / DOUBLE_BUFFER - (xInSize + xOutSize + scaleSize)) / static_cast<int64_t>(sizeof(int32_t));
}

int64_t MoeInitRoutingV3TilingArch35::CalcMaxRowIdxPerLoopFP8GroupQuant(int64_t perLoopCols)
{
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize_);
    int64_t xOutSize = AlignBytes(perLoopCols, sizeof(uint8_t));
    int64_t scaleSize = AlignBytes(Ops::Base::CeilDiv(perLoopCols, FP8_GROUP_SIZE), sizeof(float));
    int64_t tmpBufSize = std::max(scaleSize, REG_SIZE);
    return (availUbSize_ / DOUBLE_BUFFER - (xInSize + xOutSize + scaleSize + tmpBufSize * NUM_TWO)) /
           static_cast<int64_t>(sizeof(int32_t));
}

void MoeInitRoutingV3TilingArch35::Tiling4GatherOutFP8Quant()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4GatherOutFP8Quant()");

    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    bool isFp8Group = (quantMode_ == QUANT_MODE_FP8_GROUP_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_E4M3FN ||
                       quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN);
    int64_t perLoopCols = Ops::Base::CeilAlign(tilingDataPtr_->cols, FP8_PERBLOCK_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements =
        isFp8Group ? CalcMaxRowIdxPerLoopFP8GroupQuant(perLoopCols) : CalcMaxRowIdxPerLoopFP8Quant(perLoopCols);
    while (perLoopMaxIndicesElements <= 0 && perLoopCols > FP8_PERBLOCK_BLOCK_SIZE) {
        perLoopCols = Ops::Base::CeilAlign(Ops::Base::CeilDiv(perLoopCols, NUM_TWO), FP8_PERBLOCK_BLOCK_SIZE);
        perLoopMaxIndicesElements =
            isFp8Group ? CalcMaxRowIdxPerLoopFP8GroupQuant(perLoopCols) : CalcMaxRowIdxPerLoopFP8Quant(perLoopCols);
    }
    if (perLoopMaxIndicesElements <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "availUbSize and cols",
            (std::to_string(availUbSize_) + ", " + std::to_string(tilingDataPtr_->cols)).c_str(),
            "UB space is insufficient for FP8 quantization.");
        return;
    }
    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    SetLastCoreIndicesTiling(gatherOutTiling, lastCoreIndicesElements, perLoopMaxIndicesElements);

    LogGatherOutTilingData();
    return;
}

bool MoeInitRoutingV3TilingArch35::IsFullLoad()
{
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        return false;
    }
    int64_t perCoreTokens = 1;
    if (expertStart_ == 0 && expertEnd_ == expertNum_) {
        ep_ = 0;
        if (!IsAnyDynamicQuantCase()) {
            int64_t perCoreTokensEst = n_ / aivCoreNum_;
            int64_t remainder = n_ % aivCoreNum_;
            perCoreTokens = remainder <= 1 ? perCoreTokensEst + 1 : perCoreTokensEst + NUM_TWO;
        }
    } else {
        ep_ = 1;
        perCoreTokens = 1;
    }
    tilingDataPtr_->epFullload = ep_;

    if (totalLength_ > sortLoopMaxElement_) {
        return false;
    }

    int64_t tileLength = Align(totalLength_, static_cast<int64_t>(sizeof(int32_t)));
    int64_t sortNum = Ops::Base::CeilDiv(tileLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT;
    int64_t sortSpace = sortNum * sizeof(int32_t) * ONE_CORE_SORT_BUFFER;
    int64_t rowIdxSpace = sortNum * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace = Ops::Base::CeilDiv(expertNum_ * static_cast<int64_t>(sizeof(int64_t)), UB_BLOCK_SIZE) *
                          UB_BLOCK_SIZE * NUM_THREE;
    int64_t gatherSpace = Ops::Base::CeilDiv(cols_ * inputXDtypeSize_, UB_BLOCK_SIZE) * UB_BLOCK_SIZE * perCoreTokens;
    int64_t remainUb = availUbSize_ - sortSpace - rowIdxSpace - expertSpace - LENGTH_1024;

    if (quantMode_ == QUANT_MODE_UNQUANT) {
        remainUb -= (gatherSpace + UB_BLOCK_SIZE);
    } else if (quantMode_ == QUANT_MODE_STATIC) {
        int64_t xAlignedCount = Align(cols_, static_cast<int64_t>(sizeof(int8_t)));
        int64_t quantSpace = xAlignedCount * STATIC_QUANT_FULLLOAD_COLS_BUFFER * perCoreTokens;
        remainUb -= (gatherSpace + quantSpace);
    } else if (IsAnyDynamicQuantCase()) {
        if (quantMode_ == QUANT_MODE_INT4_DYNAMIC) {
            int64_t inputXInSpace = inputXDtypeSize_ == static_cast<int64_t>(sizeof(float)) ?
                                        AlignBytes(cols_, sizeof(float)) :
                                        BF16_TO_FP32_SIZE_FACTOR * AlignBytes(cols_, inputXDtypeSize_);
            inputXInSpace *= NUM_TWO;
            int64_t smoothInSpace = AlignBytes(cols_, sizeof(float));
            // INT4 output is cols/2 bytes (2 INT4 values packed per byte).
            int64_t inputXOutSpace = AlignBytes(Ops::Base::CeilDiv(cols_, NUM_TWO), sizeof(int8_t));
            int64_t calcSpace = AlignBytes(cols_, sizeof(float));
            int64_t scaleOutSpace = UB_BLOCK_SIZE * NUM_TWO;
            remainUb -= (inputXInSpace + smoothInSpace + calcSpace + inputXOutSpace + scaleOutSpace);
        } else {
            int64_t xAlignedCount = Align(cols_, UB_BLOCK_SIZE);
            int64_t quantSpace = xAlignedCount * DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER;
            int64_t scaleOutSpace = UB_BLOCK_SIZE * NUM_TWO;
            remainUb -= (quantSpace + scaleOutSpace);
        }
    }

    return remainUb > 0;
}

bool MoeInitRoutingV3TilingArch35::IsCountingSortApplicable()
{
    if (expertNum_ > MAX_EXPERT_NUM) {
        return false;
    }
    // 量化模式白名单：CS kernel 仅有非量化/静态量化两套模板，其余量化模式无对应 CS 模板
    if (quantMode_ != QUANT_MODE_UNQUANT && quantMode_ != QUANT_MODE_STATIC) {
        return false;
    }
    int64_t actualExpertNum = expertEnd_ - expertStart_;

    if (actualExpertNum <= 0 || actualExpertNum > CS_MAX_ACTUAL_EXPERT_NUM) {
        // expertIdx 值域必须能映射到 uint8（actualExpertNum ≤ 256）
        return false;
    }

    // dropless 时要求 actualExpertNum*2 <= expertNum（实际专家数不超过一半，计数排序才有收益）
    if (dropPadMode_ == DROP_PAD_MODE_DROPLESS) {
        if (actualExpertNum != expertNum_ && actualExpertNum * NUM_TWO > expertNum_) {
            return false;
        }
    } else if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) { // 非全载模板
        // dropPad 时 n >= 512
        if (n_ < MAX_EXPERT_NUM / 2) {
            return false;
        }
    }
    return true;
}

void MoeInitRoutingV3TilingArch35::ComputeCountingSortMode()
{
    countingSortMode_ = 0;
    tilingDataPtr_->countingSortParamsOp.countingSortMode = 0;

    if (isFullload_ || !IsCountingSortApplicable()) {
        return;
    }

    int64_t actualExpertNum = expertEnd_ - expertStart_;
    int64_t perCoreTokensEst = Ops::Base::CeilDiv(n_, aivCoreNum_);
    int64_t estUb = EstimateArch35CountingSortFullLoadUB(perCoreTokensEst);
    bool expandDtypeValid = (expandedXDtype_ == ge::DT_INT8 || expandedXDtype_ == ge::DT_FLOAT16 ||
                             expandedXDtype_ == ge::DT_BF16 || expandedXDtype_ == ge::DT_FLOAT);
    // bool fullLoadCond = estUb > 0 && estUb <= availUbSize_ - LENGTH_1024 && quantMode_ == QUANT_MODE_UNQUANT &&
    //                     k_ == CS_PERFORMANCE_MODE_TOP_K && dropPadMode_ == DROP_PAD_MODE_DROPLESS &&
    //                     expertTokensNumType_ == EXPERT_TOKENS_TYPE_COUNT && actualExpertNum <= NUM_32 &&
    //                     expandDtypeValid;
    bool fullLoadCond = estUb > 0 && estUb <= availUbSize_ - LENGTH_1024 && quantMode_ == QUANT_MODE_UNQUANT &&
                        dropPadMode_ == DROP_PAD_MODE_DROPLESS && actualExpertNum <= NUM_32 && expandDtypeValid;
    if (!fullLoadCond) {
        return;
    }

    ComputeArch35CountingSortFullLoadTiling();
    countingSortMode_ = tilingDataPtr_->countingSortParamsOp.countingSortMode;
}

int64_t MoeInitRoutingV3TilingArch35::EstimateArch35CountingSortFullLoadUB(int64_t perCoreTokens)
{
    // 模板1 UB 叠加模型（借鉴 A3 EstimateCountingSortFullLoadUB，SIMT DCache 已在 availUbSize_ 扣除）
    if (perCoreTokens <= 0) {
        return -1;
    }
    int64_t coreEntries = perCoreTokens * k_;
    int64_t entriesAligned = Ops::Base::CeilDiv(coreEntries, static_cast<int64_t>(64)) * 64;
    int64_t maskBytes =
        Ops::Base::CeilAlign(Ops::Base::CeilDiv(entriesAligned, static_cast<int64_t>(8)), UB_BLOCK_SIZE);
    int64_t expertCountStride = Ops::Base::CeilAlign(expertEnd_ - expertStart_, CS_ONE_BLOCK_ELEMENT);
    int64_t colsAligned = Ops::Base::CeilAlign(cols_ * inputXDtypeSize_, UB_BLOCK_SIZE) / inputXDtypeSize_;

    int64_t total = 0;
    // x 行全载
    total += Ops::Base::CeilAlign(perCoreTokens * colsAligned * inputXDtypeSize_, UB_BLOCK_SIZE);
    // expertIdx 全载
    total += Ops::Base::CeilAlign(coreEntries * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE);
    // scale 全载：per-token 标量缓冲仅非量化透传路径需要；动态量化的 (LE,H) smooth 走单行缓冲（下方量化临时区）
    if (isInputScale_ == 1 && quantMode_ != QUANT_MODE_DYNAMIC) {
        total += Ops::Base::CeilAlign(
            perCoreTokens * (UB_BLOCK_SIZE / static_cast<int64_t>(sizeof(float))) * static_cast<int64_t>(sizeof(float)),
            UB_BLOCK_SIZE);
    }
    // expertCount 本核
    total += Ops::Base::CeilAlign(expertCountStride * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE);
    // allCoreExpertCount（Phase B）
    total +=
        Ops::Base::CeilAlign(aivCoreNum_ * expertCountStride * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE);
    // expertTokens
    total += Ops::Base::CeilAlign((expertEnd_ - expertStart_) * static_cast<int64_t>(sizeof(int64_t)), UB_BLOCK_SIZE);
    // prefixSum
    total += Ops::Base::CeilAlign(expertCountStride * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE);
    // filteredPairs (pairs)
    total += Ops::Base::CeilAlign(coreEntries * NUM_TWO * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE);
    // expertIdxFp32
    total += Ops::Base::CeilAlign(entriesAligned * static_cast<int64_t>(sizeof(float)), UB_BLOCK_SIZE);
    // 3 个 mask
    total += maskBytes * NUM_THREE;
    // gatheredExpert / flatIdx / gatheredIdx
    total += Ops::Base::CeilAlign(entriesAligned * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE) * NUM_THREE;
    // 量化临时区（quantMode_ != UNQUANT）：单行 int8 + scale slot（scaleSlotSize=8 floats=32B），
    // 与 kernel ComputeUbLayout 镜像，避免 mode-1 UB 预算偏小被静默降级到 mode-2
    if (quantMode_ != QUANT_MODE_UNQUANT) {
        total += Ops::Base::CeilAlign(colsAligned * static_cast<int64_t>(sizeof(int8_t)), UB_BLOCK_SIZE);
        total += Ops::Base::CeilAlign(CS_ONE_BLOCK_ELEMENT * static_cast<int64_t>(sizeof(float)), UB_BLOCK_SIZE);
        // 动态量化 per-expert smooth 单行缓冲（(LE,H) 取一行 H 列）
        if (quantMode_ == QUANT_MODE_DYNAMIC && isInputScale_ == 1) {
            total += Ops::Base::CeilAlign(colsAligned * static_cast<int64_t>(sizeof(float)), UB_BLOCK_SIZE);
        }
    }
    // 聚合搬出相关缓冲区仅在 csAggrEnable=1 时申请：bucketBase + offsetTbl + countTbl。
    // 聚合搬出 gatherOutBuf 位于 UB 最前，固定预留 AGGRBUFBYTES_A5。
    int64_t actualExpertNum = expertEnd_ - expertStart_;
    int64_t aggrOutRows = AGGRBUFBYTES_A5 / (colsAligned * inputXDtypeSize_);
    bool aggrEnable = (aggrOutRows >= NUM_TWO) && (actualExpertNum <= CS_FULLLOAD_MAX_ACTUAL_EXPERT_NUM) &&
                      (quantMode_ == QUANT_MODE_UNQUANT);
    if (aggrEnable) {
        total += Ops::Base::CeilAlign(coreEntries * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE);
        total += Ops::Base::CeilAlign(actualExpertNum * static_cast<int64_t>(sizeof(int32_t)), UB_BLOCK_SIZE) * NUM_TWO;
        total += Ops::Base::CeilAlign(static_cast<int64_t>(AGGRBUFBYTES_A5), UB_BLOCK_SIZE);
    }
    return total;
}

void MoeInitRoutingV3TilingArch35::ComputeArch35CountingSortFullLoadTiling()
{
    auto *cs = &tilingDataPtr_->countingSortParamsOp;
    cs->countingSortMode = 1;

    int64_t perCoreTokens = Ops::Base::CeilDiv(n_, aivCoreNum_);
    int64_t needCoreNum = Ops::Base::CeilDiv(n_, perCoreTokens);
    int64_t lastCoreTokens = n_ - perCoreTokens * (needCoreNum - 1);

    cs->filterNeedCoreNum = needCoreNum;
    cs->filterPerCoreTokens = perCoreTokens;
    cs->lastCoreTokens = lastCoreTokens;
    cs->coreEntries = perCoreTokens * k_;
    cs->expertCountStride = Ops::Base::CeilAlign(expertEnd_ - expertStart_, CS_ONE_BLOCK_ELEMENT);
    cs->filterChunkSize = 0; // FullLoad 不使用 chunk
    cs->csPerLoopCols = cols_;
    cs->csColsLoops = 1;
    cs->csLastLoopCols = cols_;
    cs->maxPerLoopEntries = cs->coreEntries;

    // 聚合搬出参数（按专家外循环 + k 行切批）：仅非量化子类消费 csAggrEnable
    // 聚合区在 UB 最前独立预留 10KB（不挤占 xLocal），分桶区已计入 EstimateArch35CountingSortFullLoadUB
    int64_t colsAligned = Ops::Base::CeilAlign(cols_ * inputXDtypeSize_, UB_BLOCK_SIZE) / inputXDtypeSize_;
    int64_t aggrBufBytes = static_cast<int64_t>(AGGRBUFBYTES_A5);
    int64_t aggrOutRows = aggrBufBytes / (colsAligned * inputXDtypeSize_);
    int64_t actualExpertNum = expertEnd_ - expertStart_;
    // 启用判定：k>=2（聚合区至少容纳 2 行）、桶数受限、非量化
    // xLocal 不再被切分，搬入区完整保留，无需容量校验
    bool aggrEnable = (aggrOutRows >= NUM_TWO) && (actualExpertNum <= CS_FULLLOAD_MAX_ACTUAL_EXPERT_NUM) &&
                      (quantMode_ == QUANT_MODE_UNQUANT);
    cs->csAggrEnable = aggrEnable ? 1 : 0;
    cs->csAggrOutRows = aggrEnable ? aggrOutRows : 0;
    cs->csAggrOutBufBytes = aggrEnable ? aggrBufBytes : 0;

    OP_LOGD(context_,
            "CountingSort FullLoad: needCoreNum=%ld, perCoreTokens=%ld, coreEntries=%ld "
            "csAggrEnable=%ld, csAggrOutRows=%ld, csAggrOutBufBytes=%ld",
            needCoreNum, perCoreTokens, cs->coreEntries, cs->csAggrEnable, cs->csAggrOutRows, cs->csAggrOutBufBytes);
}

bool MoeInitRoutingV3TilingArch35::IsSupportGatherCopyKernels() const
{
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        return false;
    }
    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
        quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E5M2 || quantMode_ == QUANT_MODE_MXFP8_ROUNDSCALE_AMAX_E4M3FN ||
        quantMode_ == QUANT_MODE_FP8_GROUP_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_E4M3FN ||
        quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E5M2 || quantMode_ == QUANT_MODE_FP8_GROUP_AMAX_E4M3FN) {
        return true;
    }
    if (IsMXFP8NoQuantCase(quantMode_, xDtype_)) {
        return true;
    }
    return false;
}

void MoeInitRoutingV3TilingArch35::ComputeUseGatherCopy()
{
    if (IsSupportGatherCopyKernels()) {
        // gather搬运收益与有效专家比例、专家数k成正比，当前阈值取3。
        tilingDataPtr_->useGatherCopy = ((expertEnd_ - expertStart_) * k_ > expertNum_ * NUM_THREE);
    } else {
        tilingDataPtr_->useGatherCopy = 0;
    }
}

void MoeInitRoutingV3TilingArch35::SetLoopParams4SrcToDstDropPad(int64_t perCoreRows, int64_t lastCoreRows)
{
    auto *tilingData = &tilingDataPtr_->srcToDstDropPadParamsOp;
    int64_t rowAlign = UB_BLOCK_SIZE / static_cast<int64_t>(sizeof(int32_t));
    int64_t maxPerLoopRows = (availUbSize_ - UB_BLOCK_SIZE) / static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO;
    maxPerLoopRows = maxPerLoopRows / rowAlign * rowAlign;
    maxPerLoopRows = std::max(maxPerLoopRows, rowAlign);

    tilingData->perCorePerLoopRows = std::min(perCoreRows, maxPerLoopRows);
    tilingData->perCoreLoops = Ops::Base::CeilDiv(perCoreRows, tilingData->perCorePerLoopRows);
    tilingData->perCoreLastLoopRows = perCoreRows - (tilingData->perCoreLoops - 1) * tilingData->perCorePerLoopRows;

    tilingData->lastCorePerLoopRows = std::min(lastCoreRows, maxPerLoopRows);
    tilingData->lastCoreLoops = Ops::Base::CeilDiv(lastCoreRows, tilingData->lastCorePerLoopRows);
    tilingData->lastCoreLastLoopRows = lastCoreRows - (tilingData->lastCoreLoops - 1) * tilingData->lastCorePerLoopRows;
    tilingData->perLoopCols = cols_;
    tilingData->lastLoopCols = cols_;
    tilingData->colLoops = 1;
}

void MoeInitRoutingV3TilingArch35::Tiling4SrcToDstDropPadCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4SrcToDstDropPadCompute()");

    auto *tilingData = &tilingDataPtr_->srcToDstDropPadParamsOp;

    int64_t perCoreRows = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreRows);
    tilingData->needCoreNum = needCoreNum;
    tilingData->perCoreRows = perCoreRows;

    int64_t lastCoreRows = totalLength_ - perCoreRows * (needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;

    SetLoopParams4SrcToDstDropPad(perCoreRows, lastCoreRows);

    OP_LOGD(context_,
            "DropPad SrcToDst Tiling: needCoreNum=%ld, perCoreRows=%ld, lastCoreRows=%ld, "
            "perLoopCols=%ld, colLoops=%ld, perCoreLoops=%ld",
            tilingData->needCoreNum, tilingData->perCoreRows, tilingData->lastCoreRows, tilingData->perLoopCols,
            tilingData->colLoops, tilingData->perCoreLoops);
}

bool MoeInitRoutingV3TilingArch35::UseCompactGatherOutDropPad(int64_t outputRows) const
{
    if (outputRows > totalLength_) {
        return false;
    }
    int64_t savedIndexScanBytes = (totalLength_ - outputRows) * static_cast<int64_t>(sizeof(int32_t));
    int64_t extraSourceRows = std::max(outputRows - n_, static_cast<int64_t>(0));
    int64_t rowCopyBytes = cols_ * inputXDtypeSize_ + (isInputScale_ == 1 ? static_cast<int64_t>(sizeof(float)) : 0);
    int64_t extraCompactCopyBytes = extraSourceRows * rowCopyBytes;
    const int64_t RANDOM_ACCESS_WEIGHT = 4LL;
    return savedIndexScanBytes > extraCompactCopyBytes * RANDOM_ACCESS_WEIGHT;
}

void MoeInitRoutingV3TilingArch35::SetGatherOutDropPadCoreSplitParams(int64_t &needCoreNum,
                                                                      int64_t &perCoreIndicesElements,
                                                                      int64_t &lastCoreIndicesElements)
{
    auto *tilingData = &tilingDataPtr_->gatherOutComputeParamsOp;
    int64_t outputRows = expertNum_ * expertCapacity_;
    // 与 kernel 分支保持一致：compact 路径扫描 outputRows，fallback 原路径扫描 n*k。
    bool useCompactGatherOutDropPad = UseCompactGatherOutDropPad(outputRows);
    tilingData->useCompactGatherOutDropPad = static_cast<int64_t>(useCompactGatherOutDropPad);
    int64_t splitElements = useCompactGatherOutDropPad ? outputRows : totalLength_;
    perCoreIndicesElements = Ops::Base::CeilDiv(splitElements, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        tilingData->perCorePerLoopIndicesElements = 0;
        tilingData->lastCorePerLoopIndicesElements = 0;
        return;
    }
    needCoreNum = Ops::Base::CeilDiv(splitElements, perCoreIndicesElements);
    lastCoreIndicesElements = splitElements - (needCoreNum - 1) * perCoreIndicesElements;
}

void MoeInitRoutingV3TilingArch35::SetGatherOutDropPadLoopParams(int64_t perCoreIndicesElements,
                                                                 int64_t lastCoreIndicesElements)
{
    auto *tilingData = &tilingDataPtr_->gatherOutComputeParamsOp;
    int64_t perLoopCols = cols_;
    // compact DropPad gather_out 按输出行遍历，只需要保留单行 X queue 和 source-row map queue。
    int64_t scaleCopyInQueueSize =
        isInputScale_ == 1 ? DOUBLE_BUFFER * AlignBytes(1, static_cast<int64_t>(sizeof(float))) : 0;
    int64_t int32ElementsPerBlock = UB_BLOCK_SIZE / static_cast<int64_t>(sizeof(int32_t));
    int64_t perLoopMaxIndicesElements = 0;

    while (perLoopMaxIndicesElements <= 0 && perLoopCols > 0) {
        int64_t rowBytes = AlignBytes(perLoopCols, inputXDtypeSize_);
        int64_t xCopyInQueueSize = DOUBLE_BUFFER * rowBytes;
        int64_t remainUbForIndices = availUbSize_ - xCopyInQueueSize - scaleCopyInQueueSize;
        int64_t maxAlignedBytes = remainUbForIndices / DOUBLE_BUFFER;
        perLoopMaxIndicesElements = maxAlignedBytes / static_cast<int64_t>(sizeof(int32_t));
        perLoopMaxIndicesElements = perLoopMaxIndicesElements / int32ElementsPerBlock * int32ElementsPerBlock;

        if (perLoopMaxIndicesElements > 0 || perLoopCols <= 1) {
            break;
        }
        perLoopCols = Ops::Base::CeilDiv(perLoopCols, NUM_TWO);
    }

    int64_t colsLoops = Ops::Base::CeilDiv(cols_, perLoopCols);
    int64_t lastLoopCols = cols_ - (colsLoops - 1) * perLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;

    tilingData->colsLoops = colsLoops;
    tilingData->perLoopCols = perLoopCols;
    tilingData->lastLoopCols = lastLoopCols;
    tilingData->perCoreIndicesLoops = perCoreIndicesLoops;
    tilingData->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    tilingData->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;
    tilingData->lastCoreIndicesLoops = lastCoreIndicesLoops;
    tilingData->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    tilingData->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
    tilingData->xCopyInQueueBufferNum = DOUBLE_BUFFER;
}

void MoeInitRoutingV3TilingArch35::Tiling4GatherOutDropPadCompute()
{
    OP_LOGD(context_, "Entered MoeInitRoutingV3TilingArch35::Tiling4GatherOutDropPadCompute()");

    int64_t needCoreNum = 0;
    int64_t perCoreIndicesElements = 0;
    int64_t lastCoreIndicesElements = 0;
    SetGatherOutDropPadCoreSplitParams(needCoreNum, perCoreIndicesElements, lastCoreIndicesElements);

    if (perCoreIndicesElements <= 0) {
        return;
    }

    SetGatherOutDropPadLoopParams(perCoreIndicesElements, lastCoreIndicesElements);

    auto *tilingData = &tilingDataPtr_->gatherOutComputeParamsOp;
    tilingData->needCoreNum = needCoreNum;
    tilingData->perCoreIndicesElements = perCoreIndicesElements;
    tilingData->lastCoreIndicesElements = lastCoreIndicesElements;

    OP_LOGD(context_,
            "DropPad GatherOut Tiling: needCoreNum=%ld, perCoreIndicesElements=%ld, "
            "lastCoreIndicesElements=%ld, colsLoops=%ld, perLoopCols=%ld, perCoreIndicesLoops=%ld, "
            "perCorePerLoopIndicesElements=%ld, useCompactGatherOutDropPad=%ld",
            needCoreNum, perCoreIndicesElements, lastCoreIndicesElements, tilingData->colsLoops,
            tilingData->perLoopCols, tilingData->perCoreIndicesLoops, tilingData->perCorePerLoopIndicesElements,
            tilingData->useCompactGatherOutDropPad);
}

REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV3, MoeInitRoutingV3TilingArch35,
                             1000); // If 950, use this tiling class.
} // namespace optiling
