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
 * \file causal_conv1d_tiling.cpp
 * \brief Host-side tiling — shape parsing, UB budget, block decomposition.
 */

#include <algorithm>
#include <limits>
#include <cstring>

#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "../op_kernel/arch35/causal_conv1d_tiling_data.h"
#include "../op_kernel/arch35/causal_conv1d_tiling_key.h"

namespace optiling {
namespace CausalConv1d {

using namespace Ops::Transformer::OpTiling;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t CONV_STATES_INDEX = 2;
constexpr uint32_t BIAS_INDEX = 3;
constexpr uint32_t QUERY_START_LOC_INDEX = 4;
constexpr uint32_t CACHE_INDICES_INDEX = 5;
constexpr uint32_t INITIAL_STATE_MODE_INDEX = 6;
constexpr uint32_t NUM_ACCEPTED_TOKENS_INDEX = 7;

constexpr int64_t ASCENDC_RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr int64_t DIM_ALIGN_BYTES = 32;

static int64_t FloorPow2(int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    uint64_t result = 1;
    const uint64_t un = static_cast<uint64_t>(n);
    while (result <= un && result <= (UINT64_MAX >> 1)) {
        result <<= 1;
    }
    return static_cast<int64_t>(result >> 1);
}

static int64_t FloorPow2Multiple(int64_t n, int64_t base)
{
    if (n < base || base <= 0) {
        return base > 0 ? base : 0;
    }
    return FloorPow2(n / base) * base;
}

struct CausalConv1dCompileInfo {};

struct CausalConv1dAttrInfo {
    int64_t activationMode = 0;
    int64_t nullBlockId = 0;
};

struct CausalConv1dInputInfo {
    // shapes
    int64_t dim = 0;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t cuSeqlen = 0;
    int64_t kernelWidth = 0;
    int64_t stateLen = 0;
    int64_t numCacheLines = 0;

    // dtype
    int64_t elemBytes = 0;

    // tiling keys
    uint32_t inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN;
    uint32_t widthKey = CAUSAL_CONV1D_TPL_WIDTH_2;
    uint32_t hasBiasKey = CAUSAL_CONV1D_TPL_HAS_BIAS_FALSE;
    uint32_t activationKey = CAUSAL_CONV1D_TPL_ACTIVATION_NONE;

    // runtime flags (not in tiling key)
    bool hasBias = false;
    bool hasCacheIndices = false;
    bool hasInitialStateMode = false;
    bool hasNumAcceptedTokens = false;
};

// ============================================================
// CausalConv1dTiling
// ============================================================

class CausalConv1dTiling {
public:
    explicit CausalConv1dTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    gert::TilingContext *context_;
    const int32_t *queryStartLocData_ = nullptr;
    CausalConv1dInputInfo inputInfo_;
    uint64_t ubSize_ = 0;
    uint32_t coreNum_ = 0;
    uint64_t cacheLineSize_ = 0;
    int64_t ubLimitedBaseDim_ = 0;
    int64_t baseDim_ = 0;
    int64_t coBatch_ = 1;
    int64_t tokensPerBlock_ = 0;
    int64_t tokenBlockCnt_ = 0;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus SetWorkspaceSize(size_t workspaceSize);
    ge::graphStatus SetupFnInitStateWorkspace();
    ge::graphStatus GetAttrsInfo(CausalConv1dAttrInfo &attrInfo);
    ge::graphStatus ParseCausalConv1dInputInfo(const CausalConv1dAttrInfo &attrInfo);

    ge::graphStatus SetTilingData(const CausalConv1dAttrInfo &attrInfo);
    bool ChooseBaseDim(int64_t cuSeqlen, int64_t dim);
    static constexpr bool IsFnInputMode(uint32_t inputModeKey);
    static constexpr bool IsVarlenInputModeKey(uint32_t key)
    {
        return key == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN || key == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN;
    }
    void ResolveTilingKeys(int64_t activationMode);

    ge::graphStatus ValidateDimSize(int64_t dim, ge::DataType xDtype);
    ge::graphStatus ParseXShape(bool isFnMode);
    ge::graphStatus ParseWeightShape();
    ge::graphStatus ParseConvStatesShape();
    ge::graphStatus ParseQueryStartLoc(bool isFnMode);
    ge::graphStatus ParseCacheIndices(int64_t nullBlockId);
    ge::graphStatus ParseInitialStateMode(bool isFnMode);
    ge::graphStatus ParseNumAcceptedTokens(bool isFnMode);
    ge::graphStatus ParseBias();
    bool InferIsFnMode();

    int64_t ComputeUbLimitedBaseDim();
};

// ============================================================
// Key normalization
// ============================================================

constexpr bool CausalConv1dTiling::IsFnInputMode(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH);
}

void CausalConv1dTiling::ResolveTilingKeys(int64_t activationMode)
{
    inputInfo_.widthKey = static_cast<uint32_t>(inputInfo_.kernelWidth);
    inputInfo_.hasBiasKey = inputInfo_.hasBias ? CAUSAL_CONV1D_TPL_HAS_BIAS_TRUE : CAUSAL_CONV1D_TPL_HAS_BIAS_FALSE;
    inputInfo_.activationKey = static_cast<uint32_t>(activationMode);
}

// ============================================================
// Platform info
// ============================================================

ge::graphStatus CausalConv1dTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum_ is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_, "ubSize_ is 0"), return ge::GRAPH_FAILED);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(cacheLineSize_ == 0, OP_LOGE(context_, "cacheLineSize_ is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::SetWorkspaceSize(size_t workspaceSize)
{
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::SetupFnInitStateWorkspace()
{
    if (!IsFnInputMode(inputInfo_.inputModeKey) || !inputInfo_.hasInitialStateMode) {
        return SetWorkspaceSize(0);
    }
    int64_t snapshotBytes = 0;
    bool overflow = __builtin_mul_overflow(inputInfo_.numCacheLines, inputInfo_.stateLen, &snapshotBytes) ||
                    __builtin_mul_overflow(snapshotBytes, inputInfo_.dim, &snapshotBytes) ||
                    __builtin_mul_overflow(snapshotBytes, inputInfo_.elemBytes, &snapshotBytes);
    OP_CHECK_IF(overflow || snapshotBytes < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "workspace_size", "snapshot size",
                                                      "overflow detected"),
                return ge::GRAPH_FAILED);
    const size_t totalSize = static_cast<size_t>(ASCENDC_RESERVED_WORKSPACE_SIZE) + static_cast<size_t>(snapshotBytes);
    OP_CHECK_IF(SetWorkspaceSize(totalSize) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetScheduleMode(1) error"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Attrs
// ============================================================

static int64_t ParseActivationModeStr(const char *str)
{
    if (str == nullptr || std::strcmp(str, "") == 0 || std::strcmp(str, "silu") == 0) {
        return 1;
    }
    if (std::strcmp(str, "none") == 0) {
        return 0;
    }
    return -1;
}

ge::graphStatus CausalConv1dTiling::GetAttrsInfo(CausalConv1dAttrInfo &attrInfo)
{
    constexpr int32_t kActivationModeAttrIndex = 0;
    constexpr int32_t kNullBlockIdAttrIndex = 1;

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const char *activationModeStr = attrs->GetAttrPointer<char>(kActivationModeAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activationModeStr);
    attrInfo.activationMode = ParseActivationModeStr(activationModeStr);
    OP_CHECK_IF(attrInfo.activationMode < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "activation_mode", activationModeStr,
                                                      "only supports \"silu\" or \"none\""),
                return ge::GRAPH_FAILED);

    const int64_t *nullBlockIdPtr = attrs->GetAttrPointer<int64_t>(kNullBlockIdAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nullBlockIdPtr);
    attrInfo.nullBlockId = *nullBlockIdPtr;

    OP_LOGI(context_, "AttrsInfo: activationMode[%s->%ld], nullBlockId[%ld].", activationModeStr,
            attrInfo.activationMode, attrInfo.nullBlockId);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Validation
// ============================================================

ge::graphStatus CausalConv1dTiling::ValidateDimSize(int64_t dim, ge::DataType xDtype)
{
    const int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(xDtype));
    OP_CHECK_IF((dim * dtypeSize) % DIM_ALIGN_BYTES != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dim", std::to_string(dim).c_str(),
                                                      "must be 32-byte aligned"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Shape & dtype info
// ============================================================

ge::graphStatus CausalConv1dTiling::ParseXShape(bool isFnMode)
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    if (xShape.GetDimNum() == 2) {
        if (!isFnMode) {
            inputInfo_.inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH;
            inputInfo_.batch = xShape.GetDim(0);
            inputInfo_.dim = xShape.GetDim(1);
            inputInfo_.seqLen = 1;
            inputInfo_.cuSeqlen = inputInfo_.batch;
            OP_CHECK_IF(
                inputInfo_.batch <= 0 || inputInfo_.dim <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x",
                    ("batch=" + std::to_string(inputInfo_.batch) + ", dim=" + std::to_string(inputInfo_.dim)).c_str(),
                    "for 2D decode mode, shape must be (cu_seqlen, dim)"),
                return ge::GRAPH_FAILED);
        } else {
            inputInfo_.inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN;
            inputInfo_.cuSeqlen = xShape.GetDim(0);
            inputInfo_.dim = xShape.GetDim(1);
            inputInfo_.seqLen = 0;
            OP_CHECK_IF(inputInfo_.dim <= 0 || inputInfo_.cuSeqlen <= 0,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                                              ("dim=" + std::to_string(inputInfo_.dim) +
                                                               ", cuSeqlen=" + std::to_string(inputInfo_.cuSeqlen))
                                                                  .c_str(),
                                                              "for 2D varlen mode, shape must be (cu_seqlen, dim)"),
                        return ge::GRAPH_FAILED);
        }
    } else if (xShape.GetDimNum() == 3) {
        inputInfo_.inputModeKey =
            isFnMode ? CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH : CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH;
        inputInfo_.batch = xShape.GetDim(0);
        inputInfo_.seqLen = xShape.GetDim(1);
        inputInfo_.dim = xShape.GetDim(2);
        OP_CHECK_IF(inputInfo_.batch <= 0 || inputInfo_.dim <= 0 || inputInfo_.seqLen <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x",
                        ("batch=" + std::to_string(inputInfo_.batch) + ", dim=" + std::to_string(inputInfo_.dim) +
                         ", seqLen=" + std::to_string(inputInfo_.seqLen))
                            .c_str(),
                        "for 3D batch mode, shape must be (batch, seqlen, dim)"),
                    return ge::GRAPH_FAILED);
        int64_t cuSeqlen = 0;
        OP_CHECK_IF(__builtin_mul_overflow(inputInfo_.batch, inputInfo_.seqLen, &cuSeqlen),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "x",
                        ("batch=" + std::to_string(inputInfo_.batch) + " * seqLen=" + std::to_string(inputInfo_.seqLen))
                            .c_str(),
                        "batch * seqLen overflow"),
                    return ge::GRAPH_FAILED);
        inputInfo_.cuSeqlen = cuSeqlen;
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xShape.GetDimNum()).c_str(),
                                     "2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(), "FLOAT16 or BF16"),
        return ge::GRAPH_FAILED);
    inputInfo_.elemBytes = static_cast<int64_t>(ge::GetSizeByDataType(xDtype));
    OP_CHECK_IF(ValidateDimSize(inputInfo_.dim, xDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "dim", std::to_string(inputInfo_.dim).c_str(), "must be 32-byte aligned"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputInfo_.cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cu_seqlen",
                                              std::to_string(inputInfo_.cuSeqlen).c_str(), "exceeds int32_t max"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseWeightShape()
{
    auto wShapePtr = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(
        wShape.GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "weight", std::to_string(wShape.GetDimNum()).c_str(),
                                     "2 dimensions (kernelWidth, dim)"),
        return ge::GRAPH_FAILED);
    inputInfo_.kernelWidth = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    OP_CHECK_IF(wDim != inputInfo_.dim,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "weight", std::to_string(wDim).c_str(),
                                              ("dim=" + std::to_string(inputInfo_.dim)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.kernelWidth < 2 || inputInfo_.kernelWidth > 4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "kernel_width",
                                                      std::to_string(inputInfo_.kernelWidth).c_str(),
                                                      "only supports [2, 3, 4]"),
                return ge::GRAPH_FAILED);

    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(
        wDesc->GetDataType() != xDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "weight and x",
            (Ops::Base::ToString(wDesc->GetDataType()) + " and " + Ops::Base::ToString(xDesc->GetDataType())).c_str(),
            "weight dtype must equal x dtype"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseConvStatesShape()
{
    auto convStatesShapePtr = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShapePtr);
    auto convStatesShape = EnsureNotScalar(convStatesShapePtr->GetStorageShape());
    OP_CHECK_IF(convStatesShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "conv_states",
                                             std::to_string(convStatesShape.GetDimNum()).c_str(),
                                             "3D (num_cache_lines, state_len, dim)"),
                return ge::GRAPH_FAILED);
    inputInfo_.numCacheLines = convStatesShape.GetDim(0);
    inputInfo_.stateLen = convStatesShape.GetDim(1);
    const int64_t convStatesDim = convStatesShape.GetDim(2);
    OP_CHECK_IF(inputInfo_.numCacheLines <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "conv_states",
                                              std::to_string(inputInfo_.numCacheLines).c_str(), "num_cache_lines > 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        convStatesDim != inputInfo_.dim,
        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "conv_states", std::to_string(convStatesDim).c_str(),
                                      ("dim=" + std::to_string(inputInfo_.dim)).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.stateLen < (inputInfo_.kernelWidth - 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE(
                    context_->GetNodeName(), "conv_states", std::to_string(inputInfo_.stateLen).c_str(),
                    ("state_len >= kernelWidth-1=" + std::to_string(inputInfo_.kernelWidth - 1)).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.numCacheLines < inputInfo_.batch,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "conv_states",
                                              std::to_string(inputInfo_.numCacheLines).c_str(),
                                              ("num_cache_lines >= batch=" + std::to_string(inputInfo_.batch)).c_str()),
                return ge::GRAPH_FAILED);

    auto convStatesDesc = context_->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(convStatesDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "conv_states and x",
                                                       (Ops::Base::ToString(convStatesDesc->GetDataType()) + " and " +
                                                        Ops::Base::ToString(xDesc->GetDataType()))
                                                           .c_str(),
                                                       "conv_states dtype must equal x dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseQueryStartLoc(bool isFnMode)
{
    // Step 1: determine qsl existence and size from shape
    auto qslShapePtr = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    const gert::CompileTimeTensorDesc *qslDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    bool qslExist = false;
    int64_t qslSize = 0;
    if (qslShapePtr != nullptr) {
        const auto qslStorageShape = qslShapePtr->GetStorageShape();
        const int64_t qslDimNum = qslStorageShape.GetDimNum();
        qslExist = !(qslDimNum == 0 || (qslDimNum == 1 && qslStorageShape.GetDim(0) <= 0));
        if (qslExist) {
            auto qslShape = EnsureNotScalar(qslStorageShape);
            OP_CHECK_IF(qslShape.GetDimNum() != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "query_start_loc",
                                                     std::to_string(qslShape.GetDimNum()).c_str(), "1D"),
                        return ge::GRAPH_FAILED);
            qslSize = qslShape.GetDim(0);
            OP_CHECK_IF(qslSize < 1,
                        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "query_start_loc",
                                                      std::to_string(qslSize).c_str(), "size >= 1"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_NULL_WITH_CONTEXT(context_, qslDesc);
            OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "query_start_loc",
                                                  Ops::Base::ToString(qslDesc->GetDataType()).c_str(), "int32"),
                        return ge::GRAPH_FAILED);
        }
    }

    if (qslExist) {
        if (!isFnMode && !IsVarlenInputModeKey(inputInfo_.inputModeKey) && (qslSize - 1) != inputInfo_.batch) {
            inputInfo_.inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN;
            inputInfo_.seqLen = 0;
        }
        if (IsVarlenInputModeKey(inputInfo_.inputModeKey)) {
            inputInfo_.batch = qslSize - 1;
        } else {
            OP_CHECK_IF(qslSize != inputInfo_.batch + 1,
                        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "query_start_loc",
                                                      std::to_string(qslSize).c_str(),
                                                      ("batch+1=" + std::to_string(inputInfo_.batch + 1)).c_str()),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(IsVarlenInputModeKey(inputInfo_.inputModeKey),
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "query_start_loc", "not provided",
                                                  "required in varlen mode"),
                    return ge::GRAPH_FAILED);
        qslSize = inputInfo_.batch + 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseCacheIndices(int64_t nullBlockId)
{
    auto cIdxShapePtr = context_->GetOptionalInputShape(CACHE_INDICES_INDEX);
    if (cIdxShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto cIdxStorageShape = cIdxShapePtr->GetStorageShape();
    const int64_t cIdxDimNum = cIdxStorageShape.GetDimNum();
    inputInfo_.hasCacheIndices = (cIdxDimNum > 0) && (cIdxDimNum != 1 || cIdxStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasCacheIndices) {
        return ge::GRAPH_SUCCESS;
    }
    auto cIdxShape = EnsureNotScalar(cIdxStorageShape);
    OP_CHECK_IF(cIdxShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "cache_indices",
                                             std::to_string(cIdxShape.GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cIdxShape.GetDim(0) != inputInfo_.batch,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "cache_indices",
                                              std::to_string(cIdxShape.GetDim(0)).c_str(),
                                              ("batch=" + std::to_string(inputInfo_.batch)).c_str()),
                return ge::GRAPH_FAILED);

    auto cIdxDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cIdxDesc);
    OP_CHECK_IF(cIdxDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "cache_indices",
                                          Ops::Base::ToString(cIdxDesc->GetDataType()).c_str(), "int32"),
                return ge::GRAPH_FAILED);

    const gert::Tensor *cIdxTensor = context_->GetOptionalInputTensor(CACHE_INDICES_INDEX);
    if (cIdxTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseInitialStateMode(bool isFnMode)
{
    auto initStShapePtr = context_->GetOptionalInputShape(INITIAL_STATE_MODE_INDEX);
    if (initStShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto initStStorageShape = initStShapePtr->GetStorageShape();
    const int64_t initStDimNum = initStStorageShape.GetDimNum();
    inputInfo_.hasInitialStateMode = (initStDimNum > 0) && (initStDimNum != 1 || initStStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasInitialStateMode) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!isFnMode,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "initial_state_mode", "provided",
                                                      "only supported in prefill mode"),
                return ge::GRAPH_FAILED);
    auto initStShape = EnsureNotScalar(initStStorageShape);
    OP_CHECK_IF(initStShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "initial_state_mode",
                                             std::to_string(initStShape.GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(initStShape.GetDim(0) != inputInfo_.batch,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "initial_state_mode",
                                              std::to_string(initStShape.GetDim(0)).c_str(),
                                              ("batch=" + std::to_string(inputInfo_.batch)).c_str()),
                return ge::GRAPH_FAILED);

    auto initStDesc = context_->GetOptionalInputDesc(INITIAL_STATE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, initStDesc);
    OP_CHECK_IF(initStDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "initial_state_mode",
                                          Ops::Base::ToString(initStDesc->GetDataType()).c_str(), "int32"),
                return ge::GRAPH_FAILED);

    const gert::Tensor *initStTensor = context_->GetOptionalInputTensor(INITIAL_STATE_MODE_INDEX);
    if (initStTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseNumAcceptedTokens(bool isFnMode)
{
    auto numAccShapePtr = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX);
    if (numAccShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto numAccStorageShape = numAccShapePtr->GetStorageShape();
    const int64_t numAccDimNum = numAccStorageShape.GetDimNum();
    inputInfo_.hasNumAcceptedTokens = (numAccDimNum > 0) && (numAccDimNum != 1 || numAccStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasNumAcceptedTokens) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(isFnMode,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_accepted_tokens", "provided",
                                                      "only supported in decode/update mode"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.kernelWidth != 4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "kernel_width",
                                                      std::to_string(inputInfo_.kernelWidth).c_str(),
                                                      "num_accepted_tokens only supports kernelWidth=4"),
                return ge::GRAPH_FAILED);
    auto numAccShape = EnsureNotScalar(numAccStorageShape);
    OP_CHECK_IF(numAccShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "num_accepted_tokens",
                                             std::to_string(numAccShape.GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numAccShape.GetDim(0) != inputInfo_.batch,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "num_accepted_tokens",
                                              std::to_string(numAccShape.GetDim(0)).c_str(),
                                              ("batch=" + std::to_string(inputInfo_.batch)).c_str()),
                return ge::GRAPH_FAILED);

    if (!IsVarlenInputModeKey(inputInfo_.inputModeKey)) {
        const int64_t reqStateLen = (inputInfo_.kernelWidth - 1) + (inputInfo_.seqLen - 1);
        OP_CHECK_IF(
            inputInfo_.stateLen < reqStateLen,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "state_len", std::to_string(inputInfo_.stateLen).c_str(),
                ("spec decode requires stateLen >= (kernelWidth-1) + (seqlen-1) = " + std::to_string(reqStateLen))
                    .c_str()),
            return ge::GRAPH_FAILED);
    }

    auto numAccDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numAccDesc);
    OP_CHECK_IF(numAccDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "num_accepted_tokens",
                                          Ops::Base::ToString(numAccDesc->GetDataType()).c_str(), "int32"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseBias()
{
    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto biasStorageShape = biasShapePtr->GetStorageShape();
    const int64_t biasDimNum = biasStorageShape.GetDimNum();
    inputInfo_.hasBias = (biasDimNum > 0) && (biasDimNum != 1 || biasStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasShape = EnsureNotScalar(biasStorageShape);
    OP_CHECK_IF(biasShape.GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "bias",
                                             std::to_string(biasShape.GetDimNum()).c_str(), "1D (dim,)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        biasShape.GetDim(0) != inputInfo_.dim,
        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "bias", std::to_string(biasShape.GetDim(0)).c_str(),
                                      ("dim=" + std::to_string(inputInfo_.dim)).c_str()),
        return ge::GRAPH_FAILED);

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(biasDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "bias and x",
                    (Ops::Base::ToString(biasDesc->GetDataType()) + " and " + Ops::Base::ToString(xDesc->GetDataType()))
                        .c_str(),
                    "bias dtype must equal x dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Shape & dtype info (orchestrator)
// ============================================================

ge::graphStatus CausalConv1dTiling::ParseCausalConv1dInputInfo(const CausalConv1dAttrInfo &attrInfo)
{
    const bool isFnMode = InferIsFnMode();

    // Parse order and data flow (all functions write to inputInfo_ directly):
    //
    //  ┌─ ParseXShape ──► dim, cuSeqlen, seqLen, batch, inputModeKey
    //  ├─ ParseQueryStartLoc ──► may refine batch, seqLen, inputModeKey from x shape
    //  │
    //  ├─ ParseWeightShape ──► kernelWidth (needs dim)
    //  ├─ ParseConvStatesShape ──► stateLen, numCacheLines (needs dim, kernelWidth, final batch)
    //  └─ ParseCacheIndices ──► hasCacheIndices (needs batch, numCacheLines)
    //
    //  ┌─ ParseInitialStateMode ──► hasInitialStateMode (needs batch)
    //  ├─ ParseNumAcceptedTokens ──► hasNumAcceptedTokens (needs batch, kernelWidth, stateLen, seqLen,
    //  inputModeKey) └─ ParseBias ───────────────► hasBias (needs dim)

    // Phase 1: x shape + batch refinement
    OP_CHECK_IF(ParseXShape(isFnMode) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseXShape error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseQueryStartLoc(isFnMode) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseQueryStartLoc error"),
                return ge::GRAPH_FAILED);

    // Phase 2: weight → conv_states → cache_indices (state management chain)
    OP_CHECK_IF(ParseWeightShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseWeightShape error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseConvStatesShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseConvStatesShape error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseCacheIndices(attrInfo.nullBlockId) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseCacheIndices error"), return ge::GRAPH_FAILED);

    // Phase 3: remaining optional inputs
    OP_CHECK_IF(ParseInitialStateMode(isFnMode) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseInitialStateMode error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseNumAcceptedTokens(isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseNumAcceptedTokens error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseBias() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ParseBias error"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Tiling planning
// ============================================================

int64_t CausalConv1dTiling::ComputeUbLimitedBaseDim()
{
    constexpr int64_t kOutBufCnt = 2;

    const int64_t elemBytes = inputInfo_.elemBytes;
    const int64_t alignElems = static_cast<int64_t>(cacheLineSize_) / elemBytes;

    const int64_t ringBufCnt = inputInfo_.kernelWidth + 1;
    const int64_t calcBufCnt = inputInfo_.kernelWidth + (inputInfo_.hasBias ? 1 : 0);
    const int64_t bytesPerElem =
        (ringBufCnt * elemBytes) + (kOutBufCnt * elemBytes) + (calcBufCnt * static_cast<int64_t>(sizeof(float)));
    const int64_t rawBaseDim = static_cast<int64_t>(ubSize_) / bytesPerElem;
    if (rawBaseDim <= 0) {
        return 0;
    }
    return FloorPow2Multiple(rawBaseDim, alignElems);
}

bool CausalConv1dTiling::ChooseBaseDim(int64_t cuSeqlen, int64_t dim)
{
    if (dim > ubLimitedBaseDim_) {
        const int64_t alignElems = static_cast<int64_t>(cacheLineSize_) / inputInfo_.elemBytes;
        int64_t bestBaseDim = alignElems;
        int64_t maxActualCores = 0;

        constexpr double kCoreUsageThresh = 0.8;
        for (int64_t chunks = ubLimitedBaseDim_ / alignElems; chunks >= 1; chunks >>= 1) {
            int64_t candidate = chunks * alignElems;
            int64_t baseDimCnt = Ops::Base::CeilDiv(dim, candidate);
            int64_t group = static_cast<int64_t>(coreNum_) / baseDimCnt;
            if (group <= 0) {
                continue;
            }
            int64_t tokenBlocks = Ops::Base::CeilDiv(cuSeqlen, Ops::Base::CeilDiv(cuSeqlen, group));
            int64_t actualCores = tokenBlocks * baseDimCnt;
            if (actualCores >= static_cast<int64_t>(static_cast<double>(coreNum_) * kCoreUsageThresh)) {
                maxActualCores = actualCores;
                bestBaseDim = candidate;
                break;
            }
            if (actualCores > maxActualCores) {
                maxActualCores = actualCores;
                bestBaseDim = candidate;
            }
        }
        baseDim_ = bestBaseDim;
        coBatch_ = 1;
    } else {
        baseDim_ = dim;
        // Compute coBatch for the dim-limited case
        int64_t maxCoBatch = ubLimitedBaseDim_ / dim;
        if (inputInfo_.hasCacheIndices || inputInfo_.hasInitialStateMode || inputInfo_.hasNumAcceptedTokens ||
            IsVarlenInputModeKey(inputInfo_.inputModeKey)) {
            // CoBatch only supports simple fixed-length batch mode
            coBatch_ = 1;
        } else {
            coBatch_ = maxCoBatch;
            while (coBatch_ > 1 && (inputInfo_.batch % coBatch_ != 0)) {
                coBatch_--;
            }
        }
    }

    int64_t baseDimCnt = Ops::Base::CeilDiv(dim, baseDim_);
    int64_t group = static_cast<int64_t>(coreNum_) / baseDimCnt;
    if (group <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "dim", std::to_string(inputInfo_.dim).c_str(),
            ("too large for UB-limited baseDim: requires " + std::to_string(baseDimCnt) + " cores, only " +
             std::to_string(coreNum_) + " available")
                .c_str());
        return false;
    }
    tokensPerBlock_ = Ops::Base::CeilDiv(cuSeqlen, group);
    if (coBatch_ > 1) {
        int64_t alignedTokens = Ops::Base::CeilAlign(tokensPerBlock_, coBatch_);
        int64_t origBlockCnt = Ops::Base::CeilDiv(cuSeqlen, tokensPerBlock_);
        int64_t alignedBlockCnt = Ops::Base::CeilDiv(cuSeqlen, alignedTokens);
        constexpr double kMinCoreRatio = 0.7;
        if (static_cast<double>(alignedBlockCnt) >= static_cast<double>(origBlockCnt) * kMinCoreRatio) {
            tokensPerBlock_ = alignedTokens;
        } else {
            coBatch_ = 1;
        }
    }
    tokenBlockCnt_ = Ops::Base::CeilDiv(cuSeqlen, tokensPerBlock_);
    return true;
}

} // namespace CausalConv1d

using namespace CausalConv1d;

// ============================================================
// Inference: determine prefill vs decode from tensor shapes
// ============================================================

bool CausalConv1dTiling::InferIsFnMode()
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    if (xShapePtr == nullptr) {
        OP_LOGE(context_, "InferIsFnMode: x shape is null");
        return true; // default to prefill
    }
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
    int64_t xDims = xShape.GetDimNum();

    // num_accepted_tokens is only meaningful in decode mode — strong signal
    auto numAccShapePtr = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX);
    if (numAccShapePtr != nullptr) {
        auto numAccShape = EnsureNotScalar(numAccShapePtr->GetStorageShape());
        if (numAccShape.GetDimNum() > 0 && numAccShape.GetDim(0) > 0) {
            return false; // update
        }
    }

    if (xDims == 3) {
        // 3D: [B, L, D].  L > 1 => prefill; L == 1 => update
        int64_t seqLen = xShape.GetDim(1);
        return seqLen > 1;
    }

    // 2D: [N, D]
    auto qslShapePtr = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    if (qslShapePtr == nullptr) {
        // No query_start_loc => update batch mode
        return false;
    }
    auto qslShape = EnsureNotScalar(qslShapePtr->GetStorageShape());
    if (qslShape.GetDimNum() == 0 || qslShape.GetDim(0) == 0) {
        return false;
    }

    return true; // 2D + query_start_loc => FN_VARLEN (Update uses 3D with seqLen=1)
}

// ============================================================
// Tiling data serialization
// ============================================================

ge::graphStatus CausalConv1dTiling::SetTilingData(const CausalConv1dAttrInfo &attrInfo)
{
    CausalConv1dTilingData *tiling = context_->GetTilingData<CausalConv1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);

    tiling->activationMode = attrInfo.activationMode;
    tiling->nullBlockId = attrInfo.nullBlockId;
    tiling->dim = inputInfo_.dim;
    tiling->cuSeqlen = inputInfo_.cuSeqlen;
    tiling->seqLen = inputInfo_.seqLen;
    tiling->batch = inputInfo_.batch;
    tiling->kernelWidth = inputInfo_.kernelWidth;
    tiling->stateLen = inputInfo_.stateLen;
    tiling->numCacheLines = inputInfo_.numCacheLines;
    tiling->hasCacheIndices = inputInfo_.hasCacheIndices;
    tiling->hasInitialStateMode = inputInfo_.hasInitialStateMode;
    tiling->hasNumAcceptedTokens = inputInfo_.hasNumAcceptedTokens;
    OP_CHECK_IF(baseDim_ > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) || baseDim_ < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "baseDim",
                                                      std::to_string(baseDim_).c_str(), "exceeds int32_t range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(coBatch_ > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) || coBatch_ < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "coBatch",
                                                      std::to_string(coBatch_).c_str(), "exceeds int32_t range"),
                return ge::GRAPH_FAILED);
    tiling->baseDim = static_cast<int32_t>(baseDim_);
    tiling->coBatch = static_cast<int32_t>(coBatch_);
    tiling->hasInitStateWorkspace = inputInfo_.hasInitialStateMode;
    tiling->tokensPerBlock = tokensPerBlock_;
    tiling->tokenBlockCnt = tokenBlockCnt_;

    OP_LOGI(context_,
            "TilingData: dim[%ld], cuSeqlen[%ld], seqLen[%ld], batch[%ld], kernelWidth[%ld], "
            "stateLen[%ld], numCacheLines[%ld], activationMode[%ld], nullBlockId[%ld], "
            "baseDim[%d], coBatch[%d], tokensPerBlock[%ld], tokenBlockCnt[%ld], "
            "hasNumAcceptedTokens[%d], hasCacheIndices[%d], hasInitialStateMode[%d], "
            "hasInitStateWorkspace[%d], hasBias[%d].",
            tiling->dim, tiling->cuSeqlen, tiling->seqLen, tiling->batch, tiling->kernelWidth, tiling->stateLen,
            tiling->numCacheLines, tiling->activationMode, tiling->nullBlockId, tiling->baseDim, tiling->coBatch,
            tiling->tokensPerBlock, tiling->tokenBlockCnt, static_cast<int32_t>(tiling->hasNumAcceptedTokens),
            static_cast<int32_t>(tiling->hasCacheIndices), static_cast<int32_t>(tiling->hasInitialStateMode),
            static_cast<int32_t>(tiling->hasInitStateWorkspace), static_cast<int32_t>(inputInfo_.hasBias));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::DoTiling()
{
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);

    CausalConv1dAttrInfo attrInfo;
    OP_CHECK_IF(GetAttrsInfo(attrInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetAttrsInfo error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseCausalConv1dInputInfo(attrInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseCausalConv1dInputInfo error"), return ge::GRAPH_FAILED);

    ubLimitedBaseDim_ = ComputeUbLimitedBaseDim();
    OP_CHECK_IF(
        ubLimitedBaseDim_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ub_size", std::to_string(ubSize_).c_str(),
                                              "UB too small to allocate convolution buffers"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!ChooseBaseDim(inputInfo_.cuSeqlen, inputInfo_.dim), OP_LOGE(context_, "ChooseBaseDim failed"),
                return ge::GRAPH_FAILED);

    ResolveTilingKeys(attrInfo.activationMode);
    OP_CHECK_IF(SetupFnInitStateWorkspace() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetupWorkspace error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetTilingData(attrInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetTilingData error"),
                return ge::GRAPH_FAILED);

    const int64_t blockDimVal = tokenBlockCnt_ * Ops::Base::CeilDiv(inputInfo_.dim, baseDim_);
    OP_CHECK_IF(blockDimVal > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) || blockDimVal < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "blockDim",
                                                      std::to_string(blockDimVal).c_str(), "exceeds uint32_t range"),
                return ge::GRAPH_FAILED);
    uint32_t blockDim = static_cast<uint32_t>(blockDimVal);
    context_->SetBlockDim(blockDim);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputInfo_.inputModeKey, inputInfo_.widthKey, inputInfo_.hasBiasKey,
                                                  inputInfo_.activationKey);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext *context)
{
    CausalConv1dTiling tctx(context);
    return tctx.DoTiling();
}

static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext *context)
{
    OP_LOGD(context, "Enter TilingParseForCausalConv1d.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1d)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);

} // namespace optiling
