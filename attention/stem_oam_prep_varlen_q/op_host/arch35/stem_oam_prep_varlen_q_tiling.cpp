/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stem_oam_prep_varlen_q_tiling.h"
#include "../../op_kernel/arch35/stem_oam_prep_varlen_q_tiling_data.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include <algorithm>

namespace optiling {
namespace stem_oam_prep_varlen_q {

// ===========================================================================
// Constants
// ===========================================================================
static constexpr int64_t DIM_QK_HOST = 128;
static constexpr uint32_t MAX_BATCH_SIZE = 1024;

static constexpr uint32_t INPUT_Q = 0;
static constexpr uint32_t INPUT_QSEQLENS = 1;
static constexpr uint32_t INPUT_CUSEQLENS = 2;
static constexpr uint32_t INPUT_QSCALE = 3;

static constexpr uint32_t ATTR_STEM_BLOCK_SIZE = 0;
static constexpr uint32_t ATTR_STEM_STRIDE = 1;

static constexpr uint32_t Q_RANK = 3;
static constexpr uint32_t QSEQLENS_RANK = 1;
static constexpr uint32_t CUSEQLENS_RANK = 1;
static constexpr uint32_t QSCALE_RANK = 2;
static constexpr uint32_t QFLAT_RANK = 4;

static constexpr int64_t CURRENTLY_SUPPORTED_STEM_BLOCK_SIZE = 128;
static constexpr int64_t CURRENTLY_SUPPORTED_STEM_STRIDE = 16;
static constexpr uint32_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024;
static constexpr uint32_t SIZEOF_FP8 = 1;
static constexpr uint32_t SIZEOF_FLOAT32 = 4;
static constexpr uint32_t SIZEOF_BFLOAT16 = 2;
static constexpr uint64_t DEFAULT_TILING_KEY = 0;
static constexpr uint32_t MAX_UB_FACTOR = 2;
static constexpr uint32_t SCALE_PAD_SIZE = 8;

// ===========================================================================
// GetShapeAttrsInfo: Validate attrs + input shapes + dtypes + consistency + output
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    auto ret = ValidateAttrs();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    ret = ValidateInputShapes();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    ret = ValidateDtypes();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    ret = ValidateConsistency();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    ret = CalcExpectedMaxQb();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    return ValidateOutputShape();
}

// ===========================================================================
// GetPlatformInfo: Read platform + compile info
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const StemPrepQCompileInfo *>(context_->GetCompileInfo());
    if (compileInfo == nullptr || compileInfo->coreNum == 0) {
        OP_LOGE(opName_, "Invalid compile info or coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    coreNum_ = static_cast<uint32_t>(compileInfo->coreNum);

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(opName_, "platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);

    OP_LOGI(context_, "GetPlatformInfo: coreNum=%u ubSize=%lu", coreNum_, ubSize_);
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// IsCapable: Always return true
// ===========================================================================
bool StemOamPrepVarlenQTiling::IsCapable() { return true; }

// ===========================================================================
// DoOpTiling: Core tiling calculation (pure compute, no validation)
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::DoOpTiling()
{
    tilingData_ = context_->GetTilingData<StemPrepQTilingData>();
    if (tilingData_ == nullptr) {
        OP_LOGE(opName_, "tiling data is null");
        return ge::GRAPH_FAILED;
    }

    if (batch_ == 0 || numQHeads_ == 0) {
        tilingData_->usedCoreNum = 1;
        tilingData_->blocksPerCoreBase = 0;
        tilingData_->blocksRemainder = 0;
        blockDim_ = 1;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t B = static_cast<uint32_t>(stemBlockSize_);
    uint32_t S = static_cast<uint32_t>(stemStride_);

    tilingData_->stemBlockSize = B;
    tilingData_->stemStride = S;
    tilingData_->rVal = B / S;
    tilingData_->kflatDim = S * DIM_QK_HOST;
    tilingData_->dimQk = DIM_QK_HOST;
    tilingData_->batchSize = static_cast<uint32_t>(batch_);
    tilingData_->numQHeads = static_cast<uint32_t>(numQHeads_);
    tilingData_->totalTokens = totalTokens_;
    tilingData_->maxQb = maxQb_;

    CalcCoreDistribution();
    return CalcUBFactor();
}

// ===========================================================================
// DoLibApiTiling: Not used for this kernel
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

// ===========================================================================
// GetTilingKey: Single template, always return 0
// ===========================================================================
uint64_t StemOamPrepVarlenQTiling::GetTilingKey() const { return DEFAULT_TILING_KEY; }

// ===========================================================================
// GetWorkspaceSize: Query workspace from platform
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(opName_, "platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        OP_LOGE(opName_, "workspaceSizes info is null");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = workspaceSize;
    OP_LOGI(context_, "Workspace size: %zu bytes", workspaceSize);
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// PostTiling: Set tiling key and block dim
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::PostTiling()
{
    context_->SetTilingKey(DEFAULT_TILING_KEY);
    context_->SetBlockDim(blockDim_);
    context_->SetScheduleMode(1); // Batch mode

    OP_LOGD(context_, "========== StemPrepQTilingData ==========");
    OP_LOGD(context_, "batchSize: %u", tilingData_->batchSize);
    OP_LOGD(context_, "numQHeads: %u", tilingData_->numQHeads);
    OP_LOGD(context_, "dimQk: %u", tilingData_->dimQk);
    OP_LOGD(context_, "stemBlockSize: %u", tilingData_->stemBlockSize);
    OP_LOGD(context_, "stemStride: %u", tilingData_->stemStride);
    OP_LOGD(context_, "rVal: %u", tilingData_->rVal);
    OP_LOGD(context_, "kflatDim: %u", tilingData_->kflatDim);
    OP_LOGD(context_, "maxQb: %u", tilingData_->maxQb);
    OP_LOGD(context_, "totalTokens: %u", tilingData_->totalTokens);
    OP_LOGD(context_, "usedCoreNum: %u", tilingData_->usedCoreNum);
    OP_LOGD(context_, "blocksPerCoreBase: %u", tilingData_->blocksPerCoreBase);
    OP_LOGD(context_, "blocksRemainder: %u", tilingData_->blocksRemainder);
    OP_LOGD(context_, "ubFactor: %u", tilingData_->ubFactor);

    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: ValidateAttrs
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::ValidateAttrs()
{
    auto attrs = context_->GetAttrs();
    const int64_t *stemBlockSizePtr = attrs->GetInt(ATTR_STEM_BLOCK_SIZE);
    const int64_t *stemStridePtr = attrs->GetInt(ATTR_STEM_STRIDE);
    if (stemBlockSizePtr == nullptr || stemStridePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "stemBlockSize/stemStride");
        return ge::GRAPH_FAILED;
    }
    stemBlockSize_ = *stemBlockSizePtr;
    stemStride_ = *stemStridePtr;

    if (stemBlockSize_ != CURRENTLY_SUPPORTED_STEM_BLOCK_SIZE) {
        std::string reason = "currently only supports " + std::to_string(CURRENTLY_SUPPORTED_STEM_BLOCK_SIZE);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "stemBlockSize", std::to_string(stemBlockSize_).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (stemStride_ != CURRENTLY_SUPPORTED_STEM_STRIDE) {
        std::string reason = "currently only supports " + std::to_string(CURRENTLY_SUPPORTED_STEM_STRIDE);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "stemStride", std::to_string(stemStride_).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: ValidateInputShapes
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::ValidateInputShapes()
{
    auto qShape = context_->GetInputShape(INPUT_Q);
    auto qSeqLensShape = context_->GetInputShape(INPUT_QSEQLENS);
    auto cuSeqLensQShape = context_->GetInputShape(INPUT_CUSEQLENS);
    if (qShape == nullptr || qSeqLensShape == nullptr || cuSeqLensQShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "q/qSeqLens/cuSeqLensQ");
        return ge::GRAPH_FAILED;
    }

    if (qShape->GetStorageShape().GetDimNum() != Q_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "q", std::to_string(qShape->GetStorageShape().GetDimNum()).c_str(),
                                     std::to_string(Q_RANK).c_str());
        return ge::GRAPH_FAILED;
    }
    if (qSeqLensShape->GetStorageShape().GetDimNum() != QSEQLENS_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "qSeqLens",
                                     std::to_string(qSeqLensShape->GetStorageShape().GetDimNum()).c_str(),
                                     std::to_string(QSEQLENS_RANK).c_str());
        return ge::GRAPH_FAILED;
    }
    if (cuSeqLensQShape->GetStorageShape().GetDimNum() != CUSEQLENS_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "cuSeqLensQ",
                                     std::to_string(cuSeqLensQShape->GetStorageShape().GetDimNum()).c_str(),
                                     std::to_string(CUSEQLENS_RANK).c_str());
        return ge::GRAPH_FAILED;
    }

    batch_ = qSeqLensShape->GetStorageShape().GetDim(0);
    int64_t cuSeqLensSize = cuSeqLensQShape->GetStorageShape().GetDim(0);
    if (cuSeqLensSize != batch_ + 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cuSeqLensQ", std::to_string(cuSeqLensSize).c_str(),
                                              ("must equal batch+1=" + std::to_string(batch_ + 1)).c_str());
        return ge::GRAPH_FAILED;
    }

    numQHeads_ = qShape->GetStorageShape().GetDim(1);
    int64_t D = qShape->GetStorageShape().GetDim(2);
    if (D != DIM_QK_HOST) {
        OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(opName_, INPUT_Q, std::to_string(D).c_str(),
                                             std::to_string(DIM_QK_HOST).c_str());
        return ge::GRAPH_FAILED;
    }
    if (batch_ < 0 || batch_ > MAX_BATCH_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "batch", std::to_string(batch_).c_str(),
                                              ("must be in [0, " + std::to_string(MAX_BATCH_SIZE) + "]").c_str());
        return ge::GRAPH_FAILED;
    }
    if (numQHeads_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "numQHeads", std::to_string(numQHeads_).c_str(),
                                              "numQHeads must be >= 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: ValidateDtypes
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::ValidateDtypes()
{
    auto qDesc = context_->GetInputDesc(INPUT_Q);
    auto qSeqLensDesc = context_->GetInputDesc(INPUT_QSEQLENS);
    auto cuSeqLensQDesc = context_->GetInputDesc(INPUT_CUSEQLENS);
    if (qDesc == nullptr || qSeqLensDesc == nullptr || cuSeqLensQDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName_, "q/qSeqLens/cuSeqLensQ");
        return ge::GRAPH_FAILED;
    }

    qDtype_ = qDesc->GetDataType();
    if (qDtype_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(opName_, "q", std::to_string(static_cast<int>(qDtype_)).c_str(),
                                         "DT_FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (qSeqLensDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(
            opName_, "qSeqLens", std::to_string(static_cast<int>(qSeqLensDesc->GetDataType())).c_str(), "DT_INT64");
        return ge::GRAPH_FAILED;
    }
    if (cuSeqLensQDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(
            opName_, "cuSeqLensQ", std::to_string(static_cast<int>(cuSeqLensQDesc->GetDataType())).c_str(), "DT_INT64");
        return ge::GRAPH_FAILED;
    }

    // Validate optional qScale
    auto qScaleDesc = context_->GetOptionalInputDesc(INPUT_QSCALE);
    if (qDtype_ == ge::DT_FLOAT8_E4M3FN) {
        if (qScaleDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName_, "qScale");
            return ge::GRAPH_FAILED;
        }
        if (qScaleDesc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(
                opName_, "qScale", std::to_string(static_cast<int>(qScaleDesc->GetDataType())).c_str(), "DT_FLOAT");
            return ge::GRAPH_FAILED;
        }
        auto qScaleShape = context_->GetInputShape(INPUT_QSCALE);
        if (qScaleShape == nullptr || qScaleShape->GetStorageShape().GetDimNum() != QSCALE_RANK) {
            OP_LOGE_WITH_INVALID_INPUT(opName_, "qScale");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (qScaleDesc != nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName_, "qScale");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: ValidateConsistency
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::ValidateConsistency()
{
    if (batch_ == 0 || numQHeads_ == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (qDtype_ == ge::DT_FLOAT8_E4M3FN) {
        auto qShape = context_->GetInputShape(INPUT_Q);
        auto qScaleShape = context_->GetInputShape(INPUT_QSCALE);
        if (qShape == nullptr || qScaleShape == nullptr) {
            return ge::GRAPH_FAILED;
        }

        int64_t totalTokens = qShape->GetStorageShape().GetDim(0);
        int64_t scaleTokens = qScaleShape->GetStorageShape().GetDim(0);
        int64_t scaleHeads = qScaleShape->GetStorageShape().GetDim(1);
        if (scaleTokens != totalTokens || scaleHeads != numQHeads_) {
            std::string expected = "[" + std::to_string(totalTokens) + ", " + std::to_string(numQHeads_) + "]";
            std::string actual = "[" + std::to_string(scaleTokens) + ", " + std::to_string(scaleHeads) + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "qScale", actual.c_str(),
                                                  ("must match q shape in first two dimensions: " + expected).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: ValidateCuSeqLens
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::ValidateCuSeqLens(const int64_t *cuSeqLensData, const int64_t *qSeqLensData,
                                                            uint32_t batchU32)
{
    if (cuSeqLensData[0] != 0) {
        OP_LOGE(opName_, "cuSeqLensQ[0] must be 0");
        return ge::GRAPH_FAILED;
    }
    int64_t sumQSeqLens = 0;
    for (uint32_t b = 0; b < batchU32; b++) {
        if (cuSeqLensData[b + 1] - cuSeqLensData[b] != qSeqLensData[b]) {
            std::string expectedQLens = std::to_string(qSeqLensData[b]);
            std::string actualDiff = std::to_string(cuSeqLensData[b + 1] - cuSeqLensData[b]);
            std::string reason = "prefix sum mismatch at batch " + std::to_string(b) + ": expected qSeqLens[" +
                                 std::to_string(b) + "]=" + expectedQLens;
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cuSeqLensQ", actualDiff.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        sumQSeqLens += qSeqLensData[b];
    }
    if (cuSeqLensData[batchU32] != sumQSeqLens) {
        std::string expectedSum = std::to_string(sumQSeqLens);
        std::string actualFinal = std::to_string(cuSeqLensData[batchU32]);
        std::string reason = "final element must be equal to sum of qSeqLens=" + expectedSum;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cuSeqLensQ", actualFinal.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: ValidateOutputShape
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::ValidateOutputShape()
{
    auto qFlatShape = context_->GetOutputShape(0);
    auto qFlatDesc = context_->GetOutputDesc(0);
    if (qFlatShape == nullptr || qFlatDesc == nullptr) {
        OP_LOGE(opName_, "qFlat output is null");
        return ge::GRAPH_FAILED;
    }

    auto &shape = qFlatShape->GetStorageShape();
    if (shape.GetDimNum() != QFLAT_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "qFlat", std::to_string(shape.GetDimNum()).c_str(),
                                                 "must be 4D");
        return ge::GRAPH_FAILED;
    }

    uint32_t S = static_cast<uint32_t>(stemStride_);
    uint32_t expectedKflat = S * DIM_QK_HOST;
    int64_t actualBatch = shape.GetDim(0);
    int64_t actualHeads = shape.GetDim(1);
    int64_t actualMaxQb = shape.GetDim(2);
    int64_t actualKflat = shape.GetDim(3);

    if (actualBatch != static_cast<int64_t>(batch_)) {
        std::string reason = "must equal batchSize=" + std::to_string(batch_);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "qFlat dim 0", std::to_string(actualBatch).c_str(),
                                                 reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (actualHeads != static_cast<int64_t>(numQHeads_)) {
        std::string reason = "must equal numQHeads=" + std::to_string(numQHeads_);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "qFlat dim 1", std::to_string(actualHeads).c_str(),
                                                 reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (actualMaxQb < static_cast<int64_t>(maxQb_)) {
        std::string reason = "must be >= maxQb=" + std::to_string(maxQb_);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "qFlat dim 2", std::to_string(actualMaxQb).c_str(),
                                                 reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (actualKflat < static_cast<int64_t>(expectedKflat)) {
        std::string reason = "must be >= kflatDim=" + std::to_string(expectedKflat);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "qFlat dim 3", std::to_string(actualKflat).c_str(),
                                                 reason.c_str());
        return ge::GRAPH_FAILED;
    }

    // Check output dtype
    ge::DataType expected = (qDtype_ == ge::DT_FLOAT16) ? ge::DT_FLOAT16 : ge::DT_BF16;
    if (qFlatDesc->GetDataType() != expected) {
        const char *inputStr = (qDtype_ == ge::DT_FLOAT8_E4M3FN) ? "FP8" : (qDtype_ == ge::DT_BF16 ? "BF16" : "FP16");
        const char *outputStr = (expected == ge::DT_BF16) ? "BF16" : "FP16";
        std::string reason = std::string(inputStr) + " input requires " + std::string(outputStr) + " output";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "qFlat", outputStr, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: CalcExpectedMaxQb
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::CalcExpectedMaxQb()
{
    auto qShape = context_->GetInputShape(INPUT_Q);
    totalTokens_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
    uint32_t batchU32 = static_cast<uint32_t>(batch_);

    auto qSeqLensTensor = context_->GetInputTensor(INPUT_QSEQLENS);
    const int64_t *qSeqLensData = qSeqLensTensor->GetData<int64_t>();
    if (qSeqLensData == nullptr) {
        OP_LOGE(opName_, "qSeqLens data is null");
        return ge::GRAPH_FAILED;
    }
    auto cuSeqLensTensor = context_->GetInputTensor(INPUT_CUSEQLENS);
    const int64_t *cuSeqLensData = cuSeqLensTensor->GetData<int64_t>();
    if (cuSeqLensData == nullptr) {
        OP_LOGE(opName_, "cuSeqLensQ data is null");
        return ge::GRAPH_FAILED;
    }

    auto ret = ValidateCuSeqLens(cuSeqLensData, qSeqLensData, batchU32);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    uint32_t B = static_cast<uint32_t>(stemBlockSize_);
    maxQb_ = 0;
    for (uint32_t b = 0; b < batchU32; b++) {
        uint32_t qLenB = static_cast<uint32_t>(qSeqLensData[b]);
        uint32_t numQbB = ((qLenB + B - 1) / B);
        maxQb_ = std::max(maxQb_, numQbB);
    }

    totalBlocks_ = static_cast<uint32_t>(numQHeads_) * maxQb_ * batchU32;
    OP_LOGI(context_, "CalcExpectedMaxQb: totalTokens=%u maxQb=%u totalBlocks=%u", totalTokens_, maxQb_, totalBlocks_);
    return ge::GRAPH_SUCCESS;
}

// ===========================================================================
// Private: CalcCoreDistribution
// ===========================================================================
void StemOamPrepVarlenQTiling::CalcCoreDistribution()
{
    tilingData_->usedCoreNum = std::min(totalBlocks_, coreNum_);
    tilingData_->blocksPerCoreBase = tilingData_->usedCoreNum == 0 ? 0 : totalBlocks_ / tilingData_->usedCoreNum;
    tilingData_->blocksRemainder = tilingData_->usedCoreNum == 0 ? 0 : totalBlocks_ % tilingData_->usedCoreNum;
    blockDim_ = tilingData_->usedCoreNum;

    OP_LOGI(context_, "CalcCoreDistribution: totalBlocks=%u, usedCoreNum=%u, blocksPerCoreBase=%u, blocksRemainder=%u",
            totalBlocks_, tilingData_->usedCoreNum, tilingData_->blocksPerCoreBase, tilingData_->blocksRemainder);
}

// ===========================================================================
// Private: CalcUBFactor
// ===========================================================================
ge::graphStatus StemOamPrepVarlenQTiling::CalcUBFactor()
{
    if (ubSize_ <= SYSTEM_RESERVED_UB_SIZE) {
        OP_LOGE(opName_, "ubSize %lu too small, must be > %u", ubSize_, SYSTEM_RESERVED_UB_SIZE);
        return ge::GRAPH_FAILED;
    }

    uint32_t B = static_cast<uint32_t>(stemBlockSize_);
    uint32_t S = static_cast<uint32_t>(stemStride_);
    uint32_t perBlockUB = B * DIM_QK_HOST * SIZEOF_FLOAT32 + B * DIM_QK_HOST * SIZEOF_FP8 +
                          B * SCALE_PAD_SIZE * SIZEOF_FLOAT32 + S * DIM_QK_HOST * SIZEOF_BFLOAT16;
    uint32_t cuSeqLensBuf = static_cast<uint32_t>((batch_ + 1) * sizeof(int64_t));
    uint32_t availableUB = static_cast<uint32_t>(ubSize_ - SYSTEM_RESERVED_UB_SIZE - cuSeqLensBuf);
    uint32_t ubFactor = availableUB / perBlockUB;

    if (ubFactor < 1) {
        OP_LOGE(opName_, "UB insufficient: ubFactor=%u < 1 (available=%u, perBlock=%u)", ubFactor, availableUB,
                perBlockUB);
        return ge::GRAPH_FAILED;
    }

    tilingData_->ubFactor = std::min(ubFactor, MAX_UB_FACTOR);
    OP_LOGI(context_, "CalcUBFactor: ubSize=%lu available=%u perBlock=%u ubFactor=%u", ubSize_, availableUB, perBlockUB,
            tilingData_->ubFactor);
    return ge::GRAPH_SUCCESS;
}

} // namespace stem_oam_prep_varlen_q

// ===========================================================================
// Entry points
// ===========================================================================
ge::graphStatus Tiling4StemOamPrepVarlenQ(gert::TilingContext *context)
{
    stem_oam_prep_varlen_q::StemOamPrepVarlenQTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepare4StemOamPrepVarlenQ(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<StemPrepQCompileInfo>();
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StemOamPrepVarlenQ)
    .Tiling(Tiling4StemOamPrepVarlenQ)
    .TilingInputsDataDependency({1, 2})
    .TilingParse<StemPrepQCompileInfo>(TilingPrepare4StemOamPrepVarlenQ);

} // namespace optiling
