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
 * \file stem_oam_prep_paged_kv_tiling_simd.cpp
 * \brief StemOamPrepPagedKv simd Tiling implementation
 */
#include "stem_oam_prep_paged_kv_tiling_simd.h"
#include <string>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
namespace optiling {
using namespace Ops::Base;

ge::graphStatus StemOamPrepPagedKvTilingSimd::CheckParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char *kvLayoutStr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvLayoutStr);
    std::string kvLayout(kvLayoutStr);
    int64_t kvLayoutVal = KV_LAYOUT_BBND;
    if (kvLayout == "BNBD") {
        kvLayoutVal = KV_LAYOUT_BNBD;
    } else if (kvLayout == "BBND") {
        kvLayoutVal = KV_LAYOUT_BBND;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "kvLayout", kvLayoutStr,
                                              "kvLayout must be BBND or BNBD.");
        return ge::GRAPH_FAILED;
    }

    if ((lambdaMag_ < 0) || (lambdaMag_ > 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "lambdaMag", std::to_string(lambdaMag_).c_str(),
                                              "lambdaMag must be greater than 0 ans less than or equal to 1.");
        return ge::GRAPH_FAILED;
    }

    auto kCacheDtype = context_->GetInputDesc(INPUT_KCACHE_INDEX)->GetDataType();
    auto vCacheDtype = context_->GetInputDesc(INPUT_VCACHE_INDEX)->GetDataType();
    auto kScaleCacheDtype = context_->GetInputDesc(INPUT_K_SCALE_CACHE_INDEX)->GetDataType();
    auto vScaleDtype = context_->GetInputDesc(INPUT_V_SCALE_INDEX)->GetDataType();
    if (kCacheDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "kCache",
                                              std::to_string(static_cast<int32_t>(kCacheDtype)).c_str(),
                                              "The dtype of kCache must be FLOAT8_E4M3FN.");
        return ge::GRAPH_FAILED;
    }
    if (vCacheDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "vCache",
                                              std::to_string(static_cast<int32_t>(vCacheDtype)).c_str(),
                                              "The dtype of vCache must be FLOAT8_E4M3FN.");
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "kScaleCache",
                                              std::to_string(static_cast<int32_t>(kScaleCacheDtype)).c_str(),
                                              "The dtype of kScaleCache must be FLOAT.");
        return ge::GRAPH_FAILED;
    }
    if (vScaleDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "vScale",
                                              std::to_string(static_cast<int32_t>(vScaleDtype)).c_str(),
                                              "The dtype of vScale must be FLOAT.");
        return ge::GRAPH_FAILED;
    }

    static const char *inputNames[] = {"kCache", "vCache", "kvIndices", "kvSeqLens", "kScaleCache", "vScale"};
    for (size_t i = 0; i < INPUT_COUNT; i++) {
        auto inputShape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        if (inputShape->GetShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), inputNames[i], "0",
                                                      "does not support empty tensor.");
            return ge::GRAPH_FAILED;
        }
    }

    auto kCacheShape = context_->GetInputShape(INPUT_KCACHE_INDEX)->GetShape();
    auto vCacheShape = context_->GetInputShape(INPUT_VCACHE_INDEX)->GetShape();
    auto kScaleCacheShape = context_->GetInputShape(INPUT_K_SCALE_CACHE_INDEX)->GetShape();
    auto vScaleShape = context_->GetInputShape(INPUT_V_SCALE_INDEX)->GetShape();
    auto kvIndicesShape = context_->GetInputShape(INPUT_KV_INDICES_INDEX)->GetShape();
    int64_t kvBlockSize = (kvLayoutVal == KV_LAYOUT_BBND) ? kCacheShape.GetDim(DIM_1) : kCacheShape.GetDim(DIM_2);
    if (kCacheShape.GetDimNum() != KV_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "kCache",
                                                 std::to_string(kCacheShape.GetDimNum()).c_str(),
                                                 "The shape dim of kCache must be 4.");
        return ge::GRAPH_FAILED;
    }
    if (vCacheShape.GetDimNum() != KV_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "vCache",
                                                 std::to_string(vCacheShape.GetDimNum()).c_str(),
                                                 "The shape dim of vCache must be 4.");
        return ge::GRAPH_FAILED;
    }
    if (kScaleCacheShape.GetDimNum() != KV_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "kScaleCache",
                                                 std::to_string(kScaleCacheShape.GetDimNum()).c_str(),
                                                 "The shape dim of kScaleCache must be 4.");
        return ge::GRAPH_FAILED;
    }
    if (vScaleShape.GetDimNum() != V_SCALE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "vScale",
                                                 std::to_string(vScaleShape.GetDimNum()).c_str(),
                                                 "The shape dim of vScale must be 1.");
        return ge::GRAPH_FAILED;
    }

    if (kvLayoutVal == KV_LAYOUT_BBND) {
        if (kScaleCacheShape.GetDim(DIM_1) != kvBlockSize) {
            std::string reason =
                "kvLayout=BBND requires kScaleCacheShape[1]=kvBlockSize(" + std::to_string(kvBlockSize) + ").";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kScaleCache",
                                                  std::to_string(kScaleCacheShape.GetDim(DIM_1)).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
        if (kScaleCacheShape.GetDim(DIM_2) != vScaleShape.GetDim(DIM_0)) {
            std::string reason =
                "kvLayout=BBND requires kScaleCacheShape[2]=H_kv(" + std::to_string(vScaleShape.GetDim(DIM_0)) + ").";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kScaleCache",
                                                  std::to_string(kScaleCacheShape.GetDim(DIM_2)).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (kScaleCacheShape.GetDim(DIM_2) != kvBlockSize) {
            std::string reason =
                "kvLayout=BNBD requires kScaleCacheShape[2]=kvBlockSize(" + std::to_string(kvBlockSize) + ").";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kScaleCache",
                                                  std::to_string(kScaleCacheShape.GetDim(DIM_2)).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
        if (kScaleCacheShape.GetDim(DIM_1) != vScaleShape.GetDim(DIM_0)) {
            std::string reason =
                "kvLayout=BNBD requires kScaleCacheShape[1]=H_kv(" + std::to_string(vScaleShape.GetDim(DIM_0)) + ").";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kScaleCache",
                                                  std::to_string(kScaleCacheShape.GetDim(DIM_1)).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (numKvHeads_ > HKVMAX) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "numKvHeads",
                                              std::to_string(numKvHeads_).c_str(), "requires numKvHeads less than 8");
        return ge::GRAPH_FAILED;
    }

    if (batchSize_ > BATCHMAX) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "batchSize", std::to_string(batchSize_).c_str(),
                                              "requires batchSize less than 16");
        return ge::GRAPH_FAILED;
    }

    if (kCacheDtype == ge::DT_FLOAT8_E4M3FN) {
        if ((kCacheShape.GetDim(DIM_0) != kScaleCacheShape.GetDim(DIM_0)) ||
            (kCacheShape.GetDim(DIM_1) != kScaleCacheShape.GetDim(DIM_1)) ||
            (kCacheShape.GetDim(DIM_2) != kScaleCacheShape.GetDim(DIM_2))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kCacheAndkScaleCache", "not same",
                                                  "In the DT_FLOAT8_E4M3FN scenario, the first three dimensions of "
                                                  "kCache and kScaleCacheOptional must be the same.");
            return ge::GRAPH_FAILED;
        }
    }

    if ((kCacheShape.GetDim(DIM_0) != vCacheShape.GetDim(DIM_0)) ||
        (kCacheShape.GetDim(DIM_1) != vCacheShape.GetDim(DIM_1)) ||
        (kCacheShape.GetDim(DIM_2) != vCacheShape.GetDim(DIM_2)) ||
        (kCacheShape.GetDim(DIM_3) != vCacheShape.GetDim(DIM_3))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kCacheAndvCacheShape", "not same",
                                              "The dimensions of kCache and vCache must be the same.");
        return ge::GRAPH_FAILED;
    }

    if (batchSize_ != kvIndicesShape.GetDim(DIM_0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kvIndicesAndvkvSeqLens", "dim0 not same",
                                              "kvIndicesShape[0] and kvSeqLensShape[0] are not equal.");
        return ge::GRAPH_FAILED;
    }

    if (kvBlockSize_ != KVBLOCKSIZEONE && kvBlockSize_ != KVBLOCKSIZETWO) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kvBlockSize_", "64 or 128",
                                              "kvBlockSize must be 64 or 128.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool StemOamPrepPagedKvTilingSimd::IsCapable() { return true; }

ge::graphStatus StemOamPrepPagedKvTilingSimd::ContinuousStridesCompute(gert::Shape &shape, gert::Stride &stride,
                                                                       size_t idx)
{
    auto xStorageShape = context_->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    if (shape.GetDimNum() == 0) {
        shape = xStorageShape->GetStorageShape();
    }
    stride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t xStride = 1;
    for (int32_t j = maxDim; j >= 0; --j) {
        stride.SetStride(j, xStride);
        xStride *= shape.GetDim(j);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemOamPrepPagedKvTilingSimd::GetTensorInfo(gert::Shape &shape, gert::Stride &inputStride, size_t idx)
{
    if (context_->InputIsView(idx)) {
        auto *strides = context_->GetInputStride(idx);
        if (strides == nullptr || strides->GetDimNum() == 0) {
            ContinuousStridesCompute(shape, inputStride, idx);
        } else {
            inputStride = *strides;
            if (shape.GetDimNum() == 0) {
                auto xStorageShape = context_->GetInputShape(idx);
                if (xStorageShape != nullptr) {
                    shape = xStorageShape->GetOriginShape();
                }
            }
        }
    } else {
        ContinuousStridesCompute(shape, inputStride, idx);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemOamPrepPagedKvTilingSimd::GetPlatformInfo()
{
    OP_LOGD("StemOamPrepPagedKv", "StemOamPrepPagedKvTilingSimd GetPlatformInfo");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("StemOamPrepPagedKv", "fail to get platform info"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNumAiv <= 0),
                OP_LOGE("StemOamPrepPagedKv", "StemOamPrepPagedKvSimdTiling fail to get coreNumAiv."),
                return ge::GRAPH_FAILED);
    totalCoreNum_ = coreNumAiv;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemOamPrepPagedKvTilingSimd::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("StemOamPrepPagedKv", "context is null"), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    tilingData_ = context_->GetTilingData<StemOamPrepPagedKvTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);

    const float *lambdaMagAttr = attrs->GetAttrPointer<float>(ATTR_LAMBDA_MAG_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, lambdaMagAttr);
    const int64_t *stemBlocksAttr = attrs->GetAttrPointer<int64_t>(ATTR_STEM_BLOCK_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stemBlocksAttr);
    const int64_t *stemStrideAttr = attrs->GetAttrPointer<int64_t>(ATTR_STEM_STRIDE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stemStrideAttr);

    lambdaMag_ = *lambdaMagAttr;
    const char *kvLayoutStr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvLayoutStr);
    std::string kvLayout(kvLayoutStr);
    if (kvLayout == "BNBD") {
        kvLayout_ = KV_LAYOUT_BNBD;
    } else if (kvLayout == "BBND") {
        kvLayout_ = KV_LAYOUT_BBND;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "kvLayout", kvLayoutStr,
                                              "kvLayout must be BBND or BNBD.");
        return ge::GRAPH_FAILED;
    }
    stemBlocks_ = *stemBlocksAttr;
    stemStride_ = *stemStrideAttr;

    if (stemBlocks_ % STEM_BLOCK_SIZE_ALIGN != 0 || stemBlocks_ > STEM_BLOCK_SIZE_MAX || stemBlocks_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "stemBlockSize",
                                              std::to_string(stemBlocks_).c_str(),
                                              "stemBlockSize must be multiple of 32, >0 and <=256.");
        return ge::GRAPH_FAILED;
    }
    if (stemStride_ % STEM_STRIDE_ALIGN != 0 || stemStride_ > STEM_STRIDE_MAX || stemStride_ > stemBlocks_ ||
        stemBlocks_ % stemStride_ != 0) {
        std::string reason = "stemStride must be multiple of 16, <=64, <=stemBlockSize(" + std::to_string(stemBlocks_) +
                             "), and stemBlockSize must be a multiple of stemStride.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "stemStride",
                                              std::to_string(stemStride_).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != KV_LAYOUT_BBND && kvLayout_ != KV_LAYOUT_BNBD) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "kvLayout", std::to_string(kvLayout_).c_str(),
                                              "kvLayout must be BBND or BNBD.");
        return ge::GRAPH_FAILED;
    }

    rVal_ = stemBlocks_ / stemStride_;

    const gert::StorageShape *kvSeqLensShape = context_->GetInputShape(INPUT_KV_SEQ_LENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSeqLensShape);
    batchSize_ = kvSeqLensShape->GetShape().GetDim(DIM_0);

    auto kvSeqLensTensor = context_->GetInputTensor(INPUT_KV_SEQ_LENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvSeqLensTensor);
    int32_t maxKvLen = 0;
    for (int32_t i = 0; i < batchSize_; i++) {
        int32_t kvSeqLen = kvSeqLensTensor->GetData<int32_t>()[i];
        if (kvSeqLen > maxKvLen) {
            maxKvLen = kvSeqLen;
        }
    }

    int64_t kvLenAlign = Ops::Base::CeilAlign(static_cast<int64_t>(maxKvLen), stemBlocks_);
    maxKb_ = kvLenAlign / stemBlocks_;

    GetTensorInfo(kcacheShape_, kCacheStride_, INPUT_KCACHE_INDEX);
    GetTensorInfo(vcacheShape_, vCacheStride_, INPUT_VCACHE_INDEX);
    GetTensorInfo(kScaleCacheShape_, kScaleCacheStride_, INPUT_K_SCALE_CACHE_INDEX);

    if (kvLayout_ == KV_LAYOUT_BBND) {
        numKvHeads_ = kcacheShape_.GetDim(DIM_2);
        kvBlockSize_ = kcacheShape_.GetDim(DIM_1);
    } else {
        numKvHeads_ = kcacheShape_.GetDim(DIM_1);
        kvBlockSize_ = kcacheShape_.GetDim(DIM_2);
    }

    OP_CHECK_IF(CheckParams() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckParams failed."),
                return ge::GRAPH_FAILED);
    dimQk_ = kcacheShape_.GetDim(DIM_3);
    const gert::StorageShape *kvIndicesShape = context_->GetInputShape(INPUT_KV_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvIndicesShape);
    maxKvBlocks_ = kvIndicesShape->GetShape().GetDim(DIM_1);
    const gert::StorageShape *kFlatOutShape = context_->GetOutputShape(OUTPUT_KFLAT_INDEX);
    kflatDim_ = kFlatOutShape->GetShape().GetDim(DIM_3);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemOamPrepPagedKvTilingSimd::DoOpTiling()
{
    if (batchSize_ == 0 || numKvHeads_ == 0) {
        OP_LOGE(context_->GetNodeName(), "batchSize or numKvHeads is 0!");
        return ge::GRAPH_FAILED;
    }

    AscendC::GetMeanMaxMinTmpSize(maxKb_ * rVal_, sizeof(float), sizeof(float), false, meanMaxSize_, meanMinSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemOamPrepPagedKvTilingSimd::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t StemOamPrepPagedKvTilingSimd::GetTilingKey() const { return GET_TPL_TILING_KEY(true); }

ge::graphStatus StemOamPrepPagedKvTilingSimd::GetWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t vFlatSize =
        static_cast<uint64_t>(batchSize_) * static_cast<uint64_t>(numKvHeads_) * (maxKb_ * rVal_) * sizeof(float);
    workspaceSize_ = vFlatSize;

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_ + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StemOamPrepPagedKvTilingSimd::PostTiling()
{
    uint32_t usedCoreNum = static_cast<uint32_t>(totalCoreNum_);

    for (int32_t i = 0; i < STRIDE_DIM_NUM; i++) {
        tilingData_->kCacheStride[i] = kCacheStride_[i];
        tilingData_->vCacheStride[i] = vCacheStride_[i];
        tilingData_->kScaleCacheStride[i] = kScaleCacheStride_[i];
    }

    tilingData_->batchSize = batchSize_;
    tilingData_->numKvHeads = numKvHeads_;
    tilingData_->kflatDim = kflatDim_;
    tilingData_->maxKb = maxKb_;
    tilingData_->kvBlockSize = kvBlockSize_;
    tilingData_->dimQk = dimQk_;
    tilingData_->dimV = dimQk_;
    tilingData_->maxKvBlocks = maxKvBlocks_;
    tilingData_->stemBlockSize = stemBlocks_;
    tilingData_->stemStride = stemStride_;
    tilingData_->lambdaMag = lambdaMag_;
    tilingData_->kvLayout = kvLayout_;
    tilingData_->rVal = rVal_;
    tilingData_->meanSize = static_cast<int64_t>(meanMinSize_);

    context_->SetBlockDim(usedCoreNum);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void StemOamPrepPagedKvTilingSimd::DumpTilingInfo()
{
    std::ostringstream info;
    info << "kvLayout: " << kvLayout_ << std::endl;
    info << "kvBlockSize: " << kvBlockSize_ << std::endl;
    info << "numKvHeads: " << numKvHeads_ << std::endl;
    info << "maxKvBlocks: " << maxKvBlocks_ << std::endl;
    info << "dimQk: " << dimQk_ << std::endl;
    info << "dimV: " << dimQk_ << std::endl;
    info << "maxKb: " << maxKb_ << std::endl;
    info << "kflatDim: " << kflatDim_ << std::endl;
    info << "batchSize: " << batchSize_ << std::endl;
    info << "stemBlockSize: " << stemBlocks_ << std::endl;
    info << "stemStride: " << stemStride_ << std::endl;
    info << "lambdaMag: " << lambdaMag_ << std::endl;
    info << "rVal: " << rVal_ << std::endl;
    info << "kCacheStride: " << kCacheStride_[0] << " " << kCacheStride_[1] << " " << kCacheStride_[2] << " "
         << kCacheStride_[3] << std::endl;
    info << "vCacheStride: " << vCacheStride_[0] << " " << vCacheStride_[1] << " " << vCacheStride_[2] << " "
         << vCacheStride_[3] << std::endl;
    info << "kScaleCacheStride: " << kScaleCacheStride_[0] << " " << kScaleCacheStride_[1] << " "
         << kScaleCacheStride_[2] << " " << kScaleCacheStride_[3] << std::endl;

    OP_LOGI("StemOamPrepPagedKv", "Tiling info is: %s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(StemOamPrepPagedKv, StemOamPrepPagedKvTilingSimd, 0);

} // namespace optiling
