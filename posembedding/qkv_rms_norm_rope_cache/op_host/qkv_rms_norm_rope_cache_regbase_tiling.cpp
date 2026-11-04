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
 * \file qkv_rms_norm_rope_cache_regbase_tiling.cpp
 * \brief QkvRmsNormRopeCache arch35(Ascend950 / DAV_3510)tiling
 *
 *   - 参数校验:shape/dtype/attr 与本支持的 cache 对齐规则
 *   - 切核:按 (B*S) token 均分,每核连续一段 token(不区分 Q/K/V 核组)
 *   - UB 反推:按 regbase 内核的 buffer 布局重新计算 ubFactor
 *   - tiling key:10000
 */

#include "qkv_rms_norm_rope_cache_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckQkvValid()
{
    auto qkvShapeTuple = GetShapeTupleOfTH(context_, QKV_INDEX); // 返回[B*Sqkv, N*D]
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_BS>(qkvShapeTuple) == batchSize_ * seqLength_);
    isValid = isValid && (std::get<SHAPE_IDX_ND>(qkvShapeTuple) == numHead_ * qkvDim_);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of qkv is incorrect."), return ge::GRAPH_FAILED);
    auto qkvDesc = context_->GetInputDesc(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvDesc);
    qkvDtype_ = qkvDesc->GetDataType();
    OP_CHECK_IF(qkvDtype_ != ge::DataType::DT_FLOAT16 && qkvDtype_ != ge::DataType::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "qkv only support float16 and bfloat16 now"), return ge::GRAPH_FAILED);
    qkvDtypeSize_ = FLOAT16_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckGammaValid(int64_t gammaIdx)
{
    const gert::StorageShape* gammaShapePtr = context_->GetInputShape(gammaIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    bool isValid = true;
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDimNum() == 1);
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDim(0) == qkvDim_);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of gamma is invalid."), return ge::GRAPH_FAILED);

    auto gammaDesc = context_->GetInputDesc(gammaIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    auto gammaDtype = gammaDesc->GetDataType();
    OP_CHECK_IF(gammaDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "gamma dtype must be the same with qkv."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckCosSinValid()
{
    auto cosShapeTuple = GetShapeTupleOfTH(context_, COS_INDEX);
    auto sinShapeTuple = GetShapeTupleOfTH(context_, SIN_INDEX);
    bool isValid = true;
    isValid = isValid && ((std::get<SHAPE_IDX_BS>(cosShapeTuple) == batchSize_ * seqLength_) ||
                          (std::get<SHAPE_IDX_BS>(cosShapeTuple) == batchSize_ * NUM_ONE));
    isValid = isValid && (std::get<SHAPE_IDX_ND>(cosShapeTuple) == NUM_ONE * ropeRange_);
    isValid = isValid && ((std::get<SHAPE_IDX_BS>(sinShapeTuple) == batchSize_ * seqLength_) ||
                          (std::get<SHAPE_IDX_BS>(sinShapeTuple) == batchSize_ * NUM_ONE));
    isValid = isValid && (std::get<SHAPE_IDX_ND>(sinShapeTuple) == NUM_ONE * ropeRange_);
    auto cosSeq = std::get<SHAPE_IDX_BS>(cosShapeTuple);
    auto sinSeq = std::get<SHAPE_IDX_BS>(sinShapeTuple);
    isValid = isValid && (cosSeq == sinSeq);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of cos or sin is invalid."),
                return ge::GRAPH_FAILED);
    rope_seq_ = std::get<SHAPE_IDX_BS>(cosShapeTuple) / batchSize_;

    auto cosDesc = context_->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    auto cosDtype = cosDesc->GetDataType();
    OP_CHECK_IF(cosDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "cos dtype must be the same with qkv."),
                return ge::GRAPH_FAILED);

    auto sinDesc = context_->GetInputDesc(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    auto sinDtype = sinDesc->GetDataType();
    OP_CHECK_IF(sinDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "sin dtype must be the same with qkv."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckIndexValid()
{
    bool isValid = true;
    const gert::StorageShape* indexShapePtr = context_->GetInputShape(INDEX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexShapePtr);
    isValid = isValid && (indexShapePtr->GetStorageShape().GetDimNum() == NUM_ONE);
    isValid = isValid && (indexShapePtr->GetStorageShape().GetShapeSize() == batchSize_ * seqLength_);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "Index's shape size must be equal to B*S."),
                return ge::GRAPH_FAILED);

    auto indexDesc = context_->GetInputDesc(INDEX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
    auto indexDtype = indexDesc->GetDataType();
    OP_CHECK_IF(indexDtype != ge::DataType::DT_INT64, OP_LOGE(context_->GetNodeName(), "index dtype must be int64."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckQOutValid()
{
    bool isValid = true;
    const gert::StorageShape* qOutShapePtr = context_->GetInputShape(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOutShapePtr);
    isValid = isValid && (qOutShapePtr->GetStorageShape().GetDimNum() == NUM_TWO);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "q_out must be 2D tensor."), return ge::GRAPH_FAILED);
    isValid = isValid && (qOutShapePtr->GetStorageShape().GetDim(0) == batchSize_ * seqLength_);
    isValid = isValid && (qOutShapePtr->GetStorageShape().GetDim(1) == numHeadQ_ * qkvDim_);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of q_out is invalid."), return ge::GRAPH_FAILED);

    auto qOutDesc = context_->GetInputDesc(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qOutDesc);
    auto qOutDtype = qOutDesc->GetDataType();
    OP_CHECK_IF(qOutDtype != qkvDtype_, OP_LOGE(context_->GetNodeName(), "q_out dtype must be the same with qkv."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckKCacheValid()
{
    auto kCacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheDesc);
    auto kcacheDtype = kCacheDesc->GetDataType();
    OP_CHECK_IF(kcacheDtype != qkvDtype_ && kcacheDtype != ge::DataType::DT_INT8,
                OP_LOGE(context_->GetNodeName(), "k_cache dtype must either be the same with qkv, or be int8."),
                return ge::GRAPH_FAILED);
    isKQuant_ = kcacheDtype == ge::DataType::DT_INT8 ? 1 : 0;

    bool isValid = true;
    const gert::StorageShape* kCacheShapePtr = context_->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheShapePtr);
    isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDimNum() == NUM_FOUR);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "k_cache must be 4D tensor."), return ge::GRAPH_FAILED);
    blockNum_ = kCacheShapePtr->GetStorageShape().GetDim(0);
    blockSize_ = kCacheShapePtr->GetStorageShape().GetDim(2);
    if (isKQuant_) {
        isValid =
            isValid && (kCacheShapePtr->GetStorageShape().GetDim(1) == numHeadK_ * qkvDim_ / INT8_BLOCK_ALIGN_NUM);
        isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDim(3) == INT8_BLOCK_ALIGN_NUM);
    } else {
        isValid =
            isValid && (kCacheShapePtr->GetStorageShape().GetDim(1) == numHeadK_ * qkvDim_ / FP16_BLOCK_ALIGN_NUM);
        isValid = isValid && (kCacheShapePtr->GetStorageShape().GetDim(3) == FP16_BLOCK_ALIGN_NUM);
    }
    OP_CHECK_IF(
        blockNum_ < ((seqLength_ + blockSize_ - 1) / blockSize_ * batchSize_),
        OP_LOGE(context_->GetNodeName(), "blockNum must be greater than or equal to Ceil(Sqkv / blockSize) * Bqkv."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of k_cache is invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckVCacheValid()
{
    auto vCacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheDesc);
    auto vcacheDtype = vCacheDesc->GetDataType();
    OP_CHECK_IF(vcacheDtype != qkvDtype_ && vcacheDtype != ge::DataType::DT_INT8,
                OP_LOGE(context_->GetNodeName(), "v_cache dtype must either be the same with qkv, or be int8."),
                return ge::GRAPH_FAILED);
    isVQuant_ = vcacheDtype == ge::DataType::DT_INT8 ? 1 : 0;

    bool isValid = true;
    const gert::StorageShape* vCacheShapePtr = context_->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheShapePtr);
    isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDimNum() == NUM_FOUR);
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "v_cache must be 4D tensor."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShapePtr->GetStorageShape().GetDim(0) != blockNum_,
                OP_LOGE(context_->GetNodeName(), "the blockNums of v_cache must be equal to that of k_cache."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShapePtr->GetStorageShape().GetDim(2) != blockSize_,
                OP_LOGE(context_->GetNodeName(), "the blockSize of v_cache must be equal to that of k_cache."),
                return ge::GRAPH_FAILED);
    if (isVQuant_) {
        isValid =
            isValid && (vCacheShapePtr->GetStorageShape().GetDim(1) == numHeadV_ * qkvDim_ / INT8_BLOCK_ALIGN_NUM);
        isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDim(3) == INT8_BLOCK_ALIGN_NUM);
    } else {
        isValid =
            isValid && (vCacheShapePtr->GetStorageShape().GetDim(1) == numHeadV_ * qkvDim_ / FP16_BLOCK_ALIGN_NUM);
        isValid = isValid && (vCacheShapePtr->GetStorageShape().GetDim(3) == FP16_BLOCK_ALIGN_NUM);
    }
    OP_CHECK_IF(!isValid, OP_LOGE(context_->GetNodeName(), "the shape of v_cache is invalid."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckKScaleValid()
{
    const gert::StorageShape* kScaleShapePtr = context_->GetOptionalInputShape(K_SCALE_IDX);
    OP_CHECK_IF(kScaleShapePtr == nullptr && isKQuant_ == 1,
                OP_LOGE(context_->GetNodeName(), "when k_Cache dtype is int8, k_scale cannot be None."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kScaleShapePtr != nullptr && isKQuant_ == 0,
                OP_LOGE(context_->GetNodeName(), "when k_Cache dtype is not int8, k_scale must be None."),
                return ge::GRAPH_FAILED);
    if (kScaleShapePtr != nullptr) {
        OP_CHECK_IF(kScaleShapePtr->GetStorageShape().GetDimNum() != NUM_TWO,
                    OP_LOGE(context_->GetNodeName(), "k_scale must be 2D tensor."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleShapePtr->GetStorageShape().GetDim(0) != numHeadK_,
                    OP_LOGE(context_->GetNodeName(), "the first dimension of k_scale must be %ld.", numHeadK_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleShapePtr->GetStorageShape().GetDim(1) != qkvDim_,
                    OP_LOGE(context_->GetNodeName(), "the second dimension of k_scale must be %ld.", qkvDim_),
                    return ge::GRAPH_FAILED);
        auto kScaleDesc = context_->GetOptionalInputDesc(K_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kScaleDesc);
        auto kscaleDtype = kScaleDesc->GetDataType();
        OP_CHECK_IF(kscaleDtype != ge::DataType::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "k_scale dtype must be float32."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckVScaleValid()
{
    const gert::StorageShape* vScaleShapePtr = context_->GetOptionalInputShape(V_SCALE_IDX);
    OP_CHECK_IF(vScaleShapePtr == nullptr && isVQuant_ == 1,
                OP_LOGE(context_->GetNodeName(), "when v_Cache dtype is int8, v_scale cannot be None."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vScaleShapePtr != nullptr && isVQuant_ == 0,
                OP_LOGE(context_->GetNodeName(), "when v_Cache dtype is not int8, v_scale must be None."),
                return ge::GRAPH_FAILED);
    if (vScaleShapePtr != nullptr) {
        OP_CHECK_IF(vScaleShapePtr->GetStorageShape().GetDimNum() != NUM_TWO,
                    OP_LOGE(context_->GetNodeName(), "v_scale must be 2D tensor."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(vScaleShapePtr->GetStorageShape().GetDim(0) != numHeadV_,
                    OP_LOGE(context_->GetNodeName(), "the first dimension of v_scale must be %ld.", numHeadV_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vScaleShapePtr->GetStorageShape().GetDim(1) != qkvDim_,
                    OP_LOGE(context_->GetNodeName(), "the second dimension of v_scale must be %ld.", qkvDim_),
                    return ge::GRAPH_FAILED);
        auto vScaleDesc = context_->GetOptionalInputDesc(V_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, vScaleDesc);
        auto vscaleDtype = vScaleDesc->GetDataType();
        OP_CHECK_IF(vscaleDtype != ge::DataType::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "v_scale dtype must be float32."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckKOffsetValid()
{
    const gert::StorageShape* kOffsetShapePtr = context_->GetOptionalInputShape(K_OFFSET_IDX);
    OP_CHECK_IF(kOffsetShapePtr != nullptr && isKQuant_ == 0,
                OP_LOGE(context_->GetNodeName(), "when k_Cache dtype is not int8, k_offset must be None."),
                return ge::GRAPH_FAILED);
    if (kOffsetShapePtr != nullptr) {
        auto kOffsetDesc = context_->GetOptionalInputDesc(K_OFFSET_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kOffsetDesc);
        auto kOffsetDtype = kOffsetDesc->GetDataType();
        OP_CHECK_IF(kOffsetDtype != ge::DataType::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "k_offset dtype must be float32."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckVOffsetValid()
{
    const gert::StorageShape* vOffsetShapePtr = context_->GetOptionalInputShape(V_OFFSET_IDX);
    OP_CHECK_IF(vOffsetShapePtr != nullptr && isVQuant_ == 0,
                OP_LOGE(context_->GetNodeName(), "when v_Cache dtype is not int8, v_offset must be None."),
                return ge::GRAPH_FAILED);
    if (vOffsetShapePtr != nullptr) {
        auto vOffsetDesc = context_->GetOptionalInputDesc(V_OFFSET_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, vOffsetDesc);
        auto vOffsetDtype = vOffsetDesc->GetDataType();
        OP_CHECK_IF(vOffsetDtype != ge::DataType::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "v_offset dtype must be float32."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::GetShapeAttrsInfoInner()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."),
                return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto qkvSizePtr = attrs->GetListInt(QKV_SIZE_IDX); // 获取qkv_size
    OP_CHECK_NULL_WITH_CONTEXT(context_, qkvSizePtr);
    OP_CHECK_IF(qkvSizePtr->GetSize() != NUM_FOUR,
                OP_LOGE(context_->GetNodeName(), "Check qkv_size failed, the size of qkv_size must be 4."),
                return ge::GRAPH_FAILED);
    batchSize_ = qkvSizePtr->GetData()[SHAPE_IDX_B];
    seqLength_ = qkvSizePtr->GetData()[SHAPE_IDX_S];
    numHead_ = qkvSizePtr->GetData()[SHAPE_IDX_N];
    qkvDim_ = qkvSizePtr->GetData()[SHAPE_IDX_D];
    OP_CHECK_IF(batchSize_ <= 0 || seqLength_ <= 0 || numHead_ <= 0 || qkvDim_ <= 0,
                OP_LOGE(context_->GetNodeName(), "Check qkv_size failed, the value of qkv_size must > 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qkvDim_ != 128, OP_LOGE(context_->GetNodeName(), "Dqkv must be 128."), return ge::GRAPH_FAILED);
    reciprocal_ = 1.0 / static_cast<float>(qkvDim_);
    auto headNumsPtr = attrs->GetListInt(HEAD_NUMS_IDX); // 获取Nq、Nk、Nv
    OP_CHECK_NULL_WITH_CONTEXT(context_, headNumsPtr);
    OP_CHECK_IF(headNumsPtr->GetSize() != 3,
                OP_LOGE(context_->GetNodeName(), "Check headNums failed, the size of headNums must be 3."),
                return ge::GRAPH_FAILED);
    numHeadQ_ = headNumsPtr->GetData()[DIM_ZERO];
    numHeadK_ = headNumsPtr->GetData()[DIM_ONE];
    numHeadV_ = headNumsPtr->GetData()[DIM_TWO];
    OP_CHECK_IF(numHeadQ_ <= 0 || numHeadK_ <= 0 || numHeadV_ <= 0,
                OP_LOGE(context_->GetNodeName(), "Check headNums failed, the value of headNums must > 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHead_ != numHeadQ_ + numHeadK_ + numHeadV_,
                OP_LOGE(context_->GetNodeName(), "Check headNums failed, N is not equal to Nq + Nk + Nv."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeadK_ != numHeadV_,
                OP_LOGE(context_->GetNodeName(), "Check headNums failed, Nk is not equal to Nv."),
                return ge::GRAPH_FAILED);
    ropeRange_ = qkvDim_; // 获取ropeRange
    const float* epsilon = attrs->GetFloat(EPSILON_IDX);
    epsilon_ = epsilon == nullptr ? 1e-6 : *epsilon;     // 获取epsilon
    const char* tmpmode = attrs->GetStr(CACHE_MODE_IDX); // 获取cacheMode
    std::unordered_map<std::string, CacheMode> cacheModeMap = {{"PA_NZ", CacheMode::PA_NZ}};
    if (tmpmode == nullptr) {
        currentCacheMode_ = CacheMode::PA_NZ;
    } else {
        std::string cacheMode = tmpmode;
        if (cacheModeMap.find(cacheMode) != cacheModeMap.end()) {
            currentCacheMode_ = cacheModeMap[cacheMode];
        } else {
            OP_LOGE(context_->GetNodeName(), "Check CacheMode failed, CacheMode only support PA_NZ now.");
            return ge::GRAPH_FAILED;
        }
    }
    // 获取is_output_qkv
    const bool* isOutputQkv = attrs->GetBool(IS_OUTPUT_QKV_IDX);
    bool tmp = isOutputQkv == nullptr ? false : *isOutputQkv;
    isOutputQkv_ = tmp ? 1 : 0;
    // 校验并获取输入tensor
    OP_CHECK_IF(CheckQkvValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "qkv is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGammaValid(GAMMA_Q_INDEX) == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "gamma_q is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckGammaValid(GAMMA_K_INDEX) == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "gamma_k is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCosSinValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "cos or sin is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckIndexValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "index is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckQOutValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "qout is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKCacheValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "k_cache is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVCacheValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "v_cache is invalid."),
                return ge::GRAPH_FAILED);
    if (CheckBlockSizeAlign() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckKScaleValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "k_scale is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVScaleValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "v_scale is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKOffsetValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "k_offset is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckVOffsetValid() == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "v_offset is invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// cache BlockSize 按 dtype 判 32 Byte 对齐: int8 -> %32, fp16/bf16 -> %16, k/v 混型取严
ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckBlockSizeAlign()
{
    auto kCacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheDesc);
    auto vCacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheDesc);
    const int64_t alignBytes = BASE_BLOCK_SIZE; // 32 Byte
    const int64_t kCacheBytes = static_cast<int64_t>(ge::GetSizeByDataType(kCacheDesc->GetDataType()));
    const int64_t vCacheBytes = static_cast<int64_t>(ge::GetSizeByDataType(vCacheDesc->GetDataType()));
    OP_CHECK_IF(
        blockSize_ * kCacheBytes % alignBytes != 0 || blockSize_ * vCacheBytes % alignBytes != 0,
        OP_LOGE(context_->GetNodeName(),
                "blockSize must be 32-Byte aligned per cache dtype, but blockSize=%ld, k_cache=%ldB, v_cache=%ldB.",
                blockSize_, kCacheBytes, vCacheBytes),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool QkvRmsNormRopeCacheTilingRegbase::IsCapable()
{
    // isRegbase_ 由 base 的 GetPlatformInfo 设;platformInfo 为空时它从 CompileInfo 取
    // (CompileInfo 是 TilingPrepare 阶段用保证非空的 platformInfo 算好的),故恒可靠。
    return isRegbase_;
}

void QkvRmsNormRopeCacheTilingRegbase::CalBlockTiling()
{
    const int64_t tokenSum = batchSize_ * seqLength_;
    int64_t usedCoreNum = static_cast<int64_t>(coreNum_);
    if (usedCoreNum > tokenSum) {
        usedCoreNum = tokenSum; // 不启用空核
    }
    blockFactor_ = (tokenSum + usedCoreNum - 1) / usedCoreNum; // 每核 token 数,末核吸收余数
    blockDim_ = (tokenSum + blockFactor_ - 1) / blockFactor_;  // ceil 后实际用到的核数
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CalUbTiling()
{
    const int64_t qkvBytes = static_cast<int64_t>(ge::GetSizeByDataType(qkvDtype_));
    const int64_t kCacheBytes = (isKQuant_ != 0) ? INT8_BYTES : qkvBytes;
    const int64_t vCacheBytes = (isVQuant_ != 0) ? INT8_BYTES : qkvBytes;
    const int64_t dimBytes = qkvDim_;

    // 与 token 数无关的常驻 buffer:gamma(q_gamma + k_gamma)+ 量化因子(scale/offset × K/V)
    gammaUbBytes_ = 2 * dimBytes * qkvBytes;
    quantUbBytes_ = 2 * (numHeadK_ + numHeadV_) * dimBytes * FLOAT32_BYTES;

    // 每个 token 占用的 UB(与 arch35 内核的 buffer 布局一一对应)
    // 单 token 的各个 buffer 字节数也在这里一次算准,随 tiling 下发给内核,内核不重算
    //
    // ⚠️ cosSinUbBytes_ 的语义是【cos 或 sin 的单份】,不是"cos+sin 合计":
    // 内核里 cos / sin 是两个**独立的 queue**(regbase.h InitUbBuffers:
    //   InitBuffer(inQueueCos_, 1, cosSinUbBytes_);
    //   InitBuffer(inQueueSin_, 1, cosSinUbBytes_);),
    // 而每个 queue 只装一个 token 的 D 个元素 —— 所以两处相加才是 cos+sin。
    // 曾经把它写成 2*dimBytes*qkvBytes(合计一份)却仍按"单份"下发给两个 queue,
    // 内核实物就比记账多出 `qkvBytes*dimBytes*ubFactor_` 字节 → ubFactor 大的形态
    // 撑爆 UB,报 VEC_ERROR(实测 ubFactor=40 的 fp16 形态必挂,39 及以下正常)。
    int64_t perTokenBytes = numHead_ * dimBytes * qkvBytes; // x(整行 qkv)
    cosSinUbBytes_ = dimBytes * qkvBytes;                   // cos(或 sin)单份
    qOutUbBytes_ = numHeadQ_ * dimBytes * qkvBytes;         // q_out / q_out_before_quant
    kProtoUbBytes_ = numHeadK_ * dimBytes * qkvBytes;       // k_out_before_quant
    vProtoUbBytes_ = numHeadV_ * dimBytes * qkvBytes;       // v_out_before_quant
    kCacheUbBytes_ = numHeadK_ * dimBytes * kCacheBytes;    // k_cache 散射源
    vCacheUbBytes_ = numHeadV_ * dimBytes * vCacheBytes;    // v_cache 散射源
    perTokenBytes +=
        2 * cosSinUbBytes_ + qOutUbBytes_ + kProtoUbBytes_ + vProtoUbBytes_ + kCacheUbBytes_ + vCacheUbBytes_;

    const int64_t fixedBytes = gammaUbBytes_ + quantUbBytes_;
    const int64_t available = static_cast<int64_t>(ubSize_) - UB_RESERVED_BYTES - fixedBytes;
    OP_CHECK_IF(perTokenBytes <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "perTokenUbBytes",
                                                      std::to_string(perTokenBytes).c_str(),
                                                      "per-token UB cost must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(available <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "residentUbBytes",
                                                      std::to_string(-available).c_str(),
                                                      "ubSize is too small for the resident gamma/quant buffers"),
                return ge::GRAPH_FAILED);

    ubFactor_ = available / perTokenBytes;
    // A2 的同义检查在共享的 _tiling.cpp 里(裸 OP_LOGE);这是 A5 新增的独立校验入口,
    // 按红线 R6 用结构化宏上报(不牵连 A2 的既有实现)。
    OP_CHECK_IF(
        ubFactor_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubFactor", std::to_string(ubFactor_).c_str(),
                                              "per-token UB cost exceeds ubSize; token shape is too big"),
        return ge::GRAPH_FAILED);
    if (ubFactor_ > blockFactor_) {
        ubFactor_ = blockFactor_; // 单核 token 数不足一轮时不留空转
    }
    inUbBytes_ = ubFactor_ * numHead_ * dimBytes * qkvBytes;
    cosSinUbBytes_ *= ubFactor_;
    qOutUbBytes_ *= ubFactor_;
    kProtoUbBytes_ *= ubFactor_;
    vProtoUbBytes_ *= ubFactor_;
    kCacheUbBytes_ *= ubFactor_;
    vCacheUbBytes_ *= ubFactor_;
    return ge::GRAPH_SUCCESS;
}

// A5(ascend950)侧收紧:cos/sin 的第一维必须恰为 B*S。
//
// CheckCosSinValid 额外放行了 batch 内广播形态 [B, D],
// 但该形态:
//   ① proto / README / aclnn 三处公开资料均未声明,只承诺 [B*S, D];
//   ② 内核一律按【全局 token 行】读 cos/sin(tokenStart * D),并未实现广播 ——
//      而 cos/sin 只有 [0, B) 行有效:S>1 时 tokenStart 会落在 [B, B*S),
//      读到 cos 张量之外的内存。
ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::CheckCosSinExact()
{
    const gert::StorageShape* cosShapePtr = context_->GetInputShape(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosShapePtr);
    const gert::StorageShape* sinShapePtr = context_->GetInputShape(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinShapePtr);
    const gert::Shape& cosShape = cosShapePtr->GetStorageShape();
    const gert::Shape& sinShape = sinShapePtr->GetStorageShape();

    // 秩已由 CheckCosSinValid 校为 2,此处可直接取 dim 0/1
    const int64_t expect = batchSize_ * seqLength_;
    OP_CHECK_IF(
        cosShape.GetDim(SHAPE_IDX_BS) != expect || sinShape.GetDim(SHAPE_IDX_BS) != expect,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos and sin",
                                              Ops::Base::ToString(cosShape) + " and " + Ops::Base::ToString(sinShape),
                                              "the 0th dim of input cos and sin must be the same and equal to B*S"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::DoOpTiling()
{
    // A5 私有校验入口(实现见本文件上半部分)
    OP_CHECK_IF(GetShapeAttrsInfoInner() == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfoInner failed."), return ge::GRAPH_FAILED);
    // 基类放行了 [B,D] 广播形态但内核读不了,A5 侧单独拒掉
    OP_CHECK_IF(CheckCosSinExact() == ge::GRAPH_FAILED,
                OP_LOGE(context_->GetNodeName(), "cos or sin shape is invalid."), return ge::GRAPH_FAILED);

    CalBlockTiling();
    if (CalUbTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    tilingKey_ = static_cast<uint64_t>(TILING_KEY_REGBASE_PA_NZ);

    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqLength(seqLength_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_qkvDim(qkvDim_);
    tilingData_.set_numHeadQ(numHeadQ_);
    tilingData_.set_numHeadK(numHeadK_);
    tilingData_.set_numHeadV(numHeadV_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_blockNum(blockNum_); // index 合法上界的另一半,内核散射前据此判越界
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_isOutputQkv(isOutputQkv_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_inUbBytes(inUbBytes_);
    tilingData_.set_cosSinUbBytes(cosSinUbBytes_);
    tilingData_.set_qOutUbBytes(qOutUbBytes_);
    tilingData_.set_kProtoUbBytes(kProtoUbBytes_);
    tilingData_.set_vProtoUbBytes(vProtoUbBytes_);
    tilingData_.set_kCacheUbBytes(kCacheUbBytes_);
    tilingData_.set_vCacheUbBytes(vCacheUbBytes_);
    tilingData_.set_gammaUbBytes(gammaUbBytes_);
    tilingData_.set_quantUbBytes(quantUbBytes_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QkvRmsNormRopeCacheTilingRegbase::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkSpaceSize);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void QkvRmsNormRopeCacheTilingRegbase::DumpTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "tilingKey_:    %lu", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "coreNum_:  %lu", coreNum_);
    OP_LOGD(context_->GetNodeName(), "ubSize_:  %lu", ubSize_);
    OP_LOGD(context_->GetNodeName(), "isRegbase_:  %ld", static_cast<int64_t>(isRegbase_));
    OP_LOGD(context_->GetNodeName(), "batchSize_:  %ld", batchSize_);
    OP_LOGD(context_->GetNodeName(), "seqLength_:  %ld", seqLength_);
    OP_LOGD(context_->GetNodeName(), "numHead_:  %ld", numHead_);
    OP_LOGD(context_->GetNodeName(), "qkvDim_:  %ld", qkvDim_);
    OP_LOGD(context_->GetNodeName(), "numHeadQ_:  %ld", numHeadQ_);
    OP_LOGD(context_->GetNodeName(), "numHeadK_:  %ld", numHeadK_);
    OP_LOGD(context_->GetNodeName(), "numHeadV_:  %ld", numHeadV_);
    OP_LOGD(context_->GetNodeName(), "blockNum_:  %ld", blockNum_);
    OP_LOGD(context_->GetNodeName(), "blockSize_:  %ld", blockSize_);
    OP_LOGD(context_->GetNodeName(), "blockFactor_:  %ld", blockFactor_);
    OP_LOGD(context_->GetNodeName(), "blockDim_:  %ld", blockDim_);
    OP_LOGD(context_->GetNodeName(), "ubFactor_:  %ld", ubFactor_);
    OP_LOGD(context_->GetNodeName(), "epsilon_:  %f", epsilon_);
    OP_LOGD(context_->GetNodeName(), "reciprocal_:  %f", reciprocal_);
    OP_LOGD(context_->GetNodeName(), "isOutputQkv_:  %ld", isOutputQkv_);
    OP_LOGD(context_->GetNodeName(), "isKQuant_:  %ld", isKQuant_);
    OP_LOGD(context_->GetNodeName(), "isVQuant_:  %ld", isVQuant_);
}

REGISTER_OPS_TILING_TEMPLATE(QkvRmsNormRopeCache, QkvRmsNormRopeCacheTilingRegbase, TEMPLATE_REGBASE_PRIORITY);
} // namespace optiling
