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
 * \file kv_rms_norm_rope_cache_tiling_base.cpp
 * \brief
 */
#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"

namespace optiling {
static const std::vector<std::string> inputNames = {"kv",         "gamma",         "cos",         "sin",
                                                    "index",      "k_cache",       "ckv_cache",   "k_rope_scale",
                                                    "c_kv_scale", "k_rope_offset", "c_kv_offset", "v"};
using namespace Ops::Base;
std::tuple<int64_t, int64_t, int64_t, int64_t> KvRmsNormRopeCacheTilingBase::GetShapeTuple(
    const gert::TilingContext* context, const int64_t index)
{
    const gert::StorageShape* shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context, "Shape is nullptr."), return std::make_tuple(0, 0, 0, 0));
    // check shape length is DIM_SIZE
    int64_t dimNum = shapePtr->GetStorageShape().GetDimNum();
    if (dimNum != DIM_SIZE) {
        std::string dimNumStr = std::to_string(dimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), inputNames[index].c_str(), dimNumStr.c_str(), "4D");
        return std::make_tuple(0, 0, 0, 0);
    }
    return std::make_tuple(
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_B), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_N),
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_S), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_D));
}

std::tuple<int64_t, int64_t, int64_t, int64_t> KvRmsNormRopeCacheTilingBase::GetOptionalShapeTuple(
    const gert::TilingContext* context, const int64_t index)
{
    const gert::StorageShape* shapePtr = context->GetOptionalInputShape(index);
    OP_CHECK_IF(shapePtr == nullptr, OP_LOGE(context, "Shape is nullptr."), return std::make_tuple(0, 0, 0, 0));
    // check shape length is DIM_SIZE
    int64_t dimNum = shapePtr->GetStorageShape().GetDimNum();
    if (dimNum != DIM_SIZE) {
        std::string dimNumStr = std::to_string(dimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), inputNames[index].c_str(), dimNumStr.c_str(), "4D");
        return std::make_tuple(0, 0, 0, 0);
    }
    return std::make_tuple(
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_B), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_N),
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_S), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_D));
}

void KvRmsNormRopeCacheTilingBase::GetMethodeMode(const gert::TilingContext* context)
{
    auto vShape = context_->GetOptionalInputShape(V_IDX);
    methodMode_ = (vShape != nullptr) ? 1 : 0;
}

bool KvRmsNormRopeCacheTilingBase::IsB1SD(const gert::TilingContext* context)
{
    auto kvShapeTuple = GetShapeTuple(context, KV_INDEX);
    auto cosShapeTuple = GetShapeTuple(context, COS_INDEX);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    int64_t batchSize = std::get<SHAPE_IDX_B>(kvShapeTuple);
    if (batchSize > coreNum_ * BATCHES_FOR_EACH_CORE ||
        seqLen > 1 && std::get<SHAPE_IDX_S>(kvShapeTuple) == std::get<SHAPE_IDX_S>(cosShapeTuple)) {
        return true;
    } else {
        return false;
    }
}

bool KvRmsNormRopeCacheTilingBase::CheckKvValid(const gert::TilingContext* context, int64_t batchSize, int64_t numHead,
                                                int64_t seqLen, int64_t headSize)
{
    auto kvShapeTuple = GetShapeTuple(context, KV_INDEX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(kvShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(kvShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_S>(kvShapeTuple) == seqLen);
    isValid = isValid && (std::get<SHAPE_IDX_D>(kvShapeTuple) == headSize);

    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckVValid(const gert::TilingContext* context, int64_t batchSize, int64_t numHead,
                                               int64_t seqLen, int64_t headSize)
{
    auto vShapeTuple = GetOptionalShapeTuple(context, V_IDX);
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(vShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(vShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_S>(vShapeTuple) == seqLen);
    isValid = isValid && (std::get<SHAPE_IDX_D>(vShapeTuple) == headSize);
    if (!isValid) {
        std::string reasonMsg =
            "the B, N, S and D dimensions of v must match input kv (B=" + std::to_string(batchSize) +
            ", N=" + std::to_string(numHead) + ", S=" + std::to_string(seqLen) + ", D=" + std::to_string(headSize) +
            ")";
        std::string shapesMsg = "(" + std::to_string(std::get<SHAPE_IDX_B>(vShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_N>(vShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_S>(vShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_D>(vShapeTuple)) + ")";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "v", shapesMsg.c_str(), reasonMsg.c_str());
    }
    return isValid;
}

// index 的取值语义：Norm 下是 batch 内的行号(< Scache)，PA 系下是全局槽位号(< BlockNum * BlockSize)。
// Kernel 只能拿到 tiling 下发的上界，故这里算好下发，避免 kernel 用越界的 index 写 cache。
bool KvRmsNormRopeCacheTilingBase::GetCacheRowLimit(int64_t& cacheRowLimit)
{
    if (currentCacheMode_ == CacheMode::Norm) {
        cacheRowLimit = cacheLength_;
        return true;
    }
    const gert::StorageShape* kCacheShapePtr = context_->GetInputShape(K_CACHE_INDEX);
    // 本函数返回 bool，不能用 OP_CHECK_NULL_WITH_CONTEXT：它失败时 return ge::GRAPH_FAILED(0xFFFFFFFF)，
    // 转成 bool 是 true，会被调用方当成"取上界成功"
    OP_CHECK_IF(kCacheShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "shape of k_cache is nullptr."),
                return false);
    int64_t blockNum = kCacheShapePtr->GetStorageShape().GetDim(SHAPE_IDX_B);
    if (blockNum <= 0 || blockSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "block_num, block_size",
            ("(" + std::to_string(blockNum) + ", " + std::to_string(blockSize_) + ") of k_cache").c_str(),
            "the block_num and block_size dimensions of k_cache must be positive");
        return false;
    }
    cacheRowLimit = blockNum * blockSize_;
    return true;
}

bool KvRmsNormRopeCacheTilingBase::CheckCosSinValid(const gert::TilingContext* context, int64_t batchSize,
                                                    int64_t numHead, int64_t seqLen, int64_t headSize)
{
    auto cosShapeTuple = GetShapeTuple(context, COS_INDEX);
    auto sinShapeTuple = GetShapeTuple(context, SIN_INDEX);
    auto sinShape = context->GetInputShape(SIN_INDEX)->GetStorageShape();
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_B>(cosShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(cosShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(cosShapeTuple) == headSize);
    isValid = isValid && (std::get<SHAPE_IDX_B>(sinShapeTuple) == batchSize);
    isValid = isValid && (std::get<SHAPE_IDX_N>(sinShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(sinShapeTuple) == headSize);
    auto cosSeq = std::get<SHAPE_IDX_S>(cosShapeTuple);
    auto sinSeq = std::get<SHAPE_IDX_S>(sinShapeTuple);
    isValid = isValid && (cosSeq == sinSeq);
    if ((cosSeq != seqLen)) {
        if (cosSeq == 1) {
            cosSinNeedBrc_ = 1;
        } else {
            std::string reasonMsg = "the S dimension of cos must be seqLen (" + std::to_string(seqLen) +
                                    ") or 1 (broadcast), got " + std::to_string(cosSeq);
            auto cosShape = context->GetInputShape(COS_INDEX)->GetStorageShape();
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  reasonMsg.c_str());
            return false;
        }
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ || currentCacheMode_ == CacheMode::PA_NZ) {
        auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheDesc);
        ge::DataType kcacheDtype = kcacheDesc->GetDataType();
        bool isValidAlign = true;
        std::string reasonMsg;
        if (kcacheDtype == ge::DT_INT8 || kcacheDtype == ge::DT_HIFLOAT8 || kcacheDtype == ge::DT_FLOAT8_E5M2 ||
            kcacheDtype == ge::DT_FLOAT8_E4M3FN) {
            isValidAlign = isValidAlign && (std::get<SHAPE_IDX_D>(sinShapeTuple) % INT8_BLOCK_ALIGN_NUM == 0);
            reasonMsg = "The D-dimension of input sin must be exactly divisible by " +
                        std::to_string(INT8_BLOCK_ALIGN_NUM) +
                        " when the dtype of input k_cache is INT8, HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN, "
                        "where D is the 3rd axis";
        } else {
            isValidAlign = isValidAlign && (std::get<SHAPE_IDX_D>(sinShapeTuple) % FP16_BLOCK_ALIGN_NUM == 0);
            reasonMsg = "The D-dimension of input sin must be exactly divisible by " +
                        std::to_string(FP16_BLOCK_ALIGN_NUM) +
                        " when the dtype of input k_cache is FLOAT16 or BF16, where D is the 3rd axis";
        }
        if (!isValidAlign) {
            // 对齐失败已按参数正式上报，直接返回，避免落入尾部汇总二次上报（reason 与本失败无关，会误导排查）
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "sin", ToString(sinShape).c_str(),
                                                  reasonMsg.c_str());
            return false;
        }
    }
    if (!isValid) {
        std::string reasonMsg =
            "the B, N and D dimensions of cos and sin must match input kv (B=" + std::to_string(batchSize) +
            ", N=" + std::to_string(numHead) + ", D=" + std::to_string(headSize) +
            "), and the S dimensions of cos and sin must be the same";
        std::string shapesMsg = "cos (" + std::to_string(std::get<SHAPE_IDX_B>(cosShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_N>(cosShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_S>(cosShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_D>(cosShapeTuple)) + "), sin (" +
                                std::to_string(std::get<SHAPE_IDX_B>(sinShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_N>(sinShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_S>(sinShapeTuple)) + ", " +
                                std::to_string(std::get<SHAPE_IDX_D>(sinShapeTuple)) + ")";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapesMsg.c_str(),
                                               reasonMsg.c_str());
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckGammaValid(const gert::TilingContext* context, int64_t headSize)
{
    const gert::StorageShape* gammaShapePtr = context->GetInputShape(GAMMA_INDEX);
    bool isValid = true;
    isValid = isValid && (gammaShapePtr != nullptr);
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDimNum() == 1);
    isValid = isValid && (gammaShapePtr->GetStorageShape().GetDim(0) == headSize);

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ || currentCacheMode_ == CacheMode::PA_NZ) {
        auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, vcacheDesc);
        ge::DataType vcacheDtype = vcacheDesc->GetDataType();
        bool isValidAlign = true;
        std::string reasonMsg;
        if (vcacheDtype == ge::DT_INT8 || vcacheDtype == ge::DT_HIFLOAT8 || vcacheDtype == ge::DT_FLOAT8_E5M2 ||
            vcacheDtype == ge::DT_FLOAT8_E4M3FN) {
            isValidAlign = isValidAlign && (gammaShapePtr->GetStorageShape().GetDim(0) % INT8_BLOCK_ALIGN_NUM == 0);
            reasonMsg = "The 0th axis of input gamma must be exactly divisible by " +
                        std::to_string(INT8_BLOCK_ALIGN_NUM) +
                        " when the dtype of input ckv_cache is INT8, HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN";
        } else {
            isValidAlign = isValidAlign && (gammaShapePtr->GetStorageShape().GetDim(0) % FP16_BLOCK_ALIGN_NUM == 0);
            reasonMsg = "The 0th axis of input gamma must be exactly divisible by " +
                        std::to_string(FP16_BLOCK_ALIGN_NUM) + " when the dtype of input ckv_cache is FLOAT16 or BF16";
        }
        if (!isValidAlign) {
            // 对齐失败已按参数正式上报，直接返回，避免落入尾部汇总二次上报（reason 与本失败无关，会误导排查）
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma",
                                                  ToString(gammaShapePtr->GetStorageShape()).c_str(),
                                                  reasonMsg.c_str());
            return false;
        }
    }
    if (!isValid) {
        std::string gammaShapeStr = (gammaShapePtr == nullptr) ? "nullptr" : ToString(gammaShapePtr->GetStorageShape());
        std::string reasonMsg = "gamma must be a 1D tensor of size headDim (dv=" + std::to_string(headSize) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma", gammaShapeStr.c_str(),
                                              reasonMsg.c_str());
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckCacheValid(const gert::TilingContext* context, int64_t batchSize,
                                                   int64_t numHead, int64_t cacheLen, int64_t headSize,
                                                   size_t cacheIndex, const char* cacheName)
{
    auto cacheShapeTuple = GetShapeTuple(context, cacheIndex);
    int64_t cacheB = std::get<SHAPE_IDX_B>(cacheShapeTuple);
    int64_t cacheN = std::get<SHAPE_IDX_N>(cacheShapeTuple);
    // In CacheMode::Norm the cache is addressed linearly as (b * N + n) * cacheLen * headSize, so the B and N axes may
    // be split or merged freely, but their product must cover every (batch, head) the kernel scatters to.
    if (cacheB * cacheN < batchSize * numHead) {
        // GetShapeTuple 在 shape 为空指针时返回全 0，会走进本分支，故这里必须自己判空后再取 shape
        const gert::StorageShape* cacheShapePtr = context->GetInputShape(cacheIndex);
        std::string cacheShapeStr = (cacheShapePtr == nullptr) ? "nullptr" : ToString(cacheShapePtr->GetStorageShape());
        std::string reasonMsg = "In CacheMode::Norm, the product of the B and N dimensions of " +
                                std::string(cacheName) + " should be no less than " +
                                std::to_string(batchSize * numHead) + " (B * N of input kv), but got " +
                                std::to_string(cacheB * cacheN);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), cacheName, cacheShapeStr.c_str(),
                                              reasonMsg.c_str());
        return false;
    }
    if (cacheB != batchSize || cacheN != numHead) {
        OP_LOGW(context_->GetNodeName(),
                "In CacheMode::Norm, the B and N dimensions of %s are expected to be %ld and %ld, but got %ld and %ld. "
                "The cache is large enough, so it is addressed as if it were [%ld, %ld, %ld, %ld].",
                cacheName, batchSize, numHead, cacheB, cacheN, batchSize, numHead, cacheLen, headSize);
    }
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_S>(cacheShapeTuple) == cacheLen);
    isValid = isValid && (std::get<SHAPE_IDX_D>(cacheShapeTuple) == headSize);
    if (!isValid) {
        const gert::StorageShape* cacheShapePtr = context->GetInputShape(cacheIndex);
        std::string cacheShapeStr = (cacheShapePtr == nullptr) ? "nullptr" : ToString(cacheShapePtr->GetStorageShape());
        std::string reasonMsg = "the S dimension of " + std::string(cacheName) + " must be " +
                                std::to_string(cacheLen) + " and the D dimension must be " + std::to_string(headSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), cacheName, cacheShapeStr.c_str(),
                                              reasonMsg.c_str());
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckKCacheValid(const gert::TilingContext* context, int64_t batchSize,
                                                    int64_t numHead, int64_t cacheLen, int64_t headSize)
{
    return CheckCacheValid(context, batchSize, numHead, cacheLen, headSize, K_CACHE_INDEX, "k_cache");
}

bool KvRmsNormRopeCacheTilingBase::CheckVCacheValid(const gert::TilingContext* context, int64_t batchSize,
                                                    int64_t numHead, int64_t cacheLen, int64_t headSize)
{
    return CheckCacheValid(context, batchSize, numHead, cacheLen, headSize, V_CACHE_INDEX, "v_cache");
}

bool KvRmsNormRopeCacheTilingBase::CheckCacheValidPA(const gert::TilingContext* context, int64_t headSize,
                                                     size_t cacheIndex, const char* cacheName)
{
    // Default PA format cache check
    // In PageAttention mode, cache dimensions are in [block_num, block_size, N, hDim].
    auto cacheShapeTuple = GetShapeTuple(context, cacheIndex);
    int64_t cacheBlkNum = std::get<SHAPE_IDX_BLOCK_NUM>(cacheShapeTuple);
    int64_t cacheBlkSize = std::get<SHAPE_IDX_BLOCK_SIZE>(cacheShapeTuple);
    int64_t cacheN = std::get<DIM_TWO>(cacheShapeTuple);
    int64_t cacheD = std::get<SHAPE_IDX_D>(cacheShapeTuple);

    if (cacheBlkSize < DIM_ONE) {
        // block size must be larger than 0 under all PageAttention modes.
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), cacheName,
                                              ToString(context->GetInputShape(cacheIndex)->GetStorageShape()).c_str(),
                                              "The block_size dimension must be positive");
        return false;
    }
    if (cacheN != this->numHeadKv_) {
        const std::string reason =
            "The num_heads dimension must equal kv num_heads (" + std::to_string(this->numHeadKv_) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), cacheName,
                                              ToString(context->GetInputShape(cacheIndex)->GetStorageShape()).c_str(),
                                              reason.c_str());
        return false;
    }
    if (cacheD != headSize) {
        // Inconsist cache hDim size with the expected value headSize
        const std::string reason = "The head_size dimension must equal " + std::to_string(headSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), cacheName,
                                              ToString(context->GetInputShape(cacheIndex)->GetStorageShape()).c_str(),
                                              reason.c_str());
        return false;
    }

    // Total PA slots shall meet cache demands
    bool enoughSlots = false;
    const CacheMode& mode = currentCacheMode_;
    if (mode == CacheMode::Norm || mode == CacheMode::PA || mode == CacheMode::PA_NZ) {
        // For non-block modes, block_num * block_size ≥ Bkv × Skv
        enoughSlots = (cacheBlkNum * cacheBlkSize >= this->batchKv_ * this->seqLenKv_);
    } else {
        // For PA_BLK_* modes, block_num ≥ Ceil(Skv / block_size) ∗ Bkv
        enoughSlots = (cacheBlkNum >= ((this->seqLenKv_ + cacheBlkSize - 1) / cacheBlkSize) * this->batchKv_);
    }
    if (!enoughSlots) {
        // Note developer to check index contents if inputs are expected to be filtered into compact caches.
        OP_LOGW(context_->GetNodeName(),
                "ATTENTION: Potential insufficient cache slots for %s, make sure the slotmapping indices are safe.",
                cacheName);
    }
    return true;
}

bool KvRmsNormRopeCacheTilingBase::CheckKCacheValidPA(const gert::TilingContext* context, int64_t numHead,
                                                      int64_t headSize)
{
    auto kCacheShapeTuple = GetShapeTuple(context, K_CACHE_INDEX);
    auto kCacheShape = context->GetInputShape(K_CACHE_INDEX)->GetStorageShape();
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_S>(kCacheShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(kCacheShapeTuple) == headSize);
    if (!isValid) {
        std::string reasonMsg = "the N dimension of k_cache must be num_heads of kv (" + std::to_string(numHead) +
                                ") and the D dimension must be " + std::to_string(headSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "k_cache", ToString(kCacheShape).c_str(),
                                              reasonMsg.c_str());
    }

    auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheDesc);
    ge::DataType kcacheDtype = kcacheDesc->GetDataType();
    bool isValidAlign = true;
    std::string reasonMsg;
    if (kcacheDtype == ge::DT_INT8 || kcacheDtype == ge::DT_HIFLOAT8 || kcacheDtype == ge::DT_FLOAT8_E5M2 ||
        kcacheDtype == ge::DT_FLOAT8_E4M3FN) {
        isValidAlign = isValidAlign && (std::get<SHAPE_IDX_N>(kCacheShapeTuple) % INT8_BLOCK_ALIGN_NUM == 0);
        reasonMsg = "The N-dimension of input k_cache must be exactly divisible by " +
                    std::to_string(INT8_BLOCK_ALIGN_NUM) +
                    " when the dtype of input k_cache is INT8, HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN, "
                    "where N is the 1st axis";
    } else {
        isValidAlign = isValidAlign && (std::get<SHAPE_IDX_N>(kCacheShapeTuple) % FP16_BLOCK_ALIGN_NUM == 0);
        reasonMsg = "The N-dimension of input k_cache must be exactly divisible by " +
                    std::to_string(FP16_BLOCK_ALIGN_NUM) +
                    " when the dtype of input k_cache is FLOAT16 or BF16, where N is the 1st axis";
    }
    if (!isValidAlign) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "k_cache", ToString(kCacheShape).c_str(),
                                              reasonMsg.c_str());
    }
    isValid = isValid && isValidAlign;
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckVCacheValidPA(const gert::TilingContext* context, int64_t numHead,
                                                      int64_t headSize)
{
    auto vCacheShapeTuple = GetShapeTuple(context, V_CACHE_INDEX);
    auto vCacheShape = context->GetInputShape(V_CACHE_INDEX)->GetStorageShape();
    bool isValid = true;
    isValid = isValid && (std::get<SHAPE_IDX_S>(vCacheShapeTuple) == numHead);
    isValid = isValid && (std::get<SHAPE_IDX_D>(vCacheShapeTuple) == headSize);
    if (!isValid) {
        std::string reasonMsg = "the N dimension of ckv_cache must be num_heads of kv (" + std::to_string(numHead) +
                                ") and the D dimension must be " + std::to_string(headSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "ckv_cache", ToString(vCacheShape).c_str(),
                                              reasonMsg.c_str());
    }

    auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vcacheDesc);
    ge::DataType vcacheDtype = vcacheDesc->GetDataType();
    bool isValidAlign = true;
    std::string reasonMsg;
    if (vcacheDtype == ge::DT_INT8 || vcacheDtype == ge::DT_HIFLOAT8 || vcacheDtype == ge::DT_FLOAT8_E5M2 ||
        vcacheDtype == ge::DT_FLOAT8_E4M3FN) {
        isValidAlign = isValidAlign && (std::get<SHAPE_IDX_N>(vCacheShapeTuple) % INT8_BLOCK_ALIGN_NUM == 0);
        reasonMsg = "The N-dimension of input ckv_cache must be exactly divisible by " +
                    std::to_string(INT8_BLOCK_ALIGN_NUM) +
                    " when the dtype of input ckv_cache is INT8, HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN, "
                    "where N is the 1st axis";
    } else {
        isValidAlign = isValidAlign && (std::get<SHAPE_IDX_N>(vCacheShapeTuple) % FP16_BLOCK_ALIGN_NUM == 0);
        reasonMsg = "The N-dimension of input ckv_cache must be exactly divisible by " +
                    std::to_string(FP16_BLOCK_ALIGN_NUM) +
                    " when the dtype of input ckv_cache is FLOAT16 or BF16, where N is the 1st axis";
    }
    if (!isValidAlign) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "ckv_cache", ToString(vCacheShape).c_str(),
                                              reasonMsg.c_str());
    }
    isValid = isValid && isValidAlign;
    return isValid;
}

bool KvRmsNormRopeCacheTilingBase::CheckIndexValid(const gert::TilingContext* context, int64_t batchSize,
                                                   int64_t seqLen, int64_t pageSize, CacheMode mode)
{
    bool isValid = true;
    const gert::StorageShape* indexShapePtr = context->GetInputShape(INDEX_INDEX);
    isValid = isValid && (indexShapePtr != nullptr);
    OP_CHECK_IF(!isValid, OP_LOGE(context, "Index shape pointer is null."), return false);
    auto shapeSize = indexShapePtr->GetStorageShape().GetShapeSize();
    if (mode == CacheMode::Norm || mode == CacheMode::PA || mode == CacheMode::PA_NZ) {
        isValid = isValid && (shapeSize == batchSize * seqLen);
        if (!isValid) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "index",
                                                      std::to_string(shapeSize).c_str(),
                                                      ("the total number of elements of index must be B * S (" +
                                                       std::to_string(batchSize) + " * " + std::to_string(seqLen) + ")")
                                                          .c_str());
            return false;
        }
    } else {
        OP_CHECK_IF(pageSize <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "block_size", std::to_string(pageSize).c_str(),
                        "the block_size dimension of k_cache must be positive in PA_BLK cache mode"),
                    return false);
        int64_t page_num = (seqLen + pageSize - 1) / pageSize;
        isValid = isValid && (shapeSize == batchSize * page_num);
        if (!isValid) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "index", std::to_string(shapeSize).c_str(),
                ("the total number of elements of index must be B * page_num (" + std::to_string(batchSize) + " * " +
                 std::to_string(page_num) + ")")
                    .c_str());
            return false;
        }
    }
    return isValid;
}

int64_t KvRmsNormRopeCacheTilingBase::GetQuantMode(const gert::TilingContext* context)
{
    auto scale1Shape = context->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context->GetOptionalInputShape(C_KV_SCALE_IDX);
    bool allNullPtr = (scale1Shape == nullptr) && (scale2Shape == nullptr);
    int64_t qmode = allNullPtr ? NON_QUANT_MODE : QUANT_MODE;

    // Check and warn dissociated quant offsets
    auto offset1Shape = context->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context->GetOptionalInputShape(C_KV_OFFSET_IDX);
    if (scale1Shape == nullptr && offset1Shape != nullptr) {
        OP_LOGW(context_->GetNodeName(), "Orphan quant offset for k_cache without quant scale will be ignored.");
    }
    if (scale2Shape == nullptr && offset2Shape != nullptr) {
        OP_LOGW(context_->GetNodeName(), "Orphan quant offset for ckv_cache without quant scale will be ignored.");
    }
    return qmode;
}

ge::graphStatus KvRmsNormRopeCacheTilingBase::GetPlatformInfo()
{
    vlFp32_ = Ops::Base::GetVRegSize(context_) / FLOAT32_BYTES;
    vlFp16_ = Ops::Base::GetVRegSize(context_) / FLOAT16_BYTES;
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<KvRmsNormRopeCacheCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

/* Guarantee shape consistency between k_cache and ckv_cache except the last hDim.*/
bool KvRmsNormRopeCacheTilingBase::ValidateCacheShapes()
{
    auto kCacheShapeTuple = GetShapeTuple(context_, K_CACHE_INDEX);
    auto vCacheShapeTuple = GetShapeTuple(context_, V_CACHE_INDEX);

    if (std::get<SHAPE_IDX_BLOCK_NUM>(kCacheShapeTuple) != std::get<SHAPE_IDX_BLOCK_NUM>(vCacheShapeTuple)) {
        // Must have same blocknum or batch size for cache
        OP_LOGE(context_->GetNodeName(), "Sizes of the dim-0 MUST be CONSIST between kCache and c_kvCache.");
        return false;
    }

    this->cacheLength_ = std::get<SHAPE_IDX_S>(kCacheShapeTuple);
    this->blockSize_ = std::get<SHAPE_IDX_BLOCK_SIZE>(kCacheShapeTuple);

    if (cacheLength_ != std::get<SHAPE_IDX_S>(vCacheShapeTuple) ||
        blockSize_ != std::get<SHAPE_IDX_BLOCK_SIZE>(vCacheShapeTuple)) {
        OP_LOGE(context_->GetNodeName(),
                "Both sizes of the dim-1 and dim-2 MUST be CONSIST between kCache and c_kvCache.");
        return false;
    }
    return (this->cacheLength_ > 0 && this->blockSize_ > 0);
}

ge::graphStatus KvRmsNormRopeCacheTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        // CANNOT OP_LOGE on nullptr context_ : "context_ can not be nullptr."
        return ge::GRAPH_FAILED;
    }

    isRegbase_ = Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    // GetQuantMode
    quantMode_ = GetQuantMode(context_);
    // Basic info
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    // Dk
    auto cosShapeTuple = GetShapeTuple(context_, COS_INDEX);
    // Dv
    const gert::StorageShape* gammaShapePtr = context_->GetInputShape(GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaShapePtr);
    kv_ = std::get<SHAPE_IDX_D>(kvShapeTuple);
    dv_ = gammaShapePtr->GetStorageShape().GetDim(0);
    dk_ = std::get<SHAPE_IDX_D>(cosShapeTuple);
    reciprocal_ = 1.0 / static_cast<float>(dv_);
    auto kvDesc = context_->GetInputDesc(KV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvDesc);
    kvDtype_ = kvDesc->GetDataType();
    if (kvDtype_ == ge::DT_FLOAT) {
        kvDtypeSize_ = FLOAT32_BYTES;
    } else if (kvDtype_ == ge::DT_FLOAT16 || kvDtype_ == ge::DT_BF16) {
        kvDtypeSize_ = FLOAT16_BYTES;
    }

    int64_t batchSize = std::get<SHAPE_IDX_B>(kvShapeTuple);
    int64_t numHead = std::get<SHAPE_IDX_N>(kvShapeTuple);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    this->batchKv_ = batchSize;
    this->seqLenKv_ = seqLen;
    this->numHeadKv_ = numHead;

    OP_CHECK_IF(!ValidateCacheShapes(),
                OP_LOGE(context_->GetNodeName(),
                        "Normal axes between k_cache and v_cache must have same sizes except for the LAST DIM."),
                return ge::GRAPH_FAILED);

    isMTP_ = (seqLen > 1);
    auto kvStorageShape = context_->GetInputShape(KV_INDEX)->GetStorageShape();
    std::string kvStorageShapeStr = ToString(kvStorageShape);
    if (batchSize < 1) {
        std::string reasonMsg = "The B axis of input kv should be positive, where B refers to the " +
                                std::to_string(SHAPE_IDX_B) + "th dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv", kvStorageShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (seqLen < 1) {
        std::string reasonMsg = "The S axis of input kv should be positive, where S refers to the " +
                                std::to_string(SHAPE_IDX_S) + "th dim";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv", kvStorageShapeStr.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    GetMethodeMode(context_);
    if (methodMode_ == 0) {
        if (numHead != 1) {
            std::string reasonMsg = "The N axis of input kv should be 1 when the optional input v is not present, "
                                    "where N refers to the " +
                                    std::to_string(SHAPE_IDX_N) + "th dim";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv", kvStorageShapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (numHead != 1 && numHead != 2 && numHead != 4 && numHead != 8) {
            std::string reasonMsg =
                "The N axis of input kv should be 1, 2, 4 or 8 when the optional input v is present, "
                "where N refers to the " +
                std::to_string(SHAPE_IDX_N) + "th dim";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv", kvStorageShapeStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (methodMode_ == 1) {
        auto vShape = GetOptionalShapeTuple(context_, V_IDX);
        vlen_ = std::get<SHAPE_IDX_D>(vShape);
        // CheckVValid 失败路径已在内层正式上报，外层仅日志透传
        OP_CHECK_IF(!CheckVValid(context_, batchSize, numHead, seqLen, vlen_),
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "v shape is invalid."), return ge::GRAPH_FAILED);
    }
    // 注：CheckKvValid 的四个实参均取自 kv 自身 shape，恒为真（历史防御检查），保留控制流仅降级为日志透传
    OP_CHECK_IF(!CheckKvValid(context_, batchSize, numHead, seqLen, kv_),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "kv shape is invalid."), return ge::GRAPH_FAILED);
    // Attrs info
    // cache_mode 必须在 CheckCosSinValid / CheckGammaValid 之前解析：这两个函数里按
    // currentCacheMode_ 分支做 NZ 尾轴 C0 对齐校验(sin 的 D 维即 dk、gamma 的 0 轴即 dv)，
    // 若放在其后，它们读到的是默认值 CacheMode::Norm，NZ 分支恒不进入，两条校验形同虚设。
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsilon);
    epsilon_ = *epsilon;
    const char* tmpmode = attrs->GetStr(CACHE_MODE_IDX);
    if (tmpmode != nullptr) {
        std::string cacheMode = tmpmode;
        isPagedAttention_ = (cacheMode == "PA" || cacheMode == "PA_BNSD");
        std::unordered_map<std::string, CacheMode> cacheModeMap = {{"PA", CacheMode::PA},
                                                                   {"PA_BNSD", CacheMode::PA},
                                                                   {"PA_NZ", CacheMode::PA_NZ},
                                                                   {"PA_BLK_BNSD", CacheMode::PA_BLK_BNSD},
                                                                   {"PA_BLK_NZ", CacheMode::PA_BLK_NZ}};
        auto getCacheMode = [&cacheModeMap](const std::string& mode) -> CacheMode {
            auto it = cacheModeMap.find(mode);
            return (it != cacheModeMap.end()) ? it->second : CacheMode::Norm;
        };
        currentCacheMode_ = getCacheMode(cacheMode);
    } else {
        isPagedAttention_ = false;
        currentCacheMode_ = CacheMode::Norm;
    }
    const bool* isOutputKv = attrs->GetBool(IS_OUTPUT_KV_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, isOutputKv);
    isOutputKv_ = *isOutputKv;
    // CheckCosSinValid / CheckGammaValid / CheckIndexValid 失败路径已在内层正式上报，外层仅日志透传
    OP_CHECK_IF(!CheckCosSinValid(context_, batchSize, numHead, seqLen, dk_),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "cos or sin shape is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckGammaValid(context_, dv_),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "gamma shape is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckIndexValid(context_, batchSize, seqLen, blockSize_, currentCacheMode_),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "index shape is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t KvRmsNormRopeCacheTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ge::graphStatus Tiling4KvRmsNormRopeCache(gert::TilingContext* context)
{
    OP_LOGD(context, "TilingForKvRmsNormRopeCache running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4KvRmsNormRopeCache(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4KvRmsNormRopeCache running.");
    auto compileInfo = context->GetCompiledInfo<KvRmsNormRopeCacheCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum <= 0, OP_LOGE(context, "coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(compileInfo->ubSize <= 0, OP_LOGE(context, "ubSize must be greater than 0."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4KvRmsNormRopeCache success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvRmsNormRopeCache)
    .Tiling(Tiling4KvRmsNormRopeCache)
    .TilingParse<KvRmsNormRopeCacheCompileInfo>(TilingPrepare4KvRmsNormRopeCache);

} // namespace optiling
