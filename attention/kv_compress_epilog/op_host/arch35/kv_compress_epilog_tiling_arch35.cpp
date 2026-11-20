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
 * \file kv_compress_epilog_tiling_arch35.cpp
 * \brief Architecture-specific tiling implementation for KvCompressEpilog (arch35)
 */

#include <sstream>
#include "kv_compress_epilog_tiling_arch35.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<KvCompressEpilogCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetInputShapes()
{
    auto shapeX = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    auto shapeSlotMapping = context_->GetInputShape(SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeSlotMapping);
    auto xStorageShape = shapeX->GetStorageShape();
    auto slotMappingShape = shapeSlotMapping->GetStorageShape();
    uint32_t xDims = xStorageShape.GetDimNum();
    uint32_t slotMappingDims = slotMappingShape.GetDimNum();
    OP_CHECK_IF(xDims - 1 != slotMappingDims,
                OP_LOGE(context_->GetNodeName(), "slotMappingDims should equal xDims - 1"), return ge::GRAPH_FAILED);
    int64_t bs = 1;
    for (uint32_t i = 0; i < slotMappingDims; i++) {
        int64_t temp = xStorageShape.GetDim(i);
        OP_CHECK_IF(temp != slotMappingShape.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "slotMappingShape should equal xStorageShape in dim %d", i),
                    return ge::GRAPH_FAILED);
        bs *= temp;
    }
    d_ = xStorageShape.GetDim(xDims - 1);
    bs_ = bs;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetAttributes()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "get attrs nullptr"), return ge::GRAPH_FAILED);

    const int64_t* attrQuantGroupSize = attrs->GetAttrPointer<int64_t>(QUANT_GROUP_SIZE_ATTR_INDEX);
    if (attrQuantGroupSize != nullptr) {
        quantGroupSize_ = *attrQuantGroupSize;
    } else {
        quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    }

    const int64_t* attrQuantMode = attrs->GetAttrPointer<int64_t>(QUANT_MODE_ATTR_INDEX);
    if (attrQuantMode != nullptr) {
        quantMode_ = *attrQuantMode;
    }
    // arch35 支持 quant_mode 0/1/2/3/4；其余取值（输入值）显式报错拦截
    OP_CHECK_IF(quantMode_ != QUANT_MODE_GROUP_QUANT_BF16 && quantMode_ != QUANT_MODE_GROUP_QUANT_E8M0 &&
                    quantMode_ != QUANT_MODE_HIF8_FP4 && quantMode_ != QUANT_MODE_MXFP8 &&
                    quantMode_ != QUANT_MODE_MXFP4,
                OP_LOGE(context_->GetNodeName(), "quantMode should be 0, 1, 2, 3 or 4"), return ge::GRAPH_FAILED);

    // mode2(rope hifloat8 + nope FLOAT4_E2M1) 仅支持 16/32/64 的 per-group 量化粒度
    OP_CHECK_IF(quantMode_ == QUANT_MODE_HIF8_FP4 && quantGroupSize_ != FP4_GROUP_SIZE_16 &&
                    quantGroupSize_ != FP4_GROUP_SIZE_32 && quantGroupSize_ != FP4_GROUP_SIZE_64,
                OP_LOGE(context_->GetNodeName(), "quant_group_size should be 16/32/64 when quant_mode==2, got %ld",
                        quantGroupSize_),
                return ge::GRAPH_FAILED);

    // mode3(MXFP8) 仅支持 group=32；mode4(MXFP4 packed) 支持 16/32
    OP_CHECK_IF(
        quantMode_ == QUANT_MODE_MXFP8 && quantGroupSize_ != MXFP_GROUP_SIZE_32,
        OP_LOGE(context_->GetNodeName(), "quant_group_size must be 32 when quant_mode==3, got %ld", quantGroupSize_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode_ == QUANT_MODE_MXFP4 && quantGroupSize_ != MXFP_GROUP_SIZE_16 &&
                    quantGroupSize_ != MXFP_GROUP_SIZE_32,
                OP_LOGE(context_->GetNodeName(), "quant_group_size should be 16/32 when quant_mode==4, got %ld",
                        quantGroupSize_),
                return ge::GRAPH_FAILED);

    const bool* attrRoundScale = attrs->GetAttrPointer<bool>(ROUND_SCALE_ATTR_INDEX);
    if (attrRoundScale != nullptr) {
        roundScale_ = *attrRoundScale ? 1 : 0;
    }

    auto scaleAttr = attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    scalesAttr_ = scaleAttr == nullptr ? 1.0f : *scaleAttr;

    // mode3/4: x_scale 为保留参数，必须为 1.0
    OP_CHECK_IF(IsMxfpMode() && scalesAttr_ != 1.0f,
                OP_LOGE(context_->GetNodeName(), "x_scale is reserved and must be 1.0 when quant_mode==3/4, got %f",
                        scalesAttr_),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "quant_group_size: %ld quantMode_: %ld roundScale_: %ld", quantGroupSize_,
            quantMode_, roundScale_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetDtypeInfo()
{
    auto xDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get x desc nullptr"), return ge::GRAPH_FAILED);
    xDtype_ = xDesc->GetDataType();

    OP_CHECK_IF(xDtype_ != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "x dtype only support BF16, got %d", static_cast<int>(xDtype_)),
                return ge::GRAPH_FAILED);

    auto slotMappingDesc = context_->GetInputDesc(SLOT_MAPPING_INDEX);
    OP_CHECK_IF(slotMappingDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get slot_mapping desc nullptr"),
                return ge::GRAPH_FAILED);
    slotMappingDtype_ = slotMappingDesc->GetDataType();

    OP_CHECK_IF((slotMappingDtype_ != ge::DT_INT32 && slotMappingDtype_ != ge::DT_INT64),
                OP_LOGE(context_->GetNodeName(), "slot_mapping dtype only support INT32/INT64, got %d",
                        static_cast<int>(slotMappingDtype_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::ValidateShapes()
{
    OP_CHECK_IF(bs_ <= 0, OP_LOGE(context_->GetNodeName(), "input x first dimension must be positive, got %ld", bs_),
                return ge::GRAPH_FAILED);

    if (IsMxfpMode()) {
        // mode3/4：全行量化，d ∈ (0, 8192] 且 d % quant_group_size == 0 且 d % 128 == 0（无 rope/nope 拆分）
        OP_CHECK_IF(d_ <= 0, OP_LOGE(context_->GetNodeName(), "input x tail dimension must be positive, got %ld", d_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(d_ > MXFP_D_LENGTH_MAX,
                    OP_LOGE(context_->GetNodeName(), "input x tail dimension must not exceed 8192, got %ld", d_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            d_ % quantGroupSize_ != 0,
            OP_LOGE(context_->GetNodeName(),
                    "d %% quant_group_size should be 0 when quant_mode==3/4, got d=%ld group=%ld", d_, quantGroupSize_),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            d_ % MXFP_D_LENGTH_ALIGN != 0,
            OP_LOGE(context_->GetNodeName(), "d should be a multiple of 128 when quant_mode==3/4, got d=%ld", d_),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(d_ <= SLICE_SIZE,
                OP_LOGE(context_->GetNodeName(), "input x tail dimension must be greater than 64, got %ld", d_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(d_ > D_LENGTH_FULL_LOAD,
                OP_LOGE(context_->GetNodeName(), "input x tail dimension must be less than 8192, got %ld", d_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(d_ % SLICE_SIZE != 0, OP_LOGE(context_->GetNodeName(), "d_ %% 64 should be 0 , got %ld", d_),
                return ge::GRAPH_FAILED);

    // mode2: nope 段长度(d-64)必须能被 per-group 量化粒度整除
    OP_CHECK_IF(
        quantMode_ == QUANT_MODE_HIF8_FP4 && (d_ - SLICE_SIZE) % quantGroupSize_ != 0,
        OP_LOGE(context_->GetNodeName(),
                "(d-64) %% quant_group_size should be 0 when quant_mode==2, got d=%ld group=%ld", d_, quantGroupSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::ValidateDtypes()
{
    if (IsMxfpMode()) {
        auto cacheDesc = context_->GetInputDesc(KV_COMPRESS_CACHE_INPUT_INDEX);
        OP_CHECK_IF(cacheDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get cache desc nullptr"),
                    return ge::GRAPH_FAILED);
        const ge::DataType cacheDtype = cacheDesc->GetDataType();
        const bool validMxFp8Dtype = cacheDtype == ge::DT_FLOAT8_E4M3FN || cacheDtype == ge::DT_FLOAT8_E5M2;
        OP_CHECK_IF((quantMode_ == QUANT_MODE_MXFP8 && !validMxFp8Dtype) ||
                        (quantMode_ == QUANT_MODE_MXFP4 && cacheDtype != ge::DT_UINT8),
                    OP_LOGE(context_->GetNodeName(),
                            "cache dtype does not match quant_mode, got quant_mode=%ld, cache dtype=%d", quantMode_,
                            static_cast<int>(cacheDtype)),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // legacy mode 0/1/2：保持原有 output dtype 校验，不新增 legacy 约束
    auto kvCacheOutputDesc = context_->GetOutputDesc(KV_COMPRESS_CACHE_OUTPUT_INDEX);
    OP_CHECK_IF(kvCacheOutputDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get output desc nullptr"),
                return ge::GRAPH_FAILED);
    const ge::DataType outputDtype = kvCacheOutputDesc->GetDataType();
    OP_CHECK_IF(
        (outputDtype != ge::DT_UINT8),
        OP_LOGE(context_->GetNodeName(), "cache dtype only support UINT8, got %d", static_cast<int>(outputDtype)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool KvCompressEpilogTilingArch35::GetCacheViewLayout(size_t inputIdx, int64_t& blockSize, int64_t& rowStride,
                                                      int64_t& blockStride)
{
    if (!context_->InputIsView(inputIdx)) {
        return false;
    }
    auto* inputStride = context_->GetInputStride(inputIdx);
    if (inputStride == nullptr || inputStride->GetDimNum() != CACHE_VIEW_DIM_NUM) {
        return false;
    }
    auto* inputShape = context_->GetInputShape(inputIdx);
    if (inputShape == nullptr) {
        return false;
    }
    const auto& viewShape = inputShape->GetShape();
    if (viewShape.GetDimNum() != CACHE_VIEW_DIM_NUM) {
        return false;
    }
    // [blockNum, blockSize, 1, headDim]
    blockSize = viewShape.GetDim(CACHE_BLOCKSIZE_DIM);
    rowStride = inputStride->GetStride(CACHE_BLOCKSIZE_DIM);
    blockStride = inputStride->GetStride(CACHE_BLOCKNUM_DIM);
    if (blockSize <= 0 || rowStride <= 0 || blockStride <= 0) {
        return false;
    }
    const int64_t storageElems = inputShape->GetStorageShape().GetShapeSize();
    if (storageElems > 0 && (blockStride > storageElems || rowStride > storageElems || blockSize > storageElems)) {
        return false;
    }
    return true;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetKvCacheLayout()
{
    auto cacheShapePtr = context_->GetInputShape(KV_COMPRESS_CACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cacheShapePtr);
    const auto& cacheLogicalShape = cacheShapePtr->GetShape();
    OP_CHECK_IF(cacheLogicalShape.GetDimNum() != CACHE_VIEW_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "cache must be 4D [blockNum, blockSize, 1, headDim], got dimNum=%zu",
                        cacheLogicalShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cacheLogicalShape.GetDim(CACHE_ONE_DIM) != CACHE_ONE_DIM_VALUE,
                OP_LOGE(context_->GetNodeName(),
                        "cache dim2 (second-to-last) must be 1 (one compressed vector per token), got %ld",
                        cacheLogicalShape.GetDim(CACHE_ONE_DIM)),
                return ge::GRAPH_FAILED);

    const auto& cacheStorage = cacheShapePtr->GetStorageShape();
    int64_t cacheLastDim = cacheStorage.GetDim(cacheStorage.GetDimNum() - 1);
    kvCacheRowStride_ = cacheLastDim;
    kvCacheBlockSize_ = 1;
    kvCacheBlockStride_ = cacheLastDim;

    int64_t viewBlockSize = 0;
    int64_t viewRowStride = 0;
    int64_t viewBlockStride = 0;
    if (GetCacheViewLayout(KV_COMPRESS_CACHE_INPUT_INDEX, viewBlockSize, viewRowStride, viewBlockStride)) {
        kvCacheBlockSize_ = viewBlockSize;
        kvCacheRowStride_ = viewRowStride;
        kvCacheBlockStride_ = viewBlockStride;
    }

    int64_t contiguousBlockStride = kvCacheBlockSize_ * kvCacheRowStride_;
    OP_CHECK_IF(kvCacheBlockStride_ < contiguousBlockStride,
                OP_LOGE(context_->GetNodeName(), "block_stride %ld must be >= blockSize*headDim %ld",
                        kvCacheBlockStride_, contiguousBlockStride),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(),
            "kvCacheLayout: blockSize=%ld rowStride(headDim)=%ld blockStride=%ld kvCacheCol=%ld", kvCacheBlockSize_,
            kvCacheRowStride_, kvCacheBlockStride_, kvCacheCol_);
    return ge::GRAPH_SUCCESS;
}

bool KvCompressEpilogTilingArch35::IsMxfpMode() const
{
    return quantMode_ == QUANT_MODE_MXFP8 || quantMode_ == QUANT_MODE_MXFP4;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetMxfpCacheLayout()
{
    auto cacheShapePtr = context_->GetInputShape(KV_COMPRESS_CACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cacheShapePtr);
    // 布局推导使用逻辑 shape：aclnn/l0 执行链路下 storage shape 为展平形态（rank 1），
    // 仅 logical shape 保留原始维度（与 legacy GetKvCacheLayout 的取法一致，全链路稳健）。
    const auto& cacheLogical = cacheShapePtr->GetShape();
    const int64_t cacheRank = static_cast<int64_t>(cacheLogical.GetDimNum());
    OP_CHECK_IF(
        cacheRank != MXFP_CACHE_RANK_2D && cacheRank != MXFP_CACHE_RANK_4D,
        OP_LOGE(context_->GetNodeName(), "cache must be 2D or 4D when quant_mode==3/4, got rank %ld", cacheRank),
        return ge::GRAPH_FAILED);

    if (cacheRank == MXFP_CACHE_RANK_2D) {
        // layout1 平铺：cache [N, kvCacheCol]，slot 即行号，行落点 slot * kvCacheRowStride
        layout_ = MXFP_LAYOUT_1;
        cacheRows_ = cacheLogical.GetDim(0);
        kvCacheRowStride_ = cacheLogical.GetDim(1);
        kvCacheBlockSize_ = 1;
        layout2CacheCol_ = 0; // 仅 layout2 使用
        if (context_->InputIsView(KV_COMPRESS_CACHE_INPUT_INDEX)) {
            // 2D view：物理行步长从 view stride(0) 派生
            auto* inputStride = context_->GetInputStride(KV_COMPRESS_CACHE_INPUT_INDEX);
            if (inputStride != nullptr && inputStride->GetDimNum() == MXFP_CACHE_RANK_2D &&
                inputStride->GetStride(0) > 0) {
                kvCacheRowStride_ = inputStride->GetStride(0);
            }
        }
        // blockSize==1 退化语义：块步长恒等于派生后的行步长（须在 view 派生之后赋值，
        // 保证非连续 view 下 kvCacheBlockStride 与 kvCacheRowStride 一致，不留过期逻辑列宽）
        kvCacheBlockStride_ = kvCacheRowStride_;
    } else {
        // layout2 分页：cache [blockNum, blockSize, 1, cacheCol]，token 交错 bundle 落点为
        //     block * kvCacheBlockStride + pos * layout2TokenBundleSize
        layout_ = MXFP_LAYOUT_2;
        const int64_t blockNum = cacheLogical.GetDim(0);
        kvCacheBlockSize_ = cacheLogical.GetDim(1);
        OP_CHECK_IF(cacheLogical.GetDim(2) != CACHE_ONE_DIM_VALUE,
                    OP_LOGE(context_->GetNodeName(), "layout2 cache dim2 must be 1, got %ld", cacheLogical.GetDim(2)),
                    return ge::GRAPH_FAILED);
        kvCacheRowStride_ = cacheLogical.GetDim(MXFP_LAYOUT2_CACHE_COL_DIM);
        layout2CacheCol_ = kvCacheRowStride_;
        cacheRows_ = blockNum * kvCacheBlockSize_;
        kvCacheBlockStride_ = kvCacheBlockSize_ * kvCacheRowStride_; // 连续回退值
        int64_t viewBlockSize = 0;
        int64_t viewRowStride = 0;
        int64_t viewBlockStride = 0;
        if (GetCacheViewLayout(KV_COMPRESS_CACHE_INPUT_INDEX, viewBlockSize, viewRowStride, viewBlockStride)) {
            // 物理 block 步长从 cache descriptor 的 view stride(0) 派生（替代 v2 的 block_stride attr）
            kvCacheBlockStride_ = viewBlockStride;
        }
    }

    OP_CHECK_IF(cacheRows_ <= 0 || kvCacheRowStride_ <= 0,
                OP_LOGE(context_->GetNodeName(), "cache dimensions must be positive, got cacheRows=%ld, rowStride=%ld",
                        cacheRows_, kvCacheRowStride_),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(),
            "mxfpCacheLayout: layout=%ld cacheRows=%ld rowStride=%ld blockSize=%ld blockStride=%ld", layout_,
            cacheRows_, kvCacheRowStride_, kvCacheBlockSize_, kvCacheBlockStride_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::DoMxfpTiling()
{
    if (GetMxfpCacheLayout() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 行格式参数（全行量化，无 rope/nope 拆分）：
    //   layout1 行布局 [dataCol 数据][scaleCol 个 bf16 scale]，无 pad
    //   layout2 每 token 交错 bundle [dataCol 数据 | scaleBytes scale]
    const int64_t bf16Bytes = 2;
    scaleCol_ = d_ / quantGroupSize_;
    dataCol_ = quantMode_ == QUANT_MODE_MXFP8 ? d_ : d_ / 2;
    const int64_t concatCol = dataCol_ + scaleCol_ * bf16Bytes;
    kvCacheCol_ = concatCol;
    const int64_t padCol = 0;
    scaleBytes_ = scaleCol_ * bf16Bytes;
    layout2TokenBundleSize_ = dataCol_ + scaleBytes_;

    if (layout_ == MXFP_LAYOUT_2) {
        OP_CHECK_IF(kvCacheRowStride_ < layout2TokenBundleSize_,
                    OP_LOGE(context_->GetNodeName(), "layout2 cache row width must be at least %ld bytes, got %ld",
                            layout2TokenBundleSize_, kvCacheRowStride_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kvCacheBlockStride_ < kvCacheBlockSize_ * kvCacheRowStride_,
                    OP_LOGE(context_->GetNodeName(), "layout2 block_stride must be >= blockSize*cacheCol, got %ld",
                            kvCacheBlockStride_),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(kvCacheRowStride_ < kvCacheCol_,
                    OP_LOGE(context_->GetNodeName(), "cache row width must be at least %ld bytes, got %ld", kvCacheCol_,
                            kvCacheRowStride_),
                    return ge::GRAPH_FAILED);
    }

    // 行切分与 rowFactor 线性扫描（v2 调优策略：逐候选对 UB 五项预算做线性扫描）
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    const int64_t groupCount = d_ / quantGroupSize_;
    const int64_t scratchRowBytes = quantMode_ == QUANT_MODE_MXFP8 ?
                                        RoundUp(groupCount * static_cast<int64_t>(sizeof(float)), BLOCK_SIZE) :
                                        RoundUp(groupCount * static_cast<int64_t>(sizeof(uint16_t)), BLOCK_SIZE);
    const int64_t slotElementBytes = slotMappingDtype_ == ge::DT_INT64 ? sizeof(int64_t) : sizeof(int32_t);
    int64_t rowFactor = 0;
    for (int64_t candidate = 1; candidate <= rowOfFormerBlock_; ++candidate) {
        const int64_t xBytes = candidate * RoundUp(d_, MXFP_X_BUFFER_ALIGN) * MXFP_X_ELEMENT_BYTES * MXFP_DOUBLE_BUFFER;
        const int64_t yBytes = candidate * RoundUp(kvCacheCol_, BLOCK_SIZE) * MXFP_DOUBLE_BUFFER;
        const int64_t scratchBytes = candidate * scratchRowBytes;
        const int64_t indexBytes = RoundUp(candidate * slotElementBytes, BLOCK_SIZE);
        if (xBytes + yBytes + scratchBytes + indexBytes > static_cast<int64_t>(ubSize_)) {
            break;
        }
        rowFactor = candidate;
    }
    OP_CHECK_IF(rowFactor == 0,
                OP_LOGE(context_->GetNodeName(), "UB is insufficient for one row, ubSize=%lu, d=%ld, cacheCol=%ld",
                        ubSize_, d_, kvCacheCol_),
                return ge::GRAPH_FAILED);
    rowFactor_ = rowFactor;

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_kvCacheCol(kvCacheCol_);
    tilingData_.set_kvCacheRowStride(kvCacheRowStride_);
    tilingData_.set_kvCacheBlockSize(kvCacheBlockSize_);
    tilingData_.set_kvCacheBlockStride(kvCacheBlockStride_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_concatCol(concatCol);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_perGroupSize(quantGroupSize_);
    tilingData_.set_padCol(padCol);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_scalesAttr(scalesAttr_);
    tilingData_.set_layout(layout_);
    tilingData_.set_cacheRows(cacheRows_);
    tilingData_.set_dataCol(dataCol_);
    tilingData_.set_scaleBytes(scaleBytes_);
    tilingData_.set_layout2TokenBundleSize(layout2TokenBundleSize_);
    tilingData_.set_layout2CacheCol(layout2CacheCol_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::DoOpTiling()
{
    if (IsMxfpMode()) {
        return DoMxfpTiling();
    }

    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t concatCol;
    int64_t padCol;
    if (quantMode_ == QUANT_MODE_HIF8_FP4) {
        // mode2 行布局: [rope hifloat8 64B][nope FLOAT4_E2M1 (d-64)/2 B][nope bf16 scale nGroup*2 B][pad]
        scaleCol_ = (d_ - SLICE_SIZE) / quantGroupSize_; // nGroup, (d-64)%G==0 已在 ValidateShapes 校验
        concatCol = ROPE_HIF8_BYTES + (d_ - SLICE_SIZE) / 2 + scaleCol_ * FP4_SCALE_BYTES;
    } else {
        // mode0/1 行布局: [rope bf16 128B][nope fp8 (d-64)B][scale]
        scaleCol_ = CeilDiv(d_ - 64, static_cast<int64_t>(64));
        int64_t scaleBytes = 1;
        if (quantMode_ == QUANT_MODE_GROUP_QUANT_BF16) {
            scaleBytes = 2;
        }
        concatCol = d_ - SLICE_SIZE + SLICE_SIZE * 2 + scaleCol_ * scaleBytes;
    }
    if (quantMode_ == QUANT_MODE_GROUP_QUANT_E8M0) {
        kvCacheCol_ = concatCol;
    } else {
        kvCacheCol_ = RoundUp(concatCol, BLOCK_SIZE);
    }
    padCol = kvCacheCol_ - concatCol;

    if (GetKvCacheLayout() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(kvCacheCol_ > kvCacheRowStride_,
                OP_LOGE(context_->GetNodeName(), "cache headDim(%ld) must be >= padded length kvCacheCol(%ld)",
                        kvCacheRowStride_, kvCacheCol_),
                return ge::GRAPH_FAILED);

    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, MIN_ROW_PER_CORE);

    int64_t scratchRowBytes = 0;
    if (quantMode_ == QUANT_MODE_HIF8_FP4) {
        int64_t numChunks = (d_ - SLICE_SIZE) / SLICE_SIZE; // (d-64)/64
        scratchRowBytes = numChunks * FP4_SCRATCH_SLOTS_PER_CHUNK * FP4_SCALE_BYTES;
    }

    rowFactor_ = rowOnceLoop;
    // d全载,通过二分查找尝试搬入更多的bs
    int64_t lo = rowOnceLoop;
    int64_t hi = rowOfFormerBlock_;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int64_t xSize = mid * RoundUp(d_, X_BUFFER_ALIGN) * X_ELEMENT_BYTES * DOUBLE_BUFFER;
        int64_t ySize = mid * RoundUp(kvCacheCol_, BLOCK_SIZE) * DOUBLE_BUFFER;
        int64_t tmpBufferSize = RoundUp(mid, TMP_BUFFER_ALIGN) * TMP_BUFFER_ELEMENT_BYTES;
        int64_t scratchSize = mid * scratchRowBytes;
        int64_t totalSize = xSize + ySize + tmpBufferSize + scratchSize;
        if (totalSize <= static_cast<int64_t>(ubSize_)) {
            rowFactor_ = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_kvCacheCol(kvCacheCol_);
    tilingData_.set_kvCacheRowStride(kvCacheRowStride_);
    tilingData_.set_kvCacheBlockSize(kvCacheBlockSize_);
    tilingData_.set_kvCacheBlockStride(kvCacheBlockStride_);
    tilingData_.set_xRowStride(0); // arch22 专用字段, arch35 不使用置 0
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_concatCol(concatCol);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_perGroupSize(quantGroupSize_);
    tilingData_.set_padCol(padCol);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_scalesAttr(scalesAttr_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::PostTiling()
{
    // Set block dimension (number of AI cores to use)
    context_->SetBlockDim(usedCoreNums_);

    // Set tiling key for kernel dispatch
    context_->SetTilingKey(GetTilingKey());

    // Set workspace size（全部模式 0，与仓内 29f3b8185 整改后的存量行为一致；
    // 实测 aclnn 两阶段调用在 tiling 正常时 workspace=0 可正常执行）
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "get workspaces nullptr"),
                return ge::GRAPH_FAILED);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;

    // Save tiling data to buffer
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata nullptr"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // Log tiling information
    DumpTilingInfo();

    return ge::GRAPH_SUCCESS;
}

void KvCompressEpilogTilingArch35::DumpTilingInfo()
{
    std::ostringstream info;
    info << "bs: " << tilingData_.get_bs();
    info << ", d: " << tilingData_.get_d();
    info << ", kvCacheCol: " << tilingData_.get_kvCacheCol();
    info << ", kvCacheRowStride: " << tilingData_.get_kvCacheRowStride();
    info << ", kvCacheBlockSize: " << tilingData_.get_kvCacheBlockSize();
    info << ", kvCacheBlockStride: " << tilingData_.get_kvCacheBlockStride();
    info << ", scaleCol: " << tilingData_.get_scaleCol();
    info << ", concatCol: " << tilingData_.get_concatCol();
    info << ", quantMode: " << tilingData_.get_quantMode();
    info << ", roundScale: " << tilingData_.get_roundScale();
    info << ", padCol: " << tilingData_.get_padCol();
    info << ", rowOfFormerBlock: " << tilingData_.get_rowOfFormerBlock();
    info << ", rowOfTailBlock: " << tilingData_.get_rowOfTailBlock();
    info << ", rowLoopOfFormerBlock: " << tilingData_.get_rowLoopOfFormerBlock();
    info << ", rowLoopOfTailBlock: " << tilingData_.get_rowLoopOfTailBlock();
    info << ", rowFactor: " << tilingData_.get_rowFactor();
    info << ", tailRowFactorOfFormerBlock: " << tilingData_.get_tailRowFactorOfFormerBlock();
    info << ", tailRowFactorOfTailBlock: " << tilingData_.get_tailRowFactorOfTailBlock();
    if (tilingData_.get_quantMode() == QUANT_MODE_MXFP8 || tilingData_.get_quantMode() == QUANT_MODE_MXFP4) {
        info << ", layout: " << tilingData_.get_layout();
        info << ", cacheRows: " << tilingData_.get_cacheRows();
        info << ", dataCol: " << tilingData_.get_dataCol();
        info << ", scaleBytes: " << tilingData_.get_scaleBytes();
        info << ", layout2TokenBundleSize: " << tilingData_.get_layout2TokenBundleSize();
        info << ", layout2CacheCol: " << tilingData_.get_layout2CacheCol();
    }

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

uint64_t KvCompressEpilogTilingArch35::GetMxfpTilingKey() const
{
    // 编译期分发：百位区分布局（20xx=layout1 2D，21xx=layout2 4D），个位区分量化族
    // （0=MXFP8-g32，1=MXFP4-g32，2=MXFP4-g16）
    if (quantMode_ == QUANT_MODE_MXFP8) {
        return layout_ == MXFP_LAYOUT_2 ? 2100UL : 2000UL;
    }
    if (quantGroupSize_ == MXFP_GROUP_SIZE_16) {
        return layout_ == MXFP_LAYOUT_2 ? 2102UL : 2002UL;
    }
    return layout_ == MXFP_LAYOUT_2 ? 2101UL : 2001UL;
}

uint64_t KvCompressEpilogTilingArch35::GetTilingKey() const
{
    if (IsMxfpMode()) {
        return GetMxfpTilingKey();
    }
    // legacy mode 0/1/2 单一 tiling key; kernel 内按 tilingData->quantMode 运行时分支量化路径
    return 0UL;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context can not be nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetInputShapes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetInputShapes failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttributes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetAttributes failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetDtypeInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetDtypeInfo failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateShapes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "ValidateShapes failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateDtypes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "ValidateDtypes failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::RunTiling()
{
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfo failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalcOpTiling failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(PostTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "PostTiling failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
