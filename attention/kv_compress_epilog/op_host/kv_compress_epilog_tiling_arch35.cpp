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
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    OP_LOGE(context_->GetNodeName(), "compileInfoPtr is null"),
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
        OP_CHECK_IF(temp != slotMappingShape.GetDim(i), OP_LOGE(context_->GetNodeName(), "slotMappingShape should equal xStorageShape in dim %d", i), return ge::GRAPH_FAILED);
        bs *= temp;
    }
    d_ = xStorageShape.GetDim(xDims - 1);
    bs_ = bs;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetAttributes()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(context_->GetNodeName(), "get attrs nullptr"),
                return ge::GRAPH_FAILED);

    const int64_t *attrQuantGroupSize = attrs->GetAttrPointer<int64_t>(QUANT_GROUP_SIZE_ATTR_INDEX);
    if (attrQuantGroupSize != nullptr) {
        quantGroupSize_ = *attrQuantGroupSize;
    } else {
        quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    }

    const int64_t *attrQuantMode = attrs->GetAttrPointer<int64_t>(QUANT_MODE_ATTR_INDEX);
    if (attrQuantMode != nullptr) {
        quantMode_ = *attrQuantMode;
    }
    OP_CHECK_IF(quantMode_ != QUANT_MODE_GROUP_QUANT_BF16 && quantMode_ != QUANT_MODE_GROUP_QUANT_E8M0 &&
                    quantMode_ != QUANT_MODE_HIF8_FP4,
                OP_LOGE(context_->GetNodeName(), "quantMode should be 0, 1 or 2"),
                return ge::GRAPH_FAILED);

    // mode2(rope hifloat8 + nope FLOAT4_E2M1) 仅支持 16/32/64 的 per-group 量化粒度
    OP_CHECK_IF(quantMode_ == QUANT_MODE_HIF8_FP4 &&
                    quantGroupSize_ != FP4_GROUP_SIZE_16 && quantGroupSize_ != FP4_GROUP_SIZE_32 &&
                    quantGroupSize_ != FP4_GROUP_SIZE_64,
                OP_LOGE(context_->GetNodeName(),
                        "quant_group_size should be 16/32/64 when quant_mode==2, got %ld", quantGroupSize_),
                return ge::GRAPH_FAILED);

    const bool *attrRoundScale = attrs->GetAttrPointer<bool>(ROUND_SCALE_ATTR_INDEX);
    if (attrRoundScale != nullptr) {
        roundScale_ = *attrRoundScale ? 1 : 0;
    }

    auto scaleAttr = attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    scalesAttr_ = scaleAttr == nullptr ? 1.0f : *scaleAttr;

    OP_LOGI(context_->GetNodeName(),
            "quant_group_size: %ld quantMode_: %ld roundScale_: %ld",
            quantGroupSize_, quantMode_, roundScale_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetDtypeInfo()
{
    auto xDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "get x desc nullptr"),
                return ge::GRAPH_FAILED);
    xDtype_ = xDesc->GetDataType();

    OP_CHECK_IF(xDtype_ != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "x dtype only support BF16, got %d",
                        static_cast<int>(xDtype_)),
                return ge::GRAPH_FAILED);

    auto slotMappingDesc = context_->GetInputDesc(SLOT_MAPPING_INDEX);
    OP_CHECK_IF(slotMappingDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "get slot_mapping desc nullptr"),
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
    OP_CHECK_IF(bs_ <= 0,
                OP_LOGE(context_->GetNodeName(), "input x first dimension must be positive, got %ld", bs_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(d_ <= SLICE_SIZE,
                OP_LOGE(context_->GetNodeName(), "input x tail dimension must be greater than 64, got %ld", d_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(d_ > D_LENGTH_FULL_LOAD,
                OP_LOGE(context_->GetNodeName(), "input x tail dimension must be less than 8192, got %ld", d_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(d_ % SLICE_SIZE != 0,
                OP_LOGE(context_->GetNodeName(), "d_ %% 64 should be 0 , got %ld", d_),
                return ge::GRAPH_FAILED);

    // mode2: nope 段长度(d-64)必须能被 per-group 量化粒度整除
    OP_CHECK_IF(quantMode_ == QUANT_MODE_HIF8_FP4 && (d_ - SLICE_SIZE) % quantGroupSize_ != 0,
                OP_LOGE(context_->GetNodeName(),
                        "(d-64) %% quant_group_size should be 0 when quant_mode==2, got d=%ld group=%ld",
                        d_, quantGroupSize_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::ValidateDtypes()
{
    auto kvCacheOutputDesc = context_->GetOutputDesc(KV_COMPRESS_CACHE_OUTPUT_INDEX);
    OP_CHECK_IF(kvCacheOutputDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "get output desc nullptr"),
                return ge::GRAPH_FAILED);

    ge::DataType outputDtype = kvCacheOutputDesc->GetDataType();
    OP_CHECK_IF((outputDtype != ge::DT_UINT8),
                OP_LOGE(context_->GetNodeName(), "cache dtype only support UINT8, got %d",
                        static_cast<int>(outputDtype)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool KvCompressEpilogTilingArch35::GetCacheViewLayout(
    size_t inputIdx, int64_t &blockSize, int64_t &rowStride, int64_t &blockStride)
{
    if (!context_->InputIsView(inputIdx)) {
        return false;
    }
    auto *inputStride = context_->GetInputStride(inputIdx);
    if (inputStride == nullptr || inputStride->GetDimNum() != CACHE_VIEW_DIM_NUM) {
        return false;
    }
    auto *inputShape = context_->GetInputShape(inputIdx);
    if (inputShape == nullptr) {
        return false;
    }
    const auto &viewShape = inputShape->GetShape();
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
    const auto &cacheLogicalShape = cacheShapePtr->GetShape();
    OP_CHECK_IF(cacheLogicalShape.GetDimNum() != CACHE_VIEW_DIM_NUM,
                OP_LOGE(context_->GetNodeName(),
                        "cache must be 4D [blockNum, blockSize, 1, headDim], got dimNum=%zu",
                        cacheLogicalShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cacheLogicalShape.GetDim(CACHE_ONE_DIM) != CACHE_ONE_DIM_VALUE,
                OP_LOGE(context_->GetNodeName(),
                        "cache dim2 (second-to-last) must be 1 (one compressed vector per token), got %ld",
                        cacheLogicalShape.GetDim(CACHE_ONE_DIM)),
                return ge::GRAPH_FAILED);

    const auto &cacheStorage = cacheShapePtr->GetStorageShape();
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
                OP_LOGE(context_->GetNodeName(),
                        "block_stride %ld must be >= blockSize*headDim %ld", kvCacheBlockStride_, contiguousBlockStride),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(),
            "kvCacheLayout: blockSize=%ld rowStride(headDim)=%ld blockStride=%ld kvCacheCol=%ld",
            kvCacheBlockSize_, kvCacheRowStride_, kvCacheBlockStride_, kvCacheCol_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::DoOpTiling()
{
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
                OP_LOGE(context_->GetNodeName(),
                        "cache headDim(%ld) must be >= padded length kvCacheCol(%ld)", kvCacheRowStride_, kvCacheCol_),
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

    // Set workspace size
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                OP_LOGE(context_->GetNodeName(), "get workspaces nullptr"),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);

    // Save tiling data to buffer
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(context_->GetNodeName(), "get tilingdata nullptr"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
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

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

uint64_t KvCompressEpilogTilingArch35::GetTilingKey() const
{
    // 单一 tiling key; mode0/1/2 在 kernel 内按 tilingData->quantMode 运行时分支量化路径
    return 0UL;
}

ge::graphStatus KvCompressEpilogTilingArch35::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr,
                OP_LOGE(context_->GetNodeName(), "context can not be nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetInputShapes() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetInputShapes failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttributes() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetAttributes failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetDtypeInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetDtypeInfo failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateShapes() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ValidateShapes failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateDtypes() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ValidateDtypes failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch35::RunTiling()
{
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfo failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(DoOpTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcOpTiling failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(PostTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "PostTiling failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForKvCompressEpilog(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("KvCompressEpilog", "Tiling context is null"),
                return ge::GRAPH_FAILED);

    KvCompressEpilogTilingArch35 tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForKvCompressEpilog(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCompressEpilog)
    .Tiling(TilingForKvCompressEpilog)
    .TilingParse<KvCompressEpilogCompileInfo>(TilingPrepareForKvCompressEpilog);

} // namespace optiling
