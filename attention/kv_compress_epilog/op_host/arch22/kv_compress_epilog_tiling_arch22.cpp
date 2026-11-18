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
 * \file kv_compress_epilog_tiling_arch22.cpp
 * \brief Architecture-specific tiling implementation for KvCompressEpilog quant_mode=1 (arch22, DAV_2201)。
 *        算子 tiling 注册与按 arch 分派入口见 ../kv_compress_epilog_tiling.cpp。
 */

#include <algorithm>
#include <sstream>
#include "kv_compress_epilog_tiling_arch22.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

template <typename T>
static inline T CeilDivArch22(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

static inline int64_t RoundUpArch22(int64_t x, int64_t y)
{
    return CeilDivArch22(x, y) * y;
}

ge::graphStatus KvCompressEpilogTilingArch22::GetPlatformInfo()
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

ge::graphStatus KvCompressEpilogTilingArch22::GetInputShapes()
{
    auto shapeX = context_->GetInputShape(X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    auto shapeSlotMapping = context_->GetInputShape(SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeSlotMapping);
    auto xStorageShape = shapeX->GetStorageShape();
    auto slotMappingShape = shapeSlotMapping->GetStorageShape();
    uint32_t xDims = xStorageShape.GetDimNum();
    uint32_t slotMappingDims = slotMappingShape.GetDimNum();
    // INT8: x 恰为 2D [bs, 512] (前 448 列 nope, 后 64 列 rope)
    OP_CHECK_IF(xDims != 2,
                OP_LOGE(context_->GetNodeName(), "x must be 2D [bs, d] when quant_mode==1, got dimNum=%u", xDims),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDims - 1 != slotMappingDims,
                OP_LOGE(context_->GetNodeName(), "slotMappingDims should equal xDims - 1"), return ge::GRAPH_FAILED);
    int64_t bs = 1;
    for (uint32_t i = 0; i < slotMappingDims; i++) {
        int64_t temp = xStorageShape.GetDim(i);
        OP_CHECK_IF(temp != slotMappingShape.GetDim(i),
                    OP_LOGE(context_->GetNodeName(), "slotMappingShape should equal xStorageShape in dim %u", i),
                    return ge::GRAPH_FAILED);
        bs *= temp;
    }
    d_ = xStorageShape.GetDim(xDims - 1);
    bs_ = bs;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch22::GetAttributes()
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
    // arch22 (A2/A3) 仅支持 quant_mode=1 (int8 量化); 其余取值显式报错拦截
    OP_CHECK_IF(
        quantMode_ != QUANT_MODE_INT8,
        OP_LOGE(context_->GetNodeName(),
                "quant_mode %ld is not supported on arch22 (A2/A3); only quant_mode=1 (INT8) is supported", quantMode_),
        return ge::GRAPH_FAILED);

    // INT8: 量化分组大小必须等于 64
    OP_CHECK_IF(
        quantGroupSize_ != DEFAULT_QUANT_GROUP_SIZE,
        OP_LOGE(context_->GetNodeName(), "quant_group_size must be 64 when quant_mode==1, got %ld", quantGroupSize_),
        return ge::GRAPH_FAILED);

    const bool* attrRoundScale = attrs->GetAttrPointer<bool>(ROUND_SCALE_ATTR_INDEX);
    if (attrRoundScale != nullptr) {
        roundScale_ = *attrRoundScale ? 1 : 0;
    }

    auto scaleAttr = attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
    scalesAttr_ = scaleAttr == nullptr ? 1.0f : *scaleAttr;

    OP_LOGI(context_->GetNodeName(), "quant_group_size: %ld quantMode_: %ld roundScale_: %ld", quantGroupSize_,
            quantMode_, roundScale_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch22::GetDtypeInfo()
{
    auto xDesc = context_->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get x desc nullptr"), return ge::GRAPH_FAILED);
    xDtype_ = xDesc->GetDataType();

    OP_CHECK_IF(xDtype_ != ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "x dtype must be bfloat16, got %d", static_cast<int>(xDtype_)),
                return ge::GRAPH_FAILED);

    auto slotMappingDesc = context_->GetInputDesc(SLOT_MAPPING_INDEX);
    OP_CHECK_IF(slotMappingDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get slot_mapping desc nullptr"),
                return ge::GRAPH_FAILED);
    slotMappingDtype_ = slotMappingDesc->GetDataType();

    OP_CHECK_IF((slotMappingDtype_ != ge::DT_INT32 && slotMappingDtype_ != ge::DT_INT64),
                OP_LOGE(context_->GetNodeName(), "slot_mapping dtype only supports int32/int64, got %d",
                        static_cast<int>(slotMappingDtype_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch22::ValidateShapes()
{
    OP_CHECK_IF(bs_ <= 0, OP_LOGE(context_->GetNodeName(), "input x first dimension must be positive, got %ld", bs_),
                return ge::GRAPH_FAILED);

    // INT8: x 尾轴 d 恒为 512 (前 448 列 nope per-group int8 量化, 后 64 列 rope bf16 位型直通)
    OP_CHECK_IF(d_ != D_LENGTH_MODE3,
                OP_LOGE(context_->GetNodeName(), "d must be 512 when quant_mode==1 (448 nope + 64 rope), got %ld", d_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogTilingArch22::ValidateDtypes()
{
    auto kvCacheOutputDesc = context_->GetOutputDesc(KV_COMPRESS_CACHE_OUTPUT_INDEX);
    OP_CHECK_IF(kvCacheOutputDesc == nullptr, OP_LOGE(context_->GetNodeName(), "get output desc nullptr"),
                return ge::GRAPH_FAILED);

    ge::DataType outputDtype = kvCacheOutputDesc->GetDataType();
    OP_CHECK_IF((outputDtype != ge::DT_UINT8),
                OP_LOGE(context_->GetNodeName(), "cache dtype must be uint8, got %d", static_cast<int>(outputDtype)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool KvCompressEpilogTilingArch22::GetCacheViewLayout(size_t inputIdx, int64_t& blockSize, int64_t& rowStride,
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

ge::graphStatus KvCompressEpilogTilingArch22::GetKvCacheLayout()
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

ge::graphStatus KvCompressEpilogTilingArch22::DoOpTiling()
{
    rowOfFormerBlock_ = CeilDivArch22(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDivArch22(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    // INT8 行布局: [rope bf16 128B][nope int8 (d-64)B][scale bf16 nGroup*2B][pad 32B 对齐]
    scaleCol_ = (d_ - SLICE_SIZE) / quantGroupSize_; // nGroup = (512-64)/64 = 7, d==512 与 group==64 已校验
    int64_t concatCol = ROPE_BF16_BYTES + (d_ - SLICE_SIZE) + scaleCol_ * SCALE_BF16_BYTES; // 128+448+14=590
    kvCacheCol_ = RoundUpArch22(concatCol, BLOCK_SIZE);                                     // 608
    int64_t padCol = kvCacheCol_ - concatCol;                                               // 18

    if (GetKvCacheLayout() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(kvCacheCol_ > kvCacheRowStride_,
                OP_LOGE(context_->GetNodeName(), "cache headDim(%ld) must be >= padded length kvCacheCol(%ld)",
                        kvCacheRowStride_, kvCacheCol_),
                return ge::GRAPH_FAILED);

    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, MIN_ROW_PER_CORE);

    // d 全载, 通过二分查找尝试搬入更多的 bs; INT8 每行 UB 预算 (与 kernel_arch22.h 实际布局一致):
    //   x 搬入(双缓冲) 512x2Bx2 = 2048B | f32 工作区 1408x4B = 5632B | q 值 int16 中间 448x2B(32B 对齐) = 896B
    //   amax 槽位 128x8x4B = 4096B | 行组装输出(双缓冲) 608Bx2 = 1216B | ReduceMax tmpBuffer 固定 256B
    // 每行合计 13888B (+256B 固定), rowFactor 上限 7 (7x13888+256 = 97472B <= 192KB UB)
    int64_t rowFactorUpper = std::min(rowOfFormerBlock_, static_cast<int64_t>(7));
    rowFactor_ = rowOnceLoop;
    int64_t lo = rowOnceLoop;
    int64_t hi = rowFactorUpper;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int64_t xSize = mid * RoundUpArch22(d_, X_BUFFER_ALIGN) * X_ELEMENT_BYTES * DOUBLE_BUFFER;
        int64_t f32WorkSize = mid * 1408 * TMP_BUFFER_ELEMENT_BYTES;
        int64_t qInt16Size = mid * RoundUpArch22((d_ - SLICE_SIZE) * X_ELEMENT_BYTES, BLOCK_SIZE);
        int64_t scaleSize = mid * 128 * 8 * TMP_BUFFER_ELEMENT_BYTES;
        int64_t ySize = mid * RoundUpArch22(kvCacheCol_, BLOCK_SIZE) * DOUBLE_BUFFER;
        int64_t totalSize = xSize + f32WorkSize + qInt16Size + scaleSize + ySize + REDUCE_TMP_BUFFER_BYTES;
        if (totalSize <= static_cast<int64_t>(ubSize_)) {
            rowFactor_ = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    rowLoopOfFormerBlock_ = CeilDivArch22(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDivArch22(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_kvCacheCol(kvCacheCol_);
    tilingData_.set_kvCacheRowStride(kvCacheRowStride_);
    tilingData_.set_kvCacheBlockSize(kvCacheBlockSize_);
    tilingData_.set_kvCacheBlockStride(kvCacheBlockStride_);
    tilingData_.set_xRowStride(d_); // x 为 2D 连续 [bs, 512], 行 stride 恒为 d
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

ge::graphStatus KvCompressEpilogTilingArch22::PostTiling()
{
    // Set block dimension (number of AI cores to use)
    context_->SetBlockDim(usedCoreNums_);

    // Set tiling key for kernel dispatch
    context_->SetTilingKey(GetTilingKey());

    // Set workspace size
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

void KvCompressEpilogTilingArch22::DumpTilingInfo()
{
    std::ostringstream info;
    info << "bs: " << tilingData_.get_bs();
    info << ", d: " << tilingData_.get_d();
    info << ", kvCacheCol: " << tilingData_.get_kvCacheCol();
    info << ", kvCacheRowStride: " << tilingData_.get_kvCacheRowStride();
    info << ", kvCacheBlockSize: " << tilingData_.get_kvCacheBlockSize();
    info << ", kvCacheBlockStride: " << tilingData_.get_kvCacheBlockStride();
    info << ", xRowStride: " << tilingData_.get_xRowStride();
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

uint64_t KvCompressEpilogTilingArch22::GetTilingKey() const
{
    // 单一 tiling key 0 (单模板); slot_mapping int32/int64 差异由 kernel 模板类型参数承载
    return 0UL;
}

ge::graphStatus KvCompressEpilogTilingArch22::GetShapeAttrsInfo()
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

ge::graphStatus KvCompressEpilogTilingArch22::RunTiling()
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
