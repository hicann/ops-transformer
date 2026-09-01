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
 * \file indexer_quant_cache_tiling_arch35.cpp
 * \brief
 */

#include <sstream>
#include "log/log.h"
#include "err/ops_err.h"
#include "indexer_quant_cache_tiling_arch35.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t RoundUp(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

constexpr int64_t INPUT_CACHE_IDX = 0;
constexpr int64_t INPUT_SCALE_IDX = 1;
constexpr int64_t INPUT_X_IDX = 2;
constexpr int64_t INPUT_SLOT_MAPPING_IDX = 3;
// 4D-only cache/scale contract: cache 与 cache_scale 的逻辑 shape 必须恰为
// 4D [blockNum, blockSize, 1, headDim]:
//   - dim0 blockNum: 仅此维支持非连续(分页); dim1 blockSize: 每 block 的 token 数;
//   - dim2 固定为 1 (每 token 写出一个量化向量, 该轴不支持为变量);
//   - dim3 headDim: 每 token 行宽(行步长), cache 须 >= 写出的量化-x 长度, scale 须 >= scaleCol。
constexpr size_t CACHE_VIEW_DIM_NUM = 4;
constexpr size_t CACHE_BLOCKNUM_DIM = 0;
constexpr size_t CACHE_BLOCKSIZE_DIM = 1;
constexpr size_t CACHE_ONE_DIM = 2; // 倒数第二维, 必须 == 1
constexpr int64_t CACHE_ONE_DIM_VALUE = 1;
constexpr int64_t ATTR_QUANT_MODE_INDEX = 0;
constexpr int64_t ATTR_ROUND_SCALE_INDEX = 1;
constexpr int64_t ATTR_SCALE_INDEX = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t D_LENGTH_FULL_LOAD = 8192;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t FP4_PACK_NUM = 2; // MX-FP4: 2 fp4 values packed per byte
// per_block量化,每128个f16需要量化出一个scale, 因此切分尾轴时，以128为factor进行切分
constexpr int64_t PER_BLOCK_FP16 = 128;
// MX-FP4 (quant_mode=3) 采用标准 MX 量化块, 每32个元素一个 e8m0 scale
constexpr int64_t PER_BLOCK_MXFP4 = 32;
constexpr int64_t MXFP8_QUANT_MODE = 0;
constexpr int64_t NORMAL_QUANT_MODE = 1;
constexpr int64_t HIFLOAT_QUANT_MODE = 2;
constexpr int64_t MXFP4_QUANT_MODE = 3;
constexpr int64_t SINGLE_ROW = 1;
} // namespace

ge::graphStatus IndexerQuantCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<IndexerQuantCacheCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::GetAttr()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto quantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    quantMode_ = quantMode == nullptr ? 1 : *quantMode;

    auto roundScale = attrs->GetAttrPointer<bool>(ATTR_ROUND_SCALE_INDEX);
    roundScale_ = roundScale == nullptr ? true : *roundScale;

    auto scaleAttr = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    scalesAttr_ = scaleAttr == nullptr ? 1.0f : *scaleAttr;

    return ge::GRAPH_SUCCESS;
}

bool IndexerQuantCacheTiling::GetCacheViewLayout(
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

ge::graphStatus IndexerQuantCacheTiling::ValidateCache4D(size_t inputIdx, const char *inputName, int64_t &lastDim)
{
    // 4D-only 契约: 用逻辑(origin/view) shape 做维数门禁。连续 4D tensor 与 4D 分页 strided view
    // 的 origin shape 均为该 4D 逻辑形状(GetShape()), 与 GetCacheViewLayout 经 inputShape->GetShape()
    // 读取的 viewShape 同源; strided view 的 storage shape 可能是底层扁平 buffer。
    auto shapePtr = context_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
    const auto &logical = shapePtr->GetShape();
    OP_CHECK_IF(logical.GetDimNum() != CACHE_VIEW_DIM_NUM,
                OP_LOGE(context_->GetNodeName(),
                        "%s must be 4D [blockNum, blockSize, 1, headDim], got dimNum=%zu",
                        inputName, logical.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(logical.GetDim(CACHE_ONE_DIM) != CACHE_ONE_DIM_VALUE,
                OP_LOGE(context_->GetNodeName(),
                        "%s dim2 (second-to-last) must be 1 (one quantized vector per token), got %ld",
                        inputName, logical.GetDim(CACHE_ONE_DIM)),
                return ge::GRAPH_FAILED);
    // 连续场景的行宽(headDim)取末维; 分页 view 下随后由 view stride 覆盖, 此处仅作连续默认值。
    const auto &storage = shapePtr->GetStorageShape();
    lastDim = storage.GetDim(storage.GetDimNum() - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::GetShapeAttrsInfoInner()
{
    auto shapeX = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    auto shapeSlotMapping = context_->GetInputShape(INPUT_SLOT_MAPPING_IDX);
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
                    OP_LOGE(context_->GetNodeName(), "slotMappingShape should equal xStorageShape in dim %u", i),
                    return ge::GRAPH_FAILED);
        bs *= temp;
    }
    d_ = xStorageShape.GetDim(xDims - 1);
    bs_ = bs;

    // x 尾轴 d 必须 32 对齐: MX 模式每 32 个元素一个 scale, 且各量化分支按 32 元素块对齐处理,
    // 非 32 对齐会导致尾块读越界/scale 错位。
    OP_CHECK_IF(d_ <= 0 || d_ % BLOCK_SIZE != 0,
                OP_LOGE(context_->GetNodeName(),
                        "the last dim (d) of x should be 32-aligned, got %ld", d_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(d_ > D_LENGTH_FULL_LOAD,
                OP_LOGE(context_->GetNodeName(), "input x tail dimension must be less than 8192, got %ld", d_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode_ < 0 || quantMode_ > MXFP4_QUANT_MODE,
                OP_LOGE(context_->GetNodeName(), "quant_mode should be in [0,3], got %ld", quantMode_),
                return ge::GRAPH_FAILED);

    // 每行 scale 个数:
    //   mode0 MX-FP8 / mode3 MX-FP4 : 每 32 个元素一个 scale (标准 MX 块)
    //   mode1 Normal (整行量化) / mode2 HiFloat8 : 整行一个 scale
    if (quantMode_ == MXFP8_QUANT_MODE || quantMode_ == MXFP4_QUANT_MODE) {
        scaleCol_ = CeilDiv(d_, PER_BLOCK_MXFP4);
    } else {
        scaleCol_ = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::CalcOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    rowFactor_ = rowOnceLoop;
    int64_t scaleByteSize = 4;
    if (quantMode_ == MXFP8_QUANT_MODE || quantMode_ == MXFP4_QUANT_MODE) {
        scaleByteSize = 1; // MX 模式 scale 为 e8m0, 占1字节
    }
    int64_t perBlockScaleElemNum = BLOCK_SIZE / scaleByteSize;
    int64_t xAlign = (quantMode_ == MXFP8_QUANT_MODE) ? PER_BLOCK_FP16 : 16;
    int64_t lo = rowOnceLoop;
    int64_t hi = rowOfFormerBlock_;
    rowFactor_ = lo - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int64_t xSize = mid * RoundUp(d_, xAlign) * 2 * DOUBLE_BUFFER;
        int64_t ySize = mid * RoundUp(d_, BLOCK_SIZE) * 1 * DOUBLE_BUFFER;
        int64_t scaleSize = mid * RoundUp(scaleCol_, perBlockScaleElemNum) * scaleByteSize * DOUBLE_BUFFER;
        int64_t tmpBufferSize = RoundUp(mid, 8) * 4;
        int64_t mxScratchSize = 0;
        if (quantMode_ == MXFP8_QUANT_MODE) {
            mxScratchSize = mid * RoundUp(d_, 8) * 4 + mid * CeilDiv(d_, 128) * 16 * 4;
        }
        int64_t totalSize = xSize + ySize + scaleSize + tmpBufferSize + mxScratchSize;
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

    // 4D paged 输出布局推导 [blockNum, blockSize, 1, headDim], 仅 blockNum 非连续。
    // 与 ScatterPaKvCache 一致: 完全从 cache/scale 张量的 RUNTIME VIEW strides 推导布局
    // (aclnn 层对两个 cache 做 CreateView, 使 GE tiling context 暴露 view shape/strides),
    // 不再使用 block_size / *_block_stride 属性。无 view 时按连续 flat slots 处理
    // (blockSize=1 => slot * rowStride)。MX-FP4 的 cache 末维为打包字节 (d/2)。
    //
    // 每行写出长度(kernel 寻址单位): cache 写 cacheCol; scale 写 scaleCol。
    //   - cacheCol: MX-FP4 为打包字节 (d/2), 其余模式为 d (fp8/uint8 元素==字节)。
    //   - scaleCol: MX-FP8/MX-FP4 为 ⌈d/32⌉ 个 e8m0; Normal/HiFloat8 为 1 个 float。
    int64_t cacheCol = (quantMode_ == MXFP4_QUANT_MODE) ? (d_ + 1) / 2 : d_;

    // === Reading B + 4D-only 契约: cache 行宽(headDim)由 cache 自身决定 ===
    // cache 始终校验。末维 lastDim 为 cache 自身元素单位 (MX-FP4: fp4 元素; 其余: 字节)。
    int64_t cacheLastDim = 0;
    if (ValidateCache4D(INPUT_CACHE_IDX, "cache", cacheLastDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // MX-FP4 每字节打包 2 个 fp4 元素, headDim 为奇数时行首会落在字节的高/低半字节,
    // 现有 kernel 仅支持整字节寻址(GlobalTensor<int8_t> + 整字节 CopyOut), 无法表达半字节偏移,
    // 因此在此处直接拒绝, 而非产出错误数据(与 ScatterPaKvCache 对 fp4 head_size 的校验一致)。
    if (CheckMxfp4EvenStride(cacheLastDim, "headDim") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 转为 kernel 寻址单位: MX-FP4 字节寻址打包 cache (÷2); 其余模式元素==字节, 直接取末维。
    cacheRowStride_ = (quantMode_ == MXFP4_QUANT_MODE) ? (cacheLastDim / FP4_PACK_NUM) : cacheLastDim;

    // cache_scale 校验: 所有量化模式(0/1/2/3)均完整校验, 无例外。kernel 在 mode2(HiFloat8) 下
    // 仍会为每个 token 无条件散写 1 个 float scale 到 cacheScaleGm(scaleCol=1), 故 cache_scale
    // 必须是合法 4D 张量 [blockNum, blockSize, 1, headDim] 且行宽 >= scaleCol, 否则散写越界。
    // scale 末维即 kernel 寻址单位(e8m0 1B 或 float)。
    int64_t scaleLastDim = 0;
    if (ValidateCache4D(INPUT_SCALE_IDX, "cache_scale", scaleLastDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    scaleRowStride_ = scaleLastDim;
    blockSize_ = 1; // default: contiguous flat slots (slot * rowStride)
    cacheBlockStride_ = cacheRowStride_;
    scaleBlockStride_ = scaleRowStride_;

    int64_t cacheViewBlockSize = 0;
    int64_t cacheViewRowStride = 0;
    int64_t cacheViewBlockStride = 0;
    int64_t scaleViewBlockSize = 0;
    int64_t scaleViewRowStride = 0;
    int64_t scaleViewBlockStride = 0;
    bool cacheHasView =
        GetCacheViewLayout(INPUT_CACHE_IDX, cacheViewBlockSize, cacheViewRowStride, cacheViewBlockStride);
    bool scaleHasView =
        GetCacheViewLayout(INPUT_SCALE_IDX, scaleViewBlockSize, scaleViewRowStride, scaleViewBlockStride);
    if (cacheHasView && scaleHasView) {
        blockSize_ = cacheViewBlockSize;
        cacheRowStride_ = cacheViewRowStride;
        cacheBlockStride_ = cacheViewBlockStride;
        scaleRowStride_ = scaleViewRowStride;
        scaleBlockStride_ = scaleViewBlockStride;
        // MX-FP4 cache is a packed fp4 type (DT_FLOAT4_E2M1/E1M2, 2 values/byte). GetInputStride
        // returns the cache view strides in fp4 ELEMENTS, but the kernel byte-addresses the packed
        // cache (int8 GM), so convert the cache strides to BYTES (÷2). scale (e8m0, 1 byte) is
        // already byte-equal so it is left unchanged.
        if (quantMode_ == MXFP4_QUANT_MODE) {
            // 分页 view 下同样以整字节寻址, row/block stride(fp4 元素)必须为偶数, 否则半字节偏移
            // 无法表达(与上面连续场景 cacheLastDim 的校验同理)。
            if (CheckMxfp4EvenStride(cacheRowStride_, "paged-view row stride") != ge::GRAPH_SUCCESS ||
                CheckMxfp4EvenStride(cacheBlockStride_, "paged-view block stride") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            cacheRowStride_ = cacheRowStride_ / FP4_PACK_NUM;
            cacheBlockStride_ = cacheBlockStride_ / FP4_PACK_NUM;
        }
    }

    // === Reading B 长度校验(以最终行步长 = 行宽 判定, kernel 寻址单位) ===
    // cache 行宽必须能容纳每行写出的 cacheCol, 否则散写越界到下一行。
    // (cacheCol 与 cacheRowStride_ 单位一致: MX-FP4 皆为字节, 其余皆为元素/字节)
    OP_CHECK_IF(cacheRowStride_ < cacheCol,
                OP_LOGE(context_->GetNodeName(),
                        "cache headDim(row stride, in cache elements/bytes)=%ld must be >= per-token "
                        "quantized-x length=%ld",
                        cacheRowStride_, cacheCol),
                return ge::GRAPH_FAILED);
    // cache_scale 行宽必须能容纳每行写出的 scaleCol(mode2: scaleCol=1), 否则散写越界到下一行。
    OP_CHECK_IF(scaleRowStride_ < scaleCol_,
                OP_LOGE(context_->GetNodeName(),
                        "cache_scale last dim(row stride)=%ld must be >= scaleCol=%ld",
                        scaleRowStride_, scaleCol_),
                return ge::GRAPH_FAILED);

    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_scalesAttr(scalesAttr_);
    int64_t roundScaleData = roundScale_ ? 1 : 0;
    tilingData_.set_roundScale(roundScaleData);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_cacheRowStride(cacheRowStride_);
    tilingData_.set_cacheBlockStride(cacheBlockStride_);
    tilingData_.set_scaleRowStride(scaleRowStride_);
    tilingData_.set_scaleBlockStride(scaleBlockStride_);

    // SINGLE_ROW_MXFP8_QUANT  TILING_KEY : 10000
    // SINGLE_ROW_NORMAL_QUANT TILING_KEY : 10001
    // SINGLE_ROW_MXFP4_QUANT  TILING_KEY : 10002
    // MULTI_ROW_MXFP8_QUANT   TILING_KEY : 10010
    // MULTI_ROW_NORMAL_QUANT  TILING_KEY : 10011
    // MULTI_ROW_MXFP4_QUANT   TILING_KEY : 10012
    tilingKey_ = 10000;
    if (rowFactor_ != SINGLE_ROW) {
        tilingKey_ += 10;
    }
    if (quantMode_ == NORMAL_QUANT_MODE || quantMode_ == HIFLOAT_QUANT_MODE) {
        tilingKey_ += 1;
    } else if (quantMode_ == MXFP4_QUANT_MODE) {
        tilingKey_ += 2;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    context_->SetTilingKey(tilingKey_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerQuantCacheTiling::CheckMxfp4EvenStride(int64_t stride, const char *name)
{
    if (quantMode_ == MXFP4_QUANT_MODE && stride % FP4_PACK_NUM != 0) {
        const std::string strideStr = std::to_string(stride);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), name, strideStr.c_str(),
                                                 "MX-FP4 cache stride in fp4 elements must be an even number.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForIndexerQuantCache(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForIndexerQuantCache(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IndexerQuantCache", "Tiling context is null"),
                return ge::GRAPH_FAILED);
    IndexerQuantCacheTiling IndexerQuantCacheTiling(context);
    return IndexerQuantCacheTiling.DoOpTiling();
}

IMPL_OP_OPTILING(IndexerQuantCache)
    .Tiling(TilingForIndexerQuantCache)
    .TilingParse<IndexerQuantCacheCompileInfo>(TilingPrepareForIndexerQuantCache);

} // namespace optiling
