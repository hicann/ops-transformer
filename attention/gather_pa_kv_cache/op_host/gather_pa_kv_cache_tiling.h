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
 * \file gather_pa_kv_cache_tiling.h
 * \brief
 */

#ifndef GATHER_PA_KV_CACHE_TILING_H
#define GATHER_PA_KV_CACHE_TILING_H

#include <string>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherPaKvCacheTilingData)
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, numTokens);
TILING_DATA_FIELD_DEF(int32_t, numblkTabCol);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeK);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeV);
TILING_DATA_FIELD_DEF(int32_t, typeByte);
TILING_DATA_FIELD_DEF(int32_t, hasSeqStarts);
TILING_DATA_FIELD_DEF(int32_t, isSeqLensCumsum);
TILING_DATA_FIELD_DEF(int64_t, kCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, vCacheBlockStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GatherPaKvCache, GatherPaKvCacheTilingData);

struct Tiling4GatherPaKvCacheCompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
    uint32_t sysWorkspaceSize;
};

ge::graphStatus Tiling4GatherPaKvCache(gert::TilingContext *context);
ge::graphStatus TilingPrepare4GatherPaKvCache(gert::TilingParseContext *context);

BEGIN_TILING_DATA_DEF(GatherPaKvCacheTilingDataV35)
TILING_DATA_FIELD_DEF(uint32_t, seqLenAccumSize);
TILING_DATA_FIELD_DEF(int64_t, batchCount);
TILING_DATA_FIELD_DEF(int64_t, batchPerCore);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, blockTableWidth);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSizeK);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSizeV);
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSizeK);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSizeV);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSize);
// ===== 非连续支持字段 =====
// 0=全连续(走原有路径), 非0=有非连续tensor
// bit0: keyCache非连续, bit1: valueCache非连续
// bit2: key输出非连续, bit3: value输出非连续
// bit4: kCache blockSize轴非连续, bit5: kCache N轴非连续
// bit6: vCache blockSize轴非连续, bit7: vCache N轴非连续
// bit8: keyOut head(dim1)轴非连续, bit9: valueOut head(dim1)轴非连续
TILING_DATA_FIELD_DEF(uint32_t, nonContiguousFlag);
TILING_DATA_FIELD_DEF(int64_t, kCacheStride0);   // blockNum轴stride (元素粒度)
TILING_DATA_FIELD_DEF(int64_t, kCacheStride1);   // blockSize轴stride
TILING_DATA_FIELD_DEF(int64_t, kCacheStride2);   // N轴stride
TILING_DATA_FIELD_DEF(int64_t, vCacheStride0);   // blockNum轴stride
TILING_DATA_FIELD_DEF(int64_t, vCacheStride1);   // blockSize轴stride
TILING_DATA_FIELD_DEF(int64_t, vCacheStride2);   // N轴stride
TILING_DATA_FIELD_DEF(int64_t, keyOutStride0);   // B轴stride
TILING_DATA_FIELD_DEF(int64_t, keyOutStride1);   // S轴stride
TILING_DATA_FIELD_DEF(int64_t, valueOutStride0); // B轴stride
TILING_DATA_FIELD_DEF(int64_t, valueOutStride1); // S轴stride
TILING_DATA_FIELD_DEF(uint64_t, numHeadsK);      // key的head数量
TILING_DATA_FIELD_DEF(uint64_t, numHeadsV);      // value的head数量
END_TILING_DATA_DEF;

constexpr uint64_t TILING_KEY_1111 = 1111;
constexpr uint64_t TILING_KEY_1110 = 1110;
constexpr uint64_t TILING_KEY_1101 = 1101;
constexpr uint64_t TILING_KEY_1100 = 1100;
constexpr uint64_t TILING_KEY_1011 = 1011;
constexpr uint64_t TILING_KEY_1010 = 1010;
constexpr uint64_t TILING_KEY_1001 = 1001;
constexpr uint64_t TILING_KEY_1000 = 1000;

struct GatherPaKvCacheTilingKeyMap {
    uint64_t tilingKey;
    bool isCacheModeNorm;
    bool isSeqLenCumSum;
    bool hasSeqOffset;
};

constexpr std::array<GatherPaKvCacheTilingKeyMap, 8> tilingKeyTable = {{
    // tilingKey, isCacheModeNorm, isSeqLenCumSum, hasSeqOffset
    // ND
    {TILING_KEY_1111, true, true, true},
    {TILING_KEY_1110, true, true, false},
    {TILING_KEY_1101, true, false, true},
    {TILING_KEY_1100, true, false, false},
    // NZ
    {TILING_KEY_1011, false, true, true},
    {TILING_KEY_1010, false, true, false},
    {TILING_KEY_1001, false, false, true},
    {TILING_KEY_1000, false, false, false},
}};

REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1111, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1110, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1101, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1100, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1011, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1010, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1001, GatherPaKvCacheTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCache_1000, GatherPaKvCacheTilingDataV35);

struct GatherPaKvCacheCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class GatherPaKvCacheTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GatherPaKvCacheTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~GatherPaKvCacheTiling() override {}
    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t needCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    int64_t batchCount_ = 0;
    uint32_t keyByteSize_ = 0;
    uint32_t valueByteSize_ = 0;
    uint32_t indexByteSize_ = 0;
    int64_t blockTableWidth_ = 0;
    int64_t blockSize_ = 0;
    uint64_t hiddenSizeK_ = 0;
    uint64_t hiddenSizeV_ = 0;
    int64_t numBlocks_ = 0;
    int64_t numTokens_ = 0;
    size_t kCacheDimNum_ = 0;
    size_t vCacheDimNum_ = 0;
    gert::Shape kCacheShape_;
    gert::Shape vCacheShape_;
    gert::Shape blockTableShape_;
    gert::Shape seqLenShape_;
    gert::Shape keyShape_;
    gert::Shape valueShape_;
    gert::Shape seqOffsetShape_;
    std::string cacheMode_;
    bool isCacheModeNorm_;
    bool isSeqLenCumSum_;
    bool hasSeqOffset_;

private:
    ge::graphStatus GetAttrs();
    ge::graphStatus GetInputKeyCache();
    ge::graphStatus GetInputValueCache();
    ge::graphStatus GetInputBlockTables();
    ge::graphStatus GetInputSeqLens();
    ge::graphStatus GetInputOutputKey();
    ge::graphStatus GetInputOutputValue();
    ge::graphStatus GetInputSeqOffset();

    uint64_t tilingKey_ = 0;
    GatherPaKvCacheTilingDataV35 tilingData_;

    // ===== 非连续支持 =====
    gert::Stride kCacheStride_;
    gert::Stride vCacheStride_;
    gert::Stride keyOutStride_;
    gert::Stride valueOutStride_;

    ge::graphStatus GetContiguousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx);
    ge::graphStatus GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx);

    bool isKCacheContiguous_ = true;
    bool isVCacheContiguous_ = true;
    bool isKeyOutContiguous_ = true;
    bool isValueOutContiguous_ = true;
    bool isKeyOutHeadNonContig_ = false;   // keyOut head(dim1)轴非连续 (ND, ref dim1散写)
    bool isValueOutHeadNonContig_ = false; // valueOut head(dim1)轴非连续

    bool isKCacheSlotNonContig_ = false; // blockSize轴非连续
    bool isKCacheHeadNonContig_ = false; // N轴非连续
    bool isVCacheSlotNonContig_ = false;
    bool isVCacheHeadNonContig_ = false;

    uint64_t numHeadsK_ = 0;
    uint64_t numHeadsV_ = 0;
};
} // namespace optiling
#endif // GATHER_PA_KV_CACHE_TILING_H
