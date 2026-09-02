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
 * \file indexer_quant_cache_tiling_arch35.h
 * \brief
 */

#ifndef INDEXER_QUANT_CACHE_TILING_ARCH35_H
#define INDEXER_QUANT_CACHE_TILING_ARCH35_H

#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

namespace optiling {
// ----------公共定义----------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// ----------算子TilingData定义----------
BEGIN_TILING_DATA_DEF(IndexerQuantCacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);                   // 一行多少个scale
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);           // 头核共需要处理多少行
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);             // 尾核共需要处理多少行
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);       // 头核需要几次ub搬入
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);         // 尾核需要几次ub搬入
TILING_DATA_FIELD_DEF(int64_t, rowFactor);                  // ub一次标准处理行数
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock); // 头核最后一次ub处理行数
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);   // 尾核最后一次ub处理行数
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(float, scalesAttr);
// 4D paged layout [blockNum, blockSize, 1, headDim]; blockNum non-contiguous.
TILING_DATA_FIELD_DEF(int64_t, blockSize);        // blockSize dim (1 => contiguous flat slots)
TILING_DATA_FIELD_DEF(int64_t, cacheRowStride);   // per-position stride of cache (= headDim elems)
TILING_DATA_FIELD_DEF(int64_t, cacheBlockStride); // per-block (blockNum) stride of cache
TILING_DATA_FIELD_DEF(int64_t, scaleRowStride);   // per-position stride of scale (= scaleCol)
TILING_DATA_FIELD_DEF(int64_t, scaleBlockStride); // per-block (blockNum) stride of scale
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexerQuantCache, IndexerQuantCacheTilingData)

// ----------算子CompileInfo定义----------
struct IndexerQuantCacheCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class IndexerQuantCacheTiling {
public:
    explicit IndexerQuantCacheTiling(gert::TilingContext *tilingContext)
        : context_(tilingContext)
    {}
    ~IndexerQuantCacheTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    bool GetCacheViewLayout(size_t inputIdx, int64_t &blockSize, int64_t &rowStride, int64_t &blockStride);
    // 4D-only 契约门禁: 校验 inputIdx 张量逻辑 shape 恰为 4D [blockNum, blockSize, 1, headDim]
    // (倒数第二维 == 1), 并回传其末维(headDim, 张量自身元素单位)。失败返回 GRAPH_FAILED。
    ge::graphStatus ValidateCache4D(size_t inputIdx, const char *name, int64_t &lastDim);
    // MX-FP4 模式下校验 stride(fp4元素单位) 必须为偶数, 否则半字节偏移无法表达
    ge::graphStatus CheckMxfp4EvenStride(int64_t stride, const char *name);

private:
    gert::TilingContext *context_ = nullptr;
    IndexerQuantCacheTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t scaleCol_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t quantMode_ = 1;
    float scalesAttr_ = 0;
    bool roundScale_ = true;
    int64_t blockSize_ = 1;
    int64_t cacheRowStride_ = 0;
    int64_t cacheBlockStride_ = 0;
    int64_t scaleRowStride_ = 0;
    int64_t scaleBlockStride_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND950;
    int64_t tilingKey_ = 0;
};

} // namespace optiling
#endif // INDEXER_QUANT_CACHE_TILING_ARCH35_H
