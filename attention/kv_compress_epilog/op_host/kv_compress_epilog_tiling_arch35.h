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
 * \file kv_compress_epilog_tiling_arch35.h
 * \brief
 */

#ifndef KV_COMPRESS_EPILOG_TILING_ARCH35_H
#define KV_COMPRESS_EPILOG_TILING_ARCH35_H

#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {

// ---------- Constants ----------
constexpr int64_t KV_COMPRESS_CACHE_INPUT_INDEX = 0;
constexpr int64_t X_INPUT_INDEX = 1;
constexpr int64_t SLOT_MAPPING_INDEX = 2;
constexpr int64_t KV_COMPRESS_CACHE_OUTPUT_INDEX = 0;

constexpr int64_t QUANT_GROUP_SIZE_ATTR_INDEX = 0;
constexpr int64_t QUANT_MODE_ATTR_INDEX = 1;
constexpr int64_t ROUND_SCALE_ATTR_INDEX = 2;
constexpr int64_t SCALE_ATTR_INDEX = 3;

constexpr int64_t QUANT_MODE_GROUP_QUANT_BF16 = 0;
constexpr int64_t QUANT_MODE_GROUP_QUANT_E8M0 = 1;
// quant_mode=2: rope 段 hifloat8 静态量化(x_scale) + nope 段 per-group FLOAT4_E2M1 动态量化(bf16 scale)
constexpr int64_t QUANT_MODE_HIF8_FP4 = 2;

constexpr int64_t DEFAULT_QUANT_GROUP_SIZE = 64;
constexpr int64_t KV_CACHE_ROW_ALIGN = 128;
// mode2: rope hifloat8 输出字节数(64 个元素 x 1B)
constexpr int64_t ROPE_HIF8_BYTES = 64;
// mode2: nope FLOAT4_E2M1 每个 scale 以 bf16(2B) 写出
constexpr int64_t FP4_SCALE_BYTES = 2;
// mode2 三套特化 VF: 每 64-元素块在行内 scratch 中占用的 bf16 槽位数 (32B 对齐, 前 4 槽有效 = 4 个 16-lane datablock)
constexpr int64_t FP4_SCRATCH_SLOTS_PER_CHUNK = 16;
// mode2 仅支持的三种 per-group 量化粒度
constexpr int64_t FP4_GROUP_SIZE_16 = 16;
constexpr int64_t FP4_GROUP_SIZE_32 = 32;
constexpr int64_t FP4_GROUP_SIZE_64 = 64;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t D_LENGTH_FULL_LOAD = 8192;

constexpr int64_t SLICE_SIZE = 64;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t MIN_ROW_PER_CORE = 1;
constexpr int64_t X_BUFFER_ALIGN = 16;
constexpr int64_t X_ELEMENT_BYTES = 2;
constexpr int64_t TMP_BUFFER_ALIGN = 8;
constexpr int64_t TMP_BUFFER_ELEMENT_BYTES = 4;
// 4D-only cache contract: cache 逻辑 shape 必须恰为 4D [blockNum, blockSize, 1, headDim]。
//   - dim0 blockNum: 仅此维支持非连续(分页); dim1 blockSize: 每 block 的 token 数;
//   - dim2 固定为 1 (每 token 写出一个压缩向量, 该轴不支持为变量);
//   - dim3 headDim: 每 token 行宽(行步长), 须 >= 算子写出的 kvCacheCol。
constexpr size_t CACHE_VIEW_DIM_NUM = 4;
constexpr size_t CACHE_BLOCKNUM_DIM = 0;
constexpr size_t CACHE_BLOCKSIZE_DIM = 1;
constexpr size_t CACHE_ONE_DIM = 2; // 倒数第二维, 必须 == 1
constexpr int64_t CACHE_ONE_DIM_VALUE = 1;
// per_block量化,每128个f16需要量化出一个scale, 因此切分尾轴时，以128为factor进行切分

// ---------- TilingData Structure ----------
BEGIN_TILING_DATA_DEF(KvCompressEpilogTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, kvCacheCol);
TILING_DATA_FIELD_DEF(int64_t, kvCacheRowStride);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, concatCol);
TILING_DATA_FIELD_DEF(int64_t, padCol);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(int64_t, perGroupSize);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(float, scalesAttr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCompressEpilog, KvCompressEpilogTilingData)

// ---------- CompileInfo Structure ----------
struct KvCompressEpilogCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ---------- Tiling Class ----------
class KvCompressEpilogTilingArch35 {
public:
    explicit KvCompressEpilogTilingArch35(gert::TilingContext *context)
        : context_(context)
    {}
    ~KvCompressEpilogTilingArch35() = default;

    ge::graphStatus RunTiling();

protected:
    // Main tiling workflow methods
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

    // Helper methods
    ge::graphStatus GetInputShapes();
    ge::graphStatus GetAttributes();
    ge::graphStatus GetDtypeInfo();
    ge::graphStatus GetKvCacheLayout();
    bool GetCacheViewLayout(size_t inputIdx, int64_t &blockSize, int64_t &rowStride, int64_t &blockStride);
    void CountTilingKey();

    // Validation
    ge::graphStatus ValidateShapes();
    ge::graphStatus ValidateDtypes();

    // Utilities
    uint64_t GetTilingKey() const;
    void DumpTilingInfo();

private:
    // Context
    gert::TilingContext *context_ = nullptr;
    KvCompressEpilogTilingData tilingData_;
    uint64_t tilingKey_ = 0;

    // Platform info
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;

    // Shape info from inputs
    int64_t bs_ = 0;                 // First dimension of x (to be partitioned)
    int64_t d_ = 0;                  // Second dimension of x
    int64_t kvCacheCol_ = 0;         // kvCache cols (bytes written per token row == headDim)
    int64_t kvCacheRowStride_ = 0;   // within-block per-token row stride (physical headDim)
    int64_t kvCacheBlockSize_ = 1;   // tokens per block (logical); 1 for the 2D layout
    int64_t kvCacheBlockStride_ = 0; // physical stride between consecutive blocks (non-contiguous blockNum)
    int64_t scaleCol_ = 0;           // scale cols

    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;

    // Attributes
    int64_t quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    int64_t quantMode_ = 1;
    int64_t roundScale_ = 1;
    float scalesAttr_ = 0;

    // Data types
    ge::DataType xDtype_ = ge::DT_BF16;
    ge::DataType slotMappingDtype_ = ge::DT_INT32;
    ge::DataType kvCacheDtype_ = ge::DT_FLOAT8_E5M2;
};

} // namespace optiling

#endif // KV_COMPRESS_EPILOG_TILING_ARCH35_H
