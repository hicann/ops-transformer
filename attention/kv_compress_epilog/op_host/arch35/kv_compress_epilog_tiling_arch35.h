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
 * \brief Architecture-specific tiling for KvCompressEpilog quant_mode=0/1/2 (arch35: ascend950 / ascend960dt /
 * ascend350)。本文件与 arch22 实现相互独立，仅依赖架构无关的公共头 kv_compress_epilog_tiling.h。
 */

#ifndef KV_COMPRESS_EPILOG_TILING_ARCH35_H
#define KV_COMPRESS_EPILOG_TILING_ARCH35_H

#include "../kv_compress_epilog_tiling.h"

namespace optiling {

// ---------- arch35 (quant_mode=0/1/2) Constants ----------
constexpr int64_t QUANT_MODE_GROUP_QUANT_BF16 = 0;
constexpr int64_t QUANT_MODE_GROUP_QUANT_E8M0 = 1;
// quant_mode=2: rope 段 hifloat8 静态量化(x_scale) + nope 段 per-group FLOAT4_E2M1 动态量化(bf16 scale)
constexpr int64_t QUANT_MODE_HIF8_FP4 = 2;

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
constexpr int64_t D_LENGTH_FULL_LOAD = 8192;

constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t TMP_BUFFER_ALIGN = 8;
// per_block量化,每128个f16需要量化出一个scale, 因此切分尾轴时，以128为factor进行切分

// ---------- arch35 Tiling Class ----------
class KvCompressEpilogTilingArch35 {
public:
    explicit KvCompressEpilogTilingArch35(gert::TilingContext* context)
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
    bool GetCacheViewLayout(size_t inputIdx, int64_t& blockSize, int64_t& rowStride, int64_t& blockStride);
    void CountTilingKey();

    // Validation
    ge::graphStatus ValidateShapes();
    ge::graphStatus ValidateDtypes();

    // mode 3/4 (MXFP8/MXFP4) 路径
    bool IsMxfpMode() const;
    ge::graphStatus GetMxfpCacheLayout();
    ge::graphStatus DoMxfpTiling();
    uint64_t GetMxfpTilingKey() const;

    // Utilities
    uint64_t GetTilingKey() const;
    void DumpTilingInfo();

private:
    // Context
    gert::TilingContext* context_ = nullptr;
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

    // mode 3/4 (MXFP8/MXFP4) 路径专用
    int64_t layout_ = 0;                 // 0=legacy(不写) 1=layout1(2D) 2=layout2(4D)
    int64_t cacheRows_ = 0;              // cache 逻辑总行数（token 槽位）
    int64_t dataCol_ = 0;                // 量化数据字节数（fp8=d / fp4=d/2）
    int64_t scaleBytes_ = 0;             // scale 总字节数 = scaleCol * 2
    int64_t layout2TokenBundleSize_ = 0; // layout2 单 token bundle 大小 = dataCol + scaleBytes
    int64_t layout2CacheCol_ = 0;        // layout2 cache dim3 物理列宽

    // Attributes
    int64_t quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    int64_t quantMode_ = 1;
    int64_t roundScale_ = 1;
    float scalesAttr_ = 0;

    // Data types
    ge::DataType xDtype_ = ge::DT_BF16;
    ge::DataType slotMappingDtype_ = ge::DT_INT32;
    ge::DataType kvCacheDtype_ = ge::DT_FLOAT8_E5M2;
    ge::DataType cacheDtype_ = ge::DT_UINT8;
};

} // namespace optiling

#endif // KV_COMPRESS_EPILOG_TILING_ARCH35_H
