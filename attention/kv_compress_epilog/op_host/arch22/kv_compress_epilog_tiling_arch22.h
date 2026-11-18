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
 * \file kv_compress_epilog_tiling_arch22.h
 * \brief Architecture-specific tiling for KvCompressEpilog quant_mode=1 (arch22, DAV_2201: A2 ascend910b / A3
 * ascend910_93)。本文件与 arch35 实现相互独立，仅依赖架构无关的公共头 kv_compress_epilog_tiling.h；
 * 删除 arch35 目录不影响本文件及 arch22 quant_mode=1 的编译与运行。
 */

#ifndef KV_COMPRESS_EPILOG_TILING_ARCH22_H
#define KV_COMPRESS_EPILOG_TILING_ARCH22_H

#include "../kv_compress_epilog_tiling.h"

namespace optiling {

// ---------- arch22 (quant_mode=1, int8) Constants ----------
// quant_mode=1: rope 段 bf16 位型直通(64 元素 x 2B = 128B) + nope 段 per-group int8 对称动态量化(scale bf16)
constexpr int64_t QUANT_MODE_INT8 = 1;
// INT8: x 尾轴 d 恒为 512 (前 448 列 nope, 后 64 列 rope)
constexpr int64_t D_LENGTH_MODE3 = 512;
// INT8: rope 段 64 个 bf16 = 128B
constexpr int64_t ROPE_BF16_BYTES = 128;
// INT8: 每组 scale 以 bf16(2B) 写出
constexpr int64_t SCALE_BF16_BYTES = 2;
// INT8: 每行 UB 预算约 6.5KB (192KB UB), rowFactor 二分查找上限
constexpr int64_t ROW_FACTOR_MAX_ARCH22 = 28;
// INT8: ReduceMax sharedTmpBuffer 固定 1 份 (64 x f32)
constexpr int64_t REDUCE_TMP_BUFFER_BYTES = 256;

// ---------- arch22 Tiling Class ----------
class KvCompressEpilogTilingArch22 {
public:
    explicit KvCompressEpilogTilingArch22(gert::TilingContext* context)
        : context_(context)
    {}
    ~KvCompressEpilogTilingArch22() = default;

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

    // Validation
    ge::graphStatus ValidateShapes();
    ge::graphStatus ValidateDtypes();

    // Utilities
    uint64_t GetTilingKey() const;
    void DumpTilingInfo();

private:
    // Context
    gert::TilingContext* context_ = nullptr;
    KvCompressEpilogTilingData tilingData_;

    // Platform info
    uint64_t coreNum_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;

    // Shape info from inputs
    int64_t bs_ = 0;                 // x 第 0 维 (token 数, 核间均分维度)
    int64_t d_ = 0;                  // x 尾轴长度 (INT8 恒 512)
    int64_t kvCacheCol_ = 0;         // 每 token 行写出字节数 (INT8 = 608)
    int64_t kvCacheRowStride_ = 0;   // block 内每 token 行 stride (物理 headDim)
    int64_t kvCacheBlockSize_ = 1;   // 每 block 的 token 数 (逻辑 blockSize)
    int64_t kvCacheBlockStride_ = 0; // 相邻 block 物理 stride (blockNum 维非连续分页)
    int64_t scaleCol_ = 0;           // scale 组数 (INT8 = 7)

    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;

    // Attributes
    int64_t quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    int64_t quantMode_ = QUANT_MODE_INT8;
    int64_t roundScale_ = 1;
    float scalesAttr_ = 0;

    // Data types
    ge::DataType xDtype_ = ge::DT_BF16;
    ge::DataType slotMappingDtype_ = ge::DT_INT32;
};

} // namespace optiling

#endif // KV_COMPRESS_EPILOG_TILING_ARCH22_H
