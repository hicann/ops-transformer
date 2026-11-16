/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fused_weight_tiling_arch35.h
 * \brief EngramFusedWeight tiling 定义 (arch35 / ascend950)
 */

#ifndef ENGRAM_FUSED_WEIGHT_TILING_ARCH35_H
#define ENGRAM_FUSED_WEIGHT_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {

// ---------- 输入索引 ----------
constexpr int64_t WEIGHT_HIDDEN_INPUT_INDEX = 0;
constexpr int64_t WEIGHT_EMBED_INPUT_INDEX = 1;
constexpr int64_t WEIGHT_FUSED_OUTPUT_INDEX = 0;

// ---------- 输入范围约束 ----------
constexpr int64_t INPUT_DIM_NUM = 2;      // 两个输入均须为 2D [hcMult, hiddenSize]
constexpr int64_t TOTAL_ELEM_ALIGN = 256; // 总元素数须能被 256 整除 (0 元素允许, kernel 直接返回)

// ---------- tiling 常量 ----------
// blockSize 分档: 总量 <= 12288 -> 512; <= 131072 -> 2048;
// <= 524288 -> 4096; 否则 8192。
constexpr int64_t BLOCK_SIZE_SMALL = 512;
constexpr int64_t BLOCK_SIZE_MEDIUM = 2048;
constexpr int64_t BLOCK_SIZE_LARGE = 4096;
constexpr int64_t BLOCK_SIZE_MAX = 8192;
constexpr int64_t BLOCK_SIZE_MIN = 256;
constexpr int64_t TOTAL_ELEM_SMALL = 12288;
constexpr int64_t TOTAL_ELEM_MEDIUM = 131072;
constexpr int64_t TOTAL_ELEM_LARGE = 524288;

// UB 预算: x1/x2 双缓冲 16-bit 输入 BF16/FP16 (2*2*2B) + y 双缓冲 FP32 (2*4B) = 每元素 16 字节
constexpr int64_t UB_BYTES_PER_ELEMENT = 16;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t UB_RESERVED_BYTE = 256;

// ---------- TilingData Structure ----------
BEGIN_TILING_DATA_DEF(EngramFusedWeightTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalElements);    // 展平后的总元素数 (hcMult * hiddenSize)
TILING_DATA_FIELD_DEF(int64_t, blockSize);        // 每个 UB 块的元素数
TILING_DATA_FIELD_DEF(int64_t, perCoreElem);      // 头核要处理的元素数量
TILING_DATA_FIELD_DEF(int64_t, blockFactor);      // 头核 UB 的循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);  // 尾核 UB 的循环次数
TILING_DATA_FIELD_DEF(int64_t, perCoreTailElem);  // 头核最后一次 UB 循环加载的元素数量
TILING_DATA_FIELD_DEF(int64_t, tailCoreTailElem); // 尾核最后一次 UB 循环加载的元素数量
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);     // 实际使用的 AIV 核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EngramFusedWeight, EngramFusedWeightTilingData)

// ---------- CompileInfo Structure ----------
struct EngramFusedWeightCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ---------- Tiling Class ----------
class EngramFusedWeightTilingArch35 {
public:
    explicit EngramFusedWeightTilingArch35(gert::TilingContext* context)
        : context_(context)
    {}

    ~EngramFusedWeightTilingArch35() = default;

    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndDtypeInfo();
    ge::graphStatus ValidateShapes(const gert::Shape& whStorage, const gert::Shape& weStorage,
                                   const gert::Shape& outStorage);
    ge::graphStatus ValidateDtypes();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();
    void DumpTilingInfo();

    gert::TilingContext* context_ = nullptr;
    EngramFusedWeightTilingData tilingData_;

    // Platform info
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t usedCoreNums_ = 0;

    // Shape info
    int64_t totalElements_ = 0;
};

} // namespace optiling

#endif // ENGRAM_FUSED_WEIGHT_TILING_ARCH35_H
