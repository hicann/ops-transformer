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
 * \file engram_hash_tiling_arch35.h
 * \brief EngramHash tiling 定义 (arch35 / ascend950)
 */

#ifndef ENGRAM_HASH_TILING_ARCH35_H
#define ENGRAM_HASH_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {

// ---------- 输入索引 ----------
constexpr int64_t NGRAM_TOKEN_IDS_INPUT_INDEX = 0;
constexpr int64_t MULTIPLIERS_INPUT_INDEX = 1;
constexpr int64_t VOCAB_SIZES_INPUT_INDEX = 2;
constexpr int64_t OFFSETS_INPUT_INDEX = 3;
constexpr int64_t HASH_INDICES_OUTPUT_INDEX = 0;

// ---------- 输入范围约束 ----------
constexpr int64_t IDS_DIM_NUM = 2;
constexpr int64_t MIN_NGRAM_SIZE = 2;
constexpr int64_t VOCAB_DIM_NUM = 3;
constexpr int64_t MULTIPLIERS_DIM_NUM = 2;
constexpr int64_t OFFSETS_DIM_NUM = 2;
constexpr int64_t MAX_SIMT_THREADS = 1024;
constexpr int64_t UB_BUDGET_BYTES = 8 * 1024;

// ---------- tiling 常量 ----------
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;

// ---------- TilingData Structure ----------
BEGIN_TILING_DATA_DEF(EngramHashTilingData)
TILING_DATA_FIELD_DEF(int64_t, numTokens);        // T, 动态 token 数 (可为 0)
TILING_DATA_FIELD_DEF(int64_t, maxNgramSize);     // N, ngram_token_ids 的 dim1
TILING_DATA_FIELD_DEF(int64_t, numNgramLayers);   // L, vocab_sizes 的 dim0
TILING_DATA_FIELD_DEF(int64_t, numTables);        // tables, vocab_sizes 的 dim2
TILING_DATA_FIELD_DEF(int64_t, numOutCols);       // C = (N - 1) * tables, 输出最后一维
TILING_DATA_FIELD_DEF(int64_t, tileTokens);       // 每 tile 的 token 数
TILING_DATA_FIELD_DEF(int64_t, colChunk);         // 每 tile 内的列分块大小 (UB 预算自适应)
TILING_DATA_FIELD_DEF(int64_t, numTilesPerLayer); // 每层 tile 数 = ceil(T / tileTokens)
TILING_DATA_FIELD_DEF(int64_t, totalBlocks);      // 总块数 = L * numTilesPerLayer
TILING_DATA_FIELD_DEF(int64_t, usedCoreNums);     // 实际使用的 AIV 核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EngramHash, EngramHashTilingData)

// ---------- CompileInfo Structure ----------
struct EngramHashCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ---------- Tiling Class ----------
class EngramHashTilingArch35 {
public:
    explicit EngramHashTilingArch35(gert::TilingContext* context)
        : context_(context)
    {}

    ~EngramHashTilingArch35() = default;

    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndDtypeInfo();
    ge::graphStatus ValidateShapes(const gert::Shape& idsStorage, const gert::Shape& multStorage,
                                   const gert::Shape& vocabStorage, const gert::Shape& offsStorage,
                                   const gert::Shape& outStorage);
    ge::graphStatus ValidateDtypes();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();
    void DumpTilingInfo();

    gert::TilingContext* context_ = nullptr;
    EngramHashTilingData tilingData_;

    // Platform info
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t usedCoreNums_ = 0;

    // Shape info
    int64_t numTokens_ = 0;
    int64_t maxNgramSize_ = 0;
    int64_t numNgramLayers_ = 0;
    int64_t numTables_ = 0;
    int64_t numOutCols_ = 0;
};

} // namespace optiling

#endif // ENGRAM_HASH_TILING_ARCH35_H
