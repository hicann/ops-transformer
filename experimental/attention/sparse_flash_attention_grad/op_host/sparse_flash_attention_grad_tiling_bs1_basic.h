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
 * \file sparse_flash_attention_grad_tiling_bs1_basic.h
 * \brief
 */

#pragma once

#include "sparse_flash_attention_grad_tiling_common.h"
#include "sparse_flash_attention_grad_tiling.h"

namespace optiling {
namespace sfag {
struct TempParams {
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;
    uint32_t dataTypeSize;
    uint32_t queryType;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t d2;
    uint32_t layout;
    uint32_t selected_block_count;
    uint32_t selected_block_size;
    bool attenEnable = false;
    uint32_t singleM;
    uint32_t singleN;
    int64_t ropeDim;
    bool ropeEnable = false;
    bool deterministic = false;
    bool enableOptimizedScatter = false;
};

struct AiCoreParams {
    uint64_t ubSize;
    uint64_t blockDim;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

class SparseFlashAttentionGradBasicTiling {
public:
    explicit SparseFlashAttentionGradBasicTiling(gert::TilingContext *context)
        : context_(context) {};
    SparseFlashAttentionGradBasicTilingData tilingData;

public:
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus GetPlatformInfo();
    bool IsCapable();
    ge::graphStatus DoOpTiling();
    ge::graphStatus DoLibApiTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    uint64_t GetTilingKey() const;

private:
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus DoSftTiling();
    ge::graphStatus DoBlockTiling();
    ge::graphStatus DoCastTiling();
    TempParams tmpData;

    gert::TilingContext *context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
    AiCoreParams aicoreParams_{0, 0, 0, 0, 0, 0, 0};
};
} // namespace sfag
} // namespace optiling
