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
 * \file sparse_flash_attention_grad_tiling_bs1_basic.h
 * \brief
 */

#pragma once

#include "../sparse_flash_attention_grad_tiling_common.h"
#include "../sparse_flash_attention_grad_tiling.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_type.h"

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
    bool kvMerge = false;
    bool enableOptimizedScatter = false;
};

bool EnableOptimizedScatterPath(const TempParams &tmpData);

class SparseFlashAttentionGradBasicTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit SparseFlashAttentionGradBasicTiling(gert::TilingContext *context)
        : TilingBaseClass(context) {};
    SparseFlashAttentionGradBasicTilingData tilingData;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus DoSftTiling();
    ge::graphStatus DoBlockTiling();
    ge::graphStatus DoCastTiling();
    TempParams tmpData;

    const char *opName;
};
} // namespace sfag
} // namespace optiling
