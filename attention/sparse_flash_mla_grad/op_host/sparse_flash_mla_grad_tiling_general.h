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
 * \file sparse_flash_mla_grad_tiling_general.h
 * \brief
 */

#pragma once

#include "op_host/tiling_base.h"
#include "op_host/tiling_type.h"
#include "sparse_flash_mla_grad_tiling.h"

namespace optiling {
namespace smlag {

#define SMLAG_SWA_MODE 0
#define SMLAG_CFA_MODE 1
#define SMLAG_SCFA_MODE 2

struct TempParams {
    uint32_t dataTypeSize;
    uint32_t queryType;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;
    int64_t b;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t s3;
    int64_t d;
    uint32_t layout;
    uint32_t singleM;
    uint32_t singleN;
    uint32_t s1BasicSize;
    uint32_t selected_block_count;
    uint32_t mode;
};

struct AiCoreParams {
    uint64_t ubSize;
    uint64_t numBlocks;
    uint64_t aicNum;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

class SparseFlashMlaGradBasicTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit SparseFlashMlaGradBasicTiling(gert::TilingContext *context)
        : TilingBaseClass(context) {};
    SparseFlashMlaGradTilingData tilingData;

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
} // namespace smlag
} // namespace optiling
