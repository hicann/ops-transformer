/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_selected_attention_grad_bs1.h
 * \brief
 */

#pragma once

#include "nsa_selected_attention_grad_tiling_common.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_type.h"
#include "nsa_selected_attention_grad_tiling.h"
namespace optiling {
namespace nsa {
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
    uint32_t isDeterministic = 0;
};

class NsaSelectedAttentionGradTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit NsaSelectedAttentionGradTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    NsaSelectedAttentionGradTilingData tilingData;

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
    ge::graphStatus GetLayoutInfo();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus DoSftTiling();
    ge::graphStatus DoBlockTiling();
    ge::graphStatus DoCastTiling();
    ge::graphStatus Setmm1TilingData(matmul_tiling::DataType inputDtype, matmul_tiling::DataType outputDtype);
    ge::graphStatus Setmm2TilingData(matmul_tiling::DataType inputDtype, matmul_tiling::DataType outputDtype);
    ge::graphStatus Setmm3TilingData(matmul_tiling::DataType inputDtype, matmul_tiling::DataType outputDtype);
    ge::graphStatus Setmm4TilingData(matmul_tiling::DataType inputDtype, matmul_tiling::DataType outputDtype);
    ge::graphStatus Setmm5TilingData(matmul_tiling::DataType inputDtype, matmul_tiling::DataType outputDtype);
    ge::graphStatus SetBaseInfo(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                const gert::Shape &valueShape, int64_t dimN1);
    void PrintShapeInfo();
    void DoPreTiling();

    TempParams tmpData;
    uint32_t singleM{0};
    uint32_t singleN{0};
    uint32_t s1Ratio{0};
};
} // namespace nsa
} // namespace optiling
