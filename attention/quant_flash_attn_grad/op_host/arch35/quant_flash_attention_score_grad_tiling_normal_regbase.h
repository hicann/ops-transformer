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
 * \file quant_flash_attention_score_grad_tiling_normal_regbase.h
 * \brief
 */

#pragma once

#include "quant_flash_attention_score_grad_tiling_common_regbase.h"
// #include "../../op_kernel/arch35/quant_flash_attn_grad_template_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "err/ops_err.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace QuantFag {

class QuantFlashAttentionScoreGradTilingNormalRegbase : public TilingBaseClass {
public:
    explicit QuantFlashAttentionScoreGradTilingNormalRegbase(gert::TilingContext *curContext_)
        : TilingBaseClass(curContext_)
    {}
    ~QuantFlashAttentionScoreGradTilingNormalRegbase() override = default;

    QuantFlashAttnGradTiling *quantFagTilingData_ = nullptr;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus InitTilingData();
    void DoSplit();
    ge::graphStatus DoSparse();
    uint32_t GetDeterSparseTilingKey();
    uint8_t GetSparseType();
    void CalcleDeterParam();

    void GetWorkspaceSize4Deter(size_t &workspaceSize);
    void GetIsDeterArr();
    bool IsValid(int64_t blockIdx);
    std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
    ge::graphStatus GetSparseBlockInfo();
    void DoPreTiling();
    uint64_t DoPreSfmgTiling();
    void DoPostTiling();
    ge::graphStatus SaveToTilingData();
    ge::graphStatus GetSparsePrefixBlockInfo();
    void GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH]);
    bool CheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx = 0,
                                 int64_t blockIdx = 0);
    FuzzyBaseInfoParamsRegbase fBaseParams;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch = NpuArch::DAV_RESV;
};

} // namespace QuantFag
} // namespace optiling
