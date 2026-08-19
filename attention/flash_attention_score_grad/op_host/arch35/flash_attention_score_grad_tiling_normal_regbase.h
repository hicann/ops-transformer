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
 * \file flash_attention_score_grad_tiling_normal_regbase.h
 * \brief
 */

#pragma once

#include "flash_attention_score_grad_tiling_common_regbase.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/flash_attention_score_grad_template_tiling_key.h"
#include "err/ops_err.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace fag {

#define TND_TILING_DATA_COMMON_ASSIGN(tilingData) \
    do { \
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams; \
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams; \
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList; \
        preTilingData_ = &tilingData->preTilingData; \
        postTilingData_ = &tilingData->postTilingData; \
        tndParam_ = &tilingData->tndParam; \
    } while (0)

#define BASE_TILING_DATA_COMMON_ASSIGN(tilingData) \
    do { \
        s1s2BNGS1S2BaseParams_ = &tilingData->s1s2BNGS1S2BaseParams; \
        s1s2BNGS1S2SplitCoreParams_ = &tilingData->s1s2BNGS1S2SplitCoreParams; \
        s1s2BNGS1S2BlockNumList_ = &tilingData->s1s2BNGS1S2BlockNumList; \
        preTilingData_ = &tilingData->preTilingData; \
        postTilingData_ = &tilingData->postTilingData; \
    } while (0)

class FlashAttentionScoreGradTilingNormalRegbase : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingNormalRegbase(gert::TilingContext *curContext_)
        : TilingBaseClass(curContext_)
    {}
    ~FlashAttentionScoreGradTilingNormalRegbase() override = default;

    FlashAttentionScoreGradS1S2BNGS1S2BaseParamsRegbase *s1s2BNGS1S2BaseParams_ = nullptr;
    FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsRegbase *s1s2BNGS1S2SplitCoreParams_ = nullptr;
    BlockNumListParamsRegbase *s1s2BNGS1S2BlockNumList_ = nullptr;
    PreParamsRegbase *preTilingData_ = nullptr;
    PostParamsRegbase *postTilingData_ = nullptr;
    BaseDeterParamRegbase *baseDeterParam_ = nullptr;
    DeterParamRegbase *deterParam = nullptr;
    TndParamRegbase *tndParam_ = nullptr;
    TndSwizzleParamRegbase *tndSwizzleParam_ = nullptr;

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
    void DetermineBlockSchedule();
    ge::graphStatus DoSparse();
    bool DoBn2s2Sparse();
    uint32_t GetDeterSparseTilingKey();
    uint8_t GetSparseType();
    void CalcleDeterParam();
    virtual void CalcleTNDDeterParam() {};
    virtual void CalcTNDSwizzleParam() {};

    void GetWorkspaceSize4Deter(size_t &workspaceSize);
    void GetIsDeterArr();
    bool CheckExceedL2Cache();
    bool IsValid(int64_t blockIdx);
    std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
    ge::graphStatus GetSparseBlockInfo();
    void DoPreTiling();
    uint64_t DoPreSfmgTiling();
    void DoPostTiling();
    ge::graphStatus SaveToTilingData();
    ge::graphStatus GetSparsePrefixBlockInfo();
    virtual ge::graphStatus GetSparseUnpadBlockInfo() {};
    virtual bool GetBlockInfoOfBNS4TND() {};
    void FillBlockInfoLoadBalance(std::vector<std::vector<int64_t>> &totalBlockInfo,
                                  std::vector<std::vector<float>> &acturalBlockInfo);
    void GetParseS1S2OuterInfo(int64_t (*parseInfo)[ARRAY_LENGTH]);
    bool CheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx = 0,
                                 int64_t blockIdx = 0);
    virtual bool CheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx) {
    };
    virtual bool IsValidUnpad(int64_t blockIdx) {};
    ge::graphStatus DoBn2MultiBlkSparse();
    virtual ge::graphStatus GetBlockInfoOfTNDForBn2() {};
    ge::graphStatus GetSparseBlockInfoBn2();
    FuzzyBaseInfoParamsRegbase fBaseParams;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch = NpuArch::DAV_RESV;
    TndBaseInfo tndBaseInfo;
};

} // namespace fag
} // namespace optiling
