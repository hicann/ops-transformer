/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/sparse_lightning_indexer_kl_loss_grad_tiling_general.h"
#include "../../../../op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_tiling.h"

namespace {
constexpr uint64_t TILING_BUFFER_SIZE = 65536;
// Golden workspace sizes for B=2, S1=128, S2=256, N2=1, G=8, D=128.
constexpr size_t EXPECT_WORKSPACE_SIZE = 20506112U;
constexpr size_t EXPECT_DETERMINISTIC_WORKSPACE_SIZE = 26535424U;

const std::string SOC_INFO = R"({
    "hardware_info": {
        "UB_SIZE": 262144,
        "L2_SIZE": 134217728,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 262144,
        "CORE_NUM": 24,
        "cube_core_cnt": 24,
        "vector_core_cnt": 48,
        "socVersion": "Ascend910B"
    }
})";

gert::TilingContextPara MakeTilingContext(const std::string& layout = "BSND", ge::DataType dtype = ge::DT_FLOAT16)
{
    // The context builder requires non-null compile info; keep it alive for the returned context.
    static optiling::SparseLightningIndexerKLLossGradCompileInfo compileInfo = [] {
        optiling::SparseLightningIndexerKLLossGradCompileInfo info{};
        info.core_num = 24;
        info.aivNum = 48;
        info.aicNum = 24;
        info.ubSize = 262144;
        info.l1Size = 524288;
        info.l0cSize = 262144;
        info.l2CacheSize = 134217728;
        info.socVersion = platform_ascendc::SocVersion::ASCEND910B;
        return info;
    }();
    gert::TilingContextPara context(
        "SparseLightningIndexerKLLossGrad",
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},      // q
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},      // k
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},         // w
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // attn_softmax_l1_norm
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cu_seqlens_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cu_seqlens_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // seqused_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // seqused_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cmp_residual_k
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},                       // metadata
        },
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend910B", SOC_INFO, TILING_BUFFER_SIZE);
    if (layout == "TND") {
        context.inputTensorDesc_[0].shape_ = {{256, 8, 128}, {256, 8, 128}};
        context.inputTensorDesc_[1].shape_ = {{512, 1, 128}, {512, 1, 128}};
        context.inputTensorDesc_[2].shape_ = {{256, 8}, {256, 8}};
        context.inputTensorDesc_[3].shape_ = {{256, 1, 64}, {256, 1, 64}};
        context.inputTensorDesc_[4].shape_ = {{256, 1, 64}, {256, 1, 64}};
        context.inputTensorDesc_[5].shape_ = {{3}, {3}};
        context.inputTensorDesc_[6].shape_ = {{3}, {3}};
        for (size_t index = 0; index < 3; ++index) {
            context.outputTensorDesc_[index].shape_ = context.inputTensorDesc_[index].shape_;
        }
        context.outputTensorDesc_[3].shape_ = context.inputTensorDesc_[4].shape_;
    }
    return context;
}

void CheckTiling(const gert::TilingContextPara& context, uint8_t layout, uint64_t expectTilingKey, int32_t maskMode = 0,
                 uint32_t cmpRatio = 1)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::SparseLightningIndexerKLLossGradTilingData));
    optiling::SparseLightningIndexerKLLossGradTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingInfo.blockNum, 24U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.workspaceSizes[0],
              context.deterministicInfo_ == 1 ? EXPECT_DETERMINISTIC_WORKSPACE_SIZE : EXPECT_WORKSPACE_SIZE);
    EXPECT_EQ(tilingData.multiCoreParams.get_coreNum(), 24);
    EXPECT_EQ(tilingData.multiCoreParams.get_totalSize(), 256);
    EXPECT_EQ(tilingData.multiCoreParams.get_splitFactorSize(), 11);
    EXPECT_EQ(tilingData.initOutputParams.get_totalOutputSize(), 65536);
    EXPECT_EQ(tilingData.initOutputParams.get_singleCoreSize(), 1366U);
    EXPECT_EQ(tilingData.baseParams.get_bSize(), 2);
    EXPECT_EQ(tilingData.baseParams.get_n2Size(), 1);
    EXPECT_EQ(tilingData.baseParams.get_gSizeQueryIndex(), 8);
    EXPECT_EQ(tilingData.baseParams.get_dSizeQueryIndex(), 128);
    EXPECT_EQ(tilingData.baseParams.get_layoutType(), layout);
    EXPECT_EQ(tilingData.baseParams.get_sparseMode(), maskMode);
    EXPECT_EQ(tilingData.baseParams.get_cmpRatio(), cmpRatio);
    EXPECT_EQ(tilingData.baseParams.get_s1Size(), layout == 0 ? 128 : 256);
    EXPECT_EQ(tilingData.baseParams.get_s2Size(), layout == 0 ? 256 : 512);
    EXPECT_EQ(tilingData.baseParams.get_kSize(), 64);
}
} // namespace

class SparseLightningIndexerKLLossGradTilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseLightningIndexerKLLossGradTilingArch22 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseLightningIndexerKLLossGradTilingArch22 TearDown" << std::endl;
    }
};

TEST_F(SparseLightningIndexerKLLossGradTilingArch22, sparse_lightning_indexer_kl_loss_grad_tiling_bsnd_float16)
{
    auto context = MakeTilingContext("BSND", ge::DT_FLOAT16);
    CheckTiling(context, 0, 0U);
}

TEST_F(SparseLightningIndexerKLLossGradTilingArch22, sparse_lightning_indexer_kl_loss_grad_tiling_bsnd_bf16)
{
    auto context = MakeTilingContext("BSND", ge::DT_BF16);
    CheckTiling(context, 0, 0U);
}

TEST_F(SparseLightningIndexerKLLossGradTilingArch22, sparse_lightning_indexer_kl_loss_grad_tiling_tnd_float16)
{
    auto context = MakeTilingContext("TND", ge::DT_FLOAT16);
    CheckTiling(context, 1, 544U);
}

TEST_F(SparseLightningIndexerKLLossGradTilingArch22, sparse_lightning_indexer_kl_loss_grad_tiling_tnd_bf16)
{
    auto context = MakeTilingContext("TND", ge::DT_BF16);
    CheckTiling(context, 1, 544U);
}

TEST_F(SparseLightningIndexerKLLossGradTilingArch22, sparse_lightning_indexer_kl_loss_grad_tiling_causal)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    CheckTiling(context, 0, 8192U, 3);
}

TEST_F(SparseLightningIndexerKLLossGradTilingArch22,
       sparse_lightning_indexer_kl_loss_grad_tiling_deterministic_workspace)
{
    auto context = MakeTilingContext();
    CheckTiling(context, 0, 0U);
    context.deterministicInfo_ = 1;
    CheckTiling(context, 0, 1572864U);
}

TEST_F(SparseLightningIndexerKLLossGradTilingArch22, sparse_lightning_indexer_kl_loss_grad_tiling_empty_required_input)
{
    for (size_t index = 0; index < 5; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        auto& shape = context.inputTensorDesc_[index].shape_;
        shape.MutableOriginShape().SetDim(0, 0);
        shape.MutableStorageShape().SetDim(0, 0);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}
