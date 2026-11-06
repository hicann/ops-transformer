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
#include "../../../../op_host/sparse_lightning_indexer_kl_loss_grad_tiling_common.h"
#include "../../../../op_kernel/arch35/sparse_lightning_indexer_kl_loss_grad_tiling_regbase.h"

namespace {
constexpr uint64_t TILING_BUFFER_SIZE = 65536;
// Golden workspace sizes for B=2, S1=128, S2=256, N2=1, G=8, D=128.
constexpr size_t EXPECT_WORKSPACE_SIZE = 18251776U;
constexpr size_t EXPECT_DETERMINISTIC_WORKSPACE_SIZE = 20611072U;
// Arch35 enables both optimizations for non-deterministic TopK <= 1024.
constexpr uint64_t RESIDENT_PIPELINE_KEY = 1ULL << 22;
constexpr uint64_t BYTE_PAIR_CACHE_KEY = 1ULL << 23;
constexpr uint64_t SMALL_TOPK_OPT_KEY = RESIDENT_PIPELINE_KEY | BYTE_PAIR_CACHE_KEY;
constexpr uint64_t DETERMINISTIC_KEY = 1ULL << 21;

const std::string SOC_INFO = R"({
    "hardware_info": {
        "UB_SIZE": 262144,
        "L2_SIZE": 134217728,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 262144,
        "CORE_NUM": 32,
        "cube_core_cnt": 32,
        "vector_core_cnt": 64,
        "socVersion": "Ascend950"
    }
})";

gert::TilingContextPara MakeTilingContext(const std::string& layout = "BSND", ge::DataType dtype = ge::DT_FLOAT16,
                                          int64_t topK = 64)
{
    // The context builder requires non-null compile info; keep it alive for the returned context.
    static optiling::SparseLightningIndexerGradCompileInfo compileInfo = [] {
        optiling::SparseLightningIndexerGradCompileInfo info{};
        info.core_num = 32;
        info.aivNum = 64;
        info.aicNum = 32;
        info.ubSize = 262144;
        info.l1Size = 524288;
        info.l0cSize = 262144;
        info.l2CacheSize = 134217728;
        info.socVersion = platform_ascendc::SocVersion::ASCEND950;
        info.npuArch = NpuArch::DAV_3510;
        return info;
    }();
    gert::TilingContextPara context(
        "SparseLightningIndexerKLLossGrad",
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},          // q
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},          // k
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},             // w
            {{{2, 128, 1, topK}, {2, 128, 1, topK}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 128, 1, topK}, {2, 128, 1, topK}}, ge::DT_FLOAT, ge::FORMAT_ND}, // attn_softmax_l1_norm
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cu_seqlens_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // seqused_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                               // cmp_residual_k
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},                           // metadata
        },
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 1, topK}, {2, 128, 1, topK}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "Ascend950", SOC_INFO, TILING_BUFFER_SIZE);
    if (layout == "TND") {
        context.inputTensorDesc_[0].shape_ = {{256, 8, 128}, {256, 8, 128}};
        context.inputTensorDesc_[1].shape_ = {{512, 1, 128}, {512, 1, 128}};
        context.inputTensorDesc_[2].shape_ = {{256, 8}, {256, 8}};
        context.inputTensorDesc_[3].shape_ = {{256, 1, topK}, {256, 1, topK}};
        context.inputTensorDesc_[4].shape_ = {{256, 1, topK}, {256, 1, topK}};
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
                 uint32_t cmpRatio = 1, int64_t topK = 64, size_t expectWorkspaceSize = EXPECT_WORKSPACE_SIZE)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::SparseLightningIndexerGradRegBaseTilingData));
    optiling::SparseLightningIndexerGradRegBaseTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingInfo.blockNum, 32U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], expectWorkspaceSize);
    EXPECT_EQ(tilingData.multiCoreParams.get_coreNum(), 32);
    EXPECT_EQ(tilingData.multiCoreParams.get_totalSize(), 256);
    EXPECT_EQ(tilingData.multiCoreParams.get_splitFactorSize(), 8);
    EXPECT_EQ(tilingData.initOutputParams.get_totalOutputSize(), 65536);
    EXPECT_EQ(tilingData.initOutputParams.get_singleCoreSize(), 1024U);
    EXPECT_EQ(tilingData.baseParams.get_bSize(), 2);
    EXPECT_EQ(tilingData.baseParams.get_n2Size(), 1);
    EXPECT_EQ(tilingData.baseParams.get_gSizeQueryIndex(), 8);
    EXPECT_EQ(tilingData.baseParams.get_dSizeQueryIndex(), 128);
    EXPECT_EQ(tilingData.baseParams.get_layoutType(), layout);
    EXPECT_EQ(tilingData.baseParams.get_sparseMode(), maskMode);
    EXPECT_EQ(tilingData.baseParams.get_cmpRatio(), cmpRatio);
    EXPECT_EQ(tilingData.baseParams.get_s1Size(), layout == 0 ? 128 : 256);
    EXPECT_EQ(tilingData.baseParams.get_s2Size(), layout == 0 ? 256 : 512);
    EXPECT_EQ(tilingData.baseParams.get_kSize(), topK);
}
} // namespace

class SparseLightningIndexerKLLossGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseLightningIndexerKLLossGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseLightningIndexerKLLossGradTiling TearDown" << std::endl;
    }
};

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_bsnd_float16)
{
    auto context = MakeTilingContext("BSND", ge::DT_FLOAT16);
    CheckTiling(context, 0, SMALL_TOPK_OPT_KEY);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_bsnd_bf16)
{
    auto context = MakeTilingContext("BSND", ge::DT_BF16);
    CheckTiling(context, 0, SMALL_TOPK_OPT_KEY);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_tnd_float16)
{
    auto context = MakeTilingContext("TND", ge::DT_FLOAT16);
    CheckTiling(context, 1, 196880U | SMALL_TOPK_OPT_KEY);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_tnd_bf16)
{
    auto context = MakeTilingContext("TND", ge::DT_BF16);
    CheckTiling(context, 1, 196880U | SMALL_TOPK_OPT_KEY);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_causal)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    CheckTiling(context, 0, 4096U | SMALL_TOPK_OPT_KEY, 3);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_compressed_causal)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(128);
    context.inputTensorDesc_[9].shape_ = {{2}, {2}};
    CheckTiling(context, 0, 1052672U | SMALL_TOPK_OPT_KEY, 3, 128);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_optional_inputs_change_key)
{
    // Each optional input occupies a separate bit in the kernel dispatch key.
    const uint64_t expectTilingKeys[] = {65536U, 131072U, 262144U, 524288U, 1048576U};
    for (size_t index : {5U, 6U, 7U, 8U, 9U}) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        const int64_t length = index < 7 ? 3 : 2;
        context.inputTensorDesc_[index].shape_ = {{length}, {length}};
        CheckTiling(context, 0, expectTilingKeys[index - 5] | SMALL_TOPK_OPT_KEY);
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_deterministic_workspace)
{
    auto context = MakeTilingContext();
    CheckTiling(context, 0, SMALL_TOPK_OPT_KEY);
    context.deterministicInfo_ = 1;
    CheckTiling(context, 0, DETERMINISTIC_KEY, 0, 1, 64, EXPECT_DETERMINISTIC_WORKSPACE_SIZE);
}

TEST_F(SparseLightningIndexerKLLossGradTiling,
       sparse_lightning_indexer_kl_loss_grad_tiling_resident_pipeline_topk_boundary)
{
    struct TopKCase {
        int64_t topK;
        uint64_t optimizationKey;
        size_t workspaceSize;
        size_t deterministicWorkspaceSize;
    };
    // Keep expected dispatch choices explicit, including the unaligned value above 1024.
    // Workspace goldens use the same tensor totals as MakeTilingContext, with TopK scratch
    // aligned to 128 and an additional per-core FP32 matmul buffer in deterministic mode.
    const TopKCase cases[] = {
        {1023, SMALL_TOPK_OPT_KEY, 26738688U, 60522496U},
        {1024, SMALL_TOPK_OPT_KEY, 26738688U, 60555264U},
        {1025, 0U, 27951104U, 61800448U},
        {2048, 0U, 36438016U, 103809024U},
    };
    for (const std::string& layout : {"BSND", "TND"}) {
        const uint8_t layoutType = layout == "TND" ? 1 : 0;
        const uint64_t layoutKey = layout == "TND" ? 196880U : 0U;
        for (const auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (const auto& testCase : cases) {
                for (const int32_t maskMode : {0, 3}) {
                    SCOPED_TRACE(testing::Message() << "layout=" << layout << ", dtype=" << dtype
                                                    << ", topK=" << testCase.topK << ", maskMode=" << maskMode);
                    auto context = MakeTilingContext(layout, dtype, testCase.topK);
                    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode);
                    const uint64_t baseKey = layoutKey | (maskMode == 3 ? 4096U : 0U);
                    {
                        SCOPED_TRACE("non-deterministic");
                        CheckTiling(context, layoutType, baseKey | testCase.optimizationKey, maskMode, 1, testCase.topK,
                                    testCase.workspaceSize);
                    }
                    {
                        SCOPED_TRACE("deterministic");
                        context.deterministicInfo_ = 1;
                        CheckTiling(context, layoutType, baseKey | DETERMINISTIC_KEY, maskMode, 1, testCase.topK,
                                    testCase.deterministicWorkspaceSize);
                    }
                }
            }
        }
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_invalid_layout)
{
    for (size_t index : {0U, 1U}) {
        for (const std::string& layout : {"BNSD", "", "TND"}) {
            SCOPED_TRACE(testing::Message() << "attr=" << index << ", layout=" << layout);
            auto context = MakeTilingContext();
            context.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>(layout);
            ExecuteTestCase(context, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_invalid_mask_mode)
{
    for (int64_t mode : {-1, 1, 2, 4}) {
        SCOPED_TRACE(mode);
        auto context = MakeTilingContext();
        context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_invalid_cmp_ratio)
{
    for (int64_t ratio : {0, 129}) {
        SCOPED_TRACE(ratio);
        auto context = MakeTilingContext();
        context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(ratio);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_missing_cmp_residual)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(2);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_invalid_input_dtype)
{
    for (size_t index = 0; index < 5; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        context.inputTensorDesc_[index].dtype_ = ge::DT_INT64;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
    auto context = MakeTilingContext();
    context.inputTensorDesc_[1].dtype_ = ge::DT_BF16;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_output_shape_mismatch)
{
    for (size_t index = 0; index < 4; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        auto& shape = context.outputTensorDesc_[index].shape_;
        shape.MutableOriginShape().SetDim(0, 3);
        shape.MutableStorageShape().SetDim(0, 3);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_invalid_head_dimension)
{
    for (size_t index : {0U, 1U}) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        auto& shape = context.inputTensorDesc_[index].shape_;
        shape.MutableOriginShape().SetDim(3, 64);
        shape.MutableStorageShape().SetDim(3, 64);
        context.outputTensorDesc_[index].shape_ = shape;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_weight_head_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[2].shape_ = {{2, 128, 7}, {2, 128, 7}};
    context.outputTensorDesc_[2].shape_ = context.inputTensorDesc_[2].shape_;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_missing_metadata)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[10].shape_ = {{}, {}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_empty_required_input)
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

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_topk_exceeds_limit)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[3].shape_ = {{2, 128, 1, 2049}, {2, 128, 1, 2049}};
    context.inputTensorDesc_[4].shape_ = context.inputTensorDesc_[3].shape_;
    context.outputTensorDesc_[3].shape_ = context.inputTensorDesc_[4].shape_;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(SparseLightningIndexerKLLossGradTiling, sparse_lightning_indexer_kl_loss_grad_tiling_softmax_topk_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[4].shape_ = {{2, 128, 1, 63}, {2, 128, 1, 63}};
    context.outputTensorDesc_[3].shape_ = context.inputTensorDesc_[4].shape_;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
