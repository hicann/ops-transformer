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
#include "op_host/util/op_const_def.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/dense_lightning_indexer_kl_loss_grad_tiling_common.h"
#include "../../../../op_kernel/arch35/dense_lightning_indexer_kl_loss_grad_tiling_regbase.h"

namespace {
constexpr uint64_t TILING_BUFFER_SIZE = 65536;
// Golden workspace sizes for B=2, S1=128, S2=256, N2=1, G=8, D=128.
constexpr size_t EXPECT_WORKSPACE_SIZE = 17137664U;
constexpr size_t EXPECT_DETERMINISTIC_WORKSPACE_SIZE = 419790848U;

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
        "socVersion": "3510"
    }
})";

gert::TilingContextPara MakeTilingContext(const std::string& layout = "BSND", ge::DataType dtype = ge::DT_FLOAT16)
{
    // The context builder requires non-null compile info; keep it alive for the returned context.
    static optiling::DenseLightningIndexerKLLossGradCompileInfo compileInfo = [] {
        optiling::DenseLightningIndexerKLLossGradCompileInfo info{};
        info.core_num = 32;
        info.aivNum = 64;
        info.aicNum = 32;
        info.ubSize = 262144;
        info.l1Size = 524288;
        info.l0cSize = 262144;
        info.l2CacheSize = 134217728;
        info.npuArch = Ops::Base::DAV_3510;
        return info;
    }();
    gert::TilingContextPara context(
        "DenseLightningIndexerKLLossGrad",
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},        // q
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},        // k
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},           // w
            {{{2, 128, 1, 256}, {2, 128, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND}, // attn_softmax_l1_norm
            {{{2, 1, 128}, {2, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // softmax_lse
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cu_seqlens_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cu_seqlens_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // seqused_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // seqused_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                             // cmp_residual_k
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},                         // metadata
        },
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 128, 1, 256}, {2, 128, 1, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo, "3510", SOC_INFO, TILING_BUFFER_SIZE);
    if (layout == "TND") {
        context.inputTensorDesc_[0].shape_ = {{256, 8, 128}, {256, 8, 128}};
        context.inputTensorDesc_[1].shape_ = {{512, 1, 128}, {512, 1, 128}};
        context.inputTensorDesc_[2].shape_ = {{256, 8}, {256, 8}};
        context.inputTensorDesc_[3].shape_ = {{256, 1, 256}, {256, 1, 256}};
        context.inputTensorDesc_[4].shape_ = {{1, 256}, {1, 256}};
        context.inputTensorDesc_[5].shape_ = {{3}, {3}};
        context.inputTensorDesc_[6].shape_ = {{3}, {3}};
        for (size_t index = 0; index < 3; ++index) {
            context.outputTensorDesc_[index].shape_ = context.inputTensorDesc_[index].shape_;
        }
        context.outputTensorDesc_[3].shape_ = context.inputTensorDesc_[3].shape_;
    }
    return context;
}

void CheckTiling(const gert::TilingContextPara& context, uint8_t layout, uint64_t expectTilingKey, int32_t maskMode = 0,
                 uint32_t cmpRatio = 1)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::DenseLightningIndexerGradRegBaseTilingData));
    optiling::DenseLightningIndexerGradRegBaseTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingInfo.blockNum, 32U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.workspaceSizes[0],
              context.deterministicInfo_ == 1 ? EXPECT_DETERMINISTIC_WORKSPACE_SIZE : EXPECT_WORKSPACE_SIZE);
    EXPECT_EQ(tilingData.multiCoreParams.get_coreNum(), 32);
    EXPECT_EQ(tilingData.multiCoreParams.get_totalSize(), 256);
    EXPECT_EQ(tilingData.multiCoreParams.get_splitFactorSize(), 8);
    EXPECT_EQ(tilingData.multiCoreParams.get_qPreBlockFactor(), 4096);
    EXPECT_EQ(tilingData.multiCoreParams.get_qPreBlockTotal(), 64);
    EXPECT_EQ(tilingData.multiCoreParams.get_qPreBlockTail(), 4096);
    EXPECT_EQ(tilingData.multiCoreParams.get_kPreBlockFactor(), 1024);
    EXPECT_EQ(tilingData.multiCoreParams.get_kPreBlockTotal(), 64);
    EXPECT_EQ(tilingData.multiCoreParams.get_kPreBlockTail(), 1024);
    EXPECT_EQ(tilingData.baseParams.get_bSize(), 2);
    EXPECT_EQ(tilingData.baseParams.get_n2Size(), 1);
    EXPECT_EQ(tilingData.baseParams.get_gSizeQueryIndex(), 8);
    EXPECT_EQ(tilingData.baseParams.get_dSizeQueryIndex(), 128);
    EXPECT_EQ(tilingData.baseParams.get_layoutType(), layout);
    EXPECT_EQ(tilingData.baseParams.get_sparseMode(), maskMode);
    EXPECT_EQ(tilingData.baseParams.get_cmpRatio(), cmpRatio);
    EXPECT_EQ(tilingData.baseParams.get_s1Size(), layout == 0 ? 128 : 256);
    EXPECT_EQ(tilingData.baseParams.get_s2Size(), layout == 0 ? 256 : 512);
    EXPECT_EQ(tilingData.baseParams.get_maxSeqlenK(), 256U);
    EXPECT_EQ(tilingData.baseParams.get_mm3ResWorkSpaceOffset(), context.deterministicInfo_ == 1 ? 402751488 : 98304);
}
} // namespace

class DenseLightningIndexerKLLossGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DenseLightningIndexerKLLossGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DenseLightningIndexerKLLossGradTiling TearDown" << std::endl;
    }
};

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_bsnd_float16)
{
    auto context = MakeTilingContext("BSND", ge::DT_FLOAT16);
    CheckTiling(context, 0, 0U);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_bsnd_bf16)
{
    auto context = MakeTilingContext("BSND", ge::DT_BF16);
    CheckTiling(context, 0, 0U);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_tnd_float16)
{
    auto context = MakeTilingContext("TND", ge::DT_FLOAT16);
    CheckTiling(context, 1, 196880U);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_tnd_bf16)
{
    auto context = MakeTilingContext("TND", ge::DT_BF16);
    CheckTiling(context, 1, 196880U);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_causal)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    // SPARSE_MODE=3 is encoded as index 1 in the template's {0, 3} value list.
    CheckTiling(context, 0, 4096U, 3);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_compressed_causal)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(128);
    context.inputTensorDesc_[9].shape_ = {{2}, {2}};
    CheckTiling(context, 0, 1052672U, 3, 128);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_optional_inputs_change_key)
{
    // Each optional input occupies a separate bit in the kernel dispatch key.
    const uint64_t expectTilingKeys[] = {65536U, 131072U, 262144U, 524288U, 1048576U};
    for (size_t index : {5U, 6U, 7U, 8U, 9U}) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        const int64_t length = index < 7 ? 3 : 2;
        context.inputTensorDesc_[index].shape_ = {{length}, {length}};
        CheckTiling(context, 0, expectTilingKeys[index - 5]);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_deterministic_workspace)
{
    auto context = MakeTilingContext();
    CheckTiling(context, 0, 0U);
    context.deterministicInfo_ = 1;
    CheckTiling(context, 0, 2097152U);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_layout)
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

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_mask_mode)
{
    for (int64_t mode : {-1, 1, 2, 4}) {
        SCOPED_TRACE(mode);
        auto context = MakeTilingContext();
        context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_cmp_ratio)
{
    for (int64_t ratio : {0, 129}) {
        SCOPED_TRACE(ratio);
        auto context = MakeTilingContext();
        context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(ratio);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_missing_cmp_residual)
{
    auto context = MakeTilingContext();
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(3);
    context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(2);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_input_dtype)
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

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_output_shape_mismatch)
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

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_head_dimension)
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

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_weight_head_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[2].shape_ = {{2, 128, 7}, {2, 128, 7}};
    context.outputTensorDesc_[2].shape_ = context.inputTensorDesc_[2].shape_;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_missing_metadata)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[10].shape_ = {{}, {}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_metadata_shape)
{
    for (const gert::StorageShape& shape :
         {gert::StorageShape({{63}, {63}}), gert::StorageShape({{65}, {65}}), gert::StorageShape({{1, 64}, {1, 64}})}) {
        auto context = MakeTilingContext();
        context.inputTensorDesc_[10].shape_ = shape;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_optional_dtype)
{
    for (size_t index = 5; index < 11; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext("TND");
        if (index >= 7 && index < 10) {
            context.inputTensorDesc_[index].shape_ = {{2}, {2}};
        }
        context.inputTensorDesc_[index].dtype_ = ge::DT_INT64;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_optional_length)
{
    for (size_t index = 5; index < 10; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        context.inputTensorDesc_[index].shape_ = {{4}, {4}};
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_tnd_missing_cu_seqlens)
{
    for (size_t index : {5U, 6U}) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext("TND");
        context.inputTensorDesc_[index].shape_ = {{}, {}};
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_invalid_lse_rank)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[4].shape_ = {{2, 1, 128, 1}, {2, 1, 128, 1}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(DenseLightningIndexerKLLossGradTiling, dense_lightning_indexer_kl_loss_grad_tiling_lse_sequence_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[4].shape_ = {{2, 1, 127}, {2, 1, 127}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
