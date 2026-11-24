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
#include "../../../../op_host/quant_flash_attn_grad_tiling.h"

namespace {
constexpr uint64_t TILING_BUFFER_SIZE = 65536;
// Generated PyPTO tiling headers are force-included for this source by CMake,
// exactly as for the host tiling implementation.
struct TilingExpect {
    uint32_t layout = 0;
    uint32_t sparseType = 0;
    bool hasSequsedQ = false;
    bool hasSequsedKv = false;
    int64_t s1 = 512;
    int64_t s2 = 1024;
    int64_t metadataLength = 16;
    int64_t maskMode = 0;
    int64_t s1Token = 100000000;
    int64_t s2Token = 100000000;
};
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

gert::TilingContextPara MakeTilingContext(const std::string& layout = "BSND")
{
    // The context builder requires non-null compile info; keep it alive for the returned context.
    static optiling::QuantFlashAttnGradCompileInfo compileInfo = [] {
        optiling::QuantFlashAttnGradCompileInfo info{};
        info.coreNum = 32;
        info.aivNum = 64;
        info.aicNum = 32;
        info.ubSize = 262144;
        info.l1Size = 524288;
        info.l0cSize = 262144;
        info.l2CacheSize = 134217728;
        info.l0aSize = 65536;
        info.l0bSize = 65536;
        info.npuArch = Ops::Base::DAV_3510;
        return info;
    }();
    gert::TilingContextPara context("QuantFlashAttnGrad",
                                    {
                                        {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // q
                                        {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // k
                                        {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // v
                                        {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // dout
                                        {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // attn_out
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // q_descale
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // k_descale
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // v_descale
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // do_descale
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // p_scale
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                 // ds_scale
                                        {{{2, 4, 512}, {2, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_lse
                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                   // cu_seqlens_q
                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                   // cu_seqlens_kv
                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                   // seqused_q
                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                   // seqused_kv
                                        {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                    // sinks
                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                    // attn_mask
                                        {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},         // metadata
                                    },
                                    {
                                        {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // dq
                                        {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // dk
                                        {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // dv
                                        {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dsink
                                    },
                                    {
                                        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                                        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                        {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                        {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
                                        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
                                        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
                                    },
                                    &compileInfo, "3510", SOC_INFO, TILING_BUFFER_SIZE);
    if (layout == "BNSD") {
        context.inputTensorDesc_[0].shape_ = {{2, 4, 512, 128}, {2, 4, 512, 128}};
        context.inputTensorDesc_[1].shape_ = {{2, 4, 1024, 128}, {2, 4, 1024, 128}};
    } else if (layout == "TND") {
        context.inputTensorDesc_[0].shape_ = {{1024, 4, 128}, {1024, 4, 128}};
        context.inputTensorDesc_[1].shape_ = {{2048, 4, 128}, {2048, 4, 128}};
        context.inputTensorDesc_[11].shape_ = {{4, 1024}, {4, 1024}};
        context.inputTensorDesc_[12].shape_ = {{3}, {3}};
        context.inputTensorDesc_[13].shape_ = {{3}, {3}};
        context.inputTensorDesc_[18].shape_ = {{2, 37}, {2, 37}};
    }
    context.inputTensorDesc_[2].shape_ = context.inputTensorDesc_[1].shape_;
    context.inputTensorDesc_[3].shape_ = context.inputTensorDesc_[0].shape_;
    context.inputTensorDesc_[4].shape_ = context.inputTensorDesc_[0].shape_;
    for (size_t index = 0; index < 3; ++index) {
        context.outputTensorDesc_[index].shape_ = context.inputTensorDesc_[index].shape_;
    }
    return context;
}

void SetMaskMode(gert::TilingContextPara& context, int64_t mode)
{
    context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode);
    if (mode == 3 || mode == 4) {
        context.inputTensorDesc_[17].shape_ = {{2048, 2048}, {2048, 2048}};
    }
}

void CheckTiling(const gert::TilingContextPara& context, const TilingExpect& expect = {})
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 32U);
    // Fixed template sizes and independently specified dispatch choices.
    const uint64_t expectTilingKey = GET_TPL_TILING_KEY(
        static_cast<uint8_t>(expect.maskMode != 0), static_cast<uint8_t>(0), static_cast<uint16_t>(512),
        static_cast<uint16_t>(512), static_cast<uint8_t>(128), static_cast<uint8_t>(1),
        static_cast<uint8_t>(expect.layout), static_cast<uint8_t>(expect.sparseType),
        static_cast<uint8_t>(expect.hasSequsedQ), static_cast<uint8_t>(expect.hasSequsedKv));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    // FP32 dq/dk/dv, FP8 scratch and softmax-grad buffers, including alignment
    // and the 64 KiB + 20 MiB reserved regions. Packed inputs have the same totals.
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 31689216U);
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(QuantFlashAttnGradTiling));
    QuantFlashAttnGradTiling tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingData.b, 2);
    EXPECT_EQ(tilingData.n1, 4);
    EXPECT_EQ(tilingData.n2, 4);
    EXPECT_EQ(tilingData.g, 1);
    EXPECT_EQ(tilingData.d, 128);
    EXPECT_EQ(tilingData.s1, expect.s1);
    EXPECT_EQ(tilingData.s2, expect.s2);
    EXPECT_EQ(tilingData.t1, expect.layout == 2 ? 1024 : 0);
    EXPECT_EQ(tilingData.t2, expect.layout == 2 ? 2048 : 0);
    EXPECT_FLOAT_EQ(tilingData.softmax_scale, 0.088388f);
    EXPECT_EQ(tilingData.metadata_len, expect.metadataLength);
    EXPECT_EQ(tilingData.mask_mode, expect.maskMode);
    EXPECT_EQ(tilingData.s1_token, expect.s1Token);
    EXPECT_EQ(tilingData.s2_token, expect.s2Token);
    EXPECT_EQ(tilingData.dq_work_space_offset, 65536);
    EXPECT_EQ(tilingData.dk_work_space_offset, 2163200);
    EXPECT_EQ(tilingData.dv_work_space_offset, 6358016);
    EXPECT_EQ(tilingData.sfmg_work_space_offset, 10586112);
}
} // namespace

class QuantFlashAttnGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantFlashAttnGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantFlashAttnGradTiling TearDown" << std::endl;
    }
};

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_bsnd)
{
    auto context = MakeTilingContext("BSND");
    CheckTiling(context);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_bnsd)
{
    auto context = MakeTilingContext("BNSD");
    TilingExpect expect;
    expect.layout = 1;
    CheckTiling(context, expect);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_tnd)
{
    auto context = MakeTilingContext("TND");
    TilingExpect expect;
    expect.layout = 2;
    expect.s1 = 1024;
    expect.s2 = 2048;
    expect.metadataLength = 37;
    CheckTiling(context, expect);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_layout_keys)
{
    uint32_t layoutIndex = 0;
    for (const std::string& layout : {"BSND", "BNSD", "TND"}) {
        SCOPED_TRACE(layout);
        auto context = MakeTilingContext(layout);
        TilingExpect expect;
        expect.layout = layoutIndex++;
        if (layout == "TND") {
            expect.s1 = 1024;
            expect.s2 = 2048;
            expect.metadataLength = 37;
        }
        CheckTiling(context, expect);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_causal)
{
    auto context = MakeTilingContext();
    SetMaskMode(context, 3);
    TilingExpect expect;
    expect.sparseType = 4; // Unequal S1/S2 uses the band kernel for right-down causal.
    expect.maskMode = 3;
    expect.s1Token = 99999488;
    expect.s2Token = 512;
    CheckTiling(context, expect);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_band)
{
    auto context = MakeTilingContext();
    SetMaskMode(context, 4);
    context.attrs_[3].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(128);
    context.attrs_[4].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(0);
    TilingExpect expect;
    expect.sparseType = 4;
    expect.maskMode = 4;
    expect.s1Token = -384;
    expect.s2Token = 512;
    CheckTiling(context, expect);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_tnd_max_seqlen)
{
    auto context = MakeTilingContext("TND");
    context.attrs_[5].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(512);
    context.attrs_[6].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024);
    TilingExpect expect;
    expect.layout = 2;
    expect.metadataLength = 37;
    CheckTiling(context, expect);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_seqused_keys)
{
    for (const std::string& layout : {"BSND", "TND"}) {
        SCOPED_TRACE(layout);
        for (uint32_t flags = 0; flags < 4; ++flags) {
            SCOPED_TRACE(flags);
            auto context = MakeTilingContext(layout);
            // B=2: dense padded metadata needs 17+3B; packed metadata needs 17+10B.
            const int64_t metadataLength = layout == "TND" ? 37 : 23;
            context.inputTensorDesc_[18].shape_ = {{2, metadataLength}, {2, metadataLength}};
            if ((flags & 1U) != 0) {
                context.inputTensorDesc_[14].shape_ = {{2}, {2}};
            }
            if ((flags & 2U) != 0) {
                context.inputTensorDesc_[15].shape_ = {{2}, {2}};
            }
            TilingExpect expect;
            expect.layout = layout == "TND" ? 2 : 0;
            expect.s1 = layout == "TND" ? 1024 : 512;
            expect.s2 = layout == "TND" ? 2048 : 1024;
            expect.metadataLength = metadataLength;
            expect.hasSequsedQ = (flags & 1U) != 0;
            expect.hasSequsedKv = (flags & 2U) != 0;
            CheckTiling(context, expect);
        }
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_causal_seqused)
{
    auto context = MakeTilingContext();
    SetMaskMode(context, 3);
    context.inputTensorDesc_[14].shape_ = {{2}, {2}};
    context.inputTensorDesc_[15].shape_ = {{2}, {2}};
    context.inputTensorDesc_[18].shape_ = {{2, 37}, {2, 37}};
    TilingExpect expect;
    expect.sparseType = 4;
    expect.hasSequsedQ = true;
    expect.hasSequsedKv = true;
    expect.metadataLength = 37;
    expect.maskMode = 3;
    expect.s2Token = 0;
    CheckTiling(context, expect);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_quant_mode)
{
    for (int64_t mode : {-1, 1, 2}) {
        SCOPED_TRACE(mode);
        auto context = MakeTilingContext();
        context.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_mask_mode)
{
    for (int64_t mode : {-1, 1, 2, 5}) {
        SCOPED_TRACE(mode);
        auto context = MakeTilingContext();
        context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_window)
{
    for (size_t index : {3U, 4U}) {
        SCOPED_TRACE(index);
        auto denseContext = MakeTilingContext();
        denseContext.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(0);
        ExecuteTestCase(denseContext, ge::GRAPH_FAILED);
        auto bandContext = MakeTilingContext();
        SetMaskMode(bandContext, 4);
        bandContext.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(-2);
        ExecuteTestCase(bandContext, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_max_seqlen)
{
    for (size_t index : {5U, 6U}) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        context.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(512);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
        for (int64_t length : {0, -2}) {
            SCOPED_TRACE(length);
            auto tndContext = MakeTilingContext("TND");
            tndContext.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(length);
            ExecuteTestCase(tndContext, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_layout)
{
    for (size_t index : {7U, 8U}) {
        for (const std::string& layout : {"", "BSH", "BNSD", "bsnd"}) {
            SCOPED_TRACE(testing::Message() << "attr=" << index << ", layout=" << layout);
            auto context = MakeTilingContext();
            context.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>(layout);
            ExecuteTestCase(context, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_input_dtype)
{
    for (size_t index = 0; index < 12; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        context.inputTensorDesc_[index].dtype_ = ge::DT_FLOAT16;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_output_dtype)
{
    for (size_t index = 0; index < 3; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        context.outputTensorDesc_[index].dtype_ = ge::DT_FLOAT;
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_scale_shape)
{
    for (size_t index = 5; index <= 10; ++index) {
        SCOPED_TRACE(index);
        for (const gert::StorageShape& shape : {gert::StorageShape({{2}, {2}}), gert::StorageShape({{1, 1}, {1, 1}})}) {
            auto context = MakeTilingContext();
            context.inputTensorDesc_[index].shape_ = shape;
            ExecuteTestCase(context, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_head_dimension)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[0].shape_ = {{2, 512, 4, 64}, {2, 512, 4, 64}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_head_count_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[1].shape_ = {{2, 1024, 2, 128}, {2, 1024, 2, 128}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_empty_main_input)
{
    for (size_t index = 0; index < 5; ++index) {
        SCOPED_TRACE(index);
        auto context = MakeTilingContext();
        auto& shape = context.inputTensorDesc_[index].shape_;
        shape.MutableOriginShape().SetDim(1, 0);
        shape.MutableStorageShape().SetDim(1, 0);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_dout_sequence_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[3].shape_ = {{2, 511, 4, 128}, {2, 511, 4, 128}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_lse_shape_mismatch)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[11].shape_ = {{2, 4, 511}, {2, 4, 511}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
    auto tndContext = MakeTilingContext("TND");
    tndContext.inputTensorDesc_[11].shape_ = {{1024, 4}, {1024, 4}};
    ExecuteTestCase(tndContext, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_missing_metadata)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[18].shape_ = {{}, {}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_metadata)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[18].dtype_ = ge::DT_INT64;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
    for (const gert::StorageShape& shape : {gert::StorageShape({{32}, {32}}), gert::StorageShape({{1, 16}, {1, 16}}),
                                            gert::StorageShape({{2, 15}, {2, 15}})}) {
        auto invalidContext = MakeTilingContext();
        invalidContext.inputTensorDesc_[18].shape_ = shape;
        ExecuteTestCase(invalidContext, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_short_varlen_metadata)
{
    auto tndContext = MakeTilingContext("TND");
    tndContext.inputTensorDesc_[18].shape_ = {{2, 36}, {2, 36}};
    ExecuteTestCase(tndContext, ge::GRAPH_FAILED);
    auto denseContext = MakeTilingContext();
    denseContext.inputTensorDesc_[14].shape_ = {{2}, {2}};
    denseContext.inputTensorDesc_[18].shape_ = {{2, 22}, {2, 22}};
    ExecuteTestCase(denseContext, ge::GRAPH_FAILED);
    auto causalContext = MakeTilingContext();
    SetMaskMode(causalContext, 3);
    causalContext.inputTensorDesc_[15].shape_ = {{2}, {2}};
    causalContext.inputTensorDesc_[18].shape_ = {{2, 36}, {2, 36}};
    ExecuteTestCase(causalContext, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_cu_seqlens)
{
    for (size_t index : {12U, 13U}) {
        SCOPED_TRACE(index);
        auto missingContext = MakeTilingContext("TND");
        missingContext.inputTensorDesc_[index].shape_ = {{}, {}};
        ExecuteTestCase(missingContext, ge::GRAPH_FAILED);
        auto dtypeContext = MakeTilingContext("TND");
        dtypeContext.inputTensorDesc_[index].dtype_ = ge::DT_INT64;
        ExecuteTestCase(dtypeContext, ge::GRAPH_FAILED);
        auto lengthContext = MakeTilingContext("TND");
        lengthContext.inputTensorDesc_[index].shape_ = {{4}, {4}};
        ExecuteTestCase(lengthContext, ge::GRAPH_FAILED);
        auto paddedContext = MakeTilingContext();
        paddedContext.inputTensorDesc_[index].shape_ = {{3}, {3}};
        ExecuteTestCase(paddedContext, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_seqused)
{
    for (size_t index : {14U, 15U}) {
        SCOPED_TRACE(index);
        auto dtypeContext = MakeTilingContext();
        dtypeContext.inputTensorDesc_[index].shape_ = {{2}, {2}};
        dtypeContext.inputTensorDesc_[index].dtype_ = ge::DT_INT64;
        dtypeContext.inputTensorDesc_[18].shape_ = {{2, 23}, {2, 23}};
        ExecuteTestCase(dtypeContext, ge::GRAPH_FAILED);
        for (const gert::StorageShape& shape : {gert::StorageShape({{3}, {3}}), gert::StorageShape({{1, 2}, {1, 2}})}) {
            auto context = MakeTilingContext("TND");
            context.inputTensorDesc_[index].shape_ = shape;
            ExecuteTestCase(context, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_unsupported_sinks)
{
    auto context = MakeTilingContext();
    context.inputTensorDesc_[16].shape_ = {{4}, {4}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_mask_presence)
{
    auto denseContext = MakeTilingContext();
    denseContext.inputTensorDesc_[17].shape_ = {{2048, 2048}, {2048, 2048}};
    ExecuteTestCase(denseContext, ge::GRAPH_FAILED);
    for (int64_t mode : {3, 4}) {
        SCOPED_TRACE(mode);
        auto context = MakeTilingContext();
        context.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_mask_dtype)
{
    auto context = MakeTilingContext();
    SetMaskMode(context, 3);
    context.inputTensorDesc_[17].dtype_ = ge::DT_BOOL;
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradTilingTest, quant_flash_attn_grad_tiling_invalid_mask_shape)
{
    auto context = MakeTilingContext();
    SetMaskMode(context, 3);
    context.inputTensorDesc_[17].shape_ = {{1024, 2048}, {1024, 2048}};
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
