/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "../../../op_host/op_tiling/arch35/mhc_pre_backward_tiling.h"
#include <map>
#include <string>
#include "register/tilingdata_base.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "op_tiling_parse_context_builder.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/op_tiling/mhc_pre_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {

constexpr uint32_t N = 4;
constexpr uint32_t D = 64;
constexpr uint32_t FUSION_SIZE = N * N + 2 * N;

class MhcPreBackwardA5CoverageTilingTest : public testing::Test {};

TEST_F(MhcPreBackwardA5CoverageTilingTest, TndDenseResidual)
{
    constexpr uint32_t T = 8;
    constexpr uint32_t ND = N * D;
    optiling::MhcPreBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara context("MhcPreBackward",
                                    {
                                        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, FUSION_SIZE}, {T, FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{FUSION_SIZE}, {FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                     {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                    &compileInfo, "Ascend950");

    ExecuteTestCase(context, ge::GRAPH_SUCCESS, 0, "", {}, 0);
}

TEST_F(MhcPreBackwardA5CoverageTilingTest, RejectFactorialResidual)
{
    constexpr uint32_t T = 8;
    constexpr uint32_t ND = N * D;
    constexpr uint32_t FACTORIAL_SIZE = 24;
    optiling::MhcPreBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara context("MhcPreBackward",
                                    {
                                        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, FACTORIAL_SIZE}, {T, FACTORIAL_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, FUSION_SIZE}, {T, FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{FUSION_SIZE}, {FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                     {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
                                    &compileInfo);

    ExecuteTestCase(context, ge::GRAPH_FAILED, UINT64_MAX, "", {}, 0);
}

void CheckTiling(uint32_t T, uint32_t N, uint32_t D, int64_t mode, int mutation, ge::graphStatus expected,
                 uint64_t expectedKey = UINT64_MAX)
{
    const uint32_t ND = N * D;
    const uint32_t FUSION_SIZE = N * N + 2 * N;
    optiling::MhcPreBackwardCompileInfo compileInfo = {};
    gert::TilingContextPara context("MhcPreBackward",
                                    {
                                        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, D}, {T, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T}, {T}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, FUSION_SIZE}, {T, FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{T, N, D}, {T, N, D}}, ge::DT_BF16, ge::FORMAT_ND},
                                        {{{FUSION_SIZE, ND}, {FUSION_SIZE, ND}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{FUSION_SIZE}, {FUSION_SIZE}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{N, D}, {N, D}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {{"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                                     {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode)}},
                                    &compileInfo, "Ascend950");

    // 覆盖输入维度、序列匹配和平台资源约束。
    if (mutation == 1)
        context.inputTensorDesc_[4].shape_ = {{T + 1, N}, {T + 1, N}};
    if (mutation == 2)
        context.inputTensorDesc_[5].shape_ = {{T, N + 1, N}, {T, N + 1, N}};
    if (mutation == 3)
        context.inputTensorDesc_[3].shape_ = {{T}, {T}};
    if (mutation == 4)
        context.inputTensorDesc_[4].shape_ = {{T, 1, N}, {T, 1, N}};
    if (mutation == 5) {
        context.inputTensorDesc_[3].shape_ = {{2, T, D}, {2, T, D}};
        context.inputTensorDesc_[4].shape_ = {{2, T, N}, {2, T, N}};
        context.inputTensorDesc_[5].shape_ = {{2, T, N, N}, {2, T, N, N}};
    }
    if (mutation == 6) {
        context.inputTensorDesc_[3].shape_ = {{2, T, D}, {2, T, D}};
        context.inputTensorDesc_[4].shape_ = {{3, T, N}, {3, T, N}};
        context.inputTensorDesc_[5].shape_ = {{2, T, N, N}, {2, T, N, N}};
    }
    if (mutation == 7) {
        context.inputTensorDesc_[3].shape_ = {{2, T, D}, {2, T, D}};
        context.inputTensorDesc_[4].shape_ = {{2, T, N}, {2, T, N}};
        context.inputTensorDesc_[5].shape_ = {{2, T + 1, N, N}, {2, T + 1, N, N}};
    }
    if (expected == ge::GRAPH_SUCCESS) {
        ASSERT_NE(expectedKey, UINT64_MAX);
    }
    ExecuteTestCase(context, expected, expectedKey, "", {}, 0);

    if (expected != ge::GRAPH_SUCCESS) {
        return;
    }
    // 验收正常tiling入口实际生成的数据，而非只测试人工设置的字段。
    TilingInfo serialized;
    ASSERT_TRUE(ExecuteTiling(context, serialized));
    optiling::MhcPreBackwardTilingData schema;
    ASSERT_EQ(serialized.tilingDataSize, schema.GetDataSize());
    std::vector<uint8_t> buffer(serialized.tilingData.get(), serialized.tilingData.get() + serialized.tilingDataSize);
    optiling::MhcPreBackwardTilingData restored;
    // SDK的getter读取成员；绑定buffer不会反序列化成员，因此按schema解码后重建。
    auto checkField = [&](auto setter, auto getter, auto value) {
        using ValueType = decltype(value);
        const ValueType marker = static_cast<ValueType>(0x5A5AA5U);
        optiling::MhcPreBackwardTilingData fieldSchema;
        setter(fieldSchema, marker);
        std::vector<uint8_t> layout(fieldSchema.GetDataSize());
        fieldSchema.SaveToBuffer(layout.data(), layout.size());
        size_t fieldOffset = layout.size();
        size_t matches = 0;
        for (size_t offset = 0; offset + sizeof(ValueType) <= layout.size(); ++offset) {
            if (std::memcmp(layout.data() + offset, &marker, sizeof(ValueType)) == 0) {
                fieldOffset = offset;
                ++matches;
            }
        }
        ASSERT_EQ(matches, 1U);
        ValueType actual{};
        std::memcpy(&actual, serialized.tilingData.get() + fieldOffset, sizeof(actual));
        EXPECT_EQ(actual, value);
        setter(restored, actual);
        EXPECT_EQ(getter(restored), value);
        std::vector<uint8_t> saved(restored.GetDataSize());
        restored.SaveToBuffer(saved.data(), saved.size());
        ValueType savedValue{};
        std::memcpy(&savedValue, saved.data() + fieldOffset, sizeof(savedValue));
        EXPECT_EQ(savedValue, actual);
    };
    checkField([](auto& d, uint64_t v) { d.set_fusionSize(v); }, [](auto& d) { return d.get_fusionSize(); },
               uint64_t{FUSION_SIZE});
    checkField([](auto& d, uint64_t v) { d.set_totalLength(v); }, [](auto& d) { return d.get_totalLength(); },
               uint64_t{mutation == 5 ? 2U * T : T});
    checkField([](auto& d, uint64_t v) { d.set_nD(v); }, [](auto& d) { return d.get_nD(); }, uint64_t{ND});
    checkField([](auto& d, uint32_t v) { d.set_implMode(v); }, [](auto& d) { return d.get_implMode(); },
               static_cast<uint32_t>(mode));
    checkField([](auto& d, float v) { d.set_hcEps(v); }, [](auto& d) { return d.get_hcEps(); }, 1e-6f);
    checkField([](auto& d, uint32_t v) { d.set_c0MBlock(v); }, [](auto& d) { return d.get_c0MBlock(); }, uint32_t{512});
    checkField([](auto& d, uint32_t v) { d.set_c0NBlock(v); }, [](auto& d) { return d.get_c0NBlock(); }, uint32_t{128});
    checkField([](auto& d, uint32_t v) { d.set_c1KBlock(v); }, [](auto& d) { return d.get_c1KBlock(); }, uint32_t{512});
    checkField([](auto& d, uint32_t v) { d.set_c0ToV2Rows(v); }, [](auto& d) { return d.get_c0ToV2Rows(); },
               uint32_t{128});
    checkField([](auto& d, uint32_t v) { d.set_v2ToC1Rows(v); }, [](auto& d) { return d.get_v2ToC1Rows(); },
               uint32_t{512});
}

TEST_F(MhcPreBackwardA5CoverageTilingTest, SupportedShapesAndModes)
{
    // 模板参数按列表下标编码，HF32位在4位策略字段之后。
    constexpr uint64_t defaultKey = 0;
    constexpr uint64_t streamKey = 1;
    constexpr uint64_t tailKey = 2;
    constexpr uint64_t hf32Bit = 16;
    for (auto n : {4U, 6U, 8U}) {
        for (auto mode : {0, 1}) {
            for (auto t : {1U, 128U, 1024U, 3072U, 8192U, 65568U}) {
                SCOPED_TRACE(::testing::Message() << n << "/" << mode << "/" << t);
                uint64_t key = t < 512 ? defaultKey : streamKey;
                CheckTiling(t, n, 2560, mode, 0, ge::GRAPH_SUCCESS, key | (mode == 1 ? hf32Bit : 0));
            }
        }
    }
    CheckTiling(512, 4, 512, 1, 0, ge::GRAPH_SUCCESS, defaultKey | hf32Bit);
    CheckTiling(640, 4, 512, 0, 0, ge::GRAPH_SUCCESS, defaultKey);
    CheckTiling(1024, 4, 1536, 0, 0, ge::GRAPH_SUCCESS, defaultKey);
    CheckTiling(1024, 4, 1536, 1, 0, ge::GRAPH_SUCCESS, streamKey | hf32Bit);
    CheckTiling(2304, 4, 5120, 0, 0, ge::GRAPH_SUCCESS, tailKey);
    CheckTiling(2304, 4, 5120, 1, 0, ge::GRAPH_SUCCESS, tailKey | hf32Bit);
    CheckTiling(127, 4, 64, 0, 5, ge::GRAPH_SUCCESS, defaultKey);
}

TEST_F(MhcPreBackwardA5CoverageTilingTest, RejectInvalidDimensionsAndMode)
{
    CheckTiling(8, 5, 64, 0, 0, ge::GRAPH_FAILED);
    for (auto d : {0U, 63U, 16385U})
        CheckTiling(8, 4, d, 0, 0, ge::GRAPH_FAILED);
    CheckTiling(8, 4, 64, -1, 0, ge::GRAPH_FAILED);
    CheckTiling(8, 4, 64, 2, 0, ge::GRAPH_FAILED);
    for (int mutation : {1, 2, 3, 4, 6, 7}) {
        SCOPED_TRACE(mutation);
        CheckTiling(8, 4, 64, 0, mutation, ge::GRAPH_FAILED);
    }
}

// 通过公开工厂验证本算子的主tiling数据和嵌套MMConfig注册可创建。
TEST_F(MhcPreBackwardA5CoverageTilingTest, RegisteredTilingDataFactories)
{
    auto& factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("MhcPreBackward"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("MMConfigOp"), nullptr);
}

TEST_F(MhcPreBackwardA5CoverageTilingTest, ParseRejectsNullContext)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("MhcPreBackward");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    EXPECT_NE(impl->tiling_parse(nullptr), ge::GRAPH_SUCCESS);
}

// 通过注册入口构造真实平台解析上下文，验证资源传入compileInfo。
TEST_F(MhcPreBackwardA5CoverageTilingTest, ParsePlatformResources)
{
    optiling::MhcPreBackwardCompileInfo compileInfo{};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socResources{
        {"ai_core_cnt", "32"}, {"vector_core_cnt", "64"}, {"l2_size", "134217728"}, {"core_type_list", "AICore"}};
    platformInfo.SetPlatformRes("SoCInfo", socResources);
    std::map<std::string, std::string> coreResources{{"ub_size", "262144"},
                                                     {"l1_size", "524288"},
                                                     {"l0_a_size", "65536"},
                                                     {"l0_b_size", "65536"},
                                                     {"l0_c_size", "262144"}};
    platformInfo.SetPlatformRes("AICoreSpec", coreResources);
    auto holder = gert::OpTilingParseContextBuilder()
                      .OpType("MhcPreBackward")
                      .OpName("MhcPreBackward")
                      .IONum(12, 5)
                      .CompiledJson("{}")
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(&platformInfo)
                      .Build();
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("MhcPreBackward");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    EXPECT_EQ(impl->tiling_parse(reinterpret_cast<gert::KernelContext*>(holder.GetContext())), ge::GRAPH_SUCCESS);
    EXPECT_EQ(compileInfo.ubSize, 262144U);
    EXPECT_EQ(compileInfo.l1Size, 524288U);
}

} // namespace
