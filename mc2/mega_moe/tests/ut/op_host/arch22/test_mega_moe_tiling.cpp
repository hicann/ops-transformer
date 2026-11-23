/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <limits>
#include <utility>

#include <gtest/gtest.h>
#include <initializer_list>
#include "experiment/metadef/common/util/error_manager/error_manager.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "mc2_tiling_case_executor.h"

#include "../../../../op_kernel/arch22/mega_moe_tiling_a2a3.h"

namespace MegaMoeUT {

namespace {
template <typename Reports>
void CheckFormalReport(const Reports& reports, const std::string& errorId,
                       std::initializer_list<std::pair<std::string, std::string>> exactFields,
                       std::initializer_list<std::string> nonEmptyFields = {},
                       std::initializer_list<std::pair<std::string, std::string>> containsFields = {})
{
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_EQ(reports[0].error_id, errorId);
    const auto& fields = reports[0].args_map;
    for (const auto& item : exactFields) {
        ASSERT_EQ(fields.count(item.first), 1U) << item.first;
    }
    for (const auto& key : nonEmptyFields) {
        ASSERT_EQ(fields.count(key), 1U) << key;
    }
    for (const auto& item : containsFields) {
        ASSERT_EQ(fields.count(item.first), 1U) << item.first;
    }
    for (const auto& item : exactFields) {
        EXPECT_EQ(fields.at(item.first), item.second) << item.first;
    }
    for (const auto& key : nonEmptyFields) {
        EXPECT_FALSE(fields.at(key).empty()) << key;
    }
    for (const auto& item : containsFields) {
        EXPECT_NE(fields.at(item.first).find(item.second), std::string::npos) << item.first;
    }
}

} // namespace

class MegaMoeArch22TilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
    }
    void TearDown() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
    }

    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch22TilingTest TearDown" << std::endl;
    }
};

static void RunQuantSharedExpertCase(
    ge::DataType weightDtype, bool provideSharedScales, bool provideSharedBiases, ge::graphStatus expectedStatus,
    int64_t sharedScale1Dim = 1024, int64_t sharedScale2Dim = 1024, bool sharedGmm2Bf16 = false,
    bool provideBf16Scale2 = false, bool heterogeneousSharedW2 = false, ge::Format weightFormat = ge::FORMAT_FRACTAL_NZ,
    const char* socVersion = "Ascend910_93", int64_t kBs = 1, bool withActiveMask = false, int64_t hiddenSize = 1024,
    int64_t intermediate = 512, int64_t topk = 2, int64_t ep = 2, uint32_t expertPerRank = 2, uint32_t sharedCount = 2,
    ge::DataType sharedDtype = ge::DT_UNDEFINED, int64_t sharedIntermediate = 0, bool sharedDownNd = false,
    bool heterogeneousSharedFormat = false, int64_t combineCommMode = 0, ge::DataType topkWeightsDtype = ge::DT_FLOAT,
    TilingInfo* tilingInfo = nullptr, int64_t cclBufferSize = 2LL * 1024 * 1024 * 1024, int64_t a3RecvBudget = 4)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 24;
    constexpr uint64_t ubSize = 196608;
    const int64_t kHiddenSize = hiddenSize;
    const int64_t kN = intermediate * 2;
    const int64_t kIntermediateHidden = kN / 2;
    const uint32_t kExpertPerRank = expertPerRank;
    const uint32_t kSharedExpertNum = sharedCount;
    const int64_t sharedN = 2 * (sharedIntermediate ? sharedIntermediate : intermediate);
    if (sharedDtype == ge::DT_UNDEFINED)
        sharedDtype = weightDtype;
    const bool isInt4 = weightDtype == ge::DT_INT4;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    inputs.push_back({{{kBs, topk}, {kBs, topk}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, topk}, {kBs, topk}}, topkWeightsDtype, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, weightDtype, weightFormat});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back(
            {{{kIntermediateHidden, kHiddenSize}, {kIntermediateHidden, kHiddenSize}}, weightDtype, weightFormat});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    if (isInt4) {
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kN}, {kN}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }
    } else {
        inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});
        inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    if (withActiveMask) {
        inputs.push_back({{{kBs}, {kBs}}, ge::DT_INT8, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
        inputs.push_back({{{kHiddenSize, sharedN}, {kHiddenSize, sharedN}}, sharedDtype, weightFormat});
    }
    for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
        inputs.push_back({{{sharedN / 2, kHiddenSize}, {sharedN / 2, kHiddenSize}},
                          sharedGmm2Bf16 && !(heterogeneousSharedW2 && i == 1U) ? ge::DT_BF16 : sharedDtype,
                          sharedDownNd && !(heterogeneousSharedFormat && i == 1U) ? ge::FORMAT_ND : weightFormat});
    }
    if (provideSharedScales) {
        for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
            inputs.push_back({{{sharedScale1Dim}, {sharedScale1Dim}}, ge::DT_UINT64, ge::FORMAT_ND});
        }
        if (!sharedGmm2Bf16 || provideBf16Scale2) {
            for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
                inputs.push_back({{{sharedScale2Dim}, {sharedScale2Dim}}, ge::DT_UINT64, ge::FORMAT_ND});
            }
        } else {
            inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND});
        }
    } else {
        inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND});
        inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    if (provideSharedBiases) {
        for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
            inputs.push_back({{{sharedN}, {sharedN}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }
        for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
            inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }
    } else {
        inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});
        inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    const uint32_t routedBiasListLen = isInt4 ? kExpertPerRank : 1U;
    const uint32_t sharedScaleListLen = provideSharedScales ? kSharedExpertNum : 1U;
    const uint32_t sharedBiasListLen = provideSharedBiases ? kSharedExpertNum : 1U;
    gert::TilingContextPara tilingContextPara(
        "MegaMoe", inputs,
        {
            {{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kExpertPerRank * ep)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
         {"max_recv_token_num",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(std::string(socVersion) == "Ascend910B" ? 0 : a3RecvBudget)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"shared_expert_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"combine_comm_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(combineCommMode)}},
        {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, routedBiasListLen,
         routedBiasListLen, withActiveMask ? 1U : 0U, 0, kSharedExpertNum, kSharedExpertNum, sharedScaleListLen,
         sharedGmm2Bf16 && !provideBf16Scale2 ? 1U : sharedScaleListLen, sharedBiasListLen, sharedBiasListLen},
        {1, 1}, &compileInfo, socVersion, coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", ep}};
    if (tilingInfo != nullptr) {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        const bool ok = ExecuteTiling(tilingContextPara, *tilingInfo);
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        ASSERT_TRUE(ok);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, UINT64_MAX);
    }
}

TEST_F(MegaMoeArch22TilingTest, A3SharedWorkspaceAlignmentWithBf16TopkWeights)
{
    TilingInfo info;
    RunQuantSharedExpertCase(ge::DT_INT4, true, false, ge::GRAPH_SUCCESS, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_INT8, 0,
                             false, false, 0, ge::DT_BF16, &info);
    ASSERT_EQ(info.tilingDataSize, sizeof(MegaMoeTilingDataQuant));
    const auto* data = reinterpret_cast<const MegaMoeTilingDataQuant*>(info.tilingData.get());
    ASSERT_NE(data, nullptr);
    ASSERT_GT(data->common.sharedWorkspaceOffset, 0U);
    EXPECT_EQ(data->common.sharedWorkspaceOffset % 512U, 0U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GT(info.workspaceSizes[0], data->common.sharedWorkspaceOffset);
}

TEST_F(MegaMoeArch22TilingTest, A3SharedW8AccumulatorWorkspace)
{
    // Two shared experts, M=6, H=1024, I=1536. Both matmuls reuse the
    // larger C allocation. Each scale buffer has its own 512-byte slot.
    constexpr uint64_t systemBytes = 16U * 1024U * 1024U;
    for (bool mixedDown : {false, true}) {
        TilingInfo info;
        RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_SUCCESS, 3072, 1024, mixedDown, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 6, true, 1024, 1536, 1, 2, 8, 2, ge::DT_INT8,
                                 1536, false, false, 0, ge::DT_FLOAT, &info);
        ASSERT_EQ(info.tilingDataSize, sizeof(MegaMoeTilingDataQuant));
        const auto* data = reinterpret_cast<const MegaMoeTilingDataQuant*>(info.tilingData.get());
        ASSERT_NE(data, nullptr);
        ASSERT_EQ(info.workspaceSizes.size(), 1U);
        EXPECT_EQ(data->common.sharedWorkspaceOffset % 512U, 0U);
        const uint64_t cBytes = 2U * 6U * 3072U * (mixedDown ? sizeof(int16_t) : sizeof(int32_t));
        const uint64_t activationBytes = 2U * 6U * 1536U * (mixedDown ? sizeof(int16_t) : sizeof(int8_t));
        constexpr uint64_t inputBytes = 6U * 1024U;
        constexpr uint64_t resultBytes = 2U * 6U * 1024U * sizeof(int16_t);
        EXPECT_EQ(info.workspaceSizes[0], systemBytes + data->common.sharedWorkspaceOffset + cBytes + activationBytes +
                                              inputBytes + 2U * 512U + resultBytes);
    }
}

TEST_F(MegaMoeArch22TilingTest, Test0)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kNHalf = 14336;
    constexpr uint32_t kExpertPerRank = 8;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{32, kHiddenSize}, {32, kHiddenSize}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    inputs.push_back({{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_INT4, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kNHalf, kHiddenSize}, {kNHalf, kHiddenSize}}, ge::DT_INT4, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_UINT64, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kN}, {kN}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; i++) {
        inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara(
        "MegaMoe", inputs,
        {
            {{{32, kHiddenSize}, {32, kHiddenSize}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"dispatch_quant_mode",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // INT4/INT8 -> dispatch_quant_mode=2
         {"dispatch_quant_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // INT4/INT8 -> dispatch_quant_out_dtype=INT8
         {"shared_expert_quant_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params",
          Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank},
        {1, 1}, &compileInfo, "Ascend910B", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
}

TEST_F(MegaMoeArch22TilingTest, ActivationParamValidationWithLargeIntermediateHidden)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 20;
    constexpr uint64_t ubSize = 196608;
    constexpr int64_t kBs = 32;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kIntermediateHidden = kN / 2;
    constexpr uint32_t kExpertPerRank = 8;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back(
            {{{kIntermediateHidden, kHiddenSize}, {kIntermediateHidden, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    }

    struct ActivationCase {
        std::string activation;
        std::vector<float> params;
        ge::graphStatus expectedStatus;
        const char* expectedCode = "EZ0026";
    };
    const std::vector<ActivationCase> activationCases = {
        {"swiglu", {}, ge::GRAPH_SUCCESS},
        {"swiglustep", {}, ge::GRAPH_SUCCESS},
        {"swigluoai", {}, ge::GRAPH_SUCCESS},
        {"situglu", {}, ge::GRAPH_SUCCESS},
        {"swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS},
        {"swiglu", {0.0f}, ge::GRAPH_SUCCESS},
        {"swiglustep", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS},
        {"swigluoai", {std::numeric_limits<float>::max(), 0.0f, 0.0f}, ge::GRAPH_SUCCESS},
        {"swigluoai", {std::numeric_limits<float>::max(), -1.702f, -1.0f}, ge::GRAPH_SUCCESS},
        {"situglu", {2.0f}, ge::GRAPH_SUCCESS},
        {"situglu", {2.0f, 2.0f}, ge::GRAPH_SUCCESS},
        {"swiglu", {-1.0f}, ge::GRAPH_FAILED},
        {"swiglu", {1.0f, 2.0f}, ge::GRAPH_FAILED, "EZ0038"},
        {"swigluoai",
         {std::numeric_limits<float>::max(), std::numeric_limits<float>::quiet_NaN(), 1.0f},
         ge::GRAPH_FAILED},
        {"swigluoai",
         {std::numeric_limits<float>::max(), 1.702f, std::numeric_limits<float>::infinity()},
         ge::GRAPH_FAILED},
        {"situglu", {0.0f}, ge::GRAPH_FAILED},
        {"situglu", {1.0f, 0.0f}, ge::GRAPH_FAILED},
        {"situglu", {std::numeric_limits<float>::quiet_NaN()}, ge::GRAPH_FAILED},
        {"situglu", {1.0f, std::numeric_limits<float>::infinity()}, ge::GRAPH_FAILED},
    };
    for (const auto& activationCase : activationCases) {
        SCOPED_TRACE(activationCase.activation + " params=" + std::to_string(activationCase.params.size()));
        ErrorManager::GetInstance().GetRawErrorMessages();
        gert::TilingContextPara tilingContextPara(
            "MegaMoe", inputs,
            {
                {{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
            },
            {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
             {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2LL * 1024 * 1024 * 1024)},
             {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
             {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"dispatch_quant_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"shared_expert_quant_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
             {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>(activationCase.activation)},
             {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(activationCase.params)},
             {"activation_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
            {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, 0, 0, 0, 0, 0, 0}, {1, 1}, &compileInfo, "Ascend910B", coreNum,
            ubSize);

        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, activationCase.expectedStatus, UINT64_MAX);
        const auto reports = ErrorManager::GetInstance().GetRawErrorMessages();
        if (activationCase.expectedStatus == ge::GRAPH_FAILED) {
            const char* actualKey =
                std::string(activationCase.expectedCode) == "EZ0038" ? "incorrect_size" : "incorrect_value";
            CheckFormalReport(reports, activationCase.expectedCode, {{"op_name", "MegaMoe"}}, {actualKey, "reason"},
                              {{"param_name", "activation_params"}});
        } else {
            EXPECT_TRUE(reports.empty());
        }
        if (activationCase.activation == "swiglu" && activationCase.params.empty()) {
            const auto checkFailure = [&](const gert::TilingContextPara& invalid, const char* code, const char* field,
                                          const char* expected) {
                SCOPED_TRACE(code);
                ErrorManager::GetInstance().GetRawErrorMessages();
                Mc2ExecuteTestCase(invalid, hcomTopologyMockValues, ge::GRAPH_FAILED, UINT64_MAX);
                const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
                CheckFormalReport(errors, code, {{field, expected}});
            };
            auto missingAttr = tilingContextPara;
            missingAttr.attrs_.pop_back();
            checkFailure(missingAttr, "EZ0004", "param_name", "weight1_interleave");

            auto mismatchedTranspose = tilingContextPara;
            for (auto& attr : mismatchedTranspose.attrs_) {
                if (attr.attrName_ == "transpose_weight1") {
                    attr.attr_ = Ops::Transformer::AnyValue::CreateFrom<bool>(true);
                }
            }
            checkFailure(mismatchedTranspose, "EZ0027", "param_names", "transpose_weight1, transpose_weight2");

            auto maskCase = tilingContextPara;
            maskCase.inputInstanceNum_.resize(19, 0);
            maskCase.inputInstanceNum_[18] = 1;
            int32_t maskValues[8] = {};
            maskCase.inputTensorDesc_.push_back({{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND, true, maskValues});
            checkFailure(maskCase, "EZ0037", "param_name", "mask_buffer");
            maskCase.socVersion_ = "Ascend910_93";
            maskCase.inputTensorDesc_.back().shape_ = gert::StorageShape({7}, {7});
            checkFailure(maskCase, "EZ0009", "param_name", "mask_buffer");
            maskCase.inputTensorDesc_.back().shape_ = gert::StorageShape({8}, {8});
            maskCase.inputTensorDesc_.back().dtype_ = ge::DT_FLOAT;
            checkFailure(maskCase, "EZ0019", "param_name", "mask_buffer");
        }
    }
}

static void RunBf16SharedExpertCase(int64_t kBs, bool withRankMask = false, const char* socVersion = "Ascend910_93",
                                    int64_t sharedIntermediate = 0)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 24;
    constexpr uint64_t ubSize = 196608;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 6144;
    const int64_t kIntermediateHidden = kN / 2;
    constexpr uint32_t kExpertPerRank = 8;
    constexpr uint32_t kSharedExpertNum = 2;
    const int64_t sharedN = sharedIntermediate > 0 ? sharedIntermediate * 2 : kN;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back(
            {{{kIntermediateHidden, kHiddenSize}, {kIntermediateHidden, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // weight_scales1 placeholder
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // weight_scales2 placeholder
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // bias1 placeholder
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // bias2 placeholder
    for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
        inputs.push_back({{{kHiddenSize, sharedN}, {kHiddenSize, sharedN}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kSharedExpertNum; ++i) {
        inputs.push_back({{{sharedN / 2, kHiddenSize}, {sharedN / 2, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // shared_weight_scales1 placeholder
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // shared_weight_scales2 placeholder
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // shared_bias1 placeholder
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // shared_bias2 placeholder
    if (withRankMask) {
        inputs.push_back({{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara(
        "MegaMoe", inputs,
        {
            {{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2LL * 1024 * 1024 * 1024)},
         {"max_recv_token_num",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(std::string(socVersion) == "Ascend910B" ? 0 : 64)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"dispatch_quant_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"shared_expert_quant_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, 1, 1, 1, 1, 0, 0, kSharedExpertNum, kSharedExpertNum, 1, 1, 1, 1,
         withRankMask ? 1U : 0U},
        {1, 1}, &compileInfo, socVersion, coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(MegaMoeArch22TilingTest, Bf16SharedExpertsOnAscend910_93)
{
    RunBf16SharedExpertCase(32);
}

TEST_F(MegaMoeArch22TilingTest, Bf16SharedExpertsAcrossChunksAndRecvRounds)
{
    // M > 1024 with a tail; max_recv_token_num=64 forces receive-round workspace.
    RunBf16SharedExpertCase(1025);
}

TEST_F(MegaMoeArch22TilingTest, Bf16SharedExpertsWithRankMask)
{
    RunBf16SharedExpertCase(1025, true);
}

TEST_F(MegaMoeArch22TilingTest, Bf16SharedExpertsWithIndependentIntermediate)
{
    RunBf16SharedExpertCase(33, false, "Ascend910_93", 512);
}

TEST_F(MegaMoeArch22TilingTest, MixedSharedExpertsWithIndependentIntermediate)
{
    for (bool downNd : {false, true}) {
        RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_SUCCESS, 1536, 1024, true, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 33, false, 1024, 512, 2, 2, 2, 2, ge::DT_INT8,
                                 768, downNd);
    }
}

TEST_F(MegaMoeArch22TilingTest, Bf16NoSharedExpertsWithAclnnPlaceholdersOnAscend910_93)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 24;
    constexpr uint64_t ubSize = 196608;
    constexpr int64_t kBs = 32;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 6144;
    constexpr int64_t kIntermediateHidden = kN / 2;
    constexpr uint32_t kExpertPerRank = 8;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_INT32, ge::FORMAT_ND});
    inputs.push_back({{{kBs, 8}, {kBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND});
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_BF16, ge::FORMAT_ND});
    }
    for (uint32_t i = 0; i < kExpertPerRank; ++i) {
        inputs.push_back(
            {{{kIntermediateHidden, kHiddenSize}, {kIntermediateHidden, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
    }

    // aclnn uses one shape={0} tensor for every omitted DynamicInput so the inner op can
    // preserve the input group and its supported dtype/format.
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // weight_scales1
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // weight_scales2
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // bias1
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // bias2
    inputs.push_back({{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND});   // shared_weight1
    inputs.push_back({{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND});   // shared_weight2
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // shared_weight_scales1
    inputs.push_back({{{0}, {0}}, ge::DT_UINT64, ge::FORMAT_ND}); // shared_weight_scales2
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // shared_bias1
    inputs.push_back({{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND});  // shared_bias2

    gert::TilingContextPara tilingContextPara(
        "MegaMoe", inputs,
        {
            {{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2LL * 1024 * 1024 * 1024)},
         {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"dispatch_quant_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"shared_expert_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
         {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
         {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({})},
         {"activation_out_dtype",
          Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1}, {1, 1}, &compileInfo,
        "Ascend910_93", coreNum, ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

TEST_F(MegaMoeArch22TilingTest, SharedExpertsRejectAscend910B)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910B");
}

TEST_F(MegaMoeArch22TilingTest, Int8SharedExpertsOnAscend910_93)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch22TilingTest, Int8Bf16SharedExpertsNz)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_SUCCESS, 1024, 1024, true);
}

TEST_F(MegaMoeArch22TilingTest, Int8Bf16SharedExpertsRejectNdRouting)
{
    // Routed quantized weights require NZ even though the kernel has ND specializations.
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, true, false, false, ge::FORMAT_ND);
}

TEST_F(MegaMoeArch22TilingTest, Int8Bf16SharedExpertsRequireScale1)
{
    RunQuantSharedExpertCase(ge::DT_INT8, false, false, ge::GRAPH_FAILED, 1024, 1024, true);
}

TEST_F(MegaMoeArch22TilingTest, Int8Bf16SharedExpertsRejectScale2)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, true, true);
}

TEST_F(MegaMoeArch22TilingTest, Int8Bf16SharedExpertsRejectBias)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, true, ge::GRAPH_FAILED, 1024, 1024, true);
}

TEST_F(MegaMoeArch22TilingTest, Int8Bf16SharedExpertsRejectHeterogeneousWeight2)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, true, false, true);
}

TEST_F(MegaMoeArch22TilingTest, Int4Bf16SharedExpertsRemainUnsupported)
{
    RunQuantSharedExpertCase(ge::DT_INT4, true, true, ge::GRAPH_FAILED, 1024, 1024, true);
}

TEST_F(MegaMoeArch22TilingTest, Int4SharedExpertsOnAscend910_93)
{
    RunQuantSharedExpertCase(ge::DT_INT4, true, true, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch22TilingTest, Int8SharedExpertsRequireScales)
{
    RunQuantSharedExpertCase(ge::DT_INT8, false, false, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch22TilingTest, Int4SharedExpertsRequireBiases)
{
    RunQuantSharedExpertCase(ge::DT_INT4, true, false, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch22TilingTest, QuantSharedExpertsRejectWrongScale1Shape)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 512, 1024);
}

TEST_F(MegaMoeArch22TilingTest, QuantSharedExpertsRejectWrongScale2Shape)
{
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 512);
}

// Final A3 support contract: both shared weight dtype and intermediate are independent.
TEST_F(MegaMoeArch22TilingTest, FullRangeA3EP16MinAndMax)
{
    for (auto dtype : {ge::DT_INT8, ge::DT_INT4}) {
        RunQuantSharedExpertCase(dtype, true, dtype == ge::DT_INT4, ge::GRAPH_SUCCESS, 1024, 1024, false, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 1, 16, 4, 2);
        RunQuantSharedExpertCase(dtype, true, dtype == ge::DT_INT4, ge::GRAPH_SUCCESS, 6144, 8192, false, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 8192, 3072, 16, 16, 4, 2);
    }
}

TEST_F(MegaMoeArch22TilingTest, FullRangeA3IndependentDtypeAndIntermediate)
{
    for (auto route : {ge::DT_INT8, ge::DT_INT4}) {
        for (auto shared : {ge::DT_INT8, ge::DT_INT4}) {
            RunQuantSharedExpertCase(route, true, shared == ge::DT_INT4, ge::GRAPH_SUCCESS, 1536, 1536, false, false,
                                     false, ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1536, 512, 16, 16, 4, 2,
                                     shared, 768);
        }
    }
}

TEST_F(MegaMoeArch22TilingTest, Int4SharedExpertsWithMultipleRows)
{
    for (auto routed : {ge::DT_INT8, ge::DT_INT4}) {
        for (int64_t tokens : {2, 31, 32, 33}) {
            RunQuantSharedExpertCase(routed, true, true, ge::GRAPH_SUCCESS, 1536, 1024, false, false, false,
                                     ge::FORMAT_FRACTAL_NZ, "Ascend910_93", tokens, false, 1024, 512, 2, 2, 2, 2,
                                     ge::DT_INT4, 768);
        }
    }
}

TEST_F(MegaMoeArch22TilingTest, FullRangeA3RejectsOutsideContract)
{
    for (int64_t badH : {512, 1152, 8704}) {
        RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, badH, false, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, badH, 512, 2, 16, 4, 1);
    }
    for (int64_t badI : {256, 640}) {
        RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, badI, 2, 16, 4, 1);
        RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 2 * badI, 1024, false, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 16, 4, 1, ge::DT_INT8,
                                 badI);
    }
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 17, 16, 4, 1);
    for (uint32_t sharedCount : {3U, 4U}) {
        RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                                 ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 16, 4, sharedCount);
    }
}

TEST_F(MegaMoeArch22TilingTest, W4RoutingW8SharedOverlapShapes)
{
    // Both FFNs have two quantized layers. Only routed weights need MSD biases.
    for (uint32_t sharedCount : {0U, 1U, 2U}) {
        for (int64_t tokens : {1, 1025}) {
            for (int64_t sharedI : {512, 768, 1024}) {
                RunQuantSharedExpertCase(ge::DT_INT4, sharedCount > 0U, false, ge::GRAPH_SUCCESS, sharedI * 2, 1024,
                                         false, false, false, ge::FORMAT_FRACTAL_NZ, "Ascend910_93", tokens, true, 1024,
                                         512, 2, 2, 2, sharedCount, ge::DT_INT8, sharedI);
            }
        }
    }
}

TEST_F(MegaMoeArch22TilingTest, W4RoutingW8SharedRejectsMissingScalesAndSharedBias)
{
    RunQuantSharedExpertCase(ge::DT_INT4, false, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_INT8);
    RunQuantSharedExpertCase(ge::DT_INT4, true, true, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_INT8);
    RunQuantSharedExpertCase(ge::DT_INT4, true, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910B", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_INT8);
}

TEST_F(MegaMoeArch22TilingTest, A3MixedNzNdAndLegacyNzNz)
{
    for (uint32_t count : {1U, 2U}) {
        for (bool nd : {false, true}) {
            for (int64_t bs : {1, 1025}) {
                RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_SUCCESS, 1024, 1024, true, false, false,
                                         ge::FORMAT_FRACTAL_NZ, "Ascend910_93", bs, false, 1024, 512, 2, 2, 2, count,
                                         ge::DT_UNDEFINED, 0, nd);
            }
        }
    }
}

TEST_F(MegaMoeArch22TilingTest, RejectUnsupportedSharedDownFormats)
{
    // Quantized down still requires NZ; A2 is not extended; each expert uses
    // the same format; mixed BF16 down must not receive a quantization scale.
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_UNDEFINED,
                             0, true);
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, true, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910B", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_UNDEFINED, 0,
                             true);
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, true, false, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_UNDEFINED,
                             0, true, true);
    RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_FAILED, 1024, 1024, true, true, false,
                             ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1, false, 1024, 512, 2, 2, 2, 2, ge::DT_UNDEFINED,
                             0, true);
}
TEST_F(MegaMoeArch22TilingTest, CombineCommModeValidation)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 20;
    constexpr uint64_t ubSize = 196608;
    constexpr int64_t kBs = 32;
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kNHalf = 14336;
    constexpr uint32_t kExpertPerRank = 8;
    constexpr int64_t kEpWorldSize = 8;
    constexpr int64_t kTopK = 8;
    // 单轮接收界 = bs * epWorldSize * min(topK, expertPerRank)，取等号即可行
    constexpr int64_t kFeasibleMaxRecv = kBs * kEpWorldSize * kTopK;
    constexpr int64_t kSufficientCclBuffer = 2LL * 1024 * 1024 * 1024;

    struct DedupCase {
        std::string name;
        std::string socVersion;
        int64_t combineCommMode;
        bool isW4A8;
        int64_t maxRecvTokenNum;
        ge::graphStatus expectedStatus;
    };
    // combine 去重开关校验：值域 {0,1}；开启需 A3 + W4A8 路径（能力边界，显式报错）；
    // 单轮不满足时自动回基线（成功、Dedup 字段全 0）；各控制组（ccm=0）须成功。
    const std::vector<DedupCase> dedupCases = {
        {"a3_w4a8_ccm0_baseline", "Ascend910_93", 0, true, kFeasibleMaxRecv, ge::GRAPH_SUCCESS},
        {"a3_w4a8_ccm1_dedup_enabled", "Ascend910_93", 1, true, kFeasibleMaxRecv, ge::GRAPH_SUCCESS},
        {"a3_w4a8_ccm2_rejected", "Ascend910_93", 2, true, kFeasibleMaxRecv, ge::GRAPH_FAILED},
        {"a3_w4a8_ccm1_single_round_fallback_baseline", "Ascend910_93", 1, true, 64, ge::GRAPH_SUCCESS},
        {"a3_bf16_ccm0_baseline", "Ascend910_93", 0, false, kFeasibleMaxRecv, ge::GRAPH_SUCCESS},
        {"a3_bf16_ccm1_rejected", "Ascend910_93", 1, false, kFeasibleMaxRecv, ge::GRAPH_FAILED},
        {"a2_w4a8_ccm0_baseline", "Ascend910B", 0, true, kFeasibleMaxRecv, ge::GRAPH_SUCCESS},
        {"a2_w4a8_ccm1_rejected", "Ascend910B", 1, true, kFeasibleMaxRecv, ge::GRAPH_FAILED},
    };

    for (const auto& dedupCase : dedupCases) {
        std::vector<gert::TilingContextPara::TensorDescription> inputs;
        inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
        inputs.push_back({{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
        inputs.push_back({{{kBs, kTopK}, {kBs, kTopK}}, ge::DT_INT32, ge::FORMAT_ND});
        inputs.push_back({{{kBs, kTopK}, {kBs, kTopK}}, ge::DT_FLOAT, ge::FORMAT_ND});
        // INT4/INT8 权重按真实 GE 路径为分形布局（CheckWeight1/2Input 格式门），BF16 为 ND
        const ge::Format weightFormat = dedupCase.isW4A8 ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND;
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back(
                {{{kHiddenSize, kN}, {kHiddenSize, kN}}, dedupCase.isW4A8 ? ge::DT_INT4 : ge::DT_BF16, weightFormat});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kNHalf, kHiddenSize}, {kNHalf, kHiddenSize}},
                              dedupCase.isW4A8 ? ge::DT_INT4 : ge::DT_BF16,
                              weightFormat});
        }
        std::vector<uint32_t> inputNumList = {1, 1, 1, 1, kExpertPerRank, kExpertPerRank};
        if (dedupCase.isW4A8) {
            for (uint32_t i = 0; i < kExpertPerRank; ++i) {
                inputs.push_back({{{kN}, {kN}}, ge::DT_UINT64, ge::FORMAT_ND});
            }
            for (uint32_t i = 0; i < kExpertPerRank; ++i) {
                inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_UINT64, ge::FORMAT_ND});
            }
            for (uint32_t i = 0; i < kExpertPerRank; ++i) {
                inputs.push_back({{{kN}, {kN}}, ge::DT_FLOAT, ge::FORMAT_ND});
            }
            for (uint32_t i = 0; i < kExpertPerRank; ++i) {
                inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
            }
            inputNumList.insert(inputNumList.end(), {kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank});
        } else {
            inputNumList.insert(inputNumList.end(), {0, 0, 0, 0, 0, 0});
        }

        gert::TilingContextPara tilingContextPara(
            "MegaMoe", inputs,
            {
                {{{kBs, kHiddenSize}, {kBs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
            },
            {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kEpWorldSize)},
             {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSufficientCclBuffer)},
             {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dedupCase.maxRecvTokenNum)},
             {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dedupCase.isW4A8 ? 2 : 0)},
             {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(
                                              dedupCase.isW4A8 ? 2 : static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"shared_expert_quant_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
             {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
             {"activation_params",
              Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
             {"activation_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             // 16-18 位占位：combine_comm_mode 须落在 tiling 读取的索引 19
             {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
             {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"combine_comm_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dedupCase.combineCommMode)}},
            inputNumList, {1, 1}, &compileInfo, dedupCase.socVersion, coreNum, ubSize);

        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", kEpWorldSize}};
        std::cout << "CombineCommModeValidation case: " << dedupCase.name << std::endl;
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, dedupCase.expectedStatus, UINT64_MAX);
    }
}

// ccm=0/1 同形孪生断言 Dedup 字段透传与 workspace 增量（期望值独立硬编码，不随 dedup_layout 漂移）：
//   FP/LP = AlignUp(EP·bs·4, 32) = 416；B = min(1024·EP·min(topK,epr), maxRecv) = 784，
//   rowMeta = B×32 = 25088；增量 = 416×2 + 25088 = 25920。
TEST_F(MegaMoeArch22TilingTest, CombineCommModeDedupTilingEffect)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr uint64_t coreNum = 20;
    constexpr uint64_t ubSize = 196608;
    constexpr int64_t kBs = 49; // EP·bs=98 非 8 倍数，覆盖 32B 对齐垫量
    constexpr int64_t kHiddenSize = 7168;
    constexpr int64_t kN = 28672;
    constexpr int64_t kNHalf = 14336;
    constexpr uint32_t kExpertPerRank = 8;
    constexpr int64_t kEpWorldSize = 2;
    constexpr int64_t kTopK = 8;
    constexpr int64_t kSufficientCclBuffer = 2LL * 1024 * 1024 * 1024;

    auto buildPara = [&](int64_t combineCommMode, int64_t bs) {
        std::vector<gert::TilingContextPara::TensorDescription> inputs;
        inputs.push_back({{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND});
        inputs.push_back({{{bs, kHiddenSize}, {bs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND});
        inputs.push_back({{{bs, kTopK}, {bs, kTopK}}, ge::DT_INT32, ge::FORMAT_ND});
        inputs.push_back({{{bs, kTopK}, {bs, kTopK}}, ge::DT_FLOAT, ge::FORMAT_ND});
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kHiddenSize, kN}, {kHiddenSize, kN}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kNHalf, kHiddenSize}, {kNHalf, kHiddenSize}}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kN}, {kN}}, ge::DT_UINT64, ge::FORMAT_ND});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_UINT64, ge::FORMAT_ND});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kN}, {kN}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }
        for (uint32_t i = 0; i < kExpertPerRank; ++i) {
            inputs.push_back({{{kHiddenSize}, {kHiddenSize}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }
        return gert::TilingContextPara(
            "MegaMoe", inputs,
            {
                {{{bs, kHiddenSize}, {bs, kHiddenSize}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{kExpertPerRank}, {kExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
            },
            {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kEpWorldSize)},
             {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kSufficientCclBuffer)},
             {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs * kEpWorldSize * kTopK)},
             {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
             {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
             {"shared_expert_quant_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
             {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
             {"activation_params",
              Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
             {"activation_out_dtype",
              Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
             {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             // 16-18 位占位：combine_comm_mode 须落在 tiling 读取的索引 19
             {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
             {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"combine_comm_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(combineCommMode)}},
            {1, 1, 1, 1, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank, kExpertPerRank,
             kExpertPerRank},
            {1, 1}, &compileInfo, "Ascend910_93", coreNum, ubSize);
    };

    TilingInfo offInfo;
    TilingInfo onInfo;
    {
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", kEpWorldSize}};
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        ASSERT_TRUE(ExecuteTiling(buildPara(0, kBs), offInfo));
        ASSERT_TRUE(ExecuteTiling(buildPara(1, kBs), onInfo));
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }

    // faker blob 即裸结构体（无 Mc2 头），按 sizeof 校验后直接强转
    ASSERT_EQ(onInfo.tilingDataSize, sizeof(MegaMoeA2A3TilingData));
    ASSERT_EQ(offInfo.tilingDataSize, sizeof(MegaMoeA2A3TilingData));
    const auto* offData = reinterpret_cast<const MegaMoeA2A3TilingData*>(offInfo.tilingData.get());
    const auto* onData = reinterpret_cast<const MegaMoeA2A3TilingData*>(onInfo.tilingData.get());

    EXPECT_EQ(onData->dedupOn, 1U);
    EXPECT_EQ(offData->dedupOn, 0U);
    // 打包区基址 = 10MB 预留 + bs×(h+512) = 10862080；D 基址 = AlignUp(+pack, 128)。
    // pack（两平面 [slots|probs]，§14.10 rankOrdinal 内联）= EP·bs·topK·8B = 6272B，
    // 恰 128 倍数 → AlignUp 恒等
    EXPECT_EQ(onData->offsetExpandedDstToSrcRow, 10862080);
    EXPECT_EQ(onData->dedupOffsetD, 10868352);
    EXPECT_EQ(onData->distTokenFirstPosSize, 416);
    EXPECT_EQ(onData->distTokenLastPosSize, 416);
    EXPECT_EQ(onData->dedupRowMetaSize, 25088);
    // 关闭态：Dedup 字段全 0，kernel 走基线布局
    EXPECT_EQ(offData->offsetExpandedDstToSrcRow, 0);
    EXPECT_EQ(offData->dedupOffsetD, 0);
    EXPECT_EQ(offData->distTokenFirstPosSize, 0);
    EXPECT_EQ(offData->distTokenLastPosSize, 0);
    EXPECT_EQ(offData->dedupRowMetaSize, 0);
    // workspace 增量 = FP + LP + rowMeta（含 R5 32B 对齐垫量）
    ASSERT_EQ(onInfo.workspaceSizes.size(), offInfo.workspaceSizes.size());
    for (size_t i = 0; i < onInfo.workspaceSizes.size(); ++i) {
        EXPECT_EQ(onInfo.workspaceSizes[i] - offInfo.workspaceSizes[i], 25920) << "workspace idx " << i;
    }

    // G 收益门控：同形状小 bs（G = 8·8·dup(2,8)·7168·3/8 ≈ 129KB < θ_G）时 ccm=1 被门
    // 静默拒绝（dedupOn=0 回基线；默认 bs=49 → G≈791KB 过门开启，见上）
    TilingInfo gatedInfo;
    {
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", kEpWorldSize}};
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        ASSERT_TRUE(ExecuteTiling(buildPara(1, 8), gatedInfo));
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }
    const auto* gatedData = reinterpret_cast<const MegaMoeA2A3TilingData*>(gatedInfo.tilingData.get());
    EXPECT_EQ(gatedData->dedupOn, 0U);
}

TEST_F(MegaMoeArch22TilingTest, SharedExpertsRejectDedupRequest)
{
    // Valid shared configurations with ccm=0; ccm=1 must fail even when the
    // receive budget would otherwise cause Dedup to fall back to baseline.
    for (auto shared : {ge::DT_INT8, ge::DT_INT4}) {
        for (int64_t ccm : {0, 1}) {
            RunQuantSharedExpertCase(ge::DT_INT4, true, shared == ge::DT_INT4,
                                     ccm == 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, 1024, 1024, false, false, false,
                                     ge::FORMAT_FRACTAL_NZ, "Ascend910_93", 1025, false, 1024, 512, 2, 2, 2, 2, shared,
                                     512, false, false, ccm);
        }
    }
}

TEST_F(MegaMoeArch22TilingTest, W8A8MaskTilingKeyAndWorkspace)
{
    struct MegaMoeCompileInfo {};
    MegaMoeCompileInfo compileInfo;
    constexpr int64_t bs = 129;
    constexpr int64_t topK = 3;
    constexpr int64_t hidden = 1024;
    constexpr int64_t intermediate = 512;
    constexpr int64_t ep = 2;
    constexpr uint32_t expertsPerRank = 4;
    constexpr int64_t experts = ep * expertsPerRank;

    // 基线包含 10 MiB 保留区、512 KiB 标志区、3072 B 计数区及 387 行数据，每行 3588 B。
    constexpr int64_t baselineWindow = 12401676;
    // Mask 元数据包含 512 B 的 D 区对齐预留、每专家 160 B 的位图和两个 512 B 的就绪槽位。
    constexpr int64_t maskWindow = baselineWindow + 2816;
    constexpr uint64_t baselineKey = 1092;
    constexpr uint64_t rowKey = 5188;
    constexpr uint64_t blockedKey = 13380;

    using Tensor = gert::TilingContextPara::TensorDescription;
    std::vector<Tensor> inputs{
        {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{bs, hidden}, {bs, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{bs, topK}, {bs, topK}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{bs, topK}, {bs, topK}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    inputs.insert(inputs.end(), expertsPerRank,
                  Tensor{{{hidden, 2 * intermediate}, {hidden, 2 * intermediate}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ});
    inputs.insert(inputs.end(), expertsPerRank,
                  Tensor{{{intermediate, hidden}, {intermediate, hidden}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ});
    inputs.insert(inputs.end(), expertsPerRank,
                  Tensor{{{2 * intermediate}, {2 * intermediate}}, ge::DT_UINT64, ge::FORMAT_ND});
    inputs.insert(inputs.end(), expertsPerRank, Tensor{{{hidden}, {hidden}}, ge::DT_UINT64, ge::FORMAT_ND});

    auto buildPara = [&](int64_t window, int64_t receiveBudget) {
        return gert::TilingContextPara(
            "MegaMoe", inputs,
            {{{{bs, hidden}, {bs, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{expertsPerRank}, {expertsPerRank}}, ge::DT_INT32, ge::FORMAT_ND}},
            {{"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(experts)},
             {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ep)},
             {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2 * window)},
             {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(receiveBudget)},
             {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
             {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_INT8)},
             {"shared_expert_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
             {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
             {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
             {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bs)},
             {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
             {"activation_params",
              Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
             {"activation_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ge::DT_UNDEFINED)},
             {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
             {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
            {1, 1, 1, 1, expertsPerRank, expertsPerRank, expertsPerRank, expertsPerRank}, {1, 1}, &compileInfo,
            "Ascend910_93", 48, 196608);
    };

    struct LayoutCase {
        int64_t budget;
        uint64_t tilingKey;
        size_t workspaceIncrement;
    };
    // 预算为 64 行时 Permute 工作区更大，预算为 6144 行时 Mask 增量为 1536 + 512 * 8 * 4 + 6144 * 4 B。
    const LayoutCase cases[] = {{64, rowKey, 0}, {6144, blockedKey, 42496}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues({{"rankNum", ep}});
    for (const auto& testCase : cases) {
        SCOPED_TRACE(testing::Message() << "receive budget=" << testCase.budget);
        TilingInfo fallback;
        TilingInfo mask;
        ASSERT_TRUE(ExecuteTiling(buildPara(maskWindow - 1, testCase.budget), fallback));
        ASSERT_TRUE(ExecuteTiling(buildPara(maskWindow, testCase.budget), mask));
        EXPECT_EQ(fallback.tilingKey, baselineKey);
        EXPECT_EQ(mask.tilingKey, testCase.tilingKey);
        ASSERT_EQ(fallback.workspaceSizes.size(), 1U);
        ASSERT_EQ(mask.workspaceSizes.size(), 1U);
        EXPECT_EQ(mask.workspaceSizes[0], fallback.workspaceSizes[0] + testCase.workspaceIncrement);
    }
    TilingInfo tooSmall;
    EXPECT_FALSE(ExecuteTiling(buildPara(baselineWindow - 1, cases[0].budget), tooSmall));
}

TEST_F(MegaMoeArch22TilingTest, W8A8MaskAndFallbackWithSharedExperts)
{
    constexpr int64_t bs = 129;
    constexpr int64_t hidden = 1024;
    constexpr int64_t intermediate = 512;
    constexpr int64_t maskWindow = 12401676 + 2816;
    constexpr uint64_t systemBytes = 16U * 1024U * 1024U;
    const auto align512 = [](uint64_t bytes) { return (bytes + 511U) / 512U * 512U; };
    struct LayoutCase {
        int64_t window;
        int64_t budget;
        uint64_t tilingKey;
        uint64_t sharedOffset;
    };
    // With the smaller budget, Permute's workspace still determines the offset.
    // With the larger budget, the aligned shared offset follows the Mask metadata.
    const LayoutCase layouts[] = {
        {maskWindow - 1, 64, 1092, 16779264},
        {maskWindow, 64, 5188, 16779264},
        {maskWindow - 1, 6144, 1092, 25231872},
        {maskWindow, 6144, 13380, 25274368},
    };
    for (const auto& layout : layouts) {
        for (uint32_t sharedCount : {1U, 2U}) {
            for (bool mixedDown : {false, true}) {
                SCOPED_TRACE(testing::Message() << "key=" << layout.tilingKey << " budget=" << layout.budget
                                                << " shared=" << sharedCount << " mixedDown=" << mixedDown);
                TilingInfo info;
                RunQuantSharedExpertCase(ge::DT_INT8, true, false, ge::GRAPH_SUCCESS, 2 * intermediate, hidden,
                                         mixedDown, false, false, ge::FORMAT_FRACTAL_NZ, "Ascend910_93", bs, true,
                                         hidden, intermediate, 3, 2, 4, sharedCount, ge::DT_INT8, intermediate,
                                         mixedDown, false, 0, ge::DT_BF16, &info, 2 * layout.window, layout.budget);
                ASSERT_EQ(info.tilingDataSize, sizeof(MegaMoeTilingDataQuant));
                ASSERT_NE(info.tilingData.get(), nullptr);
                ASSERT_EQ(info.workspaceSizes.size(), 1U);
                const auto& common = reinterpret_cast<const MegaMoeTilingDataQuant*>(info.tilingData.get())->common;
                EXPECT_EQ(info.tilingKey, layout.tilingKey);
                EXPECT_EQ(common.recvRoundBudget, layout.budget);
                EXPECT_EQ(common.sharedExpertNum, sharedCount);
                EXPECT_EQ(common.sharedGmm2Bf16, mixedDown ? 1U : 0U);
                EXPECT_EQ(common.dedupOn, 0U);
                EXPECT_EQ(common.sharedWorkspaceOffset, layout.sharedOffset);
                const uint64_t sharedRows = bs * sharedCount;
                const uint64_t cBytes = align512(sharedRows * hidden * (mixedDown ? sizeof(int16_t) : sizeof(int32_t)));
                const uint64_t activationBytes =
                    align512(sharedRows * intermediate * (mixedDown ? sizeof(int16_t) : sizeof(int8_t)));
                const uint64_t sharedBytes = cBytes + activationBytes + align512(bs * hidden) +
                                             align512(bs * sizeof(float)) + align512(sharedRows * sizeof(float)) +
                                             align512(sharedRows * hidden * sizeof(int16_t));
                EXPECT_EQ(info.workspaceSizes[0], systemBytes + layout.sharedOffset + sharedBytes);
            }
        }
    }
}

} // namespace MegaMoeUT
