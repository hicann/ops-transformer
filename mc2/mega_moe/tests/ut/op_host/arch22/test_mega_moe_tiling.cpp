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

} // namespace MegaMoeUT
