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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_kernel/attention_to_ffn_v2_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace AttentionToFfnV2UT {
namespace {

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct AttentionToFfnV2CompileInfo {};

std::vector<TensorDescription> BuildInputs(bool quant, bool activeMask, uint32_t contextDim = 1U,
                                           ge::DataType contextDtype = ge::DT_INT32,
                                           ge::Format contextFormat = ge::FORMAT_ND)
{
    const gert::StorageShape contextShape =
        contextDim == 1U ? gert::StorageShape{{1}, {1}} : gert::StorageShape{{1, 1}, {1, 1}};
    std::vector<TensorDescription> inputs = {
        {contextShape, contextDtype, contextFormat},
        {{{1, 16, 7168}, {1, 16, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, 16, 8}, {1, 16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1, 9, 4}, {1, 9, 4}}, ge::DT_INT32, ge::FORMAT_ND},
    };

    if (quant || activeMask) {
        const gert::StorageShape scalesShape =
            quant ? gert::StorageShape{{9, 7168}, {9, 7168}} : gert::StorageShape{{}, {}};
        inputs.emplace_back(scalesShape, ge::DT_FLOAT, ge::FORMAT_ND);
    }
    if (activeMask) {
        inputs.emplace_back(gert::StorageShape{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND);
    }
    return inputs;
}

std::vector<OpAttr> BuildAttrs(bool quant, bool sync)
{
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
        {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7168})},
        {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quant ? 2 : 0)},
        {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sync ? 1 : 0)},
        {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
}

void ExecuteTilingCase(const std::vector<TensorDescription> &inputs, bool quant, bool sync,
                       ge::graphStatus expectedStatus, uint64_t expectedTilingKey = 0UL)
{
    AttentionToFfnV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AttentionToFfnV2", inputs, {{{}, ge::DT_INT64, ge::FORMAT_ND}},
                                              BuildAttrs(quant, sync), &compileInfo, "Ascend950");
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
        return;
    }
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus);
}

struct TilingKeyCase {
    bool quant;
    bool sync;
    bool activeMask;
    uint64_t expectedKey;
};

class AttentionToFfnV2Arch35TilingKeyTest : public testing::TestWithParam<TilingKeyCase> {};

TEST_P(AttentionToFfnV2Arch35TilingKeyTest, GeneratesA5TilingKey)
{
    const TilingKeyCase &testCase = GetParam();
    ExecuteTilingCase(BuildInputs(testCase.quant, testCase.activeMask), testCase.quant, testCase.sync,
                      ge::GRAPH_SUCCESS, testCase.expectedKey);
}

INSTANTIATE_TEST_SUITE_P(
    A5TilingKeys, AttentionToFfnV2Arch35TilingKeyTest,
    testing::Values(TilingKeyCase{false, false, false, 8UL}, TilingKeyCase{true, false, false, 9UL},
                    TilingKeyCase{false, true, false, 10UL}, TilingKeyCase{true, true, false, 11UL},
                    TilingKeyCase{false, false, true, 12UL}, TilingKeyCase{true, false, true, 13UL},
                    TilingKeyCase{false, true, true, 14UL}, TilingKeyCase{true, true, true, 15UL}));

class AttentionToFfnV2ContextValidationTest : public testing::Test {};

TEST_F(AttentionToFfnV2ContextValidationTest, RejectsTwoDimensionalContext)
{
    ExecuteTilingCase(BuildInputs(false, false, 2U), false, false, ge::GRAPH_FAILED);
}

TEST_F(AttentionToFfnV2ContextValidationTest, RejectsNonInt32Context)
{
    ExecuteTilingCase(BuildInputs(false, false, 1U, ge::DT_FLOAT), false, false, ge::GRAPH_FAILED);
}

TEST_F(AttentionToFfnV2ContextValidationTest, RejectsFractalNzContext)
{
    ExecuteTilingCase(BuildInputs(false, false, 1U, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ), false, false,
                      ge::GRAPH_FAILED);
}

} // namespace
} // namespace AttentionToFfnV2UT
