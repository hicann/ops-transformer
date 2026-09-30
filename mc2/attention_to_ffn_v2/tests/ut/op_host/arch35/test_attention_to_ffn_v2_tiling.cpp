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
#include "../../../../op_kernel/attention_to_ffn_v2_tiling_key.h"
#include "mc2_tiling_case_executor.h"

namespace AttentionToFfnV2UT {
namespace {

// GET_TPL_TILING_KEY 宏展开引用不带限定的 g_tilingDeclareParams（由 tiling_key.h 在
// namespace Mc2Tiling 内生成），此处显式引入该 TU 内符号。
using Mc2Tiling::g_tilingDeclareParams;

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct AttentionToFfnV2CompileInfo {};

// 1GB win capacity; the URMA win-layout check only enforces a lower bound.
constexpr int64_t CCL_BUFFER_SIZE = 1LL << 30;

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
        // scales 必须 3D：[expertRankTable.dim0, expertRankTable.dim1, x.dim2]，见 base tiling CheckInputForScales
        const gert::StorageShape scalesShape =
            quant ? gert::StorageShape{{1, 9, 7168}, {1, 9, 7168}} : gert::StorageShape{{}, {}};
        inputs.emplace_back(scalesShape, ge::DT_FLOAT, ge::FORMAT_ND);
    }
    if (activeMask) {
        inputs.emplace_back(gert::StorageShape{{1, 16}, {1, 16}}, ge::DT_BOOL, ge::FORMAT_ND);
    }
    return inputs;
}

std::vector<OpAttr> BuildAttrs(int64_t quantMode, bool sync)
{
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"ffn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 146})},
        // HS=7680: win 槽容量需覆盖最大单 token 载荷（MX: align256(H)+224=7392 > 7168）
        {"ffn_token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({11, 1, 16, 9, 7680})},
        {"attn_token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quantMode)},
        {"sync_flag", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sync ? 1 : 0)},
        {"ffn_start_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(CCL_BUFFER_SIZE)},
    };
}

static std::string MakeAscend950SocInfo()
{
    return R"({"hardware_info": {"UB_SIZE": 196608, "cube_core_cnt": 24, "vector_core_cnt": 48}})";
}

void ExecuteTilingCase(const std::vector<TensorDescription>& inputs, int64_t quantMode, bool sync,
                       ge::graphStatus expectedStatus, uint64_t expectedTilingKey = 0UL)
{
    AttentionToFfnV2CompileInfo compileInfo;
    // MX/MX_CLIP 模式显式 buffer 需求约 151.5KB，须小于平台 UB（196608B，与其他 arch35 UT 口径一致）。
    gert::TilingContextPara tilingContextPara("AttentionToFfnV2", inputs, {{{}, ge::DT_INT64, ge::FORMAT_ND}},
                                              BuildAttrs(quantMode, sync), &compileInfo, "3510", 48U, 196608U, 4096U,
                                              MakeAscend950SocInfo());
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
};

class AttentionToFfnV2Arch35TilingKeyTest : public testing::TestWithParam<TilingKeyCase> {};

TEST_P(AttentionToFfnV2Arch35TilingKeyTest, GeneratesA5TilingKey)
{
    const TilingKeyCase& testCase = GetParam();
    const int64_t quantMode = testCase.quant ? 2 : 0;
    const uint64_t expectedKey = GET_TPL_TILING_KEY(
        quantMode == 2 ? ATTN_FFN_TILINGKEY_PERTOKEN_INT8 : ATTN_FFN_TILINGKEY_NO_QUANT, ATTN_FFN_TILINGKEY_OUT_INT8,
        testCase.quant, testCase.sync, testCase.activeMask, TILINGKEY_TPL_A5);
    ExecuteTilingCase(BuildInputs(testCase.quant, testCase.activeMask), quantMode, testCase.sync, ge::GRAPH_SUCCESS,
                      expectedKey);
}

INSTANTIATE_TEST_SUITE_P(A5TilingKeys, AttentionToFfnV2Arch35TilingKeyTest,
                         testing::Values(TilingKeyCase{false, false, false}, TilingKeyCase{true, false, false},
                                         TilingKeyCase{false, true, false}, TilingKeyCase{true, true, false},
                                         TilingKeyCase{false, false, true}, TilingKeyCase{true, false, true},
                                         TilingKeyCase{false, true, true}, TilingKeyCase{true, true, true}));

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

class AttentionToFfnV2MxTilingTest : public testing::Test {};

// MX/MX_CLIP 模式要求 scales 缺省，走 TilingNewQuantMode 路径。
TEST_F(AttentionToFfnV2MxTilingTest, MxE5M2)
{
    ExecuteTilingCase(
        BuildInputs(false, false), 3, false, ge::GRAPH_SUCCESS,
        GET_TPL_TILING_KEY(ATTN_FFN_TILINGKEY_MX, ATTN_FFN_TILINGKEY_OUT_E5M2, false, false, false, TILINGKEY_TPL_A5));
}

TEST_F(AttentionToFfnV2MxTilingTest, MxClipE4M3)
{
    // quant_mode=7 → (MX_CLIP, E4M3)；quant_mode=6 是 (MX_CLIP, E5M2)
    ExecuteTilingCase(BuildInputs(false, false), 7, false, ge::GRAPH_SUCCESS,
                      GET_TPL_TILING_KEY(ATTN_FFN_TILINGKEY_MX_CLIP, ATTN_FFN_TILINGKEY_OUT_E4M3, false, false, false,
                                         TILINGKEY_TPL_A5));
}

TEST_F(AttentionToFfnV2MxTilingTest, RejectsInvalidQuantMode)
{
    // quant_mode=1 不在合法集合 {0, 2, 3, 4, 5, 6, 7}
    ExecuteTilingCase(BuildInputs(false, false), 1, false, ge::GRAPH_FAILED);
}

} // namespace
} // namespace AttentionToFfnV2UT
