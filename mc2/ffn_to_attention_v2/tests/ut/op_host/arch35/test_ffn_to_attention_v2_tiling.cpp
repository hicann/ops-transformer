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
#include "../../../../op_kernel/ffn_to_attention_v2_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace FFNToAttentionV2UT {
namespace {

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct FFNToAttentionV2CompileInfo {};
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t DEFAULT_CCL_BUFFER_SIZE = 2LL * MB_SIZE;

enum class InvalidInput {
    NONE,
    X_H,
    SESSION_IDS_SHAPE,
    MICRO_BATCH_IDS_SHAPE,
    TOKEN_IDS_SHAPE,
    EXPERT_OFFSETS_SHAPE,
    ACTUAL_TOKEN_NUM_SHAPE,
    ACTUAL_TOKEN_NUM_DTYPE,
    X_FORMAT,
    CONTEXT_SHAPE,
    CONTEXT_DTYPE,
    CONTEXT_FORMAT,
};

std::vector<TensorDescription> BuildInputs(bool rankTableMode, InvalidInput invalidInput = InvalidInput::NONE)
{
    std::vector<TensorDescription> inputs = {
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},       {{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1584}, {1584}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
    };
    if (rankTableMode) {
        inputs.emplace_back(gert::StorageShape{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND);
    }

    switch (invalidInput) {
        case InvalidInput::X_H:
            inputs[1] = TensorDescription{{{1584, 800}, {1584, 800}}, ge::DT_FLOAT16, ge::FORMAT_ND};
            break;
        case InvalidInput::SESSION_IDS_SHAPE:
            inputs[2] = TensorDescription{{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND};
            break;
        case InvalidInput::MICRO_BATCH_IDS_SHAPE:
            inputs[3] = TensorDescription{{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND};
            break;
        case InvalidInput::TOKEN_IDS_SHAPE:
            inputs[4] = TensorDescription{{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND};
            break;
        case InvalidInput::EXPERT_OFFSETS_SHAPE:
            inputs[5] = TensorDescription{{{1583}, {1583}}, ge::DT_INT32, ge::FORMAT_ND};
            break;
        case InvalidInput::ACTUAL_TOKEN_NUM_SHAPE:
            inputs[6] = TensorDescription{{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND};
            break;
        case InvalidInput::ACTUAL_TOKEN_NUM_DTYPE:
            inputs[6] = TensorDescription{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND};
            break;
        case InvalidInput::X_FORMAT:
            inputs[1] = TensorDescription{{{1584, 7168}, {1584, 7168}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ};
            break;
        case InvalidInput::CONTEXT_SHAPE:
            inputs[0] = TensorDescription{{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND};
            break;
        case InvalidInput::CONTEXT_DTYPE:
            inputs[0] = TensorDescription{{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND};
            break;
        case InvalidInput::CONTEXT_FORMAT:
            inputs[0] = TensorDescription{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ};
            break;
        case InvalidInput::NONE:
            break;
        default:
            break;
    }
    return inputs;
}

std::vector<OpAttr> BuildAttrs(bool invalidBs = false, bool invalidMicroBatchNum = false,
                               int64_t cclBufferSize = DEFAULT_CCL_BUFFER_SIZE)
{
    const std::vector<int64_t> tokenInfoTableShape =
        invalidMicroBatchNum ? std::vector<int64_t>{2, 16, 9} : std::vector<int64_t>{1, invalidBs ? 513 : 16, 9};
    const std::vector<int64_t> tokenDataShape = invalidMicroBatchNum ?
                                                    std::vector<int64_t>{2, 16, 9, 7168} :
                                                    std::vector<int64_t>{1, invalidBs ? 513 : 16, 9, 7168};
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tokenInfoTableShape)},
        {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tokenDataShape)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
    };
}

std::vector<OpAttr> BuildAlignmentBoundaryAttrs(int64_t cclBufferSize)
{
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 9, 15})},
        {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 9, 15, 7764})},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
    };
}

std::vector<OpAttr> BuildAttrsWithHs(int64_t hs)
{
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
        {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, hs})},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(DEFAULT_CCL_BUFFER_SIZE)},
    };
}

void ExecuteTilingCase(const std::vector<TensorDescription> &inputs, const std::vector<OpAttr> &attrs,
                       ge::graphStatus expectedStatus, uint64_t expectedTilingKey = 0UL)
{
    FFNToAttentionV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("FFNToAttentionV2", inputs, {{{}, ge::DT_INT64, ge::FORMAT_ND}}, attrs,
                                              &compileInfo, "Ascend950");
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
        return;
    }
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus);
}

class FFNToAttentionV2Arch35TilingTest : public testing::Test {};

TEST_F(FFNToAttentionV2Arch35TilingTest, NoRankTable)
{
    ExecuteTilingCase(BuildInputs(false), BuildAttrs(), ge::GRAPH_SUCCESS, 12UL);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RankTable)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(), ge::GRAPH_SUCCESS, 13UL);
}

class FFNToAttentionV2InvalidInputTest : public testing::TestWithParam<InvalidInput> {};

TEST_P(FFNToAttentionV2InvalidInputTest, RejectsInvalidInput)
{
    ExecuteTilingCase(BuildInputs(true, GetParam()), BuildAttrs(), ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(InvalidInputs, FFNToAttentionV2InvalidInputTest,
                         testing::Values(InvalidInput::X_H, InvalidInput::SESSION_IDS_SHAPE,
                                         InvalidInput::MICRO_BATCH_IDS_SHAPE, InvalidInput::TOKEN_IDS_SHAPE,
                                         InvalidInput::EXPERT_OFFSETS_SHAPE, InvalidInput::ACTUAL_TOKEN_NUM_SHAPE,
                                         InvalidInput::ACTUAL_TOKEN_NUM_DTYPE, InvalidInput::X_FORMAT,
                                         InvalidInput::CONTEXT_SHAPE, InvalidInput::CONTEXT_DTYPE,
                                         InvalidInput::CONTEXT_FORMAT));

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsInvalidBs)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(true, false), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsInvalidMicroBatchNum)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(false, true), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsHsSmallerThanH)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrsWithHs(7167), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsBufferWithoutTokenInfoAlignmentSpace)
{
    ExecuteTilingCase(BuildInputs(true), BuildAlignmentBoundaryAttrs(2LL * MB_SIZE), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, AcceptsBufferWithTokenInfoAlignmentSpace)
{
    ExecuteTilingCase(BuildInputs(true), BuildAlignmentBoundaryAttrs(4LL * MB_SIZE), ge::GRAPH_SUCCESS, 13UL);
}

} // namespace
} // namespace FFNToAttentionV2UT
