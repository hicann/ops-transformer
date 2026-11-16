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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_kernel/ffn_to_attention_v2_tiling.h"
#include "../../../../op_kernel/ffn_to_attention_v2_tilling_key.h"
#include "mc2_tiling_case_executor.h"

namespace FFNToAttentionV2UT {
namespace {

// GET_TPL_TILING_KEY 宏展开引用不带限定的 g_tilingDeclareParams（由 tilling_key.h 在
// namespace Mc2Tiling 内生成），此处显式引入该 TU 内符号。
using Mc2Tiling::g_tilingDeclareParams;

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct FFNToAttentionV2CompileInfo {};
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr uint64_t WORKSPACE_ALIGN = 512U;
constexpr uint64_t AIV_NUM = 48U;
constexpr uint64_t PAYLOAD_AIV_CAPACITY = 256U;
constexpr uint64_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr uint64_t REMOTE_ADDRESS_ENTRY_SIZE = 16U;
constexpr uint64_t LIB_API_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint64_t RESERVED_WORKSPACE_SIZE = 64U * 1024U * 1024U;
constexpr uint64_t TOKEN_DTYPE_BYTES = 2U;
constexpr int64_t DEFAULT_CCL_BUFFER_SIZE = 4LL * MB_SIZE;
constexpr int64_t MIN_CCL_BUFFER_SIZE = 2LL * MB_SIZE;
constexpr int64_t RANK_TABLE_CCL_BUFFER_SIZE = 32LL * MB_SIZE;
constexpr int64_t MULTI_MICRO_BATCH_CCL_BUFFER_SIZE = 8LL * MB_SIZE;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}
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

std::vector<OpAttr> BuildAttrs(bool invalidBs = false, int64_t microBatchNum = 1,
                               int64_t cclBufferSize = DEFAULT_CCL_BUFFER_SIZE)
{
    const std::vector<int64_t> tokenInfoTableShape = {microBatchNum, invalidBs ? 513 : 16, 9};
    const std::vector<int64_t> tokenDataShape = {microBatchNum, invalidBs ? 513 : 16, 9, 7168};
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

std::vector<OpAttr> BuildAttrsWithHs(int64_t hs, int64_t cclBufferSize = DEFAULT_CCL_BUFFER_SIZE)
{
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
        {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"token_info_table_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9})},
        {"token_data_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 16, 9, hs})},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
    };
}

static std::string MakeAscend950SocInfo()
{
    return R"({"hardware_info": {"UB_SIZE": 196608, "cube_core_cnt": 24, "vector_core_cnt": 48}})";
}

void ExecuteTilingCase(const std::vector<TensorDescription>& inputs, const std::vector<OpAttr>& attrs,
                       ge::graphStatus expectedStatus, uint64_t expectedTilingKey = 0UL)
{
    FFNToAttentionV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("FFNToAttentionV2", inputs, {{{}, ge::DT_INT64, ge::FORMAT_ND}}, attrs,
                                              &compileInfo, "3510", AIV_NUM, 196608U, 4096U, MakeAscend950SocInfo());
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
        return;
    }
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus);
}

bool RunTilingCase(const std::vector<TensorDescription>& inputs, const std::vector<OpAttr>& attrs,
                   TilingInfo& tilingInfo)
{
    FFNToAttentionV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("FFNToAttentionV2", inputs, {{{}, ge::DT_INT64, ge::FORMAT_ND}}, attrs,
                                              &compileInfo, "3510", AIV_NUM, 196608U, 4096U, MakeAscend950SocInfo());
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    const bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    return success;
}
class FFNToAttentionV2Arch35TilingTest : public testing::Test {};

TEST_F(FFNToAttentionV2Arch35TilingTest, NoRankTable)
{
    ExecuteTilingCase(BuildInputs(false), BuildAttrs(), ge::GRAPH_SUCCESS, GET_TPL_TILING_KEY(false, TILINGKEY_TPL_A5));
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RankTable)
{
    // attnRankTable dim0 (11) is the exact attention worker count.
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(false, 1, RANK_TABLE_CCL_BUFFER_SIZE), ge::GRAPH_SUCCESS,
                      GET_TPL_TILING_KEY(true, TILINGKEY_TPL_A5));
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RankTableWithLandingZoneOnlyBuffer)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(false, 1, DEFAULT_CCL_BUFFER_SIZE), ge::GRAPH_SUCCESS,
                      GET_TPL_TILING_KEY(true, TILINGKEY_TPL_A5));
}

TEST_F(FFNToAttentionV2Arch35TilingTest, WorkspaceScratchDoesNotOverlapLibApiWorkspace)
{
    // With 144 token slots, HS=7278 makes the aligned flag and return-data areas exactly 2MB.
    constexpr int64_t WINDOW_BOUNDARY_HS = 7278;
    TilingInfo tilingInfo;
    ASSERT_TRUE(
        RunTilingCase(BuildInputs(false), BuildAttrsWithHs(WINDOW_BOUNDARY_HS, MIN_CCL_BUFFER_SIZE), tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(FFNToAttentionV2TilingData));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);

    FFNToAttentionV2TilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    const auto& info = tilingData.ffnToAttentionV2Info;
    ASSERT_EQ(info.aivNum, AIV_NUM);
    const uint64_t flagWorkspaceSize = static_cast<uint64_t>(info.aivNum) * URMA_FLAG_SLOT_SIZE;
    const uint64_t rankTableStride = AlignUp(info.maxTokenNum * REMOTE_ADDRESS_ENTRY_SIZE, WORKSPACE_ALIGN);
    const uint64_t countRowStride = AlignUp(static_cast<uint64_t>(info.worldSize) * sizeof(int32_t), WORKSPACE_ALIGN);
    const uint64_t expectedFlagWorkspaceBytes = AlignUp(flagWorkspaceSize, WORKSPACE_ALIGN);
    const uint64_t expectedWorkspaceSize =
        LIB_API_WORKSPACE_SIZE + expectedFlagWorkspaceBytes + RESERVED_WORKSPACE_SIZE;
    EXPECT_EQ(static_cast<uint64_t>(tilingInfo.workspaceSizes[0]), expectedWorkspaceSize);
    EXPECT_EQ(info.cclBufferSize, static_cast<uint64_t>(MIN_CCL_BUFFER_SIZE));

    // The payload layout fits the conservative capacity allocated by the Python buffer wrapper.
    const uint64_t tokenRowStride = AlignUp(info.H * TOKEN_DTYPE_BYTES, 32U);
    const uint64_t tokenRegionBytes = AlignUp(info.maxTokenNum * tokenRowStride, WORKSPACE_ALIGN);
    const uint64_t addressTableEnd = tokenRegionBytes + rankTableStride * info.worldSize;
    const uint64_t countRowsEnd = addressTableEnd + (static_cast<uint64_t>(info.aivNum) + 1U) * countRowStride;
    const uint64_t requiredSlotBytes =
        AlignUp(countRowsEnd + AlignUp(static_cast<uint64_t>(info.aivNum) * URMA_FLAG_SLOT_SIZE, WORKSPACE_ALIGN),
                WORKSPACE_ALIGN);
    EXPECT_EQ(info.payloadTokenRowStride, tokenRowStride);
    EXPECT_EQ(info.payloadAddressTableOffset, tokenRegionBytes);
    EXPECT_EQ(info.payloadCountRowsOffset, addressTableEnd);
    EXPECT_EQ(info.payloadCountRowStride, countRowStride);
    EXPECT_EQ(info.payloadFlagOffset, countRowsEnd);
    EXPECT_EQ(info.payloadRequiredSlotBytes, requiredSlotBytes);
    const uint64_t allocatedSlotBytes = tokenRegionBytes + rankTableStride * info.worldSize +
                                        (PAYLOAD_AIV_CAPACITY + 1U) * countRowStride +
                                        AlignUp(PAYLOAD_AIV_CAPACITY * URMA_FLAG_SLOT_SIZE, WORKSPACE_ALIGN);
    EXPECT_LE(info.payloadRequiredSlotBytes, allocatedSlotBytes);

    const uint64_t tokenSlotNum = static_cast<uint64_t>(info.microBatchNum) * info.BS * info.expertNumPerToken;
    const uint64_t flagAreaBytes = AlignUp(tokenSlotNum * sizeof(int32_t), WORKSPACE_ALIGN);
    const uint64_t returnDataBytes = AlignUp(tokenSlotNum * info.HS * TOKEN_DTYPE_BYTES, WORKSPACE_ALIGN);
    const uint64_t requiredWindowSize = flagAreaBytes + returnDataBytes;
    EXPECT_EQ(requiredWindowSize, static_cast<uint64_t>(MIN_CCL_BUFFER_SIZE));
    EXPECT_LT(static_cast<uint64_t>(MIN_CCL_BUFFER_SIZE), requiredWindowSize + expectedFlagWorkspaceBytes);
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
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(true), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsZeroMicroBatchNum)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(false, 0), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsMicroBatchNumOverflow)
{
    // dim0 beyond uint32 max cannot survive the int64 -> uint32 narrowing in tilingData.
    ExecuteTilingCase(BuildInputs(true), BuildAttrs(false, 4294967296), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, AcceptsMultiMicroBatchNum)
{
    // The multi micro-batch relaxation (tokenInfoTableShape dim0 > 1) only applies to V2.
    ExecuteTilingCase(BuildInputs(false), BuildAttrs(false, 2, MULTI_MICRO_BATCH_CCL_BUFFER_SIZE), ge::GRAPH_SUCCESS,
                      GET_TPL_TILING_KEY(false, TILINGKEY_TPL_A5));
    ExecuteTilingCase(BuildInputs(false), BuildAttrs(false, 3, MULTI_MICRO_BATCH_CCL_BUFFER_SIZE), ge::GRAPH_SUCCESS,
                      GET_TPL_TILING_KEY(false, TILINGKEY_TPL_A5));
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsMoreThanThreeMicroBatches)
{
    ExecuteTilingCase(BuildInputs(false), BuildAttrs(false, 4, RANK_TABLE_CCL_BUFFER_SIZE), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsHsSmallerThanH)
{
    ExecuteTilingCase(BuildInputs(true), BuildAttrsWithHs(7167), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsBufferWithoutTokenInfoAlignmentSpace)
{
    // Landing zone = AlignUp(135 * 4B, 512) + AlignUp(135 * 7764 * 2B, 512) = 1024 + 2096640 =
    // 2097664 bytes (2MB + 512B): without the 512B flag-area alignment the raw area (2096820
    // bytes) would fit in 2MB, so the alignment alone decides between the 2MB and 4MB buffers.
    ExecuteTilingCase(BuildInputs(false), BuildAlignmentBoundaryAttrs(MIN_CCL_BUFFER_SIZE), ge::GRAPH_FAILED);
}

TEST_F(FFNToAttentionV2Arch35TilingTest, AcceptsBufferWithTokenInfoAlignmentSpace)
{
    ExecuteTilingCase(BuildInputs(false), BuildAlignmentBoundaryAttrs(4LL * MB_SIZE), ge::GRAPH_SUCCESS,
                      GET_TPL_TILING_KEY(false, TILINGKEY_TPL_A5));
}

TEST_F(FFNToAttentionV2Arch35TilingTest, RejectsRankTableBufferBelowLandingZone)
{
    ExecuteTilingCase(BuildInputs(true), BuildAlignmentBoundaryAttrs(MIN_CCL_BUFFER_SIZE), ge::GRAPH_FAILED);
}

} // namespace
} // namespace FFNToAttentionV2UT
