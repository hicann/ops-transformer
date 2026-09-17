/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include <gtest/gtest.h>

#include "../../../op_host/chunk_gated_delta_rule_compute_wy_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace ge;
using namespace optiling;

namespace {
constexpr int64_t CHUNK_SIZE = 64;
constexpr uint64_t CORE_NUM = 8;

// Prefix of ChunkGatedDeltaRuleComputeWyTilingData up to the first TCubeTiling. The
// host serializer emits fields back to back with no padding, so this mirrors the
// kernel-side #pragma pack(1) struct exactly; a layout drift breaks these reads.
#pragma pack(push, 1)
struct TilingHead {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t numChunks;
    int64_t groupSize;
    int64_t totalTasks;
    uint32_t localWorkspaceSize;
    uint32_t perCoreWorkspaceBytes;
    uint32_t usedCoreNum;
    uint64_t workspaceOffset;
};
#pragma pack(pop)

const std::string SOC_INFO_310P = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", )"
                                  R"("Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, )"
                                  R"("Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, )"
                                  R"("UB_SIZE": 196608, "L2_SIZE": 16777216, "L1_SIZE": 1048576, )"
                                  R"("L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, )"
                                  R"("CORE_NUM": 8, "socVersion":"Ascend310P"} })";

gert::TilingContextPara MakeContext(int64_t b, int64_t t, int64_t hk, int64_t hv, int64_t kDim, int64_t vDim,
                                    int64_t chunkSize, ChunkGatedDeltaRuleComputeWyCompileInfo *compileInfo)
{
    gert::StorageShape qShape = {{b, t, hk, kDim}, {b, t, hk, kDim}};
    gert::StorageShape vShape = {{b, t, hv, vDim}, {b, t, hv, vDim}};
    gert::StorageShape gShape = {{b, t, hv}, {b, t, hv}};
    gert::StorageShape qKernelShape = {{b, hk, t, kDim}, {b, hk, t, kDim}};
    gert::StorageShape wKernelShape = {{b, hv, t, kDim}, {b, hv, t, kDim}};
    gert::StorageShape uKernelShape = {{b, hv, t, vDim}, {b, hv, t, vDim}};
    gert::StorageShape gKernelShape = {{b, hv, t}, {b, hv, t}};
    return gert::TilingContextPara("ChunkGatedDeltaRuleComputeWy",
                                   {
                                       {qShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {qShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {vShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {gShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                   },
                                   {
                                       {qKernelShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {qKernelShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {wKernelShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {uKernelShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {gKernelShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                   },
                                   {
                                       {"chunk_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(chunkSize)},
                                   },
                                   compileInfo, "Ascend310P", CORE_NUM, 196608, 4096, SOC_INFO_310P);
}
} // namespace

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, GroupedHeadsFp16)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    constexpr int64_t b = 1;
    constexpr int64_t t = 256;
    constexpr int64_t hk = 2;
    constexpr int64_t hv = 4;
    constexpr int64_t kDim = 64;
    constexpr int64_t vDim = 64;
    auto context = MakeContext(b, t, hk, hv, kDim, vDim, CHUNK_SIZE, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_GT(tilingInfo.blockNum, 0U);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TilingHead));

    TilingHead head{};
    std::memcpy(&head, tilingInfo.tilingData.get(), sizeof(head));
    EXPECT_EQ(head.batch, b);
    EXPECT_EQ(head.seqlen, t);
    EXPECT_EQ(head.kNumHead, hk);
    EXPECT_EQ(head.vNumHead, hv);
    EXPECT_EQ(head.kHeadDim, kDim);
    EXPECT_EQ(head.vHeadDim, vDim);
    EXPECT_EQ(head.chunkSize, CHUNK_SIZE);
    EXPECT_EQ(head.numChunks, t / CHUNK_SIZE);
    EXPECT_EQ(head.groupSize, hv / hk);
    EXPECT_EQ(head.totalTasks, b * hv * (t / CHUNK_SIZE));
    EXPECT_EQ(head.usedCoreNum, static_cast<uint32_t>(tilingInfo.blockNum));
    EXPECT_GT(head.localWorkspaceSize, 0U);
    EXPECT_GT(head.perCoreWorkspaceBytes, 0U);

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_GE(tilingInfo.workspaceSizes[0],
              static_cast<int64_t>(head.workspaceOffset +
                                   static_cast<uint64_t>(head.usedCoreNum) * head.perCoreWorkspaceBytes));
}

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, MaxHeadDim)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    auto context = MakeContext(2, 128, 4, 8, 128, 128, CHUNK_SIZE, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_GT(tilingInfo.blockNum, 0U);
}

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, RejectsNonDefaultChunkSize)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    auto context = MakeContext(1, 256, 2, 4, 64, 64, 32, &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, RejectsUnpaddedSeqlen)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    auto context = MakeContext(1, 200, 2, 4, 64, 64, CHUNK_SIZE, &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, RejectsNonGroupedHeads)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    auto context = MakeContext(1, 256, 3, 4, 64, 64, CHUNK_SIZE, &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, RejectsHeadDimOverUbBudget)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    auto context = MakeContext(1, 256, 2, 4, 256, 256, CHUNK_SIZE, &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}

TEST(ChunkGatedDeltaRuleComputeWyTilingTest, RejectsUnalignedHeadDim)
{
    ChunkGatedDeltaRuleComputeWyCompileInfo compileInfo = {};
    auto context = MakeContext(1, 256, 2, 4, 40, 64, CHUNK_SIZE, &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(context, tilingInfo));
}
