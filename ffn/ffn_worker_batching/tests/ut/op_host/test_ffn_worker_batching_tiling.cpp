/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/ffn_worker_batching_tiling.h"
#include "../../../op_kernel/arch35/ffn_worker_batching_arch35_tiling_def.h"

using namespace ge;
using namespace optiling;

class FfnWorkerBatchingTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::StorageShape y_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    gert::StorageShape session_ids_shape = {{1024}, {1024}};
    gert::StorageShape micro_batch_ids_shape = {{1024}, {1024}};
    gert::StorageShape token_ids_shape = {{1024}, {1024}};
    gert::StorageShape expert_offsets_shape = {{1024}, {1024}};
    gert::StorageShape dynamic_scale_shape = {{1024}, {1024}};
    gert::StorageShape actual_token_num_shape = {{1}, {1}};

    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {y_shape, ge::DT_INT8, ge::FORMAT_ND},                 // y
            {group_list_shape, ge::DT_INT64, ge::FORMAT_ND},       // group_list
            {session_ids_shape, ge::DT_INT32, ge::FORMAT_ND},      // session_ids
            {micro_batch_ids_shape, ge::DT_INT32, ge::FORMAT_ND},  // micro_batch_ids
            {token_ids_shape, ge::DT_INT32, ge::FORMAT_ND},        // token_ids
            {expert_offsets_shape, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {dynamic_scale_shape, ge::DT_FLOAT, ge::FORMAT_ND},    // dynamic_scale
            {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo);
    int64_t expectTilingKey = 101;
    std::string expectTilingData = "1152 4096 0 8 32 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ---------------------------------------------------------------------------------------------------------------------
// 迭代一 A5(arch35/ascend950) 核心路径 UT
//
// 路由机制（本组用例的判定依据）：
//   socVersion="Ascend950" -> PlatformAscendC.GetCurNpuArch()==DAV_3510 -> IsRegbaseSocVersion()==true
//   -> Tiling4FfnWorkerBatching 走 TilingRegistry::DoTilingImpl -> 命中 1000 优先级模板
//      FfnWorkerBatchingRegbaseTiling(_tiling_arch35.cpp)。
//   socVersion="Ascend910B"/"Ascend910_93" -> DAV_2201 -> IsRegbaseSocVersion()==false
//   -> 走原 monolithic FfnWorkerBatchingTiling::RunFfnWorkerBatchingTiling（A2/A3，逐字不动）。
//
// arch35 TilingData 平铺 struct FfnWorkerBatchingArch35TilingData 与 A2 FfnWorkerBatchingTilingData 字段一一对应
// （8×int64：Y H tokenDtype expertNum coreNum ubSize sortLoopMaxElement sortNumWorkSpace），A5 功能对齐 A2，
// 故相同输入下 golden 值与 A2 一致 —— 本身即是「arch35 复算等价于 A2」的交叉校验。
// ---------------------------------------------------------------------------------------------------------------------

namespace {
// 构造 8 输出 desc（shape 仅供 faker 建 tensor，tiling 不读输出 shape，dtype 按 def 顺序）。
std::vector<gert::TilingContextPara::TensorDescription> MakeA5OutputDesc()
{
    static gert::StorageShape y_shape = {{1152, 4096}, {1152, 4096}};
    static gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    static gert::StorageShape idx_shape = {{1152}, {1152}};
    static gert::StorageShape scale_shape = {{1152}, {1152}};
    static gert::StorageShape actual_token_num_shape = {{1}, {1}};
    return {
        {y_shape, ge::DT_INT8, ge::FORMAT_ND},                 // y
        {group_list_shape, ge::DT_INT64, ge::FORMAT_ND},       // group_list
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // session_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // micro_batch_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // token_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // expert_offsets
        {scale_shape, ge::DT_FLOAT, ge::FORMAT_ND},            // dynamic_scale
        {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
    };
}
} // namespace

namespace {
void CheckArch35(int64_t dtype, int64_t mode, int64_t a, int64_t bs, int64_t k, int64_t experts, int64_t ub,
                 int64_t expectedCores, int64_t expectedSegments, int64_t expectedPerSegment)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(experts)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({a, bs, k, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtype)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", 64, ub);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(FfnWorkerBatchingArch35TilingData));
    const auto& d = *reinterpret_cast<const FfnWorkerBatchingArch35TilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 100 + mode);
    EXPECT_EQ(info.blockNum, expectedCores);
    EXPECT_EQ(d.Y, a * bs * k);
    EXPECT_EQ(d.H, 4096);
    EXPECT_EQ(d.tokenDtype, dtype);
    EXPECT_FALSE(d.syncFlag);
    EXPECT_EQ(d.expertNum, experts);
    EXPECT_EQ(d.ubSize, ub - 32768);
    EXPECT_EQ(d.sortSegNum, expectedSegments);
    EXPECT_EQ(d.sortPerSegElements, expectedPerSegment);
    EXPECT_EQ(d.expertStart, 1000000);
    // 各工作区不重叠且分配覆盖最后一个 gather 索引。
    EXPECT_LE(d.wsFlatIds + d.flatElements, d.wsPairA);
    EXPECT_LE(d.wsPairA + d.sortSegNum * d.sortLenPerSeg, d.wsPairB);
    EXPECT_LE(d.wsPairB + d.sortSegNum * d.sortLenPerSeg, d.wsSegCnt);
    EXPECT_LE(d.wsSegCnt + d.sortSegNum * 8, d.wsSortedIds);
    EXPECT_LE(d.wsSortedIds + d.Y, d.wsGatherIdx);
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_GE(info.workspaceSizes[0], (d.wsGatherIdx + d.Y) * sizeof(int32_t));
}
} // namespace

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_norm_key100)
{
    CheckArch35(0, 0, 16, 8, 9, 8, 262144, 36, 36, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_recv_key101)
{
    CheckArch35(0, 1, 16, 8, 9, 8, 262144, 36, 36, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_bf16_norm)
{
    CheckArch35(1, 0, 16, 8, 9, 8, 262144, 36, 36, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_int8_recv)
{
    CheckArch35(2, 1, 16, 8, 9, 8, 262144, 36, 36, 32);
}

// 零回归①：Ascend910B(A2) + need_schedule=0(NORM) -> monolithic 路径 TilingKey=100，coreNum=64。
// 验证 IsRegbaseSocVersion 对 A2 返回 false（未误走 arch35），且 A2 NORM 分支输出正确。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_ascend910b_norm_no_regression)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo,
        "Ascend910B"); // -> DAV_2201 -> monolithic A2 路径

    int64_t expectTilingKey = 100;
    std::string expectTilingData = "1152 4096 0 8 64 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 零回归②：Ascend910_93(A3) + need_schedule=1(RECV) -> monolithic 路径 TilingKey=101，coreNum=32。
// 与基线 test01（Ascend910B）语义一致，验证 A3 走原路径无回归。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_ascend910_93_recv_no_regression)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo,
        "Ascend910_93"); // -> DAV_2201 -> monolithic A3 路径

    int64_t expectTilingKey = 101;
    std::string expectTilingData = "1152 4096 0 8 32 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// arch35 平台参数注入和参数构造。
namespace {
constexpr uint64_t kArch35Ub248K = 253952; // GetCoreMemSize(UB) 真实 Ascend950 返回值（issue_20260708 收敛口径）
constexpr uint64_t kAivNum64 = 64;

// 按 (tokenDtype, needSchedule, A, BS, K, H) 构造 attrs；Y=A*BS*K，max_out_shape={A,BS,K,H}。
std::vector<gert::TilingContextPara::OpAttr> MakeArch35Attrs(int64_t tokenDtype, int64_t needSchedule, int64_t A,
                                                             int64_t BS, int64_t K, int64_t H, int64_t expertNum = 8)
{
    return {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertNum)},
            {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({A, BS, K, H})},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenDtype)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(needSchedule)},
            {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}};
}
} // namespace

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_norm_key100_ub248k_segments)
{
    CheckArch35(0, 0, 16, 8, 9, 8, 253952, 36, 36, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_key101_ub248k_corelimit)
{
    CheckArch35(0, 1, 16, 8, 9, 8, 253952, 36, 36, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_segment_capacity_runtime_derived)
{
    CheckArch35(0, 0, 1024, 8, 64, 8, 253952, 64, 87, 6080);
    CheckArch35(0, 0, 1024, 8, 64, 8, 262144, 64, 83, 6336);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_norm_multicore_largeshape)
{
    CheckArch35(0, 0, 1024, 1, 64, 8, 253952, 64, 64, 1024);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_corelimit_below_full_core_threshold)
{
    CheckArch35(0, 1, 31, 1, 64, 8, 253952, 62, 62, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_corelimit_large_y_64)
{
    CheckArch35(0, 1, 32, 1, 64, 8, 253952, 64, 64, 32);
}

// arch35 使用独立分段排序 ABI，核数随有效工作量变化。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_large_expert_num_5120)
{
    CheckArch35(0, 1, 16, 8, 9, 5120, 253952, 36, 36, 32);
}

// D1｜A3 零回归（NORM）：Ascend910_93 + need_schedule=0 -> monolithic 路径 TilingKey=100，coreNum=64。
//     补齐现有 A3 RECV 无回归用例缺失的 A3 NORM 分支；DAV_2201 预留 256B -> ubSize=261888、sortLoopMaxElement=8160
//     （与 arch35 221184/6912 不同，证明 A3 未误走 arch35 1000 档）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_ascend910_93_norm_no_regression)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend910_93"); // 默认 coreNum=64/ubSize=262144 -> DAV_2201 monolithic

    std::string expectTilingData = "1152 4096 0 8 64 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// D2｜A2 零回归（大 shape 多核 + RECV 不限核边界）：Ascend910B + RECV + Y=204800(>200000) -> monolithic 路径
//     TilingKey=101、coreNum=64（不限核）。验证 A2 monolithic 与 arch35 共享同一 RECV 限核 <= 边界语义、
//     且大 shape 下仍走 DAV_2201 路径（ubSize=261888、sortLoopMaxElement=8160），零回归。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_ascend910b_recv_largeshape_no_regression)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 800, /*BS*/ 4, /*K*/ 64, /*H*/ 4096), // Y=204800
        &compileInfo, "Ascend910B");

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t* d = reinterpret_cast<const int64_t*>(info.tilingData.get());
    std::cout << "[D2][A2][RECV largeshape] coreNum=" << d[4] << " ubSize=" << d[5] << " sortLoopMaxElement=" << d[6]
              << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[4], 64);     // Y>200000 不限核（A2 monolithic 同语义）
    EXPECT_EQ(d[5], 261888); // DAV_2201 预留 256B -> 未走 arch35（arch35 扣 32KB 会是 221184）

    // Y=204800: workspace = 16777760 + 36*204800 = 24150560
    std::string expectTilingData = "204800 4096 0 8 64 261888 8160 204800 ";
    std::vector<size_t> expectWorkspaces = {24150560};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// =====================================================================================================================
// arch35 硬校验错误路径 UT（对齐 spec.yaml boundary_conditions 的 raises_error 契约）
//
// 覆盖 arch35 FfnWorkerBatchingRegbaseTiling::CheckInputParam / GetAttrsInfo 的硬校验分支——这是与 A2 monolithic
// 校验相互独立的全新代码路径。全部经 socVersion="Ascend950" 路由到 1000 档 arch35 tiling，期望返回 ge::GRAPH_FAILED。
// 逐条对应 spec.yaml boundary_conditions：expert_num>8192 / topK+1>64 / A>1024 / max_out_shape 长度≠4 /
// token_dtype∉[0,5] / need_schedule∉[0,1]。校验阈值(EXPERT_IDX_MAX/MAX_SESSION_NUM/MAX_K_NUM)为对外规格，与代次无关。
// =====================================================================================================================

// E1｜expert_num 越界(>8192) -> arch35 GetAttrsInfo 报错（spec: index_out_of_bounds / attribute_value_out_of_range）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_expert_num_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                 MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9,
                                                 /*H*/ 4096, /*expertNum*/ 8193),
                                 &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2｜topK+1 = max_out_shape[2] 越界(>64) -> arch35 报错（spec MAX_K_NUM=64）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_topk_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 65, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E3｜A = max_out_shape[0] 越界(>1024) -> arch35 报错（spec MAX_SESSION_NUM=1024）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_session_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 1025, /*BS*/ 1, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E4｜max_out_shape 长度 != 4 -> arch35 报错（spec: 长度必须为 4）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_max_out_shape_len_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9})}, // len=3
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E5｜token_dtype 越界(>5) -> arch35 报错（取值范围 [0,5]）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_token_dtype_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 6, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E6｜need_schedule 越界(>1) -> arch35 报错（spec 取值范围 [0,1]）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_need_schedule_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 2, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_mx_types)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    for (int64_t dtype = 3; dtype <= 5; ++dtype) {
        for (int64_t mode = 0; mode <= 1; ++mode) {
            gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}},
                                         MakeA5OutputDesc(), MakeArch35Attrs(dtype, mode, 3, 7, 9, 66), &compileInfo,
                                         "Ascend950", kAivNum64, kArch35Ub248K);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            EXPECT_EQ(info.tilingKey, 100 + mode);
            const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
            EXPECT_EQ(data[0], 189);
            EXPECT_EQ(data[1], 66);
            EXPECT_EQ(data[2], dtype);
        }
    }
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_fp4_odd_h_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                 MakeArch35Attrs(5, 0, 1, 1, 1, 33), &compileInfo, "Ascend950", kAivNum64,
                                 kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_mx_rejected_on_legacy_soc)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    for (const auto* soc : {"Ascend910B", "Ascend910_93"}) {
        for (int64_t dtype = 3; dtype <= 5; ++dtype) {
            gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}},
                                         MakeA5OutputDesc(), MakeArch35Attrs(dtype, 0, 1, 1, 1, 64), &compileInfo, soc);
            ExecuteTestCase(para, ge::GRAPH_FAILED);
        }
    }
}

// 两种调度路径都覆盖布尔属性；保持 key=100/101，不把 syncFlag 编入 key。
// 对异步 RECV 再验证快照与 gather workspace 不重叠、按块对齐且总空间覆盖 A+1 个块。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_sync_flag)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    for (int64_t mode : {0, 1}) {
        for (bool syncFlag : {false, true}) {
            auto attrs = MakeArch35Attrs(0, mode, 4, 3, 3, 66);
            attrs.push_back({"sync_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(syncFlag)});
            gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}},
                                         MakeA5OutputDesc(), attrs, &compileInfo, "Ascend950", kAivNum64,
                                         kArch35Ub248K);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            const auto& d = *reinterpret_cast<const FfnWorkerBatchingArch35TilingData*>(info.tilingData.get());
            EXPECT_EQ(d.syncFlag, syncFlag);
            EXPECT_EQ(info.tilingKey, 100 + mode);
            EXPECT_GE(d.wsReady, d.wsGatherIdx + d.Y);
            EXPECT_EQ(d.wsReady % 8, 0);
            if (mode == 1 && syncFlag) {
                EXPECT_GE(info.workspaceSizes[0], (d.wsReady + 5 * 8) * sizeof(int32_t));
            }
        }
    }
}

// 根据 kernel 实际同时存活的缓冲验证预算，而不是仅核对某个 tiling 常量。
// 新字段为零时按旧实现的分配公式核算，可在修复前运行同一组测试记录失败。
TEST_F(FfnWorkerBatchingTilingTest, arch35_review_ub_capacity_contract)
{
    for (const auto& shape :
         std::vector<std::vector<int64_t>>{{1, 1024, 64, 2}, {1024, 648, 64, 2}, {1024, 257, 64, 2}, {16, 128, 4, 2}}) {
        for (int64_t mode : {0, 1}) {
            SCOPED_TRACE(::testing::Message() << "A=" << shape[0] << " BS=" << shape[1] << " mode=" << mode);
            gert::StorageShape scShape = {{1024}, {1024}};
            FfnWorkerBatchingCompileInfo compileInfo = {};
            gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}},
                                         MakeA5OutputDesc(),
                                         MakeArch35Attrs(0, mode, shape[0], shape[1], shape[2], shape[3]), &compileInfo,
                                         "Ascend950", kAivNum64, kArch35Ub248K);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info));
            const auto& d = *reinterpret_cast<const FfnWorkerBatchingArch35TilingData*>(info.tilingData.get());
            auto align = [](int64_t n, int64_t unit) { return (n + unit - 1) / unit * unit; };
            const int64_t row = align(shape[1] * shape[2], 8);
            const int64_t prepareElements =
                d.preparePerLoopElements > 0 ? d.preparePerLoopElements : d.preparePerLoopRows * row;
            EXPECT_LE(prepareElements * 4 + d.preparePerLoopRows * 32, d.ubSize) << "prepare";
            const int64_t seg = d.sortPerSegElements;
            EXPECT_LE(12 * align(seg, 64) + 24 * align(seg, 32) + 160, d.ubSize) << "sort";
            const int64_t cache = d.mergeCountCacheSegments > 0 ? d.mergeCountCacheSegments : d.sortSegNum;
            EXPECT_LE(64 * d.mergeOneLoopElements + 32 * cache + 96, d.ubSize) << "merge";
            EXPECT_LE(16 * align(d.extractPerLoopElements, 32) + 96, d.ubSize) << "extract";
            EXPECT_LE(align(d.extractPerLoopElements, 32) / 32, 255) << "Extract API repeatTime";
        }
    }
}

TEST_F(FfnWorkerBatchingTilingTest, arch35_review_context_length_contract)
{
    for (int64_t bytes : {0, 1, 1023, 1024, 1056}) {
        SCOPED_TRACE(bytes);
        gert::StorageShape scShape = {{bytes}, {bytes}};
        FfnWorkerBatchingCompileInfo compileInfo = {};
        gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                     MakeArch35Attrs(0, 0, 1, 1, 1, 2), &compileInfo, "Ascend950", kAivNum64,
                                     kArch35Ub248K);
        TilingInfo info;
        EXPECT_EQ(ExecuteTiling(para, info), bytes >= 1024);
    }
}

// Attribute arithmetic must fail before allocating outputs/workspace. Keep
// representable large indices legal: there is no artificial 2^22 limit.
TEST_F(FfnWorkerBatchingTilingTest, arch35_adopt_attribute_boundaries)
{
    struct Case {
        int64_t a, bs, k, h, dtype;
        bool pass;
    };
    const std::vector<Case> cases = {{1024, INT64_MAX, 1, 2, 0, false},
                                     {1, INT64_MAX, 64, 2, 0, false},
                                     {1, 2147483648LL, 1, 2, 0, false},
                                     {1024, 257, 64, 2, 0, true},
                                     {1, 1, 1, 2147483392LL, 0, true},
                                     {1, 1, 1, 2147483393LL, 1, false},
                                     {1, 1, 1, 4294966780LL, 2, true},
                                     {1, 1, 1, 4294966781LL, 2, false},
                                     {1, 1, 1, 4294967296LL, 5, false},
                                     // FP8 token + ceil(H/32) scale bytes exceed uint32 HS at large H.
                                     {1, 1, 1, 4294966780LL, 3, false},
                                     {1, 1, 1, 4294966780LL, 4, false},
                                     // FP4 packs H/2 bytes, so this H is representable for FP4.
                                     {1, 1, 1, 4294966780LL, 5, true}};
    for (const auto& c : cases) {
        SCOPED_TRACE(::testing::Message() << c.a << "," << c.bs << "," << c.k << "," << c.h << "," << c.dtype);
        gert::StorageShape sc = {{1024}, {1024}};
        FfnWorkerBatchingCompileInfo ci = {};
        gert::TilingContextPara para("FfnWorkerBatching", {{sc, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                     MakeArch35Attrs(c.dtype, 0, c.a, c.bs, c.k, c.h), &ci, "Ascend950", kAivNum64,
                                     kArch35Ub248K);
        TilingInfo info;
        EXPECT_EQ(ExecuteTiling(para, info), c.pass);
    }
}

TEST_F(FfnWorkerBatchingTilingTest, arch35_adopt_minimum_ub)
{
    for (uint64_t rawUb : {uint64_t(32768 + 1024), uint64_t(32768 + 19583)}) {
        gert::StorageShape sc = {{1024}, {1024}};
        FfnWorkerBatchingCompileInfo ci = {};
        gert::TilingContextPara para("FfnWorkerBatching", {{sc, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                     MakeArch35Attrs(0, 0, 1, 1, 1, 2), &ci, "Ascend950", kAivNum64, rawUb);
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(para, info));
    }
}

// Layer-derived expert IDs are exclusively an asynchronous RECV contract.
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_async_layer_contract)
{
    gert::StorageShape shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    for (int64_t mode : {0, 1}) {
        for (bool sync : {false, true}) {
            for (int64_t layers : {0, 1, 2, 3, 8}) {
                auto attrs = MakeArch35Attrs(0, mode, 3, 7, 1, 32);
                attrs.back() = {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(layers)};
                attrs.push_back({"sync_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(sync)});
                gert::TilingContextPara para("FfnWorkerBatching", {{shape, ge::DT_INT8, ge::FORMAT_ND}},
                                             MakeA5OutputDesc(), attrs, &compileInfo, "Ascend950", kAivNum64,
                                             kArch35Ub248K);
                if (mode == 1 && sync && (layers == 0 || 8 % layers != 0)) {
                    ExecuteTestCase(para, ge::GRAPH_FAILED);
                } else {
                    TilingInfo info;
                    ASSERT_TRUE(ExecuteTiling(para, info));
                    const auto& data =
                        *reinterpret_cast<const FfnWorkerBatchingArch35TilingData*>(info.tilingData.get());
                    EXPECT_EQ(data.layerNum, mode == 1 && sync ? layers : 0);
                }
            }
        }
    }
}
