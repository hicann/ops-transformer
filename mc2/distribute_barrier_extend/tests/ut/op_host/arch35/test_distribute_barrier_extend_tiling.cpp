/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../../common/op_kernel/mc2_moe_context.h"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

namespace DistributeBarrierExtendUT {
constexpr uint32_t CONTEXT_SIZE = Mc2Aclnn::MC2_CONTEXT_ELEMS;
constexpr uint64_t g_mc2TilingDataReservedLen = sizeof(Mc2InitTiling) + sizeof(Mc2CcTiling);

template <typename T>
static string ToString(void* buf, size_t size, unordered_set<size_t> mask = {})
{
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += mask.find(i) == mask.end() ? std::to_string(data[i]) : "*";
        result += " ";
    }
    return result;
}

const unordered_set<size_t> g_barrierTilingDataMask = {4};

class DistributeBarrierExtendArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DistributeBarrierExtendArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DistributeBarrierExtendArch35TilingTest TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTiling)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{CONTEXT_SIZE}, {CONTEXT_SIZE}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 1451008 0 65536 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingWorldSize1)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{CONTEXT_SIZE}, {CONTEXT_SIZE}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingWorldSize1025)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{CONTEXT_SIZE}, {CONTEXT_SIZE}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingTimeOut)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{CONTEXT_SIZE}, {CONTEXT_SIZE}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 1451008 0 65537 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingElasticInfo)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{CONTEXT_SIZE}, {CONTEXT_SIZE}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 1451008 0 65792 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingTimeOutElasticInfo)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{CONTEXT_SIZE}, {CONTEXT_SIZE}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 1451008 0 65793 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("DistributeBarrierExtend");
    ASSERT_NE(opImpl, nullptr);
    auto tilingParseFunc = opImpl->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    auto ret = tilingParseFunc(nullptr);
    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, AutomaticContextBounds)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    struct Case {
        int64_t world;
        int64_t elems;
        bool valid;
    };
    const Case cases[] = {{2, Mc2Barrier::ContextElems(2, 1), true},
                          {16, Mc2Barrier::ContextElems(16, 1), true},
                          {204800, Mc2Barrier::ContextElems(204800, 1), true},
                          {204801, Mc2Barrier::ContextElems(204801, 1), false},
                          {1, Mc2Barrier::ContextElems(1, 1), false},
                          {16, 0, false},
                          {16, 43, false},
                          {16, 45, false}};
    for (const auto& test : cases) {
        SCOPED_TRACE(::testing::PrintToString(std::make_pair(test.world, test.elems)));
        gert::TilingContextPara para(
            "DistributeBarrierExtend",
            {{{{test.elems}, {test.elems}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
            {{{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
            {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
             {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(test.world)}},
            &compileInfo, "3510", 20, 196608);
        TilingInfo info;
        ASSERT_EQ(ExecuteTiling(para, info), test.valid);
        if (test.valid) {
            const auto* data = reinterpret_cast<const uint32_t*>(info.tilingData.get() + g_mc2TilingDataReservedLen);
            EXPECT_EQ(data[0], test.world);
            EXPECT_EQ(data[6], test.elems);
            EXPECT_NE(data[8] & 0xFF000000U, 0U);
        }
    }
}

TEST_F(DistributeBarrierExtendArch35TilingTest, AutomaticElasticInfoUsesTopologySize)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    for (const auto& topology : {std::pair<uint32_t, uint32_t>{16, 2}, {204800, 1}}) {
        const uint32_t world = topology.first * topology.second;
        const int64_t elems = Mc2Barrier::ContextElems(topology.first, topology.second);
        const int64_t elasticElems = 4 + 2 * world;
        gert::TilingContextPara para("DistributeBarrierExtend",
                                     {{{{elems}, {elems}}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {{}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{{elasticElems}, {elasticElems}}, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                      {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(world)}},
                                     &compileInfo, "3510", 20, 196608);
        TilingInfo info;
        EXPECT_TRUE(ExecuteTiling(para, info));
    }
}

TEST_F(DistributeBarrierExtendArch35TilingTest, AutomaticElasticUbBudget)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    const uint32_t world = Mc2Barrier::MAX_RANKS;
    const int64_t elems = Mc2Barrier::ContextElems(16, world / 16U);
    const int64_t elasticElems = 4 + 2 * world;
    for (uint32_t cores : {2U, 20U}) {
        const uint32_t required = Mc2Barrier::AutoUbBytes(world, cores - 1U, true);
        for (uint32_t ub : {required - 1U, required}) {
            gert::TilingContextPara para("DistributeBarrierExtend",
                                         {{{{elems}, {elems}}, ge::DT_INT32, ge::FORMAT_ND},
                                          {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                          {{}, ge::DT_INT32, ge::FORMAT_ND},
                                          {{{elasticElems}, {elasticElems}}, ge::DT_INT32, ge::FORMAT_ND}},
                                         {{{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                         {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                          {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(world)}},
                                         &compileInfo, "3510", cores, ub);
            TilingInfo info;
            EXPECT_EQ(ExecuteTiling(para, info), ub >= required) << "cores=" << cores << ", UB=" << ub;
        }
    }
}

TEST_F(DistributeBarrierExtendArch35TilingTest, ContextLengthParityDisambiguatesLayouts)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    for (bool automatic : {false, true}) {
        const int64_t elements = automatic ? Mc2Barrier::ContextElems(1024, 1) : CONTEXT_SIZE;
        gert::TilingContextPara para(
            "DistributeBarrierExtend",
            {{{{elements}, {elements}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
            {{{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
            {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
             {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)}},
            &compileInfo, "3510", 20, 196608);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info));
        const auto* data = reinterpret_cast<const uint32_t*>(info.tilingData.get() + g_mc2TilingDataReservedLen);
        EXPECT_EQ((data[8] & 0xFF000000U) != 0, automatic);
    }
}

} // namespace DistributeBarrierExtendUT

TEST(DistributeBarrierTopologyTest, ChannelBatchesCoverPeersAndUseReciprocalRounds)
{
    using namespace Mc2Barrier;
    for (uint32_t count : {8U, 513U, 12800U}) {
        for (uint32_t self : {0U, count / 2, count - 1}) {
            std::vector<bool> visited(count, false);
            for (uint32_t round = 0; round < ChannelBatchRounds(count); ++round) {
                const uint32_t first = ChannelBatchBegin(self, round);
                const uint32_t end = std::min(count, first + CHANNEL_BATCH_SIZE);
                EXPECT_LE(end > first ? end - first : 0, 512U);
                for (uint32_t peer = first; peer < end; ++peer) {
                    EXPECT_FALSE(visited[peer]);
                    visited[peer] = true;
                    const uint32_t reciprocal = ChannelBatchBegin(peer, round);
                    EXPECT_GE(self, reciprocal);
                    EXPECT_LT(self, reciprocal + CHANNEL_BATCH_SIZE);
                }
            }
            EXPECT_TRUE(std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }));
        }
    }
}
