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
#include "mc2_tiling_case_executor.h"

using namespace std;

namespace MoeDistributeCombineUT {

class MoeDistributeCombineArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineArch22TilingTest, Test0)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 16UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineArch22TilingTest, Test1)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineArch22TilingTest, Test2)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 1024;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineArch22TilingTest, Test3)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 31;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
TEST_F(MoeDistributeCombineArch22TilingTest, A2)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineArch22TilingTest, A2Layered)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MoeDistributeCombineArch22TilingTest, A2GlobalBs)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 512;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
TEST_F(MoeDistributeCombineArch22TilingTest, A2Shape)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7160}, {8 * 8 * 32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
TEST_F(MoeDistributeCombineArch22TilingTest, A2EpRankId)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 33;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
TEST_F(MoeDistributeCombineArch22TilingTest, A2MoeExpertNum)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 257;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineArch22TilingTest, EpWorldSize384)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 384;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
TEST_F(MoeDistributeCombineArch22TilingTest, EpWorldSize72)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 72;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 18;
    int64_t moeExpertNum = 216;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(MoeDistributeCombineArch22TilingTest, A2Int8Quant)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}
TEST_F(MoeDistributeCombineArch22TilingTest, A2InvalidCommQuantMode)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("");
    int64_t epWorldSize = 32;
    int64_t tpWorldSize = 0;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 3;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 48;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 8 * 32, 7168}, {8 * 8 * 32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 32}, {8 * 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// ==========================================
// 补充测试用例 - 提升 moe_distribute_combine_tiling.cpp 覆盖率
// 目标覆盖: tpWorldSize>=2 校验、tp 属性参数校验
// ==========================================

// A5 模板成功路径: tpWorldSize=1, 校验 tp 属性参数
TEST_F(MoeDistributeCombineArch22TilingTest, A5_TpWorldSize1_Success)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{32 * 8, 7168}, {32 * 8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288 * 2}, {288 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板: commQuantMode=2 (INT8 通信量化), tpWorldSize=1
TEST_F(MoeDistributeCombineArch22TilingTest, A5_Int8CommQuant)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 2;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{32 * 8, 7168}, {32 * 8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板: globalBs 非零值, 覆盖 CheckBatchAttrs 中 globalBs != 0 的分支
TEST_F(MoeDistributeCombineArch22TilingTest, A5_GlobalBsNonZero)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 64;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板: epWorldSize=4, sharedExpertRankNum=2, 覆盖 A5 CheckEpWorldSize 中 epWorldSize 为 2 的倍数的路径
TEST_F(MoeDistributeCombineArch22TilingTest, A5_EpWorldSize4)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 4;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 2;
    int64_t moeExpertNum = 2;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 2, 7168}, {8 * 2, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 2}, {8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 2}, {8 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: epWorldSize 不是 2 的倍数
TEST_F(MoeDistributeCombineArch22TilingTest, A5_EpWorldSizeOdd)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 7;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 6;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: epWorldSize 不在合法列表中 (不满足 256%epWorldSize==0 且不满足 epWorldSize%144==0)
TEST_F(MoeDistributeCombineArch22TilingTest, A5_EpWorldSizeInvalid)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 6;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 5;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: expandX 数据类型为 DT_INT32
TEST_F(MoeDistributeCombineArch22TilingTest, A5_ExpandXInt32DataType)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: tpWorldSize >= 2 校验失败
TEST_F(MoeDistributeCombineArch22TilingTest, A5_TpWorldSizeGt2_Rejected)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 2;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288 * 2}, {288 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: sharedExpertNum != 1
TEST_F(MoeDistributeCombineArch22TilingTest, A5_InvalidSharedExpertNum)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: expertShardType != 0
TEST_F(MoeDistributeCombineArch22TilingTest, A5_InvalidExpertShardType)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 1;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A5 模板错误路径: sharedExpertRankNum >= epWorldSize
TEST_F(MoeDistributeCombineArch22TilingTest, A5_SharedExpertRankNumTooLarge)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 10;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend910_93", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace MoeDistributeCombineUT
