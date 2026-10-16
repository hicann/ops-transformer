/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_kernel/arch35/common/mega_moe_workspace.h"
#include "../../../../op_kernel/arch35/mega_moe_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace MegaMoeUT {

constexpr int64_t ARCH35_CCL_BUFFER_SIZE = 1024LL * 1024LL * 1024LL;

class MegaMoeArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest TearDown" << std::endl;
    }
};

static MegaMoeImpl::PeermemSizeParams MakePeermemSizeParams(const MegaMoeTilingData& tilingData)
{
    MegaMoeImpl::PeermemSizeParams peermemParams{};
    peermemParams.numMaxTokensPerRank = tilingData.numMaxTokensPerRank;
    peermemParams.topK = tilingData.topK;
    peermemParams.h = tilingData.h;
    peermemParams.moeExpertPerRank = tilingData.moeExpertPerRank;
    peermemParams.epWorldSize = tilingData.epWorldSize;
    peermemParams.yDtypeSize = sizeof(uint16_t); // 这些用例的输出类型为 BF16。
    peermemParams.elemsPerByte = MegaMoeImpl::IsA4W4GmmMode(tilingData.moeGmmMode) ? 2U : 1U;
    peermemParams.topkWeightsPrefetch =
        tilingData.topkWeightsPrefetch == 1 || MegaMoeImpl::IsCombineDedupOn(tilingData.dedupMode);
    peermemParams.isQuantCombine = tilingData.combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT;
    peermemParams.topoType = tilingData.topoType;
    peermemParams.serverNum = 1;
    if (tilingData.topoType == MegaMoeImpl::TOPO_TYPE_URMA) {
        peermemParams.serverNum = tilingData.epWorldSize / tilingData.rankNumPerServer;
    }
    return peermemParams;
}

static int64_t CalcCclCommunicationCapacity(const MegaMoeTilingData& tilingData)
{
    auto peermemParams = MakePeermemSizeParams(tilingData);
    peermemParams.elemsPerByte = 1U;
    int64_t communicationSize = MegaMoeImpl::CalcPeermemLeastSize(peermemParams);
    if (tilingData.topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        communicationSize += MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE;
    }
    return Ops::Base::CeilAlign(communicationSize, static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
}

static void ExpectCclBufferCapacity(const TilingInfo& tilingInfo, int64_t cclBufferSize = ARCH35_CCL_BUFFER_SIZE)
{
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    MegaMoeImpl::WorkspaceLayout workspaceLayout(&tilingData);
    const int64_t communicationCapacity = CalcCclCommunicationCapacity(tilingData);

    ASSERT_GT(workspaceLayout.workspaceSize, 0);
    EXPECT_GT(communicationCapacity, 0);
    EXPECT_EQ(communicationCapacity % MegaMoeImpl::ALIGN_512, 0);
    EXPECT_LE(communicationCapacity + workspaceLayout.workspaceSize, cclBufferSize);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
}

static bool RunMteCclBufferCase(int64_t cclBufferSize, TilingInfo& tilingInfo, int64_t maxRecvTokenNum = 0,
                                int64_t numMaxTokensPerRank = 0)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxRecvTokenNum)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numMaxTokensPerRank)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    return tilingSucceeded;
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 664612UL);
    ExpectCclBufferCapacity(tilingInfo);
}

TEST_F(MegaMoeArch35TilingTest, CclBufferExactCapacityAndOneByteShort)
{
    TilingInfo initialInfo;
    ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, initialInfo));
    ASSERT_GE(initialInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, initialInfo.tilingData.get(), sizeof(tilingData));
    MegaMoeImpl::WorkspaceLayout workspaceLayout(&tilingData);
    const int64_t communicationCapacity = CalcCclCommunicationCapacity(tilingData);
    const int64_t requiredSize = communicationCapacity + workspaceLayout.workspaceSize;

    TilingInfo exactInfo;
    ASSERT_TRUE(RunMteCclBufferCase(requiredSize, exactInfo));
    ExpectCclBufferCapacity(exactInfo, requiredSize);
    TilingInfo shortInfo;
    EXPECT_FALSE(RunMteCclBufferCase(requiredSize - 1, shortInfo));
    TilingInfo communicationOnlyInfo;
    EXPECT_FALSE(RunMteCclBufferCase(communicationCapacity, communicationOnlyInfo));

    TilingInfo largerInfo;
    ASSERT_TRUE(RunMteCclBufferCase(requiredSize + 512, largerInfo));
    ExpectCclBufferCapacity(largerInfo, requiredSize + 512);
}

TEST_F(MegaMoeArch35TilingTest, CclDataOffsetFitsFp4CapacityBudget)
{
    for (int64_t topoType : {MegaMoeImpl::TOPO_TYPE_MTE, MegaMoeImpl::TOPO_TYPE_URMA}) {
        for (int32_t prefetch : {0, 1}) {
            for (int32_t combineQuantMode : {0, 3, 4}) {
                for (bool combineDedup : {false, true}) {
                    if (combineDedup && (topoType == MegaMoeImpl::TOPO_TYPE_URMA ||
                                         combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT)) {
                        continue;
                    }
                    SCOPED_TRACE(testing::Message()
                                 << "topoType=" << topoType << ", prefetch=" << prefetch
                                 << ", combineQuantMode=" << combineQuantMode << ", combineDedup=" << combineDedup);
                    MegaMoeImpl::PeermemSizeParams params{};
                    params.numMaxTokensPerRank = 257;
                    params.topK = 8;
                    params.h = 4096;
                    params.moeExpertPerRank = 2;
                    params.epWorldSize = 8;
                    params.yDtypeSize = sizeof(uint16_t);
                    params.elemsPerByte = 2U;
                    params.topkWeightsPrefetch = prefetch == 1 || combineDedup;
                    params.isQuantCombine = combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT;
                    params.topoType = topoType;
                    params.serverNum = topoType == MegaMoeImpl::TOPO_TYPE_URMA ? 2 : 1;

                    auto budgetParams = params;
                    budgetParams.elemsPerByte = 1U;
                    const int64_t dumpBytes =
                        topoType == MegaMoeImpl::TOPO_TYPE_MTE ? MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE : 0;
                    const int64_t communicationCapacity = Ops::Base::CeilAlign(
                        dumpBytes + MegaMoeImpl::CalcPeermemLeastSize(budgetParams), MegaMoeImpl::ALIGN_512);
                    const int64_t communicationSize = MegaMoeImpl::CalcPeermemLeastSize(params);
                    const int64_t cclDataOffset = Ops::Base::CeilAlign(communicationSize, MegaMoeImpl::ALIGN_512);

                    // 实际布局按 FP4 压缩记录推进，业务起点位于通信末尾并保持 512B 对齐。
                    EXPECT_GE(cclDataOffset, communicationSize);
                    EXPECT_LT(cclDataOffset - communicationSize, MegaMoeImpl::ALIGN_512);
                    EXPECT_EQ(cclDataOffset % MegaMoeImpl::ALIGN_512, 0);
                    // MTE 的异常信息区在传入基址之前，保守申请容量仍覆盖实际业务起点。
                    EXPECT_LT(dumpBytes + cclDataOffset, communicationCapacity);
                    EXPECT_EQ(params.elemsPerByte, 2U);
                }
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, CclBufferRejectsReceiveCapacityTruncation)
{
    TilingInfo tilingInfo;
    const int64_t uint32Max = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, uint32Max + 1));
    // 默认接收容量为最大 token 数乘以 EP，再乘以 topK 与本卡专家数的较小值。
    EXPECT_FALSE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, 0, uint32Max));
}

TEST_F(MegaMoeArch35TilingTest, H5120_BS256_URMA_InvalidRankNumPerServer)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 6}, {256, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 6}, {256, 6}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 3072, 5120}, {8, 3072, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8, 5120, 1536}, {8, 5120, 1536}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8, 3072, 80, 2}, {8, 3072, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{8, 5120, 24, 2}, {8, 5120, 24, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H7168_BS512_MTE_RankNumIgnored)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 7168, 2048}, {2, 7168, 2048}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 4096, 112, 2}, {2, 4096, 112, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{2, 7168, 32, 2}, {2, 7168, 32, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
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
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 664612UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, DifferentNConfig)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
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
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 8995UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

static void RunUrmaTilingCase(int64_t xBs, int64_t topkIdsBs, int64_t topkWeightsBs, int64_t numMaxTokensPerRank,
                              int64_t maxRecvTokenNum, int64_t hiddenDim, const std::string& activation,
                              const std::vector<float>& activationParams, ge::graphStatus expectedStatus)
{
    ASSERT_GT(hiddenDim, 0);
    ASSERT_EQ(hiddenDim % (MegaMoeImpl::ACTIVATION_N_HALF * MegaMoeImpl::MXFP_DIVISOR_SIZE), 0);

    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{xBs, 4096}, {xBs, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{topkIdsBs, 8}, {topkIdsBs, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{topkWeightsBs, 8}, {topkWeightsBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, hiddenDim, 4096}, {8, hiddenDim, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},
            {{{8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF},
              {8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF}},
             ge::DT_FLOAT8_E4M3FN,
             ge::FORMAT_FRACTAL_NZ},
            {{{8, hiddenDim, 64, 2}, {8, hiddenDim, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF / MegaMoeImpl::MXFP_DIVISOR_SIZE, 2},
              {8, 4096, hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF / MegaMoeImpl::MXFP_DIVISOR_SIZE, 2}},
             ge::DT_FLOAT8_E8M0,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{xBs, 4096}, {xBs, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxRecvTokenNum)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numMaxTokensPerRank)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>(activation)},
            {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(activationParams)},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    const uint64_t expectedTilingKey = expectedStatus == ge::GRAPH_SUCCESS ? 42607652UL : 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);

    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ExpectCclBufferCapacity(tilingInfo);

    if (activation != "situglu") {
        return;
    }

    ASSERT_TRUE(activationParams.size() == 1U || activationParams.size() == 2U);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingData.actMode, static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU));
    EXPECT_EQ(tilingData.actSubMode,
              static_cast<uint8_t>(activationParams.size() == 2U ? MegaMoeImpl::MegaMoeActSubMode::LINEAR :
                                                                   MegaMoeImpl::MegaMoeActSubMode::DEFAULT));
    EXPECT_FLOAT_EQ(tilingData.activationAlpha, activationParams.size() == 2U ? activationParams[1] : 0.0f);
    EXPECT_FLOAT_EQ(tilingData.activationBeta, activationParams[0]);
}

static void RunUrmaVariableBsInputShapeCase(int64_t xBs, int64_t topkIdsBs, int64_t topkWeightsBs,
                                            int64_t numMaxTokensPerRank, int64_t maxRecvTokenNum,
                                            ge::graphStatus expectedStatus)
{
    RunUrmaTilingCase(xBs, topkIdsBs, topkWeightsBs, numMaxTokensPerRank, maxRecvTokenNum, 2048, "swiglu",
                      {std::numeric_limits<float>::max()}, expectedStatus);
}

static void RunUrmaVariableBsCase(int64_t bs, int64_t numMaxTokensPerRank, int64_t maxRecvTokenNum,
                                  ge::graphStatus expectedStatus)
{
    RunUrmaVariableBsInputShapeCase(bs, bs, bs, numMaxTokensPerRank, maxRecvTokenNum, expectedStatus);
}

static void RunUrmaActivationCase(int64_t hiddenDim, const std::string& activation,
                                  const std::vector<float>& activationParams, ge::graphStatus expectedStatus)
{
    RunUrmaTilingCase(128, 128, 128, 256, 0, hiddenDim, activation, activationParams, expectedStatus);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokens256_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 256, 0, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_Default_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {1.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_Linear_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {1.0f, 2.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N768_SITU_URMA_2Servers)
{
    RunUrmaActivationCase(768, "situglu", {1.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N512_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(512, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N1536_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(1536, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N8192_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(8192, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N256_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(256, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N640_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(640, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N8448_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(8448, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SwiGluStep_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "swiglustep", {}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_MissingBeta_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokensZero_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 0, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokens64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 64, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS0_NumMaxTokens256_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(0, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_TopkIdsBS64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsInputShapeCase(128, 64, 128, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_TopkWeightsBS64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsInputShapeCase(128, 128, 64, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_RouteCapacityOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / 8 + 1;
    RunUrmaVariableBsCase(128, numMaxTokensPerRank, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_OutputCapacityOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / 16 + 1;
    RunUrmaVariableBsCase(128, numMaxTokensPerRank, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_MaxRecvTokenNumOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t maxRecvTokenNum = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    RunUrmaVariableBsCase(128, 256, maxRecvTokenNum, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_MaxRecvTokenNumAboveCapacity_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 256, 4097, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, BS128_NumMaxTokens256_URMA_WorkspaceUsesCapacityMaskAndAllServers)
{
    MegaMoeTilingData tilingData{};
    tilingData.bs = 128;
    tilingData.h = 4096;
    tilingData.hiddenDim = 2048;
    tilingData.epWorldSize = 2;
    tilingData.maxOutputSize = 4096;
    tilingData.topK = 8;
    tilingData.aicNum = 28;
    tilingData.moeExpertPerRank = 8;
    tilingData.combineSyncSlotCountPerExpert = 1;
    tilingData.topoType = MegaMoeImpl::TOPO_TYPE_URMA;
    tilingData.numMaxTokensPerRank = 256;
    tilingData.rankNumPerServer = 1;

    MegaMoeImpl::WorkspaceLayout workspaceLayout(&tilingData);
    const int64_t actualMaskWorkspaceBytes =
        workspaceLayout.dispatchRelaySendQueueOffset - workspaceLayout.maskSlotOffset;
    const int64_t capacityMaskSlotBytes = MegaMoeImpl::CalcDispatchMaskAlignSize(&tilingData) + MegaMoeImpl::ALIGN_32;
    const int64_t expectedMaskWorkspaceBytes = Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize * capacityMaskSlotBytes,
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
    const int64_t actualRelayQueueBytes =
        workspaceLayout.dispatchRemoteReadyFlagSnapshotOffset - workspaceLayout.dispatchRelaySendQueueOffset;
    const int64_t serverNum = Ops::Base::CeilDiv(tilingData.epWorldSize, tilingData.rankNumPerServer);
    const int64_t expectedRelayQueueBytes = Ops::Base::CeilAlign(
        serverNum * (MegaMoeImpl::ALIGN_32 + static_cast<int64_t>(tilingData.bs) * MegaMoeImpl::ALIGN_32),
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
    const int64_t localBsMaskSlotBytes =
        MegaMoeImpl::CalcDispatchMaskAlignSizeBy(tilingData.bs, tilingData.topK) + MegaMoeImpl::ALIGN_32;
    const int64_t localBsMaskWorkspaceBytes = Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData.moeExpertPerRank) * tilingData.epWorldSize * localBsMaskSlotBytes,
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));

    EXPECT_EQ(actualMaskWorkspaceBytes, expectedMaskWorkspaceBytes);
    EXPECT_EQ(actualRelayQueueBytes, expectedRelayQueueBytes);
    EXPECT_NE(actualMaskWorkspaceBytes, localBsMaskWorkspaceBytes);
}

static void RunA8W4FormatCase(ge::Format weightOneFormat, ge::Format weightTwoFormat, ge::graphStatus expectedStatus,
                              uint64_t expectedTilingKey)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, weightOneFormat},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, weightTwoFormat},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    const auto& layout = tilingData.a8w4L1Layout;
    EXPECT_EQ(layout.aBufferNum, 3U);
    EXPECT_EQ(layout.bBufferNum, 4U);
    EXPECT_EQ(layout.tileK, 256U);
    EXPECT_EQ(layout.scaleK, 2048U);
    const uint64_t expectedBOffsets[] = {192U * 1024, 320U * 1024, 256U * 1024, 384U * 1024};
    for (uint64_t slot = 0; slot < 4U; ++slot) {
        EXPECT_EQ(layout.bOffsets[slot], expectedBOffsets[slot]);
    }
    EXPECT_EQ(layout.aOffsets[2] + layout.aSlotBytes, layout.bOffsets[0]);
    EXPECT_EQ(layout.scaleAOffsets[0], 448U * 1024);
    EXPECT_EQ(layout.scaleBOffsets[1] + layout.scaleBSlotBytes, 512U * 1024);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A8W4_URMA)
{
    RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 46802984UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A8W4_URMA_InvalidWeight2Format)
{
    RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_ND, ge::GRAPH_FAILED, 0UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_URMA)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ_C0_32},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 62138408UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

static void RunA4W4NzFormatCase(ge::Format weightOneFormat, ge::Format weightTwoFormat, ge::graphStatus expectedStatus,
                                uint64_t expectedTilingKey)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, weightOneFormat},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, weightTwoFormat},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 62138408UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA_InvalidWeight1Format)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_FAILED, 0UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA_InvalidWeight2Format)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::GRAPH_FAILED, 0UL);
}

static void RunPerExpertTensorListCase(uint32_t weightTwoTensorCount, ge::graphStatus expectedStatus)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    using TensorDescription = gert::TilingContextPara::TensorDescription;
    std::vector<TensorDescription> inputs{
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < weightTwoTensorCount; ++tensorIdx) {
        inputs.push_back({{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{2048, 64, 2}, {2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{4096, 16, 2}, {4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    }

    const std::vector<TensorDescription> outputs{
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs{
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
        {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"dispatch_quant_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
        {"shared_expert_quant_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
        {"activation_params",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
        {"activation_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
    const std::vector<uint32_t> inputInstanceNum{
        1U, 1U, 1U, 1U, 4U, weightTwoTensorCount, 4U, 4U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    };
    const std::vector<uint32_t> outputInstanceNum{1U, 1U};
    gert::TilingContextPara tilingContextPara("MegaMoe", inputs, outputs, attrs, inputInstanceNum, outputInstanceNum,
                                              &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    const uint64_t expectedTilingKey = expectedStatus == ge::GRAPH_SUCCESS ? 42607652UL : 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN_URMA_PerExpertTensorLists)
{
    RunPerExpertTensorListCase(4U, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN_URMA_PerExpertTensorListCountMismatch)
{
    RunPerExpertTensorListCase(3U, ge::GRAPH_FAILED);
}

} // namespace MegaMoeUT
