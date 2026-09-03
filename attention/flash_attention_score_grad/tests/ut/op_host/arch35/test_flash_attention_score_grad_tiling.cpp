/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/op_host/tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h"

using namespace std;

std::string A5SocInfo = "{\n"
                        "  \"hardware_info\": {\n"
                        "    \"BT_SIZE\": 0,\n"
                        "    \"load3d_constraints\": \"1\",\n"
                        "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                        "    \"Intrinsic_data_move_l12ub\": true,\n"
                        "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                        "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                        "    \"UB_SIZE\": 262144,\n"
                        "    \"L2_SIZE\": 134217728,\n"
                        "    \"L1_SIZE\": 524288,\n"
                        "    \"L0A_SIZE\": 65536,\n"
                        "    \"L0B_SIZE\": 65536,\n"
                        "    \"L0C_SIZE\": 262144,\n"
                        "    \"CORE_NUM\": 32,\n"
                        "    \"socVersion\": \"Ascend950\"\n"
                        "  }\n"
                        "}";

// SelectDeterBandSchedule requires coreNum == aicNum * 2. Default A5SocInfo only sets
// CORE_NUM, so the faker maps both AIC/AIV to 32. cube/vector counts make AIV=2*AIC.
std::string A5SocInfoHybrid = "{\n"
                              "  \"hardware_info\": {\n"
                              "    \"BT_SIZE\": 0,\n"
                              "    \"load3d_constraints\": \"1\",\n"
                              "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                              "    \"Intrinsic_data_move_l12ub\": true,\n"
                              "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                              "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                              "    \"UB_SIZE\": 262144,\n"
                              "    \"L2_SIZE\": 134217728,\n"
                              "    \"L1_SIZE\": 524288,\n"
                              "    \"L0A_SIZE\": 65536,\n"
                              "    \"L0B_SIZE\": 65536,\n"
                              "    \"L0C_SIZE\": 262144,\n"
                              "    \"CORE_NUM\": 32,\n"
                              "    \"cube_core_cnt\": 32,\n"
                              "    \"vector_core_cnt\": 64,\n"
                              "    \"socVersion\": \"Ascend950\"\n"
                              "  }\n"
                              "}";

class FlashAttentionScoreGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreGradTiling TearDown" << std::endl;
    }
};

static Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo MakeA5CompileInfo()
{
    return {64,       // aivNum
            32,       // aicNum
            196608,   // ubSize
            524288,   // l1Size
            65536,    // l0aSize
            65536,    // l0bSize
            131072,   // l0cSize
            33554432, // l2CacheSize
            32,       // coreNum
            platform_ascendc::SocVersion::ASCEND950};
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_0)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q: S1=256, B=1, H1=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577349537174544;
    std::string expectTilingData =
        "32 1 1 1 256 256 128 128 4575657222443697349 255 65536 65536 0 0 0 2 1 0 0 2 1099511627776 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 8192 4 8192 "
        "8192 4 8192 8192 4 8192 0 0 0 4 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 0 0 0 0 0 1 32768 16384 "
        "16384 1 32768 16384 16384 1 32768 16384 16384 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_1)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21562880};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_2)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{1, 1}, {1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724962;
    std::string expectTilingData =
        "32 1 1 1 1 1 1 1 4575657222473777152 255 65536 0 2 0 0 2 1 0 0 2 4294967296 281474976710656 0 0 0 1 "
        "4294967360 4294967297 1 4294967424 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 1 1 1 1 1 1 1 1 1 0 0 0 "
        "1 61440 30720 114688 30720 1 32 1 32 1 0 0 0 0 0 0 0 0 0 0 1 1 16384 1 1 1 16384 1 1 1 16384 1 65536 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_3)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 1, 64, 129}, {1, 1, 64, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{1, 1, 128, 129}, {1, 1, 128, 129}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08804509063256238f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350074044432;
    std::string expectTilingData =
        "32 1 1 1 64 128 129 129 4575657222443651324 255 2147483647 2147483647 2 0 0 3 1 0 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8192 8256 1 8256 "
        "16512 1 16512 16512 1 16512 0 0 0 1 61440 30720 114688 30720 1 8192 1 8192 1 0 0 0 0 0 0 0 0 0 0 1 8256 16384 "
        "8256 1 16512 16384 128 1 16512 16384 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_4)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{2, 8, 2000, 8}, {2, 8, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{2, 8, 2000, 8}, {2, 8, 2000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{2, 2000, 8, 128}, {2, 2000, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{2, 2000, 8, 64}, {2, 2000, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07216878364870323f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 20125462445953072;
    std::string expectTilingData =
        "32 2 8 1 2000 2000 192 128 4575657222441520442 255 2147483647 2147483647 2 0 256 1 1 0 0 0 0 "
        "72057598332895232 0 0 0 16 549755813952 343597383696 16 343597383808 549755813920 0 0 128 256 384 512 640 768 "
        "896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 "
        "3712 3840 3968 0 0 0 0 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 "
        "2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096 0 0 0 0 2000000 96000 64 96000 96000 64 "
        "96000 64000 64 64000 0 0 0 64 61440 30720 114688 30720 66 3200 600479950314048 27776 1 242442313924672 "
        "20203526216832 147 4 4 59 59 500 500 32000 6 6144000 18432 6144 6 6144000 18432 6144 4 4096000 18432 4096 "
        "65536 24678912 49292288 0 65701376 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {87698944};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_5)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{64, 16, 123, 8}, {64, 16, 123, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{64, 16, 123, 8}, {64, 16, 123, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{64, 123, 16, 389}, {64, 123, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{64, 64, 16, 389}, {64, 64, 16, 389}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.05070201265633938f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140301101207586;
    std::string expectTilingData =
        "32 64 16 1 123 64 389 389 4575657222437055722 255 2147483647 2147483647 2 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "528280977472 528280977467 1 274877907072 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 352 384 416 "
        "448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 160 192 224 256 "
        "288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 0 0 0 0 "
        "251904 1531104 32 1531104 796672 32 796672 796672 32 796672 0 0 0 32 61440 30720 114688 30720 9 6144 9 6144 1 "
        "0 0 0 0 0 0 0 0 0 0 47 48995328 16384 7168 25 25493504 16384 16384 25 25493504 16384 16384 65536 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {49349120};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_6)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{3, 1, 128, 8}, {3, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{3, 128, 128}, {3, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{3, 121, 128}, {3, 121, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140299490594850;
    std::string expectTilingData =
        "32 3 1 1 128 121 128 128 4575657222443697395 255 2147483647 2147483647 2 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "549755813952 549755813952 1 519691042944 4294967299 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15488 16384 3 16384 "
        "15488 3 15488 15488 3 15488 0 0 0 3 61440 30720 114688 30720 1 15488 1 15488 1 0 0 0 0 0 0 0 0 0 0 1 49152 "
        "16384 16384 1 46464 16384 13696 1 46464 16384 13696 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {25756160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_7)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // k
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // v
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dy
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // attention_in
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dv
            {{{1, 4096, 1, 200}, {1, 4096, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07071067811865475f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140300564337952;
    std::string expectTilingData =
        "32 1 1 1 73 4096 200 200 4561245704492732611 204 2147483647 0 2 0 0 1 1 0 0 8589934594 8796093022209 "
        "72339069014638592 0 0 0 1 313532612672 313532612617 32 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 299008 7300 2 7300 409600 2 409600 409600 2 409600 0 0 0 2 61440 30720 114688 30720 10 22528 0 30720 "
        "1099511627777 244091581366274 15255723892224 111 1 1 37 36 37 36 73 1 14600 16384 14600 25 819200 16384 16384 "
        "25 819200 16384 16384 65536 168448 3445760 0 6723072 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {27699200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_8)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350610917264;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_9)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // k
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // v
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dy
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dv
            {{{81, 2, 128}, {81, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.08838834764831843f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(40)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(70)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703249444018098;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {25756160};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_10)
{
    int64_t actual_seq_qlist[3] = {13, 14, 81};
    int64_t actual_seq_kvlist[3] = {13, 14, 81};
    int64_t prefix_list[3] = {10, 12, 60};
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        64,                                     // aivNum
        32,                                     // aicNum
        196608,                                 // ubSize
        524288,                                 // l1Size
        65536,                                  // l0aSize
        65536,                                  // l0bSize
        131072,                                 // l0cSize
        33554432,                               // l2CacheSize
        32,                                     // coreNum
        platform_ascendc::SocVersion::ASCEND950 // socVersion
    };
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            // q
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // atten_mask
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{81, 2, 8}, {81, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            // actual_seq_qlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            // actual_seq_kvlen
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{81, 2, 256}, {81, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0625f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577350610917264;
    std::string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_11_tnd_same_as_input)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData =
        "32 4 1 2 384 384 32 32 4575657222452086003 255 45 2 2 0 0 4294967300 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 3 549755813952 549755813952 3 549755814016 4294967328 0 0 1 2 3 4 5 6 7 8 9 10 11 13 "
        "14 15 17 18 19 20 22 23 24 26 27 28 29 30 31 32 33 34 35 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 13 14 15 17 18 19 "
        "20 22 23 24 26 27 28 29 30 31 32 33 34 35 36 0 0 0 0 16992 1948 32 1948 974 32 974 974 32 974 0 0 0 32 "
        "61440 30720 114688 30720 1 16992 1 16896 1 0 0 0 0 0 0 0 0 0 0 1 62336 16384 13184 1 31168 16384 14784 1 "
        "31168 16384 14784 65536 328192 459776 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 "
        "3 3 3 3 3 3 0 0 0 0 0 0 16384 16384 16384 16384 16384 16384 16384 16384 81920 81920 81920 81920 81920 81920 "
        "81920 81920 81920 81920 81920 81920 81920 81920 229376 229376 229376 229376 229376 229376 229376 229376 0 0 "
        "0 0 0 0 16384 16384 16384 16384 16384 16384 16384 16384 81920 81920 81920 81920 81920 81920 81920 81920 "
        "81920 81920 81920 81920 81920 81920 229376 229376 229376 229376 229376 229376 229376 229376 0 0 0 0 0 0 1 1 "
        "1 1 1 1 1 1 5 5 5 5 5 5 5 5 5 5 5 5 5 5 14 14 14 14 14 14 14 14 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21562880};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_12_pad_causal_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19182080395580448;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 21856 5462 3 5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "3 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_13_pad_band_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19747229372257328;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 64 64 0 0 0 1 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 2 549755813952 549755813952 2 549755814016 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 16384 4096 4 4096 4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "2 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_15_pad_prefix5)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146288;
    std::string expectTilingData =
        "32 2 2 1 256 256 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 21474836482 1099511627776 "
        "844424930131968 0 0 0 2 549755813952 549755813952 2 549755814016 4294967308 0 0 1 2 4 5 6 8 9 10 12 13 14 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 4 5 6 8 9 10 12 13 14 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 21856 5462 12 5454 5462 12 5454 5462 12 5454 0 0 0 12 61440 30720 114688 30720 1 21856 1 "
        "21728 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 16384 65536 328192 590848 "
        "0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_16_pad_prefix6)
{
    int64_t prefix_list[2] = {16, 24};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724960;
    std::string expectTilingData =
        "32 2 2 1 256 256 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 25769803778 8796093022212 "
        "844424930131968 0 0 0 2 549755813952 549755813952 2 549755814016 4294967308 0 0 1 2 4 5 6 8 9 10 12 13 14 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 4 5 6 8 9 10 12 13 14 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 21856 5462 12 5454 5462 12 5454 5462 12 5454 0 0 0 12 61440 30720 114688 30720 1 21856 1 "
        "21728 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 16384 65536 328192 590848 "
        "0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_17_tnd_sparse7)
{
    int64_t actual_seq_qlist[3] = {0, 32, 64};
    int64_t actual_seq_kvlist[3] = {0, 32, 64};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{64, 2, 8}, {64, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 2, 8}, {64, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 2, 64}, {64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData =
        "32 3 2 1 32 32 64 64 4575657222448611328 255 16 16 0 0 0 4 1 0 0 30064771074 8796093022213 844424930131968 "
        "0 0 0 1 137438953536 137438953504 1 137438953600 4294967300 2 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 2048 "
        "4 2048 2048 4 2048 2048 4 2048 0 0 0 4 61440 30720 114688 30720 1 1024 1 1024 4294967297 0 0 0 0 0 0 0 0 0 "
        "0 1 8192 16384 8192 1 8192 16384 8192 1 8192 16384 8192 65536 98816 132096 0 0 0 0 0 0 0 0 0 0 1 2 2 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21136896};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_18_tnd_sparse8_inner_pse)
{
    int64_t actual_seq_qlist[3] = {32, 64, 96};
    int64_t actual_seq_kvlist[3] = {32, 64, 96};
    int64_t q_start_list[1] = {0};
    int64_t kv_start_list[1] = {0};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, q_start_list},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, kv_start_list},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577349000304272;
    std::string expectTilingData =
        "32 3 2 1 32 32 64 64 4575657222448611328 255 16 16 0 0 0 4 25769803778 3 0 34359738370 8796093022214 "
        "844424930131968 0 0 0 1 137438953536 137438953504 1 137438953600 4294967302 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1024 2048 6 2048 2048 6 2048 2048 6 2048 0 0 0 6 61440 30720 114688 30720 1 1024 1 1024 1 0 0 0 0 0 0 "
        "0 0 0 0 1 12288 16384 12288 1 12288 16384 12288 1 12288 16384 12288 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 2 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 2048 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 1024 2048 2048 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 1 1 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_19_outer_pse_bn1s)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64}, {1, 1, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 3 4294967297 1 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 1 4096 "
        "4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_20_inner_pse_bn)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 1, 64, 8}, {2, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 64, 8}, {2, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724450;
    std::string expectTilingData =
        "32 2 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 3 21474836482 2 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 2 4096 "
        "4096 2 4096 4096 2 4096 0 0 0 2 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 8192 16384 "
        "8192 1 8192 16384 8192 1 8192 16384 8192 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_21_dropout_dropmask)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1825}, {1825}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 73, 8}, {1, 1, 73, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 73, 1, 200}, {1, 73, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 200, 1, 200}, {1, 200, 1, 200}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.07071067811865475f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140300564336928;
    std::string expectTilingData =
        "32 1 1 1 73 200 200 200 4561245704492732611 204 2147483647 2147483647 2 0 0 1 1 0 0 0 0 72058693549555712 0 "
        "0 0 1 313532612672 313532612617 2 309237645440 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7328 3650 4 "
        "3650 10000 4 10000 10000 4 10000 0 0 0 4 61440 30720 114688 30720 1 7328 0 23360 1 244091581366276 "
        "15255723892224 111 1 1 19 16 19 16 73 1 14600 16384 14600 1 40000 16384 7232 1 40000 16384 7232 65536 "
        "168448 373760 0 579072 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21555200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_22_bn2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953723938;
    std::string expectTilingData =
        "32 1 4 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 4096 4 4096 "
        "4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 16384 1 "
        "16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_23_bn2s2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953723936;
    std::string expectTilingData =
        "32 1 1 1 64 256 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 2 549755814016 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8192 2048 2 2048 "
        "8192 2 8192 8192 2 8192 0 0 0 2 61440 30720 114688 30720 1 8192 1 8192 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 16384 16384 16384 1 16384 16384 16384 65536 98816 164864 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21202432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_24_bn2_multiblk)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16, 8, 256, 8}, {16, 8, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 8, 256, 8}, {16, 8, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 8, 64}, {16, 256, 8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19773617651322930;
    std::string expectTilingData =
        "32 16 8 1 256 256 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 68719476768 0 0 16 32 48 64 80 96 112 128 144 160 176 192 208 224 "
        "240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 0 0 0 0 16 32 48 64 80 96 112 128 144 "
        "160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512 0 0 0 0 262144 "
        "65536 32 65536 65536 32 65536 65536 32 65536 0 0 0 32 61440 30720 114688 30720 9 16384 9 16384 1 0 0 0 0 0 "
        "0 0 0 0 0 2 2097152 16384 16384 2 2097152 16384 16384 2 2097152 16384 16384 65536 2424832 3604480 0 0 0 0 0 "
        "0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {25755648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_25_tnd_all_same)
{
    int64_t actual_seq_qlist[3] = {64, 128, 192};
    int64_t actual_seq_kvlist[3] = {64, 128, 192};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{192, 2, 8}, {192, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{192, 2, 8}, {192, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192, 2, 32}, {192, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145266;
    std::string expectTilingData =
        "32 3 2 1 64 64 32 32 4575657222452086003 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967302 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 2048 6 2048 2048 6 2048 "
        "2048 6 2048 0 0 0 6 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 12288 16384 12288 1 "
        "12288 16384 12288 1 12288 16384 12288 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_27_fp32_large_d)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 320}, {1, 1, 64, 320}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.055901699437f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577351147784208;
    std::string expectTilingData =
        "32 1 1 1 64 64 320 320 4575657222438451502 255 2147483647 2147483647 0 0 0 3 1 0 0 0 0 0 0 0 0 1 "
        "274877906976 274877906976 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 20480 1 "
        "20480 20480 1 20480 20480 1 20480 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 "
        "20480 16384 4096 1 20480 16384 4096 1 20480 16384 4096 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_28_atten_mask_4d)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 2 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 0 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967298 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 2 4096 "
        "4096 2 4096 4096 2 4096 0 0 0 2 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 8192 16384 "
        "8192 1 8192 16384 8192 1 8192 16384 8192 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_29_tnd_outer_pse_1d)
{
    int64_t actual_seq_qlist[3] = {32, 64, 96};
    int64_t actual_seq_kvlist[3] = {32, 64, 96};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96, 2, 8}, {96, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 3 2 1 32 32 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 4294967297 1 0 0 0 0 0 0 0 1 "
        "137438953536 137438953504 1 137438953600 4294967302 0 0 1 2 3 4 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1024 2048 6 2048 "
        "2048 6 2048 2048 6 2048 0 0 0 6 61440 30720 114688 30720 1 1024 1 1024 1 0 0 0 0 0 0 0 0 0 0 1 12288 16384 "
        "12288 1 12288 16384 12288 1 12288 16384 12288 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_30_alibi_pse)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 1024}, {1, 1, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 64}, {1, 1, 1024, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146800;
    std::string expectTilingData =
        "32 1 1 1 1024 1024 64 64 4575657222448611328 255 2147483647 0 0 0 0 3 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 8 549755813952 549755813952 8 549755814016 8589934610 0 0 2 4 6 9 11 13 15 19 21 23 "
        "28 30 36 38 45 47 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 9 11 13 15 19 21 23 28 30 36 38 45 47 55 64 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 58272 3641 18 3639 3641 18 3639 3641 18 3639 0 0 0 18 61440 30720 "
        "114688 30720 2 27552 2 27232 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 "
        "16384 65536 328192 590848 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_softmax_dim)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 2, 974, 8}, {1, 2, 974, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 974, 8}, {1, 2, 974, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 1, 32}, {974, 1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(45)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_atten_mask_shape)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dropmask_keepprob)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_fp8)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pad_tokens)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-100)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_inner_pse_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_tokens)
{
    // Prefix-sum must stay unequal so layout remains TND (all-same would rewrite to pad).
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 64}, {974, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_atten_mask_rank)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 64, 64}, {1, 1, 1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse5)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    int64_t prefix_list[4] = {8, 8, 8, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pad_sparse7)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_prefix6_empty_prefix)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pse_type)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_softmax_max_dim)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 4}, {1, 1, 64, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dscale)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_atten_mask_4d_bn)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 5, 64, 64}, {3, 5, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 64, 8}, {1, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_31_all_mask)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 4294967298 274877906944 0 "
        "0 0 0 1 274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 "
        "1 4096 4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 "
        "16384 4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_32_pad_right_down_causal)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 1 274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 4096 4096 1 4096 4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 "
        "0 0 0 0 1 4096 16384 4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_33_prefix5_empty)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724960;
    std::string expectTilingData =
        "32 2 2 1 256 256 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 4294967298 1099511627776 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 4294967312 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "16384 4096 16 4096 4096 16 4096 4096 16 4096 0 0 0 16 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 "
        "0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 16384 65536 328192 590848 0 0 0 0 0 0 0 0 0 "
        "0 0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_34_outer_pse_bnss)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 64, 64}, {2, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 4096 4 4096 "
        "4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 16384 1 "
        "16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_35_outer_pse_1nss)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145778;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 8589934593 0 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_36_atten_mask_4d_b1)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1, 64, 64}, {2, 1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 1 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_37_atten_mask_4d_11)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 2 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_38_qkv_start_idx)
{
    int64_t q_start_list[1] = {8};
    int64_t kv_start_list[1] = {4};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, q_start_list},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, kv_start_list},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145266;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 17179869192 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 1 4096 "
        "4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_39_dropout_s2_not_div8)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 65, 8}, {1, 1, 65, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 65, 8}, {1, 1, 65, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 65, 1, 64}, {1, 65, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.9f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145522;
    std::string expectTilingData =
        "32 1 1 1 65 65 64 64 4568451461326831616 229 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 279172874304 "
        "279172874241 1 279172874368 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4256 4160 1 4160 4160 1 4160 "
        "4160 1 4160 0 0 0 1 61440 30720 114688 30720 1 4256 1 4256 0 0 0 0 0 0 0 0 0 0 0 1 4160 16384 4160 1 4160 "
        "16384 4160 1 4160 16384 4160 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_40_pad_gqa)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 64, 8}, {1, 4, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 4, 64}, {1, 64, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145264;
    std::string expectTilingData =
        "32 1 2 2 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 274877907008 "
        "274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 2048 4 2048 "
        "2048 4 2048 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 16384 1 8192 "
        "16384 8192 1 8192 16384 8192 65536 147968 197632 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21218816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_41_atten_mask_bool)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146290;
    std::string expectTilingData =
        "32 1 1 1 64 64 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 2 274877906944 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 1 4096 "
        "4096 1 4096 4096 1 4096 0 0 0 1 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 4096 16384 "
        "4096 1 4096 16384 4096 1 4096 16384 4096 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_42_inner_pse_2d)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 64, 8}, {2, 2, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 2, 64}, {2, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19140298953724450;
    std::string expectTilingData =
        "32 2 2 1 64 64 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 21474836482 2 0 0 0 0 0 0 0 1 "
        "274877907008 274877907008 1 274877907072 4294967300 0 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 2 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4096 4096 4 4096 "
        "4096 4 4096 4096 4 4096 0 0 0 4 61440 30720 114688 30720 1 4096 1 4096 1 0 0 0 0 0 0 0 0 0 0 1 16384 16384 "
        "16384 1 16384 16384 16384 1 16384 16384 16384 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23396864};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_compress_mask_not_square)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 2048}, {1024, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_softmax_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_attention_in_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_attention_in_rank)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_softmax_max_rank)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 64, 8}, {1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dropmask_dtype)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_dropmask_size)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(0.8f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse7_inner_pse)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse8_inner_pse_s1s2)
{
    int64_t actual_seq_qlist[3] = {32, 80, 160};
    int64_t actual_seq_kvlist[3] = {32, 64, 96};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{160, 2, 8}, {160, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{160, 2, 8}, {160, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{160, 2, 64}, {160, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96, 2, 64}, {96, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_prefix_len)
{
    int64_t prefix_list[3] = {8, 8, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 256, 8}, {2, 2, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 2, 64}, {2, 256, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_43_prefix5_deter)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 512, 2, 64}, {2, 512, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401840;
    std::string expectTilingData =
        "32 2 2 1 512 512 64 64 4575657222448611328 255 65536 65536 0 0 0 1 1 0 0 21474836482 2199023255552 "
        "844424930131968 0 0 0 4 549755813952 549755813952 4 549755814016 8589934612 0 0 2 4 7 11 16 18 20 23 27 32 "
        "34 36 39 43 48 50 52 55 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 7 11 16 18 20 23 27 32 34 36 39 43 48 50 52 "
        "55 59 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52448 6554 20 6546 6554 20 6546 6554 20 6546 0 0 0 20 61440 30720 "
        "114688 30720 2 21728 2 21344 1 0 0 0 0 0 0 0 0 0 0 1 131072 16384 16384 1 131072 16384 16384 1 131072 16384 "
        "16384 65536 590336 1115136 0 0 0 0 0 0 0 0 0 1639936 7931392 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28909056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_44_sbh_prefix5_deter)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 2, 128}, {512, 2, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401840;
    std::string expectTilingData =
        "32 2 2 1 512 512 64 64 4575657222448611328 255 65536 65536 0 0 0 2 1 0 0 21474836482 2199023255552 "
        "844424930131968 0 0 0 4 549755813952 549755813952 4 549755814016 8589934612 0 0 2 4 7 11 16 18 20 23 27 32 "
        "34 36 39 43 48 50 52 55 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 7 11 16 18 20 23 27 32 34 36 39 43 48 50 52 "
        "55 59 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52448 6554 20 6546 6554 20 6546 6554 20 6546 0 0 0 20 61440 30720 "
        "114688 30720 2 21728 2 21344 1 0 0 0 0 0 0 0 0 0 0 1 131072 16384 16384 1 131072 16384 16384 1 131072 16384 "
        "16384 65536 590336 1115136 0 0 0 0 0 0 0 0 0 1639936 7931392 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28909056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_45_tnd_prefix6_deter)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    int64_t prefix_list[4] = {8, 8, 8, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3072, 2048}, {3072, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401968;
    std::string expectTilingData =
        "32 4 2 1 384 384 32 32 4575657222448611328 255 65536 65536 0 0 0 4 1 0 0 25769803778 8796093022212 "
        "844424930131968 0 0 0 3 549755813952 549755813952 3 549755814016 8589934610 0 0 2 4 6 8 10 12 14 16 18 20 "
        "22 24 26 28 30 32 34 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 30208 3464 18 3448 3464 18 3448 3464 18 3448 0 0 0 18 61440 30720 "
        "114688 30720 1 30208 1 30112 1 0 0 0 0 0 0 0 0 0 0 1 62336 16384 13184 1 62336 16384 13184 1 62336 16384 "
        "13184 65536 328192 590848 0 0 0 0 0 0 0 0 0 853504 7144960 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 16384 16384 16384 81920 81920 81920 81920 81920 81920 81920 81920 81920 "
        "229376 229376 229376 229376 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 16384 16384 16384 81920 81920 81920 "
        "81920 81920 81920 81920 81920 81920 229376 229376 229376 229376 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 "
        "1 5 5 5 5 5 5 5 5 5 14 14 14 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28122624};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_46_bnsd_prefix5_deter)
{
    int64_t prefix_list[2] = {16, 16};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{512, 512}, {512, 512}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 512, 8}, {2, 2, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, prefix_list},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 512, 64}, {2, 2, 512, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19705447930401840;
    std::string expectTilingData =
        "32 2 2 1 512 512 64 64 4575657222448611328 255 65536 65536 0 0 0 3 1 0 0 21474836482 2199023255552 "
        "844424930131968 0 0 0 4 549755813952 549755813952 4 549755814016 8589934612 0 0 2 4 7 11 16 18 20 23 27 32 "
        "34 36 39 43 48 50 52 55 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 7 11 16 18 20 23 27 32 34 36 39 43 48 50 52 "
        "55 59 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52448 6554 20 6546 6554 20 6546 6554 20 6546 0 0 0 20 61440 30720 "
        "114688 30720 2 21728 2 21344 1 0 0 0 0 0 0 0 0 0 0 1 131072 16384 16384 1 131072 16384 16384 1 131072 16384 "
        "16384 65536 590336 1115136 0 0 0 0 0 0 0 0 0 1639936 7931392 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28909056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_47_causal_s1_lt_s2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146288;
    std::string expectTilingData =
        "32 1 1 1 4096 16384 64 64 4575657222448611328 255 2147483647 0 0 0 528 1 1 0 0 8589934594 8796093022209 "
        "281479271677952 0 0 0 32 549755813952 549755813952 128 549755814016 73014444064 0 0 17 35 52 71 88 108 125 "
        "146 168 185 208 232 249 274 300 317 344 372 401 431 462 479 511 561 596 632 669 729 793 861 1023 0 0 0 0 17 "
        "35 52 71 88 108 125 146 168 185 208 232 249 274 300 317 344 372 401 431 462 479 511 561 596 632 669 729 793 "
        "861 1023 4096 0 0 0 0 2097152 8192 32 8192 32768 32 32768 32768 32 32768 0 0 0 32 61440 30720 114688 30720 "
        "69 8192 69 8192 1099511627777 0 0 0 0 0 0 0 0 0 0 1 262144 16384 16384 1 1048576 16384 16384 1 1048576 "
        "16384 16384 65536 1114624 5309440 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {30475776};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_48_rd_causal_s1_lt_s2)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 4096, 8}, {1, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4096, 1, 64}, {1, 4096, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16384, 1, 64}, {1, 16384, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146288;
    std::string expectTilingData =
        "32 1 1 1 4096 16384 64 64 4575657222448611328 255 2147471359 12288 0 0 0 1 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 32 549755813952 549755813952 128 549755814016 485331304480 0 0 113 226 339 452 565 "
        "678 791 904 1017 1130 1243 1356 1469 1582 1695 1808 1921 2034 2147 2260 2373 2486 2599 2712 2825 2938 3051 "
        "3167 3305 3468 3674 0 0 0 0 113 226 339 452 565 678 791 904 1017 1130 1243 1356 1469 1582 1695 1808 1921 "
        "2034 2147 2260 2373 2486 2599 2712 2825 2938 3051 3167 3305 3468 3674 4096 0 0 0 0 2097152 8192 32 8192 "
        "32768 32 32768 32768 32 32768 0 0 0 32 61440 30720 114688 30720 69 8192 69 8192 1 0 0 0 0 0 0 0 0 0 0 1 "
        "262144 16384 16384 1 1048576 16384 16384 1 1048576 16384 16384 65536 1114624 5309440 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {30475776};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_49_band_split_block)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 8, 4096, 8}, {2, 8, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 8, 4096, 8}, {2, 8, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 8, 128}, {2, 4096, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 18577349537174544;
    std::string expectTilingData =
        "32 2 8 1 4096 4096 128 128 4575657222443697349 255 256 256 0 0 154 1 1 0 0 17179869186 8796093022211 "
        "562954248388608 0 0 0 32 549755813952 549755813952 32 549755814016 330712481824 0 0 526 1024 1550 2048 2574 "
        "3072 3598 4096 4622 5120 5646 6144 6670 7168 7694 8192 8718 9216 9742 10240 10766 11264 11790 12288 12814 "
        "13312 13838 14336 14862 15360 15886 0 0 0 0 526 1024 1550 2048 2574 3072 3598 4096 4622 5120 5646 6144 6670 "
        "7168 7694 8192 8718 9216 9742 10240 10766 11264 11790 12288 12814 13312 13838 14336 14862 15360 15886 16384 "
        "0 0 0 0 8388608 262144 32 262144 262144 32 262144 262144 32 262144 0 0 0 32 61440 30720 114688 30720 274 "
        "2048 274 2048 1 0 0 0 0 0 0 0 0 0 0 8 8388608 16384 16384 8 8388608 16384 16384 8 8388608 16384 16384 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21037056};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_50_alibi_1nhs)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1024, 1024}, {1, 1, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146800;
    std::string expectTilingData =
        "32 1 1 1 1024 1024 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 8 549755813952 549755813952 8 549755814016 8589934610 0 0 2 4 6 9 11 13 15 19 21 23 "
        "28 30 36 38 45 47 55 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 9 11 13 15 19 21 23 28 30 36 38 45 47 55 64 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 58272 3641 18 3639 3641 18 3639 3641 18 3639 0 0 0 18 61440 30720 "
        "114688 30720 2 27552 2 27232 1 0 0 0 0 0 0 0 0 0 0 1 65536 16384 16384 1 65536 16384 16384 1 65536 16384 "
        "16384 65536 328192 590848 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21825024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_51_alibi_bnhs)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 1024, 1024}, {2, 2, 1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2, 2, 1024, 8}, {2, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 1024, 8}, {2, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 1024, 2, 64}, {2, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907146800;
    std::string expectTilingData =
        "32 2 2 1 1024 1024 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 8 549755813952 549755813952 8 549755814016 21474836509 0 0 5 11 18 23 31 45 63 68 74 "
        "79 86 94 103 119 131 137 142 149 157 166 182 194 199 205 212 220 229 239 0 0 0 0 0 0 0 5 11 18 23 31 45 63 "
        "68 74 79 86 94 103 119 131 137 142 149 157 166 182 194 199 205 212 220 229 239 256 0 0 0 0 0 0 0 144640 "
        "9040 29 9024 9040 29 9024 9040 29 9024 0 0 0 29 61440 30720 114688 30720 5 21760 5 21504 1 0 0 0 0 0 0 0 0 "
        "0 0 1 262144 16384 16384 1 262144 16384 16384 1 262144 16384 16384 65536 1114624 2163712 0 0 0 0 0 0 0 0 0 "
        "0 0 ";
    std::vector<size_t> expectWorkspaces = {24184320};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse4_tokens)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_sparse7_tokens)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-200)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pse_alibi_shape)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1024, 8}, {1, 1, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1024, 1, 64}, {1, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_pse_alibi_shape)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 64, 64}, {1, 2, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_pse_not_causal)
{
    int64_t actual_seq_qlist[4] = {128, 384, 768, 974};
    int64_t actual_seq_kvlist[4] = {128, 384, 768, 974};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 8, 8}, {1, 2, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{974, 2, 8}, {974, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{974, 2, 32}, {974, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_pse_pad_not_alibi)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_regbase_h_helpers)
{
    using optiling::fag::AbsCeil;
    using optiling::fag::CeilDivideBy;
    using optiling::fag::Gcd;
    using optiling::fag::SliceVector;

    EXPECT_EQ(AbsCeil(5, 3), 2);
    EXPECT_EQ(AbsCeil(-5, 3), -2);
    EXPECT_EQ(AbsCeil(5, -3), -2);
    EXPECT_EQ(Gcd(48, 18), 6);
    EXPECT_EQ(Gcd(7, 0), 7);
    EXPECT_EQ(CeilDivideBy(static_cast<int64_t>(10), static_cast<int64_t>(3)), 4);
    EXPECT_EQ(CeilDivideBy(static_cast<int64_t>(10), static_cast<int64_t>(0)), 0);
    const std::vector<int64_t> arr{1, 2, 3, 4, 5, 6};
    const std::vector<int64_t> sliced = SliceVector(arr, static_cast<int64_t>(2));
    ASSERT_EQ(sliced.size(), 3U);
    EXPECT_EQ(sliced[0], 1);
    EXPECT_EQ(sliced[1], 3);
    EXPECT_EQ(sliced[2], 5);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_52_tnd_deter_b129)
{
    constexpr int64_t kBatch = 129;
    int64_t actual_seq_qlist[kBatch];
    int64_t actual_seq_kvlist[kBatch];
    int64_t acc = 0;
    for (int64_t i = 0; i < kBatch; ++i) {
        acc += (i + 1 == kBatch) ? 256 : 32;
        actual_seq_qlist[i] = acc;
        actual_seq_kvlist[i] = acc;
    }
    const int64_t tLen = acc;
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{tLen, 2, 8}, {tLen, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{tLen, 2, 8}, {tLen, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tLen, 2, 32}, {tLen, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19742831325745328;
    std::string expectTilingData =
        "32 129 2 1 256 256 32 32 4575657222452086003 255 2147483647 2147483647 0 0 0 4 1 0 0 0 0 0 0 0 0 2 "
        "549755813952 549755813952 2 549755814016 38654705694 0 0 9 18 27 36 45 54 63 72 81 90 99 108 117 126 135 "
        "144 153 162 171 180 189 198 207 216 225 234 243 252 261 0 0 0 0 0 0 9 18 27 36 45 54 63 72 81 90 99 108 117 "
        "126 135 144 153 162 171 180 189 198 207 216 225 234 243 252 261 264 0 0 0 0 0 0 13120 9285 30 9263 9285 30 "
        "9263 9285 30 9263 0 0 0 30 61440 30720 114688 30720 1 13120 1 12736 1 0 0 0 0 0 0 0 0 0 0 1 278528 16384 "
        "16384 1 278528 16384 16384 1 278528 16384 16384 65536 1180160 2294784 0 0 0 0 0 0 0 1 9 0 0 9 1 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 2 0 2048 4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576 26624 28672 "
        "30720 32768 34816 36864 38912 40960 43008 45056 47104 49152 51200 53248 55296 57344 59392 61440 63488 65536 "
        "67584 69632 71680 73728 75776 77824 79872 81920 83968 86016 88064 90112 92160 94208 96256 98304 100352 "
        "102400 104448 106496 108544 110592 112640 114688 116736 118784 120832 122880 124928 126976 129024 131072 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 2048 4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576 26624 28672 "
        "30720 32768 34816 36864 38912 40960 43008 45056 47104 49152 51200 53248 55296 57344 59392 61440 63488 65536 "
        "67584 69632 71680 73728 75776 77824 79872 81920 83968 86016 88064 90112 92160 94208 96256 98304 100352 "
        "102400 104448 106496 108544 110592 112640 114688 116736 118784 120832 122880 124928 126976 129024 131072 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 "
        "58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 "
        "122 124 126 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 262176 -1 -1 -1 -1 -1 -1 -1 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24380928};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_query_rope_xor)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 64, 8}, {1, 1, 64, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_fail_tnd_eod_nonzero)
{
    int64_t actual_seq_qlist[3] = {32, 16, 8};
    int64_t actual_seq_kvlist[3] = {32, 16, 8};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{32, 2, 8}, {32, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 2, 8}, {32, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 2, 32}, {32, 2, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.17677669529663687f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_53_gqa_dense_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{5, 2, 1024, 8}, {5, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 2, 1024, 8}, {5, 2, 1024, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5, 1024, 2, 64}, {5, 1024, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 1024, 1, 64}, {5, 1024, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19707646953656368;
    std::string expectTilingData =
        "32 5 1 2 1024 1024 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 8 "
        "549755813952 549755813952 8 549755814016 85899345952 0 0 20 40 60 80 100 120 140 160 180 200 220 240 260 "
        "280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 0 0 0 0 20 40 60 80 100 120 140 160 "
        "180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 620 640 0 0 0 0 "
        "327680 20480 32 20480 10240 32 10240 10240 32 10240 0 0 0 32 61440 30720 114688 30720 11 20480 11 20480 1 0 "
        "0 0 0 0 0 0 0 0 0 1 655360 16384 16384 1 327680 16384 16384 1 327680 16384 16384 65536 2687488 3998720 0 0 "
        "0 0 0 0 0 1 24 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {26281472};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_55_deter_left_up)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19745030349001776;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 21856 5462 3 5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "3 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_56_deter_nomask_neg_next)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-64)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19745030349001776;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 65536 -64 0 0 0 1 1 0 0 2 1099511627776 562949953421312 0 0 "
        "0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 21856 5462 3 "
        "5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 0 0 0 0 0 0 1 16384 "
        "16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 3 0 0 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_57_bn2_multiblk_sparse)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19773617651323954;
    std::string expectTilingData =
        "32 16 16 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 103079215136 0 0 32 64 96 128 160 192 224 "
        "256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 "
        "64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 "
        "928 960 992 1024 0 0 0 0 524288 131072 32 131072 131072 32 131072 131072 32 131072 0 0 0 32 61440 30720 "
        "114688 30720 18 2048 18 2048 1 0 0 0 0 0 0 0 0 0 0 4 4194304 16384 16384 4 4194304 16384 16384 4 4194304 "
        "16384 16384 65536 2424832 3604480 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {25755648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_58_bn2_multiblk_invalid)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16, 256, 8}, {16, 16, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 256, 16, 64}, {16, 256, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19773617651323954;
    std::string expectTilingData =
        "32 16 16 1 256 256 64 64 4575657222448611328 255 16 16 0 0 0 1 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 2 549755813952 549755813952 2 549755814016 137438953504 0 0 32 64 96 128 160 192 224 "
        "256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 "
        "64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 "
        "928 960 992 1024 0 0 0 0 524288 131072 32 131072 131072 32 131072 131072 32 131072 0 0 0 32 61440 30720 "
        "114688 30720 18 2048 18 2048 1 0 0 0 0 0 0 0 0 0 0 4 4194304 16384 16384 4 4194304 16384 16384 4 4194304 "
        "16384 16384 65536 2424832 3604480 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {25755648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_59_nz_out)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 4, 2048, 8}, {1, 4, 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 4, 2048, 8}, {1, 4, 2048, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2048, 4, 72}, {1, 2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.117851f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703249444016176;
    std::string expectTilingData =
        "32 1 4 1 2048 2048 72 72 4575657222447651805 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 72057594037927936 "
        "0 0 0 16 549755813952 549755813952 16 549755814016 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 "
        "352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 "
        "160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 "
        "1024 0 0 0 0 524288 9216 64 9216 9216 64 9216 9216 64 9216 0 0 0 64 61440 30720 114688 30720 18 2048 "
        "600479950315538 2048 1 237494511599680 29686814005248 216 1 1 128 128 128 128 8192 1 589824 16384 16384 1 "
        "589824 16384 16384 1 589824 16384 16384 65536 2425344 4785152 0 7144960 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28379136};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_60_tnd_swizzle)
{
    int64_t actual_seq_qlist[1] = {2048};
    int64_t actual_seq_kvlist[1] = {2048};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 64}, {2048, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907145396;
    std::string expectTilingData =
        "32 1 4 1 2048 2048 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 4 1 0 0 0 0 0 0 0 0 16 "
        "549755813952 549755813952 16 549755814016 154618822688 0 0 32 64 96 128 160 192 224 256 288 320 352 384 416 "
        "448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 160 192 224 "
        "256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 0 0 0 "
        "0 524288 16384 32 16384 16384 32 16384 16384 32 16384 0 0 0 32 61440 30720 114688 30720 18 2048 18 2048 1 0 "
        "0 0 0 0 0 0 0 0 0 1 524288 16384 16384 1 524288 16384 16384 1 524288 16384 16384 65536 2163200 3343360 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {25495040};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_62_tnd_gqa_nz)
{
    int64_t actual_seq_qlist[1] = {2048};
    int64_t actual_seq_kvlist[1] = {2048};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048, 4, 8}, {2048, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2048, 4, 72}, {2048, 4, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 1, 72}, {2048, 1, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.117851f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703249444016176;
    std::string expectTilingData =
        "32 1 1 4 2048 2048 72 72 4575657222447651805 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 72057594037927936 "
        "0 0 0 16 549755813952 549755813952 16 549755814016 137438953504 0 0 32 64 96 128 160 192 224 256 288 320 "
        "352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 0 0 0 0 32 64 96 128 "
        "160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 "
        "1024 0 0 0 0 524288 9216 64 9216 2304 64 2304 2304 64 2304 0 0 0 64 61440 30720 114688 30720 18 2048 "
        "600479950315538 2048 1 237494511599680 29686814005248 216 1 1 128 128 128 128 8192 1 589824 16384 16384 1 "
        "147456 16384 16384 1 147456 16384 16384 65536 2425344 3015680 0 3606016 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24840192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_63_mla_rope)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 192}, {1, 128, 1, 192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.072168f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 20125462445953074;
    std::string expectTilingData =
        "32 1 1 1 128 128 192 192 4575657222441520337 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 0 0 0 0 1 "
        "549755813952 549755813952 1 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 24576 1 "
        "24576 24576 1 24576 24576 1 24576 0 0 0 1 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 0 0 0 0 0 1 "
        "24576 18432 6144 1 24576 18432 6144 1 24576 18432 6144 65536 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {28115456};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_64_sink)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 128, 8}, {1, 1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // faker drops dimNum==0 outputs; keep placeholders so dsink stays at IR index 6
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096);
    int64_t expectTilingKey = 19703248907145266;
    std::string expectTilingData =
        "32 1 1 1 128 128 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 1 1 0 0 0 0 1 2 1 0 1 "
        "549755813952 549755813952 1 549755814016 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16384 8192 1 8192 "
        "8192 1 8192 8192 1 8192 512 1 512 1 61440 30720 114688 30720 1 16384 1 16384 1 0 0 0 0 0 0 0 0 0 0 1 8192 "
        "16384 8192 1 8192 16384 8192 1 8192 16384 8192 65536 0 0 0 0 2425344 2 1 1 2 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23397376};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_65_rd_causal_deter)
{
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 256, 1, 64}, {1, 256, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 4096, 1);
    int64_t expectTilingKey = 19745030349001776;
    std::string expectTilingData =
        "32 1 1 1 256 256 64 64 4575657222448611328 255 2147483647 0 0 0 0 1 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 2 549755813952 549755813952 2 549755814016 4294967299 0 0 1 3 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 21856 5462 3 5460 5462 3 5460 5462 3 5460 0 0 0 3 61440 30720 114688 30720 1 21856 1 21824 1 0 0 0 0 "
        "0 0 0 0 0 0 1 16384 16384 16384 1 16384 16384 16384 1 16384 16384 16384 65536 131584 197632 0 0 0 0 0 0 0 1 "
        "3 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21235200};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_66_tnd_deter_bn2s2_swizzle)
{
    // Prefix-sum TND, MHA, dense+deter. Many unique (B,N,S1) tiles so CheckExceedL2Cache
    // trips the real 128MB L2; BN2S2 is kept so ConfigureTndDeterBn2S2Swizzle can run.
    int64_t actual_seq_qlist[8] = {2048, 4096, 6144, 8192, 10240, 12288, 14336, 16640};
    int64_t actual_seq_kvlist[8] = {2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{16640, 16, 8}, {16640, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16640, 16, 8}, {16640, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16640, 16, 64}, {16640, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16384, 16, 64}, {16384, 16, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 28750030580486324;
    std::string expectTilingData =
        "32 8 16 1 2304 2048 64 64 4575657222448611328 255 2147483647 2147483647 0 0 0 4 1 0 0 0 0 0 0 0 0 18 "
        "549755813952 549755813952 16 549755814016 4754528796704 0 0 1040 2080 3120 4160 5200 6240 7280 8320 9360 "
        "10400 11440 12480 13520 14560 15600 16640 17680 18720 19760 20800 21840 22880 23920 24960 26000 27040 28080 "
        "29122 30166 31210 32254 0 0 0 0 1040 2080 3120 4160 5200 6240 7280 8320 9360 10400 11440 12480 13520 14560 "
        "15600 16640 17680 18720 19760 20800 21840 22880 23920 24960 26000 27040 28080 29122 30166 31210 32254 33280 "
        "0 0 0 0 17039360 532480 32 532480 524288 32 524288 524288 32 524288 0 0 0 32 61440 30720 114688 30720 555 "
        "20480 555 20480 1 0 0 0 0 0 0 0 0 0 0 17 17039360 16384 16384 16 16777216 16384 16384 16 16777216 16384 "
        "16384 65536 68223488 69403648 0 0 0 0 0 0 0 1 1040 70583808 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 "
        "4194304 8388608 12582912 16777216 20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "4194304 8388608 12582912 16777216 20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 256 "
        "512 768 1024 1280 1536 1792 2080 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 4194304 8388608 12582912 "
        "16777216 20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4194304 8388608 12582912 16777216 "
        "20971520 25165824 29360128 34078720 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 128 256 384 512 640 768 896 1040 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {93915648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_67_band_deter_pq_le_m)
{
    // Pad BAND + deter. B*N is a multiple of AIC so blockOuter==aicNum; unique tiles
    // exceed 128MB L2; Hybrid soc reports AIV=2*AIC so SelectDeterBandSchedule runs
    // the p+q<=m branch.
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{32, 1, 4096, 8}, {32, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 1, 4096, 8}, {32, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 4096, 1, 128}, {32, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096, 1);
    int64_t expectTilingKey = 19747229909128240;
    std::string expectTilingData =
        "64 32 1 1 4096 4096 128 128 4575657222443697349 255 256 256 0 0 154 1 1 0 0 17179869186 8796093022211 "
        "72620548286316544 0 0 3 32 549755813952 549755813952 32 549755814016 661424963616 0 0 1024 2048 3072 4096 "
        "5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 "
        "23552 24576 25600 26624 27648 28672 29696 30720 31744 0 0 0 0 1024 2048 3072 4096 5120 6144 7168 8192 9216 "
        "10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600 26624 27648 "
        "28672 29696 30720 31744 32768 0 0 0 0 16777216 262144 64 262144 262144 64 262144 262144 64 262144 0 0 0 64 "
        "61440 30720 114688 30720 547 4096 600479950299136 30720 1 237494511599680 29686814005248 216 10 10 104 104 "
        "2048 2048 131072 16 16777216 16384 16384 16 16777216 16384 16384 16 16777216 16384 16384 65536 67174912 "
        "134284288 0 201393664 0 0 0 0 0 1 160 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {226560000};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_68_band_deter_causal_odd_b)
{
    // Near-causal BAND (next=0), odd B so SelectDeterBandSchedule can take the
    // lower-causal path and CalcCausalSingleBatchRound.
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{33, 1, 4096, 8}, {33, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{33, 1, 4096, 8}, {33, 1, 4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{33, 4096, 1, 128}, {33, 4096, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfoHybrid, 4096, 1);
    int64_t expectTilingKey = 19747229909128240;
    std::string expectTilingData =
        "64 33 1 1 4096 4096 128 128 4575657222443697349 255 65536 0 0 0 528 1 1 0 0 17179869186 8796093022211 "
        "72339073309605888 0 0 1 32 549755813952 549755813952 32 549755814016 2340757176352 0 0 1041 2083 3124 4167 "
        "5208 6252 7293 8338 9384 10425 11472 12520 13561 14610 15660 16701 17752 18804 19857 20911 21966 23007 "
        "24063 25137 26196 27256 28317 29401 30489 31581 32767 0 0 0 0 1041 2083 3124 4167 5208 6252 7293 8338 9384 "
        "10425 11472 12520 13561 14610 15660 16701 17752 18804 19857 20911 21966 23007 24063 25137 26196 27256 28317 "
        "29401 30489 31581 32767 33792 0 0 0 0 17301504 270336 64 270336 270336 64 270336 270336 64 270336 0 0 0 64 "
        "61440 30720 114688 30720 564 6144 600479950298607 28672 1 237494511599680 29686814005248 216 10 10 168 168 "
        "2112 2112 135168 17 17301504 16384 16384 17 17301504 16384 16384 17 17301504 16384 16384 65536 69272064 "
        "138478592 0 207685120 0 0 0 0 0 1 576 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {232982528};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_69_tnd_line_left_up)
{
    int64_t actual_seq_qlist[2] = {4096, 10240};
    int64_t actual_seq_kvlist[2] = {4096, 8192};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{10240, 5, 8}, {10240, 5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10240, 5, 8}, {10240, 5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{10240, 5, 64}, {10240, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192, 5, 64}, {8192, 5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19745030349001904;
    std::string expectTilingData =
        "32 2 5 1 6144 4096 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "844424930131968 0 0 0 48 549755813952 549755813952 32 549755814016 1052266987552 0 0 281 766 1259 1686 2232 "
        "2615 3213 3568 4195 4525 5176 5441 5743 6120 6651 6911 7207 7557 8037 8378 8662 8998 9432 9851 10129 10443 "
        "10859 11325 11590 11905 12269 0 0 0 0 281 766 1259 1686 2232 2615 3213 3568 4195 4525 5176 5441 5743 6120 "
        "6651 6911 7207 7557 8037 8378 8662 8998 9432 9851 10129 10443 10859 11325 11590 11905 12269 12800 0 0 0 0 "
        "6553600 102400 32 102400 81920 32 81920 81920 32 81920 0 0 0 32 61440 30720 114688 30720 214 10240 214 "
        "10240 1 0 0 0 0 0 0 0 0 0 0 4 3276800 16384 16384 3 2621440 16384 16384 3 2621440 16384 16384 65536 "
        "13173248 23659520 0 0 0 0 0 0 0 1 276 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 256 1 0 16777216 41943040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 16777216 41943040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 98 276 276 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {55117312};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_73_tnd_causal_gqa)
{
    int64_t actual_seq_qlist[2] = {512, 1280};
    int64_t actual_seq_kvlist[2] = {512, 1024};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1280, 6, 8}, {1280, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1280, 6, 8}, {1280, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1280, 6, 64}, {1280, 6, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024, 3, 64}, {1024, 3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19709845976913072;
    std::string expectTilingData =
        "32 2 3 2 768 512 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 6 549755813952 549755813952 4 549755814016 25769803804 0 0 7 18 27 37 48 55 66 75 85 "
        "96 103 111 120 127 135 144 151 159 168 175 183 192 199 207 216 223 231 0 0 0 0 0 0 0 0 7 18 27 37 48 55 66 "
        "75 85 96 103 111 120 127 135 144 151 159 168 175 183 192 199 207 216 223 231 240 0 0 0 0 0 0 0 0 140448 "
        "17555 28 17535 7022 28 7014 7022 28 7014 0 0 0 28 61440 30720 114688 30720 5 17568 5 17184 1 0 0 0 0 0 0 0 "
        "0 0 0 1 491520 16384 16384 1 196608 16384 16384 1 196608 16384 16384 65536 2032128 2819072 0 0 0 0 0 0 0 1 "
        "18 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 1 1 0 262144 655360 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 262144 655360 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 20 56 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 16 10 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 4 12 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24577536};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_74_tnd_causal_n1)
{
    int64_t actual_seq_qlist[2] = {512, 1536};
    int64_t actual_seq_kvlist[2] = {768, 1280};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1536, 1, 8}, {1536, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 1, 8}, {1536, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1536, 1, 64}, {1536, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 1, 64}, {1280, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19745030349001904;
    std::string expectTilingData =
        "32 2 1 1 1024 768 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "844424930131968 0 0 0 8 549755813952 549755813952 6 549755814016 8589934610 0 0 2 5 7 11 24 26 28 30 33 35 "
        "37 39 43 45 47 52 54 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 5 7 11 24 26 28 30 33 35 37 39 43 45 47 52 54 56 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 50976 5462 18 5450 4552 18 4536 4552 18 4536 0 0 0 18 61440 30720 "
        "114688 30720 2 20256 2 20192 1 0 0 0 0 0 0 0 0 0 0 1 98304 16384 16384 1 81920 16384 16384 1 81920 16384 "
        "16384 65536 459264 787456 0 0 0 0 0 0 0 1 13 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
        "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 1 0 393216 917504 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 393216 917504 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 20 72 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 6 20 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 16 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {22087168};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_75_tnd_band_bns2)
{
    int64_t actual_seq_qlist[2] = {1024, 2304};
    int64_t actual_seq_kvlist[2] = {256, 640};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{2304, 1, 8}, {2304, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2304, 1, 8}, {2304, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2304, 1, 64}, {2304, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 1, 64}, {640, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =
        "32 2 1 1 1280 384 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "844424930131968 0 0 0 10 549755813952 549755813952 3 549755814016 4294967315 0 0 5 6 7 13 14 15 21 22 23 24 "
        "25 32 33 34 35 43 44 45 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 6 7 13 14 15 21 22 23 24 25 32 33 34 35 43 44 "
        "45 46 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 39680 7761 19 7758 2156 19 2152 2156 19 2152 0 0 0 19 61440 30720 "
        "114688 30720 2 8960 2 8704 1 0 0 0 0 0 0 0 0 0 0 1 147456 16384 16384 1 40960 16384 8192 1 40960 16384 8192 "
        "65536 655872 820224 0 0 0 0 0 0 0 1 7 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 262144 753664 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 262144 753664 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12 33 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21956096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_79_tnd_line_band_unpad)
{
    // Stay TND (not isAllSame). n2>=2 + deter BAND so PreferTNDLineDeter actually calls GetTNDBandMN.
    int64_t actual_seq_qlist[2] = {4096, 10240};
    int64_t actual_seq_kvlist[2] = {2048, 5120};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{10240, 4, 8}, {10240, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10240, 4, 8}, {10240, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{10240, 4, 64}, {10240, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5120, 4, 64}, {5120, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =
        "32 2 4 1 6144 3072 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "844424930131968 0 0 0 48 549755813952 549755813952 24 549755814016 107374182432 0 0 179 344 510 661 826 991 "
        "1171 1336 1501 1653 1818 1983 2219 2464 2709 2954 3222 3467 3712 3957 4202 4426 4671 4916 5161 5406 5674 "
        "5919 6164 6409 6655 0 0 0 0 179 344 510 661 826 991 1171 1336 1501 1653 1818 1983 2219 2464 2709 2954 3222 "
        "3467 3712 3957 4202 4426 4671 4916 5161 5406 5674 5919 6164 6409 6655 6656 0 0 0 0 3407872 81920 32 81920 "
        "40960 32 40960 40960 32 40960 0 0 0 32 61440 30720 114688 30720 111 28672 111 28672 1 0 0 0 0 0 0 0 0 0 0 3 "
        "2621440 16384 16384 2 1310720 16384 16384 2 1310720 16384 16384 65536 10551808 15795200 0 0 0 0 0 0 0 1 35 "
        "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1040 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 256 1 0 8388608 27262976 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 8388608 27262976 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 14 35 35 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 ";
    std::vector<size_t> expectWorkspaces = {42010112};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_80_tnd_band_bns2_n1)
{
    // n2==1 skips line deter; per-batch square but different lengths keep TND BN2S2 for split-dk.
    int64_t actual_seq_qlist[2] = {256, 768};
    int64_t actual_seq_kvlist[2] = {256, 768};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{768, 1, 8}, {768, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{768, 1, 8}, {768, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 1, 64}, {768, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144, 1);
    int64_t expectTilingKey = 19747229372257456;
    std::string expectTilingData =
        "32 2 1 1 512 512 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 4 549755813952 549755813952 4 549755814016 4294967314 0 0 1 2 3 4 5 6 8 9 10 11 12 13 "
        "14 15 17 18 19 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 17 18 19 20 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 18208 2731 18 2725 2731 18 2725 2731 18 2725 0 0 0 18 61440 30720 114688 30720 1 "
        "18208 1 18144 1 0 0 0 0 0 0 0 0 0 0 1 49152 16384 16384 1 49152 16384 16384 1 49152 16384 16384 65536 "
        "262656 459776 0 0 0 0 0 0 0 1 4 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "1040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 65536 327680 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 65536 327680 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 20 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {21628416};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_81_tnd_bn2s2_lu_ge)
{
    // TND CASUAL needs n1 even and S1>=S2 every batch; stay unpadded via different batch sizes.
    int64_t actual_seq_qlist[2] = {512, 1280};
    int64_t actual_seq_kvlist[2] = {256, 640};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1280, 4, 8}, {1280, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1280, 4, 8}, {1280, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640, 4, 64}, {640, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907146416;
    std::string expectTilingData =
        "32 2 4 1 768 384 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 8589934594 8796093022209 "
        "281474976710656 0 0 0 6 549755813952 549755813952 3 549755814016 12884901918 0 0 3 7 10 14 17 21 24 27 31 "
        "34 37 41 46 49 52 55 59 64 67 70 73 77 82 85 88 91 95 100 103 0 0 0 0 0 0 3 7 10 14 17 21 24 27 31 34 37 41 "
        "46 49 52 55 59 64 67 70 73 77 82 85 88 91 95 100 103 104 0 0 0 0 0 0 56800 10923 30 10913 5462 30 5442 5462 "
        "30 5442 0 0 0 30 61440 30720 114688 30720 2 26080 2 26016 1 0 0 0 0 0 0 0 0 0 0 1 327680 16384 16384 1 "
        "163840 16384 16384 1 163840 16384 16384 65536 1376768 2032640 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 "
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 8 8 8 8 8 8 8 8 8 "
        "8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {23660032};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_82_tnd_bn2s2_rd_le)
{
    int64_t actual_seq_qlist[2] = {256, 768};
    int64_t actual_seq_kvlist[2] = {512, 1280};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{768, 4, 8}, {768, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{768, 4, 8}, {768, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{768, 4, 64}, {768, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1280, 4, 64}, {1280, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907146420;
    std::string expectTilingData =
        "32 2 4 1 512 768 64 64 4575657222448611328 255 2147483647 0 0 0 0 4 1 0 0 12884901890 8796093022210 "
        "281474976710656 0 0 0 4 549755813952 549755813952 6 549755814016 25769803808 0 0 2 4 8 10 12 16 18 20 24 26 "
        "28 32 36 40 44 48 56 60 64 68 72 80 84 88 92 96 104 108 112 116 120 0 0 0 0 2 4 8 10 12 16 18 20 24 26 28 "
        "32 36 40 44 48 56 60 64 68 72 80 84 88 92 96 104 108 112 116 120 128 0 0 0 0 65536 6144 32 6144 10240 32 "
        "10240 10240 32 10240 0 0 0 32 61440 30720 114688 30720 3 4096 3 4096 1 0 0 0 0 0 0 0 0 0 0 1 196608 16384 "
        "16384 1 327680 16384 16384 1 327680 16384 16384 65536 852480 2032640 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 "
        "131072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 131072 131072 131072 131072 131072 131072 131072 131072 131072 "
        "131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 131072 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24184320};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FlashAttentionScoreGradTiling, FlashAttentionScoreGrad_950_tiling_83_tnd_bn2s2_band_sq)
{
    // BAND sparseType needs isS1S2Same; different per-batch lengths keep TND (not isAllSame).
    int64_t actual_seq_qlist[2] = {512, 1536};
    int64_t actual_seq_kvlist[2] = {512, 1536};
    auto compileInfo = MakeA5CompileInfo();
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScoreGrad",
        {
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1536, 4, 8}, {1536, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1536, 4, 8}, {1536, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_qlist},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, actual_seq_kvlist},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1536, 4, 64}, {1536, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}},
        &compileInfo, "Ascend950", A5SocInfo, 262144);
    int64_t expectTilingKey = 19703248907146420;
    std::string expectTilingData =
        "32 2 4 1 1024 1024 64 64 4575657222448611328 255 256 256 0 0 0 4 1 0 0 17179869186 8796093022211 "
        "562949953421312 0 0 0 8 549755813952 549755813952 8 549755814016 42949672992 0 0 8 16 24 32 40 48 56 64 80 "
        "88 96 104 112 128 144 152 160 168 176 192 208 216 224 232 240 256 272 280 288 296 304 0 0 0 0 8 16 24 32 40 "
        "48 56 64 80 88 96 104 112 128 144 152 160 168 176 192 208 216 224 232 240 256 272 280 288 296 304 320 0 0 0 "
        "0 163840 12288 32 12288 12288 32 12288 12288 32 12288 0 0 0 32 61440 30720 114688 30720 6 10240 6 10240 1 0 "
        "0 0 0 0 0 0 0 0 0 1 393216 16384 16384 1 393216 16384 16384 1 393216 16384 16384 65536 1638912 2819072 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 "
        "262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 "
        "262144 262144 262144 262144 262144 262144 262144 262144 262144 0 0 0 0 0 0 0 0 0 0 0 0 262144 262144 262144 "
        "262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 262144 "
        "262144 262144 262144 262144 262144 262144 0 0 0 0 0 0 0 0 0 0 0 0 16 16 16 16 16 16 16 16 16 16 16 16 16 16 "
        "16 16 16 16 16 16 16 16 16 16 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {24970752};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
