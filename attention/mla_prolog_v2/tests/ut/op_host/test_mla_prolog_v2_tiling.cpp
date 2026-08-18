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
#include "../../../op_host/mla_prolog_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
// 构造版本
std::string MlaPrologV2_tiling_A2SocInfo = "{\n"
                                           "  \"hardware_info\": {\n"
                                           "    \"BT_SIZE\": 0,\n"
                                           "    \"load3d_constraints\": \"1\",\n"
                                           "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                           "    \"Intrinsic_data_move_l12ub\": true,\n"
                                           "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                           "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                           "    \"UB_SIZE\": 196608,\n"
                                           "    \"L2_SIZE\": 201326592,\n"
                                           "    \"L1_SIZE\": 524288,\n"
                                           "    \"L0A_SIZE\": 65536,\n"
                                           "    \"L0B_SIZE\": 65536,\n"
                                           "    \"L0C_SIZE\": 131072,\n"
                                           "    \"vector_core_cnt\": 40,\n"
                                           "    \"cube_core_cnt\": 20,\n"
                                           "    \"socVersion\": \"Ascend910_B3\"\n"
                                           "  }\n"
                                           "}";

class MlaPrologV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MlaPrologV3 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MlaPrologV3 TearDown" << std::endl; }
};

// 全量化kvcache pertensor量化
TEST_F(MlaPrologV2, MlaProlog_tiling_test0)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    // char* cacheMode = "PA_BSND";
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV2",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
        },
        {
            {{{8, 1, 32, 64}, {8, 1, 32, 512}}, ge::DT_INT8, ge::FORMAT_ND},      // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8, 32, 1}, {8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_q_nope
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV2_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933473;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 全量化kvcache 非量化
TEST_F(MlaPrologV2, MlaProlog_tiling_test1)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    // char* cacheMode = "PA_BSND";
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV2",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_INT8, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 512 + 64}, {7168, 512 + 64}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},               // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{8 * 1, 1}, {8 * 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},                     // dequant_scale_x
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{1, 512 + 64}, {1, 512 + 64}}, ge::DT_FLOAT, ge::FORMAT_ND},               // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
        },
        {
            {{{8, 1, 32, 64}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},      // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8, 32, 1}, {8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_q_nope
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV2_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933409;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// 半量化kvcache 量化
TEST_F(MlaPrologV2, MlaProlog_tiling_test2)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    // char* cacheMode = "PA_BSND";
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV2",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 576}, {7168, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                         // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},          // kr_cache
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dkv_kr
            {{{1, 512}, {1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // quant_scale_ckv
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},                           // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_INT8, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_INT8, ge::FORMAT_ND},   // kr_cache
            {{{8, 32, 1}, {8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_q_nope
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV2_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933345;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 半量化kvcache 非量化
TEST_F(MlaPrologV2, MlaProlog_tiling_test3)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    // char* cacheMode = "PA_BSND";
    gert::TilingContextPara tilingContextPara(
        "MlaPrologV2",
        {
            {{{8, 1, 7168}, {8, 1, 7168}}, ge::DT_BF16, ge::FORMAT_ND},                               // token_x
            {{{7168, 1536}, {7168, 1536}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                       // weight_dq
            {{{1536, 32 * (128 + 64)}, {1536, 32 * (128 + 64)}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // weight_uq_qr
            {{{32, 128, 512}, {32, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},                           // weight_uk
            {{{7168, 576}, {7168, 576}}, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},                         // weight_dkv_kr
            {{{1536}, {1536}}, ge::DT_BF16, ge::FORMAT_ND},                              // rmsnorm_gamma_cq
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},                                // rmsnorm_gamma_ckv
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_sin
            {{{8, 1, 64}, {8, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},                      // rope_cos
            {{{8, 1}, {8, 1}}, ge::DT_INT64, ge::FORMAT_ND},                             // cache_index
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},          // kr_cache
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_x
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dq
            {{{1, 32 * (128 + 64)}, {1, 32 * (128 + 64)}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dequant_scale_w_uq_qr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // dequant_scale_w_dkv_kr
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                                     // quant_scale_ckr
            {{{1, 1536}, {1, 1536}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // smooth_scales_cq
        },
        {
            {{{8, 1, 32, 512}, {8, 1, 32, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query
            {{{8, 1, 32, 64}, {8, 1, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope
            {{{16, 128, 1, 512}, {16, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // kv_cache
            {{{16, 128, 1, 64}, {16, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // kr_cache
            {{{8, 32, 1}, {8, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},              // dequant_scale_q_nope
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
        },
        &compileInfo, "Ascend910_B3", MlaPrologV2_tiling_A2SocInfo, 4096);
    int64_t expectTilingKey = 3933281;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}
